#!/usr/bin/env python3
"""実験15 (本命): 早期層 patch → CoT 自由生成 による S1→S2 因果閉鎖.

仮説 H15: 実験8 は「早期層を直せば答えが戻る」(clean CoT 強制下) を示した。
本実験は「早期層を直せば CoT の逸脱そのものが消える」を示す。損傷→分岐→搬送の
3 リンクすべてを介入で支持し、統一主張の背骨を完成させる。

手法: typo 質問プロンプトの摂動語スパン (question_span) の residual を、早期窓で
clean 値に patch したまま **CoT 全体を greedy 自由生成** する (答えだけでなく
CoT 全体)。patch は prefill で 1 回注入され、以降の decode は KV キャッシュ経由で
効果を保持する (患部は prompt 内なので decode 追加注入は不要 — exp8 の
`PatchInjector` をそのまま生成ループ全体に被せる)。

指標 (1 ペア = flip ペア: clean 正解 ∧ typo 誤答):
  - ROUGE-L(patched, clean 生成 CoT)   … 逸脱がどれだけ clean へ戻ったか
  - 発散オンセット (patched が clean 生成と分岐する最初のトークン位置)
  - flip (patched の抽出答えが正解か)   … 逸脱の解消

セル = {窓 (早期/中期/後期) × 方向 (denoise=clean→pert / noise=pert→clean)}。
統制:
  - 後期窓 patch (効かないはず)
  - sham patch (recipient 自身の活性 = no-op、生成 bit 不変を検証)
  - noise 方向 (clean 実行に typo 状態を注入 → 分岐・flip が誘発されるか = 十分性)

GPU 実行は必ず tmp/gpu-locks/run_with_gpu.sh 経由 (CUDA_VISIBLE_DEVICES は
ヘルパーが設定。このスクリプトでは変更しない)。ペア×条件ごとに JSON を書き、
config ハッシュ一致でスキップ (冪等)。シャード分割は 1 シャード = pairs[i::N]。

例 (スモーク):
  bash tmp/gpu-locks/run_with_gpu.sh \\
    /path/.venv/bin/python scripts/exp15/run_free_generation.py \\
    --model google/gemma-3-4b-it --benchmark gsm8k \\
    --baseline-dir  <arch>/outputs/baseline/gemma-3-4b-it_gsm8k \\
    --perturbed-dir-lxt <arch>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \\
    --n-pairs 16 --levels early late --directions denoise --noop-check \\
    --output-dir results/exp15/smoke_gemma_gsm8k
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.intervention.archive_loader import load_pair_records
from typo_cot.intervention.cell_builder import build_cell_inputs
from typo_cot.intervention.free_generation import (
    align_span_positions,
    cot_rouge_l_f,
    divergence_index,
    generate_ids,
    generate_ids_patched,
    locate_word_char_spans,
)
from typo_cot.intervention.patching import (
    capture_activations,
    find_decoder_layers,
    result_is_current,
    select_flip_pairs,
    shard_slice,
    span_end_token,
)
from typo_cot.intervention.records import PairRecord
from typo_cot.models.wrapper import ModelWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_free_generation")

LEVELS = ("early", "mid", "late")
DIRECTIONS = ("denoise", "noise")  # denoise=clean→pert, noise=pert→clean


class PairPreparationError(Exception):
    """ペアを実験15 に使えない理由 (除外) を表す."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--perturbed-dir-lxt", required=True)
    p.add_argument("--perturbed-dir-rnd", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-pairs", type=int, default=0, help="総ペア数 (0以下=全 flip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--site", default="residual", choices=["residual", "attn", "mlp"])
    p.add_argument("--window-width", type=int, default=6, help="自動窓の幅 (既定 6 = [0,6) 相当)")
    p.add_argument("--early", default=None, help="早期窓 'start:end' (既定: [0,width))")
    p.add_argument("--mid", default=None, help="中期窓 'start:end' (既定: 中央 width 層)")
    p.add_argument("--late", default=None, help="後期窓 'start:end' (既定: [n-width,n))")
    p.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    p.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=list(DIRECTIONS))
    p.add_argument("--noop-check", action="store_true", help="sham (恒等パッチ) の no-op 検証")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--trigger-pattern", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-store-chars", type=int, default=4000, help="保存する生成テキストの上限")
    return p.parse_args()


def config_hash(args: argparse.Namespace, windows: dict) -> str:
    payload = {
        "model": args.model,
        "benchmark": args.benchmark,
        "site": args.site,
        "windows": {k: list(v) for k, v in windows.items()},
        "levels": sorted(args.levels),
        "directions": sorted(args.directions),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "trigger_pattern": args.trigger_pattern,
        "schema": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _parse_window(s: str | None, default: tuple[int, int]) -> tuple[int, int]:
    if s is None:
        return default
    a, b = s.split(":")
    return (int(a), int(b))


def resolve_windows(
    n_layers: int, width: int, args: argparse.Namespace
) -> dict[str, tuple[int, int]]:
    """モデル層数から早期/中期/後期の 3 窓を決める (exp8: 早期 residual [0,6) が最良)."""
    early = _parse_window(args.early, (0, min(width, n_layers)))
    m0 = max(0, n_layers // 2 - width // 2)
    mid = _parse_window(args.mid, (m0, min(m0 + width, n_layers)))
    l0 = max(0, n_layers - width)
    late = _parse_window(args.late, (l0, n_layers))
    return {"early": early, "mid": mid, "late": late}


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, return_tensors=None, add_special_tokens=True)["input_ids"]


def _span_token_positions(
    tokenizer, prompt: str, question: str, words: list[str]
) -> list[int | None]:
    """質問領域の各語について offset_mapping からスパン末尾トークン位置を返す (None 可)."""
    char_spans = locate_word_char_spans(prompt, question, words)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    out: list[int | None] = []
    for span in char_spans:
        if span is None:
            out.append(None)
        else:
            out.append(span_end_token(offsets, span[0], span[1]))
    return out


class PreparedFreePair:
    """1 ペア分: clean/typo プロンプトのトークン列と摂動語スパンのトークン位置整列."""

    def __init__(self) -> None:
        self.prompt_clean: str = ""
        self.prompt_typo: str = ""
        self.ids_clean: list[int] = []
        self.ids_typo: list[int] = []
        self.clean_positions: list[int] = []
        self.pert_positions: list[int] = []
        self.meta: dict = {}


def prepare_free_pair(pair: PairRecord, tokenizer, trigger_pattern: str | None) -> PreparedFreePair:
    """flip ペアを自由生成用に準備する (プロンプト構築 + 摂動語スパン整列).

    Raises:
        PairPreparationError: 除外理由付き
    """
    perturbed_tokens = pair.extra.get("perturbed_tokens", [])
    if not perturbed_tokens:
        raise PairPreparationError("no_perturbed_tokens")
    words_clean = [str(d.get("original_token", "")).strip() for d in perturbed_tokens]
    words_typo = [str(d.get("perturbed_token", "")).strip() for d in perturbed_tokens]

    cells = build_cell_inputs(pair, trigger_pattern=trigger_pattern)
    prompt_clean = cells.prompts["A"]  # (clean 質問)
    prompt_typo = cells.prompts["B"]  # (typo 質問)

    clean_tok = _span_token_positions(tokenizer, prompt_clean, pair.question_clean, words_clean)
    pert_tok = _span_token_positions(tokenizer, prompt_typo, pair.question_typo, words_typo)
    aligned = align_span_positions(clean_tok, pert_tok)
    if not aligned.clean_positions:
        raise PairPreparationError("span_not_found")

    prepared = PreparedFreePair()
    prepared.prompt_clean = prompt_clean
    prepared.prompt_typo = prompt_typo
    prepared.ids_clean = _encode(tokenizer, prompt_clean)
    prepared.ids_typo = _encode(tokenizer, prompt_typo)
    prepared.clean_positions = aligned.clean_positions
    prepared.pert_positions = aligned.pert_positions

    # スパン位置がプロンプト長内にあることを保証 (offset_mapping 由来なので通常成立)
    if max(prepared.clean_positions) >= len(prepared.ids_clean) or max(
        prepared.pert_positions
    ) >= len(prepared.ids_typo):
        raise PairPreparationError("span_position_out_of_range")

    prepared.meta = {
        "sample_id": pair.sample_id,
        "n_span_words": aligned.n_words,
        "n_span_aligned": len(aligned.clean_positions),
        "n_span_dropped": aligned.n_dropped,
        "prompt_len_clean": len(prepared.ids_clean),
        "prompt_len_typo": len(prepared.ids_typo),
        "correct_answer": pair.correct_answer,
        "answer_clean_archive": pair.answer_clean,
        "answer_typo_archive": pair.answer_typo,
        "cell_exclude_reasons": list(cells.exclude_reasons),
    }
    return prepared


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _gen_summary(tokenizer, extractor, ids: list[int], correct: str, cap: int) -> dict:
    text = _decode(tokenizer, ids)
    answer = extractor.extract(text).extracted_answer
    return {
        "answer": answer,
        "is_correct": bool(extractor.is_correct(answer, correct)),
        "n_gen_tokens": len(ids),
        "text": text[:cap],
    }


def run_pair(model, tokenizer, layers, prepared: PreparedFreePair, extractor, args, windows) -> dict:
    """1 ペアの baseline + sham + {窓 × 方向} セルを自由生成で実行する."""
    device = next(model.parameters()).device
    correct = prepared.meta["correct_answer"]
    cap = args.max_store_chars
    site = args.site

    ids_clean_t = torch.tensor([prepared.ids_clean], device=device)
    ids_typo_t = torch.tensor([prepared.ids_typo], device=device)

    # --- baseline 自由生成 (無パッチ) --------------------------------------
    gen_clean = generate_ids(model, ids_clean_t, args.max_new_tokens)
    gen_typo = generate_ids(model, ids_typo_t, args.max_new_tokens)
    base_clean = _gen_summary(tokenizer, extractor, gen_clean, correct, cap)
    base_typo = _gen_summary(tokenizer, extractor, gen_typo, correct, cap)
    text_clean = _decode(tokenizer, gen_clean)
    text_typo = _decode(tokenizer, gen_typo)

    rouge_typo_vs_clean = cot_rouge_l_f(text_clean, text_typo)
    onset_typo_vs_clean = divergence_index(gen_clean, gen_typo)

    result: dict = {
        "baseline": {
            "clean": base_clean,
            "typo": base_typo,
            "rouge_l_typo_vs_clean": rouge_typo_vs_clean,
            "onset_typo_vs_clean": onset_typo_vs_clean,
        }
    }

    # --- 捕捉 (residual, 必要層のみ) ---------------------------------------
    # sham (--noop-check) は --levels の選択によらず常に早期窓を使うため、
    # used_layers に早期窓の層を必ず含める (さもないと --levels が early を
    # 含まない場合に ActivationCache.values が KeyError で全ペア失敗する)。
    used_layers = sorted({li for lvl in args.levels for li in range(*windows[lvl])})
    if args.noop_check:
        used_layers = sorted(set(used_layers) | set(range(*windows["early"])))
    clean_cache = capture_activations(
        model, ids_clean_t, prepared.clean_positions, sites=(site,), layers=used_layers
    )
    typo_cache = capture_activations(
        model, ids_typo_t, prepared.pert_positions, sites=(site,), layers=used_layers
    )

    # --- sham (恒等パッチ: donor=recipient 自身、早期窓) --------------------
    if args.noop_check:
        early = windows["early"]
        early_layers = list(range(*early))
        values = {li: typo_cache.values(site, li, prepared.pert_positions) for li in early_layers}
        sham_ids = generate_ids_patched(
            model, layers, ids_typo_t, site, early_layers,
            prepared.pert_positions, values, args.max_new_tokens,
        )
        sham_ans = extractor.extract(_decode(tokenizer, sham_ids)).extracted_answer
        result["sham"] = {
            "window": list(early),
            "generation_identical_to_typo": sham_ids == gen_typo,
            "answer_unchanged": sham_ans == base_typo["answer"],
        }

    # --- 本セル: 窓 × 方向 -------------------------------------------------
    cells_out = []
    for level in args.levels:
        window = windows[level]
        w_layers = list(range(*window))
        for direction in args.directions:
            if direction == "denoise":  # clean→pert: recipient=typo, donor=clean
                recip_ids_t = ids_typo_t
                dst_pos = prepared.pert_positions
                src_pos = prepared.clean_positions
                donor_cache = clean_cache
                base_gen_ids = gen_typo
            else:  # noise, pert→clean: recipient=clean, donor=typo
                recip_ids_t = ids_clean_t
                dst_pos = prepared.clean_positions
                src_pos = prepared.pert_positions
                donor_cache = typo_cache
                base_gen_ids = gen_clean

            values = {li: donor_cache.values(site, li, src_pos) for li in w_layers}
            patched_ids = generate_ids_patched(
                model, layers, recip_ids_t, site, w_layers, dst_pos, values, args.max_new_tokens
            )
            text_patched = _decode(tokenizer, patched_ids)
            answer = extractor.extract(text_patched).extracted_answer

            cell = {
                "level": level,
                "window": list(window),
                "direction": direction,
                "recipient": "typo" if direction == "denoise" else "clean",
                "rouge_l_vs_clean": cot_rouge_l_f(text_clean, text_patched),
                "rouge_l_vs_typo": cot_rouge_l_f(text_typo, text_patched),
                "onset_vs_clean": divergence_index(gen_clean, patched_ids),
                "answer": answer,
                "is_correct": bool(extractor.is_correct(answer, correct)),
                "answer_matches_clean_gen": answer == base_clean["answer"],
                "answer_matches_typo_gen": answer == base_typo["answer"],
                "n_gen_tokens": len(patched_ids),
                "identical_to_recipient_baseline": patched_ids == base_gen_ids,
                "text": text_patched[:cap],
            }
            # denoise の主指標: ROUGE 増分 (patched が clean へ戻った量)
            if direction == "denoise":
                cell["rouge_gain_vs_typo"] = cell["rouge_l_vs_clean"] - rouge_typo_vs_clean
            cells_out.append(cell)

    result["cells"] = cells_out
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # flip ペアのロード (LXT-4 / Random-4 半々)
    condition_dirs = {"lxt4": args.perturbed_dir_lxt}
    if args.perturbed_dir_rnd:
        condition_dirs["rnd4"] = args.perturbed_dir_rnd
    n_per_cond = None
    if args.n_pairs and args.n_pairs > 0:
        n_per_cond = max(1, args.n_pairs // len(condition_dirs))

    tasks: list[tuple[str, PairRecord]] = []
    for cond, pdir in condition_dirs.items():
        pairs = load_pair_records(args.baseline_dir, pdir)
        flips = select_flip_pairs(pairs, n=n_per_cond, seed=args.seed)
        logger.info("%s: flip %d 件を選定", cond, len(flips))
        tasks.extend((cond, p) for p in flips)
    tasks.sort(key=lambda t: (t[0], t[1].sample_id))
    tasks = shard_slice(tasks, args.shard_index, args.num_shards)
    logger.info("シャード %d/%d: %d タスク", args.shard_index, args.num_shards, len(tasks))

    # モデルロード (run_with_gpu.sh が CUDA_VISIBLE_DEVICES を設定済み)
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = ModelWrapper(model_name=args.model, device=device, dtype=dtype)
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    layers = find_decoder_layers(model)
    n_layers = len(layers)
    windows = resolve_windows(n_layers, args.window_width, args)
    chash = config_hash(args, windows)
    logger.info("デコーダ層 %d 層。窓=%s config_hash=%s", n_layers, windows, chash)
    extractor = create_extractor(args.benchmark)

    n_done = n_skipped = n_excluded = n_failed = 0
    excluded: dict[str, int] = {}
    for cond, pair in tqdm(tasks, desc="pairs"):
        cond_dir = out_dir / cond
        cond_dir.mkdir(exist_ok=True)
        res_path = cond_dir / f"{pair.sample_id}.json"
        if not args.force and result_is_current(res_path, chash):
            n_skipped += 1
            continue

        payload: dict = {
            "config_hash": chash,
            "sample_id": pair.sample_id,
            "condition": cond,
            "model": args.model,
            "benchmark": args.benchmark,
            "windows": {k: list(v) for k, v in windows.items()},
            "site": args.site,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            prepared = prepare_free_pair(pair, tokenizer, args.trigger_pattern)
        except PairPreparationError as e:
            payload["excluded"] = str(e)
            excluded[str(e)] = excluded.get(str(e), 0) + 1
            n_excluded += 1
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            continue

        payload["prepared"] = prepared.meta
        try:
            result = run_pair(model, tokenizer, layers, prepared, extractor, args, windows)
        except Exception as e:  # noqa: BLE001 - 1ペアの失敗で全体を止めない
            logger.exception("ペア %s (%s) の実行に失敗", pair.sample_id, cond)
            payload["error"] = f"{type(e).__name__}: {e}"
            n_failed += 1
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            continue

        payload.update(result)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        n_done += 1

    summary = {
        "config_hash": chash,
        "config": {
            "model": args.model,
            "benchmark": args.benchmark,
            "site": args.site,
            "windows": {k: list(v) for k, v in windows.items()},
            "levels": args.levels,
            "directions": args.directions,
            "max_new_tokens": args.max_new_tokens,
            "n_pairs": args.n_pairs,
            "shard": [args.shard_index, args.num_shards],
            "timestamp": datetime.now().isoformat(),
        },
        "n_tasks": len(tasks),
        "n_done": n_done,
        "n_skipped_idempotent": n_skipped,
        "n_excluded": n_excluded,
        "n_failed": n_failed,
        "excluded_reasons": excluded,
    }
    shard_tag = f"_shard{args.shard_index}" if args.num_shards > 1 else ""
    with open(out_dir / f"run_summary{shard_tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("完了: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
