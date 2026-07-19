#!/usr/bin/env python3
"""実験8 (activation patching) 実行スクリプト.

アーカイブの baseline/perturbed 生成ログから flip ペア (LXT-4/Random-4 半々) を
選び、各ペアについて 2 run (clean run = 実験1セルA / pert run = セルC 相当;
CoT 以降は同一文字列を teacher-forcing) の活性化を捕捉し、
{部位3 × 層窓 × 方向2} のスイープで donor 活性を注入して
答えトークンの Δlogit・flip・S2 (c1 分布の KL) を測定する。

設計の詳細は docs/dev_notes_08_patching.md を参照。

GPU 実行は必ず run_with_gpu.sh 経由 (CUDA_VISIBLE_DEVICES はヘルパーが設定
するため、このスクリプトでは一切変更しない)。

例 (スモーク):
    bash <...>/run_with_gpu.sh uv run python scripts/exp8/run_patching.py \
        --model google/gemma-3-4b-it --benchmark gsm8k \
        --baseline-dir <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \
        --perturbed-dir-lxt <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
        --perturbed-dir-rnd <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_random \
        --n-pairs 16 --noop-check --output-dir results/exp8/smoke_gemma3-4b_gsm8k
"""

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.intervention.archive_loader import load_pair_records
from typo_cot.intervention.cell_builder import (
    DEFAULT_TRIGGER_PATTERN,
    build_cell_inputs,
    truncate_before_answer,
)
from typo_cot.intervention.patching import (
    DIRECTIONS,
    SITES,
    PatchInjector,
    capture_activations,
    find_decoder_layers,
    first_divergence,
    iter_patch_cells,
    kl_from_logits,
    layer_windows,
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
logger = logging.getLogger("run_patching")

RUNS = ("clean", "pert")  # clean run = (Q_c, C_c) / pert run = (Q_p, C_c)

# 部位 (hook 先モジュール) と直交する「どの位置に注入するか」の 3 種別
SITE_KINDS = ("question_span", "cot_suffix", "answer_span")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HuggingFace モデル名")
    p.add_argument("--benchmark", required=True, help="ベンチマーク名 (gsm8k/mmlu/...)")
    p.add_argument("--baseline-dir", required=True, help="アーカイブ baseline ディレクトリ")
    p.add_argument(
        "--perturbed-dir-lxt", required=True, help="LXT-4 (k4_importance) の perturbed ディレクトリ"
    )
    p.add_argument(
        "--perturbed-dir-rnd", default=None, help="Random-4 (k4_random) の perturbed ディレクトリ"
    )
    p.add_argument("--output-dir", required=True, help="結果出力先")
    p.add_argument(
        "--n-pairs", type=int, default=16, help="総ペア数 (LXT/Random 半々。0以下=全 flip)"
    )
    p.add_argument("--seed", type=int, default=42, help="flip ペア選定のシード")
    p.add_argument("--window-size", type=int, default=3, help="層窓の幅")
    p.add_argument("--window-stride", type=int, default=None, help="層窓の stride (既定: 幅と同じ)")
    p.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    p.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=list(DIRECTIONS))
    p.add_argument(
        "--site-kinds", nargs="+", default=list(SITE_KINDS), choices=list(SITE_KINDS)
    )
    p.add_argument("--max-new-tokens", type=int, default=16, help="答えスパンの生成長")
    p.add_argument("--answer-token-limit", type=int, default=16, help="答え分岐探索のトークン上限")
    p.add_argument("--noop-check", action="store_true", help="恒等パッチの no-op 検証を実行")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--force", action="store_true", help="既存結果を無視して再計算")
    p.add_argument(
        "--trigger-pattern", default=None, help="答え句の正規表現 (既定: The answer is)"
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def config_hash(args: argparse.Namespace) -> str:
    """結果の冪等スキップ判定に使う設定ハッシュ."""
    payload = {
        "model": args.model,
        "benchmark": args.benchmark,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "sites": sorted(args.sites),
        "directions": sorted(args.directions),
        "site_kinds": sorted(args.site_kinds),
        "max_new_tokens": args.max_new_tokens,
        "answer_token_limit": args.answer_token_limit,
        "trigger_pattern": args.trigger_pattern,
        "seed": args.seed,
        "noop_check": args.noop_check,
        "schema": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ペア準備 (トークン化・位置整列)
# ---------------------------------------------------------------------------


class PairPreparationError(Exception):
    """ペアを実験8に使えない理由 (除外) を表す."""


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, return_tensors=None, add_special_tokens=True)["input_ids"]


def _continuation_ids(tokenizer, base_text: str, base_ids: list[int], cont_text: str) -> list[int]:
    """base_text の続きとしての cont_text のトークン列 (境界は base 側に固定)."""
    full = _encode(tokenizer, base_text + cont_text)
    if full[: len(base_ids)] != base_ids:
        raise PairPreparationError("continuation_boundary_mismatch")
    return full[len(base_ids) :]


def _locate_spans(prompt: str, question: str, words: list[str]) -> list[tuple[int, int] | None]:
    """プロンプト内の質問領域で words を順に探し、文字スパンのリストを返す.

    見つからない語の位置には None を入れる (長さ = len(words) を維持し、
    呼び出し側で clean/pert の対応語ごと落とす)。
    """
    q_start = prompt.rfind(question[:80]) if question else -1
    search_from = q_start if q_start >= 0 else 0
    spans: list[tuple[int, int] | None] = []
    cursor = search_from
    for w in words:
        if not w:
            spans.append(None)
            continue
        pos = prompt.find(w, cursor)
        if pos < 0:
            # 順序制約を緩めて質問領域全体から探す
            pos = prompt.find(w, search_from)
        if pos < 0:
            spans.append(None)
        else:
            spans.append((pos, pos + len(w)))
            cursor = pos + len(w)
    return spans


class PreparedPair:
    """1 ペア分のトークン化済み入力と位置整列の結果."""

    def __init__(self) -> None:
        self.input_ids: dict[str, list[int]] = {}
        self.prompt_len: dict[str, int] = {}
        self.span_positions: dict[str, list[int]] = {}
        self.trigger_start: dict[str, int] = {}
        self.readout_tokens: tuple[int, int] = (0, 0)  # (clean側, typo側)
        self.readout_prefix_text: str = ""  # trigger + 共通答え接頭辞 (答え抽出用)
        self.suffix_len: int = 0
        self.meta: dict = {}

    def positions_for_site(self, run: str, site_kind: str) -> list[int]:
        """位置種別 → 当該 run の絶対トークン位置リスト."""
        t = len(self.input_ids[run])
        if site_kind == "question_span":
            return self.span_positions[run]
        if site_kind == "cot_suffix":
            return list(range(self.prompt_len[run], t))
        if site_kind == "answer_span":
            return list(range(self.trigger_start[run], t))
        raise ValueError(f"未知の位置種別: {site_kind}")


def prepare_pair(
    pair: PairRecord,
    tokenizer,
    trigger_pattern: str | None,
    answer_token_limit: int = 16,
) -> PreparedPair:
    """ペアをトークン化し、両 run の位置整列と読み出し対象を決める.

    Raises:
        PairPreparationError: 除外理由付き
    """
    pattern = trigger_pattern if trigger_pattern is not None else DEFAULT_TRIGGER_PATTERN

    trunc_clean = truncate_before_answer(pair.cot_clean, pair.benchmark, trigger_pattern)
    if not trunc_clean.trigger_found:
        raise PairPreparationError("no_trigger_clean")

    match_clean = re.search(pattern, pair.cot_clean)
    trigger_text = pair.cot_clean[match_clean.start() : match_clean.end()]
    clean_ans_cont = pair.cot_clean[match_clean.end() :]

    match_typo = re.search(pattern, pair.cot_typo)
    if match_typo:
        typo_ans_cont = pair.cot_typo[match_typo.end() :]
    else:
        typo_ans_cont = " " + (pair.answer_typo or "").strip()
    if not typo_ans_cont.strip():
        raise PairPreparationError("no_typo_answer")

    cells = build_cell_inputs(pair, trigger_pattern=trigger_pattern)
    prompts = {"clean": cells.prompts["A"], "pert": cells.prompts["C"]}
    cot_prefix = cells.forced_cots["A"]  # clean CoT の切断済み prefix (両 run 共通)

    prepared = PreparedPair()

    base_ids: dict[str, list[int]] = {}
    for run in RUNS:
        prompt_ids = _encode(tokenizer, prompts[run])
        prefix_ids = _encode(tokenizer, prompts[run] + cot_prefix)
        base_text = prompts[run] + cot_prefix + trigger_text
        ids = _encode(tokenizer, base_text)
        if ids[: len(prompt_ids)] != prompt_ids:
            raise PairPreparationError("prompt_boundary_mismatch")
        prepared.prompt_len[run] = len(prompt_ids)
        prepared.trigger_start[run] = len(prefix_ids)
        base_ids[run] = ids

    # 答え継続のトークン化 (clean run 基準) と分岐トークン対
    base_text_clean = prompts["clean"] + cot_prefix + trigger_text
    cont_clean = _continuation_ids(tokenizer, base_text_clean, base_ids["clean"], clean_ans_cont)
    cont_typo = _continuation_ids(tokenizer, base_text_clean, base_ids["clean"], typo_ans_cont)
    div = first_divergence(cont_clean, cont_typo, limit=answer_token_limit)
    if div is None:
        raise PairPreparationError("no_answer_divergence")
    prepared.readout_tokens = (div.token_a, div.token_b)
    prepared.readout_prefix_text = trigger_text + tokenizer.decode(div.common)

    for run in RUNS:
        prepared.input_ids[run] = base_ids[run] + div.common

    # suffix 整列 (プロンプト以降のトークン列が両 run で一致すること)
    suffix = {run: prepared.input_ids[run][prepared.prompt_len[run] :] for run in RUNS}
    if suffix["clean"] != suffix["pert"]:
        raise PairPreparationError("token_alignment_mismatch")
    prepared.suffix_len = len(suffix["clean"])

    # trigger 開始オフセット (suffix 内相対) の一致
    rel_trigger = {run: prepared.trigger_start[run] - prepared.prompt_len[run] for run in RUNS}
    if rel_trigger["clean"] != rel_trigger["pert"]:
        raise PairPreparationError("trigger_offset_mismatch")

    # 摂動語スパン (question_span): clean run は original_token、pert run は perturbed_token
    perturbed_tokens = pair.extra.get("perturbed_tokens", [])
    words_clean = [str(d.get("original_token", "")).strip() for d in perturbed_tokens]
    words_typo = [str(d.get("perturbed_token", "")).strip() for d in perturbed_tokens]
    if not words_clean:
        raise PairPreparationError("no_perturbed_tokens")

    span_char = {
        "clean": _locate_spans(prompts["clean"], pair.question_clean, words_clean),
        "pert": _locate_spans(prompts["pert"], pair.question_typo, words_typo),
    }
    enc = {
        run: tokenizer(prompts[run], return_offsets_mapping=True, add_special_tokens=True)
        for run in RUNS
    }
    span_pos: dict[str, list[int]] = {"clean": [], "pert": []}
    n_dropped = 0
    for i in range(len(words_clean)):
        tok_pos: dict[str, int | None] = {}
        for run in RUNS:
            span = span_char[run][i]
            if span is None:
                tok_pos[run] = None
                continue
            tok_pos[run] = span_end_token(enc[run]["offset_mapping"], span[0], span[1])
        if tok_pos["clean"] is None or tok_pos["pert"] is None:
            n_dropped += 1
            continue
        span_pos["clean"].append(tok_pos["clean"])
        span_pos["pert"].append(tok_pos["pert"])
    if not span_pos["clean"]:
        raise PairPreparationError("span_not_found")
    prepared.span_positions = span_pos

    prepared.meta = {
        "sample_id": pair.sample_id,
        "n_span_words": len(span_pos["clean"]),
        "n_span_dropped": n_dropped,
        "suffix_len": prepared.suffix_len,
        "prompt_len": dict(prepared.prompt_len),
        "trigger_start": dict(prepared.trigger_start),
        "readout_tokens": list(prepared.readout_tokens),
        "readout_common_len": len(div.common),
        "answer_clean": pair.answer_clean,
        "answer_typo": pair.answer_typo,
        "correct_answer": pair.correct_answer,
        "cell_exclude_reasons": list(cells.exclude_reasons),
    }
    return prepared


# ---------------------------------------------------------------------------
# 実行 (捕捉 + パッチ付き生成)
# ---------------------------------------------------------------------------


def _generate(model, input_ids: torch.Tensor, max_new_tokens: int, return_scores: bool = False):
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = model.generation_config.eos_token_id
        if isinstance(pad_id, (list, tuple)):
            pad_id = pad_id[0]
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            output_scores=return_scores,
            return_dict_in_generate=return_scores,
        )
    if return_scores:
        gen_ids = out.sequences[0, input_ids.shape[1] :].tolist()
        scores0 = out.scores[0][0].float().cpu()
        return gen_ids, scores0
    return out[0, input_ids.shape[1] :].tolist()


def _extract_answer(tokenizer, extractor, prepared: PreparedPair, gen_ids: list[int]) -> str:
    text = prepared.readout_prefix_text + tokenizer.decode(gen_ids, skip_special_tokens=True)
    return extractor.extract(text).extracted_answer


def _c1_logits(model, hidden_last: torch.Tensor) -> torch.Tensor:
    """最終層 residual (1 位置分) から c1 分布の logits を計算する."""
    decoder = model.get_decoder()
    device = next(model.parameters()).device
    with torch.no_grad():
        h = decoder.norm(hidden_last.to(device))
        return model.get_output_embeddings()(h).float().cpu()


def run_pair(model, tokenizer, layers, prepared: PreparedPair, extractor, args) -> dict:
    """1 ペアの全セル (部位×層窓×方向 × 位置種別) を実行する."""
    n_layers = len(layers)
    device = next(model.parameters()).device
    tok_a, tok_b = prepared.readout_tokens
    stride = args.window_stride or args.window_size
    windows = layer_windows(n_layers, args.window_size, stride)

    ids = {run: torch.tensor([prepared.input_ids[run]], device=device) for run in RUNS}
    capture_pos = {
        run: sorted(
            set(prepared.span_positions[run])
            | set(range(prepared.prompt_len[run], len(prepared.input_ids[run])))
        )
        for run in RUNS
    }

    # --- pass 1: 両 run の捕捉 + 無パッチ基準 -----------------------------
    caches = {}
    baseline: dict[str, dict] = {}
    for run in RUNS:
        caches[run] = capture_activations(model, ids[run], capture_pos[run], sites=args.sites)
        with torch.no_grad():
            logits = model(input_ids=ids[run]).logits
        # 必要行のみ CPU へ (Gemma3 は語彙 262k のため全系列の転送を避ける)
        c1 = logits[0, prepared.prompt_len[run] - 1].float().cpu()  # c1 の分布
        delta = float((logits[0, -1, tok_a] - logits[0, -1, tok_b]).item())
        del logits
        gen = _generate(model, ids[run], args.max_new_tokens)
        answer = _extract_answer(tokenizer, extractor, prepared, gen)
        baseline[run] = {
            "delta_logit": delta,
            "answer": answer,
            "generated_ids": gen,
            "c1_logits": c1,
        }

    kl_unpatched = {
        # 方向 clean→pert: donor=clean, recipient=pert (KL は donor ‖ recipient)
        "clean_to_pert": kl_from_logits(
            baseline["clean"]["c1_logits"], baseline["pert"]["c1_logits"]
        ),
        "pert_to_clean": kl_from_logits(
            baseline["pert"]["c1_logits"], baseline["clean"]["c1_logits"]
        ),
    }
    result: dict = {
        "baseline": {
            run: {"delta_logit": baseline[run]["delta_logit"], "answer": baseline[run]["answer"]}
            for run in RUNS
        }
    }
    result["baseline"]["s2_kl_unpatched"] = kl_unpatched

    def _cell_run(site: str, site_kind: str, window, direction: str, identity: bool = False):
        donor_run = "clean" if direction == "clean_to_pert" else "pert"
        recip_run = "pert" if direction == "clean_to_pert" else "clean"
        if identity:
            donor_run = recip_run
        src_pos = prepared.positions_for_site(donor_run, site_kind)
        dst_pos = prepared.positions_for_site(recip_run, site_kind)
        layer_indices = list(range(window[0], window[1]))
        values = {li: caches[donor_run].values(site, li, src_pos) for li in layer_indices}

        # S2 読み出し用: 最終層 residual をプロンプト最終位置で捕捉
        c1_capture: dict = {}
        c1_pos = prepared.prompt_len[recip_run] - 1

        def c1_hook(_m, _i, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if h.shape[1] > c1_pos:
                c1_capture["h"] = h[0, c1_pos, :].detach().to("cpu")
            return None

        handle = layers[n_layers - 1].register_forward_hook(c1_hook)
        try:
            with PatchInjector(layers, site, layer_indices, dst_pos, values):
                gen, scores0 = _generate(
                    model, ids[recip_run], args.max_new_tokens, return_scores=True
                )
        finally:
            handle.remove()

        delta = float(scores0[tok_a] - scores0[tok_b])
        answer = _extract_answer(tokenizer, extractor, prepared, gen)
        cell: dict = {
            "site": site,
            "site_kind": site_kind,
            "window": list(window),
            "direction": direction,
            "delta_logit": delta,
            "answer": answer,
            "answer_matches_donor": answer == baseline[donor_run]["answer"],
            "answer_matches_recipient": answer == baseline[recip_run]["answer"],
            "generation_identical_to_recipient": gen == baseline[recip_run]["generated_ids"],
        }
        if not identity:
            d_donor = baseline[donor_run]["delta_logit"]
            d_recip = baseline[recip_run]["delta_logit"]
            gap = d_donor - d_recip
            cell["recovery"] = (delta - d_recip) / gap if abs(gap) > 1e-3 else None
        # S2: 質問スパンへのパッチのみ c1 分布が動き得る (それ以外は c1 の上流に無い)
        if site_kind == "question_span" and "h" in c1_capture:
            c1_patched = _c1_logits(model, c1_capture["h"])
            kl_patched = kl_from_logits(baseline[donor_run]["c1_logits"], c1_patched)
            cell["s2_kl_patched"] = kl_patched
            if not identity:
                kl_base = kl_unpatched[direction]
                if kl_base > 1e-9:
                    cell["s2_kl_recovery"] = 1.0 - kl_patched / kl_base
        return cell

    # --- no-op 検証 (恒等パッチ: donor = recipient 自身) -------------------
    if args.noop_check:
        mid = windows[len(windows) // 2]
        noop_cells = []
        for site in args.sites:
            for direction in args.directions:
                cell = _cell_run(site, "cot_suffix", mid, direction, identity=True)
                noop_cells.append(
                    {
                        "site": site,
                        "direction": direction,
                        "window": list(mid),
                        "generation_unchanged": cell["generation_identical_to_recipient"],
                        "answer_unchanged": cell["answer_matches_recipient"],
                    }
                )
        result["noop"] = noop_cells

    # --- 本スイープ: {部位 × 層窓 × 方向} × 位置種別 -----------------------
    cells_out = []
    for site_kind in args.site_kinds:
        for spec in iter_patch_cells(
            n_layers,
            args.window_size,
            stride,
            sites=args.sites,
            directions=args.directions,
        ):
            cells_out.append(_cell_run(spec.site, site_kind, spec.window, spec.direction))
    result["cells"] = cells_out
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chash = config_hash(args)

    # flip ペアのロード (摂動条件ごと)
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
    logger.info("デコーダ層 %d 層を検出", len(layers))
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
            "timestamp": datetime.now().isoformat(),
        }
        try:
            prepared = prepare_pair(pair, tokenizer, args.trigger_pattern, args.answer_token_limit)
        except PairPreparationError as e:
            payload["excluded"] = str(e)
            excluded[str(e)] = excluded.get(str(e), 0) + 1
            n_excluded += 1
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            continue

        payload["prepared"] = prepared.meta
        try:
            result = run_pair(model, tokenizer, layers, prepared, extractor, args)
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
            "baseline_dir": str(args.baseline_dir),
            "perturbed_dir_lxt": str(args.perturbed_dir_lxt),
            "perturbed_dir_rnd": str(args.perturbed_dir_rnd),
            "n_pairs": args.n_pairs,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "sites": args.sites,
            "directions": args.directions,
            "site_kinds": args.site_kinds,
            "max_new_tokens": args.max_new_tokens,
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
