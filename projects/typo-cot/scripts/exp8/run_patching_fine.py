#!/usr/bin/env python3
"""実験8-fine (1層分解 activation patching) 実行スクリプト.

粗い窓 (幅3) で確定した最良窓 residual[0,6) を 1 層解像度に精密化する。
部位は residual のみ、位置種別は摂動語スパン (question_span) のみに固定し、
以下の 4 系統のセルを各 flip ペアで測定する:

    single    … 単層 denoising (clean→pert): 各層の限界寄与
    cumulative… 累積 denoising (0..l 全差替): ここまで直せば何%戻るか
    noising   … 単層 noising  (pert→clean): 最良層近傍の十分性 (H8f-5)
    sham      … 単層 sham (recipient 自身の値を書き戻す): 効果ゼロのはず (統制1)

主指標 = S2 KL 回復率 (最初の CoT 語分布, s2_kl_recovery)。
副指標 = 分岐ペアの flip 逆転 (answer_matches_donor; MMLU のみ意味を持つ)。

粗い run_patching.py のペア準備・生成ユーティリティを再利用する
(prepare_pair / _generate / _extract_answer / _c1_logits)。

GPU 実行は必ず run_with_gpu.sh 経由 (CUDA_VISIBLE_DEVICES はヘルパーが設定)。

例 (スモーク):
    bash <...>/run_with_gpu.sh uv run --package typo-cot python \
        scripts/exp8/run_patching_fine.py \
        --model google/gemma-3-4b-it --benchmark gsm8k \
        --baseline-dir <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \
        --perturbed-dir-lxt <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
        --perturbed-dir-rnd <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_random \
        --n-pairs 16 --output-dir results/exp8_fine/smoke_gemma3-4b_gsm8k
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# 粗いランナーのペア準備・生成ユーティリティを再利用
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patching import (  # noqa: E402
    RUNS,
    PairPreparationError,
    PreparedPair,
    _c1_logits,
    _extract_answer,
    _generate,
    prepare_pair,
)

from typo_cot.evaluation.extractor import create_extractor  # noqa: E402
from typo_cot.intervention.archive_loader import load_pair_records  # noqa: E402
from typo_cot.intervention.patching import (  # noqa: E402
    PatchInjector,
    capture_activations,
    cumulative_windows,
    find_decoder_layers,
    kl_from_logits,
    result_is_current,
    select_flip_pairs,
    shard_slice,
    single_layer_windows,
)
from typo_cot.intervention.records import PairRecord  # noqa: E402
from typo_cot.models.wrapper import ModelWrapper  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_patching_fine")

SITE = "residual"  # 粗い窓で最良確定済み
SITE_KIND = "question_span"  # 摂動語スパンのみ


def _int_list(spec: str) -> list[int]:
    """"0-11" / "0-11,14,20,26" / "14 20 26" 形式を層 index リストに展開."""
    out: list[int] = []
    for chunk in spec.replace(" ", ",").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    # 重複除去 (順序保持)
    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--perturbed-dir-lxt", required=True)
    p.add_argument("--perturbed-dir-rnd", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-pairs", type=int, default=150, help="総ペア数 (LXT/Random 半々, 0=全 flip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single-layers", default="0-11,14,20,26", help="単層 denoising スイープ層")
    p.add_argument("--cumulative-layers", default="0-11", help="累積 denoising 終端層")
    p.add_argument("--noising-layers", default="0-7", help="単層 noising 層 (最良層±1 を含む早期帯)")
    p.add_argument("--sham-layers", default="0-11", help="単層 sham 層 (アーチファクト検出)")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--answer-token-limit", type=int, default=16)
    p.add_argument("--trigger-pattern", default=None)
    # A3 敵対的レビュー統制
    p.add_argument(
        "--no-controls", dest="controls", action="store_false",
        help="A3 統制 (other_span / all_positions) を無効化",
    )
    p.add_argument("--other-span-offset", type=int, default=2, help="other_span 統制の下流オフセット")
    p.add_argument(
        "--perturb-mode", default="typo", choices=["typo", "semantic"],
        help="typo=既存の摂動 / semantic=標的語を実語ランダム置換 (統制c)",
    )
    p.add_argument("--semantic-seed", type=int, default=1234, help="意味置換の乱数シード")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def config_hash(args: argparse.Namespace) -> str:
    payload = {
        "experiment": "exp8_fine",
        "model": args.model,
        "benchmark": args.benchmark,
        "single_layers": _int_list(args.single_layers),
        "cumulative_layers": _int_list(args.cumulative_layers),
        "noising_layers": _int_list(args.noising_layers),
        "sham_layers": _int_list(args.sham_layers),
        "max_new_tokens": args.max_new_tokens,
        "answer_token_limit": args.answer_token_limit,
        "trigger_pattern": args.trigger_pattern,
        "seed": args.seed,
        "site": SITE,
        "site_kind": SITE_KIND,
        "controls": bool(getattr(args, "controls", True)),
        "other_span_offset": args.other_span_offset,
        "perturb_mode": args.perturb_mode,
        "semantic_seed": args.semantic_seed if args.perturb_mode == "semantic" else None,
        "schema": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1 ペアの実行 (residual × question_span に固定した精密スイープ)
# ---------------------------------------------------------------------------


def run_pair_fine(model, tokenizer, layers, prepared: PreparedPair, extractor, args) -> dict:
    n_layers = len(layers)
    device = next(model.parameters()).device
    tok_a, tok_b = prepared.readout_tokens

    single_layers = [li for li in _int_list(args.single_layers) if li < n_layers]
    cumul_layers = [li for li in _int_list(args.cumulative_layers) if li < n_layers]
    noising_layers = [li for li in _int_list(args.noising_layers) if li < n_layers]
    sham_layers = [li for li in _int_list(args.sham_layers) if li < n_layers]

    ids = {run: torch.tensor([prepared.input_ids[run]], device=device) for run in RUNS}
    # 捕捉位置: 全位置 (A3 統制 other_span / all_positions で任意位置の donor 値が要る)。
    # residual 1 部位のみなので系列全域でもメモリは軽い。
    capture_pos = {run: list(range(len(prepared.input_ids[run]))) for run in RUNS}

    # --- pass 1: 両 run の捕捉 + 無パッチ基準 ---------------------------------
    caches = {}
    baseline: dict[str, dict] = {}
    for run in RUNS:
        caches[run] = capture_activations(model, ids[run], capture_pos[run], sites=(SITE,))
        with torch.no_grad():
            logits = model(input_ids=ids[run]).logits
        c1 = logits[0, prepared.prompt_len[run] - 1].float().cpu()
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
        "clean_to_pert": kl_from_logits(
            baseline["clean"]["c1_logits"], baseline["pert"]["c1_logits"]
        ),
        "pert_to_clean": kl_from_logits(
            baseline["pert"]["c1_logits"], baseline["clean"]["c1_logits"]
        ),
    }

    def run_cell(
        window,
        direction: str,
        kind: str,
        identity: bool = False,
        src_pos: list[int] | None = None,
        dst_pos: list[int] | None = None,
    ) -> dict:
        # target_run = 回復を測る基準 (denoising は clean, noising は pert)。
        # value_run   = パッチ値の供給元 (sham は recipient 自身)。
        target_run = "clean" if direction == "clean_to_pert" else "pert"
        recip_run = "pert" if direction == "clean_to_pert" else "clean"
        value_run = recip_run if identity else target_run
        if src_pos is None:
            src_pos = prepared.span_positions[value_run]
        if dst_pos is None:
            dst_pos = prepared.span_positions[recip_run]
        layer_indices = list(range(window[0], window[1]))
        values = {li: caches[value_run].values(SITE, li, src_pos) for li in layer_indices}

        c1_capture: dict = {}
        c1_pos = prepared.prompt_len[recip_run] - 1

        def c1_hook(_m, _i, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if h.shape[1] > c1_pos:
                c1_capture["h"] = h[0, c1_pos, :].detach().to("cpu")
            return None

        handle = layers[n_layers - 1].register_forward_hook(c1_hook)
        try:
            with PatchInjector(layers, SITE, layer_indices, dst_pos, values):
                gen, scores0 = _generate(
                    model, ids[recip_run], args.max_new_tokens, return_scores=True
                )
        finally:
            handle.remove()

        delta = float(scores0[tok_a] - scores0[tok_b])
        answer = _extract_answer(tokenizer, extractor, prepared, gen)
        rep_layer = window[1] - 1 if kind == "cumulative" else window[0]
        cell: dict = {
            "kind": kind,
            "window": list(window),
            "layer": rep_layer,
            "direction": direction,
            "delta_logit": delta,
            "answer": answer,
            # 回復 = 答えが target (denoising:clean / noising:pert) に一致すること
            "answer_matches_donor": answer == baseline[target_run]["answer"],
            "answer_matches_recipient": answer == baseline[recip_run]["answer"],
            "generation_identical_to_recipient": gen == baseline[recip_run]["generated_ids"],
        }
        d_target = baseline[target_run]["delta_logit"]
        d_recip = baseline[recip_run]["delta_logit"]
        gap = d_target - d_recip
        cell["recovery"] = (delta - d_recip) / gap if abs(gap) > 1e-3 else None
        # S2 KL: 基準は常に target 分布 (sham でも clean を基準にするため回復≈0 になる)
        if "h" in c1_capture:
            c1_patched = _c1_logits(model, c1_capture["h"])
            kl_patched = kl_from_logits(baseline[target_run]["c1_logits"], c1_patched)
            cell["s2_kl_patched"] = kl_patched
            kl_base = kl_unpatched[direction]
            if kl_base > 1e-9:
                cell["s2_kl_recovery"] = 1.0 - kl_patched / kl_base
        return cell

    cells: list[dict] = []
    semantic_mode = getattr(args, "perturb_mode", "typo") == "semantic"
    if semantic_mode:
        # A3 統制(c): 意味置換ペアは単層 denoising のみ (typo プロファイルとの比較用)
        for w in single_layer_windows(single_layers):
            cells.append(run_cell(w, "clean_to_pert", "semantic"))
        return {
            "n_layers": n_layers,
            "baseline": {
                run: {
                    "delta_logit": baseline[run]["delta_logit"],
                    "answer": baseline[run]["answer"],
                }
                for run in RUNS
            },
            "s2_kl_unpatched": kl_unpatched,
            "cells": cells,
        }

    # 単層 denoising (主)
    for w in single_layer_windows(single_layers):
        cells.append(run_cell(w, "clean_to_pert", "single"))
    # 累積 denoising
    for w in cumulative_windows(cumul_layers):
        cells.append(run_cell(w, "clean_to_pert", "cumulative"))
    # 単層 noising (十分性)
    for w in single_layer_windows(noising_layers):
        cells.append(run_cell(w, "pert_to_clean", "noising"))
    # 単層 sham (統制1: recipient=pert 自身の値)
    for w in single_layer_windows(sham_layers):
        cells.append(run_cell(w, "clean_to_pert", "sham_single", identity=True))

    # --- A3 敵対的レビュー統制 -------------------------------------------------
    if getattr(args, "controls", True):
        off = getattr(args, "other_span_offset", 2)
        # (a) other_span: 無摂動語 (スパンから +off の下流位置) を clean 値で patch。
        #     donor/recipient を同一 offset で対応付け、両方が有効な対のみ使う。
        src_o: list[int] = []
        dst_o: list[int] = []
        span_c = prepared.span_positions["clean"]
        span_p = prepared.span_positions["pert"]
        span_c_set, span_p_set = set(span_c), set(span_p)
        for sc, sp in zip(span_c, span_p):
            cc, cp = sc + off, sp + off
            if (
                1 <= cc < prepared.prompt_len["clean"] - 1
                and 1 <= cp < prepared.prompt_len["pert"] - 1
                and cc not in span_c_set
                and cp not in span_p_set
            ):
                src_o.append(cc)
                dst_o.append(cp)
        if src_o:
            for w in single_layer_windows(single_layers):
                cells.append(
                    run_cell(w, "clean_to_pert", "other_span", src_pos=src_o, dst_pos=dst_o)
                )

        # (b) all_positions: 全プロンプト位置を clean 値で patch (枠組みサニティ; 任意層で
        #     完全回復のはず)。トークン数一致ペアのみ (1対1 対応が取れる)。
        len_c = len(prepared.input_ids["clean"])
        len_p = len(prepared.input_ids["pert"])
        if len_c == len_p and prepared.prompt_len["clean"] == prepared.prompt_len["pert"]:
            all_pos = list(range(prepared.prompt_len["pert"]))
            sanity_layers = [li for li in single_layers if li <= 11][::3] or single_layers[:1]
            for w in single_layer_windows(sanity_layers):
                cells.append(
                    run_cell(
                        w, "clean_to_pert", "all_positions", src_pos=all_pos, dst_pos=all_pos
                    )
                )

    return {
        "n_layers": n_layers,
        "baseline": {
            run: {
                "delta_logit": baseline[run]["delta_logit"],
                "answer": baseline[run]["answer"],
            }
            for run in RUNS
        },
        "s2_kl_unpatched": kl_unpatched,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chash = config_hash(args)

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
        if args.perturb_mode == "semantic":
            # A3 統制(c): 同じ flip ペアの標的語を実語ランダム置換に差し替える
            from typo_cot.intervention.semantic_control import make_semantic_pair

            sem = [make_semantic_pair(p, seed=args.semantic_seed) for p in flips]
            flips = [p for p in sem if p is not None]
            logger.info("%s: semantic 置換 flip %d 件", cond, len(flips))
        else:
            logger.info("%s: flip %d 件を選定", cond, len(flips))
        tasks.extend((cond, p) for p in flips)

    tasks.sort(key=lambda t: (t[0], t[1].sample_id))
    tasks = shard_slice(tasks, args.shard_index, args.num_shards)
    logger.info("シャード %d/%d: %d タスク", args.shard_index, args.num_shards, len(tasks))

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
            result = run_pair_fine(model, tokenizer, layers, prepared, extractor, args)
        except Exception as e:  # noqa: BLE001
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
        "experiment": "exp8_fine",
        "config": {
            "model": args.model,
            "benchmark": args.benchmark,
            "baseline_dir": str(args.baseline_dir),
            "perturbed_dir_lxt": str(args.perturbed_dir_lxt),
            "perturbed_dir_rnd": str(args.perturbed_dir_rnd),
            "n_pairs": args.n_pairs,
            "single_layers": _int_list(args.single_layers),
            "cumulative_layers": _int_list(args.cumulative_layers),
            "noising_layers": _int_list(args.noising_layers),
            "sham_layers": _int_list(args.sham_layers),
            "site": SITE,
            "site_kind": SITE_KIND,
            "controls": bool(getattr(args, "controls", True)),
            "other_span_offset": args.other_span_offset,
            "perturb_mode": args.perturb_mode,
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
    with open(out_dir / f"run_summary_fine{shard_tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("完了: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
