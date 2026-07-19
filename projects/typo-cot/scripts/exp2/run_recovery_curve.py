#!/usr/bin/env python3
"""実験2 副実験: 回復曲線 CLI — セルC構成の p% prefix 強制と並べ替え検定.

flip 事例 (typo で答えが変わったサンプル) について、typo 質問プロンプトの下で
clean CoT の先頭 p% (p∈{0,25,50,75,100}) を強制し、続きを自由生成 → clean 答えに
回復するかを測る。回復ジャンプ位置と最上位 R_C 内容語の初出位置の一致率を
permutation 検定する (計画 §4 実験2-2-4)。

出力:
  {output_dir}/{model_short}_{benchmark}_{run_label}/
  ├── config.json / results.json (サンプル別回復曲線) / summary.json
    (p別回復率 + ジャンプ一致の並べ替え検定)

使用例 (GPU ヘルパー経由):
  bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/exp2/run_recovery_curve.py \\
    --baseline_dir <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \\
    --perturbed_dir <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \\
    --model google/gemma-3-4b-it --benchmark gsm8k --n 16
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("exp2_recovery_curve")
logger.setLevel(logging.INFO)


def resolve_gpu_id(cli_gpu_id: str, env: dict) -> str:
    env_cvd = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    return env_cvd if env_cvd else cli_gpu_id


def shard_entries(entries: list[dict], start: int | None, end: int | None) -> list[dict]:
    s = 0 if start is None else start
    e = len(entries) if end is None else end
    return entries[s:e]


def filter_pending(entries: list[dict], existing_records: list[dict]) -> list[dict]:
    done = {r["sample_id"] for r in existing_records}
    return [e for e in entries if e["sample_id"] not in done]


def save_results_atomic(out_dir: Path, records: list[dict]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "results.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    tmp.replace(out_dir / "results.json")


def assemble_case(
    cot_text: str,
    recovered_by_p: dict[int, bool],
    rc_ranking: list[dict] | None,
) -> dict:
    """1サンプルの回復曲線から並べ替え検定用の case を組み立てる.

    Returns:
        {"interval": (p_prev, p_star) | None,
         "target_word": 最上位 R_C content 語 | None,
         "target_frac": その初出位置比 | None,
         "candidate_fracs": content 候補語の初出位置比リスト (帰無分布の抽選元)}
    """
    from typo_cot.intervention.recovery_curve import find_jump, target_first_fraction
    from typo_cot.intervention.target_selector import build_candidates, select_top

    interval = find_jump(recovered_by_p)
    candidates = build_candidates(cot_text)
    content = [c for c in candidates if c.stratum == "content"]

    target_word = None
    target_frac = None
    if rc_ranking:
        top = select_top(rc_ranking, candidates, k=1, stratum="content")
        if top:
            target_word = top[0]
            target_frac = target_first_fraction(cot_text, target_word)
    candidate_fracs = (
        [c.first_char_pos / len(cot_text) for c in content] if cot_text else []
    )
    return {
        "interval": interval,
        "target_word": target_word,
        "target_frac": target_frac,
        "candidate_fracs": candidate_fracs,
    }


def build_prompt(template, benchmark: str, entry: dict) -> str:
    """run_inference / run_loo_scoring と同一のプロンプト再構築."""
    if benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
        prompt_result = template.generate(
            question=entry["question"],
            choices=entry.get("choices"),
            subject=entry.get("subset"),
        )
    elif benchmark == "gsm8k":
        prompt_result = template.generate(question=entry["question"])
    elif benchmark in ["bbh", "math", "strategy_qa"]:
        prompt_result = template.generate(
            question=entry["question"], subject=entry.get("subset")
        )
    else:
        prompt_result = template.generate(question=entry["question"])
    return prompt_result.get_full_prompt()


def main() -> None:
    parser = argparse.ArgumentParser(description="exp2: recovery curve (cell-C prefix forcing)")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--perturbed_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--n", type=int, default=None, help="flip 事例数の上限")
    parser.add_argument("--all_pairs", action="store_true",
                        help="flip 事例に限定しない (既定は clean正解∩flip のみ)")
    parser.add_argument("--rc_source", type=str, default="cot_pt",
                        choices=["cot_pt", "results"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/exp2")
    parser.add_argument("--run_label", type=str, default="recovery_curve")
    args = parser.parse_args()

    from typo_cot.evaluation.extractor import create_extractor
    from typo_cot.intervention.loo_scorer import split_generated_text
    from typo_cot.intervention.recovery_curve import (
        GRID,
        cut_prefix_by_fraction,
        match_rate_permutation_test,
    )

    baseline_dir = Path(args.baseline_dir)
    perturbed_dir = Path(args.perturbed_dir)
    with open(baseline_dir / "results.json", encoding="utf-8") as f:
        base_entries = {e["sample_id"]: e for e in json.load(f)}
    with open(perturbed_dir / "results.json", encoding="utf-8") as f:
        pert_entries = json.load(f)

    # 対象: clean 正解 ∩ flip (typo で答えが変わった事例)。--all_pairs で全対に拡張
    joined = []
    for pert in pert_entries:
        base = base_entries.get(pert["sample_id"])
        if base is None:
            continue
        flip = str(pert.get("extracted_answer") or "").strip() != str(
            base.get("extracted_answer") or ""
        ).strip()
        if args.all_pairs or (base.get("is_correct") and flip):
            joined.append({"sample_id": base["sample_id"], "base": base, "pert": pert})
    pairs = shard_entries(joined, args.start, args.end)
    if args.n is not None:
        pairs = pairs[: args.n]

    model_short = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) / f"{model_short}_{args.benchmark}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if (out_dir / "results.json").exists():
        with open(out_dir / "results.json", encoding="utf-8") as f:
            existing = json.load(f)
    pending = filter_pending(pairs, existing)

    config = {
        "experiment": "exp2_recovery_curve",
        "model": args.model,
        "benchmark": args.benchmark,
        "baseline_dir": str(baseline_dir),
        "perturbed_dir": str(perturbed_dir),
        "grid": list(GRID),
        "n": args.n,
        "start": args.start,
        "end": args.end,
        "all_pairs": args.all_pairs,
        "rc_source": args.rc_source,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decoding": "greedy",
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    from typo_cot.models.prompts import create_prompt_template

    template = create_prompt_template(args.benchmark)

    # load_rc_ranking は run_target_deletion.py と同一規約 (スクリプト間 import 不可のため再掲)
    def load_rc(sample_id: str, entry: dict) -> list[dict] | None:
        if args.rc_source == "results":
            return entry.get("cot_top_k_words")
        import torch

        from typo_cot.intervention.loo_scorer import rc_word_ranking_from_cot_pt

        path = baseline_dir / "importance_scores" / f"{sample_id}_cot.pt"
        if not path.exists():
            return None
        # word_scores 結合不良 (Mistral アーカイブ) は token_scores から再構築
        full_text = build_prompt(template, args.benchmark, entry) + entry.get(
            "generated_text", ""
        )
        return rc_word_ranking_from_cot_pt(
            torch.load(path, map_location="cpu", weights_only=False),
            full_text=full_text,
        )

    records = list(existing)
    if pending:
        from typo_cot.models.wrapper import create_model_wrapper

        gpu_id = resolve_gpu_id(args.gpu_id, os.environ)
        logger.info(f"モデルをロード: {args.model} (GPU {gpu_id})")
        wrapper = create_model_wrapper(
            model_name=args.model, gpu_id=gpu_id, wrap_for_lxt=False
        )
        extractor = create_extractor(args.benchmark)

        t0 = time.time()
        for i in range(0, len(pending), args.save_every):
            chunk = pending[i : i + args.save_every]
            tasks: list[tuple[int, int, str, str]] = []  # (chunk_idx, p, prefix, ctx)
            metas: list[dict] = []
            for j, pair in enumerate(chunk):
                base, pert = pair["base"], pair["pert"]
                split = split_generated_text(base["generated_text"])
                if split is None:
                    metas.append({"skip_reason": "no_answer_pattern"})
                    continue
                prompt_typo = build_prompt(template, args.benchmark, pert)
                metas.append(
                    {
                        "skip_reason": None,
                        "split": split,
                        "clean_answer": str(base.get("extracted_answer") or "").strip(),
                    }
                )
                for p in GRID:
                    prefix = cut_prefix_by_fraction(split.cot_text, p)
                    tasks.append((j, p, prefix, prompt_typo + prefix))

            continuations: dict[tuple[int, int], tuple[str, str]] = {}
            for b in range(0, len(tasks), args.batch_size):
                batch = tasks[b : b + args.batch_size]
                results = wrapper.generate_batch(
                    [ctx for _, _, _, ctx in batch],
                    max_new_tokens=args.max_new_tokens,
                )
                for (j, p, prefix, _), res in zip(batch, results, strict=True):
                    continuations[(j, p)] = (prefix, res.generated_text)

            for j, pair in enumerate(chunk):
                meta = metas[j]
                rec: dict = {
                    "sample_id": pair["sample_id"],
                    "skip_reason": meta.get("skip_reason"),
                }
                if meta.get("skip_reason") is None:
                    split = meta["split"]
                    recovered: dict[int, bool] = {}
                    curve: dict[str, dict] = {}
                    for p in GRID:
                        prefix, cont = continuations[(j, p)]
                        answer = extractor.extract(prefix + cont).extracted_answer.strip()
                        recovered[p] = answer == meta["clean_answer"]
                        curve[str(p)] = {"answer": answer, "recovered": recovered[p]}
                    case = assemble_case(
                        split.cot_text, recovered, load_rc(pair["sample_id"], pair["base"])
                    )
                    rec.update(
                        clean_answer=meta["clean_answer"],
                        curve=curve,
                        interval=list(case["interval"]) if case["interval"] else None,
                        target_word=case["target_word"],
                        target_frac=case["target_frac"],
                        candidate_fracs=case["candidate_fracs"],
                    )
                records.append(rec)
            save_results_atomic(out_dir, records)
            done = min(i + args.save_every, len(pending))
            logger.info(
                f"{done}/{len(pending)} 事例処理済 "
                f"({(time.time() - t0) / max(1, done):.1f}s/case)"
            )

    save_results_atomic(out_dir, records)

    valid = [r for r in records if r.get("skip_reason") is None]
    recovery_rates = {
        str(p): (
            sum(1 for r in valid if r["curve"][str(p)]["recovered"]) / len(valid)
            if valid
            else None
        )
        for p in (0, 25, 50, 75, 100)
    }
    cases = [
        {
            "interval": tuple(r["interval"]) if r.get("interval") else None,
            "target_frac": r.get("target_frac"),
            "candidate_fracs": r.get("candidate_fracs", []),
        }
        for r in valid
    ]
    perm = match_rate_permutation_test(cases, n_perm=args.n_perm, seed=args.seed)
    summary = {
        "experiment_info": {
            "experiment": "exp2_recovery_curve",
            "model": args.model,
            "benchmark": args.benchmark,
            "timestamp": datetime.now().isoformat(),
        },
        "n_records": len(records),
        "n_valid": len(valid),
        "recovery_rates": recovery_rates,
        "jump_match_permutation": perm,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"完了: {out_dir}")
    logger.info(json.dumps(summary["recovery_rates"], ensure_ascii=False))
    logger.info(json.dumps(perm, ensure_ascii=False))


if __name__ == "__main__":
    main()
