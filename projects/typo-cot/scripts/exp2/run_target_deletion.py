#!/usr/bin/env python3
"""実験2: R_C 標的削除の要因計画 — teacher-forcing 短生成 CLI.

アーカイブの baseline run ディレクトリ (results.json / importance_scores/*.pt、
読み取り専用) を入力に、腕プリセット (core / smoke / full) の各セルについて
clean CoT を編集 → Q_clean の下で teacher-forcing → 答えスパンを短生成 →
基準腕との flip を判定する。

- シャード: --start/--end (baseline results.json の行インデックス)
- 進捗保存: --save_every サンプルごとに results.json を原子的に置換
- 冪等: 既存 results.json の sample_id はスキップ (resume)

出力:
  {output_dir}/{model_short}_{benchmark}_{run_label}/
  ├── config.json    # 実行設定
  ├── results.json   # サンプル×腕の record (スキーマ: dev notes §4)
  └── summary.json   # 腕別集計 (content/numeric 層分離) + コア対比 + 用量反応

使用例 (GPU ヘルパー経由):
  bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/exp2/run_target_deletion.py \\
    --baseline_dir <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \\
    --model google/gemma-3-4b-it --benchmark gsm8k --arms smoke \\
    --clean_correct_only --n 24 --output_dir results/exp2_smoke
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
logger = logging.getLogger("exp2_target_deletion")
logger.setLevel(logging.INFO)


# ============================================================
# 純粋ヘルパー (tests/test_exp2_cli.py で検証)
# ============================================================


def resolve_gpu_id(cli_gpu_id: str, env: dict) -> str:
    """run_with_gpu.sh 等の外部 CUDA_VISIBLE_DEVICES を優先して GPU ID を解決."""
    env_cvd = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    return env_cvd if env_cvd else cli_gpu_id


def shard_entries(entries: list[dict], start: int | None, end: int | None) -> list[dict]:
    """--start/--end による行インデックスシャード."""
    s = 0 if start is None else start
    e = len(entries) if end is None else end
    return entries[s:e]


def resolve_arms(preset: str):
    """腕プリセット名 → ArmSpec リスト."""
    from typo_cot.intervention.deletion_runner import core_arms, full_grid_arms, smoke_arms

    presets = {"core": core_arms, "smoke": smoke_arms, "full": full_grid_arms}
    if preset not in presets:
        raise ValueError(f"unknown arms preset: {preset!r} (expected {sorted(presets)})")
    return presets[preset]()


def filter_pending(entries: list[dict], existing_records: list[dict]) -> list[dict]:
    """既存 results.json に記録済みの sample_id を除外する (冪等 resume)."""
    done = {r["sample_id"] for r in existing_records}
    return [e for e in entries if e["sample_id"] not in done]


def load_rc_ranking(
    run_dir: Path, sample_id: str, entry: dict, rc_source: str
) -> list[dict] | None:
    """R_C ランキングをロードする.

    rc_source:
        "cot_pt": importance_scores/{sid}_cot.pt の word_scores (CoT 領域、完全ランキング)
        "results": results.json の cot_top_k_words (上位のみ)
    本番は実験4の fixed-target 版ランキング (上流依存) — ディレクトリ差し替えで対応。
    """
    if rc_source == "results":
        return entry.get("cot_top_k_words")
    if rc_source == "cot_pt":
        import torch

        from typo_cot.intervention.loo_scorer import rc_word_ranking_from_cot_pt

        path = Path(run_dir) / "importance_scores" / f"{sample_id}_cot.pt"
        if not path.exists():
            return None
        data = torch.load(path, map_location="cpu", weights_only=False)
        return rc_word_ranking_from_cot_pt(data)
    raise ValueError(f"unknown rc_source: {rc_source!r}")


def load_loo_rankings(path: str | Path) -> dict[str, list[dict]]:
    """exp/06 run_loo_scoring の results.json から sample_id → LOO ランキングを作る."""
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["sample_id"]: e.get("loo_word_scores", []) for e in entries}


def save_results_atomic(out_dir: Path, records: list[dict]) -> None:
    """results.json を tmp 経由で原子的に置換する (進捗保存)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "results.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    tmp.replace(out_dir / "results.json")


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


def build_dose_trends(records: list[dict], n_perm: int, seed: int) -> list[dict]:
    """同一 (target_kind, op, stratum) で用量が2水準以上ある系列の単調性検定."""
    from typo_cot.intervention.deletion_stats import _arm_flip_maps, dose_trend_test

    arm_meta: dict[str, dict] = {}
    for rec in records:
        for name, arm in rec.get("arms", {}).items():
            arm_meta.setdefault(
                name,
                {
                    "target_kind": arm["target_kind"],
                    "op": arm["op"],
                    "k": arm["k"],
                    "stratum": arm["stratum"],
                },
            )
    series: dict[tuple, dict[int, str]] = {}
    for name, meta in arm_meta.items():
        key = (meta["target_kind"], meta["op"], meta["stratum"])
        series.setdefault(key, {})[meta["k"]] = name
    trends = []
    for (kind, op, stratum), by_k in sorted(series.items(), key=str):
        if len(by_k) < 2:
            continue
        flips_by_dose = {}
        for k, name in by_k.items():
            _, cc = _arm_flip_maps(records, name)
            if cc:
                flips_by_dose[k] = cc
        if len(flips_by_dose) < 2:
            continue
        res = dose_trend_test(flips_by_dose, n_perm=n_perm, seed=seed)
        trends.append({"target_kind": kind, "op": op, "stratum": stratum, **res})
    return trends


# ============================================================
# main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="exp2: targeted CoT deletion factorial")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="アーカイブ baseline run ディレクトリ (読み取り専用)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--arms", type=str, default="core",
                        choices=["core", "smoke", "full"])
    parser.add_argument("--start", type=int, default=None, help="シャード開始行")
    parser.add_argument("--end", type=int, default=None, help="シャード終了行 (排他)")
    parser.add_argument("--n", type=int, default=None,
                        help="処理するサンプル数上限 (フィルタ後)")
    parser.add_argument("--clean_correct_only", action="store_true",
                        help="clean 正解サンプルのみ処理 (主推定量の対象)")
    parser.add_argument("--rc_source", type=str, default="cot_pt",
                        choices=["cot_pt", "results"])
    parser.add_argument("--loo_results", type=str, default=None,
                        help="exp6 run_loo_scoring の results.json (top_loo 腕の供給源)")
    parser.add_argument("--loo_inline", action="store_true",
                        help="LOO ランキングをその場で計算 (高コスト、スモーク用)")
    parser.add_argument("--loo_deletion_mode", type=str, default="occurrence",
                        choices=["occurrence", "type"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=8,
                        help="このサンプル数ごとに results.json を原子的保存")
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="results/exp2")
    parser.add_argument("--run_label", type=str, default="target_deletion")
    args = parser.parse_args()

    run_dir = Path(args.baseline_dir)
    with open(run_dir / "results.json", encoding="utf-8") as f:
        entries = json.load(f)
    entries = shard_entries(entries, args.start, args.end)
    if args.clean_correct_only:
        entries = [e for e in entries if e.get("is_correct")]
    if args.n is not None:
        entries = entries[: args.n]

    model_short = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) / f"{model_short}_{args.benchmark}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume (冪等): 既存レコードを読み込み、済み sample_id をスキップ
    existing: list[dict] = []
    if (out_dir / "results.json").exists():
        with open(out_dir / "results.json", encoding="utf-8") as f:
            existing = json.load(f)
        logger.info(f"resume: 既存 {len(existing)} 件をスキップ対象にする")
    pending = filter_pending(entries, existing)

    arms = resolve_arms(args.arms)
    need_loo = any(a.target_kind == "top_loo" for a in arms)
    need_replace = any(a.op == "replace" for a in arms)

    config = {
        "experiment": "exp2_target_deletion",
        "model": args.model,
        "benchmark": args.benchmark,
        "baseline_dir": str(run_dir),
        "arms_preset": args.arms,
        "arms": [
            {
                "name": a.name,
                "target_kind": a.target_kind,
                "op": a.op,
                "k": a.k,
                "stratum": a.stratum,
            }
            for a in arms
        ],
        "start": args.start,
        "end": args.end,
        "n": args.n,
        "clean_correct_only": args.clean_correct_only,
        "rc_source": args.rc_source,
        "loo_results": args.loo_results,
        "loo_inline": args.loo_inline,
        "loo_deletion_mode": args.loo_deletion_mode,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decoding": "greedy",
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not pending:
        logger.info("処理対象なし (すべて既存) — summary のみ再計算")

    from typo_cot.intervention.deletion_runner import run_samples
    from typo_cot.intervention.deletion_stats import aggregate_results
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper

    loo_map: dict[str, list[dict]] = {}
    if need_loo and args.loo_results:
        loo_map = load_loo_rankings(args.loo_results)
        logger.info(f"LOO ランキングをロード: {len(loo_map)} 件")

    records = list(existing)
    if pending:
        gpu_id = resolve_gpu_id(args.gpu_id, os.environ)
        logger.info(f"モデルをロード: {args.model} (GPU {gpu_id})")
        wrapper = create_model_wrapper(
            model_name=args.model, gpu_id=gpu_id, wrap_for_lxt=False
        )
        template = create_prompt_template(args.benchmark)

        def generate_fn(contexts: list[str]) -> list[str]:
            results = wrapper.generate_batch(
                contexts, max_new_tokens=args.max_new_tokens
            )
            return [r.generated_text for r in results]

        replacement_sampler = None
        if need_replace:
            from typo_cot.intervention.replacement import ReplacementSampler

            replacement_sampler = ReplacementSampler()

        t0 = time.time()
        n_done = 0
        for i in range(0, len(pending), args.save_every):
            chunk = pending[i : i + args.save_every]
            prompts = [build_prompt(template, args.benchmark, e) for e in chunk]
            rc_rankings = [
                load_rc_ranking(run_dir, e["sample_id"], e, args.rc_source)
                for e in chunk
            ]
            loo_rankings: list[list[dict] | None] = []
            if need_loo:
                for e, prompt in zip(chunk, prompts, strict=True):
                    sid = e["sample_id"]
                    if sid in loo_map:
                        loo_rankings.append(loo_map[sid])
                    elif args.loo_inline:
                        from typo_cot.intervention.loo_scorer import score_sample_loo

                        loo = score_sample_loo(
                            wrapper.model,
                            wrapper.tokenizer,
                            prompt,
                            e["generated_text"],
                            batch_size=args.batch_size,
                            deletion_mode=args.loo_deletion_mode,
                        )
                        loo_rankings.append(loo["word_scores"] if loo else None)
                    else:
                        loo_rankings.append(None)
            else:
                loo_rankings = [None] * len(chunk)

            chunk_records = run_samples(
                entries=chunk,
                prompts=prompts,
                rc_rankings=rc_rankings,
                loo_rankings=loo_rankings,
                arms=arms,
                generate_fn=generate_fn,
                benchmark=args.benchmark,
                seed=args.seed,
                batch_size=args.batch_size,
                replacement_sampler=replacement_sampler,
            )
            records.extend(chunk_records)
            save_results_atomic(out_dir, records)
            n_done += len(chunk)
            rate = (time.time() - t0) / max(1, n_done)
            logger.info(f"{n_done}/{len(pending)} サンプル処理済 ({rate:.1f}s/sample)")

    save_results_atomic(out_dir, records)

    summary = {
        "experiment_info": {
            "experiment": "exp2_target_deletion",
            "model": args.model,
            "benchmark": args.benchmark,
            "arms_preset": args.arms,
            "rc_source": args.rc_source,
            "timestamp": datetime.now().isoformat(),
        },
        **aggregate_results(records, n_boot=args.n_perm, seed=args.seed),
        "dose_trends": build_dose_trends(records, n_perm=args.n_perm, seed=args.seed),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"完了: {out_dir}")
    for stratum, block in summary.get("strata", {}).items():
        for name, arm in block["arms"].items():
            logger.info(
                f"[{stratum}] {name}: flip_rate={arm['flip_rate']} "
                f"(n={arm['n']}, all={arm['flip_rate_all']})"
            )
    for c in summary.get("contrasts", []):
        logger.info(
            f"contrast {c['arm_a']} vs {c['arm_b']}: RD={c['risk_difference']} "
            f"CI95={c['rd_ci95']} McNemar p={c['mcnemar_p']}"
        )


if __name__ == "__main__":
    main()
