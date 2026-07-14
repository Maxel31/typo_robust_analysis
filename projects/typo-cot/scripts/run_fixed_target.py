#!/usr/bin/env python3
"""実験4: fixed-target 帰属 (R_C^fixed) の全設定一般化ランナー.

rebuttal 期 scripts/rebuttal/run_fixed_target_attribution.py (4設定版) の一般化。
コアロジックは typo_cot.attribution.fixed_target (テスト済み) に分離してある。

追加機能 (rebuttal 版との差分):
- --flip_only: flip (splice) 事例のみ AttnLRP を再実行 (非flip は定義上 default と
  同値なので再計算不要 — 全設定展開時の計算節約)
- --sample_ids / --sample_ids_file: 対象サンプルの明示指定 (スモーク用)
- --compare_dir: 参照 fixed_target ディレクトリ (rebuttal 出力) と同一サンプルの
  _cot.pt を比較し comparison.json に保存 (スモーク検証 (a))
- --compare_default: 非flip事例について摂動 run の default _cot.pt と比較
  (スモーク検証 (b))

使用例 (スモーク):
  uv run python scripts/run_fixed_target.py \
    --baseline_dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
    --perturbed_dir $ARCHIVE/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
    --model google/gemma-3-4b-it --benchmark gsm8k \
    --output_dir results/smoke/fixed_target \
    --sample_ids_file results/smoke/sample_ids_gsm8k.json \
    --compare_dir $ARCHIVE/outputs/rebuttal/fixed_target/gemma-3-4b-it_gsm8k_k4_fixed_target \
    --compare_default
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from typo_cot.attribution.fixed_target import (
    analyze_cot_fixed,
    build_prompt,
    compare_cot_payloads,
    fixed_target_entry,
    plan_run,
)
from typo_cot.data import run_io

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_fixed_target")
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-Target Attribution (実験4)")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--perturbed_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample_ids", type=str, default=None, help="カンマ区切りの sample_id リスト"
    )
    parser.add_argument(
        "--sample_ids_file", type=str, default=None,
        help="sample_id の JSON リストを含むファイル",
    )
    parser.add_argument(
        "--flip_only", action="store_true",
        help="flip (splice) 事例のみ再計算する (非flip は default と同値)",
    )
    parser.add_argument(
        "--compare_dir", type=str, default=None,
        help="参照 fixed_target ディレクトリ (同一サンプルの _cot.pt と比較)",
    )
    parser.add_argument(
        "--compare_default", action="store_true",
        help="非flip事例を摂動 run の default _cot.pt と比較",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    perturbed_dir = Path(args.perturbed_dir)

    baseline_results = run_io.load_results_by_id(baseline_dir)
    perturbed_results = run_io.load_results_list(perturbed_dir)
    perturbed_by_id = {r["sample_id"]: r for r in perturbed_results}
    perturbed_config = run_io.load_run_config(perturbed_dir)

    sample_ids: set[str] | None = None
    if args.sample_ids:
        sample_ids = set(args.sample_ids.split(","))
    if args.sample_ids_file:
        with open(args.sample_ids_file, encoding="utf-8") as f:
            sample_ids = (sample_ids or set()) | set(json.load(f))

    plans, stats = plan_run(
        baseline_results, perturbed_results, limit=args.limit, sample_ids=sample_ids
    )
    all_plans = list(plans)  # 出力順 (= 摂動 results.json 順) の復元用
    nonflip_plans: list = []
    if args.flip_only:
        nonflip_plans = [p for p in plans if not p.spliced]
        plans = [p for p in plans if p.spliced]
        stats["flip_only"] = True
        stats["nonflip_reused"] = len(nonflip_plans)
    logger.info(
        f"計画: total={stats['total']} processed対象={len(plans)} "
        f"(spliced={stats['spliced']}, identical={stats['identical']})"
    )

    num_perturbations = perturbed_config.get("perturbed_metadata", {}).get(
        "num_perturbations", "unknown"
    )
    model_short = args.model.split("/")[-1]
    out_dir = (
        Path(args.output_dir)
        / f"{model_short}_{args.benchmark}_k{num_perturbations}_fixed_target"
    )
    scores_dir = out_dir / "importance_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    config = dict(perturbed_config)
    config["fixed_target_source"] = str(perturbed_dir)
    config["baseline_source"] = str(baseline_dir)
    config["timestamp"] = datetime.now().isoformat()
    if "perturbed_metadata" in config:
        config["perturbed_metadata"] = dict(config["perturbed_metadata"])
        config["perturbed_metadata"]["perturbation_mode"] = "fixed_target"
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # モデル・分析器 (rebuttal 実装と同一の構成)
    import torch

    from typo_cot.lrp.analyzer import create_analyzer
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper

    logger.info(f"モデルをロード: {args.model} (GPU {args.gpu_id})")
    wrapper = create_model_wrapper(
        model_name=args.model, gpu_id=args.gpu_id, wrap_for_lxt=True
    )
    template = create_prompt_template(args.benchmark)
    analyzer = create_analyzer(
        model=wrapper.model, tokenizer=wrapper.tokenizer, top_k=None,
        device=wrapper.device,
    )

    ref_dir = Path(args.compare_dir) if args.compare_dir else None
    comparisons: dict[str, dict] = {}
    computed: dict[str, dict] = {}
    t0 = time.time()

    for idx, plan in enumerate(plans):
        sid = plan.sample_id
        entry = perturbed_by_id[sid]
        try:
            prompt = build_prompt(template, args.benchmark, entry)
            cot_data = analyze_cot_fixed(analyzer, prompt, plan.spliced_text)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["processed"] -= 1
            stats["spliced" if plan.spliced else "identical"] -= 1
            stats["skipped_ids"][sid] = f"error: {exc}"
            logger.warning(f"{sid} 失敗: {exc}")
            continue

        cot_data["fixed_target_baseline_answer"] = plan.baseline_answer
        cot_data["fixed_target_perturbed_answer"] = plan.perturbed_answer
        cot_data["fixed_target_spliced"] = plan.spliced
        torch.save(cot_data, scores_dir / f"{sid}_cot.pt")

        # 質問側 .pt は摂動 run と同一 (固定ターゲット化の影響なし) → シンボリックリンク
        src_q = run_io.question_scores_path(perturbed_dir, sid).resolve()
        dst_q = scores_dir / f"{sid}.pt"
        if src_q.exists() and not dst_q.exists():
            dst_q.symlink_to(src_q)

        computed[sid] = fixed_target_entry(entry, plan)

        # スモーク検証 (a): rebuttal 参照との比較
        if ref_dir is not None and run_io.cot_scores_path(ref_dir, sid).exists():
            ref = run_io.load_cot_scores(ref_dir, sid)
            rep = compare_cot_payloads(cot_data, ref)
            rep["kind"] = "vs_reference_fixed"
            rep["spliced"] = plan.spliced
            comparisons[sid] = rep
        # スモーク検証 (b): 非flip事例は default _cot.pt と同値のはず
        if (
            args.compare_default
            and not plan.spliced
            and run_io.cot_scores_path(perturbed_dir, sid).exists()
        ):
            ref = run_io.load_cot_scores(perturbed_dir, sid)
            rep = compare_cot_payloads(cot_data, ref)
            rep["kind"] = "vs_default"
            key = f"{sid}::default"
            comparisons[key] = rep

        if (idx + 1) % 20 == 0:
            rate = (time.time() - t0) / (idx + 1)
            logger.info(f"{idx + 1}/{len(plans)} 処理済 ({rate:.2f}s/sample)")

    # 非flip の再利用 (--flip_only): default 側 .pt を symlink し、エントリを合流。
    # これで出力ディレクトリが analyzer にそのまま掛けられる完全な run になる。
    nonflip_missing_cot = []
    nonflip_ids = {p.sample_id for p in nonflip_plans}
    for plan in nonflip_plans:
        linked = run_io.link_reused_scores(perturbed_dir, scores_dir, plan.sample_id)
        if not linked["cot"]:
            nonflip_missing_cot.append(plan.sample_id)
            stats["skipped_ids"][plan.sample_id] = "nonflip_default_cot_pt_missing"
    if nonflip_missing_cot:
        logger.warning(
            f"非flip {len(nonflip_missing_cot)} 件で default _cot.pt が見つからず "
            "再利用不能 (skipped_ids に記録)"
        )
    stats["nonflip_missing_default_cot"] = len(nonflip_missing_cot)
    missing_set = set(nonflip_missing_cot)

    out_results = []
    for plan in all_plans:
        sid = plan.sample_id
        if sid in computed:
            out_results.append(computed[sid])
        elif sid in nonflip_ids and sid not in missing_set:
            out_results.append(fixed_target_entry(perturbed_by_id[sid], plan))

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out_results, f, ensure_ascii=False, indent=2)

    stats["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out_dir / "fixed_target_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if comparisons:
        n_ref = sum(1 for r in comparisons.values() if r["kind"] == "vs_reference_fixed")
        n_def = sum(1 for r in comparisons.values() if r["kind"] == "vs_default")
        summary = {
            "n_vs_reference": n_ref,
            "n_vs_default": n_def,
            "min_top10_jaccard_vs_reference": min(
                (r["top10_jaccard"] for r in comparisons.values()
                 if r["kind"] == "vs_reference_fixed"), default=None
            ),
            "min_top10_jaccard_vs_default": min(
                (r["top10_jaccard"] for r in comparisons.values()
                 if r["kind"] == "vs_default"), default=None
            ),
            "all_cot_range_match": all(
                r["cot_range_match"] for r in comparisons.values()
            ),
            "details": comparisons,
        }
        with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(
            f"比較: vs_reference n={n_ref} min_top10_jaccard="
            f"{summary['min_top10_jaccard_vs_reference']}, "
            f"vs_default n={n_def} min_top10_jaccard="
            f"{summary['min_top10_jaccard_vs_default']}, "
            f"cot_range_match all={summary['all_cot_range_match']}"
        )

    logger.info(f"完了: {out_dir}")
    logger.info(json.dumps(
        {k: v for k, v in stats.items() if k != "skipped_ids"}, ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
