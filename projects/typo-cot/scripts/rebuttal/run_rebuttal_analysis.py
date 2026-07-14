#!/usr/bin/env python3
"""Rebuttal 用分析ドライバ: union 除外を既存一括分析と同一に再現して個別ペアを分析する.

既存 scripts/run_analysis.py の一括モードは outputs/perturbed の
{model}_{bench}_k{N}_{importance|random} 群から (model, bench) 単位の union 除外集合を
計算して各ペアに適用するが、個別モード (--before_dir/--after_dir) では union 除外が
使えない。本スクリプトは新規条件ディレクトリ (fixed_target / spellfix 等) に対して
既存と同一の union 除外集合 (+ 追加除外) を適用した分析を実行する。

使用例:
  uv run --no-sync python scripts/rebuttal/run_rebuttal_analysis.py \
    --baseline_dir outputs/baseline/gemma-3-4b-it_mmlu \
    --perturbed_root outputs/perturbed \
    --after_dirs outputs/perturbed/gemma-3-4b-it_mmlu_k4_importance \
                 outputs/rebuttal/fixed_target/gemma-3-4b-it_mmlu_k4_fixed_target \
    --extra_excluded outputs/rebuttal/fixed_target/gemma-3-4b-it_mmlu_k4_fixed_target/fixed_target_stats.json \
    --output_dir outputs/rebuttal/analysis
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path


from typo_cot.analysis import compute_unified_exclusion, run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rebuttal_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuttal 分析 (union 除外つき個別ペア)")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument(
        "--perturbed_root", type=str, default="outputs/perturbed",
        help="union 除外の計算対象となる既存摂動ディレクトリ群のルート",
    )
    parser.add_argument("--after_dirs", type=str, nargs="+", required=True)
    parser.add_argument(
        "--extra_excluded", type=str, default=None,
        help="追加除外 sample_id を含む JSON (fixed_target_stats.json の skipped_ids 等)",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    base_name = baseline_dir.name  # 例: gemma-3-4b-it_mmlu

    with open(baseline_dir / "config.json", encoding="utf-8") as f:
        dataset = json.load(f).get("benchmark")

    # 既存一括分析 (run_analysis.py:115-143, 282-322) と同じ対象で union 除外を計算:
    # {base_name}_k{N}_{importance|random} にマッチする既存摂動ディレクトリ全部
    pattern = re.compile(rf"^{re.escape(base_name)}_k(\d+)_(importance|random)$")
    perturbed_root = Path(args.perturbed_root)
    union_dirs = sorted(
        d for d in perturbed_root.iterdir() if d.is_dir() and pattern.match(d.name)
    )
    if not union_dirs:
        logger.error(f"union 対象ディレクトリが見つかりません: {base_name}")
        sys.exit(1)
    logger.info(f"union 除外の対象条件: {[d.name for d in union_dirs]}")

    excluded = compute_unified_exclusion(baseline_dir, union_dirs, dataset)
    logger.info(f"union 除外: {len(excluded)} サンプル")

    if args.extra_excluded:
        with open(args.extra_excluded, encoding="utf-8") as f:
            extra_data = json.load(f)
        if isinstance(extra_data, dict) and "skipped_ids" in extra_data:
            extra_ids = set(extra_data["skipped_ids"].keys())
        elif isinstance(extra_data, list):
            extra_ids = set(extra_data)
        else:
            extra_ids = set(extra_data)
        before_n = len(excluded)
        excluded = excluded | extra_ids
        logger.info(
            f"追加除外: {len(extra_ids)} サンプル (union と合わせて {len(excluded)}, "
            f"純増 {len(excluded) - before_n})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 除外集合を記録 (再現性のため)
    with open(output_dir / f"excluded_ids_{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(sorted(excluded), f, ensure_ascii=False, indent=2)

    for after in args.after_dirs:
        after_dir = Path(after)
        logger.info(f"=== 分析: {baseline_dir.name} vs {after_dir.name} ===")
        result = run_analysis(
            before_dir=baseline_dir,
            after_dir=after_dir,
            output_dir=output_dir,
            excluded_sample_ids=excluded,
        )
        logger.info(
            f"完了: 総サンプル {result.total_samples}, 回答変化 {result.answer_changed_count}"
        )


if __name__ == "__main__":
    main()
