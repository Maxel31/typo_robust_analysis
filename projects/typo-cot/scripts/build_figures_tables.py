#!/usr/bin/env python3
"""論文Figure 2/3 と Table 3/5/6 を一括生成する CLI スクリプト.

入力:
- `outputs/analysis/<dataset>/<model>/k<k>_<type>/` 配下の exp*.json / full_results.json
- `outputs/baseline/<model>_<bench>/summary.json`
- `outputs/perturbed/<model>_<bench>_k<k>_<type>/summary.json`

出力（既定 `outputs/figures/`）:
- `figure2a.png`, `figure2b.png`, `figure3.png`
- `table3.csv` + `table3.tex`
- `table5.csv` + `table5.tex`
- `table6.csv` + `table6.tex`
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


from typo_cot.visualization import (
    DEFAULT_DATASET_ORDER,
    DEFAULT_MODEL_ORDER,
    collect_accuracy_summary,
    collect_exclusion_stats,
    collect_overall_metrics,
    collect_partial_correlations,
    collect_q_cot_correlations,
    iter_analysis_dirs,
    make_exclusion_summary,
    make_table3,
    make_table5,
    make_table6,
    plot_figure2a,
    plot_figure2b,
    plot_figure3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_n_changed(analysis_dir: Path) -> tuple[int | None, int | None]:
    """`full_results.json` の `answer_change` から (unchanged, changed) を取得."""
    path = analysis_dir / "full_results.json"
    if not path.exists():
        return None, None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    ac = data.get("answer_change", {})
    return ac.get("unchanged"), ac.get("changed")


def _build_n_changed_df(analysis_root: Path, k: int, pert_type: str):
    import pandas as pd

    rows: list[dict] = []
    for ds, model, k_val, ptype, dir_path in iter_analysis_dirs(analysis_root):
        if ptype != pert_type or k_val != k:
            continue
        n_unchanged, n_changed = _load_n_changed(dir_path)
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "n_unchanged": n_unchanged,
                "n_changed": n_changed,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="論文Figure/Table 一括生成")
    p.add_argument("--analysis_dir", type=Path, default=Path("outputs/analysis"))
    p.add_argument("--baseline_dir", type=Path, default=Path("outputs/baseline"))
    p.add_argument("--perturbed_dir", type=Path, default=Path("outputs/perturbed"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_ORDER,
        help="集計対象モデル（baseline/perturbed の prefix 名）",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_DATASET_ORDER,
        help="集計対象ベンチマーク",
    )
    p.add_argument("--k_fig", type=int, default=4, help="Figure 3 / Table 3 の基準 k")
    p.add_argument("--k_partial", type=int, default=10, help="Table 3 の偏相関 k")
    p.add_argument(
        "--skip", nargs="*", default=[],
        choices=["fig2a", "fig2b", "fig3", "table3", "table5", "table6", "exclusion"],
        help="スキップする生成対象",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("論文 Figure / Table 一括生成")
    logger.info(f"  analysis_dir : {args.analysis_dir}")
    logger.info(f"  baseline_dir : {args.baseline_dir}")
    logger.info(f"  perturbed_dir: {args.perturbed_dir}")
    logger.info(f"  output_dir   : {args.output_dir}")
    logger.info("=" * 60)

    if "fig2a" not in args.skip or "fig2b" not in args.skip:
        logger.info("[Figure 2] overall_metrics を集約中…")
        overall = collect_overall_metrics(args.analysis_dir)
        if overall.empty:
            logger.warning("overall_metrics が空。analysis_dir を確認してください。")
        else:
            if "fig2a" not in args.skip:
                plot_figure2a(overall, args.output_dir / "figure2a.png", model_order=args.models)
            if "fig2b" not in args.skip:
                plot_figure2b(overall, args.output_dir / "figure2b.png", model_order=args.models)

    if "fig3" not in args.skip:
        logger.info(f"[Figure 3] Q↔CoT 相関 (k={args.k_fig}) を集約中…")
        qcot = collect_q_cot_correlations(args.analysis_dir, k=args.k_fig)
        if qcot.empty:
            logger.warning("Q↔CoT correlations が空")
        else:
            plot_figure3(qcot, args.output_dir / "figure3.png", model_order=args.models)

    if "table3" not in args.skip:
        logger.info(f"[Table 3] 偏相関 (k={args.k_partial}) を集約中…")
        partial = collect_partial_correlations(args.analysis_dir, k=args.k_fig)
        n_df = _build_n_changed_df(args.analysis_dir, k=args.k_fig, pert_type="importance")
        make_table3(
            partial,
            n_df,
            out_csv=args.output_dir / "table3.csv",
            out_tex=args.output_dir / "table3.tex",
            k=args.k_partial,
        )

    if "table5" not in args.skip:
        logger.info("[Table 5] accuracy summary を集約中…")
        acc = collect_accuracy_summary(
            args.baseline_dir,
            args.perturbed_dir,
            models=args.models,
            benchmarks=args.benchmarks,
        )
        make_table5(
            acc,
            out_csv=args.output_dir / "table5.csv",
            out_tex=args.output_dir / "table5.tex",
        )

    if "table6" not in args.skip:
        logger.info("[Table 6] C→I 偏相関を再計算中…")
        make_table6(
            args.analysis_dir,
            out_csv=args.output_dir / "table6.csv",
            out_tex=args.output_dir / "table6.tex",
        )

    if "exclusion" not in args.skip:
        logger.info("[exclusion_summary] 回答スパン未検出による除外統計を集約中…")
        excl = collect_exclusion_stats(args.analysis_dir)
        make_exclusion_summary(
            excl,
            out_csv=args.output_dir / "exclusion_summary.csv",
            out_tex=args.output_dir / "exclusion_summary.tex",
        )

    logger.info("=" * 60)
    logger.info(f"完了: {args.output_dir} に出力されました")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
