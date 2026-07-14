"""可視化モジュール.

- `heatmap`: トークン重要度のヒートマップ生成（per-sample）
- `aggregators`: 分析結果JSONの横断的集約
- `figures`: 論文 Figure 2 / Figure 3 を再生成
- `tables`: 論文 Table 3 / Table 5 / Table 6 を再生成
"""

from .aggregators import (
    collect_accuracy_summary,
    collect_exclusion_stats,
    collect_overall_metrics,
    collect_partial_correlations,
    collect_q_cot_correlations,
    iter_analysis_dirs,
    load_sample_results,
)
from .figures import (
    DEFAULT_DATASET_ORDER,
    DEFAULT_MODEL_ORDER,
    plot_figure2a,
    plot_figure2b,
    plot_figure3,
)
from .tables import (
    compute_partial_corr_c2i,
    make_exclusion_summary,
    make_table3,
    make_table5,
    make_table6,
)

__all__ = [
    "DEFAULT_DATASET_ORDER",
    "DEFAULT_MODEL_ORDER",
    "collect_accuracy_summary",
    "collect_exclusion_stats",
    "collect_overall_metrics",
    "collect_partial_correlations",
    "collect_q_cot_correlations",
    "compute_partial_corr_c2i",
    "iter_analysis_dirs",
    "load_sample_results",
    "make_exclusion_summary",
    "make_table3",
    "make_table5",
    "make_table6",
    "plot_figure2a",
    "plot_figure2b",
    "plot_figure3",
]
