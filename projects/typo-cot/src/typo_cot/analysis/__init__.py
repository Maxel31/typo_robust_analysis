"""Phase 4: 摂動前後の分析モジュール."""

from .analyzer import (
    AnalysisResult,
    PerturbationAnalyzer,
    SamplePairResult,
    compute_unified_exclusion,
    run_analysis,
)
from .metrics import (
    cohens_d,
    js_divergence,
    mann_whitney_u_test,
    normalize_distribution,
    pearson_correlation,
    rouge_l_score,
    shannon_entropy,
    spearman_correlation,
    top_k_concentration,
    top_k_jaccard,
)

__all__ = [
    # Analyzer
    "AnalysisResult",
    "PerturbationAnalyzer",
    "SamplePairResult",
    "compute_unified_exclusion",
    "run_analysis",
    # Metrics
    "cohens_d",
    "js_divergence",
    "mann_whitney_u_test",
    "normalize_distribution",
    "pearson_correlation",
    "rouge_l_score",
    "shannon_entropy",
    "spearman_correlation",
    "top_k_concentration",
    "top_k_jaccard",
]
