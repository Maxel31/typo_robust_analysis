"""Public API for the clean pre-answer token-prefix scan."""

from typo_cot.experiments.clean_prefix_scan.metrics import (
    PrefixTrajectorySummary,
    summarize_prefix_correctness,
)
from typo_cot.experiments.clean_prefix_scan.planning import (
    ABSOLUTE_BUDGETS,
    RELATIVE_BUDGETS,
    CleanCotAlignment,
    PrefixBudget,
    PrefixInputPlan,
    align_clean_cot_suffixes,
    build_budget_grid,
    build_prefix_input_plan,
    select_extension_sample_ids,
)
from typo_cot.experiments.clean_prefix_scan.runner import (
    CleanPrefixGeneration,
    CleanPrefixInputUse,
    CleanPrefixPairScan,
    CleanPrefixPointScan,
    CleanPrefixScanConfig,
    CleanPrefixScanResult,
    CleanPrefixScanRunError,
    CleanPrefixScanRuntime,
    run_clean_prefix_scan,
)

__all__ = [
    "ABSOLUTE_BUDGETS",
    "RELATIVE_BUDGETS",
    "CleanCotAlignment",
    "CleanPrefixGeneration",
    "CleanPrefixInputUse",
    "CleanPrefixPairScan",
    "CleanPrefixPointScan",
    "CleanPrefixScanConfig",
    "CleanPrefixScanResult",
    "CleanPrefixScanRunError",
    "CleanPrefixScanRuntime",
    "PrefixBudget",
    "PrefixInputPlan",
    "PrefixTrajectorySummary",
    "align_clean_cot_suffixes",
    "build_budget_grid",
    "build_prefix_input_plan",
    "run_clean_prefix_scan",
    "select_extension_sample_ids",
    "summarize_prefix_correctness",
]
