"""Public API for the Appendix E typo-warning prompt audit."""

from typo_cot.experiments.typo_warning_prompt.aggregation import (
    BuildTypoWarningSummaryConfig,
    BuildTypoWarningSummaryResult,
    TypoWarningSummaryInputError,
    build_typo_warning_summary,
)
from typo_cot.experiments.typo_warning_prompt.planning import (
    WarningPromptArmPlan,
    WarningPromptPlan,
    build_warning_prompt_plan,
    inject_typo_warning,
)
from typo_cot.experiments.typo_warning_prompt.protocol import (
    ARM_ORDER,
    PAPER_BENCHMARKS,
    PAPER_GRID,
    PAPER_MODELS,
    PROMPT_MARKERS,
    WARNING_TEXT,
)
from typo_cot.experiments.typo_warning_prompt.runner import (
    WarningPromptArmScan,
    WarningPromptConfig,
    WarningPromptGeneration,
    WarningPromptInputUse,
    WarningPromptResult,
    WarningPromptRunError,
    WarningPromptScan,
    run_typo_warning_prompt,
)
from typo_cot.experiments.typo_warning_prompt.source import (
    WarningSourceBundle,
    WarningSourceCase,
    load_warning_sources,
    select_warning_cases,
    validate_source_snapshot,
)
from typo_cot.experiments.typo_warning_prompt.statistics import exact_mcnemar

__all__ = [
    "ARM_ORDER",
    "PAPER_BENCHMARKS",
    "PAPER_GRID",
    "PAPER_MODELS",
    "PROMPT_MARKERS",
    "WARNING_TEXT",
    "BuildTypoWarningSummaryConfig",
    "BuildTypoWarningSummaryResult",
    "TypoWarningSummaryInputError",
    "WarningPromptArmPlan",
    "WarningPromptArmScan",
    "WarningPromptConfig",
    "WarningPromptGeneration",
    "WarningPromptInputUse",
    "WarningPromptPlan",
    "WarningPromptResult",
    "WarningPromptRunError",
    "WarningPromptScan",
    "WarningSourceBundle",
    "WarningSourceCase",
    "build_typo_warning_summary",
    "build_warning_prompt_plan",
    "exact_mcnemar",
    "inject_typo_warning",
    "load_warning_sources",
    "run_typo_warning_prompt",
    "select_warning_cases",
    "validate_source_snapshot",
]
