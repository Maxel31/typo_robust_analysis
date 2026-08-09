"""Public API for the Appendix E input-corrector audit."""

from typo_cot.experiments.input_corrector_audit.aggregation import (
    BuildInputCorrectorSummaryConfig,
    BuildInputCorrectorSummaryResult,
    InputCorrectorSummaryInputError,
    run_build_input_corrector_summary,
)
from typo_cot.experiments.input_corrector_audit.correctors import (
    CORRECTOR_IDS,
    PySpellCheckerCorrector,
    QwenCorrector,
    T5LargeSpellCorrector,
    create_corrector,
)
from typo_cot.experiments.input_corrector_audit.protocol import (
    PAPER_MODELS,
    SUPPORTED_BENCHMARKS,
)
from typo_cot.experiments.input_corrector_audit.restoration import (
    RestorationResult,
    classify_restoration,
    is_byte_exact_equal,
)
from typo_cot.experiments.input_corrector_audit.runner import (
    CorrectionOutcome,
    InputCorrectorAuditConfig,
    InputCorrectorAuditResult,
    InputCorrectorAuditRunError,
    SamePromptGeneration,
    run_input_corrector_audit,
)

__all__ = [
    "BuildInputCorrectorSummaryConfig",
    "BuildInputCorrectorSummaryResult",
    "CORRECTOR_IDS",
    "CorrectionOutcome",
    "InputCorrectorAuditConfig",
    "InputCorrectorAuditResult",
    "InputCorrectorAuditRunError",
    "InputCorrectorSummaryInputError",
    "PAPER_MODELS",
    "PySpellCheckerCorrector",
    "QwenCorrector",
    "RestorationResult",
    "SUPPORTED_BENCHMARKS",
    "SamePromptGeneration",
    "T5LargeSpellCorrector",
    "classify_restoration",
    "create_corrector",
    "is_byte_exact_equal",
    "run_build_input_corrector_summary",
    "run_input_corrector_audit",
]
