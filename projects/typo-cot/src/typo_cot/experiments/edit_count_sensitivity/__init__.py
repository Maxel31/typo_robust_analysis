"""Public API for Appendix C/Table 8 reproduction."""

from typo_cot.experiments.edit_count_sensitivity.protocol import PUBLISHED_REFERENCE
from typo_cot.experiments.edit_count_sensitivity.runner import (
    EditCountSensitivityConfig,
    EditCountSensitivityResult,
    run_edit_count_sensitivity,
)
from typo_cot.experiments.edit_count_sensitivity.source import (
    EditCountSensitivityInputError,
)

__all__ = [
    "PUBLISHED_REFERENCE",
    "EditCountSensitivityConfig",
    "EditCountSensitivityInputError",
    "EditCountSensitivityResult",
    "run_edit_count_sensitivity",
]
