"""Public API for the Appendix C/Table 9 CPU artifact builder."""

from typo_cot.experiments.model_scale_cot_swap.protocol import PUBLISHED_REFERENCE
from typo_cot.experiments.model_scale_cot_swap.runner import (
    ModelScaleCotSwapConfig,
    ModelScaleCotSwapResult,
    run_model_scale_cot_swap,
)
from typo_cot.experiments.model_scale_cot_swap.source import (
    ModelScaleCotSwapInputError,
)

__all__ = [
    "PUBLISHED_REFERENCE",
    "ModelScaleCotSwapConfig",
    "ModelScaleCotSwapInputError",
    "ModelScaleCotSwapResult",
    "run_model_scale_cot_swap",
]
