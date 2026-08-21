"""Public API for fixed-window free-answer activation patching."""

from typo_cot.experiments.fixed_window_answer_patching.runner import (
    DIRECTION_NAMES,
    FIXED_WINDOW_ANSWER_PATCHING_PROTOCOL,
    FixedWindowAnswerPatchingConfig,
    FixedWindowAnswerPatchingResult,
    FixedWindowAnswerPatchingRunError,
    LayerWindow,
    parse_layer_window,
    run_fixed_window_answer_patching,
)

__all__ = [
    "DIRECTION_NAMES",
    "FIXED_WINDOW_ANSWER_PATCHING_PROTOCOL",
    "FixedWindowAnswerPatchingConfig",
    "FixedWindowAnswerPatchingResult",
    "FixedWindowAnswerPatchingRunError",
    "LayerWindow",
    "parse_layer_window",
    "run_fixed_window_answer_patching",
]
