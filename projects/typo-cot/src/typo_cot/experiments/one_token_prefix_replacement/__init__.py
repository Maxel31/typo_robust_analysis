"""Public API for the one-token clean-prefix replacement diagnostic."""

from typo_cot.experiments.one_token_prefix_replacement.planning import (
    DistantPositionSelection,
    OneTokenArmSpec,
    OneTokenInputPlan,
    OneTokenProfile,
    build_arm_specs,
    choose_adjacent_position,
    choose_distant_positions,
)
from typo_cot.experiments.one_token_prefix_replacement.protocol import (
    ADJACENT_SETTINGS,
    EXTENSION_SETTINGS,
    HISTORICAL_REFERENCE,
    POSITION_CONTROLS,
    PROTOCOL,
)
from typo_cot.experiments.one_token_prefix_replacement.runner import (
    OneTokenGeneration,
    OneTokenPrefixReplacementConfig,
    OneTokenPrefixReplacementResult,
    OneTokenPrefixReplacementRunError,
    OneTokenPrefixReplacementRuntime,
    classify_one_token_events,
    run_one_token_prefix_replacement,
)
from typo_cot.experiments.one_token_prefix_replacement.runtime import (
    HuggingFaceOneTokenPrefixReplacementRuntime,
    OneTokenBoundaryInvalid,
)

__all__ = [
    "ADJACENT_SETTINGS",
    "EXTENSION_SETTINGS",
    "HISTORICAL_REFERENCE",
    "POSITION_CONTROLS",
    "PROTOCOL",
    "DistantPositionSelection",
    "HuggingFaceOneTokenPrefixReplacementRuntime",
    "OneTokenArmSpec",
    "OneTokenBoundaryInvalid",
    "OneTokenGeneration",
    "OneTokenInputPlan",
    "OneTokenPrefixReplacementConfig",
    "OneTokenPrefixReplacementResult",
    "OneTokenPrefixReplacementRunError",
    "OneTokenPrefixReplacementRuntime",
    "OneTokenProfile",
    "build_arm_specs",
    "choose_adjacent_position",
    "choose_distant_positions",
    "classify_one_token_events",
    "run_one_token_prefix_replacement",
]
