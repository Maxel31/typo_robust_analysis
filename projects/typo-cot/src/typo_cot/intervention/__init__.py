"""介入実験モジュール (実験1: CoT移植 2×2 / 実験3: divergence / 実験8: patching)."""

from typo_cot.intervention.analysis import (
    bootstrap_ci,
    bootstrap_flip_cis,
    flip_table,
    glmm_decomposition,
)
from typo_cot.intervention.archive_loader import load_pair_records
from typo_cot.intervention.cell_builder import (
    CellInputs,
    TruncationResult,
    build_cell_inputs,
    truncate_before_answer,
)
from typo_cot.intervention.divergence import (
    align_cot_targets,
    divergence_onset,
    positionwise_divergence,
    precision_at_k,
    shuffle_null_precision,
)
from typo_cot.intervention.patching import (
    DIRECTIONS,
    SITES,
    ActivationCache,
    FirstDivergence,
    PatchCell,
    PatchInjector,
    capture_activations,
    find_decoder_layers,
    first_divergence,
    get_site_module,
    iter_patch_cells,
    kl_from_logits,
    layer_windows,
    result_is_current,
    span_end_token,
)
from typo_cot.intervention.records import PairRecord
from typo_cot.intervention.runner import CellOutcome, run_cells

__all__ = [
    "DIRECTIONS",
    "SITES",
    "ActivationCache",
    "CellInputs",
    "CellOutcome",
    "FirstDivergence",
    "PairRecord",
    "PatchCell",
    "PatchInjector",
    "TruncationResult",
    "align_cot_targets",
    "capture_activations",
    "find_decoder_layers",
    "first_divergence",
    "get_site_module",
    "iter_patch_cells",
    "kl_from_logits",
    "layer_windows",
    "result_is_current",
    "span_end_token",
    "bootstrap_ci",
    "bootstrap_flip_cis",
    "build_cell_inputs",
    "divergence_onset",
    "flip_table",
    "glmm_decomposition",
    "load_pair_records",
    "positionwise_divergence",
    "precision_at_k",
    "run_cells",
    "shuffle_null_precision",
    "truncate_before_answer",
]
