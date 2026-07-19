"""介入実験モジュール (実験1: CoT移植 2×2 / 実験2: ターゲット単語削除LOO / 実験3: forced-decoding divergence / 実験6: AttnLRP帰属手法収束性 / 実験8: activation patching)."""

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
    align_by_relative_depth,
    capture_activations,
    cumulative_windows,
    find_decoder_layers,
    first_divergence,
    get_site_module,
    iter_patch_cells,
    kl_from_logits,
    layer_windows,
    relative_depth,
    result_is_current,
    single_layer_windows,
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
    "align_by_relative_depth",
    "align_cot_targets",
    "bootstrap_ci",
    "bootstrap_flip_cis",
    "build_cell_inputs",
    "capture_activations",
    "cumulative_windows",
    "divergence_onset",
    "find_decoder_layers",
    "first_divergence",
    "flip_table",
    "get_site_module",
    "glmm_decomposition",
    "iter_patch_cells",
    "kl_from_logits",
    "layer_windows",
    "load_pair_records",
    "positionwise_divergence",
    "precision_at_k",
    "relative_depth",
    "result_is_current",
    "run_cells",
    "shuffle_null_precision",
    "single_layer_windows",
    "span_end_token",
    "truncate_before_answer",
]
