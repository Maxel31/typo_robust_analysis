"""Numerical helpers for the layerwise first-CoT-token KL experiment.

The functions in this module intentionally keep the paper's estimands small
and explicit: KL is evaluated in float32 from logits, pair scores are never
clipped, and all setting-level summaries preserve the pairing across layers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

KL_DENOMINATOR_EPSILON = 1e-9
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 42
MCB_CONFIDENCE_LEVEL = 0.95


def kl_from_logits(reference_logits: torch.Tensor, comparison_logits: torch.Tensor) -> float:
    """Return ``KL(reference || comparison)`` for one next-token distribution.

    Model logits may be produced in bfloat16 or float16, but the paper's KL
    readout is evaluated after converting both inputs to float32.  Leading
    singleton dimensions (for example ``[1, vocab]``) are harmless; all
    probability mass is reduced to the returned scalar.

    Non-finite model output is deliberately propagated as a non-finite return
    value.  The complete-grid validator can then exclude the affected
    pair/direction without turning a scientific exclusion into a process
    failure.
    """

    if reference_logits.shape != comparison_logits.shape:
        raise ValueError("KL logits must have identical shapes")
    if reference_logits.ndim == 0 or reference_logits.shape[-1] == 0:
        raise ValueError("KL logits must contain a non-empty vocabulary dimension")

    reference_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
    comparison_log_probs = torch.log_softmax(comparison_logits.float(), dim=-1)
    divergence = torch.sum(reference_log_probs.exp() * (reference_log_probs - comparison_log_probs))
    return float(divergence.detach().cpu().item())


def normalized_kl_score(*, denominator_kl: float, patched_kl: float) -> float:
    """Compute the paper's unclipped normalized KL restoration score.

    The untreated denominator is admissible only when it is finite and
    strictly greater than :data:`KL_DENOMINATOR_EPSILON`.  Negative scores are
    meaningful (the intervention moved farther from the reference) and are
    therefore retained.
    """

    if not math.isfinite(denominator_kl):
        raise ValueError("nonfinite_denominator")
    if denominator_kl <= KL_DENOMINATOR_EPSILON:
        raise ValueError("denominator_le_1e-9")
    if not math.isfinite(patched_kl):
        raise ValueError("nonfinite_layer_value")

    score = 1.0 - patched_kl / denominator_kl
    if not math.isfinite(score):
        raise ValueError("nonfinite_layer_value")
    return score


def summarize_direction(
    pair_scores: Sequence[Sequence[float]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize complete pair-by-layer score grids for one direction.

    Layer profiles use across-pair medians.  The observed peak is the lowest
    layer attaining the maximum median.  Hsu's comparisons-with-the-best set
    is estimated from paired bootstrap medians of
    ``score[layer] - score[observed_peak]`` using a shared resample sequence
    for every layer.
    """

    scores = _validated_score_matrix(pair_scores)
    _validate_bootstrap_arguments(bootstrap_resamples, seed)

    num_pairs, num_layers = scores.shape
    layer_medians = np.median(scores, axis=0)
    maximum_median = float(np.max(layer_medians))
    tied_layers = np.flatnonzero(layer_medians == maximum_median)
    peak_layer = int(tied_layers[0])

    layer_profile = [
        {
            "layer_index": layer,
            "relative_depth": layer / num_layers,
            "layer_center_relative_depth": (layer + 0.5) / num_layers,
            "median_normalized_kl": float(layer_medians[layer]),
        }
        for layer in range(num_layers)
    ]

    mcb = _paired_hsu_mcb(
        scores,
        peak_layer=peak_layer,
        tied_layers=tied_layers,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )

    return {
        "included_pairs": num_pairs,
        "num_layers": num_layers,
        "layer_profile": layer_profile,
        "peak": {
            "layer_index": peak_layer,
            "tied_layer_indices": [int(layer) for layer in tied_layers],
            "relative_depth": peak_layer / num_layers,
            "median_normalized_kl": maximum_median,
        },
        "mcb": mcb,
        "depth_thirds": _summarize_depth_thirds(scores),
    }


def _validated_score_matrix(pair_scores: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a rectangular, complete, finite grid to float64 NumPy storage."""

    try:
        materialized = tuple(tuple(row) for row in pair_scores)
    except TypeError as error:
        raise ValueError("pair scores must be a two-dimensional sequence") from error

    if not materialized:
        raise ValueError("at least one complete pair score grid is required")
    if not materialized[0]:
        raise ValueError("pair score grids must contain at least one layer")

    num_layers = len(materialized[0])
    if any(len(row) != num_layers for row in materialized):
        raise ValueError("pair score grids must have the same number of layers")

    try:
        scores = np.asarray(materialized, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("pair scores must be real numbers") from error
    if scores.ndim != 2:
        raise ValueError("pair scores must be a two-dimensional sequence")
    if not np.isfinite(scores).all():
        raise ValueError("pair score grids must contain only finite values")
    return scores


def _validate_bootstrap_arguments(bootstrap_resamples: int, seed: int) -> None:
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _paired_hsu_mcb(
    scores: np.ndarray,
    *,
    peak_layer: int,
    tied_layers: np.ndarray,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Return the paired, one-sided percentile MCB result."""

    num_pairs, num_layers = scores.shape
    differences = scores - scores[:, [peak_layer]]
    observed_difference_medians = np.median(differences, axis=0)

    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(
        0,
        num_pairs,
        size=(bootstrap_resamples, num_pairs),
    )
    bootstrap_difference_medians = np.empty((bootstrap_resamples, num_layers), dtype=np.float64)
    for bootstrap_index, pair_indices in enumerate(resample_indices):
        bootstrap_difference_medians[bootstrap_index] = np.median(differences[pair_indices], axis=0)

    upper_bounds = np.percentile(
        bootstrap_difference_medians,
        100.0 * MCB_CONFIDENCE_LEVEL,
        axis=0,
    )
    members = upper_bounds >= 0.0
    # Every observed maximizer is, by definition, a candidate for the best;
    # forcing exact ties also avoids arbitrary removal by the lowest-index
    # reference-layer tie break.
    members[tied_layers] = True

    layer_results = [
        {
            "layer_index": layer,
            "median_paired_difference_from_peak": float(observed_difference_medians[layer]),
            "upper_bound_one_sided_95": float(upper_bounds[layer]),
            "is_member": bool(members[layer]),
        }
        for layer in range(num_layers)
    ]
    return {
        "method": "paired-bootstrap-median-difference-hsu-mcb/v1",
        "reference_peak_layer_index": peak_layer,
        "confidence_level": MCB_CONFIDENCE_LEVEL,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "member_layer_indices": [int(layer) for layer in np.flatnonzero(members)],
        "layers": layer_results,
    }


def _summarize_depth_thirds(scores: np.ndarray) -> list[dict[str, Any]]:
    """Aggregate pair-first means in layer-center depth thirds."""

    _, num_layers = scores.shape
    centers = (np.arange(num_layers, dtype=np.float64) + 0.5) / num_layers
    third_specs = (
        ("early", centers < 1.0 / 3.0),
        ("middle", (centers >= 1.0 / 3.0) & (centers < 2.0 / 3.0)),
        ("late", centers >= 2.0 / 3.0),
    )

    summaries: list[dict[str, Any]] = []
    for name, mask in third_specs:
        layer_indices = np.flatnonzero(mask)
        setting_value: float | None
        if layer_indices.size == 0:
            setting_value = None
        else:
            pair_means = np.mean(scores[:, layer_indices], axis=1)
            setting_value = float(np.median(pair_means))
        summaries.append(
            {
                "name": name,
                "layer_indices": [int(layer) for layer in layer_indices],
                "setting_median_pair_mean": setting_value,
            }
        )
    return summaries
