"""Binary fixed-denominator summaries for free-answer layer scans."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 42


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the descriptive 95% Wilson score interval."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return (center - half) / denominator, (center + half) / denominator


def _event_matrix(pair_events: Sequence[Sequence[bool]]) -> np.ndarray:
    try:
        materialized = tuple(tuple(row) for row in pair_events)
    except TypeError as exc:
        raise ValueError("pair events must be a two-dimensional sequence") from exc
    if not materialized or not materialized[0]:
        raise ValueError("at least one complete pair-by-layer event grid is required")
    layer_count = len(materialized[0])
    if any(len(row) != layer_count for row in materialized):
        raise ValueError("pair event grids must have the same number of layers")
    if any(type(value) is not bool for row in materialized for value in row):
        raise ValueError("pair event grids must contain booleans")
    return np.asarray(materialized, dtype=np.float64)


def _paired_binary_mcb(
    events: np.ndarray,
    *,
    peak_layer: int,
    tied_layers: np.ndarray,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    pair_count, layer_count = events.shape
    differences = events - events[:, [peak_layer]]
    observed = np.mean(differences, axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, pair_count, size=(bootstrap_resamples, pair_count))
    bootstrapped = np.empty((bootstrap_resamples, layer_count), dtype=np.float64)
    for bootstrap_index, sampled in enumerate(indices):
        bootstrapped[bootstrap_index] = np.mean(differences[sampled], axis=0)
    upper = np.percentile(bootstrapped, 95.0, axis=0)
    members = upper >= 0.0
    members[tied_layers] = True
    return {
        "method": "paired-bootstrap-binary-risk-difference-hsu-mcb/v1",
        "reference_peak_layer_index": peak_layer,
        "confidence_level": 0.95,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "member_layer_indices": [int(index) for index in np.flatnonzero(members)],
        "layers": [
            {
                "layer_index": layer,
                "mean_paired_difference_from_peak": float(observed[layer]),
                "upper_bound_one_sided_95": float(upper[layer]),
                "is_member": bool(members[layer]),
            }
            for layer in range(layer_count)
        ],
    }


def summarize_binary_layers(
    pair_events: Sequence[Sequence[bool]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize a complete fixed-cohort binary layer grid."""

    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples <= 0
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    events = _event_matrix(pair_events)
    pair_count, layer_count = events.shape
    successes = np.sum(events, axis=0).astype(int)
    rates = successes / pair_count
    maximum = float(np.max(rates))
    tied = np.flatnonzero(rates == maximum)
    peak = int(tied[0])
    profile: list[dict[str, object]] = []
    for layer in range(layer_count):
        low, high = wilson_interval(int(successes[layer]), pair_count)
        profile.append(
            {
                "layer_index": layer,
                "relative_depth": layer / layer_count,
                "layer_center_relative_depth": (layer + 0.5) / layer_count,
                "successes": int(successes[layer]),
                "total": pair_count,
                "rate": float(rates[layer]),
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return {
        "included_pairs": pair_count,
        "num_layers": layer_count,
        "layer_profile": profile,
        "peak": {
            "layer_index": peak,
            "tied_layer_indices": [int(index) for index in tied],
            "relative_depth": peak / layer_count,
            "rate": maximum,
        },
        "mcb": _paired_binary_mcb(
            events,
            peak_layer=peak,
            tied_layers=tied,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
    }
