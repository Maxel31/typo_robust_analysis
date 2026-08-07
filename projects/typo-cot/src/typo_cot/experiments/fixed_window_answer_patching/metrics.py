"""Paired binary summaries for fixed-window answer comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42
PAIRED_BOOTSTRAP_METHOD = "paired-percentile-bootstrap-binary-risk-difference/v1"


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a descriptive 95% Wilson score interval."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return (center - half) / denominator, (center + half) / denominator


def _binary_events(values: Sequence[bool], *, field: str) -> tuple[bool, ...]:
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of booleans") from exc
    if not materialized:
        raise ValueError(f"{field} must not be empty")
    if any(type(value) is not bool for value in materialized):
        raise ValueError(f"{field} must contain only booleans")
    return materialized


def paired_binary_difference(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare paired binary rates with a pair-resampled bootstrap interval.

    Each bootstrap draw samples pair indices once and applies those indices to
    both arms.  This retains the within-pair dependence required by the final
    paper's prespecified MMLU-Pro window comparison.
    """

    left_events = _binary_events(left, field="left events")
    right_events = _binary_events(right, field="right events")
    if len(left_events) != len(right_events):
        raise ValueError("left and right events must contain the same pairs")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    left_array = np.asarray(left_events, dtype=np.float64)
    right_array = np.asarray(right_events, dtype=np.float64)
    paired_differences = left_array - right_array
    pair_count = len(left_events)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, pair_count, size=(resamples, pair_count))
    bootstrap_differences = np.mean(paired_differences[sampled_indices], axis=1)
    low, high = np.percentile(bootstrap_differences, (2.5, 97.5))

    left_successes = int(np.sum(left_array))
    right_successes = int(np.sum(right_array))
    return {
        "method": PAIRED_BOOTSTRAP_METHOD,
        "pairs": pair_count,
        "left_successes": left_successes,
        "right_successes": right_successes,
        "left_rate": left_successes / pair_count,
        "right_rate": right_successes / pair_count,
        "difference": float(np.mean(paired_differences)),
        "confidence_level": 0.95,
        "confidence_interval": [float(low), float(high)],
        "bootstrap_resamples": resamples,
        "seed": seed,
    }
