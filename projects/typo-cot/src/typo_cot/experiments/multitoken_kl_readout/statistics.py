"""Pair-bootstrap summaries for multi-token KL estimands."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _numpy() -> Any:
    """Load the GPU-extra numerical dependency only when analysis runs."""

    import numpy

    return numpy


def bootstrap_median(
    values: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    """Return a deterministic percentile interval for the sample median."""

    np = _numpy()
    try:
        array = np.asarray(tuple(values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("bootstrap values must be numeric") from exc
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    rng = np.random.default_rng(seed)
    medians = np.empty(replicates, dtype=np.float64)
    chunk_size = 512
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        medians[start:stop] = np.median(array[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(medians, (alpha, 1.0 - alpha))
    return {
        "estimate": float(np.median(array)),
        "lower": float(lower),
        "upper": float(upper),
    }


__all__ = ["bootstrap_median"]
