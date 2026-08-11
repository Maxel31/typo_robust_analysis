"""Numerically explicit metrics for teacher-forced multi-token KL readout."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def kl_trajectory_from_logits(
    reference_logits: Any,
    comparison_logits: Any,
    *,
    negative_roundoff_tolerance: float = 1e-12,
) -> tuple[float, ...]:
    """Return token-wise ``KL(reference || comparison)`` in float64.

    Runtime callers first materialize model logits as CPU float32. This helper
    then converts them to float64 before log-softmax and reduction. Tiny
    negative roundoff is clamped to zero; a more negative value fails closed.
    """

    import torch

    if reference_logits.shape != comparison_logits.shape:
        raise ValueError("KL logits must have identical shapes")
    if reference_logits.ndim != 2 or reference_logits.shape[0] == 0:
        raise ValueError("KL logits must have shape [tokens, vocabulary]")
    if reference_logits.shape[1] == 0:
        raise ValueError("KL logits must contain a non-empty vocabulary")
    if reference_logits.dtype != torch.float32 or comparison_logits.dtype != torch.float32:
        raise ValueError("KL logits must be materialized as float32")
    if not math.isfinite(negative_roundoff_tolerance) or negative_roundoff_tolerance < 0.0:
        raise ValueError("negative_roundoff_tolerance must be finite and non-negative")

    reference_log_probs = torch.log_softmax(reference_logits.double(), dim=-1)
    comparison_log_probs = torch.log_softmax(comparison_logits.double(), dim=-1)
    values = torch.sum(
        reference_log_probs.exp() * (reference_log_probs - comparison_log_probs),
        dim=-1,
    )
    result: list[float] = []
    for raw_value in values.detach().cpu().tolist():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("nonfinite_kl")
        if value < -negative_roundoff_tolerance:
            raise ValueError("negative_kl_beyond_roundoff_tolerance")
        result.append(max(value, 0.0))
    return tuple(result)


def _trajectory(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric sequence") from exc
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError(f"{field} must contain finite non-negative KL values")
    return normalized


def restoration_score(
    *,
    untreated_kl: Sequence[float],
    patched_kl: Sequence[float],
    token_range: tuple[int, int],
    denominator_epsilon: float,
) -> float:
    """Return unclipped normalized restoration for a 1-indexed inclusive range."""

    untreated = _trajectory(untreated_kl, field="untreated_kl")
    patched = _trajectory(patched_kl, field="patched_kl")
    if len(untreated) != len(patched):
        raise ValueError("KL trajectories must have equal length")
    if (
        not isinstance(token_range, tuple)
        or len(token_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_range)
    ):
        raise ValueError("token_range must be a two-integer tuple")
    start, stop = token_range
    if not 1 <= start <= stop <= len(untreated):
        raise ValueError("token_range is outside the KL trajectory")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0.0:
        raise ValueError("denominator_epsilon must be finite and positive")
    selection = slice(start - 1, stop)
    denominator = sum(untreated[selection]) / (stop - start + 1)
    if denominator <= denominator_epsilon:
        raise ValueError("denominator_le_1e-9")
    patched_mean = sum(patched[selection]) / (stop - start + 1)
    value = 1.0 - patched_mean / denominator
    if not math.isfinite(value):
        raise ValueError("nonfinite_restoration")
    return value


__all__ = ["kl_trajectory_from_logits", "restoration_score"]
