"""Pure gold-correctness metrics for clean-prefix trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixTrajectorySummary:
    """Paper-defined point, stable-prefix, and non-monotonicity readouts."""

    k0_correct: bool
    full_correct: bool
    k_star: int | None
    r_star: float | None
    correctness_transitions: int
    non_monotone: bool
    stable_at_k: dict[int, bool]
    stable_le_0_2: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "k0_correct": self.k0_correct,
            "full_correct": self.full_correct,
            "k_star": self.k_star,
            "r_star": self.r_star,
            "correctness_transitions": self.correctness_transitions,
            "non_monotone": self.non_monotone,
            "stable_at_k": dict(self.stable_at_k),
            "stable_le_0_2": self.stable_le_0_2,
        }


def summarize_prefix_correctness(
    points: Sequence[tuple[int, bool]],
    *,
    cot_token_count: int,
) -> PrefixTrajectorySummary:
    """Summarize one complete, distinct-k gold-correctness trajectory."""

    if (
        not isinstance(cot_token_count, int)
        or isinstance(cot_token_count, bool)
        or cot_token_count <= 0
    ):
        raise ValueError("cot_token_count must be a positive integer")
    try:
        raw_points = tuple(points)
    except TypeError as exc:
        raise ValueError("prefix correctness points must be a sequence") from exc
    if not raw_points:
        raise ValueError("prefix correctness points must not be empty")

    correctness_by_k: dict[int, bool] = {}
    for point in raw_points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError("each prefix correctness point must be a (k, bool) pair")
        k, correct = point
        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= cot_token_count:
            raise ValueError("point budget k must be between zero and cot_token_count")
        if type(correct) is not bool:
            raise ValueError("point correctness must be boolean")
        if k in correctness_by_k:
            raise ValueError("prefix correctness points must use distinct k values")
        correctness_by_k[k] = correct

    if 0 not in correctness_by_k or cot_token_count not in correctness_by_k:
        raise ValueError("prefix correctness points must include k=0 and k=cot_token_count")

    ordered_k = tuple(sorted(correctness_by_k))
    flags = tuple(correctness_by_k[k] for k in ordered_k)
    transitions = sum(left != right for left, right in zip(flags, flags[1:]))

    stable_at_k: dict[int, bool] = {}
    suffix_is_correct = True
    for k in reversed(ordered_k):
        suffix_is_correct = correctness_by_k[k] and suffix_is_correct
        stable_at_k[k] = suffix_is_correct
    stable_at_k = {k: stable_at_k[k] for k in ordered_k}
    k_star = next((k for k in ordered_k if stable_at_k[k]), None)
    r_star = k_star / cot_token_count if k_star is not None else None

    return PrefixTrajectorySummary(
        k0_correct=correctness_by_k[0],
        full_correct=correctness_by_k[cot_token_count],
        k_star=k_star,
        r_star=r_star,
        correctness_transitions=transitions,
        non_monotone=transitions >= 2,
        stable_at_k=stable_at_k,
        # Cross multiplication makes the submitted <= .2 event exact.
        stable_le_0_2=k_star is not None and 5 * k_star <= cot_token_count,
    )


__all__ = ["PrefixTrajectorySummary", "summarize_prefix_correctness"]
