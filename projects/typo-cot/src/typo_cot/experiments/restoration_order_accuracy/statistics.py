"""Dependency-free paired statistics used by the Table 13 builder."""

from __future__ import annotations

import math
from fractions import Fraction


def exact_mcnemar_p_value(first_only: int, second_only: int) -> float:
    """Return the two-sided exact McNemar/binomial p value at ``p=0.5``."""
    for name, value in (("first_only", first_only), ("second_only", second_only)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    numerator = sum(math.comb(discordant, count) for count in range(tail + 1))
    probability = Fraction(2 * numerator, 1 << discordant)
    return float(min(Fraction(1), probability))


__all__ = ["exact_mcnemar_p_value"]
