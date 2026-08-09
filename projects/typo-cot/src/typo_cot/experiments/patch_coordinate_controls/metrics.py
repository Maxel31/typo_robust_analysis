"""Exact paired binary statistics for the coordinate-control experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal, localcontext
from typing import Any

EXACT_MCNEMAR_METHOD = "exact-mcnemar-two-sided-conditional-binomial/v1"


def _events(values: Sequence[bool], *, field: str) -> tuple[bool, ...]:
    try:
        events = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of booleans") from exc
    if not events:
        raise ValueError(f"{field} must not be empty")
    if any(type(event) is not bool for event in events):
        raise ValueError(f"{field} must contain only booleans")
    return events


def _two_sided_binomial_half_p(left_only: int, right_only: int) -> float:
    """Return the exact two-sided conditional-binomial probability.

    Under the McNemar null, either direction is equiprobable for each
    discordant pair.  Integer binomial coefficients keep the tail exact; a
    high-precision decimal division avoids a SciPy dependency and premature
    floating-point underflow in the published small-p-value comparisons.
    """

    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower_tail = min(left_only, right_only)
    numerator = 2 * sum(math.comb(discordant, k) for k in range(lower_tail + 1))
    denominator = 1 << discordant
    with localcontext() as context:
        context.prec = max(50, len(str(denominator)) + 5)
        probability = min(Decimal(1), Decimal(numerator) / Decimal(denominator))
    return float(probability)


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    """Summarize two paired event vectors and run exact two-sided McNemar.

    The test is the conditional exact binomial test over discordant pairs with
    null probability 0.5.  It is the analysis reported for the final paper's
    correct-coordinate versus offset and cross-item comparisons.
    """

    left_events = _events(left, field="left events")
    right_events = _events(right, field="right events")
    if len(left_events) != len(right_events):
        raise ValueError("left and right events must contain the same pairs")

    both_success = 0
    left_only = 0
    right_only = 0
    both_failure = 0
    for left_event, right_event in zip(left_events, right_events, strict=True):
        if left_event and right_event:
            both_success += 1
        elif left_event:
            left_only += 1
        elif right_event:
            right_only += 1
        else:
            both_failure += 1

    pairs = len(left_events)
    left_successes = both_success + left_only
    right_successes = both_success + right_only
    discordant = left_only + right_only
    return {
        "method": EXACT_MCNEMAR_METHOD,
        "alternative": "two-sided",
        "null_discordant_probability": 0.5,
        "pairs": pairs,
        "left_successes": left_successes,
        "right_successes": right_successes,
        "left_rate": left_successes / pairs,
        "right_rate": right_successes / pairs,
        "rate_difference": (left_successes - right_successes) / pairs,
        "both_success": both_success,
        "left_only": left_only,
        "right_only": right_only,
        "both_failure": both_failure,
        "discordant": discordant,
        "p_value": _two_sided_binomial_half_p(left_only, right_only),
    }


__all__ = ["EXACT_MCNEMAR_METHOD", "exact_mcnemar"]
