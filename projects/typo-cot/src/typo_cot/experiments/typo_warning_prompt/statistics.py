"""Integer-only paired statistics for typo-warning outcomes."""

from __future__ import annotations

import math


def exact_mcnemar(*, b10: int, b01: int) -> dict[str, object]:
    """Return the exact two-sided McNemar/binomial result for discordant pairs."""

    for field, value in (("b10", b10), ("b01", b01)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    discordant = b10 + b01
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(b10, b01)
        numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
        p_value = min(1.0, numerator / (2**discordant))
    return {
        "b10": b10,
        "b01": b01,
        "discordant": discordant,
        "p_value": p_value,
        "method": "exact-two-sided-mcnemar-binomial/v1",
    }
