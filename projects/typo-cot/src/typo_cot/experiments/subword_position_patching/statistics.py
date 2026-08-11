"""Three-arm paired omnibus inference for subword-position patching."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

_METHOD = "cochran-q-chi-square-df2/v1"


def _events(values: Sequence[bool], *, field: str) -> tuple[bool, ...]:
    events = tuple(values)
    if not events or any(type(value) is not bool for value in events):
        raise ValueError(f"{field} must be a non-empty boolean sequence")
    return events


def cochran_q_three(arms: Mapping[str, Sequence[bool]]) -> dict[str, object]:
    """Run Cochran's Q and the chi-square(df=2) reference test on three arms."""

    if len(arms) != 3 or any(not isinstance(label, str) or not label for label in arms):
        raise ValueError("subword Cochran's Q requires exactly three named arms")
    normalized = {label: _events(values, field=f"{label} events") for label, values in arms.items()}
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("subword Cochran's Q arms must contain the same paired items")
    labels = tuple(normalized)
    pairs = next(iter(lengths))
    column_sums = [sum(normalized[label]) for label in labels]
    row_sums = [sum(normalized[label][index] for label in labels) for index in range(pairs)]
    total = sum(column_sums)
    numerator = 2 * (3 * sum(value * value for value in column_sums) - total * total)
    denominator = 3 * total - sum(value * value for value in row_sums)
    statistic = 0.0 if denominator == 0 else numerator / denominator
    if statistic < 0.0 or not math.isfinite(statistic):
        raise ValueError("subword Cochran's Q produced an invalid statistic")
    return {
        "method": _METHOD,
        "pairs": pairs,
        "arms": 3,
        "degrees_of_freedom": 2,
        "statistic": float(statistic),
        "p_value": math.exp(-float(statistic) / 2.0),
        "arm_order": list(labels),
        "successes": dict(zip(labels, column_sums, strict=True)),
    }


__all__ = ["cochran_q_three"]
