"""Omnibus inference for four paired binary source/write arms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

_METHOD = "cochran-q-chi-square-df3/v1"


def _events(values: Sequence[bool], *, field: str) -> tuple[bool, ...]:
    result = tuple(values)
    if not result or any(type(value) is not bool for value in result):
        raise ValueError(f"{field} must be a non-empty boolean sequence")
    return result


def _chi_square_three_survival(statistic: float) -> float:
    if statistic <= 0.0:
        return 1.0
    z = statistic / 2.0
    return min(
        1.0,
        max(
            0.0,
            math.erfc(math.sqrt(z)) + 2.0 / math.sqrt(math.pi) * math.sqrt(z) * math.exp(-z),
        ),
    )


def cochran_q(arms: Mapping[str, Sequence[bool]]) -> dict[str, object]:
    """Run Cochran's Q on exactly four paired binary arms."""

    if len(arms) != 4 or any(not isinstance(label, str) or not label for label in arms):
        raise ValueError("Cochran's Q requires four uniquely named arms")
    normalized = {label: _events(values, field=f"{label} events") for label, values in arms.items()}
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("Cochran's Q arms must contain the same paired items")
    labels = tuple(normalized)
    pairs = next(iter(lengths))
    column_sums = [sum(normalized[label]) for label in labels]
    row_sums = [sum(normalized[label][index] for label in labels) for index in range(pairs)]
    arm_count = len(labels)
    total = sum(column_sums)
    numerator = (arm_count - 1) * (
        arm_count * sum(value * value for value in column_sums) - total * total
    )
    denominator = arm_count * total - sum(value * value for value in row_sums)
    statistic = 0.0 if denominator == 0 else numerator / denominator
    if statistic < 0.0 or not math.isfinite(statistic):
        raise ValueError("Cochran's Q produced an invalid statistic")
    return {
        "method": _METHOD,
        "pairs": pairs,
        "arms": arm_count,
        "degrees_of_freedom": arm_count - 1,
        "statistic": float(statistic),
        "p_value": _chi_square_three_survival(float(statistic)),
        "arm_order": list(labels),
        "successes": dict(zip(labels, column_sums, strict=True)),
    }


__all__ = ["cochran_q"]
