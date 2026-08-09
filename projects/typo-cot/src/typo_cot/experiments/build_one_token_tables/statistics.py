"""Deterministic clustered inference used by the one-token table builder."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TypeVar

import numpy as np


Row = TypeVar("Row", bound=Mapping[str, object])


def _finite_values(
    rows: Sequence[Row],
    *,
    value: Callable[[Row], float],
    cluster_key: Callable[[Row], Hashable],
) -> tuple[list[float], dict[Hashable, list[float]]]:
    if not rows:
        raise ValueError("clustered inference requires at least one record")
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    values: list[float] = []
    for row in rows:
        number = float(value(row))
        if not math.isfinite(number):
            raise ValueError("clustered inference values must be finite")
        key = cluster_key(row)
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError("cluster keys must be hashable") from exc
        grouped[key].append(number)
        values.append(number)
    return values, grouped


def _ordered_clusters(grouped: Mapping[Hashable, Sequence[float]]) -> list[Hashable]:
    """Give heterogeneous but hashable keys a deterministic total order."""

    return sorted(grouped, key=lambda key: (type(key).__qualname__, repr(key)))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction (Numerical Recipes)."""

    maximum_iterations = 300
    epsilon = 3e-15
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail_twice = _regularized_beta(x, degrees_of_freedom / 2.0, 0.5)
    return 1.0 - 0.5 * tail_twice if value > 0 else 0.5 * tail_twice


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not 0.5 < probability < 1.0:
        raise ValueError("only upper Student-t quantiles are supported")
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
        if upper > 1e12:
            raise ArithmeticError("Student-t quantile search did not bracket the result")
    for _ in range(160):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def cluster_robust_mean_ci(
    rows: Sequence[Row],
    *,
    value: Callable[[Row], float],
    cluster_key: Callable[[Row], Hashable],
) -> dict[str, int | float | str]:
    """Return the intercept-only CR1 mean interval used for distant controls."""

    values, grouped = _finite_values(rows, value=value, cluster_key=cluster_key)
    n_records = len(values)
    n_clusters = len(grouped)
    if n_clusters < 2:
        raise ValueError("cluster-robust inference requires at least two clusters")
    estimate = math.fsum(sorted(values)) / n_records
    cluster_residual_sums = [
        math.fsum(number - estimate for number in sorted(grouped[key]))
        for key in _ordered_clusters(grouped)
    ]
    variance = (
        n_clusters
        / (n_clusters - 1)
        * math.fsum(residual * residual for residual in cluster_residual_sums)
        / (n_records * n_records)
    )
    standard_error = math.sqrt(max(variance, 0.0))
    degrees_of_freedom = n_clusters - 1
    critical = _student_t_quantile(0.975, degrees_of_freedom)
    if standard_error == 0.0:
        p_value = 1.0 if estimate == 0.0 else 0.0
    else:
        p_value = 2.0 * (1.0 - _student_t_cdf(abs(estimate / standard_error), degrees_of_freedom))
    return {
        "method": "intercept-only CR1 cluster-robust Student-t interval",
        "n_records": n_records,
        "n_clusters": n_clusters,
        "estimate": estimate,
        "se_cluster": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value_975": critical,
        "ci95_low": estimate - critical * standard_error,
        "ci95_high": estimate + critical * standard_error,
        "p_cluster_t": max(0.0, min(1.0, p_value)),
    }


def _derived_seed(seed: int, label: str) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(label, str) or not label:
        raise ValueError("bootstrap label must be a non-empty string")
    digest = hashlib.sha256(f"{seed}|{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def percentile_cluster_bootstrap(
    rows: Sequence[Row],
    *,
    value: Callable[[Row], float],
    cluster_key: Callable[[Row], Hashable],
    draws: int = 50_000,
    seed: int = 42,
    label: str,
) -> dict[str, int | float | str]:
    """Return the submitted NumPy percentile interval for adjacent controls."""

    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")
    values, grouped = _finite_values(rows, value=value, cluster_key=cluster_key)
    clusters = _ordered_clusters(grouped)
    counts = np.asarray([len(grouped[key]) for key in clusters], dtype=np.int64)
    totals = np.asarray(
        [math.fsum(sorted(grouped[key])) for key in clusters],
        dtype=np.float64,
    )
    derived_seed = _derived_seed(seed, label)
    rng = np.random.default_rng(derived_seed)
    estimates = np.empty(draws, dtype=np.float64)
    # Preserve the exact row-major RNG stream while avoiding a potentially
    # multi-gigabyte draws-by-clusters index matrix on full paper cohorts.
    # The submitted adjacent-control aggregate generated 2,000 bootstrap rows
    # per RNG call. Freezing this boundary preserves its exact PCG64 stream.
    rows_per_chunk = 2_000
    offset = 0
    while offset < draws:
        chunk = min(rows_per_chunk, draws - offset)
        sampled = rng.integers(0, len(clusters), size=(chunk, len(clusters)))
        estimates[offset : offset + chunk] = totals[sampled].sum(axis=1) / counts[sampled].sum(
            axis=1
        )
        offset += chunk
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "method": "percentile cluster bootstrap",
        "n_records": len(values),
        "n_clusters": len(clusters),
        "estimate": math.fsum(sorted(values)) / len(values),
        "draws": draws,
        "base_seed": seed,
        "derived_seed": derived_seed,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def cluster_sign_flip(
    rows: Sequence[Row],
    *,
    value: Callable[[Row], float],
    cluster_key: Callable[[Row], Hashable],
    draws: int = 200_000,
    seed: int = 42,
) -> dict[str, int | float | str]:
    """Run the submitted two-sided Monte Carlo cluster sign-flip test."""

    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise ValueError("sign-flip draws must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    values, grouped = _finite_values(rows, value=value, cluster_key=cluster_key)
    contributions = [math.fsum(sorted(grouped[key])) for key in _ordered_clusters(grouped)]
    nonzero = [number for number in contributions if number != 0.0]
    observed = math.fsum(nonzero)
    if not nonzero:
        extreme = draws
    else:
        magnitude_counts = Counter(abs(number) for number in nonzero)
        rng = random.Random(seed)
        extreme = 0
        for _ in range(draws):
            simulated = 0.0
            for magnitude in sorted(magnitude_counts):
                count = magnitude_counts[magnitude]
                positive = rng.getrandbits(count).bit_count()
                simulated += magnitude * (2 * positive - count)
            extreme += abs(simulated) >= abs(observed) - 1e-12
    return {
        "method": "Monte Carlo cluster sign-flip with plus-one correction",
        "n_records": len(values),
        "n_clusters": len(grouped),
        "n_nonzero_clusters": len(nonzero),
        "estimate": math.fsum(sorted(values)) / len(values),
        "observed_sum": observed,
        "n_extreme_draws": extreme,
        "p_two_sided": (extreme + 1) / (draws + 1),
        "draws": draws,
        "seed": seed,
    }


__all__ = [
    "cluster_robust_mean_ci",
    "cluster_sign_flip",
    "percentile_cluster_bootstrap",
]
