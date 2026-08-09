"""Statistical contracts for the CPU-only one-token table builder.

The submitted analyses treat ``(benchmark, sample_id)`` as the resampling
cluster.  These tests deliberately use plain mappings and callbacks so the
statistics remain independent of the artifact-loading and rendering layers.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from typo_cot.experiments.build_one_token_tables.statistics import (
    cluster_robust_mean_ci,
    cluster_sign_flip,
    percentile_cluster_bootstrap,
)


Row = Mapping[str, object]
Value = Callable[[Row], float]
ClusterKey = Callable[[Row], tuple[str, str]]


def _row(
    benchmark: str,
    sample_id: str,
    delta: float,
    *,
    cohort: str = "extension",
) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "sample_id": sample_id,
        "delta": delta,
        "cohort": cohort,
    }


def _value(row: Row) -> float:
    return float(row["delta"])


def _cluster_key(row: Row) -> tuple[str, str]:
    return str(row["benchmark"]), str(row["sample_id"])


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _reference_cluster_bootstrap(
    rows: Sequence[Row],
    *,
    draws: int,
    seed: int,
    label: str,
    value: Value = _value,
    cluster_key: ClusterKey = _cluster_key,
) -> tuple[int, np.ndarray]:
    """Independent transcription of the submitted NumPy percentile method."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[cluster_key(row)].append(value(row))
    clusters = sorted(grouped)
    cluster_n = np.asarray([len(grouped[key]) for key in clusters], dtype=np.int64)
    cluster_total = np.asarray(
        [sum(grouped[key]) for key in clusters],
        dtype=np.float64,
    )
    derived_seed = _derived_seed(seed, label)
    rng = np.random.default_rng(derived_seed)
    sampled = rng.integers(0, len(clusters), size=(draws, len(clusters)))
    estimates = cluster_total[sampled].sum(axis=1) / cluster_n[sampled].sum(axis=1)
    return derived_seed, np.quantile(estimates, [0.025, 0.975])


def _reference_sign_flip(
    rows: Sequence[Row],
    *,
    draws: int,
    seed: int,
    value: Value = _value,
    cluster_key: ClusterKey = _cluster_key,
) -> tuple[float, int]:
    """Reproduce cluster aggregation, sign draws, and plus-one correction."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[cluster_key(row)].append(value(row))
    contributions = [sum(grouped[key]) for key in sorted(grouped)]
    nonzero = [contribution for contribution in contributions if contribution != 0]
    observed = sum(nonzero)
    if not nonzero:
        return 1.0, draws

    magnitude_counts = Counter(abs(contribution) for contribution in nonzero)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        simulated = 0.0
        for magnitude in sorted(magnitude_counts):
            count = magnitude_counts[magnitude]
            positive = rng.getrandbits(count).bit_count()
            simulated += magnitude * (2 * positive - count)
        extreme += abs(simulated) >= abs(observed) - 1e-12
    return (extreme + 1) / (draws + 1), extreme


def test_cluster_robust_ci_uses_student_t_critical_value() -> None:
    """Three clusters imply t(2), whose interval is much wider than z=1.96."""
    rows = [
        _row("a", "one", 1.0),
        _row("a", "one", 1.0),
        _row("a", "two", -1.0),
        _row("a", "two", -1.0),
        _row("b", "three", 0.0),
        _row("b", "three", 0.0),
    ]

    result = cluster_robust_mean_ci(
        rows,
        value=_value,
        cluster_key=_cluster_key,
    )

    # Intercept-only CR1: G/(G-1) * sum_g(sum_i residual_i)^2 / N^2.
    expected_se = math.sqrt((3 / 2) * (2**2 + (-2) ** 2 + 0**2) / 6**2)
    t_975_df2 = 4.302652729911275
    assert result["n_records"] == 6
    assert result["n_clusters"] == 3
    assert result["estimate"] == pytest.approx(0.0)
    assert result["se_cluster"] == pytest.approx(expected_se)
    assert result["ci95_low"] == pytest.approx(-t_975_df2 * expected_se)
    assert result["ci95_high"] == pytest.approx(t_975_df2 * expected_se)
    assert result["ci95_high"] > 1.96 * expected_se
    assert result["p_cluster_t"] == pytest.approx(1.0)


def test_cluster_robust_ci_is_invariant_to_input_order() -> None:
    rows = [
        _row("b", "2", 0.5),
        _row("a", "1", 1.0),
        _row("b", "2", -0.5),
        _row("c", "3", -1.0),
    ]

    forward = cluster_robust_mean_ci(rows, value=_value, cluster_key=_cluster_key)
    reverse = cluster_robust_mean_ci(
        list(reversed(rows)),
        value=_value,
        cluster_key=_cluster_key,
    )

    assert reverse == forward


def test_percentile_cluster_bootstrap_matches_sha_derived_numpy_method() -> None:
    """Unequal cluster sizes catch resampling cluster means instead of records."""
    rows = [
        _row("a", "one", 1.0),
        _row("a", "one", 1.0),
        _row("a", "two", -1.0),
        _row("b", "three", 0.0),
        _row("b", "three", 0.0),
        _row("b", "three", 0.0),
    ]
    draws = 4_096
    seed = 42
    label = "table11/all-extensions/distant-factorial"
    expected_seed, expected_interval = _reference_cluster_bootstrap(
        rows,
        draws=draws,
        seed=seed,
        label=label,
    )

    result = percentile_cluster_bootstrap(
        rows,
        value=_value,
        cluster_key=_cluster_key,
        draws=draws,
        seed=seed,
        label=label,
    )

    assert result["method"] == "percentile cluster bootstrap"
    assert result["n_records"] == len(rows)
    assert result["n_clusters"] == 3
    assert result["estimate"] == pytest.approx(1 / 6)
    assert result["draws"] == draws
    assert result["base_seed"] == seed
    assert result["derived_seed"] == expected_seed
    assert result["ci95_low"] == pytest.approx(expected_interval[0])
    assert result["ci95_high"] == pytest.approx(expected_interval[1])


def test_percentile_cluster_bootstrap_is_invariant_to_input_order() -> None:
    rows = [
        _row("b", "2", 0.0),
        _row("a", "1", 1.0),
        _row("b", "2", -0.5),
        _row("c", "3", -1.0),
    ]
    kwargs = {
        "value": _value,
        "cluster_key": _cluster_key,
        "draws": 1_024,
        "seed": 73,
        "label": "stable-population-label",
    }

    forward = percentile_cluster_bootstrap(rows, **kwargs)
    reverse = percentile_cluster_bootstrap(list(reversed(rows)), **kwargs)

    assert reverse == forward


def test_cluster_sign_flip_aggregates_records_before_drawing_signs() -> None:
    rows = [
        _row("a", "one", 0.5),
        _row("a", "one", 0.5),
        _row("b", "two", -0.5),
        _row("c", "three", 0.25),
        _row("c", "three", 0.25),
    ]
    draws = 2_048
    seed = 42
    expected_p, expected_extreme = _reference_sign_flip(
        rows,
        draws=draws,
        seed=seed,
    )

    result = cluster_sign_flip(
        rows,
        value=_value,
        cluster_key=_cluster_key,
        draws=draws,
        seed=seed,
    )

    assert result["method"] == "Monte Carlo cluster sign-flip with plus-one correction"
    assert result["n_records"] == 5
    assert result["n_clusters"] == 3
    assert result["n_nonzero_clusters"] == 3
    assert result["estimate"] == pytest.approx(0.2)
    assert result["observed_sum"] == pytest.approx(1.0)
    assert result["n_extreme_draws"] == expected_extreme
    assert result["p_two_sided"] == pytest.approx(expected_p)
    assert result["draws"] == draws
    assert result["seed"] == seed


def test_cluster_sign_flip_submission_defaults_and_input_order_are_stable() -> None:
    parameters = inspect.signature(cluster_sign_flip).parameters
    assert parameters["draws"].default == 200_000
    assert parameters["seed"].default == 42

    rows = [
        _row("c", "3", 0.5),
        _row("a", "1", 1.0),
        _row("b", "2", -0.5),
        _row("c", "3", 0.0),
    ]
    kwargs = {
        "value": _value,
        "cluster_key": _cluster_key,
        "draws": 1_024,
        "seed": 9,
    }

    forward = cluster_sign_flip(rows, **kwargs)
    reverse = cluster_sign_flip(list(reversed(rows)), **kwargs)

    assert reverse == forward


def test_all_zero_differences_have_degenerate_intervals_and_null_p_value() -> None:
    rows = [
        _row("a", "one", 0.0),
        _row("a", "one", 0.0),
        _row("b", "two", 0.0),
    ]

    robust = cluster_robust_mean_ci(rows, value=_value, cluster_key=_cluster_key)
    bootstrap = percentile_cluster_bootstrap(
        rows,
        value=_value,
        cluster_key=_cluster_key,
        draws=128,
        seed=42,
        label="zero-difference",
    )
    sign_flip = cluster_sign_flip(
        rows,
        value=_value,
        cluster_key=_cluster_key,
        draws=128,
        seed=42,
    )

    assert robust["estimate"] == 0.0
    assert robust["se_cluster"] == 0.0
    assert robust["ci95_low"] == 0.0
    assert robust["ci95_high"] == 0.0
    assert robust["p_cluster_t"] == 1.0
    assert bootstrap["estimate"] == 0.0
    assert bootstrap["ci95_low"] == 0.0
    assert bootstrap["ci95_high"] == 0.0
    assert sign_flip["estimate"] == 0.0
    assert sign_flip["observed_sum"] == 0.0
    assert sign_flip["n_nonzero_clusters"] == 0
    assert sign_flip["n_extreme_draws"] == 128
    assert sign_flip["p_two_sided"] == 1.0
