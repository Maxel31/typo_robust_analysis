"""Deterministic paired and cross-setting inference for patch controls."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

_BOOTSTRAP_METHOD = "percentile-paired-bootstrap-pcg64/v1"
_NESTED_METHOD = "equal-setting-nested-bootstrap-pcg64/v1"


def _numpy() -> Any:
    """Load the GPU-extra numerical dependency only when analysis runs."""

    import numpy

    return numpy


def _events(values: Sequence[bool], *, field: str) -> Any:
    np = _numpy()
    events = tuple(values)
    if not events or any(type(value) is not bool for value in events):
        raise ValueError(f"{field} must be a non-empty boolean sequence")
    return np.asarray(events, dtype=np.float64)


def _bootstrap_arguments(
    *, replicates: int, confidence_level: float, seed: int
) -> tuple[int, float, int]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("bootstrap confidence_level must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    return replicates, float(confidence_level), seed


def derived_seed(seed: int, label: str) -> int:
    """Derive an order-independent PCG64 seed for one named contrast."""

    _bootstrap_arguments(replicates=1, confidence_level=0.95, seed=seed)
    if not isinstance(label, str) or not label:
        raise ValueError("bootstrap seed label must be non-empty")
    digest = hashlib.sha256(f"six-setting-bootstrap/v1\0{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _interval(values: Any, confidence_level: float) -> tuple[float, float]:
    np = _numpy()
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(lower), float(upper)


def paired_bootstrap_risk_difference(
    correct: Sequence[bool],
    control: Sequence[bool],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the paired success-rate difference as a proportion."""

    replicates, confidence_level, seed = _bootstrap_arguments(
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    left = _events(correct, field="correct events")
    right = _events(control, field="control events")
    if left.size != right.size:
        raise ValueError("paired bootstrap arms must contain the same pairs")
    differences = left - right
    np = _numpy()
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, differences.size, size=(replicates, differences.size))
    samples = differences[indices].mean(axis=1)
    lower, upper = _interval(samples, confidence_level)
    return {
        "estimate": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "method": _BOOTSTRAP_METHOD,
    }


def nested_macro_bootstrap_risk_difference(
    setting_pairs: Mapping[str, tuple[Sequence[bool], Sequence[bool]]],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Resample settings, then pairs within each sampled setting."""

    replicates, confidence_level, seed = _bootstrap_arguments(
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    if not setting_pairs:
        raise ValueError("nested bootstrap requires at least one setting")
    np = _numpy()
    labels = tuple(sorted(setting_pairs))
    differences: dict[str, Any] = {}
    for label in labels:
        correct, control = setting_pairs[label]
        left = _events(correct, field=f"{label} correct events")
        right = _events(control, field=f"{label} control events")
        if left.size != right.size:
            raise ValueError(f"nested bootstrap arms differ in {label}")
        differences[label] = left - right
    estimate = float(np.mean([values.mean() for values in differences.values()]))
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(labels), size=len(labels))
        setting_estimates: list[float] = []
        for setting_index in selected:
            values = differences[labels[int(setting_index)]]
            pair_indices = rng.integers(0, values.size, size=values.size)
            setting_estimates.append(float(values[pair_indices].mean()))
        samples[replicate] = float(np.mean(setting_estimates))
    lower, upper = _interval(samples, confidence_level)
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "settings": len(labels),
        "method": _NESTED_METHOD,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise adjustment with stable tie ordering."""

    if not p_values:
        raise ValueError("Holm adjustment requires at least one p-value")
    normalized: dict[str, float] = {}
    for label, value in p_values.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Holm labels must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"Holm p-value is invalid for {label!r}")
        normalized[label] = float(value)
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[label] = running
    return {label: adjusted[label] for label in normalized}


__all__ = [
    "derived_seed",
    "holm_adjust",
    "nested_macro_bootstrap_risk_difference",
    "paired_bootstrap_risk_difference",
]
