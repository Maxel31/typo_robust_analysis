"""Recompute the single-layer causal gate from raw KL trajectories."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from typo_robust_training.state_gate.config import SingleLayerGateProtocol


_CONDITIONS = ("correct", "offset", "cross", "self_copy")
_SCIENTIFIC_INVALID_REASONS = frozenset(
    {
        "fewer-than-16-clean-corpus-tokens-after-final-edit",
        "untreated-kl-at-or-below-1e-9",
    }
)


def _kl_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gate observation KL values must be real JSON numbers")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(
            "gate observation KL values must be finite and non-negative"
        ) from exc
    if not finite or value < 0:
        raise ValueError("gate observation KL values must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class GateObservation:
    pair_id: str
    source_group_sha256: str
    stratum: str
    transition_layer: int
    untreated_kl_2_16: tuple[float, ...]
    patched_kl_2_16: Mapping[str, tuple[float, ...]]
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("gate observation pair_id must be non-empty")
        if not isinstance(self.source_group_sha256, str) or len(self.source_group_sha256) != 64:
            raise ValueError("gate observation source group must be one SHA-256")
        if not isinstance(self.stratum, str) or not self.stratum:
            raise ValueError("gate observation stratum must be non-empty")
        if (
            isinstance(self.transition_layer, bool)
            or not isinstance(self.transition_layer, int)
            or self.transition_layer < 1
        ):
            raise ValueError("gate observation transition layer must be positive")
        untreated_values = tuple(self.untreated_kl_2_16)
        patched_values = {
            key: tuple(values) for key, values in self.patched_kl_2_16.items()
        }
        untreated = tuple(_kl_number(value) for value in untreated_values)
        patched = {
            key: tuple(_kl_number(value) for value in values)
            for key, values in patched_values.items()
        }
        if untreated:
            if len(untreated) != 15 or set(patched) != set(_CONDITIONS):
                raise ValueError("valid gate observation must contain every R_2:16 condition")
            if any(len(values) != 15 for values in patched.values()):
                raise ValueError("gate condition trajectories must contain offsets 2 through 16")
            if self.invalid_reason is not None:
                raise ValueError("valid gate observation cannot have an invalid reason")
        elif patched or self.invalid_reason not in _SCIENTIFIC_INVALID_REASONS:
            raise ValueError(
                "invalid gate observation must contain one preregistered scientific reason"
            )
        object.__setattr__(self, "untreated_kl_2_16", untreated)
        object.__setattr__(self, "patched_kl_2_16", MappingProxyType(patched))

    def restorations(self) -> Mapping[str, float]:
        if not self.untreated_kl_2_16:
            raise ValueError("invalid gate observation has no restoration")
        denominator = sum(self.untreated_kl_2_16) / len(self.untreated_kl_2_16)
        return MappingProxyType(
            {
                condition: 1.0 - (sum(values) / len(values)) / denominator
                for condition, values in self.patched_kl_2_16.items()
            }
        )


@dataclass(frozen=True, slots=True)
class GateScore:
    estimate: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True, slots=True)
class GateScoreResult:
    valid_records: int
    valid_strata: Mapping[str, int]
    scores: Mapping[str, GateScore]
    maximum_absolute_self_copy_restoration: float
    failure_reasons: tuple[str, ...]
    passed: bool


def _interval(values: np.ndarray, *, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def score_single_layer_gate(
    observations: Sequence[GateObservation],
    *,
    protocol: SingleLayerGateProtocol,
    transition_layer: int,
) -> GateScoreResult:
    """Group-bootstrap raw trajectories; never trust an artifact's pass bit."""

    rows = tuple(observations)
    if len(rows) != protocol.records or len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("single-layer gate observations differ from the frozen cohort size")
    if any(row.transition_layer != transition_layer for row in rows):
        raise ValueError("single-layer gate patched a layer other than the parent transition")
    all_strata = Counter(row.stratum for row in rows)
    if all_strata != Counter(protocol.stratum_counts):
        raise ValueError("single-layer gate observation strata differ from preregistration")
    valid = tuple(row for row in rows if row.untreated_kl_2_16)
    observed_valid_strata = Counter(row.stratum for row in valid)
    valid_strata = {
        key: observed_valid_strata[key] for key in sorted(protocol.stratum_counts)
    }
    failures: list[str] = []
    if len(valid) < protocol.minimum_valid_records:
        failures.append("minimum-valid-records")
    failures.extend(
        f"minimum-valid-stratum:{key}"
        for key, minimum in sorted(protocol.minimum_valid_per_stratum.items())
        if valid_strata[key] < minimum
    )
    per_group: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    maximum_self = 0.0
    for row in valid:
        denominator = sum(row.untreated_kl_2_16) / 15
        if denominator <= protocol.denominator_min_exclusive:
            raise ValueError("valid gate observation violates the KL denominator threshold")
        values = row.restorations()
        maximum_self = max(maximum_self, abs(values["self_copy"]))
        per_group[row.source_group_sha256].append(values)
    if len(per_group) < 2:
        failures.append("minimum-independent-source-groups")
    group_rows: list[dict[str, float]] = []
    for _group, values in sorted(per_group.items()):
        means = {
            condition: sum(row[condition] for row in values) / len(values)
            for condition in _CONDITIONS
        }
        means["correct_minus_offset"] = means["correct"] - means["offset"]
        means["correct_minus_cross"] = means["correct"] - means["cross"]
        group_rows.append(means)
    metrics = (
        "correct",
        "offset",
        "cross",
        "self_copy",
        "correct_minus_offset",
        "correct_minus_cross",
    )
    scores: dict[str, GateScore] = {}
    if group_rows:
        matrix = np.asarray(
            [[row[key] for key in metrics] for row in group_rows], dtype=np.float64
        )
        rng = np.random.default_rng(protocol.bootstrap_seed)
        indices = rng.integers(
            0,
            len(group_rows),
            size=(protocol.bootstrap_resamples, len(group_rows)),
        )
        draws = matrix[indices].mean(axis=1)
        for index, key in enumerate(metrics):
            lower, upper = _interval(draws[:, index], confidence=protocol.confidence)
            scores[key] = GateScore(
                estimate=float(matrix[:, index].mean()),
                ci_lower=lower,
                ci_upper=upper,
            )
        score_thresholds = (
            ("correct", protocol.minimum_correct_ci_lower),
            (
                "correct_minus_offset",
                protocol.minimum_correct_minus_offset_ci_lower,
            ),
            (
                "correct_minus_cross",
                protocol.minimum_correct_minus_cross_ci_lower,
            ),
        )
        failures.extend(
            f"{key.replace('_', '-')}-ci-lower-not-above-threshold"
            for key, threshold in score_thresholds
            if scores[key].ci_lower <= threshold
        )
    if maximum_self > protocol.maximum_absolute_self_copy_restoration:
        failures.append("self-copy-restoration-exceeds-maximum")
    failure_reasons = tuple(failures)
    passed = not failure_reasons
    return GateScoreResult(
        valid_records=len(valid),
        valid_strata=MappingProxyType(valid_strata),
        scores=MappingProxyType(scores),
        maximum_absolute_self_copy_restoration=maximum_self,
        failure_reasons=failure_reasons,
        passed=passed,
    )


__all__ = [
    "GateObservation",
    "GateScore",
    "GateScoreResult",
    "score_single_layer_gate",
]
