"""Pure, source-group-bootstrap scoring for semantic-subspace kill tests."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from typo_robust_training.probe.subspace_kill_config import SemanticSubspaceKillProtocol


PATCH_OPERATORS = (
    "full-state",
    "semantic-rank16",
    "clean-fit-pca-rank16",
    "deterministic-haar-random-rank16",
    "semantic-complement-rank16",
)
CONTROL_OPERATORS = PATCH_OPERATORS[2:]


def _kl(values: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != 15:
        raise ValueError(f"{field} must contain exactly fifteen KL values")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{field} must contain finite non-negative KL values")
    return result


@dataclass(frozen=True, slots=True)
class SubspaceKillScoreRow:
    pair_id: str
    source_group_sha256: str
    transition_layer: int
    clean_word_final_token: int
    typo_word_final_token: int
    untreated_kl_2_16: tuple[float, ...]
    patched_kl_2_16: Mapping[str, tuple[float, ...]]
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("semantic kill pair id must be non-empty")
        if (
            not isinstance(self.source_group_sha256, str)
            or len(self.source_group_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.source_group_sha256)
        ):
            raise ValueError("semantic kill source group must be a SHA-256")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.transition_layer,
                self.clean_word_final_token,
                self.typo_word_final_token,
            )
        ):
            raise ValueError("semantic kill patch coordinates differ")
        untreated = tuple(self.untreated_kl_2_16)
        patched = dict(self.patched_kl_2_16)
        if self.invalid_reason is None:
            untreated = _kl(untreated, field="untreated semantic kill KL")
            if set(patched) != set(PATCH_OPERATORS):
                raise ValueError("semantic kill patch operator inventory differs")
            patched = {
                operator: _kl(values, field=f"{operator} semantic kill KL")
                for operator, values in patched.items()
            }
        else:
            if not isinstance(self.invalid_reason, str) or not self.invalid_reason:
                raise ValueError("semantic kill invalid reason must be non-empty")
            if untreated or patched:
                raise ValueError("invalid semantic kill row cannot contain KL values")
        object.__setattr__(self, "untreated_kl_2_16", untreated)
        object.__setattr__(self, "patched_kl_2_16", MappingProxyType(patched))

    def restoration(self, operator: str, *, denominator_min_exclusive: float) -> float:
        if operator not in self.patched_kl_2_16:
            raise ValueError("semantic kill restoration operator is unavailable")
        denominator = sum(self.untreated_kl_2_16) / 15
        if denominator <= denominator_min_exclusive:
            raise ValueError("semantic kill untreated denominator is ineligible")
        numerator = sum(self.patched_kl_2_16[operator]) / 15
        return 1.0 - numerator / denominator


@dataclass(frozen=True, slots=True)
class SemanticSubspaceKillSummary:
    records: int
    valid_records: int
    restoration: Mapping[str, float]
    ci_lower: Mapping[str, float]
    semantic_full_ratio_ci_lower: float | None
    semantic_minus_control_ci_lower: Mapping[str, float]
    passed: bool


def _lower(values: np.ndarray, confidence: float) -> float:
    ordered = np.sort(values)
    index = max(0, math.ceil(((1.0 - confidence) / 2.0) * len(ordered)) - 1)
    return float(ordered[index])


def score_semantic_subspace_kill(
    rows: Sequence[SubspaceKillScoreRow],
    *,
    protocol: SemanticSubspaceKillProtocol,
    transition_layer: int,
) -> SemanticSubspaceKillSummary:
    """Recompute all gates from raw per-pair KL trajectories."""

    observations = tuple(rows)
    if not observations:
        raise ValueError("semantic kill score inventory must not be empty")
    if len({row.pair_id for row in observations}) != len(observations):
        raise ValueError("semantic kill score pair ids must be unique")
    if any(row.transition_layer != transition_layer for row in observations):
        raise ValueError("semantic kill score uses the wrong patch layer")
    valid = tuple(
        row
        for row in observations
        if row.invalid_reason is None
        and sum(row.untreated_kl_2_16) / 15 > protocol.denominator_min_exclusive
    )
    if (
        len(valid) < protocol.minimum_valid
        or len(valid) / len(observations) < protocol.minimum_valid_fraction
    ):
        raise ValueError("semantic kill valid cohort is below the preregistered gate")
    grouped: dict[str, list[SubspaceKillScoreRow]] = defaultdict(list)
    for row in valid:
        grouped[row.source_group_sha256].append(row)
    if len(grouped) < 2:
        raise ValueError("semantic kill bootstrap requires independent source groups")
    group_names = tuple(sorted(grouped))
    group_values = np.empty((len(group_names), len(PATCH_OPERATORS)), dtype=np.float64)
    for group_index, group in enumerate(group_names):
        group_rows = grouped[group]
        for operator_index, operator in enumerate(PATCH_OPERATORS):
            group_values[group_index, operator_index] = np.mean(
                [
                    row.restoration(
                        operator,
                        denominator_min_exclusive=protocol.denominator_min_exclusive,
                    )
                    for row in group_rows
                ]
            )
    restoration_values = group_values.mean(axis=0)
    rng = np.random.default_rng(protocol.bootstrap_seed)
    draws = rng.integers(
        0,
        len(group_names),
        size=(protocol.bootstrap_resamples, len(group_names)),
    )
    boot = group_values[draws].mean(axis=1)
    ci_lower = {
        operator: _lower(boot[:, index], protocol.bootstrap_confidence)
        for index, operator in enumerate(PATCH_OPERATORS)
    }
    full_index = PATCH_OPERATORS.index("full-state")
    semantic_index = PATCH_OPERATORS.index("semantic-rank16")
    full_samples = boot[:, full_index]
    if np.any(full_samples <= protocol.denominator_min_exclusive):
        ratio_lower = None
    else:
        ratio_lower = _lower(
            boot[:, semantic_index] / full_samples,
            protocol.bootstrap_confidence,
        )
    differences = {
        control: _lower(
            boot[:, semantic_index] - boot[:, PATCH_OPERATORS.index(control)],
            protocol.bootstrap_confidence,
        )
        for control in CONTROL_OPERATORS
    }
    passed = (
        ci_lower["full-state"] > 0.0
        and ci_lower["semantic-rank16"] > 0.0
        and ratio_lower is not None
        and ratio_lower >= protocol.semantic_full_ratio_lower
        and all(
            lower > protocol.control_difference_lower for lower in differences.values()
        )
    )
    return SemanticSubspaceKillSummary(
        records=len(observations),
        valid_records=len(valid),
        restoration=MappingProxyType(
            {
                operator: float(restoration_values[index])
                for index, operator in enumerate(PATCH_OPERATORS)
            }
        ),
        ci_lower=MappingProxyType(ci_lower),
        semantic_full_ratio_ci_lower=ratio_lower,
        semantic_minus_control_ci_lower=MappingProxyType(differences),
        passed=passed,
    )


__all__ = [
    "CONTROL_OPERATORS",
    "PATCH_OPERATORS",
    "SemanticSubspaceKillSummary",
    "SubspaceKillScoreRow",
    "score_semantic_subspace_kill",
]
