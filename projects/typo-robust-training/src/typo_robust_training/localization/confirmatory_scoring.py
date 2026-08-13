"""Pairwise-median scoring for confirmatory joint-window patching."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from typo_robust_training.localization.confirmatory_records import JointWindowScan


def _ci(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pairwise_restoration(scan: JointWindowScan, start: int) -> float:
    untreated = _mean(scan.untreated_kl_2_16)
    patched = _mean(scan.patched_kl_2_16_by_window[start])
    return 1.0 - patched / untreated


def _eligible_rows(
    scans: Sequence[JointWindowScan],
    *,
    role: str,
    denominator_min_exclusive: float,
    minimum_eligible: int,
    minimum_eligible_fraction: float,
) -> tuple[JointWindowScan, ...]:
    rows = tuple(scans)
    if not rows or any(not isinstance(row, JointWindowScan) for row in rows):
        raise ValueError("joint-window scoring requires JointWindowScan records")
    ids = [row.record_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("joint-window scan IDs are duplicated")
    if any(row.role != role for row in rows):
        raise ValueError(f"joint-window scoring requires {role} records")
    dimensions = {(row.decoder_layers, row.window_width) for row in rows}
    if len(dimensions) != 1:
        raise ValueError("joint-window scans disagree on layer dimensions")
    if (
        isinstance(denominator_min_exclusive, bool)
        or not isinstance(denominator_min_exclusive, (int, float))
        or not math.isfinite(float(denominator_min_exclusive))
        or denominator_min_exclusive <= 0.0
    ):
        raise ValueError("joint-window denominator threshold must be positive")
    if (
        isinstance(minimum_eligible, bool)
        or not isinstance(minimum_eligible, int)
        or minimum_eligible <= 0
        or isinstance(minimum_eligible_fraction, bool)
        or not isinstance(minimum_eligible_fraction, (int, float))
        or not 0.0 < float(minimum_eligible_fraction) <= 1.0
    ):
        raise ValueError("joint-window eligibility requirements are invalid")
    eligible = tuple(
        row
        for row in rows
        if row.untreated_mean_kl is not None
        and row.untreated_mean_kl > float(denominator_min_exclusive)
    )
    if len(eligible) < minimum_eligible:
        raise ValueError("joint-window scoring has too few KL-eligible records")
    if len(eligible) / len(rows) < float(minimum_eligible_fraction):
        raise ValueError("joint-window scoring KL-eligible fraction is too small")
    return eligible


def _random_control_window(
    *,
    decoder_layers: int,
    window_width: int,
    selected_window: tuple[int, int],
    seed: int,
) -> tuple[int, int]:
    candidates = [
        start
        for start in range(decoder_layers - window_width + 1)
        if start + window_width <= selected_window[0] or start >= selected_window[1]
    ]
    if not candidates:
        raise ValueError("selected window leaves no non-overlapping same-width control")
    material = (
        "confirmatory-random-window/v1\0"
        f"{seed}\0{selected_window[0]}:{selected_window[1]}\0"
        + ",".join(str(value) for value in candidates)
    ).encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(candidates)
    start = candidates[index]
    return start, start + window_width


@dataclass(frozen=True, slots=True)
class JointWindowSelectionSummary:
    records: int
    eligible_records: int
    decoder_layers: int
    window_width: int
    selected_window: tuple[int, int]
    random_control_window: tuple[int, int]
    window_scores: Mapping[int, float]
    selected_confidence_interval: tuple[float, float]
    selection_frequency: Mapping[int, float]
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-joint-window-selection/v1",
            "operation": "select-generic-joint-patch-window",
            "records": self.records,
            "eligible_records": self.eligible_records,
            "decoder_layers": self.decoder_layers,
            "window_width": self.window_width,
            "selection_metric": "median-pairwise-kl-restoration/v1",
            "tie_break": "smallest-start-layer/v1",
            "selected_window": {
                "start": self.selected_window[0],
                "stop": self.selected_window[1],
                "median_pairwise_restoration": self.window_scores[self.selected_window[0]],
                "confidence_interval": list(self.selected_confidence_interval),
            },
            "random_control_window": {
                "start": self.random_control_window[0],
                "stop": self.random_control_window[1],
                "rule": "sha256-drawn-nonoverlapping-same-width/v1",
            },
            "windows": [
                {
                    "start": start,
                    "stop": start + self.window_width,
                    "median_pairwise_restoration": score,
                    "bootstrap_selection_frequency": self.selection_frequency[start],
                }
                for start, score in sorted(self.window_scores.items())
            ],
            "bootstrap": {
                "method": "pair-bootstrap-median/v1",
                "replicates": self.bootstrap_replicates,
                "seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
            },
        }


@dataclass(frozen=True, slots=True)
class JointWindowValidationSummary:
    records: int
    eligible_records: int
    selected_window: tuple[int, int]
    median_pairwise_restoration: float
    confidence_interval: tuple[float, float]
    passed: bool
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-joint-window-validation/v1",
            "operation": "validate-generic-joint-patch-window",
            "records": self.records,
            "eligible_records": self.eligible_records,
            "selected_window": {
                "start": self.selected_window[0],
                "stop": self.selected_window[1],
            },
            "median_pairwise_restoration": self.median_pairwise_restoration,
            "confidence_interval": list(self.confidence_interval),
            "validation_rule": "bootstrap-95ci-lower-strictly-positive/v1",
            "passed": self.passed,
            "bootstrap": {
                "method": "pair-bootstrap-median/v1",
                "replicates": self.bootstrap_replicates,
                "seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
            },
        }


def select_joint_patch_window(
    scans: Sequence[JointWindowScan],
    *,
    denominator_min_exclusive: float,
    minimum_eligible: int,
    minimum_eligible_fraction: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
    random_control_seed: int,
) -> JointWindowSelectionSummary:
    rows = tuple(scans)
    eligible = _eligible_rows(
        rows,
        role="selection",
        denominator_min_exclusive=denominator_min_exclusive,
        minimum_eligible=minimum_eligible,
        minimum_eligible_fraction=minimum_eligible_fraction,
    )
    decoder_layers, window_width = eligible[0].decoder_layers, eligible[0].window_width
    starts = tuple(range(decoder_layers - window_width + 1))
    values = np.asarray(
        [[_pairwise_restoration(row, start) for start in starts] for row in eligible],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("joint-window pairwise restoration is non-finite")
    scores_array = np.median(values, axis=0)
    selected_start = min(starts, key=lambda start: (-float(scores_array[start]), start))

    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
        or not 0.0 < confidence_level <= 1.0
    ):
        raise ValueError("joint-window bootstrap settings are invalid")
    rng = np.random.default_rng(bootstrap_seed)
    selected_samples = np.empty(bootstrap_replicates, dtype=np.float64)
    winner_counts = np.zeros(len(starts), dtype=np.int64)
    for replicate in range(bootstrap_replicates):
        sampled = rng.integers(0, len(eligible), size=len(eligible))
        bootstrap_scores = np.median(values[sampled], axis=0)
        winner = int(np.argmax(bootstrap_scores))
        winner_counts[winner] += 1
        selected_samples[replicate] = bootstrap_scores[selected_start]
    selected_window = (selected_start, selected_start + window_width)
    return JointWindowSelectionSummary(
        records=len(rows),
        eligible_records=len(eligible),
        decoder_layers=decoder_layers,
        window_width=window_width,
        selected_window=selected_window,
        random_control_window=_random_control_window(
            decoder_layers=decoder_layers,
            window_width=window_width,
            selected_window=selected_window,
            seed=random_control_seed,
        ),
        window_scores={start: float(scores_array[start]) for start in starts},
        selected_confidence_interval=_ci(selected_samples, confidence_level),
        selection_frequency={
            start: int(winner_counts[start]) / bootstrap_replicates for start in starts
        },
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def validate_joint_patch_window(
    scans: Sequence[JointWindowScan],
    *,
    selected_window: tuple[int, int],
    denominator_min_exclusive: float,
    minimum_eligible: int,
    minimum_eligible_fraction: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> JointWindowValidationSummary:
    rows = tuple(scans)
    eligible = _eligible_rows(
        rows,
        role="validation",
        denominator_min_exclusive=denominator_min_exclusive,
        minimum_eligible=minimum_eligible,
        minimum_eligible_fraction=minimum_eligible_fraction,
    )
    start, stop = selected_window
    if (
        stop - start != eligible[0].window_width
        or not 0 <= start < stop <= eligible[0].decoder_layers
    ):
        raise ValueError("frozen selected window differs from validation dimensions")
    if any(set(row.patched_kl_2_16_by_window) != {start} for row in eligible):
        raise ValueError("validation scans do not contain exactly the frozen window")
    values = np.asarray([_pairwise_restoration(row, start) for row in eligible], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("validation pairwise restoration is non-finite")
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        sampled = rng.integers(0, len(eligible), size=len(eligible))
        samples[replicate] = np.median(values[sampled])
    confidence_interval = _ci(samples, confidence_level)
    median = float(np.median(values))
    return JointWindowValidationSummary(
        records=len(rows),
        eligible_records=len(eligible),
        selected_window=selected_window,
        median_pairwise_restoration=median,
        confidence_interval=confidence_interval,
        passed=confidence_interval[0] > 0.0,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


__all__ = [
    "JointWindowSelectionSummary",
    "JointWindowValidationSummary",
    "select_joint_patch_window",
    "validate_joint_patch_window",
]
