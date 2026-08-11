"""Denominator-safe layer scoring and deterministic stratified bootstrap."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from typo_robust_training.localization.config import LayerSelectionProtocol
from typo_robust_training.localization.records import LayerScan


@dataclass(frozen=True, slots=True)
class TaskCounts:
    total: int
    kl_eligible: int
    repair: int
    harm: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "kl_eligible": self.kl_eligible,
            "repair": self.repair,
            "harm": self.harm,
        }


@dataclass(frozen=True, slots=True)
class TaskLayerScore:
    kl_restoration: float
    answer_restoration: float
    harm: float
    composite: float

    def as_dict(self) -> dict[str, float]:
        return {
            "kl_restoration": self.kl_restoration,
            "answer_restoration": self.answer_restoration,
            "harm": self.harm,
            "composite": self.composite,
        }


@dataclass(frozen=True, slots=True)
class LayerSummary:
    layer: int
    macro_score: float
    tasks: Mapping[str, TaskLayerScore]

    def as_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "macro_score": self.macro_score,
            "tasks": {task: score.as_dict() for task, score in sorted(self.tasks.items())},
        }


@dataclass(frozen=True, slots=True)
class WindowSummary:
    window: tuple[int, int]
    macro_score: float
    rank: int

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.window[0],
            "stop": self.window[1],
            "macro_score": self.macro_score,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class LayerSelectionResult:
    selected_window: tuple[int, int]
    task_counts: Mapping[str, TaskCounts]
    layer_summaries: tuple[LayerSummary, ...]
    window_summaries: tuple[WindowSummary, ...]
    bootstrap: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-layer-selection/v1",
            "selected_window": {
                "start": self.selected_window[0],
                "stop": self.selected_window[1],
            },
            "task_counts": {
                task: counts.as_dict() for task, counts in sorted(self.task_counts.items())
            },
            "layer_summaries": [summary.as_dict() for summary in self.layer_summaries],
            "window_summaries": [summary.as_dict() for summary in self.window_summaries],
            "bootstrap": dict(self.bootstrap),
        }


def _eligible(scan: LayerScan, *, threshold: float) -> bool:
    mean = scan.untreated_mean_kl
    return mean is not None and math.isfinite(mean) and mean > threshold


def _cohort(scan: LayerScan) -> str:
    if scan.clean_correct and not scan.typo_correct:
        return "repair"
    if scan.clean_correct and scan.typo_correct:
        return "harm"
    return "other"


def _validate_and_group(
    scans: Sequence[LayerScan], *, protocol: LayerSelectionProtocol
) -> tuple[dict[str, tuple[LayerScan, ...]], int, dict[str, TaskCounts]]:
    rows = tuple(scans)
    if not rows:
        raise ValueError("layer selection requires diagnostic scans")
    if any(not isinstance(scan, LayerScan) for scan in rows):
        raise TypeError("layer selection scans must be LayerScan records")
    record_ids = [scan.record_id for scan in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("layer selection record IDs must be unique")
    layer_counts = {scan.num_layers for scan in rows}
    if len(layer_counts) != 1:
        raise ValueError("layer selection scans disagree on decoder-layer count")
    (num_layers,) = layer_counts
    if num_layers < protocol.window_width:
        raise ValueError("model has fewer layers than the frozen selection window")
    unexpected = {scan.task for scan in rows} - set(protocol.tasks)
    if unexpected:
        raise ValueError(f"layer selection received unexpected tasks: {sorted(unexpected)}")
    grouped = {task: tuple(scan for scan in rows if scan.task == task) for task in protocol.tasks}
    counts: dict[str, TaskCounts] = {}
    for task, task_rows in grouped.items():
        if not task_rows:
            raise ValueError(f"layer selection has no records for task {task}")
        eligible = sum(
            _eligible(scan, threshold=protocol.untreated_mean_kl_min_exclusive)
            for scan in task_rows
        )
        repair = sum(_cohort(scan) == "repair" for scan in task_rows)
        harm = sum(_cohort(scan) == "harm" for scan in task_rows)
        if eligible < protocol.minimum_kl_eligible_per_task:
            raise ValueError(
                f"{task} KL-eligible count {eligible} is below "
                f"{protocol.minimum_kl_eligible_per_task}"
            )
        fraction = eligible / len(task_rows)
        if fraction < protocol.minimum_kl_eligible_fraction_per_task:
            raise ValueError(
                f"{task} KL-eligible fraction {fraction:.6f} is below "
                f"{protocol.minimum_kl_eligible_fraction_per_task:.6f}"
            )
        if repair < protocol.minimum_answer_cohort_per_task:
            raise ValueError(
                f"{task} repair cohort {repair} is below {protocol.minimum_answer_cohort_per_task}"
            )
        if harm < protocol.minimum_answer_cohort_per_task:
            raise ValueError(
                f"{task} harm cohort {harm} is below {protocol.minimum_answer_cohort_per_task}"
            )
        counts[task] = TaskCounts(len(task_rows), eligible, repair, harm)
    return grouped, num_layers, counts


def _task_layer_scores(
    rows: Sequence[LayerScan], *, protocol: LayerSelectionProtocol, num_layers: int
) -> tuple[TaskLayerScore, ...]:
    eligible = [
        scan for scan in rows if _eligible(scan, threshold=protocol.untreated_mean_kl_min_exclusive)
    ]
    untreated_mean = sum(scan.untreated_mean_kl or 0.0 for scan in eligible) / len(eligible)
    repair = [scan for scan in rows if _cohort(scan) == "repair"]
    harm = [scan for scan in rows if _cohort(scan) == "harm"]
    scores: list[TaskLayerScore] = []
    for layer in range(num_layers):
        patched_mean = sum(
            sum(scan.patched_kl_2_16_by_layer[layer]) / 15 for scan in eligible
        ) / len(eligible)
        kl_restoration = 1.0 - patched_mean / untreated_mean
        answer_restoration = sum(
            scan.patched_correct_by_layer[layer] is True for scan in repair
        ) / len(repair)
        harm_rate = sum(scan.patched_correct_by_layer[layer] is False for scan in harm) / len(harm)
        composite = kl_restoration + protocol.beta * answer_restoration - protocol.gamma * harm_rate
        for value in (kl_restoration, answer_restoration, harm_rate, composite):
            if not math.isfinite(value):
                raise ValueError("layer selection produced a non-finite score")
        scores.append(
            TaskLayerScore(
                kl_restoration=kl_restoration,
                answer_restoration=answer_restoration,
                harm=harm_rate,
                composite=composite,
            )
        )
    return tuple(scores)


def _bootstrap(
    grouped: Mapping[str, tuple[LayerScan, ...]],
    *,
    protocol: LayerSelectionProtocol,
    num_layers: int,
) -> dict[str, object]:
    rng = np.random.default_rng(protocol.bootstrap_seed)
    width = protocol.window_width
    num_windows = num_layers - width + 1
    window_scores = np.empty((protocol.bootstrap_replicates, num_windows), dtype=np.float64)
    batch_size = min(256, protocol.bootstrap_replicates)
    for batch_start in range(0, protocol.bootstrap_replicates, batch_size):
        batch_stop = min(batch_start + batch_size, protocol.bootstrap_replicates)
        batch = batch_stop - batch_start
        task_composites = np.zeros((batch, num_layers), dtype=np.float64)
        for task in protocol.tasks:
            rows = grouped[task]
            strata: dict[tuple[bool, str], list[int]] = defaultdict(list)
            for index, scan in enumerate(rows):
                strata[
                    (
                        _eligible(
                            scan,
                            threshold=protocol.untreated_mean_kl_min_exclusive,
                        ),
                        _cohort(scan),
                    )
                ].append(index)
            kl_untreated = np.zeros(batch, dtype=np.float64)
            kl_patched = np.zeros((batch, num_layers), dtype=np.float64)
            kl_count = 0
            repair_correct = np.zeros((batch, num_layers), dtype=np.float64)
            repair_count = 0
            harm_wrong = np.zeros((batch, num_layers), dtype=np.float64)
            harm_count = 0
            for (is_eligible, cohort), indices_list in sorted(strata.items()):
                indices = np.asarray(indices_list, dtype=np.int64)
                sampled = indices[rng.integers(0, len(indices), size=(batch, len(indices)))]
                if is_eligible:
                    untreated = np.asarray(
                        [rows[index].untreated_mean_kl for index in range(len(rows))],
                        dtype=np.float64,
                    )
                    patched = np.asarray(
                        [
                            [
                                sum(values) / len(values)
                                for values in rows[index].patched_kl_2_16_by_layer
                            ]
                            for index in range(len(rows))
                        ],
                        dtype=np.float64,
                    )
                    kl_untreated += untreated[sampled].sum(axis=1)
                    kl_patched += patched[sampled].sum(axis=1)
                    kl_count += len(indices)
                if cohort == "repair":
                    correctness = np.asarray(
                        [
                            [value is True for value in rows[index].patched_correct_by_layer]
                            for index in range(len(rows))
                        ],
                        dtype=np.float64,
                    )
                    repair_correct += correctness[sampled].sum(axis=1)
                    repair_count += len(indices)
                elif cohort == "harm":
                    wrong = np.asarray(
                        [
                            [value is False for value in rows[index].patched_correct_by_layer]
                            for index in range(len(rows))
                        ],
                        dtype=np.float64,
                    )
                    harm_wrong += wrong[sampled].sum(axis=1)
                    harm_count += len(indices)
            task_composites += (
                1.0
                - (kl_patched / kl_count) / (kl_untreated[:, None] / kl_count)
                + protocol.beta * (repair_correct / repair_count)
                - protocol.gamma * (harm_wrong / harm_count)
            )
        macro_layers = task_composites / len(protocol.tasks)
        for start in range(num_windows):
            window_scores[batch_start:batch_stop, start] = macro_layers[
                :, start : start + width
            ].mean(axis=1)
    alpha = (1.0 - protocol.confidence_level) / 2.0
    selected = np.argmax(window_scores, axis=1)
    windows: list[dict[str, object]] = []
    for start in range(num_windows):
        values = window_scores[:, start]
        windows.append(
            {
                "start": start,
                "stop": start + width,
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                "selection_frequency": float(np.mean(selected == start)),
            }
        )
    return {
        "method": "task-and-base-cohort-stratified-pair-bootstrap/v1",
        "replicates": protocol.bootstrap_replicates,
        "seed": protocol.bootstrap_seed,
        "confidence_level": protocol.confidence_level,
        "windows": windows,
    }


def compute_layer_selection(
    scans: Sequence[LayerScan], *, protocol: LayerSelectionProtocol
) -> LayerSelectionResult:
    """Score every layer and freeze one fixed-width contiguous window."""

    grouped, num_layers, counts = _validate_and_group(scans, protocol=protocol)
    by_task = {
        task: _task_layer_scores(rows, protocol=protocol, num_layers=num_layers)
        for task, rows in grouped.items()
    }
    layers: list[LayerSummary] = []
    for layer in range(num_layers):
        task_scores = {task: by_task[task][layer] for task in protocol.tasks}
        macro = sum(score.composite for score in task_scores.values()) / len(task_scores)
        layers.append(LayerSummary(layer=layer, macro_score=macro, tasks=task_scores))

    raw_windows: list[tuple[tuple[int, int], float]] = []
    for start in range(num_layers - protocol.window_width + 1):
        stop = start + protocol.window_width
        score = sum(layer.macro_score for layer in layers[start:stop]) / protocol.window_width
        raw_windows.append(((start, stop), score))
    ranking = sorted(raw_windows, key=lambda item: (-item[1], item[0][0]))
    ranks = {window: rank for rank, (window, _score) in enumerate(ranking, 1)}
    windows = tuple(
        WindowSummary(window=window, macro_score=score, rank=ranks[window])
        for window, score in raw_windows
    )
    return LayerSelectionResult(
        selected_window=ranking[0][0],
        task_counts=counts,
        layer_summaries=tuple(layers),
        window_summaries=windows,
        bootstrap=_bootstrap(
            grouped,
            protocol=protocol,
            num_layers=num_layers,
        ),
    )


__all__ = [
    "LayerSelectionResult",
    "LayerSummary",
    "TaskCounts",
    "TaskLayerScore",
    "WindowSummary",
    "compute_layer_selection",
]
