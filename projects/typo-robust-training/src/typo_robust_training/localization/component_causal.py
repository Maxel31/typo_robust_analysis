"""Cross-task causal validation and fail-closed component selection."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from typo_robust_training.localization.component_config import (
    ComponentLocalizationProtocol,
)
from typo_robust_training.localization.components import ComponentRef


@dataclass(frozen=True, slots=True)
class ComponentCausalObservation:
    """One candidate patched on one frozen diagnostic record."""

    record_id: str
    task: str
    component: ComponentRef
    untreated_mean_kl: float | None
    patched_mean_kl: float | None
    clean_correct: bool
    typo_correct: bool
    patched_correct: bool | None
    audit: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("causal observation record_id must be non-empty")
        if self.task not in {"gsm8k", "mmlu", "arc"}:
            raise ValueError("causal observation task is unsupported")
        if not isinstance(self.component, ComponentRef):
            raise TypeError("causal observation component must be ComponentRef")
        if (self.untreated_mean_kl is None) != (self.patched_mean_kl is None):
            raise ValueError("untreated and patched KL must be jointly available or unavailable")
        for field_name, value in (
            ("untreated_mean_kl", self.untreated_mean_kl),
            ("patched_mean_kl", self.patched_mean_kl),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if type(self.clean_correct) is not bool or type(self.typo_correct) is not bool:
            raise TypeError("causal baseline correctness must be boolean")
        if self.clean_correct and type(self.patched_correct) is not bool:
            raise ValueError("clean-correct causal observations require a patched answer")
        if not self.clean_correct and self.patched_correct is not None:
            raise ValueError("clean-wrong causal observations must skip patched answers")
        if not isinstance(self.audit, Mapping):
            raise TypeError("causal observation audit must be an object")
        audit = dict(self.audit)
        try:
            json.dumps(audit, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("causal observation audit must be canonical JSON") from exc
        object.__setattr__(self, "audit", MappingProxyType(audit))

    def as_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "task": self.task,
            "component": self.component.as_dict(),
            "untreated_mean_kl": self.untreated_mean_kl,
            "patched_mean_kl": self.patched_mean_kl,
            "clean_correct": self.clean_correct,
            "typo_correct": self.typo_correct,
            "patched_correct": self.patched_correct,
            "audit": dict(self.audit),
        }

    @classmethod
    def from_dict(cls, value: object) -> ComponentCausalObservation:
        if not isinstance(value, Mapping) or set(value) != {
            "record_id",
            "task",
            "component",
            "untreated_mean_kl",
            "patched_mean_kl",
            "clean_correct",
            "typo_correct",
            "patched_correct",
            "audit",
        }:
            raise ValueError("component causal observation fields differ")
        return cls(
            record_id=value["record_id"],  # type: ignore[arg-type]
            task=value["task"],  # type: ignore[arg-type]
            component=ComponentRef.from_dict(value["component"]),
            untreated_mean_kl=value["untreated_mean_kl"],  # type: ignore[arg-type]
            patched_mean_kl=value["patched_mean_kl"],  # type: ignore[arg-type]
            clean_correct=value["clean_correct"],  # type: ignore[arg-type]
            typo_correct=value["typo_correct"],  # type: ignore[arg-type]
            patched_correct=value["patched_correct"],  # type: ignore[arg-type]
            audit=value["audit"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CausalTaskScore:
    records: int
    kl_eligible: int
    repair: int
    harm_cohort: int
    kl_restoration: float
    answer_restoration: float
    harm_rate: float
    composite: float


@dataclass(frozen=True, slots=True)
class CausalComponentResult:
    component: ComponentRef
    task_scores: Mapping[str, CausalTaskScore]
    macro_score: float
    beneficial_tasks: int
    rejection_reasons: tuple[str, ...]
    selected: bool
    weight: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            **self.component.as_dict(),
            "macro_score": self.macro_score,
            "beneficial_tasks": self.beneficial_tasks,
            "rejection_reasons": list(self.rejection_reasons),
            "selected": self.selected,
            "weight": self.weight,
            "task_scores": {
                task: {
                    "records": score.records,
                    "kl_eligible": score.kl_eligible,
                    "repair": score.repair,
                    "harm_cohort": score.harm_cohort,
                    "kl_restoration": score.kl_restoration,
                    "answer_restoration": score.answer_restoration,
                    "harm_rate": score.harm_rate,
                    "composite": score.composite,
                }
                for task, score in sorted(self.task_scores.items())
            },
        }


@dataclass(frozen=True, slots=True)
class ComponentSelectionResult:
    components: tuple[CausalComponentResult, ...]
    selected: tuple[CausalComponentResult, ...]
    bootstrap: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-component-selection/v1",
            "components": [component.as_dict() for component in self.components],
            "selected": [component.as_dict() for component in self.selected],
            "bootstrap": dict(self.bootstrap),
        }


def _cohort(row: ComponentCausalObservation) -> str:
    if row.clean_correct and not row.typo_correct:
        return "repair"
    if row.clean_correct and row.typo_correct:
        return "harm"
    return "other"


def _validate_pair_grid(
    rows: Sequence[ComponentCausalObservation],
    *,
    candidates: tuple[ComponentRef, ...],
) -> Mapping[str, Mapping[ComponentRef, ComponentCausalObservation]]:
    candidate_set = set(candidates)
    by_record: dict[str, dict[ComponentRef, ComponentCausalObservation]] = defaultdict(dict)
    for row in rows:
        record_rows = by_record[row.record_id]
        if row.component in record_rows:
            raise ValueError("causal observation record/component keys are duplicated")
        record_rows[row.component] = row
    for record_id, record_rows in by_record.items():
        if set(record_rows) != candidate_set:
            raise ValueError("causal observations do not form a complete record/component grid")
        baseline = next(iter(record_rows.values()))
        signature = (
            baseline.task,
            baseline.untreated_mean_kl,
            baseline.clean_correct,
            baseline.typo_correct,
        )
        if any(
            (
                row.task,
                row.untreated_mean_kl,
                row.clean_correct,
                row.typo_correct,
            )
            != signature
            for row in record_rows.values()
        ):
            raise ValueError(f"causal observations disagree on baseline for record {record_id}")
    return MappingProxyType(
        {record_id: MappingProxyType(record_rows) for record_id, record_rows in by_record.items()}
    )


def _bootstrap(
    grid: Mapping[str, Mapping[ComponentRef, ComponentCausalObservation]],
    *,
    candidates: tuple[ComponentRef, ...],
    protocol: ComponentLocalizationProtocol,
) -> dict[str, object]:
    """Pair-bootstrap records while preserving every frozen task/cohort denominator."""

    rng = np.random.default_rng(protocol.bootstrap_seed)
    candidate_count = len(candidates)
    macro_scores = np.empty((protocol.bootstrap_replicates, candidate_count), dtype=np.float64)
    gate_passes = np.empty((protocol.bootstrap_replicates, candidate_count), dtype=np.bool_)
    batch_size = min(256, protocol.bootstrap_replicates)
    for batch_start in range(0, protocol.bootstrap_replicates, batch_size):
        batch_stop = min(batch_start + batch_size, protocol.bootstrap_replicates)
        batch = batch_stop - batch_start
        macro = np.zeros((batch, candidate_count), dtype=np.float64)
        beneficial = np.zeros((batch, candidate_count), dtype=np.int64)
        harm_violation = np.zeros((batch, candidate_count), dtype=np.bool_)
        for task in protocol.tasks:
            record_ids = sorted(
                record_id
                for record_id, record_rows in grid.items()
                if next(iter(record_rows.values())).task == task
            )
            baseline = [grid[record_id][candidates[0]] for record_id in record_ids]
            eligible = np.asarray(
                [
                    row.untreated_mean_kl is not None
                    and row.untreated_mean_kl > protocol.untreated_mean_kl_min_exclusive
                    for row in baseline
                ],
                dtype=np.bool_,
            )
            untreated = np.asarray(
                [row.untreated_mean_kl or 0.0 for row in baseline], dtype=np.float64
            )
            patched = np.asarray(
                [
                    [grid[record_id][component].patched_mean_kl or 0.0 for component in candidates]
                    for record_id in record_ids
                ],
                dtype=np.float64,
            )
            repair_correct = np.asarray(
                [
                    [grid[record_id][component].patched_correct is True for component in candidates]
                    for record_id in record_ids
                ],
                dtype=np.float64,
            )
            harm_wrong = np.asarray(
                [
                    [
                        grid[record_id][component].patched_correct is False
                        for component in candidates
                    ]
                    for record_id in record_ids
                ],
                dtype=np.float64,
            )
            strata: dict[tuple[bool, str], list[int]] = defaultdict(list)
            for index, row in enumerate(baseline):
                strata[(bool(eligible[index]), _cohort(row))].append(index)
            kl_untreated = np.zeros(batch, dtype=np.float64)
            kl_patched = np.zeros((batch, candidate_count), dtype=np.float64)
            repair_hits = np.zeros((batch, candidate_count), dtype=np.float64)
            harm_hits = np.zeros((batch, candidate_count), dtype=np.float64)
            kl_count = repair_count = harm_count = 0
            for (is_eligible, cohort), index_list in sorted(strata.items()):
                indices = np.asarray(index_list, dtype=np.int64)
                sampled = indices[rng.integers(0, len(indices), size=(batch, len(indices)))]
                if is_eligible:
                    kl_untreated += untreated[sampled].sum(axis=1)
                    kl_patched += patched[sampled].sum(axis=1)
                    kl_count += len(indices)
                if cohort == "repair":
                    repair_hits += repair_correct[sampled].sum(axis=1)
                    repair_count += len(indices)
                elif cohort == "harm":
                    harm_hits += harm_wrong[sampled].sum(axis=1)
                    harm_count += len(indices)
            task_score = (
                1.0
                - kl_patched / kl_untreated[:, None]
                + protocol.beta * repair_hits / repair_count
                - protocol.gamma * harm_hits / harm_count
            )
            harm_rate = harm_hits / harm_count
            macro += task_score / len(protocol.tasks)
            beneficial += task_score > 0.0
            harm_violation |= harm_rate > protocol.maximum_harm_rate_per_task
        macro_scores[batch_start:batch_stop] = macro
        gate_passes[batch_start:batch_stop] = (
            (beneficial >= protocol.minimum_beneficial_tasks) & ~harm_violation & (macro > 0.0)
        )
    alpha = (1.0 - protocol.confidence_level) / 2.0
    return {
        "method": "task-and-base-cohort-stratified-pair-bootstrap/v1",
        "replicates": protocol.bootstrap_replicates,
        "seed": protocol.bootstrap_seed,
        "confidence_level": protocol.confidence_level,
        "components": [
            {
                "identifier": component.identifier,
                "macro_ci_lower": float(np.quantile(macro_scores[:, index], alpha)),
                "macro_ci_upper": float(np.quantile(macro_scores[:, index], 1.0 - alpha)),
                "positive_frequency": float(np.mean(macro_scores[:, index] > 0.0)),
                "gate_pass_frequency": float(np.mean(gate_passes[:, index])),
            }
            for index, component in enumerate(candidates)
        ],
    }


def _task_score(
    rows: Sequence[ComponentCausalObservation],
    *,
    task: str,
    protocol: ComponentLocalizationProtocol,
) -> CausalTaskScore:
    eligible = [
        row
        for row in rows
        if row.untreated_mean_kl is not None
        and row.untreated_mean_kl > protocol.untreated_mean_kl_min_exclusive
    ]
    if len(eligible) < protocol.minimum_kl_eligible_per_task:
        raise ValueError(f"{task} component KL-eligible count is below the frozen minimum")
    if len(eligible) / len(rows) < protocol.minimum_kl_eligible_fraction_per_task:
        raise ValueError(f"{task} component KL-eligible fraction is below the frozen minimum")
    repair = [row for row in rows if row.clean_correct and not row.typo_correct]
    harm = [row for row in rows if row.clean_correct and row.typo_correct]
    if len(repair) < protocol.minimum_answer_cohort_per_task:
        raise ValueError(f"{task} component repair cohort is below the frozen minimum")
    if len(harm) < protocol.minimum_answer_cohort_per_task:
        raise ValueError(f"{task} component harm cohort is below the frozen minimum")
    untreated = sum(float(row.untreated_mean_kl) for row in eligible) / len(eligible)
    patched = sum(float(row.patched_mean_kl) for row in eligible) / len(eligible)
    kl_restoration = 1.0 - patched / untreated
    answer_restoration = sum(row.patched_correct is True for row in repair) / len(repair)
    harm_rate = sum(row.patched_correct is False for row in harm) / len(harm)
    composite = kl_restoration + protocol.beta * answer_restoration - protocol.gamma * harm_rate
    return CausalTaskScore(
        records=len(rows),
        kl_eligible=len(eligible),
        repair=len(repair),
        harm_cohort=len(harm),
        kl_restoration=kl_restoration,
        answer_restoration=answer_restoration,
        harm_rate=harm_rate,
        composite=composite,
    )


def select_training_components(
    observations: Sequence[ComponentCausalObservation],
    *,
    candidates: Sequence[ComponentRef],
    protocol: ComponentLocalizationProtocol,
) -> ComponentSelectionResult:
    """Select only causally beneficial, cross-task, low-harm components."""

    normalized_candidates = tuple(candidates)
    if not normalized_candidates or len(set(normalized_candidates)) != len(normalized_candidates):
        raise ValueError("causal candidates must be non-empty and unique")
    if any(not isinstance(component, ComponentRef) for component in normalized_candidates):
        raise TypeError("causal candidates must be ComponentRef values")
    rows = tuple(observations)
    if not rows or any(not isinstance(row, ComponentCausalObservation) for row in rows):
        raise ValueError("causal observations must be non-empty validated records")
    unexpected = {row.component for row in rows} - set(normalized_candidates)
    if unexpected:
        raise ValueError("causal observations contain an unrequested component")
    ordered_candidates = tuple(sorted(normalized_candidates, key=lambda value: value.identifier))
    grid = _validate_pair_grid(rows, candidates=ordered_candidates)
    results: list[CausalComponentResult] = []
    for component in ordered_candidates:
        component_rows = [row for row in rows if row.component == component]
        task_scores: dict[str, CausalTaskScore] = {}
        for task in protocol.tasks:
            task_rows = [row for row in component_rows if row.task == task]
            if not task_rows:
                raise ValueError(f"component {component.identifier} has no {task} observations")
            task_scores[task] = _task_score(task_rows, task=task, protocol=protocol)
        macro = sum(score.composite for score in task_scores.values()) / len(task_scores)
        beneficial = sum(score.composite > 0.0 for score in task_scores.values())
        reasons: list[str] = []
        if beneficial < protocol.minimum_beneficial_tasks:
            reasons.append(f"beneficial_tasks_lt_{protocol.minimum_beneficial_tasks}")
        reasons.extend(
            f"harm_rate_gt_{protocol.maximum_harm_rate_per_task:.2f}:{task}"
            for task, score in task_scores.items()
            if score.harm_rate > protocol.maximum_harm_rate_per_task
        )
        if macro <= 0.0:
            reasons.append("nonpositive_macro_causal_score")
        results.append(
            CausalComponentResult(
                component=component,
                task_scores=task_scores,
                macro_score=macro,
                beneficial_tasks=beneficial,
                rejection_reasons=tuple(reasons),
                selected=not reasons,
            )
        )
    selected_indices = [index for index, result in enumerate(results) if result.selected]
    if len(selected_indices) < protocol.minimum_selected_components:
        raise ValueError("no causally validated component survived the frozen selection gate")
    denominator = sum(max(results[index].macro_score, 0.0) for index in selected_indices)
    if denominator <= 0.0:
        raise ValueError("selected component weights have no positive causal mass")
    for index in selected_indices:
        results[index] = replace(
            results[index],
            weight=max(results[index].macro_score, 0.0) / denominator,
        )
    selected = tuple(results[index] for index in selected_indices)
    return ComponentSelectionResult(
        components=tuple(results),
        selected=selected,
        bootstrap=_bootstrap(
            grid,
            candidates=ordered_candidates,
            protocol=protocol,
        ),
    )


__all__ = [
    "CausalComponentResult",
    "CausalTaskScore",
    "ComponentCausalObservation",
    "ComponentSelectionResult",
    "select_training_components",
]
