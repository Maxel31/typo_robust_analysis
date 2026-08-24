"""Clustered paired macro statistics for confirmatory evaluation v2."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from typo_robust_training.evaluation.calibration_v2 import EvaluationV2Protocol


_FIELDS = {
    "model_id",
    "task",
    "record_id",
    "variant",
    "condition",
    "seed",
    "clean_correct",
    "typo_correct",
}


@dataclass(frozen=True, slots=True)
class ClusteredPairedContrast:
    left_condition: str
    right_condition: str
    outcome: str
    point_difference_points: float
    ci95_points: tuple[float, float]
    one_sided_95_lower_points: float
    source_items: int


def _outcome(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"evaluation v2 {field} outcome must be boolean")
    return value


def clustered_paired_macro_contrast(
    rows: Sequence[Mapping[str, object]],
    *,
    protocol: EvaluationV2Protocol,
    left_condition: str,
    right_condition: str,
    outcome: str,
) -> ClusteredPairedContrast:
    """Bootstrap source items, never variants or training seeds as independent rows.

    The returned difference is ``right - left``.  Within each source item we
    first average the two typo realizations and all three learning seeds.  We
    then average items inside each model/task cell and give models and tasks
    equal macro weight.
    """

    if outcome not in {"clean_correct", "typo_correct"}:
        raise ValueError("evaluation v2 contrast outcome is unsupported")
    if left_condition == right_condition:
        raise ValueError("evaluation v2 contrast conditions must differ")
    allowed_conditions = set(protocol.arms)
    if left_condition not in allowed_conditions or right_condition not in allowed_conditions:
        raise ValueError("evaluation v2 contrast condition is not preregistered")

    selected = []
    for value in rows:
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("evaluation v2 confirmatory outcome fields differ")
        if value["condition"] not in {left_condition, right_condition}:
            continue
        model_id, task, record_id = value["model_id"], value["task"], value["record_id"]
        if (
            model_id not in {model.model_id for model in protocol.models}
            or task not in protocol.tasks
            or not isinstance(record_id, str)
            or not record_id
        ):
            raise ValueError("evaluation v2 confirmatory outcome identity differs")
        variant = value["variant"]
        if (
            isinstance(variant, bool)
            or not isinstance(variant, int)
            or variant not in range(protocol.confirmatory_typo_variants_per_item)
        ):
            raise ValueError("evaluation v2 confirmatory variant inventory differs")
        condition = str(value["condition"])
        seed = value["seed"]
        if condition == "base":
            if seed is not None:
                raise ValueError("evaluation v2 Base outcome cannot have a training seed")
        elif seed not in protocol.training_seeds:
            raise ValueError("evaluation v2 adapter seed inventory differs")
        selected.append(
            (
                str(model_id),
                str(task),
                record_id,
                variant,
                condition,
                seed,
                _outcome(value[outcome], field=outcome),
            )
        )
    if not selected:
        raise ValueError("evaluation v2 contrast has no selected outcomes")

    per_item_condition: dict[
        tuple[str, str, str, str], list[tuple[int, int | None, float]]
    ] = defaultdict(list)
    seen: set[tuple[str, str, str, int, str, int | None]] = set()
    records_by_model_task: dict[tuple[str, str], set[str]] = defaultdict(set)
    for model_id, task, record_id, variant, condition, seed, correct in selected:
        identity = (model_id, task, record_id, variant, condition, seed)
        if identity in seen:
            raise ValueError("evaluation v2 confirmatory outcomes contain duplicates")
        seen.add(identity)
        records_by_model_task[(model_id, task)].add(record_id)
        per_item_condition[(model_id, task, record_id, condition)].append(
            (variant, seed, float(correct))
        )

    expected_base_cells = {
        (variant, None)
        for variant in range(protocol.confirmatory_typo_variants_per_item)
    }
    expected_adapter_cells = {
        (variant, seed)
        for variant in range(protocol.confirmatory_typo_variants_per_item)
        for seed in protocol.training_seeds
    }
    for (model_id, task), record_ids in records_by_model_task.items():
        if len(record_ids) != protocol.confirmatory_records_per_task:
            raise ValueError("evaluation v2 confirmatory task sample size differs")
        for record_id in record_ids:
            for condition in (left_condition, right_condition):
                values = per_item_condition.get((model_id, task, record_id, condition), [])
                cells = {(variant, seed) for variant, seed, _correct in values}
                expected = expected_base_cells if condition == "base" else expected_adapter_cells
                if cells != expected:
                    raise ValueError(
                        "evaluation v2 confirmatory item/variant/seed coverage differs"
                    )

    expected_model_tasks = {
        (model.model_id, task) for model in protocol.models for task in protocol.tasks
    }
    if set(records_by_model_task) != expected_model_tasks:
        raise ValueError("evaluation v2 confirmatory model/task coverage differs")
    for task in protocol.tasks:
        model_record_sets = [
            records_by_model_task[(model.model_id, task)] for model in protocol.models
        ]
        if any(records != model_record_sets[0] for records in model_record_sets[1:]):
            raise ValueError("evaluation v2 confirmatory models must share source-item IDs")

    item_differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    for model in protocol.models:
        for task in protocol.tasks:
            for record_id in sorted(records_by_model_task[(model.model_id, task)]):
                left = fmean(
                    value
                    for _variant, _seed, value in per_item_condition[
                        (model.model_id, task, record_id, left_condition)
                    ]
                )
                right = fmean(
                    value
                    for _variant, _seed, value in per_item_condition[
                        (model.model_id, task, record_id, right_condition)
                    ]
                )
                item_differences[(model.model_id, task)].append(right - left)

    point = 100.0 * fmean(
        fmean(values) for _cell, values in sorted(item_differences.items())
    )
    import numpy as np

    label = f"{left_condition}\0{right_condition}\0{outcome}"
    label_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(protocol.bootstrap_seed ^ label_seed)
    samples = np.empty(protocol.bootstrap_replicates, dtype=np.float64)
    arrays = {
        cell: np.asarray(values, dtype=np.float64) for cell, values in item_differences.items()
    }
    for replicate in range(protocol.bootstrap_replicates):
        cell_means = []
        for cell in sorted(arrays):
            values = arrays[cell]
            indices = rng.integers(0, values.size, size=values.size)
            cell_means.append(values[indices].mean())
        samples[replicate] = fmean(cell_means) * 100.0
    lower, one_sided, upper = np.quantile(samples, [0.025, 0.05, 0.975]).tolist()
    return ClusteredPairedContrast(
        left_condition=left_condition,
        right_condition=right_condition,
        outcome=outcome,
        point_difference_points=point,
        ci95_points=(float(lower), float(upper)),
        one_sided_95_lower_points=float(one_sided),
        source_items=sum(
            len(records_by_model_task[(protocol.models[0].model_id, task)])
            for task in protocol.tasks
        ),
    )


__all__ = ["ClusteredPairedContrast", "clustered_paired_macro_contrast"]
