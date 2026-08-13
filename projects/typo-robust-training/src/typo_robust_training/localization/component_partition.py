"""Deterministic diagnostic split for screening versus causal validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typo_robust_training.localization.component_config import (
    ComponentLocalizationProtocol,
)


@dataclass(frozen=True, slots=True)
class ComponentDiagnosticPartition:
    screening: tuple[str, ...]
    causal_validation: tuple[str, ...]
    screening_by_task: Mapping[str, tuple[str, ...]]
    causal_validation_by_task: Mapping[str, tuple[str, ...]]


def _key(record_id: str, *, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"component-partition/v1\0{seed}\0{record_id}".encode("utf-8")
    ).hexdigest()
    return digest, record_id


def partition_diagnostic_ids(
    records: Sequence[Mapping[str, object]],
    *,
    protocol: ComponentLocalizationProtocol,
) -> ComponentDiagnosticPartition:
    """Split SHA-ordered IDs in half independently within every task."""

    by_task: dict[str, list[str]] = {task: [] for task in protocol.tasks}
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"diagnostic record {index} must be an object")
        record_id, task = record.get("record_id"), record.get("task")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"diagnostic record {index} has no record_id")
        if record_id in seen:
            raise ValueError("diagnostic record IDs are duplicated")
        if task not in by_task:
            raise ValueError(f"diagnostic record {record_id} has an unexpected task")
        seen.add(record_id)
        by_task[str(task)].append(record_id)
    screening_by_task: dict[str, tuple[str, ...]] = {}
    causal_by_task: dict[str, tuple[str, ...]] = {}
    for task in protocol.tasks:
        ordered = tuple(
            sorted(by_task[task], key=lambda value: _key(value, seed=protocol.partition_seed))
        )
        if len(ordered) < 2:
            raise ValueError(f"component partition requires at least two {task} records")
        split = (len(ordered) + 1) // 2
        screening_by_task[task] = ordered[:split]
        causal_by_task[task] = ordered[split:]
    screening = tuple(sorted(value for values in screening_by_task.values() for value in values))
    causal = tuple(sorted(value for values in causal_by_task.values() for value in values))
    if set(screening) & set(causal) or set(screening) | set(causal) != seen:
        raise RuntimeError("component diagnostic partition is not disjoint and exhaustive")
    return ComponentDiagnosticPartition(
        screening=screening,
        causal_validation=causal,
        screening_by_task=screening_by_task,
        causal_validation_by_task=causal_by_task,
    )


__all__ = ["ComponentDiagnosticPartition", "partition_diagnostic_ids"]
