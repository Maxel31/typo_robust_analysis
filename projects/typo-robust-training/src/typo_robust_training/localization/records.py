"""Validated per-record outputs from the all-layer diagnostic scan."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _trajectory(values: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be a sequence")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{field_name} must contain finite non-negative values")
    return result


@dataclass(frozen=True, slots=True)
class LayerScan:
    """One record's invariant baselines and one result per decoder layer."""

    record_id: str
    task: str
    target_token_ids: tuple[int, ...]
    untreated_kl_2_16: tuple[float, ...]
    patched_kl_2_16_by_layer: tuple[tuple[float, ...], ...]
    clean_correct: bool
    typo_correct: bool
    patched_correct_by_layer: tuple[bool | None, ...]
    kl_invalid_reason: str | None
    audit: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("record_id must be non-empty")
        if self.task not in {"gsm8k", "mmlu", "arc"}:
            raise ValueError("task is outside the diagnostic inventory")
        if not isinstance(self.target_token_ids, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.target_token_ids
        ):
            raise ValueError("target_token_ids must be non-negative integers")
        untreated = _trajectory(self.untreated_kl_2_16, field_name="untreated_kl_2_16")
        patched = tuple(
            _trajectory(values, field_name=f"patched_kl_2_16_by_layer[{index}]")
            for index, values in enumerate(self.patched_kl_2_16_by_layer)
        )
        if not patched:
            raise ValueError("patched KL must contain every decoder layer")
        if len(self.patched_correct_by_layer) != len(patched):
            raise ValueError("patched correctness count differs from decoder layers")
        if type(self.clean_correct) is not bool or type(self.typo_correct) is not bool:
            raise TypeError("baseline correctness values must be boolean")
        if any(
            value is not None and type(value) is not bool for value in self.patched_correct_by_layer
        ):
            raise TypeError("patched correctness values must be boolean or null")
        if self.clean_correct:
            if any(value is None for value in self.patched_correct_by_layer):
                raise ValueError("clean-correct records require patched answer outcomes")
        elif any(value is not None for value in self.patched_correct_by_layer):
            raise ValueError("clean-wrong records must skip patched answer generation")
        if untreated:
            if len(untreated) != 15 or any(len(values) != 15 for values in patched):
                raise ValueError("valid layer scans require token 2--16 KL trajectories")
            if len(self.target_token_ids) != 16:
                raise ValueError("valid layer scans require sixteen target token IDs")
            if self.kl_invalid_reason is not None:
                raise ValueError("valid KL trajectories cannot have an invalid reason")
        else:
            if any(values for values in patched):
                raise ValueError("invalid KL scans cannot contain patched trajectories")
            if not isinstance(self.kl_invalid_reason, str) or not self.kl_invalid_reason:
                raise ValueError("invalid KL scans require an explicit reason")
        if not isinstance(self.audit, Mapping):
            raise TypeError("audit must be an object")
        audit = dict(self.audit)
        try:
            json.dumps(audit, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("audit must be canonical-JSON-compatible") from exc
        object.__setattr__(self, "untreated_kl_2_16", untreated)
        object.__setattr__(self, "patched_kl_2_16_by_layer", patched)
        object.__setattr__(self, "audit", MappingProxyType(audit))

    @property
    def num_layers(self) -> int:
        return len(self.patched_kl_2_16_by_layer)

    @property
    def untreated_mean_kl(self) -> float | None:
        if not self.untreated_kl_2_16:
            return None
        return sum(self.untreated_kl_2_16) / len(self.untreated_kl_2_16)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-layer-scan/v1",
            "record_id": self.record_id,
            "task": self.task,
            "target_token_ids": list(self.target_token_ids),
            "untreated_kl_2_16": list(self.untreated_kl_2_16),
            "patched_kl_2_16_by_layer": [list(values) for values in self.patched_kl_2_16_by_layer],
            "clean_correct": self.clean_correct,
            "typo_correct": self.typo_correct,
            "patched_correct_by_layer": list(self.patched_correct_by_layer),
            "kl_invalid_reason": self.kl_invalid_reason,
            "audit": dict(self.audit),
        }

    @classmethod
    def from_dict(cls, value: object) -> LayerScan:
        if not isinstance(value, Mapping):
            raise ValueError("layer scan payload must be an object")
        expected = {
            "schema_version",
            "record_id",
            "task",
            "target_token_ids",
            "untreated_kl_2_16",
            "patched_kl_2_16_by_layer",
            "clean_correct",
            "typo_correct",
            "patched_correct_by_layer",
            "kl_invalid_reason",
            "audit",
        }
        if set(value) != expected or value.get("schema_version") != "robustness-layer-scan/v1":
            raise ValueError("layer scan payload fields or schema differ")
        patched = value["patched_kl_2_16_by_layer"]
        if not isinstance(patched, list):
            raise ValueError("patched KL payload must be a list")
        correctness = value["patched_correct_by_layer"]
        if not isinstance(correctness, list):
            raise ValueError("patched correctness payload must be a list")
        targets = value["target_token_ids"]
        untreated = value["untreated_kl_2_16"]
        if not isinstance(targets, list) or not isinstance(untreated, list):
            raise ValueError("layer scan token payloads must be lists")
        return cls(
            record_id=value["record_id"],  # type: ignore[arg-type]
            task=value["task"],  # type: ignore[arg-type]
            target_token_ids=tuple(targets),
            untreated_kl_2_16=tuple(untreated),
            patched_kl_2_16_by_layer=tuple(tuple(row) for row in patched),
            clean_correct=value["clean_correct"],  # type: ignore[arg-type]
            typo_correct=value["typo_correct"],  # type: ignore[arg-type]
            patched_correct_by_layer=tuple(correctness),  # type: ignore[arg-type]
            kl_invalid_reason=value["kl_invalid_reason"],  # type: ignore[arg-type]
            audit=value["audit"],  # type: ignore[arg-type]
        )


__all__ = ["LayerScan"]
