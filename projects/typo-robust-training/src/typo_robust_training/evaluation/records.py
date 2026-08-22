"""Validated per-pair outputs shared by evaluation resume and aggregation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


_SHA64 = re.compile(r"[0-9a-f]{64}")
_CONDITIONS = frozenset(
    {
        "base",
        "noisy-language-model",
        "output-matching",
        "global-state-alignment",
        "localized-state-distillation",
        "random-window-state-distillation",
        "probe-transition-output-matching",
        "probe-transition-single-layer-state-distillation",
        "causal-probe-subspace-distillation",
        "probe-semantic-subspace-distillation",
    }
)
_EVALUATION_CONDITIONS = frozenset(
    {
        "random-1",
        "random-2",
        "random-4",
        "transposition-2",
        "natural-injection",
        "natural-lm-pair",
    }
)
_TASKS = frozenset({"gsm8k", "mmlu", "arc", "mmlu_pro", "math_500", "commonsense_qa"})
_STRATA = ("same-task", "unseen-task", "unseen-content", "unseen-typo")
_TOKENIZATION_STRATA = frozenset(
    {
        "same-subtoken-count",
        "fragmentation-increased",
        "fragmentation-decreased",
        "mixed-subtoken-change",
    }
)
_FIELDS = {
    "schema_version",
    "record_id",
    "condition",
    "seed",
    "evaluation_condition",
    "source",
    "task",
    "operation",
    "edit_count",
    "mechanistic_audit",
    "strata",
    "clean_answer",
    "typo_answer",
    "patched_answer",
    "clean_correct",
    "typo_correct",
    "patched_correct",
    "target_token_ids",
    "untreated_kl_2_16",
    "patched_kl_2_16",
    "kl_invalid_reason",
    "patch_invalid_reason",
    "clean_subtoken_counts",
    "typo_subtoken_counts",
    "tokenization_stratum",
    "audit",
}
_CORPUS_FIELDS = {
    "schema_version",
    "record_id",
    "condition",
    "seed",
    "kind",
    "source",
    "clean_nll_sum",
    "clean_nll_tokens",
    "typo_nll_sum",
    "typo_nll_tokens",
    "base_clean_kl_sum",
    "base_clean_kl_tokens",
    "natural_clean_typo_kl_sum",
    "natural_clean_typo_kl_tokens",
}


def _trajectory(value: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError(f"{field_name} must contain finite non-negative values")
    return result


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationObservation:
    record_id: str
    condition: str
    seed: int | None
    evaluation_condition: str
    source: str
    task: str | None
    operation: str
    edit_count: int
    mechanistic_audit: bool
    strata: tuple[str, ...]
    clean_answer: str | None
    typo_answer: str | None
    patched_answer: str | None
    clean_correct: bool | None
    typo_correct: bool | None
    patched_correct: bool | None
    target_token_ids: tuple[int, ...]
    untreated_kl_2_16: tuple[float, ...]
    patched_kl_2_16: tuple[float, ...]
    kl_invalid_reason: str | None
    patch_invalid_reason: str | None
    clean_subtoken_counts: tuple[int, ...]
    typo_subtoken_counts: tuple[int, ...]
    tokenization_stratum: str
    audit: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or _SHA64.fullmatch(self.record_id) is None:
            raise ValueError("evaluation observation record_id must be a SHA-256 digest")
        if self.condition not in _CONDITIONS:
            raise ValueError("evaluation observation condition is unsupported")
        if self.condition == "base":
            if self.seed is not None:
                raise ValueError("base evaluation observation must not have a seed")
        elif isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("adapter evaluation observation must have a non-negative seed")
        if self.evaluation_condition not in _EVALUATION_CONDITIONS:
            raise ValueError("evaluation observation typo condition is unsupported")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("evaluation observation source must be non-empty")
        if self.task is not None and self.task not in _TASKS:
            raise ValueError("evaluation observation task is unsupported")
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("evaluation observation operation must be non-empty")
        if (
            isinstance(self.edit_count, bool)
            or not isinstance(self.edit_count, int)
            or self.edit_count <= 0
        ):
            raise ValueError("evaluation observation edit_count must be a positive integer")
        if type(self.mechanistic_audit) is not bool:
            raise ValueError("evaluation observation mechanistic_audit must be boolean")
        if self.mechanistic_audit and self.evaluation_condition != "random-2":
            raise ValueError("mechanistic audit observations must use random-2")
        if not self.mechanistic_audit and (
            self.patched_answer is not None
            or self.patched_correct is not None
            or self.patched_kl_2_16
            or self.patch_invalid_reason != "not-mechanistic-audit"
        ):
            raise ValueError("non-audit observations cannot contain patch outputs")
        if tuple(item for item in _STRATA if item in self.strata) != self.strata or not self.strata:
            raise ValueError("evaluation observation strata must be unique and canonically ordered")
        if self.task is None:
            if any(
                value is not None
                for value in (
                    self.clean_answer,
                    self.typo_answer,
                    self.patched_answer,
                    self.clean_correct,
                    self.typo_correct,
                    self.patched_correct,
                )
            ):
                raise ValueError("non-task evaluation observations cannot contain answer outcomes")
        else:
            if not isinstance(self.clean_answer, str) or not isinstance(self.typo_answer, str):
                raise ValueError("task evaluation observations require clean and typo answers")
            if type(self.clean_correct) is not bool or type(self.typo_correct) is not bool:
                raise ValueError("task evaluation observations require clean and typo correctness")
            if self.patch_invalid_reason is None:
                if (
                    not isinstance(self.patched_answer, str)
                    or type(self.patched_correct) is not bool
                ):
                    raise ValueError("valid task patches require an answer outcome")
            elif self.patched_answer is not None or self.patched_correct is not None:
                raise ValueError("invalid task patches cannot contain an answer outcome")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.target_token_ids
        ):
            raise ValueError("evaluation target token IDs must be non-negative integers")
        untreated = _trajectory(self.untreated_kl_2_16, field_name="untreated_kl_2_16")
        patched = _trajectory(self.patched_kl_2_16, field_name="patched_kl_2_16")
        if untreated:
            if len(untreated) != 15 or len(self.target_token_ids) != 16:
                raise ValueError("valid untreated KL requires sixteen targets and fifteen values")
            if self.kl_invalid_reason is not None:
                raise ValueError("valid untreated KL cannot have an invalid reason")
        elif (
            self.target_token_ids
            or not isinstance(self.kl_invalid_reason, str)
            or not self.kl_invalid_reason
        ):
            raise ValueError("invalid untreated KL requires an explicit reason and no targets")
        if patched:
            if len(patched) != 15:
                raise ValueError("valid patched KL requires fifteen values")
            if not untreated or self.patch_invalid_reason is not None:
                raise ValueError(
                    "valid patched KL requires valid untreated KL and no invalid reason"
                )
        elif not isinstance(self.patch_invalid_reason, str) or not self.patch_invalid_reason:
            raise ValueError("invalid patched KL requires an explicit reason")
        for field_name, counts in (
            ("clean_subtoken_counts", self.clean_subtoken_counts),
            ("typo_subtoken_counts", self.typo_subtoken_counts),
        ):
            if not counts or any(
                isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for count in counts
            ):
                raise ValueError(f"evaluation {field_name} must contain positive integers")
        if len(self.clean_subtoken_counts) != len(self.typo_subtoken_counts):
            raise ValueError("evaluation clean and typo subtoken counts must align by edit")
        if len(self.clean_subtoken_counts) != self.edit_count:
            raise ValueError("evaluation edit_count must match the aligned edit inventory")
        if self.tokenization_stratum not in _TOKENIZATION_STRATA:
            raise ValueError("evaluation tokenization stratum is unsupported")
        if not isinstance(self.audit, Mapping):
            raise ValueError("evaluation observation audit must be an object")
        audit = dict(self.audit)
        try:
            json.dumps(audit, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation observation audit must be canonical JSON") from exc
        object.__setattr__(self, "untreated_kl_2_16", untreated)
        object.__setattr__(self, "patched_kl_2_16", patched)
        object.__setattr__(self, "audit", MappingProxyType(audit))

    @property
    def condition_id(self) -> str:
        return "base" if self.condition == "base" else f"{self.condition}:seed-{self.seed}"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-evaluation-observation/v4",
            "record_id": self.record_id,
            "condition": self.condition,
            "seed": self.seed,
            "evaluation_condition": self.evaluation_condition,
            "source": self.source,
            "task": self.task,
            "operation": self.operation,
            "edit_count": self.edit_count,
            "mechanistic_audit": self.mechanistic_audit,
            "strata": list(self.strata),
            "clean_answer": self.clean_answer,
            "typo_answer": self.typo_answer,
            "patched_answer": self.patched_answer,
            "clean_correct": self.clean_correct,
            "typo_correct": self.typo_correct,
            "patched_correct": self.patched_correct,
            "target_token_ids": list(self.target_token_ids),
            "untreated_kl_2_16": list(self.untreated_kl_2_16),
            "patched_kl_2_16": list(self.patched_kl_2_16),
            "kl_invalid_reason": self.kl_invalid_reason,
            "patch_invalid_reason": self.patch_invalid_reason,
            "clean_subtoken_counts": list(self.clean_subtoken_counts),
            "typo_subtoken_counts": list(self.typo_subtoken_counts),
            "tokenization_stratum": self.tokenization_stratum,
            "audit": dict(self.audit),
        }

    @classmethod
    def from_dict(cls, value: object) -> EvaluationObservation:
        if (
            not isinstance(value, Mapping)
            or set(value) != _FIELDS
            or value.get("schema_version") != "robustness-evaluation-observation/v4"
        ):
            raise ValueError("evaluation observation fields or schema differ")
        list_fields = (
            "strata",
            "target_token_ids",
            "untreated_kl_2_16",
            "patched_kl_2_16",
            "clean_subtoken_counts",
            "typo_subtoken_counts",
        )
        if any(not isinstance(value[field], list) for field in list_fields):
            raise ValueError("evaluation observation sequence fields must be lists")
        return cls(
            record_id=value["record_id"],  # type: ignore[arg-type]
            condition=value["condition"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            evaluation_condition=value["evaluation_condition"],  # type: ignore[arg-type]
            source=value["source"],  # type: ignore[arg-type]
            task=value["task"],  # type: ignore[arg-type]
            operation=value["operation"],  # type: ignore[arg-type]
            edit_count=value["edit_count"],  # type: ignore[arg-type]
            mechanistic_audit=value["mechanistic_audit"],  # type: ignore[arg-type]
            strata=tuple(value["strata"]),  # type: ignore[arg-type]
            clean_answer=_optional_string(value["clean_answer"], field_name="clean_answer"),
            typo_answer=_optional_string(value["typo_answer"], field_name="typo_answer"),
            patched_answer=_optional_string(value["patched_answer"], field_name="patched_answer"),
            clean_correct=value["clean_correct"],  # type: ignore[arg-type]
            typo_correct=value["typo_correct"],  # type: ignore[arg-type]
            patched_correct=value["patched_correct"],  # type: ignore[arg-type]
            target_token_ids=tuple(value["target_token_ids"]),  # type: ignore[arg-type]
            untreated_kl_2_16=tuple(value["untreated_kl_2_16"]),  # type: ignore[arg-type]
            patched_kl_2_16=tuple(value["patched_kl_2_16"]),  # type: ignore[arg-type]
            kl_invalid_reason=_optional_string(
                value["kl_invalid_reason"], field_name="kl_invalid_reason"
            ),
            patch_invalid_reason=_optional_string(
                value["patch_invalid_reason"], field_name="patch_invalid_reason"
            ),
            clean_subtoken_counts=tuple(value["clean_subtoken_counts"]),  # type: ignore[arg-type]
            typo_subtoken_counts=tuple(value["typo_subtoken_counts"]),  # type: ignore[arg-type]
            tokenization_stratum=value["tokenization_stratum"],  # type: ignore[arg-type]
            audit=value["audit"],  # type: ignore[arg-type]
        )


def _finite_nonnegative(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"evaluation corpus {field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"evaluation corpus {field_name} must be finite and non-negative")
    return result


def _token_count(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"evaluation corpus {field_name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class CorpusEvaluationObservation:
    """Per-document sufficient statistics for frozen corpus preservation gates."""

    record_id: str
    condition: str
    seed: int | None
    kind: str
    source: str
    clean_nll_sum: float
    clean_nll_tokens: int
    typo_nll_sum: float
    typo_nll_tokens: int
    base_clean_kl_sum: float
    base_clean_kl_tokens: int
    natural_clean_typo_kl_sum: float
    natural_clean_typo_kl_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or _SHA64.fullmatch(self.record_id) is None:
            raise ValueError("evaluation corpus observation record_id must be a SHA-256 digest")
        if self.condition not in _CONDITIONS:
            raise ValueError("evaluation corpus observation condition is unsupported")
        if self.condition == "base":
            if self.seed is not None:
                raise ValueError("base corpus observation must not have a seed")
        elif isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("adapter corpus observation must have a non-negative seed")
        if self.kind not in {"clean-corpus", "natural"}:
            raise ValueError("evaluation corpus observation kind is unsupported")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("evaluation corpus observation source must be non-empty")
        for field_name in (
            "clean_nll_sum",
            "typo_nll_sum",
            "base_clean_kl_sum",
            "natural_clean_typo_kl_sum",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_nonnegative(getattr(self, field_name), field_name=field_name),
            )
        for field_name, minimum in (
            ("clean_nll_tokens", 1),
            ("typo_nll_tokens", 0),
            ("base_clean_kl_tokens", 1),
            ("natural_clean_typo_kl_tokens", 0),
        ):
            _token_count(getattr(self, field_name), field_name=field_name, minimum=minimum)
        if self.kind == "clean-corpus":
            if self.typo_nll_tokens != 0 or self.natural_clean_typo_kl_tokens != 0:
                raise ValueError("clean corpus observations cannot contain natural-pair metrics")
            if self.typo_nll_sum != 0.0 or self.natural_clean_typo_kl_sum != 0.0:
                raise ValueError("clean corpus observations cannot contain natural-pair sums")
        elif self.typo_nll_tokens < 1 or self.natural_clean_typo_kl_tokens < 1:
            raise ValueError("natural corpus observations require typo likelihood and aligned KL")

    @property
    def condition_id(self) -> str:
        return "base" if self.condition == "base" else f"{self.condition}:seed-{self.seed}"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-evaluation-corpus-observation/v1",
            "record_id": self.record_id,
            "condition": self.condition,
            "seed": self.seed,
            "kind": self.kind,
            "source": self.source,
            "clean_nll_sum": self.clean_nll_sum,
            "clean_nll_tokens": self.clean_nll_tokens,
            "typo_nll_sum": self.typo_nll_sum,
            "typo_nll_tokens": self.typo_nll_tokens,
            "base_clean_kl_sum": self.base_clean_kl_sum,
            "base_clean_kl_tokens": self.base_clean_kl_tokens,
            "natural_clean_typo_kl_sum": self.natural_clean_typo_kl_sum,
            "natural_clean_typo_kl_tokens": self.natural_clean_typo_kl_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> CorpusEvaluationObservation:
        if (
            not isinstance(value, Mapping)
            or set(value) != _CORPUS_FIELDS
            or value.get("schema_version") != "robustness-evaluation-corpus-observation/v1"
        ):
            raise ValueError("evaluation corpus observation fields or schema differ")
        return cls(
            record_id=value["record_id"],  # type: ignore[arg-type]
            condition=value["condition"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            kind=value["kind"],  # type: ignore[arg-type]
            source=value["source"],  # type: ignore[arg-type]
            clean_nll_sum=value["clean_nll_sum"],  # type: ignore[arg-type]
            clean_nll_tokens=value["clean_nll_tokens"],  # type: ignore[arg-type]
            typo_nll_sum=value["typo_nll_sum"],  # type: ignore[arg-type]
            typo_nll_tokens=value["typo_nll_tokens"],  # type: ignore[arg-type]
            base_clean_kl_sum=value["base_clean_kl_sum"],  # type: ignore[arg-type]
            base_clean_kl_tokens=value["base_clean_kl_tokens"],  # type: ignore[arg-type]
            natural_clean_typo_kl_sum=value["natural_clean_typo_kl_sum"],  # type: ignore[arg-type]
            natural_clean_typo_kl_tokens=value["natural_clean_typo_kl_tokens"],  # type: ignore[arg-type]
        )


__all__ = ["CorpusEvaluationObservation", "EvaluationObservation"]
