"""Strict, versioned configuration for robustness-training data."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_REVISION = re.compile(r"[0-9a-f]{40}")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "stage",
    "seed",
    "model",
    "model_revision",
    "training",
    "preprocessing",
    "typos",
    "splits",
    "sampling",
    "streaming",
    "sources",
}
_SOURCE_FIELDS = {
    "dataset",
    "revision",
    "subset",
    "splits",
    "role",
    "license",
    "streaming",
    "task",
}
_SOURCE_NAMES = (
    "fineweb_edu",
    "gsm8k",
    "mmlu",
    "arc",
    "github_typo_corpus",
    "dolma",
    "mmlu_pro",
    "math_500",
    "commonsense_qa",
)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def strict_loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON in {context}: {exc}") from exc


def _mapping(value: object, *, field: str, fields: set[str] | None = None) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if fields is not None and set(value) != fields:
        raise ValueError(f"{field} fields differ")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _probability(value: object, *, field: str, allow_zero: bool = True) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or float(value) > 1.0
        or (not allow_zero and float(value) == 0.0)
    ):
        raise ValueError(f"{field} must be a finite probability")
    return float(value)


def _probability_map(
    value: object,
    *,
    field: str,
    keys: tuple[str, ...],
) -> Mapping[str, float]:
    payload = _mapping(value, field=field)
    if set(payload) != set(keys):
        raise ValueError(f"{field} keys differ")
    result = {key: _probability(payload[key], field=f"{field}.{key}") for key in keys}
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field} probabilities must sum to one")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """One pinned dataset and its one prespecified scientific role."""

    name: str
    dataset: str
    revision: str
    subset: str | None
    splits: tuple[str, ...]
    role: str
    license: str
    streaming: bool
    task: str | None


@dataclass(frozen=True, slots=True)
class TrainingDataProtocol:
    """Resolved immutable data protocol."""

    schema_version: str
    stage: str
    seed: int
    model: str
    model_revision: str
    training_token_budget: int
    max_sequence_length: int
    document_character_window: int
    training_mixture: Mapping[str, float]
    explicit_clean_pair_probability: float
    edit_count_probabilities: Mapping[str, float]
    training_operations: tuple[str, ...]
    operation_probabilities: Mapping[str, float]
    held_out_operations: tuple[str, ...]
    minimum_word_letters: int
    distinct_words: bool
    fineweb_content_split: Mapping[str, float]
    natural_repository_split: Mapping[str, float]
    held_out_repository_evaluation_split: Mapping[str, float]
    diagnostic_per_task: int
    fixed_pairs_per_source_split: int
    shuffle_buffer_size: int
    sources: Mapping[str, DatasetSource]
    config_sha256: str

    @property
    def diagnostic_tasks(self) -> tuple[str, ...]:
        return ("gsm8k", "mmlu", "arc")

    @property
    def unseen_tasks(self) -> tuple[str, ...]:
        return ("mmlu_pro", "math_500", "commonsense_qa")

    @property
    def unseen_domain(self) -> str:
        return "dolma"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "seed": self.seed,
            "model": self.model,
            "model_revision": self.model_revision,
            "training": {
                "token_budget": self.training_token_budget,
                "max_sequence_length": self.max_sequence_length,
                "mixture": dict(self.training_mixture),
                "explicit_clean_pair_probability": self.explicit_clean_pair_probability,
            },
            "preprocessing": {
                "document_character_window": self.document_character_window,
            },
            "typos": {
                "edit_count_probabilities": dict(self.edit_count_probabilities),
                "training_operations": list(self.training_operations),
                "operation_probabilities": dict(self.operation_probabilities),
                "held_out_operations": list(self.held_out_operations),
                "minimum_word_letters": self.minimum_word_letters,
                "distinct_words": self.distinct_words,
            },
            "splits": {
                "fineweb_content": dict(self.fineweb_content_split),
                "natural_repository": dict(self.natural_repository_split),
                "held_out_repository_evaluation": dict(self.held_out_repository_evaluation_split),
            },
            "sampling": {
                "diagnostic_per_task": self.diagnostic_per_task,
                "fixed_pairs_per_source_split": self.fixed_pairs_per_source_split,
            },
            "streaming": {"shuffle_buffer_size": self.shuffle_buffer_size},
            "sources": {
                name: {
                    "dataset": source.dataset,
                    "revision": source.revision,
                    "subset": source.subset,
                    "splits": list(source.splits),
                    "role": source.role,
                    "license": source.license,
                    "streaming": source.streaming,
                    "task": source.task,
                }
                for name, source in self.sources.items()
            },
        }


def _load_source(name: str, value: object) -> DatasetSource:
    payload = _mapping(value, field=f"sources.{name}", fields=_SOURCE_FIELDS)
    revision = _string(payload["revision"], field=f"sources.{name}.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError(f"sources.{name}.revision must be a pinned 40-character SHA")
    subset = payload["subset"]
    if subset is not None and (not isinstance(subset, str) or not subset):
        raise ValueError(f"sources.{name}.subset must be null or a non-empty string")
    splits = payload["splits"]
    if (
        not isinstance(splits, list)
        or not splits
        or any(not isinstance(split, str) or not split for split in splits)
    ):
        raise ValueError(f"sources.{name}.splits must be non-empty strings")
    if len(set(splits)) != len(splits):
        raise ValueError(f"sources.{name}.splits must be unique")
    streaming = payload["streaming"]
    if type(streaming) is not bool:
        raise ValueError(f"sources.{name}.streaming must be boolean")
    task = payload["task"]
    if task is not None and (not isinstance(task, str) or not task):
        raise ValueError(f"sources.{name}.task must be null or a non-empty string")
    return DatasetSource(
        name=name,
        dataset=_string(payload["dataset"], field=f"sources.{name}.dataset"),
        revision=revision,
        subset=subset,
        splits=tuple(splits),
        role=_string(payload["role"], field=f"sources.{name}.role"),
        license=_string(payload["license"], field=f"sources.{name}.license"),
        streaming=streaming,
        task=task,
    )


def load_training_data_config(path: Path) -> TrainingDataProtocol:
    """Load the strict v1 data protocol without accepting silent drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"training data config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"training data config is not UTF-8: {resolved}") from exc
    root = _mapping(
        strict_loads(text, context=str(resolved)), field="config", fields=_TOP_LEVEL_FIELDS
    )
    if root["schema_version"] != "robustness-training-data-config/v1":
        raise ValueError("training data schema_version differs")
    if root["stage"] != "sanity" or root["seed"] != 42:
        raise ValueError("training data stage or seed differs")

    training = _mapping(
        root["training"],
        field="training",
        fields={
            "token_budget",
            "max_sequence_length",
            "mixture",
            "explicit_clean_pair_probability",
        },
    )
    mixture = _probability_map(
        training["mixture"],
        field="training.mixture",
        keys=("fineweb_edu", "reasoning", "natural_typo"),
    )
    preprocessing = _mapping(
        root["preprocessing"],
        field="preprocessing",
        fields={"document_character_window"},
    )
    typos = _mapping(
        root["typos"],
        field="typos",
        fields={
            "edit_count_probabilities",
            "training_operations",
            "operation_probabilities",
            "held_out_operations",
            "minimum_word_letters",
            "distinct_words",
        },
    )
    edit_counts = _probability_map(
        typos["edit_count_probabilities"],
        field="typos.edit_count_probabilities",
        keys=("1", "2", "3-4"),
    )
    training_operations = typos["training_operations"]
    held_out_operations = typos["held_out_operations"]
    if (
        not isinstance(training_operations, list)
        or not training_operations
        or any(not isinstance(value, str) or not value for value in training_operations)
    ):
        raise ValueError("typos.training_operations must be non-empty strings")
    if (
        not isinstance(held_out_operations, list)
        or not held_out_operations
        or any(not isinstance(value, str) or not value for value in held_out_operations)
    ):
        raise ValueError("typos.held_out_operations must be non-empty strings")
    if set(training_operations) & set(held_out_operations):
        raise ValueError("training and held-out typo operations overlap")
    operation_probabilities = _probability_map(
        typos["operation_probabilities"],
        field="typos.operation_probabilities",
        keys=tuple(training_operations),
    )
    if tuple(held_out_operations) != ("adjacent-transposition",):
        raise ValueError("v1 held-out typo operation differs")
    distinct_words = typos["distinct_words"]
    if distinct_words is not True:
        raise ValueError("v1 typo edits must target distinct words")

    splits = _mapping(
        root["splits"],
        field="splits",
        fields={
            "fineweb_content",
            "natural_repository",
            "held_out_repository_evaluation",
        },
    )
    fineweb_split = _probability_map(
        splits["fineweb_content"],
        field="splits.fineweb_content",
        keys=("train", "tune", "pre_pr_gate", "final_test"),
    )
    natural_split = _probability_map(
        splits["natural_repository"],
        field="splits.natural_repository",
        keys=("train", "tune", "held_out"),
    )
    heldout_natural_split = _probability_map(
        splits["held_out_repository_evaluation"],
        field="splits.held_out_repository_evaluation",
        keys=("pre_pr_gate", "final_test"),
    )
    sampling = _mapping(
        root["sampling"],
        field="sampling",
        fields={"diagnostic_per_task", "fixed_pairs_per_source_split"},
    )
    streaming = _mapping(
        root["streaming"],
        field="streaming",
        fields={"shuffle_buffer_size"},
    )
    sources_payload = _mapping(root["sources"], field="sources")
    if set(sources_payload) != set(_SOURCE_NAMES):
        raise ValueError("training data source inventory differs")
    sources = MappingProxyType(
        {name: _load_source(name, sources_payload[name]) for name in _SOURCE_NAMES}
    )

    protocol = TrainingDataProtocol(
        schema_version="robustness-training-data-config/v1",
        stage="sanity",
        seed=42,
        model=_string(root["model"], field="model"),
        model_revision=_string(root["model_revision"], field="model_revision"),
        training_token_budget=_positive_int(
            training["token_budget"], field="training.token_budget"
        ),
        max_sequence_length=_positive_int(
            training["max_sequence_length"], field="training.max_sequence_length"
        ),
        document_character_window=_positive_int(
            preprocessing["document_character_window"],
            field="preprocessing.document_character_window",
        ),
        training_mixture=mixture,
        explicit_clean_pair_probability=_probability(
            training["explicit_clean_pair_probability"],
            field="training.explicit_clean_pair_probability",
        ),
        edit_count_probabilities=edit_counts,
        training_operations=tuple(training_operations),
        operation_probabilities=operation_probabilities,
        held_out_operations=tuple(held_out_operations),
        minimum_word_letters=_positive_int(
            typos["minimum_word_letters"], field="typos.minimum_word_letters"
        ),
        distinct_words=True,
        fineweb_content_split=fineweb_split,
        natural_repository_split=natural_split,
        held_out_repository_evaluation_split=heldout_natural_split,
        diagnostic_per_task=_positive_int(
            sampling["diagnostic_per_task"], field="sampling.diagnostic_per_task"
        ),
        fixed_pairs_per_source_split=_positive_int(
            sampling["fixed_pairs_per_source_split"],
            field="sampling.fixed_pairs_per_source_split",
        ),
        shuffle_buffer_size=_positive_int(
            streaming["shuffle_buffer_size"], field="streaming.shuffle_buffer_size"
        ),
        sources=sources,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    if protocol.max_sequence_length != 512:
        raise ValueError("sanity max_sequence_length must remain 512")
    if protocol.document_character_window != 8_192:
        raise ValueError("sanity document_character_window must remain 8192")
    if _REVISION.fullmatch(protocol.model_revision) is None:
        raise ValueError("model_revision must be a pinned 40-character SHA")
    return protocol


__all__ = [
    "DatasetSource",
    "TrainingDataProtocol",
    "load_training_data_config",
    "strict_loads",
]
