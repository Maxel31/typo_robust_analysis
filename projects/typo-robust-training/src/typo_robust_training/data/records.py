"""Validated records shared by data construction and training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


_SHA40_LENGTH = 40


def _nonempty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be an object")
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be canonical-JSON-compatible") from exc
    return MappingProxyType(dict(value))


def record_id_for(*, source: str, source_revision: str, source_id: str) -> str:
    payload = json.dumps(
        {
            "source": source,
            "source_id": source_id,
            "source_revision": source_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CleanRecord:
    """One normalized clean source example before synthetic perturbation."""

    source: str
    source_revision: str
    source_split: str
    source_id: str
    group_id: str
    text: str
    task: str | None
    answer: str | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "source",
            "source_revision",
            "source_split",
            "source_id",
            "group_id",
            "text",
        ):
            _nonempty(getattr(self, field_name), field_name=field_name)
        if len(self.source_revision) != _SHA40_LENGTH:
            raise ValueError("source_revision must be a pinned 40-character SHA")
        if self.task is not None:
            _nonempty(self.task, field_name="task")
        if self.answer is not None and not isinstance(self.answer, str):
            raise TypeError("answer must be null or a string")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def record_id(self) -> str:
        return record_id_for(
            source=self.source,
            source_revision=self.source_revision,
            source_id=self.source_id,
        )


@dataclass(frozen=True, slots=True)
class NaturalTypoRecord:
    """One licensed natural clean/typo correction pair."""

    source: str
    source_revision: str
    source_split: str
    source_id: str
    group_id: str
    clean_text: str
    typo_text: str
    repository: str
    repository_license: str
    operation: str
    training_eligible: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "source",
            "source_revision",
            "source_split",
            "source_id",
            "group_id",
            "clean_text",
            "typo_text",
            "repository",
            "repository_license",
            "operation",
        ):
            _nonempty(getattr(self, field_name), field_name=field_name)
        if len(self.source_revision) != _SHA40_LENGTH:
            raise ValueError("source_revision must be a pinned 40-character SHA")
        if self.clean_text == self.typo_text:
            raise ValueError("natural clean and typo text must differ")
        if type(self.training_eligible) is not bool:
            raise TypeError("training_eligible must be boolean")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def record_id(self) -> str:
        return record_id_for(
            source=self.source,
            source_revision=self.source_revision,
            source_id=self.source_id,
        )


@dataclass(frozen=True, slots=True)
class TypoEdit:
    """One aligned original-word to typo-word edit."""

    operation: str
    clean_word: str
    typo_word: str
    clean_char_span: tuple[int, int]
    typo_char_span: tuple[int, int]

    def __post_init__(self) -> None:
        for field_name in ("operation", "clean_word", "typo_word"):
            _nonempty(getattr(self, field_name), field_name=field_name)
        if self.clean_word == self.typo_word:
            raise ValueError("typo edit must change its word")
        for field_name in ("clean_char_span", "typo_char_span"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
                or value[0] < 0
                or value[1] <= value[0]
            ):
                raise ValueError(f"{field_name} must be an increasing integer span")


@dataclass(frozen=True, slots=True)
class TypoPair:
    """A deterministically generated clean/typo pair for one epoch."""

    record_id: str
    source_id: str
    clean_text: str
    typo_text: str
    seed: int
    epoch: int
    variant: int
    edits: tuple[TypoEdit, ...]
    is_noop: bool = False

    def __post_init__(self) -> None:
        for field_name in ("record_id", "source_id", "clean_text", "typo_text"):
            _nonempty(getattr(self, field_name), field_name=field_name)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.seed, self.epoch, self.variant)
        ):
            raise ValueError("seed, epoch, and variant must be non-negative integers")
        if not isinstance(self.edits, tuple) or any(
            not isinstance(edit, TypoEdit) for edit in self.edits
        ):
            raise TypeError("edits must be a tuple of TypoEdit records")
        if type(self.is_noop) is not bool:
            raise TypeError("is_noop must be boolean")
        if self.is_noop:
            if self.edits or self.clean_text != self.typo_text:
                raise ValueError("noop pairs must be identical and contain no edits")
        elif not self.edits or self.clean_text == self.typo_text:
            raise ValueError("perturbed pairs must differ and contain edits")


def finite_number(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


__all__ = [
    "CleanRecord",
    "NaturalTypoRecord",
    "TypoEdit",
    "TypoPair",
    "finite_number",
    "record_id_for",
]
