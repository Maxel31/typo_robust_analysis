"""Validated records shared by data construction and training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


_SOURCE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_LETTER = re.compile(r"[A-Za-z]")


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
        if _SOURCE_REVISION.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a pinned 40- or 64-character SHA")
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
        if _SOURCE_REVISION.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a pinned 40- or 64-character SHA")
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


def _is_word_character(text: str, index: int) -> bool:
    if not 0 <= index < len(text):
        return False
    if _LETTER.fullmatch(text[index]):
        return True
    return (
        text[index] == "'"
        and index > 0
        and index + 1 < len(text)
        and _LETTER.fullmatch(text[index - 1]) is not None
        and _LETTER.fullmatch(text[index + 1]) is not None
    )


def _word_span(text: str, start: int, stop: int) -> tuple[int, int]:
    anchor = start
    if start == stop and not _is_word_character(text, anchor):
        anchor -= 1
    if not _is_word_character(text, anchor):
        raise ValueError("natural typo change is not inside an English word")
    left = anchor
    right = max(anchor + 1, stop)
    while _is_word_character(text, left - 1):
        left -= 1
    while _is_word_character(text, right):
        right += 1
    return left, right


def infer_single_word_typo_edit(
    clean_text: str,
    typo_text: str,
    *,
    operation: str,
) -> TypoEdit:
    """Infer the one edited word shared by natural training and evaluation."""

    for field_name, value in (
        ("clean_text", clean_text),
        ("typo_text", typo_text),
        ("operation", operation),
    ):
        _nonempty(value, field_name=field_name)
    if clean_text == typo_text:
        raise ValueError("natural typo pair must differ")
    prefix = 0
    while prefix < min(len(clean_text), len(typo_text)) and clean_text[prefix] == typo_text[prefix]:
        prefix += 1
    clean_stop, typo_stop = len(clean_text), len(typo_text)
    while (
        clean_stop > prefix
        and typo_stop > prefix
        and clean_text[clean_stop - 1] == typo_text[typo_stop - 1]
    ):
        clean_stop -= 1
        typo_stop -= 1
    clean_span = _word_span(clean_text, prefix, clean_stop)
    typo_span = _word_span(typo_text, prefix, typo_stop)
    if (
        clean_text[: clean_span[0]] != typo_text[: typo_span[0]]
        or clean_text[clean_span[1] :] != typo_text[typo_span[1] :]
    ):
        raise ValueError("natural typo does not isolate one aligned edited word")
    return TypoEdit(
        operation=operation,
        clean_word=clean_text[slice(*clean_span)],
        typo_word=typo_text[slice(*typo_span)],
        clean_char_span=clean_span,
        typo_char_span=typo_span,
    )


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
