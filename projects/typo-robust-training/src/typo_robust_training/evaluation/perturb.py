"""Meaning-preserving, text-level perturbations for the frozen evaluation."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from typo_robust_training.data.perturb import (
    apply_typo_operation_to_word,
    eligible_word_spans,
)
from typo_robust_training.data.records import CleanRecord, TypoEdit


_MULTIPLE_CHOICE_TASKS = frozenset({"mmlu", "arc", "mmlu_pro", "commonsense_qa"})
_SUPPORTED = frozenset(
    {
        "keyboard-neighbor-substitution",
        "deletion",
        "duplication",
        "adjacent-transposition",
    }
)
FROZEN_EVALUATION_TYPO_VERSION = "frozen-evaluation-typo/v2"
_ANSWER_WORD = re.compile(r"[A-Za-z]{3,}")
_DOLLAR_MATH_SPAN = re.compile(r"(?<!\\)\$(?!\s)(?:\\.|[^$\\\n])*(?<![\s\\])\$")
_MATH_SPANS = (
    re.compile(r"\\\([^\n]*?\\\)"),
    re.compile(r"\\\[[^\n]*?\\\]"),
    re.compile(r"\\[A-Za-z]+(?:\{[^{}]*\})*"),
)


@dataclass(frozen=True, slots=True)
class FrozenEvaluationTypo:
    """One deterministic realized typo condition for one clean source item."""

    record_id: str
    clean_text: str
    typo_text: str
    edits: tuple[TypoEdit, ...]
    condition: str
    seed: int
    variant: int
    metadata: Mapping[str, object]


class NoNaturalInjectionTargetError(ValueError):
    """The validated dictionary has no eligible target in one record."""


@dataclass(frozen=True, slots=True)
class NaturalInjectionDictionary:
    """One validated, immutable natural-typo dictionary reusable across records."""

    replacements: Mapping[str, tuple[str, ...]]
    minimum_word_letters: int


def freeze_natural_injection_dictionary(
    replacements: Mapping[str, Sequence[str]],
    *,
    minimum_word_letters: int = 3,
) -> NaturalInjectionDictionary:
    """Validate and normalize a natural-typo dictionary exactly once."""

    if (
        isinstance(minimum_word_letters, bool)
        or not isinstance(minimum_word_letters, int)
        or minimum_word_letters <= 0
    ):
        raise ValueError("minimum_word_letters must be a positive integer")
    if not isinstance(replacements, Mapping):
        raise TypeError("natural injection dictionary must be a mapping")
    normalized: dict[str, tuple[str, ...]] = {}
    for clean, typos in replacements.items():
        if not isinstance(typos, Sequence) or isinstance(typos, (str, bytes)):
            raise ValueError("natural injection dictionary contains an invalid entry")
        values = tuple(sorted(set(typos)))
        if (
            not isinstance(clean, str)
            or not clean.isascii()
            or not clean.isalpha()
            or len(clean) < minimum_word_letters
            or not values
            or any(
                not isinstance(typo, str)
                or not typo.isascii()
                or not typo.isalpha()
                or typo.casefold() == clean.casefold()
                for typo in values
            )
        ):
            raise ValueError("natural injection dictionary contains an invalid entry")
        key = clean.casefold()
        if key in normalized:
            raise ValueError("natural injection dictionary contains duplicate corrected words")
        normalized[key] = values
    return NaturalInjectionDictionary(
        replacements=MappingProxyType(normalized),
        minimum_word_letters=minimum_word_letters,
    )


def _question_stop(record: CleanRecord) -> int:
    if record.task in _MULTIPLE_CHOICE_TASKS:
        separator = record.text.find("\n")
        if separator <= 0:
            raise ValueError("multiple-choice evaluation record has no option separator")
        return separator
    return len(record.text)


def _question_span_metadata(record: CleanRecord, typo_text: str) -> Mapping[str, object]:
    clean_stop = _question_stop(record)
    if record.task in _MULTIPLE_CHOICE_TASKS:
        typo_stop = typo_text.find("\n")
        if typo_stop <= 0:
            raise RuntimeError("multiple-choice typo text has no option separator")
    else:
        typo_stop = len(typo_text)
    return MappingProxyType(
        {
            "clean_question_char_span": [0, clean_stop],
            "typo_question_char_span": [0, typo_stop],
        }
    )


def evaluation_eligible_word_spans(
    record: CleanRecord,
    *,
    minimum_word_letters: int,
) -> tuple[tuple[int, int], ...]:
    """Return question-only spans after answer and structural exclusions."""

    if not isinstance(record, CleanRecord) or record.task is None or record.answer is None:
        raise TypeError("evaluation typo targets require a task CleanRecord with an answer")
    stop = _question_stop(record)
    question = record.text[:stop]
    answer_words = {word.casefold() for word in _ANSWER_WORD.findall(record.answer)}
    math_patterns = (*_MATH_SPANS, _DOLLAR_MATH_SPAN) if record.task == "math_500" else _MATH_SPANS
    math_spans = tuple(
        match.span() for pattern in math_patterns for match in pattern.finditer(question)
    )
    return tuple(
        (start, end)
        for start, end in eligible_word_spans(
            question,
            minimum_letters=minimum_word_letters,
        )
        if question[start:end].casefold() not in answer_words
        and not any(
            start < blocked_end and blocked_start < end for blocked_start, blocked_end in math_spans
        )
    )


def _rng(
    record: CleanRecord,
    *,
    condition: str,
    seed: int,
    role: str,
    variant: int,
) -> random.Random:
    material = (
        f"{FROZEN_EVALUATION_TYPO_VERSION}\0{seed}\0{role}\0{condition}\0{variant}\0"
        f"{record.record_id}"
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _record_id(
    record: CleanRecord,
    *,
    condition: str,
    role: str,
    seed: int,
    variant: int,
    edit_count: int,
) -> str:
    return hashlib.sha256(
        (
            "frozen-evaluation-pair/v2\0"
            f"{role}\0{condition}\0{seed}\0{variant}\0{edit_count}\0{record.record_id}"
        ).encode()
    ).hexdigest()


def generate_evaluation_typo(
    record: CleanRecord,
    *,
    condition: str,
    edit_count: int,
    operations: Sequence[str],
    seed: int,
    role: str,
    variant: int,
    minimum_word_letters: int = 3,
) -> FrozenEvaluationTypo:
    """Generate one exact, replayable evaluation text without shared RNG state."""

    if not isinstance(record, CleanRecord) or record.task is None or record.answer is None:
        raise TypeError("evaluation typo generation requires a task CleanRecord")
    if not isinstance(condition, str) or not condition or not isinstance(role, str) or not role:
        raise ValueError("evaluation condition and role must be non-empty")
    if isinstance(edit_count, bool) or not isinstance(edit_count, int) or edit_count < 0:
        raise ValueError("evaluation edit_count must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("evaluation seed must be a non-negative integer")
    if isinstance(variant, bool) or not isinstance(variant, int) or variant < 0:
        raise ValueError("evaluation variant must be a non-negative integer")
    operation_inventory = tuple(operations)
    if edit_count == 0:
        if operation_inventory:
            raise ValueError("zero-edit evaluation cannot declare operations")
        return FrozenEvaluationTypo(
            record_id=_record_id(
                record,
                condition=condition,
                role=role,
                seed=seed,
                variant=variant,
                edit_count=0,
            ),
            clean_text=record.text,
            typo_text=record.text,
            edits=(),
            condition=condition,
            seed=seed,
            variant=variant,
            metadata=MappingProxyType(
                {
                    "evaluation_condition": condition,
                    "base_record_id": record.record_id,
                    **_question_span_metadata(record, record.text),
                }
            ),
        )
    if (
        not operation_inventory
        or len(set(operation_inventory)) != len(operation_inventory)
        or any(operation not in _SUPPORTED for operation in operations)
    ):
        raise ValueError("evaluation operation inventory is empty or unsupported")
    spans = evaluation_eligible_word_spans(
        record,
        minimum_word_letters=minimum_word_letters,
    )

    def compatible_operations(word: str) -> tuple[str, ...]:
        transposable = any(left != right for left, right in zip(word, word[1:]))
        return tuple(
            operation
            for operation in operation_inventory
            if operation != "adjacent-transposition" or transposable
        )

    spans_by_word: dict[str, list[tuple[int, int]]] = {}
    for span in spans:
        word = record.text[slice(*span)]
        if compatible_operations(word):
            spans_by_word.setdefault(word.casefold(), []).append(span)
    if len(spans_by_word) < edit_count:
        raise ValueError(
            "evaluation record has "
            f"{len(spans_by_word)} eligible distinct words but requires {edit_count}"
        )
    rng = _rng(record, condition=condition, seed=seed, role=role, variant=variant)
    selected_word_keys = rng.sample(tuple(spans_by_word), edit_count)
    selected = sorted(rng.choice(spans_by_word[key]) for key in selected_word_keys)
    replacements: list[tuple[tuple[int, int], str, str, str]] = []
    for span in selected:
        clean_word = record.text[slice(*span)]
        operation = rng.choice(compatible_operations(clean_word))
        typo_word = apply_typo_operation_to_word(clean_word, operation, rng)
        if typo_word == clean_word:
            raise RuntimeError("evaluation typo operation produced an identity edit")
        replacements.append((span, operation, clean_word, typo_word))

    chunks: list[str] = []
    edits: list[TypoEdit] = []
    clean_cursor = 0
    typo_cursor = 0
    for (start, stop), operation, clean_word, typo_word in replacements:
        prefix = record.text[clean_cursor:start]
        chunks.extend((prefix, typo_word))
        typo_cursor += len(prefix)
        typo_start = typo_cursor
        typo_stop = typo_start + len(typo_word)
        edits.append(
            TypoEdit(
                operation=operation,
                clean_word=clean_word,
                typo_word=typo_word,
                clean_char_span=(start, stop),
                typo_char_span=(typo_start, typo_stop),
            )
        )
        clean_cursor = stop
        typo_cursor = typo_stop
    chunks.append(record.text[clean_cursor:])
    typo_text = "".join(chunks)
    if len(edits) != edit_count or len({edit.clean_char_span for edit in edits}) != edit_count:
        raise RuntimeError("evaluation typo generator violated exact distinct-edit count")
    if len({edit.clean_word.casefold() for edit in edits}) != edit_count:
        raise RuntimeError("evaluation typo generator violated exact distinct-word count")
    for edit in edits:
        if (
            record.text[slice(*edit.clean_char_span)] != edit.clean_word
            or typo_text[slice(*edit.typo_char_span)] != edit.typo_word
        ):
            raise RuntimeError("evaluation typo spans do not round-trip")
    return FrozenEvaluationTypo(
        record_id=_record_id(
            record,
            condition=condition,
            role=role,
            seed=seed,
            variant=variant,
            edit_count=edit_count,
        ),
        clean_text=record.text,
        typo_text=typo_text,
        edits=tuple(edits),
        condition=condition,
        seed=seed,
        variant=variant,
        metadata=MappingProxyType(
            {
                "evaluation_condition": condition,
                "base_record_id": record.record_id,
                **_question_span_metadata(record, typo_text),
            }
        ),
    )


def generate_natural_injection(
    record: CleanRecord,
    *,
    replacements: Mapping[str, Sequence[str]] | NaturalInjectionDictionary,
    seed: int,
    role: str,
    variant: int,
    minimum_word_letters: int = 3,
) -> FrozenEvaluationTypo:
    """Inject one held-out real misspelling into an eligible question word."""

    dictionary = (
        replacements
        if isinstance(replacements, NaturalInjectionDictionary)
        else freeze_natural_injection_dictionary(
            replacements,
            minimum_word_letters=minimum_word_letters,
        )
    )
    if dictionary.minimum_word_letters != minimum_word_letters:
        raise ValueError("natural injection dictionary eligibility threshold differs")
    normalized = dictionary.replacements
    candidates = [
        span
        for span in evaluation_eligible_word_spans(
            record,
            minimum_word_letters=minimum_word_letters,
        )
        if record.text[slice(*span)].casefold() in normalized
    ]
    if not candidates:
        raise NoNaturalInjectionTargetError(
            "evaluation record has no held-out natural dictionary target"
        )
    rng = _rng(
        record,
        condition="natural-injection",
        seed=seed,
        role=role,
        variant=variant,
    )
    start, stop = rng.choice(candidates)
    clean_word = record.text[start:stop]
    typo_word = rng.choice(normalized[clean_word.casefold()])
    if clean_word.isupper():
        typo_word = typo_word.upper()
    elif clean_word[:1].isupper():
        typo_word = typo_word[:1].upper() + typo_word[1:]
    typo_text = record.text[:start] + typo_word + record.text[stop:]
    edit = TypoEdit(
        operation="natural-dictionary-substitution",
        clean_word=clean_word,
        typo_word=typo_word,
        clean_char_span=(start, stop),
        typo_char_span=(start, start + len(typo_word)),
    )
    return FrozenEvaluationTypo(
        record_id=_record_id(
            record,
            condition="natural-injection",
            role=role,
            seed=seed,
            variant=variant,
            edit_count=1,
        ),
        clean_text=record.text,
        typo_text=typo_text,
        edits=(edit,),
        condition="natural-injection",
        seed=seed,
        variant=variant,
        metadata=MappingProxyType(
            {
                "evaluation_condition": "natural-injection",
                "base_record_id": record.record_id,
                **_question_span_metadata(record, typo_text),
            }
        ),
    )


__all__ = [
    "FROZEN_EVALUATION_TYPO_VERSION",
    "FrozenEvaluationTypo",
    "NaturalInjectionDictionary",
    "NoNaturalInjectionTargetError",
    "evaluation_eligible_word_spans",
    "freeze_natural_injection_dictionary",
    "generate_evaluation_typo",
    "generate_natural_injection",
]
