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
_ANSWER_WORD = re.compile(r"[A-Za-z]{3,}")
_MATH_SPANS = (
    re.compile(r"\$[^$\n]+\$"),
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


def _question_stop(record: CleanRecord) -> int:
    if record.task in _MULTIPLE_CHOICE_TASKS:
        separator = record.text.find("\n")
        if separator <= 0:
            raise ValueError("multiple-choice evaluation record has no option separator")
        return separator
    return len(record.text)


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
    math_spans = tuple(
        match.span() for pattern in _MATH_SPANS for match in pattern.finditer(question)
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
        f"frozen-evaluation-typo/v1\0{seed}\0{role}\0{condition}\0{variant}\0{record.record_id}"
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _record_id(record: CleanRecord, *, condition: str, role: str) -> str:
    return hashlib.sha256(
        f"frozen-evaluation-pair/v1\0{role}\0{condition}\0{record.record_id}".encode()
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
            record_id=_record_id(record, condition=condition, role=role),
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
                    "question_char_span": [0, _question_stop(record)],
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
    if operation_inventory == ("adjacent-transposition",):
        spans = tuple(
            span
            for span in spans
            if any(
                left != right
                for left, right in zip(
                    record.text[slice(*span)],
                    record.text[slice(*span)][1:],
                )
            )
        )
    if len(spans) < edit_count:
        raise ValueError(
            f"evaluation record has {len(spans)} eligible words but requires {edit_count}"
        )
    rng = _rng(record, condition=condition, seed=seed, role=role, variant=variant)
    selected = sorted(rng.sample(spans, edit_count))
    replacements: list[tuple[tuple[int, int], str, str, str]] = []
    for span in selected:
        operation = rng.choice(operation_inventory)
        clean_word = record.text[slice(*span)]
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
    for edit in edits:
        if (
            record.text[slice(*edit.clean_char_span)] != edit.clean_word
            or typo_text[slice(*edit.typo_char_span)] != edit.typo_word
        ):
            raise RuntimeError("evaluation typo spans do not round-trip")
    return FrozenEvaluationTypo(
        record_id=_record_id(record, condition=condition, role=role),
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
                "question_char_span": [0, _question_stop(record)],
            }
        ),
    )


def generate_natural_injection(
    record: CleanRecord,
    *,
    replacements: Mapping[str, Sequence[str]],
    seed: int,
    role: str,
    variant: int,
    minimum_word_letters: int = 3,
) -> FrozenEvaluationTypo:
    """Inject one held-out real misspelling into an eligible question word."""

    normalized: dict[str, tuple[str, ...]] = {}
    for clean, typos in replacements.items():
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
    candidates = [
        span
        for span in evaluation_eligible_word_spans(
            record,
            minimum_word_letters=minimum_word_letters,
        )
        if record.text[slice(*span)].casefold() in normalized
    ]
    if not candidates:
        raise ValueError("evaluation record has no held-out natural dictionary target")
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
        record_id=_record_id(record, condition="natural-injection", role=role),
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
                "question_char_span": [0, _question_stop(record)],
            }
        ),
    )


__all__ = [
    "FrozenEvaluationTypo",
    "evaluation_eligible_word_spans",
    "generate_evaluation_typo",
    "generate_natural_injection",
]
