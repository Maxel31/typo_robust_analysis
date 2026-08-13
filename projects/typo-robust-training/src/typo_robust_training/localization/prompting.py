"""Build paper prompts while transporting raw-text edit coordinates exactly."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptSide:
    """One complete prompt and its edited-word character spans."""

    text: str
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class DiagnosticPrompts:
    """Clean and typo prompts built from one frozen diagnostic pair."""

    record_id: str
    task: str
    answer: str
    clean: PromptSide
    typo: PromptSide


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _raw_span(value: object, *, field: str, text: str, expected_word: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{field} must contain two integers")
    start, stop = value
    if not 0 <= start < stop <= len(text):
        raise ValueError(f"{field} is outside its raw text")
    if text[start:stop] != expected_word:
        raise ValueError(f"{field} does not select the recorded word")
    return start, stop


def build_diagnostic_prompts(record: Mapping[str, object]) -> DiagnosticPrompts:
    """Insert clean and typo text into the exact paper few-shot templates."""

    if not isinstance(record, Mapping):
        raise TypeError("diagnostic record must be an object")
    record_id = _text(record.get("record_id"), field="record_id")
    task = _text(record.get("task"), field="task")
    if task not in {"gsm8k", "mmlu", "arc"}:
        raise ValueError("diagnostic task is unsupported")
    answer = _text(record.get("answer"), field="answer")
    clean_text = _text(record.get("clean_text"), field="clean_text")
    typo_text = _text(record.get("typo_text"), field="typo_text")
    edits = record.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("diagnostic edits must be a non-empty list")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("diagnostic metadata must be an object")
    subject = metadata.get("subject")
    if subject is not None and not isinstance(subject, str):
        raise ValueError("diagnostic metadata.subject must be a string or null")

    from typo_cot.models.prompts import create_prompt_template

    template = create_prompt_template(task)
    clean_result = template.generate(question=clean_text, subject=subject)
    typo_result = template.generate(question=typo_text, subject=subject)
    clean_prompt = clean_result.get_full_prompt()
    typo_prompt = typo_result.get_full_prompt()
    clean_offset = int(clean_result.question_start_in_full)
    typo_offset = int(typo_result.question_start_in_full)
    clean_spans: list[tuple[int, int]] = []
    typo_spans: list[tuple[int, int]] = []
    for index, raw_edit in enumerate(edits):
        if not isinstance(raw_edit, Mapping):
            raise ValueError(f"edits[{index}] must be an object")
        clean_word = _text(raw_edit.get("clean_word"), field=f"edits[{index}].clean_word")
        typo_word = _text(raw_edit.get("typo_word"), field=f"edits[{index}].typo_word")
        clean_start, clean_stop = _raw_span(
            raw_edit.get("clean_char_span"),
            field=f"edits[{index}].clean_char_span",
            text=clean_text,
            expected_word=clean_word,
        )
        typo_start, typo_stop = _raw_span(
            raw_edit.get("typo_char_span"),
            field=f"edits[{index}].typo_char_span",
            text=typo_text,
            expected_word=typo_word,
        )
        clean_spans.append((clean_offset + clean_start, clean_offset + clean_stop))
        typo_spans.append((typo_offset + typo_start, typo_offset + typo_stop))
    if len(set(clean_spans)) != len(clean_spans) or len(set(typo_spans)) != len(typo_spans):
        raise ValueError("diagnostic edited-word spans are duplicated")
    for prompt, spans, side in (
        (clean_prompt, clean_spans, "clean"),
        (typo_prompt, typo_spans, "typo"),
    ):
        if any(not 0 <= start < stop <= len(prompt) for start, stop in spans):
            raise ValueError(f"{side} prompt span is outside the complete prompt")
    return DiagnosticPrompts(
        record_id=record_id,
        task=task,
        answer=answer,
        clean=PromptSide(clean_prompt, tuple(clean_spans)),
        typo=PromptSide(typo_prompt, tuple(typo_spans)),
    )


def _flat(value: object, *, field: str) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError(f"tokenizer {field} must be one sequence")
    return value


def word_final_token_positions(
    tokenizer: Any,
    *,
    text: str,
    spans: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Resolve exact word-final token indices from fast-tokenizer offsets."""

    if not isinstance(text, str) or not text:
        raise ValueError("prompt text must be non-empty")
    normalized = tuple(spans)
    if not normalized:
        raise ValueError("edited-word spans must not be empty")
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise ValueError("tokenizer must expose exact offset_mapping") from exc
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer output must be an object")
    input_ids = _flat(encoded.get("input_ids"), field="input_ids")
    offsets = _flat(encoded.get("offset_mapping"), field="offset_mapping")
    if len(input_ids) != len(offsets):
        raise ValueError("tokenizer offsets and input IDs differ in length")
    normalized_offsets: list[tuple[int, int]] = []
    for index, raw in enumerate(offsets):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(f"tokenizer offset_mapping[{index}] is invalid")
        start, stop = int(raw[0]), int(raw[1])
        while start < stop and text[start].isspace():
            start += 1
        while start < stop and text[stop - 1].isspace():
            stop -= 1
        normalized_offsets.append((start, stop))

    positions: list[int] = []
    for index, span in enumerate(normalized):
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in span)
        ):
            raise ValueError(f"spans[{index}] must contain two integers")
        start, stop = span
        if not 0 <= start < stop <= len(text):
            raise ValueError(f"spans[{index}] is outside the prompt")
        selected = text[start:stop]
        if re.search(r"\s", selected):
            raise ValueError(f"spans[{index}] crosses a word boundary")
        if start and text[start - 1].isalnum():
            raise ValueError(f"spans[{index}] starts inside a word boundary")
        if stop < len(text) and text[stop].isalnum():
            raise ValueError(f"spans[{index}] ends inside a word boundary")
        overlapping = [
            token_index
            for token_index, (token_start, token_stop) in enumerate(normalized_offsets)
            if token_stop > token_start and token_start < stop and start < token_stop
        ]
        if not overlapping:
            raise ValueError(f"spans[{index}] has no tokenizer overlap")
        first_start = normalized_offsets[overlapping[0]][0]
        final_stop = normalized_offsets[overlapping[-1]][1]
        if first_start != start or final_stop != stop:
            raise ValueError(f"spans[{index}] token boundary differs from the word boundary")
        positions.append(overlapping[-1])
    if len(set(positions)) != len(positions):
        raise ValueError("edited words resolve to duplicate final-token positions")
    return tuple(positions)


__all__ = [
    "DiagnosticPrompts",
    "PromptSide",
    "build_diagnostic_prompts",
    "word_final_token_positions",
]
