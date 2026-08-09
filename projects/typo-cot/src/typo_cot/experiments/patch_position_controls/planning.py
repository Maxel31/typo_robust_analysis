"""Deterministic token coordinates for patch-position controls.

The source pair records store character spans and the edited-word token
alignment produced during pair preparation.  This module resolves the three
paper positions from those records and fresh tokenizer offsets, before any GPU
intervention is run.  In particular, ``question-final`` is the last token
overlapping each side's recorded editable span; it is never inferred by
searching prompt text or by copying an index from the clean prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


POSITION_NAMES = ("edited-word", "prompt-final", "question-final")


def _positions(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of token indices") from exc
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in normalized):
        raise ValueError(f"{field} must contain only integer token indices")
    if any(value < 0 for value in normalized):
        raise ValueError(f"{field} must contain only non-negative token indices")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicate token indices")
    return normalized


@dataclass(frozen=True, slots=True)
class PositionCoordinates:
    """Clean donor and edited recipient coordinates for one control arm."""

    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]
    locator: str

    def __post_init__(self) -> None:
        source = _positions(self.source_positions, field="source_positions")
        destination = _positions(
            self.destination_positions,
            field="destination_positions",
        )
        if len(source) != len(destination):
            raise ValueError(
                "source_positions and destination_positions must have equal cardinality"
            )
        if not isinstance(self.locator, str) or not self.locator.strip():
            raise ValueError("locator must be a non-empty string")
        object.__setattr__(self, "source_positions", source)
        object.__setattr__(self, "destination_positions", destination)


@dataclass(frozen=True, slots=True)
class _Side:
    prompt: str
    offsets: tuple[tuple[int, int], ...]
    editable_span: tuple[int, int]


def _span(value: object, *, field: str, prompt_length: int) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    start, end = value.get("start"), value.get("end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= prompt_length
    ):
        raise ValueError(f"{field} is not a valid prompt character span")
    return start, end


def _offsets(
    values: Sequence[Sequence[int]],
    *,
    field: str,
    prompt_length: int,
) -> tuple[tuple[int, int], ...]:
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of offset pairs") from exc
    if not materialized:
        raise ValueError(f"{field} must not be empty")

    normalized: list[tuple[int, int]] = []
    for index, raw_offset in enumerate(materialized):
        try:
            offset = tuple(raw_offset)
        except TypeError as exc:
            raise ValueError(f"{field}[{index}] must be an offset pair") from exc
        if len(offset) != 2:
            raise ValueError(f"{field}[{index}] must contain start and end")
        start, end = offset
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start <= end <= prompt_length
        ):
            raise ValueError(f"{field}[{index}] is not a valid prompt offset")
        normalized.append((start, end))
    return tuple(normalized)


def _side(
    pair: Mapping[str, object],
    *,
    name: str,
    raw_offsets: Sequence[Sequence[int]],
) -> _Side:
    payload = pair.get(name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"pair.{name} must be an object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{name}.prompt must be non-empty")
    token_count = payload.get("prompt_token_count")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError(f"{name}.prompt_token_count must be a positive integer")
    offsets = _offsets(
        raw_offsets,
        field=f"{name}_offsets",
        prompt_length=len(prompt),
    )
    if len(offsets) != token_count:
        raise ValueError(
            f"{name}_offsets length differs from {name}.prompt_token_count: "
            f"{len(offsets)} != {token_count}"
        )
    editable_span = _span(
        payload.get("editable_prompt_span"),
        field=f"{name}.editable_prompt_span",
        prompt_length=len(prompt),
    )
    return _Side(prompt=prompt, offsets=offsets, editable_span=editable_span)


def _overlapping_tokens(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
) -> tuple[int, ...]:
    span_start, span_end = span
    return tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < span_end and span_start < end
    )


def _aligned_final_positions(
    pair: Mapping[str, object],
    *,
    side_name: str,
    side: _Side,
) -> tuple[int, ...]:
    aligned_words = pair.get("aligned_words")
    if not isinstance(aligned_words, list) or not aligned_words:
        raise ValueError("aligned_words must be a non-empty list")
    recorded_count = pair.get("num_aligned_words")
    if recorded_count is not None and (
        isinstance(recorded_count, bool)
        or not isinstance(recorded_count, int)
        or recorded_count != len(aligned_words)
    ):
        raise ValueError("num_aligned_words differs from aligned_words length")

    final_positions: list[int] = []
    for index, raw_word in enumerate(aligned_words):
        field = f"aligned_words[{index}]"
        if not isinstance(raw_word, Mapping):
            raise ValueError(f"{field} must be an object")
        token_field = f"{side_name}_token_indices"
        final_field = f"{side_name}_final_token"
        span_field = f"{side_name}_prompt_span"
        token_indices = _positions(
            raw_word.get(token_field),  # type: ignore[arg-type]
            field=f"{field}.{token_field}",
        )
        final_token = raw_word.get(final_field)
        if isinstance(final_token, bool) or not isinstance(final_token, int):
            raise ValueError(f"{field}.{final_field} must be an integer")
        if final_token != token_indices[-1]:
            raise ValueError(f"{field}.{final_field} differs from {token_field}")
        if final_token >= len(side.offsets):
            raise ValueError(f"{field}.{final_field} is outside the prompt token sequence")

        prompt_span = _span(
            raw_word.get(span_field),
            field=f"{field}.{span_field}",
            prompt_length=len(side.prompt),
        )
        overlapping = _overlapping_tokens(side.offsets, prompt_span)
        if not overlapping:
            raise ValueError(f"{field}.{span_field} has no overlapping prompt token")
        if token_indices != overlapping:
            raise ValueError(
                f"{field}.{token_field} does not match tokenizer offsets for {span_field}"
            )
        if final_token != overlapping[-1]:
            raise ValueError(
                f"{field}.{final_field} does not match tokenizer offsets for {span_field}"
            )
        final_positions.append(final_token)

    return _positions(final_positions, field=f"{side_name} aligned final-token coordinates")


def _question_final(side: _Side, *, side_name: str) -> int:
    overlapping = _overlapping_tokens(side.offsets, side.editable_span)
    if not overlapping:
        raise ValueError(f"{side_name}.editable_prompt_span has no overlapping prompt token")
    return overlapping[-1]


def locate_position_coordinates(
    pair: Mapping[str, object],
    clean_offsets: Sequence[Sequence[int]],
    edited_offsets: Sequence[Sequence[int]],
) -> dict[str, PositionCoordinates]:
    """Resolve all clean-to-edited position arms from one validated pair.

    ``clean_offsets`` and ``edited_offsets`` must come from the tokenizer used
    by the model runtime.  The stored alignment is cross-checked against those
    offsets so stale pair records or tokenizer revisions fail before patching.
    """

    if not isinstance(pair, Mapping):
        raise ValueError("pair must be an object")
    clean = _side(pair, name="clean", raw_offsets=clean_offsets)
    edited = _side(pair, name="edited", raw_offsets=edited_offsets)
    clean_word_positions = _aligned_final_positions(
        pair,
        side_name="clean",
        side=clean,
    )
    edited_word_positions = _aligned_final_positions(
        pair,
        side_name="edited",
        side=edited,
    )
    if len(clean_word_positions) != len(edited_word_positions):
        raise ValueError("clean and edited aligned-word coordinate cardinalities differ")

    return {
        "edited-word": PositionCoordinates(
            clean_word_positions,
            edited_word_positions,
            "aligned-edited-word-final-tokens/v1",
        ),
        "prompt-final": PositionCoordinates(
            (len(clean.offsets) - 1,),
            (len(edited.offsets) - 1,),
            "prompt-final-token/v1",
        ),
        "question-final": PositionCoordinates(
            (_question_final(clean, side_name="clean"),),
            (_question_final(edited, side_name="edited"),),
            "editable-prompt-span-final-overlap/v1",
        ),
    }


__all__ = ["POSITION_NAMES", "PositionCoordinates", "locate_position_coordinates"]
