"""Render and tokenize clean/typo training pairs with exact causal masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from typo_robust_training.training.pairs import (
    TrainingPair,
    align_unchanged_token_positions,
    edited_word_final_token_positions,
)


_QUESTION_PREFIX = "Question:\n"
_ANSWER_PROMPT_SUFFIX = "\nAnswer:"


@dataclass(frozen=True, slots=True)
class RenderedTrainingPair:
    clean_text: str
    typo_text: str
    clean_edit_spans: tuple[tuple[int, int], ...]
    typo_edit_spans: tuple[tuple[int, int], ...]
    clean_answer_span: tuple[int, int] | None
    typo_answer_span: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class PairedEncoding:
    record_id: str
    clean_input_ids: tuple[int, ...]
    typo_input_ids: tuple[int, ...]
    clean_attention_mask: tuple[int, ...]
    typo_attention_mask: tuple[int, ...]
    output_logit_pairs: tuple[tuple[int, int], ...]
    global_state_token_pairs: tuple[tuple[int, int], ...]
    clean_edit_positions: tuple[int, ...]
    typo_edit_positions: tuple[int, ...]
    answer_targets: tuple[tuple[int, int], ...]
    student_tokens: int
    is_noop: bool


def render_training_pair(pair: TrainingPair) -> RenderedTrainingPair:
    """Use a short shared task wrapper while preserving raw edit coordinates."""

    if not isinstance(pair, TrainingPair):
        raise TypeError("training rendering requires TrainingPair")
    if pair.task is None:
        if pair.answer is not None:
            raise ValueError("general-text training pair cannot contain an answer")
        clean_text, typo_text = pair.clean_text, pair.typo_text
        clean_shift = typo_shift = 0
        clean_answer_span = typo_answer_span = None
    else:
        if pair.answer is None:
            raise ValueError("reasoning training pair requires an answer")
        prefix, prompt_suffix = _QUESTION_PREFIX, _ANSWER_PROMPT_SUFFIX
        answer_suffix = f" {pair.answer}\n"
        clean_prompt = prefix + pair.clean_text + prompt_suffix
        typo_prompt = prefix + pair.typo_text + prompt_suffix
        clean_text = clean_prompt + answer_suffix
        typo_text = typo_prompt + answer_suffix
        clean_shift = typo_shift = len(prefix)
        clean_answer_span = len(clean_prompt), len(clean_prompt) + 1 + len(pair.answer)
        typo_answer_span = len(typo_prompt), len(typo_prompt) + 1 + len(pair.answer)
    clean_spans = tuple(
        (edit.clean_char_span[0] + clean_shift, edit.clean_char_span[1] + clean_shift)
        for edit in pair.edits
    )
    typo_spans = tuple(
        (edit.typo_char_span[0] + typo_shift, edit.typo_char_span[1] + typo_shift)
        for edit in pair.edits
    )
    return RenderedTrainingPair(
        clean_text=clean_text,
        typo_text=typo_text,
        clean_edit_spans=clean_spans,
        typo_edit_spans=typo_spans,
        clean_answer_span=clean_answer_span,
        typo_answer_span=typo_answer_span,
    )


def _flat(value: object, *, field: str) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError(f"tokenizer {field} must be one sequence")
    return value


def _tokenize(
    tokenizer: Any, *, text: str, max_length: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping):
        raise ValueError("training tokenizer must return a mapping")
    ids_raw = _flat(encoded.get("input_ids"), field="input_ids")
    mask_raw = _flat(encoded.get("attention_mask"), field="attention_mask")
    offsets_raw = _flat(encoded.get("offset_mapping"), field="offset_mapping")
    if not ids_raw or len(ids_raw) != len(mask_raw) or len(ids_raw) != len(offsets_raw):
        raise ValueError("training tokenizer returned inconsistent sequence fields")
    if len(ids_raw) > max_length:
        raise ValueError("training tokenizer ignored max_length")
    ids: list[int] = []
    mask: list[int] = []
    offsets: list[tuple[int, int]] = []
    for index, (token, attended, offset) in enumerate(
        zip(ids_raw, mask_raw, offsets_raw, strict=True)
    ):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"tokenizer input_ids[{index}] is invalid")
        if attended not in {0, 1}:
            raise ValueError(f"tokenizer attention_mask[{index}] is invalid")
        if (
            not isinstance(offset, (tuple, list))
            or len(offset) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offset)
        ):
            raise ValueError(f"tokenizer offset_mapping[{index}] is invalid")
        ids.append(token)
        mask.append(int(attended))
        offsets.append((offset[0], offset[1]))
    if sum(mask) < 2:
        raise ValueError("training sequence must contain at least two attended tokens")
    return tuple(ids), tuple(mask), tuple(offsets)


def _answer_targets(
    *,
    ids: tuple[int, ...],
    offsets: tuple[tuple[int, int], ...],
    span: tuple[int, int] | None,
    required: bool,
) -> tuple[tuple[int, int], ...]:
    if span is None:
        return ()
    start, stop = span
    if stop > max((token_stop for _token_start, token_stop in offsets), default=0):
        if required:
            raise ValueError("reasoning answer suffix was truncated from the training sequence")
        return ()
    targets = tuple(
        (index - 1, ids[index])
        for index, (token_start, token_stop) in enumerate(offsets)
        if index > 0 and token_stop > token_start and start <= token_start and token_stop <= stop
    )
    if not targets:
        raise ValueError("reasoning answer suffix was truncated from the training sequence")
    return targets


def encode_training_pair(
    pair: TrainingPair,
    *,
    tokenizer: Any,
    max_length: int,
    require_answer_targets: bool = False,
    require_all_edits_visible: bool = False,
) -> PairedEncoding:
    """Tokenize one pair and derive all masks without heuristic token alignment."""

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("training max_length must be at least two")
    rendered = render_training_pair(pair)
    clean_ids, clean_mask, clean_offsets = _tokenize(
        tokenizer, text=rendered.clean_text, max_length=max_length
    )
    typo_ids, typo_mask, typo_offsets = _tokenize(
        tokenizer, text=rendered.typo_text, max_length=max_length
    )
    aligned = align_unchanged_token_positions(
        clean_text=rendered.clean_text,
        typo_text=rendered.typo_text,
        clean_edit_spans=rendered.clean_edit_spans,
        typo_edit_spans=rendered.typo_edit_spans,
        clean_offsets=clean_offsets,
        typo_offsets=typo_offsets,
    )
    aligned = tuple(
        (clean_index, typo_index)
        for clean_index, typo_index in aligned
        if clean_mask[clean_index]
        and typo_mask[typo_index]
        and clean_ids[clean_index] == typo_ids[typo_index]
    )
    output_pairs = tuple(
        (clean_index - 1, typo_index - 1)
        for clean_index, typo_index in aligned
        if clean_index > 0 and typo_index > 0
    )
    if not output_pairs:
        raise ValueError("training pair has no aligned non-edited next-token targets")
    if pair.edits:
        clean_extent = max((stop for _start, stop in clean_offsets), default=0)
        typo_extent = max((stop for _start, stop in typo_offsets), default=0)
        visible_spans = tuple(
            (clean_span, typo_span)
            for clean_span, typo_span in zip(
                rendered.clean_edit_spans,
                rendered.typo_edit_spans,
                strict=True,
            )
            if clean_span[1] <= clean_extent and typo_span[1] <= typo_extent
        )
        if require_all_edits_visible and len(visible_spans) != len(pair.edits):
            raise ValueError("training typo edit falls outside the retained token window")
        clean_edit_positions = edited_word_final_token_positions(
            clean_offsets,
            tuple(clean_span for clean_span, _typo_span in visible_spans),
            text=rendered.clean_text,
        )
        typo_edit_positions = edited_word_final_token_positions(
            typo_offsets,
            tuple(typo_span for _clean_span, typo_span in visible_spans),
            text=rendered.typo_text,
        )
    else:
        clean_edit_positions = typo_edit_positions = ()
    if len(clean_edit_positions) != len(typo_edit_positions):
        raise ValueError("training edited-word token cardinalities differ")
    return PairedEncoding(
        record_id=pair.record_id,
        clean_input_ids=clean_ids,
        typo_input_ids=typo_ids,
        clean_attention_mask=clean_mask,
        typo_attention_mask=typo_mask,
        output_logit_pairs=output_pairs,
        global_state_token_pairs=aligned,
        clean_edit_positions=clean_edit_positions,
        typo_edit_positions=typo_edit_positions,
        answer_targets=_answer_targets(
            ids=typo_ids,
            offsets=typo_offsets,
            span=rendered.typo_answer_span,
            required=require_answer_targets,
        ),
        student_tokens=sum(typo_mask),
        is_noop=pair.is_noop,
    )


def retained_clean_character_extent(
    pair: TrainingPair,
    *,
    tokenizer: Any,
    max_length: int,
) -> int:
    """Return the raw clean-text prefix fully retained after token truncation."""

    rendered = render_training_pair(pair)
    _ids, _mask, offsets = _tokenize(
        tokenizer,
        text=rendered.clean_text,
        max_length=max_length,
    )
    retained_stop = max((stop for _start, stop in offsets), default=0)
    shift = len(_QUESTION_PREFIX) if pair.task is not None else 0
    return max(0, min(len(pair.clean_text), retained_stop - shift))


__all__ = [
    "PairedEncoding",
    "RenderedTrainingPair",
    "encode_training_pair",
    "retained_clean_character_extent",
    "render_training_pair",
]
