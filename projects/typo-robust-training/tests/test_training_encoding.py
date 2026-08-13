"""Rendered clean/typo sequences retain exact edit and answer coordinates."""

from __future__ import annotations

import re

import pytest

from typo_robust_training.data.records import TypoEdit
from typo_robust_training.training.encoding import encode_training_pair, render_training_pair
from typo_robust_training.training.pairs import TrainingPair


class _WordTokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {"<bos>": 1}

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        pieces = [
            (match.group(), match.span()) for match in re.finditer(r"[A-Za-z]+|\d+|[^\w\s]", text)
        ]
        ids = [1]
        offsets: list[tuple[int, int]] = [(0, 0)]
        for piece, span in pieces:
            ids.append(self.vocabulary.setdefault(piece, len(self.vocabulary) + 1))
            offsets.append(span)
        if kwargs.get("truncation"):
            maximum = int(kwargs["max_length"])
            ids = ids[:maximum]
            offsets = offsets[:maximum]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


class _MarkupTokenizer(_WordTokenizer):
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        pieces = [(match.group(), match.span()) for match in re.finditer(r"\S+", text)]
        ids = [1]
        offsets: list[tuple[int, int]] = [(0, 0)]
        for piece, span in pieces:
            ids.append(self.vocabulary.setdefault(piece, len(self.vocabulary) + 1))
            offsets.append(span)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


def _pair(*, task: str | None = None, answer: str | None = None) -> TrainingPair:
    return TrainingPair(
        record_id="a" * 64,
        clean_text="The airport works.",
        typo_text="The arport works.",
        task=task,
        answer=answer,
        metadata={},
        edits=(
            TypoEdit(
                operation="deletion",
                clean_word="airport",
                typo_word="arport",
                clean_char_span=(4, 11),
                typo_char_span=(4, 10),
            ),
        ),
        is_noop=False,
        epoch=0,
    )


def test_rendered_reasoning_pair_shifts_edits_and_marks_only_answer_suffix() -> None:
    rendered = render_training_pair(_pair(task="gsm8k", answer="72"))
    assert rendered.clean_text.startswith("Question:\nThe airport works.\nAnswer:")
    assert rendered.typo_text.startswith("Question:\nThe arport works.\nAnswer:")
    assert rendered.clean_text[slice(*rendered.clean_edit_spans[0])] == "airport"
    assert rendered.typo_text[slice(*rendered.typo_edit_spans[0])] == "arport"
    assert rendered.clean_text[slice(*rendered.clean_answer_span)] == " 72"
    assert rendered.typo_text[slice(*rendered.typo_answer_span)] == " 72"


def test_encoding_excludes_edited_targets_and_exposes_word_final_state_positions() -> None:
    encoding = encode_training_pair(
        _pair(task="gsm8k", answer="72"),
        tokenizer=_WordTokenizer(),
        max_length=512,
    )
    assert encoding.clean_edit_positions
    assert encoding.typo_edit_positions
    assert len(encoding.clean_edit_positions) == len(encoding.typo_edit_positions) == 1
    assert encoding.output_logit_pairs
    edited_clean = encoding.clean_edit_positions[0]
    edited_typo = encoding.typo_edit_positions[0]
    assert all(
        clean_logit + 1 != edited_clean and typo_logit + 1 != edited_typo
        for clean_logit, typo_logit in encoding.output_logit_pairs
    )
    assert encoding.answer_targets
    assert all(target_id >= 0 for _logit_position, target_id in encoding.answer_targets)
    assert encoding.student_tokens == len(encoding.typo_input_ids)


def test_noop_pair_has_zero_edit_positions_but_keeps_all_exact_alignment() -> None:
    pair = TrainingPair(
        record_id="b" * 64,
        clean_text="Clean text remains stable.",
        typo_text="Clean text remains stable.",
        task=None,
        answer=None,
        metadata={},
        edits=(),
        is_noop=True,
        epoch=2,
    )
    encoding = encode_training_pair(pair, tokenizer=_WordTokenizer(), max_length=512)
    assert encoding.clean_edit_positions == ()
    assert encoding.typo_edit_positions == ()
    assert encoding.output_logit_pairs == tuple(
        (index, index) for index in range(len(encoding.clean_input_ids) - 1)
    )
    assert encoding.answer_targets == ()


def test_encoding_ignores_only_edits_truncated_from_either_side() -> None:
    pair = TrainingPair(
        record_id="c" * 64,
        clean_text="airport works terminal",
        typo_text="arport works termnal",
        task=None,
        answer=None,
        metadata={},
        edits=(
            TypoEdit("deletion", "airport", "arport", (0, 7), (0, 6)),
            TypoEdit("deletion", "terminal", "termnal", (14, 22), (13, 20)),
        ),
        is_noop=False,
        epoch=0,
    )

    encoding = encode_training_pair(pair, tokenizer=_WordTokenizer(), max_length=3)

    assert len(encoding.clean_edit_positions) == len(encoding.typo_edit_positions) == 1
    assert encoding.clean_input_ids != encoding.typo_input_ids

    with pytest.raises(ValueError, match="outside the retained token window"):
        encode_training_pair(
            pair,
            tokenizer=_WordTokenizer(),
            max_length=3,
            require_all_edits_visible=True,
        )


def test_encoding_skips_answer_ce_when_the_answer_suffix_is_truncated() -> None:
    encoding = encode_training_pair(
        _pair(task="gsm8k", answer="72"),
        tokenizer=_WordTokenizer(),
        max_length=6,
    )

    assert encoding.clean_edit_positions
    assert encoding.typo_edit_positions
    assert encoding.answer_targets == ()

    with pytest.raises(ValueError, match="answer suffix was truncated"):
        encode_training_pair(
            _pair(task="gsm8k", answer="72"),
            tokenizer=_WordTokenizer(),
            max_length=6,
            require_answer_targets=True,
        )


def test_encoding_maps_an_edited_word_inside_a_punctuation_token() -> None:
    pair = TrainingPair(
        record_id="d" * 64,
        clean_text="Before </div> after",
        typo_text="Before </piv> after",
        task=None,
        answer=None,
        metadata={},
        edits=(
            TypoEdit(
                "natural-statistics-substitution",
                "div",
                "piv",
                (9, 12),
                (9, 12),
            ),
        ),
        is_noop=False,
        epoch=0,
    )

    encoding = encode_training_pair(pair, tokenizer=_MarkupTokenizer(), max_length=512)

    assert encoding.clean_edit_positions == (2,)
    assert encoding.typo_edit_positions == (2,)
    assert encoding.output_logit_pairs
