"""Paper prompt construction preserves every edited-word character coordinate."""

from __future__ import annotations

import re

from typo_robust_training.localization.prompting import (
    build_diagnostic_prompts,
    word_final_token_positions,
)


class _WordTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {
            "input_ids": list(range(len(spans))),
            "attention_mask": [1] * len(spans),
            "offset_mapping": spans,
        }


def test_prompt_builder_shifts_raw_spans_into_same_paper_template() -> None:
    clean = "The airport is in Chicago."
    typo = "The arport is in Chicago."
    record = {
        "record_id": "fixture",
        "task": "gsm8k",
        "clean_text": clean,
        "typo_text": typo,
        "answer": "2",
        "metadata": {},
        "edits": [
            {
                "operation": "deletion",
                "clean_word": "airport",
                "typo_word": "arport",
                "clean_char_span": [4, 11],
                "typo_char_span": [4, 10],
            }
        ],
    }
    prompts = build_diagnostic_prompts(record)

    assert prompts.clean.text[prompts.clean.spans[0][0] : prompts.clean.spans[0][1]] == "airport"
    assert prompts.typo.text[prompts.typo.spans[0][0] : prompts.typo.spans[0][1]] == "arport"
    clean_prefix = prompts.clean.text[: prompts.clean.spans[0][0]]
    typo_prefix = prompts.typo.text[: prompts.typo.spans[0][0]]
    assert clean_prefix == typo_prefix
    assert prompts.clean.text.endswith("\n\nSolution:")
    assert prompts.typo.text.endswith("\n\nSolution:")


def test_word_final_coordinates_support_multiple_subwords_and_reject_ambiguity() -> None:
    tokenizer = _WordTokenizer()
    text = "alpha airport omega"
    assert word_final_token_positions(tokenizer, text=text, spans=((6, 13),)) == (1,)

    # A span that extends into the following word is not an exact word/token alignment.
    try:
        word_final_token_positions(tokenizer, text=text, spans=((6, 15),))
    except ValueError as exc:
        assert "boundary" in str(exc)
    else:  # pragma: no cover - explicit failure gives a clearer contract message
        raise AssertionError("ambiguous span was accepted")
