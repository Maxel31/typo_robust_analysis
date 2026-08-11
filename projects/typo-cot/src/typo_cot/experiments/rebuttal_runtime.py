"""Shared adapters for running rebuttal interventions from pair manifests."""

from __future__ import annotations

from collections.abc import Mapping


def require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    """Return an object-like manifest field or fail with its qualified name."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def manifest_token_positions(value: object, *, field: str) -> tuple[int, ...]:
    """Validate and normalize a non-empty list of unique token positions."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty integer list")
    positions = tuple(value)
    if any(
        not isinstance(position, int) or isinstance(position, bool) or position < 0
        for position in positions
    ):
        raise ValueError(f"{field} contains an invalid token position")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{field} contains duplicate token positions")
    return positions


def manifest_runtime_pair(record: Mapping[str, object]) -> dict[str, object]:
    """Adapt a normalized rebuttal pair to shared tokenizer runtime checks."""

    edits = record.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("rebuttal runtime record must contain aligned edits")
    aligned_words: list[dict[str, object]] = []
    for index, raw_edit in enumerate(edits):
        edit = require_mapping(raw_edit, field=f"edits[{index}]")
        clean_span = edit.get("clean_char_span")
        typo_span = edit.get("typo_char_span")
        if (
            not isinstance(clean_span, list)
            or len(clean_span) != 2
            or not isinstance(typo_span, list)
            or len(typo_span) != 2
        ):
            raise ValueError("runtime edit character spans must contain two integers")
        aligned_words.append(
            {
                "clean_prompt_span": {"start": clean_span[0], "end": clean_span[1]},
                "edited_prompt_span": {"start": typo_span[0], "end": typo_span[1]},
                "clean_final_token": edit.get("clean_word_final_token"),
                "edited_final_token": edit.get("typo_word_final_token"),
            }
        )
    return {
        "sample_id": record.get("pair_id"),
        "gold_answer": record.get("gold_answer"),
        "targeting": record.get("target_rule"),
        "clean": {
            "prompt": record.get("clean_text"),
            "prompt_token_count": record.get("clean_prompt_token_count"),
        },
        "edited": {
            "prompt": record.get("typo_text"),
            "prompt_token_count": record.get("typo_prompt_token_count"),
        },
        "aligned_words": aligned_words,
    }


__all__ = ["manifest_runtime_pair", "manifest_token_positions", "require_mapping"]
