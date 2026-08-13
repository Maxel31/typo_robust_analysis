"""Deterministic corrected-word roles for natural-typo training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping

from typo_robust_training.data.records import NaturalTypoRecord, infer_single_word_typo_edit
from typo_robust_training.data.splits import stable_weighted_split


_WORD_SPLIT_NAMESPACE = "github-typo-corrected-word-split/v1"


def natural_dictionary_role_for_word(
    word: str,
    *,
    seed: int,
    weights: Mapping[str, float],
) -> str:
    """Assign one already-normalized corrected word to a deterministic role."""

    if not isinstance(word, str) or not word or not word.isascii() or not word.isalpha():
        raise ValueError("natural dictionary corrected word must be normalized ASCII letters")
    return stable_weighted_split(
        word.casefold(),
        seed=seed,
        namespace=_WORD_SPLIT_NAMESPACE,
        weights=weights,
    )


def natural_corrected_word(record: NaturalTypoRecord) -> str | None:
    """Return one normalized corrected word, or None for an unusable pair."""

    if not isinstance(record, NaturalTypoRecord):
        raise TypeError("natural corrected-word extraction requires NaturalTypoRecord")
    try:
        edit = infer_single_word_typo_edit(
            record.clean_text,
            record.typo_text,
            operation=record.operation,
        )
    except ValueError:
        return None
    word = edit.clean_word.casefold()
    return word if word.isascii() and word.isalpha() else None


def natural_dictionary_word_role(
    record: NaturalTypoRecord,
    *,
    seed: int,
    weights: Mapping[str, float],
) -> str | None:
    """Assign every corrected word atomically so exact words cannot leak across roles."""

    word = natural_corrected_word(record)
    if word is None:
        return None
    return natural_dictionary_role_for_word(
        word,
        seed=seed,
        weights=weights,
    )


__all__ = [
    "natural_corrected_word",
    "natural_dictionary_role_for_word",
    "natural_dictionary_word_role",
]
