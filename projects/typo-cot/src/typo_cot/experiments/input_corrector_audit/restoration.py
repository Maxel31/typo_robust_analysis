"""Submitted whitespace-word restoration metrics for the input-corrector audit."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

_LETTERS = "ABCDEFGHIJ"


def build_reference(original_question: str, choices: list[str] | None) -> str:
    """Render the clean editable text in the submitted multiple-choice format."""
    if choices:
        options = " ".join(f"({_LETTERS[index]}) {choice}" for index, choice in enumerate(choices))
        return f"{original_question}\n{options}"
    return original_question


def aligned_word_changes(reference: str, hypothesis: str) -> list[tuple[int, str, str]]:
    """Return equal-length replacement words after whitespace tokenization."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    changes: list[tuple[int, str, str]] = []
    matcher = difflib.SequenceMatcher(
        a=reference_words,
        b=hypothesis_words,
        autojunk=False,
    )
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag != "replace" or ref_end - ref_start != hyp_end - hyp_start:
            continue
        for offset in range(ref_end - ref_start):
            changes.append(
                (
                    hyp_start + offset,
                    reference_words[ref_start + offset],
                    hypothesis_words[hyp_start + offset],
                )
            )
    return changes


def diff_word_positions(
    reference: str,
    edited: str,
) -> list[tuple[int, str | None, str]]:
    """Return submitted clean-to-edited word positions and unalignable words."""
    reference_words = reference.split()
    edited_words = edited.split()
    changes: list[tuple[int, str | None, str]] = []
    matcher = difflib.SequenceMatcher(
        a=reference_words,
        b=edited_words,
        autojunk=False,
    )
    for tag, ref_start, ref_end, edit_start, edit_end in matcher.get_opcodes():
        if tag == "replace" and ref_end - ref_start == edit_end - edit_start:
            for offset in range(ref_end - ref_start):
                changes.append(
                    (
                        edit_start + offset,
                        reference_words[ref_start + offset],
                        edited_words[edit_start + offset],
                    )
                )
        elif tag != "equal":
            changes.extend(
                (index, None, edited_words[index]) for index in range(edit_start, edit_end)
            )
    return changes


def is_whitespace_normalized_equal(left: str, right: str) -> bool:
    """Compare text with the historical whitespace-normalized predicate."""
    return " ".join(left.split()) == " ".join(right.split())


def is_byte_exact_equal(left: str, right: str) -> bool:
    """Compare the exact UTF-8 bytes used by the Table 12 prompt endpoint."""
    return left.encode("utf-8") == right.encode("utf-8")


@dataclass(slots=True)
class RestorationResult:
    """One item's submitted word-restoration and collateral classification."""

    n_perturbed_words: int
    n_restored: int
    n_unalignable: int
    restored_flags: list[tuple[str, str, bool]]
    whitespace_normalized_full: bool
    all_perturbed_restored: bool
    n_collateral: int
    collateral: list[tuple[int, str, str]] = field(default_factory=list)


def classify_restoration(
    reference: str,
    edited: str,
    corrected: str,
) -> RestorationResult:
    """Classify restoration with the exact heuristic used by the submitted run."""
    changes = diff_word_positions(reference, edited)
    corrected_words = corrected.split()

    perturbed_words = 0
    restored_words = 0
    unalignable = 0
    restored_flags: list[tuple[str, str, bool]] = []
    for edited_index, clean_word, edited_word in changes:
        if clean_word is None:
            unalignable += 1
            continue
        perturbed_words += 1
        restored = (
            edited_index < len(corrected_words) and corrected_words[edited_index] == clean_word
        )
        restored_words += int(restored)
        restored_flags.append((clean_word, edited_word, restored))

    perturbed_positions = {index for index, _, _ in changes}
    collateral = [
        change
        for change in aligned_word_changes(reference, corrected)
        if change[0] not in perturbed_positions
    ]
    return RestorationResult(
        n_perturbed_words=perturbed_words,
        n_restored=restored_words,
        n_unalignable=unalignable,
        restored_flags=restored_flags,
        whitespace_normalized_full=is_whitespace_normalized_equal(reference, corrected),
        all_perturbed_restored=(perturbed_words > 0 and restored_words == perturbed_words),
        n_collateral=len(collateral),
        collateral=collateral,
    )
