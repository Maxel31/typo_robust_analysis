"""Contracts for submitted whitespace-token restoration classification."""

from __future__ import annotations

from typo_cot.experiments.input_corrector_audit.restoration import (
    RestorationResult,
    aligned_word_changes,
    build_reference,
    classify_restoration,
    diff_word_positions,
    is_byte_exact_equal,
    is_whitespace_normalized_equal,
)


def test_build_reference_matches_the_submitted_multiple_choice_rendering() -> None:
    assert build_reference("What is 2+2?", None) == "What is 2+2?"
    assert build_reference("Pick one.", ["cat", "dog", "fox"]) == (
        "Pick one.\n(A) cat (B) dog (C) fox"
    )


def test_diff_word_positions_aligns_equal_length_replacements_after_whitespace_split() -> None:
    assert diff_word_positions(
        "The quick brown fox",
        "The qick brwn fox",
    ) == [
        (1, "quick", "qick"),
        (2, "brown", "brwn"),
    ]


def test_diff_word_positions_marks_unequal_replace_words_unalignable() -> None:
    changes = diff_word_positions("The quick fox", "The qu ick fox")

    assert changes == [(1, None, "qu"), (2, None, "ick")]


def test_diff_word_positions_disables_difflib_autojunk() -> None:
    reference = " ".join(["x"] * 250 + ["quick"] + ["x"] * 250)
    perturbed = " ".join(["x"] * 250 + ["qick"] + ["x"] * 250)

    assert diff_word_positions(reference, perturbed) == [(250, "quick", "qick")]


def test_aligned_word_changes_reports_only_position_aligned_replacements() -> None:
    assert aligned_word_changes("a b c d", "a x c y") == [
        (1, "b", "x"),
        (3, "d", "y"),
    ]


def test_classify_restoration_counts_full_and_partial_word_restoration() -> None:
    reference = "The quick brown fox jumps"
    perturbed = "The qick brwn fox jumps"

    full = classify_restoration(reference, perturbed, reference)
    partial = classify_restoration(reference, perturbed, "The quick brwn fox jumps")

    assert isinstance(full, RestorationResult)
    assert full.n_perturbed_words == 2
    assert full.n_restored == 2
    assert full.n_unalignable == 0
    assert full.restored_flags == [
        ("quick", "qick", True),
        ("brown", "brwn", True),
    ]
    assert full.whitespace_normalized_full is True
    assert full.all_perturbed_restored is True
    assert full.n_collateral == 0

    assert partial.n_perturbed_words == 2
    assert partial.n_restored == 1
    assert partial.restored_flags == [
        ("quick", "qick", True),
        ("brown", "brwn", False),
    ]
    assert partial.whitespace_normalized_full is False
    assert partial.all_perturbed_restored is False


def test_classify_restoration_keeps_collateral_separate_from_target_restoration() -> None:
    reference = "The quick brown fox jumps"
    perturbed = "The qick brwn fox jumps"
    corrected = "The quick brown fax jumps"

    result = classify_restoration(reference, perturbed, corrected)

    assert result.n_restored == 2
    assert result.all_perturbed_restored is True
    assert result.whitespace_normalized_full is False
    assert result.n_collateral == 1
    assert result.collateral == [(3, "fox", "fax")]


def test_classify_restoration_counts_unalignable_perturbed_words() -> None:
    result = classify_restoration(
        "The quick brown fox",
        "The qu ick brown fox",
        "The quick brown fox",
    )

    assert result.n_perturbed_words == 0
    assert result.n_restored == 0
    assert result.n_unalignable == 2
    assert result.all_perturbed_restored is False


def test_whitespace_normalized_full_is_not_byte_exact_equality() -> None:
    reference = "The quick brown fox\n(A) yes (B) no"
    corrected = "The quick  brown fox\n(A) yes (B) no"
    result = classify_restoration(
        reference,
        "The qick brown fox\n(A) yes (B) no",
        corrected,
    )

    assert is_whitespace_normalized_equal(reference, corrected) is True
    assert result.whitespace_normalized_full is True
    assert is_byte_exact_equal(reference, corrected) is False
    assert is_byte_exact_equal(reference, reference) is True
