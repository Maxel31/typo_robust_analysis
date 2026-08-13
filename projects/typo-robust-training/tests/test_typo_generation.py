"""Deterministic synthetic and natural clean--typo pair contracts."""

from __future__ import annotations

import pytest

from typo_robust_training.data.natural_typos import parse_github_typo_commit
from typo_robust_training.data.perturb import (
    TRAINING_OPERATIONS,
    TypoGenerator,
    classify_character_edit,
    eligible_word_spans,
)
from typo_robust_training.data.records import CleanRecord


NATURAL_SUBSTITUTIONS = {
    character: {"z" if character != "z" else "x": 1} for character in "abcdefghijklmnopqrstuvwxyz"
}


def _record(source_id: str, text: str) -> CleanRecord:
    return CleanRecord(
        source="fixture",
        source_revision="b" * 40,
        source_split="train",
        source_id=source_id,
        group_id=source_id,
        text=text,
        task=None,
        answer=None,
        metadata={},
    )


def test_eligible_words_exclude_urls_emails_identifiers_and_short_words() -> None:
    text = (
        "A robust airport, ordinary words, user@example.com, https://example.com, "
        "snake_case and camelCase remain untouched."
    )
    words = [text[start:stop] for start, stop in eligible_word_spans(text)]
    assert "robust" in words
    assert "airport" in words
    assert "ordinary" in words
    assert "user" not in words
    assert "example" not in words
    assert "snake" not in words
    assert "camelCase" not in words
    assert "A" not in words


@pytest.mark.parametrize("operation", sorted(TRAINING_OPERATIONS))
def test_every_training_operation_produces_valid_replayable_spans(operation: str) -> None:
    record = _record("row-1", "The airport supports reliable international travel.")
    generator = TypoGenerator(seed=42, natural_substitutions=NATURAL_SUBSTITUTIONS)
    first = generator.generate(
        record,
        epoch=3,
        variant=1,
        force_operations=(operation,),
        force_edit_count=1,
    )
    second = generator.generate(
        record,
        epoch=3,
        variant=1,
        force_operations=(operation,),
        force_edit_count=1,
    )
    assert first == second
    assert first.clean_text == record.text
    assert first.typo_text != first.clean_text
    assert len(first.edits) == 1
    edit = first.edits[0]
    assert edit.operation == operation
    assert first.clean_text[slice(*edit.clean_char_span)] == edit.clean_word
    assert first.typo_text[slice(*edit.typo_char_span)] == edit.typo_word
    assert edit.clean_word != edit.typo_word


def test_counter_based_generation_is_independent_of_iteration_and_resume_order() -> None:
    generator = TypoGenerator(seed=44, natural_substitutions=NATURAL_SUBSTITUTIONS)
    records = tuple(
        _record(f"row-{index}", f"Educational passage number {index} contains useful context.")
        for index in range(12)
    )
    forward = {record.source_id: generator.generate(record, epoch=2) for record in records}
    resumed = {
        record.source_id: generator.generate(record, epoch=2)
        for record in (*records[7:], *records[:7])
    }
    assert resumed == forward
    assert generator.generate(records[0], epoch=3) != forward[records[0].source_id]


def test_adjacent_transposition_is_rejected_from_training_but_classified_for_evaluation() -> None:
    assert classify_character_edit(clean="airport", typo="ariport") == "adjacent-transposition"
    generator = TypoGenerator(seed=42, natural_substitutions=NATURAL_SUBSTITUTIONS)
    with pytest.raises(ValueError, match="held-out"):
        generator.generate(
            _record("row", "Airport traffic remains predictable."),
            epoch=0,
            force_operations=("adjacent-transposition",),
            force_edit_count=1,
        )


def test_natural_statistics_substitution_uses_only_the_supplied_character_table() -> None:
    generator = TypoGenerator(
        seed=42,
        operation_weights={"natural-statistics-substitution": 1.0},
        natural_substitutions={"a": {"q": 7, "x": 0}},
    )
    pair = generator.generate(
        _record("natural", "Aaaaa aaaaa aaaaa."),
        epoch=0,
        force_edit_count=1,
    )
    edit = pair.edits[0]
    differences = [
        (clean, typo)
        for clean, typo in zip(edit.clean_word.lower(), edit.typo_word.lower(), strict=True)
        if clean != typo
    ]
    assert differences == [("a", "q")]


def test_github_typo_corpus_orientation_repository_gate_and_operation_filter() -> None:
    repository = "https://github.com/example/permissive-project"
    payload = {
        "repo": repository,
        "commit": "0123456789abcdef",
        "message": "fix typo",
        "edits": [
            {
                "src": {
                    "text": "The arport is busy.",
                    "path": "README.md",
                    "lang": "eng",
                },
                "tgt": {
                    "text": "The airport is busy.",
                    "path": "README.md",
                    "lang": "eng",
                },
                "prob_typo": 0.99,
                "is_typo": True,
            },
            {
                "src": {"text": "The ariport is busy.", "path": "README.md", "lang": "eng"},
                "tgt": {"text": "The airport is busy.", "path": "README.md", "lang": "eng"},
                "prob_typo": 0.98,
                "is_typo": True,
            },
        ],
    }
    records = parse_github_typo_commit(
        payload,
        approved_repositories={repository: "MIT"},
    )
    assert len(records) == 2
    assert records[0].clean_text == "The airport is busy."
    assert records[0].typo_text == "The arport is busy."
    assert records[0].operation == "deletion"
    assert records[0].repository == repository
    assert records[0].repository_license == "MIT"
    assert records[1].operation == "adjacent-transposition"

    assert not parse_github_typo_commit(payload, approved_repositories={})
    assert all(
        record.operation != "adjacent-transposition"
        for record in records
        if record.training_eligible
    )
