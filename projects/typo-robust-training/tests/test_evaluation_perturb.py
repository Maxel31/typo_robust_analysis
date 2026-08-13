"""Frozen evaluation typos preserve task semantics and replay byte-for-byte."""

from __future__ import annotations

import pytest

from typo_robust_training.data.records import CleanRecord
from typo_robust_training.evaluation.perturb import (
    NoNaturalInjectionTargetError,
    evaluation_eligible_word_spans,
    freeze_natural_injection_dictionary,
    generate_evaluation_typo,
    generate_natural_injection,
)


def _record(*, task: str = "mmlu", answer: str = "B") -> CleanRecord:
    return CleanRecord(
        source=task,
        source_revision="a" * 40,
        source_split="test",
        source_id=f"{task}:fixture-1",
        group_id=f"{task}:fixture-1",
        text=(
            "Which reliable airport serves the northern research district?\n"
            "A. Alpha terminal\n"
            "B. Northern airport\n"
            "C. Third terminal"
        ),
        task=task,
        answer=answer,
        metadata={"fixture": True},
    )


def test_multiple_choice_targets_only_question_and_forbids_gold_or_option_text() -> None:
    record = _record()
    spans = evaluation_eligible_word_spans(record, minimum_word_letters=3)

    words = tuple(record.text[start:stop] for start, stop in spans)
    assert words == (
        "Which",
        "reliable",
        "airport",
        "serves",
        "the",
        "northern",
        "research",
        "district",
    )
    assert all(stop <= record.text.index("\n") for _, stop in spans)


def test_math_and_identifier_spans_are_never_typo_targets() -> None:
    record = CleanRecord(
        source="math_500",
        source_revision="a" * 40,
        source_split="test",
        source_id="math:fixture",
        group_id="math:fixture",
        text=(
            "Carefully evaluate $triangle + perimeter$ and \\frac{numerator}{denominator} "
            "before describing result_value in ordinary prose."
        ),
        task="math_500",
        answer="42",
        metadata={},
    )

    spans = evaluation_eligible_word_spans(record, minimum_word_letters=3)
    words = {record.text[start:stop] for start, stop in spans}

    assert {"triangle", "perimeter", "frac", "numerator", "denominator"}.isdisjoint(words)
    assert "result" not in words
    assert {"Carefully", "evaluate", "before", "describing", "ordinary", "prose"} <= words


def test_fixed_evaluation_typo_has_exact_distinct_edits_and_is_replayable() -> None:
    record = _record()
    first = generate_evaluation_typo(
        record,
        condition="random-2",
        edit_count=2,
        operations=("keyboard-neighbor-substitution", "deletion", "duplication"),
        seed=42,
        role="final_test",
        variant=17,
    )
    replay = generate_evaluation_typo(
        record,
        condition="random-2",
        edit_count=2,
        operations=("keyboard-neighbor-substitution", "deletion", "duplication"),
        seed=42,
        role="final_test",
        variant=17,
    )

    assert replay == first
    assert first.typo_text != record.text
    assert len(first.edits) == 2
    assert len({edit.clean_char_span for edit in first.edits}) == 2
    assert first.typo_text[first.typo_text.index("\n") :] == record.text[record.text.index("\n") :]
    assert first.record_id != record.record_id
    assert first.metadata["evaluation_condition"] == "random-2"
    assert first.metadata["base_record_id"] == record.record_id


def test_severity_and_transposition_have_frozen_exact_edit_counts() -> None:
    record = CleanRecord(
        source="gsm8k",
        source_revision="a" * 40,
        source_split="test",
        source_id="gsm8k:fixture-2",
        group_id="gsm8k:fixture-2",
        text=(
            "Reliable gardeners carefully measure several rectangular flowerbeds before "
            "calculating the combined perimeter for tomorrow morning."
        ),
        task="gsm8k",
        answer="42",
        metadata={},
    )

    severity = generate_evaluation_typo(
        record,
        condition="random-4",
        edit_count=4,
        operations=("keyboard-neighbor-substitution", "deletion", "duplication"),
        seed=42,
        role="final_test",
        variant=3,
    )
    held_out = generate_evaluation_typo(
        record,
        condition="transposition-2",
        edit_count=2,
        operations=("adjacent-transposition",),
        seed=42,
        role="final_test",
        variant=3,
    )

    assert len(severity.edits) == 4
    assert len(held_out.edits) == 2
    assert {edit.operation for edit in held_out.edits} == {"adjacent-transposition"}


def test_zero_edit_noop_is_byte_identical() -> None:
    record = _record()
    no_op = generate_evaluation_typo(
        record,
        condition="no-op",
        edit_count=0,
        operations=(),
        seed=42,
        role="monitor",
        variant=0,
    )

    assert no_op.typo_text == record.text
    assert no_op.edits == ()


def test_frozen_record_identity_includes_seed_variant_and_edit_count() -> None:
    record = _record()
    common = {
        "record": record,
        "condition": "identity-fixture",
        "operations": ("deletion",),
        "role": "tune",
    }
    one = generate_evaluation_typo(**common, edit_count=1, seed=42, variant=0)
    two = generate_evaluation_typo(**common, edit_count=2, seed=42, variant=0)
    variant = generate_evaluation_typo(**common, edit_count=1, seed=42, variant=1)
    seed = generate_evaluation_typo(**common, edit_count=1, seed=43, variant=0)

    assert len({one.record_id, two.record_id, variant.record_id, seed.record_id}) == 4


def test_mixed_operation_inventory_never_applies_identity_transposition() -> None:
    record = CleanRecord(
        source="mmlu",
        source_revision="a" * 40,
        source_split="test",
        source_id="mmlu:repeated-letter-fixture",
        group_id="mmlu:repeated-letter-fixture",
        text="aaa lll eee nnn\nA. option one\nB. option two",
        task="mmlu",
        answer="B",
        metadata={"fixture": True},
    )
    typo = generate_evaluation_typo(
        record,
        condition="mixed-operation-fixture",
        edit_count=4,
        operations=("adjacent-transposition", "deletion"),
        seed=42,
        role="tune",
        variant=0,
    )

    assert len(typo.edits) == 4
    assert {edit.operation for edit in typo.edits} == {"deletion"}


def test_natural_injection_uses_eval_dictionary_and_keeps_options_unchanged() -> None:
    record = _record()
    injected = generate_natural_injection(
        record,
        replacements={"airport": ("arport",), "reliable": ("relyable",)},
        seed=42,
        role="final_test",
        variant=1,
    )

    assert len(injected.edits) == 1
    assert injected.edits[0].operation == "natural-dictionary-substitution"
    assert injected.edits[0].clean_word.casefold() in {"airport", "reliable"}
    assert (
        injected.typo_text[injected.typo_text.index("\n") :]
        == record.text[record.text.index("\n") :]
    )
    assert injected.metadata["evaluation_condition"] == "natural-injection"


def test_natural_injection_dictionary_is_validated_once_and_no_target_is_distinct() -> None:
    dictionary = freeze_natural_injection_dictionary(
        {"airport": ("arport", "arport")},
        minimum_word_letters=3,
    )
    injected = generate_natural_injection(
        _record(),
        replacements=dictionary,
        seed=42,
        role="final_test",
        variant=0,
    )
    assert injected.edits[0].clean_word.casefold() == "airport"

    with pytest.raises(ValueError, match="invalid entry"):
        freeze_natural_injection_dictionary({"airport": ("airport",)})
    with pytest.raises(NoNaturalInjectionTargetError, match="no held-out"):
        generate_natural_injection(
            _record(),
            replacements=freeze_natural_injection_dictionary({"unseenword": ("unseenwrd",)}),
            seed=42,
            role="final_test",
            variant=0,
        )
