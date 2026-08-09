"""Pure reconstruction contracts for the Appendix E restoration-order audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

import typo_cot.experiments.restoration_order_accuracy.protocol as protocol_module
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    PAPER_SOURCE_RECORD_COUNTS,
    PROTOCOL,
    EditGroupingError,
    build_edit_groups,
    restore_edit_groups,
)


def _attempt(token_index: int, relevance: float) -> dict[str, object]:
    return {
        "selection_rank": token_index,
        "target_token_index": token_index,
        "relevance": relevance,
    }


def test_build_edit_groups_uses_sequence_matcher_without_autojunk_and_merges_adjacent_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Matcher:
        def __init__(self, *, a: str, b: str, autojunk: bool) -> None:
            calls.append({"a": a, "b": b, "autojunk": autojunk})

        def get_opcodes(self) -> Sequence[tuple[str, int, int, int, int]]:
            # A delete immediately followed by an insert is one edit event, not two.
            return (
                ("equal", 0, 1, 0, 1),
                ("delete", 1, 2, 1, 1),
                ("insert", 2, 2, 1, 3),
                ("equal", 2, 4, 3, 5),
            )

    monkeypatch.setattr(protocol_module.difflib, "SequenceMatcher", Matcher)

    groups = build_edit_groups("abcd", "aXYcd", [_attempt(7, -3.5)])

    assert calls == [{"a": "abcd", "b": "aXYcd", "autojunk": False}]
    assert len(groups) == 1
    group = groups[0]
    assert group.index == 0
    assert group.clean_text == "b"
    assert group.edited_text == "XY"
    assert group.clean_start == 1
    assert group.clean_end == 2
    assert group.edited_start == 1
    assert group.edited_end == 3
    assert group.target_token_index == 7
    assert group.relevance == -3.5


def test_edit_groups_are_bound_one_to_one_to_attempts_in_left_to_right_token_order() -> None:
    clean = "aa bb cc dd ee"
    edited = "ax bx cx dx ee"
    attempts = (
        _attempt(30, -3.0),
        _attempt(10, 1.0),
        _attempt(40, 4.0),
        _attempt(20, -2.0),
    )

    groups = build_edit_groups(clean, edited, attempts)

    assert [group.index for group in groups] == [0, 1, 2, 3]
    assert [group.target_token_index for group in groups] == [10, 20, 30, 40]
    assert [group.relevance for group in groups] == [1.0, -2.0, -3.0, 4.0]
    assert [(group.clean_text, group.edited_text) for group in groups] == [
        ("a", "x"),
        ("b", "x"),
        ("c", "x"),
        ("d", "x"),
    ]


@pytest.mark.parametrize(
    "attempts",
    [
        [_attempt(10, 1.0)],
        [_attempt(10, 1.0), _attempt(10, 2.0)],
        [_attempt(10, 1.0), _attempt(20, 2.0), _attempt(30, 3.0)],
    ],
)
def test_edit_grouping_rejects_non_bijective_target_attempts(
    attempts: Sequence[Mapping[str, object]],
) -> None:
    with pytest.raises(EditGroupingError, match="one-to-one|target_attempt|edit group|unique"):
        build_edit_groups("aa bb", "ax bx", attempts)


def test_restore_edit_groups_preserves_exact_endpoints_and_partial_bytes() -> None:
    clean = "the quick brown fox jumps"
    edited = "the quikk brywn fox jumpz"
    groups = build_edit_groups(
        clean,
        edited,
        (_attempt(10, 9.0), _attempt(20, -2.0), _attempt(30, 1.0)),
    )

    assert restore_edit_groups(clean, edited, groups, ()) == edited
    assert restore_edit_groups(clean, edited, groups, range(len(groups))) == clean
    assert restore_edit_groups(clean, edited, groups, (0, 2)) == (
        "the quick brywn fox jumps"
    )


def test_restore_edit_groups_rejects_an_unknown_or_duplicate_group_index() -> None:
    clean = "aa bb"
    edited = "ax bx"
    groups = build_edit_groups(clean, edited, (_attempt(1, 2.0), _attempt(2, 1.0)))

    with pytest.raises(ValueError, match="group|index|duplicate"):
        restore_edit_groups(clean, edited, groups, (0, 0))
    with pytest.raises(ValueError, match="group|index|range"):
        restore_edit_groups(clean, edited, groups, (2,))


def test_dataset_cohort_sizes_are_public_implementation_evidence_not_pdf_facts() -> None:
    assert "source_records" not in PROTOCOL["paper_defined"]
    assert PROTOCOL["public_reproduction"]["source_records"] == PAPER_SOURCE_RECORD_COUNTS
    assert PROTOCOL["public_reproduction"]["source_outcome_revalidation"] == (
        "stored-continuation-final-pdf-extraction-match/v1"
    )
