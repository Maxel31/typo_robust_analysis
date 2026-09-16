"""Pure acceptance tests for rebuttal-v2 coordinates and sharding."""

from __future__ import annotations

import copy

import pytest
from typo_cot.experiments.rebuttal_v2.coordinates import (
    assign_shards,
    plan_offset,
)
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError


def _edit(
    clean_indices: list[int],
    typo_indices: list[int],
) -> dict[str, object]:
    return {
        "edit_indices": [99],
        "clean_token_indices": clean_indices,
        "typo_token_indices": typo_indices,
        "clean_endpoint": clean_indices[-1],
        "typo_endpoint": typo_indices[-1],
        "ignored_provenance": {"source": "input-audit"},
    }


def test_offset_uses_word_final_endpoints_but_excludes_all_subword_tokens() -> None:
    edits = [_edit([2, 3], [2]), _edit([7], [8, 9])]

    result = plan_offset(edits, clean_length=15, typo_length=15)

    assert result == {
        "eligibility": "valid",
        "source_positions": [5, 9],
        "write_positions": [4, 11],
        "reason_codes": [],
    }


def test_offset_retains_all_four_shifted_positions_when_one_is_invalid() -> None:
    edits = [
        _edit([1], [2]),
        _edit([4], [5]),
        _edit([7], [8]),
        _edit([12], [11]),
    ]

    result = plan_offset(edits, clean_length=15, typo_length=16)

    assert result["eligibility"] == "invalid"
    assert result["source_positions"] == [3, 6, 9, 14]
    assert result["write_positions"] == [4, 7, 10, 13]
    assert result["reason_codes"] == ["clean_offset_outside_prompt_interior"]


def test_offset_reports_boundaries_on_both_sides_without_dropping_coordinates() -> None:
    result = plan_offset([_edit([7], [8])], clean_length=10, typo_length=11)

    assert result == {
        "eligibility": "invalid",
        "source_positions": [9],
        "write_positions": [10],
        "reason_codes": [
            "clean_offset_outside_prompt_interior",
            "typo_offset_outside_prompt_interior",
        ],
    }


def test_offset_rejects_positions_inside_any_edited_word_subtoken() -> None:
    edits = [_edit([2], [3]), _edit([4, 5], [5, 6])]

    result = plan_offset(edits, clean_length=12, typo_length=13)

    assert result["source_positions"] == [4, 7]
    assert result["write_positions"] == [5, 8]
    assert result["eligibility"] == "invalid"
    assert result["reason_codes"] == [
        "clean_offset_overlaps_edited_token",
        "typo_offset_overlaps_edited_token",
    ]


def test_offset_rejects_repeated_writes_but_preserves_the_repetitions() -> None:
    result = plan_offset(
        [_edit([2], [3]), _edit([6], [3])],
        clean_length=12,
        typo_length=12,
    )

    assert result == {
        "eligibility": "invalid",
        "source_positions": [4, 8],
        "write_positions": [5, 5],
        "reason_codes": ["duplicate_write_positions"],
    }


def test_offset_empty_edits_are_independently_invalid_even_for_empty_prompts() -> None:
    assert plan_offset([], clean_length=0, typo_length=0) == {
        "eligibility": "invalid",
        "source_positions": [],
        "write_positions": [],
        "reason_codes": ["no_aligned_edits"],
    }


@pytest.mark.parametrize(
    ("edits", "clean_length", "typo_length", "message"),
    [
        ((), 10, 10, "aligned_edits must be a JSON array"),
        ([{"clean_endpoint": 1}], 10, 10, "missing required fields"),
        ([_edit([2], [3]) | {"clean_endpoint": True}], 10, 10, "not a boolean"),
        ([_edit([2], [3]) | {"clean_token_indices": (2,)}], 10, 10, "JSON array"),
        ([_edit([2, 2], [3])], 10, 10, "strictly increasing"),
        ([_edit([2], [3]) | {"clean_endpoint": 1}], 10, 10, "must equal"),
        ([_edit([10], [3])], 10, 10, "outside its prompt"),
        ([_edit([2], [3])], -1, 10, "integer >= 0"),
    ],
)
def test_offset_rejects_malformed_or_internally_inconsistent_alignment(
    edits: object,
    clean_length: object,
    typo_length: object,
    message: str,
) -> None:
    with pytest.raises(IntakeError, match=message):
        plan_offset(edits, clean_length, typo_length)  # type: ignore[arg-type]


def test_offset_does_not_mutate_alignment_input() -> None:
    edits = [_edit([2, 3], [4, 5])]
    before = copy.deepcopy(edits)

    plan_offset(edits, clean_length=12, typo_length=12)

    assert edits == before


def _plan_rows() -> list[dict[str, object]]:
    return [
        {
            "plan_id": "plan-z",
            "pair_id": "pair-2",
            "original_problem_group_id": "group-alpha",
            "setting": "R4",
            "arm": "offset",
            "window": [6, 12],
        },
        {
            "plan_id": "plan-a",
            "pair_id": "pair-1",
            "original_problem_group_id": "group-alpha",
            "setting": "R3",
            "arm": "clean",
            "window": None,
        },
        {
            "plan_id": "plan-b",
            "pair_id": "pair-1",
            "original_problem_group_id": "group-alpha",
            "setting": "R4",
            "arm": "correct",
            "window": [0, 6],
        },
        {
            "plan_id": "plan-u",
            "pair_id": "pair-unicode",
            "original_problem_group_id": "問題-三",
            "setting": "R3",
            "arm": "typo",
            "window": None,
        },
        {
            "plan_id": "plan-c",
            "pair_id": "pair-3",
            "original_problem_group_id": "group-beta",
            "setting": "R3",
            "arm": "cross",
            "window": [0, 6],
        },
    ]


def test_shards_use_the_frozen_utf8_hash_oracle_for_every_allowed_count() -> None:
    expected = {
        "group-alpha": [0, 1, 2, 1, 2, 5],
        "group-beta": [0, 1, 1, 1, 0, 1],
        "問題-三": [0, 1, 1, 1, 2, 1],
    }

    for num_shards in range(1, 7):
        assignment = assign_shards(_plan_rows(), num_shards)
        observed = {
            group["original_problem_group_id"]: group["shard_index"]
            for group in assignment["groups"]
        }
        assert observed == {
            group_id: indices[num_shards - 1] for group_id, indices in expected.items()
        }


def test_shards_keep_all_variants_arms_and_windows_in_one_sorted_group() -> None:
    result = assign_shards(list(reversed(_plan_rows())), 3)

    assert result["schema_version"] == "rebuttal-shard-assignment/v2"
    assert result["num_shards"] == 3
    alpha = next(
        group for group in result["groups"] if group["original_problem_group_id"] == "group-alpha"
    )
    assert alpha == {
        "original_problem_group_id": "group-alpha",
        "shard_index": 2,
        "pair_ids": ["pair-1", "pair-2"],
        "plan_ids": ["plan-a", "plan-b", "plan-z"],
    }


def test_shards_reject_all_unknown_groups_and_report_every_blocking_pair() -> None:
    rows = _plan_rows()
    rows[0]["original_problem_group_id"] = None
    rows[3]["original_problem_group_id"] = None

    with pytest.raises(IntakeError) as caught:
        assign_shards(rows, 2)

    assert "pair-2" in str(caught.value)
    assert "pair-unicode" in str(caught.value)


def test_shards_reject_duplicate_plan_ids_and_conflicting_pair_groups() -> None:
    duplicate = _plan_rows()
    duplicate[1]["plan_id"] = duplicate[0]["plan_id"]
    with pytest.raises(IntakeError, match="duplicate plan_id"):
        assign_shards(duplicate, 2)

    conflicting = _plan_rows()
    conflicting[2]["original_problem_group_id"] = "group-beta"
    with pytest.raises(IntakeError, match="conflicting original_problem_group_id"):
        assign_shards(conflicting, 2)


@pytest.mark.parametrize("num_shards", [False, 0, 7])
def test_shards_reject_counts_outside_one_through_six(num_shards: object) -> None:
    with pytest.raises(IntakeError, match="num_shards"):
        assign_shards([], num_shards)  # type: ignore[arg-type]


def test_shards_are_order_independent_and_do_not_mutate_plan_rows() -> None:
    rows = _plan_rows()
    before = copy.deepcopy(rows)

    forward = assign_shards(rows, 4)
    reverse = assign_shards(list(reversed(rows)), 4)

    assert forward == reverse
    assert rows == before
