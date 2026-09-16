"""Hand-specified token versus word targeting fixtures, without tokenizer replay."""

from __future__ import annotations

from copy import deepcopy

import pytest

from typo_cot.experiments.rebuttal_v2.input_audit import _intended_hit


def _fixture(
    target: int = 2,
    clean_span: tuple[int, int] = (3, 4),
    typo_span: tuple[int, int] = (3, 4),
) -> tuple[dict, dict, dict]:
    # Exact clean prompt: "abcd / abcd". The current query is the SECOND word.
    # Tokens 0/1 belong to the few-shot repeat; 2/3 split the current query ab|cd.
    # Token 4 is a zero-width special token, not an editable character span.
    pair = {
        "clean_query_span": [7, 11],
        "intended_edits": [
            {
                "audit_adapter": "rebuttal-intended-token/v2",
                "clean_token_index": target,
                "actual_clean_span": list(clean_span),
                "actual_typo_span": list(typo_span),
            }
        ],
    }
    span_audit = {
        "edit_spans": [
            {
                "clean_start": clean_span[0],
                "clean_end": clean_span[1],
                "typo_start": typo_span[0],
                "typo_end": typo_span[1],
            }
        ]
    }
    alignment = {
        "eligibility": "valid",
        "clean": {
            "prompt_token_ids": [101, 102, 101, 102, 1],
            "offset_mapping": [[0, 2], [2, 4], [7, 9], [9, 11], [0, 0]],
        },
        "aligned_edits": [{"edit_indices": [0], "clean_token_indices": [2, 3]}],
    }
    return pair, span_audit, alignment


@pytest.mark.parametrize(("target", "hit"), [(2, False), (3, True)])
def test_same_changed_word_does_not_make_every_subtoken_a_target_hit(target, hit) -> None:
    result = _intended_hit(*_fixture(target))
    row = result["records"][0]
    assert row["word_hit"] is True
    assert row["hit"] is hit
    assert result["status"] == ("hit" if hit else "miss")
    assert row["reason_codes"] == ([] if hit else ["intended_token_misses_actual_edit"])


@pytest.mark.parametrize("target", [0, 1])
def test_repeated_fewshot_token_is_not_selected_and_query_offset_is_applied(target) -> None:
    result = _intended_hit(*_fixture(target))
    assert result["status"] == "miss"
    assert result["records"][0]["hit"] is False
    assert result["records"][0]["word_hit"] is False


@pytest.mark.parametrize(
    ("target", "clean_span", "expected"),
    [(2, (0, 1), True), (3, (0, 1), False), (2, (1, 3), True), (3, (1, 3), True)],
)
def test_positive_actual_edit_overlap_is_evaluated_in_full_prompt_coordinates(
    target, clean_span, expected
) -> None:
    result = _intended_hit(*_fixture(target, clean_span, clean_span))
    assert result["records"][0]["hit"] is expected


@pytest.mark.parametrize(
    ("target", "position", "expected"),
    [(2, 1, True), (3, 3, True), (0, 1, False), (2, 3, False), (3, 1, False)],
)
def test_insertion_strictly_inside_intended_token_hits_and_disjoint_token_misses(
    target, position, expected
) -> None:
    result = _intended_hit(*_fixture(target, (position, position), (position, position + 1)))
    assert result["records"][0]["hit"] is expected
    assert result["status"] == ("hit" if expected else "miss")


@pytest.mark.parametrize(("target", "position"), [(2, 0), (2, 2), (3, 2), (3, 4)])
def test_insertion_at_intended_token_boundary_stays_unknown(target, position) -> None:
    result = _intended_hit(*_fixture(target, (position, position), (position, position + 1)))
    row = result["records"][0]
    assert row["word_hit"] is True
    assert row["hit"] is None
    assert result["status"] == "unknown"
    assert row["reason_codes"] == ["insertion_at_intended_token_boundary"]


def test_zero_width_special_token_cannot_receive_a_character_edit_hit() -> None:
    result = _intended_hit(*_fixture(target=4))
    assert result["status"] == "unknown"
    assert result["records"][0]["hit"] is None
    assert result["records"][0]["word_hit"] is False
    assert result["records"][0]["reason_codes"] == ["intended_token_has_no_character_span"]


@pytest.mark.parametrize("field", ["actual_clean_span", "actual_typo_span"])
def test_archived_actual_span_must_bind_the_independent_edit_on_both_sides(field) -> None:
    pair, span, alignment = _fixture(target=3)
    pair["intended_edits"][0][field] = [1, 2]
    result = _intended_hit(pair, span, alignment)
    assert result["status"] == "unknown"
    assert result["records"][0]["reason_codes"] == [
        "intended_edit_not_uniquely_bound_to_independent_diff"
    ]
    assert "word_hit" not in result["records"][0]


def test_independent_edit_cannot_be_bound_to_two_different_word_rows() -> None:
    pair, span, alignment = _fixture(target=3)
    alignment["aligned_edits"].append({"edit_indices": [0], "clean_token_indices": [0, 1]})
    result = _intended_hit(pair, span, alignment)
    assert result["status"] == "unknown"
    assert result["records"][0]["hit"] is None


@pytest.mark.parametrize(
    "entry",
    [None, [], {}, {"clean_token_index": 3}, {"audit_adapter": "legacy-unknown"}],
)
def test_unsupported_archived_coordinate_metadata_is_unknown(entry) -> None:
    pair, span, alignment = _fixture()
    pair["intended_edits"] = [entry]
    result = _intended_hit(pair, span, alignment)
    assert result["status"] == "unknown"
    assert result["records"][0]["reason_codes"] == ["intended_coordinate_adapter_unavailable"]


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        (True, "intended_coordinate_adapter_unavailable"),
        (-1, "intended_token_out_of_bounds"),
        (5, "intended_token_out_of_bounds"),
    ],
)
def test_invalid_target_token_indices_do_not_become_hits_or_misses(target, reason) -> None:
    result = _intended_hit(*_fixture(target=target))
    assert result["status"] == "unknown"
    assert result["records"][0]["reason_codes"] == [reason]


@pytest.mark.parametrize("eligibility", ["invalid", "unknown"])
def test_unavailable_independent_alignment_is_not_rescued_by_archived_metadata(eligibility) -> None:
    pair, span, alignment = _fixture(target=3)
    alignment["eligibility"] = eligibility
    result = _intended_hit(pair, span, alignment)
    assert result["status"] == "unknown"
    assert result["records"][0]["reason_codes"] == ["independent_alignment_unavailable"]


@pytest.mark.parametrize(
    ("intended", "status", "reason"),
    [
        (None, "unknown", "intended_metadata_unavailable"),
        ([], "not_applicable", "no_recorded_intended_edits"),
    ],
)
def test_absent_and_explicitly_empty_attempt_histories_are_distinct(
    intended, status, reason
) -> None:
    pair, span, alignment = _fixture()
    pair["intended_edits"] = intended
    result = _intended_hit(pair, span, alignment)
    assert result == {"status": status, "records": [], "reason_codes": [reason]}


@pytest.mark.parametrize(("second_target", "expected_status"), [(2, "mixed"), (4, "unknown")])
def test_per_attempt_hit_and_unknown_results_remain_visible_in_pair_status(
    second_target, expected_status
) -> None:
    pair, span, alignment = _fixture(target=3)
    second = deepcopy(pair["intended_edits"][0])
    second["clean_token_index"] = second_target
    pair["intended_edits"].append(second)
    result = _intended_hit(pair, span, alignment)
    assert result["status"] == expected_status
    assert result["records"][0]["hit"] is True
    assert result["records"][1]["hit"] is (False if second_target == 2 else None)
