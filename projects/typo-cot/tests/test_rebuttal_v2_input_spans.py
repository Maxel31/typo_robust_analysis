"""Independent exact-offset, ambiguity, and non-semantic diagnostic fixtures."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
import unicodedata

import pytest

from typo_cot.experiments.rebuttal_v2 import input_spans
from typo_cot.experiments.rebuttal_v2.input_spans import (
    classify_edit_regions,
    diff_actual_spans,
    word_spans,
)


def _span(left: int, right: int, other_left: int, other_right: int) -> dict[str, int]:
    return {
        "clean_start": left,
        "clean_end": right,
        "typo_start": other_left,
        "typo_end": other_right,
    }


def _record(clean: str, typo: str, *, kind: str = "stem") -> dict:
    prefix = "Example: cat cat.\n"
    return {
        "clean_text": clean,
        "typo_text": typo,
        "clean_prompt": prefix + clean,
        "typo_prompt": prefix + typo,
        "clean_query_span": [len(prefix), len(prefix) + len(clean)],
        "typo_query_span": [len(prefix), len(prefix) + len(typo)],
        "prompt_regions": {
            "clean": [{"kind": kind, "start": len(prefix), "end": len(prefix) + len(clean)}],
            "typo": [{"kind": kind, "start": len(prefix), "end": len(prefix) + len(typo)}],
        },
    }


def test_all_disjoint_edits_and_same_word_deduplication() -> None:
    result = diff_actual_spans("abcde fox", "AbcdE fix")
    assert result.diff_status == "changed"
    assert result.edit_distance == 3
    assert result.optimal_path_count == 1
    assert result.edit_spans == [_span(0, 1, 0, 1), _span(4, 5, 4, 5), _span(7, 8, 7, 8)]
    assert result.changed_words == {"clean": [[0, 5], [6, 9]], "typo": [[0, 5], [6, 9]]}
    assert result.aligned_words == [
        {"clean_span": [0, 5], "typo_span": [0, 5], "edit_indices": [0, 1]},
        {"clean_span": [6, 9], "typo_span": [6, 9], "edit_indices": [2]},
    ]
    assert result.alignment_status == "aligned"


def test_repeated_words_are_positioned_by_diff_not_first_substring() -> None:
    result = diff_actual_spans("cat cat cat", "cat cot cat")
    assert result.edit_spans == [_span(5, 6, 5, 6)]
    assert result.aligned_words == [
        {"clean_span": [4, 7], "typo_span": [4, 7], "edit_indices": [0]}
    ]


def test_original_four_edit_intervention_is_not_collapsed_to_one_edit() -> None:
    result = diff_actual_spans("one two six ten", "One tWo sIx teN")
    assert result.alignment_status == "aligned"
    assert result.edit_distance == 4
    assert result.edit_spans == [
        _span(0, 1, 0, 1),
        _span(5, 6, 5, 6),
        _span(9, 10, 9, 10),
        _span(14, 15, 14, 15),
    ]
    assert len(result.aligned_words) == 4


@pytest.mark.parametrize(
    ("clean", "typo", "expected"),
    [
        (" café", " cafè", _span(4, 5, 4, 5)),
        ("cafe\u0301", "cafe\u0300", _span(4, 5, 4, 5)),
        ("猫を見た", "犬を見た", _span(0, 1, 0, 1)),
        ("🧠 cat", "🧠 cot", _span(3, 4, 3, 4)),
        ("cat", "cats", _span(3, 3, 3, 4)),
        ("cats", "cat", _span(3, 4, 3, 3)),
    ],
)
def test_unicode_offsets_are_codepoints_and_insertions_keep_both_spans(
    clean, typo, expected
) -> None:
    result = diff_actual_spans(clean, typo)
    assert result.diff_status == "changed"
    assert result.edit_spans == [expected]
    assert result.alignment_status == "aligned"


def test_unicode_is_not_normalized_and_segmentation_version_is_recorded() -> None:
    result = diff_actual_spans("é", "e\u0301")
    assert result.diff_status != "no_change"
    assert result.algorithm_versions["unicode_database"] == unicodedata.unidata_version
    assert "word_segmentation" in result.algorithm_versions
    assert word_spans(" café e\u0301 can't can’t a_b 😀 １２") == [
        [1, 5],
        [6, 8],
        [9, 14],
        [15, 20],
        [21, 22],
        [23, 24],
        [27, 29],
    ]


def test_apostrophes_are_internal_and_isolated_combining_marks_are_not_words() -> None:
    assert word_spans("'a' a''b \u0301 c\u0301 ’d’") == [[1, 2], [4, 5], [7, 8], [11, 13], [15, 16]]


@pytest.mark.parametrize(("clean", "typo"), [("a", "aa"), ("ab", "ba"), ("red dog", "red  dog")])
def test_multiple_optimal_paths_remain_ambiguous_with_representative_edits(clean, typo) -> None:
    result = diff_actual_spans(clean, typo)
    assert result.diff_status == result.alignment_status == "ambiguous"
    assert result.optimal_path_count == 2
    assert "multiple_optimal_diff_paths" in result.reasons
    assert result.edit_spans
    assert asdict(result) == asdict(diff_actual_spans(clean, typo))


def test_fixed_backtrace_priority_keeps_substitutions_and_leftmost_repeated_insertion() -> None:
    assert diff_actual_spans("ab", "ba").edit_spans == [_span(0, 2, 0, 2)]
    assert diff_actual_spans("a", "aa").edit_spans == [_span(0, 0, 0, 1)]


@pytest.mark.parametrize(
    ("clean", "typo"), [("blue", "bl ue"), ("", "cat"), ("cat", ""), ("x!", "x?")]
)
def test_unresolvable_word_endpoints_are_not_certified(clean, typo) -> None:
    result = diff_actual_spans(clean, typo)
    assert result.diff_status == "changed"
    assert result.alignment_status == "unsupported"
    assert result.aligned_words == []
    assert "edited_word_pair_not_unique" in result.reasons


@pytest.mark.parametrize("text", ["", "same", "同じ e\u0301"])
def test_no_net_change_does_not_invent_cancellation_history(text) -> None:
    result = diff_actual_spans(text, text)
    assert result.diff_status == result.alignment_status == "no_change"
    assert result.edit_distance == 0
    assert result.edit_spans == result.aligned_words == []
    assert "cancellation_not_identifiable_from_texts" in result.reasons


def test_resource_limits_are_unavailable_not_no_change_or_fake_spans(monkeypatch) -> None:
    monkeypatch.setattr(input_spans, "MAX_TEXT_CODEPOINTS", 3)
    result = diff_actual_spans("same", "same")
    assert result.diff_status == "unavailable"
    assert result.edit_distance is None
    assert result.optimal_path_count == 0
    assert result.edit_spans == []
    monkeypatch.setattr(input_spans, "MAX_TEXT_CODEPOINTS", 100)
    monkeypatch.setattr(input_spans, "MAX_DIFF_CELLS", 8)
    result = diff_actual_spans("ab", "cd")
    assert result.diff_status == "unavailable"
    assert result.reasons == ["diff_matrix_resource_limit"]
    assert diff_actual_spans("same", "same").diff_status == "no_change"


@pytest.mark.parametrize(("clean", "typo"), [(None, "x"), ("x", b"x"), (0, False)])
def test_invalid_diff_api_arguments_are_errors(clean, typo) -> None:
    with pytest.raises(ValueError, match="exact clean and typo strings"):
        diff_actual_spans(clean, typo)


def _all_path_costs(clean: str, typo: str) -> list[int]:
    """Tiny exhaustive oracle, enumerating paths rather than sharing DP logic."""
    if not clean:
        return [len(typo)]
    if not typo:
        return [len(clean)]
    return (
        [cost + (clean[0] != typo[0]) for cost in _all_path_costs(clean[1:], typo[1:])]
        + [cost + 1 for cost in _all_path_costs(clean[1:], typo)]
        + [cost + 1 for cost in _all_path_costs(clean, typo[1:])]
    )


def test_exhaustive_small_oracle_checks_distance_ambiguity_and_edit_reconstruction() -> None:
    texts = [""] + ["".join(parts) for size in (1, 2, 3) for parts in product("ab", repeat=size)]
    for clean, typo in product(texts, repeat=2):
        result = diff_actual_spans(clean, typo)
        costs = _all_path_costs(clean, typo)
        expected = min(costs)
        assert result.edit_distance == expected, (clean, typo)
        assert result.optimal_path_count == min(2, costs.count(expected)), (clean, typo)
        rebuilt = clean
        for edit in reversed(result.edit_spans):
            rebuilt = (
                rebuilt[: edit["clean_start"]]
                + typo[edit["typo_start"] : edit["typo_end"]]
                + rebuilt[edit["clean_end"] :]
            )
        assert rebuilt == typo, (clean, typo)


@pytest.mark.parametrize(
    ("clean", "typo", "flag"),
    [
        ("12 cats", "13 cats", "number"),
        ("5 cats", "-5 cats", "number"),
        ("15 cats", "1.5 cats", "number"),
        ("５ cats", "６ cats", "number"),
        ("each cat", "eacg cat", "quantity"),
        ("few cats", "fex cats", "quantity"),
        ("5 kg", "5 kx", "unit"),
        ("5$", "5€", "unit"),
        ("not cats", "nit cats", "negation"),
        ("can't go", "canxt go", "negation"),
        ("cannot go", "can not go", "negation"),
    ],
)
def test_risk_diagnostics_include_stem_numbers_units_quantities_and_negation(
    clean, typo, flag
) -> None:
    result = classify_edit_regions(_record(clean, typo))
    assert result["flags"][flag] is True
    assert result["flags"]["stem"] is True
    assert result["flags"]["option_content"] is False
    assert "semantic_label" not in result


def test_unchanged_risky_words_do_not_set_edit_flags() -> None:
    result = classify_edit_regions(_record("not 12 kg cats", "not 12 kg cots"))
    assert {name: result["flags"][name] for name in ("number", "unit", "quantity", "negation")} == {
        "number": False,
        "unit": False,
        "quantity": False,
        "negation": False,
    }
    assert result["flags"]["entity"] is None


def _option_record(gold: str = "A") -> dict:
    record = _record("stem\nA. cat\nB. dog", "stem\nA. cot\nB. dog")
    record.update(canonical_gold=gold, valid_labels=["A", "B"])
    prefix = record["clean_query_span"][0]
    regions = [
        {"kind": "stem", "start": prefix, "end": prefix + 4},
        {"kind": "option-label", "start": prefix + 5, "end": prefix + 6, "option_label": "A"},
        {"kind": "option-content", "start": prefix + 8, "end": prefix + 11, "option_label": "A"},
        {"kind": "option-label", "start": prefix + 12, "end": prefix + 13, "option_label": "B"},
        {"kind": "option-content", "start": prefix + 15, "end": prefix + 18, "option_label": "B"},
    ]
    record["prompt_regions"] = {"clean": regions, "typo": regions}
    return record


@pytest.mark.parametrize(("gold", "expected"), [("A", True), ("B", False)])
def test_option_content_edit_and_gold_membership_are_separate(gold, expected) -> None:
    result = classify_edit_regions(_option_record(gold))["flags"]
    assert result["option_content"] is True
    assert result["option_label"] is False
    assert result["stem"] is False
    assert result["gold_option"] is expected


def test_option_label_edit_is_not_an_option_content_edit() -> None:
    record = _option_record()
    record["typo_text"] = "stem\nC. cat\nB. dog"
    record["typo_prompt"] = (
        record["typo_prompt"][: record["typo_query_span"][0]] + record["typo_text"]
    )
    result = classify_edit_regions(record)["flags"]
    assert result["option_label"] is True
    assert result["option_content"] is False
    assert result["gold_option"] is True


@pytest.mark.parametrize("field", ["canonical_gold", "valid_labels"])
def test_missing_gold_information_is_not_non_gold_option(field) -> None:
    record = _option_record()
    record[field] = None
    result = classify_edit_regions(record)["flags"]
    assert result["option_content"] is True
    assert result["gold_option"] is None


def test_option_membership_requires_label_annotations() -> None:
    record = _option_record()
    for side in ("clean", "typo"):
        record["prompt_regions"][side] = [
            {name: value for name, value in region.items() if name != "option_label"}
            for region in record["prompt_regions"][side]
        ]
    flags = classify_edit_regions(record)["flags"]
    assert flags["option_content"] is True
    assert flags["gold_option"] is None


def test_entities_require_explicit_regions_not_capitalization_heuristics() -> None:
    record = _record("Alice saw cats", "Alicf saw cats")
    assert classify_edit_regions(record)["flags"]["entity"] is None
    for side in ("clean", "typo"):
        start = record[f"{side}_query_span"][0]
        record["prompt_regions"][side].append({"kind": "entity", "start": start, "end": start + 5})
    assert classify_edit_regions(record)["flags"]["entity"] is True


@pytest.mark.parametrize(
    "missing",
    ["prompt_regions", "clean_prompt", "typo_prompt", "clean_query_span", "typo_query_span"],
)
def test_missing_layout_is_unknown_not_false_while_lexical_flags_remain_available(missing) -> None:
    record = _record("12 cats", "13 cats")
    record[missing] = None
    result = classify_edit_regions(record)
    assert result["flags"]["number"] is True
    assert result["flags"]["option_content"] is None
    # A positive explicit region hit on the other known side is still evidence.
    assert result["flags"]["stem"] is (None if missing == "prompt_regions" else True)


def test_uncovered_spans_and_wrong_query_coordinates_do_not_use_substring_fallback() -> None:
    record = _record("cat cat", "cat cot")
    record["clean_query_span"] = [0, 7]
    record["typo_query_span"] = [0, 7]
    result = classify_edit_regions(record)
    assert result["flags"]["stem"] is None
    assert "clean_query_span_text_mismatch" in result["reasons"]["stem"]
    record = _record("cat cat", "cat cot")
    record["prompt_regions"] = {"clean": [], "typo": []}
    assert classify_edit_regions(record)["flags"]["stem"] is None


@pytest.mark.parametrize(
    "region",
    [
        None,
        {"kind": []},
        {"kind": "stem", "start": True, "end": 20},
        {"kind": "stem", "start": -1, "end": 20},
        {"kind": "stem", "start": 0, "end": 999},
        {"kind": "unsupported", "start": 0, "end": 1},
    ],
)
def test_unsupported_region_annotations_are_unknown_not_errors(region) -> None:
    record = _record("cat", "cot")
    record["prompt_regions"] = {"clean": [region], "typo": [region]}
    assert classify_edit_regions(record)["flags"]["stem"] is None


def test_ambiguous_diff_does_not_use_representative_path_to_claim_risk_or_no_risk() -> None:
    result = classify_edit_regions(_record("a", "aa"))
    assert all(value is None for value in result["flags"].values())
    assert all(value == ["diff_ambiguous"] for value in result["reasons"].values())


def test_no_change_is_no_edit_risk_not_a_preserved_semantic_label() -> None:
    record = {"clean_text": "not 12 kg", "typo_text": "not 12 kg"}
    result = classify_edit_regions(record)
    assert all(value is False for value in result["flags"].values())
    assert "semantic_label" not in result


def test_missing_exact_texts_preserves_unknown_all_dimensions() -> None:
    assert all(value is None for value in classify_edit_regions({})["flags"].values())


def test_results_do_not_depend_on_archived_positions_or_model_outcomes() -> None:
    record = _option_record()
    initial = classify_edit_regions(record)
    record.update(
        archived_token_positions={"clean": [0], "typo": [999]},
        intended_edits=[{"start": 0, "end": 1}],
        archived_generations=[{"arm": "correct", "gold_match": True}],
        semantic_label="unique_preserved",
    )
    assert classify_edit_regions(record) == initial
