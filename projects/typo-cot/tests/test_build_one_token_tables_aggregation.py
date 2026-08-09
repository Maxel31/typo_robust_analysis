"""Aggregation contracts for the CPU-only one-token table builder.

The records below are deliberately tiny.  A ``ValidatedOneTokenRun`` represents
an input whose schema, hashes, completion state, and paper protocol have already
been checked by the loading boundary.  These tests freeze the scientific
pooling rules independently of filesystem discovery and rendering.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import pytest

from typo_cot.experiments.build_one_token_tables import (
    InferenceProtocol,
    ValidatedOneTokenRun,
    assemble_one_token_tables,
)


Setting = tuple[str, str, str]
RecordFactory = Callable[[str, str, str, int], Mapping[str, object]]

PRIMARY: Setting = (
    "gemma4b_gsm8k",
    "google/gemma-3-4b-it",
    "gsm8k",
)
EXTENSIONS: tuple[Setting, ...] = (
    ("gemma1b_gsm8k", "google/gemma-3-1b-it", "gsm8k"),
    ("gemma1b_mmlu", "google/gemma-3-1b-it", "mmlu"),
    ("gemma1b_arc", "google/gemma-3-1b-it", "arc"),
    ("gemma4b_mmlu", "google/gemma-3-4b-it", "mmlu"),
    ("gemma4b_arc", "google/gemma-3-4b-it", "arc"),
    ("llama1b_gsm8k", "meta-llama/Llama-3.2-1B-Instruct", "gsm8k"),
    ("llama1b_mmlu", "meta-llama/Llama-3.2-1B-Instruct", "mmlu"),
    ("llama1b_arc", "meta-llama/Llama-3.2-1B-Instruct", "arc"),
    ("llama3b_gsm8k", "meta-llama/Llama-3.2-3B-Instruct", "gsm8k"),
    ("llama3b_mmlu", "meta-llama/Llama-3.2-3B-Instruct", "mmlu"),
    ("llama3b_arc", "meta-llama/Llama-3.2-3B-Instruct", "arc"),
    ("mistral7b_gsm8k", "mistralai/Mistral-7B-Instruct-v0.3", "gsm8k"),
    ("mistral7b_mmlu", "mistralai/Mistral-7B-Instruct-v0.3", "mmlu"),
    ("mistral7b_arc", "mistralai/Mistral-7B-Instruct-v0.3", "arc"),
)
ADJACENT_SETTINGS = frozenset(
    {
        "gemma1b_gsm8k",
        "llama3b_arc",
        "mistral7b_mmlu",
    }
)
FAST_INFERENCE = InferenceProtocol(
    seed=42,
    bootstrap_draws=64,
    sign_flip_draws=64,
)


def _record(
    *,
    setting_id: str,
    benchmark: str,
    index: int,
    table10_selected: bool = True,
    table10_control: bool = False,
    literal_eligible: bool = True,
    literal_selected_events: int = 1,
    literal_control_events: int = 0,
    submitted_eligible: bool = True,
    submitted_selected_events: int | None = None,
    submitted_control_events: int | None = None,
    exclusion_reasons: tuple[str, ...] = (),
    selected_before_control: bool = True,
    adjacent: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return one producer-shaped record with preclassified integer events."""

    selected_position, distant_position = (0, 4) if selected_before_control else (4, 0)
    if submitted_selected_events is None:
        submitted_selected_events = literal_selected_events if submitted_eligible else 0
    if submitted_control_events is None:
        submitted_control_events = literal_control_events if submitted_eligible else 0
    clean_ids = [10, 11, 12, 13, 14]
    kl = [0.1, 0.5, 0.2, 0.5, 0.1]
    kl[selected_position] = 2.0
    return {
        "sample_id": f"{benchmark}_synthetic_{index:03d}",
        "benchmark": benchmark,
        "targeting": "attribution-4",
        "input_plan": {"clean_cot_ids": clean_ids},
        "profile": {
            "clean_to_edited_kl": kl,
            "clean_token_rank_under_clean": [1, 1, 1, 1, 1],
            "clean_token_rank_under_edited": [2, 2, 1, 2, 2],
            "edited_top1_ids": [20, 21, 22, 23, 24],
        },
        "positions": {
            "selected": selected_position,
            "distant": distant_position,
            "adjacent": (
                (1 if selected_before_control else 3) if setting_id in ADJACENT_SETTINGS else None
            ),
        },
        "arms": [],
        "events": {
            "table10": {
                "eligible": True,
                "selected_correct_to_incorrect": table10_selected,
                "control_correct_to_incorrect": table10_control,
            },
            "distant_factorial": {
                "eligible": literal_eligible,
                "selected_loss_events": literal_selected_events if literal_eligible else 0,
                "control_loss_events": literal_control_events if literal_eligible else 0,
                "event_opportunities_per_position": 2,
                "selected_before_control": selected_before_control,
            },
            "distant_factorial_submitted_producer": {
                "eligible": submitted_eligible,
                "selected_loss_events": submitted_selected_events,
                "control_loss_events": submitted_control_events,
                "event_opportunities_per_position": 2,
                "selected_before_control": selected_before_control,
                "exclusion_reasons": list(exclusion_reasons),
            },
            "adjacent": None if adjacent is None else dict(adjacent),
        },
    }


def _run(
    setting: Setting,
    *,
    cohort: str,
    record: Mapping[str, object],
) -> ValidatedOneTokenRun:
    setting_id, model, benchmark = setting
    return ValidatedOneTokenRun(
        setting_id=setting_id,
        model=model,
        benchmark=benchmark,
        cohort=cohort,
        adjacent_requested=setting_id in ADJACENT_SETTINGS,
        records=(record,),
    )


def _primary_run(record: Mapping[str, object] | None = None) -> ValidatedOneTokenRun:
    setting_id, _, benchmark = PRIMARY
    return _run(
        PRIMARY,
        cohort="primary",
        record=(
            _record(setting_id=setting_id, benchmark=benchmark, index=999)
            if record is None
            else record
        ),
    )


def _extension_runs(factory: RecordFactory | None = None) -> list[ValidatedOneTokenRun]:
    runs: list[ValidatedOneTokenRun] = []
    for index, setting in enumerate(EXTENSIONS):
        setting_id, _, benchmark = setting
        record = (
            _record(
                setting_id=setting_id,
                benchmark=benchmark,
                index=index,
                adjacent=(
                    {
                        "eligible": True,
                        "selected_correct_to_incorrect": True,
                        "control_correct_to_incorrect": False,
                    }
                    if setting_id in ADJACENT_SETTINGS
                    else None
                ),
            )
            if factory is None
            else factory(setting_id, setting[1], benchmark, index)
        )
        runs.append(_run(setting, cohort="extension", record=record))
    return runs


def _assemble(runs: list[ValidatedOneTokenRun]) -> dict[str, object]:
    return assemble_one_token_tables(runs, inference=FAST_INFERENCE)


def test_table10_extension_pool_is_byte_stable_when_primary_outcomes_change() -> None:
    """The separately reported 172-pair primary cell can never affect 14 extensions."""

    extensions = _extension_runs()
    primary_a = _primary_run()
    primary_b = _primary_run(
        _record(
            setting_id=PRIMARY[0],
            benchmark=PRIMARY[2],
            index=999,
            table10_selected=False,
            table10_control=True,
            literal_selected_events=0,
            literal_control_events=2,
            submitted_selected_events=0,
            submitted_control_events=2,
        )
    )

    first = _assemble([primary_a, *extensions])
    second = _assemble([primary_b, *extensions])
    first_pool = first["table10"]["extension_pool"]  # type: ignore[index]
    second_pool = second["table10"]["extension_pool"]  # type: ignore[index]

    assert first_pool == second_pool
    assert json.dumps(first_pool, sort_keys=True) == json.dumps(second_pool, sort_keys=True)
    assert first_pool["paired_eligible"] == 14
    assert first_pool["selected"] == {"numerator": 14, "denominator": 14, "rate": 1.0}
    assert first_pool["control"] == {"numerator": 0, "denominator": 14, "rate": 0.0}
    assert first["table10"]["primary"] != second["table10"]["primary"]  # type: ignore[index]


def test_distant_pdf_literal_and_submitted_denominators_are_never_conflated() -> None:
    def factory(
        setting_id: str,
        _model: str,
        benchmark: str,
        index: int,
    ) -> Mapping[str, object]:
        if index == 0:
            return _record(
                setting_id=setting_id,
                benchmark=benchmark,
                index=index,
                literal_selected_events=2,
                literal_control_events=1,
                submitted_eligible=False,
                exclusion_reasons=("selected-and-control-tokens-identical",),
            )
        return _record(
            setting_id=setting_id,
            benchmark=benchmark,
            index=index,
            literal_selected_events=1,
            literal_control_events=0,
        )

    result = _assemble([_primary_run(), *_extension_runs(factory)])
    distant = result["table11"]["distant"]  # type: ignore[index]
    literal = distant["pdf_literal"]["extension_pool"]
    submitted = distant["submitted_producer"]["extension_pool"]

    assert literal["paired_eligible"] == 14
    assert literal["event_opportunities_per_position"] == 28
    assert literal["selected"]["numerator"] == 15
    assert literal["control"]["numerator"] == 1
    assert submitted["paired_eligible"] == 13
    assert submitted["event_opportunities_per_position"] == 26
    assert submitted["selected"]["numerator"] == 13
    assert submitted["control"]["numerator"] == 0
    assert submitted["attrition_from_pdf_literal"] == {
        "count": 1,
        "reasons": {"selected-and-control-tokens-identical": 1},
    }


def test_distant_position_order_strata_use_separate_common_denominators() -> None:
    def factory(
        setting_id: str,
        _model: str,
        benchmark: str,
        index: int,
    ) -> Mapping[str, object]:
        before = index < 7
        return _record(
            setting_id=setting_id,
            benchmark=benchmark,
            index=index,
            literal_selected_events=2 if before else 0,
            literal_control_events=0 if before else 2,
            submitted_selected_events=2 if before else 0,
            submitted_control_events=0 if before else 2,
            selected_before_control=before,
        )

    result = _assemble([_primary_run(), *_extension_runs(factory)])
    literal = result["table11"]["distant"]["pdf_literal"]["extension_pool"]  # type: ignore[index]
    strata = literal["position_order_strata"]
    before = strata["selected_before_control"]
    after = strata["selected_after_control"]

    assert before["paired_eligible"] == 7
    assert before["selected"] == {"numerator": 14, "denominator": 14, "rate": 1.0}
    assert before["control"] == {"numerator": 0, "denominator": 14, "rate": 0.0}
    assert before["difference_percentage_points"] == pytest.approx(100.0)
    assert after["paired_eligible"] == 7
    assert after["selected"] == {"numerator": 0, "denominator": 14, "rate": 0.0}
    assert after["control"] == {"numerator": 14, "denominator": 14, "rate": 1.0}
    assert after["difference_percentage_points"] == pytest.approx(-100.0)


def test_adjacent_pool_uses_exactly_three_prespecified_complete_settings() -> None:
    complete = _assemble([_primary_run(), *_extension_runs()])
    adjacent = complete["table11"]["adjacent"]  # type: ignore[index]

    assert set(adjacent["settings"]) == ADJACENT_SETTINGS
    assert adjacent["extension_pool"]["paired_eligible"] == 3
    assert adjacent["extension_pool"]["selected"] == {
        "numerator": 3,
        "denominator": 3,
        "rate": 1.0,
    }
    assert adjacent["extension_pool"]["control"] == {
        "numerator": 0,
        "denominator": 3,
        "rate": 0.0,
    }

    without_one_adjacent = [
        run for run in [_primary_run(), *_extension_runs()] if run.setting_id != "llama3b_arc"
    ]
    partial = _assemble(without_one_adjacent)
    assert partial["table11"]["adjacent"]["extension_pool"] is None  # type: ignore[index]


def test_partial_coverage_never_emits_a_paper_labelled_pool() -> None:
    one_extension = _extension_runs()[0]
    result = _assemble([_primary_run(), one_extension])
    coverage = result["coverage"]

    assert coverage["complete_15_setting_grid"] is False
    assert coverage["complete_14_extension_grid"] is False
    assert coverage["complete_3_adjacent_grid"] is False
    assert len(coverage["missing_settings"]) == 13
    assert result["table10"]["extension_pool"] is None  # type: ignore[index]
    assert result["table11"]["distant"]["pdf_literal"]["extension_pool"] is None  # type: ignore[index]
    assert (
        result["table11"]["distant"]["submitted_producer"]["extension_pool"]  # type: ignore[index]
        is None
    )
    assert result["table11"]["adjacent"]["extension_pool"] is None  # type: ignore[index]
    assert result["historical_comparison"]["table10"]["extension_pool"] == {
        "status": "not_comparable",
        "reason": "complete-14-extension-grid-required",
    }


def test_historical_values_are_reference_comparisons_not_acceptance_thresholds() -> None:
    """A fresh cohort may differ while remaining a valid completed analysis."""

    result = _assemble([_primary_run(), *_extension_runs()])
    comparison = result["historical_comparison"]

    assert comparison["contract"] == "reference-only-not-runtime-acceptance"
    table10 = comparison["table10"]["extension_pool"]
    assert table10["status"] == "differs"
    assert table10["reference"]["paired_eligible"] == 1629
    assert table10["reference"]["selected"]["numerator"] == 492
    assert table10["observed"]["paired_eligible"] == 14
    submitted = comparison["table11"]["distant_submitted_producer"]
    assert submitted["status"] == "differs"
    assert submitted["reference"]["paired_eligible"] == 1575
    assert submitted["observed"]["paired_eligible"] == 14
    assert set(comparison["table11"]["distant_submitted_position_order_strata"]) == {
        "selected_before_control",
        "selected_after_control",
    }
    adjacent = comparison["table11"]["adjacent"]
    assert adjacent["status"] == "differs"
    assert adjacent["reference"]["paired_eligible"] == 391
    assert adjacent["observed"]["paired_eligible"] == 3


def _figure5_record() -> dict[str, object]:
    clean_ids = list(range(1_000, 1_061))
    clean_kl = [0.1] * 61
    clean_kl[23] = 8.785974
    clean_ranks = [1] * 61
    edited_ranks = [2] * 61
    edited_ranks[23] = 8
    edited_top1 = list(range(2_000, 2_061))
    arms = [
        {
            "name": "selected_keep",
            "forced_token_id": clean_ids[23],
            "generation": {"extracted_answer": "160", "is_correct": True},
            "is_noop": True,
        },
        {
            "name": "selected_from_selected",
            "forced_token_id": edited_top1[23],
            "generation": {"extracted_answer": "120", "is_correct": False},
            "is_noop": False,
        },
        {
            "name": "distant_keep",
            "forced_token_id": clean_ids[60],
            "generation": {"extracted_answer": "160", "is_correct": True},
            "is_noop": True,
        },
        {
            "name": "distant_from_distant",
            "forced_token_id": edited_top1[60],
            "generation": {"extracted_answer": "160", "is_correct": True},
            "is_noop": False,
        },
    ]
    record = _record(
        setting_id=PRIMARY[0],
        benchmark=PRIMARY[2],
        index=556,
    )
    record.update(
        {
            "sample_id": "gsm8k_00556",
            "targeting": "attribution-4",
            "input_plan": {"clean_cot_ids": clean_ids},
            "profile": {
                "clean_to_edited_kl": clean_kl,
                "clean_token_rank_under_clean": clean_ranks,
                "clean_token_rank_under_edited": edited_ranks,
                "edited_top1_ids": edited_top1,
            },
            "positions": {"selected": 23, "distant": 60, "adjacent": None},
            "arms": arms,
        }
    )
    return record


def test_figure5_validates_stored_fields_but_not_unstored_token_text() -> None:
    result = _assemble([_primary_run(_figure5_record()), *_extension_runs()])
    figure = result["figure5_validation"]
    checks = figure["checks"]

    assert figure["identity"] == {
        "setting_id": "gemma4b_gsm8k",
        "targeting": "attribution-4",
        "sample_id": "gsm8k_00556",
    }
    for field, expected in {
        "clean_cot_token_count": 61,
        "selected_position": 23,
        "distant_position": 60,
        "selected_kl": 8.785974,
        "clean_token_rank_under_clean": 1,
        "clean_token_rank_under_edited": 8,
        "selected_keep_answer": "160",
        "selected_replacement_answer": "120",
        "distant_keep_answer": "160",
        "distant_replacement_answer": "160",
    }.items():
        assert checks[field] == {
            "expected": expected,
            "observed": expected,
            "status": "match",
        }

    assert checks["clean_token_text"] == {
        "expected": " thrice",
        "observed": None,
        "status": "unverifiable-from-one-token-record",
    }
    assert checks["selected_edited_top1_text"] == {
        "expected": " twice",
        "observed": None,
        "status": "unverifiable-from-one-token-record",
    }
