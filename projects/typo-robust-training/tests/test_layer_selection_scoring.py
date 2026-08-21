"""Selection statistics are denominator-safe, deterministic, and fail closed."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from typo_robust_training.localization.config import load_layer_selection_config
from typo_robust_training.localization.records import LayerScan
from typo_robust_training.localization.scoring import compute_layer_selection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-layer-selection.yaml"


def _scan(
    task: str,
    index: int,
    *,
    untreated: float = 1.0,
    patched: tuple[float, ...] = (0.8, 0.4, 0.5, 0.9),
    cohort: str = "other",
) -> LayerScan:
    if cohort == "repair":
        clean_correct, typo_correct = True, False
        patched_correct: tuple[bool | None, ...] = (False, True, True, False)
    elif cohort == "harm":
        clean_correct, typo_correct = True, True
        patched_correct = (True, True, False, True)
    else:
        clean_correct, typo_correct = False, False
        patched_correct = (None, None, None, None)
    return LayerScan(
        record_id=f"{task}-{index:04d}",
        task=task,
        target_token_ids=tuple(range(16)),
        untreated_kl_2_16=(untreated,) * 15,
        patched_kl_2_16_by_layer=tuple((value,) * 15 for value in patched),
        clean_correct=clean_correct,
        typo_correct=typo_correct,
        patched_correct_by_layer=patched_correct,
        kl_invalid_reason=None,
    )


def _protocol(**changes: object):
    base = load_layer_selection_config(DEFAULT_CONFIG)
    defaults = {
        "minimum_kl_eligible_per_task": 1,
        "minimum_kl_eligible_fraction_per_task": 0.0,
        "minimum_answer_cohort_per_task": 1,
        "window_width": 2,
        "bootstrap_replicates": 100,
    }
    defaults.update(changes)
    return replace(base, **defaults)


def _cohort() -> tuple[LayerScan, ...]:
    rows: list[LayerScan] = []
    for task in ("gsm8k", "mmlu", "arc"):
        rows.extend(_scan(task, index, cohort="repair") for index in range(10))
        rows.extend(_scan(task, 10 + index, cohort="harm") for index in range(10))
        rows.extend(_scan(task, 20 + index) for index in range(40))
    return tuple(rows)


def test_selection_uses_equal_task_macro_and_contiguous_layer_scores() -> None:
    result = compute_layer_selection(_cohort(), protocol=_protocol())

    assert result.selected_window == (0, 2)
    assert [summary.window for summary in result.window_summaries] == [
        (0, 2),
        (1, 3),
        (2, 4),
    ]
    first_task = result.layer_summaries[0].tasks["gsm8k"]
    assert first_task.kl_restoration == pytest.approx(0.2)
    assert first_task.answer_restoration == 0.0
    assert first_task.harm == 0.0
    assert result.layer_summaries[1].macro_score == pytest.approx(1.1)
    assert result.layer_summaries[2].macro_score == pytest.approx(0.0)


def test_kl_restoration_is_ratio_of_cohort_means_not_mean_of_ratios() -> None:
    rows: list[LayerScan] = []
    for task in ("gsm8k", "mmlu", "arc"):
        rows.extend(
            [
                _scan(task, 0, untreated=1.0, patched=(0.0, 0.0, 0.0, 0.0), cohort="repair"),
                _scan(task, 1, untreated=3.0, patched=(3.0, 3.0, 3.0, 3.0), cohort="harm"),
            ]
        )
    result = compute_layer_selection(rows, protocol=_protocol())
    # 1 - mean(0, 3) / mean(1, 3) = .25; mean(1 - patched/untreated) would be .5.
    assert result.layer_summaries[0].tasks["gsm8k"].kl_restoration == pytest.approx(0.25)


def test_ineligible_kl_stays_in_answer_terms_but_never_gets_normalized() -> None:
    rows = list(_cohort())
    low = _scan("gsm8k", 999, untreated=1e-7, cohort="repair")
    rows.append(low)
    result = compute_layer_selection(rows, protocol=_protocol())
    counts = result.task_counts["gsm8k"]
    assert counts.total == 61
    assert counts.kl_eligible == 60
    assert counts.repair == 11
    assert result.layer_summaries[1].tasks["gsm8k"].answer_restoration == 1.0


def test_bootstrap_skips_runtime_invalid_kl_trajectories() -> None:
    rows = list(_cohort())
    rows.append(
        LayerScan(
            record_id="gsm8k-runtime-invalid",
            task="gsm8k",
            target_token_ids=(),
            untreated_kl_2_16=(),
            patched_kl_2_16_by_layer=((), (), (), ()),
            clean_correct=True,
            typo_correct=False,
            patched_correct_by_layer=(False, True, True, False),
            kl_invalid_reason="fewer-than-16-generated-tokens",
        )
    )

    result = compute_layer_selection(rows, protocol=_protocol())

    assert result.task_counts["gsm8k"].total == 61
    assert result.task_counts["gsm8k"].kl_eligible == 60
    assert result.bootstrap["replicates"] == 100


def test_selection_fails_closed_for_low_kl_or_answer_denominators() -> None:
    rows = _cohort()
    with pytest.raises(ValueError, match="KL-eligible"):
        compute_layer_selection(
            rows,
            protocol=_protocol(minimum_kl_eligible_per_task=61),
        )
    with pytest.raises(ValueError, match="repair cohort"):
        compute_layer_selection(
            rows,
            protocol=_protocol(minimum_answer_cohort_per_task=11),
        )


def test_ties_choose_smallest_start_and_bootstrap_is_reproducible() -> None:
    rows = tuple(
        replace(
            scan,
            patched_kl_2_16_by_layer=((0.5,) * 15,) * 4,
            patched_correct_by_layer=(scan.patched_correct_by_layer[0],) * 4,
        )
        for scan in _cohort()
    )
    first = compute_layer_selection(rows, protocol=_protocol())
    second = compute_layer_selection(rows, protocol=_protocol())
    assert first.selected_window == (0, 2)
    assert first.bootstrap == second.bootstrap
