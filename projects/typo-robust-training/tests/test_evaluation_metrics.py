"""Paired robustness metrics and the pre-PR gate are frozen before runtime work."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from typo_robust_training.evaluation.config import load_robustness_evaluation_config
from typo_robust_training.evaluation.metrics import build_evaluation_report
from typo_robust_training.evaluation.records import EvaluationObservation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_robustness_evaluation_config(PROJECT_ROOT / "configs/gemma4b-evaluation.yaml")


def _observation(
    index: int,
    *,
    condition: str,
    seed: int | None,
    clean_correct: bool,
    typo_correct: bool,
    patch_gain: float,
    edit_count: int = 1,
    evaluation_condition: str = "random-2",
) -> EvaluationObservation:
    unseen = index >= 10
    strata = ["unseen-task"] if unseen else ["same-task"]
    if evaluation_condition in {"transposition-2", "natural-injection"}:
        strata.append("unseen-typo")
    return EvaluationObservation(
        record_id=f"{index:064x}",
        condition=condition,
        seed=seed,
        evaluation_condition=evaluation_condition,
        source="mmlu_pro" if unseen else "gsm8k",
        task="mmlu_pro" if unseen else "gsm8k",
        operation=(
            "multiple" if edit_count > 1 else "adjacent-transposition" if unseen else "deletion"
        ),
        edit_count=edit_count,
        strata=tuple(strata),
        clean_answer="A" if clean_correct else "B",
        typo_answer="A" if typo_correct else "B",
        patched_answer="A" if typo_correct or patch_gain > 0.0 else "B",
        clean_correct=clean_correct,
        typo_correct=typo_correct,
        patched_correct=typo_correct or patch_gain > 0.0,
        target_token_ids=tuple(range(16)),
        untreated_kl_2_16=(1.0,) * 15,
        patched_kl_2_16=(1.0 - patch_gain,) * 15,
        kl_invalid_reason=None,
        patch_invalid_reason=None,
        clean_subtoken_counts=(1,) * edit_count,
        typo_subtoken_counts=(2,) * edit_count,
        tokenization_stratum="fragmentation-increased",
        audit={},
    )


def _passing_observations() -> tuple[EvaluationObservation, ...]:
    observations: list[EvaluationObservation] = []
    for index in range(20):
        base_correct = index < 10
        observations.append(
            _observation(
                index,
                condition="base",
                seed=None,
                clean_correct=True,
                typo_correct=base_correct,
                patch_gain=0.50,
                edit_count=2 if index == 19 else 1,
            )
        )
        for seed in (42, 43):
            observations.append(
                _observation(
                    index,
                    condition="localized-state-distillation",
                    seed=seed,
                    clean_correct=True,
                    typo_correct=base_correct or 10 <= index < 14,
                    patch_gain=0.25,
                    edit_count=2 if index == 19 else 1,
                )
            )
        observations.append(
            _observation(
                index,
                condition="localized-state-distillation",
                seed=44,
                clean_correct=True,
                typo_correct=base_correct,
                patch_gain=0.50,
                edit_count=2 if index == 19 else 1,
            )
        )
    return tuple(observations)


def test_report_computes_paired_transitions_strata_patch_reliance_and_gate() -> None:
    report = build_evaluation_report(_passing_observations(), protocol=PROTOCOL)
    comparison = report["comparisons"]["localized-state-distillation:seed-42"]

    assert comparison["overall"]["n_answer"] == 20
    assert comparison["overall"]["base_typo_accuracy"] == pytest.approx(0.50)
    assert comparison["overall"]["adapter_typo_accuracy"] == pytest.approx(0.70)
    assert comparison["overall"]["typo_accuracy_gain_points"] == pytest.approx(20.0)
    assert comparison["overall"]["wrong_to_right"] == 4
    assert comparison["overall"]["right_to_wrong"] == 0
    assert comparison["overall"]["clean_harm"] == 0
    assert comparison["primary_strata"]["unseen-task"][
        "typo_accuracy_gain_points"
    ] == pytest.approx(40.0)
    assert comparison["overall"]["base_mean_patch_gain"] == pytest.approx(0.50)
    assert comparison["overall"]["adapter_mean_patch_gain"] == pytest.approx(0.25)
    assert comparison["overall"]["patch_gain_reduction_fraction"] == pytest.approx(0.50)
    assert comparison["overall"]["typo_accuracy_gain_ci95_points"][0] <= 20.0
    assert comparison["overall"]["typo_accuracy_gain_ci95_points"][1] >= 20.0
    assert report["gate"]["directional_seeds"] == [42, 43]
    assert report["gate"]["passed"] is True

    seed_checks = report["gate"]["seed_checks"]
    assert all(seed_checks[str(seed)]["passed"] for seed in (42, 43))
    assert seed_checks["44"]["passed"] is False
    assert "fragmentation-increased" in report["conditions"]["base"]["tokenization_strata"]
    assert report["conditions"]["base"]["edit_counts"]["1"]["n_records"] == 19
    assert report["conditions"]["base"]["edit_counts"]["2"]["n_records"] == 1
    method = report["method_comparisons"]["localized-state-distillation"]
    assert method["n_seeds"] == 3
    assert method["seed_inventory_complete"] is True
    assert method["typo_accuracy_gain_points_mean"] == pytest.approx(40.0 / 3.0)
    assert method["typo_accuracy_gain_points_sd"] > 0.0
    assert method["typo_accuracy_gain_ci95_points"][0] <= 40.0 / 3.0
    assert method["typo_accuracy_gain_ci95_points"][1] >= 40.0 / 3.0


def test_report_rejects_unpaired_conditions_and_detects_clean_harm() -> None:
    observations = list(_passing_observations())
    observations = [
        observation
        for observation in observations
        if not (
            observation.condition == "localized-state-distillation"
            and observation.seed == 42
            and observation.record_id == f"{19:064x}"
        )
    ]
    with pytest.raises(ValueError, match="paired record IDs"):
        build_evaluation_report(observations, protocol=PROTOCOL)

    observations = list(_passing_observations())
    harmed: list[EvaluationObservation] = []
    for observation in observations:
        if (
            observation.condition == "localized-state-distillation"
            and observation.seed in {42, 43}
            and observation.record_id == f"{0:064x}"
        ):
            observation = _observation(
                0,
                condition=observation.condition,
                seed=observation.seed,
                clean_correct=False,
                typo_correct=observation.typo_correct is True,
                patch_gain=0.25,
            )
        harmed.append(observation)
    report = build_evaluation_report(harmed, protocol=PROTOCOL)
    assert report["gate"]["seed_checks"]["42"]["checks"]["clean_preservation"] is False
    assert report["gate"]["passed"] is False


def test_observation_round_trip_rejects_nonfinite_or_misaligned_kl() -> None:
    observation = _observation(
        1,
        condition="base",
        seed=None,
        clean_correct=True,
        typo_correct=False,
        patch_gain=0.5,
    )
    assert EvaluationObservation.from_dict(observation.as_dict()) == observation

    payload = observation.as_dict()
    payload["untreated_kl_2_16"] = [1.0] * 14
    with pytest.raises(ValueError, match="fifteen"):
        EvaluationObservation.from_dict(payload)

    payload = observation.as_dict()
    payload["patched_kl_2_16"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        EvaluationObservation.from_dict(payload)


def test_random_window_control_is_a_valid_observation_condition() -> None:
    observation = _observation(
        1,
        condition="random-window-state-distillation",
        seed=42,
        clean_correct=True,
        typo_correct=False,
        patch_gain=0.5,
    )

    assert EvaluationObservation.from_dict(observation.as_dict()) == observation


def test_patch_readout_coverage_is_diagnostic_and_does_not_block_gate() -> None:
    observations = list(_passing_observations())
    for index, observation in enumerate(observations):
        if (
            observation.condition == "localized-state-distillation"
            and observation.seed == 42
            and observation.record_id == f"{0:064x}"
        ):
            observations[index] = replace(
                observation,
                patched_answer=None,
                patched_correct=None,
                patched_kl_2_16=(),
                patch_invalid_reason="insufficient-readout",
                audit=dict(observation.audit),
            )
            break
    report = build_evaluation_report(observations, protocol=PROTOCOL)
    seed = report["gate"]["seed_checks"]["42"]
    overall = report["comparisons"]["localized-state-distillation:seed-42"]["overall"]
    assert overall["patch_readout_coverage_fraction"] < 1.0
    assert "patch_readout_coverage" not in seed["checks"]
    assert "patch_reliance_reduction" not in seed["checks"]
    assert seed["passed"] is True
    assert report["gate"]["passed"] is True


def test_near_zero_untreated_kl_is_logged_without_failing_readout_coverage() -> None:
    observations = list(_passing_observations())
    record_id = f"{0:064x}"
    for index, observation in enumerate(observations):
        if observation.record_id == record_id and (
            observation.condition == "base"
            or (observation.condition == "localized-state-distillation" and observation.seed == 42)
        ):
            observations[index] = replace(
                observation,
                untreated_kl_2_16=(1e-8,) * 15,
                patched_kl_2_16=(1e-8,) * 15,
                audit=dict(observation.audit),
            )

    report = build_evaluation_report(observations, protocol=PROTOCOL)
    overall = report["comparisons"]["localized-state-distillation:seed-42"]["overall"]
    seed = report["gate"]["seed_checks"]["42"]

    assert overall["n_patch_readout_valid"] == overall["n_records"]
    assert overall["n_paired_patch_gain"] == overall["n_records"] - 1
    assert overall["patch_gain_exclusions"] == {
        "adapter:near-zero-untreated-kl": 1,
        "base:near-zero-untreated-kl": 1,
    }
    assert seed["passed"] is True


def test_confirmatory_endpoint_excludes_secondary_typo_conditions() -> None:
    observations: list[EvaluationObservation] = []
    for index in range(10):
        observations.extend(
            (
                _observation(
                    index,
                    condition="base",
                    seed=None,
                    clean_correct=True,
                    typo_correct=True,
                    patch_gain=0.5,
                ),
                _observation(
                    index,
                    condition="localized-state-distillation",
                    seed=42,
                    clean_correct=True,
                    typo_correct=index != 0,
                    patch_gain=0.25,
                ),
            )
        )
    for index in range(100, 120):
        observations.extend(
            (
                _observation(
                    index,
                    condition="base",
                    seed=None,
                    clean_correct=True,
                    typo_correct=False,
                    patch_gain=0.5,
                    evaluation_condition="random-4",
                    edit_count=4,
                ),
                _observation(
                    index,
                    condition="localized-state-distillation",
                    seed=42,
                    clean_correct=True,
                    typo_correct=True,
                    patch_gain=0.25,
                    evaluation_condition="random-4",
                    edit_count=4,
                ),
            )
        )

    report = build_evaluation_report(observations, protocol=PROTOCOL)
    comparison = report["comparisons"]["localized-state-distillation:seed-42"]

    assert comparison["primary_condition"] == "random-2"
    assert comparison["overall"]["n_records"] == 10
    assert comparison["overall"]["typo_accuracy_gain_points"] == pytest.approx(-10.0)
    assert comparison["all_conditions"]["typo_accuracy_gain_points"] == pytest.approx(1900.0 / 30.0)
    assert comparison["evaluation_conditions"]["random-4"][
        "typo_accuracy_gain_points"
    ] == pytest.approx(100.0)


def test_secondary_held_out_conditions_populate_descriptive_not_primary_strata() -> None:
    observations: list[EvaluationObservation] = []
    for index in range(20):
        for condition, seed in (("base", None), ("localized-state-distillation", 42)):
            observations.append(
                _observation(
                    index,
                    condition=condition,
                    seed=seed,
                    clean_correct=True,
                    typo_correct=index % 2 == 0,
                    patch_gain=0.5,
                )
            )
    for index in range(100, 120):
        for condition, seed in (("base", None), ("localized-state-distillation", 42)):
            observations.append(
                _observation(
                    index,
                    condition=condition,
                    seed=seed,
                    clean_correct=True,
                    typo_correct=index % 2 == 0,
                    patch_gain=0.5,
                    evaluation_condition="transposition-2",
                    edit_count=2,
                )
            )

    report = build_evaluation_report(observations, protocol=PROTOCOL)
    base = report["conditions"]["base"]
    comparison = report["comparisons"]["localized-state-distillation:seed-42"]

    assert base["strata"]["unseen-typo"]["n_records"] == 20
    assert base["primary_strata"]["unseen-typo"]["n_records"] == 0
    assert comparison["strata"]["unseen-typo"]["n_records"] == 20
    assert comparison["primary_strata"]["unseen-typo"]["n_records"] == 0
