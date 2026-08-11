"""Paired robustness metrics and the pre-PR gate are frozen before runtime work."""

from __future__ import annotations

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
) -> EvaluationObservation:
    unseen = index >= 10
    return EvaluationObservation(
        record_id=f"{index:064x}",
        condition=condition,
        seed=seed,
        source="mmlu_pro" if unseen else "gsm8k",
        task="mmlu_pro" if unseen else "gsm8k",
        operation="adjacent-transposition" if unseen else "deletion",
        strata=("unseen-task", "unseen-typo") if unseen else ("same-task",),
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
        clean_subtoken_counts=(1,),
        typo_subtoken_counts=(2,),
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
    assert comparison["strata"]["unseen-task"]["typo_accuracy_gain_points"] == pytest.approx(
        40.0
    )
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
