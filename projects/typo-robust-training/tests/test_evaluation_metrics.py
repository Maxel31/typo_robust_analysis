"""Paired robustness metrics and the pre-PR gate are frozen before runtime work."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from typo_robust_training.evaluation.config import load_robustness_evaluation_config
from typo_robust_training.evaluation.metrics import (
    _method_paired_metrics,
    build_evaluation_report,
)
from typo_robust_training.evaluation.records import (
    CorpusEvaluationObservation,
    EvaluationObservation,
)
from typo_robust_training.evaluation.study import load_evaluation_study_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_robustness_evaluation_config(PROJECT_ROOT / "configs/gemma4b-evaluation.yaml")
STUDY = load_evaluation_study_protocol(PROJECT_ROOT / "configs/robustness-evaluation-v1.yaml")


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
    task: str | None = None,
    mechanistic_audit: bool | None = None,
) -> EvaluationObservation:
    task_name = task or ("mmlu_pro" if index >= 10 else "gsm8k")
    audit_selected = (
        evaluation_condition == "random-2" if mechanistic_audit is None else mechanistic_audit
    )
    unseen = task_name in {"mmlu_pro", "math_500", "commonsense_qa"}
    strata = ["unseen-task"] if unseen else ["same-task"]
    if evaluation_condition in {"transposition-2", "natural-injection"}:
        strata.append("unseen-typo")
    return EvaluationObservation(
        record_id=f"{index:064x}",
        condition=condition,
        seed=seed,
        evaluation_condition=evaluation_condition,
        source=task_name,
        task=task_name,
        operation=(
            "multiple" if edit_count > 1 else "adjacent-transposition" if unseen else "deletion"
        ),
        edit_count=edit_count,
        mechanistic_audit=audit_selected,
        strata=tuple(strata),
        clean_answer="A" if clean_correct else "B",
        typo_answer="A" if typo_correct else "B",
        patched_answer=("A" if typo_correct or patch_gain > 0.0 else "B")
        if audit_selected
        else None,
        clean_correct=clean_correct,
        typo_correct=typo_correct,
        patched_correct=(typo_correct or patch_gain > 0.0) if audit_selected else None,
        target_token_ids=tuple(range(16)),
        untreated_kl_2_16=(1.0,) * 15,
        patched_kl_2_16=(1.0 - patch_gain,) * 15 if audit_selected else (),
        kl_invalid_reason=None,
        patch_invalid_reason=None if audit_selected else "not-mechanistic-audit",
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
    for index in range(100, 120):
        base_correct = index < 110
        observations.append(
            _observation(
                index,
                condition="base",
                seed=None,
                clean_correct=True,
                typo_correct=base_correct,
                patch_gain=0.50,
                evaluation_condition="natural-injection",
            )
        )
        for seed in (42, 43, 44):
            observations.append(
                _observation(
                    index,
                    condition="localized-state-distillation",
                    seed=seed,
                    clean_correct=True,
                    typo_correct=base_correct,
                    patch_gain=0.25,
                    evaluation_condition="natural-injection",
                )
            )
    return tuple(observations)


def _passing_corpus_observations() -> tuple[CorpusEvaluationObservation, ...]:
    observations: list[CorpusEvaluationObservation] = []
    records = (
        ("fineweb_edu", "clean-corpus"),
        ("dolma", "clean-corpus"),
        ("github_typo_corpus", "natural"),
    )
    for index, (source, kind) in enumerate(records, 1):
        for condition, seed in (
            ("base", None),
            ("localized-state-distillation", 42),
            ("localized-state-distillation", 43),
            ("localized-state-distillation", 44),
        ):
            natural = kind == "natural"
            observations.append(
                CorpusEvaluationObservation(
                    record_id=f"{1000 + index:064x}",
                    condition=condition,
                    seed=seed,
                    kind=kind,
                    source=source,
                    clean_nll_sum=10.0,
                    clean_nll_tokens=10,
                    typo_nll_sum=10.0 if natural else 0.0,
                    typo_nll_tokens=10 if natural else 0,
                    base_clean_kl_sum=0.0 if condition == "base" else 0.01,
                    base_clean_kl_tokens=10,
                    natural_clean_typo_kl_sum=(
                        1.0 if condition == "base" and natural else 0.9 if natural else 0.0
                    ),
                    natural_clean_typo_kl_tokens=10 if natural else 0,
                )
            )
    return tuple(observations)


def _report(
    observations: tuple[EvaluationObservation, ...] | list[EvaluationObservation],
    *,
    corpus_observations: tuple[CorpusEvaluationObservation, ...] | None = None,
) -> dict[str, object]:
    condition_ids = {observation.condition_id for observation in observations}
    corpus_rows = (
        _passing_corpus_observations() if corpus_observations is None else corpus_observations
    )
    return build_evaluation_report(
        observations,
        protocol=PROTOCOL,
        study=STUDY,
        corpus_observations=tuple(row for row in corpus_rows if row.condition_id in condition_ids),
    )


def test_report_computes_paired_transitions_strata_patch_reliance_and_gate() -> None:
    report = _report(_passing_observations())
    comparison = report["comparisons"]["localized-state-distillation:seed-42"]

    assert comparison["overall"]["n_answer"] == 20
    assert comparison["overall"]["base_typo_accuracy"] == pytest.approx(0.50)
    assert comparison["overall"]["adapter_typo_accuracy"] == pytest.approx(0.70)
    assert comparison["overall"]["typo_accuracy_gain_points"] == pytest.approx(20.0)
    assert comparison["overall"]["wrong_to_right"] == 4
    assert comparison["overall"]["right_to_wrong"] == 0
    assert comparison["overall"]["typo_transition"] == {
        "wrong_to_wrong": 6,
        "wrong_to_right": 4,
        "right_to_wrong": 0,
        "right_to_right": 10,
    }
    assert comparison["overall"]["typo_exact_mcnemar_pvalue"] == pytest.approx(0.125)
    assert comparison["overall"]["clean_harm"] == 0
    assert comparison["overall"]["clean_transition"] == {
        "wrong_to_wrong": 0,
        "wrong_to_right": 0,
        "right_to_wrong": 0,
        "right_to_right": 20,
    }
    assert comparison["overall"]["clean_exact_mcnemar_pvalue"] == pytest.approx(1.0)
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
    corpus = report["corpus_comparisons"]["localized-state-distillation:seed-42"]
    assert corpus["clean_perplexity_ratio"] == pytest.approx(1.0)
    assert corpus["clean_base_forward_kl_median"] == pytest.approx(0.001)
    assert corpus["base_natural_typo_perplexity"] == pytest.approx(2.718281828459045)
    assert corpus["adapter_natural_typo_perplexity"] == pytest.approx(2.718281828459045)


def test_method_bootstrap_is_invariant_to_observation_order() -> None:
    """The same paired cohort must produce one canonical finite bootstrap draw."""

    protocol = replace(
        PROTOCOL,
        bootstrap_replicates=400,
        bootstrap_seed=12_345,
        seed_inventory=(42,),
    )
    outcome_pairs = (
        (True, False),
        (True, False),
        (True, False),
        (False, True),
        (False, False),
        (False, True),
    )
    base = tuple(
        _observation(
            index,
            condition="base",
            seed=None,
            clean_correct=index % 2 == 0,
            typo_correct=base_typo,
            patch_gain=0.0,
            task="gsm8k",
            mechanistic_audit=False,
        )
        for index, (base_typo, _adapter_typo) in enumerate(outcome_pairs)
    )
    adapter = tuple(
        _observation(
            index,
            condition="localized-state-distillation",
            seed=42,
            clean_correct=index % 3 == 0,
            typo_correct=adapter_typo,
            patch_gain=0.0,
            task="gsm8k",
            mechanistic_audit=False,
        )
        for index, (_base_typo, adapter_typo) in enumerate(outcome_pairs)
    )

    forward = _method_paired_metrics(
        base,
        {42: adapter},
        condition="localized-state-distillation",
        protocol=protocol,
    )
    backward = _method_paired_metrics(
        tuple(reversed(base)),
        {42: tuple(reversed(adapter))},
        condition="localized-state-distillation",
        protocol=protocol,
    )

    assert forward["typo_accuracy_gain_points_mean"] == backward["typo_accuracy_gain_points_mean"]
    assert forward == backward


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
        _report(observations)

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
    report = _report(harmed)
    assert report["gate"]["checks"]["clean_macro_ci_noninferiority"] is False
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
    report = _report(observations)
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

    report = _report(observations)
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

    report = _report(observations)
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

    report = _report(observations)
    base = report["conditions"]["base"]
    comparison = report["comparisons"]["localized-state-distillation:seed-42"]

    assert base["strata"]["unseen-typo"]["n_records"] == 20
    assert base["primary_strata"]["unseen-typo"]["n_records"] == 0
    assert comparison["strata"]["unseen-typo"]["n_records"] == 20
    assert comparison["primary_strata"]["unseen-typo"]["n_records"] == 0


def _fixed_gate_observations(
    *,
    primary_records_per_task: int | tuple[int, int],
    primary_adapter_outcome,
    primary_base_clean=None,
    primary_adapter_clean=None,
) -> tuple[EvaluationObservation, ...]:
    observations: list[EvaluationObservation] = []
    task_names = ("gsm8k", "mmlu_pro")
    task_counts = (
        (primary_records_per_task,) * len(task_names)
        if isinstance(primary_records_per_task, int)
        else primary_records_per_task
    )
    for task_index, (task, task_count) in enumerate(zip(task_names, task_counts, strict=True)):
        for item_index in range(task_count):
            index = 10_000 + sum(task_counts[:task_index]) + item_index
            base_clean = (
                True if primary_base_clean is None else bool(primary_base_clean(task, item_index))
            )
            observations.append(
                _observation(
                    index,
                    condition="base",
                    seed=None,
                    clean_correct=base_clean,
                    typo_correct=False,
                    patch_gain=0.50,
                    edit_count=2,
                    task=task,
                )
            )
            for seed in (42, 43, 44):
                adapter_clean = (
                    base_clean
                    if primary_adapter_clean is None
                    else bool(primary_adapter_clean(task, item_index))
                )
                observations.append(
                    _observation(
                        index,
                        condition="localized-state-distillation",
                        seed=seed,
                        clean_correct=adapter_clean,
                        typo_correct=bool(primary_adapter_outcome(task, item_index, seed)),
                        patch_gain=0.25,
                        edit_count=2,
                        task=task,
                    )
                )
    for item_index in range(20):
        index = 20_000 + item_index
        for condition, seed in (
            ("base", None),
            ("localized-state-distillation", 42),
            ("localized-state-distillation", 43),
            ("localized-state-distillation", 44),
        ):
            observations.append(
                _observation(
                    index,
                    condition=condition,
                    seed=seed,
                    clean_correct=True,
                    typo_correct=item_index % 2 == 0,
                    patch_gain=0.25 if seed is not None else 0.50,
                    evaluation_condition="natural-injection",
                    task="mmlu_pro",
                )
            )
    return tuple(observations)


def test_exact_two_point_typo_gain_cannot_pass_without_ci_superiority() -> None:
    """Counterexample: the point threshold alone must not open the fixed gate."""

    observations = _fixed_gate_observations(
        primary_records_per_task=(20, 100),
        primary_adapter_outcome=lambda _task, item, seed: seed in {42, 43} and item == 0,
    )

    report = _report(observations)

    assert report["method_comparisons"]["localized-state-distillation"][
        "typo_accuracy_gain_points_mean"
    ] == pytest.approx(2.0)
    assert report["gate"]["checks"]["primary_typo_point_superiority"] is True
    assert report["gate"]["checks"]["primary_typo_ci_superiority"] is False
    assert report["gate"]["passed"] is False


def test_task_clean_collapse_is_blocking_even_when_macro_point_is_preserved() -> None:
    observations = _fixed_gate_observations(
        primary_records_per_task=100,
        primary_adapter_outcome=lambda _task, item, _seed: item < 5,
        primary_base_clean=lambda task, item: not (task == "mmlu_pro" and item < 3),
        primary_adapter_clean=lambda task, item: not (task == "gsm8k" and item < 3),
    )

    report = _report(observations)

    method = report["method_comparisons"]["localized-state-distillation"]
    assert method["clean_accuracy_change_points_mean"] == pytest.approx(0.0)
    assert method["task_clean_accuracy_change_points_mean"]["gsm8k"] == pytest.approx(-3.0)
    assert report["gate"]["checks"]["no_task_clean_collapse"] is False
    assert report["gate"]["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (
            lambda row: (
                replace(row, clean_nll_sum=11.0)
                if row.condition != "base" and row.kind == "clean-corpus"
                else row
            ),
            "clean_perplexity",
        ),
        (
            lambda row: (
                replace(row, base_clean_kl_sum=0.5)
                if row.condition != "base"
                and row.kind == "clean-corpus"
                and row.source == "fineweb_edu"
                else row
            ),
            "clean_forward_kl",
        ),
        (
            lambda row: (
                replace(row, natural_clean_typo_kl_sum=1.1)
                if row.condition != "base" and row.kind == "natural"
                else row
            ),
            "natural_corpus_kl_nondegradation",
        ),
    ),
)
def test_corpus_harm_is_blocking(mutation, failed_check: str) -> None:
    corpus = tuple(mutation(row) for row in _passing_corpus_observations())

    report = _report(_passing_observations(), corpus_observations=corpus)

    assert report["gate"]["checks"][failed_check] is False
    assert report["gate"]["passed"] is False


def test_one_harmful_seed_cannot_hide_behind_mean_corpus_preservation() -> None:
    corpus = tuple(
        replace(
            row,
            clean_nll_sum=10.0 * (1.0 + math.log(1.059)),
            base_clean_kl_sum=0.89,
        )
        if row.condition == "localized-state-distillation"
        and row.seed == 44
        and row.kind == "clean-corpus"
        else replace(row, base_clean_kl_sum=0.0)
        if row.condition == "localized-state-distillation" and row.kind == "clean-corpus"
        else row
        for row in _passing_corpus_observations()
    )

    report = _report(_passing_observations(), corpus_observations=corpus)
    method = report["corpus_method_comparisons"]["localized-state-distillation"]

    assert method["clean_perplexity_ratio_mean"] < 1.02
    assert method["clean_perplexity_ratio_max"] == pytest.approx(1.059)
    assert method["clean_base_forward_kl_median_mean"] < 0.03
    assert method["clean_base_forward_kl_median_max"] == pytest.approx(0.089)
    assert report["gate"]["checks"]["clean_perplexity"] is False
    assert report["gate"]["checks"]["clean_forward_kl"] is False
    assert report["gate"]["passed"] is False


def test_patch_diagnostics_only_use_the_frozen_mechanistic_audit_cohort() -> None:
    excluded_id = f"{0:064x}"
    observations = tuple(
        replace(
            row,
            mechanistic_audit=False,
            patched_answer=None,
            patched_correct=None,
            patched_kl_2_16=(),
            patch_invalid_reason="not-mechanistic-audit",
        )
        if row.record_id == excluded_id and row.evaluation_condition == "random-2"
        else row
        for row in _passing_observations()
    )

    report = _report(observations)
    overall = report["comparisons"]["localized-state-distillation:seed-42"]["overall"]

    assert overall["n_records"] == 20
    assert overall["n_patch_audit_records"] == 19
    assert overall["n_patch_readout_valid"] == 19
    assert overall["n_paired_patch_gain"] == 19
    assert overall["patch_readout_coverage_fraction"] == 1.0
    assert overall["patch_gain_coverage_fraction"] == 1.0
    assert overall["patch_gain_exclusions"] == {}


def test_non_audit_observation_rejects_accidental_patch_outputs() -> None:
    with pytest.raises(ValueError, match="non-audit observations cannot contain patch outputs"):
        replace(
            _observation(
                0,
                condition="base",
                seed=None,
                clean_correct=True,
                typo_correct=False,
                patch_gain=0.0,
                mechanistic_audit=False,
            ),
            patched_answer="A",
            patched_correct=True,
            patched_kl_2_16=(0.5,) * 15,
            patch_invalid_reason=None,
        )


def test_natural_typo_harm_is_blocking() -> None:
    observations = tuple(
        replace(row, typo_answer="B", typo_correct=False)
        if row.condition != "base"
        and row.evaluation_condition == "natural-injection"
        and row.typo_correct is True
        else row
        for row in _passing_observations()
    )

    report = _report(observations)

    assert report["gate"]["checks"]["natural_typo_point_nondegradation"] is False
    assert report["gate"]["checks"]["natural_typo_ci_nondegradation"] is False
    assert report["gate"]["passed"] is False


def test_corpus_observation_round_trip_rejects_invalid_natural_counts() -> None:
    observation = next(row for row in _passing_corpus_observations() if row.kind == "natural")
    assert CorpusEvaluationObservation.from_dict(observation.as_dict()) == observation

    payload = observation.as_dict()
    payload["natural_clean_typo_kl_tokens"] = 0
    with pytest.raises(ValueError, match="require typo likelihood and aligned KL"):
        CorpusEvaluationObservation.from_dict(payload)
