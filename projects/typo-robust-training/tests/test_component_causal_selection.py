"""Only cross-task causal benefits without frozen harm violations reach training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from typo_robust_training.localization.component_causal import (
    ComponentCausalObservation,
    select_training_components,
)
from typo_robust_training.localization.component_config import (
    load_component_localization_config,
)
from typo_robust_training.localization.components import ComponentRef


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-component-localization.yaml"


def _protocol():
    return replace(
        load_component_localization_config(DEFAULT_CONFIG),
        minimum_kl_eligible_per_task=1,
        minimum_kl_eligible_fraction_per_task=0.0,
        minimum_answer_cohort_per_task=1,
        bootstrap_replicates=20,
    )


def _observations() -> tuple[ComponentCausalObservation, ...]:
    good = ComponentRef("mlp-neuron", 0, 1)
    one_task = ComponentRef("attention-head", 0, 0)
    harmful = ComponentRef("mlp-neuron", 0, 2)
    rows: list[ComponentCausalObservation] = []
    for task in ("gsm8k", "mmlu", "arc"):
        for component in (good, one_task, harmful):
            rows.append(
                ComponentCausalObservation(
                    record_id=f"{task}-{component.identifier}-repair",
                    task=task,
                    component=component,
                    untreated_mean_kl=2.0,
                    patched_mean_kl=(1.0 if component == good else 1.0 if task == "gsm8k" else 2.2),
                    clean_correct=True,
                    typo_correct=False,
                    patched_correct=component != one_task or task == "gsm8k",
                )
            )
            rows.append(
                ComponentCausalObservation(
                    record_id=f"{task}-{component.identifier}-harm",
                    task=task,
                    component=component,
                    untreated_mean_kl=2.0,
                    patched_mean_kl=1.0 if component != one_task else 2.2,
                    clean_correct=True,
                    typo_correct=True,
                    patched_correct=component != harmful,
                )
            )
    return tuple(rows)


def test_causal_gate_requires_two_beneficial_tasks_and_no_harm_violation() -> None:
    candidates = (
        ComponentRef("mlp-neuron", 0, 1),
        ComponentRef("attention-head", 0, 0),
        ComponentRef("mlp-neuron", 0, 2),
    )
    result = select_training_components(
        _observations(), candidates=candidates, protocol=_protocol()
    )
    assert [selected.component.identifier for selected in result.selected] == ["mlp-neuron:L0:N1"]
    assert result.selected[0].weight == pytest.approx(1.0)
    rejected = {item.component.identifier: item.rejection_reasons for item in result.components}
    assert "beneficial_tasks_lt_2" in rejected["attention-head:L0:H0"]
    assert any(reason.startswith("harm_rate_gt_0.05") for reason in rejected["mlp-neuron:L0:N2"])


def test_causal_gate_fails_closed_when_no_component_survives() -> None:
    with pytest.raises(ValueError, match="no causally validated component"):
        select_training_components(
            tuple(row for row in _observations() if row.component.index != 1),
            candidates=(ComponentRef("attention-head", 0, 0),),
            protocol=_protocol(),
        )
