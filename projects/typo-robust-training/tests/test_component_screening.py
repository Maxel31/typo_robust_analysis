"""Approximate screens rank candidates without claiming causal selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from typo_robust_training.localization.component_config import (
    load_component_localization_config,
)
from typo_robust_training.localization.component_screening import (
    ComponentScreenMetric,
    rank_component_screen,
)
from typo_robust_training.localization.components import ComponentRef


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-component-localization.yaml"


def _protocol():
    return replace(
        load_component_localization_config(DEFAULT_CONFIG),
        mlp_intermediate_size=4,
        attention_heads=2,
        mlp_shortlist_per_layer=2,
        attention_shortlist_per_layer=2,
        causal_candidate_limits={"mlp-neuron": 2, "attention-head": 1},
    )


def test_screen_uses_equal_task_percentile_ranks_and_type_quotas() -> None:
    metrics: list[ComponentScreenMetric] = []
    for task_index, task in enumerate(("gsm8k", "mmlu", "arc")):
        for kind, count in (("mlp-neuron", 4), ("attention-head", 2)):
            for index in range(count):
                # Candidate zero remains best on every task; task scale differs by 100x.
                scale = 100.0 if task_index == 2 else 1.0
                metrics.append(
                    ComponentScreenMetric(
                        component=ComponentRef(kind, layer=0, index=index),
                        task=task,
                        records=50,
                        activation_difference=scale * (count - index),
                        gradient_attribution=scale * (count - index),
                    )
                )
    result = rank_component_screen(metrics, selected_layers=(0,), protocol=_protocol())

    assert len(result.universe) == 6
    assert [candidate.component.identifier for candidate in result.causal_candidates] == [
        "attention-head:L0:H0",
        "mlp-neuron:L0:N0",
        "mlp-neuron:L0:N1",
    ]
    assert all(candidate.screen_only for candidate in result.universe)
    assert all(not candidate.causally_validated for candidate in result.universe)


def test_screen_excludes_candidates_without_positive_attribution_on_two_tasks() -> None:
    metrics: list[ComponentScreenMetric] = []
    for task in ("gsm8k", "mmlu", "arc"):
        for index in range(4):
            metrics.append(
                ComponentScreenMetric(
                    component=ComponentRef("mlp-neuron", layer=0, index=index),
                    task=task,
                    records=50,
                    activation_difference=4.0 - index,
                    gradient_attribution=(
                        4.0 - index if index else (-1.0 if task != "gsm8k" else 4.0)
                    ),
                )
            )
        for index in range(2):
            metrics.append(
                ComponentScreenMetric(
                    component=ComponentRef("attention-head", layer=0, index=index),
                    task=task,
                    records=50,
                    activation_difference=2.0 - index,
                    gradient_attribution=2.0 - index,
                )
            )
    result = rank_component_screen(metrics, selected_layers=(0,), protocol=_protocol())
    assert "mlp-neuron:L0:N0" not in {
        candidate.component.identifier for candidate in result.causal_candidates
    }
