"""Screening statistics follow the frozen neuron/head component definitions."""

from __future__ import annotations

import pytest
import torch

from typo_robust_training.localization.component_runtime import (
    HuggingFaceComponentLocalizationRuntime,
    component_statistics,
)


def test_mlp_statistics_use_mean_absolute_difference_and_signed_first_order_effect() -> None:
    clean = torch.tensor([[3.0, 1.0], [5.0, 4.0]])
    typo = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    gradient = torch.tensor([[0.5, 1.0], [0.5, -1.0]])
    activation, attribution = component_statistics(
        kind="mlp-neuron",
        clean=clean,
        typo=typo,
        gradient=gradient,
        attention_head_dim=2,
    )
    assert activation == pytest.approx((3.0, 1.5))
    # -mean(gradient * (clean - typo)) for each post-SwiGLU coordinate.
    assert attribution == pytest.approx((-1.5, 1.5))


def test_attention_statistics_keep_complete_head_slices() -> None:
    clean = torch.tensor([[3.0, 4.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
    typo = torch.zeros_like(clean)
    gradient = torch.tensor([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, -1.0, -1.0]])
    activation, attribution = component_statistics(
        kind="attention-head",
        clean=clean,
        typo=typo,
        gradient=gradient,
        attention_head_dim=2,
    )
    assert activation == pytest.approx((2.5, (2.0 + 2.0**0.5 * 2.0) / 2.0))
    # Negative mean over positions of the within-head gradient dot donor delta.
    assert attribution == pytest.approx((-1.5, 3.0))


def test_component_statistics_reject_shape_or_kind_drift() -> None:
    values = torch.ones(2, 4)
    with pytest.raises(ValueError, match="kind"):
        component_statistics(
            kind="residual",
            clean=values,
            typo=values,
            gradient=values,
            attention_head_dim=2,
        )
    with pytest.raises(ValueError, match="shape"):
        component_statistics(
            kind="mlp-neuron",
            clean=values,
            typo=values[:, :2],
            gradient=values,
            attention_head_dim=2,
        )


def test_kl_trajectory_clamps_only_floating_point_roundoff_below_zero() -> None:
    runtime = object.__new__(HuggingFaceComponentLocalizationRuntime)
    runtime._torch = torch
    reference = torch.tensor([[0.011102902702987194, -0.016897989436984062]])
    comparison = torch.tensor([[0.011102803982794285, -0.01689789444208145]])
    trajectory = runtime._kl_trajectory(reference, comparison)
    assert trajectory.tolist() == [0.0]
