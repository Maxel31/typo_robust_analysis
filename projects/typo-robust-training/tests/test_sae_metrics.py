"""SAE acceptance metrics and OOD energy score."""

from __future__ import annotations

import pytest
import torch

from typo_robust_training.sae.metrics import (
    SaeStatisticsAccumulator,
    energy_score,
    evaluate_sae_gates,
)
from typo_robust_training.sae.runner import select_l1_coefficients


def test_streaming_statistics_compute_fvu_l0_firing_rates_and_scale() -> None:
    accumulator = SaeStatisticsAccumulator(d_sae=3)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    reconstruction = torch.tensor([[0.5, 0.0], [0.0, 2.0], [1.0, 1.0]])
    features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 3.0]])
    accumulator.update(inputs, reconstruction, features)
    result = accumulator.finalize()

    expected_error = torch.tensor([0.25, 0.0, 1.0])
    expected_variance = ((inputs - inputs.mean(dim=0)) ** 2).sum().item()
    assert result.fvu == pytest.approx(expected_error.sum().item() / expected_variance)
    assert result.median_l0 == 2.0
    assert result.firing_probabilities == pytest.approx((2 / 3, 1 / 3, 2 / 3))
    assert result.reconstruction_error_median == pytest.approx(0.25)


def test_streaming_fvu_is_stable_for_large_offset_activations() -> None:
    accumulator = SaeStatisticsAccumulator(d_sae=1)
    inputs = torch.tensor(
        [
            [100_000.0, 100_000.0],
            [100_001.0, 99_999.0],
            [100_002.0, 99_998.0],
            [100_003.0, 99_997.0],
        ],
        dtype=torch.float32,
    )
    reconstruction = inputs + 0.5
    features = torch.ones((4, 1))

    accumulator.update(inputs[:2], reconstruction[:2], features[:2])
    accumulator.update(inputs[2:], reconstruction[2:], features[2:])
    result = accumulator.finalize()

    expected_error = ((inputs.double() - reconstruction.double()) ** 2).sum().item()
    expected_variance = ((inputs.double() - inputs.double().mean(dim=0)) ** 2).sum().item()
    assert result.fvu == pytest.approx(expected_error / expected_variance)


def test_energy_score_penalizes_rare_feature_more_than_common_feature() -> None:
    common = energy_score(
        reconstruction_error=torch.tensor([1.0]),
        features=torch.tensor([[1.0, 0.0]]),
        firing_probabilities=torch.tensor([0.5, 0.01]),
        reconstruction_error_scale=2.0,
    )
    rare = energy_score(
        reconstruction_error=torch.tensor([1.0]),
        features=torch.tensor([[0.0, 1.0]]),
        firing_probabilities=torch.tensor([0.5, 0.01]),
        reconstruction_error_scale=2.0,
    )
    assert rare.item() > common.item()


def test_wp2_gate_reports_each_preregistered_condition() -> None:
    result = evaluate_sae_gates(
        fvu=0.30,
        median_l0=50.0,
        dead_feature_rate=0.10,
        splice_kl_median=0.12,
        statistics_saved=True,
        fvu_max=0.35,
        median_l0_range=(30, 150),
        dead_feature_rate_max=0.20,
        splice_kl_max=0.15,
    )
    assert result.passed is True
    assert all(result.conditions.values())

    failed = evaluate_sae_gates(
        fvu=0.36,
        median_l0=50.0,
        dead_feature_rate=0.10,
        splice_kl_median=0.12,
        statistics_saved=True,
        fvu_max=0.35,
        median_l0_range=(30, 150),
        dead_feature_rate_max=0.20,
        splice_kl_max=0.15,
    )
    assert failed.passed is False
    assert failed.conditions["fvu"] is False


def test_l1_selection_requires_in_range_l0_then_uses_lowest_fvu() -> None:
    selected = select_l1_coefficients(
        {
            5: {
                0.0001: {"median_l0": 200.0, "fvu": 0.05},
                0.0003: {"median_l0": 80.0, "fvu": 0.20},
                0.001: {"median_l0": 40.0, "fvu": 0.25},
            },
            20: {
                0.0001: {"median_l0": 100.0, "fvu": 0.30},
                0.0003: {"median_l0": 90.0, "fvu": 0.20},
                0.001: {"median_l0": 10.0, "fvu": 0.01},
            },
        },
        median_l0_range=(30, 150),
    )
    assert selected == {5: 0.0003, 20: 0.0003}

    with pytest.raises(RuntimeError, match="no L1 candidate"):
        select_l1_coefficients(
            {5: {0.001: {"median_l0": 5.0, "fvu": 0.01}}},
            median_l0_range=(30, 150),
        )
