from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from typo_robust_training.probe.config import ProbeProducerProtocol
from typo_robust_training.probe.producer import _fit_probe
from typo_robust_training.probe.partition import build_probe_fit_partitions


def _protocol(*, hidden_size: int, decoder_layers: int = 2) -> ProbeProducerProtocol:
    return ProbeProducerProtocol(
        model="test/model",
        model_revision="a" * 40,
        code_revision="b" * 40,
        decoder_layers=decoder_layers,
        hidden_size=hidden_size,
        input_sha256={
            name: "c" * 64
            for name in (
                "class_inventory",
                "fit_manifest",
                "selection_manifest",
                "validation_manifest",
                "protected_split_registry",
            )
        },
        records_per_class={"fit": 2, "selection": 2, "validation": 2},
        min_source_groups_per_class={"fit": 2, "selection": 2, "validation": 2},
        stratum_counts={"selection": {"x|1|same": 1}, "validation": {"x|1|same": 1}},
        probe_seeds=(42, 43),
        learning_rate=None,
        weight_decay=None,
        beta1=None,
        beta2=None,
        epsilon=None,
        epochs=None,
        batch_size=None,
        bootstrap_resamples=10_000,
        bootstrap_seed=1729,
        bootstrap_confidence=0.95,
        config_sha256="d" * 64,
        schema_version="typo-linear-probe-producer-config/v3",
        optimizer="full-batch-lbfgs-strong-wolfe-float64/v1",
        standardization="fit-only-per-layer-scalar-rms-folded/v1",
        l2_penalty="unit-prior-sum-loss/v1",
        fit_partition_rule="class-stratified-record-id-sha256-balanced-halves/v1",
        max_iterations=1000,
        max_evaluations=1250,
        history_size=100,
        gradient_tolerance=1e-7,
        change_tolerance=0.0,
        folded_logit_tolerance=1e-8,
        serialized_logit_tolerance=1e-5,
    )


def _problem() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(2718)
    sample_count, layers, hidden_size, classes = 30, 2, 40, 3
    labels = np.arange(sample_count, dtype=np.int64) % classes
    values = generator.normal(scale=0.25, size=(sample_count, layers, hidden_size))
    for layer in range(layers):
        values[np.arange(sample_count), layer, labels] += 2.0 + layer
    return values.astype(np.float32), labels


def _raw_logits(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.einsum("nld,ldc->nlc", values, weight, dtype=np.float64) + bias[None]


def test_n_less_than_d_zero_start_is_deterministic() -> None:
    values, labels = _problem()
    protocol = _protocol(hidden_size=values.shape[2])
    fits = {
        seed: _fit_probe(values, labels, class_count=3, seed=seed, protocol=protocol)
        for seed in protocol.probe_seeds
    }

    assert np.array_equal(fits[42].weights.weight, fits[43].weights.weight)
    assert np.array_equal(fits[42].weights.bias, fits[43].weights.bias)
    assert all(
        max(row.gradient_inf_norm for row in fit.diagnostics) <= 1e-7 for fit in fits.values()
    )


def test_fit_partitions_are_balanced_disjoint_and_order_invariant() -> None:
    from types import SimpleNamespace

    records = tuple(
        SimpleNamespace(record_id=f"class-{class_id}-record-{record}", class_id=class_id)
        for class_id in range(3)
        for record in range(4)
    )
    original = build_probe_fit_partitions(records, seeds=(42, 43))
    reversed_partitions = build_probe_fit_partitions(tuple(reversed(records)), seeds=(42, 43))

    assert set(original[42].record_ids).isdisjoint(original[43].record_ids)
    assert set(original[42].record_ids) | set(original[43].record_ids) == {
        record.record_id for record in records
    }
    assert original[42].class_counts == original[43].class_counts == ((0, 2), (1, 2), (2, 2))
    assert {seed: partition.identity_sha256 for seed, partition in original.items()} == {
        seed: partition.identity_sha256 for seed, partition in reversed_partitions.items()
    }


def test_fit_partitions_fail_closed_on_class_imbalance() -> None:
    from types import SimpleNamespace

    records = tuple(SimpleNamespace(record_id=f"record-{index}", class_id=0) for index in range(3))
    with pytest.raises(ValueError, match="even number"):
        build_probe_fit_partitions(records, seeds=(42, 43))


def test_fit_partitions_fail_closed_on_duplicate_membership_identity() -> None:
    from types import SimpleNamespace

    records = (
        SimpleNamespace(record_id="duplicate", class_id=0),
        SimpleNamespace(record_id="duplicate", class_id=0),
    )
    with pytest.raises(ValueError, match="record ids must be unique"):
        build_probe_fit_partitions(records, seeds=(42, 43))


def test_fit_only_scalar_rms_is_affine_layer_scale_invariant() -> None:
    values, labels = _problem()
    protocol = _protocol(hidden_size=values.shape[2])
    original = _fit_probe(values, labels, class_count=3, seed=42, protocol=protocol)
    scales = np.asarray((7.0, 0.125), dtype=np.float32)
    shifts = np.linspace(-3.0, 2.0, values.shape[2], dtype=np.float32)[None, None]
    transformed_values = values * scales[None, :, None] + shifts
    transformed = _fit_probe(
        transformed_values,
        labels,
        class_count=3,
        seed=42,
        protocol=protocol,
    )

    assert np.allclose(
        original.standardized_weight,
        transformed.standardized_weight,
        rtol=2e-7,
        atol=2e-7,
    )
    assert np.allclose(
        _raw_logits(values, original.weights.weight, original.weights.bias),
        _raw_logits(
            transformed_values,
            transformed.weights.weight,
            transformed.weights.bias,
        ),
        rtol=2e-5,
        atol=2e-5,
    )


def test_standardized_and_serialized_raw_logits_are_equivalent() -> None:
    values, labels = _problem()
    fit = _fit_probe(
        values,
        labels,
        class_count=3,
        seed=42,
        protocol=_protocol(hidden_size=values.shape[2]),
    )
    assert fit.standardized_weight is not None
    assert fit.standardized_bias is not None
    assert fit.layer_mean is not None
    assert fit.layer_scale is not None
    standardized = (values - fit.layer_mean[None]) / fit.layer_scale[None, :, None]
    expected = (
        np.einsum("nld,ldc->nlc", standardized, fit.standardized_weight)
        + fit.standardized_bias[None]
    )
    observed = _raw_logits(values, fit.weights.weight, fit.weights.bias)
    assert np.max(np.abs(expected - observed)) <= 1e-5


def test_one_lbfgs_iteration_fails_closed() -> None:
    values, labels = _problem()
    protocol = replace(
        _protocol(hidden_size=values.shape[2]),
        max_iterations=1,
        max_evaluations=2,
    )
    with pytest.raises(FloatingPointError, match="gradient gate"):
        _fit_probe(values, labels, class_count=3, seed=42, protocol=protocol)
