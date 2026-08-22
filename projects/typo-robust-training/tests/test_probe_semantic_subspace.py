from __future__ import annotations

import numpy as np
import pytest

from typo_robust_training.probe.subspace import (
    SemanticProbeSubspace,
    derive_semantic_probe_subspace,
    validate_orthonormal_rows,
)


def _classifier(*, classes: int = 24, hidden: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    return rng.normal(size=(classes, hidden)), rng.normal(size=(classes,))


def test_common_class_weight_direction_does_not_change_semantic_subspace() -> None:
    weight, bias = _classifier()
    common = np.random.default_rng(77).normal(size=(weight.shape[1],))

    original = derive_semantic_probe_subspace(weight, bias)
    translated = derive_semantic_probe_subspace(weight + common[None, :], bias)

    assert np.allclose(original.basis.T @ original.basis, translated.basis.T @ translated.basis)
    assert np.allclose(original.projected_class_weights, translated.projected_class_weights)


def test_projection_and_logits_are_invariant_to_basis_sign_and_rotation() -> None:
    weight, bias = _classifier()
    semantic = derive_semantic_probe_subspace(weight, bias)
    values = np.random.default_rng(4).normal(size=(5, semantic.hidden_size))
    rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(16, 16)))
    rotated_basis = rotation @ semantic.basis
    rotated = SemanticProbeSubspace(
        rank=semantic.rank,
        hidden_size=semantic.hidden_size,
        class_count=semantic.class_count,
        basis=rotated_basis,
        projected_class_weights=semantic.projected_class_weights @ rotation.T,
        classifier_bias=semantic.classifier_bias,
        singular_values=semantic.singular_values,
    )
    signed_basis = semantic.basis.copy()
    signed_basis[::2] *= -1
    signed_weights = semantic.projected_class_weights.copy()
    signed_weights[:, ::2] *= -1
    signed = SemanticProbeSubspace(
        rank=semantic.rank,
        hidden_size=semantic.hidden_size,
        class_count=semantic.class_count,
        basis=signed_basis,
        projected_class_weights=signed_weights,
        classifier_bias=semantic.classifier_bias,
        singular_values=semantic.singular_values,
    )

    assert np.allclose(semantic.project(values), rotated.project(values), atol=1e-10)
    assert np.allclose(semantic.logits(values), rotated.logits(values), atol=1e-10)
    assert np.allclose(semantic.project(values), signed.project(values), atol=1e-10)
    assert np.allclose(semantic.logits(values), signed.logits(values), atol=1e-10)


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda weight: weight[:16], "cannot support"),
        (lambda weight: np.zeros_like(weight), "numerical rank"),
        (
            lambda weight: np.concatenate(([np.nan], weight.ravel()[1:])).reshape(weight.shape),
            "finite",
        ),
    ],
)
def test_invalid_classifier_rank_or_values_fail_closed(mutator, match: str) -> None:
    weight, bias = _classifier()
    changed = mutator(weight)
    changed_bias = bias[: changed.shape[0]]

    with pytest.raises(ValueError, match=match):
        derive_semantic_probe_subspace(changed, changed_bias)


def test_nonorthogonal_or_wrong_shape_basis_fails_closed() -> None:
    weight, bias = _classifier()
    semantic = derive_semantic_probe_subspace(weight, bias)
    broken = semantic.basis.copy()
    broken[1] = broken[0]

    with pytest.raises(ValueError, match="not orthonormal"):
        validate_orthonormal_rows(broken)
    with pytest.raises(ValueError, match="shapes differ"):
        SemanticProbeSubspace(
            rank=16,
            hidden_size=32,
            class_count=24,
            basis=semantic.basis[:, :-1],
            projected_class_weights=semantic.projected_class_weights,
            classifier_bias=semantic.classifier_bias,
            singular_values=semantic.singular_values,
        )
