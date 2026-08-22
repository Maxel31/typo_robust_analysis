"""Falsification tests for frozen probe-semantic distillation."""

from __future__ import annotations

import pytest

from typo_robust_training.training.losses import probe_semantic_classifier_kl


torch = pytest.importorskip("torch")


def _inputs() -> tuple[list[object], list[object], object, object, object]:
    generator = torch.Generator().manual_seed(41)
    teacher = [torch.randn(1, 3, 20, generator=generator) for _ in range(4)]
    student = [value.clone().requires_grad_(True) for value in teacher]
    raw = torch.randn(20, 4, generator=generator)
    q = torch.linalg.qr(raw, mode="reduced").Q.T.detach()
    weights = torch.randn(19, 4, generator=generator)
    bias = torch.randn(19, generator=generator)
    return teacher, student, q, weights, bias


def _loss(teacher, student, q, weights, bias):
    return probe_semantic_classifier_kl(
        teacher,
        student,
        layer_index=2,
        clean_positions=(1,),
        typo_positions=(1,),
        decoder_layers=3,
        basis=q,
        projected_class_weights=weights,
        classifier_bias=bias,
    )


def test_probe_semantic_loss_only_updates_student() -> None:
    teacher, student, q, weights, bias = _inputs()
    student[3] = (student[3] + 0.2).detach().requires_grad_(True)
    value = _loss(teacher, student, q, weights, bias)
    value.backward()
    assert student[3].grad is not None
    assert all(item.grad is None for item in teacher)
    assert q.grad is None and weights.grad is None and bias.grad is None


def test_probe_semantic_loss_is_basis_rotation_and_sign_invariant() -> None:
    teacher, student, q, weights, bias = _inputs()
    rotation = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(5))).Q
    expected = _loss(teacher, student, q, weights, bias)
    observed = _loss(teacher, student, rotation @ q, weights @ rotation.T, bias)
    assert torch.allclose(expected, observed, atol=1e-6, rtol=1e-6)


def test_probe_semantic_loss_noop_is_exact_zero() -> None:
    teacher, student, q, weights, bias = _inputs()
    value = probe_semantic_classifier_kl(
        teacher,
        student,
        layer_index=2,
        clean_positions=(),
        typo_positions=(),
        decoder_layers=3,
        basis=q,
        projected_class_weights=weights,
        classifier_bias=bias,
    )
    assert value.item() == 0.0
    value.backward()
    assert student[3].grad is not None


def test_probe_semantic_loss_is_finite_for_extreme_logits() -> None:
    teacher, student, q, weights, bias = _inputs()
    student[3] = (student[3] * 1e10).detach().requires_grad_(True)
    value = _loss(teacher, student, q, weights * 1e10, bias * 1e10)
    assert torch.isfinite(value)


@pytest.mark.parametrize("failure", ["rank", "nan", "nonorthogonal", "shape", "trainable"])
def test_probe_semantic_loss_rejects_invalid_classifier(failure: str) -> None:
    teacher, student, q, weights, bias = _inputs()
    if failure == "rank":
        q = q[:2]
    elif failure == "nan":
        q = q.clone()
        q[0, 0] = float("nan")
    elif failure == "nonorthogonal":
        q = q.clone()
        q[1] = q[0]
    elif failure == "shape":
        weights = weights[:, :3]
    elif failure == "trainable":
        q = q.requires_grad_(True)
    with pytest.raises(ValueError):
        _loss(teacher, student, q, weights, bias)
