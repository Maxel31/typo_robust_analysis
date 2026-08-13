"""Loss masks and stop-gradient boundaries match the frozen method."""

from __future__ import annotations

import pytest
import torch

from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.training.losses import (
    aligned_output_kl,
    answer_cross_entropy,
    next_token_cross_entropy,
    normalized_component_state_loss,
    residual_window_cosine_loss,
)


def test_output_kl_uses_only_aligned_nonedited_targets_and_stops_teacher_gradient() -> None:
    teacher = torch.tensor(
        [[[2.0, 0.0], [8.0, -8.0], [0.0, 2.0], [1.0, 1.0]]],
        requires_grad=True,
    )
    student = torch.tensor(
        [[[0.0, 2.0], [-8.0, 8.0], [2.0, 0.0], [1.0, 1.0]]],
        requires_grad=True,
    )
    pairs = ((0, 0), (2, 2))
    first = aligned_output_kl(teacher, student, logit_pairs=pairs, temperature=1.0)
    changed_excluded = student.detach().clone()
    changed_excluded[:, 1, :] = torch.tensor([100.0, -100.0])
    second = aligned_output_kl(
        teacher,
        changed_excluded.requires_grad_(True),
        logit_pairs=pairs,
        temperature=1.0,
    )
    assert first.item() == pytest.approx(second.item())
    first.backward()
    assert teacher.grad is None
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[:, 1, :]) == 0


def test_component_state_loss_is_zero_for_identity_and_uses_only_selected_components() -> None:
    selected = ComponentRef("mlp-neuron", 0, 3)
    teacher = {selected: torch.tensor([[2.0], [4.0]], requires_grad=True)}
    identical_student = {selected: teacher[selected].detach().clone().requires_grad_(True)}
    identity = normalized_component_state_loss(
        teacher,
        identical_student,
        weights={selected: 1.0},
        epsilon=1e-6,
    )
    assert identity.item() == pytest.approx(0.0)

    changed_student = {selected: torch.tensor([[1.0], [1.0]], requires_grad=True)}
    loss = normalized_component_state_loss(
        teacher,
        changed_student,
        weights={selected: 1.0},
        epsilon=1e-6,
    )
    loss.backward()
    assert loss.item() > 0.0
    assert teacher[selected].grad is None
    assert changed_student[selected].grad is not None


def test_answer_and_noisy_lm_losses_use_only_declared_causal_targets() -> None:
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]]], requires_grad=True)
    answer = answer_cross_entropy(logits, targets=((1, 1),))
    assert answer.item() < 1e-5

    input_ids = torch.tensor([[0, 0, 1, 0]])
    language_model = next_token_cross_entropy(logits, input_ids)
    assert language_model.item() < 1e-5

    with pytest.raises(ValueError, match="answer targets"):
        answer_cross_entropy(logits, targets=())


def test_residual_window_cosine_is_bounded_position_local_and_stops_teacher_gradient() -> None:
    teacher = tuple(torch.randn(1, 4, 3, requires_grad=True) for _ in range(4))
    student = tuple(value.detach().clone().requires_grad_(True) for value in teacher)
    identity = residual_window_cosine_loss(
        teacher,
        student,
        layer_indices=(0, 1),
        clean_positions=(2,),
        typo_positions=(2,),
        decoder_layers=3,
        epsilon=1e-8,
    )
    assert identity.item() == pytest.approx(0.0, abs=1e-6)

    changed = [value.detach().clone().requires_grad_(True) for value in teacher]
    changed[1].data[0, 2, :] *= -1.0
    loss = residual_window_cosine_loss(
        teacher,
        changed,
        layer_indices=(0, 1),
        clean_positions=(2,),
        typo_positions=(2,),
        decoder_layers=3,
        epsilon=1e-8,
    )
    assert 0.0 <= loss.item() <= 2.0
    assert loss.item() == pytest.approx(1.0, abs=1e-6)
    loss.backward()
    assert all(value.grad is None for value in teacher)
    assert changed[1].grad is not None
    assert torch.count_nonzero(changed[1].grad[0, :2, :]) == 0
    assert changed[3].grad is None
