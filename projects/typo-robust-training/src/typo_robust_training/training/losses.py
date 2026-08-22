"""Numerically explicit losses with teacher stop-gradient boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from typo_robust_training.localization.components import ComponentRef


def _logits(value: Any, *, field: str) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 3 or int(value.shape[0]) != 1:
        raise ValueError(f"{field} must have shape [1, sequence, vocabulary]")
    if int(value.shape[1]) == 0 or int(value.shape[2]) < 2:
        raise ValueError(f"{field} must be non-empty")
    return value


def next_token_cross_entropy(logits: Any, input_ids: Any) -> Any:
    """Ordinary causal-LM loss on every non-padding token of one sequence."""

    import torch
    from torch.nn import functional as F

    values = _logits(logits, field="noisy LM logits")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or tuple(input_ids.shape) != (1, int(values.shape[1]) + 1)
    ):
        raise ValueError("noisy LM input IDs must contain one more token than logits")
    return F.cross_entropy(values[0], input_ids[0, 1:].to(device=values.device))


def answer_cross_entropy(logits: Any, *, targets: Sequence[tuple[int, int]]) -> Any:
    """Cross entropy at only the declared answer-suffix causal positions."""

    import torch
    from torch.nn import functional as F

    values = _logits(logits, field="answer logits")
    normalized = tuple(targets)
    if not normalized:
        raise ValueError("answer targets must not be empty")
    positions: list[int] = []
    token_ids: list[int] = []
    for position, token_id in normalized:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < int(values.shape[1])
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < int(values.shape[2])
        ):
            raise ValueError("answer targets contain an invalid causal coordinate")
        positions.append(position)
        token_ids.append(token_id)
    selected = values[0, positions, :]
    target = torch.tensor(token_ids, dtype=torch.long, device=values.device)
    return F.cross_entropy(selected, target)


def aligned_output_kl(
    teacher_logits: Any,
    student_logits: Any,
    *,
    logit_pairs: Sequence[tuple[int, int]],
    temperature: float,
) -> Any:
    """Mean forward KL on exact non-edited target-token alignments."""

    import torch

    teacher = _logits(teacher_logits, field="teacher logits").detach()
    student = _logits(student_logits, field="student logits")
    if int(teacher.shape[2]) != int(student.shape[2]):
        raise ValueError("teacher and student vocabularies differ")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("distillation temperature must be positive")
    pairs = tuple(logit_pairs)
    if not pairs:
        raise ValueError("output KL requires aligned logit pairs")
    clean_positions: list[int] = []
    typo_positions: list[int] = []
    for clean_position, typo_position in pairs:
        if (
            isinstance(clean_position, bool)
            or not isinstance(clean_position, int)
            or not 0 <= clean_position < int(teacher.shape[1])
            or isinstance(typo_position, bool)
            or not isinstance(typo_position, int)
            or not 0 <= typo_position < int(student.shape[1])
        ):
            raise ValueError("output KL contains an invalid causal coordinate")
        clean_positions.append(clean_position)
        typo_positions.append(typo_position)
    scale = float(temperature)
    teacher_log = torch.log_softmax(teacher[0, clean_positions, :].float() / scale, dim=-1)
    student_log = torch.log_softmax(student[0, typo_positions, :].float() / scale, dim=-1)
    return (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1).clamp_min(0.0).mean() * (
        scale**2
    )


def normalized_component_state_loss(
    teacher: Mapping[ComponentRef, Any],
    student: Mapping[ComponentRef, Any],
    *,
    weights: Mapping[ComponentRef, float],
    epsilon: float,
) -> Any:
    """Normalize per component, then apply frozen causal component weights."""

    import torch

    if not weights or set(teacher) != set(student) or set(teacher) != set(weights):
        raise ValueError("component state maps and weights must have identical non-empty keys")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("component state epsilon must be positive")
    total_weight = sum(float(weight) for weight in weights.values())
    if any(not math.isfinite(float(weight)) or float(weight) < 0.0 for weight in weights.values()):
        raise ValueError("component state weights must be finite and non-negative")
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise ValueError("component state weights must sum to one")
    losses: list[Any] = []
    for component in sorted(weights, key=lambda value: value.identifier):
        teacher_value = teacher[component]
        student_value = student[component]
        if (
            not isinstance(teacher_value, torch.Tensor)
            or not isinstance(student_value, torch.Tensor)
            or teacher_value.ndim != 2
            or teacher_value.shape != student_value.shape
            or int(teacher_value.numel()) == 0
        ):
            raise ValueError("component state tensors must have identical non-empty matrices")
        target = teacher_value.detach().to(device=student_value.device, dtype=torch.float32)
        prediction = student_value.float()
        denominator = target.square().mean() + float(epsilon)
        losses.append(
            float(weights[component]) * (prediction - target).square().mean() / denominator
        )
    return torch.stack(losses).sum()


def normalized_global_state_loss(
    teacher_hidden_states: Sequence[Any],
    student_hidden_states: Sequence[Any],
    *,
    token_pairs: Sequence[tuple[int, int]],
    decoder_layers: int,
    epsilon: float,
) -> Any:
    """Equal-layer normalized MSE over every exactly aligned non-edited token."""

    import torch

    teacher = tuple(teacher_hidden_states)
    student = tuple(student_hidden_states)
    if (
        isinstance(decoder_layers, bool)
        or not isinstance(decoder_layers, int)
        or decoder_layers <= 0
        or len(teacher) != decoder_layers + 1
        or len(student) != decoder_layers + 1
    ):
        raise ValueError("global state hidden-state inventory differs from decoder layers")
    pairs = tuple(token_pairs)
    if not pairs:
        raise ValueError("global state alignment requires aligned tokens")
    clean_positions, typo_positions = zip(*pairs, strict=True)
    losses: list[Any] = []
    for layer in range(decoder_layers):
        target_raw = teacher[layer + 1]
        prediction_raw = student[layer + 1]
        if (
            not isinstance(target_raw, torch.Tensor)
            or not isinstance(prediction_raw, torch.Tensor)
            or target_raw.ndim != 3
            or prediction_raw.ndim != 3
            or int(target_raw.shape[0]) != 1
            or int(prediction_raw.shape[0]) != 1
            or int(target_raw.shape[2]) != int(prediction_raw.shape[2])
            or max(clean_positions) >= int(target_raw.shape[1])
            or max(typo_positions) >= int(prediction_raw.shape[1])
        ):
            raise ValueError("global state tensors or aligned positions differ")
        target = target_raw[0, clean_positions, :].detach().float().to(prediction_raw.device)
        prediction = prediction_raw[0, typo_positions, :].float()
        losses.append(
            (prediction - target).square().mean() / (target.square().mean() + float(epsilon))
        )
    return torch.stack(losses).mean()


def residual_window_cosine_loss(
    teacher_hidden_states: Sequence[Any],
    student_hidden_states: Sequence[Any],
    *,
    layer_indices: Sequence[int],
    clean_positions: Sequence[int],
    typo_positions: Sequence[int],
    decoder_layers: int,
    epsilon: float,
) -> Any:
    """Bounded cosine distance at edited-word residual coordinates only."""

    import torch
    from torch.nn import functional as F

    teacher = tuple(teacher_hidden_states)
    student = tuple(student_hidden_states)
    layers = tuple(layer_indices)
    clean = tuple(clean_positions)
    typo = tuple(typo_positions)
    if (
        isinstance(decoder_layers, bool)
        or not isinstance(decoder_layers, int)
        or decoder_layers <= 0
        or len(teacher) != decoder_layers + 1
        or len(student) != decoder_layers + 1
    ):
        raise ValueError("residual state hidden-state inventory differs from decoder layers")
    if (
        not layers
        or tuple(sorted(set(layers))) != layers
        or layers[0] < 0
        or layers[-1] >= decoder_layers
    ):
        raise ValueError("residual state layers must be non-empty, ordered, and in range")
    if (
        not clean
        or len(clean) != len(typo)
        or len(set(clean)) != len(clean)
        or len(set(typo)) != len(typo)
        or min((*clean, *typo)) < 0
    ):
        raise ValueError("residual state edited positions must align one-to-one")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("residual state epsilon must be positive")
    losses: list[Any] = []
    for layer in layers:
        target_raw = teacher[layer + 1]
        prediction_raw = student[layer + 1]
        if (
            not isinstance(target_raw, torch.Tensor)
            or not isinstance(prediction_raw, torch.Tensor)
            or target_raw.ndim != 3
            or prediction_raw.ndim != 3
            or int(target_raw.shape[0]) != 1
            or int(prediction_raw.shape[0]) != 1
            or int(target_raw.shape[2]) != int(prediction_raw.shape[2])
            or max(clean) >= int(target_raw.shape[1])
            or max(typo) >= int(prediction_raw.shape[1])
        ):
            raise ValueError("residual state tensors or edited positions differ")
        target = target_raw[0, clean, :].detach().float().to(prediction_raw.device)
        prediction = prediction_raw[0, typo, :].float()
        similarity = F.cosine_similarity(target, prediction, dim=-1, eps=float(epsilon))
        losses.append((1.0 - similarity).clamp(0.0, 2.0).mean())
    return torch.stack(losses).mean()


def probe_semantic_classifier_kl(
    teacher_hidden_states: Sequence[Any],
    student_hidden_states: Sequence[Any],
    *,
    layer_index: int,
    clean_positions: Sequence[int],
    typo_positions: Sequence[int],
    decoder_layers: int,
    basis: Any,
    projected_class_weights: Any,
    classifier_bias: Any,
) -> Any:
    """Forward KL through one frozen rank-limited probe classifier.

    ``basis`` contains the rows of Q and ``projected_class_weights`` is the
    class-centred probe matrix multiplied by Q.T.  The implementation therefore
    evaluates ``z = A_s (Q h) + b`` and never constructs the dense Q.T Q
    projector.  Only the typo/student hidden state remains in the autograd graph.
    """

    import torch

    teacher = tuple(teacher_hidden_states)
    student = tuple(student_hidden_states)
    clean = tuple(clean_positions)
    typo = tuple(typo_positions)
    if (
        isinstance(decoder_layers, bool)
        or not isinstance(decoder_layers, int)
        or decoder_layers <= 0
        or len(teacher) != decoder_layers + 1
        or len(student) != decoder_layers + 1
        or isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < decoder_layers
    ):
        raise ValueError("semantic state hidden-state inventory or layer differs")
    if len(clean) != len(typo) or len(set(clean)) != len(clean) or len(set(typo)) != len(typo):
        raise ValueError("semantic state edited positions must align one-to-one")
    target_raw = teacher[layer_index + 1]
    prediction_raw = student[layer_index + 1]
    if (
        not isinstance(target_raw, torch.Tensor)
        or not isinstance(prediction_raw, torch.Tensor)
        or target_raw.ndim != 3
        or prediction_raw.ndim != 3
        or int(target_raw.shape[0]) != 1
        or int(prediction_raw.shape[0]) != 1
        or int(target_raw.shape[2]) != int(prediction_raw.shape[2])
    ):
        raise ValueError("semantic state hidden tensors differ")
    tensors = (basis, projected_class_weights, classifier_bias)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("semantic classifier tensors must be torch tensors")
    rank = int(basis.shape[0]) if basis.ndim == 2 else -1
    hidden = int(prediction_raw.shape[2])
    classes = int(projected_class_weights.shape[0]) if projected_class_weights.ndim == 2 else -1
    if (
        basis.ndim != 2
        or tuple(basis.shape) != (rank, hidden)
        or projected_class_weights.ndim != 2
        or tuple(projected_class_weights.shape) != (classes, rank)
        or classifier_bias.ndim != 1
        or tuple(classifier_bias.shape) != (classes,)
        or rank <= 0
        or classes <= rank
        or any(value.requires_grad for value in tensors)
        or any(not bool(torch.isfinite(value).all()) for value in tensors)
    ):
        raise ValueError("semantic classifier shape, values, or frozen boundary differs")
    gram = basis.detach().double() @ basis.detach().double().T
    if not torch.allclose(
        gram,
        torch.eye(rank, dtype=torch.float64, device=gram.device),
        # Evidence is derived and checked in float64, then transferred to the
        # accelerator as float32.  This tolerance admits only that cast error.
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("semantic classifier basis is not orthonormal")
    if not clean:
        return prediction_raw.float().sum() * 0.0
    if min((*clean, *typo)) < 0 or max(clean) >= int(target_raw.shape[1]) or max(typo) >= int(
        prediction_raw.shape[1]
    ):
        raise ValueError("semantic state edited position is out of range")

    device = prediction_raw.device
    q = basis.detach().to(device=device, dtype=torch.float32)
    weights = projected_class_weights.detach().to(device=device, dtype=torch.float32)
    bias = classifier_bias.detach().to(device=device, dtype=torch.float32)
    target = target_raw[0, clean, :].detach().to(device=device, dtype=torch.float32)
    prediction = prediction_raw[0, typo, :].float()
    teacher_logits = (target @ q.T) @ weights.T + bias
    student_logits = (prediction @ q.T) @ weights.T + bias
    teacher_log = torch.log_softmax(teacher_logits, dim=-1)
    student_log = torch.log_softmax(student_logits, dim=-1)
    loss = (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1).clamp_min(0.0).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("semantic classifier KL is non-finite")
    return loss


__all__ = [
    "aligned_output_kl",
    "answer_cross_entropy",
    "next_token_cross_entropy",
    "normalized_component_state_loss",
    "normalized_global_state_loss",
    "probe_semantic_classifier_kl",
    "residual_window_cosine_loss",
]
