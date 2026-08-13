"""One teacher/student forward with condition-specific frozen losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.training.activations import forward_with_component_activations
from typo_robust_training.training.config import AdapterTrainingProtocol
from typo_robust_training.training.encoding import PairedEncoding
from typo_robust_training.training.losses import (
    aligned_output_kl,
    answer_cross_entropy,
    next_token_cross_entropy,
    normalized_component_state_loss,
    normalized_global_state_loss,
)


@dataclass(frozen=True, slots=True)
class TrainingStepOutput:
    loss: Any
    losses: Mapping[str, Any]
    student_tokens: int


def _tensor(values: tuple[int, ...], *, device: Any) -> Any:
    import torch

    return torch.tensor([values], dtype=torch.long, device=device)


def _forward(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    output_hidden_states: bool,
) -> Any:
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )


def compute_training_step(
    *,
    teacher: Any,
    student: Any,
    encoding: PairedEncoding,
    protocol: AdapterTrainingProtocol,
    component_weights: Mapping[ComponentRef, float] | None,
    attention_head_dim: int,
) -> TrainingStepOutput:
    """Compute all configured losses while keeping the teacher fully detached."""

    import torch

    from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers

    if not isinstance(encoding, PairedEncoding) or not isinstance(
        protocol, AdapterTrainingProtocol
    ):
        raise TypeError("training step requires validated encoding and protocol")
    selected_state = protocol.state_scope == "selected-components-edited-word-final-tokens"
    global_state = protocol.state_scope == "all-layers-all-aligned-tokens"
    if selected_state and not component_weights:
        raise ValueError("localized state training requires causal component weights")
    if not selected_state and component_weights:
        raise ValueError("non-localized training cannot consume component weights")
    teacher.requires_grad_(False)
    teacher.eval()
    student.train()
    try:
        device = next(parameter.device for parameter in student.parameters())
    except StopIteration as exc:
        raise ValueError("student has no parameters") from exc
    clean_ids = _tensor(encoding.clean_input_ids, device=device)
    typo_ids = _tensor(encoding.typo_input_ids, device=device)
    clean_mask = _tensor(encoding.clean_attention_mask, device=device)
    typo_mask = _tensor(encoding.typo_attention_mask, device=device)
    require_teacher = any(
        protocol.loss_weights[name] > 0.0 for name in ("output", "state", "clean")
    )
    components = tuple(component_weights or ())
    teacher_output = teacher_components = None
    if require_teacher:
        with torch.no_grad():
            if selected_state and encoding.clean_edit_positions:
                teacher_output, teacher_components = forward_with_component_activations(
                    teacher,
                    components=components,
                    positions=encoding.clean_edit_positions,
                    attention_head_dim=attention_head_dim,
                    forward=lambda: _forward(
                        teacher,
                        input_ids=clean_ids,
                        attention_mask=clean_mask,
                        output_hidden_states=global_state,
                    ),
                )
            else:
                teacher_output = _forward(
                    teacher,
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    output_hidden_states=global_state,
                )

    if selected_state and encoding.typo_edit_positions:
        student_output, student_components = forward_with_component_activations(
            student,
            components=components,
            positions=encoding.typo_edit_positions,
            attention_head_dim=attention_head_dim,
            forward=lambda: _forward(
                student,
                input_ids=typo_ids,
                attention_mask=typo_mask,
                output_hidden_states=False,
            ),
        )
    else:
        student_output = _forward(
            student,
            input_ids=typo_ids,
            attention_mask=typo_mask,
            output_hidden_states=global_state,
        )
        student_components = None
    logits = student_output.logits
    zero = logits.float().sum() * 0.0
    losses: dict[str, Any] = {name: zero for name in protocol.loss_weights}
    if protocol.loss_weights["noisy_language_model"] > 0.0:
        losses["noisy_language_model"] = next_token_cross_entropy(logits[:, :-1, :], typo_ids)
    if protocol.loss_weights["answer"] > 0.0 and encoding.answer_targets:
        losses["answer"] = answer_cross_entropy(logits, targets=encoding.answer_targets)
    if protocol.loss_weights["output"] > 0.0:
        if teacher_output is None:
            raise RuntimeError("output matching is missing the clean teacher")
        losses["output"] = aligned_output_kl(
            teacher_output.logits,
            logits,
            logit_pairs=encoding.output_logit_pairs,
            temperature=protocol.temperature,
        )
    if protocol.loss_weights["state"] > 0.0:
        if selected_state:
            if not encoding.clean_edit_positions:
                losses["state"] = zero
            else:
                if teacher_components is None or student_components is None:
                    raise RuntimeError("localized state loss is missing component activations")
                losses["state"] = normalized_component_state_loss(
                    teacher_components,
                    student_components,
                    weights=component_weights or {},
                    epsilon=protocol.epsilon,
                )
        elif global_state:
            if teacher_output is None or teacher_output.hidden_states is None:
                raise RuntimeError("global state loss is missing teacher hidden states")
            if student_output.hidden_states is None:
                raise RuntimeError("global state loss is missing student hidden states")
            losses["state"] = normalized_global_state_loss(
                teacher_output.hidden_states,
                student_output.hidden_states,
                token_pairs=encoding.global_state_token_pairs,
                decoder_layers=len(find_decoder_layers(student)),
                epsilon=protocol.epsilon,
            )
    if protocol.loss_weights["clean"] > 0.0:
        if teacher_output is None:
            raise RuntimeError("clean preservation is missing the clean teacher")
        student_clean = _forward(
            student,
            input_ids=clean_ids,
            attention_mask=clean_mask,
            output_hidden_states=False,
        )
        clean_pairs = tuple(
            (position, position) for position in range(len(encoding.clean_input_ids) - 1)
        )
        losses["clean"] = aligned_output_kl(
            teacher_output.logits,
            student_clean.logits,
            logit_pairs=clean_pairs,
            temperature=protocol.temperature,
        )
    total = sum(protocol.loss_weights[name] * losses[name] for name in protocol.loss_weights)
    if not torch.isfinite(total):
        raise FloatingPointError("training step produced a non-finite total loss")
    return TrainingStepOutput(
        loss=total,
        losses=losses,
        student_tokens=encoding.student_tokens,
    )


__all__ = ["TrainingStepOutput", "compute_training_step"]
