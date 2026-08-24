"""One teacher/student forward with condition-specific frozen losses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.training.activations import forward_with_component_activations
from typo_robust_training.training.config import (
    AdapterTrainingProtocol,
    is_kojima_faithful_protocol,
)
from typo_robust_training.training.encoding import (
    PairedEncoding,
    output_logit_pairs_for_scope,
)
from typo_robust_training.training.losses import (
    aligned_output_kl,
    aligned_soft_cross_entropy,
    answer_cross_entropy,
    next_token_cross_entropy,
    normalized_component_state_loss,
    normalized_global_state_loss,
    probe_semantic_classifier_kl,
    residual_window_cosine_loss,
)


def _output_matching_loss(protocol: AdapterTrainingProtocol):
    """Use public soft CE only for the named reproduction arm."""

    return (
        aligned_soft_cross_entropy
        if is_kojima_faithful_protocol(protocol)
        else aligned_output_kl
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
    state_layers: Sequence[int] | None = None,
    state_weight: float = 1.0,
    semantic_basis: Any | None = None,
    semantic_projected_class_weights: Any | None = None,
    semantic_classifier_bias: Any | None = None,
    teacher_adapter_disabled: bool = False,
) -> TrainingStepOutput:
    """Compute all configured losses while keeping the teacher fully detached."""

    import torch

    from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers

    if not isinstance(encoding, PairedEncoding) or not isinstance(
        protocol, AdapterTrainingProtocol
    ):
        raise TypeError("training step requires validated encoding and protocol")
    component_state = protocol.state_scope == "selected-components-edited-word-final-tokens"
    legacy_global_state = protocol.state_scope == "all-layers-all-aligned-tokens"
    residual_state = protocol.state_scope in {
        "causal-window-edited-word-final-tokens",
        "random-window-edited-word-final-tokens",
        "all-layers-edited-word-final-tokens",
        "probe-transition-single-layer-edited-word-final-token/v1",
    }
    semantic_state = (
        protocol.state_scope == "probe-semantic-subspace-edited-word-final-token"
    )
    resolved_state_layers = tuple(state_layers or ())
    semantic_tensors = (
        semantic_basis,
        semantic_projected_class_weights,
        semantic_classifier_bias,
    )
    if component_state and not component_weights:
        raise ValueError("localized state training requires causal component weights")
    if not component_state and component_weights:
        raise ValueError("non-localized training cannot consume component weights")
    if (residual_state or semantic_state) != bool(resolved_state_layers):
        raise ValueError("residual state training layers differ from the objective")
    if semantic_state:
        if len(resolved_state_layers) != 1 or any(value is None for value in semantic_tensors):
            raise ValueError("semantic state training requires one layer and frozen classifier")
    elif any(value is not None for value in semantic_tensors):
        raise ValueError("non-semantic training cannot consume a semantic classifier")
    if not math.isfinite(float(state_weight)) or float(state_weight) < 0.0:
        raise ValueError("state loss weight must be finite and non-negative")
    if teacher_adapter_disabled:
        if teacher is not student or not hasattr(student, "disable_adapter"):
            raise ValueError(
                "adapter-disabled teacher requires the same PEFT student model"
            )
        # The public runtime never switches the PEFT model back to train mode.
        # Gradients remain enabled for LoRA weights in eval mode (dropout is 0).
        student.eval()
    else:
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
        adapter_context = (
            student.disable_adapter() if teacher_adapter_disabled else nullcontext()
        )
        with torch.no_grad(), adapter_context:
            if component_state and encoding.clean_edit_positions:
                teacher_output, teacher_components = forward_with_component_activations(
                    teacher,
                    components=components,
                    positions=encoding.clean_edit_positions,
                    attention_head_dim=attention_head_dim,
                    forward=lambda: _forward(
                        teacher,
                        input_ids=clean_ids,
                        attention_mask=clean_mask,
                        output_hidden_states=legacy_global_state,
                    ),
                )
            else:
                teacher_output = _forward(
                    teacher,
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    output_hidden_states=legacy_global_state
                    or ((residual_state or semantic_state) and bool(encoding.clean_edit_positions)),
                )

    if component_state and encoding.typo_edit_positions:
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
            output_hidden_states=legacy_global_state
            or ((residual_state or semantic_state) and bool(encoding.typo_edit_positions)),
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
        output_loss = _output_matching_loss(protocol)
        losses["output"] = output_loss(
            teacher_output.logits,
            logits,
            logit_pairs=output_logit_pairs_for_scope(
                encoding,
                output_scope=protocol.output_scope,
            ),
            temperature=protocol.temperature,
        )
    if protocol.loss_weights["state"] > 0.0:
        if component_state:
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
        elif legacy_global_state:
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
        elif residual_state:
            if not encoding.clean_edit_positions:
                losses["state"] = zero
            else:
                if teacher_output is None or teacher_output.hidden_states is None:
                    raise RuntimeError("residual state loss is missing teacher hidden states")
                if student_output.hidden_states is None:
                    raise RuntimeError("residual state loss is missing student hidden states")
                losses["state"] = residual_window_cosine_loss(
                    teacher_output.hidden_states,
                    student_output.hidden_states,
                    layer_indices=resolved_state_layers,
                    clean_positions=encoding.clean_edit_positions,
                    typo_positions=encoding.typo_edit_positions,
                    decoder_layers=len(find_decoder_layers(student)),
                    epsilon=protocol.epsilon,
                )
        elif semantic_state:
            if not encoding.clean_edit_positions:
                losses["state"] = zero
            else:
                if teacher_output is None or teacher_output.hidden_states is None:
                    raise RuntimeError("semantic state loss is missing teacher hidden states")
                if student_output.hidden_states is None:
                    raise RuntimeError("semantic state loss is missing student hidden states")
                losses["state"] = probe_semantic_classifier_kl(
                    teacher_output.hidden_states,
                    student_output.hidden_states,
                    layer_index=resolved_state_layers[0],
                    clean_positions=encoding.clean_edit_positions,
                    typo_positions=encoding.typo_edit_positions,
                    decoder_layers=len(find_decoder_layers(student)),
                    basis=semantic_basis,
                    projected_class_weights=semantic_projected_class_weights,
                    classifier_bias=semantic_classifier_bias,
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
    total = sum(
        protocol.loss_weights[name]
        * (float(state_weight) if name == "state" else 1.0)
        * losses[name]
        for name in protocol.loss_weights
    )
    if not torch.isfinite(total):
        raise FloatingPointError("training step produced a non-finite total loss")
    return TrainingStepOutput(
        loss=total,
        losses=losses,
        student_tokens=encoding.student_tokens,
    )


__all__ = ["TrainingStepOutput", "compute_training_step"]
