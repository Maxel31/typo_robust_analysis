"""A real tiny Transformer step preserves the teacher/base gradient boundary."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.training.adapters import attach_lora_adapters
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.encoding import PairedEncoding
from typo_robust_training.training import step as training_step_module
from typo_robust_training.training.step import compute_training_step


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/gemma4b-targeted-lora.yaml"
CYCLE2_CONFIG = PROJECT_ROOT / "configs/cycle2/gemma4b-causal-window-100step.yaml"


def _model() -> Gemma3ForCausalLM:
    return Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
            sliding_window=32,
            layer_types=["full_attention", "full_attention"],
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def _protocol(*, gradient_checkpointing: bool = False):
    return replace(
        load_adapter_training_config(CONFIG),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        gradient_checkpointing=gradient_checkpointing,
    )


def _cycle2_protocol():
    return replace(
        load_adapter_training_config(CYCLE2_CONFIG),
        decoder_layers=2,
        lora_rank=2,
        lora_alpha=1.0,
        gradient_checkpointing=False,
    )


def _encoding(*, typo: tuple[int, ...]) -> PairedEncoding:
    clean = (1, 4, 5, 6, 7)
    return PairedEncoding(
        record_id="a" * 64,
        clean_input_ids=clean,
        typo_input_ids=typo,
        clean_attention_mask=(1,) * len(clean),
        typo_attention_mask=(1,) * len(typo),
        output_logit_pairs=((0, 0), (2, 2), (3, 3)),
        global_state_token_pairs=((1, 1), (3, 3), (4, 4)),
        clean_edit_positions=(2,),
        typo_edit_positions=(2,),
        answer_targets=(),
        student_tokens=len(typo),
        is_noop=False,
    )


def test_training_step_has_no_teacher_or_base_gradients_and_nonzero_lora_gradient() -> None:
    torch.manual_seed(42)
    teacher = _model()
    student_base = copy.deepcopy(teacher)
    student = attach_lora_adapters(
        student_base,
        protocol=_protocol(),
        decoder_layers=(0,),
    )
    components = {
        ComponentRef("mlp-neuron", 0, 3): 0.6,
        ComponentRef("attention-head", 0, 1): 0.4,
    }
    result = compute_training_step(
        teacher=teacher,
        student=student,
        encoding=_encoding(typo=(1, 4, 8, 6, 7)),
        protocol=_protocol(),
        component_weights=components,
        attention_head_dim=4,
    )
    assert torch.isfinite(result.loss)
    assert result.loss.item() > 0.0
    assert result.losses["state"].item() > 0.0
    result.loss.backward()

    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in student.named_parameters()
        if "lora_" not in name
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in student.named_parameters()
        if "lora_" in name
    )


def test_clean_equals_typo_has_near_zero_component_state_loss_before_training() -> None:
    torch.manual_seed(7)
    teacher = _model()
    student = attach_lora_adapters(
        copy.deepcopy(teacher),
        protocol=_protocol(),
        decoder_layers=(0,),
    )
    components = {ComponentRef("mlp-neuron", 0, 3): 1.0}
    result = compute_training_step(
        teacher=teacher,
        student=student,
        encoding=_encoding(typo=(1, 4, 5, 6, 7)),
        protocol=_protocol(),
        component_weights=components,
        attention_head_dim=4,
    )
    assert result.losses["state"].item() < 1e-10


def test_component_capture_is_compatible_with_nonreentrant_gradient_checkpointing() -> None:
    torch.manual_seed(11)
    protocol = _protocol(gradient_checkpointing=True)
    teacher = _model()
    student = attach_lora_adapters(
        copy.deepcopy(teacher),
        protocol=protocol,
        decoder_layers=(0,),
    )
    result = compute_training_step(
        teacher=teacher,
        student=student,
        encoding=_encoding(typo=(1, 4, 8, 6, 7)),
        protocol=protocol,
        component_weights={
            ComponentRef("mlp-neuron", 0, 3): 0.6,
            ComponentRef("attention-head", 0, 1): 0.4,
        },
        attention_head_dim=4,
    )

    result.loss.backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in student.named_parameters()
        if "lora_" in name
    )


def test_cycle2_residual_cosine_uses_only_declared_layers_and_updates_only_lora() -> None:
    torch.manual_seed(17)
    protocol = _cycle2_protocol()
    teacher = _model()
    student = attach_lora_adapters(
        copy.deepcopy(teacher),
        protocol=protocol,
        decoder_layers=(0, 1),
    )
    output = compute_training_step(
        teacher=teacher,
        student=student,
        encoding=_encoding(typo=(1, 4, 8, 6, 7)),
        protocol=protocol,
        component_weights=None,
        attention_head_dim=4,
        state_layers=(0,),
        state_weight=0.05,
    )
    assert 0.0 <= output.losses["state"].item() <= 2.0
    output.loss.backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in student.named_parameters()
        if "lora_" not in name
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in student.named_parameters()
        if "lora_" in name
    )


def test_cycle2_clean_identity_has_zero_residual_state_loss() -> None:
    torch.manual_seed(19)
    protocol = _cycle2_protocol()
    teacher = _model()
    student = attach_lora_adapters(
        copy.deepcopy(teacher),
        protocol=protocol,
        decoder_layers=(0, 1),
    )
    output = compute_training_step(
        teacher=teacher,
        student=student,
        encoding=_encoding(typo=(1, 4, 5, 6, 7)),
        protocol=protocol,
        component_weights=None,
        attention_head_dim=4,
        state_layers=(0,),
        state_weight=0.05,
    )
    assert output.losses["state"].item() < 1e-7


def test_cycle2_noop_pair_does_not_request_unused_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    protocol = _cycle2_protocol()
    teacher = _model()
    student = attach_lora_adapters(
        copy.deepcopy(teacher),
        protocol=protocol,
        decoder_layers=(0, 1),
    )
    encoding = replace(
        _encoding(typo=(1, 4, 5, 6, 7)),
        clean_edit_positions=(),
        typo_edit_positions=(),
        is_noop=True,
    )
    calls: list[tuple[str, bool]] = []
    original_forward = training_step_module._forward

    def recording_forward(
        model: object,
        *,
        input_ids: object,
        attention_mask: object,
        output_hidden_states: bool,
    ) -> object:
        calls.append(("teacher" if model is teacher else "student", output_hidden_states))
        return original_forward(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    monkeypatch.setattr(training_step_module, "_forward", recording_forward)
    output = compute_training_step(
        teacher=teacher,
        student=student,
        encoding=encoding,
        protocol=protocol,
        component_weights=None,
        attention_head_dim=4,
        state_layers=(0,),
        state_weight=0.05,
    )

    assert output.losses["state"].item() == 0.0
    assert calls == [("teacher", False), ("student", False)]
