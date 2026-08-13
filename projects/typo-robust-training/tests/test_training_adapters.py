"""LoRA updates only frozen-config modules in the intended decoder layers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from typo_robust_training.training.adapters import (
    attach_lora_adapters,
    trainable_parameter_report,
)
from typo_robust_training.training.config import load_adapter_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ABLATION_CONFIG = PROJECT_ROOT / "configs/ablations/gemma4b-component-state-cycle1.yaml"
NOISY_CONFIG = PROJECT_ROOT / "configs/baselines/noisy-language-model.yaml"


def _tiny_model() -> Gemma3ForCausalLM:
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        sliding_window=32,
        layer_types=["full_attention"] * 3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Gemma3ForCausalLM(config)


def _small_protocol(path: Path):
    return replace(
        load_adapter_training_config(path),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        gradient_checkpointing=False,
    )


def test_targeted_lora_freezes_base_and_updates_only_component_containing_layer() -> None:
    adapted = attach_lora_adapters(
        _tiny_model(),
        protocol=_small_protocol(COMPONENT_ABLATION_CONFIG),
        decoder_layers=(1,),
    )
    report = trainable_parameter_report(
        adapted,
        expected_layers=(1,),
        expected_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    assert report.trainable_parameters > 0
    assert report.decoder_layers == (1,)
    assert report.modules == (
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    )
    assert all("lora_" in name for name in report.parameter_names)
    assert all(
        not parameter.requires_grad
        for name, parameter in adapted.named_parameters()
        if "lora_" not in name
    )

    output = adapted(input_ids=torch.tensor([[1, 5, 6, 7]]), use_cache=False)
    output.logits.float().square().mean().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in adapted.named_parameters()
        if "lora_" in name
    )


def test_all_layer_baseline_inserts_rank_matched_adapters_in_every_decoder_layer() -> None:
    adapted = attach_lora_adapters(
        _tiny_model(),
        protocol=_small_protocol(NOISY_CONFIG),
        decoder_layers=(0, 1, 2),
    )
    report = trainable_parameter_report(
        adapted,
        expected_layers=(0, 1, 2),
        expected_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    assert report.decoder_layers == (0, 1, 2)
