"""PEFT LoRA insertion with fail-closed decoder-layer and module scope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from typo_robust_training.training.config import AdapterTrainingProtocol


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True, slots=True)
class TrainableParameterReport:
    trainable_parameters: int
    total_parameters: int
    parameter_names: tuple[str, ...]
    decoder_layers: tuple[int, ...]
    modules: tuple[str, ...]


def _layers(value: tuple[int, ...]) -> tuple[int, ...]:
    if (
        not value
        or tuple(sorted(set(value))) != value
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in value
        )
    ):
        raise ValueError("LoRA decoder layers must be unique and strictly increasing")
    return value


def attach_lora_adapters(
    model: Any,
    *,
    protocol: AdapterTrainingProtocol,
    decoder_layers: tuple[int, ...],
) -> Any:
    """Freeze the base model and insert LoRA only at the declared sites."""

    if not isinstance(protocol, AdapterTrainingProtocol):
        raise TypeError("adapter protocol must be AdapterTrainingProtocol")
    layers = _layers(decoder_layers)
    if not hasattr(model, "requires_grad_") or not hasattr(model, "named_parameters"):
        raise TypeError("LoRA model must expose torch module parameters")
    from peft import LoraConfig, TaskType, get_peft_model

    model.requires_grad_(False)
    if hasattr(model, "config"):
        model.config.use_cache = False
    config = LoraConfig(
        r=protocol.lora_rank,
        lora_alpha=protocol.lora_alpha,
        lora_dropout=protocol.lora_dropout,
        bias=protocol.adapter_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(protocol.lora_target_modules),
        layers_to_transform=list(layers),
        layers_pattern="layers",
    )
    adapted = get_peft_model(model, config)
    if protocol.gradient_checkpointing:
        adapted.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        enable_inputs = getattr(adapted, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    trainable_parameter_report(
        adapted,
        expected_layers=layers,
        expected_modules=protocol.lora_target_modules,
    )
    return adapted


def trainable_parameter_report(
    model: Any,
    *,
    expected_layers: tuple[int, ...],
    expected_modules: tuple[str, ...],
) -> TrainableParameterReport:
    """Prove that every and only LoRA parameters are trainable in scope."""

    layers = _layers(expected_layers)
    if not expected_modules or len(set(expected_modules)) != len(expected_modules):
        raise ValueError("expected LoRA modules must be non-empty and unique")
    names: list[str] = []
    observed_layers: set[int] = set()
    observed_modules: set[str] = set()
    trainable = total = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if not parameter.requires_grad:
            continue
        names.append(name)
        trainable += count
        if "lora_" not in name:
            raise ValueError(f"non-LoRA parameter is trainable: {name}")
        match = _LAYER.search(name)
        if match is None:
            raise ValueError(f"trainable LoRA parameter has no decoder-layer coordinate: {name}")
        observed_layers.add(int(match.group(1)))
        matches = [module for module in expected_modules if f".{module}." in name]
        if len(matches) != 1:
            raise ValueError(f"trainable LoRA parameter has an unexpected module: {name}")
        observed_modules.add(matches[0])
    if not names or trainable <= 0:
        raise ValueError("adapter exposes no trainable parameters")
    if tuple(sorted(observed_layers)) != layers:
        raise ValueError("trainable LoRA decoder layers differ from the intended scope")
    if observed_modules != set(expected_modules):
        raise ValueError("trainable LoRA modules differ from the intended scope")
    return TrainableParameterReport(
        trainable_parameters=trainable,
        total_parameters=total,
        parameter_names=tuple(sorted(names)),
        decoder_layers=tuple(sorted(observed_layers)),
        modules=tuple(sorted(observed_modules)),
    )


__all__ = [
    "TrainableParameterReport",
    "attach_lora_adapters",
    "trainable_parameter_report",
]
