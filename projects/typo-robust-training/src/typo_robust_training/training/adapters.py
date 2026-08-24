"""PEFT LoRA insertion with fail-closed decoder-layer and module scope."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from typo_robust_training.training.config import AdapterTrainingProtocol


_DECODER_LAYER = re.compile(
    r"(?:^|\.)(?:language_model|model)\.layers\.(\d+)(?:\.|$)"
)
_VISION_SCOPE = re.compile(
    r"(?:^|\.)(?:vision|vision_model|vision_tower)(?:\.|$)"
)
_GLOBAL_TARGET_MODULES = ("embed_tokens", "lm_head")


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


def _modules(value: tuple[str, ...]) -> tuple[str, ...]:
    if (
        not value
        or len(set(value)) != len(value)
        or any(not isinstance(module, str) or not module or "." in module for module in value)
    ):
        raise ValueError("expected LoRA modules must be non-empty, unique leaf names")
    return value


def _partition_target_modules(
    value: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    modules = _modules(value)
    global_modules = tuple(module for module in modules if module in _GLOBAL_TARGET_MODULES)
    decoder_modules = tuple(module for module in modules if module not in _GLOBAL_TARGET_MODULES)
    if global_modules and set(global_modules) != set(_GLOBAL_TARGET_MODULES):
        raise ValueError("global LoRA scope must include both embed_tokens and lm_head")
    if not decoder_modules:
        raise ValueError("LoRA scope must include decoder projection modules")
    return decoder_modules, global_modules


def _decoder_target_module_names(
    model: Any,
    *,
    decoder_layers: tuple[int, ...],
    target_modules: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve exact decoder projection paths before PEFT mutates the model."""

    layers = _layers(decoder_layers)
    modules = _modules(target_modules)
    expected = {(layer, module) for layer in layers for module in modules}
    observed: dict[tuple[int, str], str] = {}
    for name, _module in model.named_modules():
        match = _DECODER_LAYER.search(name)
        leaf = name.rsplit(".", 1)[-1]
        if match is None or leaf not in modules:
            continue
        coordinate = (int(match.group(1)), leaf)
        if coordinate not in expected:
            continue
        if coordinate in observed:
            raise ValueError(f"multiple decoder modules resolve to {coordinate}: {name}")
        observed[coordinate] = name
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise ValueError(f"decoder LoRA target modules are missing: {missing}")
    return tuple(observed[coordinate] for coordinate in sorted(observed))


def _global_target_module_names(
    model: Any,
    *,
    target_modules: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the one text embedding and one LM head without crossing into vision."""

    modules = _modules(target_modules)
    observed: dict[str, str] = {}
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in modules or _DECODER_LAYER.search(name) is not None:
            continue
        if _VISION_SCOPE.search(name) is not None:
            raise ValueError(f"vision module collides with the global LoRA scope: {name}")
        if leaf in observed:
            raise ValueError(f"multiple global LoRA modules resolve to {leaf}: {name}")
        observed[leaf] = name
    if set(observed) != set(modules):
        missing = sorted(set(modules) - set(observed))
        raise ValueError(f"global LoRA target modules are missing: {missing}")
    return tuple(observed[module] for module in modules)


def attach_lora_adapters(
    model: Any,
    *,
    protocol: AdapterTrainingProtocol,
    decoder_layers: tuple[int, ...],
    initialization_seed: int | None = None,
) -> Any:
    """Freeze the base model and insert LoRA only at the declared sites."""

    if not isinstance(protocol, AdapterTrainingProtocol):
        raise TypeError("adapter protocol must be AdapterTrainingProtocol")
    layers = _layers(decoder_layers)
    if not hasattr(model, "requires_grad_") or not hasattr(model, "named_parameters"):
        raise TypeError("LoRA model must expose torch module parameters")
    from peft import LoraConfig, TaskType, get_peft_model

    modules = _modules(protocol.lora_target_modules)
    layer_keyed_initialization = (
        protocol.adapter_initialization_policy == "sha256-layer-keyed-kaiming-a-zero-b/v1"
    )
    if layer_keyed_initialization and (
        isinstance(initialization_seed, bool)
        or not isinstance(initialization_seed, int)
        or initialization_seed < 0
    ):
        raise ValueError("layer-keyed LoRA initialization requires a non-negative seed")
    if not layer_keyed_initialization and (
        protocol.adapter_initialization_policy != "peft-default/v1"
    ):
        raise ValueError("LoRA initialization policy is unsupported")
    decoder_modules, global_modules = _partition_target_modules(modules)
    decoder_target_names = _decoder_target_module_names(
        model,
        decoder_layers=layers,
        target_modules=decoder_modules,
    )
    global_target_names = (
        _global_target_module_names(model, target_modules=global_modules)
        if global_modules
        else ()
    )
    target_names = (*decoder_target_names, *global_target_names)
    model.requires_grad_(False)
    if hasattr(model, "config"):
        model.config.use_cache = False
    config = LoraConfig(
        r=protocol.lora_rank,
        lora_alpha=protocol.lora_alpha,
        lora_dropout=protocol.lora_dropout,
        bias=protocol.adapter_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(target_names),
    )
    # PEFT initializes every inserted adapter with the process-global RNG.
    # Different factorial scopes insert different numbers of modules, so merely
    # overwriting their final values with coordinate-keyed tensors would still
    # leave a different RNG stream for the subsequent training computation.
    # Preserve and restore both CPU and already-initialized CUDA RNG states for
    # the frozen layer-keyed policy.  Legacy PEFT-default runs retain their
    # historical RNG semantics.
    import torch

    cpu_rng_state = torch.get_rng_state() if layer_keyed_initialization else None
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if layer_keyed_initialization and torch.cuda.is_initialized()
        else None
    )
    try:
        adapted = get_peft_model(model, config)
        if layer_keyed_initialization:
            assert isinstance(initialization_seed, int)  # validated above
            initialize_layer_keyed_lora(
                adapted,
                seed=initialization_seed,
                expected_layers=layers,
                expected_modules=decoder_modules,
            )
    finally:
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
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
        expected_modules=modules,
    )
    return adapted


def _coordinate_seed(*, seed: int, layer: int, module: str, role: str) -> int:
    payload = (
        f"typo-robust-lora-init/v1\0seed={seed}\0layer={layer}\0module={module}\0role={role}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def initialize_layer_keyed_lora(
    model: Any,
    *,
    seed: int,
    expected_layers: tuple[int, ...],
    expected_modules: tuple[str, ...],
) -> None:
    """Initialize every LoRA coordinate independently of the arm's layer set.

    A shared ``(seed, layer, module)`` coordinate is therefore bit-identical in
    an all-layer, suffix, or random-freeze arm.  This removes PEFT insertion
    order and global RNG consumption as factorial confounds.
    """

    import torch

    layers = _layers(expected_layers)
    modules = _modules(expected_modules)
    expected = {
        (layer, module, role) for layer in layers for module in modules for role in ("A", "B")
    }
    observed: set[tuple[int, str, str]] = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or "lora_" not in name:
                continue
            layer_match = _DECODER_LAYER.search(name)
            module_matches = [module for module in modules if f".{module}." in name]
            role = "A" if ".lora_A." in name else "B" if ".lora_B." in name else None
            if layer_match is None or len(module_matches) != 1 or role is None:
                raise ValueError(f"cannot resolve layer-keyed LoRA coordinate: {name}")
            coordinate = (int(layer_match.group(1)), module_matches[0], role)
            if coordinate not in expected or coordinate in observed:
                raise ValueError(f"unexpected or duplicate layer-keyed LoRA coordinate: {name}")
            observed.add(coordinate)
            if role == "B":
                parameter.zero_()
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _coordinate_seed(
                    seed=seed,
                    layer=coordinate[0],
                    module=coordinate[1],
                    role=role,
                )
            )
            value = torch.empty(tuple(parameter.shape), dtype=torch.float32, device="cpu")
            torch.nn.init.kaiming_uniform_(
                value,
                a=math.sqrt(5),
                generator=generator,
            )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    if observed != expected:
        raise ValueError(f"layer-keyed LoRA coordinates are missing: {sorted(expected - observed)}")


def trainable_parameter_report(
    model: Any,
    *,
    expected_layers: tuple[int, ...],
    expected_modules: tuple[str, ...],
) -> TrainableParameterReport:
    """Prove that every and only LoRA parameters are trainable in scope."""

    layers = _layers(expected_layers)
    modules = _modules(expected_modules)
    decoder_modules, global_modules = _partition_target_modules(modules)
    names: list[str] = []
    observed_layers: set[int] = set()
    observed_modules: set[str] = set()
    observed_coordinates: set[tuple[int, str]] = set()
    observed_global_modules: set[str] = set()
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
        match = _DECODER_LAYER.search(name)
        matches = [module for module in modules if f".{module}." in name]
        if len(matches) != 1:
            raise ValueError(f"trainable LoRA parameter has an unexpected module: {name}")
        module = matches[0]
        observed_modules.add(module)
        if match is None:
            if module not in global_modules or _VISION_SCOPE.search(name) is not None:
                raise ValueError(
                    f"trainable LoRA parameter has no allowed decoder/global coordinate: {name}"
                )
            observed_global_modules.add(module)
        else:
            if module not in decoder_modules:
                raise ValueError(f"global LoRA module appeared inside a decoder layer: {name}")
            layer = int(match.group(1))
            observed_layers.add(layer)
            observed_coordinates.add((layer, module))
    if not names or trainable <= 0:
        raise ValueError("adapter exposes no trainable parameters")
    if tuple(sorted(observed_layers)) != layers:
        raise ValueError("trainable LoRA decoder layers differ from the intended scope")
    if observed_modules != set(modules):
        raise ValueError("trainable LoRA modules differ from the intended scope")
    expected_coordinates = {
        (layer, module) for layer in layers for module in decoder_modules
    }
    if observed_coordinates != expected_coordinates:
        raise ValueError("trainable LoRA decoder coordinates differ from the intended scope")
    if observed_global_modules != set(global_modules):
        raise ValueError("trainable global LoRA modules differ from the intended scope")
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
    "initialize_layer_keyed_lora",
    "trainable_parameter_report",
]
