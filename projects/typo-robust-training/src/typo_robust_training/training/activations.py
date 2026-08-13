"""Differentiable capture of selected post-SwiGLU neurons and attention heads."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typo_robust_training.localization.components import ComponentRef


def forward_with_component_activations(
    model: Any,
    *,
    components: Sequence[ComponentRef],
    positions: Sequence[int],
    attention_head_dim: int,
    forward: Callable[[], Any],
) -> tuple[Any, Mapping[ComponentRef, Any]]:
    """Run one forward and retain only declared component coordinates."""

    import torch

    from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers

    selected = tuple(components)
    coordinates = tuple(positions)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("component activation capture requires unique components")
    if (
        not coordinates
        or len(set(coordinates)) != len(coordinates)
        or any(
            isinstance(position, bool) or not isinstance(position, int) or position < 0
            for position in coordinates
        )
    ):
        raise ValueError("component activation positions must be unique non-negative integers")
    if (
        isinstance(attention_head_dim, bool)
        or not isinstance(attention_head_dim, int)
        or attention_head_dim <= 0
    ):
        raise ValueError("attention head dimension must be positive")
    layers = find_decoder_layers(model)
    grouped: dict[tuple[str, int], list[ComponentRef]] = defaultdict(list)
    for component in selected:
        if component.layer >= len(layers):
            raise ValueError("selected component layer is outside the model")
        grouped[(component.kind, component.layer)].append(component)
    captured: dict[tuple[str, int], Any] = {}
    handles: list[Any] = []

    def make_hook(key: tuple[str, int]):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if key in captured:
                raise RuntimeError("selected component module ran more than once")
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("selected component module input must start with a tensor")
            hidden = inputs[0]
            if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
                raise ValueError("selected component input must have shape [1, sequence, width]")
            if max(coordinates) >= int(hidden.shape[1]):
                raise ValueError("selected edited-word position is outside the sequence")
            # Keep the hook side-effect free with respect to autograd.  In
            # particular, indexing here would become part of a checkpointed
            # decoder block's original forward, while the hook is removed
            # before backward recomputation.  Retaining the module input and
            # slicing it after the forward keeps both checkpoint passes
            # structurally identical.
            captured[key] = hidden

        return hook

    try:
        for key in sorted(grouped):
            kind, layer_index = key
            layer = layers[layer_index]
            module = (
                getattr(getattr(layer, "mlp", None), "down_proj", None)
                if kind == "mlp-neuron"
                else getattr(getattr(layer, "self_attn", None), "o_proj", None)
            )
            if module is None:
                raise ValueError("selected component module is absent from the decoder layer")
            handles.append(module.register_forward_pre_hook(make_hook(key)))
        output = forward()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(grouped):
        raise RuntimeError("model forward skipped a selected component module")
    result: dict[ComponentRef, Any] = {}
    for key, components_at_site in grouped.items():
        hidden = captured[key][0, coordinates, :]
        for component in components_at_site:
            if component.kind == "mlp-neuron":
                start, stop = component.index, component.index + 1
            else:
                start = component.index * attention_head_dim
                stop = start + attention_head_dim
            if stop > int(hidden.shape[1]):
                raise ValueError("selected component index is outside the captured module width")
            result[component] = hidden[:, start:stop]
    return output, result


__all__ = ["forward_with_component_activations"]
