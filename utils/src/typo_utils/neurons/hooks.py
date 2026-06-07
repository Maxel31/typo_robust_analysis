"""FFN intermediate-activation forward hooks and neuron-index conventions.

Contract (projects/quant_typo_neuron/README.md §4.1):
- "neuron n" == one FFN intermediate dimension. For SwiGLU models the
  ``gate_proj``/``up_proj`` output dim == ``down_proj`` input dim == neuron n.
- ``NeuronIndex`` = (layer, dim).
- A neuron mask is ``dict[int, list[int]]`` (layer -> dims) or a bool tensor
  of shape ``[num_layers, d_ff]``.

The FFN intermediate activation a_n = SiLU(gate.x) * (up.x) is exactly the
*input* to ``down_proj``, so we capture it with a forward-pre-hook on each
layer's ``down_proj`` module. Shared by M0 (identification) and M4 (shift).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

NeuronIndex = tuple[int, int]  # (layer, dim)
NeuronMask = dict[int, list[int]]  # layer -> list of intermediate dims

# matches e.g. "model.layers.7.mlp.down_proj"
_DOWN_PROJ_RE = re.compile(r"layers\.(\d+)\..*down_proj$")


def find_ffn_down_projs(model: Any) -> "dict[int, nn.Module]":
    """Return ``{layer_index: down_proj_module}`` for every decoder layer.

    Works for Llama/Qwen/Gemma-style HF models (``model.layers.N.mlp.down_proj``)
    and any module whose qualified name ends in ``layers.<i>...down_proj``.
    """
    out: dict[int, Any] = {}
    for name, module in model.named_modules():
        m = _DOWN_PROJ_RE.search(name)
        if m is not None:
            out[int(m.group(1))] = module
    if not out:
        raise ValueError("no FFN down_proj modules found; pass layer_modules explicitly")
    return dict(sorted(out.items()))


class FFNActivationHook:
    """Context manager registering forward-pre-hooks on each ``down_proj`` to
    capture the per-layer FFN intermediate activation ``[..., d_ff]``.

    Usage::

        with FFNActivationHook(model) as h:
            model(**inputs)
        acts = h.activations  # {layer: tensor[..., d_ff]}
    """

    def __init__(
        self, model: Any, layer_modules: "dict[int, nn.Module] | None" = None
    ) -> None:
        self.model = model
        self.modules = (
            layer_modules if layer_modules is not None else find_ffn_down_projs(model)
        )
        self.activations: dict[int, Any] = {}
        self._handles: list[Any] = []

    def _make_hook(self, layer: int):
        def pre_hook(module: Any, args: tuple) -> None:
            # args[0] is the input to down_proj == the FFN intermediate activation
            self.activations[layer] = args[0].detach()

        return pre_hook

    def __enter__(self) -> "FFNActivationHook":
        self.activations = {}
        self._handles = [
            mod.register_forward_pre_hook(self._make_hook(layer))
            for layer, mod in self.modules.items()
        ]
        return self

    def __exit__(self, *exc: object) -> bool:
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def collect_ffn_activations(
    model: Any, inputs: Any, layer_modules: "dict[int, nn.Module] | None" = None
) -> "dict[int, torch.Tensor]":
    """Run one forward pass under :class:`FFNActivationHook` and return the
    per-layer FFN intermediate activations.

    ``inputs`` may be a tensor (positional) or a dict of kwargs (e.g. the output
    of a tokenizer).
    """
    import torch

    with FFNActivationHook(model, layer_modules) as hook, torch.no_grad():
        if isinstance(inputs, dict):
            model(**inputs)
        else:
            model(inputs)
    return hook.activations


def activation_at_position(
    activations: "dict[int, torch.Tensor]", position: int = -1
) -> "dict[int, torch.Tensor]":
    """Select activations at a single sequence position (e.g. the gold token).

    Input tensors are ``[batch, seq, d_ff]``; output tensors are ``[batch, d_ff]``.
    """
    return {layer: a[:, position, :] for layer, a in activations.items()}


__all__ = [
    "NeuronIndex",
    "NeuronMask",
    "FFNActivationHook",
    "find_ffn_down_projs",
    "collect_ffn_activations",
    "activation_at_position",
]
