"""FFN intermediate-activation forward hooks and neuron-index conventions.

Contract (projects/quant_typo_neuron/README.md §4.1):
- "neuron n" == one FFN intermediate dimension. For SwiGLU models the
  ``gate_proj``/``up_proj`` output dim == ``down_proj`` input dim == neuron n.
- ``NeuronIndex`` = (layer, dim).
- A neuron mask is ``dict[int, list[int]]`` (layer -> dims) or a bool tensor
  of shape ``[num_layers, d_ff]``.

STATUS: stub. Implemented in feature/quant_typo_neuron/m0-ffn-hooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

NeuronIndex = tuple[int, int]  # (layer, dim)
NeuronMask = dict[int, list[int]]  # layer -> list of intermediate dims


@dataclass
class FFNActivationHook:
    """Register forward hooks on each layer's ``down_proj`` input (= FFN
    intermediate activation) and accumulate per-neuron responsibility.

    STATUS: stub.
    """

    model: Any

    def __enter__(self) -> "FFNActivationHook":
        raise NotImplementedError("implemented in feature/quant_typo_neuron/m0-ffn-hooks")

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError("implemented in feature/quant_typo_neuron/m0-ffn-hooks")


def collect_ffn_activations(model: Any, inputs: Any) -> "dict[int, torch.Tensor]":
    """Return per-layer FFN intermediate activations ``[..., d_ff]``. STATUS: stub."""
    raise NotImplementedError("implemented in feature/quant_typo_neuron/m0-ffn-hooks")
