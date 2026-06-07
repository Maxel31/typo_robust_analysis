"""Unified loader for fp16 + quantized model variants (variant registry).

Contract (README §4.2): the KV cache is FP16-fixed across all variants so that
weight quantization is the only manipulated variable. ``group_size`` /
``zero_point`` / library version are recorded on :class:`QuantVariant` for
reproducibility.

STATUS: stub. Implemented in feature/quant_typo_neuron/m1-quant-interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantVariant:
    name: str                       # e.g. "gptq_w4", "nf4", "rtn_w4", "fp16"
    method: str                     # fp16 | gptq | awq | nf4 | int8 | rtn
    bits: int | None = None         # 4 | 8 | None (fp16)
    group_size: int | None = None
    lib_version: str = ""           # recorded for reproducibility
    extra: dict[str, Any] = field(default_factory=dict)  # zero_point, etc.


def load_variant(name: str, **kwargs: Any) -> tuple[Any, QuantVariant]:
    """Load a ``(model, QuantVariant)`` pair by registry name. STATUS: stub."""
    raise NotImplementedError("implemented in feature/quant_typo_neuron/m1-quant-interface")
