"""Custom round-to-nearest (RTN) weight quantization (group-wise).

Shared by M1 (rtn variant) and M5 (mixed-precision neuron protection).
STATUS: stub. Implemented in feature/quant_typo_neuron/m1-rtn.
"""
from __future__ import annotations

from typing import Any


def rtn_quantize(weight: Any, bits: int = 4, group_size: int = 128) -> Any:
    """Round-to-nearest quantize a weight tensor group-wise. STATUS: stub."""
    raise NotImplementedError("implemented in feature/quant_typo_neuron/m1-rtn")
