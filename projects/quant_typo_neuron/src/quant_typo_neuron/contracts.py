"""Shared contracts for the quant_typo_neuron project (README §4).

Single import surface: re-exports the cross-module type conventions defined in
``typo_utils`` and defines the project-specific :class:`ItemResult` evaluation
record. Full JSONL I/O + long-form conversion live in
``quant_typo_neuron.m2.schema`` (feature/quant_typo_neuron/m2-result-schema).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from typo_utils.neurons import NeuronIndex, NeuronMask  # noqa: F401 (re-export)
from typo_utils.quant import QuantVariant  # noqa: F401 (re-export)


@dataclass
class ItemResult:
    """One evaluation record = one item x one condition (README §4.3).

    Item-level 0/1 correctness is stored raw (NOT averaged) because M3's GLMM
    uses ``item`` as a random effect.
    """

    model: str
    method: str            # fp16 | gptq | awq | nf4 | int8 | rtn
    bit: int | None
    typo_type: str         # sub_keyboard | insert | delete | transpose
    eps: float             # 1 (one char) or ratio (0.05/0.10/0.20)
    dataset: str
    seed: int
    item_id: str
    correct_clean: int     # 0/1
    correct_typo: int      # 0/1
    conf: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ItemResult":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


__all__ = ["NeuronIndex", "NeuronMask", "QuantVariant", "ItemResult"]
