"""Stable identifiers for causally localized MLP neurons and attention heads."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True, order=True, slots=True)
class ComponentRef:
    """One component inside one selected decoder layer."""

    kind: str
    layer: int
    index: int

    def __post_init__(self) -> None:
        if self.kind not in {"mlp-neuron", "attention-head"}:
            raise ValueError("component kind must be mlp-neuron or attention-head")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.layer, self.index)
        ):
            raise ValueError("component layer and index must be non-negative integers")

    @property
    def identifier(self) -> str:
        label = "N" if self.kind == "mlp-neuron" else "H"
        return f"{self.kind}:L{self.layer}:{label}{self.index}"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "layer": self.layer,
            "index": self.index,
            "identifier": self.identifier,
        }

    @classmethod
    def from_dict(cls, value: object) -> ComponentRef:
        if not isinstance(value, Mapping) or set(value) != {
            "kind",
            "layer",
            "index",
            "identifier",
        }:
            raise ValueError("component reference fields differ")
        component = cls(
            kind=value["kind"],  # type: ignore[arg-type]
            layer=value["layer"],  # type: ignore[arg-type]
            index=value["index"],  # type: ignore[arg-type]
        )
        if value["identifier"] != component.identifier:
            raise ValueError("component identifier differs from its coordinates")
        return component


__all__ = ["ComponentRef"]
