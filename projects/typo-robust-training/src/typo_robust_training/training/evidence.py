"""Hash-bound layer/component evidence consumed by Proposed training only."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from typo_robust_training.data.config import strict_loads
from typo_robust_training.localization.components import ComponentRef


@dataclass(frozen=True, slots=True)
class LocalizationEvidence:
    selected_window: tuple[int, int]
    adapter_layers: tuple[int, ...]
    component_weights: Mapping[ComponentRef, float]
    layer_selection_sha256: str
    component_selection_sha256: str


def _load(path: Path, *, artifact: str) -> tuple[Mapping[str, object], str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{artifact} is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact} is not UTF-8") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{artifact} must contain an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_localization_evidence(
    *,
    layer_selection_path: Path,
    component_selection_path: Path,
    model: str,
    model_revision: str,
    decoder_layers: int,
    mlp_intermediate_size: int,
    attention_heads: int,
) -> LocalizationEvidence:
    """Validate causal evidence and return only selected components and layers."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (decoder_layers, mlp_intermediate_size, attention_heads)
    ):
        raise ValueError("training architecture dimensions must be positive integers")
    layers, layer_sha256 = _load(layer_selection_path, artifact="layer selection")
    required_layer = {
        "schema_version": "robustness-layer-selection/v1",
        "operation": "select-distillation-layers",
        "model": model,
        "model_revision": model_revision,
    }
    if any(layers.get(field) != expected for field, expected in required_layer.items()):
        raise ValueError("layer selection identity differs from training")
    window = layers.get("selected_window")
    if not isinstance(window, Mapping) or set(window) != {"start", "stop"}:
        raise ValueError("layer selection window fields differ")
    start = _positive_int(window.get("start"), field="selected_window.start")
    stop = _positive_int(window.get("stop"), field="selected_window.stop")
    if not 0 <= start < stop <= decoder_layers:
        raise ValueError("layer selection window is outside the training model")

    components, component_sha256 = _load(component_selection_path, artifact="component selection")
    required_component = {
        "schema_version": "robustness-component-selection/v1",
        "operation": "localize-robustness-components",
        "model": model,
        "model_revision": model_revision,
    }
    if any(components.get(field) != expected for field, expected in required_component.items()):
        raise ValueError("component selection identity differs from training")
    if components.get("layer_selection_sha256") != layer_sha256:
        raise ValueError("component selection layer-selection hash differs")
    selected = components.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("component selection has no selected components")
    weights: dict[ComponentRef, float] = {}
    for index, raw in enumerate(selected):
        if not isinstance(raw, Mapping):
            raise ValueError(f"selected component {index} must be an object")
        if raw.get("causally_validated") is not True:
            raise ValueError("training components must be causally validated")
        component = ComponentRef(
            kind=raw.get("kind"),  # type: ignore[arg-type]
            layer=raw.get("layer"),  # type: ignore[arg-type]
            index=raw.get("index"),  # type: ignore[arg-type]
        )
        if raw.get("identifier") != component.identifier:
            raise ValueError("selected component identifier differs")
        if not start <= component.layer < stop:
            raise ValueError("selected component is outside the selected layer window")
        width = mlp_intermediate_size if component.kind == "mlp-neuron" else attention_heads
        if component.index >= width:
            raise ValueError("selected component index is outside the training architecture")
        if component in weights:
            raise ValueError("selected components are duplicated")
        weight = raw.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise ValueError("selected component weight must be finite and positive")
        weights[component] = float(weight)
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("selected component weights must sum to one")
    return LocalizationEvidence(
        selected_window=(start, stop),
        adapter_layers=tuple(sorted({component.layer for component in weights})),
        component_weights=MappingProxyType(weights),
        layer_selection_sha256=layer_sha256,
        component_selection_sha256=component_sha256,
    )


__all__ = ["LocalizationEvidence", "load_localization_evidence"]
