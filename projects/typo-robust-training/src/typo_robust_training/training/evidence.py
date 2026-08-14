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


@dataclass(frozen=True, slots=True)
class ResidualStateEvidence:
    selected_window: tuple[int, int]
    state_layers: tuple[int, ...]
    policy: str
    layer_selection_sha256: str
    validation_sha256: str | None
    evidence_sha256: str


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


def load_residual_state_evidence(
    *,
    layer_selection_path: Path,
    window_validation_path: Path | None = None,
    model: str,
    model_revision: str,
    decoder_layers: int,
    policy: str,
) -> ResidualStateEvidence:
    """Resolve a causal or same-width random residual window without outcomes."""

    if (
        isinstance(decoder_layers, bool)
        or not isinstance(decoder_layers, int)
        or decoder_layers < 2
    ):
        raise ValueError("residual evidence decoder_layers must be at least two")
    layers, layer_sha256 = _load(layer_selection_path, artifact="layer selection")
    schema = layers.get("schema_version")
    if schema == "robustness-layer-selection/v1":
        required = {
            "operation": "select-distillation-layers",
            "model": model,
            "model_revision": model_revision,
        }
    elif schema == "robustness-joint-window-selection/v1":
        required = {
            "operation": "select-generic-joint-patch-window",
            "model": model,
            "model_revision": model_revision,
        }
    else:
        raise ValueError("layer selection identity differs from residual training")
    if any(layers.get(field) != expected for field, expected in required.items()):
        raise ValueError("layer selection identity differs from residual training")
    window = layers.get("selected_window")
    expected_window_fields = (
        {"start", "stop"}
        if schema == "robustness-layer-selection/v1"
        else {"start", "stop", "median_pairwise_restoration", "confidence_interval"}
    )
    if not isinstance(window, Mapping) or set(window) != expected_window_fields:
        raise ValueError("layer selection window fields differ")
    start = _positive_int(window.get("start"), field="selected_window.start")
    stop = _positive_int(window.get("stop"), field="selected_window.stop")
    if not 0 <= start < stop <= decoder_layers:
        raise ValueError("layer selection window is outside the residual model")
    width = stop - start
    validation_sha256: str | None = None
    if schema == "robustness-joint-window-selection/v1":
        if window_validation_path is None:
            raise ValueError("confirmatory layer selection requires independent validation")
        validation, validation_sha256 = _load(
            window_validation_path,
            artifact="window validation",
        )
        validation_required = {
            "schema_version": "robustness-joint-window-validation/v1",
            "operation": "validate-generic-joint-patch-window",
            "model": model,
            "model_revision": model_revision,
            "config_sha256": layers.get("config_sha256"),
            "window_selection_sha256": layer_sha256,
            "validation_rule": "bootstrap-95ci-lower-strictly-positive/v1",
        }
        for field, expected in validation_required.items():
            if validation.get(field) != expected:
                if field == "window_selection_sha256":
                    raise ValueError("window validation selection hash differs")
                raise ValueError("window validation identity differs from residual training")
        if validation.get("passed") is not True:
            raise ValueError("confirmatory window did not pass independent validation")
        validation_window = validation.get("selected_window")
        if (
            not isinstance(validation_window, Mapping)
            or set(validation_window) != {"start", "stop"}
            or validation_window.get("start") != start
            or validation_window.get("stop") != stop
        ):
            raise ValueError("window validation selected window differs")
        interval = validation.get("confidence_interval")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in interval
            )
            or float(interval[0]) <= 0.0
            or float(interval[0]) > float(interval[1])
        ):
            raise ValueError("window validation confidence interval is not strictly positive")
    elif window_validation_path is not None:
        raise ValueError("legacy layer selection cannot bind confirmatory validation")
    if policy == "frozen-causal-window/v1":
        selected = tuple(range(start, stop))
    elif policy == "sha256-seed42-middle-late-nonoverlap-same-width/v1":
        first_start = math.ceil(0.4 * decoder_layers)
        if schema == "robustness-joint-window-selection/v1":
            control = layers.get("random_control_window")
            if (
                not isinstance(control, Mapping)
                or set(control) != {"start", "stop", "rule"}
                or control.get("rule") != "sha256-drawn-nonoverlapping-same-width/v1"
            ):
                raise ValueError("confirmatory random control window fields differ")
            selected_start = _positive_int(
                control.get("start"), field="random_control_window.start"
            )
            selected_stop = _positive_int(control.get("stop"), field="random_control_window.stop")
            if (
                selected_stop - selected_start != width
                or selected_stop > decoder_layers
                or selected_start < first_start
                or not (selected_stop <= start or stop <= selected_start)
            ):
                raise ValueError("confirmatory random control window is invalid")
        else:
            candidates = tuple(
                candidate
                for candidate in range(first_start, decoder_layers - width + 1)
                if candidate + width <= start or stop <= candidate
            )
            if not candidates:
                raise ValueError("model has no eligible same-width random control window")
            selected_start = min(
                candidates,
                key=lambda candidate: hashlib.sha256(
                    (
                        "random-residual-window/v1\0seed-42\0"
                        f"{model}\0{model_revision}\0{layer_sha256}\0{candidate}"
                    ).encode()
                ).hexdigest(),
            )
            selected_stop = selected_start + width
        selected = tuple(range(selected_start, selected_stop))
    else:
        raise ValueError("residual evidence window policy is unsupported")
    evidence_sha = hashlib.sha256(
        (
            f"residual-state-evidence/v1\0{layer_sha256}\0{validation_sha256}\0{policy}\0"
            + ",".join(map(str, selected))
        ).encode()
    ).hexdigest()
    return ResidualStateEvidence(
        selected_window=(selected[0], selected[-1] + 1),
        state_layers=selected,
        policy=policy,
        layer_selection_sha256=layer_sha256,
        validation_sha256=validation_sha256,
        evidence_sha256=evidence_sha,
    )


__all__ = [
    "LocalizationEvidence",
    "ResidualStateEvidence",
    "load_localization_evidence",
    "load_residual_state_evidence",
]
