"""Proposed training consumes only hash-bound causally validated components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.training.evidence import load_localization_evidence


MODEL = "google/gemma-3-4b-it"
REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    layers = tmp_path / "layer_selection.json"
    _write(
        layers,
        {
            "schema_version": "robustness-layer-selection/v1",
            "operation": "select-distillation-layers",
            "model": MODEL,
            "model_revision": REVISION,
            "selected_window": {"start": 0, "stop": 6},
        },
    )
    components = tmp_path / "component_selection.json"
    _write(
        components,
        {
            "schema_version": "robustness-component-selection/v1",
            "operation": "localize-robustness-components",
            "model": MODEL,
            "model_revision": REVISION,
            "layer_selection_sha256": hashlib.sha256(layers.read_bytes()).hexdigest(),
            "selected": [
                {
                    "kind": "mlp-neuron",
                    "layer": 1,
                    "index": 17,
                    "identifier": "mlp-neuron:L1:N17",
                    "weight": 0.75,
                    "causally_validated": True,
                },
                {
                    "kind": "attention-head",
                    "layer": 3,
                    "index": 2,
                    "identifier": "attention-head:L3:H2",
                    "weight": 0.25,
                    "causally_validated": True,
                },
            ],
        },
    )
    return layers, components


def test_evidence_uses_component_layers_for_lora_and_normalized_causal_weights(
    tmp_path: Path,
) -> None:
    layers, components = _fixture(tmp_path)
    evidence = load_localization_evidence(
        layer_selection_path=layers,
        component_selection_path=components,
        model=MODEL,
        model_revision=REVISION,
        decoder_layers=34,
        mlp_intermediate_size=10240,
        attention_heads=8,
    )
    assert evidence.selected_window == (0, 6)
    assert evidence.adapter_layers == (1, 3)
    assert evidence.component_weights == {
        ComponentRef("mlp-neuron", 1, 17): 0.75,
        ComponentRef("attention-head", 3, 2): 0.25,
    }


def test_evidence_rejects_tampering_noncausal_or_out_of_window_components(
    tmp_path: Path,
) -> None:
    layers, components = _fixture(tmp_path)
    layer_payload = json.loads(layers.read_text(encoding="utf-8"))
    layer_payload["selected_window"] = {"start": 1, "stop": 7}
    _write(layers, layer_payload)
    with pytest.raises(ValueError, match="layer-selection hash"):
        load_localization_evidence(
            layer_selection_path=layers,
            component_selection_path=components,
            model=MODEL,
            model_revision=REVISION,
            decoder_layers=34,
            mlp_intermediate_size=10240,
            attention_heads=8,
        )

    layers, components = _fixture(tmp_path)
    component_payload = json.loads(components.read_text(encoding="utf-8"))
    component_payload["selected"][0]["causally_validated"] = False
    _write(components, component_payload)
    with pytest.raises(ValueError, match="causally validated"):
        load_localization_evidence(
            layer_selection_path=layers,
            component_selection_path=components,
            model=MODEL,
            model_revision=REVISION,
            decoder_layers=34,
            mlp_intermediate_size=10240,
            attention_heads=8,
        )
