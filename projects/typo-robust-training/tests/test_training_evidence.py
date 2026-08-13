"""Proposed training consumes only hash-bound causally validated components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.training.evidence import (
    load_localization_evidence,
    load_residual_state_evidence,
)


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


def _confirmatory_fixture(tmp_path: Path) -> tuple[Path, Path]:
    selection = tmp_path / "window_selection.json"
    _write(
        selection,
        {
            "schema_version": "robustness-joint-window-selection/v1",
            "operation": "select-generic-joint-patch-window",
            "model": MODEL,
            "model_revision": REVISION,
            "config_sha256": "a" * 64,
            "selected_window": {
                "start": 0,
                "stop": 6,
                "median_pairwise_restoration": 0.81,
                "confidence_interval": [0.78, 0.86],
            },
            "random_control_window": {
                "start": 20,
                "stop": 26,
                "rule": "sha256-drawn-nonoverlapping-same-width/v1",
            },
        },
    )
    validation = tmp_path / "window_validation.json"
    _write(
        validation,
        {
            "schema_version": "robustness-joint-window-validation/v1",
            "operation": "validate-generic-joint-patch-window",
            "model": MODEL,
            "model_revision": REVISION,
            "config_sha256": "a" * 64,
            "selected_window": {"start": 0, "stop": 6},
            "window_selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
            "validation_rule": "bootstrap-95ci-lower-strictly-positive/v1",
            "confidence_interval": [0.68, 0.82],
            "passed": True,
        },
    )
    return selection, validation


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


def test_residual_evidence_selects_causal_and_deterministic_nonoverlap_control(
    tmp_path: Path,
) -> None:
    layers, _components = _fixture(tmp_path)
    causal = load_residual_state_evidence(
        layer_selection_path=layers,
        model=MODEL,
        model_revision=REVISION,
        decoder_layers=34,
        policy="frozen-causal-window/v1",
    )
    random_first = load_residual_state_evidence(
        layer_selection_path=layers,
        model=MODEL,
        model_revision=REVISION,
        decoder_layers=34,
        policy="sha256-seed42-middle-late-nonoverlap-same-width/v1",
    )
    random_second = load_residual_state_evidence(
        layer_selection_path=layers,
        model=MODEL,
        model_revision=REVISION,
        decoder_layers=34,
        policy="sha256-seed42-middle-late-nonoverlap-same-width/v1",
    )
    assert causal.state_layers == tuple(range(6))
    assert random_first == random_second
    assert len(random_first.state_layers) == len(causal.state_layers)
    assert set(random_first.state_layers).isdisjoint(causal.state_layers)
    assert random_first.state_layers[0] >= 14
    assert random_first.evidence_sha256 != causal.evidence_sha256


def test_residual_evidence_rejects_model_mismatch_and_impossible_control(
    tmp_path: Path,
) -> None:
    layers, _components = _fixture(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        load_residual_state_evidence(
            layer_selection_path=layers,
            model="different/model",
            model_revision=REVISION,
            decoder_layers=34,
            policy="frozen-causal-window/v1",
        )
    with pytest.raises(ValueError, match="no eligible"):
        load_residual_state_evidence(
            layer_selection_path=layers,
            model=MODEL,
            model_revision=REVISION,
            decoder_layers=7,
            policy="sha256-seed42-middle-late-nonoverlap-same-width/v1",
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


def test_residual_evidence_requires_passed_independent_confirmatory_validation(
    tmp_path: Path,
) -> None:
    selection, validation = _confirmatory_fixture(tmp_path)

    causal = load_residual_state_evidence(
        layer_selection_path=selection,
        window_validation_path=validation,
        model=MODEL,
        model_revision=REVISION,
        decoder_layers=34,
        policy="frozen-causal-window/v1",
    )
    random_control = load_residual_state_evidence(
        layer_selection_path=selection,
        window_validation_path=validation,
        model=MODEL,
        model_revision=REVISION,
        decoder_layers=34,
        policy="sha256-seed42-middle-late-nonoverlap-same-width/v1",
    )

    assert causal.selected_window == (0, 6)
    assert causal.state_layers == tuple(range(6))
    assert causal.validation_sha256 == hashlib.sha256(validation.read_bytes()).hexdigest()
    assert random_control.selected_window == (20, 26)
    assert random_control.state_layers == tuple(range(20, 26))
    assert random_control.validation_sha256 == causal.validation_sha256
    assert random_control.evidence_sha256 != causal.evidence_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"passed": False}, "did not pass"),
        ({"confidence_interval": [0.0, 0.82]}, "confidence interval"),
        ({"window_selection_sha256": "f" * 64}, "selection hash"),
        ({"selected_window": {"start": 1, "stop": 7}}, "selected window"),
    ],
)
def test_residual_evidence_rejects_invalid_confirmatory_validation(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    selection, validation = _confirmatory_fixture(tmp_path)
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload.update(mutation)
    _write(validation, payload)

    with pytest.raises(ValueError, match=message):
        load_residual_state_evidence(
            layer_selection_path=selection,
            window_validation_path=validation,
            model=MODEL,
            model_revision=REVISION,
            decoder_layers=34,
            policy="frozen-causal-window/v1",
        )


def test_confirmatory_residual_evidence_rejects_missing_validation(tmp_path: Path) -> None:
    selection, _validation = _confirmatory_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires independent validation"):
        load_residual_state_evidence(
            layer_selection_path=selection,
            model=MODEL,
            model_revision=REVISION,
            decoder_layers=34,
            policy="frozen-causal-window/v1",
        )
