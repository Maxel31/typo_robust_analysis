"""Fail-closed configuration tests for probe-semantic training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.methods import (
    ProbeSemanticSubspaceTrainingEvidence,
    materialize_probe_semantic_subspace_training_config,
    resolve_training_method,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/proposals/gemma4b-probe-semantic-subspace-10m.template.yaml"
MODEL = "google/gemma-3-4b-it"
REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"


def _evidence(*, digest: str = "a" * 64) -> ProbeSemanticSubspaceTrainingEvidence:
    basis = np.eye(16, 32, dtype=np.float64)
    return ProbeSemanticSubspaceTrainingEvidence(
        model=MODEL,
        model_revision=REVISION,
        decoder_layers=34,
        transition_layer=11,
        primary_probe_seed=42,
        basis=basis,
        projected_class_weights=np.ones((20, 16), dtype=np.float64),
        classifier_bias=np.zeros(20, dtype=np.float64),
        evidence_sha256=digest,
    )


def _bound(tmp_path: Path, **mutations: object) -> Path:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    payload["schema_version"] = "robustness-adapter-training-config/v6"
    payload["method_evidence"]["artifact_sha256"] = "a" * 64
    for path, value in mutations.items():
        target = payload
        parts = path.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    output = tmp_path / f"bound-{len(list(tmp_path.iterdir()))}.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    return output


def test_semantic_template_is_deliberately_non_runnable() -> None:
    with pytest.raises(ValueError):
        load_adapter_training_config(TEMPLATE)


def test_semantic_protocol_and_suffix_are_exact(tmp_path: Path) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    method = resolve_training_method(protocol, evidence=_evidence())
    assert protocol.condition == "probe-semantic-subspace-distillation"
    assert protocol.state_gradient_ratio == 0.05
    assert protocol.calibration_micro_batches == 8
    assert method.adapter_layers == tuple(range(11, 34))
    assert method.state_layers == (11,)
    assert method.state_target == "probe-semantic-subspace-rank16"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("objective__state_gradient_ratio", 0.1),
        ("objective__calibration_micro_batches", 7),
        ("objective__weights__answer", 1),
        ("objective__weights__state", 0),
        ("objective__state_scope", "none"),
        ("objective__state_distance", "cosine-residual/v1"),
        ("adapter__layer_scope", "all-decoder-layers"),
        ("adapter__layer_policy", "validated-linear-probe-transition-suffix/v1"),
    ],
)
def test_semantic_protocol_rejects_rho_loss_and_scope_drift(
    tmp_path: Path, path: str, value: object
) -> None:
    with pytest.raises(ValueError):
        load_adapter_training_config(_bound(tmp_path, **{path: value}))


def test_semantic_evidence_arrays_are_immutable() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError):
        evidence.basis[0, 0] = 3.0


def test_semantic_materializer_revalidates_and_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path = tmp_path / "kill.json"
    evidence_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "bound.json"
    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_semantic_subspace_training_evidence",
        lambda *args, **kwargs: _evidence(),
    )
    protocol = materialize_probe_semantic_subspace_training_config(
        TEMPLATE,
        evidence_path=evidence_path,
        output_path=output,
    )
    assert protocol.expected_method_evidence_sha256 == "a" * 64
    assert json.loads(output.read_text())["method_evidence"]["artifact_sha256"] == "a" * 64
    with pytest.raises(FileExistsError):
        materialize_probe_semantic_subspace_training_config(
            TEMPLATE,
            evidence_path=evidence_path,
            output_path=output,
        )
