"""Fail-closed configuration tests for probe-semantic training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from _tokenizer_attestation import tokenizer_attestation_provenance
from typo_robust_training.cli import register_commands
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.methods import (
    ProbeSemanticSubspaceTrainingEvidence,
    materialize_probe_semantic_subspace_training_config,
    resolve_training_method,
)
from typo_robust_training.training.runtime import _resolve_probe_transition_runtime_method
from typo_robust_training.training.runner import _resolved_method_presentation_layers
from typo_robust_training.training.tracking import build_wandb_run_presentation


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
        tokenizer_snapshot_attestation=tokenizer_attestation_provenance(MODEL, REVISION),
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
        ("condition", "probe-transition-output-matching"),
        ("objective__state_gradient_ratio", 0.1),
        ("objective__calibration_micro_batches", 7),
        ("objective__weights__answer", 1),
        ("objective__weights__state", 0),
        ("objective__state_scope", "none"),
        ("objective__state_distance", "cosine-residual/v1"),
        ("objective__temperature", 9.0),
        ("objective__epsilon", 0.25),
        ("adapter__layer_scope", "all-decoder-layers"),
        ("adapter__layer_policy", "validated-linear-probe-transition-suffix/v1"),
        ("adapter__rank", 8),
        ("adapter__alpha", 16),
        ("adapter__dropout", 0.1),
        ("model__id", "different/model"),
        ("model__decoder_layers", 35),
        ("sequence__max_length", 256),
        ("optimization__learning_rate", 1.0),
        ("optimization__weight_decay", 99.0),
        ("optimization__warmup_ratio", 1.0),
        ("optimization__gradient_checkpointing", False),
        ("optimization__gradient_accumulation_steps", 2),
        ("optimization__max_optimizer_steps", 1),
        ("optimization__max_student_tokens", 1),
        ("optimization__max_grad_norm", 2.0),
        ("optimization__checkpoint_every_optimizer_steps", 1),
        ("optimization__log_every_micro_steps", 2),
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


def test_semantic_runtime_resolves_evidence_before_cuda(tmp_path: Path) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    resolved = _resolve_probe_transition_runtime_method(protocol, _evidence())
    assert resolved is not None and resolved.adapter_layers[0] == 11
    with pytest.raises(ValueError):
        _resolve_probe_transition_runtime_method(protocol, None)


def test_semantic_cli_requires_explicit_kill_evidence() -> None:
    parser = argparse.ArgumentParser()
    register_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "train-probe-semantic-subspace-distillation",
            "--config",
            "bound.json",
            "--training-data",
            "data",
            "--probe-selection",
            "kill.json",
            "--seed",
            "42",
            "--gpu-id",
            "0",
            "--wandb-project",
            "test",
            "--output-dir",
            "output",
        ]
    )
    assert args._training_condition == "probe-semantic-subspace-distillation"
    assert args.probe_selection == Path("kill.json")


def test_semantic_config_reaches_standard_wandb_presentation(tmp_path: Path) -> None:
    """A valid semantic run must not fail while constructing its standard tracker identity."""

    protocol = load_adapter_training_config(_bound(tmp_path))
    method = resolve_training_method(protocol, evidence=_evidence())
    presentation = build_wandb_run_presentation(
        condition=protocol.condition,
        schema_version=protocol.schema_version,
        model=protocol.model,
        seed=42,
        max_optimizer_steps=protocol.max_optimizer_steps,
        max_student_tokens=protocol.max_student_tokens,
        state_gradient_ratio=protocol.state_gradient_ratio,
        state_layers=method.state_layers,
    )

    assert presentation.name.startswith(
        "Probe-subspace proposal · Rank-16 semantic-subspace distillation · L11"
    )
    assert presentation.job_type == "proposed-probe-semantic-subspace"
    assert "arm:probe-semantic-subspace-distillation" in presentation.tags
    assert "frozen-classifier forward-KL" in presentation.notes
    assert _resolved_method_presentation_layers(condition=protocol.condition, method=method) == (
        11,
    )


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
