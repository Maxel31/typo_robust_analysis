from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.methods import (
    ProbeTransitionTrainingEvidence,
    materialize_probe_transition_training_config,
    resolve_training_method,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "tests/fixtures/gemma4b-probe-transition-output-10m.bound.json"
TEMPLATE = (
    PROJECT_ROOT / "configs/proposals/gemma4b-probe-transition-output-10m.template.yaml"
)


def _evidence(*, transition: int = 7) -> ProbeTransitionTrainingEvidence:
    return ProbeTransitionTrainingEvidence(
        model="google/gemma-3-4b-it",
        model_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        decoder_layers=34,
        selected_transition_layer=transition,
        evidence_sha256="a" * 64,
    )


def test_probe_transition_v4_config_is_output_only_and_suffix_scoped() -> None:
    protocol = load_adapter_training_config(CONFIG)

    assert protocol.schema_version == "robustness-adapter-training-config/v4"
    assert protocol.condition == "probe-transition-output-matching"
    assert protocol.layer_scope == "probe-transition-suffix"
    assert protocol.layer_policy == "validated-linear-probe-transition-suffix/v1"
    assert protocol.expected_method_evidence_sha256 == "a" * 64
    assert protocol.loss_weights == {
        "noisy_language_model": 0.0,
        "answer": 0.0,
        "output": 1.0,
        "state": 0.0,
        "clean": 0.0,
    }
    assert protocol.state_scope == "none"
    assert protocol.state_distance == "none"
    assert protocol.state_gradient_ratio is None
    assert protocol.calibration_micro_batches == 0


def test_probe_transition_template_is_deliberately_not_runnable() -> None:
    with pytest.raises(ValueError, match="config fields differ|schema"):
        load_adapter_training_config(TEMPLATE)


def test_materializer_atomically_binds_validated_artifact_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def load(path: Path, **identity: object) -> ProbeTransitionTrainingEvidence:
        calls.append(path)
        assert identity == {
            "model": "google/gemma-3-4b-it",
            "model_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
            "decoder_layers": 34,
        }
        return _evidence()

    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_training_evidence",
        load,
    )
    evidence_path = tmp_path / "probe-artifact.json"
    evidence_path.write_text("fixture", encoding="utf-8")
    output = tmp_path / "materialized.json"

    protocol = materialize_probe_transition_training_config(
        TEMPLATE,
        evidence_path=evidence_path,
        output_path=output,
    )

    assert calls == [evidence_path]
    assert output.is_file()
    assert protocol.expected_method_evidence_sha256 == "a" * 64
    assert json.loads(output.read_text(encoding="utf-8"))["method_evidence"] == {
        "schema_version": "probe-transition-evidence-binding/v1",
        "artifact_sha256": "a" * 64,
    }
    assert not any(path.name.startswith(f".{output.name}.") for path in tmp_path.iterdir())


def test_materialized_config_hash_changes_with_evidence_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {"digest": "a" * 64}

    def load(_path: Path, **_identity: object) -> ProbeTransitionTrainingEvidence:
        return replace(_evidence(), evidence_sha256=current["digest"])

    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_training_evidence",
        load,
    )
    evidence_path = tmp_path / "probe-artifact.json"
    evidence_path.write_text("fixture", encoding="utf-8")
    first = materialize_probe_transition_training_config(
        TEMPLATE,
        evidence_path=evidence_path,
        output_path=tmp_path / "first.json",
    )
    current["digest"] = "b" * 64
    second = materialize_probe_transition_training_config(
        TEMPLATE,
        evidence_path=evidence_path,
        output_path=tmp_path / "second.json",
    )

    assert first.config_sha256 != second.config_sha256


def test_materializer_rejects_symlink_evidence_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("fixture", encoding="utf-8")
    link = tmp_path / "artifact-link.json"
    link.symlink_to(target)
    called = False

    def load(*_args: object, **_kwargs: object) -> ProbeTransitionTrainingEvidence:
        nonlocal called
        called = True
        return _evidence()

    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_training_evidence",
        load,
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        materialize_probe_transition_training_config(
            TEMPLATE,
            evidence_path=link,
            output_path=tmp_path / "output.json",
        )
    assert called is False


def test_probe_transition_resolves_exact_suffix_and_no_state_layers() -> None:
    method = resolve_training_method(
        load_adapter_training_config(CONFIG),
        evidence=_evidence(transition=7),
    )

    assert method.adapter_layers == tuple(range(7, 34))
    assert method.state_layers == ()
    assert method.state_target == "none"
    assert method.method_evidence_sha256 == "a" * 64


@pytest.mark.parametrize("transition", [-1, 0, 34, 35])
def test_probe_transition_rejects_empty_or_out_of_range_suffix(transition: int) -> None:
    with pytest.raises(ValueError, match="transition layer"):
        _evidence(transition=transition)


@pytest.mark.parametrize(
    ("transition", "expected"),
    [
        (1, tuple(range(1, 34))),
        (33, (33,)),
    ],
)
def test_probe_transition_accepts_only_nonempty_interior_suffix_boundaries(
    transition: int,
    expected: tuple[int, ...],
) -> None:
    method = resolve_training_method(
        load_adapter_training_config(CONFIG),
        evidence=_evidence(transition=transition),
    )

    assert method.adapter_layers == expected


def test_probe_transition_rejects_model_revision_mismatch() -> None:
    evidence = replace(_evidence(), model_revision="b" * 40)

    with pytest.raises(ValueError, match="identity"):
        resolve_training_method(load_adapter_training_config(CONFIG), evidence=evidence)


def test_probe_transition_rejects_valid_but_unregistered_artifact_hash() -> None:
    evidence = replace(_evidence(), evidence_sha256="b" * 64)

    with pytest.raises(ValueError, match="preregistered training config"):
        resolve_training_method(load_adapter_training_config(CONFIG), evidence=evidence)


def test_probe_transition_v4_requires_hash_bound_method_evidence(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    del payload["method_evidence"]
    path = tmp_path / "unbound-config.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config fields differ"):
        load_adapter_training_config(path)


def test_probe_transition_v4_rejects_state_or_calibration_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["objective"]["weights"]["state"] = 1
    payload["objective"]["state_gradient_ratio"] = 0.05
    payload["objective"]["calibration_micro_batches"] = 8
    path = tmp_path / "state-drift.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="condition and objective"):
        load_adapter_training_config(path)


def test_probe_transition_v4_requires_explicit_layer_policy(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    del payload["adapter"]["layer_policy"]
    path = tmp_path / "missing-policy.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter fields differ"):
        load_adapter_training_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("rank", 1), ("alpha", 999), ("dropout", 0.75)),
)
def test_probe_transition_v4_rejects_lora_capacity_or_dropout_drift(
    tmp_path: Path,
    field: str,
    value: int | float,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["adapter"][field] = value
    path = tmp_path / f"drifted-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="probe-transition LoRA must use rank 16, alpha 8, and dropout 0",
    ):
        load_adapter_training_config(path)


def test_probe_transition_cli_requires_probe_selection() -> None:
    parser = argparse.ArgumentParser()
    register_commands(parser.add_subparsers(dest="command"))

    args = parser.parse_args(
        [
            "train-probe-transition-output-matching",
            "--config",
            str(CONFIG),
            "--training-data",
            "data",
            "--probe-selection",
            "probe.json",
            "--seed",
            "42",
            "--gpu-id",
            "0",
            "--wandb-project",
            "typo-robustness-training",
            "--output-dir",
            "output",
        ]
    )

    assert args.probe_selection == Path("probe.json")
    assert args._training_condition == "probe-transition-output-matching"


def test_probe_transition_config_materializer_cli_is_explicit() -> None:
    parser = argparse.ArgumentParser()
    register_commands(parser.add_subparsers(dest="command"))

    args = parser.parse_args(
        [
            "materialize-probe-transition-training-config",
            "--template",
            "template.json",
            "--probe-selection",
            "probe.json",
            "--output-config",
            "bound.json",
        ]
    )

    assert args.template == Path("template.json")
    assert args.probe_selection == Path("probe.json")
    assert args.output_config == Path("bound.json")
