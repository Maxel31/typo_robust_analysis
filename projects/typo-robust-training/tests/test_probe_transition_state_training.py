from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from typo_robust_training.cli import register_commands
from typo_robust_training.evaluation.checkpoints import load_adapter_descriptors
from typo_robust_training.training.adapters import attach_lora_adapters
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.checkpoint import TrainingCursor, write_training_checkpoint
from typo_robust_training.training.encoding import PairedEncoding
from typo_robust_training.training.methods import (
    ProbeTransitionStateTrainingEvidence,
    materialize_probe_transition_state_training_config,
    resolve_training_method,
)
from typo_robust_training.training.runtime import (
    HuggingFaceAdapterTrainingRuntime,
    _resolve_probe_transition_runtime_method,
    _validate_replayed_priority_b_calibration,
    validate_resume_state_calibration,
)
from typo_robust_training.training.runner import (
    AdapterTrainingRunConfig,
    _load_evidence,
    run_adapter_training,
)
from typo_robust_training.training.step import compute_training_step

from test_probe_transition_runner_runtime import _bundle as _training_bundle
from test_evaluation_inputs import (
    PROTOCOL as EVALUATION_PROTOCOL,
    _adapter as _evaluation_adapter,
    _tree_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/proposals/gemma4b-probe-transition-single-layer-state-10m.template.yaml"


def _bound(tmp_path: Path, *, digest: str = "a" * 64) -> Path:
    payload = json.loads(TEMPLATE.read_text())
    payload["schema_version"] = "robustness-adapter-training-config/v5"
    payload["method_evidence"]["artifact_sha256"] = digest
    path = tmp_path / "bound-state-training.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evidence(*, transition: int = 7, digest: str = "a" * 64):
    return ProbeTransitionStateTrainingEvidence(
        model="google/gemma-3-4b-it",
        model_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        decoder_layers=34,
        selected_transition_layer=transition,
        parent_probe_artifact_sha256="b" * 64,
        evidence_sha256=digest,
    )


def test_v5_is_exactly_output_plus_single_layer_cosine_with_one_shot_rho(
    tmp_path: Path,
) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    resolved = resolve_training_method(protocol, evidence=_evidence())

    assert protocol.condition == "probe-transition-single-layer-state-distillation"
    assert protocol.loss_weights == {
        "noisy_language_model": 0.0,
        "answer": 0.0,
        "output": 1.0,
        "state": 1.0,
        "clean": 0.0,
    }
    assert protocol.state_gradient_ratio == 0.05
    assert protocol.calibration_micro_batches == 8
    assert protocol.temperature == 1.0
    assert protocol.epsilon == 1e-8
    assert protocol.state_distance == "cosine-residual/v1"
    assert resolved.adapter_layers == tuple(range(7, 34))
    assert resolved.state_layers == (7,)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("objective", "state_gradient_ratio", 0.1), "dosage"),
        (("objective", "calibration_micro_batches", 7), "dosage"),
        (("objective", "temperature", 2.0), "dosage"),
        (("objective", "epsilon", 1e-6), "dosage"),
        (("objective", "state_scope", "all-layers-edited-word-final-tokens"), "objective"),
        (("objective", "state_window_policy", "all-decoder-layers/v1"), "objective"),
        (("adapter", "layer_scope", "all-decoder-layers"), "objective"),
        (("objective.weights", "answer", 1), "objective"),
    ],
)
def test_v5_rejects_dosage_loss_or_scope_drift(
    tmp_path: Path,
    mutation: tuple[str, str, object],
    message: str,
) -> None:
    payload = json.loads(TEMPLATE.read_text())
    payload["schema_version"] = "robustness-adapter-training-config/v5"
    payload["method_evidence"]["artifact_sha256"] = "a" * 64
    section, field, value = mutation
    if section == "objective.weights":
        payload["objective"]["weights"][field] = value
    else:
        payload[section][field] = value
    path = tmp_path / f"drift-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_adapter_training_config(path)


def test_runtime_requires_state_gate_evidence_and_exact_single_layer(
    tmp_path: Path,
) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    resolved = _resolve_probe_transition_runtime_method(protocol, _evidence())
    assert resolved is not None
    assert resolved.state_layers == (7,)
    with pytest.raises(ValueError, match="requires probe evidence"):
        _resolve_probe_transition_runtime_method(protocol, None)


def _calibration(*, state_weight: float = 0.025) -> dict[str, object]:
    return {
        "schema_version": "state-gradient-calibration/v3",
        "micro_batches": 8,
        "noisy_micro_batches": 8,
        "record_ids": [f"record-{index}" for index in range(8)],
        "output_gradient_norms": [2.0] * 8,
        "state_gradient_norms": [4.0] * 8,
        "mean_output_gradient_norm": 2.0,
        "mean_state_gradient_norm": 4.0,
        "target_gradient_ratio": 0.05,
        "state_weight": state_weight,
        "achieved_initial_ratio": state_weight * 2.0,
        "replay_relative_tolerance": 1e-6,
        "replay_absolute_tolerance": 1e-8,
    }


def _self_consistent_forged_calibration() -> dict[str, object]:
    state_weight = 999.0
    mean_output = 2.0
    mean_state = 0.05 * mean_output / state_weight
    return {
        **_calibration(state_weight=state_weight),
        "record_ids": [f"forged-{index}" for index in range(8)],
        "state_gradient_norms": [mean_state] * 8,
        "mean_state_gradient_norm": mean_state,
        "state_weight": state_weight,
        "achieved_initial_ratio": 0.05,
    }


def _resume_state(
    path: Path,
    *,
    protocol: object,
    state_weight: float = 0.025,
    calibration: object | None = None,
) -> Path:
    torch.save(
        {
            "schema_version": "robustness-adapter-runtime-state/v3",
            "condition": "probe-transition-single-layer-state-distillation",
            "config_sha256": protocol.config_sha256,
            "seed": 42,
            "state_weight": state_weight,
            "state_calibration": (
                _calibration(state_weight=state_weight)
                if calibration is None
                else calibration
            ),
        },
        path,
    )
    return path


@pytest.mark.parametrize(
    ("state_weight", "calibration", "message"),
    [
        (0.0, _calibration(state_weight=0.0), "state weight differs"),
        (999.0, _calibration(state_weight=999.0), "calibrated state weight differs"),
        (0.025, {}, "calibration fields differ"),
        (
            0.025,
            {**_calibration(), "output_gradient_norms": [float("nan"), *([2.0] * 7)]},
            "output gradients differs",
        ),
        (0.025, {**_calibration(), "record_ids": []}, "calibration records differ"),
        (
            0.025,
            {**_calibration(), "replay_relative_tolerance": 1e-3},
            "replay tolerance differs",
        ),
        (
            0.025,
            {**_calibration(), "record_ids": ["duplicate-record"] * 8},
            "calibration records differ",
        ),
        (
            0.025,
            {**_calibration(), "achieved_initial_ratio": float("inf")},
            "achieved ratio differs",
        ),
    ],
)
def test_resume_calibration_rejects_zero_999_empty_or_nonfinite_evidence(
    tmp_path: Path,
    state_weight: float,
    calibration: object,
    message: str,
) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    path = _resume_state(
        tmp_path / "runtime-state.pt",
        protocol=protocol,
        state_weight=state_weight,
        calibration=calibration,
    )

    with pytest.raises(ValueError, match=message):
        validate_resume_state_calibration(path, protocol=protocol, seed=42)


def test_structural_validation_alone_cannot_authenticate_self_consistent_forgery(
    tmp_path: Path,
) -> None:
    """The replay layer, not checkpoint self-consistency, proves initial dosage."""

    protocol = load_adapter_training_config(_bound(tmp_path))
    forged = _self_consistent_forged_calibration()
    path = _resume_state(
        tmp_path / "self-consistent-forgery.pt",
        protocol=protocol,
        state_weight=999.0,
        calibration=forged,
    )
    validate_resume_state_calibration(path, protocol=protocol, seed=42)

    with pytest.raises(ValueError, match="replay records differ"):
        _validate_replayed_priority_b_calibration(
            protocol=protocol,
            saved_state_weight=999.0,
            saved_calibration=forged,
            replayed_calibration=_calibration(),
        )


def test_runtime_replays_real_calibration_claim_before_authorizing_load(
    tmp_path: Path,
) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    state_path = _resume_state(tmp_path / "runtime-state.pt", protocol=protocol)
    runtime = HuggingFaceAdapterTrainingRuntime.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime.protocol = protocol
    runtime.seed = 42
    runtime.state_weight = None
    runtime.state_calibration = None
    runtime._torch = torch
    runtime._prepared_encodings = deque()
    runtime._verified_resume_state_path = None
    runtime._verified_resume_state_sha256 = None
    zeroed: list[bool] = []
    runtime._measure_state_calibration = lambda pairs: (
        _calibration() if len(tuple(pairs)) == 8 else pytest.fail("wrong replay dosage")
    )
    runtime.zero_grad = lambda: zeroed.append(True)
    runtime._trainable_parameters = lambda: ()
    pairs = tuple(SimpleNamespace(record_id=f"record-{index}") for index in range(8))

    runtime.verify_resume_state_calibration(state_path, pairs)

    assert runtime._verified_resume_state_path == state_path.resolve()
    assert runtime._verified_resume_state_sha256 is not None
    assert zeroed == [True]


def test_runtime_replay_rejects_self_consistent_forged_calibration(
    tmp_path: Path,
) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    forged = _self_consistent_forged_calibration()
    state_path = _resume_state(
        tmp_path / "forged-runtime-state.pt",
        protocol=protocol,
        state_weight=999.0,
        calibration=forged,
    )
    runtime = HuggingFaceAdapterTrainingRuntime.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime.protocol = protocol
    runtime.seed = 42
    runtime.state_weight = None
    runtime.state_calibration = None
    runtime._torch = torch
    runtime._prepared_encodings = deque()
    runtime._verified_resume_state_path = None
    runtime._verified_resume_state_sha256 = None
    runtime._measure_state_calibration = lambda _pairs: _calibration()
    runtime.zero_grad = lambda: None
    runtime._trainable_parameters = lambda: ()

    with pytest.raises(ValueError, match="replay records differ"):
        runtime.verify_resume_state_calibration(state_path, tuple(range(8)))
    assert runtime._verified_resume_state_path is None


def test_priority_b_runtime_refuses_direct_checkpoint_load_without_replay(
    tmp_path: Path,
) -> None:
    protocol = load_adapter_training_config(_bound(tmp_path))
    state_path = _resume_state(tmp_path / "unverified-runtime-state.pt", protocol=protocol)
    runtime = HuggingFaceAdapterTrainingRuntime.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime.protocol = protocol
    runtime._verified_resume_state_path = None
    runtime._verified_resume_state_sha256 = None

    with pytest.raises(ValueError, match="must pass exact calibration replay"):
        runtime.load_state(state_path)


def test_state_runtime_provenance_binds_gate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cuda:
        @staticmethod
        def get_device_name(_index: int) -> str:
            return "fixture-gpu"

        @staticmethod
        def get_device_properties(_index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=1)

    runtime = HuggingFaceAdapterTrainingRuntime.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime._torch = SimpleNamespace(version=SimpleNamespace(cuda="fixture"), cuda=_Cuda())
    runtime.protocol = load_adapter_training_config(_bound(tmp_path))
    runtime.teacher = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    runtime.student = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    runtime.teacher_revision = runtime.protocol.model_revision
    runtime.student_revision = runtime.protocol.model_revision
    runtime.tokenizer_revision = runtime.protocol.model_revision
    runtime.code_revision = "f" * 40
    runtime.source_tree_sha256 = "e" * 64
    runtime.seed = 42
    runtime.device = "cuda:0"
    runtime.num_decoder_layers = 34
    runtime.adapter_layers = tuple(range(7, 34))
    runtime.state_layers = (7,)
    runtime.evidence = _evidence()
    runtime.state_weight = 0.025
    runtime.state_calibration = _calibration()
    runtime.parameter_report = SimpleNamespace(
        modules=("q_proj",),
        trainable_parameters=1,
        total_parameters=2,
    )
    runtime.attention_head_dim = 256
    monkeypatch.setattr("typo_robust_training.training.runtime._version", lambda _name: "fixture")

    provenance = runtime.provenance()

    assert provenance["method_evidence_sha256"] == "a" * 64


def test_evaluation_requires_exact_priority_b_run_runtime_provenance(tmp_path: Path) -> None:
    adapter = _evaluation_adapter(
        tmp_path,
        condition="probe-transition-single-layer-state-distillation",
        seed=42,
        runtime_version="v2",
        method_evidence_sha256="9" * 64,
    )
    runtime_path = adapter / "training_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["method_evidence_sha256"] = "9" * 64
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    run_path = adapter.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["runtime"] = runtime
    run["outputs"]["adapter"]["sha256"] = _tree_sha256(adapter)
    run_path.write_text(json.dumps(run), encoding="utf-8")

    descriptor = load_adapter_descriptors((adapter,), protocol=EVALUATION_PROTOCOL)[0]
    assert descriptor.method_evidence_sha256 == "9" * 64

    run["runtime"] = {**runtime, "method_evidence_sha256": "8" * 64}
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="run/runtime provenance differs"):
        load_adapter_descriptors((adapter,), protocol=EVALUATION_PROTOCOL)


def _run_config(
    tmp_path: Path,
    *,
    config_path: Path,
    state_gate_path: Path | None,
    resume: bool = False,
) -> AdapterTrainingRunConfig:
    return AdapterTrainingRunConfig(
        condition="probe-transition-single-layer-state-distillation",
        config_path=config_path,
        training_data_dir=tmp_path,
        layer_selection_path=None,
        component_selection_path=None,
        seed=42,
        gpu_id="0",
        wandb_project=None,
        wandb_entity=None,
        output_dir=tmp_path / "output",
        resume=resume,
        state_gate_path=state_gate_path,
    )


def test_state_evidence_loader_receives_only_exact_gate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _bound(tmp_path)
    protocol = load_adapter_training_config(config_path)
    gate = tmp_path / "gate.json"
    calls: list[tuple[Path, str, str, int]] = []

    def load(
        path: Path,
        *,
        model: str,
        model_revision: str,
        decoder_layers: int,
    ) -> ProbeTransitionStateTrainingEvidence:
        calls.append((path, model, model_revision, decoder_layers))
        return _evidence()

    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_state_training_evidence",
        load,
    )
    run_config = _run_config(
        tmp_path,
        config_path=config_path,
        state_gate_path=gate,
    )
    assert _load_evidence(run_config, protocol=protocol) == _evidence()
    assert calls == [(gate, protocol.model, protocol.model_revision, 34)]

    ambiguous = replace(run_config, probe_selection_path=tmp_path / "parent.json")
    with pytest.raises(ValueError, match="requires only one gate artifact"):
        _load_evidence(ambiguous, protocol=protocol)


def test_materializer_binds_gate_hash_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "gate.json"
    evidence_path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_state_training_evidence",
        lambda *_args, **_kwargs: _evidence(),
    )
    output = tmp_path / "materialized.json"

    protocol = materialize_probe_transition_state_training_config(
        TEMPLATE,
        evidence_path=evidence_path,
        output_path=output,
    )

    assert protocol.expected_method_evidence_sha256 == "a" * 64
    assert json.loads(output.read_text())["method_evidence"] == {
        "schema_version": "probe-transition-state-gate-binding/v1",
        "artifact_sha256": "a" * 64,
    }


def _tiny_model() -> Gemma3ForCausalLM:
    return Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
            sliding_window=32,
            layer_types=["full_attention"] * 3,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def test_state_gradient_reaches_transition_lora_but_not_teacher_or_lower_layer(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_adapter_training_config(_bound(tmp_path)),
        decoder_layers=3,
        lora_rank=2,
        lora_alpha=4.0,
        gradient_checkpointing=False,
    )
    teacher = _tiny_model()
    student = attach_lora_adapters(
        _tiny_model(),
        protocol=protocol,
        decoder_layers=(1, 2),
    )
    encoding = PairedEncoding(
        record_id="pair",
        clean_input_ids=(1, 5, 6, 7),
        typo_input_ids=(1, 8, 6, 7),
        clean_attention_mask=(1, 1, 1, 1),
        typo_attention_mask=(1, 1, 1, 1),
        output_logit_pairs=((0, 0), (2, 2)),
        global_state_token_pairs=((0, 0), (2, 2)),
        clean_edit_positions=(1,),
        typo_edit_positions=(1,),
        answer_targets=(),
        student_tokens=4,
        is_noop=False,
    )

    output = compute_training_step(
        teacher=teacher,
        student=student,
        encoding=encoding,
        protocol=protocol,
        component_weights=None,
        attention_head_dim=4,
        state_layers=(1,),
        state_weight=1.0,
    )
    output.losses["state"].backward()

    transition_grads = [
        parameter.grad
        for name, parameter in student.named_parameters()
        if ".layers.1." in name and "lora_" in name
    ]
    assert transition_grads
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in transition_grads)
    assert not any(".layers.0." in name and "lora_" in name for name, _ in student.named_parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_cli_exposes_only_explicit_gate_for_state_training() -> None:
    parser = argparse.ArgumentParser()
    register_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "train-probe-transition-single-layer-state-distillation",
            "--config", "config.json",
            "--training-data", "data",
            "--state-gate", "gate.json",
            "--seed", "42",
            "--gpu-id", "0",
            "--wandb-project", "project",
            "--output-dir", "output",
        ]
    )
    assert args._training_condition == "probe-transition-single-layer-state-distillation"
    assert args.state_gate == Path("gate.json")


def test_resume_rejects_changed_state_gate_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _bound(tmp_path)
    protocol = load_adapter_training_config(config_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    state_path = output_dir / "runtime-state.pt"
    state_path.write_bytes(b"opaque runtime state")
    monitor_protocol_sha = "e" * 64
    monitor_data_sha = "f" * 64
    write_training_checkpoint(
        output_dir / "checkpoint.json",
        cursor=TrainingCursor(0, 0, 0, 0, 0),
        state_path=state_path,
        bindings={
            "config_sha256": protocol.config_sha256,
            "training_data_sha256": "d" * 64,
            "localization_sha256": None,
            "method_evidence_sha256": "9" * 64,
            "monitor_protocol_sha256": monitor_protocol_sha,
            "monitor_data_sha256": monitor_data_sha,
            "seed": 42,
        },
    )
    study = SimpleNamespace(
        config_sha256=monitor_protocol_sha,
        tune_fineweb_documents=1,
        tune_natural_pairs=1,
        gates={
            "maximum_clean_kl_nats_per_token": 0.03,
            "maximum_clean_ppl_ratio": 1.02,
        },
    )
    monitor_bundle = SimpleNamespace(
        records=(
            SimpleNamespace(source="fineweb_edu", kind="clean"),
            SimpleNamespace(source="github_typo_corpus", kind="natural"),
        ),
        manifest_sha256=monitor_data_sha,
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.study.load_evaluation_study_protocol",
        lambda _path: study,
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.data.load_evaluation_corpus_bundle",
        lambda *_args, **_kwargs: monitor_bundle,
    )
    runtime_constructed = False

    def forbidden_runtime(*_args: object, **_kwargs: object) -> object:
        nonlocal runtime_constructed
        runtime_constructed = True
        raise AssertionError("runtime must not be constructed for mismatched evidence")

    monkeypatch.setattr(
        "typo_robust_training.training.runtime.HuggingFaceAdapterTrainingRuntime",
        forbidden_runtime,
    )
    run_config = replace(
        _run_config(
            tmp_path,
            config_path=config_path,
            state_gate_path=tmp_path / "gate.json",
            resume=True,
        ),
        evaluation_protocol_path=tmp_path / "evaluation.json",
        monitor_data_dir=tmp_path / "monitor",
    )

    with pytest.raises(ValueError, match="training checkpoint bindings differ"):
        run_adapter_training(
            run_config,
            runtime=None,
            data_bundle=_training_bundle(tmp_path),
            evidence=_evidence(),
        )
    assert runtime_constructed is False


def test_resume_rejects_invalid_calibration_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _bound(tmp_path)
    protocol = load_adapter_training_config(config_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    state_path = _resume_state(
        output_dir / "runtime-state.pt",
        protocol=protocol,
        state_weight=999.0,
        calibration=_calibration(state_weight=999.0),
    )
    monitor_protocol_sha = "e" * 64
    monitor_data_sha = "f" * 64
    write_training_checkpoint(
        output_dir / "checkpoint.json",
        cursor=TrainingCursor(0, 0, 0, 0, 0),
        state_path=state_path,
        bindings={
            "config_sha256": protocol.config_sha256,
            "training_data_sha256": "d" * 64,
            "localization_sha256": None,
            "method_evidence_sha256": "a" * 64,
            "monitor_protocol_sha256": monitor_protocol_sha,
            "monitor_data_sha256": monitor_data_sha,
            "seed": 42,
        },
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.study.load_evaluation_study_protocol",
        lambda _path: SimpleNamespace(
            config_sha256=monitor_protocol_sha,
            tune_fineweb_documents=1,
            tune_natural_pairs=1,
            gates={
                "maximum_clean_kl_nats_per_token": 0.03,
                "maximum_clean_ppl_ratio": 1.02,
            },
        ),
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.data.load_evaluation_corpus_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            records=(
                SimpleNamespace(source="fineweb_edu", kind="clean"),
                SimpleNamespace(source="github_typo_corpus", kind="natural"),
            ),
            manifest_sha256=monitor_data_sha,
        ),
    )
    runtime_constructed = False

    def forbidden_runtime(*_args: object, **_kwargs: object) -> object:
        nonlocal runtime_constructed
        runtime_constructed = True
        raise AssertionError("runtime must not be constructed before calibration validation")

    monkeypatch.setattr(
        "typo_robust_training.training.runtime.HuggingFaceAdapterTrainingRuntime",
        forbidden_runtime,
    )
    run_config = replace(
        _run_config(
            tmp_path,
            config_path=config_path,
            state_gate_path=tmp_path / "gate.json",
            resume=True,
        ),
        evaluation_protocol_path=tmp_path / "evaluation.json",
        monitor_data_dir=tmp_path / "monitor",
    )

    with pytest.raises(ValueError, match="calibrated state weight differs"):
        run_adapter_training(
            run_config,
            runtime=None,
            data_bundle=_training_bundle(tmp_path),
            evidence=_evidence(),
        )
    assert runtime_constructed is False


def test_runner_replays_initial_eight_noisy_pairs_before_loading_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(TEMPLATE.read_text())
    payload["schema_version"] = "robustness-adapter-training-config/v5"
    payload["method_evidence"]["artifact_sha256"] = "a" * 64
    payload["optimization"]["max_optimizer_steps"] = 1
    payload["optimization"]["checkpoint_every_optimizer_steps"] = 50
    config_path = tmp_path / "resume-config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    protocol = load_adapter_training_config(config_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    state_path = _resume_state(output_dir / "runtime-state.pt", protocol=protocol)
    monitor_protocol_sha = "e" * 64
    monitor_data_sha = "f" * 64
    write_training_checkpoint(
        output_dir / "checkpoint.json",
        cursor=TrainingCursor(0, 0, 0, 1, protocol.max_student_tokens),
        state_path=state_path,
        bindings={
            "config_sha256": protocol.config_sha256,
            "training_data_sha256": "d" * 64,
            "localization_sha256": None,
            "method_evidence_sha256": "a" * 64,
            "monitor_protocol_sha256": monitor_protocol_sha,
            "monitor_data_sha256": monitor_data_sha,
            "seed": 42,
        },
    )
    (output_dir / "adapter-step-000001").mkdir()
    monkeypatch.setattr(
        "typo_robust_training.evaluation.study.load_evaluation_study_protocol",
        lambda _path: SimpleNamespace(
            config_sha256=monitor_protocol_sha,
            tune_fineweb_documents=1,
            tune_natural_pairs=1,
            gates={
                "maximum_clean_kl_nats_per_token": 0.03,
                "maximum_clean_ppl_ratio": 1.02,
            },
        ),
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.data.load_evaluation_corpus_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            records=(
                SimpleNamespace(source="fineweb_edu", kind="clean"),
                SimpleNamespace(source="github_typo_corpus", kind="natural"),
            ),
            manifest_sha256=monitor_data_sha,
        ),
    )

    class ReplayRuntime:
        replayed = False
        loaded = False

        @staticmethod
        def pair_is_usable(_pair: object) -> bool:
            return True

        @staticmethod
        def retained_clean_character_extent(pair: object) -> int:
            return len(pair.clean_text)

        def verify_resume_state_calibration(
            self,
            path: Path,
            pairs: tuple[object, ...],
        ) -> None:
            assert path == state_path.resolve()
            assert len(pairs) == 8
            assert all(not pair.is_noop and pair.edits for pair in pairs)
            self.replayed = True

        def load_state(self, path: Path) -> None:
            assert self.replayed
            assert path == state_path.resolve()
            self.loaded = True

        @staticmethod
        def zero_grad() -> None:
            return None

        @staticmethod
        def save_state(path: Path) -> None:
            path.write_text("saved", encoding="utf-8")

        @staticmethod
        def save_adapter(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter.txt").write_text("saved", encoding="utf-8")

        @staticmethod
        def provenance() -> dict[str, object]:
            return {"runtime": "replay-order-fixture/v1"}

    runtime = ReplayRuntime()
    with pytest.raises(RuntimeError, match="metrics are missing optimizer step"):
        run_adapter_training(
            replace(
                _run_config(
                    tmp_path,
                    config_path=config_path,
                    state_gate_path=tmp_path / "gate.json",
                    resume=True,
                ),
                output_dir=output_dir,
                evaluation_protocol_path=tmp_path / "evaluation.json",
                monitor_data_dir=tmp_path / "monitor",
            ),
            runtime=runtime,
            data_bundle=_training_bundle(tmp_path),
            evidence=_evidence(),
        )
    assert runtime.replayed is True
    assert runtime.loaded is True
