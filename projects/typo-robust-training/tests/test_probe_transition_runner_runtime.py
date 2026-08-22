"""Falsification checks for the probe-transition training consumer boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.checkpoint import (
    TrainingCursor,
    write_training_checkpoint,
)
from typo_robust_training.training.data import TrainingDataBundle
from typo_robust_training.training.methods import ProbeTransitionTrainingEvidence
from typo_robust_training.training.pairs import TrainingPair, TrainingSource
from typo_robust_training.training.runner import (
    AdapterTrainingRunConfig,
    TrainingMicroStepResult,
    TrainingMicroStepScales,
    _load_evidence,
    run_adapter_training,
)
from typo_robust_training.training.runtime import (
    HuggingFaceAdapterTrainingRuntime,
    _require_exact_training_wrapper_revision,
    _resolve_probe_transition_runtime_method,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = PROJECT_ROOT / "tests/fixtures/gemma4b-probe-transition-output-10m.bound.json"
OUTPUT_CONFIG = PROJECT_ROOT / "configs/cycle2/gemma4b-output-matching-100step.yaml"
REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
EVIDENCE_SHA256 = "a" * 64


def _evidence(*, transition: int = 7) -> ProbeTransitionTrainingEvidence:
    return ProbeTransitionTrainingEvidence(
        model="google/gemma-3-4b-it",
        model_revision=REVISION,
        decoder_layers=34,
        selected_transition_layer=transition,
        evidence_sha256=EVIDENCE_SHA256,
    )


def _run_config(
    tmp_path: Path,
    *,
    config_path: Path = PROBE_CONFIG,
    probe_selection_path: Path | None = None,
) -> AdapterTrainingRunConfig:
    return AdapterTrainingRunConfig(
        condition="probe-transition-output-matching",
        config_path=config_path,
        training_data_dir=tmp_path,
        layer_selection_path=None,
        component_selection_path=None,
        seed=42,
        gpu_id="0",
        wandb_project=None,
        wandb_entity=None,
        output_dir=tmp_path / "output",
        probe_selection_path=probe_selection_path,
    )


def test_probe_evidence_loader_receives_exact_training_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = tmp_path / "probe-selection.json"
    calls: list[tuple[Path, str, str, int]] = []

    def load(
        path: Path,
        *,
        model: str,
        model_revision: str,
        decoder_layers: int,
    ) -> ProbeTransitionTrainingEvidence:
        calls.append((path, model, model_revision, decoder_layers))
        return _evidence()

    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_training_evidence",
        load,
    )
    protocol = load_adapter_training_config(PROBE_CONFIG)

    assert (
        _load_evidence(
            _run_config(tmp_path, probe_selection_path=selection),
            protocol=protocol,
        )
        == _evidence()
    )
    assert calls == [(selection, protocol.model, protocol.model_revision, 34)]


def test_probe_evidence_loader_rejects_missing_or_ambiguous_artifacts(tmp_path: Path) -> None:
    protocol = load_adapter_training_config(PROBE_CONFIG)
    with pytest.raises(ValueError, match="only one probe selection"):
        _load_evidence(_run_config(tmp_path), protocol=protocol)

    config = replace(
        _run_config(tmp_path, probe_selection_path=tmp_path / "probe.json"),
        layer_selection_path=tmp_path / "legacy-layer.json",
    )
    with pytest.raises(ValueError, match="only one probe selection"):
        _load_evidence(config, protocol=protocol)


def test_non_probe_condition_cannot_consume_probe_evidence_path(tmp_path: Path) -> None:
    protocol = load_adapter_training_config(OUTPUT_CONFIG)
    config = replace(
        _run_config(tmp_path, probe_selection_path=tmp_path / "probe.json"),
        condition=protocol.condition,
        config_path=OUTPUT_CONFIG,
    )
    with pytest.raises(ValueError, match="non-probe training"):
        _load_evidence(config, protocol=protocol)


def test_runtime_resolves_suffix_and_fails_closed_without_probe_evidence() -> None:
    protocol = load_adapter_training_config(PROBE_CONFIG)
    resolved = _resolve_probe_transition_runtime_method(protocol, _evidence(transition=7))

    assert resolved is not None
    assert resolved.adapter_layers == tuple(range(7, 34))
    assert resolved.state_layers == ()
    assert resolved.state_target == "none"
    with pytest.raises(ValueError, match="requires probe evidence"):
        _resolve_probe_transition_runtime_method(protocol, None)


@pytest.mark.parametrize(
    "loss_name",
    ["noisy_language_model", "answer", "state", "clean"],
)
def test_runtime_rejects_any_reenabled_non_output_objective(loss_name: str) -> None:
    protocol = load_adapter_training_config(PROBE_CONFIG)
    weights = dict(protocol.loss_weights)
    weights[loss_name] = 1.0
    drifted = replace(protocol, loss_weights=weights)

    with pytest.raises(ValueError, match="must disable state training"):
        _resolve_probe_transition_runtime_method(drifted, _evidence())


def test_runtime_rejects_disabled_output_objective() -> None:
    protocol = load_adapter_training_config(PROBE_CONFIG)
    weights = dict(protocol.loss_weights)
    weights["output"] = 0.0
    drifted = replace(protocol, loss_weights=weights)

    with pytest.raises(ValueError, match="must disable state training"):
        _resolve_probe_transition_runtime_method(drifted, _evidence())


def _revision_wrapper(
    *,
    model_revision: str | None = REVISION,
    tokenizer_revision: str | None = REVISION,
) -> SimpleNamespace:
    tokenizer_kwargs = (
        {} if tokenizer_revision is None else {"_commit_hash": tokenizer_revision}
    )
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash=model_revision)),
        tokenizer=SimpleNamespace(init_kwargs=tokenizer_kwargs),
    )


def test_training_runtime_requires_independently_observable_exact_model_and_tokenizer() -> None:
    assert _require_exact_training_wrapper_revision(
        _revision_wrapper(),
        expected=REVISION,
        role="teacher",
    ) == (REVISION, REVISION)
    assert _require_exact_training_wrapper_revision(
        _revision_wrapper(),
        expected=REVISION,
        role="student",
    ) == (REVISION, REVISION)

    with pytest.raises(ValueError, match="model revision is not observable"):
        _require_exact_training_wrapper_revision(
            _revision_wrapper(model_revision=None),
            expected=REVISION,
            role="teacher",
        )
    with pytest.raises(ValueError, match="model revision differs"):
        _require_exact_training_wrapper_revision(
            _revision_wrapper(model_revision="b" * 40),
            expected=REVISION,
            role="student",
        )
    with pytest.raises(ValueError, match="tokenizer revision is not observable"):
        _require_exact_training_wrapper_revision(
            _revision_wrapper(tokenizer_revision=None),
            expected=REVISION,
            role="teacher",
        )
    with pytest.raises(ValueError, match="tokenizer revision differs"):
        _require_exact_training_wrapper_revision(
            _revision_wrapper(tokenizer_revision="b" * 40),
            expected=REVISION,
            role="student",
        )


def _source(index: int) -> TrainingSource:
    text = f"Educational airport passage number {index} remains useful."
    return TrainingSource.from_dict(
        {
            "schema_version": "robustness-clean-record/v1",
            "kind": "clean",
            "record_id": f"{index:064x}",
            "source": "fineweb_edu",
            "source_revision": "b" * 40,
            "source_split": "train",
            "source_id": f"source-{index}",
            "group_id": f"group-{index}",
            "split": "train",
            "text": text,
            "task": None,
            "answer": None,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "normalized_content_sha256": normalized_content_sha256(text),
            "metadata": {},
            "token_count": 9,
        }
    )


def _bundle(tmp_path: Path) -> TrainingDataBundle:
    substitutions = {
        character: {"z" if character != "z" else "x": 1.0}
        for character in "abcdefghijklmnopqrstuvwxyz"
    }
    return TrainingDataBundle(
        root=tmp_path,
        sources=tuple(_source(index) for index in range(2)),
        generator=TypoGenerator(seed=42, natural_substitutions=substitutions),
        data_identity_sha256="c" * 64,
        training_data_sha256="d" * 64,
        artifact_sha256={},
    )


class _ProbeRuntime:
    def prepare_accumulation(
        self,
        pairs: tuple[TrainingPair, ...],
    ) -> tuple[TrainingMicroStepScales, ...]:
        assert len(pairs) == 2
        return tuple(TrainingMicroStepScales(output=0.5, state=0.0) for _ in pairs)

    def train_micro_step(
        self,
        pair: TrainingPair,
        *,
        loss_scale: float,
        measure_gradient_ratio: bool = False,
        output_loss_scale: float | None = None,
        state_loss_scale: float | None = None,
    ) -> TrainingMicroStepResult:
        assert loss_scale == pytest.approx(0.5)
        assert measure_gradient_ratio is False
        assert output_loss_scale == pytest.approx(0.5)
        assert state_loss_scale == 0.0
        return TrainingMicroStepResult(
            losses={"output": 1.0, "state": 0.0},
            total_loss=1.0,
            student_tokens=7,
        )

    def optimizer_step(self, *, max_grad_norm: float) -> tuple[float, float]:
        assert max_grad_norm == 1.0
        return 0.1, 0.0001

    def zero_grad(self) -> None:
        return None

    def save_state(self, path: Path) -> None:
        path.write_text("runtime-state", encoding="utf-8")

    def load_state(self, path: Path) -> None:
        raise AssertionError(f"unexpected resume from {path}")

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter.txt").write_text("adapter", encoding="utf-8")

    def provenance(self) -> dict[str, object]:
        return {"runtime": "probe-consumer-fixture/v1"}


def test_runner_binds_probe_hash_to_run_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(PROBE_CONFIG.read_text(encoding="utf-8"))
    payload["optimization"].update(
        {
            "gradient_accumulation_steps": 2,
            "max_optimizer_steps": 1,
            "max_student_tokens": 14,
            "checkpoint_every_optimizer_steps": 1,
        }
    )
    config_path = tmp_path / "probe-training.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    captured_bindings: list[dict[str, object]] = []

    def write_checkpoint(
        path: Path,
        *,
        cursor: object,
        state_path: Path,
        bindings: dict[str, object],
    ) -> None:
        assert state_path.is_file()
        captured_bindings.append(dict(bindings))
        path.write_text(json.dumps({"cursor": str(cursor)}), encoding="utf-8")

    monkeypatch.setattr(
        "typo_robust_training.training.runner.write_training_checkpoint",
        write_checkpoint,
    )
    result = run_adapter_training(
        _run_config(tmp_path, config_path=config_path),
        runtime=_ProbeRuntime(),
        data_bundle=_bundle(tmp_path),
        evidence=_evidence(),
    )

    assert captured_bindings == [
        {
            "config_sha256": load_adapter_training_config(config_path).config_sha256,
            "training_data_sha256": "d" * 64,
            "localization_sha256": None,
            "method_evidence_sha256": EVIDENCE_SHA256,
            "seed": 42,
        }
    ]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["method_evidence_sha256"] == EVIDENCE_SHA256
    assert run["localization_sha256"] is None


def test_resume_rejects_changed_probe_evidence_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_adapter_training_config(PROBE_CONFIG)
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
        _run_config(tmp_path),
        output_dir=output_dir,
        resume=True,
        evaluation_protocol_path=tmp_path / "evaluation.json",
        monitor_data_dir=tmp_path / "monitor",
    )

    with pytest.raises(ValueError, match="training checkpoint bindings differ"):
        run_adapter_training(
            run_config,
            runtime=None,
            data_bundle=_bundle(tmp_path),
            evidence=_evidence(),
        )
    assert runtime_constructed is False


def test_runtime_provenance_reports_probe_evidence_hash(
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
    runtime.protocol = load_adapter_training_config(PROBE_CONFIG)
    runtime.teacher = SimpleNamespace(config=SimpleNamespace(_commit_hash=REVISION))
    runtime.student = SimpleNamespace(config=SimpleNamespace(_commit_hash=REVISION))
    runtime.teacher_revision = REVISION
    runtime.student_revision = REVISION
    runtime.tokenizer_revision = REVISION
    runtime.code_revision = "f" * 40
    runtime.source_tree_sha256 = "e" * 64
    runtime.seed = 42
    runtime.device = "cuda:0"
    runtime.num_decoder_layers = 34
    runtime.adapter_layers = tuple(range(7, 34))
    runtime.state_layers = ()
    runtime.evidence = _evidence()
    runtime.state_weight = 0.0
    runtime.state_calibration = None
    runtime.parameter_report = SimpleNamespace(
        modules=("q_proj",),
        trainable_parameters=1,
        total_parameters=2,
    )
    runtime.attention_head_dim = 256
    monkeypatch.setattr("typo_robust_training.training.runtime._version", lambda _name: "fixture")

    provenance = runtime.provenance()

    assert provenance["method_evidence_sha256"] == EVIDENCE_SHA256
    assert provenance["teacher_revision"] == REVISION
    assert provenance["student_revision"] == REVISION
    assert provenance["tokenizer_revision"] == REVISION
    assert provenance["code_revision"] == "f" * 40
    assert provenance["adapter_layers"] == list(range(7, 34))
    assert provenance["state_layers"] == []
