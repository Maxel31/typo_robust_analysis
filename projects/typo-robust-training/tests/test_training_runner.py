"""The shared runner checkpoints at optimizer boundaries and resumes exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.evaluation.study import load_evaluation_study_protocol
from typo_robust_training.training.checkpoint import TrainingCursor
from typo_robust_training.training.data import TrainingDataBundle
from typo_robust_training.training.evidence import ResidualStateEvidence
from typo_robust_training.training.pairs import TrainingPair, TrainingSource
from typo_robust_training.training.runner import (
    AdapterTrainingRunConfig,
    TrainingMicroStepResult,
    TrainingMicroStepScales,
    _materialize_usable_pair,
    _monitor_violation_streak,
    _next_usable_training_pair,
    _optimizer_step_telemetry,
    normalized_accumulation_scales,
    run_adapter_training,
)
from typo_robust_training.training.tracking import WandbRunPresentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs/baselines/noisy-language-model.yaml"
LEGACY_GLOBAL_STATE_CONFIG = PROJECT_ROOT / "configs/baselines/global-state-alignment.yaml"
CYCLE2_OUTPUT_CONFIG = PROJECT_ROOT / "configs/cycle2/gemma4b-output-matching-100step.yaml"
CYCLE3_CAUSAL_CONFIG = PROJECT_ROOT / "configs/cycle3/gemma4b-causal-window-10m.yaml"
EVALUATION_PROTOCOL = PROJECT_ROOT / "configs/robustness-evaluation-v1.yaml"
NATURAL_SUBSTITUTIONS = {
    character: {"z" if character != "z" else "x": 1.0} for character in "abcdefghijklmnopqrstuvwxyz"
}


def _source(index: int) -> TrainingSource:
    text = f"Educational airport passage number {index} remains useful."
    return TrainingSource.from_dict(
        {
            "schema_version": "robustness-clean-record/v1",
            "kind": "clean",
            "record_id": f"{index:064x}",
            "source": "fineweb_edu",
            "source_revision": "a" * 40,
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
    return TrainingDataBundle(
        root=tmp_path,
        sources=tuple(_source(index) for index in range(5)),
        generator=TypoGenerator(seed=42, natural_substitutions=NATURAL_SUBSTITUTIONS),
        data_identity_sha256="b" * 64,
        training_data_sha256="c" * 64,
        artifact_sha256={},
    )


def _config(tmp_path: Path) -> Path:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    payload["optimization"]["gradient_accumulation_steps"] = 2
    payload["optimization"]["max_optimizer_steps"] = 3
    payload["optimization"]["checkpoint_every_optimizer_steps"] = 1
    path = tmp_path / "training.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Runtime:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.seen: list[tuple[int, str, str]] = []
        self.optimizer_steps = 0

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
        assert output_loss_scale is None
        assert state_loss_scale is None
        if self.fail_after is not None and len(self.seen) >= self.fail_after:
            raise RuntimeError("injected interruption")
        self.seen.append((pair.epoch, pair.record_id, pair.typo_text))
        return TrainingMicroStepResult(
            losses={"noisy_language_model": 1.0},
            total_loss=1.0,
            student_tokens=7,
        )

    def optimizer_step(self, *, max_grad_norm: float) -> tuple[float, float]:
        assert max_grad_norm == 1.0
        self.optimizer_steps += 1
        return 0.25, 0.0002

    def zero_grad(self) -> None:
        return None

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps({"optimizer_steps": self.optimizer_steps}), encoding="utf-8")

    def load_state(self, path: Path) -> None:
        self.optimizer_steps = json.loads(path.read_text(encoding="utf-8"))["optimizer_steps"]

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter.txt").write_text(str(self.optimizer_steps), encoding="utf-8")

    def provenance(self) -> dict[str, object]:
        return {"runtime": "offline-training-fixture/v1", "gpu_id": "3"}


def _run_config(
    tmp_path: Path,
    config: Path,
    output: Path,
    *,
    resume: bool,
    tracking: bool = False,
):
    return AdapterTrainingRunConfig(
        condition="noisy-language-model",
        config_path=config,
        training_data_dir=tmp_path,
        layer_selection_path=None,
        component_selection_path=None,
        seed=42,
        gpu_id="3",
        wandb_project="typo-robustness-training" if tracking else None,
        wandb_entity="fixture-entity" if tracking else None,
        output_dir=output,
        resume=resume,
    )


class _Tracker:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, int | float]]] = []
        self.finished: list[tuple[str, dict[str, int | float]]] = []

    def log_optimizer_step(
        self,
        metrics: dict[str, int | float],
        *,
        optimizer_step: int,
    ) -> None:
        self.logged.append((optimizer_step, dict(metrics)))

    def finish(self, *, status: str, summary: dict[str, int | float]) -> None:
        self.finished.append((status, dict(summary)))

    def provenance(self) -> dict[str, object]:
        return {
            "provider": "wandb",
            "project": "typo-robustness-training",
            "entity": "fixture-entity",
            "run_id": "fixture-run",
            "url": "https://wandb.example/fixture-run",
        }


class _Cycle2Runtime(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.noop_sequence: list[bool] = []
        self.gradient_checks: list[bool] = []
        self.scales: list[tuple[float, float]] = []
        self.monitor_calls = 0

    def prepare_accumulation(
        self,
        pairs: tuple[TrainingPair, ...],
    ) -> tuple[TrainingMicroStepScales, ...]:
        assert len(pairs) == 4
        return tuple(
            TrainingMicroStepScales(
                output=0.25,
                state=0.0,
            )
            for _ in pairs
        )

    def train_micro_step(
        self,
        pair: TrainingPair,
        *,
        loss_scale: float,
        measure_gradient_ratio: bool = False,
        output_loss_scale: float | None = None,
        state_loss_scale: float | None = None,
    ) -> TrainingMicroStepResult:
        assert loss_scale == pytest.approx(0.25)
        self.seen.append((pair.epoch, pair.record_id, pair.typo_text))
        self.noop_sequence.append(pair.is_noop)
        self.gradient_checks.append(measure_gradient_ratio)
        assert output_loss_scale is not None
        assert state_loss_scale is not None
        self.scales.append((output_loss_scale, state_loss_scale))
        return TrainingMicroStepResult(
            losses={"output": 1.0, "state": 0.0},
            total_loss=1.0,
            student_tokens=7,
        )

    def monitor(self, records: tuple[object, ...]) -> dict[str, float]:
        assert records
        self.monitor_calls += 1
        return {
            "clean_kl_nats_per_token": 0.001,
            "fineweb_edu_ppl_ratio": 1.0,
        }


class _CalibrationFailureRuntime(_Cycle2Runtime):
    def calibrate_state_weight(self, pairs: tuple[TrainingPair, ...]) -> None:
        assert len(pairs) == 8
        raise FloatingPointError("state calibration produced an invalid gradient norm")


class _LegacyGlobalStateRuntime(_Runtime):
    def train_micro_step(
        self,
        pair: TrainingPair,
        *,
        loss_scale: float,
        measure_gradient_ratio: bool = False,
        output_loss_scale: float | None = None,
        state_loss_scale: float | None = None,
    ) -> TrainingMicroStepResult:
        assert loss_scale == pytest.approx(1.0 / 32.0)
        assert measure_gradient_ratio is False
        assert output_loss_scale is None
        assert state_loss_scale is None
        self.seen.append((pair.epoch, pair.record_id, pair.typo_text))
        return TrainingMicroStepResult(
            losses={"answer": 1.0, "output": 1.0, "state": 1.0, "clean": 1.0},
            total_loss=3.0,
            student_tokens=7,
        )


def test_runner_resume_matches_uninterrupted_sample_sequence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle = _bundle(tmp_path)
    uninterrupted_runtime = _Runtime()
    uninterrupted = run_adapter_training(
        _run_config(tmp_path, config, tmp_path / "uninterrupted", resume=False),
        runtime=uninterrupted_runtime,
        data_bundle=bundle,
    )
    assert uninterrupted.optimizer_steps == 3
    assert uninterrupted.student_tokens == 42
    assert {
        path.name
        for path in (tmp_path / "uninterrupted").iterdir()
        if path.is_dir() and path.name.startswith("adapter-step-")
    } == {
        "adapter-step-000001",
        "adapter-step-000002",
        "adapter-step-000003",
    }

    interrupted_runtime = _Runtime(fail_after=4)
    output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_adapter_training(
            _run_config(tmp_path, config, output, resume=False),
            runtime=interrupted_runtime,
            data_bundle=bundle,
        )
    failed_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert failed_run["status"] == "failed"

    resumed_runtime = _Runtime()
    resumed = run_adapter_training(
        _run_config(tmp_path, config, output, resume=True),
        runtime=resumed_runtime,
        data_bundle=bundle,
    )
    assert interrupted_runtime.seen + resumed_runtime.seen == uninterrupted_runtime.seen
    assert resumed.optimizer_steps == 3
    assert resumed.checkpoint_path.is_file()
    assert (resumed.adapter_path / "adapter.txt").read_text(encoding="utf-8") == "3"
    completed = json.loads(resumed.run_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert len(completed["outputs"]["adapter"]["sha256"]) == 64
    assert completed["adapter_checkpoints"] == [
        {
            "optimizer_step": 1,
            "path": "adapter-step-000001",
            "student_tokens": 14,
        },
        {
            "optimizer_step": 2,
            "path": "adapter-step-000002",
            "student_tokens": 28,
        },
        {
            "optimizer_step": 3,
            "path": "adapter-step-000003",
            "student_tokens": 42,
        },
    ]


def test_resume_rejects_a_missing_adapter_checkpoint_before_more_training(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    bundle = _bundle(tmp_path)
    output = tmp_path / "missing-adapter"
    interrupted = _Runtime(fail_after=4)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_adapter_training(
            _run_config(tmp_path, config, output, resume=False),
            runtime=interrupted,
            data_bundle=bundle,
        )
    (output / "adapter-step-000001").rename(output / "adapter-step-000001-missing")

    resumed = _Runtime()
    with pytest.raises(RuntimeError, match="adapter checkpoint is missing at step 1"):
        run_adapter_training(
            _run_config(tmp_path, config, output, resume=True),
            runtime=resumed,
            data_bundle=bundle,
        )
    assert resumed.seen == []


def test_state_pair_generation_retries_until_typo_tokens_are_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(0)
    attempts: list[int | None] = []

    def materialize(
        candidate_source: TrainingSource,
        *,
        generator: TypoGenerator,
        epoch: int,
        variant: int = 0,
        force_noop: bool | None,
        maximum_target_stop: int | None = None,
    ) -> TrainingPair:
        del generator, force_noop, variant
        attempts.append(maximum_target_stop)
        return TrainingPair(
            record_id=candidate_source.record_id,
            clean_text=candidate_source.clean_text,
            typo_text=candidate_source.clean_text.replace("airport", "arport"),
            task=None,
            answer=None,
            metadata={"maximum_target_stop": maximum_target_stop},
            edits=(),
            is_noop=False,
            epoch=epoch,
        )

    class Runtime:
        @staticmethod
        def pair_is_usable(pair: TrainingPair) -> bool:
            limit = pair.metadata["maximum_target_stop"]
            return isinstance(limit, int) and limit <= 24

        @staticmethod
        def retained_clean_character_extent(_pair: TrainingPair) -> int:
            return 32

    monkeypatch.setattr(
        "typo_robust_training.training.runner.materialize_training_pair",
        materialize,
    )
    pair = _materialize_usable_pair(
        source=source,
        generator=_bundle(tmp_path).generator,
        epoch=0,
        force_noop=False,
        protocol=SimpleNamespace(schema_version="robustness-adapter-training-config/v3"),
        runtime=Runtime(),
    )

    assert pair.metadata["maximum_target_stop"] == 19
    assert attempts == [None, 27, 19]


def test_state_pair_generation_checks_every_reachable_word_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "abcdefghijklmnopqr " + ("tailword " * 10).strip()
    source = TrainingSource.from_dict(
        {
            "schema_version": "robustness-clean-record/v1",
            "kind": "clean",
            "record_id": "f" * 64,
            "source": "fineweb_edu",
            "source_revision": "a" * 40,
            "source_split": "train",
            "source_id": "backoff-counterexample",
            "group_id": "backoff-counterexample",
            "split": "train",
            "text": text,
            "task": None,
            "answer": None,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "normalized_content_sha256": normalized_content_sha256(text),
            "metadata": {},
            "token_count": 11,
        }
    )
    attempts: list[int | None] = []

    def materialize(
        candidate_source: TrainingSource,
        *,
        generator: TypoGenerator,
        epoch: int,
        variant: int = 0,
        force_noop: bool | None,
        maximum_target_stop: int | None = None,
    ) -> TrainingPair:
        del generator, force_noop
        attempts.append(maximum_target_stop)
        return TrainingPair(
            record_id=candidate_source.record_id,
            clean_text=candidate_source.clean_text,
            typo_text=candidate_source.clean_text,
            task=None,
            answer=None,
            metadata={"maximum_target_stop": maximum_target_stop},
            edits=(),
            is_noop=False,
            epoch=epoch,
            variant=variant,
        )

    class Runtime:
        @staticmethod
        def pair_is_usable(pair: TrainingPair) -> bool:
            return pair.metadata["maximum_target_stop"] == 18

        @staticmethod
        def retained_clean_character_extent(_pair: TrainingPair) -> int:
            return len(text)

    monkeypatch.setattr(
        "typo_robust_training.training.runner.materialize_training_pair",
        materialize,
    )
    pair = _materialize_usable_pair(
        source=source,
        generator=_bundle(tmp_path).generator,
        epoch=0,
        force_noop=False,
        protocol=SimpleNamespace(schema_version="robustness-adapter-training-config/v3"),
        runtime=Runtime(),
    )

    assert pair is not None
    assert pair.metadata["maximum_target_stop"] == 18
    assert attempts[-1] == 18


def test_state_pair_generation_retries_deterministic_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[int, int | None]] = []

    def materialize(
        candidate_source: TrainingSource,
        *,
        generator: TypoGenerator,
        epoch: int,
        variant: int = 0,
        force_noop: bool | None,
        maximum_target_stop: int | None = None,
    ) -> TrainingPair:
        del generator, force_noop
        attempts.append((variant, maximum_target_stop))
        return TrainingPair(
            record_id=candidate_source.record_id,
            clean_text=candidate_source.clean_text,
            typo_text=candidate_source.clean_text.replace("airport", "arport"),
            task=None,
            answer=None,
            metadata={"variant": variant},
            edits=(),
            is_noop=False,
            epoch=epoch,
            variant=variant,
        )

    class Runtime:
        @staticmethod
        def pair_is_usable(pair: TrainingPair) -> bool:
            return pair.variant == 1

        @staticmethod
        def retained_clean_character_extent(_pair: TrainingPair) -> int:
            return 32

    monkeypatch.setattr(
        "typo_robust_training.training.runner.materialize_training_pair",
        materialize,
    )
    pair = _materialize_usable_pair(
        source=_source(0),
        generator=_bundle(tmp_path).generator,
        epoch=0,
        force_noop=False,
        protocol=SimpleNamespace(schema_version="robustness-adapter-training-config/v3"),
        runtime=Runtime(),
    )

    assert pair is not None
    assert pair.variant == 1
    assert attempts == [(0, None), (0, 27), (0, 19), (0, 11), (1, 27)]


def test_training_stream_skips_unusable_source_without_consuming_micro_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    attempted: list[str] = []

    def materialize(**kwargs: object) -> TrainingPair | None:
        source = kwargs["source"]
        assert isinstance(source, TrainingSource)
        attempted.append(source.record_id)
        if len(attempted) == 1:
            return None
        return TrainingPair(
            record_id=source.record_id,
            clean_text=source.clean_text,
            typo_text=source.clean_text,
            task=None,
            answer=None,
            metadata={},
            edits=(),
            is_noop=True,
            epoch=int(kwargs["epoch"]),
        )

    monkeypatch.setattr(
        "typo_robust_training.training.runner._materialize_usable_pair",
        materialize,
    )
    cursor = TrainingCursor(0, 0, 1, 0, 0)
    pair, _epoch, next_cursor = _next_usable_training_pair(
        bundle=bundle,
        cursor=cursor,
        seed=42,
        protocol=SimpleNamespace(pairing_policy="exact-alternating-clean-noisy/v1"),
        runtime=object(),
    )

    assert pair.record_id == attempted[1]
    assert len(attempted) == 2
    assert next_cursor.micro_steps == cursor.micro_steps + 1
    assert next_cursor.source_index == cursor.source_index + 2


def test_cycle2_runner_enforces_exact_clean_noisy_pairs_per_optimizer_step(
    tmp_path: Path,
) -> None:
    payload = json.loads(CYCLE2_OUTPUT_CONFIG.read_text(encoding="utf-8"))
    payload["optimization"]["gradient_accumulation_steps"] = 4
    payload["optimization"]["max_optimizer_steps"] = 2
    payload["optimization"]["checkpoint_every_optimizer_steps"] = 1
    config_path = tmp_path / "cycle2-output.yaml"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    runtime = _Cycle2Runtime()
    run_adapter_training(
        AdapterTrainingRunConfig(
            condition="output-matching",
            config_path=config_path,
            training_data_dir=tmp_path,
            layer_selection_path=None,
            component_selection_path=None,
            seed=42,
            gpu_id="1",
            wandb_project=None,
            wandb_entity=None,
            output_dir=tmp_path / "cycle2-run",
        ),
        runtime=runtime,
        data_bundle=_bundle(tmp_path),
    )
    assert runtime.noop_sequence == [True, False, True, False] * 2
    assert runtime.gradient_checks == [False] * 8
    assert runtime.scales == [(0.25, 0.0)] * 8


def test_long_run_stops_at_first_optimizer_boundary_past_student_token_budget(
    tmp_path: Path,
) -> None:
    payload = json.loads(CYCLE2_OUTPUT_CONFIG.read_text(encoding="utf-8"))
    payload["schema_version"] = "robustness-adapter-training-config/v3"
    payload["optimization"]["gradient_accumulation_steps"] = 4
    payload["optimization"]["max_optimizer_steps"] = 10
    payload["optimization"]["max_student_tokens"] = 50
    payload["optimization"]["checkpoint_every_optimizer_steps"] = 1
    config_path = tmp_path / "long-output.yaml"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    runtime = _Cycle2Runtime()

    result = run_adapter_training(
        AdapterTrainingRunConfig(
            condition="output-matching",
            config_path=config_path,
            training_data_dir=tmp_path,
            layer_selection_path=None,
            component_selection_path=None,
            seed=42,
            gpu_id="1",
            wandb_project=None,
            wandb_entity=None,
            output_dir=tmp_path / "long-run",
        ),
        runtime=runtime,
        data_bundle=_bundle(tmp_path),
    )

    assert result.optimizer_steps == 2
    assert result.student_tokens == 56
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["requested_student_tokens"] == 50
    assert run["student_token_overshoot"] == 6


def test_cycle2_runner_loads_and_executes_the_frozen_tune_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(CYCLE2_OUTPUT_CONFIG.read_text(encoding="utf-8"))
    payload["optimization"]["gradient_accumulation_steps"] = 4
    payload["optimization"]["max_optimizer_steps"] = 10
    payload["optimization"]["checkpoint_every_optimizer_steps"] = 10
    config_path = tmp_path / "cycle2-output.yaml"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    study = load_evaluation_study_protocol(EVALUATION_PROTOCOL)
    loader_calls: list[dict[str, object]] = []

    def load_monitor(_root: Path, **kwargs: object) -> SimpleNamespace:
        loader_calls.append(dict(kwargs))
        return SimpleNamespace(
            records=(
                *(
                    SimpleNamespace(source="fineweb_edu", kind="clean-corpus")
                    for _ in range(study.tune_fineweb_documents)
                ),
                *(
                    SimpleNamespace(source="github_typo_corpus", kind="natural")
                    for _ in range(study.tune_natural_pairs)
                ),
            ),
            manifest_sha256="e" * 64,
        )

    monkeypatch.setattr(
        "typo_robust_training.evaluation.data.load_evaluation_corpus_bundle",
        load_monitor,
    )
    runtime = _Cycle2Runtime()
    result = run_adapter_training(
        AdapterTrainingRunConfig(
            condition="output-matching",
            config_path=config_path,
            training_data_dir=tmp_path,
            layer_selection_path=None,
            component_selection_path=None,
            seed=42,
            gpu_id="1",
            wandb_project=None,
            wandb_entity=None,
            output_dir=tmp_path / "monitored-run",
            evaluation_protocol_path=EVALUATION_PROTOCOL,
            monitor_data_dir=tmp_path / "monitor-data",
        ),
        runtime=runtime,
        data_bundle=_bundle(tmp_path),
    )

    assert runtime.monitor_calls == 1
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["monitor"] == {
        "protocol_sha256": study.config_sha256,
        "data_sha256": "e" * 64,
        "records": study.tune_fineweb_documents + study.tune_natural_pairs,
        "interval_optimizer_steps": 10,
        "task_accuracy_allowed": False,
    }
    assert loader_calls == [
        {
            "evaluation_role": "tune",
            "study_protocol_sha256": study.config_sha256,
            "access_binding_sha256": study.config_sha256,
            "experiment_binding_sha256": study.config_sha256,
            "output_dir": tmp_path / "monitored-run",
            "confirm_sealed_role": False,
            "resume": False,
        }
    ]


def test_state_calibration_failure_is_recorded_before_training_starts(tmp_path: Path) -> None:
    output_dir = tmp_path / "calibration-failure"
    evidence = ResidualStateEvidence(
        selected_window=(0, 6),
        state_layers=tuple(range(6)),
        policy="frozen-causal-window/v1",
        layer_selection_sha256="d" * 64,
        validation_sha256="e" * 64,
        evidence_sha256="f" * 64,
    )

    with pytest.raises(
        FloatingPointError,
        match="state calibration produced an invalid gradient norm",
    ):
        run_adapter_training(
            AdapterTrainingRunConfig(
                condition="localized-state-distillation",
                config_path=CYCLE3_CAUSAL_CONFIG,
                training_data_dir=tmp_path,
                layer_selection_path=None,
                component_selection_path=None,
                seed=42,
                gpu_id="1",
                wandb_project=None,
                wandb_entity=None,
                output_dir=output_dir,
            ),
            runtime=_CalibrationFailureRuntime(),
            data_bundle=_bundle(tmp_path),
            evidence=evidence,
        )

    failed = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["cursor"] == {
        "epoch": 0,
        "micro_steps": 0,
        "optimizer_steps": 0,
        "source_index": 0,
        "student_tokens": 0,
    }
    assert failed["error"] == {
        "type": "FloatingPointError",
        "message": "state calibration produced an invalid gradient norm",
    }


def test_accumulation_scales_use_total_token_and_edited_coordinate_denominators() -> None:
    scales = normalized_accumulation_scales(
        output_token_counts=(2, 6, 4, 8),
        state_coordinate_counts=(0, 1, 0, 3),
        state_active=True,
    )
    assert [scale.output for scale in scales] == pytest.approx([0.1, 0.3, 0.2, 0.4])
    assert [scale.state for scale in scales] == pytest.approx([0.0, 0.25, 0.0, 0.75])


def test_cycle2_telemetry_reports_the_exact_accumulation_objective() -> None:
    micro_rows = [
        {
            "student_tokens": 3,
            "edit_count": 0,
            "is_noop": True,
            "total_loss": 2.0,
            "losses": {
                "output": 2.0,
                "state": 0.0,
                "weighted_state": 0.0,
                "output_accumulation_scale": 0.1,
                "state_accumulation_scale": 0.0,
                "backward_contribution": 0.2,
            },
        },
        {
            "student_tokens": 7,
            "edit_count": 1,
            "is_noop": False,
            "total_loss": 6.5,
            "losses": {
                "output": 6.0,
                "state": 1.0,
                "weighted_state": 0.5,
                "output_accumulation_scale": 0.9,
                "state_accumulation_scale": 1.0,
                "backward_contribution": 5.9,
            },
        },
    ]

    metrics = _optimizer_step_telemetry(
        micro_rows,
        optimizer_step=1,
        micro_steps=2,
        cumulative_student_tokens=10,
        gradient_norm=0.25,
        learning_rate=1e-4,
        elapsed_seconds=1.0,
        runtime=_Runtime(),
    )

    assert metrics["train/total_loss"] == pytest.approx(6.1)
    assert metrics["train/objective/output"] == pytest.approx(5.6)
    assert metrics["train/objective/state"] == pytest.approx(1.0)
    assert metrics["train/objective/weighted_state"] == pytest.approx(0.5)


def test_monitor_violation_streak_is_reconstructed_from_completed_metrics(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    for step, clean_kl in ((10, 0.01), (20, 0.04), (30, 0.05)):
        path = work / "metrics" / f"optimizer-step-{step:06d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "aggregates": {
                        "monitor/clean_kl_nats_per_token": clean_kl,
                        "monitor/fineweb_edu_ppl_ratio": 1.0,
                    }
                }
            ),
            encoding="utf-8",
        )

    assert (
        _monitor_violation_streak(
            work_dir=work,
            optimizer_steps=30,
            monitor_interval=10,
            clean_kl_limit=0.03,
            ppl_limit=1.02,
        )
        == 2
    )


def test_runner_uploads_only_aggregate_optimizer_step_telemetry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tracker = _Tracker()
    runtime = _Runtime()
    result = run_adapter_training(
        _run_config(tmp_path, config, tmp_path / "tracked", resume=False, tracking=True),
        runtime=runtime,
        data_bundle=_bundle(tmp_path),
        tracker=tracker,
    )

    assert result.optimizer_steps == 3
    assert [step for step, _ in tracker.logged] == [1, 2, 3]
    for step, metrics in tracker.logged:
        assert metrics["train/optimizer_step"] == step
        assert metrics["train/total_loss"] == pytest.approx(1.0)
        assert metrics["train/loss/noisy_language_model"] == pytest.approx(1.0)
        assert metrics["train/gradient_norm"] == pytest.approx(0.25)
        assert metrics["train/learning_rate"] == pytest.approx(0.0002)
        assert metrics["train/student_tokens_this_step"] == 14
        assert metrics["train/student_tokens"] == step * 14
        assert metrics["train/student_tokens_per_second"] > 0
        assert all(isinstance(value, (int, float)) for value in metrics.values())
        assert not any("record" in name or "text" in name or "prompt" in name for name in metrics)
    assert tracker.finished == [
        (
            "completed",
            {
                "optimizer_steps": 3,
                "micro_steps": 6,
                "student_tokens": 42,
            },
        )
    ]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["tracking"] == tracker.provenance()


def test_legacy_global_state_readme_config_reaches_wandb_tracker_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _Tracker()
    presentations: list[WandbRunPresentation] = []

    def start_tracker(**kwargs: object) -> _Tracker:
        presentation = kwargs["presentation"]
        assert isinstance(presentation, WandbRunPresentation)
        presentations.append(presentation)
        return tracker

    monkeypatch.setattr(
        "typo_robust_training.training.runner.start_wandb_training_tracker",
        start_tracker,
    )
    result = run_adapter_training(
        AdapterTrainingRunConfig(
            condition="global-state-alignment",
            config_path=LEGACY_GLOBAL_STATE_CONFIG,
            training_data_dir=tmp_path,
            layer_selection_path=None,
            component_selection_path=None,
            seed=42,
            gpu_id="3",
            wandb_project="typo-robustness-training",
            wandb_entity="fixture-entity",
            output_dir=tmp_path / "legacy-global-state",
        ),
        runtime=_LegacyGlobalStateRuntime(),
        data_bundle=_bundle(tmp_path),
    )

    assert result.optimizer_steps == 100
    assert len(presentations) == 1
    presentation = presentations[0]
    assert presentation.name == (
        "Legacy · Global relative-MSE state pilot · Gemma-3-4B-IT · 100 steps · seed 42"
    )
    assert "condition:global-state-alignment" in presentation.tags
    assert "State layers:" not in presentation.notes
