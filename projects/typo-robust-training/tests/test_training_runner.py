"""The shared runner checkpoints at optimizer boundaries and resumes exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import typo_robust_training.training.runner as runner_module
from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.training.data import TrainingDataBundle
from typo_robust_training.training.pairs import TrainingPair, TrainingSource
from typo_robust_training.training.runner import (
    AdapterTrainingRunConfig,
    TrainingMicroStepResult,
    run_adapter_training,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs/baselines/noisy-language-model.yaml"
GLOBAL_STATE_CONFIG = PROJECT_ROOT / "configs/baselines/global-state-alignment.yaml"
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


def _global_state_config(tmp_path: Path) -> Path:
    payload = json.loads(GLOBAL_STATE_CONFIG.read_text(encoding="utf-8"))
    payload["optimization"]["gradient_accumulation_steps"] = 2
    payload["optimization"]["max_optimizer_steps"] = 3
    payload["optimization"]["checkpoint_every_optimizer_steps"] = 1
    path = tmp_path / "global-state-training.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Runtime:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.seen: list[tuple[int, str, str]] = []
        self.optimizer_steps = 0

    def train_micro_step(self, pair: TrainingPair, *, loss_scale: float) -> TrainingMicroStepResult:
        assert loss_scale == pytest.approx(0.5)
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
        return {
            "runtime": "offline-training-fixture/v1",
            "gpu_id": "3",
            "decoder_layers": 4,
        }


def _run_config(
    tmp_path: Path,
    config: Path,
    output: Path,
    *,
    resume: bool,
    tracking: bool = False,
    condition: str = "noisy-language-model",
):
    return AdapterTrainingRunConfig(
        condition=condition,
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


def test_runner_starts_wandb_with_a_self_explanatory_series_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    tracker = _Tracker()

    def start_tracker(**kwargs: object) -> _Tracker:
        captured.update(kwargs)
        return tracker

    monkeypatch.setattr(runner_module, "start_wandb_training_tracker", start_tracker)
    result = run_adapter_training(
        _run_config(
            tmp_path,
            _config(tmp_path),
            tmp_path / "descriptive-series",
            resume=False,
            tracking=True,
        ),
        runtime=_Runtime(),
        data_bundle=_bundle(tmp_path),
    )

    presentation = captured["presentation"]
    assert presentation.name == (
        "Historical baseline · Noisy-language-model training · Gemma-3-4B-IT · 3 steps · seed 42"
    )
    assert presentation.group == "Historical Cycle 1 · Gemma-3-4B-IT · 3 steps"
    assert "gpu_id" not in captured["bindings"]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["gpu_id"] == "3"


def test_runner_names_all_layer_state_control_from_runtime_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    tracker = _Tracker()

    def start_tracker(**kwargs: object) -> _Tracker:
        captured.update(kwargs)
        return tracker

    monkeypatch.setattr(runner_module, "start_wandb_training_tracker", start_tracker)
    run_adapter_training(
        _run_config(
            tmp_path,
            _global_state_config(tmp_path),
            tmp_path / "global-state-series",
            resume=False,
            tracking=True,
            condition="global-state-alignment",
        ),
        runtime=_Runtime(),
        data_bundle=_bundle(tmp_path),
    )

    presentation = captured["presentation"]
    assert presentation.name == (
        "Historical control · Global relative-MSE state alignment · "
        "L0–3 · Gemma-3-4B-IT · 3 steps · seed 42"
    )
