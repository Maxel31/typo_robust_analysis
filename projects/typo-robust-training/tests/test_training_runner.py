"""The shared runner checkpoints at optimizer boundaries and resumes exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.training.data import TrainingDataBundle
from typo_robust_training.training.pairs import TrainingPair, TrainingSource
from typo_robust_training.training.runner import (
    AdapterTrainingRunConfig,
    TrainingMicroStepResult,
    TrainingMicroStepScales,
    _optimizer_step_telemetry,
    normalized_accumulation_scales,
    run_adapter_training,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs/baselines/noisy-language-model.yaml"
CYCLE2_OUTPUT_CONFIG = PROJECT_ROOT / "configs/cycle2/gemma4b-output-matching-100step.yaml"
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
    assert runtime.gradient_checks == [False, True, False, False] * 2
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
