"""W&B tracking is resumable, scalar-only, and never persists credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typo_robust_training.training.tracking import (
    build_wandb_run_presentation,
    start_wandb_training_tracker,
)


class _Run:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.url = f"https://wandb.example/entity/project/runs/{run_id}"
        self.summary: dict[str, object] = {}
        self.defined: list[tuple[str, dict[str, object]]] = []
        self.logged: list[tuple[int | None, dict[str, int | float]]] = []
        self.finished: list[int] = []

    def define_metric(self, name: str, **kwargs: object) -> None:
        self.defined.append((name, dict(kwargs)))

    def log(self, metrics: dict[str, int | float], *, step: int | None = None) -> None:
        self.logged.append((step, dict(metrics)))

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class _Wandb:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.runs: list[_Run] = []

    def init(self, **kwargs: object) -> _Run:
        self.calls.append(dict(kwargs))
        resume_from = kwargs.get("resume_from")
        run_id = str(resume_from).split("?", 1)[0] if resume_from is not None else str(kwargs["id"])
        run = _Run(run_id)
        self.runs.append(run)
        return run


class _FailingMetricRun(_Run):
    def define_metric(self, name: str, **kwargs: object) -> None:
        del name, kwargs
        raise RuntimeError("injected metric registration failure")


class _FailingWandb(_Wandb):
    def init(self, **kwargs: object) -> _Run:
        self.calls.append(dict(kwargs))
        run = _FailingMetricRun(str(kwargs["id"]))
        self.runs.append(run)
        return run


def _bindings() -> dict[str, object]:
    return {
        "condition": "localized-state-distillation",
        "config_sha256": "a" * 64,
        "training_data_sha256": "b" * 64,
        "localization_sha256": "c" * 64,
        "seed": 42,
        "gpu_id": "3",
    }


def test_wandb_presentation_names_the_scientific_arm_and_operation() -> None:
    proposed = build_wandb_run_presentation(
        condition="localized-state-distillation",
        schema_version="robustness-adapter-training-config/v2",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=100,
        state_gradient_ratio=0.05,
        state_layers=tuple(range(1, 7)),
    )
    assert proposed.name == (
        "Proposed method · Causal-window localized state distillation · "
        "L1–6 · Gemma-3-4B-IT · 100 steps · seed 42"
    )
    assert proposed.group == "Cycle 2 · Gemma-3-4B-IT · 100 steps"
    assert proposed.job_type == "proposed-causal-window"
    assert proposed.tags == (
        "typo-robustness",
        "cycle:2",
        "arm:causal-window-localized-state-distillation",
        "role:proposed",
        "model:gemma-3-4b-it",
        "budget:100-steps",
        "state-gradient-ratio:0.05",
    )
    assert "Activation Patching" in proposed.notes
    assert "edited-word-final" in proposed.notes

    baseline = build_wandb_run_presentation(
        condition="output-matching",
        schema_version="robustness-adapter-training-config/v2",
        model="google/gemma-3-4b-it",
        seed=44,
        max_optimizer_steps=300,
        state_gradient_ratio=None,
        state_layers=(),
    )
    assert baseline.name == (
        "Kojima baseline · Output-distribution matching · Gemma-3-4B-IT · 300 steps · seed 44"
    )
    assert baseline.job_type == "baseline-output-matching"
    assert "arm:kojima-output-matching" in baseline.tags
    assert "state-gradient-ratio" not in " ".join(baseline.tags)

    legacy = build_wandb_run_presentation(
        condition="localized-state-distillation",
        schema_version="robustness-adapter-training-config/v1",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=100,
        state_gradient_ratio=None,
        state_layers=(2, 3, 4, 5, 6),
    )
    assert legacy.name.startswith("Legacy · Component-level state distillation")
    assert legacy.job_type == "legacy-component-state-pilot"
    assert "arm:Legacy" in legacy.tags
    assert "relative-MSE" in legacy.notes


def test_long_run_presentation_uses_the_student_token_budget() -> None:
    presentation = build_wandb_run_presentation(
        condition="output-matching",
        schema_version="robustness-adapter-training-config/v3",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=10_000,
        max_student_tokens=64_000_000,
        state_gradient_ratio=None,
        state_layers=(),
    )

    assert "64M tokens" in presentation.name
    assert "64M tokens" in presentation.group
    assert "budget:64000000-student-tokens" in presentation.tags
    assert "10000 steps" not in presentation.name


def test_wandb_requires_environment_credential_without_persisting_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        start_wandb_training_tracker(
            output_dir=tmp_path,
            project="typo-robustness-training",
            entity=None,
            bindings=_bindings(),
            resume=False,
            resume_optimizer_step=0,
            environment={},
            wandb_module=_Wandb(),
            run_id_factory=lambda: "fixed-run-id",
        )

    assert not (tmp_path / "wandb_run.json").exists()


def test_wandb_post_init_failure_finishes_and_records_the_remote_run(tmp_path: Path) -> None:
    module = _FailingWandb()

    with pytest.raises(RuntimeError, match="metric registration failure"):
        start_wandb_training_tracker(
            output_dir=tmp_path,
            project="typo-robustness-training",
            entity=None,
            bindings=_bindings(),
            resume=False,
            resume_optimizer_step=0,
            environment={"WANDB_API_KEY": "secret"},
            wandb_module=module,
            run_id_factory=lambda: "failed-run-id",
        )

    assert module.runs[0].finished == [1]
    metadata = json.loads((tmp_path / "wandb_run.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "failed-run-id"
    assert metadata["status"] == "failed"


def test_wandb_logs_scalars_and_resumes_same_hash_bound_run(tmp_path: Path) -> None:
    module = _Wandb()
    secret = "fixture-secret-that-must-never-be-written"
    presentation = build_wandb_run_presentation(
        condition="localized-state-distillation",
        schema_version="robustness-adapter-training-config/v2",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=100,
        state_gradient_ratio=0.05,
        state_layers=tuple(range(1, 7)),
    )
    tracker = start_wandb_training_tracker(
        output_dir=tmp_path,
        project="typo-robustness-training",
        entity="fixture-entity",
        bindings=_bindings(),
        presentation=presentation,
        resume=False,
        resume_optimizer_step=0,
        environment={"WANDB_API_KEY": secret},
        wandb_module=module,
        run_id_factory=lambda: "fixed-run-id",
    )
    tracker.log_optimizer_step(
        {"train/optimizer_step": 1, "train/total_loss": 0.75},
        optimizer_step=1,
    )
    tracker.finish(status="completed", summary={"optimizer_steps": 1})

    assert module.calls[0]["id"] == "fixed-run-id"
    assert module.calls[0]["resume"] == "never"
    assert module.calls[0]["project"] == "typo-robustness-training"
    assert module.calls[0]["entity"] == "fixture-entity"
    assert module.calls[0]["name"] == presentation.name
    assert module.calls[0]["group"] == presentation.group
    assert module.calls[0]["job_type"] == presentation.job_type
    assert module.calls[0]["tags"] == list(presentation.tags)
    assert module.calls[0]["notes"] == presentation.notes
    assert secret not in json.dumps(module.calls[0], sort_keys=True)
    assert module.runs[0].logged == [(1, {"train/optimizer_step": 1, "train/total_loss": 0.75})]
    assert module.runs[0].finished == [0]

    metadata_path = tmp_path / "wandb_run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "fixed-run-id"
    assert metadata["last_logged_optimizer_step"] == 1
    assert secret not in metadata_path.read_text(encoding="utf-8")

    resumed = start_wandb_training_tracker(
        output_dir=tmp_path,
        project="typo-robustness-training",
        entity="fixture-entity",
        bindings=_bindings(),
        presentation=presentation,
        resume=True,
        resume_optimizer_step=1,
        environment={"WANDB_API_KEY": secret},
        wandb_module=module,
        run_id_factory=lambda: "must-not-be-used",
    )
    assert module.calls[1]["id"] == "fixed-run-id"
    assert module.calls[1]["resume"] == "must"
    assert "resume_from" not in module.calls[1]
    resumed.finish(status="failed", summary={"optimizer_steps": 1})
    assert module.runs[1].finished == [1]


def test_wandb_resume_rejects_binding_drift_or_remote_history_behind_checkpoint(
    tmp_path: Path,
) -> None:
    module = _Wandb()
    tracker = start_wandb_training_tracker(
        output_dir=tmp_path,
        project="typo-robustness-training",
        entity=None,
        bindings=_bindings(),
        resume=False,
        resume_optimizer_step=0,
        environment={"WANDB_API_KEY": "secret"},
        wandb_module=module,
        run_id_factory=lambda: "fixed-run-id",
    )
    tracker.log_optimizer_step(
        {"train/optimizer_step": 1, "train/total_loss": 1.0},
        optimizer_step=1,
    )
    tracker.finish(status="completed", summary={"optimizer_steps": 1})

    changed = {**_bindings(), "seed": 43}
    with pytest.raises(ValueError, match="bindings differ"):
        start_wandb_training_tracker(
            output_dir=tmp_path,
            project="typo-robustness-training",
            entity=None,
            bindings=changed,
            resume=True,
            resume_optimizer_step=1,
            environment={"WANDB_API_KEY": "secret"},
            wandb_module=module,
        )
    replayed = start_wandb_training_tracker(
        output_dir=tmp_path,
        project="typo-robustness-training",
        entity=None,
        bindings=_bindings(),
        resume=True,
        resume_optimizer_step=0,
        environment={"WANDB_API_KEY": "secret"},
        wandb_module=module,
    )
    assert module.calls[-1]["id"] == "fixed-run-id"
    assert module.calls[-1]["resume"] == "must"
    replayed.log_optimizer_step(
        {"train/optimizer_step": 1, "train/total_loss": 1.0},
        optimizer_step=1,
    )
    assert module.runs[-1].logged == []
    replayed.log_optimizer_step(
        {"train/optimizer_step": 2, "train/total_loss": 0.9},
        optimizer_step=2,
    )
    assert module.runs[-1].logged == [(2, {"train/optimizer_step": 2, "train/total_loss": 0.9})]
    replayed.finish(status="failed", summary={"optimizer_steps": 2})

    with pytest.raises(ValueError, match="behind the local checkpoint"):
        start_wandb_training_tracker(
            output_dir=tmp_path,
            project="typo-robustness-training",
            entity=None,
            bindings=_bindings(),
            resume=True,
            resume_optimizer_step=3,
            environment={"WANDB_API_KEY": "secret"},
            wandb_module=module,
        )
