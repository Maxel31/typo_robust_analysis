"""W&B tracking is resumable, scalar-only, and never persists credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typo_robust_training.training.tracking import start_wandb_training_tracker


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
        run_id = (
            str(resume_from).split("?", 1)[0]
            if resume_from is not None
            else str(kwargs["id"])
        )
        run = _Run(run_id)
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


def test_wandb_logs_scalars_and_resumes_same_hash_bound_run(tmp_path: Path) -> None:
    module = _Wandb()
    secret = "fixture-secret-that-must-never-be-written"
    tracker = start_wandb_training_tracker(
        output_dir=tmp_path,
        project="typo-robustness-training",
        entity="fixture-entity",
        bindings=_bindings(),
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
    assert secret not in json.dumps(module.calls[0], sort_keys=True)
    assert module.runs[0].logged == [
        (1, {"train/optimizer_step": 1, "train/total_loss": 0.75})
    ]
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
        resume=True,
        resume_optimizer_step=1,
        environment={"WANDB_API_KEY": secret},
        wandb_module=module,
        run_id_factory=lambda: "must-not-be-used",
    )
    assert module.calls[1]["resume_from"] == "fixed-run-id?_step=1"
    assert "id" not in module.calls[1]
    resumed.finish(status="failed", summary={"optimizer_steps": 1})
    assert module.runs[1].finished == [1]


def test_wandb_resume_rejects_binding_or_boundary_drift(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="resume boundary differs"):
        start_wandb_training_tracker(
            output_dir=tmp_path,
            project="typo-robustness-training",
            entity=None,
            bindings=_bindings(),
            resume=True,
            resume_optimizer_step=0,
            environment={"WANDB_API_KEY": "secret"},
            wandb_module=module,
        )
