"""W&B tracking is resumable, scalar-only, and never persists credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typo_robust_training.training.tracking import (
    WandbRunPresentation,
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


def _bindings() -> dict[str, object]:
    return {
        "condition": "localized-state-distillation",
        "config_sha256": "a" * 64,
        "training_data_sha256": "b" * 64,
        "localization_sha256": "c" * 64,
        "seed": 42,
        "gpu_id": "3",
    }


def test_wandb_presentation_uses_self_explanatory_scientific_names() -> None:
    proposed = build_wandb_run_presentation(
        condition="localized-state-distillation",
        schema_version="robustness-adapter-training-config/v2",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=100,
        state_layers=tuple(range(6)),
    )
    assert proposed.name == (
        "Proposed method · Causal-window localized state distillation · "
        "L0–5 · Gemma-3-4B-IT · 100 steps · seed 42"
    )
    assert proposed.group == "Confirmatory comparison · Gemma-3-4B-IT · 100 steps"
    assert proposed.job_type == "proposed-causal-window-state-distillation"
    assert "role:proposed-method" in proposed.tags
    assert all(tag not in {"arm:B1", "arm:T1", "arm:C1", "arm:C2"} for tag in proposed.tags)

    baseline = build_wandb_run_presentation(
        condition="output-matching",
        schema_version="robustness-adapter-training-config/v2",
        model="google/gemma-3-4b-it",
        seed=44,
        max_optimizer_steps=300,
        state_layers=(),
    )
    assert baseline.name == (
        "Kojima baseline · Output-distribution matching · Gemma-3-4B-IT · 300 steps · seed 44"
    )
    assert baseline.job_type == "kojima-output-distribution-matching"

    historical = build_wandb_run_presentation(
        condition="localized-state-distillation",
        schema_version="robustness-adapter-training-config/v1",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=100,
        state_layers=(2, 3, 4, 5, 6),
    )
    assert historical.name.startswith(
        "Historical ablation · Component-level relative-MSE state distillation · L2–6"
    )
    assert historical.group.startswith("Historical Cycle 1 ·")
    assert "not the confirmatory method" in historical.notes


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
    presentation = build_wandb_run_presentation(
        condition="localized-state-distillation",
        schema_version="robustness-adapter-training-config/v1",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=100,
        state_layers=(2, 3, 4, 5, 6),
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
    assert metadata["presentation"] == {
        "group": presentation.group,
        "job_type": presentation.job_type,
        "name": presentation.name,
        "notes": presentation.notes,
        "tags": list(presentation.tags),
    }
    assert secret not in metadata_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="presentation differs"):
        start_wandb_training_tracker(
            output_dir=tmp_path,
            project="typo-robustness-training",
            entity="fixture-entity",
            bindings=_bindings(),
            presentation=WandbRunPresentation(
                name="Changed name",
                group=presentation.group,
                job_type=presentation.job_type,
                tags=presentation.tags,
                notes=presentation.notes,
            ),
            resume=True,
            resume_optimizer_step=1,
            environment={"WANDB_API_KEY": secret},
            wandb_module=module,
        )

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
    assert module.calls[1]["resume_from"] == "fixed-run-id?_step=1"
    assert "id" not in module.calls[1]
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
    rewound = start_wandb_training_tracker(
        output_dir=tmp_path,
        project="typo-robustness-training",
        entity=None,
        bindings=_bindings(),
        resume=True,
        resume_optimizer_step=0,
        environment={"WANDB_API_KEY": "secret"},
        wandb_module=module,
    )
    assert module.calls[-1]["resume_from"] == "fixed-run-id?_step=0"
    rewound.finish(status="failed", summary={"optimizer_steps": 0})

    with pytest.raises(ValueError, match="behind the local checkpoint"):
        start_wandb_training_tracker(
            output_dir=tmp_path,
            project="typo-robustness-training",
            entity=None,
            bindings=_bindings(),
            resume=True,
            resume_optimizer_step=2,
            environment={"WANDB_API_KEY": "secret"},
            wandb_module=module,
        )
