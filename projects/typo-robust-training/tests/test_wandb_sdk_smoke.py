"""Exercise the locked real W&B SDK's rewind argument without network access."""

from __future__ import annotations

import inspect
from pathlib import Path

import wandb


def test_locked_wandb_sdk_accepts_resume_from_query(tmp_path: Path) -> None:
    assert wandb.__version__ == "0.28.1"
    assert "resume_from" in inspect.signature(wandb.init).parameters

    run_id = "locked-sdk-resume-smoke"
    settings = wandb.Settings(silent=True)
    initial = wandb.init(
        project="typo-robustness-sdk-smoke",
        id=run_id,
        mode="offline",
        dir=str(tmp_path),
        reinit="create_new",
        settings=settings,
    )
    assert initial is not None
    initial.log({"train/optimizer_step": 1}, step=1)
    initial.finish()

    resumed = wandb.init(
        project="typo-robustness-sdk-smoke",
        resume_from=f"{run_id}?_step=0",
        mode="offline",
        dir=str(tmp_path),
        reinit="create_new",
        settings=settings,
    )
    assert resumed is not None
    assert resumed.id == run_id
    resumed.finish()
