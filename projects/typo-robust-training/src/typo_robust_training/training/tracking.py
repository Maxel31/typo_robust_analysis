"""Scalar-only, hash-bound Weights & Biases training telemetry."""

from __future__ import annotations

import importlib
import json
import math
import os
import secrets
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol


_SCHEMA = "robustness-wandb-training-run/v1"
_METADATA_FIELDS = {
    "schema_version",
    "provider",
    "project",
    "entity",
    "run_id",
    "url",
    "bindings",
    "last_logged_optimizer_step",
    "status",
}
_FORBIDDEN_METRIC_FRAGMENTS = ("record_id", "text", "prompt", "api_key", "secret")


class TrainingTracker(Protocol):
    """The runner-facing boundary for one external scalar tracker."""

    def log_optimizer_step(
        self,
        metrics: Mapping[str, int | float],
        *,
        optimizer_step: int,
    ) -> None: ...

    def finish(self, *, status: str, summary: Mapping[str, int | float]) -> None: ...

    def provenance(self) -> Mapping[str, object]: ...


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_bindings(bindings: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("W&B bindings must be a non-empty object")
    try:
        encoded = json.dumps(dict(bindings), sort_keys=True, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("W&B bindings must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError("W&B bindings must be an object")
    if any(
        fragment in str(key).lower()
        for key in normalized
        for fragment in ("api_key", "credential", "secret", "token")
    ):
        raise ValueError("W&B bindings must not contain credentials")
    return normalized


def _scalar_metrics(values: Mapping[str, int | float]) -> dict[str, int | float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("W&B metrics must be a non-empty object")
    normalized: dict[str, int | float] = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or not name
            or any(fragment in name.lower() for fragment in _FORBIDDEN_METRIC_FRAGMENTS)
        ):
            raise ValueError("W&B metric name is empty or may expose raw training data")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("W&B metrics must contain only numeric scalars")
        if not math.isfinite(float(value)):
            raise ValueError("W&B metrics must be finite")
        normalized[name] = value
    return normalized


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError("W&B resume requires wandb_run.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("W&B run metadata is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _METADATA_FIELDS:
        raise ValueError("W&B run metadata fields differ")
    if payload.get("schema_version") != _SCHEMA or payload.get("provider") != "wandb":
        raise ValueError("W&B run metadata schema differs")
    return payload


class WandbTrainingTracker:
    """One online W&B run whose local identity is safe to resume exactly."""

    def __init__(
        self,
        *,
        run: Any,
        metadata_path: Path,
        metadata: Mapping[str, object],
        sdk_version: str | None,
    ) -> None:
        self._run = run
        self._metadata_path = metadata_path
        self._metadata = dict(metadata)
        self._sdk_version = sdk_version
        self._finished = False

    def log_optimizer_step(
        self,
        metrics: Mapping[str, int | float],
        *,
        optimizer_step: int,
    ) -> None:
        if self._finished:
            raise RuntimeError("W&B run is already finished")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step <= 0
        ):
            raise ValueError("W&B optimizer step must be positive")
        prior = self._metadata["last_logged_optimizer_step"]
        if not isinstance(prior, int) or optimizer_step != prior + 1:
            raise ValueError("W&B optimizer steps must be consecutive")
        payload = _scalar_metrics(metrics)
        if payload.get("train/optimizer_step") != optimizer_step:
            raise ValueError("W&B metrics must contain the matching optimizer step")
        self._run.log(payload, step=optimizer_step)
        self._metadata.update(
            {"last_logged_optimizer_step": optimizer_step, "status": "running"}
        )
        _write_json(self._metadata_path, self._metadata)

    def finish(self, *, status: str, summary: Mapping[str, int | float]) -> None:
        if self._finished:
            return
        if status not in {"completed", "failed"}:
            raise ValueError("W&B finish status must be completed or failed")
        values = _scalar_metrics(summary)
        self._run.summary.update({f"summary/{name}": value for name, value in values.items()})
        self._run.summary["run/status"] = status
        self._run.finish(exit_code=0 if status == "completed" else 1)
        self._metadata["status"] = status
        _write_json(self._metadata_path, self._metadata)
        self._finished = True

    def provenance(self) -> Mapping[str, object]:
        return {
            "provider": "wandb",
            "sdk_version": self._sdk_version,
            "project": self._metadata["project"],
            "entity": self._metadata["entity"],
            "run_id": self._metadata["run_id"],
            "url": self._metadata["url"],
            "metadata_path": str(self._metadata_path),
            "raw_training_data_uploaded": False,
        }


def start_wandb_training_tracker(
    *,
    output_dir: Path,
    project: str,
    entity: str | None,
    bindings: Mapping[str, object],
    resume: bool,
    resume_optimizer_step: int,
    environment: Mapping[str, str] | None = None,
    wandb_module: Any | None = None,
    run_id_factory: Callable[[], str] | None = None,
) -> WandbTrainingTracker:
    """Start or rewind one W&B run without copying its credential into artifacts."""

    if not isinstance(project, str) or not project.strip():
        raise ValueError("W&B project must be non-empty")
    if entity is not None and (not isinstance(entity, str) or not entity.strip()):
        raise ValueError("W&B entity must be null or non-empty")
    if (
        isinstance(resume_optimizer_step, bool)
        or not isinstance(resume_optimizer_step, int)
        or resume_optimizer_step < 0
    ):
        raise ValueError("W&B resume optimizer step must be non-negative")
    values = os.environ if environment is None else environment
    api_key = values.get("WANDB_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("W&B online tracking requires WANDB_API_KEY")
    resolved_entity = entity or values.get("WANDB_ENTITY")
    if resolved_entity is not None and not resolved_entity:
        resolved_entity = None
    frozen_bindings = _canonical_bindings(bindings)
    root = Path(output_dir).resolve()
    metadata_path = root / "wandb_run.json"
    factory = run_id_factory or (lambda: secrets.token_hex(8))

    if resume:
        metadata = _load_metadata(metadata_path)
        if (
            metadata["project"] != project
            or metadata["entity"] != resolved_entity
            or metadata["bindings"] != frozen_bindings
        ):
            raise ValueError("W&B resume bindings differ")
        prior_step = metadata["last_logged_optimizer_step"]
        if not isinstance(prior_step, int) or prior_step < resume_optimizer_step:
            raise ValueError("W&B history is behind the local checkpoint")
        run_id = metadata["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("W&B run ID is invalid")
        init_resume = {"resume_from": f"{run_id}?_step={resume_optimizer_step}"}
    else:
        if metadata_path.exists():
            raise FileExistsError("fresh W&B run cannot overwrite wandb_run.json")
        run_id = factory()
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("W&B run ID factory returned an invalid ID")
        metadata = {
            "schema_version": _SCHEMA,
            "provider": "wandb",
            "project": project,
            "entity": resolved_entity,
            "run_id": run_id,
            "url": None,
            "bindings": frozen_bindings,
            "last_logged_optimizer_step": 0,
            "status": "initializing",
        }
        init_resume = {"id": run_id, "resume": "never"}

    module = wandb_module or importlib.import_module("wandb")
    local_dir = root / ".wandb"
    local_dir.mkdir(parents=True, exist_ok=True)
    condition = str(frozen_bindings.get("condition", "adapter-training"))
    seed = frozen_bindings.get("seed")
    run = module.init(
        project=project,
        entity=resolved_entity,
        name=f"{condition}-seed-{seed}",
        group=condition,
        job_type="adapter-training",
        tags=["typo-robustness", condition],
        config=frozen_bindings,
        dir=str(local_dir),
        mode="online",
        reinit="create_new",
        **init_resume,
    )
    if run is None or getattr(run, "id", None) != run_id:
        raise RuntimeError("W&B initialized a different or missing run ID")
    run.define_metric("train/optimizer_step")
    run.define_metric("train/*", step_metric="train/optimizer_step")
    run.define_metric("system/*", step_metric="train/optimizer_step")
    metadata.update(
        {
            "url": getattr(run, "url", None),
            "last_logged_optimizer_step": resume_optimizer_step,
            "status": "running",
        }
    )
    _write_json(metadata_path, metadata)
    sdk_version = getattr(module, "__version__", None)
    return WandbTrainingTracker(
        run=run,
        metadata_path=metadata_path,
        metadata=metadata,
        sdk_version=str(sdk_version) if sdk_version is not None else None,
    )


__all__ = [
    "TrainingTracker",
    "WandbTrainingTracker",
    "start_wandb_training_tracker",
]
