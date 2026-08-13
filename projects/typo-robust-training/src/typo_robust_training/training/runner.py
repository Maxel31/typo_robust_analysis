"""Shared hash-bound runner for every frozen adapter-training condition."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typo_robust_training.integrity import sha256_tree
from typo_robust_training.training.checkpoint import (
    TrainingCursor,
    load_training_checkpoint,
    next_training_source,
    write_training_checkpoint,
)
from typo_robust_training.training.config import (
    AdapterTrainingProtocol,
    load_adapter_training_config,
)
from typo_robust_training.training.data import (
    TrainingDataBundle,
    load_training_data_bundle,
)
from typo_robust_training.training.evidence import (
    LocalizationEvidence,
    load_localization_evidence,
)
from typo_robust_training.training.json_io import write_json_atomic
from typo_robust_training.training.pairs import TrainingPair, materialize_training_pair
from typo_robust_training.training.tracking import (
    TrainingTracker,
    build_wandb_run_presentation,
    start_wandb_training_tracker,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AdapterTrainingRunConfig:
    condition: str
    config_path: Path
    training_data_dir: Path
    layer_selection_path: Path | None
    component_selection_path: Path | None
    seed: int
    gpu_id: str
    wandb_project: str | None
    wandb_entity: str | None
    output_dir: Path
    resume: bool = False


@dataclass(frozen=True, slots=True)
class TrainingMicroStepResult:
    losses: Mapping[str, float]
    total_loss: float
    student_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.student_tokens, bool)
            or not isinstance(self.student_tokens, int)
            or self.student_tokens <= 0
        ):
            raise ValueError("training micro-step student_tokens must be positive")


class AdapterTrainingRuntime(Protocol):
    def train_micro_step(
        self, pair: TrainingPair, *, loss_scale: float
    ) -> TrainingMicroStepResult: ...

    def optimizer_step(self, *, max_grad_norm: float) -> tuple[float, float]: ...

    def zero_grad(self) -> None: ...

    def save_state(self, path: Path) -> None: ...

    def load_state(self, path: Path) -> None: ...

    def save_adapter(self, path: Path) -> None: ...

    def provenance(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class AdapterTrainingRunResult:
    optimizer_steps: int
    micro_steps: int
    student_tokens: int
    adapter_path: Path
    checkpoint_path: Path
    metrics_path: Path
    run_path: Path


def _load_evidence(
    config: AdapterTrainingRunConfig,
    *,
    protocol: AdapterTrainingProtocol,
) -> LocalizationEvidence | None:
    localized = protocol.condition == "localized-state-distillation"
    if localized:
        if config.layer_selection_path is None or config.component_selection_path is None:
            raise ValueError("localized training requires layer and component selections")
        return load_localization_evidence(
            layer_selection_path=config.layer_selection_path,
            component_selection_path=config.component_selection_path,
            model=protocol.model,
            model_revision=protocol.model_revision,
            decoder_layers=34,
            mlp_intermediate_size=10240,
            attention_heads=8,
        )
    if config.layer_selection_path is not None or config.component_selection_path is not None:
        raise ValueError("baseline training cannot consume localization evidence")
    return None


def _metrics_step_path(work_dir: Path, optimizer_step: int) -> Path:
    return work_dir / "metrics" / f"optimizer-step-{optimizer_step:06d}.json"


def _assemble_metrics(path: Path, *, work_dir: Path, optimizer_steps: int) -> None:
    rows: list[bytes] = []
    for step in range(1, optimizer_steps + 1):
        source = _metrics_step_path(work_dir, step)
        if not source.is_file():
            raise RuntimeError(f"training metrics are missing optimizer step {step}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows.append(
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(b"".join(rows))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _optimizer_step_telemetry(
    micro_rows: list[dict[str, object]],
    *,
    optimizer_step: int,
    micro_steps: int,
    cumulative_student_tokens: int,
    gradient_norm: float,
    learning_rate: float,
    elapsed_seconds: float,
    runtime: AdapterTrainingRuntime,
) -> dict[str, int | float]:
    if not micro_rows:
        raise ValueError("optimizer telemetry requires micro-batches")
    elapsed = max(float(elapsed_seconds), 1e-12)
    step_tokens = sum(int(row["student_tokens"]) for row in micro_rows)
    metrics: dict[str, int | float] = {
        "train/optimizer_step": optimizer_step,
        "train/micro_steps": micro_steps,
        "train/student_tokens": cumulative_student_tokens,
        "train/student_tokens_this_step": step_tokens,
        "train/total_loss": sum(float(row["total_loss"]) for row in micro_rows) / len(micro_rows),
        "train/gradient_norm": float(gradient_norm),
        "train/learning_rate": float(learning_rate),
        "train/mean_edit_count": sum(int(row["edit_count"]) for row in micro_rows)
        / len(micro_rows),
        "train/noop_fraction": sum(bool(row["is_noop"]) for row in micro_rows) / len(micro_rows),
        "train/step_seconds": elapsed,
        "train/student_tokens_per_second": step_tokens / elapsed,
    }
    losses: dict[str, list[float]] = {}
    for row in micro_rows:
        values = row["losses"]
        if not isinstance(values, Mapping):
            raise TypeError("micro-batch loss telemetry must be an object")
        for name, value in values.items():
            losses.setdefault(str(name), []).append(float(value))
    metrics.update(
        {f"train/loss/{name}": sum(values) / len(values) for name, values in sorted(losses.items())}
    )
    telemetry = getattr(runtime, "telemetry", None)
    if callable(telemetry):
        values = telemetry()
        if not isinstance(values, Mapping):
            raise TypeError("training runtime telemetry must be an object")
        for name, value in values.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError("training runtime telemetry must contain numeric scalars")
            metrics[f"system/{name}"] = value
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError("optimizer telemetry contains a non-finite scalar")
    return metrics


def run_adapter_training(
    config: AdapterTrainingRunConfig,
    *,
    runtime: AdapterTrainingRuntime | None = None,
    data_bundle: TrainingDataBundle | None = None,
    evidence: LocalizationEvidence | None = None,
    tracker: TrainingTracker | None = None,
) -> AdapterTrainingRunResult:
    """Train one explicit condition and checkpoint only completed optimizer steps."""

    if not isinstance(config, AdapterTrainingRunConfig):
        raise TypeError("training run config must be AdapterTrainingRunConfig")
    protocol = load_adapter_training_config(config.config_path)
    if config.condition != protocol.condition:
        raise ValueError("training command condition differs from its config")
    if config.seed not in protocol.seed_inventory:
        raise ValueError("training seed is outside the frozen seed inventory")
    if not config.gpu_id or "," in config.gpu_id:
        raise ValueError("--gpu-id must name one physical GPU")
    if config.wandb_project is None and config.wandb_entity is not None:
        raise ValueError("W&B entity requires a W&B project")
    bundle = data_bundle or load_training_data_bundle(
        config.training_data_dir,
        protocol=protocol,
        seed=config.seed,
    )
    if evidence is None:
        evidence = _load_evidence(config, protocol=protocol)
    elif protocol.condition != "localized-state-distillation":
        raise ValueError("baseline training cannot consume injected localization evidence")
    localization_sha = evidence.component_selection_sha256 if evidence is not None else None
    bindings = {
        "config_sha256": protocol.config_sha256,
        "training_data_sha256": bundle.training_data_sha256,
        "localization_sha256": localization_sha,
        "seed": config.seed,
    }

    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume:
        raise FileExistsError(f"training output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / ".train-adapter-work"
    work_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    metrics_path = output_dir / "metrics.jsonl"
    run_path = output_dir / "run.json"
    adapter_path = output_dir / "adapter"
    if config.resume and not checkpoint_path.is_file():
        raise ValueError("--resume requires a completed optimizer-boundary checkpoint")

    if runtime is None:
        from typo_robust_training.training.runtime import HuggingFaceAdapterTrainingRuntime

        runtime = HuggingFaceAdapterTrainingRuntime(
            protocol=protocol,
            seed=config.seed,
            gpu_id=config.gpu_id,
            evidence=evidence,
        )
    provenance = dict(runtime.provenance())
    cursor = TrainingCursor(0, 0, 0, 0, 0)
    if config.resume:
        checkpoint = load_training_checkpoint(
            checkpoint_path,
            expected_bindings=bindings,
        )
        cursor = checkpoint.cursor
        runtime.load_state(checkpoint.state_path)
    if cursor.optimizer_steps > protocol.max_optimizer_steps:
        raise ValueError("training checkpoint exceeds the configured optimizer steps")
    runtime.zero_grad()
    started_at = _now()
    run_base: dict[str, object] = {
        "schema_version": "robustness-adapter-training-run/v1",
        "operation": f"train-{protocol.condition}",
        "condition": protocol.condition,
        "config_sha256": protocol.config_sha256,
        "training_data_sha256": bundle.training_data_sha256,
        "data_identity_sha256": bundle.data_identity_sha256,
        "localization_sha256": localization_sha,
        "seed": config.seed,
        "gpu_id": config.gpu_id,
        "resume": config.resume,
        "python": platform.python_version(),
        "runtime": provenance,
        "tracking": (
            dict(tracker.provenance())
            if tracker is not None
            else {
                "provider": "wandb" if config.wandb_project is not None else "disabled",
                "project": config.wandb_project,
                "entity": config.wandb_entity,
            }
        ),
    }
    write_json_atomic(run_path, {**run_base, "status": "running", "started_at": started_at})
    tracking_finished = False
    try:
        if tracker is None and config.wandb_project is not None:
            if protocol.loss_weights["state"] <= 0.0:
                state_layers: tuple[int, ...] = ()
            elif evidence is not None:
                state_layers = evidence.adapter_layers
            else:
                decoder_layers = provenance.get("decoder_layers")
                if (
                    isinstance(decoder_layers, bool)
                    or not isinstance(decoder_layers, int)
                    or decoder_layers <= 0
                ):
                    raise ValueError(
                        "state-training W&B presentation requires the decoder-layer count"
                    )
                state_layers = tuple(range(decoder_layers))
            tracker = start_wandb_training_tracker(
                output_dir=output_dir,
                project=config.wandb_project,
                entity=config.wandb_entity,
                bindings={
                    **bindings,
                    "condition": protocol.condition,
                },
                presentation=build_wandb_run_presentation(
                    condition=protocol.condition,
                    schema_version=protocol.schema_version,
                    model=protocol.model,
                    seed=config.seed,
                    max_optimizer_steps=protocol.max_optimizer_steps,
                    state_layers=state_layers,
                ),
                resume=config.resume,
                resume_optimizer_step=cursor.optimizer_steps,
            )
        if tracker is not None:
            run_base["tracking"] = dict(tracker.provenance())
            write_json_atomic(
                run_path,
                {**run_base, "status": "running", "started_at": started_at},
            )
        while cursor.optimizer_steps < protocol.max_optimizer_steps:
            step_started = time.perf_counter()
            micro_rows: list[dict[str, object]] = []
            for accumulation_index in range(protocol.gradient_accumulation_steps):
                source, epoch, next_cursor = next_training_source(
                    bundle.sources,
                    cursor=cursor,
                    seed=config.seed,
                )
                pair = materialize_training_pair(
                    source,
                    generator=bundle.generator,
                    epoch=epoch,
                )
                result = runtime.train_micro_step(
                    pair,
                    loss_scale=1.0 / protocol.gradient_accumulation_steps,
                )
                if not isinstance(result, TrainingMicroStepResult):
                    raise TypeError("training runtime returned an invalid micro-step result")
                cursor = replace(
                    next_cursor,
                    student_tokens=cursor.student_tokens + result.student_tokens,
                )
                micro_rows.append(
                    {
                        "accumulation_index": accumulation_index,
                        "epoch": epoch,
                        "record_id": pair.record_id,
                        "is_noop": pair.is_noop,
                        "edit_count": len(pair.edits),
                        "student_tokens": result.student_tokens,
                        "total_loss": result.total_loss,
                        "losses": dict(result.losses),
                    }
                )
            grad_norm, learning_rate = runtime.optimizer_step(max_grad_norm=protocol.max_grad_norm)
            runtime.zero_grad()
            cursor = replace(cursor, optimizer_steps=cursor.optimizer_steps + 1)
            step_payload = {
                "schema_version": "robustness-adapter-training-step/v1",
                "optimizer_step": cursor.optimizer_steps,
                "micro_steps": cursor.micro_steps,
                "student_tokens": cursor.student_tokens,
                "gradient_norm": grad_norm,
                "learning_rate": learning_rate,
                "micro_batches": micro_rows,
            }
            telemetry = _optimizer_step_telemetry(
                micro_rows,
                optimizer_step=cursor.optimizer_steps,
                micro_steps=cursor.micro_steps,
                cumulative_student_tokens=cursor.student_tokens,
                gradient_norm=grad_norm,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - step_started,
                runtime=runtime,
            )
            step_payload["aggregates"] = telemetry
            write_json_atomic(_metrics_step_path(work_dir, cursor.optimizer_steps), step_payload)
            if tracker is not None:
                tracker.log_optimizer_step(
                    telemetry,
                    optimizer_step=cursor.optimizer_steps,
                )
            if (
                cursor.optimizer_steps % protocol.checkpoint_every_optimizer_steps == 0
                or cursor.optimizer_steps == protocol.max_optimizer_steps
            ):
                state_path = work_dir / f"runtime-state-step-{cursor.optimizer_steps:06d}.pt"
                runtime.save_state(state_path)
                write_training_checkpoint(
                    checkpoint_path,
                    cursor=cursor,
                    state_path=state_path,
                    bindings=bindings,
                )
        _assemble_metrics(
            metrics_path,
            work_dir=work_dir,
            optimizer_steps=cursor.optimizer_steps,
        )
        runtime.save_adapter(adapter_path)
        if tracker is not None:
            tracker.finish(
                status="completed",
                summary={
                    "optimizer_steps": cursor.optimizer_steps,
                    "micro_steps": cursor.micro_steps,
                    "student_tokens": cursor.student_tokens,
                },
            )
            tracking_finished = True
        write_json_atomic(
            run_path,
            {
                **run_base,
                "status": "completed",
                "completed_at": _now(),
                "optimizer_steps": cursor.optimizer_steps,
                "micro_steps": cursor.micro_steps,
                "student_tokens": cursor.student_tokens,
                "outputs": {
                    "adapter": {"sha256": sha256_tree(adapter_path)},
                    "checkpoint.json": {"sha256": _sha256_file(checkpoint_path)},
                    "metrics.jsonl": {"sha256": _sha256_file(metrics_path)},
                },
            },
        )
    except Exception as exc:
        tracking_error: dict[str, str] | None = None
        if tracker is not None and not tracking_finished:
            try:
                tracker.finish(
                    status="failed",
                    summary={
                        "optimizer_steps": cursor.optimizer_steps,
                        "micro_steps": cursor.micro_steps,
                        "student_tokens": cursor.student_tokens,
                    },
                )
            except Exception as finish_exc:
                tracking_error = {
                    "type": type(finish_exc).__name__,
                    "message": str(finish_exc),
                }
        write_json_atomic(
            run_path,
            {
                **run_base,
                "status": "failed",
                "failed_at": _now(),
                "cursor": cursor.as_dict(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                **({"tracking_error": tracking_error} if tracking_error is not None else {}),
            },
        )
        raise
    return AdapterTrainingRunResult(
        optimizer_steps=cursor.optimizer_steps,
        micro_steps=cursor.micro_steps,
        student_tokens=cursor.student_tokens,
        adapter_path=adapter_path,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        run_path=run_path,
    )


__all__ = [
    "AdapterTrainingRunConfig",
    "AdapterTrainingRunResult",
    "AdapterTrainingRuntime",
    "TrainingMicroStepResult",
    "run_adapter_training",
]
