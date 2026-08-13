"""Shared hash-bound runner for every frozen adapter-training condition."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

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
from typo_robust_training.training.pairs import TrainingPair, materialize_training_pair


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
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


@dataclass(frozen=True, slots=True)
class AdapterTrainingRunConfig:
    condition: str
    config_path: Path
    training_data_dir: Path
    layer_selection_path: Path | None
    component_selection_path: Path | None
    seed: int
    gpu_id: str
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


def run_adapter_training(
    config: AdapterTrainingRunConfig,
    *,
    runtime: AdapterTrainingRuntime | None = None,
    data_bundle: TrainingDataBundle | None = None,
    evidence: LocalizationEvidence | None = None,
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
    run_base = {
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
    }
    _write_json(run_path, {**run_base, "status": "running", "started_at": _now()})
    try:
        while cursor.optimizer_steps < protocol.max_optimizer_steps:
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
            _write_json(_metrics_step_path(work_dir, cursor.optimizer_steps), step_payload)
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
        _write_json(
            run_path,
            {
                **run_base,
                "status": "completed",
                "completed_at": _now(),
                "optimizer_steps": cursor.optimizer_steps,
                "micro_steps": cursor.micro_steps,
                "student_tokens": cursor.student_tokens,
                "outputs": {
                    "checkpoint.json": {"sha256": _sha256_file(checkpoint_path)},
                    "metrics.jsonl": {"sha256": _sha256_file(metrics_path)},
                },
            },
        )
    except Exception as exc:
        _write_json(
            run_path,
            {
                **run_base,
                "status": "failed",
                "failed_at": _now(),
                "cursor": cursor.as_dict(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
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
