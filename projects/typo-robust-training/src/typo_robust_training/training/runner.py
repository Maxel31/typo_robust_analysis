"""Shared hash-bound runner for every frozen adapter-training condition."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typo_robust_training.data.perturb import TypoGenerator, eligible_word_spans
from typo_robust_training.integrity import sha256_tree
from typo_robust_training.training.checkpoint import (
    EpochSourceOrderCache,
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
    ResidualStateEvidence,
    load_localization_evidence,
    load_residual_state_evidence,
)
from typo_robust_training.training.json_io import write_json_atomic as _write_json
from typo_robust_training.training import methods as training_methods
from typo_robust_training.training.pairs import (
    TrainingPair,
    TrainingSource,
    materialize_training_pair,
)
from typo_robust_training.training.provenance import validate_condition_evidence
from typo_robust_training.training.tracking import (
    TrainingTracker,
    build_wandb_run_presentation,
    start_wandb_training_tracker,
)


_MAX_SYNTHETIC_PAIR_VARIANTS = 32
_TRAINING_MONITOR_INTERVAL_OPTIMIZER_STEPS = 10


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
    evaluation_protocol_path: Path | None = None
    monitor_data_dir: Path | None = None
    window_validation_path: Path | None = None
    method_evidence_sha256: str | None = None
    probe_selection_path: Path | None = None
    state_gate_path: Path | None = None


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


@dataclass(frozen=True, slots=True)
class TrainingMicroStepScales:
    output: float
    state: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in (self.output, self.state)
        ):
            raise ValueError("training micro-step scales must be finite and non-negative")


def normalized_accumulation_scales(
    *,
    output_token_counts: Sequence[int],
    state_coordinate_counts: Sequence[int],
    state_active: bool,
) -> tuple[TrainingMicroStepScales, ...]:
    """Weight per-record means by the accumulation batch's exact denominators."""

    output_counts = tuple(output_token_counts)
    state_counts = tuple(state_coordinate_counts)
    if not output_counts or len(output_counts) != len(state_counts):
        raise ValueError("accumulation loss-count vectors differ")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in output_counts
    ):
        raise ValueError("each accumulation record needs output supervision")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in state_counts
    ):
        raise ValueError("accumulation state-coordinate counts are invalid")
    output_total = sum(output_counts)
    state_total = sum(state_counts)
    if state_active and state_total <= 0:
        raise ValueError("state training accumulation has no edited coordinates")
    if not state_active and state_total != 0:
        raise ValueError("output-only accumulation cannot contain state coordinates")
    return tuple(
        TrainingMicroStepScales(
            output=output_count / output_total,
            state=(state_count / state_total if state_active else 0.0),
        )
        for output_count, state_count in zip(output_counts, state_counts, strict=True)
    )


def factorial_group_balanced_accumulation_scales(
    *,
    output_token_counts: Sequence[int],
    is_noop_rows: Sequence[bool],
) -> tuple[TrainingMicroStepScales, ...]:
    """Give clean and noisy factorial targets exactly one half of one update.

    The horizon arms deliberately expose far fewer noisy targets than clean
    targets.  A single token denominator would therefore turn the frozen 1:1
    document mixture into an almost entirely clean objective.  Normalize
    selected targets within each row group, then assign the two preregistered
    groups equal mass.  The one-half constants are implied by the frozen 1:1
    pairing policy and are not tunable method coefficients.
    """

    output_counts = tuple(output_token_counts)
    noop_rows = tuple(is_noop_rows)
    if not output_counts or len(output_counts) != len(noop_rows):
        raise ValueError("factorial accumulation counts and row groups differ")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in output_counts
    ):
        raise ValueError("each factorial accumulation row needs output supervision")
    if any(type(is_noop) is not bool for is_noop in noop_rows):
        raise ValueError("factorial accumulation row groups must be boolean")
    expected_noop_rows = tuple(index % 2 == 0 for index in range(len(noop_rows)))
    if noop_rows != expected_noop_rows:
        raise ValueError("factorial accumulation must alternate clean and noisy rows")
    clean_count = sum(noop_rows)
    noisy_count = len(noop_rows) - clean_count
    if clean_count == 0 or noisy_count == 0 or clean_count != noisy_count:
        raise ValueError("factorial accumulation requires an exact 1:1 clean/noisy mixture")
    clean_total = sum(
        count for count, is_noop in zip(output_counts, noop_rows, strict=True) if is_noop
    )
    noisy_total = sum(
        count for count, is_noop in zip(output_counts, noop_rows, strict=True) if not is_noop
    )
    if clean_total <= 0 or noisy_total <= 0:  # Defensive: positive rows imply this.
        raise ValueError("factorial accumulation is missing one target group")
    return tuple(
        TrainingMicroStepScales(
            output=(
                0.5 * output_count / clean_total if is_noop else 0.5 * output_count / noisy_total
            ),
            state=0.0,
        )
        for output_count, is_noop in zip(output_counts, noop_rows, strict=True)
    )


class AdapterTrainingRuntime(Protocol):
    def train_micro_step(
        self,
        pair: TrainingPair,
        *,
        loss_scale: float,
        measure_gradient_ratio: bool = False,
        output_loss_scale: float | None = None,
        state_loss_scale: float | None = None,
    ) -> TrainingMicroStepResult: ...

    def prepare_accumulation(
        self,
        pairs: Sequence[TrainingPair],
    ) -> Sequence[TrainingMicroStepScales]: ...

    def calibrate_state_weight(
        self,
        pairs: Sequence[TrainingPair],
    ) -> Mapping[str, object]: ...

    def verify_resume_state_calibration(
        self,
        path: Path,
        pairs: Sequence[TrainingPair],
    ) -> None: ...

    def optimizer_step(self, *, max_grad_norm: float) -> tuple[float, float]: ...

    def zero_grad(self) -> None: ...

    def save_state(self, path: Path) -> None: ...

    def load_state(
        self,
        path: Path,
        *,
        expected_state_calibration: Mapping[str, object] | None = None,
    ) -> None: ...

    def save_adapter(self, path: Path) -> None: ...

    def provenance(self) -> Mapping[str, object]: ...


def _resolved_method_presentation_layers(
    *, condition: str, method: training_methods.ResolvedTrainingMethod
) -> tuple[int, ...]:
    """Expose the scientific intervention coordinate, not the LoRA support."""

    return (
        method.state_layers
        if condition
        in {
            "probe-transition-single-layer-state-distillation",
            "probe-semantic-subspace-distillation",
        }
        else method.adapter_layers
    )


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
) -> (
    LocalizationEvidence
    | ResidualStateEvidence
    | training_methods.ProbeTransitionTrainingEvidence
    | training_methods.ProbeTransitionStateTrainingEvidence
    | training_methods.ProbeSemanticSubspaceTrainingEvidence
    | None
):
    if protocol.condition in {
        "probe-transition-output-matching",
        "probe-semantic-subspace-distillation",
    } | set(training_methods.PROBE_FACTORIAL_CONDITIONS):
        if (
            config.probe_selection_path is None
            or protocol.decoder_layers is None
            or config.layer_selection_path is not None
            or config.window_validation_path is not None
            or config.component_selection_path is not None
            or config.state_gate_path is not None
        ):
            message = (
                "semantic training requires only one kill evidence artifact"
                if protocol.condition == "probe-semantic-subspace-distillation"
                else "probe-transition training requires only one probe selection artifact"
            )
            raise ValueError(message)
        loader = (
            training_methods.load_probe_semantic_subspace_training_evidence
            if protocol.condition == "probe-semantic-subspace-distillation"
            else training_methods.load_probe_transition_training_evidence
        )
        return loader(
            config.probe_selection_path,
            model=protocol.model,
            model_revision=protocol.model_revision,
            decoder_layers=protocol.decoder_layers,
        )
    if protocol.condition == "probe-transition-single-layer-state-distillation":
        if (
            config.state_gate_path is None
            or config.probe_selection_path is not None
            or config.layer_selection_path is not None
            or config.window_validation_path is not None
            or config.component_selection_path is not None
            or protocol.decoder_layers is None
        ):
            raise ValueError("probe-transition state training requires only one gate artifact")
        return training_methods.load_probe_transition_state_training_evidence(
            config.state_gate_path,
            model=protocol.model,
            model_revision=protocol.model_revision,
            decoder_layers=protocol.decoder_layers,
        )
    if config.probe_selection_path is not None or config.state_gate_path is not None:
        raise ValueError("non-probe training cannot consume probe-transition evidence")
    if not protocol.schema_version.endswith("/v1"):
        if config.component_selection_path is not None:
            raise ValueError("cycle-2 training cannot consume component selection evidence")
        if protocol.condition in {
            "localized-state-distillation",
            "random-window-state-distillation",
        }:
            if config.layer_selection_path is None or protocol.decoder_layers is None:
                raise ValueError("window state training requires a layer selection")
            return load_residual_state_evidence(
                layer_selection_path=config.layer_selection_path,
                window_validation_path=config.window_validation_path,
                model=protocol.model,
                model_revision=protocol.model_revision,
                decoder_layers=protocol.decoder_layers,
                policy=protocol.state_window_policy,
            )
        if config.layer_selection_path is not None or config.window_validation_path is not None:
            raise ValueError("output/all-layer training cannot consume a layer selection")
        return None
    if protocol.condition == "localized-state-distillation":
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
    if (
        config.layer_selection_path is not None
        or config.window_validation_path is not None
        or config.component_selection_path is not None
    ):
        raise ValueError("baseline training cannot consume localization evidence")
    return None


def _forced_noop(protocol: AdapterTrainingProtocol, *, micro_step: int) -> bool | None:
    if protocol.pairing_policy == "exact-alternating-clean-noisy/v1":
        return micro_step % 2 == 0
    return None


def _state_calibration_pairs(
    *,
    bundle: TrainingDataBundle,
    protocol: AdapterTrainingProtocol,
    seed: int,
    runtime: AdapterTrainingRuntime,
) -> tuple[TrainingPair, ...]:
    """Replay the initial stream and retain the configured number of noisy pairs."""

    required = protocol.calibration_micro_batches
    if required == 0:
        return ()
    cursor = TrainingCursor(0, 0, 0, 0, 0)
    order_cache = EpochSourceOrderCache(bundle.sources, seed=seed)
    pairs: list[TrainingPair] = []
    while len(pairs) < required:
        pair, _epoch, cursor = _next_usable_training_pair(
            bundle=bundle,
            cursor=cursor,
            seed=seed,
            protocol=protocol,
            runtime=runtime,
            order_cache=order_cache,
        )
        if not pair.is_noop:
            pairs.append(pair)
    return tuple(pairs)


def _materialize_usable_pair(
    *,
    source: TrainingSource,
    generator: TypoGenerator,
    epoch: int,
    force_noop: bool | None,
    protocol: AdapterTrainingProtocol,
    runtime: AdapterTrainingRuntime,
) -> TrainingPair | None:
    """Keep every cycle-2/3 typo target inside the tokenizer-retained prefix."""

    pair = materialize_training_pair(
        source,
        generator=generator,
        epoch=epoch,
        force_noop=force_noop,
    )
    if protocol.schema_version.endswith("/v1") or pair.is_noop:
        return pair
    validator = getattr(runtime, "pair_is_usable", None)
    if not callable(validator) or validator(pair):
        return pair
    if source.kind == "natural":
        return None
    extent_reader = getattr(runtime, "retained_clean_character_extent", None)
    if not callable(extent_reader):
        raise TypeError("cycle-2 runtime cannot constrain typo targets to retained tokens")
    maximum_target_stop = extent_reader(pair)
    if (
        isinstance(maximum_target_stop, bool)
        or not isinstance(maximum_target_stop, int)
        or maximum_target_stop <= 0
    ):
        raise ValueError("retained token window contains no eligible typo target")
    candidate_stops = tuple(
        sorted(
            {
                stop
                for _start, stop in eligible_word_spans(
                    source.clean_text,
                    minimum_letters=generator.minimum_word_letters,
                )
                if stop <= maximum_target_stop
            },
            reverse=True,
        )
    )
    for target_stop in candidate_stops:
        try:
            constrained = materialize_training_pair(
                source,
                generator=generator,
                epoch=epoch,
                variant=0,
                force_noop=force_noop,
                maximum_target_stop=target_stop,
            )
        except ValueError as exc:
            if "contains no eligible typo target" not in str(exc):
                raise
            continue
        if validator(constrained):
            return constrained
    for variant in range(1, _MAX_SYNTHETIC_PAIR_VARIANTS):
        for target_stop in candidate_stops:
            constrained = materialize_training_pair(
                source,
                generator=generator,
                epoch=epoch,
                variant=variant,
                force_noop=force_noop,
                maximum_target_stop=target_stop,
            )
            if validator(constrained):
                return constrained
    return None


def _next_usable_training_pair(
    *,
    bundle: TrainingDataBundle,
    cursor: TrainingCursor,
    seed: int,
    protocol: AdapterTrainingProtocol,
    runtime: AdapterTrainingRuntime,
    order_cache: EpochSourceOrderCache | None = None,
) -> tuple[TrainingPair, int, TrainingCursor]:
    """Advance past unusable sources without consuming a training micro-step."""

    source_cursor = cursor
    for _attempt in range(len(bundle.sources)):
        source, epoch, next_cursor = next_training_source(
            bundle.sources,
            cursor=source_cursor,
            seed=seed,
            order_cache=order_cache,
        )
        pair = _materialize_usable_pair(
            source=source,
            generator=bundle.generator,
            epoch=epoch,
            force_noop=_forced_noop(protocol, micro_step=cursor.micro_steps),
            protocol=protocol,
            runtime=runtime,
        )
        if pair is not None:
            return pair, epoch, next_cursor
        source_cursor = replace(next_cursor, micro_steps=cursor.micro_steps)
    raise ValueError("training source inventory contains no usable pair for this micro-step")


def _monitor_violation_streak(
    *,
    work_dir: Path,
    optimizer_steps: int,
    monitor_interval: int,
    clean_kl_limit: float,
    ppl_limit: float,
) -> int:
    """Reconstruct the consecutive completed unsafe-monitor count on resume."""

    if optimizer_steps <= 0 or monitor_interval <= 0:
        return 0
    streak = 0
    step = optimizer_steps - (optimizer_steps % monitor_interval)
    while step > 0:
        path = _metrics_step_path(work_dir, step)
        if not path.is_file():
            break
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregates = payload.get("aggregates")
        if not isinstance(aggregates, Mapping):
            break
        clean_kl = aggregates.get("monitor/clean_kl_nats_per_token")
        ppl_ratio = aggregates.get("monitor/fineweb_edu_ppl_ratio")
        if not isinstance(clean_kl, (int, float)) or not isinstance(ppl_ratio, (int, float)):
            break
        if float(clean_kl) <= clean_kl_limit and float(ppl_ratio) <= ppl_limit:
            break
        streak += 1
        step -= monitor_interval
    return streak


def _metrics_step_path(work_dir: Path, optimizer_step: int) -> Path:
    return work_dir / "metrics" / f"optimizer-step-{optimizer_step:06d}.json"


def _expected_adapter_checkpoint_steps(
    *,
    optimizer_steps: int,
    interval: int,
) -> tuple[int, ...]:
    """List every periodic checkpoint plus the current completed boundary."""

    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps < 0
        or isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
    ):
        raise ValueError("adapter checkpoint inventory inputs are invalid")
    if optimizer_steps == 0:
        return ()
    periodic = set(range(interval, optimizer_steps + 1, interval))
    periodic.add(optimizer_steps)
    return tuple(sorted(periodic))


def _validate_adapter_checkpoints(
    output_dir: Path,
    *,
    optimizer_steps: int,
    interval: int,
) -> tuple[int, ...]:
    """Fail before training when a resumed checkpoint inventory is incomplete."""

    expected = _expected_adapter_checkpoint_steps(
        optimizer_steps=optimizer_steps,
        interval=interval,
    )
    for step in expected:
        if not (output_dir / f"adapter-step-{step:06d}").is_dir():
            raise RuntimeError(f"training adapter checkpoint is missing at step {step}")
    return expected


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
    loss_rows: list[Mapping[str, object]] = []
    for row in micro_rows:
        values = row["losses"]
        if not isinstance(values, Mapping):
            raise TypeError("micro-batch loss telemetry must be an object")
        loss_rows.append(values)
        for name, value in values.items():
            losses.setdefault(str(name), []).append(float(value))
    metrics.update(
        {f"train/loss/{name}": sum(values) / len(values) for name, values in sorted(losses.items())}
    )
    accumulation_fields = {
        "output",
        "state",
        "weighted_state",
        "output_accumulation_scale",
        "state_accumulation_scale",
        "backward_contribution",
    }
    if loss_rows and all(accumulation_fields <= row.keys() for row in loss_rows):
        metrics["train/total_loss"] = sum(float(row["backward_contribution"]) for row in loss_rows)
        metrics["train/objective/output"] = sum(
            float(row["output"]) * float(row["output_accumulation_scale"]) for row in loss_rows
        )
        metrics["train/objective/state"] = sum(
            float(row["state"]) * float(row["state_accumulation_scale"]) for row in loss_rows
        )
        metrics["train/objective/weighted_state"] = sum(
            float(row["weighted_state"]) * float(row["state_accumulation_scale"])
            for row in loss_rows
        )
        if all(type(row.get("is_noop")) is bool for row in micro_rows):
            # Preserve the exact within-group token weighting used by the
            # backward objective, but divide out each group's assigned mass.
            # These two diagnostic series make noisy-learning progress visible
            # even when the mixed objective is dominated by normal batch noise.
            for suffix, is_noop in (("clean", True), ("noisy", False)):
                group = tuple(
                    loss
                    for row, loss in zip(micro_rows, loss_rows, strict=True)
                    if row["is_noop"] is is_noop
                )
                group_mass = sum(float(row["output_accumulation_scale"]) for row in group)
                if group and group_mass > 0.0:
                    metrics[f"train/objective/output_{suffix}_mean"] = (
                        sum(
                            float(row["output"]) * float(row["output_accumulation_scale"])
                            for row in group
                        )
                        / group_mass
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
    evidence: (
        LocalizationEvidence
        | ResidualStateEvidence
        | training_methods.ProbeTransitionTrainingEvidence
        | training_methods.ProbeTransitionStateTrainingEvidence
        | training_methods.ProbeSemanticSubspaceTrainingEvidence
        | None
    ) = None,
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
    monitor_records: tuple[object, ...] = ()
    monitor_protocol_sha: str | None = None
    monitor_data_sha: str | None = None
    monitor_interval = 0
    monitor_clean_kl_limit = monitor_ppl_limit = math.inf
    supplied_monitor = (
        config.evaluation_protocol_path is not None or config.monitor_data_dir is not None
    )
    if supplied_monitor:
        if config.evaluation_protocol_path is None or config.monitor_data_dir is None:
            raise ValueError("training monitor requires protocol and frozen data together")
        from typo_robust_training.evaluation.data import load_evaluation_corpus_bundle
        from typo_robust_training.evaluation.study import load_evaluation_study_protocol

        study = load_evaluation_study_protocol(config.evaluation_protocol_path)
        monitor_bundle = load_evaluation_corpus_bundle(
            config.monitor_data_dir,
            evaluation_role="tune",
            study_protocol_sha256=study.config_sha256,
            access_binding_sha256=study.config_sha256,
            experiment_binding_sha256=study.config_sha256,
            output_dir=config.output_dir,
            confirm_sealed_role=False,
            resume=config.resume,
        )
        monitor_records = tuple(monitor_bundle.records)
        clean_count = sum(record.source == "fineweb_edu" for record in monitor_records)
        paired_count = sum(record.kind == "natural" for record in monitor_records)
        if (
            clean_count != study.tune_fineweb_documents
            or paired_count != study.tune_natural_pairs
            or len(monitor_records) != clean_count + paired_count
        ):
            raise ValueError("training monitor frozen record inventory differs")
        monitor_protocol_sha = study.config_sha256
        monitor_data_sha = monitor_bundle.manifest_sha256
        # Monitor scheduling is an operational training-run concern.  Evaluation
        # v1.4 freezes the tune inventory and safety gates, but deliberately does
        # not own the cadence (see robustness_evaluation_protocol_v1.md).
        monitor_interval = _TRAINING_MONITOR_INTERVAL_OPTIMIZER_STEPS
        monitor_clean_kl_limit = float(study.gates["maximum_clean_kl_nats_per_token"])
        monitor_ppl_limit = float(study.gates["maximum_clean_ppl_ratio"])
    elif not protocol.schema_version.endswith("/v1") and runtime is None:
        raise ValueError("cycle-2 training requires the frozen T0 monitor data")
    if evidence is None:
        evidence = _load_evidence(config, protocol=protocol)
    elif protocol.condition in {
        "probe-transition-output-matching",
        "probe-transition-single-layer-state-distillation",
        "probe-semantic-subspace-distillation",
    } | set(training_methods.PROBE_FACTORIAL_CONDITIONS):
        expected_type = {
            "probe-transition-output-matching": (training_methods.ProbeTransitionTrainingEvidence),
            "probe-transition-single-layer-state-distillation": (
                training_methods.ProbeTransitionStateTrainingEvidence
            ),
            "probe-semantic-subspace-distillation": (
                training_methods.ProbeSemanticSubspaceTrainingEvidence
            ),
            **{
                condition: training_methods.ProbeTransitionTrainingEvidence
                for condition in training_methods.PROBE_FACTORIAL_CONDITIONS
            },
        }[protocol.condition]
        if not isinstance(evidence, expected_type):
            raise ValueError("injected probe method evidence differs from the condition")
    else:
        expected_evidence = protocol.condition in {
            "localized-state-distillation",
            "random-window-state-distillation",
        }
        if expected_evidence != isinstance(evidence, (LocalizationEvidence, ResidualStateEvidence)):
            raise ValueError("injected localization evidence differs from the condition")
    resolved_method = (
        training_methods.resolve_training_method(protocol, evidence=evidence)
        if isinstance(
            evidence,
            (
                training_methods.ProbeTransitionTrainingEvidence,
                training_methods.ProbeTransitionStateTrainingEvidence,
                training_methods.ProbeSemanticSubspaceTrainingEvidence,
            ),
        )
        else None
    )
    localization_sha = (
        evidence.component_selection_sha256
        if isinstance(evidence, LocalizationEvidence)
        else evidence.evidence_sha256
        if isinstance(evidence, ResidualStateEvidence)
        else None
    )
    resolved_method_sha = (
        resolved_method.method_evidence_sha256 if resolved_method is not None else None
    )
    if (
        config.method_evidence_sha256 is not None
        and resolved_method_sha is not None
        and config.method_evidence_sha256 != resolved_method_sha
    ):
        raise ValueError("injected method evidence hash differs from the resolved artifact")
    localization_sha, method_evidence_sha = validate_condition_evidence(
        condition=protocol.condition,
        localization_sha256=localization_sha,
        method_evidence_sha256=(
            resolved_method_sha
            if resolved_method_sha is not None
            else config.method_evidence_sha256
        ),
    )
    bindings = {
        "config_sha256": protocol.config_sha256,
        "training_data_sha256": bundle.training_data_sha256,
        "localization_sha256": localization_sha,
        "seed": config.seed,
    }
    if monitor_protocol_sha is not None and monitor_data_sha is not None:
        bindings.update(
            {
                "monitor_protocol_sha256": monitor_protocol_sha,
                "monitor_data_sha256": monitor_data_sha,
            }
        )
    if method_evidence_sha is not None:
        bindings["method_evidence_sha256"] = method_evidence_sha

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

    cursor = TrainingCursor(0, 0, 0, 0, 0)
    checkpoint = None
    if config.resume:
        checkpoint = load_training_checkpoint(
            checkpoint_path,
            expected_bindings=bindings,
        )
        if protocol.condition == "probe-transition-single-layer-state-distillation":
            from typo_robust_training.training.runtime import validate_resume_state_calibration

            validate_resume_state_calibration(
                checkpoint.state_path,
                protocol=protocol,
                seed=config.seed,
            )
        cursor = checkpoint.cursor
        _validate_adapter_checkpoints(
            output_dir,
            optimizer_steps=cursor.optimizer_steps,
            interval=protocol.checkpoint_every_optimizer_steps,
        )
    if runtime is None:
        from typo_robust_training.training.runtime import HuggingFaceAdapterTrainingRuntime

        runtime = HuggingFaceAdapterTrainingRuntime(
            protocol=protocol,
            seed=config.seed,
            gpu_id=config.gpu_id,
            evidence=evidence,
        )
    if checkpoint is not None:
        if protocol.condition == "probe-transition-single-layer-state-distillation":
            replay = getattr(runtime, "verify_resume_state_calibration", None)
            if not callable(replay):
                raise TypeError("state training runtime cannot replay its calibration")
            replay(
                checkpoint.state_path,
                _state_calibration_pairs(
                    bundle=bundle,
                    protocol=protocol,
                    seed=config.seed,
                    runtime=runtime,
                ),
            )
            runtime.load_state(checkpoint.state_path)
        elif protocol.calibration_micro_batches:
            calibration = getattr(runtime, "calibrate_state_weight", None)
            if not callable(calibration):
                raise TypeError("state training runtime cannot revalidate its loss weight")
            expected_state_calibration = dict(
                calibration(
                    _state_calibration_pairs(
                        bundle=bundle,
                        protocol=protocol,
                        seed=config.seed,
                        runtime=runtime,
                    )
                )
            )
            runtime.load_state(
                checkpoint.state_path,
                expected_state_calibration=expected_state_calibration,
            )
        else:
            runtime.load_state(checkpoint.state_path)
    order_cache = EpochSourceOrderCache(bundle.sources, seed=config.seed)
    provenance = dict(runtime.provenance())
    started_at = _now()
    run_base: dict[str, object] = {
        "schema_version": "robustness-adapter-training-run/v1",
        "operation": f"train-{protocol.condition}",
        "condition": protocol.condition,
        "config_sha256": protocol.config_sha256,
        "training_data_sha256": bundle.training_data_sha256,
        "data_identity_sha256": bundle.data_identity_sha256,
        "localization_sha256": localization_sha,
        **(
            {"method_evidence_sha256": method_evidence_sha}
            if method_evidence_sha is not None
            else {}
        ),
        "seed": config.seed,
        "gpu_id": config.gpu_id,
        "resume": config.resume,
        "python": platform.python_version(),
        "runtime": provenance,
        "monitor": {
            "protocol_sha256": monitor_protocol_sha,
            "data_sha256": monitor_data_sha,
            "records": len(monitor_records),
            "interval_optimizer_steps": monitor_interval,
            "task_accuracy_allowed": False,
        },
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
    _write_json(run_path, {**run_base, "status": "running", "started_at": started_at})
    tracking_finished = False
    try:
        if not config.resume and protocol.calibration_micro_batches:
            calibration = getattr(runtime, "calibrate_state_weight", None)
            if not callable(calibration):
                raise TypeError("state training runtime cannot calibrate its loss weight")
            calibration(
                _state_calibration_pairs(
                    bundle=bundle,
                    protocol=protocol,
                    seed=config.seed,
                    runtime=runtime,
                )
            )
            run_base["runtime"] = dict(runtime.provenance())
            _write_json(
                run_path,
                {**run_base, "status": "running", "started_at": started_at},
            )
        if cursor.optimizer_steps > protocol.max_optimizer_steps:
            raise ValueError("training checkpoint exceeds the configured optimizer steps")
        runtime.zero_grad()
        if tracker is None and config.wandb_project is not None:
            if isinstance(evidence, ResidualStateEvidence):
                presentation_layers = evidence.state_layers
            elif isinstance(evidence, LocalizationEvidence):
                presentation_layers = evidence.adapter_layers
            elif resolved_method is not None:
                presentation_layers = _resolved_method_presentation_layers(
                    condition=protocol.condition,
                    method=resolved_method,
                )
            elif protocol.condition == "global-state-alignment":
                # Cycle 1 predates the explicit decoder-layer inventory and its
                # Legacy presentation intentionally carries no layer label.
                presentation_layers = (
                    tuple(range(protocol.decoder_layers))
                    if protocol.decoder_layers is not None
                    else ()
                )
            else:
                presentation_layers = ()
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
                    max_student_tokens=protocol.max_student_tokens,
                    state_gradient_ratio=protocol.state_gradient_ratio,
                    state_layers=presentation_layers,
                ),
                resume=config.resume,
                resume_optimizer_step=cursor.optimizer_steps,
            )
        if tracker is not None:
            run_base["tracking"] = dict(tracker.provenance())
            _write_json(
                run_path,
                {**run_base, "status": "running", "started_at": started_at},
            )
        consecutive_monitor_violations = (
            _monitor_violation_streak(
                work_dir=work_dir,
                optimizer_steps=cursor.optimizer_steps,
                monitor_interval=monitor_interval,
                clean_kl_limit=monitor_clean_kl_limit,
                ppl_limit=monitor_ppl_limit,
            )
            if config.resume and monitor_records
            else 0
        )
        while cursor.optimizer_steps < protocol.max_optimizer_steps and (
            protocol.max_student_tokens is None
            or cursor.student_tokens < protocol.max_student_tokens
        ):
            step_started = time.perf_counter()
            micro_rows: list[dict[str, object]] = []
            pending: list[tuple[int, int, TrainingPair]] = []
            for accumulation_index in range(protocol.gradient_accumulation_steps):
                pair, epoch, next_cursor = _next_usable_training_pair(
                    bundle=bundle,
                    cursor=cursor,
                    seed=config.seed,
                    protocol=protocol,
                    runtime=runtime,
                    order_cache=order_cache,
                )
                pending.append((accumulation_index, epoch, pair))
                cursor = next_cursor
            if not protocol.schema_version.endswith("/v1"):
                prepare = getattr(runtime, "prepare_accumulation", None)
                if not callable(prepare):
                    raise TypeError("cycle-2 runtime cannot normalize its accumulation batch")
                scales = tuple(prepare(tuple(pair for _index, _epoch, pair in pending)))
                if (
                    len(scales) != len(pending)
                    or any(not isinstance(scale, TrainingMicroStepScales) for scale in scales)
                    or not math.isclose(sum(scale.output for scale in scales), 1.0, abs_tol=1e-9)
                    or (
                        protocol.loss_weights["state"] > 0.0
                        and not math.isclose(
                            sum(scale.state for scale in scales), 1.0, abs_tol=1e-9
                        )
                    )
                ):
                    raise ValueError("cycle-2 accumulation scales differ from the batch")
            else:
                scales = tuple(
                    TrainingMicroStepScales(
                        output=1.0 / protocol.gradient_accumulation_steps,
                        state=1.0 / protocol.gradient_accumulation_steps,
                    )
                    for _ in pending
                )
            gradient_probe_index = next(
                (index for index, _epoch, candidate in pending if not candidate.is_noop),
                None,
            )
            for (accumulation_index, epoch, pair), scales_for_pair in zip(
                pending, scales, strict=True
            ):
                result = runtime.train_micro_step(
                    pair,
                    loss_scale=1.0 / protocol.gradient_accumulation_steps,
                    measure_gradient_ratio=(
                        not protocol.schema_version.endswith("/v1")
                        and protocol.loss_weights["state"] > 0.0
                        and accumulation_index == gradient_probe_index
                    ),
                    output_loss_scale=(
                        scales_for_pair.output
                        if not protocol.schema_version.endswith("/v1")
                        else None
                    ),
                    state_loss_scale=(
                        scales_for_pair.state
                        if not protocol.schema_version.endswith("/v1")
                        else None
                    ),
                )
                if not isinstance(result, TrainingMicroStepResult):
                    raise TypeError("training runtime returned an invalid micro-step result")
                cursor = replace(
                    cursor,
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
            reached_token_budget = (
                protocol.max_student_tokens is not None
                and cursor.student_tokens >= protocol.max_student_tokens
            )
            monitor_metrics: dict[str, float] = {}
            safety_stop = False
            if monitor_records and cursor.optimizer_steps % monitor_interval == 0:
                monitor = getattr(runtime, "monitor", None)
                if not callable(monitor):
                    raise TypeError("training runtime cannot execute the frozen T0 monitor")
                raw_monitor = monitor(monitor_records)
                if not isinstance(raw_monitor, Mapping):
                    raise TypeError("training T0 monitor must return scalar metrics")
                monitor_metrics = {str(name): float(value) for name, value in raw_monitor.items()}
                if any(not math.isfinite(value) for value in monitor_metrics.values()):
                    raise FloatingPointError("training T0 monitor contains a non-finite scalar")
                unsafe = (
                    monitor_metrics.get("clean_kl_nats_per_token", math.inf)
                    > monitor_clean_kl_limit
                    or monitor_metrics.get("fineweb_edu_ppl_ratio", math.inf) > monitor_ppl_limit
                )
                consecutive_monitor_violations = consecutive_monitor_violations + 1 if unsafe else 0
                safety_stop = consecutive_monitor_violations >= 2
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
            telemetry.update(
                {f"monitor/{name}": value for name, value in sorted(monitor_metrics.items())}
            )
            step_payload["aggregates"] = telemetry
            _write_json(_metrics_step_path(work_dir, cursor.optimizer_steps), step_payload)
            if tracker is not None:
                tracker.log_optimizer_step(
                    telemetry,
                    optimizer_step=cursor.optimizer_steps,
                )
            if (
                cursor.optimizer_steps % protocol.checkpoint_every_optimizer_steps == 0
                or cursor.optimizer_steps == protocol.max_optimizer_steps
                or reached_token_budget
            ):
                state_path = work_dir / f"runtime-state-step-{cursor.optimizer_steps:06d}.pt"
                runtime.save_state(state_path)
                runtime.save_adapter(output_dir / f"adapter-step-{cursor.optimizer_steps:06d}")
                write_training_checkpoint(
                    checkpoint_path,
                    cursor=cursor,
                    state_path=state_path,
                    bindings=bindings,
                )
            if safety_stop:
                raise RuntimeError("frozen T0 clean-harm monitor exceeded its KL/PPL limits twice")
        if (
            protocol.max_student_tokens is not None
            and cursor.student_tokens < protocol.max_student_tokens
        ):
            raise RuntimeError(
                "optimizer-step safety cap was reached before the student-token budget"
            )
        _assemble_metrics(
            metrics_path,
            work_dir=work_dir,
            optimizer_steps=cursor.optimizer_steps,
        )
        runtime.save_adapter(adapter_path)
        expected_adapter_steps = _expected_adapter_checkpoint_steps(
            optimizer_steps=cursor.optimizer_steps,
            interval=protocol.checkpoint_every_optimizer_steps,
        )
        adapter_checkpoints: list[dict[str, int | str]] = []
        for step in expected_adapter_steps:
            checkpoint_adapter = output_dir / f"adapter-step-{step:06d}"
            if not checkpoint_adapter.is_dir():
                raise RuntimeError(f"training adapter checkpoint is missing at step {step}")
            step_metrics = json.loads(
                _metrics_step_path(work_dir, step).read_text(encoding="utf-8")
            )
            student_tokens_at_step = step_metrics.get("student_tokens")
            if (
                isinstance(student_tokens_at_step, bool)
                or not isinstance(student_tokens_at_step, int)
                or student_tokens_at_step <= 0
            ):
                raise RuntimeError("training checkpoint token count is invalid")
            adapter_checkpoints.append(
                {
                    "optimizer_step": step,
                    "student_tokens": student_tokens_at_step,
                    "path": checkpoint_adapter.relative_to(output_dir).as_posix(),
                }
            )
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
        _write_json(
            run_path,
            {
                **run_base,
                "status": "completed",
                "completed_at": _now(),
                "optimizer_steps": cursor.optimizer_steps,
                "micro_steps": cursor.micro_steps,
                "student_tokens": cursor.student_tokens,
                "requested_student_tokens": protocol.max_student_tokens,
                "student_token_overshoot": (
                    cursor.student_tokens - protocol.max_student_tokens
                    if protocol.max_student_tokens is not None
                    else None
                ),
                "adapter_checkpoints": adapter_checkpoints,
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
        _write_json(
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
    "TrainingMicroStepScales",
    "normalized_accumulation_scales",
    "run_adapter_training",
]
