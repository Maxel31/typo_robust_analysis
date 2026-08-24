"""Single-GPU Hugging Face/PEFT runtime for all training conditions."""

from __future__ import annotations

import importlib.metadata
import hashlib
import io
import json
import math
import os
import platform
import random
import stat
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from typo_robust_training.probe.runtime import (
    _require_exact_model_revision,
)
from typo_robust_training.state_gate.runtime import _checkout_source_attestation
from typo_robust_training.integrity import sha256_file
from typo_robust_training.training.adapters import (
    TrainableParameterReport,
    attach_lora_adapters,
    trainable_parameter_report,
)
from typo_robust_training.training.config import (
    AdapterTrainingProtocol,
    is_kojima_faithful_protocol,
    is_mistral_factorial_protocol,
    is_probe_factorial_protocol,
)
from typo_robust_training.training.encoding import (
    PairedEncoding,
    encode_training_pair,
    output_logit_pairs_for_scope,
    retained_clean_character_extent,
)
from typo_robust_training.training.evidence import (
    LocalizationEvidence,
    ResidualStateEvidence,
)
from typo_robust_training.training.methods import (
    PROBE_FACTORIAL_CONDITIONS,
    ProbeTransitionStateTrainingEvidence,
    ProbeSemanticSubspaceTrainingEvidence,
    ProbeTransitionTrainingEvidence,
    ResolvedTrainingMethod,
    resolve_training_method,
)
from typo_robust_training.training.pairs import TrainingPair, UnusableTrainingPairError
from typo_robust_training.training.kojima_faithful import (
    UnusableKojimaFaithfulPairError,
    encode_kojima_faithful_pair,
)
from typo_robust_training.training.runner import (
    factorial_group_balanced_accumulation_scales,
    TrainingMicroStepResult,
    TrainingMicroStepScales,
    normalized_accumulation_scales,
)
from typo_robust_training.training.step import compute_training_step


def _mistral_attested_state_buffer(path: Path, *, expected_sha256: str) -> io.BytesIO:
    """Return one immutable buffer bound to the checkpoint's recorded digest."""

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("adapter runtime expected state hash differs")
    state_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(state_path, flags)
    except OSError as exc:
        raise ValueError("Mistral runtime state must be one unlinked regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Mistral runtime state must be one unlinked regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw_state = handle.read()
        final_metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        visible_metadata = state_path.lstat()
    except OSError as exc:
        raise ValueError("Mistral runtime state changed during attested loading") from exc
    if (
        final_metadata.st_nlink != 1
        or (metadata.st_dev, metadata.st_ino) != (final_metadata.st_dev, final_metadata.st_ino)
        or (metadata.st_dev, metadata.st_ino) != (visible_metadata.st_dev, visible_metadata.st_ino)
    ):
        raise ValueError("Mistral runtime state changed during attested loading")
    if hashlib.sha256(raw_state).hexdigest() != expected_sha256:
        raise ValueError("Mistral runtime state changed before attested loading")
    return io.BytesIO(raw_state)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _positive_architecture_integer(config: object, field: str) -> int:
    value = getattr(config, field, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"training model architecture field {field} is unavailable")
    return value


def resolve_attention_head_dim(config: object) -> int:
    """Resolve a canonical head width while preserving Gemma's explicit GQA width."""

    text_config = getattr(config, "text_config", config)
    hidden_size = _positive_architecture_integer(text_config, "hidden_size")
    attention_heads = _positive_architecture_integer(text_config, "num_attention_heads")
    if hidden_size % attention_heads != 0:
        raise ValueError("training model hidden_size must be divisible by num_attention_heads")
    derived = hidden_size // attention_heads
    explicit = getattr(text_config, "head_dim", None)
    if explicit is None:
        return derived
    if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit <= 0:
        raise ValueError("training model explicit head_dim must be a positive integer")
    model_type = getattr(text_config, "model_type", None)
    if explicit != derived and model_type not in {
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
    }:
        raise ValueError(
            "training model explicit head_dim disagrees with hidden_size/num_attention_heads"
        )
    return explicit


def _cpu_cuda_rng_states(states: object) -> tuple[object, ...]:
    """Normalize serialized CUDA generator states for ``set_rng_state_all``."""

    import torch

    if (
        not isinstance(states, Sequence)
        or isinstance(states, (str, bytes))
        or not states
        or any(
            not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1
            for state in states
        )
    ):
        raise ValueError("CUDA RNG states must be non-empty one-dimensional byte tensors")
    return tuple(state.detach().cpu().contiguous() for state in states)


def _finite_ppl_ratio(log_nll_delta: float) -> float:
    """Exponentiate a log-PPL delta without bypassing the monitor safety gate."""

    if not math.isfinite(float(log_nll_delta)):
        raise ValueError("log PPL ratio must be finite")
    maximum_log = math.log(sys.float_info.max)
    minimum_log = math.log(sys.float_info.min)
    return math.exp(min(max(float(log_nll_delta), minimum_log), maximum_log))


def _require_exact_training_wrapper_revision(
    wrapper: object,
    *,
    expected: str,
    role: str,
) -> tuple[str, str]:
    """Bind one independently loaded model and tokenizer to the requested commit."""

    model = getattr(wrapper, "model", None)
    tokenizer = getattr(wrapper, "tokenizer", None)
    config = getattr(model, "config", None)
    if model is None or config is None or tokenizer is None:
        raise ValueError(f"loaded {role} model/tokenizer identity is unavailable")
    model_revision = _require_exact_model_revision(
        model_config=config,
        tokenizer=tokenizer,
        expected=expected,
    )
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    tokenizer_revision = (
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, Mapping) else None
    )
    if not isinstance(tokenizer_revision, str) or not tokenizer_revision:
        raise ValueError(f"loaded {role} tokenizer revision is not observable")
    if tokenizer_revision != expected:
        raise ValueError(f"loaded {role} tokenizer revision differs from the requested revision")
    return model_revision, tokenizer_revision


_RUNTIME_STATE_SCHEMA = "robustness-adapter-runtime-state/v3"
_CALIBRATION_REPLAY_REL_TOLERANCE = 1e-6
_CALIBRATION_REPLAY_ABS_TOLERANCE = 1e-12
_CALIBRATION_FIELDS = {
    "schema_version",
    "micro_batches",
    "record_ids",
    "output_gradient_norms",
    "state_gradient_norms",
    "mean_output_gradient_norm",
    "mean_state_gradient_norm",
    "target_gradient_ratio",
    "state_weight",
    "achieved_initial_ratio",
}


def _calibration_replay_matches(stored: object, replayed: object) -> bool:
    """Compare attested replay evidence with a strict, frozen numeric tolerance."""

    if isinstance(stored, Mapping) and isinstance(replayed, Mapping):
        return set(stored) == set(replayed) and all(
            _calibration_replay_matches(stored[key], replayed[key]) for key in stored
        )
    if isinstance(stored, list) and isinstance(replayed, list):
        return len(stored) == len(replayed) and all(
            _calibration_replay_matches(left, right)
            for left, right in zip(stored, replayed, strict=True)
        )
    if (
        not isinstance(stored, bool)
        and not isinstance(replayed, bool)
        and isinstance(stored, (int, float))
        and isinstance(replayed, (int, float))
    ):
        return (
            math.isfinite(float(stored))
            and math.isfinite(float(replayed))
            and math.isclose(
                float(stored),
                float(replayed),
                rel_tol=_CALIBRATION_REPLAY_REL_TOLERANCE,
                abs_tol=_CALIBRATION_REPLAY_ABS_TOLERANCE,
            )
        )
    return type(stored) is type(replayed) and stored == replayed


def _validated_resume_state_calibration(
    *,
    protocol: AdapterTrainingProtocol | object,
    state_weight: object,
    calibration: object,
    expected_calibration: Mapping[str, object] | None,
) -> tuple[float, dict[str, object] | None]:
    """Re-derive the immutable one-shot calibration contract on resume."""

    if (
        isinstance(state_weight, bool)
        or not isinstance(state_weight, (int, float))
        or not math.isfinite(float(state_weight))
        or float(state_weight) < 0.0
    ):
        raise ValueError("adapter runtime checkpoint state weight differs")
    weight = float(state_weight)
    ratio = getattr(protocol, "state_gradient_ratio", None)
    if ratio is None:
        expected_weight = float(getattr(protocol, "loss_weights", {}).get("state", 0.0) > 0.0)
        if weight != expected_weight or calibration is not None:
            raise ValueError("adapter runtime checkpoint calibration differs")
        return weight, None
    if (
        weight <= 0.0
        or not isinstance(calibration, Mapping)
        or not isinstance(expected_calibration, Mapping)
        or not _calibration_replay_matches(calibration, expected_calibration)
    ):
        raise ValueError("adapter runtime checkpoint calibration differs")
    row = dict(calibration)
    if set(row) != _CALIBRATION_FIELDS or row.get("schema_version") != (
        "state-gradient-calibration/v1"
    ):
        raise ValueError("adapter runtime checkpoint calibration differs")
    count = getattr(protocol, "calibration_micro_batches", None)
    record_ids = row.get("record_ids")
    output_norms = row.get("output_gradient_norms")
    state_norms = row.get("state_gradient_norms")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or row.get("micro_batches") != count
        or not isinstance(record_ids, list)
        or len(record_ids) != count
        or len(set(record_ids)) != count
        or any(not isinstance(value, str) or not value for value in record_ids)
        or not isinstance(output_norms, list)
        or not isinstance(state_norms, list)
        or len(output_norms) != count
        or len(state_norms) != count
    ):
        raise ValueError("adapter runtime checkpoint calibration differs")

    def positive(values: list[object]) -> tuple[float, ...]:
        normalized: list[float] = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError("adapter runtime checkpoint calibration differs")
            normalized.append(float(value))
        return tuple(normalized)

    outputs = positive(output_norms)
    states = positive(state_norms)
    mean_output = sum(outputs) / count
    mean_state = sum(states) / count
    expected_ratio = float(ratio)
    expected_values = {
        "mean_output_gradient_norm": mean_output,
        "mean_state_gradient_norm": mean_state,
        "target_gradient_ratio": expected_ratio,
        "state_weight": weight,
        "achieved_initial_ratio": weight * mean_state / mean_output,
    }
    if any(
        isinstance(row.get(field), bool)
        or not isinstance(row.get(field), (int, float))
        or not math.isfinite(float(row[field]))
        or not math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=0.0)
        for field, value in expected_values.items()
    ) or not math.isclose(
        expected_values["achieved_initial_ratio"],
        expected_ratio,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("adapter runtime checkpoint calibration differs")
    return weight, row


_ADAPTER_SCOPE_SCHEMA = "decoder-lora-optimizer-scope/v1"
_STATE_CALIBRATION_SCHEMA = "state-gradient-calibration/v3"
_STATE_CALIBRATION_REPLAY_REL_TOL = 1e-6
_STATE_CALIBRATION_REPLAY_ABS_TOL = 1e-8
_STATE_CALIBRATION_FIELDS = {
    "schema_version",
    "micro_batches",
    "noisy_micro_batches",
    "record_ids",
    "output_gradient_norms",
    "state_gradient_norms",
    "mean_output_gradient_norm",
    "mean_state_gradient_norm",
    "target_gradient_ratio",
    "state_weight",
    "achieved_initial_ratio",
    "replay_relative_tolerance",
    "replay_absolute_tolerance",
}


def _positive_finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"adapter runtime checkpoint {field} differs")
    return float(value)


def _positive_finite_vector(value: object, *, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"adapter runtime checkpoint {field} differs")
    return tuple(_positive_finite_number(item, field=field) for item in value)


def _validate_priority_b_calibration(
    *,
    protocol: AdapterTrainingProtocol,
    state_weight: object,
    calibration: object,
) -> None:
    """Validate the immutable, one-shot Priority B calibration evidence."""

    if protocol.condition != "probe-transition-single-layer-state-distillation":
        return
    if protocol.state_gradient_ratio != 0.05 or protocol.calibration_micro_batches != 8:
        raise ValueError("adapter runtime checkpoint calibration protocol differs")
    if not isinstance(calibration, Mapping) or set(calibration) != _STATE_CALIBRATION_FIELDS:
        raise ValueError("adapter runtime checkpoint calibration fields differ")
    if calibration["schema_version"] != _STATE_CALIBRATION_SCHEMA:
        raise ValueError("adapter runtime checkpoint calibration schema differs")
    if calibration["micro_batches"] != 8 or calibration["noisy_micro_batches"] != 8:
        raise ValueError("adapter runtime checkpoint calibration dosage differs")
    if (
        calibration["replay_relative_tolerance"] != _STATE_CALIBRATION_REPLAY_REL_TOL
        or calibration["replay_absolute_tolerance"] != _STATE_CALIBRATION_REPLAY_ABS_TOL
    ):
        raise ValueError("adapter runtime checkpoint calibration replay tolerance differs")
    record_ids = calibration["record_ids"]
    if (
        not isinstance(record_ids, Sequence)
        or isinstance(record_ids, (str, bytes))
        or len(record_ids) != 8
        or any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
        or len(set(record_ids)) != 8
    ):
        raise ValueError("adapter runtime checkpoint calibration records differ")
    output_norms = _positive_finite_vector(
        calibration["output_gradient_norms"],
        field="calibration output gradients",
        length=8,
    )
    state_norms = _positive_finite_vector(
        calibration["state_gradient_norms"],
        field="calibration state gradients",
        length=8,
    )
    mean_output = _positive_finite_number(
        calibration["mean_output_gradient_norm"],
        field="calibration mean output gradient",
    )
    mean_state = _positive_finite_number(
        calibration["mean_state_gradient_norm"],
        field="calibration mean state gradient",
    )
    if not math.isclose(mean_output, sum(output_norms) / 8, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("adapter runtime checkpoint calibration output mean differs")
    if not math.isclose(mean_state, sum(state_norms) / 8, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("adapter runtime checkpoint calibration state mean differs")
    target = _positive_finite_number(
        calibration["target_gradient_ratio"],
        field="calibration target ratio",
    )
    if target != 0.05:
        raise ValueError("adapter runtime checkpoint calibration target ratio differs")
    derived_weight = target * mean_output / mean_state
    stored_weight = _positive_finite_number(
        state_weight,
        field="state weight",
    )
    calibration_weight = _positive_finite_number(
        calibration["state_weight"],
        field="calibration state weight",
    )
    if not math.isclose(
        stored_weight, derived_weight, rel_tol=1e-12, abs_tol=1e-12
    ) or not math.isclose(
        calibration_weight,
        derived_weight,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("adapter runtime checkpoint calibrated state weight differs")
    achieved = _positive_finite_number(
        calibration["achieved_initial_ratio"],
        field="calibration achieved ratio",
    )
    derived_ratio = stored_weight * mean_state / mean_output
    if not math.isclose(achieved, derived_ratio, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(
        derived_ratio,
        target,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("adapter runtime checkpoint calibration achieved ratio differs")


def _validate_replayed_priority_b_calibration(
    *,
    protocol: AdapterTrainingProtocol,
    saved_state_weight: object,
    saved_calibration: object,
    replayed_calibration: Mapping[str, object],
) -> None:
    """Compare a checkpoint claim with gradients replayed from the bound initial stream."""

    _validate_priority_b_calibration(
        protocol=protocol,
        state_weight=saved_state_weight,
        calibration=saved_calibration,
    )
    _validate_priority_b_calibration(
        protocol=protocol,
        state_weight=replayed_calibration.get("state_weight"),
        calibration=replayed_calibration,
    )
    assert isinstance(saved_calibration, Mapping)  # established above
    if tuple(saved_calibration["record_ids"]) != tuple(replayed_calibration["record_ids"]):
        raise ValueError("adapter runtime checkpoint calibration replay records differ")

    scalar_fields = (
        "mean_output_gradient_norm",
        "mean_state_gradient_norm",
        "state_weight",
        "achieved_initial_ratio",
    )
    vector_fields = ("output_gradient_norms", "state_gradient_norms")
    for field in scalar_fields:
        if not math.isclose(
            float(saved_calibration[field]),
            float(replayed_calibration[field]),
            rel_tol=_STATE_CALIBRATION_REPLAY_REL_TOL,
            abs_tol=_STATE_CALIBRATION_REPLAY_ABS_TOL,
        ):
            raise ValueError(f"adapter runtime checkpoint calibration replay {field} differs")
    for field in vector_fields:
        saved_values = tuple(float(value) for value in saved_calibration[field])
        replayed_values = tuple(float(value) for value in replayed_calibration[field])
        if len(saved_values) != len(replayed_values) or any(
            not math.isclose(
                saved,
                replayed,
                rel_tol=_STATE_CALIBRATION_REPLAY_REL_TOL,
                abs_tol=_STATE_CALIBRATION_REPLAY_ABS_TOL,
            )
            for saved, replayed in zip(saved_values, replayed_values, strict=True)
        ):
            raise ValueError(f"adapter runtime checkpoint calibration replay {field} differs")
    if not math.isclose(
        float(saved_state_weight),
        float(replayed_calibration["state_weight"]),
        rel_tol=_STATE_CALIBRATION_REPLAY_REL_TOL,
        abs_tol=_STATE_CALIBRATION_REPLAY_ABS_TOL,
    ):
        raise ValueError("adapter runtime checkpoint calibration replay state weight differs")


def validate_resume_state_calibration(
    path: Path,
    *,
    protocol: AdapterTrainingProtocol,
    seed: int,
) -> None:
    """Reject invalid Priority B calibration before constructing a GPU runtime."""

    if protocol.condition != "probe-transition-single-layer-state-distillation":
        return
    import torch

    payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != _RUNTIME_STATE_SCHEMA
        or payload.get("condition") != protocol.condition
        or payload.get("config_sha256") != protocol.config_sha256
        or payload.get("seed") != seed
    ):
        raise ValueError("adapter runtime checkpoint identity differs")
    _validate_priority_b_calibration(
        protocol=protocol,
        state_weight=payload.get("state_weight"),
        calibration=payload.get("state_calibration"),
    )


def _optimizer_group_sizes(state: object) -> tuple[int, ...]:
    """Return checkpoint optimizer group sizes without mutating an optimizer."""

    if not isinstance(state, Mapping):
        raise ValueError("adapter runtime checkpoint optimizer state differs")
    groups = state.get("param_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or not groups:
        raise ValueError("adapter runtime checkpoint optimizer groups differ")
    sizes: list[int] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("adapter runtime checkpoint optimizer group differs")
        parameters = group.get("params")
        if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes)):
            raise ValueError("adapter runtime checkpoint optimizer parameters differ")
        sizes.append(len(parameters))
    return tuple(sizes)


def _adapter_tensor_spec(state: object) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Describe every serialized adapter tensor for fail-closed scope checks."""

    import torch

    if not isinstance(state, Mapping) or not state:
        raise ValueError("adapter runtime checkpoint adapter state differs")
    spec: list[tuple[str, tuple[int, ...], str]] = []
    for name, tensor in state.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError("adapter runtime checkpoint adapter tensors differ")
        spec.append((name, tuple(int(size) for size in tensor.shape), str(tensor.dtype)))
    return tuple(sorted(spec))


def _adapter_scope_contract(
    *,
    adapter_state: object,
    optimizer_state: object,
    optimizer_parameter_names: Sequence[str],
) -> dict[str, object]:
    """Build the exact adapter/optimizer ordering contract stored in v3 states."""

    names = tuple(optimizer_parameter_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("adapter runtime optimizer parameter names differ")
    group_sizes = _optimizer_group_sizes(optimizer_state)
    if sum(group_sizes) != len(names):
        raise ValueError("adapter runtime optimizer parameter order differs")
    return {
        "schema_version": _ADAPTER_SCOPE_SCHEMA,
        "adapter_tensors": _adapter_tensor_spec(adapter_state),
        "optimizer_group_sizes": group_sizes,
        "optimizer_parameter_names": names,
    }


def _validate_adapter_scope_before_resume(
    *,
    checkpoint_adapter: object,
    checkpoint_optimizer: object,
    checkpoint_scope: object | None,
    current_adapter: object,
    current_optimizer: object,
    current_optimizer_parameter_names: Sequence[str],
) -> None:
    """Reject incompatible LoRA scopes before PEFT or Torch mutate runtime state."""

    checkpoint_tensors = _adapter_tensor_spec(checkpoint_adapter)
    current_tensors = _adapter_tensor_spec(current_adapter)
    checkpoint_group_sizes = _optimizer_group_sizes(checkpoint_optimizer)
    current_group_sizes = _optimizer_group_sizes(current_optimizer)
    current_names = tuple(current_optimizer_parameter_names)
    compatible = (
        checkpoint_tensors == current_tensors and checkpoint_group_sizes == current_group_sizes
    )
    if checkpoint_scope is not None:
        if not isinstance(checkpoint_scope, Mapping) or set(checkpoint_scope) != {
            "schema_version",
            "adapter_tensors",
            "optimizer_group_sizes",
            "optimizer_parameter_names",
        }:
            raise ValueError("adapter runtime checkpoint scope fields differ")
        recorded_scope = {
            "schema_version": checkpoint_scope["schema_version"],
            "adapter_tensors": tuple(checkpoint_scope["adapter_tensors"]),
            "optimizer_group_sizes": tuple(checkpoint_scope["optimizer_group_sizes"]),
            "optimizer_parameter_names": tuple(checkpoint_scope["optimizer_parameter_names"]),
        }
        derived_checkpoint = {
            "schema_version": _ADAPTER_SCOPE_SCHEMA,
            "adapter_tensors": checkpoint_tensors,
            "optimizer_group_sizes": checkpoint_group_sizes,
            "optimizer_parameter_names": recorded_scope["optimizer_parameter_names"],
        }
        if recorded_scope != derived_checkpoint:
            raise ValueError("adapter runtime checkpoint scope metadata differs")
        compatible = compatible and recorded_scope["optimizer_parameter_names"] == current_names
    if not compatible:
        raise ValueError(
            "adapter runtime checkpoint LoRA/optimizer scope differs from the current "
            "decoder-only scope; resume with the checkpoint-producing code revision "
            "instead of silently dropping adapter tensors"
        )


def next_gradient_ratio_violations(
    violations: int,
    *,
    ratio: float,
    optimizer_steps: int,
    guard_steps: int,
) -> int:
    """Count startup-only ratio violations and keep later ratios diagnostic."""

    if (
        isinstance(violations, bool)
        or not isinstance(violations, int)
        or violations < 0
        or not math.isfinite(float(ratio))
        or float(ratio) < 0.0
        or isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps < 0
        or isinstance(guard_steps, bool)
        or not isinstance(guard_steps, int)
        or guard_steps <= 0
    ):
        raise ValueError("gradient-ratio guard inputs are invalid")
    if optimizer_steps >= guard_steps:
        return 0
    return violations + 1 if ratio > 0.5 else 0


def _resolve_probe_transition_runtime_method(
    protocol: AdapterTrainingProtocol,
    evidence: (
        LocalizationEvidence
        | ResidualStateEvidence
        | ProbeTransitionTrainingEvidence
        | ProbeTransitionStateTrainingEvidence
        | ProbeSemanticSubspaceTrainingEvidence
        | None
    ),
) -> ResolvedTrainingMethod | None:
    """Fail closed on probe evidence and objectives before CUDA setup."""

    expected_types = {
        "probe-transition-output-matching": ProbeTransitionTrainingEvidence,
        "probe-transition-single-layer-state-distillation": (ProbeTransitionStateTrainingEvidence),
        "probe-semantic-subspace-distillation": ProbeSemanticSubspaceTrainingEvidence,
        **{condition: ProbeTransitionTrainingEvidence for condition in PROBE_FACTORIAL_CONDITIONS},
    }
    expected_type = expected_types.get(protocol.condition)
    if expected_type is None:
        if isinstance(
            evidence,
            (
                ProbeTransitionTrainingEvidence,
                ProbeTransitionStateTrainingEvidence,
                ProbeSemanticSubspaceTrainingEvidence,
            ),
        ):
            raise ValueError("probe method evidence cannot configure this condition")
        return None
    if not isinstance(evidence, expected_type):
        raise ValueError("probe training requires probe evidence matching its exact condition")
    resolved = resolve_training_method(protocol, evidence=evidence)
    if protocol.condition in PROBE_FACTORIAL_CONDITIONS:
        expected_weights = {
            "noisy_language_model": 0.0,
            "answer": 0.0,
            "output": 1.0,
            "state": 0.0,
            "clean": 0.0,
        }
        if (
            dict(protocol.loss_weights) != expected_weights
            or resolved.state_layers
            or resolved.state_target != "none"
            or protocol.state_scope != "none"
            or protocol.state_distance != "none"
            or protocol.state_gradient_ratio is not None
            or protocol.calibration_micro_batches != 0
        ):
            raise ValueError("probe-factorial output matching must disable auxiliary losses")
        return resolved
    if protocol.condition == "probe-transition-output-matching":
        expected_weights = {
            "noisy_language_model": 0.0,
            "answer": 0.0,
            "output": 1.0,
            "state": 0.0,
            "clean": 0.0,
        }
        if (
            dict(protocol.loss_weights) != expected_weights
            or resolved.state_layers
            or resolved.state_target != "none"
            or protocol.state_scope != "none"
            or protocol.state_distance != "none"
            or protocol.state_gradient_ratio is not None
            or protocol.calibration_micro_batches != 0
        ):
            raise ValueError("probe-transition output matching must disable state training")
        return resolved
    if protocol.condition == "probe-transition-single-layer-state-distillation":
        expected_weights = {
            "noisy_language_model": 0.0,
            "answer": 0.0,
            "output": 1.0,
            "state": 1.0,
            "clean": 0.0,
        }
        if (
            not isinstance(evidence, ProbeTransitionStateTrainingEvidence)
            or dict(protocol.loss_weights) != expected_weights
            or resolved.state_layers != (evidence.selected_transition_layer,)
            or resolved.state_target
            != "complete-decoder-block-residual-output-at-edited-word-final/v1"
            or protocol.state_scope != "probe-transition-single-layer-edited-word-final-token/v1"
            or protocol.state_distance != "cosine-residual/v1"
            or protocol.state_gradient_ratio != 0.05
            or protocol.calibration_micro_batches != 8
            or protocol.temperature != 1.0
            or protocol.epsilon != 1e-8
        ):
            raise ValueError("probe-transition state training objective or evidence differs")
        return resolved
    expected_weights = {
        "noisy_language_model": 0.0,
        "answer": 0.0,
        "output": 1.0,
        "state": 1.0,
        "clean": 0.0,
    }
    if (
        not isinstance(evidence, ProbeSemanticSubspaceTrainingEvidence)
        or dict(protocol.loss_weights) != expected_weights
        or resolved.state_layers != (evidence.transition_layer,)
        or resolved.state_target != "probe-semantic-subspace-rank16"
        or protocol.state_scope != "probe-semantic-subspace-edited-word-final-token"
        or protocol.state_distance != "frozen-probe-classifier-forward-kl/v1"
        or protocol.state_gradient_ratio != 0.05
        or protocol.calibration_micro_batches != 8
        or protocol.temperature != 1.0
        or protocol.epsilon != 1e-8
    ):
        raise ValueError("probe semantic training objective differs")
    return resolved


class HuggingFaceAdapterTrainingRuntime:
    """Frozen clean teacher plus a typo student whose LoRA is the only update."""

    def __init__(
        self,
        *,
        protocol: AdapterTrainingProtocol,
        seed: int,
        gpu_id: str,
        evidence: (
            LocalizationEvidence
            | ResidualStateEvidence
            | ProbeTransitionTrainingEvidence
            | ProbeTransitionStateTrainingEvidence
            | ProbeSemanticSubspaceTrainingEvidence
            | None
        ),
    ) -> None:
        if not isinstance(protocol, AdapterTrainingProtocol):
            raise TypeError("training runtime protocol must be AdapterTrainingProtocol")
        if seed not in protocol.seed_inventory:
            raise ValueError("training runtime seed is outside the frozen inventory")
        resolved_method = _resolve_probe_transition_runtime_method(protocol, evidence)
        self.code_revision, self.source_tree_sha256 = _checkout_source_attestation()
        from typo_cot.models.tokenizer_attestation import (
            preflight_frozen_tokenizer_attestation,
        )

        frozen_tokenizer_attestation = preflight_frozen_tokenizer_attestation(
            expected_model=protocol.model,
            expected_revision=protocol.model_revision,
        )
        if isinstance(
            evidence,
            (
                ProbeTransitionTrainingEvidence,
                ProbeTransitionStateTrainingEvidence,
                ProbeSemanticSubspaceTrainingEvidence,
            ),
        ):
            inherited = evidence.tokenizer_snapshot_attestation
            if inherited is not None and dict(inherited) != (
                frozen_tokenizer_attestation.provenance_dict()
            ):
                raise ValueError("probe evidence tokenizer provenance differs from training")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("adapter training requires exactly one requested CUDA GPU")
        from transformers import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.tokenizer_attestation import require_frozen_tokenizer_attestation
        from typo_cot.models.wrapper import create_model_wrapper

        self.protocol = protocol
        self.seed = seed
        self.gpu_id = gpu_id
        self.evidence = evidence
        self._torch = torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.teacher_wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self.teacher = self.teacher_wrapper.model
        self.teacher_revision, teacher_tokenizer_revision = (
            _require_exact_training_wrapper_revision(
                self.teacher_wrapper,
                expected=protocol.model_revision,
                role="teacher",
            )
        )
        teacher_tokenizer_attestation = require_frozen_tokenizer_attestation(
            self.teacher_wrapper,
            expected_model=protocol.model,
            expected_revision=protocol.model_revision,
        )
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.student_wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        student_base = self.student_wrapper.model
        self.student_revision, student_tokenizer_revision = (
            _require_exact_training_wrapper_revision(
                self.student_wrapper,
                expected=protocol.model_revision,
                role="student",
            )
        )
        student_tokenizer_attestation = require_frozen_tokenizer_attestation(
            self.student_wrapper,
            expected_model=protocol.model,
            expected_revision=protocol.model_revision,
        )
        if teacher_tokenizer_revision != student_tokenizer_revision:
            raise ValueError("teacher and student tokenizer revisions differ")
        if teacher_tokenizer_attestation.provenance_dict() != (
            student_tokenizer_attestation.provenance_dict()
        ):
            raise ValueError("teacher and student tokenizer attestations differ")
        if student_tokenizer_attestation.provenance_dict() != (
            frozen_tokenizer_attestation.provenance_dict()
        ):
            raise ValueError("training tokenizer attestation changed after preflight")
        self.tokenizer_snapshot_attestation = student_tokenizer_attestation
        self.tokenizer_revision = student_tokenizer_revision
        base_layers = find_decoder_layers(student_base)
        self.num_decoder_layers = len(base_layers)
        if (
            protocol.decoder_layers is not None
            and protocol.decoder_layers != self.num_decoder_layers
        ):
            raise ValueError("training model decoder layers differ from the frozen config")
        text_config = getattr(student_base.config, "text_config", student_base.config)
        teacher_text_config = getattr(self.teacher.config, "text_config", self.teacher.config)
        self.attention_head_dim = resolve_attention_head_dim(text_config)
        teacher_attention_head_dim = resolve_attention_head_dim(teacher_text_config)
        self.mlp_intermediate_size = _positive_architecture_integer(
            text_config, "intermediate_size"
        )
        self.attention_heads = _positive_architecture_integer(text_config, "num_attention_heads")
        teacher_architecture = (
            teacher_attention_head_dim,
            _positive_architecture_integer(teacher_text_config, "intermediate_size"),
            _positive_architecture_integer(teacher_text_config, "num_attention_heads"),
        )
        if teacher_architecture != (
            self.attention_head_dim,
            self.mlp_intermediate_size,
            self.attention_heads,
        ):
            raise ValueError("teacher and student architecture fields differ")
        if resolved_method is not None:
            adapter_layers = resolved_method.adapter_layers
        elif protocol.layer_scope in {
            "all-decoder-layers",
            "all-linear-lora-including-embedding-and-lm-head",
        }:
            adapter_layers = tuple(range(self.num_decoder_layers))
        else:
            if not isinstance(evidence, LocalizationEvidence):
                raise ValueError("targeted adapter training requires localization evidence")
            adapter_layers = evidence.adapter_layers
        self.adapter_layers = adapter_layers
        self.component_weights = (
            evidence.component_weights if isinstance(evidence, LocalizationEvidence) else None
        )
        if resolved_method is not None:
            self.state_layers = resolved_method.state_layers
        elif isinstance(evidence, ResidualStateEvidence):
            self.state_layers = evidence.state_layers
        elif protocol.state_scope == "all-layers-edited-word-final-tokens":
            self.state_layers = tuple(range(self.num_decoder_layers))
        else:
            self.state_layers = ()
        residual_scope = protocol.state_scope in {
            "causal-window-edited-word-final-tokens",
            "random-window-edited-word-final-tokens",
            "all-layers-edited-word-final-tokens",
            "probe-transition-single-layer-edited-word-final-token/v1",
        }
        semantic_scope = protocol.state_scope == "probe-semantic-subspace-edited-word-final-token"
        if (residual_scope or semantic_scope) != bool(self.state_layers):
            raise ValueError("residual state layers differ from the training objective")
        self.student = attach_lora_adapters(
            student_base,
            protocol=protocol,
            decoder_layers=adapter_layers,
            initialization_seed=seed,
        )
        self.device = next(self.student.parameters()).device
        if next(self.teacher.parameters()).device != self.device:
            raise ValueError("teacher and student must share one training device")
        self.semantic_basis = self.semantic_projected_class_weights = None
        self.semantic_classifier_bias = None
        if isinstance(evidence, ProbeSemanticSubspaceTrainingEvidence):
            self.semantic_basis = torch.as_tensor(
                evidence.basis.copy(), dtype=torch.float32, device=self.device
            ).detach()
            self.semantic_projected_class_weights = torch.as_tensor(
                evidence.projected_class_weights.copy(),
                dtype=torch.float32,
                device=self.device,
            ).detach()
            self.semantic_classifier_bias = torch.as_tensor(
                evidence.classifier_bias.copy(), dtype=torch.float32, device=self.device
            ).detach()
            if any(
                value.requires_grad
                for value in (
                    self.semantic_basis,
                    self.semantic_projected_class_weights,
                    self.semantic_classifier_bias,
                )
            ):
                raise RuntimeError("semantic classifier must remain frozen")
        self.tokenizer = self.student_wrapper.tokenizer
        if getattr(self.tokenizer, "is_fast", False) is not True:
            raise ValueError("adapter training requires a fast tokenizer with offsets")
        self.tokenizer.padding_side = "left"
        trainable_items = tuple(
            (name, parameter)
            for name, parameter in self.student.named_parameters()
            if parameter.requires_grad
        )
        self._optimizer_parameter_names = tuple(name for name, _ in trainable_items)
        trainable = [parameter for _, parameter in trainable_items]
        self.parameter_report: TrainableParameterReport = trainable_parameter_report(
            self.student,
            expected_layers=adapter_layers,
            expected_modules=protocol.lora_target_modules,
        )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
        )
        warmup_steps = round(protocol.max_optimizer_steps * protocol.warmup_ratio)
        if protocol.scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=protocol.max_optimizer_steps,
            )
        elif protocol.scheduler == "constant-with-warmup":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError("training scheduler is unsupported")
        self.state_weight: float | None = (
            None
            if protocol.state_gradient_ratio is not None
            else float(protocol.loss_weights["state"] > 0.0)
        )
        self.state_calibration: dict[str, object] | None = None
        self._last_gradient_diagnostics: dict[str, float] = {}
        self._gradient_ratio_violations = 0
        self._optimizer_steps = 0
        self._prepared_encodings: deque[tuple[TrainingPair, PairedEncoding]] = deque()
        self._verified_resume_state_path: Path | None = None
        self._verified_resume_state_sha256: str | None = None
        self._monitor_base_clean: tuple[float, int] | None = None
        self._monitor_base_natural: tuple[float, int] | None = None
        torch.cuda.reset_peak_memory_stats()

    def _encode_pair(self, pair: TrainingPair) -> PairedEncoding:
        lightweight_kojima_protocol = (
            not isinstance(self.protocol, AdapterTrainingProtocol)
            and getattr(self.protocol, "schema_version", None)
            == "robustness-adapter-training-config/v7"
            and getattr(self.protocol, "condition", None) == "kojima-faithful-output-matching"
        )
        if is_kojima_faithful_protocol(self.protocol) or lightweight_kojima_protocol:
            return encode_kojima_faithful_pair(pair, tokenizer=self.tokenizer)
        encoding_options: dict[str, object] = {}
        if is_mistral_factorial_protocol(self.protocol):
            encoding_options["add_special_tokens"] = False
        return encode_training_pair(
            pair,
            tokenizer=self.tokenizer,
            max_length=self.protocol.max_sequence_length,
            require_answer_targets=self.protocol.loss_weights["answer"] > 0.0,
            require_all_edits_visible=not self.protocol.schema_version.endswith("/v1"),
            require_downstream_targets=(
                is_probe_factorial_protocol(self.protocol) and not pair.is_noop
            ),
            **encoding_options,
        )

    def pair_is_usable(self, pair: TrainingPair) -> bool:
        """Return whether a generated pair supplies every frozen training target."""

        try:
            self._encode_pair(pair)
        except (UnusableTrainingPairError, UnusableKojimaFaithfulPairError):
            return False
        return True

    def retained_clean_character_extent(self, pair: TrainingPair) -> int:
        """Expose the tokenizer-exact raw prefix available for typo selection."""

        return retained_clean_character_extent(
            pair,
            tokenizer=self.tokenizer,
            max_length=self.protocol.max_sequence_length,
        )

    def _trainable_parameters(self) -> tuple[object, ...]:
        return tuple(
            parameter for parameter in self.student.parameters() if parameter.requires_grad
        )

    def _gradient_norm(self, loss: object, *, retain_graph: bool) -> float:
        torch = self._torch
        gradients = torch.autograd.grad(
            loss,
            self._trainable_parameters(),
            retain_graph=retain_graph,
            allow_unused=True,
        )
        squared = sum(
            gradient.detach().float().square().sum()
            for gradient in gradients
            if gradient is not None
        )
        if not isinstance(squared, torch.Tensor):
            return 0.0
        return float(squared.sqrt().cpu())

    def _step_output(self, pair: TrainingPair, *, state_weight: float):
        if self._prepared_encodings:
            prepared_pair, encoding = self._prepared_encodings.popleft()
            if prepared_pair != pair:
                self._prepared_encodings.clear()
                raise RuntimeError("prepared accumulation order differs from training order")
        else:
            encoding = self._encode_pair(pair)
        return compute_training_step(
            teacher=(self.student if is_kojima_faithful_protocol(self.protocol) else self.teacher),
            student=self.student,
            encoding=encoding,
            protocol=self.protocol,
            component_weights=self.component_weights,
            attention_head_dim=self.attention_head_dim,
            state_layers=self.state_layers,
            state_weight=state_weight,
            semantic_basis=self.semantic_basis,
            semantic_projected_class_weights=self.semantic_projected_class_weights,
            semantic_classifier_bias=self.semantic_classifier_bias,
            teacher_adapter_disabled=is_kojima_faithful_protocol(self.protocol),
        )

    def prepare_accumulation(
        self,
        pairs: Sequence[TrainingPair],
    ) -> Sequence[TrainingMicroStepScales]:
        """Tokenize once and freeze exact token/coordinate denominators for one update."""

        if self._prepared_encodings:
            raise RuntimeError("the previous prepared accumulation was not consumed")
        rows = tuple(pairs)
        if len(rows) != self.protocol.gradient_accumulation_steps:
            raise ValueError("prepared accumulation size differs from the config")
        encodings = tuple(self._encode_pair(pair) for pair in rows)
        state_active = self.protocol.loss_weights["state"] > 0.0
        output_token_counts = tuple(
            len(
                output_logit_pairs_for_scope(
                    encoding,
                    output_scope=self.protocol.output_scope,
                )
            )
            for encoding in encodings
        )
        if self.protocol.condition in PROBE_FACTORIAL_CONDITIONS:
            if state_active:
                raise ValueError("probe-factorial accumulation cannot contain state supervision")
            scales = factorial_group_balanced_accumulation_scales(
                output_token_counts=output_token_counts,
                is_noop_rows=tuple(encoding.is_noop for encoding in encodings),
            )
        else:
            scales = normalized_accumulation_scales(
                output_token_counts=output_token_counts,
                state_coordinate_counts=tuple(
                    len(encoding.clean_edit_positions) if state_active else 0
                    for encoding in encodings
                ),
                state_active=state_active,
            )
        self._prepared_encodings.extend(zip(rows, encodings, strict=True))
        return scales

    def _measure_state_calibration(
        self,
        pairs: Sequence[TrainingPair],
    ) -> dict[str, object]:
        """Measure the preregistered initial gradient dosage without mutating weights."""

        if self.protocol.state_gradient_ratio is None:
            if pairs:
                raise ValueError("output-only training cannot calibrate a state loss")
            return {}
        rows = tuple(pairs)
        if len(rows) != self.protocol.calibration_micro_batches:
            raise ValueError("state calibration pair count differs from the config")
        output_norms: list[float] = []
        state_norms: list[float] = []
        record_ids: list[str] = []
        for pair in rows:
            if pair.is_noop or not pair.edits:
                raise ValueError("state calibration requires noisy edited pairs")
            output = self._step_output(pair, state_weight=1.0)
            output_norm = self._gradient_norm(output.losses["output"], retain_graph=True)
            state_norm = self._gradient_norm(output.losses["state"], retain_graph=False)
            if (
                not math.isfinite(output_norm)
                or not math.isfinite(state_norm)
                or output_norm <= 0.0
                or state_norm <= 0.0
            ):
                raise FloatingPointError("state calibration produced an invalid gradient norm")
            output_norms.append(output_norm)
            state_norms.append(state_norm)
            record_ids.append(pair.record_id)
        mean_output = sum(output_norms) / len(output_norms)
        mean_state = sum(state_norms) / len(state_norms)
        rho = float(self.protocol.state_gradient_ratio)
        state_weight = rho * mean_output / mean_state
        if not math.isfinite(state_weight) or state_weight <= 0.0:
            raise FloatingPointError("calibrated state weight is invalid")
        if self.protocol.condition == "probe-transition-single-layer-state-distillation":
            calibration: dict[str, object] = {
                "schema_version": _STATE_CALIBRATION_SCHEMA,
                "micro_batches": len(rows),
                "noisy_micro_batches": len(rows),
                "record_ids": record_ids,
                "output_gradient_norms": output_norms,
                "state_gradient_norms": state_norms,
                "mean_output_gradient_norm": mean_output,
                "mean_state_gradient_norm": mean_state,
                "target_gradient_ratio": rho,
                "state_weight": state_weight,
                "achieved_initial_ratio": state_weight * mean_state / mean_output,
                "replay_relative_tolerance": _STATE_CALIBRATION_REPLAY_REL_TOL,
                "replay_absolute_tolerance": _STATE_CALIBRATION_REPLAY_ABS_TOL,
            }
            _validate_priority_b_calibration(
                protocol=self.protocol,
                state_weight=state_weight,
                calibration=calibration,
            )
        else:
            calibration = {
                "schema_version": "state-gradient-calibration/v1",
                "micro_batches": len(rows),
                "record_ids": record_ids,
                "output_gradient_norms": output_norms,
                "state_gradient_norms": state_norms,
                "mean_output_gradient_norm": mean_output,
                "mean_state_gradient_norm": mean_state,
                "target_gradient_ratio": rho,
                "state_weight": state_weight,
                "achieved_initial_ratio": state_weight * mean_state / mean_output,
            }
        return calibration

    def calibrate_state_weight(
        self,
        pairs: Sequence[TrainingPair],
    ) -> Mapping[str, object]:
        """Freeze lambda from initial output/state LoRA gradient norms."""

        if self.state_weight is not None or self.state_calibration is not None:
            raise RuntimeError("state gradient weight is already calibrated")
        try:
            calibration = self._measure_state_calibration(pairs)
        finally:
            self._prepared_encodings.clear()
            self.zero_grad()
        self.state_weight = float(calibration["state_weight"])
        self.state_calibration = calibration
        return dict(calibration)

    def verify_resume_state_calibration(
        self,
        path: Path,
        pairs: Sequence[TrainingPair],
    ) -> None:
        """Replay calibration on the attested initial model before checkpoint mutation."""

        if self.protocol.condition != "probe-transition-single-layer-state-distillation":
            raise ValueError("calibration replay is exclusive to Priority B state training")
        if self.state_weight is not None or self.state_calibration is not None:
            raise RuntimeError("calibration replay requires an unmodified initial runtime")
        state_path = Path(path).resolve()
        before_sha = sha256_file(state_path)
        payload = self._torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != _RUNTIME_STATE_SCHEMA
            or payload.get("condition") != self.protocol.condition
            or payload.get("config_sha256") != self.protocol.config_sha256
            or payload.get("seed") != self.seed
        ):
            raise ValueError("adapter runtime checkpoint identity differs")
        try:
            replayed = self._measure_state_calibration(pairs)
            _validate_replayed_priority_b_calibration(
                protocol=self.protocol,
                saved_state_weight=payload.get("state_weight"),
                saved_calibration=payload.get("state_calibration"),
                replayed_calibration=replayed,
            )
        finally:
            self._prepared_encodings.clear()
            self.zero_grad()
        after_sha = sha256_file(state_path)
        if before_sha != after_sha:
            raise ValueError("adapter runtime checkpoint changed during calibration replay")
        if any(parameter.grad is not None for parameter in self._trainable_parameters()):
            raise RuntimeError("calibration replay left trainable parameter gradients")
        self._verified_resume_state_path = state_path
        self._verified_resume_state_sha256 = after_sha

    def train_micro_step(
        self,
        pair: TrainingPair,
        *,
        loss_scale: float,
        measure_gradient_ratio: bool = False,
        output_loss_scale: float | None = None,
        state_loss_scale: float | None = None,
    ) -> TrainingMicroStepResult:
        if not math.isfinite(float(loss_scale)) or float(loss_scale) <= 0.0:
            raise ValueError("training micro-step loss scale must be positive")
        if self.state_weight is None:
            raise RuntimeError("state loss must be calibrated before training")
        output = self._step_output(pair, state_weight=self.state_weight)
        normalized_objective = not self.protocol.schema_version.endswith("/v1")
        if normalized_objective:
            if (
                output_loss_scale is None
                or state_loss_scale is None
                or not math.isfinite(float(output_loss_scale))
                or not math.isfinite(float(state_loss_scale))
                or float(output_loss_scale) < 0.0
                or float(state_loss_scale) < 0.0
            ):
                raise ValueError("cycle-2 loss scales must be finite and non-negative")
            backward_loss = output.losses["output"] * float(
                self.protocol.loss_weights["output"]
            ) * float(output_loss_scale) + output.losses["state"] * float(
                self.protocol.loss_weights["state"]
            ) * float(self.state_weight) * float(state_loss_scale)
        else:
            if output_loss_scale is not None or state_loss_scale is not None:
                raise ValueError("legacy training cannot consume cycle-2 loss scales")
            backward_loss = output.loss * float(loss_scale)
        if measure_gradient_ratio:
            output_norm = self._gradient_norm(
                output.losses["output"]
                * float(output_loss_scale if normalized_objective else loss_scale),
                retain_graph=True,
            )
            weighted_state_norm = 0.0
            if self.protocol.loss_weights["state"] > 0.0 and not pair.is_noop:
                weighted_state_norm = self._gradient_norm(
                    output.losses["state"]
                    * self.state_weight
                    * float(state_loss_scale if normalized_objective else loss_scale),
                    retain_graph=True,
                )
            ratio = weighted_state_norm / max(output_norm, 1e-12)
            share = weighted_state_norm / max(output_norm + weighted_state_norm, 1e-12)
            self._last_gradient_diagnostics = {
                "output_gradient_norm": output_norm,
                "weighted_state_gradient_norm": weighted_state_norm,
                "state_to_output_gradient_ratio": ratio,
                "state_gradient_share": share,
            }
            self._gradient_ratio_violations = next_gradient_ratio_violations(
                self._gradient_ratio_violations,
                ratio=ratio,
                optimizer_steps=self._optimizer_steps,
                guard_steps=self.protocol.gradient_ratio_guard_optimizer_steps,
            )
            if self._gradient_ratio_violations >= 3:
                raise RuntimeError("state gradient ratio exceeded 0.5 for three startup checks")
        backward_loss.backward()
        losses = {
            name: float(value.detach().float().cpu()) for name, value in output.losses.items()
        }
        losses["state_weight"] = float(self.state_weight)
        losses["weighted_state"] = losses.get("state", 0.0) * float(self.state_weight)
        if normalized_objective:
            losses["output_accumulation_scale"] = float(output_loss_scale or 0.0)
            losses["state_accumulation_scale"] = float(state_loss_scale or 0.0)
        losses["backward_contribution"] = float(backward_loss.detach().float().cpu())
        total = float(output.loss.detach().float().cpu())
        if not math.isfinite(total) or any(not math.isfinite(value) for value in losses.values()):
            raise FloatingPointError("training micro-step produced non-finite logged losses")
        return TrainingMicroStepResult(
            losses=losses,
            total_loss=total,
            student_tokens=output.student_tokens,
        )

    def optimizer_step(self, *, max_grad_norm: float) -> tuple[float, float]:
        torch = self._torch
        parameters = [
            parameter for parameter in self.student.parameters() if parameter.requires_grad
        ]
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
        value = float(norm.detach().float().cpu())
        if not math.isfinite(value):
            raise FloatingPointError("adapter gradient norm is non-finite")
        self.optimizer.step()
        self.scheduler.step()
        self._optimizer_steps += 1
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        return value, learning_rate

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def telemetry(self) -> dict[str, int | float]:
        torch = self._torch
        return {
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "gpu_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "state_weight": float(self.state_weight or 0.0),
            **self._last_gradient_diagnostics,
        }

    def _monitor_tokenize(self, text: str) -> tuple[object, object, tuple[tuple[int, int], ...]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.protocol.max_sequence_length,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("training monitor tokenizer must return a mapping")
        ids = encoded.get("input_ids")
        mask = encoded.get("attention_mask")
        offsets = encoded.get("offset_mapping")
        if (
            not isinstance(ids, list)
            or not isinstance(mask, list)
            or not isinstance(offsets, list)
            or len(ids) < 2
            or len(ids) != len(mask)
            or len(ids) != len(offsets)
        ):
            raise ValueError("training monitor tokenizer fields differ")
        normalized_offsets: list[tuple[int, int]] = []
        for offset in offsets:
            if (
                not isinstance(offset, (tuple, list))
                or len(offset) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in offset)
            ):
                raise ValueError("training monitor tokenizer offsets differ")
            normalized_offsets.append((offset[0], offset[1]))
        return (
            self._torch.tensor([ids], dtype=self._torch.long, device=self.device),
            self._torch.tensor([mask], dtype=self._torch.long, device=self.device),
            tuple(normalized_offsets),
        )

    def monitor(self, records: Sequence[object]) -> Mapping[str, float]:
        """Measure fixed T0 clean drift and natural typo KL without task accuracy."""

        from typo_robust_training.evaluation.data import EvaluationCorpusRecord
        from typo_robust_training.evaluation.runtime import (
            aligned_forward_kl_sum,
            causal_nll_and_forward_kl,
            causal_nll_sum,
        )
        from typo_robust_training.training.pairs import align_unchanged_token_positions

        rows = tuple(records)
        if not rows or any(not isinstance(row, EvaluationCorpusRecord) for row in rows):
            raise ValueError("training monitor requires frozen evaluation corpus records")
        if self._monitor_base_clean is None:
            base_nll, clean_tokens = 0.0, 0
        else:
            base_nll, clean_tokens = self._monitor_base_clean
        student_nll = 0.0
        student_clean_tokens = 0
        clean_kl = 0.0
        clean_kl_tokens = 0
        if self._monitor_base_natural is None:
            base_natural_kl, natural_tokens = 0.0, 0
        else:
            base_natural_kl, natural_tokens = self._monitor_base_natural
        student_natural_kl = 0.0
        student_natural_tokens = 0
        was_training = self.student.training
        self.student.eval()
        with self._torch.inference_mode():
            for record in rows:
                clean_ids, clean_mask, clean_offsets = self._monitor_tokenize(record.clean_text)
                student_clean = self.student(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    use_cache=False,
                )
                if record.source == "fineweb_edu":
                    teacher_clean = self.teacher(
                        input_ids=clean_ids,
                        attention_mask=clean_mask,
                        use_cache=False,
                    )
                    candidate_nll, candidate_count, kl_sum, kl_count = causal_nll_and_forward_kl(
                        student_clean.logits,
                        clean_ids,
                        base_logits=teacher_clean.logits,
                    )
                    if self._monitor_base_clean is None:
                        teacher_nll, count = causal_nll_sum(teacher_clean.logits, clean_ids)
                        if candidate_count != count:
                            raise RuntimeError("training monitor clean token counts differ")
                        base_nll += teacher_nll
                        clean_tokens += count
                    student_nll += candidate_nll
                    student_clean_tokens += candidate_count
                    clean_kl += kl_sum
                    clean_kl_tokens += kl_count
                if record.kind == "natural":
                    if record.typo_text is None or len(record.edits) != 1:
                        raise ValueError("training monitor natural pair lost its edit")
                    typo_ids, typo_mask, typo_offsets = self._monitor_tokenize(record.typo_text)
                    student_typo = self.student(
                        input_ids=typo_ids,
                        attention_mask=typo_mask,
                        use_cache=False,
                    )
                    edit = record.edits[0]
                    aligned = align_unchanged_token_positions(
                        clean_text=record.clean_text,
                        typo_text=record.typo_text,
                        clean_edit_spans=(edit.clean_char_span,),
                        typo_edit_spans=(edit.typo_char_span,),
                        clean_offsets=clean_offsets,
                        typo_offsets=typo_offsets,
                    )
                    student_gap, student_count = aligned_forward_kl_sum(
                        student_clean.logits,
                        student_typo.logits,
                        token_pairs=aligned,
                    )
                    if self._monitor_base_natural is None:
                        teacher_clean = self.teacher(
                            input_ids=clean_ids,
                            attention_mask=clean_mask,
                            use_cache=False,
                        )
                        teacher_typo = self.teacher(
                            input_ids=typo_ids,
                            attention_mask=typo_mask,
                            use_cache=False,
                        )
                        teacher_gap, teacher_count = aligned_forward_kl_sum(
                            teacher_clean.logits,
                            teacher_typo.logits,
                            token_pairs=aligned,
                        )
                        if student_count != teacher_count:
                            raise RuntimeError("training monitor natural token counts differ")
                        base_natural_kl += teacher_gap
                        natural_tokens += teacher_count
                    elif student_count <= 0:
                        raise RuntimeError("training monitor natural token counts differ")
                    student_natural_kl += student_gap
                    student_natural_tokens += student_count
        if was_training:
            self.student.train()
        if (
            clean_tokens <= 0
            or student_clean_tokens != clean_tokens
            or clean_kl_tokens <= 0
            or natural_tokens <= 0
            or student_natural_tokens != natural_tokens
        ):
            raise ValueError("training monitor has no valid frozen tokens")
        if self._monitor_base_clean is None:
            self._monitor_base_clean = (base_nll, clean_tokens)
        if self._monitor_base_natural is None:
            self._monitor_base_natural = (base_natural_kl, natural_tokens)
        base_gap = base_natural_kl / natural_tokens
        student_gap = student_natural_kl / natural_tokens
        log_ppl_ratio = student_nll / clean_tokens - base_nll / clean_tokens
        return {
            "clean_kl_nats_per_token": clean_kl / clean_kl_tokens,
            "fineweb_edu_ppl_ratio": _finite_ppl_ratio(log_ppl_ratio),
            "base_natural_clean_typo_kl": base_gap,
            "natural_clean_typo_kl": student_gap,
            "natural_clean_typo_kl_ratio": student_gap / max(base_gap, 1e-12),
            "clean_documents": float(sum(record.source == "fineweb_edu" for record in rows)),
            "natural_pairs": float(sum(record.kind == "natural" for record in rows)),
        }

    def save_state(self, path: Path) -> None:
        from peft import get_peft_model_state_dict

        state_path = Path(path).resolve()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_state = get_peft_model_state_dict(self.student)
        optimizer_state = self.optimizer.state_dict()
        payload = {
            "schema_version": _RUNTIME_STATE_SCHEMA,
            "condition": self.protocol.condition,
            "config_sha256": self.protocol.config_sha256,
            "seed": self.seed,
            "optimizer_steps": self._optimizer_steps,
            "adapter": adapter_state,
            "optimizer": optimizer_state,
            "adapter_scope": _adapter_scope_contract(
                adapter_state=adapter_state,
                optimizer_state=optimizer_state,
                optimizer_parameter_names=self._optimizer_parameter_names,
            ),
            "scheduler": self.scheduler.state_dict(),
            "state_weight": self.state_weight,
            "state_calibration": self.state_calibration,
            "gradient_ratio_violations": self._gradient_ratio_violations,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": self._torch.get_rng_state(),
            "cuda_rng": self._torch.cuda.get_rng_state_all(),
        }
        temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
        try:
            self._torch.save(payload, temporary)
            os.replace(temporary, state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_state(
        self,
        path: Path,
        *,
        expected_state_calibration: Mapping[str, object] | None = None,
        expected_state_sha256: str | None = None,
    ) -> None:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        supplied_state_path = Path(path)
        state_path = supplied_state_path.resolve()
        serialized_state: io.BytesIO | Path = state_path
        if expected_state_sha256 is not None:
            if not is_mistral_factorial_protocol(self.protocol):
                raise ValueError("attested runtime-state loading is reserved for Mistral v8")
            # Deserialize exactly the bytes whose digest was checked.  Reopening
            # the path inside torch.load would leave a check/use substitution gap.
            serialized_state = _mistral_attested_state_buffer(
                supplied_state_path,
                expected_sha256=expected_state_sha256,
            )
        if self.protocol.condition == "probe-transition-single-layer-state-distillation":
            if (
                self._verified_resume_state_path != state_path
                or self._verified_resume_state_sha256 is None
                or sha256_file(state_path) != self._verified_resume_state_sha256
            ):
                raise ValueError(
                    "Priority B checkpoint must pass exact calibration replay before loading"
                )
        payload = self._torch.load(
            serialized_state,
            map_location="cpu",
            weights_only=False,
        )
        if (
            self.protocol.condition == "probe-transition-single-layer-state-distillation"
            and sha256_file(state_path) != self._verified_resume_state_sha256
        ):
            raise ValueError("Priority B checkpoint changed after calibration replay")
        expected_v1 = {
            "schema_version",
            "condition",
            "config_sha256",
            "seed",
            "optimizer_steps",
            "adapter",
            "optimizer",
            "scheduler",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
        }
        expected_v2 = expected_v1 | {
            "state_weight",
            "state_calibration",
            "gradient_ratio_violations",
        }
        expected_v3 = expected_v2 | {"adapter_scope"}
        schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
        expected_fields = (
            expected_v3
            if schema_version == _RUNTIME_STATE_SCHEMA
            else expected_v2
            if isinstance(schema_version, str) and schema_version.endswith("/v2")
            else expected_v1
        )
        if (
            not isinstance(payload, dict)
            or schema_version
            not in {
                "robustness-adapter-runtime-state/v1",
                "robustness-adapter-runtime-state/v2",
                _RUNTIME_STATE_SCHEMA,
            }
            or set(payload) != expected_fields
        ):
            raise ValueError("adapter runtime checkpoint fields differ")
        if (
            payload["condition"] != self.protocol.condition
            or payload["config_sha256"] != self.protocol.config_sha256
            or payload["seed"] != self.seed
            or isinstance(payload["optimizer_steps"], bool)
            or not isinstance(payload["optimizer_steps"], int)
            or payload["optimizer_steps"] < 0
            or (
                is_mistral_factorial_protocol(self.protocol)
                and payload["optimizer_steps"] > self.protocol.max_optimizer_steps
            )
        ):
            raise ValueError("adapter runtime checkpoint identity differs")
        resumed_state: tuple[float, dict[str, object] | None] | None = None
        if payload["schema_version"].endswith(("/v2", "/v3")):
            if self.protocol.condition == "probe-transition-single-layer-state-distillation":
                if expected_state_calibration is not None:
                    raise ValueError("Priority B resume cannot substitute calibration evidence")
                _validate_priority_b_calibration(
                    protocol=self.protocol,
                    state_weight=payload["state_weight"],
                    calibration=payload["state_calibration"],
                )
                resumed_state = (
                    float(payload["state_weight"]),
                    dict(payload["state_calibration"]),
                )
            else:
                resumed_state = _validated_resume_state_calibration(
                    protocol=self.protocol,
                    state_weight=payload["state_weight"],
                    calibration=payload["state_calibration"],
                    expected_calibration=expected_state_calibration,
                )
            violations = payload["gradient_ratio_violations"]
            if isinstance(violations, bool) or not isinstance(violations, int) or violations < 0:
                raise ValueError("adapter runtime checkpoint gradient counter differs")
        elif self.protocol.state_gradient_ratio is not None:
            raise ValueError("cycle-2 state training cannot resume a legacy runtime state")
        _validate_adapter_scope_before_resume(
            checkpoint_adapter=payload["adapter"],
            checkpoint_optimizer=payload["optimizer"],
            checkpoint_scope=payload.get("adapter_scope"),
            current_adapter=get_peft_model_state_dict(self.student),
            current_optimizer=self.optimizer.state_dict(),
            current_optimizer_parameter_names=self._optimizer_parameter_names,
        )
        set_peft_model_state_dict(self.student, payload["adapter"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self._optimizer_steps = int(payload["optimizer_steps"])
        if resumed_state is not None:
            self.state_weight, self.state_calibration = resumed_state
            self._gradient_ratio_violations = payload["gradient_ratio_violations"]
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        self._torch.set_rng_state(payload["torch_rng"].cpu())
        self._torch.cuda.set_rng_state_all(_cpu_cuda_rng_states(payload["cuda_rng"]))
        self._verified_resume_state_path = None
        self._verified_resume_state_sha256 = None

    def verify_resume_optimizer_step(self, expected_optimizer_steps: int) -> None:
        """Bind the opaque optimizer/scheduler state to the runner checkpoint cursor."""

        if (
            isinstance(expected_optimizer_steps, bool)
            or not isinstance(expected_optimizer_steps, int)
            or expected_optimizer_steps < 0
            or self._optimizer_steps != expected_optimizer_steps
            or getattr(self.scheduler, "last_epoch", None) != expected_optimizer_steps
        ):
            raise ValueError("adapter runtime optimizer state differs from the resume cursor")

    def save_adapter(self, path: Path) -> None:
        output = Path(path).resolve()
        output.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(output, safe_serialization=True)
        self.tokenizer.save_pretrained(output)
        (output / "training_runtime.json").write_text(
            json.dumps(self.provenance(), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def provenance(self) -> dict[str, object]:
        torch = self._torch
        return {
            "runtime": "HuggingFaceAdapterTrainingRuntime/v2",
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "accelerate": _version("accelerate"),
            "peft": _version("peft"),
            "model": self.protocol.model,
            "requested_revision": self.protocol.model_revision,
            "teacher_revision": self.teacher_revision,
            "student_revision": self.student_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_snapshot_attestation": (
                self.tokenizer_snapshot_attestation.provenance_dict()
            ),
            "code_revision": self.code_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "condition": self.protocol.condition,
            "method_identity": self.protocol.method_identity,
            "seed": self.seed,
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "decoder_layers": self.num_decoder_layers,
            "adapter_layers": list(self.adapter_layers),
            "adapter_initialization_policy": self.protocol.adapter_initialization_policy,
            "output_scope": self.protocol.output_scope,
            "state_layers": list(self.state_layers),
            **(
                {"method_evidence_sha256": self.evidence.evidence_sha256}
                if isinstance(
                    self.evidence,
                    (
                        ProbeTransitionTrainingEvidence,
                        ProbeTransitionStateTrainingEvidence,
                        ProbeSemanticSubspaceTrainingEvidence,
                    ),
                )
                else {}
            ),
            "state_weight": self.state_weight,
            "state_calibration": self.state_calibration,
            "adapter_modules": list(self.parameter_report.modules),
            "trainable_parameters": self.parameter_report.trainable_parameters,
            "total_parameters": self.parameter_report.total_parameters,
            "attention_head_dim": self.attention_head_dim,
            "training_teacher_mode": (
                "same-student-model-with-adapter-disabled"
                if is_kojima_faithful_protocol(self.protocol)
                else "separate-frozen-clean-model"
            ),
            "teacher_frozen": True,
            "student_base_frozen": True,
        }


__all__ = [
    "HuggingFaceAdapterTrainingRuntime",
    "next_gradient_ratio_violations",
    "resolve_attention_head_dim",
    "validate_resume_state_calibration",
]
