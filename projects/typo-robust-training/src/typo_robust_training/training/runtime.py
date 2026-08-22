"""Single-GPU Hugging Face/PEFT runtime for all training conditions."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from typo_robust_training.probe.runtime import (
    _checkout_code_revision,
    _require_exact_model_revision,
)
from typo_robust_training.training.adapters import (
    TrainableParameterReport,
    attach_lora_adapters,
    trainable_parameter_report,
)
from typo_robust_training.training.config import AdapterTrainingProtocol
from typo_robust_training.training.encoding import (
    PairedEncoding,
    encode_training_pair,
    retained_clean_character_extent,
)
from typo_robust_training.training.evidence import (
    LocalizationEvidence,
    ResidualStateEvidence,
)
from typo_robust_training.training.methods import (
    ProbeTransitionStateTrainingEvidence,
    ProbeTransitionTrainingEvidence,
    ResolvedTrainingMethod,
    resolve_training_method,
)
from typo_robust_training.training.pairs import TrainingPair, UnusableTrainingPairError
from typo_robust_training.training.runner import (
    TrainingMicroStepResult,
    TrainingMicroStepScales,
    normalized_accumulation_scales,
)
from typo_robust_training.training.step import compute_training_step


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


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
_ADAPTER_SCOPE_SCHEMA = "decoder-lora-optimizer-scope/v1"
_STATE_CALIBRATION_SCHEMA = "state-gradient-calibration/v2"
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
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
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
    if not math.isclose(stored_weight, derived_weight, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(
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
        | None
    ),
) -> ResolvedTrainingMethod | None:
    """Fail closed on the v4 evidence and output-only boundary before CUDA setup."""

    is_probe_condition = protocol.condition in {
        "probe-transition-output-matching",
        "probe-transition-single-layer-state-distillation",
    }
    if not is_probe_condition:
        if isinstance(
            evidence,
            (ProbeTransitionTrainingEvidence, ProbeTransitionStateTrainingEvidence),
        ):
            raise ValueError("probe-transition evidence cannot configure this condition")
        return None
    if not isinstance(
        evidence,
        (ProbeTransitionTrainingEvidence, ProbeTransitionStateTrainingEvidence),
    ):
        raise ValueError("probe-transition output matching requires probe evidence")
    resolved = resolve_training_method(protocol, evidence=evidence)
    if protocol.condition == "probe-transition-output-matching":
        expected_weights = {
            "noisy_language_model": 0.0,
            "answer": 0.0,
            "output": 1.0,
            "state": 0.0,
            "clean": 0.0,
        }
        valid = (
            dict(protocol.loss_weights) == expected_weights
            and not resolved.state_layers
            and resolved.state_target == "none"
            and protocol.state_scope == "none"
            and protocol.state_distance == "none"
            and protocol.state_gradient_ratio is None
            and protocol.calibration_micro_batches == 0
        )
    else:
        expected_weights = {
            "noisy_language_model": 0.0,
            "answer": 0.0,
            "output": 1.0,
            "state": 1.0,
            "clean": 0.0,
        }
        valid = (
            isinstance(evidence, ProbeTransitionStateTrainingEvidence)
            and dict(protocol.loss_weights) == expected_weights
            and resolved.state_layers == (evidence.selected_transition_layer,)
            and resolved.state_target
            == "complete-decoder-block-residual-output-at-edited-word-final/v1"
            and protocol.state_scope
            == "probe-transition-single-layer-edited-word-final-token/v1"
            and protocol.state_distance == "cosine-residual/v1"
            and protocol.state_gradient_ratio == 0.05
            and protocol.calibration_micro_batches == 8
            and protocol.temperature == 1.0
            and protocol.epsilon == 1e-8
        )
    if not valid:
        if protocol.condition == "probe-transition-output-matching":
            raise ValueError(
                "probe-transition output matching must disable state training"
            )
        raise ValueError("probe-transition state training objective or evidence differs")
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
            | None
        ),
    ) -> None:
        if not isinstance(protocol, AdapterTrainingProtocol):
            raise TypeError("training runtime protocol must be AdapterTrainingProtocol")
        if seed not in protocol.seed_inventory:
            raise ValueError("training runtime seed is outside the frozen inventory")
        resolved_method = _resolve_probe_transition_runtime_method(protocol, evidence)
        self.code_revision = _checkout_code_revision()
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
        if teacher_tokenizer_revision != student_tokenizer_revision:
            raise ValueError("teacher and student tokenizer revisions differ")
        self.tokenizer_revision = student_tokenizer_revision
        base_layers = find_decoder_layers(student_base)
        self.num_decoder_layers = len(base_layers)
        if (
            protocol.decoder_layers is not None
            and protocol.decoder_layers != self.num_decoder_layers
        ):
            raise ValueError("training model decoder layers differ from the frozen config")
        text_config = getattr(student_base.config, "text_config", student_base.config)
        self.attention_head_dim = getattr(text_config, "head_dim", None)
        self.mlp_intermediate_size = getattr(text_config, "intermediate_size", None)
        self.attention_heads = getattr(text_config, "num_attention_heads", None)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                self.attention_head_dim,
                self.mlp_intermediate_size,
                self.attention_heads,
            )
        ):
            raise ValueError("training model architecture fields are unavailable")
        if resolved_method is not None:
            adapter_layers = resolved_method.adapter_layers
        elif protocol.layer_scope == "all-decoder-layers":
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
        if residual_scope != bool(self.state_layers):
            raise ValueError("residual state layers differ from the training objective")
        self.student = attach_lora_adapters(
            student_base,
            protocol=protocol,
            decoder_layers=adapter_layers,
        )
        self.device = next(self.student.parameters()).device
        if next(self.teacher.parameters()).device != self.device:
            raise ValueError("teacher and student must share one training device")
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
        self._monitor_base_clean: tuple[float, int] | None = None
        self._monitor_base_natural: tuple[float, int] | None = None
        torch.cuda.reset_peak_memory_stats()

    def _encode_pair(self, pair: TrainingPair) -> PairedEncoding:
        return encode_training_pair(
            pair,
            tokenizer=self.tokenizer,
            max_length=self.protocol.max_sequence_length,
            require_answer_targets=self.protocol.loss_weights["answer"] > 0.0,
            require_all_edits_visible=not self.protocol.schema_version.endswith("/v1"),
        )

    def pair_is_usable(self, pair: TrainingPair) -> bool:
        """Return whether a generated pair supplies every frozen training target."""

        try:
            self._encode_pair(pair)
        except UnusableTrainingPairError:
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
            teacher=self.teacher,
            student=self.student,
            encoding=encoding,
            protocol=self.protocol,
            component_weights=self.component_weights,
            attention_head_dim=self.attention_head_dim,
            state_layers=self.state_layers,
            state_weight=state_weight,
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
        scales = normalized_accumulation_scales(
            output_token_counts=tuple(len(encoding.output_logit_pairs) for encoding in encodings),
            state_coordinate_counts=tuple(
                len(encoding.clean_edit_positions) if state_active else 0 for encoding in encodings
            ),
            state_active=state_active,
        )
        self._prepared_encodings.extend(zip(rows, encodings, strict=True))
        return scales

    def calibrate_state_weight(
        self,
        pairs: Sequence[TrainingPair],
    ) -> Mapping[str, object]:
        """Freeze lambda from initial output/state LoRA gradient norms."""

        if self.protocol.state_gradient_ratio is None:
            if pairs:
                raise ValueError("output-only training cannot calibrate a state loss")
            return {}
        rows = tuple(pairs)
        if len(rows) != self.protocol.calibration_micro_batches:
            raise ValueError("state calibration pair count differs from the config")
        if self.state_weight is not None or self.state_calibration is not None:
            raise RuntimeError("state gradient weight is already calibrated")
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
        self.state_weight = rho * mean_output / mean_state
        if not math.isfinite(self.state_weight) or self.state_weight <= 0.0:
            raise FloatingPointError("calibrated state weight is invalid")
        self.state_calibration = {
            "schema_version": _STATE_CALIBRATION_SCHEMA,
            "micro_batches": len(rows),
            "noisy_micro_batches": len(rows),
            "record_ids": record_ids,
            "output_gradient_norms": output_norms,
            "state_gradient_norms": state_norms,
            "mean_output_gradient_norm": mean_output,
            "mean_state_gradient_norm": mean_state,
            "target_gradient_ratio": rho,
            "state_weight": self.state_weight,
            "achieved_initial_ratio": self.state_weight * mean_state / mean_output,
        }
        _validate_priority_b_calibration(
            protocol=self.protocol,
            state_weight=self.state_weight,
            calibration=self.state_calibration,
        )
        self.zero_grad()
        return dict(self.state_calibration)

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

    def load_state(self, path: Path) -> None:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        payload = self._torch.load(
            Path(path).resolve(),
            map_location="cpu",
            weights_only=False,
        )
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
        ):
            raise ValueError("adapter runtime checkpoint identity differs")
        if payload["schema_version"].endswith(("/v2", "/v3")):
            _validate_priority_b_calibration(
                protocol=self.protocol,
                state_weight=payload["state_weight"],
                calibration=payload["state_calibration"],
            )
        elif self.protocol.condition == "probe-transition-single-layer-state-distillation":
            raise ValueError("probe-transition state training cannot resume a legacy runtime state")
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
        if payload["schema_version"].endswith(("/v2", "/v3")):
            state_weight = payload["state_weight"]
            if (
                isinstance(state_weight, bool)
                or not isinstance(state_weight, (int, float))
                or not math.isfinite(float(state_weight))
                or float(state_weight) < 0.0
            ):
                raise ValueError("adapter runtime checkpoint state weight differs")
            self.state_weight = float(state_weight)
            calibration = payload["state_calibration"]
            if calibration is not None and not isinstance(calibration, dict):
                raise ValueError("adapter runtime checkpoint calibration differs")
            self.state_calibration = calibration
            violations = payload["gradient_ratio_violations"]
            if isinstance(violations, bool) or not isinstance(violations, int) or violations < 0:
                raise ValueError("adapter runtime checkpoint gradient counter differs")
            self._gradient_ratio_violations = violations
        elif self.protocol.state_gradient_ratio is not None:
            raise ValueError("cycle-2 state training cannot resume a legacy runtime state")
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        self._torch.set_rng_state(payload["torch_rng"].cpu())
        self._torch.cuda.set_rng_state_all(_cpu_cuda_rng_states(payload["cuda_rng"]))

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
            "code_revision": self.code_revision,
            "condition": self.protocol.condition,
            "seed": self.seed,
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "decoder_layers": self.num_decoder_layers,
            "adapter_layers": list(self.adapter_layers),
            "state_layers": list(self.state_layers),
            **(
                {"method_evidence_sha256": self.evidence.evidence_sha256}
                if isinstance(
                    self.evidence,
                    (ProbeTransitionTrainingEvidence, ProbeTransitionStateTrainingEvidence),
                )
                else {}
            ),
            "state_weight": self.state_weight,
            "state_calibration": self.state_calibration,
            "adapter_modules": list(self.parameter_report.modules),
            "trainable_parameters": self.parameter_report.trainable_parameters,
            "total_parameters": self.parameter_report.total_parameters,
            "attention_head_dim": self.attention_head_dim,
            "teacher_frozen": True,
            "student_base_frozen": True,
        }


__all__ = [
    "HuggingFaceAdapterTrainingRuntime",
    "next_gradient_ratio_violations",
    "validate_resume_state_calibration",
]
