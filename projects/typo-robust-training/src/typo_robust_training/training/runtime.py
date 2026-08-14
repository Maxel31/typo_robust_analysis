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
from typo_robust_training.training.pairs import TrainingPair
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


class HuggingFaceAdapterTrainingRuntime:
    """Frozen clean teacher plus a typo student whose LoRA is the only update."""

    def __init__(
        self,
        *,
        protocol: AdapterTrainingProtocol,
        seed: int,
        gpu_id: str,
        evidence: LocalizationEvidence | ResidualStateEvidence | None,
    ) -> None:
        if not isinstance(protocol, AdapterTrainingProtocol):
            raise TypeError("training runtime protocol must be AdapterTrainingProtocol")
        if seed not in protocol.seed_inventory:
            raise ValueError("training runtime seed is outside the frozen inventory")
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
        if protocol.layer_scope == "all-decoder-layers":
            adapter_layers = tuple(range(self.num_decoder_layers))
        else:
            if not isinstance(evidence, LocalizationEvidence):
                raise ValueError("targeted adapter training requires localization evidence")
            adapter_layers = evidence.adapter_layers
        self.adapter_layers = adapter_layers
        self.component_weights = (
            evidence.component_weights if isinstance(evidence, LocalizationEvidence) else None
        )
        if isinstance(evidence, ResidualStateEvidence):
            self.state_layers = evidence.state_layers
        elif protocol.state_scope == "all-layers-edited-word-final-tokens":
            self.state_layers = tuple(range(self.num_decoder_layers))
        else:
            self.state_layers = ()
        residual_scope = protocol.state_scope in {
            "causal-window-edited-word-final-tokens",
            "random-window-edited-word-final-tokens",
            "all-layers-edited-word-final-tokens",
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
        trainable = [
            parameter for parameter in self.student.parameters() if parameter.requires_grad
        ]
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
        except ValueError as exc:
            if str(exc) in {
                "edited words resolve to duplicate token positions",
                "training edited-word token cardinalities differ",
                "training pair has no aligned non-edited next-token targets",
                "training typo edit falls outside the retained token window",
            }:
                return False
            raise
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
            "schema_version": "state-gradient-calibration/v1",
            "micro_batches": len(rows),
            "record_ids": record_ids,
            "output_gradient_norms": output_norms,
            "state_gradient_norms": state_norms,
            "mean_output_gradient_norm": mean_output,
            "mean_state_gradient_norm": mean_state,
            "target_gradient_ratio": rho,
            "state_weight": self.state_weight,
            "achieved_initial_ratio": self.state_weight * mean_state / mean_output,
        }
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
        payload = {
            "schema_version": "robustness-adapter-runtime-state/v2",
            "condition": self.protocol.condition,
            "config_sha256": self.protocol.config_sha256,
            "seed": self.seed,
            "optimizer_steps": self._optimizer_steps,
            "adapter": get_peft_model_state_dict(self.student),
            "optimizer": self.optimizer.state_dict(),
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
        from peft import set_peft_model_state_dict

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
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            not in {
                "robustness-adapter-runtime-state/v1",
                "robustness-adapter-runtime-state/v2",
            }
            or set(payload)
            != (expected_v2 if payload.get("schema_version", "").endswith("/v2") else expected_v1)
        ):
            raise ValueError("adapter runtime checkpoint fields differ")
        if (
            payload["condition"] != self.protocol.condition
            or payload["config_sha256"] != self.protocol.config_sha256
            or payload["seed"] != self.seed
        ):
            raise ValueError("adapter runtime checkpoint identity differs")
        set_peft_model_state_dict(self.student, payload["adapter"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self._optimizer_steps = int(payload["optimizer_steps"])
        if payload["schema_version"].endswith("/v2"):
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
            "teacher_revision": getattr(self.teacher.config, "_commit_hash", None),
            "student_revision": getattr(self.student.config, "_commit_hash", None),
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
            "state_weight": self.state_weight,
            "state_calibration": self.state_calibration,
            "adapter_modules": list(self.parameter_report.modules),
            "trainable_parameters": self.parameter_report.trainable_parameters,
            "total_parameters": self.parameter_report.total_parameters,
            "attention_head_dim": self.attention_head_dim,
            "teacher_frozen": True,
            "student_base_frozen": True,
        }


__all__ = ["HuggingFaceAdapterTrainingRuntime", "next_gradient_ratio_violations"]
