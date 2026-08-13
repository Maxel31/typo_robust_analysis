"""Single-GPU Hugging Face/PEFT runtime for all training conditions."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from typo_robust_training.training.adapters import (
    TrainableParameterReport,
    attach_lora_adapters,
    trainable_parameter_report,
)
from typo_robust_training.training.config import AdapterTrainingProtocol
from typo_robust_training.training.encoding import encode_training_pair
from typo_robust_training.training.evidence import LocalizationEvidence
from typo_robust_training.training.pairs import TrainingPair
from typo_robust_training.training.runner import TrainingMicroStepResult
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


class HuggingFaceAdapterTrainingRuntime:
    """Frozen clean teacher plus a typo student whose LoRA is the only update."""

    def __init__(
        self,
        *,
        protocol: AdapterTrainingProtocol,
        seed: int,
        gpu_id: str,
        evidence: LocalizationEvidence | None,
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
        from transformers import get_cosine_schedule_with_warmup
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
            if evidence is None:
                raise ValueError("targeted adapter training requires localization evidence")
            adapter_layers = evidence.adapter_layers
        self.adapter_layers = adapter_layers
        self.component_weights = evidence.component_weights if evidence is not None else None
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
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=protocol.max_optimizer_steps,
        )
        self._optimizer_steps = 0
        torch.cuda.reset_peak_memory_stats()

    def train_micro_step(self, pair: TrainingPair, *, loss_scale: float) -> TrainingMicroStepResult:
        if not math.isfinite(float(loss_scale)) or float(loss_scale) <= 0.0:
            raise ValueError("training micro-step loss scale must be positive")
        encoding = encode_training_pair(
            pair,
            tokenizer=self.tokenizer,
            max_length=self.protocol.max_sequence_length,
        )
        output = compute_training_step(
            teacher=self.teacher,
            student=self.student,
            encoding=encoding,
            protocol=self.protocol,
            component_weights=self.component_weights,
            attention_head_dim=self.attention_head_dim,
        )
        (output.loss * float(loss_scale)).backward()
        losses = {
            name: float(value.detach().float().cpu()) for name, value in output.losses.items()
        }
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

    def telemetry(self) -> dict[str, int]:
        torch = self._torch
        return {
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_peak_memory_allocated_bytes_since_start": int(torch.cuda.max_memory_allocated()),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "gpu_peak_memory_reserved_bytes_since_start": int(torch.cuda.max_memory_reserved()),
        }

    def save_state(self, path: Path) -> None:
        from peft import get_peft_model_state_dict

        state_path = Path(path).resolve()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "robustness-adapter-runtime-state/v1",
            "condition": self.protocol.condition,
            "config_sha256": self.protocol.config_sha256,
            "seed": self.seed,
            "optimizer_steps": self._optimizer_steps,
            "adapter": get_peft_model_state_dict(self.student),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
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
        expected = {
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
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("adapter runtime checkpoint fields differ")
        if (
            payload["schema_version"] != "robustness-adapter-runtime-state/v1"
            or payload["condition"] != self.protocol.condition
            or payload["config_sha256"] != self.protocol.config_sha256
            or payload["seed"] != self.seed
        ):
            raise ValueError("adapter runtime checkpoint identity differs")
        set_peft_model_state_dict(self.student, payload["adapter"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self._optimizer_steps = int(payload["optimizer_steps"])
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
            "runtime": "HuggingFaceAdapterTrainingRuntime/v1",
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
            "adapter_modules": list(self.parameter_report.modules),
            "trainable_parameters": self.parameter_report.trainable_parameters,
            "total_parameters": self.parameter_report.total_parameters,
            "attention_head_dim": self.attention_head_dim,
            "teacher_frozen": True,
            "student_base_frozen": True,
        }


__all__ = ["HuggingFaceAdapterTrainingRuntime"]
