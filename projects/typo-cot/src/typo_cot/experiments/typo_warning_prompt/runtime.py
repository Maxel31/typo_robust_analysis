"""Hugging Face GPU runtime for submitted-compatible typo-warning generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import typo_cot.experiments.typo_warning_prompt.protocol as warning_protocol
from typo_cot.experiments.typo_warning_prompt.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.typo_warning_prompt.planning import WarningPromptPlan
from typo_cot.experiments.typo_warning_prompt.scoring import extract_submitted_answer

if TYPE_CHECKING:
    from typo_cot.experiments.typo_warning_prompt.runner import (
        WarningPromptConfig,
        WarningPromptArmScan,
        WarningPromptGeneration,
    )

ARM_ORDER = warning_protocol.ARM_ORDER
_ANSWER_EXTRACTION = warning_protocol.ANSWER_EXTRACTION
_ANSWER_SPAN_DECODING = warning_protocol.ANSWER_SPAN_DECODING
_BATCHING = warning_protocol.BATCHING
_GENERATION = warning_protocol.GENERATION
_IMPLEMENTATION = warning_protocol.IMPLEMENTATION
_PROMPT_INTERVENTION = warning_protocol.PROMPT_INTERVENTION


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"tokenizer returned no token IDs for {field}")
    ids = tuple(value)
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in ids):
        raise ValueError(f"tokenizer returned invalid token IDs for {field}")
    return ids


def _ids_sha256(token_ids: Sequence[int]) -> str:
    encoded = json.dumps(
        list(token_ids),
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HuggingFaceWarningPromptRuntime:
    """Load one pinned causal LM and generate sorted same-arm batches of up to eight."""

    def __init__(self, config: WarningPromptConfig, *, revision: str) -> None:
        if not revision:
            raise ValueError("typo-warning generation requires a pinned source revision")
        self.revision = revision
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != config.gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={config.gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("typo-warning generation requires the requested CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "typo-warning generation requires exactly one visible GPU; set "
                "CUDA_VISIBLE_DEVICES to one physical device"
            )

        from typo_cot.models.wrapper import create_model_wrapper

        self.config = config
        self._torch = torch
        self.wrapper = create_model_wrapper(
            model_name=config.model,
            gpu_id=config.gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=revision,
        )
        self.model = self.wrapper.model
        self.model.eval()
        self.tokenizer = self.wrapper.tokenizer
        self.tokenizer.padding_side = str(_GENERATION["padding_side"])
        self.device = next(self.model.parameters()).device
        (
            self.effective_eos_token_ids,
            self.effective_eos_token_ids_source,
        ) = self._resolve_eos_token_ids()

    @staticmethod
    def _normalized_token_ids(value: object) -> tuple[int, ...]:
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        return tuple(
            sorted(
                {
                    token
                    for token in values
                    if isinstance(token, int) and not isinstance(token, bool) and token >= 0
                }
            )
        )

    def _resolve_eos_token_ids(self) -> tuple[tuple[int, ...], str]:
        generation_config = self.model.generation_config
        if getattr(generation_config, "stop_strings", None):
            raise ValueError("typo-warning generation does not support stop_strings")
        if getattr(generation_config, "forced_eos_token_id", None) is not None:
            raise ValueError("typo-warning generation does not support forced_eos_token_id")
        resolved = self._normalized_token_ids(getattr(generation_config, "eos_token_id", None))
        if resolved:
            return resolved, "model-generation-config"
        resolved = self._normalized_token_ids(getattr(self.tokenizer, "eos_token_id", None))
        if resolved:
            return resolved, "tokenizer-fallback"
        raise ValueError("model generation config and tokenizer expose no EOS token ID")

    def _encode_unpadded(self, text: str, *, field: str) -> tuple[int, ...]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError(f"tokenizer returned no input_ids for {field}")
        return _token_ids(encoded["input_ids"], field=field)

    def _generation_result(
        self,
        token_ids: tuple[int, ...],
        text: str,
        gold_answer: str,
        *,
        stop_reason: str,
        stop_token_id: int | None,
    ) -> WarningPromptGeneration:
        from typo_cot.experiments.typo_warning_prompt.runner import WarningPromptGeneration

        extraction = extract_submitted_answer(
            text,
            benchmark=self.config.benchmark,
            correct_answer=gold_answer,
        )
        return WarningPromptGeneration(
            token_ids=token_ids,
            text=text,
            value=extraction.value,
            is_extracted=extraction.is_extracted,
            is_correct=extraction.is_correct,
            method=extraction.method,
            primary_method=extraction.primary_method,
            stop_reason=stop_reason,  # type: ignore[arg-type]
            stop_token_id=stop_token_id,
        )

    def scan_arm_batch(
        self,
        pairs: Sequence[dict[str, object]],
        plans: Sequence[WarningPromptPlan],
        *,
        arm: str,
    ) -> Sequence[WarningPromptArmScan]:
        """Generate one warning arm for up to eight sorted submitted samples."""

        from typo_cot.experiments.typo_warning_prompt.runner import (
            WarningPromptArmScan,
            WarningPromptInputUse,
        )

        if arm not in ARM_ORDER:
            raise ValueError(f"runtime received an unsupported warning arm: {arm!r}")
        if len(pairs) != len(plans) or not pairs:
            raise ValueError("runtime pairs and plans must be equally sized and non-empty")
        if len(pairs) > int(_BATCHING["batch_size"]):
            raise ValueError("runtime batch exceeds the submitted batch size")
        arm_plans = []
        sample_ids: list[str] = []
        for pair, plan in zip(pairs, plans, strict=True):
            sample_id = pair.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("submitted pair sample_id must be non-empty")
            if plan.sample_id != sample_id:
                raise ValueError("warning plan sample_id does not match the submitted input")
            if tuple(item.arm for item in plan.arms) != ARM_ORDER:
                raise ValueError("runtime requires warning arms in protocol order")
            arm_plans.append(next(item for item in plan.arms if item.arm == arm))
            sample_ids.append(sample_id)
        if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("runtime same-arm batch sample IDs must be sorted and unique")
        prompt_ids = [
            self._encode_unpadded(item.prompt, field=f"{sample_id} {arm} prompt")
            for sample_id, item in zip(sample_ids, arm_plans, strict=True)
        ]
        encoded = self.tokenizer(
            [item.prompt for item in arm_plans],
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer returned no batched input_ids")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != len(arm_plans):
            raise ValueError("tokenizer must return one sequence for each batch item")
        if attention_mask is None or attention_mask.shape != input_ids.shape:
            raise ValueError("tokenizer returned an invalid batched attention mask")
        for index, (sample_id, arm_plan) in enumerate(zip(sample_ids, arm_plans, strict=True)):
            batch_ids = tuple(
                int(token)
                for token, attended in zip(
                    input_ids[index].detach().cpu().tolist(),
                    attention_mask[index].detach().cpu().tolist(),
                    strict=True,
                )
                if int(attended) == 1
            )
            if batch_ids != prompt_ids[index]:
                raise ValueError(
                    f"batched tokenization differs from exact prompt for {sample_id} {arm}"
                )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        generation_arguments = {
            key: value for key, value in _GENERATION.items() if key != "padding_side"
        }
        max_new_tokens = int(_GENERATION["max_new_tokens"])
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_arguments,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.effective_eos_token_ids),
            )
        if (
            not hasattr(output_ids, "ndim")
            or output_ids.ndim != 2
            or int(output_ids.shape[0]) != len(arm_plans)
            or int(output_ids.shape[1]) <= int(input_ids.shape[1])
            or int(output_ids.shape[1]) > int(input_ids.shape[1]) + max_new_tokens
        ):
            raise ValueError("generation must return one capped answer span per batch item")

        scans: list[WarningPromptArmScan] = []
        prompt_width = int(input_ids.shape[1])
        for index, (_pair, plan, arm_plan) in enumerate(zip(pairs, plans, arm_plans, strict=True)):
            raw_continuation = tuple(
                int(token) for token in output_ids[index, prompt_width:].detach().cpu().tolist()
            )
            if not raw_continuation:
                raise ValueError(f"{plan.sample_id} {arm} generated no continuation token")
            eos_index = next(
                (
                    position
                    for position, token in enumerate(raw_continuation)
                    if token in self.effective_eos_token_ids
                ),
                None,
            )
            if eos_index is None:
                if len(raw_continuation) != max_new_tokens:
                    raise ValueError(
                        f"{plan.sample_id} {arm} stopped without EOS before the "
                        f"{max_new_tokens}-token cap"
                    )
                continuation = raw_continuation
                stop_reason = "max_new_tokens"
                stop_token_id = None
            else:
                continuation = raw_continuation[: eos_index + 1]
                stop_reason = "eos_token"
                stop_token_id = continuation[-1]
            text = self.tokenizer.decode(
                list(continuation),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            use = WarningPromptInputUse(
                arm=arm,
                prompt_text_sha256=arm_plan.prompt_sha256,
                prompt_char_count=len(arm_plan.prompt),
                prompt_token_count=len(prompt_ids[index]),
                prompt_ids_sha256=_ids_sha256(prompt_ids[index]),
            )
            scans.append(
                WarningPromptArmScan(
                    sample_id=plan.sample_id,
                    arm=arm,
                    input_use=use,
                    generation=self._generation_result(
                        continuation,
                        text,
                        plan.gold_answer,
                        stop_reason=stop_reason,
                        stop_token_id=stop_token_id,
                    ),
                )
            )
        return tuple(scans)

    def provenance(self) -> dict[str, object]:
        torch = self._torch
        model_metadata_revision = getattr(self.model.config, "_commit_hash", None)
        model_revision = model_metadata_revision or self.revision
        tokenizer_metadata_revision = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        tokenizer_revision = tokenizer_metadata_revision or self.revision
        return {
            "operation": "typo-warning-prompt",
            "runtime": "HuggingFaceWarningPromptRuntime",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "model": self.config.model,
            "requested_revision": self.revision,
            "model_revision": model_revision,
            "model_revision_source": (
                "model-config-metadata" if model_metadata_revision else "explicit-load-revision"
            ),
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_revision_source": (
                "tokenizer-init-metadata"
                if tokenizer_metadata_revision
                else "explicit-load-revision"
            ),
            "dtype": "bfloat16",
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "generation": dict(_GENERATION),
            "implementation": _IMPLEMENTATION,
            "batching": dict(_BATCHING),
            "answer_span_decoding": dict(_ANSWER_SPAN_DECODING),
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.effective_eos_token_ids_source,
            "answer_extraction": _ANSWER_EXTRACTION,
            "prompt_intervention": dict(_PROMPT_INTERVENTION),
            "implementation_code": implementation_code_identity(),
        }


__all__ = ["HuggingFaceWarningPromptRuntime"]
