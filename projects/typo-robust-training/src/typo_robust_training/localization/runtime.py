"""Hugging Face runtime for diagnostic single-layer causal scans."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

from typo_robust_training.localization.config import LayerSelectionProtocol
from typo_robust_training.localization.prompting import (
    PromptSide,
    build_diagnostic_prompts,
    word_final_token_positions,
)
from typo_robust_training.localization.records import LayerScan


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def clean_teacher_targets(
    generated_token_ids: Sequence[int],
    *,
    termination: str,
    effective_eos_token_ids: Sequence[int],
    count: int,
) -> tuple[tuple[int, ...], str | None]:
    """Strip terminal EOS and freeze at most ``count`` clean target tokens."""

    if termination not in {"eos", "length-cap"}:
        raise ValueError("clean generation termination must be eos or length-cap")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("clean target count must be a positive integer")
    raw = tuple(generated_token_ids)
    if not raw or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in raw
    ):
        raise ValueError("clean generated token IDs must be non-empty non-negative integers")
    eos = frozenset(
        token
        for token in effective_eos_token_ids
        if isinstance(token, int) and not isinstance(token, bool) and token >= 0
    )
    if not eos:
        raise ValueError("effective EOS token IDs must not be empty")
    eos_indices = [index for index, token in enumerate(raw) if token in eos]
    if termination == "eos":
        if not eos_indices or eos_indices[0] != len(raw) - 1:
            raise ValueError("clean continuation must end at its first EOS token")
        content = raw[:-1]
    else:
        if eos_indices:
            raise ValueError("length-cap continuation unexpectedly contains EOS")
        content = raw
    targets = content[:count]
    reason = None if len(targets) == count else f"clean-continuation-lt-{count}-before-eos"
    return targets, reason


def readout_kl_2_16(
    trajectory: Sequence[float], *, teacher_forced_tokens: int
) -> tuple[float, ...]:
    """Return the prespecified token 2--16 trajectory, excluding token one."""

    values = tuple(float(value) for value in trajectory)
    if teacher_forced_tokens != 16 or len(values) != 16:
        raise ValueError("multi-token layer selection requires exactly sixteen KL values")
    return values[1:16]


def should_generate_patched_answers(*, clean_correct: bool) -> bool:
    """Skip outcome-irrelevant layer generations for clean-wrong records."""

    if type(clean_correct) is not bool:
        raise TypeError("clean correctness must be boolean")
    return clean_correct


class HuggingFaceLayerSelectionRuntime:
    """Run invariant baselines and every clean-to-typo single-layer patch."""

    def __init__(self, *, protocol: LayerSelectionProtocol, gpu_id: str) -> None:
        if not isinstance(protocol, LayerSelectionProtocol):
            raise TypeError("protocol must be LayerSelectionProtocol")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("select-distillation-layers requires the requested CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "select-distillation-layers requires exactly one visible GPU; "
                "set CUDA_VISIBLE_DEVICES to one physical device"
            )
        from typo_cot.evaluation.generation import resolve_effective_eos_token_ids
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.wrapper import create_model_wrapper

        self.protocol = protocol
        self.gpu_id = gpu_id
        self._torch = torch
        self.wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self.model = self.wrapper.model
        self.model.eval()
        self.tokenizer = self.wrapper.tokenizer
        self.tokenizer.padding_side = "left"
        self.layers = find_decoder_layers(self.model)
        self.num_layers = len(self.layers)
        if self.num_layers < protocol.window_width:
            raise ValueError("model has fewer layers than the frozen selection window")
        self.device = next(self.model.parameters()).device
        self.effective_eos_token_ids, self.eos_source = resolve_effective_eos_token_ids(
            generation_config=self.model.generation_config,
            tokenizer=self.tokenizer,
            operation="select-distillation-layers",
        )
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision is not None and actual_revision != protocol.model_revision:
            raise ValueError("loaded model revision differs from the pinned selection revision")

    def _encode(self, side: PromptSide) -> tuple[Any, Any, tuple[int, ...]]:
        positions = word_final_token_positions(
            self.tokenizer,
            text=side.text,
            spans=side.spans,
        )
        encoded = self.tokenizer(
            side.text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=False,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("tokenizer must return mapping-like tensors")
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("tokenizer must return one unpadded input sequence")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
            raise ValueError("diagnostic prompts must be unpadded and fully attended")
        if max(positions) >= int(input_ids.shape[1]):
            raise ValueError("edited-word final token lies outside the prompt")
        return input_ids.to(self.device), attention_mask.to(self.device), positions

    def _generate(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        answer: str,
        task: str,
        field: str,
        patch: Any | None = None,
    ) -> dict[str, object]:
        from typo_cot.evaluation.fallback import extract_with_fallback
        from typo_cot.evaluation.generation import classify_generated_token_ids

        context = patch if patch is not None else nullcontext()
        with context:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(self.protocol.generation["max_new_tokens"]),
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                return_dict_in_generate=False,
                output_scores=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.effective_eos_token_ids),
            )
        prompt_length = int(input_ids.shape[1])
        raw_ids = output_ids[0, prompt_length:].detach().cpu().tolist()
        token_ids, termination = classify_generated_token_ids(
            raw_ids,
            effective_eos_token_ids=self.effective_eos_token_ids,
            max_new_tokens=int(self.protocol.generation["max_new_tokens"]),
            field=field,
        )
        text = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        extraction = extract_with_fallback(
            text,
            benchmark=task,
            correct_answer=answer,
            allow_positional=termination == "eos",
        )
        return {
            "token_ids": list(token_ids),
            "text": text,
            "termination": termination,
            "value": extraction.value,
            "is_extracted": extraction.is_extracted,
            "is_correct": extraction.is_correct,
            "method": extraction.method,
            "primary_method": extraction.primary_method,
        }

    def _capture_donors(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        positions: tuple[int, ...],
    ) -> list[Any]:
        from typo_cot.experiments.layerwise_kl_patching.patching import capture_block_outputs

        return capture_block_outputs(
            self.layers,
            positions=positions,
            forward=lambda: self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ),
        )

    def _append_targets(
        self, input_ids: Any, attention_mask: Any, targets: tuple[int, ...]
    ) -> tuple[Any, Any]:
        if len(targets) != self.protocol.teacher_forced_tokens:
            raise ValueError("teacher-forced targets differ from the frozen token count")
        prefix = self._torch.tensor(
            [targets[:-1]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return (
            self._torch.cat((input_ids, prefix), dim=1),
            self._torch.cat(
                (attention_mask, self._torch.ones_like(prefix, dtype=attention_mask.dtype)),
                dim=1,
            ),
        )

    def _target_logits(self, output: Any, *, prompt_tokens: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("model output must contain [1, sequence, vocabulary] logits")
        start = prompt_tokens - 1
        stop = start + self.protocol.teacher_forced_tokens
        selected = logits[0, start:stop, :]
        if int(selected.shape[0]) != self.protocol.teacher_forced_tokens:
            raise ValueError("model output does not cover every teacher-forced target")
        return selected.detach().float().cpu()

    @staticmethod
    def _prompt_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def scan_pair(self, record: dict[str, object]) -> LayerScan:
        """Scan all layers using one invariant clean continuation and donor cache."""

        from typo_cot.experiments.layerwise_answer_patching.patching import (
            PrefillBlockOutputPatch,
        )
        from typo_cot.experiments.layerwise_kl_patching.patching import BlockOutputPatch
        from typo_cot.experiments.multitoken_kl_readout.metrics import (
            kl_trajectory_from_logits,
        )

        prompts = build_diagnostic_prompts(record)
        clean_ids, clean_mask, clean_positions = self._encode(prompts.clean)
        typo_ids, typo_mask, typo_positions = self._encode(prompts.typo)
        if len(clean_positions) != len(typo_positions):
            raise ValueError("clean and typo edited-word coordinate cardinalities differ")

        with self._torch.inference_mode():
            clean_answer = self._generate(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                answer=prompts.answer,
                task=prompts.task,
                field=f"{prompts.record_id}:clean",
            )
            typo_answer = self._generate(
                input_ids=typo_ids,
                attention_mask=typo_mask,
                answer=prompts.answer,
                task=prompts.task,
                field=f"{prompts.record_id}:typo",
            )
            target_ids, invalid_reason = clean_teacher_targets(
                clean_answer["token_ids"],  # type: ignore[arg-type]
                termination=str(clean_answer["termination"]),
                effective_eos_token_ids=self.effective_eos_token_ids,
                count=self.protocol.teacher_forced_tokens,
            )
            target_text = self.tokenizer.convert_ids_to_tokens(list(target_ids))
            if not isinstance(target_text, list) or any(
                not isinstance(piece, str) for piece in target_text
            ):
                raise ValueError("tokenizer must expose every target vocabulary piece")

            untreated_readout: tuple[float, ...] = ()
            patched_readouts: list[tuple[float, ...]] = [()] * self.num_layers
            untreated_audit: tuple[float, ...] = ()
            clean_full: tuple[Any, Any] | None = None
            typo_full: tuple[Any, Any] | None = None
            clean_logits: Any | None = None
            if invalid_reason is None:
                clean_full = self._append_targets(clean_ids, clean_mask, target_ids)
                typo_full = self._append_targets(typo_ids, typo_mask, target_ids)
                clean_output = self.model(
                    input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False
                )
                typo_output = self.model(
                    input_ids=typo_full[0], attention_mask=typo_full[1], use_cache=False
                )
                clean_logits = self._target_logits(
                    clean_output,
                    prompt_tokens=int(clean_ids.shape[1]),
                )
                typo_logits = self._target_logits(
                    typo_output,
                    prompt_tokens=int(typo_ids.shape[1]),
                )
                del clean_output, typo_output
                untreated_audit = readout_kl_2_16(
                    kl_trajectory_from_logits(clean_logits, typo_logits),
                    teacher_forced_tokens=self.protocol.teacher_forced_tokens,
                )
                untreated_mean = sum(untreated_audit) / len(untreated_audit)
                if untreated_mean > self.protocol.untreated_mean_kl_min_exclusive:
                    untreated_readout = untreated_audit
                else:
                    invalid_reason = "untreated-kl-at-or-below-1e-6"

            clean_correct = bool(clean_answer["is_correct"])
            generate_answers = should_generate_patched_answers(clean_correct=clean_correct)
            require_donors = bool(untreated_readout) or generate_answers
            donors = (
                self._capture_donors(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    positions=clean_positions,
                )
                if require_donors
                else None
            )
            patched_correct: list[bool | None] = []
            patched_answers: list[dict[str, object] | None] = []
            for layer in range(self.num_layers):
                if untreated_readout:
                    if donors is None or typo_full is None or clean_logits is None:
                        raise RuntimeError("eligible KL scan is missing its frozen donor or target")
                    with BlockOutputPatch(
                        self.layers,
                        layer_index=layer,
                        positions=typo_positions,
                        donor_values=donors[layer],
                    ):
                        patched_output = self.model(
                            input_ids=typo_full[0],
                            attention_mask=typo_full[1],
                            use_cache=False,
                        )
                    patched_logits = self._target_logits(
                        patched_output,
                        prompt_tokens=int(typo_ids.shape[1]),
                    )
                    del patched_output
                    patched_readouts[layer] = readout_kl_2_16(
                        kl_trajectory_from_logits(clean_logits, patched_logits),
                        teacher_forced_tokens=self.protocol.teacher_forced_tokens,
                    )
                if generate_answers:
                    if donors is None:
                        raise RuntimeError("clean-correct scan is missing its frozen donor")
                    patched_answer = self._generate(
                        input_ids=typo_ids,
                        attention_mask=typo_mask,
                        answer=prompts.answer,
                        task=prompts.task,
                        field=f"{prompts.record_id}:layer-{layer}",
                        patch=PrefillBlockOutputPatch(
                            self.layers,
                            layer_index=layer,
                            positions=typo_positions,
                            donor_values=donors[layer],
                        ),
                    )
                    patched_correct.append(bool(patched_answer["is_correct"]))
                    patched_answers.append(patched_answer)
                else:
                    patched_correct.append(None)
                    patched_answers.append(None)

        audit = {
            "clean_prompt_sha256": self._prompt_sha256(prompts.clean.text),
            "typo_prompt_sha256": self._prompt_sha256(prompts.typo.text),
            "clean_prompt_token_count": int(clean_ids.shape[1]),
            "typo_prompt_token_count": int(typo_ids.shape[1]),
            "clean_word_final_tokens": list(clean_positions),
            "typo_word_final_tokens": list(typo_positions),
            "target_token_text": target_text,
            "untreated_kl_2_16_before_eligibility": list(untreated_audit),
            "clean_answer": clean_answer,
            "typo_answer": typo_answer,
            "patched_answers_by_layer": patched_answers,
        }
        return LayerScan(
            record_id=prompts.record_id,
            task=prompts.task,
            target_token_ids=target_ids,
            untreated_kl_2_16=untreated_readout,
            patched_kl_2_16_by_layer=tuple(patched_readouts),
            clean_correct=clean_correct,
            typo_correct=bool(typo_answer["is_correct"]),
            patched_correct_by_layer=tuple(patched_correct),
            kl_invalid_reason=invalid_reason,
            audit=audit,
        )

    def provenance(self) -> dict[str, object]:
        torch = self._torch
        tokenizer_revision = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        return {
            "runtime": "HuggingFaceLayerSelectionRuntime/v1",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "model": self.protocol.model,
            "requested_revision": self.protocol.model_revision,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_revision": tokenizer_revision or self.protocol.model_revision,
            "num_decoder_layers": self.num_layers,
            "decoder_adapter": (
                f"{type(self.model.get_decoder()).__module__}."
                f"{type(self.model.get_decoder()).__name__}.layers"
            ),
            "dtype": self.protocol.dtype,
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.eos_source,
            "prompt_template": self.protocol.prompt_template,
            "patch_direction": self.protocol.patch_direction,
            "patch_site": self.protocol.patch_site,
            "coordinate": self.protocol.coordinate,
            "teacher_forced_targets": "single-clean-generation-reused-for-all-layers/v1",
            "readout": "tokens-2-through-16-inclusive/v1",
            "answer_patch_policy": "clean-correct-records-only/v1",
        }


__all__ = [
    "HuggingFaceLayerSelectionRuntime",
    "clean_teacher_targets",
    "readout_kl_2_16",
    "should_generate_patched_answers",
]
