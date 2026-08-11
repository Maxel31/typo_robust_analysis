"""Single-GPU base/PEFT runtime for paired held-out robustness scans."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any

from typo_robust_training.evaluation.checkpoints import AdapterDescriptor, PatchWindow
from typo_robust_training.evaluation.config import RobustnessEvaluationProtocol
from typo_robust_training.evaluation.data import EvaluationPair
from typo_robust_training.evaluation.prompting import (
    EvaluationPrompts,
    build_evaluation_prompts,
    classify_tokenization_counts,
)
from typo_robust_training.evaluation.records import EvaluationObservation


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _flat(value: object, *, field: str) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError(f"evaluation tokenizer {field} must be one sequence")
    return value


def _token_overlap_profile(
    *,
    text: str,
    spans: Sequence[tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    positions: list[int] = []
    counts: list[int] = []
    for index, (start, stop) in enumerate(spans):
        overlapping = [
            token_index
            for token_index, (token_start, token_stop) in enumerate(offsets)
            if token_stop > token_start and token_start < stop and start < token_stop
        ]
        if not overlapping:
            raise ValueError(f"evaluation edit span {index} has no tokenizer overlap")
        if offsets[overlapping[0]][0] != start or offsets[overlapping[-1]][1] != stop:
            raise ValueError(f"evaluation edit span {index} differs from tokenizer boundaries")
        if text[start:stop].isspace():
            raise ValueError(f"evaluation edit span {index} contains only whitespace")
        positions.append(overlapping[-1])
        counts.append(len(overlapping))
    if len(set(positions)) != len(positions):
        raise ValueError("evaluation edited words resolve to duplicate token positions")
    return tuple(positions), tuple(counts)


@dataclass(frozen=True, slots=True)
class PromptTokenizationProfile:
    clean_input_ids: tuple[int, ...]
    typo_input_ids: tuple[int, ...]
    clean_attention_mask: tuple[int, ...]
    typo_attention_mask: tuple[int, ...]
    clean_positions: tuple[int, ...]
    typo_positions: tuple[int, ...]
    clean_subtoken_counts: tuple[int, ...]
    typo_subtoken_counts: tuple[int, ...]
    tokenization_stratum: str


def _tokenize_side(tokenizer: Any, *, text: str, spans: Sequence[tuple[int, int]], max_tokens: int):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping):
        raise ValueError("evaluation tokenizer must return a mapping")
    ids = _flat(encoded.get("input_ids"), field="input_ids")
    mask = _flat(encoded.get("attention_mask"), field="attention_mask")
    raw_offsets = _flat(encoded.get("offset_mapping"), field="offset_mapping")
    if not ids or len(ids) != len(mask) or len(ids) != len(raw_offsets):
        raise ValueError("evaluation tokenizer returned inconsistent prompt fields")
    if len(ids) > max_tokens:
        raise ValueError("evaluation prompt exceeds max_input_tokens")
    normalized_ids: list[int] = []
    normalized_mask: list[int] = []
    offsets: list[tuple[int, int]] = []
    for index, (token, attended, raw_offset) in enumerate(zip(ids, mask, raw_offsets, strict=True)):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"evaluation tokenizer input_ids[{index}] is invalid")
        if attended != 1:
            raise ValueError("evaluation prompts must be unpadded and fully attended")
        if (
            not isinstance(raw_offset, (tuple, list))
            or len(raw_offset) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_offset)
        ):
            raise ValueError(f"evaluation tokenizer offset_mapping[{index}] is invalid")
        start, stop = raw_offset
        while start < stop and text[start].isspace():
            start += 1
        while start < stop and text[stop - 1].isspace():
            stop -= 1
        normalized_ids.append(token)
        normalized_mask.append(1)
        offsets.append((start, stop))
    positions, counts = _token_overlap_profile(text=text, spans=spans, offsets=offsets)
    return (
        tuple(normalized_ids),
        tuple(normalized_mask),
        positions,
        counts,
    )


def prompt_tokenization_profile(
    tokenizer: Any,
    *,
    prompts: EvaluationPrompts,
    max_tokens: int,
) -> PromptTokenizationProfile:
    """Resolve prompt token IDs plus aligned edited-word tokenization strata."""

    if not isinstance(prompts, EvaluationPrompts):
        raise TypeError("evaluation tokenization requires EvaluationPrompts")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 2:
        raise ValueError("evaluation max_tokens must be at least two")
    clean = _tokenize_side(
        tokenizer,
        text=prompts.clean.text,
        spans=prompts.clean.spans,
        max_tokens=max_tokens,
    )
    typo = _tokenize_side(
        tokenizer,
        text=prompts.typo.text,
        spans=prompts.typo.spans,
        max_tokens=max_tokens,
    )
    if len(clean[2]) != len(typo[2]):
        raise ValueError("evaluation clean and typo edited-word token coordinates differ")
    return PromptTokenizationProfile(
        clean_input_ids=clean[0],
        clean_attention_mask=clean[1],
        clean_positions=clean[2],
        clean_subtoken_counts=clean[3],
        typo_input_ids=typo[0],
        typo_attention_mask=typo[1],
        typo_positions=typo[2],
        typo_subtoken_counts=typo[3],
        tokenization_stratum=classify_tokenization_counts(clean[3], typo[3]),
    )


def teacher_forced_kl_readout(
    clean_logits: Any,
    candidate_logits: Any,
    *,
    teacher_forced_tokens: int,
) -> tuple[float, ...]:
    """Compute KL(clean || candidate) and return prespecified tokens 2--16."""

    if teacher_forced_tokens != 16:
        raise ValueError("robustness evaluation requires exactly sixteen teacher-forced tokens")
    if (
        clean_logits.ndim != 2
        or candidate_logits.ndim != 2
        or int(clean_logits.shape[0]) != teacher_forced_tokens
        or clean_logits.shape != candidate_logits.shape
    ):
        raise ValueError("evaluation KL logits must share shape [16, vocabulary]")
    from typo_cot.experiments.multitoken_kl_readout.metrics import (
        kl_trajectory_from_logits,
    )

    trajectory = kl_trajectory_from_logits(
        clean_logits.detach().float().cpu(),
        candidate_logits.detach().float().cpu(),
    )
    return tuple(trajectory[1:16])


def window_patched_forward(
    layers: Sequence[Any],
    *,
    layer_indices: Sequence[int],
    positions: Sequence[int],
    donor_values: Sequence[Any],
    forward: Callable[[], Any],
) -> Any:
    """Patch every selected block output during one ordinary full forward."""

    from typo_cot.experiments.layerwise_kl_patching.patching import BlockOutputPatch

    indices = tuple(layer_indices)
    donors = tuple(donor_values)
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("evaluation patch layers must be unique and strictly increasing")
    if len(indices) != len(donors):
        raise ValueError("evaluation patch donor count differs from selected layers")
    with ExitStack() as stack:
        for layer_index, donor in zip(indices, donors, strict=True):
            stack.enter_context(
                BlockOutputPatch(
                    layers,
                    layer_index=layer_index,
                    positions=positions,
                    donor_values=donor,
                )
            )
        return forward()


class HuggingFaceRobustnessEvaluationRuntime:
    """Evaluate one base or adapter condition on one requested physical GPU."""

    def __init__(
        self,
        *,
        protocol: RobustnessEvaluationProtocol,
        gpu_id: str,
        descriptor: AdapterDescriptor | None,
        patch_window: PatchWindow,
    ) -> None:
        if not isinstance(protocol, RobustnessEvaluationProtocol):
            raise TypeError("evaluation runtime protocol must be validated")
        if descriptor is not None and not isinstance(descriptor, AdapterDescriptor):
            raise TypeError("evaluation runtime adapter descriptor is invalid")
        if not isinstance(patch_window, PatchWindow):
            raise TypeError("evaluation runtime patch window is invalid")
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
            raise RuntimeError("robustness evaluation requires exactly one requested CUDA GPU")
        from typo_cot.evaluation.generation import resolve_effective_eos_token_ids
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.wrapper import create_model_wrapper

        self.protocol = protocol
        self.gpu_id = gpu_id
        self.descriptor = descriptor
        self.patch_window = patch_window
        self.condition = "base" if descriptor is None else descriptor.condition
        self.seed = None if descriptor is None else descriptor.seed
        self._torch = torch
        self.wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        base_model = self.wrapper.model
        revision = getattr(base_model.config, "_commit_hash", None)
        if revision is not None and revision != protocol.model_revision:
            raise ValueError("evaluation loaded model revision differs")
        self.layers = find_decoder_layers(base_model)
        if not patch_window.layers or patch_window.stop > len(self.layers):
            raise ValueError("evaluation patch window is outside the model")
        if descriptor is None:
            self.model = base_model
        else:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                base_model,
                descriptor.path,
                is_trainable=False,
            )
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = self.wrapper.tokenizer
        if getattr(self.tokenizer, "is_fast", False) is not True:
            raise ValueError("robustness evaluation requires a fast tokenizer with offsets")
        self.tokenizer.padding_side = "left"
        self.device = next(self.model.parameters()).device
        self.effective_eos_token_ids, self.eos_source = resolve_effective_eos_token_ids(
            generation_config=self.model.generation_config,
            tokenizer=self.tokenizer,
            operation="evaluate-typo-robustness",
        )
        self._closed = False

    def _tensor(self, values: tuple[int, ...]) -> Any:
        return self._torch.tensor([values], dtype=self._torch.long, device=self.device)

    def _generate(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        field: str,
        patch: Any | None = None,
    ) -> dict[str, object]:
        from typo_cot.evaluation.generation import classify_generated_token_ids

        context = patch if patch is not None else nullcontext()
        with context:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.protocol.max_new_tokens,
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
        raw_ids = output[0, int(input_ids.shape[1]) :].detach().cpu().tolist()
        token_ids, termination = classify_generated_token_ids(
            raw_ids,
            effective_eos_token_ids=self.effective_eos_token_ids,
            max_new_tokens=self.protocol.max_new_tokens,
            field=field,
        )
        text = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {"token_ids": token_ids, "termination": termination, "text": text}

    @staticmethod
    def _answer(generation: Mapping[str, object], *, task: str, gold: str) -> dict[str, object]:
        termination = generation["termination"]
        text = generation["text"]
        if not isinstance(termination, str) or not isinstance(text, str):
            raise RuntimeError("validated generation changed type")
        if task in {"gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa"}:
            from typo_cot.evaluation.fallback import extract_with_fallback

            extraction = extract_with_fallback(
                text,
                benchmark=task,
                correct_answer=gold,
                allow_positional=termination == "eos",
            )
            return {
                "value": extraction.value,
                "correct": extraction.is_correct,
                "method": extraction.method,
                "primary_method": extraction.primary_method,
            }
        if task == "math":
            from typo_cot.evaluation.extractor import create_extractor

            extractor = create_extractor("math")
            extracted = extractor.extract(text)
            return {
                "value": extracted.extracted_answer,
                "correct": extractor.is_correct(extracted.extracted_answer, gold),
                "method": f"primary:{extracted.extraction_method}",
                "primary_method": extracted.extraction_method,
            }
        raise ValueError(f"evaluation answer task is unsupported: {task}")

    def _append_targets(self, ids: Any, mask: Any, targets: tuple[int, ...]):
        prefix = self._torch.tensor(
            [targets[:-1]],
            dtype=ids.dtype,
            device=ids.device,
        )
        return (
            self._torch.cat((ids, prefix), dim=1),
            self._torch.cat((mask, self._torch.ones_like(prefix, dtype=mask.dtype)), dim=1),
        )

    def _target_logits(self, output: Any, *, prompt_tokens: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("evaluation model output must contain batched logits")
        selected = logits[
            0,
            prompt_tokens - 1 : prompt_tokens - 1 + self.protocol.teacher_forced_tokens,
            :,
        ]
        if int(selected.shape[0]) != self.protocol.teacher_forced_tokens:
            raise ValueError("evaluation output omits teacher-forced target positions")
        return selected.detach().float().cpu()

    def _capture_donors(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        positions: tuple[int, ...],
    ) -> tuple[Any, ...]:
        from typo_cot.experiments.layerwise_kl_patching.patching import capture_block_outputs

        selected_layers = [self.layers[index] for index in self.patch_window.layers]
        return tuple(
            capture_block_outputs(
                selected_layers,
                positions=positions,
                forward=lambda: self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ),
            )
        )

    def scan_pair(self, pair: EvaluationPair) -> EvaluationObservation:
        """Generate paired answers and one clean-target KL/patch audit."""

        from typo_robust_training.localization.runtime import clean_teacher_targets

        prompts = build_evaluation_prompts(pair)
        profile = prompt_tokenization_profile(
            self.tokenizer,
            prompts=prompts,
            max_tokens=self.protocol.max_input_tokens,
        )
        clean_ids = self._tensor(profile.clean_input_ids)
        typo_ids = self._tensor(profile.typo_input_ids)
        clean_mask = self._tensor(profile.clean_attention_mask)
        typo_mask = self._tensor(profile.typo_attention_mask)
        with self._torch.inference_mode():
            clean_generation = self._generate(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                field=f"{pair.record_id}:{self.condition}:clean",
            )
            typo_generation = None
            clean_answer = typo_answer = None
            if prompts.task_for_extractor is not None:
                if prompts.answer is None:
                    raise RuntimeError("task evaluation prompt lost its gold answer")
                typo_generation = self._generate(
                    input_ids=typo_ids,
                    attention_mask=typo_mask,
                    field=f"{pair.record_id}:{self.condition}:typo",
                )
                clean_answer = self._answer(
                    clean_generation,
                    task=prompts.task_for_extractor,
                    gold=prompts.answer,
                )
                typo_answer = self._answer(
                    typo_generation,
                    task=prompts.task_for_extractor,
                    gold=prompts.answer,
                )
            targets, invalid_reason = clean_teacher_targets(
                clean_generation["token_ids"],  # type: ignore[arg-type]
                termination=str(clean_generation["termination"]),
                effective_eos_token_ids=self.effective_eos_token_ids,
                count=self.protocol.teacher_forced_tokens,
            )
            untreated: tuple[float, ...] = ()
            patched: tuple[float, ...] = ()
            patch_invalid_reason = invalid_reason
            patched_generation = None
            patched_answer = None
            if invalid_reason is None:
                clean_full = self._append_targets(clean_ids, clean_mask, targets)
                typo_full = self._append_targets(typo_ids, typo_mask, targets)
                clean_output = self.model(
                    input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False
                )
                typo_output = self.model(
                    input_ids=typo_full[0], attention_mask=typo_full[1], use_cache=False
                )
                clean_logits = self._target_logits(
                    clean_output,
                    prompt_tokens=len(profile.clean_input_ids),
                )
                typo_logits = self._target_logits(
                    typo_output,
                    prompt_tokens=len(profile.typo_input_ids),
                )
                untreated = teacher_forced_kl_readout(
                    clean_logits,
                    typo_logits,
                    teacher_forced_tokens=self.protocol.teacher_forced_tokens,
                )
                donors = self._capture_donors(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    positions=profile.clean_positions,
                )
                patched_output = window_patched_forward(
                    self.layers,
                    layer_indices=self.patch_window.layers,
                    positions=profile.typo_positions,
                    donor_values=donors,
                    forward=lambda: self.model(
                        input_ids=typo_full[0],
                        attention_mask=typo_full[1],
                        use_cache=False,
                    ),
                )
                patched_logits = self._target_logits(
                    patched_output,
                    prompt_tokens=len(profile.typo_input_ids),
                )
                patched = teacher_forced_kl_readout(
                    clean_logits,
                    patched_logits,
                    teacher_forced_tokens=self.protocol.teacher_forced_tokens,
                )
                patch_invalid_reason = None
                if prompts.task_for_extractor is not None:
                    from typo_cot.experiments.fixed_window_answer_patching.patching import (
                        PrefillBlockOutputWindowPatch,
                    )

                    patched_generation = self._generate(
                        input_ids=typo_ids,
                        attention_mask=typo_mask,
                        field=f"{pair.record_id}:{self.condition}:patched",
                        patch=PrefillBlockOutputWindowPatch(
                            self.layers,
                            layer_indices=self.patch_window.layers,
                            positions=profile.typo_positions,
                            donor_values=donors,
                        ),
                    )
                    if prompts.answer is None:
                        raise RuntimeError("task evaluation prompt lost its gold answer")
                    patched_answer = self._answer(
                        patched_generation,
                        task=prompts.task_for_extractor,
                        gold=prompts.answer,
                    )
                del clean_output, typo_output, patched_output

        task = pair.task
        return EvaluationObservation(
            record_id=pair.record_id,
            condition=self.condition,
            seed=self.seed,
            source=pair.source,
            task=task,
            operation=pair.operation,
            strata=pair.strata,
            clean_answer=None if clean_answer is None else str(clean_answer["value"]),
            typo_answer=None if typo_answer is None else str(typo_answer["value"]),
            patched_answer=(None if patched_answer is None else str(patched_answer["value"])),
            clean_correct=None if clean_answer is None else bool(clean_answer["correct"]),
            typo_correct=None if typo_answer is None else bool(typo_answer["correct"]),
            patched_correct=(None if patched_answer is None else bool(patched_answer["correct"])),
            target_token_ids=targets,
            untreated_kl_2_16=untreated,
            patched_kl_2_16=patched,
            kl_invalid_reason=invalid_reason,
            patch_invalid_reason=patch_invalid_reason,
            clean_subtoken_counts=profile.clean_subtoken_counts,
            typo_subtoken_counts=profile.typo_subtoken_counts,
            tokenization_stratum=profile.tokenization_stratum,
            audit={
                "clean_prompt_sha256": hashlib.sha256(prompts.clean.text.encode()).hexdigest(),
                "typo_prompt_sha256": hashlib.sha256(prompts.typo.text.encode()).hexdigest(),
                "clean_prompt_tokens": len(profile.clean_input_ids),
                "typo_prompt_tokens": len(profile.typo_input_ids),
                "clean_word_final_tokens": list(profile.clean_positions),
                "typo_word_final_tokens": list(profile.typo_positions),
                "patch_layers": list(self.patch_window.layers),
                "clean_generation": clean_generation,
                "typo_generation": typo_generation,
                "patched_generation": patched_generation,
                "clean_extraction": clean_answer,
                "typo_extraction": typo_answer,
                "patched_extraction": patched_answer,
            },
        )

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "HuggingFaceRobustnessEvaluationRuntime/v1",
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
            "model": self.protocol.model,
            "requested_revision": self.protocol.model_revision,
            "condition": self.condition,
            "seed": self.seed,
            "adapter_sha256": (None if self.descriptor is None else self.descriptor.adapter_sha256),
            "patch_layers": list(self.patch_window.layers),
            "device": str(self.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": self._torch.cuda.get_device_name(0),
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.eos_source,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        model = self.model
        wrapper = self.wrapper
        del self.model, self.layers, self.wrapper
        del model, wrapper
        gc.collect()
        self._torch.cuda.empty_cache()


__all__ = [
    "HuggingFaceRobustnessEvaluationRuntime",
    "PromptTokenizationProfile",
    "prompt_tokenization_profile",
    "teacher_forced_kl_readout",
    "window_patched_forward",
]
