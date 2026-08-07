"""Hugging Face GPU runtime for fixed-window free-answer patching."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from typo_cot.experiments.layerwise_answer_patching.runtime import (
    continuation_token_ids,
)

if TYPE_CHECKING:
    from typo_cot.experiments.fixed_window_answer_patching.runner import (
        AnswerGeneration,
        BaselineScan,
        FixedWindowAnswerPatchingConfig,
        LayerWindow,
        PairWindowScan,
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


class HuggingFaceFixedWindowAnswerPatchingRuntime:
    """Load one causal LM and patch each requested layer window in one pass."""

    def __init__(
        self,
        config: FixedWindowAnswerPatchingConfig,
        *,
        revision: str,
    ) -> None:
        if not revision:
            raise ValueError("fixed-window-answer-patching requires a pinned source revision")
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
            raise RuntimeError("fixed-window-answer-patching requires the requested CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "fixed-window-answer-patching requires exactly one visible GPU; "
                "set CUDA_VISIBLE_DEVICES to one physical device"
            )

        from typo_cot.experiments.layerwise_kl_patching.patching import (
            find_decoder_layers,
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
        self.tokenizer.padding_side = "left"
        self.layers = find_decoder_layers(self.model)
        self.num_layers = len(self.layers)
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _span(value: object, *, field: str, prompt_length: int) -> tuple[int, int]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{field} must be an object")
        start, end = value.get("start"), value.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= prompt_length
        ):
            raise ValueError(f"{field} is not a valid prompt character span")
        return start, end

    def _tokenize_and_validate(
        self,
        pair: Mapping[str, object],
        *,
        side: str,
    ) -> tuple[Any, Any, tuple[int, ...]]:
        payload = pair.get(side)
        if not isinstance(payload, Mapping):
            raise TypeError(f"pair.{side} must be an object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"pair.{side}.prompt must be non-empty")
        try:
            encoded = self.tokenizer(
                prompt,
                add_special_tokens=True,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
        except (NotImplementedError, TypeError) as exc:
            raise ValueError(
                "the tokenizer must provide offset_mapping for aligned final-token validation"
            ) from exc

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        raw_offsets = encoded.get("offset_mapping")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("tokenizer must return one unpadded input sequence")
        if input_ids.shape[1] != payload.get("prompt_token_count"):
            raise ValueError(f"runtime {side} token count differs from the pair record")
        if raw_offsets is None:
            raise ValueError("tokenizer returned no offset_mapping")
        offsets = raw_offsets[0].tolist()
        if len(offsets) != input_ids.shape[1]:
            raise ValueError("tokenizer offset_mapping length differs from input_ids")

        aligned_words = pair.get("aligned_words")
        if not isinstance(aligned_words, list) or not aligned_words:
            raise ValueError("runtime received a pair without aligned words")
        positions: list[int] = []
        for index, raw_word in enumerate(aligned_words):
            if not isinstance(raw_word, Mapping):
                raise TypeError(f"aligned_words[{index}] must be an object")
            final = raw_word.get(f"{side}_final_token")
            if not isinstance(final, int) or isinstance(final, bool):
                raise TypeError(f"aligned_words[{index}].{side}_final_token must be integer")
            word_start, word_end = self._span(
                raw_word.get(f"{side}_prompt_span"),
                field=f"aligned_words[{index}].{side}_prompt_span",
                prompt_length=len(prompt),
            )
            overlapping = [
                token_index
                for token_index, (raw_start, raw_end) in enumerate(offsets)
                if int(raw_end) > int(raw_start)
                and int(raw_start) < word_end
                and word_start < int(raw_end)
            ]
            if not overlapping or final != overlapping[-1]:
                raise ValueError(
                    f"runtime {side} final-token coordinate does not match tokenizer offsets "
                    f"for aligned_words[{index}]"
                )
            positions.append(final)
        if len(positions) != len(set(positions)):
            raise ValueError(f"runtime {side} final-token coordinates are duplicated")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
            raise ValueError("tokenizer must return one unpadded, fully attended prompt")
        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            tuple(positions),
        )

    @staticmethod
    def _sample_id(pair: Mapping[str, object]) -> str:
        sample_id = pair.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("pair sample_id must be non-empty")
        return sample_id

    @staticmethod
    def _gold_answer(pair: Mapping[str, object]) -> str:
        gold_answer = pair.get("gold_answer")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError("pair gold_answer must be non-empty")
        return gold_answer

    @property
    def _extraction_benchmark(self) -> str:
        return "mmlu_pro" if self.config.benchmark == "mmlu-pro" else self.config.benchmark

    def _generate(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        correct_answer: str,
        patch: Any | None = None,
    ) -> AnswerGeneration:
        from typo_cot.evaluation.fallback import extract_with_fallback
        from typo_cot.experiments.fixed_window_answer_patching.runner import (
            AnswerGeneration,
        )

        context = patch if patch is not None else nullcontext()
        with context:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
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
            )
        token_ids = continuation_token_ids(
            output_ids,
            prompt_length=int(input_ids.shape[1]),
        )
        text = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        extraction = extract_with_fallback(
            text,
            benchmark=self._extraction_benchmark,
            correct_answer=correct_answer,
        )
        return AnswerGeneration(
            token_ids=token_ids,
            text=text,
            value=extraction.value,
            is_extracted=extraction.is_extracted,
            is_correct=extraction.is_correct,
            method=extraction.method,
            primary_method=extraction.primary_method,
        )

    def _capture(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        positions: tuple[int, ...],
    ) -> list[Any]:
        from typo_cot.experiments.layerwise_kl_patching.patching import (
            capture_block_outputs,
        )

        return capture_block_outputs(
            self.layers,
            positions=positions,
            forward=lambda: self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ),
        )

    def regenerate_baseline(self, pair: dict[str, object]) -> BaselineScan:
        from typo_cot.experiments.fixed_window_answer_patching.runner import BaselineScan

        sample_id = self._sample_id(pair)
        correct_answer = self._gold_answer(pair)
        clean_ids, clean_mask, _ = self._tokenize_and_validate(pair, side="clean")
        edited_ids, edited_mask, _ = self._tokenize_and_validate(pair, side="edited")
        with self._torch.inference_mode():
            clean = self._generate(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                correct_answer=correct_answer,
            )
            edited = self._generate(
                input_ids=edited_ids,
                attention_mask=edited_mask,
                correct_answer=correct_answer,
            )
        return BaselineScan(sample_id=sample_id, clean=clean, edited=edited)

    def scan_pair(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        windows: tuple[LayerWindow, ...],
        directions: tuple[str, ...],
    ) -> PairWindowScan:
        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )
        from typo_cot.experiments.fixed_window_answer_patching.runner import (
            DirectionWindowScan,
            PairWindowScan,
        )

        sample_id = self._sample_id(pair)
        if baseline.sample_id != sample_id:
            raise ValueError("baseline sample_id does not match the pair")
        if not windows:
            raise ValueError("windows must contain at least one layer window")
        correct_answer = self._gold_answer(pair)
        clean_ids, clean_mask, clean_positions = self._tokenize_and_validate(
            pair,
            side="clean",
        )
        edited_ids, edited_mask, edited_positions = self._tokenize_and_validate(
            pair,
            side="edited",
        )
        if len(clean_positions) != len(edited_positions):
            raise ValueError("clean and edited alignment cardinalities differ")

        for window in windows:
            if not 0 <= window.start < window.stop <= self.num_layers:
                raise ValueError(f"layer window {window.label} is outside [0, {self.num_layers})")

        with self._torch.inference_mode():
            clean_cache = self._capture(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                positions=clean_positions,
            )
            edited_cache = self._capture(
                input_ids=edited_ids,
                attention_mask=edited_mask,
                positions=edited_positions,
            )
            direction_results: dict[str, DirectionWindowScan] = {}
            for direction in directions:
                if direction == "clean-to-edited":
                    recipient_ids, recipient_mask = edited_ids, edited_mask
                    recipient_positions, donor_cache = edited_positions, clean_cache
                elif direction == "edited-to-clean":
                    recipient_ids, recipient_mask = clean_ids, clean_mask
                    recipient_positions, donor_cache = clean_positions, edited_cache
                else:
                    raise ValueError(f"unsupported direction: {direction!r}")

                patched_by_window: dict[str, AnswerGeneration] = {}
                for window in windows:
                    layer_indices = tuple(range(window.start, window.stop))
                    patch = PrefillBlockOutputWindowPatch(
                        self.layers,
                        layer_indices=layer_indices,
                        positions=recipient_positions,
                        donor_values=tuple(donor_cache[index] for index in layer_indices),
                    )
                    patched_by_window[window.label] = self._generate(
                        input_ids=recipient_ids,
                        attention_mask=recipient_mask,
                        correct_answer=correct_answer,
                        patch=patch,
                    )
                direction_results[direction] = DirectionWindowScan(
                    patched_by_window=patched_by_window
                )
        return PairWindowScan(sample_id=sample_id, directions=direction_results)

    def provenance(self) -> dict[str, object]:
        torch = self._torch
        model_metadata_revision = getattr(self.model.config, "_commit_hash", None)
        model_revision = model_metadata_revision or self.revision
        model_revision_source = (
            "model-config-metadata" if model_metadata_revision else "explicit-load-revision"
        )
        tokenizer_metadata_revision = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        tokenizer_revision = tokenizer_metadata_revision or self.revision
        tokenizer_revision_source = (
            "tokenizer-init-metadata" if tokenizer_metadata_revision else "explicit-load-revision"
        )
        decoder = self.model.get_decoder()
        layer_container = getattr(decoder, "layers", None)
        adapter = (
            f"{type(decoder).__module__}.{type(decoder).__name__}.layers"
            if layer_container is not None
            else "unknown"
        )
        return {
            "operation": "fixed-window-answer-patching",
            "runtime": "HuggingFaceFixedWindowAnswerPatchingRuntime",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "lxt": _package_version("lxt"),
            "model": self.config.model,
            "requested_revision": self.revision,
            "model_revision": model_revision,
            "model_revision_source": model_revision_source,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_revision_source": tokenizer_revision_source,
            "decoder_adapter": adapter,
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_new_tokens": 512,
                "use_cache": True,
                "return_dict_in_generate": False,
                "output_scores": False,
                "padding_side": getattr(self.tokenizer, "padding_side", "left"),
                "patch_application": "all-window-layers-on-prompt-prefill-exactly-once",
            },
            "answer_extraction": "primary-then-empty-only-fallback/v1",
        }
