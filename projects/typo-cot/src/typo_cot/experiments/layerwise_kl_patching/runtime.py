"""Hugging Face GPU runtime for prompt-final layerwise KL patching."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typo_cot.experiments.layerwise_kl_patching.runner import (
        LayerwiseKLPatchingConfig,
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


class HuggingFaceLayerwiseKLPatchingRuntime:
    """Load one causal LM and scan prompt-only edited-word block patches."""

    def __init__(self, config: LayerwiseKLPatchingConfig, *, revision: str) -> None:
        if not revision:
            raise ValueError("layerwise-kl-patching requires a pinned source revision")
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
            raise RuntimeError("layerwise-kl-patching requires the requested CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "layerwise-kl-patching requires exactly one visible GPU; "
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
            raise ValueError(f"{field} must be an object")
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
            raise ValueError(f"pair.{side} must be an object")
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
                raise ValueError(f"aligned_words[{index}] must be an object")
            final = raw_word.get(f"{side}_final_token")
            if not isinstance(final, int) or isinstance(final, bool):
                raise ValueError(f"aligned_words[{index}].{side}_final_token must be integer")
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
        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            tuple(positions),
        )

    def _untreated(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        positions: tuple[int, ...],
    ) -> tuple[Any, dict[int, Any]]:
        from typo_cot.experiments.layerwise_kl_patching.patching import (
            capture_block_outputs,
        )

        output_holder: dict[str, Any] = {}

        def forward() -> Any:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            output_holder["output"] = output
            return output

        cache = capture_block_outputs(self.layers, positions=positions, forward=forward)
        output = output_holder["output"]
        logits = output.logits[0, -1, :].detach().float().cpu()
        return logits, cache

    def _patched_logits(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        layer_index: int,
        positions: tuple[int, ...],
        donor_values: Any,
    ) -> Any:
        from typo_cot.experiments.layerwise_kl_patching.patching import BlockOutputPatch

        with BlockOutputPatch(
            self.layers,
            layer_index=layer_index,
            positions=positions,
            donor_values=donor_values,
        ):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return output.logits[0, -1, :].detach().float().cpu()

    def scan_pair(
        self,
        pair: dict[str, object],
        directions: tuple[str, ...],
    ) -> Any:
        from typo_cot.experiments.layerwise_kl_patching.metrics import (
            KL_DENOMINATOR_EPSILON,
            kl_from_logits,
        )
        from typo_cot.experiments.layerwise_kl_patching.runner import (
            DirectionScan,
            PairScan,
        )

        sample_id = pair.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("pair sample_id must be non-empty")
        clean_ids, clean_mask, clean_positions = self._tokenize_and_validate(pair, side="clean")
        edited_ids, edited_mask, edited_positions = self._tokenize_and_validate(
            pair, side="edited"
        )
        if len(clean_positions) != len(edited_positions):
            raise ValueError("clean and edited alignment cardinalities differ")

        with self._torch.inference_mode():
            clean_logits, clean_cache = self._untreated(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                positions=clean_positions,
            )
            edited_logits, edited_cache = self._untreated(
                input_ids=edited_ids,
                attention_mask=edited_mask,
                positions=edited_positions,
            )

            direction_results: dict[str, DirectionScan] = {}
            for direction in directions:
                if direction == "clean-to-edited":
                    reference_logits, recipient_logits = clean_logits, edited_logits
                    recipient_ids, recipient_mask = edited_ids, edited_mask
                    recipient_positions = edited_positions
                    donor_cache = clean_cache
                elif direction == "edited-to-clean":
                    reference_logits, recipient_logits = edited_logits, clean_logits
                    recipient_ids, recipient_mask = clean_ids, clean_mask
                    recipient_positions = clean_positions
                    donor_cache = edited_cache
                else:  # Config validation should make this unreachable.
                    raise ValueError(f"unsupported direction: {direction!r}")

                denominator = kl_from_logits(reference_logits, recipient_logits)
                patched_values: list[float] = []
                if self._torch.isfinite(self._torch.tensor(denominator)) and (
                    denominator > KL_DENOMINATOR_EPSILON
                ):
                    for layer_index in range(self.num_layers):
                        patched_logits = self._patched_logits(
                            input_ids=recipient_ids,
                            attention_mask=recipient_mask,
                            layer_index=layer_index,
                            positions=recipient_positions,
                            donor_values=donor_cache[layer_index],
                        )
                        patched_values.append(kl_from_logits(reference_logits, patched_logits))
                direction_results[direction] = DirectionScan(
                    denominator_kl=denominator,
                    patched_kl_by_layer=tuple(patched_values),
                )
        return PairScan(sample_id=sample_id, directions=direction_results)

    def provenance(self) -> dict[str, object]:
        torch = self._torch
        model_revision = getattr(self.model.config, "_commit_hash", None)
        tokenizer_metadata_revision = getattr(self.tokenizer, "init_kwargs", {}).get(
            "_commit_hash"
        )
        tokenizer_revision = tokenizer_metadata_revision or self.revision
        tokenizer_revision_source = (
            "tokenizer-init-metadata"
            if tokenizer_metadata_revision
            else "explicit-load-revision"
        )
        layer_container = getattr(self.model.get_decoder(), "layers", None)
        adapter = (
            f"{type(self.model.get_decoder()).__module__}."
            f"{type(self.model.get_decoder()).__name__}.layers"
            if layer_container is not None
            else "unknown"
        )
        return {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "lxt": _package_version("lxt"),
            "model": self.config.model,
            "requested_revision": self.revision,
            "model_revision": model_revision,
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
        }
