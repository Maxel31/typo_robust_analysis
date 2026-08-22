"""Read-only Hugging Face runtime for the transition-layer causal gate."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from collections.abc import Mapping, Sequence
from typing import Any

from typo_robust_training.localization.corpus_targets import clean_corpus_targets
from typo_robust_training.localization.prompting import word_final_token_positions
from typo_robust_training.state_gate.artifacts import SingleLayerGateRecord
from typo_robust_training.state_gate.config import SingleLayerGateProtocol


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


class HuggingFaceSingleLayerGateProvider:
    """Patch exactly the parent transition layer and its frozen controls."""

    def __init__(self, *, protocol: SingleLayerGateProtocol, gpu_id: str) -> None:
        if not isinstance(protocol, SingleLayerGateProtocol):
            raise TypeError("single-layer gate runtime requires its validated protocol")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError("CUDA_VISIBLE_DEVICES conflicts with the requested gate GPU")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("single-layer gate requires exactly one requested CUDA GPU")
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.wrapper import create_model_wrapper

        wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self.protocol = protocol
        self.gpu_id = gpu_id
        self._torch = torch
        self.model = wrapper.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = wrapper.tokenizer
        if getattr(self.tokenizer, "is_fast", False) is not True:
            raise ValueError("single-layer gate requires a fast tokenizer")
        self.layers = tuple(find_decoder_layers(self.model))
        self.device = next(self.model.parameters()).device
        if len(self.layers) != protocol.decoder_layers or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise ValueError("single-layer gate model identity or freeze boundary differs")
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision is not None and actual_revision != protocol.model_revision:
            raise ValueError("single-layer gate loaded a different model revision")

    def _tokenize(self, text: str) -> tuple[Any, Any, tuple[int, ...], tuple[tuple[int, int], ...]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("single-layer gate tokenizer must return one mapping")
        ids = encoded.get("input_ids")
        mask = encoded.get("attention_mask")
        offsets = encoded.get("offset_mapping")
        if ids is None or ids.ndim != 2 or int(ids.shape[0]) != 1:
            raise ValueError("single-layer gate tokenizer must return one input sequence")
        if mask is None:
            mask = self._torch.ones_like(ids)
        if offsets is None or offsets.ndim != 3 or tuple(offsets.shape[:2]) != tuple(ids.shape):
            raise ValueError("single-layer gate tokenizer must return exact offsets")
        if mask.shape != ids.shape or not bool(mask.all()):
            raise ValueError("single-layer gate input must be unpadded")
        return (
            ids.to(self.device),
            mask.to(self.device),
            tuple(int(value) for value in ids[0].tolist()),
            tuple((int(row[0]), int(row[1])) for row in offsets[0].tolist()),
        )

    def _append_targets(self, ids: Any, mask: Any, targets: tuple[int, ...]) -> tuple[Any, Any]:
        prefix = self._torch.tensor([targets[:-1]], dtype=ids.dtype, device=ids.device)
        return (
            self._torch.cat((ids, prefix), dim=1),
            self._torch.cat((mask, self._torch.ones_like(prefix)), dim=1),
        )

    def _target_logits(self, output: Any, *, prompt_tokens: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("single-layer gate model output lacks rank-three logits")
        selected = logits[0, prompt_tokens - 1 : prompt_tokens + 15, :]
        if int(selected.shape[0]) != 16:
            raise ValueError("single-layer gate output lacks sixteen target logits")
        return selected.detach()

    def _kl(self, reference: Any, comparison: Any) -> tuple[float, ...]:
        reference_log = self._torch.log_softmax(reference.double(), dim=-1)
        comparison_log = self._torch.log_softmax(comparison.double(), dim=-1)
        values = (reference_log.exp() * (reference_log - comparison_log)).sum(dim=-1).clamp_min(0.0)
        return tuple(float(value) for value in values[1:16].cpu().tolist())

    def _donor(self, ids: Any, mask: Any, *, layer: int, position: int) -> Any:
        from typo_cot.experiments.layerwise_kl_patching.patching import capture_block_outputs

        donors = capture_block_outputs(
            self.layers,
            positions=(position,),
            forward=lambda: self.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
            ),
        )
        return donors[layer]

    def _patched(
        self,
        *,
        ids: Any,
        mask: Any,
        layer: int,
        position: int,
        donor: Any,
        prompt_tokens: int,
        reference: Any,
    ) -> tuple[float, ...]:
        from typo_cot.experiments.layerwise_kl_patching.patching import BlockOutputPatch

        with BlockOutputPatch(
            self.layers,
            layer_index=layer,
            positions=(position,),
            donor_values=donor,
        ):
            output = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        return self._kl(reference, self._target_logits(output, prompt_tokens=prompt_tokens))

    def scan(
        self,
        records: Sequence[SingleLayerGateRecord],
        *,
        donor_plan: Mapping[str, str],
        transition_layer: int,
    ) -> Sequence[Mapping[str, object]]:
        if not 1 <= transition_layer < len(self.layers):
            raise ValueError("single-layer gate transition lies outside the decoder")
        prepared: dict[str, dict[str, object]] = {}
        with self._torch.inference_mode():
            for record in records:
                clean_prefix = record.clean_text[: record.clean_word_char_span[1]]
                typo_prefix = record.typo_text[: record.typo_word_char_span[1]]
                clean_ids, clean_mask, clean_tokens, clean_offsets = self._tokenize(clean_prefix)
                typo_ids, typo_mask, typo_tokens, typo_offsets = self._tokenize(typo_prefix)
                _full_ids, _full_mask, full_clean_tokens, _full_offsets = self._tokenize(record.clean_text)
                clean_position = word_final_token_positions(
                    self.tokenizer,
                    text=clean_prefix,
                    spans=(record.clean_word_char_span,),
                )[0]
                typo_position = word_final_token_positions(
                    self.tokenizer,
                    text=typo_prefix,
                    spans=(record.typo_word_char_span,),
                )[0]
                if clean_position != len(clean_tokens) - 1 or typo_position != len(typo_tokens) - 1:
                    raise ValueError("single-layer gate prefix must end at edited-word-final token")
                targets, reason = clean_corpus_targets(
                    full_clean_token_ids=full_clean_tokens,
                    clean_prompt_token_ids=clean_tokens,
                    final_edited_token=clean_position,
                    count=self.protocol.teacher_forced_tokens,
                )
                prepared[record.pair_id] = {
                    "record": record,
                    "clean_ids": clean_ids,
                    "clean_mask": clean_mask,
                    "typo_ids": typo_ids,
                    "typo_mask": typo_mask,
                    "clean_tokens": clean_tokens,
                    "typo_tokens": typo_tokens,
                    "clean_offsets": clean_offsets,
                    "typo_offsets": typo_offsets,
                    "clean_position": clean_position,
                    "typo_position": typo_position,
                    "targets": targets,
                    "invalid_reason": reason,
                    "clean_donor": self._donor(
                        clean_ids,
                        clean_mask,
                        layer=transition_layer,
                        position=clean_position,
                    ),
                    "typo_donor": self._donor(
                        typo_ids,
                        typo_mask,
                        layer=transition_layer,
                        position=typo_position,
                    ),
                }
            output_rows: list[dict[str, object]] = []
            for record in records:
                item = prepared[record.pair_id]
                donor_item = prepared[donor_plan[record.pair_id]]
                targets = item["targets"]
                reason = item["invalid_reason"]
                clean_position = int(item["clean_position"])
                typo_position = int(item["typo_position"])
                base = {
                    "pair_id": record.pair_id,
                    "source_group_sha256": record.source_group_sha256,
                    "stratum": record.stratum,
                    "transition_layer": transition_layer,
                    "clean_word_final_token": clean_position,
                    "typo_word_final_token": typo_position,
                    "offset_patch_token": typo_position + self.protocol.offset_control_tokens,
                    "cross_donor_pair_id": donor_plan[record.pair_id],
                    "cross_donor_clean_word_final_token": int(donor_item["clean_position"]),
                    "cross_donor_clean_prompt_offsets": [
                        list(row) for row in donor_item["clean_offsets"]
                    ],
                    "clean_prompt_offsets": [list(row) for row in item["clean_offsets"]],
                    "typo_prompt_offsets": [list(row) for row in item["typo_offsets"]],
                }
                if reason is not None:
                    output_rows.append(
                        {
                            **base,
                            "target_token_ids": [],
                            "untreated_kl_2_16": [],
                            "correct_kl_2_16": [],
                            "offset_kl_2_16": [],
                            "cross_kl_2_16": [],
                            "self_copy_kl_2_16": [],
                            "invalid_reason": reason,
                        }
                    )
                    continue
                clean_full = self._append_targets(item["clean_ids"], item["clean_mask"], targets)
                typo_full = self._append_targets(item["typo_ids"], item["typo_mask"], targets)
                clean_output = self.model(
                    input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False
                )
                reference = self._target_logits(
                    clean_output, prompt_tokens=len(item["clean_tokens"])
                )
                typo_output = self.model(
                    input_ids=typo_full[0], attention_mask=typo_full[1], use_cache=False
                )
                untreated = self._kl(
                    reference,
                    self._target_logits(typo_output, prompt_tokens=len(item["typo_tokens"])),
                )
                if sum(untreated) / len(untreated) <= self.protocol.denominator_min_exclusive:
                    output_rows.append(
                        {
                            **base,
                            "target_token_ids": [],
                            "untreated_kl_2_16": [],
                            "correct_kl_2_16": [],
                            "offset_kl_2_16": [],
                            "cross_kl_2_16": [],
                            "self_copy_kl_2_16": [],
                            "invalid_reason": "untreated-kl-at-or-below-1e-9",
                        }
                    )
                    continue
                patch_common = {
                    "ids": typo_full[0],
                    "mask": typo_full[1],
                    "layer": transition_layer,
                    "prompt_tokens": len(item["typo_tokens"]),
                    "reference": reference,
                }
                output_rows.append(
                    {
                        **base,
                        "target_token_ids": list(targets),
                        "untreated_kl_2_16": list(untreated),
                        "correct_kl_2_16": list(
                            self._patched(
                                **patch_common,
                                position=typo_position,
                                donor=item["clean_donor"],
                            )
                        ),
                        "offset_kl_2_16": list(
                            self._patched(
                                **patch_common,
                                position=typo_position + self.protocol.offset_control_tokens,
                                donor=item["clean_donor"],
                            )
                        ),
                        "cross_kl_2_16": list(
                            self._patched(
                                **patch_common,
                                position=typo_position,
                                donor=donor_item["clean_donor"],
                            )
                        ),
                        "self_copy_kl_2_16": list(
                            self._patched(
                                **patch_common,
                                position=typo_position,
                                donor=item["typo_donor"],
                            )
                        ),
                        "invalid_reason": None,
                    }
                )
        return tuple(output_rows)

    def provenance(self) -> Mapping[str, object]:
        torch = self._torch
        return {
            "schema_version": "single-layer-gate-runtime/v1",
            "provider": "hugging-face-single-layer-gate/v1",
            "model": self.protocol.model,
            "model_revision": self.protocol.model_revision,
            "code_revision": self.protocol.code_revision,
            "decoder_layers": len(self.layers),
            "dtype": "bfloat16",
            "hook_site": "complete-decoder-block-residual-output",
            "coordinate": "edited-word-final-token/v1",
            "readout": "teacher-forced-tokens-2-through-16-inclusive/v1",
            "base_model_frozen": True,
            "packages": {
                "python": platform.python_version(),
                "torch": _version("torch"),
                "transformers": _version("transformers"),
                "typo-cot": _version("typo-cot"),
            },
            "hardware": {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        }


__all__ = ["HuggingFaceSingleLayerGateProvider"]
