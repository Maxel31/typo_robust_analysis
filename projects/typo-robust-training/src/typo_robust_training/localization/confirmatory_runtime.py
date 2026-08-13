"""GPU runtime for confirmatory generic-text joint-window patching."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

from typo_robust_training.localization.confirmatory_config import (
    ConfirmatoryLocalizationProtocol,
)
from typo_robust_training.localization.confirmatory_records import JointWindowScan
from typo_robust_training.localization.generic_runtime import clean_corpus_targets
from typo_robust_training.localization.prompting import word_final_token_positions


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@contextmanager
def joint_block_output_patch(
    layers: Sequence[Any],
    *,
    window: tuple[int, int],
    positions: Sequence[int],
    donor_values_by_layer: Sequence[Any],
) -> Iterator[None]:
    """Patch every complete-block residual output in one contiguous window."""

    decoder_layers = tuple(layers)
    donors = tuple(donor_values_by_layer)
    start, stop = window
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= len(decoder_layers)
        or len(donors) != len(decoder_layers)
    ):
        raise ValueError("joint block-output patch dimensions are invalid")
    from typo_cot.experiments.layerwise_kl_patching.patching import BlockOutputPatch

    with ExitStack() as stack:
        for layer_index in range(start, stop):
            stack.enter_context(
                BlockOutputPatch(
                    decoder_layers,
                    layer_index=layer_index,
                    positions=positions,
                    donor_values=donors[layer_index],
                )
            )
        yield


class HuggingFaceConfirmatoryJointWindowRuntime:
    """Measure actual joint-window patches on fixed generic-text pairs."""

    def __init__(self, *, protocol: ConfirmatoryLocalizationProtocol, gpu_id: str) -> None:
        if not isinstance(protocol, ConfirmatoryLocalizationProtocol):
            raise TypeError("protocol must be ConfirmatoryLocalizationProtocol")
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
            raise RuntimeError(
                "confirmatory joint-window localization requires exactly one requested CUDA GPU"
            )
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
        self.model.requires_grad_(False)
        self.tokenizer = self.wrapper.tokenizer
        self.tokenizer.padding_side = "left"
        self.layers = tuple(find_decoder_layers(self.model))
        self.device = next(self.model.parameters()).device
        if len(self.layers) != protocol.decoder_layers:
            raise ValueError("confirmatory runtime decoder-layer count differs")
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision is not None and actual_revision != protocol.model_revision:
            raise ValueError("confirmatory runtime model revision differs")

    @staticmethod
    def _edit(record: Mapping[str, object]) -> tuple[tuple[int, int], tuple[int, int]]:
        edits = record.get("edits")
        if not isinstance(edits, list) or len(edits) != 1 or not isinstance(edits[0], Mapping):
            raise ValueError("confirmatory runtime requires exactly one edit")
        clean = edits[0].get("clean_char_span")
        typo = edits[0].get("typo_char_span")
        if (
            not isinstance(clean, list)
            or len(clean) != 2
            or not isinstance(typo, list)
            or len(typo) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in (*clean, *typo)
            )
            or not 0 <= clean[0] < clean[1]
            or not 0 <= typo[0] < typo[1]
        ):
            raise ValueError("confirmatory runtime edit spans are invalid")
        return (clean[0], clean[1]), (typo[0], typo[1])

    def _tokenize(self, text: str) -> tuple[Any, Any, tuple[int, ...]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("confirmatory tokenizer must return an object")
        ids = encoded.get("input_ids")
        mask = encoded.get("attention_mask")
        if ids is None or ids.ndim != 2 or int(ids.shape[0]) != 1:
            raise ValueError("confirmatory tokenizer must return one sequence")
        if mask is None:
            mask = self._torch.ones_like(ids)
        if mask.shape != ids.shape or not bool(mask.all()):
            raise ValueError("confirmatory text must be unpadded and fully attended")
        token_ids = tuple(int(value) for value in ids[0].tolist())
        return ids.to(self.device), mask.to(self.device), token_ids

    def _append_targets(self, ids: Any, mask: Any, targets: tuple[int, ...]) -> tuple[Any, Any]:
        prefix = self._torch.tensor([targets[:-1]], dtype=ids.dtype, device=ids.device)
        combined_ids = self._torch.cat((ids, prefix), dim=1)
        combined_mask = self._torch.cat(
            (mask, self._torch.ones_like(prefix, dtype=mask.dtype)), dim=1
        )
        if int(combined_ids.shape[1]) > self.protocol.max_sequence_length:
            raise ValueError("confirmatory teacher-forced input exceeds the sequence limit")
        return combined_ids, combined_mask

    @staticmethod
    def _target_logits(output: Any, *, prompt_tokens: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("confirmatory model output must contain rank-three logits")
        selected = logits[0, prompt_tokens - 1 : prompt_tokens + 15, :]
        if int(selected.shape[0]) != 16:
            raise ValueError("confirmatory output does not cover sixteen targets")
        return selected

    def _kl_2_16(self, reference: Any, comparison: Any) -> tuple[float, ...]:
        reference_log = self._torch.log_softmax(reference.float(), dim=-1)
        comparison_log = self._torch.log_softmax(comparison.float(), dim=-1)
        values = (reference_log.exp() * (reference_log - comparison_log)).sum(dim=-1).clamp_min(0.0)
        return tuple(float(value) for value in values[1:16].detach().cpu().tolist())

    def _invalid(
        self,
        *,
        record_id: str,
        role: str,
        reason: str,
        audit: Mapping[str, object],
    ) -> JointWindowScan:
        return JointWindowScan(
            record_id=record_id,
            role=role,
            decoder_layers=len(self.layers),
            window_width=self.protocol.window_width,
            target_token_ids=(),
            untreated_kl_2_16=(),
            patched_kl_2_16_by_window={},
            invalid_reason=reason,
            audit=audit,
        )

    def _scan(
        self,
        record: dict[str, object],
        *,
        role: str,
        selected_window: tuple[int, int] | None,
    ) -> JointWindowScan:
        from typo_cot.experiments.layerwise_kl_patching.patching import capture_block_outputs

        record_id = record.get("record_id")
        clean_text, typo_text = record.get("clean_text"), record.get("typo_text")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record.get("source") != self.protocol.selection_source
            or record.get("split") != f"localization-{role}"
            or not isinstance(clean_text, str)
            or not isinstance(typo_text, str)
        ):
            raise ValueError("confirmatory runtime record identity differs")
        clean_span, typo_span = self._edit(record)
        clean_prefix, typo_prefix = clean_text[: clean_span[1]], typo_text[: typo_span[1]]
        clean_ids, clean_mask, clean_token_ids = self._tokenize(clean_prefix)
        typo_ids, typo_mask, typo_token_ids = self._tokenize(typo_prefix)
        _, _, full_clean_ids = self._tokenize(clean_text)
        clean_positions = word_final_token_positions(
            self.tokenizer, text=clean_prefix, spans=(clean_span,)
        )
        typo_positions = word_final_token_positions(
            self.tokenizer, text=typo_prefix, spans=(typo_span,)
        )
        targets, invalid_reason = clean_corpus_targets(
            full_clean_token_ids=full_clean_ids,
            clean_prompt_token_ids=clean_token_ids,
            final_edited_token=clean_positions[-1],
            count=self.protocol.teacher_forced_tokens,
        )
        if typo_positions[-1] != len(typo_token_ids) - 1:
            raise ValueError("confirmatory typo prompt must end at the edited-word-final token")
        audit: dict[str, object] = {
            "source": record["source"],
            "role": role,
            "clean_prompt_sha256": hashlib.sha256(clean_prefix.encode()).hexdigest(),
            "typo_prompt_sha256": hashlib.sha256(typo_prefix.encode()).hexdigest(),
            "clean_prompt_token_count": len(clean_token_ids),
            "typo_prompt_token_count": len(typo_token_ids),
            "clean_word_final_token": clean_positions[-1],
            "typo_word_final_token": typo_positions[-1],
        }
        if invalid_reason is not None:
            return self._invalid(record_id=record_id, role=role, reason=invalid_reason, audit=audit)
        clean_full = self._append_targets(clean_ids, clean_mask, targets)
        typo_full = self._append_targets(typo_ids, typo_mask, targets)
        with self._torch.inference_mode():
            clean_output = self.model(
                input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False
            )
            reference = self._target_logits(
                clean_output, prompt_tokens=len(clean_token_ids)
            ).detach()
            del clean_output
            typo_output = self.model(
                input_ids=typo_full[0], attention_mask=typo_full[1], use_cache=False
            )
            untreated = self._kl_2_16(
                reference,
                self._target_logits(typo_output, prompt_tokens=len(typo_token_ids)),
            )
            del typo_output
            denominator = sum(untreated) / len(untreated)
            audit["untreated_mean_kl_2_16"] = denominator
            if denominator <= self.protocol.denominator_min_exclusive:
                return self._invalid(
                    record_id=record_id,
                    role=role,
                    reason="untreated-kl-at-or-below-1e-9",
                    audit=audit,
                )
            donors = capture_block_outputs(
                self.layers,
                positions=clean_positions,
                forward=lambda: self.model(
                    input_ids=clean_ids, attention_mask=clean_mask, use_cache=False
                ),
            )
            if role == "selection":
                windows = tuple(
                    (start, start + self.protocol.window_width)
                    for start in range(len(self.layers) - self.protocol.window_width + 1)
                )
            else:
                if selected_window is None:
                    raise ValueError("validation requires one frozen selected window")
                windows = (selected_window,)
            patched: dict[int, tuple[float, ...]] = {}
            for window in windows:
                with joint_block_output_patch(
                    self.layers,
                    window=window,
                    positions=typo_positions,
                    donor_values_by_layer=donors,
                ):
                    output = self.model(
                        input_ids=typo_full[0],
                        attention_mask=typo_full[1],
                        use_cache=False,
                    )
                patched[window[0]] = self._kl_2_16(
                    reference,
                    self._target_logits(output, prompt_tokens=len(typo_token_ids)),
                )
                del output
        audit["target_token_text"] = self.tokenizer.convert_ids_to_tokens(list(targets))
        if selected_window is not None:
            audit["selected_window"] = list(selected_window)
        return JointWindowScan(
            record_id=record_id,
            role=role,
            decoder_layers=len(self.layers),
            window_width=self.protocol.window_width,
            target_token_ids=targets,
            untreated_kl_2_16=untreated,
            patched_kl_2_16_by_window=patched,
            invalid_reason=None,
            audit=audit,
        )

    def scan_selection_pair(self, record: dict[str, object]) -> JointWindowScan:
        return self._scan(record, role="selection", selected_window=None)

    def scan_validation_pair(
        self, record: dict[str, object], selected_window: tuple[int, int]
    ) -> JointWindowScan:
        return self._scan(record, role="validation", selected_window=selected_window)

    def provenance(self) -> Mapping[str, object]:
        torch = self._torch
        return {
            "runtime": "HuggingFaceConfirmatoryJointWindowRuntime/v1",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "model": self.protocol.model,
            "requested_revision": self.protocol.model_revision,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "num_decoder_layers": len(self.layers),
            "window_width": self.protocol.window_width,
            "dtype": self.protocol.dtype,
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "selection_metric": self.protocol.window_selection_metric,
            "patch_direction": "clean-to-typo",
            "patch_site": "complete-decoder-block-residual-output",
            "readout": "teacher-forced-tokens-2-through-16-inclusive/v1",
        }


__all__ = [
    "HuggingFaceConfirmatoryJointWindowRuntime",
    "joint_block_output_patch",
]
