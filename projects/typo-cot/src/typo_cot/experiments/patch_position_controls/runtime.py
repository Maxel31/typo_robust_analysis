"""Hugging Face GPU runtime for clean-to-edited patch-position controls."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typo_cot.experiments.layerwise_kl_patching.runtime import (
    HuggingFaceLayerwiseKLPatchingRuntime,
)
from typo_cot.experiments.patch_position_controls.planning import (
    PositionCoordinates,
    locate_position_coordinates,
)

if TYPE_CHECKING:
    from typo_cot.experiments.patch_position_controls.runner import PositionControlConfig


_ALTERNATIVE_POSITIONS = ("prompt-final", "question-final")
POSITION_RUNTIME_PROTOCOL = {
    "direction": "clean-to-edited",
    "donor_capture": "one-clean-untreated-forward-over-position-union",
    "recipient_baseline": "one-edited-untreated-forward",
    "readout": "actual-model-output-logits-at-prompt-final-token",
    "prompt-final": "final-token-index-of-unpadded-prompt",
    "question-final": "last-token-overlapping-recorded-editable-prompt-span",
    "correct_arm_source": "verified-layerwise-kl-reference-run",
}


@dataclass(frozen=True, slots=True)
class AlternativePositionScan:
    """Coordinates and one patched KL value per decoder layer."""

    coordinates: PositionCoordinates
    patched_kl_by_layer: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PositionControlPairScan:
    """Untreated denominator and alternative-position scans for one pair."""

    sample_id: str
    denominator_kl: float
    positions: Mapping[str, AlternativePositionScan]


class HuggingFacePositionControlRuntime(HuggingFaceLayerwiseKLPatchingRuntime):
    """Patch clean donor states into alternative edited-prompt positions.

    The clean untreated forward captures the union of all requested donor
    positions from every decoder block.  A separate edited untreated forward
    supplies the common denominator.  Each subsequent intervention reads
    ``output.logits[0, -1]`` after the patched decoder block has run, including
    when that block is the final layer.
    """

    def __init__(self, config: PositionControlConfig, *, revision: str) -> None:
        if not revision:
            raise ValueError("patch-position-controls requires a pinned source revision")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != config.gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={config.gpu_id!r}"
            )
        super().__init__(config, revision=revision)

    def _tokenize_for_positions(
        self,
        pair: Mapping[str, object],
        *,
        side: str,
    ) -> tuple[Any, Any, tuple[tuple[int, int], ...]]:
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
                "the tokenizer must provide offset_mapping for patch-position controls"
            ) from exc

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        raw_offsets = encoded.get("offset_mapping")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("tokenizer must return one unpadded input sequence")
        recorded_count = payload.get("prompt_token_count")
        if input_ids.shape[1] != recorded_count:
            raise ValueError(f"runtime {side} token count differs from the pair record")
        if raw_offsets is None:
            raise ValueError("tokenizer returned no offset_mapping")
        try:
            offset_row = raw_offsets[0]
            raw_values = offset_row.tolist() if hasattr(offset_row, "tolist") else offset_row
            offsets = tuple(tuple(value) for value in raw_values)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("tokenizer returned malformed offset_mapping") from exc
        if len(offsets) != input_ids.shape[1]:
            raise ValueError("tokenizer offset_mapping length differs from input_ids")

        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("tokenizer attention_mask shape differs from input_ids")
        if not bool(self._torch.all(attention_mask == 1).item()):
            raise ValueError("patch-position controls require one unpadded prompt sequence")

        # locate_position_coordinates performs the detailed integer/range
        # validation.  Keeping the offsets on CPU also avoids retaining model
        # tensors solely for coordinate provenance.
        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            offsets,  # type: ignore[return-value]
        )

    def _plain_logits(self, *, input_ids: Any, attention_mask: Any) -> Any:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return output.logits[0, -1, :].detach().float().cpu()

    @staticmethod
    def _validate_positions(positions: tuple[str, ...]) -> tuple[str, ...]:
        try:
            normalized = tuple(positions)
        except TypeError as exc:
            raise ValueError("positions must be a sequence") from exc
        if not normalized:
            raise ValueError("positions must contain at least one alternative position")
        if len(normalized) != len(set(normalized)):
            raise ValueError("positions must not contain duplicates")
        unsupported = [name for name in normalized if name not in _ALTERNATIVE_POSITIONS]
        if unsupported:
            raise ValueError(f"unsupported runtime position: {unsupported[0]!r}")
        return normalized

    def scan_pair(
        self,
        pair: dict[str, object],
        positions: tuple[str, ...],
    ) -> PositionControlPairScan:
        """Run every requested alternative position for one source pair."""

        from typo_cot.experiments.layerwise_kl_patching.metrics import (
            KL_DENOMINATOR_EPSILON,
            kl_from_logits,
        )

        sample_id = pair.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("pair sample_id must be non-empty")
        requested = self._validate_positions(positions)
        clean_ids, clean_mask, clean_offsets = self._tokenize_for_positions(
            pair,
            side="clean",
        )
        edited_ids, edited_mask, edited_offsets = self._tokenize_for_positions(
            pair,
            side="edited",
        )
        all_coordinates = locate_position_coordinates(pair, clean_offsets, edited_offsets)

        # Capture a donor row only once even when two controls resolve to the
        # same clean token.  Per-control row selections below restore the exact
        # order specified by PositionCoordinates.
        donor_positions = tuple(
            dict.fromkeys(
                position
                for name in requested
                for position in all_coordinates[name].source_positions
            )
        )
        donor_row = {position: index for index, position in enumerate(donor_positions)}

        with self._torch.inference_mode():
            clean_logits, clean_cache = self._untreated(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                positions=donor_positions,
            )
            edited_logits = self._plain_logits(
                input_ids=edited_ids,
                attention_mask=edited_mask,
            )
            denominator = kl_from_logits(clean_logits, edited_logits)

            scans: dict[str, AlternativePositionScan] = {}
            for name in requested:
                coordinates = all_coordinates[name]
                patched_values: list[float] = []
                if math.isfinite(denominator) and denominator > KL_DENOMINATOR_EPSILON:
                    row_indices = [donor_row[position] for position in coordinates.source_positions]
                    for layer_index in range(self.num_layers):
                        donor_values = clean_cache[layer_index][row_indices, :]
                        patched_logits = self._patched_logits(
                            input_ids=edited_ids,
                            attention_mask=edited_mask,
                            layer_index=layer_index,
                            positions=coordinates.destination_positions,
                            donor_values=donor_values,
                        )
                        patched_values.append(kl_from_logits(clean_logits, patched_logits))
                scans[name] = AlternativePositionScan(
                    coordinates=coordinates,
                    patched_kl_by_layer=tuple(patched_values),
                )

        return PositionControlPairScan(
            sample_id=sample_id,
            denominator_kl=denominator,
            positions=scans,
        )

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload["operation"] = "patch-position-controls"
        payload["runtime"] = "HuggingFacePositionControlRuntime"
        payload["position_controls"] = dict(POSITION_RUNTIME_PROTOCOL)
        return payload


__all__ = [
    "AlternativePositionScan",
    "HuggingFacePositionControlRuntime",
    "POSITION_RUNTIME_PROTOCOL",
    "PositionControlPairScan",
]
