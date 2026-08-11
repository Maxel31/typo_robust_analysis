"""GPU runtime for the donor-source by write-position coordinate grid."""

from __future__ import annotations

from collections.abc import Mapping

from typo_cot.experiments.rebuttal_runtime import (
    manifest_runtime_pair,
    manifest_token_positions,
    require_mapping,
)
from typo_cot.experiments.six_setting_patch_controls import ControlArmResult
from typo_cot.experiments.six_setting_patch_controls.runtime import (
    HuggingFaceSixSettingPatchControlsRuntime,
)

_GENERATED_ARMS = ("E->O", "O->E", "O->O")


class HuggingFaceSourceWriteCoordinateGridRuntime(HuggingFaceSixSettingPatchControlsRuntime):
    """Generate the three missing cells while reusing the fixed E-to-E cell."""

    def scan_grid(
        self,
        pair: Mapping[str, object],
    ) -> Mapping[str, ControlArmResult]:
        if self.num_layers < 6:
            raise ValueError("source/write grid requires at least six decoder layers")
        plans = require_mapping(pair.get("controls"), field="controls")
        correct = require_mapping(plans.get("correct"), field="controls.correct")
        offset = require_mapping(plans.get("offset_2"), field="controls.offset_2")
        if correct.get("valid") is not True or offset.get("valid") is not True:
            raise ValueError("source/write runtime requires a common-valid coordinate plan")

        recipient = manifest_runtime_pair(pair)
        clean_ids, clean_mask, clean_final = self._tokenize_and_validate(
            recipient,
            side="clean",
        )
        typo_ids, typo_mask, typo_final = self._tokenize_and_validate(
            recipient,
            side="edited",
        )
        edited_source = manifest_token_positions(
            correct.get("source_positions"), field="controls.correct.source_positions"
        )
        edited_write = manifest_token_positions(
            correct.get("destination_positions"),
            field="controls.correct.destination_positions",
        )
        offset_source = manifest_token_positions(
            offset.get("source_positions"), field="controls.offset_2.source_positions"
        )
        offset_write = manifest_token_positions(
            offset.get("destination_positions"),
            field="controls.offset_2.destination_positions",
        )
        if edited_source != clean_final or edited_write != typo_final:
            raise ValueError("source/write edited coordinates differ from tokenizer alignment")
        cardinalities = {
            len(edited_source),
            len(edited_write),
            len(offset_source),
            len(offset_write),
        }
        if len(cardinalities) != 1:
            raise ValueError("source/write coordinate cardinalities differ")

        coordinates = {
            "E->O": (edited_source, offset_write),
            "O->E": (offset_source, edited_write),
            "O->O": (offset_source, offset_write),
        }
        gold_answer = self._gold_answer(recipient)
        results: dict[str, ControlArmResult] = {}
        with self._torch.inference_mode():
            donor_cache = {
                "E": self._capture(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    positions=edited_source,
                ),
                "O": self._capture(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    positions=offset_source,
                ),
            }
            for arm in _GENERATED_ARMS:
                source_positions, destination_positions = coordinates[arm]
                generation = self._generate_control(
                    input_ids=typo_ids,
                    attention_mask=typo_mask,
                    correct_answer=gold_answer,
                    patch=self._window_patch(
                        destination_positions=destination_positions,
                        donor_cache=donor_cache[arm[0]],
                    ),
                    field=f"{pair.get('pair_id')}:{arm}",
                )
                results[arm] = ControlArmResult(
                    generation=generation,
                    source_positions=source_positions,
                    destination_positions=destination_positions,
                )
        return results

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload.pop("diagnostic_controls", None)
        payload.update(
            {
                "operation": "source-write-coordinate-grid",
                "runtime": "HuggingFaceSourceWriteCoordinateGridRuntime",
                "generated_arms": list(_GENERATED_ARMS),
            }
        )
        return payload


__all__ = ["HuggingFaceSourceWriteCoordinateGridRuntime"]
