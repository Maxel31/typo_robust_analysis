"""Hugging Face GPU runtime for the final-paper coordinate controls."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from typo_cot.experiments.fixed_window_answer_patching.runtime import (
    HuggingFaceFixedWindowAnswerPatchingRuntime,
)

if TYPE_CHECKING:
    from typo_cot.experiments.fixed_window_answer_patching.runner import (
        BaselineScan,
        LayerWindow,
    )
    from typo_cot.experiments.patch_coordinate_controls.runner import (
        ControlScan,
        CoordinateControlConfig,
    )


_RUNTIME_CONTROLS = frozenset({"offset-2", "cross-item", "self-copy"})


class HuggingFaceCoordinateControlRuntime(HuggingFaceFixedWindowAnswerPatchingRuntime):
    """Generate the same-item offset, cross-item, and identity controls.

    The fixed-window run supplies the paper's correct-coordinate arm.  This
    runtime regenerates its clean and edited baselines, then runs only the
    three diagnostic arms against the same edited recipient.  In every arm,
    complete residual-stream rows after each decoder block in ``window`` are
    copied during prompt prefill; cached one-token decoding steps are left
    untouched by :class:`PrefillBlockOutputWindowPatch`.
    """

    def __init__(
        self,
        config: CoordinateControlConfig,
        *,
        revision: str,
    ) -> None:
        if not revision:
            raise ValueError("patch-coordinate-controls requires a pinned source revision")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != config.gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={config.gpu_id!r}"
            )

        # The shared implementation enforces one visible CUDA device, loads
        # the pinned model/tokenizer, and provides the paper's exact greedy
        # generation and answer-extraction protocol.
        super().__init__(config, revision=revision)

    @staticmethod
    def _validate_controls(controls: tuple[str, ...]) -> None:
        if not controls:
            raise ValueError("controls must contain at least one diagnostic control")
        if len(controls) != len(set(controls)):
            raise ValueError("controls must not contain duplicates")
        unsupported = sorted(set(controls) - _RUNTIME_CONTROLS)
        if unsupported:
            raise ValueError(f"unsupported runtime controls: {unsupported}")

    def _window_patch(
        self,
        *,
        window: LayerWindow,
        positions: tuple[int, ...],
        donor_cache: list[Any],
    ) -> Any:
        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )

        layer_indices = tuple(range(window.start, window.stop))
        return PrefillBlockOutputWindowPatch(
            self.layers,
            layer_indices=layer_indices,
            positions=positions,
            donor_values=tuple(donor_cache[index] for index in layer_indices),
        )

    def scan_controls(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        donor_pair: dict[str, object] | None,
        window: LayerWindow,
        controls: tuple[str, ...],
    ) -> ControlScan:
        """Generate each requested diagnostic arm for one edited recipient."""

        from typo_cot.experiments.patch_coordinate_controls.runner import ControlScan

        self._validate_controls(controls)
        sample_id = self._sample_id(pair)
        if baseline.sample_id != sample_id:
            raise ValueError("baseline sample_id does not match the recipient pair")
        if not 0 <= window.start < window.stop <= self.num_layers:
            raise ValueError(f"layer window {window.label} is outside [0, {self.num_layers})")

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

        offset_plan: Any | None = None
        if "offset-2" in controls:
            from typo_cot.experiments.patch_coordinate_controls.planning import (
                plan_offset_positions,
            )

            offset_plan = plan_offset_positions(
                clean_positions,
                edited_positions,
                int(clean_ids.shape[1]),
                int(edited_ids.shape[1]),
                offset=2,
            )
            if not offset_plan.source_positions:
                raise ValueError(f"offset-2 has no valid coordinate pairs for {sample_id!r}")

        donor_ids: Any | None = None
        donor_mask: Any | None = None
        donor_positions: tuple[int, ...] | None = None
        if "cross-item" in controls:
            if donor_pair is None:
                raise ValueError("cross-item requires a donor pair")
            donor_id = self._sample_id(donor_pair)
            if donor_id == sample_id:
                raise ValueError("cross-item donor must differ from the recipient")
            recipient_targeting = pair.get("targeting")
            donor_targeting = donor_pair.get("targeting")
            if (
                not isinstance(recipient_targeting, str)
                or not recipient_targeting
                or not isinstance(donor_targeting, str)
                or not donor_targeting
            ):
                raise ValueError("cross-item pairs must have non-empty targeting labels")
            if donor_targeting != recipient_targeting:
                raise ValueError(
                    "cross-item donor and recipient targeting labels differ: "
                    f"{donor_targeting!r} != {recipient_targeting!r}"
                )
            donor_ids, donor_mask, donor_positions = self._tokenize_and_validate(
                donor_pair,
                side="clean",
            )
            if len(donor_positions) != len(edited_positions):
                raise ValueError(
                    "cross-item donor and recipient alignment cardinalities differ: "
                    f"{len(donor_positions)} != {len(edited_positions)}"
                )

        generations: dict[str, Any] = {}
        with self._torch.inference_mode():
            for control in controls:
                if control == "offset-2":
                    assert offset_plan is not None
                    donor_cache = self._capture(
                        input_ids=clean_ids,
                        attention_mask=clean_mask,
                        positions=offset_plan.source_positions,
                    )
                    destination_positions = offset_plan.destination_positions
                elif control == "cross-item":
                    assert donor_ids is not None
                    assert donor_mask is not None
                    assert donor_positions is not None
                    donor_cache = self._capture(
                        input_ids=donor_ids,
                        attention_mask=donor_mask,
                        positions=donor_positions,
                    )
                    destination_positions = edited_positions
                else:
                    donor_cache = self._capture(
                        input_ids=edited_ids,
                        attention_mask=edited_mask,
                        positions=edited_positions,
                    )
                    destination_positions = edited_positions

                generations[control] = self._generate(
                    input_ids=edited_ids,
                    attention_mask=edited_mask,
                    correct_answer=correct_answer,
                    patch=self._window_patch(
                        window=window,
                        positions=destination_positions,
                        donor_cache=donor_cache,
                    ),
                )

        return ControlScan(sample_id=sample_id, generations=generations)

    def provenance(self) -> dict[str, object]:
        """Return the shared runtime provenance with control-specific semantics."""

        payload = super().provenance()
        payload["operation"] = "patch-coordinate-controls"
        payload["runtime"] = "HuggingFaceCoordinateControlRuntime"
        generation = payload.get("generation")
        if isinstance(generation, dict):
            generation["patch_application"] = "all-window-layers-on-prompt-prefill-exactly-once"
        payload["coordinate_controls"] = {
            "offset-2": "same-item clean and edited coordinates shifted by +2 tokens",
            "cross-item": "different-item clean edited-word states at recipient coordinates",
            "self-copy": "recipient edited-word states copied onto identical coordinates",
            "correct_arm_source": "verified fixed-window reference run",
        }
        return payload
