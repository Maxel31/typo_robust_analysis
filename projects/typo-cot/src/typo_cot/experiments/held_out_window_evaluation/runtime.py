"""GPU runtime for diagnostic window scoring and held-out answer generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typo_cot.evaluation.generation import resolve_effective_eos_token_ids
from typo_cot.experiments.multitoken_kl_readout.runtime import (
    CleanContinuationTooShort,
    CleanPromptPrefixMismatch,
    clean_continuation_target_ids,
)
from typo_cot.experiments.rebuttal_runtime import manifest_runtime_pair
from typo_cot.experiments.six_setting_patch_controls.runtime import (
    HuggingFaceSixSettingPatchControlsRuntime,
)


class HuggingFaceHeldOutWindowRuntime(HuggingFaceSixSettingPatchControlsRuntime):
    """Reuse one clean capture for frozen diagnostic or evaluation windows."""

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        super().__init__(model=model, task=task, revision=revision, gpu_id=gpu_id)
        self.effective_eos_token_ids, self.eos_source = resolve_effective_eos_token_ids(
            generation_config=self.model.generation_config,
            tokenizer=self.tokenizer,
            operation="held-out-window-evaluation",
        )
        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )

        self._patch_type = PrefillBlockOutputWindowPatch

    @staticmethod
    def _last_logits(output: Any) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("held-out diagnostic output must contain [1, sequence, vocabulary]")
        if int(logits.shape[1]) == 0 or int(logits.shape[2]) == 0:
            raise ValueError("held-out diagnostic logits must be non-empty")
        return logits[0, -1:, :].detach().float().cpu()

    @staticmethod
    def _windows(
        windows: Mapping[str, tuple[int, int]], *, num_layers: int
    ) -> dict[str, tuple[int, int]]:
        normalized = dict(windows)
        if set(normalized) != {"selected", "runner-up"}:
            raise ValueError("held-out evaluation windows must contain selected and runner-up")
        if len(set(normalized.values())) != 2:
            raise ValueError("held-out evaluation windows must be distinct")
        for arm, window in normalized.items():
            if (
                not isinstance(window, tuple)
                or len(window) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in window)
                or not 0 <= window[0] < window[1] <= num_layers
                or window[1] - window[0] != 6
            ):
                raise ValueError(f"held-out {arm} window is invalid")
        return normalized

    def scan_selection(
        self,
        pair: Mapping[str, object],
        *,
        candidate_windows: tuple[tuple[int, int], ...],
    ) -> Any:
        """Measure first-token KL restoration for every frozen candidate."""

        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )
        from typo_cot.experiments.layerwise_kl_patching.patching import capture_block_outputs
        from typo_cot.experiments.multitoken_kl_readout.metrics import kl_trajectory_from_logits
        from typo_cot.experiments.held_out_window_evaluation.runner import WindowSelectionScan

        if candidate_windows != ((0, 6), (6, 12), (12, 18), (18, 24), (22, 28)):
            raise ValueError("held-out diagnostic candidate windows differ from the protocol")
        if self.num_layers < max(stop for _start, stop in candidate_windows):
            raise ValueError("held-out diagnostic model has fewer than 28 decoder layers")
        clean_prompt = pair.get("clean_text")
        clean_continuation = pair.get("clean_continuation")
        if not isinstance(clean_prompt, str) or not isinstance(clean_continuation, str):
            raise ValueError("held-out manifest clean prompt/continuation fields differ")
        try:
            target_ids, target_text = clean_continuation_target_ids(
                self.tokenizer,
                clean_prompt=clean_prompt,
                clean_continuation=clean_continuation,
                count=1,
            )
        except CleanContinuationTooShort:
            return WindowSelectionScan(
                available=False,
                invalid_reason="clean_continuation_lt_1",
                target_token_id=None,
                target_token_text=None,
                untreated_kl=None,
                patched_kl=(),
            )
        except CleanPromptPrefixMismatch:
            return WindowSelectionScan(
                available=False,
                invalid_reason="clean_prompt_not_exact_token_prefix",
                target_token_id=None,
                target_token_text=None,
                untreated_kl=None,
                patched_kl=(),
            )
        runtime_pair = manifest_runtime_pair(pair)
        clean_ids, clean_mask, clean_positions = self._tokenize_and_validate(
            runtime_pair, side="clean"
        )
        typo_ids, typo_mask, typo_positions = self._tokenize_and_validate(
            runtime_pair, side="edited"
        )
        if len(clean_positions) != len(typo_positions):
            raise ValueError("held-out clean and typo coordinate cardinalities differ")
        clean_holder: dict[str, Any] = {}

        def clean_forward() -> Any:
            output = self.model(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                use_cache=False,
            )
            clean_holder["output"] = output
            return output

        with self._torch.inference_mode():
            donor_cache = capture_block_outputs(
                self.layers[:28],
                positions=clean_positions,
                forward=clean_forward,
            )
            clean_logits = self._last_logits(clean_holder.pop("output"))
            typo_output = self.model(
                input_ids=typo_ids,
                attention_mask=typo_mask,
                use_cache=False,
            )
            typo_logits = self._last_logits(typo_output)
            del typo_output
            untreated = kl_trajectory_from_logits(clean_logits, typo_logits)[0]
            patched_values: list[float] = []
            for start, stop in candidate_windows:
                with PrefillBlockOutputWindowPatch(
                    self.layers,
                    layer_indices=tuple(range(start, stop)),
                    positions=typo_positions,
                    donor_values=tuple(donor_cache[start:stop]),
                ):
                    patched_output = self.model(
                        input_ids=typo_ids,
                        attention_mask=typo_mask,
                        use_cache=False,
                    )
                patched_logits = self._last_logits(patched_output)
                del patched_output
                patched_values.append(kl_trajectory_from_logits(clean_logits, patched_logits)[0])
        return WindowSelectionScan(
            available=True,
            invalid_reason=None,
            target_token_id=target_ids[0],
            target_token_text=target_text[0],
            untreated_kl=untreated,
            patched_kl=tuple(patched_values),
        )

    def scan_evaluation(
        self,
        pair: Mapping[str, object],
        *,
        windows: Mapping[str, tuple[int, int]],
    ) -> Any:
        """Generate selected and runner-up arms after diagnostic selection commits."""

        from typo_cot.experiments.held_out_window_evaluation.runner import (
            HeldOutGeneration,
            WindowEvaluationScan,
        )

        normalized = self._windows(windows, num_layers=self.num_layers)
        runtime_pair = manifest_runtime_pair(pair)
        clean_ids, clean_mask, clean_positions = self._tokenize_and_validate(
            runtime_pair, side="clean"
        )
        typo_ids, typo_mask, typo_positions = self._tokenize_and_validate(
            runtime_pair, side="edited"
        )
        if len(clean_positions) != len(typo_positions):
            raise ValueError("held-out clean and typo coordinate cardinalities differ")
        correct_answer = pair.get("clean_answer")
        if not isinstance(correct_answer, str) or not correct_answer:
            raise ValueError("held-out manifest clean answer must be non-empty")
        generations: dict[str, HeldOutGeneration] = {}
        with self._torch.inference_mode():
            donor_cache = self._capture(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                positions=clean_positions,
            )
            for arm in ("selected", "runner-up"):
                start, stop = normalized[arm]
                generation = self._generate_control(
                    input_ids=typo_ids,
                    attention_mask=typo_mask,
                    correct_answer=correct_answer,
                    patch=self._patch_type(
                        self.layers,
                        layer_indices=tuple(range(start, stop)),
                        positions=typo_positions,
                        donor_values=tuple(donor_cache[start:stop]),
                    ),
                    field=f"{pair.get('pair_id')}:{arm}",
                    allow_positional_after_length_cap=True,
                )
                generations[arm] = HeldOutGeneration(
                    token_ids=generation.token_ids,
                    text=generation.text,
                    termination=generation.termination,
                    value=generation.value,
                    is_extracted=generation.is_extracted,
                    is_correct=generation.is_correct,
                    method=generation.method,
                    primary_method=generation.primary_method,
                )
        return WindowEvaluationScan(generations=generations)

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload.update(
            {
                "operation": "held-out-window-evaluation",
                "runtime": "HuggingFaceHeldOutWindowRuntime",
                "task": self.config.benchmark,
                "direction": "clean-to-typo",
                "coordinate_source": "manifest-edited-word-final-token/v1",
                "diagnostic_readout": "first-clean-continuation-token-distribution/v1",
                "candidate_windows": [[0, 6], [6, 12], [12, 18], [18, 24], [22, 28]],
                "effective_eos_token_ids": list(self.effective_eos_token_ids),
                "effective_eos_token_ids_source": self.eos_source,
                "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
                "answer_extraction": "primary-then-empty-only-positional/v1",
            }
        )
        payload.pop("diagnostic_controls", None)
        generation = dict(payload["generation"])  # type: ignore[arg-type]
        generation["patch_application"] = "selected-window-on-typo-prefill-exactly-once/v1"
        payload["generation"] = generation
        return payload


__all__ = ["HuggingFaceHeldOutWindowRuntime"]
