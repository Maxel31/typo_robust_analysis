"""Hugging Face GPU runtime for the Table 2 patch/text crossing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from typo_cot.experiments.fixed_window_answer_patching.runtime import (
    HuggingFaceFixedWindowAnswerPatchingRuntime,
)
from typo_cot.experiments.patch_text_combination.planning import (
    PRE_ANSWER_BOUNDARY_METHOD,
)

if TYPE_CHECKING:
    from typo_cot.experiments.fixed_window_answer_patching import LayerWindow
    from typo_cot.experiments.patch_text_combination.runner import (
        CompleteTextInputUse,
        CompleteTextScan,
        PatchTextCombinationConfig,
    )


class HuggingFacePatchTextCombinationRuntime(HuggingFaceFixedWindowAnswerPatchingRuntime):
    """Generate complete-text cells with and without the frozen patch."""

    def __init__(
        self,
        config: PatchTextCombinationConfig,
        *,
        revision: str,
    ) -> None:
        super().__init__(config, revision=revision)  # type: ignore[arg-type]

    @staticmethod
    def _record_positions(pair: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        aligned = pair.get("aligned_words")
        if not isinstance(aligned, list) or not aligned:
            raise ValueError("runtime received a pair without aligned words")
        clean_positions: list[int] = []
        edited_positions: list[int] = []
        for index, raw_word in enumerate(aligned):
            if not isinstance(raw_word, Mapping):
                raise TypeError(f"aligned_words[{index}] must be an object")
            clean = raw_word.get("clean_final_token")
            edited = raw_word.get("edited_final_token")
            if (
                not isinstance(clean, int)
                or isinstance(clean, bool)
                or clean < 0
                or not isinstance(edited, int)
                or isinstance(edited, bool)
                or edited < 0
            ):
                raise ValueError(f"aligned_words[{index}] has invalid final-token coordinates")
            clean_positions.append(clean)
            edited_positions.append(edited)
        return tuple(clean_positions), tuple(edited_positions)

    def _tokenize_complete_input(
        self,
        pair: Mapping[str, object],
        pre_answer_text: str,
        edited_prompt_ids: Any,
    ) -> tuple[Any, Any, CompleteTextInputUse]:
        from typo_cot.experiments.patch_text_combination.runner import CompleteTextInputUse

        edited = pair.get("edited")
        if not isinstance(edited, Mapping):
            raise TypeError("pair.edited must be an object")
        prompt = edited.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("pair.edited.prompt must be non-empty")
        if not isinstance(pre_answer_text, str):
            raise TypeError("pre_answer_text must be a string")
        encoded = self.tokenizer(
            prompt + pre_answer_text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("tokenizer must return one unpadded complete-text input")
        prompt_count = int(edited_prompt_ids.shape[1])
        full_count = int(input_ids.shape[1])
        if full_count < prompt_count:
            raise ValueError("complete-text input is shorter than the edited prompt")
        prompt_prefix = input_ids[:, :prompt_count]
        boundary_stable = bool(self._torch.equal(prompt_prefix.cpu(), edited_prompt_ids.cpu()))
        if not boundary_stable:
            raise ValueError(
                "tokenizing the edited prompt plus complete text changed the edited "
                "prompt-token prefix"
            )
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
            raise ValueError("tokenizer must return one fully attended complete-text input")
        full_token_ids = [int(value) for value in input_ids[0].tolist()]
        serialized_ids = json.dumps(full_token_ids, separators=(",", ":")).encode("utf-8")
        clean_positions, edited_positions = self._record_positions(pair)
        use = CompleteTextInputUse(
            pre_answer_text_sha256=hashlib.sha256(pre_answer_text.encode("utf-8")).hexdigest(),
            pre_answer_char_count=len(pre_answer_text),
            pre_answer_token_count=full_count - prompt_count,
            edited_prompt_token_count=prompt_count,
            full_input_token_count=full_count,
            full_input_ids_sha256=hashlib.sha256(serialized_ids).hexdigest(),
            clean_positions=clean_positions,
            edited_positions=edited_positions,
            boundary_stable=True,
        )
        return input_ids.to(self.device), attention_mask.to(self.device), use

    def _window_patch(
        self,
        *,
        window: LayerWindow,
        positions: tuple[int, ...],
        donor_cache: list[Any],
        recipient_ids: Any,
    ) -> Any:
        del recipient_ids  # Included in the seam so tests can audit the recipient input.
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

    def scan_complete_text(
        self,
        pair: dict[str, object],
        pre_answer_text: str,
        window: LayerWindow,
    ) -> CompleteTextScan:
        from typo_cot.experiments.patch_text_combination.runner import CompleteTextScan

        sample_id = self._sample_id(pair)
        correct_answer = self._gold_answer(pair)
        clean_ids, clean_mask, clean_positions = self._tokenize_and_validate(pair, side="clean")
        edited_ids, _, edited_positions = self._tokenize_and_validate(pair, side="edited")
        if len(clean_positions) != len(edited_positions):
            raise ValueError("clean and edited alignment cardinalities differ")
        if not 0 <= window.start < window.stop <= self.num_layers:
            raise ValueError(f"layer window {window.label} is outside [0, {self.num_layers})")
        full_ids, full_mask, input_use = self._tokenize_complete_input(
            pair,
            pre_answer_text,
            edited_ids,
        )
        if (
            input_use.clean_positions != clean_positions
            or input_use.edited_positions != edited_positions
        ):
            raise ValueError("complete-text input coordinates differ from runtime alignment")

        with self._torch.inference_mode():
            donor_cache = self._capture(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                positions=clean_positions,
            )
            no_patch = self._generate(
                input_ids=full_ids,
                attention_mask=full_mask,
                correct_answer=correct_answer,
            )
            patch = self._window_patch(
                window=window,
                positions=edited_positions,
                donor_cache=donor_cache,
                recipient_ids=full_ids,
            )
            fixed_window_patch = self._generate(
                input_ids=full_ids,
                attention_mask=full_mask,
                correct_answer=correct_answer,
                patch=patch,
            )
        return CompleteTextScan(
            sample_id=sample_id,
            input_use=input_use,
            no_patch=no_patch,
            fixed_window_patch=fixed_window_patch,
        )

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload.update(
            {
                "operation": "patch-text-combination",
                "runtime": "HuggingFacePatchTextCombinationRuntime",
                "text_intervention": {
                    "source": "prepared-clean-continuation",
                    "boundary": PRE_ANSWER_BOUNDARY_METHOD,
                    "recipient": "edited-prompt-plus-complete-clean-pre-answer-text",
                    "donor": "clean-question-only-prompt",
                    "tokenization": "single-concatenated-text-call-with-prompt-prefix-check",
                },
            }
        )
        return payload
