"""GPU runtime for first-, final-, and all-subword answer patching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from typo_cot.evaluation.generation import resolve_effective_eos_token_ids
from typo_cot.experiments.rebuttal_runtime import manifest_runtime_pair, require_mapping
from typo_cot.experiments.six_setting_patch_controls.runtime import (
    HuggingFaceSixSettingPatchControlsRuntime,
)
from typo_cot.experiments.subword_position_patching.planning import plan_subword_patch


class HuggingFaceSubwordPositionPatchingRuntime(HuggingFaceSixSettingPatchControlsRuntime):
    """Transfer clean residual states at frozen within-word coordinates."""

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        super().__init__(model=model, task=task, revision=revision, gpu_id=gpu_id)
        self.effective_eos_token_ids, self.eos_source = resolve_effective_eos_token_ids(
            generation_config=self.model.generation_config,
            tokenizer=self.tokenizer,
            operation="subword-position-patching",
        )
        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )

        self._patch_type = PrefillBlockOutputWindowPatch

    def _validated_word_tokens(
        self, pair: Mapping[str, object], side: str
    ) -> tuple[tuple[int, ...], ...]:
        if side not in {"clean", "typo"}:
            raise ValueError("subword token validation side must be clean or typo")
        text_field = "clean_text" if side == "clean" else "typo_text"
        count_field = "clean_prompt_token_count" if side == "clean" else "typo_prompt_token_count"
        span_field = "clean_char_span" if side == "clean" else "typo_char_span"
        token_field = "clean_token_indices" if side == "clean" else "typo_token_indices"
        final_field = "clean_word_final_token" if side == "clean" else "typo_word_final_token"
        prompt = pair.get(text_field)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{text_field} must be non-empty")
        try:
            encoded = self.tokenizer(
                prompt,
                add_special_tokens=True,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
        except (NotImplementedError, TypeError) as exc:
            raise ValueError("subword patching requires tokenizer offset mappings") from exc
        input_ids = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        if (
            input_ids.ndim != 2
            or int(input_ids.shape[0]) != 1
            or int(input_ids.shape[1]) != pair.get(count_field)
            or offsets is None
        ):
            raise ValueError(f"runtime {side} tokenization differs from the manifest")
        raw_offsets = offsets[0].tolist()
        edits = pair.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("subword patching requires aligned edits")
        words: list[tuple[int, ...]] = []
        for index, raw_edit in enumerate(edits):
            edit = require_mapping(raw_edit, field=f"edits[{index}]")
            span = edit.get(span_field)
            stored = edit.get(token_field)
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
                or not 0 <= span[0] < span[1] <= len(prompt)
                or not isinstance(stored, list)
                or not stored
            ):
                raise ValueError(f"edits[{index}] {side} token/span fields are invalid")
            overlapping = tuple(
                token_index
                for token_index, (raw_start, raw_end) in enumerate(raw_offsets)
                if int(raw_end) > int(raw_start)
                and int(raw_start) < span[1]
                and span[0] < int(raw_end)
            )
            if tuple(stored) != overlapping or edit.get(final_field) != overlapping[-1]:
                raise ValueError(
                    f"runtime {side} subword coordinates differ from tokenizer offsets "
                    f"for edits[{index}]"
                )
            words.append(overlapping)
        flattened = tuple(position for word in words for position in word)
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"runtime {side} edited-word subtoken coordinates overlap")
        return tuple(words)

    @staticmethod
    def _runtime_edits(
        clean_words: Sequence[Sequence[int]], typo_words: Sequence[Sequence[int]]
    ) -> tuple[dict[str, object], ...]:
        if len(clean_words) != len(typo_words) or not clean_words:
            raise ValueError("clean and typo edited-word inventories differ")
        return tuple(
            {
                "clean_token_indices": list(clean),
                "typo_token_indices": list(typo),
                "clean_word_final_token": clean[-1],
                "typo_word_final_token": typo[-1],
            }
            for clean, typo in zip(clean_words, typo_words, strict=True)
        )

    def _reindex_donor(self, values: Any, indices: tuple[int, ...]) -> Any:
        if hasattr(values, "index_select"):
            index = self._torch.tensor(indices, dtype=self._torch.long, device=values.device)
            return values.index_select(0, index)
        return [values[index] for index in indices]

    def _generate_subword(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        correct_answer: str,
        patch: Any,
        field: str,
    ) -> Any:
        from typo_cot.experiments.subword_position_patching.runner import SubwordGeneration

        generation = self._generate_control(
            input_ids=input_ids,
            attention_mask=attention_mask,
            correct_answer=correct_answer,
            patch=patch,
            field=field,
            allow_positional_after_length_cap=True,
        )
        return SubwordGeneration(
            token_ids=generation.token_ids,
            text=generation.text,
            termination=generation.termination,
            value=generation.value,
            is_extracted=generation.is_extracted,
            is_correct=generation.is_correct,
            method=generation.method,
            primary_method=generation.primary_method,
        )

    def scan_pair(self, pair: Mapping[str, object], *, modes: tuple[str, ...]) -> Any:
        from typo_cot.experiments.subword_position_patching.runner import (
            SubwordModeScan,
            SubwordPairScan,
        )

        if (
            not modes
            or len(modes) != len(set(modes))
            or set(modes)
            - {
                "first",
                "final",
                "all",
            }
        ):
            raise ValueError("subword runtime modes must be unique first/final/all labels")
        if self.num_layers < 6:
            raise ValueError("subword-position patching requires at least six decoder layers")
        cohorts = require_mapping(pair.get("cohorts"), field="cohorts")
        if (
            cohorts.get("restoration") is not True
            or pair.get("clean_correct") is not True
            or pair.get("typo_correct") is not False
        ):
            raise ValueError("subword runtime requires a clean-correct/typo-wrong pair")
        runtime_pair = manifest_runtime_pair(pair)
        clean_ids, clean_mask, clean_final = self._tokenize_and_validate(runtime_pair, side="clean")
        typo_ids, typo_mask, typo_final = self._tokenize_and_validate(runtime_pair, side="edited")
        clean_words = self._validated_word_tokens(pair, "clean")
        typo_words = self._validated_word_tokens(pair, "typo")
        if (
            tuple(word[-1] for word in clean_words) != clean_final
            or tuple(word[-1] for word in typo_words) != typo_final
        ):
            raise ValueError("subword and word-final tokenizer validations disagree")
        edits = self._runtime_edits(clean_words, typo_words)
        plans = {mode: plan_subword_patch(edits, mode=mode) for mode in modes}
        capture_positions = tuple(position for word in clean_words for position in word)
        source_index = {position: index for index, position in enumerate(capture_positions)}
        layer_indices = tuple(range(6))
        correct_answer = self._gold_answer(runtime_pair)
        scans: dict[str, SubwordModeScan] = {}
        with self._torch.inference_mode():
            donor_cache = self._capture(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                positions=capture_positions,
            )
            for mode in modes:
                plan = plans[mode]
                row_indices = tuple(source_index[position] for position in plan.source_positions)
                patch = self._patch_type(
                    self.layers,
                    layer_indices=layer_indices,
                    positions=plan.destination_positions,
                    donor_values=tuple(
                        self._reindex_donor(donor_cache[layer], row_indices)
                        for layer in layer_indices
                    ),
                )
                generation = self._generate_subword(
                    input_ids=typo_ids,
                    attention_mask=typo_mask,
                    correct_answer=correct_answer,
                    patch=patch,
                    field=f"{pair.get('pair_id')}:{mode}",
                )
                scans[mode] = SubwordModeScan(
                    generation=generation,
                    source_positions=plan.source_positions,
                    destination_positions=plan.destination_positions,
                )
        return SubwordPairScan(scans=scans)

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload.update(
            {
                "operation": "subword-position-patching",
                "runtime": "HuggingFaceSubwordPositionPatchingRuntime",
                "task": self.config.benchmark,
                "direction": "clean-to-typo",
                "coordinate_source": "manifest-edit-token-indices-retokenized/v1",
                "layer_window": [0, 6],
                "modes": ["first", "final", "all"],
                "token_count_policy": "equal-count-primary",
                "secondary_alignment": ("nearest-normalized-position-half-up-endpoints/v1"),
                "effective_eos_token_ids": list(self.effective_eos_token_ids),
                "effective_eos_token_ids_source": self.eos_source,
                "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
                "answer_extraction": "primary-then-empty-only-positional/v1",
            }
        )
        payload.pop("diagnostic_controls", None)
        generation = dict(payload["generation"])  # type: ignore[arg-type]
        generation["patch_application"] = "layers-0-5-on-typo-prompt-prefill-exactly-once/v1"
        payload["generation"] = generation
        return payload


__all__ = ["HuggingFaceSubwordPositionPatchingRuntime"]
