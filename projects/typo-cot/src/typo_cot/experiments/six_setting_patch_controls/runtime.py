"""GPU runtime for manifest-planned six-setting patch controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from typo_cot.evaluation.generation import (
    classify_generated_token_ids,
    resolve_effective_eos_token_ids,
)
from typo_cot.experiments.fixed_window_answer_patching.runtime import (
    HuggingFaceFixedWindowAnswerPatchingRuntime,
)


@dataclass(frozen=True, slots=True)
class _RuntimeConfig:
    model: str
    benchmark: str
    gpu_id: str


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _positions(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty integer list")
    positions = tuple(value)
    if any(
        not isinstance(position, int) or isinstance(position, bool) or position < 0
        for position in positions
    ):
        raise ValueError(f"{field} contains an invalid token position")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{field} contains duplicate token positions")
    return positions


def _runtime_pair(record: Mapping[str, object]) -> dict[str, object]:
    """Adapt the normalized manifest record to the shared tokenizer checks."""

    edits = record.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("six-setting runtime record must contain aligned edits")
    aligned_words: list[dict[str, object]] = []
    for index, raw_edit in enumerate(edits):
        edit = _mapping(raw_edit, field=f"edits[{index}]")
        clean_span = edit.get("clean_char_span")
        typo_span = edit.get("typo_char_span")
        if (
            not isinstance(clean_span, list)
            or len(clean_span) != 2
            or not isinstance(typo_span, list)
            or len(typo_span) != 2
        ):
            raise ValueError("runtime edit character spans must contain two integers")
        aligned_words.append(
            {
                "clean_prompt_span": {"start": clean_span[0], "end": clean_span[1]},
                "edited_prompt_span": {"start": typo_span[0], "end": typo_span[1]},
                "clean_final_token": edit.get("clean_word_final_token"),
                "edited_final_token": edit.get("typo_word_final_token"),
            }
        )
    return {
        "sample_id": record.get("pair_id"),
        "gold_answer": record.get("gold_answer"),
        "targeting": record.get("target_rule"),
        "clean": {
            "prompt": record.get("clean_text"),
            "prompt_token_count": record.get("clean_prompt_token_count"),
        },
        "edited": {
            "prompt": record.get("typo_text"),
            "prompt_token_count": record.get("typo_prompt_token_count"),
        },
        "aligned_words": aligned_words,
    }


class HuggingFaceSixSettingPatchControlsRuntime(HuggingFaceFixedWindowAnswerPatchingRuntime):
    """Run offset and cross-item arms at manifest-specified coordinates."""

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        super().__init__(
            _RuntimeConfig(model=model, benchmark=task, gpu_id=gpu_id),  # type: ignore[arg-type]
            revision=revision,
        )
        self.effective_eos_token_ids, self.eos_source = resolve_effective_eos_token_ids(
            generation_config=self.model.generation_config,
            tokenizer=self.tokenizer,
            operation="six-setting-patch-controls",
        )

    def _generate_control(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        correct_answer: str,
        patch: Any,
        field: str,
    ) -> Any:
        from typo_cot.evaluation.fallback import extract_with_fallback
        from typo_cot.experiments.six_setting_patch_controls.runner import (
            ControlGeneration,
        )

        with patch:
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
                eos_token_id=list(self.effective_eos_token_ids),
            )
        prompt_length = int(input_ids.shape[1])
        if (
            not hasattr(output_ids, "ndim")
            or output_ids.ndim != 2
            or int(output_ids.shape[0]) != 1
            or int(output_ids.shape[1]) <= prompt_length
            or int(output_ids.shape[1]) > prompt_length + 512
            or not bool(output_ids[:, :prompt_length].equal(input_ids))
        ):
            raise ValueError(f"{field} generation returned an invalid capped sequence")
        raw = tuple(int(token) for token in output_ids[0, prompt_length:].detach().cpu().tolist())
        continuation, termination = classify_generated_token_ids(
            raw,
            effective_eos_token_ids=self.effective_eos_token_ids,
            max_new_tokens=512,
            field=field,
        )
        text = self.tokenizer.decode(
            list(continuation),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        extracted = extract_with_fallback(
            text,
            benchmark=self._extraction_benchmark,
            correct_answer=correct_answer,
            allow_positional=termination == "eos",
        )
        return ControlGeneration(
            token_ids=continuation,
            text=text,
            termination=termination,
            value=extracted.value,
            is_extracted=extracted.is_extracted,
            is_correct=extracted.is_correct,
            method=extracted.method,
            primary_method=extracted.primary_method,
        )

    def _window_patch(
        self,
        *,
        destination_positions: tuple[int, ...],
        donor_cache: list[Any],
    ) -> Any:
        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )

        layer_indices = tuple(range(0, 6))
        return PrefillBlockOutputWindowPatch(
            self.layers,
            layer_indices=layer_indices,
            positions=destination_positions,
            donor_values=tuple(donor_cache[index] for index in layer_indices),
        )

    def scan_controls(
        self,
        pair: Mapping[str, object],
        donor_pair: Mapping[str, object] | None,
        controls: tuple[str, ...],
    ) -> Mapping[str, Any]:
        """Generate exactly the requested diagnostic arms for one recipient."""

        from typo_cot.experiments.six_setting_patch_controls.runner import (
            ControlArmResult,
        )

        if not controls or set(controls) - {"offset-2", "cross-item"}:
            raise ValueError("runtime controls must be offset-2 and/or cross-item")
        if len(controls) != len(set(controls)):
            raise ValueError("runtime controls must not contain duplicates")
        if self.num_layers < 6:
            raise ValueError("six-setting controls require at least six decoder layers")
        recipient = _runtime_pair(pair)
        clean_ids, clean_mask, _clean_final = self._tokenize_and_validate(
            recipient,
            side="clean",
        )
        typo_ids, typo_mask, typo_final = self._tokenize_and_validate(
            recipient,
            side="edited",
        )
        gold_answer = self._gold_answer(recipient)
        plans = _mapping(pair.get("controls"), field="controls")
        results: dict[str, ControlArmResult] = {}
        with self._torch.inference_mode():
            for control in controls:
                if control == "offset-2":
                    offset = _mapping(plans.get("offset_2"), field="controls.offset_2")
                    if offset.get("valid") is not True:
                        raise ValueError("runtime received an invalid offset-2 plan")
                    source_positions = _positions(
                        offset.get("source_positions"),
                        field="controls.offset_2.source_positions",
                    )
                    destination_positions = _positions(
                        offset.get("destination_positions"),
                        field="controls.offset_2.destination_positions",
                    )
                    donor_ids, donor_mask = clean_ids, clean_mask
                else:
                    cross = _mapping(plans.get("cross_item"), field="controls.cross_item")
                    if cross.get("valid") is not True or donor_pair is None:
                        raise ValueError("runtime received an invalid cross-item plan")
                    if donor_pair.get("pair_id") != cross.get("donor_pair_id"):
                        raise ValueError("cross-item donor differs from the manifest plan")
                    donor = _runtime_pair(donor_pair)
                    donor_ids, donor_mask, source_positions = self._tokenize_and_validate(
                        donor,
                        side="clean",
                    )
                    destination_positions = typo_final
                if len(source_positions) != len(destination_positions):
                    raise ValueError(f"{control} coordinate cardinalities differ")
                donor_cache = self._capture(
                    input_ids=donor_ids,
                    attention_mask=donor_mask,
                    positions=source_positions,
                )
                generation = self._generate_control(
                    input_ids=typo_ids,
                    attention_mask=typo_mask,
                    correct_answer=gold_answer,
                    patch=self._window_patch(
                        destination_positions=destination_positions,
                        donor_cache=donor_cache,
                    ),
                    field=f"{pair.get('pair_id')}:{control}",
                )
                results[control] = ControlArmResult(
                    generation=generation,
                    source_positions=source_positions,
                    destination_positions=destination_positions,
                )
        return results

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload.update(
            {
                "operation": "six-setting-patch-controls",
                "runtime": "HuggingFaceSixSettingPatchControlsRuntime",
                "task": self.config.benchmark,
                "effective_eos_token_ids": list(self.effective_eos_token_ids),
                "effective_eos_token_ids_source": self.eos_source,
                "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
                "coordinate_source": "rebuttal-pair-manifest/v1",
                "layer_window": [0, 6],
                "diagnostic_controls": ["offset-2", "cross-item"],
                "answer_extraction": ("primary-then-empty-only-positional-only-after-eos/v1"),
            }
        )
        return payload


__all__ = ["HuggingFaceSixSettingPatchControlsRuntime"]
