"""Hugging Face GPU runtime for the correct-answer patch harm audit."""

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
from typo_cot.experiments.rebuttal_runtime import (
    manifest_runtime_pair,
    manifest_token_positions,
    require_mapping,
)


@dataclass(frozen=True, slots=True)
class _RuntimeConfig:
    model: str
    benchmark: str
    gpu_id: str


class HuggingFacePatchHarmAuditRuntime(HuggingFaceFixedWindowAnswerPatchingRuntime):
    """Patch clean edited-word states into typo-correct recipient prompts."""

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        super().__init__(
            _RuntimeConfig(model=model, benchmark=task, gpu_id=gpu_id),  # type: ignore[arg-type]
            revision=revision,
        )
        self.effective_eos_token_ids, self.eos_source = resolve_effective_eos_token_ids(
            generation_config=self.model.generation_config,
            tokenizer=self.tokenizer,
            operation="patch-harm-audit",
        )

    def _generate_harm(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        correct_answer: str,
        patch: Any,
        field: str,
    ) -> Any:
        from typo_cot.evaluation.fallback import extract_with_fallback
        from typo_cot.experiments.patch_harm_audit.runner import PatchHarmGeneration

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
        extraction = extract_with_fallback(
            text,
            benchmark=self._extraction_benchmark,
            correct_answer=correct_answer,
            allow_positional=termination == "eos",
        )
        return PatchHarmGeneration(
            token_ids=continuation,
            text=text,
            termination=termination,
            value=extraction.value,
            is_extracted=extraction.is_extracted,
            is_correct=extraction.is_correct,
            method=extraction.method,
            primary_method=extraction.primary_method,
        )

    def scan_pair(self, pair: Mapping[str, object]) -> Any:
        """Run the frozen [0,6) correct-coordinate intervention for one pair."""

        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )
        from typo_cot.experiments.patch_harm_audit.runner import PatchHarmScan

        if self.num_layers < 6:
            raise ValueError("patch harm audit requires at least six decoder layers")
        cohorts = require_mapping(pair.get("cohorts"), field="cohorts")
        if (
            cohorts.get("harm") is not True
            or pair.get("clean_correct") is not True
            or pair.get("typo_correct") is not True
        ):
            raise ValueError("patch harm runtime requires a clean-correct/typo-correct pair")

        runtime_pair = manifest_runtime_pair(pair)
        clean_ids, clean_mask, clean_final = self._tokenize_and_validate(
            runtime_pair,
            side="clean",
        )
        typo_ids, typo_mask, typo_final = self._tokenize_and_validate(
            runtime_pair,
            side="edited",
        )
        controls = require_mapping(pair.get("controls"), field="controls")
        correct = require_mapping(controls.get("correct"), field="controls.correct")
        if correct.get("valid") is not True:
            raise ValueError("patch harm runtime requires a valid correct-coordinate plan")
        source_positions = manifest_token_positions(
            correct.get("source_positions"),
            field="controls.correct.source_positions",
        )
        destination_positions = manifest_token_positions(
            correct.get("destination_positions"),
            field="controls.correct.destination_positions",
        )
        if source_positions != clean_final or destination_positions != typo_final:
            raise ValueError("patch harm coordinates differ from tokenizer alignment")
        if len(source_positions) != len(destination_positions):
            raise ValueError("patch harm coordinate cardinalities differ")

        gold_answer = self._gold_answer(runtime_pair)
        layer_indices = tuple(range(6))
        with self._torch.inference_mode():
            donor_cache = self._capture(
                input_ids=clean_ids,
                attention_mask=clean_mask,
                positions=source_positions,
            )
            patch = PrefillBlockOutputWindowPatch(
                self.layers,
                layer_indices=layer_indices,
                positions=destination_positions,
                donor_values=tuple(donor_cache[index] for index in layer_indices),
            )
            generation = self._generate_harm(
                input_ids=typo_ids,
                attention_mask=typo_mask,
                correct_answer=gold_answer,
                patch=patch,
                field=f"{pair.get('pair_id')}:correct-coordinate",
            )
        return PatchHarmScan(
            generation=generation,
            source_positions=source_positions,
            destination_positions=destination_positions,
        )

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload.update(
            {
                "operation": "patch-harm-audit",
                "runtime": "HuggingFacePatchHarmAuditRuntime",
                "task": self.config.benchmark,
                "coordinate_source": "rebuttal-pair-manifest-correct/v1",
                "layer_window": [0, 6],
                "cohort": "clean-correct-typo-correct",
                "generated_arm": "correct-coordinate-clean-to-typo/v1",
                "baseline_source": ("manifest-stored-deterministically-reextracted-typo-answer/v1"),
                "effective_eos_token_ids": list(self.effective_eos_token_ids),
                "effective_eos_token_ids_source": self.eos_source,
                "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
                "answer_extraction": ("primary-then-empty-only-positional-by-termination/v1"),
            }
        )
        generation = dict(payload["generation"])  # type: ignore[arg-type]
        generation["patch_application"] = "layers-0-5-on-typo-prompt-prefill-exactly-once/v1"
        payload["generation"] = generation
        return payload


__all__ = ["HuggingFacePatchHarmAuditRuntime"]
