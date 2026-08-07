"""GPU/AttnLRP runtime for the paper's pair-preparation operation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from typo_cot.experiments.prepare_edited_pairs.protocol import (
    PairProtocolError,
    apply_paper_edits,
    build_aligned_words,
    eligible_candidates,
    order_candidates,
)

if TYPE_CHECKING:
    from typo_cot.experiments.prepare_edited_pairs.runner import PrepareEditedPairsConfig

_BENCHMARK_NAMES = {
    "gsm8k": "gsm8k",
    "math-500": "math",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}
_SAMPLES_PER_SUBSET = {"mmlu": 50, "mmlu_pro": 100}
_HISTORICAL_COMPATIBILITY_NOTES = [
    "stable-sha256-seeds-replace-process-random-python-hash",
    "mistral-attnlrp-rules-target-mistral-classes",
    "actual-word-final-alignment-replaces-token-substring-coordinates",
    "parenthesized-choice-markers-use-recorded-choice-boundary",
    "arc-numeric-answer-keys-normalized-to-prompt-letters",
]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def preload_provenance(
    config: PrepareEditedPairsConfig,
    *,
    torch_module: Any | None = None,
) -> dict[str, object]:
    """Collect environment and protocol identity without loading model weights."""
    if torch_module is None:
        import torch as torch_module

    gpu_names = (
        [
            torch_module.cuda.get_device_name(index)
            for index in range(torch_module.cuda.device_count())
        ]
        if torch_module.cuda.is_available()
        else []
    )
    return {
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "lxt": _package_version("lxt"),
        "datasets": _package_version("datasets"),
        "cuda": torch_module.version.cuda,
        "gpu_names": gpu_names,
        "model": config.model,
        "benchmark_dataset_loader": _BENCHMARK_NAMES[config.benchmark],
        "random_seed_algorithm": "sha256-first-64-bits/v1",
        "target_position": "maximum-logit-after-first-cot-token",
        "alignment": "actual-edited-word-final-token",
        "historical_compatibility_notes": list(_HISTORICAL_COMPATIBILITY_NOTES),
    }


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise PairProtocolError(
            "the selected tokenizer must provide exact offset_mapping for word-final alignment"
        ) from exc

    input_ids = encoded["input_ids"]
    raw_offsets = encoded.get("offset_mapping")
    if raw_offsets is None or len(input_ids) != len(raw_offsets):
        raise PairProtocolError("tokenizer returned incomplete offset_mapping")

    offsets: list[tuple[int, int]] = []
    for raw_start, raw_end in raw_offsets:
        start, end = int(raw_start), int(raw_end)
        # The paper run used AttnLRPAnalyzer's fallback locator, which lstrips
        # decoded token text before locating it. Fast tokenizers commonly attach
        # the preceding separator to a word token; trimming it here preserves the
        # reported item-initial target and edit spans while retaining exact ends.
        while start < end and text[start].isspace():
            start += 1
        while start < end and text[end - 1].isspace():
            end -= 1
        offsets.append((start, end))
    return list(input_ids), offsets


class HuggingFacePairPreparationRuntime:
    """Load one paper setting and produce schema-v1 pair records."""

    def __init__(self, config: PrepareEditedPairsConfig) -> None:
        import numpy as np
        import torch

        from typo_cot.evaluation.extractor import create_extractor
        from typo_cot.lrp.analyzer import create_analyzer
        from typo_cot.models.prompts import create_prompt_template
        from typo_cot.models.wrapper import create_model_wrapper

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.config = config
        self.internal_benchmark = _BENCHMARK_NAMES[config.benchmark]
        self.wrapper = create_model_wrapper(
            model_name=config.model,
            gpu_id=config.gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=True,
        )
        self.tokenizer = self.wrapper.tokenizer
        self.tokenizer.padding_side = "left"
        self.analyzer = create_analyzer(
            model=self.wrapper.model,
            tokenizer=self.tokenizer,
            top_k=None,
            device=self.wrapper.device,
        )
        self.template = create_prompt_template(self.internal_benchmark)
        self.extractor = create_extractor(self.internal_benchmark)
        self._torch = torch
        self._dataset_provenance: dict[str, object] = {}

    def load_samples(self, config: PrepareEditedPairsConfig) -> list[Any]:
        from typo_cot.data.loader import create_loader

        loader = create_loader(
            benchmark=self.internal_benchmark,
            samples_per_subset=_SAMPLES_PER_SUBSET.get(self.internal_benchmark, 50),
            seed=config.seed,
            num_samples=None,
        )
        samples = loader.load()
        fingerprint_rows = [
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "choices": sample.choices,
                "correct_answer": sample.correct_answer,
                "subset": sample.subset,
            }
            for sample in sorted(samples, key=lambda item: item.sample_id)
        ]
        serialized = json.dumps(
            fingerprint_rows,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._dataset_provenance = {
            "dataset_sample_count": len(samples),
            "dataset_records_sha256": hashlib.sha256(serialized).hexdigest(),
        }
        return samples

    def _prompt(self, sample: Any) -> tuple[Any, str]:
        if self.internal_benchmark in {"mmlu", "mmlu_pro", "arc", "commonsense_qa"}:
            result = self.template.generate(
                question=sample.question,
                choices=sample.choices,
                subject=sample.subset,
            )
        else:
            result = self.template.generate(question=sample.question, subject=sample.subset)
        return result, result.get_full_prompt()

    def _generate(self, prompt: str) -> tuple[list[int], list[int], str]:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.wrapper.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.wrapper.device)
        with self._torch.no_grad():
            output_ids = self.wrapper.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_length = input_ids.shape[1]
        generated_ids = output_ids[0, prompt_length:].tolist()
        if not generated_ids:
            raise PairProtocolError("model generated no continuation token")
        continuation = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return input_ids[0].tolist(), generated_ids, continuation

    def _relevance_after_first_cot(
        self, prompt_ids: list[int], generated_ids: list[int]
    ) -> list[float]:
        if not generated_ids:
            raise PairProtocolError("cannot attribute an empty clean continuation")
        input_ids = self._torch.tensor(
            [prompt_ids + generated_ids],
            dtype=self._torch.long,
            device=self.wrapper.device,
        )
        # The target position contains the first generated CoT token; its logits
        # are therefore the distribution immediately after that token. Keeping
        # the complete clean generation matches the reported AttnLRP pipeline.
        relevance = self.analyzer.compute_relevance(input_ids, target_position=len(prompt_ids))
        return [float(value) for value in relevance[: len(prompt_ids)].detach().cpu().tolist()]

    def _answer(self, continuation: str, correct_answer: str) -> dict[str, object]:
        extraction = self.extractor.extract(continuation)
        answer = extraction.extracted_answer
        return {
            "value": answer,
            "is_extracted": bool(answer),
            "is_correct": self.extractor.is_correct(answer, correct_answer),
            "method": extraction.extraction_method,
            "confidence": extraction.confidence,
        }

    def prepare_pair(self, sample: Any, config: PrepareEditedPairsConfig) -> dict[str, object]:
        prompt_result, clean_prompt = self._prompt(sample)
        clean_prompt_ids, clean_generated_ids, clean_continuation = self._generate(clean_prompt)
        token_ids, clean_offsets = _tokenize_with_offsets(self.tokenizer, clean_prompt)
        if token_ids != clean_prompt_ids:
            raise PairProtocolError("generation and alignment tokenization disagree")

        relevances = self._relevance_after_first_cot(clean_prompt_ids, clean_generated_ids)
        token_texts = [
            self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids
        ]
        editable_start = int(prompt_result.question_start_in_full)
        choice_list_start = int(prompt_result.question_end_in_full)
        editable_end = int(prompt_result.question_with_choices_end)
        candidates = eligible_candidates(
            prompt_text=clean_prompt,
            token_texts=token_texts,
            relevances=relevances,
            offsets=clean_offsets,
            editable_prompt_start=editable_start,
            editable_prompt_end=editable_end,
            choice_list_start=choice_list_start,
        )
        ordering = order_candidates(
            candidates,
            targeting=config.targeting,
            num_edits=config.num_edits,
            seed=config.seed,
            sample_id=sample.sample_id,
        )
        clean_editable = clean_prompt[editable_start:editable_end]
        application = apply_paper_edits(
            editable_text=clean_editable,
            editable_prompt_start=editable_start,
            candidate_order=ordering.candidates,
            num_edits=config.num_edits,
            seed=config.seed,
            sample_id=sample.sample_id,
        )
        if not application.attempts:
            raise PairProtocolError("no eligible character edit could be applied")

        edited_prompt = (
            clean_prompt[:editable_start] + application.edited_text + clean_prompt[editable_end:]
        )
        edited_prompt_ids, edited_generated_ids, edited_continuation = self._generate(edited_prompt)
        aligned_ids, edited_offsets = _tokenize_with_offsets(self.tokenizer, edited_prompt)
        if aligned_ids != edited_prompt_ids:
            raise PairProtocolError("edited generation and alignment tokenization disagree")
        aligned_words = build_aligned_words(
            clean_editable=clean_editable,
            edited_editable=application.edited_text,
            editable_prompt_start=editable_start,
            attempts=application.attempts,
            clean_token_offsets=clean_offsets,
            edited_token_offsets=edited_offsets,
        )
        if not aligned_words:
            raise PairProtocolError("edits produced no alignable changed word")

        clean_answer = self._answer(clean_continuation, sample.correct_answer)
        edited_answer = self._answer(edited_continuation, sample.correct_answer)
        return {
            "schema_version": "prepare-edited-pairs/v1",
            "sample_id": sample.sample_id,
            "model": config.model,
            "benchmark": config.benchmark,
            "targeting": config.targeting,
            "seed": config.seed,
            "num_edits_requested": config.num_edits,
            "num_candidates": len(candidates),
            "num_target_attempts": len(application.attempts),
            "num_aligned_words": len(aligned_words),
            "gold_answer": sample.correct_answer,
            "subset": sample.subset,
            "attribution_target": {
                "definition": "maximum-logit-after-first-cot-token",
                "position": len(clean_prompt_ids),
                "first_cot_token_id": clean_generated_ids[0],
                "first_cot_token_text": self.tokenizer.decode(
                    [clean_generated_ids[0]],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "context": "complete-clean-generation",
            },
            "clean": {
                "question": sample.question,
                "choices": sample.choices,
                "editable_text": clean_editable,
                "editable_prompt_span": {"start": editable_start, "end": editable_end},
                "prompt": clean_prompt,
                "prompt_token_count": len(clean_prompt_ids),
                "continuation": clean_continuation,
                "continuation_token_count": len(clean_generated_ids),
                "answer": clean_answer,
            },
            "edited": {
                "editable_text": application.edited_text,
                "editable_prompt_span": {
                    "start": editable_start,
                    "end": editable_start + len(application.edited_text),
                },
                "prompt": edited_prompt,
                "prompt_token_count": len(edited_prompt_ids),
                "continuation": edited_continuation,
                "continuation_token_count": len(edited_generated_ids),
                "answer": edited_answer,
            },
            "answer_changed": clean_answer["value"] != edited_answer["value"],
            "excluded_attribution_tokens": [asdict(item) for item in ordering.excluded_top],
            "target_attempts": [asdict(item) for item in application.attempts],
            "aligned_words": [asdict(item) for item in aligned_words],
        }

    def provenance(self) -> dict[str, object]:
        provenance = preload_provenance(self.config, torch_module=self._torch)
        provenance.update(
            {
                "model_revision": getattr(self.wrapper.model.config, "_commit_hash", None),
            }
        )
        provenance.update(self._dataset_provenance)
        return provenance
