"""Single source of truth for the public CoT-swap runtime protocol."""

from __future__ import annotations

import copy
import hashlib
import json

from typo_cot.experiments.cot_swap.planning import CELL_ORDER, CELL_SIDES

BENCHMARK_DATASET_NAMES = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}
GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 16,
    "use_cache": True,
    "return_dict_in_generate": False,
    "output_scores": False,
    "padding_side": "left",
}
TEXT_INTERVENTION = {
    "boundary": "submitted-first-[Tt]he-answer-is-filter/v1",
    "assembly": "recorded-prompt-plus-decoded-pre-answer-text-retokenized/v1",
}
ANSWER_EXTRACTION = "primary-then-empty-only-fallback-symmetric-a-b-c-d-cap-aware/v2"
IMPLEMENTATION = "huggingface-cot-swap-four-cell-batch/v1"
BATCHING = {
    "policy": "one-pair-four-cells/v1",
    "batch_size": 4,
    "cell_order": list(CELL_ORDER),
}
ANSWER_SPAN_DECODING = {
    "source": "generated-token-ids-only/v1",
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": False,
}
EDIT_VALIDITY = {
    "policy": "stored-prompts-differ-and-positive-target-attempts/v1",
    "requires_prompt_difference": True,
    "requires_positive_target_attempts": True,
    "zero_edit_restoration": "undefined-excluded-before-template-filter",
}
ANSWER_EXTRACTION_DETAIL = {
    "primary_precedence": "task-extractor-preserve-nonempty/v1",
    "fallback_invocation": "empty-primary-only-symmetric-a-b-c-d/v1",
    "max_token_cap_gate": "disable-positional-numeric-n4-n5-only/v1",
    "regex_and_cap_gate_source": "legacy-backed-detail-not-specified-by-final-pdf",
}
PROGRESS_PERSISTENCE = "atomic-checkpoints-power-of-two-manifest-flush/v1"
PROTOCOL_BASE = {
    "schema_version": "cot-swap-protocol/v1",
    "source": "completed-unlimited-prepare-edited-pairs-v1",
    "edit_validity": dict(EDIT_VALIDITY),
    "template_filter": {
        "policy": "submitted-first-[Tt]he-answer-is-filter/v1",
        "implementation_source": "legacy-backed-detail-not-specified-by-final-pdf",
        "requires_both_sides": True,
        "requires_one_trigger": True,
        "early_trigger_ratio_exclusive": 0.25,
        "rejects_residual_answer_fragment": True,
    },
    "cells": {cell: list(CELL_SIDES[cell]) for cell in CELL_ORDER},
    "implementation": IMPLEMENTATION,
    "batching": dict(BATCHING),
    "answer_span_decoding": dict(ANSWER_SPAN_DECODING),
    "generation": dict(GENERATION),
    "answer_extraction": ANSWER_EXTRACTION,
    "answer_extraction_detail": dict(ANSWER_EXTRACTION_DETAIL),
    "progress_persistence": PROGRESS_PERSISTENCE,
    "change_denominator": (
        "edit-valid-template-eligible-successfully-executed-regenerated-a-correct"
    ),
    "both_changed": "canonical-b-does-not-equal-canonical-a",
    "question_only_changed": "canonical-c-does-not-equal-canonical-a",
    "cot_only_changed": "canonical-d-does-not-equal-canonical-a",
    "restoration": "canonical-c-equals-canonical-a-conditioned-on-b-not-equal-a",
    "unextractable": "non-equality-failure-retained-in-applicable-denominator",
    "analysis": "descriptive-four-cell-counts-only",
    "historical_conflict": (
        "final-pdf-requires-symmetric-fallback-but-printed-headline-kept-legacy-a-correct"
    ),
}


def protocol_for(source_num_edits: int) -> dict[str, object]:
    """Return the public CoT-swap protocol bound to a prepared edit count."""
    if source_num_edits not in {1, 2, 4} or isinstance(source_num_edits, bool):
        raise ValueError("source_num_edits must be one of 1, 2, or 4")
    protocol = copy.deepcopy(PROTOCOL_BASE)
    protocol["source_generation"] = {
        "seed": 42,
        "num_edits_requested": source_num_edits,
        "max_new_tokens": 512,
    }
    return protocol


def protocol_sha256_for(source_num_edits: int) -> str:
    """Fingerprint the canonical public protocol for one edit count."""
    return hashlib.sha256(
        json.dumps(
            protocol_for(source_num_edits),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


PROTOCOL = protocol_for(4)
