"""Single source of truth for the clean-prefix generation protocol."""

from __future__ import annotations

BENCHMARK_DATASET_NAMES = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "arc": "arc",
}

GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 512,
    "use_cache": True,
    "return_dict_in_generate": False,
    "output_scores": False,
    "padding_side": "left",
    "batch_size": 1,
}

ANSWER_DECODING = "generated-token-ids-only/v1"
ANSWER_EXTRACTION = "primary-then-empty-only-fallback-cap-aware/v1"
PRE_ANSWER_BOUNDARY = "first-submitted-[Tt]he-answer-is/v1"
SELECTION_ALIGNMENT = "prompt-length-suffix-equality/v1"
SCAN_BOUNDARY = "edited-full-must-preserve-separately-tokenized-prompt/v1"
INPUT_ASSEMBLY = "edited-prompt-ids-plus-clean-cot-id-prefix/v1"
IMPLEMENTATION = "huggingface-clean-prefix-independent-point-generation/v1"
