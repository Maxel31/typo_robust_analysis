"""Single source of truth for the public CoT-swap runtime protocol."""

from __future__ import annotations

from typo_cot.experiments.cot_swap.planning import CELL_ORDER

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
