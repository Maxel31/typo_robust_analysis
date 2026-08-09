"""Single source of truth for the final-PDF answer-line deletion protocol."""

from __future__ import annotations

import typo_cot.experiments.cot_swap.protocol as cot_swap_protocol

ARM_ORDER = ("complete", "answer-line-deleted")
BENCHMARK_DATASET_NAMES = {
    "gsm8k": cot_swap_protocol.BENCHMARK_DATASET_NAMES["gsm8k"],
    "mmlu": cot_swap_protocol.BENCHMARK_DATASET_NAMES["mmlu"],
}

# Appendix A applies the same answer-span generation contract to the control.
# Keep the object identity so a future CoT-swap protocol change cannot silently
# leave this downstream operation behind.
GENERATION = cot_swap_protocol.GENERATION
ANSWER_SPAN_DECODING = cot_swap_protocol.ANSWER_SPAN_DECODING
ANSWER_EXTRACTION = "primary-then-empty-only-fallback-symmetric-complete-and-deleted-cap-aware/v1"
TEXT_INTERVENTION = {
    "source_boundary": cot_swap_protocol.TEXT_INTERVENTION["boundary"],
    "deletion": "submitted-final-nonempty-line/v1",
    "assembly": cot_swap_protocol.TEXT_INTERVENTION["assembly"],
}
IMPLEMENTATION = "huggingface-answer-line-deletion-two-arm-batch/v1"
BATCHING = {
    "policy": "one-pair-two-arms/v1",
    "batch_size": 2,
    "arm_order": list(ARM_ORDER),
}
