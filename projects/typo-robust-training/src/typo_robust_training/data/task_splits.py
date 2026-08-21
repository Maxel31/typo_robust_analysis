"""Canonical task-split inventories shared by data construction and evaluation."""

from __future__ import annotations


REASONING_TRAINING_SPLITS = {
    "gsm8k": frozenset({"train"}),
    "mmlu": frozenset({"auxiliary_train"}),
    "arc": frozenset({"train"}),
}

REASONING_DIAGNOSTIC_SPLITS = {
    "gsm8k": frozenset({"train"}),
    "mmlu": frozenset({"dev"}),
    "arc": frozenset({"train"}),
}

# The broad builder-held-out inventory predates the frozen confirmatory study and
# remains explicit here so it cannot silently diverge from the evaluation loader.
TRAINING_DATA_EVALUATION_SPLITS = {
    "gsm8k": frozenset({"test"}),
    "mmlu": frozenset({"validation", "test"}),
    "arc": frozenset({"validation", "test"}),
}

# The frozen study deliberately uses the benchmark test split where available.
FROZEN_TASK_EVALUATION_SPLITS = {
    "gsm8k": frozenset({"test"}),
    "mmlu": frozenset({"test"}),
    "arc": frozenset({"test"}),
    "mmlu_pro": frozenset({"test"}),
    "math_500": frozenset({"test"}),
    "commonsense_qa": frozenset({"validation"}),
}


__all__ = [
    "FROZEN_TASK_EVALUATION_SPLITS",
    "REASONING_DIAGNOSTIC_SPLITS",
    "REASONING_TRAINING_SPLITS",
    "TRAINING_DATA_EVALUATION_SPLITS",
]
