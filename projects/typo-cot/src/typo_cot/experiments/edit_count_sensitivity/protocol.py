"""Frozen final-PDF contract for Appendix C/Table 8."""

from __future__ import annotations

import hashlib
import json

EDIT_COUNTS = (1, 2, 4)
ACCURACY_BENCHMARKS = ("arc", "csqa", "gsm8k", "math-500", "mmlu", "mmlu-pro")
_FULL_ACCURACY_MODELS = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
EXPECTED_ACCURACY_SETTINGS = tuple(
    (model, benchmark) for model in _FULL_ACCURACY_MODELS for benchmark in ACCURACY_BENCHMARKS
) + tuple(("Qwen/Qwen2.5-3B-Instruct", benchmark) for benchmark in ("gsm8k", "mmlu", "mmlu-pro"))
EXPECTED_ACCURACY_SETTING_COUNT = 51
if len(EXPECTED_ACCURACY_SETTINGS) != EXPECTED_ACCURACY_SETTING_COUNT:
    raise AssertionError("the recovered Table 8 accuracy grid must contain exactly 51 settings")
RESTORATION_MODELS = (
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
RESTORATION_BENCHMARKS = ("gsm8k", "mmlu")
EXPECTED_RESTORATION_SETTINGS = tuple(
    (model, benchmark) for model in RESTORATION_MODELS for benchmark in RESTORATION_BENCHMARKS
)

_SETTING_LABELS = {
    "google/gemma-3-4b-it": "Gemma / {benchmark}",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama / {benchmark}",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B / {benchmark}",
}


def restoration_setting_label(model: str, benchmark: str) -> str:
    """Return the literal compact label used by the submitted Table 8."""
    return _SETTING_LABELS[model].format(benchmark=benchmark.upper())


PUBLISHED_REFERENCE: dict[str, object] = {
    "source": "final-pdf-appendix-c-table-8",
    "accuracy": {
        "equal_setting_mean": {"0": 0.521, "1": 0.500, "2": 0.483, "4": 0.460},
        "matched_81812_items": {"0": 0.546, "1": 0.525, "2": 0.509, "4": 0.488},
        "clean_above_four_settings": {"numerator": 51, "denominator": 51},
    },
    "restoration_settings": {
        "google/gemma-3-4b-it::gsm8k": {
            "label": "Gemma / GSM8K",
            "1": {"restored": 55, "denominator": 57, "rate": 55 / 57},
            "2": {"restored": 63, "denominator": 65, "rate": 63 / 65},
            "4": {"restored": 95, "denominator": 102, "rate": 95 / 102},
        },
        "google/gemma-3-4b-it::mmlu": {
            "label": "Gemma / MMLU",
            "1": {"restored": 169, "denominator": 202, "rate": 169 / 202},
            "2": {"restored": 203, "denominator": 249, "rate": 203 / 249},
            "4": {"restored": 233, "denominator": 301, "rate": 233 / 301},
        },
        "meta-llama/Llama-3.2-3B-Instruct::gsm8k": {
            "label": "Llama / GSM8K",
            "1": {"restored": 96, "denominator": 97, "rate": 96 / 97},
            "2": {"restored": 124, "denominator": 129, "rate": 124 / 129},
            "4": {"restored": 165, "denominator": 167, "rate": 165 / 167},
        },
        "meta-llama/Llama-3.2-3B-Instruct::mmlu": {
            "label": "Llama / MMLU",
            "1": {"restored": 248, "denominator": 275, "rate": 248 / 275},
            "2": {"restored": 290, "denominator": 327, "rate": 290 / 327},
            "4": {"restored": 332, "denominator": 384, "rate": 332 / 384},
        },
        "mistralai/Mistral-7B-Instruct-v0.3::gsm8k": {
            "label": "Mistral-7B / GSM8K",
            "1": {"restored": 87, "denominator": 87, "rate": 1.0},
            "2": {"restored": 100, "denominator": 101, "rate": 100 / 101},
            "4": {"restored": 129, "denominator": 131, "rate": 129 / 131},
        },
        "mistralai/Mistral-7B-Instruct-v0.3::mmlu": {
            "label": "Mistral-7B / MMLU",
            "1": {"restored": 156, "denominator": 190, "rate": 156 / 190},
            "2": {"restored": 208, "denominator": 252, "rate": 208 / 252},
            "4": {"restored": 263, "denominator": 330, "rate": 263 / 330},
        },
    },
    "restoration_pooled": {
        "1": {"restored": 811, "denominator": 908, "rate": 811 / 908},
        "2": {"restored": 988, "denominator": 1123, "rate": 988 / 1123},
        "4": {"restored": 1217, "denominator": 1415, "rate": 1217 / 1415},
    },
    "notes": [
        "accuracy-uses-51-complete-settings-over-six-benchmarks",
        "matched-accuracy-intersects-81812-sample-ids-over-all-four-conditions",
        "restoration-is-undefined-at-zero-edits",
        "restoration-cohorts-differ-by-edit-count",
    ],
}

ANALYSIS_PROTOCOL: dict[str, object] = {
    "schema_version": "edit-count-sensitivity-protocol/v1",
    "paper_location": ["Appendix C", "Table 8"],
    "edit_counts": list(EDIT_COUNTS),
    "accuracy": {
        "source": "completed-unlimited-attribution-4-prepare-edited-pairs/v1",
        "setting_grid": [list(setting) for setting in EXPECTED_ACCURACY_SETTINGS],
        "setting_grid_provenance": (
            "final-pdf-count-and-benchmarks-with-setting-identities-recovered-from-"
            "submitted-table-source/v1"
        ),
        "clean_condition": "clean.answer.is_correct",
        "edited_condition": "edited.answer.is_correct",
        "equal_setting_mean": "unweighted-mean-of-full-setting-accuracies/v1",
        "matched_pool": "sample-id-intersection-over-clean-one-two-four/v1",
        "clean_consistency": "exact-clean-prompt-continuation-answer-and-gold/v1",
    },
    "restoration": {
        "source": "completed-unlimited-attribution-4-cot-swap/v1",
        "denominator": "regenerated-a-correct-and-b-not-equal-a-per-edit-count",
        "event": "c-equals-a",
        "pooling": "micro-sum-six-settings-separately-per-edit-count/v1",
        "zero_edits": "undefined",
    },
    "inference": "descriptive-integer-counts-only",
    "historical_identity": "not-claimed",
}
ANALYSIS_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        ANALYSIS_PROTOCOL,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


__all__ = [
    "ACCURACY_BENCHMARKS",
    "ANALYSIS_PROTOCOL",
    "ANALYSIS_PROTOCOL_SHA256",
    "EDIT_COUNTS",
    "EXPECTED_ACCURACY_SETTING_COUNT",
    "EXPECTED_ACCURACY_SETTINGS",
    "EXPECTED_RESTORATION_SETTINGS",
    "PUBLISHED_REFERENCE",
    "RESTORATION_BENCHMARKS",
    "RESTORATION_MODELS",
    "restoration_setting_label",
]
