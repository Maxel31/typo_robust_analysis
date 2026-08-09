"""Frozen final-PDF contract for Appendix C/Table 9."""

from __future__ import annotations

import hashlib
import json

from typo_cot.data.cohorts import (
    MODEL_SCALE_COHORT_ID,
    MODEL_SCALE_COHORT_SAMPLES_PER_SUBSET,
    MODEL_SCALE_COHORT_SELECTED_SAMPLE_COUNTS,
    MODEL_SCALE_COHORT_SELECTED_SAMPLE_IDS_SHA256,
    MODEL_SCALE_COHORT_SELECTION,
    MODEL_SCALE_COHORT_SAMPLE_IDS_SHA256,
)

EXPECTED_MODELS = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-72B-Instruct",
)
EXPECTED_SETTINGS = tuple((model, "mmlu") for model in EXPECTED_MODELS)
MODEL_LABELS = {
    "google/gemma-3-1b-it": "Gemma-3-1B",
    "google/gemma-3-4b-it": "Gemma-3-4B",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "google/gemma-3-27b-it": "Gemma-3-27B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
}
MODEL_SAMPLES_PER_SUBSET = dict(MODEL_SCALE_COHORT_SAMPLES_PER_SUBSET)
MODEL_SELECTED_SAMPLE_COUNTS = dict(MODEL_SCALE_COHORT_SELECTED_SAMPLE_COUNTS)
MODEL_SELECTED_SAMPLE_IDS_SHA256 = dict(MODEL_SCALE_COHORT_SELECTED_SAMPLE_IDS_SHA256)
if set(MODEL_SAMPLES_PER_SUBSET) != set(EXPECTED_MODELS):
    raise AssertionError("the Table 9 model grid and cohort coverage must match")

# Tuple order: n_s, Both, Question only, CoT only, restored, n_B.
PUBLISHED_REFERENCE = {
    "google/gemma-3-1b-it": (65, 19, 14, 12, 9, 19),
    "google/gemma-3-4b-it": (129, 32, 10, 28, 23, 32),
    "google/gemma-3-12b-it": (351, 41, 11, 29, 33, 41),
    "google/gemma-3-27b-it": (383, 33, 8, 36, 30, 33),
    "meta-llama/Llama-3.2-1B-Instruct": (119, 49, 24, 29, 30, 49),
    "meta-llama/Llama-3.2-3B-Instruct": (142, 36, 11, 32, 27, 36),
    "meta-llama/Llama-3.1-70B-Instruct": (411, 35, 2, 33, 33, 35),
    "mistralai/Mistral-7B-Instruct-v0.3": (137, 28, 8, 25, 24, 28),
    "Qwen/Qwen2.5-72B-Instruct": (331, 10, 7, 12, 8, 10),
}

COHORT_ID = MODEL_SCALE_COHORT_ID
COHORT_SAMPLE_COUNT = 500
COHORT_SAMPLE_IDS_SHA256 = MODEL_SCALE_COHORT_SAMPLE_IDS_SHA256
COHORT_SELECTION = MODEL_SCALE_COHORT_SELECTION

ANALYSIS_PROTOCOL: dict[str, object] = {
    "schema_version": "model-scale-cot-swap-protocol/v1",
    "paper_location": ["Appendix C", "Table 9"],
    "benchmark": "mmlu",
    "targeting": "attribution-4",
    "source_num_edits": 4,
    "models": list(EXPECTED_MODELS),
    "cohort": {
        "cohort_id": COHORT_ID,
        "sample_count": COHORT_SAMPLE_COUNT,
        "sample_ids_sha256": COHORT_SAMPLE_IDS_SHA256,
        "selection": COHORT_SELECTION,
        "intersection": "model-specific-final-paper-mmlu-source-cohort/v1",
        "model_selected_sample_counts": dict(MODEL_SELECTED_SAMPLE_COUNTS),
        "model_selected_sample_ids_sha256": dict(MODEL_SELECTED_SAMPLE_IDS_SHA256),
    },
    "change_denominator": "executed-regenerated-a-correct-per-model",
    "both_event": "b-not-equal-a",
    "question_only_event": "c-not-equal-a",
    "cot_only_event": "d-not-equal-a",
    "restoration_denominator": "regenerated-a-correct-and-b-not-equal-a-per-model",
    "restoration_event": "c-equals-a",
    "inference": "descriptive-integer-counts-only",
    "qwen_72b_interpretation": "directional-only-n_b-10",
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


def published_reference_payload() -> dict[str, dict[str, int]]:
    """Return the final-PDF integers with named fields."""
    return {
        model: {
            "n_s": values[0],
            "both": values[1],
            "question_only": values[2],
            "cot_only": values[3],
            "restored": values[4],
            "n_b": values[5],
        }
        for model, values in PUBLISHED_REFERENCE.items()
    }


__all__ = [
    "ANALYSIS_PROTOCOL",
    "ANALYSIS_PROTOCOL_SHA256",
    "COHORT_ID",
    "COHORT_SAMPLE_COUNT",
    "COHORT_SAMPLE_IDS_SHA256",
    "COHORT_SELECTION",
    "EXPECTED_MODELS",
    "EXPECTED_SETTINGS",
    "MODEL_LABELS",
    "MODEL_SAMPLES_PER_SUBSET",
    "MODEL_SELECTED_SAMPLE_COUNTS",
    "PUBLISHED_REFERENCE",
    "published_reference_payload",
]
