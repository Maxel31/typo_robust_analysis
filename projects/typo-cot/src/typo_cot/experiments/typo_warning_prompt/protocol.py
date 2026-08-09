"""Frozen public protocol for the Appendix E typo-warning audit."""

from __future__ import annotations

import hashlib
import json

ARM_ORDER = ("without-warning", "with-warning")
PAPER_MODELS = (
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
PAPER_BENCHMARKS = ("gsm8k", "mmlu")
PAPER_GRID = tuple((model, benchmark) for model in PAPER_MODELS for benchmark in PAPER_BENCHMARKS)

WARNING_TEXT = (
    "IMPORTANT: The question may contain typos or misspelled words caused by "
    "keyboard errors. Before solving, first silently correct any obvious typos to "
    "recover the intended wording, then reason step by step over the corrected text."
)
PROMPT_MARKERS = {
    "gsm8k": "Now solve the following problem:",
    "mmlu": "Now answer the following question:",
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
}
BATCHING = {
    "policy": "same-warning-arm-sorted-samples/v1",
    "batch_size": 8,
    "arm_order": list(ARM_ORDER),
    "final_partial_batch": True,
    "checkpoint_unit": "sample-arm",
    "detail_source": "legacy-backed-not-printed-by-final-pdf",
}
ANSWER_SPAN_DECODING = {
    "source": "generated-token-ids-only/v1",
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": False,
}
ANSWER_EXTRACTION = "submitted-task-extractor-primary-only/v1"
IMPLEMENTATION = "huggingface-typo-warning-same-arm-batch/v1"
RUNTIME_ENVIRONMENT = {
    "runtime": "HuggingFaceWarningPromptRuntime",
    "minimum_python": [3, 12],
    "package_versions": {
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
    },
    "device": "single-visible-cuda-gpu/v1",
}
PROMPT_INTERVENTION = {
    "warning_text": WARNING_TEXT,
    "insertion": "immediately-before-unique-task-final-marker/v1",
    "markers": dict(PROMPT_MARKERS),
    "without_warning": "submitted-attribution-edited-prompt-byte-identical/v1",
    "with_warning": "warning-plus-two-newlines-before-marker/v1",
    "ambiguous_boundary": "fail-closed/v1",
    "question_text": "unchanged/v1",
    "detail_source": "legacy-backed-not-printed-by-final-pdf",
}
SELECTION = {
    "source": "bundled-output-free-submitted-input-edit-manifest/v1",
    "historical_algorithm": "sample-id-sort-python-random-42-shuffle-take-sort/v1",
    "seed": 42,
    "pairs_per_setting": 300,
    "evaluated_targeting": "attribution-4",
    "random_4_role": "historical-shared-id-selector-only",
    "detail_source": "legacy-backed-not-printed-by-final-pdf",
}
PROGRESS_PERSISTENCE = "atomic-per-sample-arm-checkpoints-and-manifest/v1"
COMPARABILITY_LIMITATIONS = (
    "exact-warning-text-model-grid-and-selection-are-legacy-backed-details",
    "model-revisions-are-recovered-from-the-submission-environment-not-recorded-by-the-producer",
    "input-edit-manifest-reconstructs-submitted-text-without-publishing-generated-outputs",
    "one-instruction-and-two-tasks-do-not-generalize-self-correction",
    "not-a-performance-comparison-with-activation-patching",
)
PROTOCOL = {
    "schema_version": "typo-warning-prompt-protocol/v2",
    "paper_scope": "appendix-e-one-instruction-two-task-accuracy-audit",
    "source": "bundled-output-free-submitted-input-edit-manifest/v1",
    "selection": dict(SELECTION),
    "arms": list(ARM_ORDER),
    "prompt_intervention": dict(PROMPT_INTERVENTION),
    "generation": dict(GENERATION),
    "batching": dict(BATCHING),
    "answer_span_decoding": dict(ANSWER_SPAN_DECODING),
    "answer_extraction": ANSWER_EXTRACTION,
    "implementation": IMPLEMENTATION,
    "runtime_environment": RUNTIME_ENVIRONMENT,
    "paired_test": {
        "method": "exact-two-sided-mcnemar-binomial/v1",
        "detail_source": "legacy-backed-not-printed-by-final-pdf",
    },
    "unextractable": "retain-as-incorrect-in-common-paired-denominator/v1",
    "progress_persistence": PROGRESS_PERSISTENCE,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        PROTOCOL,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

HISTORICAL_REFERENCE = {
    "role": "descriptive-reference-not-acceptance-target",
    "printed_result_source": "final-pdf-appendix-e",
    "pooled_by_benchmark": {
        "gsm8k": {
            "n": 900,
            "n_source": "legacy-backed-not-printed-by-final-pdf",
            "without_warning_accuracy": 0.601,
            "with_warning_accuracy": 0.541,
            "significant": True,
        },
        "mmlu": {
            "n": 900,
            "n_source": "legacy-backed-not-printed-by-final-pdf",
            "without_warning_accuracy": 0.576,
            "with_warning_accuracy": 0.562,
            "significant": False,
        },
    },
}
