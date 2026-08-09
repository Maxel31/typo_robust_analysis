"""Frozen paper and legacy-backed protocol for the input-corrector audit."""

from __future__ import annotations

import hashlib
import json
from itertools import product

from typo_cot.experiments.input_corrector_audit.correctors import CORRECTOR_IDS

PAPER_MODELS = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
CORE_BENCHMARKS = ("gsm8k", "mmlu", "mmlu-pro", "arc", "csqa")
MATH_BENCHMARK = "math-500"
SUPPORTED_BENCHMARKS = (*CORE_BENCHMARKS, MATH_BENCHMARK)
MATH_DIAGNOSTIC_CORRECTORS = ("t5-large-spell", "qwen2.5-7b-instruct")
PAPER_BENCHMARK_ITEM_COUNTS = {
    "gsm8k": 1319,
    "mmlu": 2850,
    "mmlu-pro": 1400,
    "arc": 1172,
    "csqa": 1221,
    "math-500": 500,
}

CORE_SETTINGS = tuple(product(PAPER_MODELS, CORE_BENCHMARKS, CORRECTOR_IDS))
MATH_DIAGNOSTIC_SETTINGS = tuple(
    product(PAPER_MODELS, (MATH_BENCHMARK,), MATH_DIAGNOSTIC_CORRECTORS)
)

CORRECTOR_MODELS = {
    "pyspellchecker": {
        "implementation": "pyspellchecker==0.9.0",
        "revision": None,
        "revision_evidence": "paper-package-version",
    },
    "t5-large-spell": {
        "implementation": "ai-forever/T5-large-spell",
        "revision": "c32125039b7df52d4d7cda059a0f2f74c29e8e02",
        "revision_evidence": "public-reproduction-pin-not-recorded-by-submitted-run",
    },
    "qwen2.5-7b-instruct": {
        "implementation": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "revision_evidence": "public-reproduction-pin-not-recorded-by-submitted-run",
    },
}

SOURCE_PROTOCOL = {
    "schema": "prepare-edited-pairs/v1",
    "run_schema": "prepare-edited-pairs-run/v1",
    "status": "completed",
    "targeting": "attribution-4",
    "seed": 42,
    "num_edits": 4,
    "max_new_tokens": 512,
    "limit": None,
    "records_per_model_benchmark": PAPER_BENCHMARK_ITEM_COUNTS,
}

WORD_METRIC = {
    "version": "submitted-whitespace-sequencematcher-equal-replace/v1",
    "tokenization": "str.split-whitespace",
    "alignment": "difflib.SequenceMatcher-autojunk-false",
    "aligned_changes": "equal-length-replace-spans-only",
    "setting_metric": "exact-clean-word-restorations/aligned-edited-words",
    "table_metric": "unweighted-mean-of-25-setting-rates",
}

EXACT_CLEAN = {
    "version": "full-prompt-utf8-byte-equality/v1",
    "normalization": "none",
}

SAME_BATCH = {
    "version": "adjacent-duplicate-prompt-pairs/v1",
    "pairs_per_batch": 2,
    "row_order": "p0,p0,p1,p1",
    "checkpoint_unit": "complete-generation-call",
}

GENERATION = {
    "strategy": "greedy",
    "dtype": "bfloat16",
    "padding_side": "left",
    "max_new_tokens": 512,
    "do_sample": False,
    "num_beams": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

PUBLISHED_REFERENCE = {
    "role": "descriptive-historical-reference-not-acceptance-target",
    "methods": {
        "pyspellchecker": {
            "paper_label": "Dictionary",
            "word": 0.663,
            "exact_clean": 7548,
            "same_changed": 0,
            "archive_changed": 708,
        },
        "t5-large-spell": {
            "paper_label": "T5-large",
            "word": 0.886,
            "exact_clean": 21306,
            "same_changed": 0,
            "archive_changed": 1874,
        },
        "qwen2.5-7b-instruct": {
            "paper_label": "Qwen2.5",
            "word": 0.734,
            "exact_clean": 16787,
            "same_changed": 0,
            "archive_changed": 1780,
        },
    },
    "total": {
        "exact_clean": 45641,
        "same_changed": 0,
        "archive_changed": 4362,
    },
    "math_intact_word_changes_per_item": {
        "t5-large-spell": 2.562,
        "qwen2.5-7b-instruct": 0.182,
    },
}

PROTOCOL = {
    "version": "input-corrector-audit-protocol/v1",
    "paper_defined": {
        "models": list(PAPER_MODELS),
        "core_benchmarks": list(CORE_BENCHMARKS),
        "correctors": list(CORRECTOR_IDS),
        "word_metric": {
            "setting_metric": "exact-restored-edited-words/edited-words",
            "table_metric": "unweighted-mean-of-25-setting-rates",
        },
        "exact_clean": EXACT_CLEAN,
        "same_batch": {
            "comparison": "duplicated-byte-identical-prompts-within-one-batch",
        },
    },
    "legacy_backed": {
        "source": SOURCE_PROTOCOL,
        "corrector_models": CORRECTOR_MODELS,
        "generation": GENERATION,
        "word_metric_implementation": WORD_METRIC,
        "same_batch_implementation": SAME_BATCH,
        "math_diagnostic_correctors": list(MATH_DIAGNOSTIC_CORRECTORS),
        "archive_endpoint": "submitted-corrected-generation-vs-separate-archived-clean-run",
    },
}


def canonical_sha256(payload: object) -> str:
    """Hash finite canonical JSON for protocol and checkpoint identities."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)
