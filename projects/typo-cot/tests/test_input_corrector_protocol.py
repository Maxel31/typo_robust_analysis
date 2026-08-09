"""Provenance boundaries for the Appendix E input-corrector protocol."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import product

import pytest

from typo_cot.experiments.input_corrector_audit.correctors import CORRECTOR_IDS
from typo_cot.experiments.input_corrector_audit.protocol import (
    CORE_BENCHMARKS,
    CORE_SETTINGS,
    CORRECTOR_MODELS,
    EXACT_CLEAN,
    GENERATION,
    MATH_BENCHMARK,
    MATH_DIAGNOSTIC_CORRECTORS,
    MATH_DIAGNOSTIC_SETTINGS,
    PAPER_BENCHMARK_ITEM_COUNTS,
    PAPER_MODELS,
    PROTOCOL,
    PROTOCOL_SHA256,
    PUBLISHED_REFERENCE,
    SAME_BATCH,
    SOURCE_PROTOCOL,
    SUPPORTED_BENCHMARKS,
    WORD_METRIC,
    canonical_sha256,
)


EXPECTED_MODELS = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
EXPECTED_CORE_BENCHMARKS = ("gsm8k", "mmlu", "mmlu-pro", "arc", "csqa")
EXPECTED_CORRECTORS = (
    "pyspellchecker",
    "t5-large-spell",
    "qwen2.5-7b-instruct",
)


def _independent_canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_public_names_and_complete_setting_grids_are_frozen() -> None:
    assert PAPER_MODELS == EXPECTED_MODELS
    assert CORE_BENCHMARKS == EXPECTED_CORE_BENCHMARKS
    assert MATH_BENCHMARK == "math-500"
    assert SUPPORTED_BENCHMARKS == (*EXPECTED_CORE_BENCHMARKS, "math-500")
    assert CORRECTOR_IDS == EXPECTED_CORRECTORS
    assert MATH_DIAGNOSTIC_CORRECTORS == (
        "t5-large-spell",
        "qwen2.5-7b-instruct",
    )

    expected_core = tuple(product(EXPECTED_MODELS, EXPECTED_CORE_BENCHMARKS, EXPECTED_CORRECTORS))
    expected_math = tuple(
        product(
            EXPECTED_MODELS,
            ("math-500",),
            ("t5-large-spell", "qwen2.5-7b-instruct"),
        )
    )
    assert CORE_SETTINGS == expected_core
    assert MATH_DIAGNOSTIC_SETTINGS == expected_math
    assert len(CORE_SETTINGS) == len(set(CORE_SETTINGS)) == 75
    assert len(MATH_DIAGNOSTIC_SETTINGS) == len(set(MATH_DIAGNOSTIC_SETTINGS)) == 10
    assert all(setting[2] != "pyspellchecker" for setting in MATH_DIAGNOSTIC_SETTINGS)


def test_public_corrector_implementations_and_reproduction_pins_are_frozen() -> None:
    assert CORRECTOR_MODELS == {
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


def test_pdf_defined_semantics_exclude_recovered_implementation_details() -> None:
    paper = PROTOCOL["paper_defined"]

    assert paper == {
        "models": list(EXPECTED_MODELS),
        "core_benchmarks": list(EXPECTED_CORE_BENCHMARKS),
        "correctors": list(EXPECTED_CORRECTORS),
        "word_metric": {
            "setting_metric": "exact-restored-edited-words/edited-words",
            "table_metric": "unweighted-mean-of-25-setting-rates",
        },
        "exact_clean": {
            "version": "full-prompt-utf8-byte-equality/v1",
            "normalization": "none",
        },
        "same_batch": {
            "comparison": "duplicated-byte-identical-prompts-within-one-batch",
        },
    }

    serialized = json.dumps(paper, ensure_ascii=False, sort_keys=True)
    for implementation_detail in (
        "SequenceMatcher",
        "str.split-whitespace",
        "pairs_per_batch",
        "p0,p0,p1,p1",
        "complete-generation-call",
    ):
        assert implementation_detail not in serialized


def test_recovered_word_alignment_and_batch_layout_are_legacy_backed() -> None:
    legacy = PROTOCOL["legacy_backed"]

    assert WORD_METRIC == {
        "version": "submitted-whitespace-sequencematcher-equal-replace/v1",
        "tokenization": "str.split-whitespace",
        "alignment": "difflib.SequenceMatcher-autojunk-false",
        "aligned_changes": "equal-length-replace-spans-only",
        "setting_metric": "exact-clean-word-restorations/aligned-edited-words",
        "table_metric": "unweighted-mean-of-25-setting-rates",
    }
    assert SAME_BATCH == {
        "version": "adjacent-duplicate-prompt-pairs/v1",
        "pairs_per_batch": 2,
        "row_order": "p0,p0,p1,p1",
        "checkpoint_unit": "complete-generation-call",
    }
    assert legacy["word_metric_implementation"] == WORD_METRIC
    assert legacy["same_batch_implementation"] == SAME_BATCH

    serialized = json.dumps(legacy, ensure_ascii=False, sort_keys=True)
    for recovered_detail in (
        "SequenceMatcher",
        "str.split-whitespace",
        "pairs_per_batch",
        "p0,p0,p1,p1",
        "complete-generation-call",
    ):
        assert recovered_detail in serialized


def test_source_exact_identity_and_evaluator_generation_contracts_are_frozen() -> None:
    assert PAPER_BENCHMARK_ITEM_COUNTS == {
        "gsm8k": 1319,
        "mmlu": 2850,
        "mmlu-pro": 1400,
        "arc": 1172,
        "csqa": 1221,
        "math-500": 500,
    }
    assert SOURCE_PROTOCOL == {
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
    assert EXACT_CLEAN == {
        "version": "full-prompt-utf8-byte-equality/v1",
        "normalization": "none",
    }
    assert GENERATION == {
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

    legacy = PROTOCOL["legacy_backed"]
    assert legacy["source"] == SOURCE_PROTOCOL
    assert legacy["corrector_models"] == CORRECTOR_MODELS
    assert legacy["generation"] == GENERATION
    assert legacy["math_diagnostic_correctors"] == list(MATH_DIAGNOSTIC_CORRECTORS)


def test_published_values_remain_descriptive_names_not_acceptance_targets() -> None:
    assert PUBLISHED_REFERENCE["role"] == ("descriptive-historical-reference-not-acceptance-target")
    assert PUBLISHED_REFERENCE["methods"] == {
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
    }
    assert PUBLISHED_REFERENCE["total"] == {
        "exact_clean": 45641,
        "same_changed": 0,
        "archive_changed": 4362,
    }
    assert PUBLISHED_REFERENCE["math_intact_word_changes_per_item"] == {
        "t5-large-spell": 2.562,
        "qwen2.5-7b-instruct": 0.182,
    }


def test_protocol_sha256_is_finite_canonical_and_order_independent() -> None:
    assert PROTOCOL["version"] == "input-corrector-audit-protocol/v1"
    assert re.fullmatch(r"[0-9a-f]{64}", PROTOCOL_SHA256)
    assert PROTOCOL_SHA256 == canonical_sha256(PROTOCOL)
    assert PROTOCOL_SHA256 == _independent_canonical_sha256(PROTOCOL)

    first = {"z": [3, 2, 1], "a": {"β": True, "x": None}}
    reordered = {"a": {"x": None, "β": True}, "z": [3, 2, 1]}
    assert canonical_sha256(first) == canonical_sha256(reordered)
    assert canonical_sha256(first) != canonical_sha256({**first, "z": [3, 2, 0]})

    with pytest.raises(ValueError, match="compliant|range|JSON|nan|NaN"):
        canonical_sha256({"invalid": float("nan")})
