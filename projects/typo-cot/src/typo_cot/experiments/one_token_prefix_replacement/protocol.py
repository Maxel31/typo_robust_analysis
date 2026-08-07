"""Versioned protocol and final-PDF references for the one-token diagnostic."""

from __future__ import annotations

from typing import Final

POSITION_CONTROLS: Final = ("distant", "adjacent")
PRIMARY_SETTING: Final = ("google/gemma-3-4b-it", "gsm8k")
LEGACY_SETTING_IDS: Final = {
    ("google/gemma-3-1b-it", "arc"): "gemma1b_arc",
    ("google/gemma-3-1b-it", "gsm8k"): "gemma1b_gsm8k",
    ("google/gemma-3-1b-it", "mmlu"): "gemma1b_mmlu",
    ("google/gemma-3-4b-it", "arc"): "gemma4b_arc",
    ("google/gemma-3-4b-it", "gsm8k"): "gemma4b_gsm8k",
    ("google/gemma-3-4b-it", "mmlu"): "gemma4b_mmlu",
    ("meta-llama/Llama-3.2-1B-Instruct", "arc"): "llama1b_arc",
    ("meta-llama/Llama-3.2-1B-Instruct", "gsm8k"): "llama1b_gsm8k",
    ("meta-llama/Llama-3.2-1B-Instruct", "mmlu"): "llama1b_mmlu",
    ("meta-llama/Llama-3.2-3B-Instruct", "arc"): "llama3b_arc",
    ("meta-llama/Llama-3.2-3B-Instruct", "gsm8k"): "llama3b_gsm8k",
    ("meta-llama/Llama-3.2-3B-Instruct", "mmlu"): "llama3b_mmlu",
    ("mistralai/Mistral-7B-Instruct-v0.3", "arc"): "mistral7b_arc",
    ("mistralai/Mistral-7B-Instruct-v0.3", "gsm8k"): "mistral7b_gsm8k",
    ("mistralai/Mistral-7B-Instruct-v0.3", "mmlu"): "mistral7b_mmlu",
}
EXTENSION_SETTINGS: Final = frozenset(LEGACY_SETTING_IDS) - {PRIMARY_SETTING}
ADJACENT_SETTINGS: Final = frozenset(
    {
        ("google/gemma-3-1b-it", "gsm8k"),
        ("meta-llama/Llama-3.2-3B-Instruct", "arc"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mmlu"),
    }
)
LEGACY_TARGETING_CODES: Final = {
    "attribution-4": "lxt4",
    "random-4": "rnd4",
}

PAPER_SETTINGS: Final = {
    "all_cells": [
        {"model": model, "benchmark": benchmark, "setting_id": setting_id}
        for (model, benchmark), setting_id in sorted(
            LEGACY_SETTING_IDS.items(), key=lambda item: item[1]
        )
    ],
    "primary_cell": {
        "model": PRIMARY_SETTING[0],
        "benchmark": PRIMARY_SETTING[1],
        "setting_id": LEGACY_SETTING_IDS[PRIMARY_SETTING],
    },
    "extension_cells": [
        {
            "model": model,
            "benchmark": benchmark,
            "setting_id": LEGACY_SETTING_IDS[(model, benchmark)],
        }
        for model, benchmark in sorted(
            EXTENSION_SETTINGS, key=lambda item: LEGACY_SETTING_IDS[item]
        )
    ],
    "adjacent_cells": [
        {
            "model": model,
            "benchmark": benchmark,
            "setting_id": LEGACY_SETTING_IDS[(model, benchmark)],
        }
        for model, benchmark in sorted(ADJACENT_SETTINGS, key=lambda item: LEGACY_SETTING_IDS[item])
    ],
}

GENERATION: Final = {
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

IMPLEMENTATION: Final = "one-token-prefix-replacement-huggingface/v1"
ANSWER_DECODING: Final = "generated-token-ids-only/v1"
ANSWER_EXTRACTION: Final = "primary-then-empty-only-fallback-cap-aware/v1"
PRE_ANSWER_BOUNDARY: Final = "first-submitted-[Tt]he-answer-is/v1"
PROFILE_INPUT: Final = "tokenized-full-clean-cot-under-clean-and-edited-prompts/v1"
GENERATION_INPUT: Final = "clean-prompt-ids-plus-clean-cot-before-site-plus-one-token/v1"
TOKEN_ADMISSIBILITY: Final = {
    "implementation": "submitted-producer-tokenizer-candidate-pool/v1",
    "admissible_token_ids_sha256_algorithm": "sorted-decimal-lines/v1",
    "marker_regex": (
        r"^(?:<unused\d+>|<\|reserved_special_token_\d+\|>"
        r"|\[control_\d+\]|\[unused\d+\])$"
    ),
}

BENCHMARK_DATASET_NAMES: Final = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "arc": "arc_challenge",
}

PAPER_DEFINED: Final = {
    "purpose": "supplementary-clean-question-one-token-answer-sensitivity",
    "not_typo_repair": True,
    "eligibility": {
        "clean_answer": "correct",
        "edited_answer": "incorrect",
        "aligned_edit": True,
        "exact_token_boundaries": True,
        "clean_cot_tokens": {"minimum": 8, "maximum": 512},
    },
    "settings": PAPER_SETTINGS,
    "candidate": "clean-token-rank-under-edited-context-greater-than-one",
    "selected_position": "maximum-clean-to-edited-next-token-kl",
    "distant_control": "lower-median-kl-candidate-at-distance-at-least-three",
    "intervention": GENERATION_INPUT,
    "endpoint": "correct-keep-to-incorrect-replacement",
    "table10_readout": "paired-selected-versus-distant-diagonal-replacement",
    "distant_factorial_denominator": "both-keeps-correct-and-all-four-replacements-nonnoop",
    "adjacent_denominator": "both-keeps-correct-and-selected-token-nonnoop-at-both-sites",
    "aggregate": "fourteen-extensions-only",
    "cluster_key": ["benchmark", "sample_id"],
    "generation": "greedy-bfloat16-left-padded-at-most-512-new-tokens",
}

LEGACY_BACKED: Final = {
    "pre_answer_locator": PRE_ANSWER_BOUNDARY,
    "selection_stage_alignment": "prompt-length-suffix-equality-before-exact-boundary-audit",
    "profile_alignment": "shared-clean-cot-suffix",
    "position_indexing": "zero-based-clean-cot-token-index",
    "selected_exact_tie": "smallest-token-index",
    "distant_median": "statistics.median_low-then-smallest-token-index",
    "adjacent": "nearest-strictly-lower-kl-position",
    "adjacent_side_tie": "sha256(short-setting|legacy-condition|sample-id)-parity",
    "table10_denominator": "both-keeps-correct-and-both-diagonal-replacements-nonnoop",
    "submitted_factorial_token_guard": (
        "selected-and-control-edited-top1-tokens-admissible-and-distinct"
    ),
    "batch_size": 1,
    "seed": 42,
    "answer_decoding": ANSWER_DECODING,
    "answer_extraction": ANSWER_EXTRACTION,
}

PUBLIC_IMPLEMENTATION: Final = {
    "selection": "preserve-clean-prefix-selected-ids-without-boundary-backfill",
    "exact_boundary_audit": (
        "clean-and-edited-prompt-prefix-preservation-plus-shared-clean-cot-suffix"
    ),
}

HISTORICAL_REFERENCE: Final = {
    "table10_extension_aggregate": {
        "includes_primary": False,
        "paired_eligible": 1629,
        "selected": {"numerator": 492, "percent": 30.2},
        "control": {"numerator": 296, "percent": 18.2},
    },
    "table10_primary_gemma4b_gsm8k": {
        "includes_primary": True,
        "paired_eligible": 153,
        "selected": {"numerator": 41, "percent": 26.8},
        "control": {"numerator": 23, "percent": 15.0},
    },
    "table11_distant_pooled": {
        "includes_primary": False,
        "paired_eligible": 1575,
        "selected_percent": 28.3,
        "control_percent": 20.1,
        "difference_percentage_points": 8.2,
        "confidence_interval_95": [6.0, 10.4],
    },
    "table11_distant_submitted_producer_exact_counts": {
        "includes_primary": False,
        "paired_eligible": 1575,
        "event_opportunities_per_position": 3150,
        "selected": {"numerator": 892, "percent": 28.3},
        "control": {"numerator": 633, "percent": 20.1},
        "difference_percentage_points": 8.2,
    },
    "table11_distant_final_pdf_literal_reclassification": {
        "includes_primary": False,
        "paired_eligible": 1603,
        "event_opportunities_per_position": 3206,
        "selected": {"numerator": 912, "percent": 28.4},
        "control": {"numerator": 647, "percent": 20.2},
        "difference_percentage_points": 8.3,
        "submitted_producer_attrition": {
            "count": 28,
            "selected_and_control_tokens_identical": 28,
            "selected_token_not_admissible": 0,
            "control_token_not_admissible": 0,
        },
    },
    "table11_distant_strata": {
        "selected_before_control": {
            "paired_eligible": 1044,
            "selected_percent": 32.3,
            "control_percent": 16.6,
            "difference_percentage_points": 15.7,
            "confidence_interval_95": [13.0, 18.4],
        },
        "selected_after_control": {
            "paired_eligible": 531,
            "selected_percent": 20.5,
            "control_percent": 27.0,
            "difference_percentage_points": -6.5,
            "confidence_interval_95": [-10.0, -3.0],
        },
    },
    "table11_adjacent_pooled": {
        "includes_primary": False,
        "paired_eligible": 391,
        "selected_percent": 31.7,
        "control_percent": 28.6,
        "difference_percentage_points": 3.1,
        "confidence_interval_95": [-1.8, 7.9],
    },
    "figure5_worked_case": {
        "setting_id": "gemma4b_gsm8k",
        "targeting_code": "lxt4",
        "sample_id": "gsm8k_00556",
        "selected_position": 23,
        "distant_position": 60,
        "selected_kl": 8.785974,
        "clean_token_rank_under_clean": 1,
        "clean_token_rank_under_edited": 8,
        "clean_token_text": " thrice",
        "selected_edited_top1_text": " twice",
        "selected_keep_answer": "160",
        "selected_replacement_answer": "120",
        "distant_keep_answer": "160",
        "distant_replacement_answer": "160",
    },
    "table10_cells": {
        "gemma1b_gsm8k": {
            "paired_eligible": 120,
            "selected": {"numerator": 46, "percent": 38.3},
            "control": {"numerator": 35, "percent": 29.2},
        },
        "gemma1b_mmlu": {
            "paired_eligible": 119,
            "selected": {"numerator": 33, "percent": 27.7},
            "control": {"numerator": 27, "percent": 22.7},
        },
        "gemma1b_arc": {
            "paired_eligible": 118,
            "selected": {"numerator": 22, "percent": 18.6},
            "control": {"numerator": 12, "percent": 10.2},
        },
        "gemma4b_gsm8k": {
            "paired_eligible": 153,
            "selected": {"numerator": 41, "percent": 26.8},
            "control": {"numerator": 23, "percent": 15.0},
        },
        "gemma4b_mmlu": {
            "paired_eligible": 125,
            "selected": {"numerator": 23, "percent": 18.4},
            "control": {"numerator": 19, "percent": 15.2},
        },
        "gemma4b_arc": {
            "paired_eligible": 125,
            "selected": {"numerator": 21, "percent": 16.8},
            "control": {"numerator": 14, "percent": 11.2},
        },
        "llama1b_gsm8k": {
            "paired_eligible": 106,
            "selected": {"numerator": 50, "percent": 47.2},
            "control": {"numerator": 32, "percent": 30.2},
        },
        "llama1b_mmlu": {
            "paired_eligible": 107,
            "selected": {"numerator": 40, "percent": 37.4},
            "control": {"numerator": 24, "percent": 22.4},
        },
        "llama1b_arc": {
            "paired_eligible": 106,
            "selected": {"numerator": 24, "percent": 22.6},
            "control": {"numerator": 19, "percent": 17.9},
        },
        "llama3b_gsm8k": {
            "paired_eligible": 113,
            "selected": {"numerator": 43, "percent": 38.1},
            "control": {"numerator": 22, "percent": 19.5},
        },
        "llama3b_mmlu": {
            "paired_eligible": 119,
            "selected": {"numerator": 44, "percent": 37.0},
            "control": {"numerator": 18, "percent": 15.1},
        },
        "llama3b_arc": {
            "paired_eligible": 121,
            "selected": {"numerator": 31, "percent": 25.6},
            "control": {"numerator": 17, "percent": 14.0},
        },
        "mistral7b_gsm8k": {
            "paired_eligible": 103,
            "selected": {"numerator": 40, "percent": 38.8},
            "control": {"numerator": 25, "percent": 24.3},
        },
        "mistral7b_mmlu": {
            "paired_eligible": 119,
            "selected": {"numerator": 37, "percent": 31.1},
            "control": {"numerator": 17, "percent": 14.3},
        },
        "mistral7b_arc": {
            "paired_eligible": 128,
            "selected": {"numerator": 38, "percent": 29.7},
            "control": {"numerator": 15, "percent": 11.7},
        },
    },
    "table11_adjacent_cells": {
        "gemma1b_gsm8k": {
            "paired_eligible": 127,
            "selected_percent": 37.0,
            "control_percent": 38.6,
            "difference_percentage_points": -1.6,
            "confidence_interval_95": [-10.2, 7.1],
        },
        "llama3b_arc": {
            "paired_eligible": 133,
            "selected_percent": 24.8,
            "control_percent": 21.8,
            "difference_percentage_points": 3.0,
            "confidence_interval_95": [-5.3, 11.4],
        },
        "mistral7b_mmlu": {
            "paired_eligible": 131,
            "selected_percent": 33.6,
            "control_percent": 26.0,
            "difference_percentage_points": 7.6,
            "confidence_interval_95": [-0.8, 15.8],
        },
    },
    "reference_provenance": {
        "printed_pdf_fields": "paired denominators, rounded percentages, differences, and intervals",
        "integer_numerators": "submitted-producer counts consistent with printed percentages",
        "table10_exact_denominator": "submitted-producer operationalization of printed n1",
    },
    "comparability": {
        "fresh_public_pair_preparation_may_change_membership": True,
        "historical_values_are_acceptance_targets": False,
        "submitted_factorial_distinct_token_guard_not_in_final_pdf": True,
        "submitted_factorial_admissible_token_guard_not_in_final_pdf": True,
        "stored_extensions_reclassified_under_final_pdf_literal": True,
    },
}

PROTOCOL: Final = {
    "paper_defined": PAPER_DEFINED,
    "legacy_backed": LEGACY_BACKED,
    "public_implementation": PUBLIC_IMPLEMENTATION,
    "implementation": IMPLEMENTATION,
}


__all__ = [
    "ADJACENT_SETTINGS",
    "ANSWER_DECODING",
    "ANSWER_EXTRACTION",
    "BENCHMARK_DATASET_NAMES",
    "EXTENSION_SETTINGS",
    "GENERATION",
    "GENERATION_INPUT",
    "HISTORICAL_REFERENCE",
    "IMPLEMENTATION",
    "LEGACY_BACKED",
    "LEGACY_SETTING_IDS",
    "LEGACY_TARGETING_CODES",
    "PAPER_DEFINED",
    "PAPER_SETTINGS",
    "POSITION_CONTROLS",
    "PRE_ANSWER_BOUNDARY",
    "PRIMARY_SETTING",
    "PROFILE_INPUT",
    "PROTOCOL",
    "PUBLIC_IMPLEMENTATION",
    "TOKEN_ADMISSIBILITY",
]
