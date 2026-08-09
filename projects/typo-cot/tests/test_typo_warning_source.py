"""Fail-closed source reconstruction tests for the compact warning fixture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import typo_cot.models.prompts as prompt_templates
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.typo_warning_prompt.planning import (
    WARNING_TEXT,
    inject_typo_warning,
)
from typo_cot.experiments.typo_warning_prompt.protocol import PAPER_GRID
from typo_cot.experiments.typo_warning_prompt.source import (
    DEFAULT_SUBMITTED_INPUTS,
    canonical_sha256,
    load_warning_sources,
    validate_source_snapshot,
)
from typo_cot.experiments.typo_warning_prompt.submitted_inputs import (
    _AGGREGATE_HASH_SEMANTICS,
    _PATCH_SEMANTICS,
)


def _text_records_sha256(sample_ids: list[str], texts: list[str]) -> str:
    return canonical_sha256(
        [
            {"sample_id": sample_id, "text": text}
            for sample_id, text in zip(sample_ids, texts, strict=True)
        ]
    )


def _fixture_payload() -> tuple[dict[str, object], dict[str, object], str]:
    model = "test/model"
    sample_id = "gsm8k_00000"
    question = "Two plus zero?"
    edited = "Txy plus zero?"
    sample = {
        "sample_id": sample_id,
        "question": question,
        "choices": None,
        "correct_answer": "2",
        "subset": "default",
    }
    template = prompt_templates.create_prompt_template("gsm8k")
    clean_prompt = template.generate(question=question).get_full_prompt()
    edited_prompt = template.generate(question=edited).get_full_prompt()
    warned_prompt = inject_typo_warning(edited_prompt, benchmark="gsm8k", enabled=True)
    patches = [[[1, "w", "x"], [2, "o", "y"]]]
    gsm_ids = [sample_id]
    mmlu_ids = ["mmlu_test_subject_0000"]
    fixture = {
        "schema_version": "typo-warning-submitted-input-patches/v2",
        "paper_sha256": PAPER_SHA256,
        "artifact_role": "input-only-no-generated-outputs",
        "canonicalization": "json-utf8-sort-keys-compact-no-nan/v1",
        "provenance": {
            "contains_generated_outputs": False,
            "selection": "submitted-attribution-random-intersection-seed-42/v1",
            "legacy_input_source_fingerprints": {
                f"{model}|gsm8k": {
                    "submitted_attribution_full_input_sha256": "1" * 64,
                    "submitted_selected_input_sha256": "2" * 64,
                    "submitted_selector_manifest_sha256": "3" * 64,
                }
            },
            "prompt_template_file_sha256": hashlib.sha256(
                Path(prompt_templates.__file__).read_bytes()
            ).hexdigest(),
            "warning_text_sha256": hashlib.sha256(WARNING_TEXT.encode()).hexdigest(),
        },
        "dataset_revisions": {"gsm8k": "a" * 40, "mmlu": "b" * 40},
        "model_revisions": {model: {"revision": "c" * 40, "evidence": "test-fixture"}},
        "aggregate_hash_semantics": dict(_AGGREGATE_HASH_SEMANTICS),
        "patch_semantics": dict(_PATCH_SEMANTICS),
        "benchmarks": {
            "gsm8k": {
                "sample_count": 1,
                "sample_ids": gsm_ids,
                "ordered_sample_ids_sha256": canonical_sha256(gsm_ids),
                "dataset_records_sha256": canonical_sha256([sample]),
                "clean_editable_records_sha256": _text_records_sha256(gsm_ids, [question]),
                "clean_prompt_records_sha256": _text_records_sha256(gsm_ids, [clean_prompt]),
            },
            "mmlu": {
                "sample_count": 1,
                "sample_ids": mmlu_ids,
                "ordered_sample_ids_sha256": canonical_sha256(mmlu_ids),
                "dataset_records_sha256": "4" * 64,
                "clean_editable_records_sha256": "5" * 64,
                "clean_prompt_records_sha256": "6" * 64,
            },
        },
        "settings": [
            {
                "model": model,
                "benchmark": "gsm8k",
                "sample_count": 1,
                "historical_target_attempts": 4,
                "net_edit_count_distribution": {"2": 1, "3": 0, "4": 0},
                "records": patches,
                "records_sha256": canonical_sha256(patches),
                "edited_editable_records_sha256": _text_records_sha256(gsm_ids, [edited]),
                "without_warning_prompt_records_sha256": _text_records_sha256(
                    gsm_ids, [edited_prompt]
                ),
                "with_warning_prompt_records_sha256": _text_records_sha256(
                    gsm_ids, [warned_prompt]
                ),
            }
        ],
    }
    return fixture, sample, edited_prompt


def test_source_reconstructs_and_binds_compact_sequential_patches(tmp_path: Path) -> None:
    payload, sample, edited_prompt = _fixture_payload()
    path = tmp_path / "submitted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    source = load_warning_sources(
        path,
        model="test/model",
        benchmark="gsm8k",
        samples=[sample],
    )

    assert source.shared_sample_count == 1
    assert source.cases[0].submitted_record["edited"]["prompt"] == edited_prompt
    assert source.fixture_is_bundled_submitted_input is False
    assert source.to_dict()["contains_generated_outputs"] is False


def test_source_rejects_dataset_drift_even_when_the_sample_id_matches(tmp_path: Path) -> None:
    payload, sample, _edited_prompt = _fixture_payload()
    path = tmp_path / "submitted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    changed = {**sample, "question": "Two plus one?"}

    with pytest.raises(ValueError, match="dataset records SHA-256"):
        load_warning_sources(
            path,
            model="test/model",
            benchmark="gsm8k",
            samples=[changed],
        )


def test_source_snapshot_detects_fixture_mutation_after_loading(tmp_path: Path) -> None:
    payload, sample, _edited_prompt = _fixture_payload()
    path = tmp_path / "submitted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    source = load_warning_sources(
        path,
        model="test/model",
        benchmark="gsm8k",
        samples=[sample],
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source snapshot changed"):
        validate_source_snapshot(source)


@pytest.mark.skipif(
    os.environ.get("TYPO_COT_RUN_NETWORK_TESTS") != "1",
    reason="set TYPO_COT_RUN_NETWORK_TESTS=1 to load pinned public datasets",
)
def test_bundled_sources_reconstruct_all_settings_from_pinned_public_datasets() -> None:
    expected = {
        (
            "google/gemma-3-4b-it",
            "gsm8k",
        ): "e53f69bf13bfd6f9dae57f5ddb67e13be33b0fbf2fd7b586d45f8830e90c7c56",
        (
            "google/gemma-3-4b-it",
            "mmlu",
        ): "a73791adf239bbf7695d935039f5690a5cf0c69af9a3ed7345ae51f6e6943304",
        (
            "meta-llama/Llama-3.2-3B-Instruct",
            "gsm8k",
        ): "d7dea51f6ff458f8e16010991f2dca72fbad6ba3ab0edda480e64fd761cd0c26",
        (
            "meta-llama/Llama-3.2-3B-Instruct",
            "mmlu",
        ): "ceee5b83b47bb362810e761d885c7c267afb74cc9d52b0aaa249eeb363dbe61b",
        (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "gsm8k",
        ): "fe90db1bb040e82e02c443804e6f1ab26c3ce4c75b48b9cafe6b483423f1c46c",
        (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mmlu",
        ): "0ceb1ed5ed3c5b3dc7a4c33526eed740858d5cac20208800438c45ecae46ce73",
    }

    reconstructed = {
        (model, benchmark): load_warning_sources(
            DEFAULT_SUBMITTED_INPUTS,
            model=model,
            benchmark=benchmark,
        )
        for model, benchmark in PAPER_GRID
    }

    assert {
        key: (source.shared_sample_count, source.source_sha256)
        for key, source in reconstructed.items()
    } == {key: (300, digest) for key, digest in expected.items()}
