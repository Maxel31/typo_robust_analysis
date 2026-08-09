"""Focused contracts for typo-warning readable output and publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from typo_cot.experiments.typo_warning_prompt import aggregation
from typo_cot.experiments.typo_warning_prompt.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.typo_warning_prompt.protocol import PROTOCOL


def _metric() -> dict[str, object]:
    return {
        "n": 1,
        "counts": {
            "without_warning_correct": 1,
            "with_warning_correct": 0,
        },
        "accuracy": {
            "without_warning": 1.0,
            "with_warning": 0.0,
            "with_minus_without": -1.0,
        },
        "mcnemar": {
            "b10": 1,
            "b01": 0,
            "discordant": 1,
            "p_value": 1.0,
            "method": "exact-two-sided-mcnemar-binomial/v1",
        },
    }


def test_latex_summary_labels_its_first_column_as_model() -> None:
    payload = {
        "settings": [
            {
                "model": "test/model",
                "benchmark": "gsm8k",
                **_metric(),
            }
        ],
        "pooled_by_benchmark": {
            "gsm8k": {"model": "all-three-models", **_metric()},
            "mmlu": {"model": "all-three-models", **_metric()},
        },
    }

    latex = aggregation._latex_text(payload)

    assert r"Model & Benchmark & $n$" in latex
    assert r"Scope & Benchmark & $n$" not in latex
    assert "test/model & gsm8k" in latex


def test_stage_publication_removes_renamed_output_when_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / ".summary.stage"
    output = tmp_path / "summary"
    stage.mkdir()
    (stage / "run.json").write_text("{}\n", encoding="utf-8")

    def fail_parent_fsync(_path: Path) -> None:
        raise OSError("directory fsync failed")

    monkeypatch.setattr(aggregation, "_fsync_directory", fail_parent_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        aggregation._publish_staged_directory(stage, output)

    assert not stage.exists()
    assert not output.exists()


def test_stage_publication_never_deletes_a_concurrent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / ".summary.stage"
    output = tmp_path / "summary"
    moved_stage = tmp_path / "owned-stage-after-race"
    stage.mkdir()
    (stage / "run.json").write_text("{}\n", encoding="utf-8")

    def replace_output_then_fail(_path: Path) -> None:
        output.replace(moved_stage)
        output.mkdir()
        (output / "concurrent.txt").write_text("keep\n", encoding="utf-8")
        raise OSError("directory fsync failed")

    monkeypatch.setattr(aggregation, "_fsync_directory", replace_output_then_fail)

    with pytest.raises(RuntimeError, match="could not be safely removed"):
        aggregation._publish_staged_directory(stage, output)

    assert (output / "concurrent.txt").read_text(encoding="utf-8") == "keep\n"
    assert (moved_stage / "run.json").is_file()


def test_warning_document_separates_pdf_results_from_legacy_protocol() -> None:
    document = (Path(__file__).parents[1] / "docs" / "typo-warning-prompt.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(document.split())

    assert "benchmark-level edited-input accuracy" in normalized
    assert "pooling, pairing, and exact-test implementation" in normalized
    assert "recovered from the submitted producer" in normalized


def test_paper_builder_rejects_non_paper_dependency_version() -> None:
    revision = "a" * 40
    runtime = {
        "operation": "typo-warning-prompt",
        "runtime": "HuggingFaceWarningPromptRuntime",
        "python": "3.12.11",
        "torch": "2.9.0",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
        "model": "test/model",
        "requested_revision": revision,
        "model_revision": revision,
        "model_revision_source": "model-config-metadata",
        "tokenizer_revision": revision,
        "tokenizer_revision_source": "tokenizer-init-metadata",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "cuda": "12.8",
        "cuda_visible_devices": "0",
        "gpu_name": "Fixture GPU",
        "gpu_total_memory_bytes": 1024,
        "generation": PROTOCOL["generation"],
        "implementation": PROTOCOL["implementation"],
        "implementation_code": implementation_code_identity(),
        "batching": PROTOCOL["batching"],
        "answer_span_decoding": PROTOCOL["answer_span_decoding"],
        "answer_extraction": PROTOCOL["answer_extraction"],
        "prompt_intervention": PROTOCOL["prompt_intervention"],
        "effective_eos_token_ids": [2],
        "effective_eos_token_ids_source": "fixture",
    }

    with pytest.raises(ValueError, match="torch"):
        aggregation._validate_setting_runtime(
            {"runtime": runtime},
            model="test/model",
            model_revision=revision,
            gpu_id="0",
        )
