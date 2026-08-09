"""Regression contracts for the submitted Appendix E warning inputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from typo_cot.experiments.typo_warning_prompt.planning import (
    WARNING_TEXT,
    build_warning_prompt_plan,
)
from typo_cot.experiments.typo_warning_prompt.protocol import (
    ANSWER_EXTRACTION,
    BATCHING,
    RUNTIME_ENVIRONMENT,
)
from typo_cot.experiments.typo_warning_prompt.source import DEFAULT_SUBMITTED_INPUTS
from typo_cot.experiments.typo_warning_prompt.submitted_inputs import (
    SubmittedInputPatch,
    apply_submitted_patches,
    load_submitted_input_fixture,
)


def test_submitted_protocol_uses_same_arm_batches_and_primary_scorer() -> None:
    assert BATCHING == {
        "policy": "same-warning-arm-sorted-samples/v1",
        "batch_size": 8,
        "arm_order": ["without-warning", "with-warning"],
        "final_partial_batch": True,
        "checkpoint_unit": "sample-arm",
        "detail_source": "legacy-backed-not-printed-by-final-pdf",
    }
    assert ANSWER_EXTRACTION == "submitted-task-extractor-primary-only/v1"
    assert RUNTIME_ENVIRONMENT == {
        "runtime": "HuggingFaceWarningPromptRuntime",
        "minimum_python": [3, 12],
        "package_versions": {
            "torch": "2.10.0",
            "transformers": "4.57.6",
            "accelerate": "1.12.0",
        },
        "device": "single-visible-cuda-gpu/v1",
    }


def test_readmes_explain_gated_model_access_and_authentication() -> None:
    project = Path(__file__).parents[1]
    for name in ("README.md", "README.ja.md"):
        readme = (project / name).read_text(encoding="utf-8")
        assert "https://huggingface.co/google/gemma-3-4b-it" in readme
        assert "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct" in readme
        assert "uv run --project projects/typo-cot --extra lrp hf auth login" in readme


def test_submitted_patch_application_is_ordered_and_fail_closed() -> None:
    clean = "The ducks lay eggs."
    patches = (
        SubmittedInputPatch(start=4, expected="ducks", replacement="dicks"),
        SubmittedInputPatch(start=14, expected="", replacement="g"),
    )

    edited = apply_submitted_patches(clean, patches)

    assert edited == "The dicks lay geggs."
    with pytest.raises(ValueError, match="expected text"):
        apply_submitted_patches("The geese lay eggs.", patches)


def test_fixture_rejects_generated_answers_or_outcomes(tmp_path: Path) -> None:
    fixture = tmp_path / "submitted.json"
    payload = json.loads(DEFAULT_SUBMITTED_INPUTS.read_text(encoding="utf-8"))
    payload["generated_text"] = "must never be published in this fixture"
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generated|output-only|unexpected"):
        load_submitted_input_fixture(fixture)


def test_fixture_validates_setting_and_record_hashes(tmp_path: Path) -> None:
    fixture = tmp_path / "submitted.json"
    payload = json.loads(DEFAULT_SUBMITTED_INPUTS.read_text(encoding="utf-8"))
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_submitted_input_fixture(fixture)

    assert loaded.setting(model="google/gemma-3-4b-it", benchmark="gsm8k").records[0].sample_id == (
        "gsm8k_00002"
    )
    payload["settings"][0]["records"][0][0][2] = "x"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="records SHA-256"):
        load_submitted_input_fixture(fixture)


def test_plan_rejects_same_length_non_warning_insertion() -> None:
    pair = {
        "sample_id": "sample",
        "benchmark": "gsm8k",
        "gold_answer": "18",
        "edited": {
            "prompt": (
                "few-shot\n\nNow solve the following problem:\n\n"
                "Problem: typo question\n\nSolution:"
            )
        },
    }
    plan = build_warning_prompt_plan(pair, source_record_sha256="a" * 64)
    warned = plan.arms[1]
    insertion = WARNING_TEXT + "\n\n"
    marker_index = warned.marker_index
    forged_prompt = (
        warned.prompt[: marker_index - len(insertion)]
        + "x" * len(insertion)
        + warned.prompt[marker_index:]
    )
    forged_arm = replace(warned, prompt=forged_prompt)

    with pytest.raises(ValueError, match="warning insertion"):
        replace(plan, arms=(plan.arms[0], forged_arm))
