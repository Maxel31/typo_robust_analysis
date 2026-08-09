"""Runner contracts for submitted-compatible Appendix E generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

import typo_cot.experiments.typo_warning_prompt.runner as warning_runner
from typo_cot.experiments.typo_warning_prompt.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.typo_warning_prompt.planning import WarningPromptPlan
from typo_cot.experiments.typo_warning_prompt.runner import (
    WarningPromptArmScan,
    WarningPromptConfig,
    WarningPromptGeneration,
    WarningPromptInputUse,
    WarningPromptRunError,
    run_typo_warning_prompt,
)
from typo_cot.experiments.typo_warning_prompt.scoring import extract_submitted_answer
from typo_cot.experiments.typo_warning_prompt.source import (
    SourceFileSnapshot,
    WarningSourceBundle,
    WarningSourceCase,
    canonical_sha256,
)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source(tmp_path: Path, *, count: int = 10) -> WarningSourceBundle:
    fixture = tmp_path / "submitted-inputs.json"
    fixture.write_text('{"input_only":true}\n', encoding="utf-8")
    cases: list[WarningSourceCase] = []
    for index in range(count):
        sample_id = f"gsm8k_{index:05d}"
        prompt = (
            "few-shot\n\nNow solve the following problem:\n\n"
            f"Problem: typo question {index}\n\nSolution:"
        )
        submitted = {
            "schema_version": "typo-warning-submitted-input/v1",
            "sample_id": sample_id,
            "model": "google/gemma-3-4b-it",
            "benchmark": "gsm8k",
            "subset": "default",
            "gold_answer": "2",
            "clean": {
                "prompt_sha256": "1" * 64,
                "editable_text_sha256": "2" * 64,
            },
            "edited": {
                "prompt": prompt,
                "prompt_sha256": _text_sha256(prompt),
                "editable_text_sha256": "3" * 64,
            },
            "fixture_record_sha256": "4" * 64,
        }
        submitted_sha = canonical_sha256(
            {
                **submitted,
                "edited": {
                    "prompt_sha256": _text_sha256(prompt),
                    "editable_text_sha256": "3" * 64,
                },
            }
        )
        selector_sha = canonical_sha256([sample_id, "submitted-selector"])
        cases.append(
            WarningSourceCase(
                sample_id=sample_id,
                submitted_record=submitted,
                submitted_record_sha256=submitted_sha,
                selector_record_sha256=selector_sha,
            )
        )
    snapshot = SourceFileSnapshot(
        role="submitted-input-edit-manifest",
        path=fixture,
        sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
    )
    return WarningSourceBundle(
        model="google/gemma-3-4b-it",
        benchmark="gsm8k",
        model_revision="a" * 40,
        model_revision_evidence="test-fixture",
        dataset_revision="b" * 40,
        dataset_records_sha256=canonical_sha256([case.sample_id for case in cases]),
        dataset_sample_count=count,
        shared_sample_count=count,
        cases=tuple(cases),
        source_files=(snapshot,),
        source_sha256=canonical_sha256([case.identity for case in cases]),
        fixture_sha256=snapshot.sha256,
        fixture_canonical_sha256="8" * 64,
        fixture_is_bundled_submitted_input=False,
        setting_records_sha256="5" * 64,
        ordered_sample_ids_sha256=canonical_sha256([case.sample_id for case in cases]),
    )


def _input_use(arm: str, plan: WarningPromptPlan) -> WarningPromptInputUse:
    arm_plan = next(item for item in plan.arms if item.arm == arm)
    return WarningPromptInputUse(
        arm=arm,
        prompt_text_sha256=arm_plan.prompt_sha256,
        prompt_char_count=len(arm_plan.prompt),
        prompt_token_count=10,
        prompt_ids_sha256=("6" if arm == "without-warning" else "7") * 64,
    )


def _generation(*, correct: bool, benchmark: str, gold: str) -> WarningPromptGeneration:
    answer = gold if correct else "3"
    text = f"The answer is {answer}."
    extraction = extract_submitted_answer(
        text,
        benchmark=benchmark,
        correct_answer=gold,
    )
    return WarningPromptGeneration(
        token_ids=(2,),
        text=text,
        value=extraction.value,
        is_extracted=extraction.is_extracted,
        is_correct=extraction.is_correct,
        method=extraction.method,
        primary_method=extraction.primary_method,
        stop_reason="eos_token",
        stop_token_id=2,
    )


class FakeRuntime:
    def __init__(
        self,
        *,
        fail_arm: str | None = None,
        reverse_results: bool = False,
    ) -> None:
        self.fail_arm = fail_arm
        self.reverse_results = reverse_results
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def provenance(self) -> dict[str, object]:
        return {
            "operation": "typo-warning-prompt",
            "runtime": "HuggingFaceWarningPromptRuntime",
            "python": "3.12.11",
            "torch": "2.10.0",
            "transformers": "4.57.6",
            "accelerate": "1.12.0",
            "model": "google/gemma-3-4b-it",
            "requested_revision": "a" * 40,
            "model_revision": "a" * 40,
            "model_revision_source": "model-config-metadata",
            "tokenizer_revision": "a" * 40,
            "tokenizer_revision_source": "tokenizer-init-metadata",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "cuda": "12.8",
            "cuda_visible_devices": "0",
            "gpu_name": "Fixture GPU",
            "gpu_total_memory_bytes": 1024,
            "generation": dict(warning_runner._GENERATION),
            "implementation": warning_runner._IMPLEMENTATION,
            "implementation_code": implementation_code_identity(),
            "batching": dict(warning_runner._BATCHING),
            "answer_span_decoding": dict(warning_runner._ANSWER_SPAN_DECODING),
            "answer_extraction": warning_runner._ANSWER_EXTRACTION,
            "prompt_intervention": dict(warning_runner._PROMPT_INTERVENTION),
            "effective_eos_token_ids": [2],
            "effective_eos_token_ids_source": "fixture",
        }

    def scan_arm_batch(
        self,
        pairs: Sequence[dict[str, object]],
        plans: Sequence[WarningPromptPlan],
        *,
        arm: str,
    ) -> Sequence[WarningPromptArmScan]:
        ids = tuple(str(pair["sample_id"]) for pair in pairs)
        self.calls.append((arm, ids))
        if arm == self.fail_arm:
            self.fail_arm = None
            raise RuntimeError("fixture failure")
        scans = [
            WarningPromptArmScan(
                sample_id=str(pair["sample_id"]),
                arm=arm,
                input_use=_input_use(arm, plan),
                generation=_generation(
                    correct=arm == "without-warning",
                    benchmark=str(pair["benchmark"]),
                    gold=plan.gold_answer,
                ),
            )
            for pair, plan in zip(pairs, plans, strict=True)
        ]
        return tuple(reversed(scans)) if self.reverse_results else tuple(scans)


def _config(source: WarningSourceBundle, output_dir: Path) -> WarningPromptConfig:
    return WarningPromptConfig(
        model=source.model,
        benchmark="gsm8k",
        input_fixture=source.source_files[0].path,
        output_dir=output_dir,
        gpu_id="0",
    )


def _install_source(
    monkeypatch: pytest.MonkeyPatch,
    source: WarningSourceBundle,
) -> None:
    def load(
        input_fixture: Path | None,
        *,
        model: str,
        benchmark: str,
    ) -> WarningSourceBundle:
        assert Path(input_fixture or "").resolve() == source.source_files[0].path
        assert (model, benchmark) == (source.model, source.benchmark)
        return source

    monkeypatch.setattr(warning_runner, "load_warning_sources", load)


def test_runner_uses_submitted_same_arm_batches_and_publishes_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    _install_source(monkeypatch, source)
    runtime = FakeRuntime()
    config = _config(source, tmp_path / "out")

    result = run_typo_warning_prompt(config, runtime=runtime)

    ids = tuple(case.sample_id for case in source.cases)
    assert runtime.calls == [
        ("without-warning", ids[:8]),
        ("without-warning", ids[8:]),
        ("with-warning", ids[:8]),
        ("with-warning", ids[8:]),
    ]
    records = _read_jsonl(result.records_path)
    assert len(records) == 10
    assert all(row["events"]["without_warning_only"] is True for row in records)
    manifest = _read_json(result.run_path)
    assert manifest["schema_version"] == "typo-warning-prompt-run/v2"
    assert manifest["counts"]["completed_arms"] == 20
    assert manifest["counts"]["completed_pairs"] == 10
    assert manifest["checkpoints"] == {}
    assert not (config.output_dir / ".typo-warning-work").exists()


def test_runner_resumes_only_missing_arms_after_batch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    _install_source(monkeypatch, source)
    config = _config(source, tmp_path / "out")
    first = FakeRuntime(fail_arm="with-warning")

    with pytest.raises(WarningPromptRunError, match="with-warning"):
        run_typo_warning_prompt(config, runtime=first)

    failed = _read_json(config.output_dir / "run.json")
    assert failed["counts"]["completed_arms"] == 10
    assert failed["counts"]["completed_pairs"] == 0
    assert len(failed["checkpoints"]) == 10
    assert all(key.endswith("|without-warning") for key in failed["checkpoints"])

    resumed_runtime = FakeRuntime()
    result = run_typo_warning_prompt(
        replace(config, resume=True),
        runtime=resumed_runtime,
    )

    ids = tuple(case.sample_id for case in source.cases)
    assert resumed_runtime.calls == [
        ("with-warning", ids[:8]),
        ("with-warning", ids[8:]),
    ]
    assert result.records == 10


def test_runner_rejects_runtime_batch_reordering_without_checkpointing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    _install_source(monkeypatch, source)
    config = _config(source, tmp_path / "out")

    with pytest.raises(WarningPromptRunError, match="sorted sample-arm order"):
        run_typo_warning_prompt(config, runtime=FakeRuntime(reverse_results=True))

    failed = _read_json(config.output_dir / "run.json")
    assert failed["counts"]["completed_arms"] == 0
    assert failed["checkpoints"] == {}


def test_completed_resume_revalidates_outputs_without_loading_a_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, count=2)
    _install_source(monkeypatch, source)
    config = _config(source, tmp_path / "out")
    result = run_typo_warning_prompt(config, runtime=FakeRuntime())

    resumed = run_typo_warning_prompt(replace(config, resume=True), runtime=None)

    assert resumed == result


def test_completed_resume_does_not_recompute_the_derived_setting_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, count=2)
    _install_source(monkeypatch, source)
    config = _config(source, tmp_path / "out")
    result = run_typo_warning_prompt(config, runtime=FakeRuntime())

    def reject_recomputation(*, b10: int, b01: int) -> dict[str, object]:
        raise AssertionError(f"statistics were recomputed: {b10=}, {b01=}")

    monkeypatch.setattr(warning_runner, "exact_mcnemar", reject_recomputation)

    resumed = run_typo_warning_prompt(replace(config, resume=True), runtime=None)

    assert resumed == result


def test_completed_resume_rejects_semantically_tampered_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, count=2)
    _install_source(monkeypatch, source)
    config = _config(source, tmp_path / "out")
    result = run_typo_warning_prompt(config, runtime=FakeRuntime())
    rows = _read_jsonl(result.records_path)
    rows[0]["events"]["with_warning_correct"] = True
    result.records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = _read_json(result.run_path)
    output = manifest["outputs"][result.records_path.name]
    output["sha256"] = hashlib.sha256(result.records_path.read_bytes()).hexdigest()
    output["bytes"] = result.records_path.stat().st_size
    result.run_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(WarningPromptRunError, match="completed run validation"):
        run_typo_warning_prompt(replace(config, resume=True), runtime=None)


def test_runtime_provenance_rejects_stale_implementation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, count=1)
    _install_source(monkeypatch, source)
    runtime = FakeRuntime()
    provenance = runtime.provenance()
    provenance["implementation_code"] = {
        **implementation_code_identity(),
        "sha256": "0" * 64,
    }
    monkeypatch.setattr(runtime, "provenance", lambda: provenance)

    with pytest.raises(ValueError, match="implementation_code"):
        run_typo_warning_prompt(_config(source, tmp_path / "out"), runtime=runtime)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("runtime", "FakeRuntime"),
        ("torch", "2.9.0"),
        ("transformers", "4.57.5"),
        ("accelerate", "1.11.0"),
    ),
)
def test_runtime_provenance_rejects_non_paper_runtime_or_dependency_versions(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    source = _source(tmp_path, count=1)
    provenance = {
        **FakeRuntime().provenance(),
        "runtime": "HuggingFaceWarningPromptRuntime",
        "python": "3.12.11",
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
        "model_revision_source": "model-config-metadata",
        "tokenizer_revision_source": "tokenizer-init-metadata",
        "device": "cuda:0",
        "cuda": "12.8",
        "gpu_name": "Fixture GPU",
        "gpu_total_memory_bytes": 1024,
    }
    provenance[field] = wrong_value

    with pytest.raises(ValueError, match=field):
        warning_runner._validate_runtime_provenance(
            provenance,
            config=_config(source, tmp_path / "out"),
            source=source,
        )


def test_limit_run_retains_every_custom_comparability_limitation(tmp_path: Path) -> None:
    source = replace(
        _source(tmp_path, count=1),
        model="custom/model",
        fixture_is_bundled_submitted_input=False,
    )
    config = replace(
        _config(source, tmp_path / "out"),
        model="custom/model",
        limit=1,
    )

    comparability = warning_runner._comparability(config, source)

    assert comparability["status"] == "partial-smoke-run"
    assert set(comparability["limitations"]) >= {
        "run-is-limit-truncated",
        "input-fixture-is-not-the-bundled-submitted-artifact",
        "model-is-outside-submitted-grid",
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"benchmark": "arc"}, "benchmark"),
        ({"gpu_id": "0,1"}, "single"),
        ({"limit": 0}, "limit"),
    ),
)
def test_config_rejects_invalid_arguments(
    tmp_path: Path,
    changes: Mapping[str, object],
    message: str,
) -> None:
    source = _source(tmp_path, count=1)
    with pytest.raises(ValueError, match=message):
        replace(_config(source, tmp_path / "out"), **changes)
