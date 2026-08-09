"""Contracts for the CPU-only final-paper Table 12 builder.

The synthetic producers are intentionally tiny, but their grid has the same
identity as the submitted experiment: five evaluator models, five core tasks,
and three correctors.  The tests freeze the scientific denominators separately
from rendering so a pooled word rate, a whitespace-only prompt comparison, or
the wrong cross-run endpoint cannot silently reproduce a plausible table.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pytest

from typo_cot.experiments.input_corrector_audit.aggregation import (
    BuildInputCorrectorSummaryConfig,
    InputCorrectorSummaryInputError,
    run_build_input_corrector_summary,
)
from typo_cot.experiments.input_corrector_audit.integrity import (
    analysis_code_identity,
    implementation_code_identity,
)
from typo_cot.experiments.input_corrector_audit.protocol import PROTOCOL_SHA256
from typo_cot.experiments.input_corrector_audit.protocol import (
    CORRECTOR_MODELS,
    GENERATION,
    canonical_sha256,
)


PAPER_SHA256 = "2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0"
MODELS = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
CORE_BENCHMARKS = ("gsm8k", "mmlu", "mmlu-pro", "arc", "csqa")
CORE_CORRECTORS = ("pyspellchecker", "t5-large-spell", "qwen2.5-7b-instruct")
MATH_DIAGNOSTIC_CORRECTORS = ("t5-large-spell", "qwen2.5-7b-instruct")
RECORDS_NAME = "corrector_records.jsonl"
SETTING_SUMMARY_NAME = "corrector_audit_summary.json"
BUILDER_OUTPUTS = {
    "input_corrector_summary.json",
    "table12_input_correctors.csv",
    "table12_input_correctors.md",
    "table12_input_correctors.tex",
    "run.json",
}

Setting = tuple[str, str, str]
IMPLEMENTATION_CODE = implementation_code_identity()
_BENCHMARK_EXTRACTORS = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}


@pytest.fixture(autouse=True)
def _use_one_record_paper_cohorts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the 75/10-cell logic without materializing 119,430 rows."""
    import typo_cot.experiments.input_corrector_audit.aggregation as aggregation_module

    monkeypatch.setattr(
        aggregation_module,
        "PAPER_BENCHMARK_ITEM_COUNTS",
        {benchmark: 1 for benchmark in (*CORE_BENCHMARKS, "math-500")},
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _prompt_sha256(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _output_metadata(path: Path, *, records: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        payload["records"] = records
    return payload


def _source_identity(
    *,
    model: str,
    benchmark: str,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cohort = f"{model}\n{benchmark}".encode()
    sample_ids = [str(record["sample_id"]) for record in records]
    return {
        "input_kind": "completed-prepare-edited-pairs/v1",
        "model": model,
        "benchmark": benchmark,
        "model_revision": "1" * 40,
        "records": len(records),
        "pairs_sha256": _sha256_bytes(b"pairs\n" + cohort),
        "run_sha256": _sha256_bytes(b"run\n" + cohort),
        "ordered_sample_ids_sha256": canonical_sha256(sample_ids),
        "dataset_records_sha256": _sha256_bytes(f"dataset\n{benchmark}".encode()),
    }


def _neural_environment() -> dict[str, object]:
    return {
        "profile": "neural",
        "python": "3.12.0",
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
        "device": "cuda:0",
        "cuda": "12.8",
        "cuda_visible_devices": "0",
        "gpu_name": "synthetic-test-gpu",
        "gpu_total_memory_bytes": 1,
    }


def _correction_runtime(corrector: str) -> dict[str, object]:
    specification = CORRECTOR_MODELS.get(corrector, CORRECTOR_MODELS["pyspellchecker"])
    requested_revision = specification["revision"]
    common: dict[str, object] = {
        "operation": "input-corrector-audit",
        "runtime": "ProductionCorrectionRuntime",
        "corrector": corrector,
        "python": "3.12.0",
        "requested_revision": requested_revision,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_code": IMPLEMENTATION_CODE,
    }
    if requested_revision is None:
        common.update(
            {
                "profile": "pyspellchecker",
                "pyspellchecker": "0.9.0",
                "device": "cpu",
                "dictionary_language": "en",
                "dictionary_sha256": "2" * 64,
                "model_revision": None,
                "tokenizer_revision": None,
            }
        )
        return common
    common.update(_neural_environment())
    common.update(
        {
            "model_revision": requested_revision,
            "model_revision_source": "model-config-metadata",
            "tokenizer_revision": requested_revision,
            "tokenizer_revision_source": "tokenizer-init-metadata",
        }
    )
    return common


def _generation_runtime(*, model: str, benchmark: str) -> dict[str, object]:
    revision = "1" * 40
    generation = {
        "padding_side": "left",
        **{
            key: value
            for key, value in GENERATION.items()
            if key not in {"strategy", "dtype", "padding_side"}
        },
        "num_return_sequences": 1,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
    }
    return {
        "operation": "input-corrector-audit",
        "runtime": "HuggingFaceSamePromptRuntime",
        **_neural_environment(),
        "model": model,
        "requested_revision": revision,
        "model_revision": revision,
        "model_revision_source": "model-config-metadata",
        "tokenizer_revision": revision,
        "tokenizer_revision_source": "tokenizer-init-metadata",
        "dtype": "bfloat16",
        "protocol_sha256": PROTOCOL_SHA256,
        "generation": generation,
        "effective_eos_token_ids": [1],
        "effective_eos_token_ids_source": "model-generation-config",
        "answer_extraction": "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1",
        "benchmark_extractor": _BENCHMARK_EXTRACTORS.get(benchmark, "gsm8k"),
        "implementation_code": IMPLEMENTATION_CODE,
    }


def _setting_index(model: str, benchmark: str) -> int:
    return MODELS.index(model) * len(CORE_BENCHMARKS) + CORE_BENCHMARKS.index(benchmark)


def _word_counts(*, setting_index: int, corrector: str) -> tuple[int, int]:
    """Make every setting equally weighted but make pooling visibly different."""

    total = 1 if setting_index == 0 else 100
    last_restored_setting = {
        "pyspellchecker": 0,
        "t5-large-spell": 1,
        "qwen2.5-7b-instruct": 2,
    }[corrector]
    restored = total if setting_index <= last_restored_setting else 0
    return restored, total


def _is_exact_clean(*, setting_index: int, corrector: str) -> bool:
    if corrector == "pyspellchecker":
        return True
    if corrector == "t5-large-spell":
        return setting_index % 2 == 0
    return setting_index < 5


def _same_changed(*, setting_index: int, corrector: str) -> bool:
    changed = {
        "pyspellchecker": {0},
        "t5-large-spell": {0, 2},
        "qwen2.5-7b-instruct": {0, 1, 2},
    }
    return setting_index in changed[corrector]


def _separate_source_changed(*, setting_index: int, corrector: str) -> bool:
    changed = {
        "pyspellchecker": {0, 1},
        "t5-large-spell": {0, 2, 4},
        "qwen2.5-7b-instruct": {0, 1, 2, 3},
    }
    return setting_index in changed[corrector]


def _generation_evidence(*, sample_id: str, answer: str, token: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "token_ids": [token],
        "text": f"The answer is {answer}.",
        "extracted_answer": answer,
        "is_extracted": True,
        "is_correct": True,
        "method": "primary:fixture",
        "primary_method": "fixture",
    }


def _core_record(*, model: str, benchmark: str, corrector: str) -> dict[str, object]:
    index = _setting_index(model, benchmark)
    restored, total = _word_counts(setting_index=index, corrector=corrector)
    exact = _is_exact_clean(setting_index=index, corrector=corrector)
    clean_prompt_sha256 = _prompt_sha256(f"clean prompt {model} {benchmark} {index}")
    corrected_prompt_sha256 = (
        clean_prompt_sha256
        if exact
        else _prompt_sha256(f"non-exact corrected prompt {model} {benchmark} {index}")
    )
    same = None
    separate_source = None
    sample_id = f"{benchmark}_synthetic_000"
    if exact:
        first_answer = "same-answer"
        duplicate_answer = (
            "different-answer"
            if _same_changed(setting_index=index, corrector=corrector)
            else first_answer
        )
        same = {
            "protocol": "adjacent-duplicate-prompt-pair/v1",
            "first_extracted_answer": first_answer,
            "duplicate_extracted_answer": duplicate_answer,
            "first": _generation_evidence(
                sample_id=sample_id,
                answer=first_answer,
                token=1,
            ),
            "duplicate": _generation_evidence(
                sample_id=sample_id,
                answer=duplicate_answer,
                token=2,
            ),
        }
        source_answer = (
            "different-archive-answer"
            if _separate_source_changed(setting_index=index, corrector=corrector)
            else duplicate_answer
        )
        separate_source = {
            # This fresh diagnostic compares the same-batch corrected endpoint
            # with the clean answer stored by the separate pair-source run. It
            # is deliberately not named or accepted as the paper archive cell.
            "comparison": "same_batch_corrected_vs_source_pair_clean",
            "same_batch_corrected_extracted_answer": duplicate_answer,
            "source_pair_clean_extracted_answer": source_answer,
        }
    corrected_text = f"corrected {model} {benchmark}"
    return {
        "schema_version": "input-corrector-audit-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "input-corrector-audit",
        "model": model,
        "benchmark": benchmark,
        "corrector": corrector,
        "sample_id": sample_id,
        "source_record_sha256": _prompt_sha256(f"source {model} {benchmark} {sample_id}"),
        "correction": {
            "corrected_text": corrected_text,
            "parse_failed": False,
            "n_calls": 1,
            "raw_response": None,
            "input_sha256": _prompt_sha256(f"edited {model} {benchmark} {sample_id}"),
            "corrected_sha256": _prompt_sha256(corrected_text),
        },
        "edited_words": {"restored": restored, "total": total, "unalignable": 0},
        "prompt_endpoints": {
            "clean_sha256": clean_prompt_sha256,
            "corrected_sha256": corrected_prompt_sha256,
            "exact_utf8": exact,
        },
        "same_batch_answers": same,
        "separate_source_answers": separate_source,
        "diagnostics": {
            "whitespace_normalized_full": exact,
            "all_perturbed_restored": restored == total,
            "intact_word_changes": 0,
            "collateral_changes": [],
        },
    }


def _math_record(*, model: str, corrector: str) -> dict[str, object]:
    model_index = MODELS.index(model)
    collateral = (
        (1, 2, 3, 4, 5)[model_index]
        if corrector == "t5-large-spell"
        else (0, 1, 0, 1, 0)[model_index]
    )
    prompt_sha256 = _prompt_sha256(f"math prompt {model}")
    corrected_prompt_sha256 = (
        prompt_sha256
        if collateral == 0
        else _prompt_sha256(f"math corrected prompt {model} {corrector}")
    )
    sample_id = "math_synthetic_000"
    corrected_text = f"math corrected {model}"
    return {
        "schema_version": "input-corrector-audit-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "input-corrector-audit",
        "model": model,
        "benchmark": "math-500",
        "corrector": corrector,
        "sample_id": sample_id,
        "source_record_sha256": _prompt_sha256(f"source {model} math {sample_id}"),
        "correction": {
            "corrected_text": corrected_text,
            "parse_failed": False,
            "n_calls": 1,
            "raw_response": None,
            "input_sha256": _prompt_sha256(f"edited {model} math {sample_id}"),
            "corrected_sha256": _prompt_sha256(corrected_text),
        },
        "edited_words": {"restored": 1, "total": 1, "unalignable": 0},
        "prompt_endpoints": {
            "clean_sha256": prompt_sha256,
            "corrected_sha256": corrected_prompt_sha256,
            "exact_utf8": collateral == 0,
        },
        "same_batch_answers": None,
        "separate_source_answers": None,
        "diagnostics": {
            "whitespace_normalized_full": collateral == 0,
            "all_perturbed_restored": True,
            "intact_word_changes": collateral,
            "collateral_changes": [
                {
                    "word_index": offset,
                    "clean": f"clean-{offset}",
                    "corrected": f"corrected-{offset}",
                }
                for offset in range(collateral)
            ],
        },
    }


def _derive_setting_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    word_restored = 0
    word_total = 0
    exact_records: list[Mapping[str, object]] = []
    intact_word_changes = 0
    for record in records:
        edited = record["edited_words"]
        assert isinstance(edited, Mapping)
        word_restored += int(edited["restored"])
        word_total += int(edited["total"])
        prompts = record["prompt_endpoints"]
        assert isinstance(prompts, Mapping)
        if prompts["clean_sha256"] == prompts["corrected_sha256"]:
            exact_records.append(record)
        diagnostics = record["diagnostics"]
        assert isinstance(diagnostics, Mapping)
        intact_word_changes += int(diagnostics["intact_word_changes"])

    same_changed = 0
    separate_source_changed = 0
    for record in exact_records:
        same = record["same_batch_answers"]
        separate = record["separate_source_answers"]
        if record["benchmark"] == "math-500" and same is None and separate is None:
            continue
        assert isinstance(same, Mapping)
        assert isinstance(separate, Mapping)
        same_changed += same["first_extracted_answer"] != same["duplicate_extracted_answer"]
        separate_source_changed += (
            separate["same_batch_corrected_extracted_answer"]
            != separate["source_pair_clean_extracted_answer"]
        )
    return {
        "records": len(records),
        "word_restored": word_restored,
        "word_total": word_total,
        "word_restoration_rate": word_restored / word_total,
        "exact_clean": len(exact_records),
        "same_changed": same_changed,
        "separate_source_changed": separate_source_changed,
        "intact_word_changes": intact_word_changes,
    }


def _safe_name(value: str) -> str:
    return value.replace("/", "--").replace("_", "-")


def _write_completed_run(
    directory: Path,
    *,
    model: str,
    benchmark: str,
    corrector: str,
    records: Sequence[Mapping[str, object]],
    status: str = "completed",
    limit: int | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / RECORDS_NAME
    summary_path = directory / SETTING_SUMMARY_NAME
    _write_jsonl(records_path, records)
    metrics = _derive_setting_metrics(records)
    source = _source_identity(model=model, benchmark=benchmark, records=records)
    runtime: dict[str, object] = {"correction": _correction_runtime(corrector)}
    if benchmark != "math-500" and int(metrics["exact_clean"]) > 0:
        runtime["generation"] = _generation_runtime(model=model, benchmark=benchmark)
    _write_json(
        summary_path,
        {
            "schema_version": "input-corrector-audit-summary/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "input-corrector-audit",
            "status": status,
            "scope": "paper-setting",
            "model": model,
            "benchmark": benchmark,
            "corrector": corrector,
            "metrics": metrics,
        },
    )
    run_path = directory / "run.json"
    _write_json(
        run_path,
        {
            "schema_version": "input-corrector-audit-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "input-corrector-audit",
            "status": status,
            "scope": "paper-setting",
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_code": IMPLEMENTATION_CODE,
            "arguments": {
                "model": model,
                "benchmark": benchmark,
                "corrector": corrector,
                "pairs": f"/paper-inputs/{_safe_name(model)}--{benchmark}/pairs.jsonl",
                "gpu_id": "0",
                "limit": limit,
                "output_dir": str(directory.resolve()),
            },
            "source": source,
            "selected_records": len(records),
            "runtime": runtime,
            "checkpoints": {},
            "outputs": {
                RECORDS_NAME: _output_metadata(records_path, records=len(records)),
                SETTING_SUMMARY_NAME: _output_metadata(summary_path),
            },
            "started_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:01:00+00:00",
            "failure": None,
        },
    )
    return run_path


def _write_grid(
    runs_root: Path,
    *,
    omitted: Iterable[Setting] = (),
    include_core: bool = True,
    include_math: bool = False,
    omitted_math: Iterable[Setting] = (),
) -> dict[Setting, Path]:
    omitted_set = set(omitted)
    omitted_math_set = set(omitted_math)
    paths: dict[Setting, Path] = {}
    if include_core:
        for model in MODELS:
            for benchmark in CORE_BENCHMARKS:
                for corrector in CORE_CORRECTORS:
                    setting = (model, benchmark, corrector)
                    if setting in omitted_set:
                        continue
                    directory = runs_root / (f"core--{_safe_name(model)}--{benchmark}--{corrector}")
                    paths[setting] = _write_completed_run(
                        directory,
                        model=model,
                        benchmark=benchmark,
                        corrector=corrector,
                        records=[
                            _core_record(
                                model=model,
                                benchmark=benchmark,
                                corrector=corrector,
                            )
                        ],
                    )
    if include_math:
        for model in MODELS:
            for corrector in MATH_DIAGNOSTIC_CORRECTORS:
                setting = (model, "math-500", corrector)
                if setting in omitted_math_set:
                    continue
                directory = runs_root / f"math--{_safe_name(model)}--{corrector}"
                paths[setting] = _write_completed_run(
                    directory,
                    model=model,
                    benchmark="math-500",
                    corrector=corrector,
                    records=[_math_record(model=model, corrector=corrector)],
                )
    return paths


def _refresh_manifest_output(run_path: Path, output_name: str) -> None:
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    output_path = run_path.parent / output_name
    records = None
    if output_name == RECORDS_NAME:
        records = sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line)
    manifest["outputs"][output_name] = _output_metadata(output_path, records=records)
    _write_json(run_path, manifest)


def _build(
    runs_root: Path,
    output_dir: Path,
    *,
    math_runs_root: Path | None = None,
) -> object:
    return run_build_input_corrector_summary(
        BuildInputCorrectorSummaryConfig(
            runs_root=runs_root,
            math_runs_root=math_runs_root,
            output_dir=output_dir,
        )
    )


def test_core_summary_uses_setting_means_and_recomputes_record_endpoints(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "summary"
    _write_grid(runs_root)

    _build(runs_root, output_dir)

    assert {path.name for path in output_dir.iterdir()} == BUILDER_OUTPUTS
    payload = json.loads((output_dir / "input_corrector_summary.json").read_text())
    assert payload["coverage"]["core"] == {
        "complete_grid": True,
        "expected_settings": 75,
        "present_settings": 75,
        "missing_settings": [],
        "unexpected_settings": [],
    }

    dictionary = payload["methods"]["pyspellchecker"]
    t5 = payload["methods"]["t5-large-spell"]
    qwen = payload["methods"]["qwen2.5-7b-instruct"]
    assert dictionary["word"]["setting_mean_exact_restoration"] == pytest.approx(1 / 25)
    assert dictionary["word"]["pooled_exact_restoration"] == pytest.approx(1 / 2401)
    assert dictionary["word"]["setting_mean_exact_restoration"] != pytest.approx(
        dictionary["word"]["pooled_exact_restoration"]
    )
    assert t5["word"]["setting_mean_exact_restoration"] == pytest.approx(2 / 25)
    assert qwen["word"]["setting_mean_exact_restoration"] == pytest.approx(3 / 25)

    assert dictionary["exact_clean"] == 25
    assert dictionary["same"] == {"denominator": 25, "changed": 1, "rate": 1 / 25}
    assert dictionary["separate_source"] == {
        "denominator": 25,
        "changed": 2,
        "rate": 2 / 25,
    }
    assert t5["exact_clean"] == 13
    assert t5["same"] == {"denominator": 13, "changed": 2, "rate": 2 / 13}
    assert t5["separate_source"] == {
        "denominator": 13,
        "changed": 3,
        "rate": 3 / 13,
    }
    assert qwen["exact_clean"] == 5
    assert qwen["same"] == {"denominator": 5, "changed": 3, "rate": 3 / 5}
    assert qwen["separate_source"] == {
        "denominator": 5,
        "changed": 4,
        "rate": 4 / 5,
    }

    with (output_dir / "table12_input_correctors.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    assert set(rows) == set(CORE_CORRECTORS)
    assert float(rows["pyspellchecker"]["word_setting_mean"]) == pytest.approx(1 / 25)
    assert int(rows["pyspellchecker"]["exact_clean"]) == 25
    assert int(rows["pyspellchecker"]["same_changed"]) == 1
    assert int(rows["pyspellchecker"]["separate_source_changed"]) == 2

    markdown = (output_dir / "table12_input_correctors.md").read_text(encoding="utf-8")
    latex = (output_dir / "table12_input_correctors.tex").read_text(encoding="utf-8")
    for method in CORE_CORRECTORS:
        assert method in markdown
        assert method in latex

    builder_run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert builder_run["status"] == "completed"
    assert builder_run["operation"] == "build-input-corrector-summary"
    assert builder_run["analysis_code"] == analysis_code_identity()
    for output_name in BUILDER_OUTPUTS - {"run.json"}:
        metadata = builder_run["outputs"][output_name]
        path = output_dir / output_name
        assert metadata["sha256"] == _sha256_file(path)
        assert metadata["bytes"] == path.stat().st_size


def test_published_archive_counts_are_context_not_a_fresh_run_acceptance_target(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "summary"
    _write_grid(runs_root)

    _build(runs_root, output_dir)

    payload = json.loads((output_dir / "input_corrector_summary.json").read_text())
    assert payload["published_reference"]["methods"] == {
        "pyspellchecker": {"exact_clean": 7548, "same_changed": 0, "archive_changed": 708},
        "t5-large-spell": {"exact_clean": 21306, "same_changed": 0, "archive_changed": 1874},
        "qwen2.5-7b-instruct": {"exact_clean": 16787, "same_changed": 0, "archive_changed": 1780},
    }
    assert payload["comparability"]["reference_is_acceptance_target"] is False
    assert payload["methods"]["pyspellchecker"]["separate_source"]["changed"] == 2
    comparison = payload["published_comparison"]
    assert comparison["role"] == "descriptive-comparable-cells-only"
    assert comparison["all_comparable_integer_cells_match"] is False
    dictionary = comparison["by_method"]["pyspellchecker"]
    assert dictionary["exact_clean"] is False
    assert dictionary["same_changed"] is False
    assert dictionary["archive"] == {
        "comparable": False,
        "published_changed": 708,
        "fresh_endpoint": "separate_source",
    }
    assert "separate_source_vs_historical_archive" not in dictionary


def test_builder_accepts_no_math_diagnostic_runs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "summary"
    _write_grid(runs_root)

    _build(runs_root, output_dir)

    payload = json.loads((output_dir / "input_corrector_summary.json").read_text())
    assert payload["math_intact_word_changes"] == {
        "status": "not-run",
        "expected_settings": 10,
        "present_settings": 0,
        "methods": {},
    }


def test_complete_math_diagnostic_grid_is_separate_from_the_core_table(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    math_runs_root = tmp_path / "math-runs"
    output_dir = tmp_path / "summary"
    _write_grid(runs_root)
    _write_grid(math_runs_root, include_core=False, include_math=True)

    _build(runs_root, output_dir, math_runs_root=math_runs_root)

    payload = json.loads((output_dir / "input_corrector_summary.json").read_text())
    assert payload["coverage"]["core"]["present_settings"] == 75
    assert payload["methods"]["pyspellchecker"]["word"][
        "setting_mean_exact_restoration"
    ] == pytest.approx(1 / 25)
    diagnostic = payload["math_intact_word_changes"]
    assert diagnostic["status"] == "complete-diagnostic-grid"
    assert diagnostic["present_settings"] == 10
    assert diagnostic["methods"]["t5-large-spell"] == {
        "settings": 5,
        "items": 5,
        "intact_word_changes": 15,
        "changes_per_item": 3.0,
    }
    assert diagnostic["methods"]["qwen2.5-7b-instruct"] == {
        "settings": 5,
        "items": 5,
        "intact_word_changes": 2,
        "changes_per_item": 0.4,
    }


def test_builder_rejects_a_missing_core_setting_without_publishing(tmp_path: Path) -> None:
    missing = (MODELS[-1], CORE_BENCHMARKS[-1], CORE_CORRECTORS[-1])
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "summary"
    _write_grid(runs_root, omitted=[missing])

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, output_dir)

    assert not output_dir.exists()


def test_builder_rejects_a_duplicate_core_setting(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    setting = (MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])
    original_record = _core_record(model=setting[0], benchmark=setting[1], corrector=setting[2])
    original_record["sample_id"] = "duplicate-setting-record"
    _write_completed_run(
        runs_root / "duplicate-setting",
        model=setting[0],
        benchmark=setting[1],
        corrector=setting[2],
        records=[original_record],
    )
    assert paths[setting].is_file()

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary")


@pytest.mark.parametrize(
    ("model", "benchmark", "corrector"),
    [
        ("unexpected/model", "gsm8k", "pyspellchecker"),
        (MODELS[0], "unexpected_task", "pyspellchecker"),
        (MODELS[0], "gsm8k", "unexpected-corrector"),
    ],
)
def test_builder_rejects_unexpected_core_identity(
    tmp_path: Path,
    model: str,
    benchmark: str,
    corrector: str,
) -> None:
    runs_root = tmp_path / "runs"
    _write_grid(runs_root)
    record = _core_record(
        model=MODELS[0], benchmark=CORE_BENCHMARKS[0], corrector=CORE_CORRECTORS[0]
    )
    record["model"] = model
    record["benchmark"] = benchmark
    record["corrector"] = corrector
    _write_completed_run(
        runs_root / "unexpected-setting",
        model=model,
        benchmark=benchmark,
        corrector=corrector,
        records=[record],
    )

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary")


@pytest.mark.parametrize(("status", "limit"), [("running", None), ("completed", 1)])
def test_builder_rejects_partial_core_runs(
    tmp_path: Path,
    status: str,
    limit: int | None,
) -> None:
    runs_root = tmp_path / "runs"
    setting = (MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])
    _write_grid(runs_root, omitted=[setting])
    directory = runs_root / "replacement-partial-setting"
    _write_completed_run(
        directory,
        model=setting[0],
        benchmark=setting[1],
        corrector=setting[2],
        records=[_core_record(model=setting[0], benchmark=setting[1], corrector=setting[2])],
        status=status,
        limit=limit,
    )

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_duplicate_sample_ids_even_when_counts_and_hashes_match(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    omitted = (MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])
    _write_grid(runs_root, omitted=[omitted])
    record = _core_record(model=omitted[0], benchmark=omitted[1], corrector=omitted[2])
    _write_completed_run(
        runs_root / "duplicate-records",
        model=omitted[0],
        benchmark=omitted[1],
        corrector=omitted[2],
        records=[record, dict(record)],
    )

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_stale_record_digest(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    records_path = run_path.parent / RECORDS_NAME
    records_path.write_text(records_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_an_exact_utf8_flag_that_disagrees_with_prompt_hashes(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    records_path = run_path.parent / RECORDS_NAME
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["prompt_endpoints"]["exact_utf8"] = False
    _write_jsonl(records_path, records)
    _refresh_manifest_output(run_path, RECORDS_NAME)

    with pytest.raises(InputCorrectorSummaryInputError, match="exact|prompt|endpoint"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_a_record_missing_source_and_correction_evidence(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    records_path = run_path.parent / RECORDS_NAME
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    del records[0]["source_record_sha256"]
    del records[0]["correction"]
    _write_jsonl(records_path, records)
    _refresh_manifest_output(run_path, RECORDS_NAME)

    with pytest.raises(InputCorrectorSummaryInputError, match="record|source|correction"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_nested_same_evidence_that_disagrees_with_top_level_answer(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    records_path = run_path.parent / RECORDS_NAME
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["same_batch_answers"]["first"]["extracted_answer"] = "tampered"
    _write_jsonl(records_path, records)
    _refresh_manifest_output(run_path, RECORDS_NAME)

    with pytest.raises(InputCorrectorSummaryInputError, match="Same|nested|answer|generation"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_internally_inconsistent_correction_or_diagnostics(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    records_path = run_path.parent / RECORDS_NAME
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["correction"]["corrected_sha256"] = "9" * 64
    records[0]["diagnostics"]["intact_word_changes"] = 1
    _write_jsonl(records_path, records)
    _refresh_manifest_output(run_path, RECORDS_NAME)

    with pytest.raises(InputCorrectorSummaryInputError, match="correction|diagnostic|collateral"):
        _build(runs_root, tmp_path / "summary")


@pytest.mark.parametrize("field", ("protocol_sha256", "implementation_code"))
def test_builder_rejects_protocol_or_producer_code_drift(
    tmp_path: Path,
    field: str,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if field == "protocol_sha256":
        manifest[field] = "0" * 64
    else:
        manifest[field]["sha256"] = "0" * 64
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="protocol|code|implementation"):
        _build(runs_root, tmp_path / "summary")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scope", "custom-smoke", "scope|paper-setting"),
        ("selected_records", 0, "selected_records|record"),
        ("failure", "ignored failure", "failure|completed"),
        ("checkpoints", {"corrections": 1}, "checkpoint"),
    ],
)
def test_builder_rejects_incomplete_completed_manifest_state(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match=message):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_a_fake_correction_runtime(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["runtime"]["correction"]["runtime"] = "FakeCorrectionRuntime"
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="correction|runtime|production"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_neural_corrector_revision_drift(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], "t5-large-spell")]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["runtime"]["correction"]["requested_revision"] = "9" * 40
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="correction|revision|pin"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_runtime_gpu_that_differs_from_manifest_argument(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], "mmlu", "t5-large-spell")]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["arguments"]["gpu_id"] = "1"
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="GPU|visible|argument"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_missing_same_prompt_runtime_provenance(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], "pyspellchecker")]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    del manifest["runtime"]["generation"]
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="generation|runtime|provenance"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_source_identity_that_disagrees_with_output_sample_ids(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["source"]["ordered_sample_ids_sha256"] = "9" * 64
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="source|sample|cohort"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_a_truncated_declared_paper_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.input_corrector_audit.aggregation as aggregation_module

    runs_root = tmp_path / "runs"
    _write_grid(runs_root)
    monkeypatch.setattr(
        aggregation_module,
        "PAPER_BENCHMARK_ITEM_COUNTS",
        {benchmark: 2 if benchmark == "gsm8k" else 1 for benchmark in CORE_BENCHMARKS},
    )

    with pytest.raises(InputCorrectorSummaryInputError, match="paper.*cohort|record"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_different_source_cohorts_across_correctors(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], "t5-large-spell")]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["source"]["pairs_sha256"] = "9" * 64
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="source|cohort|corrector"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_source_record_drift_across_correctors(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], "t5-large-spell")]
    records_path = run_path.parent / RECORDS_NAME
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["source_record_sha256"] = "9" * 64
    _write_jsonl(records_path, records)
    _refresh_manifest_output(run_path, RECORDS_NAME)

    with pytest.raises(InputCorrectorSummaryInputError, match="source|cohort|corrector"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_model_revision_drift_across_benchmarks(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    revision = "8" * 40
    for corrector in CORE_CORRECTORS:
        run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], corrector)]
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        manifest["source"]["model_revision"] = revision
        generation = manifest["runtime"].get("generation")
        if generation is not None:
            generation["requested_revision"] = revision
            generation["model_revision"] = revision
            generation["tokenizer_revision"] = revision
        _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="revision|benchmark|model"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_dataset_snapshot_drift_across_models(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    for corrector in CORE_CORRECTORS:
        run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], corrector)]
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        manifest["source"]["dataset_records_sha256"] = "8" * 64
        _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="dataset|cohort|benchmark"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_math_evaluator_revision_that_differs_from_core(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    math_runs_root = tmp_path / "math-runs"
    _write_grid(runs_root)
    paths = _write_grid(math_runs_root, include_core=False, include_math=True)
    for corrector in MATH_DIAGNOSTIC_CORRECTORS:
        run_path = paths[(MODELS[0], "math-500", corrector)]
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        manifest["source"]["model_revision"] = "8" * 40
        _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="MATH|core|revision"):
        _build(runs_root, tmp_path / "summary", math_runs_root=math_runs_root)


def test_builder_rejects_dictionary_identity_drift_across_settings(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], "pyspellchecker")]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["runtime"]["correction"]["dictionary_sha256"] = "9" * 64
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="dictionary|pyspellchecker"):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_generation_runtime_in_math_only_diagnostic(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    math_runs_root = tmp_path / "math-runs"
    _write_grid(runs_root)
    paths = _write_grid(math_runs_root, include_core=False, include_math=True)
    run_path = paths[(MODELS[0], "math-500", "t5-large-spell")]
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["runtime"]["generation"] = _generation_runtime(
        model=MODELS[0],
        benchmark="gsm8k",
    )
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSummaryInputError, match="MATH|generation|runtime"):
        _build(runs_root, tmp_path / "summary", math_runs_root=math_runs_root)


def test_builder_accepts_a_relocated_completed_run_tree(tmp_path: Path) -> None:
    original_root = tmp_path / "original-runs"
    relocated_root = tmp_path / "relocated-runs"
    _write_grid(original_root)
    original_root.rename(relocated_root)

    result = _build(relocated_root, tmp_path / "summary")

    assert result.settings == 75


def test_builder_fails_closed_if_a_producer_output_changes_during_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.input_corrector_audit.aggregation as aggregation_module

    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    records_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])].parent / (
        RECORDS_NAME
    )
    original_write_text = aggregation_module._write_text
    mutated = False

    def mutate_input_then_write(path: Path, value: str) -> None:
        nonlocal mutated
        if not mutated:
            records_path.write_bytes(records_path.read_bytes() + b"\n")
            mutated = True
        original_write_text(path, value)

    monkeypatch.setattr(aggregation_module, "_write_text", mutate_input_then_write)

    with pytest.raises(InputCorrectorSummaryInputError, match="changed|hash|producer"):
        _build(runs_root, tmp_path / "summary")

    assert not (tmp_path / "summary").exists()


def test_builder_recomputes_records_and_rejects_a_rehashed_summary_tamper(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    paths = _write_grid(runs_root)
    run_path = paths[(MODELS[0], CORE_BENCHMARKS[0], CORE_CORRECTORS[0])]
    summary_path = run_path.parent / SETTING_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["exact_clean"] = 0
    _write_json(summary_path, summary)
    _refresh_manifest_output(run_path, SETTING_SUMMARY_NAME)

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary")


def test_builder_rejects_a_partial_math_diagnostic_grid(tmp_path: Path) -> None:
    omitted = (MODELS[-1], "math-500", MATH_DIAGNOSTIC_CORRECTORS[-1])
    runs_root = tmp_path / "runs"
    math_runs_root = tmp_path / "math-runs"
    _write_grid(runs_root)
    _write_grid(
        math_runs_root,
        include_core=False,
        include_math=True,
        omitted_math=[omitted],
    )

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary", math_runs_root=math_runs_root)


def test_builder_rejects_dictionary_math_as_outside_the_submitted_diagnostic(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    math_runs_root = tmp_path / "math-runs"
    _write_grid(runs_root)
    _write_grid(math_runs_root, include_core=False, include_math=True)
    record = _math_record(model=MODELS[0], corrector="t5-large-spell")
    record["corrector"] = "pyspellchecker"
    _write_completed_run(
        math_runs_root / "unexpected-math-dictionary",
        model=MODELS[0],
        benchmark="math-500",
        corrector="pyspellchecker",
        records=[record],
    )

    with pytest.raises(InputCorrectorSummaryInputError):
        _build(runs_root, tmp_path / "summary", math_runs_root=math_runs_root)


def test_builder_rejects_same_generation_fields_in_the_math_only_diagnostic(
    tmp_path: Path,
) -> None:
    setting = (MODELS[0], "math-500", "t5-large-spell")
    runs_root = tmp_path / "runs"
    math_runs_root = tmp_path / "math-runs"
    _write_grid(runs_root)
    _write_grid(
        math_runs_root,
        include_core=False,
        include_math=True,
        omitted_math=[setting],
    )
    record = _math_record(model=setting[0], corrector=setting[2])
    record["same_batch_answers"] = {
        "first_extracted_answer": "1",
        "duplicate_extracted_answer": "1",
    }
    record["separate_source_answers"] = {
        "same_batch_corrected_extracted_answer": "1",
        "source_pair_clean_extracted_answer": "1",
    }
    _write_completed_run(
        math_runs_root / "replacement-with-generation",
        model=setting[0],
        benchmark=setting[1],
        corrector=setting[2],
        records=[record],
    )

    with pytest.raises(InputCorrectorSummaryInputError, match="MATH|Same|generation"):
        _build(runs_root, tmp_path / "summary", math_runs_root=math_runs_root)


def test_input_error_type_is_public() -> None:
    assert issubclass(InputCorrectorSummaryInputError, ValueError)
