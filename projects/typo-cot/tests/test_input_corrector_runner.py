"""Runner contracts for the Appendix E input-corrector audit.

The fixtures are completed ``prepare-edited-pairs`` runs.  Both runtime seams
are CPU fakes: the tests exercise prompt splicing, exact-byte selection,
legacy restoration accounting, complete-batch checkpointing, and publication
integrity without importing model libraries.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.input_corrector_audit.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.input_corrector_audit.runner import (
    CorrectionOutcome,
    InputCorrectorAuditConfig,
    InputCorrectorAuditResult,
    InputCorrectorAuditRunError,
    SamePromptGeneration,
    run_input_corrector_audit,
)


MODEL = "google/gemma-3-1b-it"
RECORDS_NAME = "corrector_records.jsonl"
SUMMARY_NAME = "corrector_audit_summary.json"

_CASES = (
    {
        "sample_id": "sample-a",
        "clean": "Alpha",
        "edited": "Alpga",
        "operation": "substitution",
        "clean_answer": "A",
    },
    {
        "sample_id": "sample-b",
        "clean": "Beta",
        "edited": "Beeta",
        "operation": "duplication",
        "clean_answer": "C",
    },
    {
        "sample_id": "sample-c",
        "clean": "Gamma",
        "edited": "Gbmma",
        "operation": "substitution",
        "clean_answer": "D",
    },
    {
        "sample_id": "sample-d",
        "clean": "Delta stable",
        "edited": "Ddlta stable",
        "operation": "substitution",
        "clean_answer": "B",
    },
    {
        "sample_id": "sample-e",
        "clean": "Epsilon",
        "edited": "Epsilom",
        "operation": "substitution",
        "clean_answer": "A",
    },
)

_SUBMITTED_CORRECTIONS = {
    "Alpga": "Alpha",
    "Beeta": "Beta",
    # Whitespace-normalized full restoration is deliberately not byte exact.
    "Gbmma": "Gamma ",
    # The targeted word is restored, while one intact word is changed.
    "Ddlta stable": "Delta altered",
    "Epsilom": "Epsilom",
}


@pytest.fixture(autouse=True)
def _use_tiny_structurally_valid_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep runner unit fixtures small; production uses the strict default."""
    import typo_cot.experiments.input_corrector_audit.runner as runner_module
    import typo_cot.experiments.input_corrector_audit.source as source_module

    def load_tiny_source(
        pairs_path: Path,
        *,
        model: str,
        benchmark: str,
    ) -> object:
        return source_module.load_input_corrector_source(
            pairs_path,
            model=model,
            benchmark=benchmark,
            require_paper_cohort_size=False,
        )

    monkeypatch.setattr(runner_module, "load_input_corrector_source", load_tiny_source)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_sha256(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _answer(value: str, *, correct: bool) -> dict[str, object]:
    return {
        "value": value,
        "is_extracted": True,
        "is_correct": correct,
        "method": "primary:fixture",
        "primary_method": "fixture",
        "confidence": 1.0,
    }


def _pair_record(
    case: Mapping[str, str],
    *,
    benchmark: str,
) -> dict[str, object]:
    sample_id = case["sample_id"]
    clean_editable = case["clean"]
    edited_editable = case["edited"]
    clean_word = clean_editable.split()[0]
    edited_word = edited_editable.split()[0]
    prefix = f"FEWSHOT::{sample_id}\nQuestion: "
    suffix = "\nAnswer:"
    clean_prompt = prefix + clean_editable + suffix
    edited_prompt = prefix + edited_editable + suffix
    editable_start = len(prefix)
    clean_end = editable_start + len(clean_editable)
    edited_end = editable_start + len(edited_editable)
    operation = case["operation"]
    character_index = (
        1
        if operation == "duplication"
        else next(
            index
            for index, (clean_character, edited_character) in enumerate(
                zip(clean_word, edited_word, strict=True)
            )
            if clean_character != edited_character
        )
    )
    new_character = (
        clean_word[character_index] if operation == "duplication" else edited_word[character_index]
    )
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": MODEL,
        "benchmark": benchmark,
        "targeting": "attribution-4",
        "seed": 42,
        "num_edits_requested": 4,
        "num_candidates": 4,
        "num_target_attempts": 1,
        "num_aligned_words": 1,
        "gold_answer": case["clean_answer"],
        "subset": "fixture",
        "attribution_target": {
            "definition": "maximum-logit-after-first-cot-token",
            "position": 9,
            "first_cot_token_id": 123,
            "first_cot_token_text": " First",
            "context": "complete-clean-generation",
        },
        "clean": {
            "question": clean_editable,
            "choices": None,
            "editable_text": clean_editable,
            "editable_prompt_span": {"start": editable_start, "end": clean_end},
            "prompt": clean_prompt,
            "prompt_token_count": 12,
            "continuation": f"Source clean answer: {case['clean_answer']}",
            "continuation_token_count": 4,
            "termination": "eos",
            "answer": _answer(case["clean_answer"], correct=True),
        },
        "edited": {
            "editable_text": edited_editable,
            "editable_prompt_span": {"start": editable_start, "end": edited_end},
            "prompt": edited_prompt,
            "prompt_token_count": 12,
            "continuation": "Source edited answer: Z",
            "continuation_token_count": 4,
            "termination": "eos",
            "answer": _answer("Z", correct=False),
        },
        "answer_changed": True,
        "excluded_attribution_tokens": [],
        "target_attempts": [
            {
                "selection_rank": 1,
                "attribution_rank": 1,
                "target_token_index": 4,
                "target_token_text": clean_word,
                "relevance": 1.0,
                "intended_prompt_span": {
                    "start": editable_start,
                    "end": editable_start + len(clean_word),
                },
                "intended_editable_span": {"start": 0, "end": len(clean_word)},
                "landed_editable_span_before": {"start": 0, "end": len(clean_word)},
                "landed_text_before": clean_word,
                "landed_origin_index": character_index,
                "landed_on_intended_token": True,
                "intended_word_index": 0,
                "landed_word_index": 0,
                "operation": operation,
                "character_index": character_index,
                "original_character": clean_word[character_index],
                "new_character": new_character,
                "edited_token_text": edited_word,
            }
        ],
        "aligned_words": [
            {
                "word_index": 0,
                "clean_text": clean_word,
                "edited_text": edited_word,
                "clean_editable_span": {"start": 0, "end": len(clean_word)},
                "edited_editable_span": {"start": 0, "end": len(edited_word)},
                "clean_prompt_span": {
                    "start": editable_start,
                    "end": editable_start + len(clean_word),
                },
                "edited_prompt_span": {
                    "start": editable_start,
                    "end": editable_start + len(edited_word),
                },
                "target_ranks": [1],
                "target_token_indices": [4],
                "clean_token_indices": [4],
                "edited_token_indices": [4],
                "clean_final_token": 4,
                "edited_final_token": 4,
            }
        ],
    }


def _dataset_loader(benchmark: str) -> str:
    return {
        "gsm8k": "gsm8k",
        "mmlu": "mmlu",
        "mmlu-pro": "mmlu_pro",
        "arc": "arc",
        "csqa": "commonsense_qa",
        "math-500": "math",
    }[benchmark]


def _source_manifest(
    directory: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    benchmark: str,
) -> dict[str, object]:
    dataset_identity = [
        {
            "sample_id": row["sample_id"],
            "question": row["clean"]["question"],  # type: ignore[index]
            "choices": row["clean"]["choices"],  # type: ignore[index]
            "correct_answer": row["gold_answer"],
            "subset": row["subset"],
        }
        for row in rows
    ]
    dataset_sha256 = _canonical_sha256(dataset_identity)
    return {
        "schema_version": "prepare-edited-pairs-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "prepare-edited-pairs",
        "status": "completed",
        "arguments": {
            "model": MODEL,
            "benchmark": benchmark,
            "targeting": "attribution-4",
            "num_edits": 4,
            "seed": 42,
            "max_new_tokens": 512,
            "gpu_id": "0",
            "limit": None,
            "output_dir": str(directory.resolve()),
        },
        "counts": {
            "discovered": len(rows),
            "written": len(rows),
            "failed": 0,
        },
        "failures": [],
        "outputs": {
            "pairs": {
                "path": "pairs.jsonl",
                "sha256": _sha256_file(directory / "pairs.jsonl"),
                "records": len(rows),
            }
        },
        "decoding": {
            "strategy": "greedy",
            "dtype": "bfloat16",
            "padding_side": "left",
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "use_cache": True,
            "return_dict_in_generate": False,
            "output_scores": False,
        },
        "provenance": {
            "model": MODEL,
            "model_revision": "1" * 40,
            "benchmark_dataset_loader": _dataset_loader(benchmark),
            "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
            "dataset_samples_per_subset": (
                50 if benchmark == "mmlu" else 100 if benchmark == "mmlu-pro" else None
            ),
            "dataset_sample_count": len(rows),
            "dataset_records_sha256": dataset_sha256,
            "random_seed_algorithm": "sha256-first-64-bits/v1",
            "generation_protocol": "explicit-greedy-generation/v1",
            "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
            "target_position": "maximum-logit-after-first-cot-token",
            "alignment": "actual-edited-word-final-token",
            "historical_compatibility_notes": [],
        },
    }


def _write_completed_source(
    directory: Path,
    *,
    benchmark: str = "gsm8k",
    cases: Sequence[Mapping[str, str]] = _CASES,
) -> tuple[Path, list[dict[str, object]]]:
    directory.mkdir(parents=True)
    rows = [_pair_record(case, benchmark=benchmark) for case in cases]
    rows.sort(key=lambda row: str(row["sample_id"]))
    pairs_path = directory / "pairs.jsonl"
    _write_jsonl(pairs_path, rows)
    _write_json(
        directory / "run.json",
        _source_manifest(directory, rows, benchmark=benchmark),
    )
    return pairs_path, rows


class FakeCorrectionRuntime:
    def __init__(
        self,
        corrections: Mapping[str, str],
        *,
        corrector: str = "pyspellchecker",
        events: list[str] | None = None,
    ) -> None:
        self.corrections = dict(corrections)
        self.corrector = corrector
        self.events = events
        self.calls: list[str] = []
        self.closed = False

    def correct(self, text: str) -> CorrectionOutcome:
        self.calls.append(text)
        if self.events is not None:
            self.events.append(f"correction:{text}")
        return CorrectionOutcome(
            corrected_text=self.corrections[text],
            parse_failed=False,
            n_calls=1,
            raw_response=None,
        )

    def provenance(self) -> dict[str, object]:
        return {
            "operation": "input-corrector-audit",
            "runtime": "FakeCorrectionRuntime",
            "corrector": self.corrector,
            "implementation_revision": "2" * 40,
            "protocol_sha256": "3" * 64,
        }

    def close(self) -> None:
        self.closed = True
        if self.events is not None:
            self.events.append("correction:close")


class FakeGenerationRuntime:
    def __init__(
        self,
        *,
        answers_by_call: Sequence[Sequence[str]] = (),
        fail_on_call: int | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.answers_by_call = tuple(tuple(answers) for answers in answers_by_call)
        self.fail_on_call = fail_on_call
        self.events = events
        self.calls: list[dict[str, tuple[str, ...]]] = []
        self.closed = False

    def generate_duplicate_batch(
        self,
        prompts: Sequence[str],
        *,
        sample_ids: Sequence[str],
        gold_answers: Sequence[str],
    ) -> Sequence[SamePromptGeneration]:
        call_number = len(self.calls) + 1
        call = {
            "prompts": tuple(prompts),
            "sample_ids": tuple(sample_ids),
            "gold_answers": tuple(gold_answers),
        }
        self.calls.append(call)
        if self.events is not None:
            self.events.append(f"generation:{','.join(sample_ids)}")
        if call_number == self.fail_on_call:
            raise RuntimeError("injected complete-batch failure")
        answers = (
            self.answers_by_call[call_number - 1]
            if call_number <= len(self.answers_by_call)
            else tuple(gold_answers)
        )
        assert len(answers) == len(prompts)
        return tuple(
            SamePromptGeneration(
                sample_id=sample_id,
                token_ids=tuple(answer.encode("utf-8")),
                text=f"The answer is {answer}.",
                extracted_answer=answer,
                is_extracted=True,
                is_correct=answer == gold,
            )
            for sample_id, gold, answer in zip(
                sample_ids,
                gold_answers,
                answers,
                strict=True,
            )
        )

    def provenance(self) -> dict[str, object]:
        return {
            "operation": "input-corrector-audit",
            "runtime": "FakeGenerationRuntime",
            "model": MODEL,
            "requested_revision": "1" * 40,
            "model_revision": "1" * 40,
            "tokenizer_revision": "1" * 40,
            "protocol_sha256": "4" * 64,
            "generation": {
                "strategy": "greedy",
                "padding_side": "left",
                "max_new_tokens": 512,
                "do_sample": False,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            },
        }

    def close(self) -> None:
        self.closed = True
        if self.events is not None:
            self.events.append("generation:close")


def _config(
    pairs_path: Path,
    output_dir: Path,
    *,
    benchmark: str = "gsm8k",
    corrector: str = "pyspellchecker",
    limit: int | None = None,
    resume: bool = False,
) -> InputCorrectorAuditConfig:
    return InputCorrectorAuditConfig(
        corrector=corrector,
        model=MODEL,
        benchmark=benchmark,
        pairs=pairs_path,
        output_dir=output_dir,
        gpu_id="0",
        limit=limit,
        resume=resume,
    )


def _corrected_prompt(source: Mapping[str, object], corrected_text: str) -> str:
    edited = source["edited"]
    assert isinstance(edited, Mapping)
    prompt = str(edited["prompt"])
    span = edited["editable_prompt_span"]
    assert isinstance(span, Mapping)
    return prompt[: int(span["start"])] + corrected_text + prompt[int(span["end"]) :]


def test_runner_splices_corrected_text_and_publishes_auditable_setting_outputs(
    tmp_path: Path,
) -> None:
    pairs_path, source_rows = _write_completed_source(tmp_path / "source")
    by_source_id = {str(row["sample_id"]): row for row in source_rows}
    output_dir = tmp_path / "out"
    events: list[str] = []
    correction = FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS, events=events)
    generation = FakeGenerationRuntime(
        answers_by_call=(("A", "B", "C", "C"),),
        events=events,
    )

    result = run_input_corrector_audit(
        _config(pairs_path, output_dir),
        correction_runtime=correction,
        generation_runtime=generation,
    )

    assert isinstance(result, InputCorrectorAuditResult)
    assert result.records_path == output_dir / RECORDS_NAME
    assert result.summary_path == output_dir / SUMMARY_NAME
    assert result.run_path == output_dir / "run.json"
    assert result.records == 5
    assert correction.calls == [case["edited"] for case in _CASES]
    assert correction.closed is True
    assert generation.closed is True
    assert events.index("correction:close") < next(
        index for index, event in enumerate(events) if event.startswith("generation:")
    )

    clean_a = str(by_source_id["sample-a"]["clean"]["prompt"])  # type: ignore[index]
    clean_b = str(by_source_id["sample-b"]["clean"]["prompt"])  # type: ignore[index]
    assert generation.calls == [
        {
            "prompts": (clean_a, clean_a, clean_b, clean_b),
            "sample_ids": ("sample-a", "sample-a", "sample-b", "sample-b"),
            "gold_answers": ("A", "A", "C", "C"),
        }
    ]

    records = _read_jsonl(result.records_path)
    assert [record["sample_id"] for record in records] == [
        "sample-a",
        "sample-b",
        "sample-c",
        "sample-d",
        "sample-e",
    ]
    assert all(record["schema_version"] == "input-corrector-audit-record/v1" for record in records)
    assert all(record["paper_sha256"] == PAPER_SHA256 for record in records)
    assert all(record["operation"] == "input-corrector-audit" for record in records)
    by_id = {str(record["sample_id"]): record for record in records}

    for sample_id, source in by_source_id.items():
        assert by_id[sample_id]["source_record_sha256"] == _canonical_sha256(source)

    exact_a = by_id["sample-a"]
    assert exact_a["correction"]["corrected_text"] == "Alpha"
    assert exact_a["edited_words"] == {
        "restored": 1,
        "total": 1,
        "unalignable": 0,
    }
    assert exact_a["prompt_endpoints"] == {
        "clean_sha256": _sha256_text(clean_a),
        "corrected_sha256": _sha256_text(clean_a),
        "exact_utf8": True,
    }
    assert exact_a["same_batch_answers"]["first_extracted_answer"] == "A"
    assert exact_a["same_batch_answers"]["duplicate_extracted_answer"] == "B"
    assert exact_a["separate_source_answers"]["comparison"] == (
        "same_batch_corrected_vs_source_pair_clean"
    )
    assert exact_a["separate_source_answers"]["same_batch_corrected_extracted_answer"] == "B"
    assert exact_a["separate_source_answers"]["source_pair_clean_extracted_answer"] == "A"

    whitespace_only = by_id["sample-c"]
    corrected_c = _corrected_prompt(by_source_id["sample-c"], "Gamma ")
    assert whitespace_only["diagnostics"]["whitespace_normalized_full"] is True
    assert whitespace_only["prompt_endpoints"]["exact_utf8"] is False
    assert whitespace_only["prompt_endpoints"]["corrected_sha256"] == _sha256_text(corrected_c)
    assert whitespace_only["same_batch_answers"] is None
    assert whitespace_only["separate_source_answers"] is None

    collateral = by_id["sample-d"]
    assert collateral["edited_words"] == {
        "restored": 1,
        "total": 1,
        "unalignable": 0,
    }
    assert collateral["diagnostics"]["intact_word_changes"] == 1
    assert collateral["prompt_endpoints"]["exact_utf8"] is False

    not_restored = by_id["sample-e"]
    assert not_restored["edited_words"]["restored"] == 0
    assert not_restored["edited_words"]["total"] == 1
    assert not_restored["prompt_endpoints"]["exact_utf8"] is False

    summary = _read_json(result.summary_path)
    assert summary["schema_version"] == "input-corrector-audit-summary/v1"
    assert summary["operation"] == "input-corrector-audit"
    assert summary["status"] == "completed"
    assert summary["model"] == MODEL
    assert summary["benchmark"] == "gsm8k"
    assert summary["corrector"] == "pyspellchecker"
    assert summary["metrics"] == {
        "records": 5,
        "word_restored": 4,
        "word_total": 5,
        "word_restoration_rate": 0.8,
        "exact_clean": 2,
        "same_changed": 1,
        "separate_source_changed": 1,
        "intact_word_changes": 1,
    }

    manifest = _read_json(result.run_path)
    assert manifest["schema_version"] == "input-corrector-audit-run/v1"
    assert manifest["operation"] == "input-corrector-audit"
    assert manifest["status"] == "completed"
    assert manifest["arguments"]["limit"] is None
    assert manifest["implementation_code"] == implementation_code_identity()
    assert manifest["source"]["pairs_sha256"] == _sha256_file(pairs_path)
    assert manifest["source"]["run_sha256"] == _sha256_file(pairs_path.parent / "run.json")
    assert manifest["outputs"] == {
        RECORDS_NAME: {
            "sha256": _sha256_file(result.records_path),
            "bytes": result.records_path.stat().st_size,
            "records": 5,
        },
        SUMMARY_NAME: {
            "sha256": _sha256_file(result.summary_path),
            "bytes": result.summary_path.stat().st_size,
        },
    }
    assert not (output_dir / ".input-corrector-audit-work").exists()


def test_limit_selects_a_sorted_smoke_prefix_and_labels_the_manifest(
    tmp_path: Path,
) -> None:
    pairs_path, _rows = _write_completed_source(tmp_path / "source")
    correction = FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS)
    generation = FakeGenerationRuntime()
    config = _config(pairs_path, tmp_path / "out", limit=2)

    result = run_input_corrector_audit(
        config,
        correction_runtime=correction,
        generation_runtime=generation,
    )

    assert correction.calls == ["Alpga", "Beeta"]
    assert result.records == 2
    assert [row["sample_id"] for row in _read_jsonl(result.records_path)] == [
        "sample-a",
        "sample-b",
    ]
    manifest = _read_json(result.run_path)
    summary = _read_json(result.summary_path)
    assert manifest["arguments"]["limit"] == 2
    assert manifest["scope"] == "custom-smoke"
    assert summary["scope"] == "custom-smoke"


def test_math_diagnostic_never_loads_or_runs_same_prompt_generation(
    tmp_path: Path,
) -> None:
    pairs_path, _rows = _write_completed_source(
        tmp_path / "source",
        benchmark="math-500",
        cases=_CASES[:2],
    )
    correction = FakeCorrectionRuntime(
        _SUBMITTED_CORRECTIONS,
        corrector="t5-large-spell",
    )
    config = _config(
        pairs_path,
        tmp_path / "out",
        benchmark="math-500",
        corrector="t5-large-spell",
    )

    result = run_input_corrector_audit(
        config,
        correction_runtime=correction,
        generation_runtime=None,
    )

    records = _read_jsonl(result.records_path)
    assert len(records) == 2
    assert all(record["same_batch_answers"] is None for record in records)
    assert all(record["separate_source_answers"] is None for record in records)
    summary = _read_json(result.summary_path)
    assert summary["benchmark"] == "math-500"
    assert summary["metrics"]["exact_clean"] == 2
    assert summary["metrics"]["same_changed"] == 0
    assert summary["metrics"]["separate_source_changed"] == 0


def test_default_correction_runtime_uses_the_production_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.input_corrector_audit.runtime as runtime_module

    pairs_path, _rows = _write_completed_source(
        tmp_path / "source",
        cases=_CASES[-1:],
    )
    correction = FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS)
    constructed: list[InputCorrectorAuditConfig] = []

    def factory(config: InputCorrectorAuditConfig) -> FakeCorrectionRuntime:
        constructed.append(config)
        return correction

    monkeypatch.setattr(runtime_module, "ProductionCorrectionRuntime", factory)
    config = _config(pairs_path, tmp_path / "out")

    result = run_input_corrector_audit(config)

    assert constructed == [config]
    assert correction.calls == ["Epsilom"]
    assert result.records == 1


def test_default_generation_runtime_uses_source_revision_and_full_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.input_corrector_audit.runtime as runtime_module

    pairs_path, _rows = _write_completed_source(
        tmp_path / "source",
        cases=_CASES[:1],
    )
    correction = FakeCorrectionRuntime({"Alpga": "Alpha"})
    generation = FakeGenerationRuntime()
    constructed: list[tuple[InputCorrectorAuditConfig, str]] = []

    def factory(
        config: InputCorrectorAuditConfig,
        *,
        revision: str,
    ) -> FakeGenerationRuntime:
        constructed.append((config, revision))
        return generation

    monkeypatch.setattr(runtime_module, "HuggingFaceSamePromptRuntime", factory)
    config = _config(pairs_path, tmp_path / "out")

    result = run_input_corrector_audit(
        config,
        correction_runtime=correction,
    )

    assert constructed == [(config, "1" * 40)]
    assert len(generation.calls) == 1
    assert generation.calls[0]["sample_ids"] == ("sample-a", "sample-a")
    assert result.records == 1


def test_manifest_captures_correction_provenance_after_lazy_model_use(
    tmp_path: Path,
) -> None:
    pairs_path, _rows = _write_completed_source(
        tmp_path / "source",
        cases=_CASES[-1:],
    )

    class LazyRevisionRuntime(FakeCorrectionRuntime):
        def provenance(self) -> dict[str, object]:
            payload = super().provenance()
            payload["model_revision_source"] = (
                "model-config-metadata" if self.calls else "explicit-load-revision"
            )
            return payload

    correction = LazyRevisionRuntime(_SUBMITTED_CORRECTIONS)
    result = run_input_corrector_audit(
        _config(pairs_path, tmp_path / "out"),
        correction_runtime=correction,
    )

    manifest = _read_json(result.run_path)
    assert manifest["runtime"]["correction"]["model_revision_source"] == ("model-config-metadata")


def test_resume_reuses_corrections_and_only_complete_same_batch_checkpoints(
    tmp_path: Path,
) -> None:
    pairs_path, rows = _write_completed_source(
        tmp_path / "source",
        cases=_CASES[:4],
    )
    all_exact = {
        str(row["edited"]["editable_text"]): str(row["clean"]["editable_text"]) for row in rows
    }
    config = _config(pairs_path, tmp_path / "out")
    initial_correction = FakeCorrectionRuntime(all_exact)
    initial_generation = FakeGenerationRuntime(
        answers_by_call=(("A", "B", "C", "C"),),
        fail_on_call=2,
    )

    with pytest.raises(InputCorrectorAuditRunError, match="generation|batch"):
        run_input_corrector_audit(
            config,
            correction_runtime=initial_correction,
            generation_runtime=initial_generation,
        )

    assert initial_correction.calls == [
        "Alpga",
        "Beeta",
        "Gbmma",
        "Ddlta stable",
    ]
    assert [call["sample_ids"] for call in initial_generation.calls] == [
        ("sample-a", "sample-a", "sample-b", "sample-b"),
        ("sample-c", "sample-c", "sample-d", "sample-d"),
    ]
    assert not (config.output_dir / RECORDS_NAME).exists()
    assert not (config.output_dir / SUMMARY_NAME).exists()
    failed = _read_json(config.output_dir / "run.json")
    assert failed["status"] == "failed"

    resumed_correction = FakeCorrectionRuntime(all_exact)
    resumed_generation = FakeGenerationRuntime()
    result = run_input_corrector_audit(
        replace(config, resume=True),
        correction_runtime=resumed_correction,
        generation_runtime=resumed_generation,
    )

    assert resumed_correction.calls == []
    assert [call["sample_ids"] for call in resumed_generation.calls] == [
        ("sample-c", "sample-c", "sample-d", "sample-d"),
    ]
    assert result.records == 4
    records = {row["sample_id"]: row for row in _read_jsonl(result.records_path)}
    # The first batch came from the original generation call, not the resumed runtime.
    assert records["sample-a"]["same_batch_answers"]["first_extracted_answer"] == "A"
    assert records["sample-a"]["same_batch_answers"]["duplicate_extracted_answer"] == "B"
    assert not (config.output_dir / ".input-corrector-audit-work").exists()


def test_resume_rejects_complete_generation_checkpoints_without_runtime_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.input_corrector_audit.runner as runner_module

    pairs_path, rows = _write_completed_source(tmp_path / "source", cases=_CASES[:2])
    all_exact = {
        str(row["edited"]["editable_text"]): str(row["clean"]["editable_text"])
        for row in rows
    }
    config = _config(pairs_path, tmp_path / "out")

    def fail_publication(path: Path, payload: object) -> None:
        raise OSError(f"injected publication failure for {path.name}")

    with monkeypatch.context() as publication_patch:
        publication_patch.setattr(runner_module, "_write_jsonl_atomic", fail_publication)
        with pytest.raises(InputCorrectorAuditRunError, match="publication|injected"):
            run_input_corrector_audit(
                config,
                correction_runtime=FakeCorrectionRuntime(all_exact),
                generation_runtime=FakeGenerationRuntime(),
            )

    manifest = _read_json(config.output_dir / "run.json")
    assert manifest["checkpoints"]["complete_same_batches"] == 1
    del manifest["runtime"]["generation"]
    _write_json(config.output_dir / "run.json", manifest)

    with pytest.raises(InputCorrectorAuditRunError, match="generation.*provenance|provenance"):
        run_input_corrector_audit(replace(config, resume=True))

    assert not (config.output_dir / RECORDS_NAME).exists()
    assert not (config.output_dir / SUMMARY_NAME).exists()


def test_checkpoint_cleanup_failure_does_not_undo_completed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.input_corrector_audit.runner as runner_module

    pairs_path, _rows = _write_completed_source(
        tmp_path / "source",
        cases=_CASES[-1:],
    )
    config = _config(pairs_path, tmp_path / "out")

    def fail_cleanup(path: Path) -> None:
        assert path.name == ".input-corrector-audit-work"
        raise OSError("injected checkpoint cleanup failure")

    monkeypatch.setattr(runner_module.shutil, "rmtree", fail_cleanup)

    result = run_input_corrector_audit(
        config,
        correction_runtime=FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS),
    )

    assert result.records == 1
    assert result.records_path.is_file()
    assert result.summary_path.is_file()
    assert _read_json(result.run_path)["status"] == "completed"
    assert (config.output_dir / ".input-corrector-audit-work").is_dir()


def _write_failed_run(tmp_path: Path) -> tuple[InputCorrectorAuditConfig, Path]:
    pairs_path, _rows = _write_completed_source(tmp_path / "source", cases=_CASES[:2])
    config = _config(pairs_path, tmp_path / "out")
    with pytest.raises(InputCorrectorAuditRunError):
        run_input_corrector_audit(
            config,
            correction_runtime=FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS),
            generation_runtime=FakeGenerationRuntime(fail_on_call=1),
        )
    assert not (config.output_dir / RECORDS_NAME).exists()
    return config, pairs_path


def test_resume_rejects_argument_mutation_before_runtime_work(tmp_path: Path) -> None:
    config, _pairs_path = _write_failed_run(tmp_path)
    correction = FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS)
    generation = FakeGenerationRuntime()

    with pytest.raises(ValueError, match="argument|resume"):
        run_input_corrector_audit(
            replace(config, limit=1, resume=True),
            correction_runtime=correction,
            generation_runtime=generation,
        )

    assert correction.calls == []
    assert generation.calls == []


def test_resume_rejects_source_byte_mutation_before_runtime_work(tmp_path: Path) -> None:
    config, pairs_path = _write_failed_run(tmp_path)
    pairs_path.write_bytes(pairs_path.read_bytes() + b"\n")
    correction = FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS)
    generation = FakeGenerationRuntime()

    with pytest.raises(ValueError, match="source|hash|SHA-256|changed"):
        run_input_corrector_audit(
            replace(config, resume=True),
            correction_runtime=correction,
            generation_runtime=generation,
        )

    assert correction.calls == []
    assert generation.calls == []
    assert not (config.output_dir / RECORDS_NAME).exists()


def test_source_mutation_during_work_fails_closed_before_publication(
    tmp_path: Path,
) -> None:
    pairs_path, _rows = _write_completed_source(tmp_path / "source", cases=_CASES[:2])
    config = _config(pairs_path, tmp_path / "out")

    class MutatingCorrectionRuntime(FakeCorrectionRuntime):
        def correct(self, text: str) -> CorrectionOutcome:
            outcome = super().correct(text)
            if len(self.calls) == 1:
                pairs_path.write_bytes(pairs_path.read_bytes() + b"\n")
            return outcome

    correction = MutatingCorrectionRuntime(_SUBMITTED_CORRECTIONS)
    generation = FakeGenerationRuntime()

    with pytest.raises(
        InputCorrectorAuditRunError,
        match="source|hash|SHA-256|changed",
    ):
        run_input_corrector_audit(
            config,
            correction_runtime=correction,
            generation_runtime=generation,
        )

    assert correction.closed is True
    assert generation.closed is True
    assert not (config.output_dir / RECORDS_NAME).exists()
    assert not (config.output_dir / SUMMARY_NAME).exists()
    assert _read_json(config.output_dir / "run.json")["status"] == "failed"


def test_completed_resume_revalidates_public_outputs_without_loading_models(
    tmp_path: Path,
) -> None:
    pairs_path, _rows = _write_completed_source(tmp_path / "source", cases=_CASES[:2])
    config = _config(pairs_path, tmp_path / "out")
    expected = run_input_corrector_audit(
        config,
        correction_runtime=FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS),
        generation_runtime=FakeGenerationRuntime(),
    )

    resumed = run_input_corrector_audit(replace(config, resume=True))

    assert resumed == expected


def test_resume_rejects_an_executable_code_identity_mismatch(tmp_path: Path) -> None:
    pairs_path, _rows = _write_completed_source(tmp_path / "source", cases=_CASES[:2])
    config = _config(pairs_path, tmp_path / "out")
    run_input_corrector_audit(
        config,
        correction_runtime=FakeCorrectionRuntime(_SUBMITTED_CORRECTIONS),
        generation_runtime=FakeGenerationRuntime(),
    )
    manifest = _read_json(config.output_dir / "run.json")
    implementation = manifest["implementation_code"]
    assert isinstance(implementation, dict)
    implementation["sha256"] = "0" * 64
    _write_json(config.output_dir / "run.json", manifest)

    with pytest.raises(ValueError, match="code|implementation|executable"):
        run_input_corrector_audit(replace(config, resume=True))
