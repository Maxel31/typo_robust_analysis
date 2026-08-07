"""Final-PDF contract tests for the answer-line deletion control."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
import typo_cot.experiments.answer_line_deletion.source as deletion_source_module
from typo_cot.cli import main
from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.answer_line_deletion import (
    ARM_ORDER,
    AnswerLineDeletionConfig,
    AnswerLineDeletionGeneration,
    AnswerLineDeletionInputUse,
    AnswerLineDeletionResult,
    AnswerLineDeletionRunError,
    AnswerLineDeletionScan,
    build_answer_line_deletion_plan,
    run_answer_line_deletion,
    strip_final_nonempty_line,
)
from typo_cot.experiments.answer_line_deletion.runtime import (
    HuggingFaceAnswerLineDeletionRuntime,
)
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.cot_swap import (
    CELL_ORDER,
    CotSwapConfig,
    CotSwapGeneration,
    CotSwapInputUse,
    CotSwapScan,
    run_cot_swap,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _answer_payload(value: str, *, correct: bool) -> dict[str, object]:
    return {
        "value": value,
        "is_extracted": bool(value),
        "is_correct": correct,
        "method": "primary:fixture" if value else "unextractable",
        "primary_method": "fixture" if value else "no_match",
        "confidence": 1.0 if value else 0.0,
    }


def _pair(sample_id: str) -> dict[str, object]:
    clean_prompt = f"few-shot context\nQuestion: clean {sample_id}\nAnswer:"
    edited_prompt = f"few-shot context\nQuestion: edited {sample_id}\nAnswer:"
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": "test/model",
        "benchmark": "gsm8k",
        "targeting": "random-4",
        "seed": 42,
        "num_edits_requested": 4,
        "num_candidates": 8,
        "num_target_attempts": 1,
        "num_aligned_words": 0,
        "gold_answer": "2",
        "subset": None,
        "clean": {
            "prompt": clean_prompt,
            "prompt_token_count": 12,
            "continuation": (
                "We first calculate one quantity.\nThe final computation gives 2.\nThe answer is 2."
            ),
            "continuation_token_count": 18,
            "answer": _answer_payload("2", correct=True),
        },
        "edited": {
            "prompt": edited_prompt,
            "prompt_token_count": 12,
            "continuation": ("The altered calculation gives three.\nThe answer is 3."),
            "continuation_token_count": 15,
            "answer": _answer_payload("3", correct=False),
        },
        "answer_changed": True,
        "target_attempts": [{"token_index": 3, "token_text": "clean", "operation": "substitution"}],
        "aligned_words": [],
    }


def _write_pair_source(root: Path, sample_ids: list[str]) -> Path:
    root.mkdir(parents=True)
    pairs = [_pair(sample_id) for sample_id in sorted(sample_ids)]
    pairs_path = root / "pairs.jsonl"
    _write_jsonl(pairs_path, pairs)
    _write_json(
        root / "run.json",
        {
            "schema_version": "prepare-edited-pairs-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "prepare-edited-pairs",
            "status": "completed",
            "arguments": {
                "model": "test/model",
                "benchmark": "gsm8k",
                "targeting": "random-4",
                "num_edits": 4,
                "seed": 42,
                "max_new_tokens": 512,
                "gpu_id": "0",
                "limit": None,
                "output_dir": str(root.resolve()),
            },
            "counts": {
                "discovered": len(pairs),
                "written": len(pairs),
                "failed": 0,
            },
            "failures": [],
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
                "model": "test/model",
                "model_revision": "source-revision",
                "benchmark_dataset_loader": "gsm8k",
                "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
                "dataset_sample_count": len(pairs),
                "dataset_records_sha256": "d" * 64,
                "dataset_samples_per_subset": None,
                "random_seed_algorithm": "sha256-first-64-bits/v1",
                "generation_protocol": "explicit-greedy-generation/v1",
                "target_position": "maximum-logit-after-first-cot-token",
                "alignment": "actual-edited-word-final-token",
                "historical_compatibility_notes": [],
            },
        },
    )
    return pairs_path


def _cot_generation(value: str, *, cell: str) -> CotSwapGeneration:
    text = f"The answer is {value}." if value else "No extractable answer."
    extraction = extract_with_fallback(
        text,
        benchmark="gsm8k",
        correct_answer="2",
    )
    return CotSwapGeneration(
        token_ids=(ord(cell),),
        text=text,
        value=extraction.value,
        is_extracted=extraction.is_extracted,
        is_correct=extraction.is_correct,
        method=extraction.method,
        primary_method=extraction.primary_method,
        stop_reason="eos_token",
        stop_token_id=ord(cell),
    )


class _CotSwapRuntime:
    def __init__(self, answers: Mapping[str, Mapping[str, str]]) -> None:
        self.answers = {sample_id: dict(values) for sample_id, values in answers.items()}

    def provenance(self) -> dict[str, object]:
        return {
            "operation": "cot-swap",
            "runtime": "fixture-runtime",
            "model": "test/model",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "cuda_visible_devices": "0",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_new_tokens": 16,
                "use_cache": True,
                "return_dict_in_generate": False,
                "output_scores": False,
                "padding_side": "left",
                "eos_token_id": [1, *[ord(cell) for cell in CELL_ORDER]],
            },
            "answer_extraction": (
                "primary-then-empty-only-fallback-symmetric-a-b-c-d-cap-aware/v2"
            ),
            "implementation": "huggingface-cot-swap-four-cell-batch/v1",
            "batching": {
                "policy": "one-pair-four-cells/v1",
                "batch_size": 4,
                "cell_order": list(CELL_ORDER),
            },
            "answer_span_decoding": {
                "source": "generated-token-ids-only/v1",
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
            "effective_eos_token_ids": [1, *[ord(cell) for cell in CELL_ORDER]],
            "effective_eos_token_ids_source": "model-generation-config",
            "text_intervention": {
                "boundary": "submitted-first-[Tt]he-answer-is-filter/v1",
                "assembly": "recorded-prompt-plus-decoded-pre-answer-text-retokenized/v1",
            },
        }

    def scan_pair(self, pair: dict[str, object], plan: object) -> CotSwapScan:
        sample_id = str(pair["sample_id"])
        values = self.answers[sample_id]
        uses: dict[str, CotSwapInputUse] = {}
        generations: dict[str, CotSwapGeneration] = {}
        for cell in plan.cells:  # type: ignore[attr-defined]
            uses[cell.cell] = CotSwapInputUse(
                cell=cell.cell,
                prompt_text_sha256=cell.prompt_sha256,
                pre_answer_text_sha256=cell.pre_answer_sha256,
                full_input_text_sha256=cell.full_input_sha256,
                prompt_char_count=len(cell.prompt),
                pre_answer_char_count=len(cell.pre_answer_text),
                full_input_char_count=len(cell.full_input),
                prompt_token_count=cell.prompt_token_count,
                full_input_token_count=cell.prompt_token_count + 4,
                full_input_ids_sha256=hashlib.sha256(cell.full_input.encode()).hexdigest(),
                prompt_prefix_token_stable=True,
            )
            generations[cell.cell] = _cot_generation(values[cell.cell], cell=cell.cell)
        return CotSwapScan(
            sample_id=sample_id,
            input_uses=uses,
            generations=generations,
        )


def _completed_cot_swap(
    root: Path,
    *,
    answers: Mapping[str, Mapping[str, str]],
) -> Path:
    pairs = _write_pair_source(root / "prepared", list(answers))
    output = root / "cot-swap"
    run_cot_swap(
        CotSwapConfig(
            model="test/model",
            benchmark="gsm8k",
            pairs=pairs,
            targeting="random-4",
            output_dir=output,
        ),
        runtime=_CotSwapRuntime(answers),
    )
    return output


def _generation(value: str, *, arm: str, capped: bool = False) -> AnswerLineDeletionGeneration:
    text = f"The answer is {value}." if value else "No extractable answer."
    extraction = extract_with_fallback(
        text,
        benchmark="gsm8k",
        correct_answer="2",
        allow_positional=not capped,
    )
    token = 101 if arm == "complete" else 102
    return AnswerLineDeletionGeneration(
        token_ids=tuple(range(200, 216)) if capped else (token,),
        text=text,
        value=extraction.value,
        is_extracted=extraction.is_extracted,
        is_correct=extraction.is_correct,
        method=extraction.method,
        primary_method=extraction.primary_method,
        stop_reason="max_new_tokens" if capped else "eos_token",
        stop_token_id=None if capped else token,
    )


class _Runtime:
    def __init__(
        self,
        *,
        answers: Mapping[str, Mapping[str, str]] | None = None,
        failure_for: str | None = None,
        provenance_changes: Mapping[str, object] | None = None,
        input_mismatch_for: tuple[str, str] | None = None,
    ) -> None:
        self.answers = {sample_id: dict(values) for sample_id, values in (answers or {}).items()}
        self.failure_for = failure_for
        self.provenance_changes = dict(provenance_changes or {})
        self.input_mismatch_for = input_mismatch_for
        self.calls: list[str] = []

    def provenance(self) -> dict[str, object]:
        protocol = importlib.import_module("typo_cot.experiments.answer_line_deletion.protocol")
        payload: dict[str, object] = {
            "operation": "answer-line-deletion",
            "runtime": "fixture-runtime",
            "model": "test/model",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "cuda_visible_devices": "0",
            "generation": {**protocol.GENERATION, "eos_token_id": [101, 102]},
            "answer_extraction": protocol.ANSWER_EXTRACTION,
            "implementation": protocol.IMPLEMENTATION,
            "batching": dict(protocol.BATCHING),
            "answer_span_decoding": dict(protocol.ANSWER_SPAN_DECODING),
            "effective_eos_token_ids": [101, 102],
            "effective_eos_token_ids_source": "model-generation-config",
            "text_intervention": dict(protocol.TEXT_INTERVENTION),
        }
        payload.update(self.provenance_changes)
        return payload

    def scan_pair(self, pair: dict[str, object], plan: object) -> AnswerLineDeletionScan:
        sample_id = str(pair["sample_id"])
        self.calls.append(sample_id)
        if sample_id == self.failure_for:
            raise RuntimeError("synthetic answer-line-deletion GPU failure")
        values = self.answers.get(
            sample_id,
            {"complete": "2", "answer-line-deleted": "3"},
        )
        uses: dict[str, AnswerLineDeletionInputUse] = {}
        generations: dict[str, AnswerLineDeletionGeneration] = {}
        for arm in plan.arms:  # type: ignore[attr-defined]
            full_hash = arm.full_input_sha256
            if self.input_mismatch_for == (sample_id, arm.arm):
                full_hash = "0" * 64
            uses[arm.arm] = AnswerLineDeletionInputUse(
                arm=arm.arm,
                prompt_text_sha256=arm.prompt_sha256,
                pre_answer_text_sha256=arm.pre_answer_sha256,
                full_input_text_sha256=full_hash,
                prompt_char_count=len(arm.prompt),
                pre_answer_char_count=len(arm.pre_answer_text),
                full_input_char_count=len(arm.full_input),
                prompt_token_count=arm.prompt_token_count,
                full_input_token_count=arm.prompt_token_count + 4,
                full_input_ids_sha256=hashlib.sha256(arm.full_input.encode()).hexdigest(),
                prompt_prefix_token_stable=True,
            )
            generations[arm.arm] = _generation(values[arm.arm], arm=arm.arm)
        return AnswerLineDeletionScan(
            sample_id=sample_id,
            input_uses=uses,
            generations=generations,
        )


def _config(source: Path, output: Path, **changes: object) -> AnswerLineDeletionConfig:
    config = AnswerLineDeletionConfig(
        model="test/model",
        benchmark="gsm8k",
        cot_swap_run=source,
        max_pairs=150,
        output_dir=output,
    )
    return replace(config, **changes)


def test_catalog_and_cli_expose_completed_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = get_experiment("answer-line-deletion")
    assert spec.status == "implemented"
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--cot-swap-run",
        "--max-pairs",
        "--output-dir",
    )
    assert spec.outputs == (
        "answer_line_deletion_records.jsonl",
        "pair_status_records.jsonl",
        "answer_line_deletion_summary.json",
        "run.json",
    )

    captured: list[AnswerLineDeletionConfig] = []

    def fake_run(config: AnswerLineDeletionConfig) -> AnswerLineDeletionResult:
        captured.append(config)
        return AnswerLineDeletionResult(
            records_path=config.output_dir / "answer_line_deletion_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "answer_line_deletion_summary.json",
            run_path=config.output_dir / "run.json",
            source_cohort_pairs=3,
            executed_pairs=1,
            records=1,
        )

    monkeypatch.setattr(cli_module, "run_answer_line_deletion", fake_run)
    assert (
        main(
            [
                "answer-line-deletion",
                "--model",
                "test/model",
                "--benchmark",
                "gsm8k",
                "--cot-swap-run",
                "results/cot-swap",
                "--max-pairs",
                "150",
                "--gpu-id",
                "0",
                "--limit",
                "1",
                "--output-dir",
                "results/deletion",
                "--resume",
            ]
        )
        == 0
    )
    assert captured == [
        AnswerLineDeletionConfig(
            model="test/model",
            benchmark="gsm8k",
            cot_swap_run=Path("results/cot-swap"),
            max_pairs=150,
            output_dir=Path("results/deletion"),
            gpu_id="0",
            limit=1,
            resume=True,
        )
    ]
    assert "wrote 1 paired control record(s) from 3 source-cohort pair(s)" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"model": ""}, "model must not be empty"),
        ({"benchmark": "mmlu-pro"}, "unsupported benchmark"),
        ({"gpu_id": "0,1"}, "single non-negative integer"),
        ({"max_pairs": 0}, "max_pairs must be a positive integer"),
        ({"limit": 0}, "limit must be a positive integer"),
    ),
)
def test_config_rejects_ambiguous_or_out_of_scope_arguments(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "source", tmp_path / "output", **changes)


def test_strip_final_nonempty_line_freezes_submitted_character_rule() -> None:
    deletion = strip_final_nonempty_line("first\n\nlast answer\n\n")
    assert deletion.original_text == "first\n\nlast answer\n\n"
    assert deletion.deleted_text == "first\n\n"
    assert deletion.deleted_line == "last answer"
    assert deletion.deleted_line_index == 2
    assert deletion.prefix_became_empty is False
    assert deletion.method == "submitted-final-nonempty-line/v1"

    assert strip_final_nonempty_line("only answer\n").deleted_text == ""
    assert strip_final_nonempty_line("only answer\n").prefix_became_empty is True
    assert strip_final_nonempty_line("\n \n").deleted_text == ""
    assert strip_final_nonempty_line("\n \n").deleted_line is None


def test_plan_builds_exact_complete_and_deleted_inputs() -> None:
    pair = _pair("case-1")
    plan = build_answer_line_deletion_plan(
        pair,
        source_a_answer="2",
        source_c_answer="2",
        source_record_sha256="a" * 64,
        prepared_record_sha256="b" * 64,
    )
    assert tuple(arm.arm for arm in plan.arms) == ARM_ORDER
    assert plan.arms[0].pre_answer_text == (
        "We first calculate one quantity.\nThe final computation gives 2.\n"
    )
    assert plan.arms[1].pre_answer_text == "We first calculate one quantity.\n"
    assert plan.deletion.deleted_line == "The final computation gives 2."
    assert plan.deletion.prefix_became_empty is False
    assert all(arm.prompt == pair["edited"]["prompt"] for arm in plan.arms)  # type: ignore[index]
    assert len(plan.fingerprint) == 64


def test_runner_uses_sorted_a_correct_b_changed_cohort_then_cap_then_limit(
    tmp_path: Path,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={
            "case-3": {"A": "2", "B": "3", "C": "2", "D": "3"},
            "case-1": {"A": "2", "B": "3", "C": "2", "D": "3"},
            "case-4": {"A": "2", "B": "3", "C": "3", "D": "3"},
            "case-2": {"A": "2", "B": "2", "C": "2", "D": "3"},
        },
    )
    runtime = _Runtime(
        answers={
            "case-1": {"complete": "2", "answer-line-deleted": "3"},
        }
    )
    result = run_answer_line_deletion(
        _config(source, tmp_path / "output", max_pairs=2, limit=1),
        runtime=runtime,
    )

    assert result.source_cohort_pairs == 3
    assert result.executed_pairs == 1
    assert runtime.calls == ["case-1"]
    records = _read_jsonl(result.records_path)
    assert [row["sample_id"] for row in records] == ["case-1"]
    assert records[0]["events"] == {
        "complete_matches_source_c": True,
        "complete_restored_to_a": True,
        "answer_line_deleted_restored_to_a": False,
        "restoration_lost_after_deletion": True,
        "restoration_gained_after_deletion": False,
    }
    statuses = _read_jsonl(result.pair_status_records_path)
    assert [row["sample_id"] for row in statuses] == [
        "case-1",
        "case-2",
        "case-3",
        "case-4",
    ]
    assert [row["execution_status"] for row in statuses] == [
        "completed",
        "not-restoration-denominator",
        "limit-truncated",
        "beyond-max-pairs",
    ]
    assert statuses[1]["source_a_correct"] is True
    assert statuses[1]["source_b_changed_from_a"] is False
    assert statuses[1]["cohort_exclusion_reason"] == "source-b-equals-a"
    assert statuses[0]["cohort_exclusion_reason"] is None
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["source_restoration_cohort_pairs"] == 3
    assert summary["counts"]["capped_pairs"] == 2
    assert summary["metrics"]["complete_restoration"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert summary["metrics"]["answer_line_deleted_restoration"] == {
        "numerator": 0,
        "denominator": 1,
        "rate": 0.0,
    }
    assert summary["metrics"]["restoration_rate_difference_deleted_minus_complete"] == -1.0
    assert "paired_restoration_rate_difference" not in summary["metrics"]
    assert summary["comparability"]["status"] == "partial-smoke-run"


def test_unextractable_source_a_is_retained_as_an_a_incorrect_status(
    tmp_path: Path,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={
            "case-0": {"A": "", "B": "3", "C": "2", "D": "3"},
            "case-1": {"A": "2", "B": "3", "C": "2", "D": "3"},
        },
    )
    runtime = _Runtime(answers={"case-1": {"complete": "2", "answer-line-deleted": "2"}})
    result = run_answer_line_deletion(
        _config(source, tmp_path / "output"),
        runtime=runtime,
    )
    assert runtime.calls == ["case-1"]
    statuses = _read_jsonl(result.pair_status_records_path)
    assert statuses[0]["sample_id"] == "case-0"
    assert statuses[0]["source_a_correct"] is False
    assert statuses[0]["source_b_changed_from_a"] is True
    assert statuses[0]["source_restoration_denominator"] is False
    assert statuses[0]["cohort_exclusion_reason"] == "source-a-not-correct"
    assert statuses[0]["execution_status"] == "not-restoration-denominator"


def test_runner_records_empty_prefix_stratum_and_published_integer_references(
    tmp_path: Path,
) -> None:
    _completed_cot_swap(
        tmp_path,
        answers={
            "case-1": {"A": "2", "B": "3", "C": "2", "D": "3"},
            "case-2": {"A": "2", "B": "3", "C": "2", "D": "3"},
        },
    )
    pairs_path = tmp_path / "prepared" / "pairs.jsonl"
    pairs = _read_jsonl(pairs_path)
    pairs[0]["clean"]["continuation"] = "Single forced line.\nThe answer is 2."  # type: ignore[index]
    _write_jsonl(pairs_path, pairs)
    # Rebuild the upstream run because it binds every prepared-pair byte.
    for path in (tmp_path / "cot-swap").iterdir():
        if path.is_file():
            path.unlink()
    run_cot_swap(
        CotSwapConfig(
            model="test/model",
            benchmark="gsm8k",
            pairs=pairs_path,
            targeting="random-4",
            output_dir=tmp_path / "cot-swap",
        ),
        runtime=_CotSwapRuntime(
            {
                "case-1": {"A": "2", "B": "3", "C": "2", "D": "3"},
                "case-2": {"A": "2", "B": "3", "C": "2", "D": "3"},
            }
        ),
    )
    result = run_answer_line_deletion(
        _config(tmp_path / "cot-swap", tmp_path / "output"),
        runtime=_Runtime(
            answers={
                "case-1": {"complete": "2", "answer-line-deleted": ""},
                "case-2": {"complete": "2", "answer-line-deleted": "2"},
            }
        ),
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["prefix_became_empty_pairs"] == 1
    assert (
        summary["metrics_by_prefix_became_empty"]["true"]["answer_line_deleted_restoration"][
            "denominator"
        ]
        == 1
    )
    assert summary["metrics_by_prefix_became_empty"]["false"][
        "answer_line_deleted_restoration"
    ] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert summary["diagnostics_by_prefix_became_empty"]["true"]["unextractable_by_arm"] == {
        "complete": 0,
        "answer-line-deleted": 1,
    }
    assert summary["diagnostics_by_prefix_became_empty"]["false"]["unextractable_by_arm"] == {
        "complete": 0,
        "answer-line-deleted": 0,
    }
    assert summary["diagnostics_by_prefix_became_empty"]["true"]["stop_reason_by_arm"][
        "answer-line-deleted"
    ] == {"eos_token": 1, "max_new_tokens": 0}
    reference = summary["published_reference"]
    assert reference["final_pdf_protocol"]["max_new_tokens"] == 16
    assert reference["printed_table_source"] == {
        "max_new_tokens": 256,
        "gsm8k": {
            "n": 333,
            "complete_restored": 317,
            "deleted_restored": 163,
            "deleted_unextractable": 29,
        },
        "mmlu": {
            "n": 450,
            "complete_restored": 370,
            "deleted_restored": 131,
            "deleted_unextractable": 7,
        },
    }
    assert reference["archived_16_token_control"] == {
        "max_new_tokens": 16,
        "gsm8k": {
            "n": 333,
            "complete_restored": 315,
            "deleted_restored": 14,
            "complete_unextractable": 13,
            "deleted_unextractable": 262,
        },
        "mmlu": {
            "n": 450,
            "complete_restored": 369,
            "deleted_restored": 54,
            "complete_unextractable": 7,
            "deleted_unextractable": 232,
        },
    }
    assert reference["historical_prefix_became_empty"] == {
        "gsm8k": {"numerator": 179, "denominator": 333},
        "mmlu": {"numerator": 334, "denominator": 450},
    }
    assert reference["use_as_acceptance_target"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("status", "source CoT-swap run must be completed"),
        ("limit", "source CoT-swap run must be unlimited"),
        ("targeting", "source CoT-swap run must use random-4"),
        ("output-hash", "completed run validation failed"),
    ),
)
def test_invalid_upstream_contract_fails_before_runtime(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={"case-1": {"A": "2", "B": "3", "C": "2", "D": "3"}},
    )
    manifest_path = source / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "status":
        manifest["status"] = "failed"
        _write_json(manifest_path, manifest)
    elif mutation == "limit":
        manifest["arguments"]["limit"] = 1
        _write_json(manifest_path, manifest)
    elif mutation == "targeting":
        manifest["arguments"]["targeting"] = "attribution-4"
        _write_json(manifest_path, manifest)
    else:
        with (source / "cot_swap_records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")

    runtime = _Runtime()
    with pytest.raises((ValueError, AnswerLineDeletionRunError), match=message):
        run_answer_line_deletion(_config(source, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("changed_file", "message"),
    (
        ("cot-swap-run", "source CoT-swap run changed during validation"),
        ("prepared-run", "source prepared-pair run changed during validation"),
    ),
)
def test_source_manifest_replacement_during_validation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_file: str,
    message: str,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={"case-1": {"A": "2", "B": "3", "C": "2", "D": "3"}},
    )
    real_validate = deletion_source_module.run_cot_swap

    def validate_then_replace(config: CotSwapConfig) -> object:
        result = real_validate(config)
        path = (
            source / "run.json"
            if changed_file == "cot-swap-run"
            else tmp_path / "prepared" / "run.json"
        )
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(deletion_source_module, "run_cot_swap", validate_then_replace)
    with pytest.raises(ValueError, match=message):
        run_answer_line_deletion(_config(source, tmp_path / "output"), runtime=_Runtime())


def test_runtime_provenance_and_fixed_input_are_verified_before_publication(
    tmp_path: Path,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={"case-1": {"A": "2", "B": "3", "C": "2", "D": "3"}},
    )
    with pytest.raises(ValueError, match="max_new_tokens"):
        run_answer_line_deletion(
            _config(source, tmp_path / "bad-provenance"),
            runtime=_Runtime(
                provenance_changes={
                    "generation": {
                        **_Runtime().provenance()["generation"],  # type: ignore[dict-item]
                        "max_new_tokens": 256,
                    }
                }
            ),
        )
    with pytest.raises(AnswerLineDeletionRunError, match="fixed input"):
        run_answer_line_deletion(
            _config(source, tmp_path / "bad-input"),
            runtime=_Runtime(input_mismatch_for=("case-1", "answer-line-deleted")),
        )


def test_failed_pair_resumes_only_missing_work_and_completed_resume_is_model_free(
    tmp_path: Path,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={
            "case-1": {"A": "2", "B": "3", "C": "2", "D": "3"},
            "case-2": {"A": "2", "B": "3", "C": "2", "D": "3"},
        },
    )
    output = tmp_path / "output"
    first = _Runtime(failure_for="case-2")
    with pytest.raises(AnswerLineDeletionRunError, match=r"1 pair\(s\) failed"):
        run_answer_line_deletion(_config(source, output), runtime=first)
    assert first.calls == ["case-1", "case-2"]

    resumed = _Runtime()
    result = run_answer_line_deletion(
        _config(source, output, resume=True),
        runtime=resumed,
    )
    assert resumed.calls == ["case-2"]
    assert result.records == 2

    class _ForbiddenRuntime:
        def provenance(self) -> Mapping[str, object]:
            raise AssertionError("completed validation must not inspect a new runtime")

        def scan_pair(self, pair: dict[str, object], plan: object) -> AnswerLineDeletionScan:
            raise AssertionError("completed validation must not run GPU work")

    validated = run_answer_line_deletion(
        _config(source, output, resume=True),
        runtime=_ForbiddenRuntime(),
    )
    assert validated.records == 2


def test_registered_checkpoint_with_extra_fields_is_rejected_before_runtime(
    tmp_path: Path,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={
            "case-1": {"A": "2", "B": "3", "C": "2", "D": "3"},
            "case-2": {"A": "2", "B": "3", "C": "2", "D": "3"},
        },
    )
    output = tmp_path / "output"
    with pytest.raises(AnswerLineDeletionRunError):
        run_answer_line_deletion(
            _config(source, output),
            runtime=_Runtime(failure_for="case-2"),
        )
    manifest_path = output / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity, metadata = next(iter(manifest["checkpoints"].items()))
    checkpoint_path = output / ".answer-line-deletion-work" / "checkpoints" / metadata["file"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["unexpected"] = "must-not-be-ignored"
    _write_json(checkpoint_path, checkpoint)
    manifest["checkpoints"][identity]["sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    _write_json(manifest_path, manifest)

    resumed = _Runtime()
    with pytest.raises(ValueError, match="checkpoint payload does not match"):
        run_answer_line_deletion(
            _config(source, output, resume=True),
            runtime=resumed,
        )
    assert resumed.calls == []


def test_source_drift_during_gpu_work_prevents_publication_and_keeps_checkpoint(
    tmp_path: Path,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={"case-1": {"A": "2", "B": "3", "C": "2", "D": "3"}},
    )

    class _DriftingRuntime(_Runtime):
        def scan_pair(self, pair: dict[str, object], plan: object) -> AnswerLineDeletionScan:
            scan = super().scan_pair(pair, plan)
            source_run = source / "run.json"
            source_run.write_text(
                source_run.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            return scan

    output = tmp_path / "output"
    with pytest.raises(AnswerLineDeletionRunError, match="source snapshot changed"):
        run_answer_line_deletion(
            _config(source, output),
            runtime=_DriftingRuntime(),
        )
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["counts"]["checkpointed_pairs"] == 1
    assert not (output / "answer_line_deletion_records.jsonl").exists()
    assert not (output / "pair_status_records.jsonl").exists()
    assert not (output / "answer_line_deletion_summary.json").exists()


def test_completed_resume_rejects_semantic_tampering_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    source = _completed_cot_swap(
        tmp_path,
        answers={"case-1": {"A": "2", "B": "3", "C": "2", "D": "3"}},
    )
    output = tmp_path / "output"
    run_answer_line_deletion(_config(source, output), runtime=_Runtime())
    records_path = output / "answer_line_deletion_records.jsonl"
    records = _read_jsonl(records_path)
    records[0]["events"]["answer_line_deleted_restored_to_a"] = True  # type: ignore[index]
    _write_jsonl(records_path, records)
    manifest_path = output / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][records_path.name]["sha256"] = hashlib.sha256(
        records_path.read_bytes()
    ).hexdigest()
    _write_json(manifest_path, manifest)

    with pytest.raises(AnswerLineDeletionRunError, match="completed run validation failed"):
        run_answer_line_deletion(_config(source, output, resume=True))


def test_runtime_and_runner_share_final_pdf_generation_constants() -> None:
    deletion_protocol = importlib.import_module(
        "typo_cot.experiments.answer_line_deletion.protocol"
    )
    cot_protocol = importlib.import_module("typo_cot.experiments.cot_swap.protocol")
    runner = importlib.import_module("typo_cot.experiments.answer_line_deletion.runner")
    runtime = importlib.import_module("typo_cot.experiments.answer_line_deletion.runtime")
    assert deletion_protocol.GENERATION is cot_protocol.GENERATION
    assert runner._GENERATION is deletion_protocol.GENERATION
    assert runtime._GENERATION is deletion_protocol.GENERATION
    assert deletion_protocol.GENERATION["max_new_tokens"] == 16


def test_huggingface_runtime_batches_both_arms_and_trims_each_row_at_its_eos() -> None:
    pytest.importorskip("torch")
    import torch

    class _Tokenizer:
        padding_side = "left"
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, value: object, **kwargs: object) -> dict[str, object]:
            texts = value if isinstance(value, list) else [value]
            rows = [[1, *[ord(character) for character in str(text)]] for text in texts]
            if not kwargs.get("padding"):
                return {"input_ids": rows[0]}
            width = max(map(len, rows))
            padded = [[0] * (width - len(row)) + row for row in rows]
            masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(masks),
            }

        def decode(self, token_ids: list[int], **kwargs: object) -> str:
            visible = [token for token in token_ids if token != 99]
            return "The answer is 2." if visible[0] == 41 else "The answer is 3."

    class _Model:
        def generate(self, *, input_ids: object, **kwargs: object) -> object:
            assert kwargs["max_new_tokens"] == 16
            assert kwargs["do_sample"] is False
            assert kwargs["eos_token_id"] == [99]
            prefix = input_ids
            suffix = torch.tensor([[41, 99, 0], [42, 43, 99]], device=prefix.device)
            return torch.cat((prefix, suffix), dim=1)

    runtime = object.__new__(HuggingFaceAnswerLineDeletionRuntime)
    runtime.config = _config(Path("source"), Path("output"))
    runtime._torch = torch
    runtime.tokenizer = _Tokenizer()
    runtime.model = _Model()
    runtime.device = torch.device("cpu")
    runtime.effective_eos_token_ids = (99,)
    runtime_pair = _pair("case-1")
    for side in ("clean", "edited"):
        side_payload = runtime_pair[side]
        assert isinstance(side_payload, dict)
        side_payload["prompt_token_count"] = 1 + len(str(side_payload["prompt"]))
    plan = build_answer_line_deletion_plan(
        runtime_pair,
        source_a_answer="2",
        source_c_answer="2",
        source_record_sha256="a" * 64,
        prepared_record_sha256="b" * 64,
    )
    scan = runtime.scan_pair(runtime_pair, plan)
    assert tuple(scan.generations) == ARM_ORDER
    assert scan.generations["complete"].token_ids == (41, 99)
    assert scan.generations["answer-line-deleted"].token_ids == (42, 43, 99)
    assert scan.generations["complete"].value == "2"
    assert scan.generations["answer-line-deleted"].value == "3"


def test_docs_expose_gpu0_command_protocol_conflict_and_empty_prefix_diagnostic() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    docs = (root / "docs" / "answer-line-deletion.md").read_text(encoding="utf-8")
    for text in (readme, docs):
        assert "CUDA_VISIBLE_DEVICES=0" in text
        assert "--cot-swap-run" in text
        assert "--max-pairs 150" in text
        assert "16" in text
        assert "256" in text
    assert "179/333" in docs
    assert "334/450" in docs
    assert "prefix_became_empty" in docs
