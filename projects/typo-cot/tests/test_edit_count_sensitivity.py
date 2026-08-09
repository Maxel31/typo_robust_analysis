"""Table 8 edit-count sensitivity contracts written before implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.cot_swap.protocol import protocol_for as cot_swap_protocol_for
from typo_cot.experiments.edit_count_sensitivity import (
    PUBLISHED_REFERENCE,
    EditCountSensitivityConfig,
    EditCountSensitivityInputError,
    EditCountSensitivityResult,
    run_edit_count_sensitivity,
)
from typo_cot.experiments.edit_count_sensitivity.protocol import (
    ACCURACY_BENCHMARKS,
    EXPECTED_ACCURACY_SETTING_COUNT,
    EXPECTED_ACCURACY_SETTINGS,
)


OUTPUT_NAMES = {
    "edit_count_records.jsonl",
    "edit_count_summary.json",
    "table8_edit_count.csv",
    "table8_edit_count.md",
    "table8_edit_count.tex",
    "run.json",
}
GEMMA = "google/gemma-3-4b-it"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _answer(value: str, correct: bool) -> dict[str, object]:
    return {
        "value": value,
        "is_extracted": bool(value),
        "is_correct": correct,
        "method": "primary:fixture" if value else "unextractable",
        "primary_method": "fixture" if value else "no_match",
        "confidence": 1.0 if value else 0.0,
    }


def _pair(
    *,
    model: str,
    benchmark: str,
    count: int,
    sample_id: str,
    clean_correct: bool,
    edited_correct: bool,
) -> dict[str, object]:
    clean_value = "2" if clean_correct else "3"
    edited_value = "2" if edited_correct else "3"
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": model,
        "benchmark": benchmark,
        "targeting": "attribution-4",
        "seed": 42,
        "num_edits_requested": count,
        "num_candidates": 4,
        "num_target_attempts": count,
        "num_aligned_words": count,
        "gold_answer": "2",
        "subset": None,
        "clean": {
            "prompt": f"Question {sample_id}\nSolution:",
            "prompt_token_count": 8,
            "continuation": f"Clean reasoning for {sample_id}.\nThe answer is {clean_value}.",
            "continuation_token_count": 12,
            "answer": _answer(clean_value, clean_correct),
        },
        "edited": {
            "prompt": f"Edited-{count} question {sample_id}\nSolution:",
            "prompt_token_count": 9,
            "continuation": f"Edited reasoning for {sample_id}.\nThe answer is {edited_value}.",
            "continuation_token_count": 12,
            "answer": _answer(edited_value, edited_correct),
        },
        "answer_changed": clean_value != edited_value,
        "target_attempts": [
            {
                "token_index": index,
                "token_text": f"word-{index}",
                "operation": "substitution",
            }
            for index in range(count)
        ],
        "aligned_words": [
            {
                "clean_token_index": index,
                "edited_token_index": index,
                "clean_word": f"word-{index}",
                "edited_word": f"wprd-{index}",
            }
            for index in range(count)
        ],
    }


def _write_prepare_run(
    directory: Path,
    *,
    model: str = GEMMA,
    benchmark: str = "gsm8k",
    count: int,
    correctness: Sequence[tuple[bool, bool]],
) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    rows = [
        _pair(
            model=model,
            benchmark=benchmark,
            count=count,
            sample_id=f"sample-{index}",
            clean_correct=clean_correct,
            edited_correct=edited_correct,
        )
        for index, (clean_correct, edited_correct) in enumerate(correctness)
    ]
    pairs_path = directory / "pairs.jsonl"
    _write_jsonl(pairs_path, rows)
    dataset_hash = hashlib.sha256(
        json.dumps([row["sample_id"] for row in rows], separators=(",", ":")).encode()
    ).hexdigest()
    run_path = directory / "run.json"
    _write_json(
        run_path,
        {
            "schema_version": "prepare-edited-pairs-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "prepare-edited-pairs",
            "status": "completed",
            "arguments": {
                "model": model,
                "benchmark": benchmark,
                "targeting": "attribution-4",
                "num_edits": count,
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
                "model": model,
                "model_revision": "fixture-revision",
                "benchmark_dataset_loader": benchmark,
                "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
                "dataset_sample_count": len(rows),
                "dataset_records_sha256": dataset_hash,
                "dataset_samples_per_subset": None,
                "random_seed_algorithm": "sha256-first-64-bits/v1",
                "generation_protocol": "explicit-greedy-generation/v1",
                "target_position": "maximum-logit-after-first-cot-token",
                "alignment": "actual-edited-word-final-token",
                "historical_compatibility_notes": [],
            },
        },
    )
    return pairs_path, run_path


def _metric(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _write_cot_swap_run(
    directory: Path,
    *,
    pairs_path: Path,
    source_run_path: Path,
    model: str = GEMMA,
    benchmark: str = "gsm8k",
    count: int,
    events: Sequence[tuple[bool, bool]],
) -> Path:
    directory.mkdir(parents=True)
    source_manifest = json.loads(source_run_path.read_text(encoding="utf-8"))
    source_provenance = source_manifest["provenance"]
    source_rows = {
        json.loads(line)["sample_id"]: hashlib.sha256(line.encode("utf-8")).hexdigest()
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
    }
    rows: list[dict[str, object]] = []
    for index, (denominator, restored) in enumerate(events):
        answers = {
            "A": "2",
            "B": "3" if denominator else "2",
            "C": "2" if restored or not denominator else "3",
            "D": "2",
        }
        rows.append(
            {
                "schema_version": "cot-swap-record/v1",
                "paper_sha256": PAPER_SHA256,
                "operation": "cot-swap",
                "model": model,
                "benchmark": benchmark,
                "targeting": "attribution-4",
                "source_num_edits": count,
                "sample_id": f"sample-{index}",
                "gold_answer": "2",
                "source": {
                    "pairs_sha256": _sha256(pairs_path),
                    "source_run_sha256": _sha256(source_run_path),
                    "source_record_sha256": source_rows[f"sample-{index}"],
                },
                "cells": {
                    cell: {
                        "answer": _answer(value, value == "2"),
                        "equal_to_a": cell == "A" or value == answers["A"],
                    }
                    for cell, value in answers.items()
                },
                "events": {
                    "clean_correct": True,
                    "both_changed": denominator,
                    "question_only_changed": answers["C"] != answers["A"],
                    "cot_only_changed": False,
                    "restoration_denominator": denominator,
                    "b_to_c_restored": restored if denominator or restored else None,
                },
            }
        )
    records_path = directory / "cot_swap_records.jsonl"
    _write_jsonl(records_path, rows)
    statuses_path = directory / "pair_status_records.jsonl"
    _write_jsonl(
        statuses_path,
        [
            {
                "schema_version": "cot-swap-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": model,
                "benchmark": benchmark,
                "targeting": "attribution-4",
                "sample_id": row["sample_id"],
                "source_record_sha256": source_rows[str(row["sample_id"])],
                "edit_valid": True,
                "template_eligible": True,
                "exclusion_reasons": [],
                "selected_for_execution": True,
                "execution_status": "completed",
                "included_in_change_denominator": True,
                "included_in_restoration_denominator": row["events"]["restoration_denominator"],
            }
            for row in rows
        ],
    )
    denominator = sum(flag for flag, _ in events)
    restored = sum(restored for flag, restored in events if flag)
    summary_path = directory / "cot_swap_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": "cot-swap-summary/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "cot-swap",
            "model": model,
            "benchmark": benchmark,
            "targeting": "attribution-4",
            "source_num_edits": count,
            "source": {
                "pairs": str(pairs_path.resolve()),
                "pairs_sha256": _sha256(pairs_path),
                "source_run": str(source_run_path.resolve()),
                "source_run_sha256": _sha256(source_run_path),
                "source_schema": "prepare-edited-pairs/v1",
                "model_revision": source_provenance["model_revision"],
                "dataset_records_sha256": source_provenance["dataset_records_sha256"],
                "dataset_sample_count": source_provenance["dataset_sample_count"],
                "record_count": sum(1 for _ in pairs_path.open(encoding="utf-8")),
            },
            "counts": {
                "source_records": len(source_rows),
                "executed_pairs": len(rows),
                "failed_pairs": 0,
            },
            "metrics": {"restoration": _metric(restored, denominator)},
            "comparability": {
                "status": "fresh-edit-count-sensitivity-setting",
                "historical_cohort_identity": False,
            },
        },
    )
    run_path = directory / "run.json"
    outputs = {
        path.name: {
            "records": len(rows) if path.suffix == ".jsonl" else 1,
            "sha256": _sha256(path),
        }
        for path in (records_path, statuses_path, summary_path)
    }
    protocol = cot_swap_protocol_for(count)
    source = {
        "pairs": str(pairs_path.resolve()),
        "pairs_sha256": _sha256(pairs_path),
        "source_run": str(source_run_path.resolve()),
        "source_run_sha256": _sha256(source_run_path),
        "source_schema": "prepare-edited-pairs/v1",
        "model_revision": source_provenance["model_revision"],
        "dataset_records_sha256": source_provenance["dataset_records_sha256"],
        "dataset_sample_count": source_provenance["dataset_sample_count"],
        "record_count": len(source_rows),
    }
    _write_json(
        run_path,
        {
            "schema_version": "cot-swap-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "cot-swap",
            "status": "completed",
            "arguments": {
                "model": model,
                "benchmark": benchmark,
                "pairs": str(pairs_path.resolve()),
                "targeting": "attribution-4",
                "source_num_edits": count,
                "gpu_id": "0",
                "limit": None,
                "output_dir": str(directory.resolve()),
            },
            "protocol": protocol,
            "protocol_sha256": _canonical_sha256(protocol),
            "source": source,
            "comparability": {
                "status": "fresh-edit-count-sensitivity-setting",
                "historical_cohort_identity": False,
            },
            "counts": {
                "source_records": len(source_rows),
                "selected_pairs": len(rows),
                "checkpointed_pairs": len(rows),
                "failed_pairs": 0,
                "publication_failures": 0,
            },
            "failures": [],
            "outputs": outputs,
        },
    )
    return run_path


def _write_partial_fixture(
    root: Path,
) -> tuple[Path, Path, dict[int, tuple[Path, Path]]]:
    pairs_root = root / "pairs"
    cot_root = root / "cot"
    sources: dict[int, tuple[Path, Path]] = {}
    correctness = {
        1: ((True, True), (False, True)),
        2: ((True, True), (False, False)),
        4: ((True, False), (False, False)),
    }
    cot_events = {
        1: ((True, True), (False, False)),
        2: ((True, True), (True, False)),
        4: ((True, False), (False, False)),
    }
    for count in (1, 2, 4):
        source = _write_prepare_run(
            pairs_root / "arbitrary" / str(count),
            count=count,
            correctness=correctness[count],
        )
        sources[count] = source
        _write_cot_swap_run(
            cot_root / "arbitrary" / str(count),
            pairs_path=source[0],
            source_run_path=source[1],
            count=count,
            events=cot_events[count],
        )
    return pairs_root, cot_root, sources


def _command_parser() -> argparse.ArgumentParser:
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices["edit-count-sensitivity"]


def test_catalog_and_parser_freeze_the_documented_cpu_command() -> None:
    spec = get_experiment("edit-count-sensitivity")
    assert spec.status == "implemented"
    assert spec.compute == "cpu"
    assert spec.required_arguments == (
        "--pairs-root",
        "--cot-swap-runs-root",
        "--edit-counts",
        "--output-dir",
    )
    assert spec.outputs == (
        "edit_count_records.jsonl",
        "edit_count_summary.json",
        "table8_edit_count.csv",
        "table8_edit_count.md",
        "table8_edit_count.tex",
        "run.json",
    )

    args = cli_module._parser().parse_args(
        [
            "edit-count-sensitivity",
            "--pairs-root",
            "results/edit-count-pairs",
            "--cot-swap-runs-root",
            "results/edit-count-cot-swap",
            "--edit-counts",
            "1",
            "2",
            "4",
            "--output-dir",
            "results/edit-count-sensitivity",
        ]
    )
    assert args.pairs_root == Path("results/edit-count-pairs")
    assert args.cot_swap_runs_root == Path("results/edit-count-cot-swap")
    assert args.edit_counts == [1, 2, 4]
    assert args.output_dir == Path("results/edit-count-sensitivity")


@pytest.mark.parametrize(
    "argv",
    (
        [
            "--cot-swap-runs-root",
            "cot",
            "--edit-counts",
            "1",
            "2",
            "4",
            "--output-dir",
            "out",
        ],
        [
            "--pairs-root",
            "pairs",
            "--edit-counts",
            "1",
            "2",
            "4",
            "--output-dir",
            "out",
        ],
    ),
)
def test_parser_requires_both_distinct_producer_roots(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _command_parser().parse_args(argv)
    assert exc_info.value.code == 2


def test_cli_passes_all_public_arguments_to_the_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[EditCountSensitivityConfig] = []

    def fake_run(config: EditCountSensitivityConfig) -> EditCountSensitivityResult:
        captured.append(config)
        return EditCountSensitivityResult(
            output_dir=config.output_dir.resolve(),
            records_path=config.output_dir / "edit_count_records.jsonl",
            summary_path=config.output_dir / "edit_count_summary.json",
            csv_path=config.output_dir / "table8_edit_count.csv",
            markdown_path=config.output_dir / "table8_edit_count.md",
            latex_path=config.output_dir / "table8_edit_count.tex",
            run_path=config.output_dir / "run.json",
            accuracy_settings=1,
            restoration_settings=1,
        )

    monkeypatch.setattr(cli_module, "run_edit_count_sensitivity", fake_run)
    argv = [
        "edit-count-sensitivity",
        "--pairs-root",
        str(tmp_path / "pairs"),
        "--cot-swap-runs-root",
        str(tmp_path / "cot"),
        "--edit-counts",
        "1",
        "2",
        "4",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert main(argv) == 0
    assert captured == [
        EditCountSensitivityConfig(
            pairs_root=tmp_path / "pairs",
            cot_swap_runs_root=tmp_path / "cot",
            edit_counts=(1, 2, 4),
            output_dir=tmp_path / "out",
        )
    ]
    assert "1 accuracy setting(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("edit_counts", "message"),
    (
        ((1, 1, 4), "unique"),
        ((1, 3, 4), "1, 2, or 4"),
        ((2, 1, 4), "ascending"),
        ((), "must not be empty"),
    ),
)
def test_config_rejects_ambiguous_edit_count_sets(
    tmp_path: Path,
    edit_counts: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EditCountSensitivityConfig(
            pairs_root=tmp_path / "pairs",
            cot_swap_runs_root=tmp_path / "cot",
            edit_counts=edit_counts,
            output_dir=tmp_path / "out",
        )


def test_runner_recomputes_the_two_table8_denominators_without_intersection(
    tmp_path: Path,
) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    output = tmp_path / "table8"

    result = run_edit_count_sensitivity(
        EditCountSensitivityConfig(
            pairs_root=pairs_root,
            cot_swap_runs_root=cot_root,
            edit_counts=(1, 2, 4),
            output_dir=output,
        )
    )

    assert result.accuracy_settings == 1
    assert result.restoration_settings == 1
    assert {path.name for path in output.iterdir()} == OUTPUT_NAMES
    rows = [
        json.loads(line) for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["analysis"] for row in rows] == ["accuracy", "restoration"]
    accuracy = rows[0]
    assert accuracy["full_conditions"] == {
        "0": {"correct": 1, "denominator": 2, "rate": 0.5},
        "1": {"correct": 2, "denominator": 2, "rate": 1.0},
        "2": {"correct": 1, "denominator": 2, "rate": 0.5},
        "4": {"correct": 0, "denominator": 2, "rate": 0.0},
    }
    assert accuracy["matched_conditions"]["sample_count"] == 2
    assert accuracy["clean_above_four_edits"] is True
    restoration = rows[1]
    assert restoration["by_edit_count"] == {
        "1": {"denominator": 1, "restored": 1, "rate": 1.0},
        "2": {"denominator": 2, "restored": 1, "rate": 0.5},
        "4": {"denominator": 1, "restored": 0, "rate": 0.0},
    }

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["accuracy"]["equal_setting_mean"] == {
        "0": 0.5,
        "1": 1.0,
        "2": 0.5,
        "4": 0.0,
    }
    assert summary["accuracy"]["matched_items"]["sample_count"] == 2
    assert summary["restoration"]["pooled"]["2"] == {
        "denominator": 2,
        "restored": 1,
        "rate": 0.5,
    }
    assert summary["coverage"]["complete_accuracy_grid"] is False
    assert len(summary["coverage"]["accuracy"]["missing_expected_settings"]) == 50
    assert summary["coverage"]["accuracy"]["unexpected_settings"] == []
    assert summary["coverage"]["complete_restoration_grid"] is False
    assert summary["comparability"]["status"] == "partial-valid-analysis"
    assert summary["published_reference"] == PUBLISHED_REFERENCE

    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "edit-count-sensitivity-run/v1"
    assert manifest["paper_sha256"] == PAPER_SHA256
    assert manifest["operation"] == "edit-count-sensitivity"
    assert manifest["status"] == "completed"
    assert set(manifest["outputs"]) == OUTPUT_NAMES - {"run.json"}
    for name, metadata in manifest["outputs"].items():
        path = output / name
        assert metadata == {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def test_published_reference_is_the_literal_final_pdf_table8() -> None:
    assert PUBLISHED_REFERENCE["accuracy"] == {
        "equal_setting_mean": {"0": 0.521, "1": 0.500, "2": 0.483, "4": 0.460},
        "matched_81812_items": {"0": 0.546, "1": 0.525, "2": 0.509, "4": 0.488},
        "clean_above_four_settings": {"numerator": 51, "denominator": 51},
    }
    assert PUBLISHED_REFERENCE["restoration_pooled"] == {
        "1": {"restored": 811, "denominator": 908, "rate": 811 / 908},
        "2": {"restored": 988, "denominator": 1123, "rate": 988 / 1123},
        "4": {"restored": 1217, "denominator": 1415, "rate": 1217 / 1415},
    }
    assert len(PUBLISHED_REFERENCE["restoration_settings"]) == 6


def test_recovered_accuracy_grid_has_the_papers_exact_51_setting_shape() -> None:
    assert EXPECTED_ACCURACY_SETTING_COUNT == 51
    assert len(EXPECTED_ACCURACY_SETTINGS) == EXPECTED_ACCURACY_SETTING_COUNT
    assert len(set(EXPECTED_ACCURACY_SETTINGS)) == EXPECTED_ACCURACY_SETTING_COUNT
    assert {benchmark for _, benchmark in EXPECTED_ACCURACY_SETTINGS} == set(ACCURACY_BENCHMARKS)
    assert (
        "Qwen/Qwen2.5-3B-Instruct",
        "arc",
    ) not in EXPECTED_ACCURACY_SETTINGS
    assert {
        benchmark
        for model, benchmark in EXPECTED_ACCURACY_SETTINGS
        if model == "Qwen/Qwen2.5-3B-Instruct"
    } == {"gsm8k", "mmlu", "mmlu-pro"}


def test_clean_condition_must_agree_across_edit_count_sources(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    cot_root = tmp_path / "cot"
    for count in (1, 2, 4):
        correctness = (
            ((False, True), (False, False))
            if count == 2
            else (
                (True, True),
                (False, False),
            )
        )
        source = _write_prepare_run(
            pairs_root / str(count),
            count=count,
            correctness=correctness,
        )
        _write_cot_swap_run(
            cot_root / str(count),
            pairs_path=source[0],
            source_run_path=source[1],
            count=count,
            events=((True, True), (False, False)),
        )

    with pytest.raises(EditCountSensitivityInputError, match="clean condition"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )
    assert not (tmp_path / "out").exists()


def test_cot_swap_protocol_hash_tampering_fails_before_publication(tmp_path: Path) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    manifest_path = next(cot_root.rglob("run.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["source_generation"]["seed"] = 7
    _write_json(manifest_path, manifest)

    with pytest.raises(EditCountSensitivityInputError, match="protocol SHA-256"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_self_consistent_nonpublic_cot_swap_protocol_is_rejected(tmp_path: Path) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    manifest_path = next(cot_root.rglob("run.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["forged_rule"] = "self-consistent-but-not-the-public-protocol"
    manifest["protocol_sha256"] = _canonical_sha256(manifest["protocol"])
    _write_json(manifest_path, manifest)

    with pytest.raises(EditCountSensitivityInputError, match="public CoT-swap protocol"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_cot_swap_completed_records_must_match_completed_statuses(tmp_path: Path) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    statuses_path = next(cot_root.rglob("pair_status_records.jsonl"))
    statuses = [json.loads(line) for line in statuses_path.read_text(encoding="utf-8").splitlines()]
    statuses[0]["execution_status"] = "template-excluded"
    _write_jsonl(statuses_path, statuses)
    manifest_path = statuses_path.parent / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][statuses_path.name]["sha256"] = _sha256(statuses_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(EditCountSensitivityInputError, match="completed status IDs"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_accuracy_counts_must_share_model_revision_and_dataset_cohort(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    cot_root = tmp_path / "cot"
    for count in (1, 2, 4):
        source = _write_prepare_run(
            pairs_root / str(count),
            count=count,
            correctness=((True, True), (False, False)),
        )
        if count == 2:
            manifest = json.loads(source[1].read_text(encoding="utf-8"))
            manifest["provenance"]["dataset_records_sha256"] = "0" * 64
            _write_json(source[1], manifest)
        _write_cot_swap_run(
            cot_root / str(count),
            pairs_path=source[0],
            source_run_path=source[1],
            count=count,
            events=((True, True), (False, False)),
        )

    with pytest.raises(EditCountSensitivityInputError, match="dataset cohort"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_cot_swap_output_hash_tampering_fails_before_publication(tmp_path: Path) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    records = next(cot_root.rglob("cot_swap_records.jsonl"))
    records.write_text(records.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(EditCountSensitivityInputError, match="SHA-256"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_restoration_cannot_succeed_outside_its_denominator(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    cot_root = tmp_path / "cot"
    for count in (1, 2, 4):
        source = _write_prepare_run(
            pairs_root / str(count),
            count=count,
            correctness=((True, False),),
        )
        _write_cot_swap_run(
            cot_root / str(count),
            pairs_path=source[0],
            source_run_path=source[1],
            count=count,
            events=((False, count == 1),),
        )

    with pytest.raises(EditCountSensitivityInputError, match="restored.*denominator"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_restoration_denominator_cannot_omit_an_a_correct_b_change(tmp_path: Path) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    records_path = next(cot_root.rglob("cot_swap_records.jsonl"))
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records[0]["events"]["restoration_denominator"] = False
    records[0]["events"]["b_to_c_restored"] = None
    _write_jsonl(records_path, records)

    statuses_path = records_path.parent / "pair_status_records.jsonl"
    statuses = [json.loads(line) for line in statuses_path.read_text(encoding="utf-8").splitlines()]
    statuses[0]["included_in_restoration_denominator"] = False
    _write_jsonl(statuses_path, statuses)

    summary_path = records_path.parent / "cot_swap_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    denominator = sum(row["events"]["restoration_denominator"] is True for row in records)
    restored = sum(row["events"]["b_to_c_restored"] is True for row in records)
    summary["metrics"]["restoration"] = _metric(restored, denominator)
    _write_json(summary_path, summary)

    manifest_path = records_path.parent / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for output_path in (records_path, statuses_path, summary_path):
        manifest["outputs"][output_path.name]["sha256"] = _sha256(output_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(EditCountSensitivityInputError, match="denominator contradicts"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_duplicate_prepare_setting_and_count_is_rejected(tmp_path: Path) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    _write_prepare_run(
        pairs_root / "duplicate",
        count=1,
        correctness=((True, True), (False, True)),
    )

    with pytest.raises(EditCountSensitivityInputError, match="duplicate.*prepare"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=tmp_path / "out",
            )
        )


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "user.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_edit_count_sensitivity(
            EditCountSensitivityConfig(
                pairs_root=pairs_root,
                cot_swap_runs_root=cot_root,
                edit_counts=(1, 2, 4),
                output_dir=output,
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_cli_executes_the_documented_command_on_a_partial_verified_grid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pairs_root, cot_root, _ = _write_partial_fixture(tmp_path)
    output = tmp_path / "out"

    assert (
        main(
            [
                "edit-count-sensitivity",
                "--pairs-root",
                str(pairs_root),
                "--cot-swap-runs-root",
                str(cot_root),
                "--edit-counts",
                "1",
                "2",
                "4",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert "1 accuracy setting(s)" in capsys.readouterr().out
    assert {path.name for path in output.iterdir()} == OUTPUT_NAMES
