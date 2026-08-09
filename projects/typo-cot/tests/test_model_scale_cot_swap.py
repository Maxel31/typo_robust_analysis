"""Appendix C/Table 9 model-scale CoT-swap contracts written before implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
import typo_cot.experiments.model_scale_cot_swap.runner as runner_module
from typo_cot.cli import main
from typo_cot.data.cohorts import load_sample_id_cohort
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.cot_swap import (
    CELL_ORDER,
    CotSwapConfig,
    CotSwapGeneration,
    CotSwapInputUse,
    CotSwapScan,
    run_cot_swap,
)
from typo_cot.experiments.model_scale_cot_swap import (
    PUBLISHED_REFERENCE,
    ModelScaleCotSwapConfig,
    ModelScaleCotSwapResult,
    run_model_scale_cot_swap,
)
from typo_cot.experiments.model_scale_cot_swap.aggregation import build_analysis
from typo_cot.experiments.model_scale_cot_swap.protocol import (
    ANALYSIS_PROTOCOL,
    EXPECTED_MODELS,
    MODEL_LABELS,
)
from typo_cot.experiments.model_scale_cot_swap.source import ModelScaleInputs
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PrepareEditedPairsConfig,
    run_prepare_edited_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = PROJECT_ROOT / "data" / "cohorts" / "model_scale_mmlu_first500.json"
OUTPUT_NAMES = {
    "model_scale_records.jsonl",
    "model_scale_summary.json",
    "table9_model_scale.csv",
    "table9_model_scale.md",
    "table9_model_scale.tex",
    "run.json",
}


def test_implementation_identity_covers_shared_validation_dependencies() -> None:
    identity = runner_module._code_identity()
    paths = [entry["path"] for entry in identity["files"]]

    assert paths == sorted(paths)
    assert identity["python_file_count"] == len(paths)
    assert {
        "data/cohorts.py",
        "experiments/edit_count_sensitivity/source.py",
        "experiments/model_scale_cot_swap/aggregation.py",
    }.issubset(paths)


def _command_parser() -> argparse.ArgumentParser:
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices["model-scale-cot-swap"]


def _event_record(
    sample_id: str,
    *,
    clean_correct: bool,
    both_changed: bool,
    question_changed: bool,
    cot_changed: bool,
    restored: bool | None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "events": {
            "clean_correct": clean_correct,
            "both_changed": both_changed,
            "question_only_changed": question_changed,
            "cot_only_changed": cot_changed,
            "restoration_denominator": clean_correct and both_changed,
            "b_to_c_restored": restored,
        },
    }


def _run(
    model: str,
    records: tuple[dict[str, object], ...],
    *,
    source_records: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        benchmark="mmlu",
        edit_count=4,
        records=records,
        run_path=Path(f"/{model.rsplit('/', 1)[-1]}/run.json"),
        records_path=Path(f"/{model.rsplit('/', 1)[-1]}/cot_swap_records.jsonl"),
        summary_path=Path(f"/{model.rsplit('/', 1)[-1]}/cot_swap_summary.json"),
        run_sha256="a" * 64,
        records_sha256="b" * 64,
        source_pairs_sha256="c" * 64,
        source_run_sha256="d" * 64,
        source_record_count=source_records,
    )


def test_catalog_and_parser_freeze_the_documented_cpu_builder() -> None:
    spec = get_experiment("model-scale-cot-swap")
    assert spec.status == "implemented"
    assert spec.compute == "cpu"
    assert spec.required_arguments == (
        "--pairs-root",
        "--cot-swap-runs-root",
        "--cohort",
        "--output-dir",
    )
    assert spec.outputs == tuple(sorted(OUTPUT_NAMES - {"run.json"})) + ("run.json",)

    args = cli_module._parser().parse_args(
        [
            "model-scale-cot-swap",
            "--pairs-root",
            "results/model-scale-pairs",
            "--cot-swap-runs-root",
            "results/model-scale-cot-swap-runs",
            "--cohort",
            "projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json",
            "--output-dir",
            "results/model-scale-cot-swap",
        ]
    )
    assert args.pairs_root == Path("results/model-scale-pairs")
    assert args.cot_swap_runs_root == Path("results/model-scale-cot-swap-runs")
    assert args.cohort == Path("projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json")
    assert args.output_dir == Path("results/model-scale-cot-swap")


@pytest.mark.parametrize(
    "missing",
    ("--pairs-root", "--cot-swap-runs-root", "--cohort", "--output-dir"),
)
def test_parser_requires_every_distinct_builder_input(missing: str) -> None:
    arguments = {
        "--pairs-root": "pairs",
        "--cot-swap-runs-root": "cot",
        "--cohort": "cohort.json",
        "--output-dir": "out",
    }
    argv = [item for pair in arguments.items() if pair[0] != missing for item in pair]
    with pytest.raises(SystemExit) as exc_info:
        _command_parser().parse_args(argv)
    assert exc_info.value.code == 2


def test_cli_passes_all_builder_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[ModelScaleCotSwapConfig] = []

    def fake_run(config: ModelScaleCotSwapConfig) -> ModelScaleCotSwapResult:
        captured.append(config)
        return ModelScaleCotSwapResult(
            output_dir=config.output_dir.resolve(),
            records_path=config.output_dir / "model_scale_records.jsonl",
            summary_path=config.output_dir / "model_scale_summary.json",
            csv_path=config.output_dir / "table9_model_scale.csv",
            markdown_path=config.output_dir / "table9_model_scale.md",
            latex_path=config.output_dir / "table9_model_scale.tex",
            run_path=config.output_dir / "run.json",
            settings=1,
        )

    monkeypatch.setattr(cli_module, "run_model_scale_cot_swap", fake_run)
    argv = [
        "model-scale-cot-swap",
        "--pairs-root",
        str(tmp_path / "pairs"),
        "--cot-swap-runs-root",
        str(tmp_path / "cot"),
        "--cohort",
        str(tmp_path / "cohort.json"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert main(argv) == 0
    assert captured == [
        ModelScaleCotSwapConfig(
            pairs_root=tmp_path / "pairs",
            cot_swap_runs_root=tmp_path / "cot",
            cohort=tmp_path / "cohort.json",
            output_dir=tmp_path / "out",
        )
    ]
    assert "built Table 9 from 1 model setting(s)" in capsys.readouterr().out


def test_public_cohort_freezes_the_recovered_first500_selector() -> None:
    cohort = load_sample_id_cohort(COHORT_PATH)
    assert cohort.schema_version == "sample-id-cohort/v1"
    assert cohort.paper_sha256 == PAPER_SHA256
    assert cohort.cohort_id == "model-scale-mmlu-first500"
    assert cohort.benchmark == "mmlu"
    assert cohort.provenance == "submitted-source-recovered"
    assert len(cohort.sample_ids) == len(set(cohort.sample_ids)) == 500
    assert cohort.sample_ids_sha256 == (
        "7663efab7085892e60ba7a68c6b3c857101468aef2a2cff5acda998d7b6c637d"
    )
    assert cohort.sample_ids[:3] == (
        "mmlu_abstract_algebra_0000",
        "mmlu_abstract_algebra_0001",
        "mmlu_abstract_algebra_0002",
    )
    assert cohort.model_selected_sample_counts == {
        model: (
            500
            if model in EXPECTED_MODELS[2:4] + EXPECTED_MODELS[6:7] + EXPECTED_MODELS[8:9]
            else 250
        )
        for model in EXPECTED_MODELS
    }


def test_known_table9_cohort_rejects_a_self_consistent_reordered_selector(
    tmp_path: Path,
) -> None:
    payload = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    payload["sample_ids"][0], payload["sample_ids"][1] = (
        payload["sample_ids"][1],
        payload["sample_ids"][0],
    )
    payload["sample_ids_sha256"] = hashlib.sha256(
        json.dumps(payload["sample_ids"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen cohort.*ordered ID SHA-256"):
        load_sample_id_cohort(changed)


def test_protocol_and_reference_are_literal_final_pdf_table9() -> None:
    assert EXPECTED_MODELS == (
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-72B-Instruct",
    )
    assert PUBLISHED_REFERENCE == {
        "google/gemma-3-1b-it": (65, 19, 14, 12, 9, 19),
        "google/gemma-3-4b-it": (129, 32, 10, 28, 23, 32),
        "google/gemma-3-12b-it": (351, 41, 11, 29, 33, 41),
        "google/gemma-3-27b-it": (383, 33, 8, 36, 30, 33),
        "meta-llama/Llama-3.2-1B-Instruct": (119, 49, 24, 29, 30, 49),
        "meta-llama/Llama-3.2-3B-Instruct": (142, 36, 11, 32, 27, 36),
        "meta-llama/Llama-3.1-70B-Instruct": (411, 35, 2, 33, 33, 35),
        "mistralai/Mistral-7B-Instruct-v0.3": (137, 28, 8, 25, 24, 28),
        "Qwen/Qwen2.5-72B-Instruct": (331, 10, 7, 12, 8, 10),
    }
    assert ANALYSIS_PROTOCOL["inference"] == "descriptive-integer-counts-only"
    assert ANALYSIS_PROTOCOL["qwen_72b_interpretation"] == "directional-only-n_b-10"
    assert MODEL_LABELS[EXPECTED_MODELS[6]] == "Llama-3.1-70B"


def test_aggregation_uses_clean_correct_and_flip_specific_denominators() -> None:
    records = (
        _event_record(
            "a",
            clean_correct=True,
            both_changed=True,
            question_changed=False,
            cot_changed=False,
            restored=True,
        ),
        _event_record(
            "b",
            clean_correct=True,
            both_changed=True,
            question_changed=True,
            cot_changed=True,
            restored=False,
        ),
        _event_record(
            "c",
            clean_correct=True,
            both_changed=False,
            question_changed=True,
            cot_changed=False,
            restored=None,
        ),
        _event_record(
            "d",
            clean_correct=False,
            both_changed=True,
            question_changed=True,
            cot_changed=True,
            restored=None,
        ),
    )
    rows, summary = build_analysis((_run(EXPECTED_MODELS[0], records, source_records=250),))

    assert rows == (
        {
            "schema_version": "model-scale-cot-swap-record/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "model-scale-cot-swap",
            "model": EXPECTED_MODELS[0],
            "label": "Gemma-3-1B",
            "source_records": 250,
            "executed_pairs": 4,
            "n_s": 3,
            "both": {"numerator": 2, "denominator": 3, "rate": 2 / 3},
            "question_only": {"numerator": 2, "denominator": 3, "rate": 2 / 3},
            "cot_only": {"numerator": 1, "denominator": 3, "rate": 1 / 3},
            "restoration": {"numerator": 1, "denominator": 2, "rate": 0.5},
        },
    )
    assert summary["coverage"]["complete_grid"] is False
    assert summary["coverage"]["present_models"] == [EXPECTED_MODELS[0]]
    assert summary["coverage"]["missing_models"] == list(EXPECTED_MODELS[1:])
    assert summary["comparability"]["status"] == "partial-valid-analysis"
    assert summary["published_reference"] == {
        model: {
            "n_s": values[0],
            "both": values[1],
            "question_only": values[2],
            "cot_only": values[3],
            "restored": values[4],
            "n_b": values[5],
        }
        for model, values in PUBLISHED_REFERENCE.items()
    }


def test_runner_publishes_partial_valid_grid_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = SimpleNamespace(
        path=(tmp_path / "cohort.json").resolve(),
        artifact_sha256="e" * 64,
        sample_ids_sha256="f" * 64,
        sample_ids=tuple(f"id-{index}" for index in range(500)),
        to_dict=lambda: {"cohort_id": "model-scale-mmlu-first500"},
    )
    cot_run = _run(
        EXPECTED_MODELS[0],
        (
            _event_record(
                "a",
                clean_correct=True,
                both_changed=True,
                question_changed=False,
                cot_changed=False,
                restored=True,
            ),
        ),
        source_records=250,
    )
    inputs = ModelScaleInputs(cohort=cohort, prepared_runs=(), cot_swap_runs=(cot_run,))
    monkeypatch.setattr(runner_module, "discover_model_scale_inputs", lambda **kwargs: inputs)
    output = tmp_path / "out"

    result = run_model_scale_cot_swap(
        ModelScaleCotSwapConfig(
            pairs_root=tmp_path / "pairs",
            cot_swap_runs_root=tmp_path / "cot",
            cohort=tmp_path / "cohort.json",
            output_dir=output,
        )
    )

    assert result.settings == 1
    assert {path.name for path in output.iterdir()} == OUTPUT_NAMES
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "model-scale-cot-swap-run/v1"
    assert manifest["paper_sha256"] == PAPER_SHA256
    assert manifest["status"] == "completed"
    assert set(manifest["outputs"]) == OUTPUT_NAMES - {"run.json"}
    for name, metadata in manifest["outputs"].items():
        path = output / name
        assert metadata == {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    latex = result.latex_path.read_text(encoding="utf-8")
    assert r"100.0\%" in latex


def test_existing_output_directory_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "user.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(
        runner_module,
        "discover_model_scale_inputs",
        lambda **kwargs: pytest.fail("input discovery must not run when output exists"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_model_scale_cot_swap(
            ModelScaleCotSwapConfig(
                pairs_root=tmp_path / "pairs",
                cot_swap_runs_root=tmp_path / "cot",
                cohort=tmp_path / "cohort.json",
                output_dir=output,
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


class _PairRuntime:
    def __init__(self, sample_ids: tuple[str, ...]) -> None:
        self.sample_ids = sample_ids

    def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
        return [{"sample_id": sample_id} for sample_id in self.sample_ids]

    def prepare_pair(
        self,
        sample: Mapping[str, str],
        config: PrepareEditedPairsConfig,
    ) -> dict[str, object]:
        sample_id = sample["sample_id"]
        return {
            "schema_version": "prepare-edited-pairs/v1",
            "sample_id": sample_id,
            "model": config.model,
            "benchmark": config.benchmark,
            "targeting": config.targeting,
            "seed": 42,
            "num_edits_requested": 4,
            "num_candidates": 4,
            "num_target_attempts": 1,
            "num_aligned_words": 0,
            "gold_answer": "A",
            "subset": "fixture",
            "clean": {
                "prompt": f"Question: clean {sample_id}\nAnswer:",
                "prompt_token_count": 8,
                "continuation": "Clean reasoning.\nThe answer is (A).",
                "continuation_token_count": 8,
                "answer": {
                    "value": "A",
                    "is_extracted": True,
                    "is_correct": True,
                    "method": "primary:pattern_1",
                    "primary_method": "pattern_1",
                    "confidence": 1.0,
                },
            },
            "edited": {
                "prompt": f"Question: edited {sample_id}\nAnswer:",
                "prompt_token_count": 8,
                "continuation": "Edited reasoning.\nThe answer is (B).",
                "continuation_token_count": 8,
                "answer": {
                    "value": "B",
                    "is_extracted": True,
                    "is_correct": False,
                    "method": "primary:pattern_1",
                    "primary_method": "pattern_1",
                    "confidence": 1.0,
                },
            },
            "answer_changed": True,
            "target_attempts": [
                {"token_index": 3, "token_text": "clean", "operation": "substitution"}
            ],
            "aligned_words": [],
        }

    def provenance(self) -> dict[str, object]:
        return {
            "model": EXPECTED_MODELS[0],
            "model_revision": "fixture-revision",
            "benchmark_dataset_loader": "mmlu",
            "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
            "dataset_sample_count": len(self.sample_ids),
            "dataset_records_sha256": "d" * 64,
            "dataset_samples_per_subset": 50,
            "random_seed_algorithm": "sha256-first-64-bits/v1",
            "generation_protocol": "explicit-greedy-generation/v1",
            "target_position": "maximum-logit-after-first-cot-token",
            "alignment": "actual-edited-word-final-token",
            "historical_compatibility_notes": [],
        }


class _CotRuntime:
    def __init__(self, sample_ids: tuple[str, ...]) -> None:
        self.sample_ids = sample_ids

    def provenance(self) -> dict[str, object]:
        eos = [1, *[ord(cell) for cell in CELL_ORDER]]
        return {
            "operation": "cot-swap",
            "runtime": "fixture-runtime",
            "model": EXPECTED_MODELS[0],
            "requested_revision": "fixture-revision",
            "model_revision": "fixture-revision",
            "tokenizer_revision": "fixture-revision",
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
                "eos_token_id": eos,
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
            "effective_eos_token_ids": eos,
            "effective_eos_token_ids_source": "model-generation-config",
            "text_intervention": {
                "boundary": "submitted-first-[Tt]he-answer-is-filter/v1",
                "assembly": "recorded-prompt-plus-decoded-pre-answer-text-retokenized/v1",
            },
        }

    def scan_pair(self, pair: dict[str, object], plan: object) -> CotSwapScan:
        sample_id = str(pair["sample_id"])
        index = self.sample_ids.index(sample_id)
        if index == 0:
            values = {"A": "A", "B": "B", "C": "A", "D": "B"}
        elif index == 1:
            values = {"A": "A", "B": "B", "C": "B", "D": "A"}
        else:
            values = {cell: "B" for cell in CELL_ORDER}
        uses: dict[str, CotSwapInputUse] = {}
        generations: dict[str, CotSwapGeneration] = {}
        for cell in plan.cells:
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
                full_input_ids_sha256=hashlib.sha256(cell.full_input.encode("utf-8")).hexdigest(),
                prompt_prefix_token_stable=True,
            )
            value = values[cell.cell]
            generations[cell.cell] = CotSwapGeneration(
                token_ids=(ord(cell.cell),),
                text=f"The answer is ({value}).",
                value=value,
                is_extracted=True,
                is_correct=value == "A",
                method="primary:pattern_1",
                primary_method="pattern_1",
                stop_reason="eos_token",
                stop_token_id=ord(cell.cell),
            )
        return CotSwapScan(
            sample_id=sample_id,
            input_uses=uses,
            generations=generations,
        )


def test_documented_cpu_command_accepts_a_real_verified_partial_producer_grid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cohort = load_sample_id_cohort(COHORT_PATH)
    # The submitted 50-per-subject source intersects the five 100-ID blocks this way.
    sample_ids = tuple(
        sample_id
        for block_start in range(0, 500, 100)
        for sample_id in cohort.sample_ids[block_start : block_start + 50]
    )
    pairs_dir = tmp_path / "pairs" / "gemma-1b"
    pair_result = run_prepare_edited_pairs(
        PrepareEditedPairsConfig(
            model=EXPECTED_MODELS[0],
            benchmark="mmlu",
            targeting="attribution-4",
            num_edits=4,
            sample_ids=COHORT_PATH,
            output_dir=pairs_dir,
        ),
        runtime=_PairRuntime(sample_ids),
    )
    cot_dir = tmp_path / "cot" / "gemma-1b"
    run_cot_swap(
        CotSwapConfig(
            model=EXPECTED_MODELS[0],
            benchmark="mmlu",
            pairs=pair_result.pairs_path,
            targeting="attribution-4",
            output_dir=cot_dir,
        ),
        runtime=_CotRuntime(tuple(sorted(sample_ids))),
    )
    output = tmp_path / "table9"

    assert (
        main(
            [
                "model-scale-cot-swap",
                "--pairs-root",
                str(tmp_path / "pairs"),
                "--cot-swap-runs-root",
                str(tmp_path / "cot"),
                "--cohort",
                str(COHORT_PATH),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert "built Table 9 from 1 model setting(s)" in capsys.readouterr().out
    summary = json.loads((output / "model_scale_summary.json").read_text(encoding="utf-8"))
    row = summary["models"][EXPECTED_MODELS[0]]
    assert row["source_records"] == 250
    assert row["n_s"] == 2
    assert row["both"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert row["restoration"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert summary["coverage"]["complete_grid"] is False


def test_docs_define_separate_producers_gpu0_and_cpu_builder() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    doc = (PROJECT_ROOT / "docs" / "model-scale-cot-swap.md").read_text(encoding="utf-8")
    for text in (readme, doc):
        assert "model-scale-cot-swap" in text
        assert "first 500" in text
        assert "250" in text and "500" in text
        assert "n_s" in text and "n_B" in text
        assert "directional" in text
    assert 'MODEL_SCALE_GPU_IDS="${MODEL_SCALE_GPU_IDS:-0}"' in readme
    assert "typo-cot prepare-edited-pairs" in readme
    assert '--sample-ids "${MODEL_SCALE_COHORT}"' in readme
    assert "typo-cot cot-swap" in readme
    assert "--cot-swap-runs-root results/model-scale-cot-swap-runs" in readme
    assert "CPU-only" in doc
