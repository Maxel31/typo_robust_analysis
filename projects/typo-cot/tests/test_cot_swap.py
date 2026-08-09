"""Final-PDF contract tests for the complete pre-answer text CoT swap."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
import typo_cot.experiments.cot_swap.runner as cot_runner
import typo_cot.experiments.cot_swap.runtime as cot_runtime
from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.cot_swap import (
    CELL_ORDER,
    CELL_SIDES,
    CotSwapConfig,
    CotSwapGeneration,
    CotSwapInputUse,
    CotSwapResult,
    CotSwapRunError,
    CotSwapScan,
    build_cell_plan,
    locate_pre_answer,
    run_cot_swap,
)
from typo_cot.experiments.cot_swap.runtime import HuggingFaceCotSwapRuntime


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


def test_runner_and_runtime_share_one_protocol_constant_source() -> None:
    protocol = importlib.import_module("typo_cot.experiments.cot_swap.protocol")
    aliases = (
        ("BENCHMARK_DATASET_NAMES", "_BENCHMARK_NAMES", "_BENCHMARK_NAMES"),
        ("GENERATION", "_GENERATION", "_GENERATION"),
        ("TEXT_INTERVENTION", "_TEXT_INTERVENTION", "_TEXT_INTERVENTION"),
        ("ANSWER_EXTRACTION", "_ANSWER_EXTRACTION", "_ANSWER_EXTRACTION"),
        ("IMPLEMENTATION", "_IMPLEMENTATION", "_IMPLEMENTATION"),
        ("BATCHING", "_BATCHING", "_BATCHING"),
        ("ANSWER_SPAN_DECODING", "_ANSWER_SPAN_DECODING", "_ANSWER_SPAN_DECODING"),
    )

    for protocol_name, runner_name, runtime_name in aliases:
        shared = getattr(protocol, protocol_name)
        assert getattr(cot_runner, runner_name) is shared
        assert getattr(cot_runtime, runtime_name) is shared


def _answer_payload(value: str, *, correct: bool) -> dict[str, object]:
    return {
        "value": value,
        "is_extracted": bool(value),
        "is_correct": correct,
        "method": "primary:fixture" if value else "unextractable",
        "primary_method": "fixture" if value else "no_match",
        "confidence": 1.0 if value else 0.0,
    }


def _pair(
    sample_id: str,
    *,
    model: str = "test/model",
    benchmark: str = "gsm8k",
    targeting: str = "attribution-4",
    clean_continuation: str = (
        "We calculate the quantities carefully and obtain two.\nThe answer is 2."
    ),
    edited_continuation: str = ("The edited calculation instead produces three.\nThe answer is 3."),
    edited: bool = True,
    num_edits: int = 4,
) -> dict[str, object]:
    clean_prompt = f"few-shot context\nQuestion: clean {sample_id}\nAnswer:"
    edited_prompt = (
        f"few-shot context\nQuestion: edited {sample_id}\nAnswer:" if edited else clean_prompt
    )
    attempts = (
        [
            {
                "token_index": 3,
                "token_text": "clean",
                "operation": "substitution",
            }
        ]
        if edited
        else []
    )
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": model,
        "benchmark": benchmark,
        "targeting": targeting,
        "seed": 42,
        "num_edits_requested": num_edits,
        "num_candidates": 8,
        "num_target_attempts": len(attempts),
        "num_aligned_words": 0,
        "gold_answer": "2",
        "subset": None,
        "clean": {
            "prompt": clean_prompt,
            "prompt_token_count": 12,
            "continuation": clean_continuation,
            "continuation_token_count": 15,
            "answer": _answer_payload("2", correct=True),
        },
        "edited": {
            "prompt": edited_prompt,
            "prompt_token_count": 12,
            "continuation": edited_continuation if edited else clean_continuation,
            "continuation_token_count": 15,
            "answer": _answer_payload("3" if edited else "2", correct=not edited),
        },
        "answer_changed": edited,
        "target_attempts": attempts,
        "aligned_words": [],
    }


def _write_pair_source(
    root: Path,
    pairs: list[dict[str, object]],
    *,
    model: str = "test/model",
    benchmark: str = "gsm8k",
    targeting: str = "attribution-4",
    limit: int | None = None,
    num_edits: int = 4,
    sample_id_cohort: Mapping[str, object] | None = None,
) -> Path:
    root.mkdir(parents=True)
    ordered = sorted(pairs, key=lambda row: str(row["sample_id"]))
    pairs_path = root / "pairs.jsonl"
    _write_jsonl(pairs_path, ordered)
    _write_json(
        root / "run.json",
        {
            "schema_version": "prepare-edited-pairs-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "prepare-edited-pairs",
            "status": "completed",
            "arguments": {
                "model": model,
                "benchmark": benchmark,
                "targeting": targeting,
                "num_edits": num_edits,
                "seed": 42,
                "max_new_tokens": 512,
                "gpu_id": "0",
                "limit": limit,
                "output_dir": str(root.resolve()),
                **(
                    {"sample_ids": str((root / "cohort.json").resolve())}
                    if sample_id_cohort is not None
                    else {}
                ),
            },
            "counts": {
                "discovered": len(ordered),
                "written": len(ordered),
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
                "model_revision": "source-revision",
                "benchmark_dataset_loader": {
                    "mmlu-pro": "mmlu_pro",
                    "csqa": "commonsense_qa",
                }.get(benchmark, benchmark),
                "dataset_cohort_rule": (
                    "explicit-sample-id-cohort/v1"
                    if sample_id_cohort is not None
                    else "paper-model-benchmark-cohort/v1"
                ),
                "dataset_sample_count": len(ordered),
                "dataset_records_sha256": "d" * 64,
                "dataset_samples_per_subset": {
                    "mmlu": 50,
                    "mmlu-pro": 100,
                }.get(benchmark),
                "random_seed_algorithm": "sha256-first-64-bits/v1",
                "generation_protocol": "explicit-greedy-generation/v1",
                "target_position": "maximum-logit-after-first-cot-token",
                "alignment": "actual-edited-word-final-token",
                "historical_compatibility_notes": [],
                **(
                    {"sample_id_cohort": dict(sample_id_cohort)}
                    if sample_id_cohort is not None
                    else {}
                ),
            },
        },
    )
    return pairs_path


def _generation(value: str, *, cell: str, gold: str = "2") -> CotSwapGeneration:
    return CotSwapGeneration(
        token_ids=(ord(cell),),
        text=f"The answer is {value}." if value else "No extractable answer.",
        value=value,
        is_extracted=bool(value),
        is_correct=value == gold,
        method="primary:pattern_1" if value else "unextractable",
        primary_method="pattern_1" if value else "no_match",
        stop_reason="eos_token",
        stop_token_id=ord(cell),
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
        self.answers = {key: dict(value) for key, value in (answers or {}).items()}
        self.failure_for = failure_for
        self.provenance_changes = dict(provenance_changes or {})
        self.input_mismatch_for = input_mismatch_for
        self.calls: list[str] = []

    def provenance(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        payload.update(self.provenance_changes)
        return payload

    def scan_pair(self, pair: dict[str, object], plan: object) -> CotSwapScan:
        sample_id = str(pair["sample_id"])
        self.calls.append(sample_id)
        if sample_id == self.failure_for:
            raise RuntimeError("synthetic cot-swap GPU failure")
        plan_cells = tuple(plan.cells)  # type: ignore[attr-defined]
        uses: dict[str, CotSwapInputUse] = {}
        generations: dict[str, CotSwapGeneration] = {}
        values = self.answers.get(sample_id, {"A": "2", "B": "3", "C": "2", "D": "3"})
        for cell in plan_cells:
            full_hash = cell.full_input_sha256
            if self.input_mismatch_for == (sample_id, cell.cell):
                full_hash = "0" * 64
            uses[cell.cell] = CotSwapInputUse(
                cell=cell.cell,
                prompt_text_sha256=cell.prompt_sha256,
                pre_answer_text_sha256=cell.pre_answer_sha256,
                full_input_text_sha256=full_hash,
                prompt_char_count=len(cell.prompt),
                pre_answer_char_count=len(cell.pre_answer_text),
                full_input_char_count=len(cell.full_input),
                prompt_token_count=cell.prompt_token_count,
                full_input_token_count=cell.prompt_token_count + 4,
                full_input_ids_sha256=hashlib.sha256(cell.full_input.encode("utf-8")).hexdigest(),
                prompt_prefix_token_stable=True,
            )
            generations[cell.cell] = _generation(values[cell.cell], cell=cell.cell)
        return CotSwapScan(
            sample_id=sample_id,
            input_uses=uses,
            generations=generations,
        )


def _config(pairs: Path, output: Path, **changes: object) -> CotSwapConfig:
    config = CotSwapConfig(
        model="test/model",
        benchmark="gsm8k",
        pairs=pairs,
        targeting="attribution-4",
        output_dir=output,
    )
    return replace(config, **changes)


def test_catalog_and_cli_expose_the_completed_cot_swap_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = get_experiment("cot-swap")
    assert spec.status == "implemented"
    assert spec.paper_question == "RQ2"
    assert spec.paper_sections == ("§3.4", "§4.2", "Table 1", "Appendix C")
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--pairs",
        "--targeting",
        "--output-dir",
    )
    assert spec.outputs == (
        "cot_swap_records.jsonl",
        "pair_status_records.jsonl",
        "cot_swap_summary.json",
        "run.json",
    )

    captured: list[CotSwapConfig] = []

    def fake_run(config: CotSwapConfig) -> CotSwapResult:
        captured.append(config)
        return CotSwapResult(
            records_path=config.output_dir / "cot_swap_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "cot_swap_summary.json",
            run_path=config.output_dir / "run.json",
            executed_pairs=1,
            records=1,
        )

    monkeypatch.setattr(cli_module, "run_cot_swap", fake_run)
    assert (
        main(
            [
                "cot-swap",
                "--model",
                "test/model",
                "--benchmark",
                "gsm8k",
                "--pairs",
                "results/prepared/pairs.jsonl",
                "--targeting",
                "attribution-4",
                "--source-num-edits",
                "2",
                "--gpu-id",
                "0",
                "--limit",
                "1",
                "--output-dir",
                "results/cot-swap",
                "--resume",
            ]
        )
        == 0
    )
    assert captured == [
        CotSwapConfig(
            model="test/model",
            benchmark="gsm8k",
            pairs=Path("results/prepared/pairs.jsonl"),
            targeting="attribution-4",
            output_dir=Path("results/cot-swap"),
            source_num_edits=2,
            gpu_id="0",
            limit=1,
            resume=True,
        )
    ]
    assert "wrote 1 CoT-swap record(s) for 1 pair(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"model": ""}, "model must not be empty"),
        ({"benchmark": "math-500"}, "unsupported benchmark"),
        ({"targeting": "top-4"}, "unsupported targeting"),
        ({"gpu_id": "0,0"}, "unique"),
        ({"gpu_id": "00"}, "comma-separated"),
        ({"limit": 0}, "positive integer"),
        ({"source_num_edits": 3}, "one of 1, 2, or 4"),
    ),
)
def test_config_rejects_non_paper_or_ambiguous_arguments(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "pairs.jsonl", tmp_path / "out", **changes)


def test_config_accepts_ordered_unique_model_parallel_gpu_ids(tmp_path: Path) -> None:
    config = _config(
        tmp_path / "pairs.jsonl",
        tmp_path / "out",
        gpu_id="0,2,3",
    )

    assert config.gpu_id == "0,2,3"
    assert config.public_arguments()["gpu_id"] == "0,2,3"


def test_explicit_sample_id_cohort_is_preserved_in_cot_swap_sources(
    tmp_path: Path,
) -> None:
    cohort = {
        "schema_version": "sample-id-cohort/v1",
        "cohort_id": "fixture-mmlu-first500",
        "benchmark": "gsm8k",
        "selection": "fixture-order/v1",
        "provenance": "test-fixture",
        "sample_count": 1,
        "sample_ids_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "model_samples_per_subset": None,
        "selected_sample_count": 1,
    }
    source = _write_pair_source(
        tmp_path / "prepared",
        [_pair("sample", model="test/model", benchmark="gsm8k")],
        model="test/model",
        benchmark="gsm8k",
        sample_id_cohort=cohort,
    )
    runtime = _Runtime(
        provenance_changes={
            "model": "test/model",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
        }
    )

    result = run_cot_swap(
        CotSwapConfig(
            model="test/model",
            benchmark="gsm8k",
            pairs=source,
            targeting="attribution-4",
            output_dir=tmp_path / "cot",
        ),
        runtime=runtime,
    )

    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert manifest["source"]["sample_id_cohort"] == cohort
    assert summary["source"]["sample_id_cohort"] == cohort


def test_runner_labels_a_two_edit_cot_swap_as_table8_sensitivity(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(
        tmp_path / "prepared",
        [_pair("sample", num_edits=2)],
        num_edits=2,
    )

    result = run_cot_swap(
        _config(source, tmp_path / "output", source_num_edits=2),
        runtime=_Runtime(),
    )

    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert manifest["arguments"]["source_num_edits"] == 2
    assert manifest["protocol"]["source_generation"]["num_edits_requested"] == 2
    assert manifest["comparability"]["requirements"]["requested_edit_count"] == 2
    assert manifest["comparability"]["status"] == "fresh-edit-count-sensitivity-setting"
    assert summary["source_num_edits"] == 2
    assert summary["comparability"] == manifest["comparability"]


@pytest.mark.parametrize(
    ("continuation", "prefix", "found", "count", "start", "early", "residual"),
    (
        (
            "Long reasoning over several intermediate steps.\nThe answer is 2.",
            "Long reasoning over several intermediate steps.\n",
            True,
            1,
            48,
            False,
            False,
        ),
        (
            "Long reasoning over several intermediate steps. the answer is 2.",
            "Long reasoning over several intermediate steps. ",
            True,
            1,
            48,
            False,
            False,
        ),
        ("The answer is 2.", "", True, 1, 0, True, False),
        (
            "Reasoning. The answer is discussed. The answer is 2.",
            "Reasoning. ",
            True,
            2,
            11,
            True,
            False,
        ),
        ("No canonical trigger.", "No canonical trigger.", False, 0, None, False, False),
        (
            "Answer: perhaps two after a long derivation. The answer is 2.",
            "Answer: perhaps two after a long derivation. ",
            True,
            1,
            45,
            False,
            True,
        ),
    ),
)
def test_pre_answer_locator_freezes_the_submitted_template_rule(
    continuation: str,
    prefix: str,
    found: bool,
    count: int,
    start: int | None,
    early: bool,
    residual: bool,
) -> None:
    boundary = locate_pre_answer(continuation)
    assert boundary.text == prefix
    assert boundary.trigger_found is found
    assert boundary.trigger_count == count
    assert boundary.trigger_char_start == start
    assert boundary.early_trigger is early
    assert boundary.residual_fragment is residual
    assert boundary.method == "submitted-first-[Tt]he-answer-is-filter/v1"


def test_cell_plan_crosses_exact_recorded_prompts_and_pre_answer_texts() -> None:
    pair = _pair("sample")
    plan = build_cell_plan(pair)
    assert plan.sample_id == "sample"
    assert plan.exclusion_reasons == ()
    assert [cell.cell for cell in plan.cells] == list(CELL_ORDER)
    assert [(cell.question_side, cell.cot_side) for cell in plan.cells] == [
        CELL_SIDES[cell] for cell in CELL_ORDER
    ]
    clean_prompt = pair["clean"]["prompt"]  # type: ignore[index]
    edited_prompt = pair["edited"]["prompt"]  # type: ignore[index]
    clean_prefix = locate_pre_answer(pair["clean"]["continuation"]).text  # type: ignore[index]
    edited_prefix = locate_pre_answer(pair["edited"]["continuation"]).text  # type: ignore[index]
    assert [cell.full_input for cell in plan.cells] == [
        clean_prompt + clean_prefix,
        edited_prompt + edited_prefix,
        edited_prompt + clean_prefix,
        clean_prompt + edited_prefix,
    ]
    assert all(len(cell.full_input_sha256) == 64 for cell in plan.cells)


@pytest.mark.parametrize("invalid_attempts", (None, "0", False, -1, 5))
def test_cell_plan_rejects_invalid_target_attempt_counts(invalid_attempts: object) -> None:
    pair = _pair("sample")
    pair["num_target_attempts"] = invalid_attempts

    with pytest.raises(ValueError, match="num_target_attempts"):
        build_cell_plan(pair)


@pytest.mark.parametrize(
    ("pair", "reason"),
    (
        (_pair("zero", edited=False), "no-applied-edit"),
        (
            _pair("clean-missing", clean_continuation="No template after long reasoning."),
            "no-trigger-clean",
        ),
        (
            _pair(
                "edited-multi",
                edited_continuation=(
                    "Long reasoning. The answer is discussed. Later The answer is 3."
                ),
            ),
            "multiple-trigger-edited",
        ),
        (_pair("clean-early", clean_continuation="The answer is 2."), "early-trigger-clean"),
        (
            _pair(
                "edited-residual",
                edited_continuation=(
                    "Answer: maybe three after a long derivation. The answer is 3."
                ),
            ),
            "residual-fragment-edited",
        ),
    ),
)
def test_cell_plan_records_every_template_or_edit_exclusion(
    pair: dict[str, object],
    reason: str,
) -> None:
    plan = build_cell_plan(pair)
    assert reason in plan.exclusion_reasons
    assert plan.eligible is False


def test_runner_emits_four_cells_and_uses_the_paper_denominators(tmp_path: Path) -> None:
    pairs = [_pair(sample_id) for sample_id in ("a", "b", "c", "d")]
    pairs.append(_pair("excluded", clean_continuation="No answer template here."))
    source = _write_pair_source(tmp_path / "prepared", pairs)
    runtime = _Runtime(
        answers={
            "a": {"A": "2", "B": "3", "C": "2", "D": "3"},
            "b": {"A": "2", "B": "2", "C": "3", "D": "2"},
            "c": {"A": "3", "B": "3", "C": "2", "D": "3"},
            "d": {"A": "2", "B": "", "C": "", "D": ""},
        }
    )
    result = run_cot_swap(_config(source, tmp_path / "output"), runtime=runtime)

    assert result.executed_pairs == 4
    assert result.records == 4
    assert runtime.calls == ["a", "b", "c", "d"]
    rows = _read_jsonl(result.records_path)
    assert [row["sample_id"] for row in rows] == ["a", "b", "c", "d"]
    assert all(row["schema_version"] == "cot-swap-record/v1" for row in rows)
    assert all(list(row["cells"]) == list(CELL_ORDER) for row in rows)
    row_a = rows[0]
    assert row_a["events"] == {
        "clean_correct": True,
        "both_changed": True,
        "question_only_changed": False,
        "cot_only_changed": True,
        "restoration_denominator": True,
        "b_to_c_restored": True,
    }

    statuses = _read_jsonl(result.pair_status_records_path)
    assert len(statuses) == 5
    excluded = next(row for row in statuses if row["sample_id"] == "excluded")
    assert excluded["selected_for_execution"] is False
    assert excluded["exclusion_reasons"] == ["no-trigger-clean"]

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"] == {
        "source_records": 5,
        "no_applied_edit_records": 0,
        "template_excluded_records": 1,
        "eligible_pairs": 4,
        "selected_pairs": 4,
        "executed_pairs": 4,
        "failed_pairs": 0,
        "a_correct_pairs": 3,
    }
    assert summary["metrics"]["both_changed"] == {
        "numerator": 2,
        "denominator": 3,
        "rate": 2 / 3,
    }
    assert summary["metrics"]["question_only_changed"] == {
        "numerator": 2,
        "denominator": 3,
        "rate": 2 / 3,
    }
    assert summary["metrics"]["cot_only_changed"] == {
        "numerator": 2,
        "denominator": 3,
        "rate": 2 / 3,
    }
    assert summary["extraction_method_by_cell_and_stop_reason"]["B"] == {
        "eos_token": {"primary:pattern_1": 3, "unextractable": 1},
        "max_new_tokens": {},
    }
    assert summary["metrics"]["restoration"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert summary["answer_extraction"]["symmetric_cells"] == list(CELL_ORDER)
    assert summary["answer_extraction"]["implementation_detail"] == {
        "primary_precedence": "task-extractor-preserve-nonempty/v1",
        "fallback_invocation": "empty-primary-only-symmetric-a-b-c-d/v1",
        "max_token_cap_gate": "disable-positional-numeric-n4-n5-only/v1",
        "regex_and_cap_gate_source": ("legacy-backed-detail-not-specified-by-final-pdf"),
    }
    assert summary["stop_reason_by_cell"] == {
        cell: {"eos_token": 4, "max_new_tokens": 0} for cell in CELL_ORDER
    }
    assert summary["published_reference"]["scope"] == "task-pooled-attribution-4"
    assert summary["comparability"]["historical_cohort_identity"] is False
    serialized = json.dumps(summary, sort_keys=True).lower()
    for forbidden in ("te_rate", "de_rate", "ie_rate", "mediation", "direct_effect"):
        assert forbidden not in serialized

    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["paper_sha256"] == PAPER_SHA256
    assert run["status"] == "completed"
    assert (
        run["protocol"]["answer_extraction_detail"]
        == summary["answer_extraction"]["implementation_detail"]
    )
    assert (
        "fallback-regex-and-cap-gate-are-legacy-backed-and-not-specified-by-final-pdf"
        in run["comparability"]["limitations"]
    )
    assert not (result.run_path.parent / ".cot-swap-work").exists()


def test_unextractable_comparison_answers_are_failures_not_exclusions(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    runtime = _Runtime(answers={"a": {"A": "2", "B": "", "C": "", "D": ""}})
    result = run_cot_swap(_config(source, tmp_path / "output"), runtime=runtime)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["a_correct_pairs"] == 1
    assert summary["metrics"]["both_changed"]["numerator"] == 1
    assert summary["metrics"]["question_only_changed"]["numerator"] == 1
    assert summary["metrics"]["cot_only_changed"]["numerator"] == 1
    assert summary["metrics"]["restoration"] == {
        "numerator": 0,
        "denominator": 1,
        "rate": 0.0,
    }
    status = _read_jsonl(result.pair_status_records_path)[0]
    assert status["included_in_change_denominator"] is True


def test_limit_is_applied_after_full_planning_and_statuses_retain_the_population(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(
        tmp_path / "prepared",
        [_pair("a"), _pair("b"), _pair("excluded", edited=False)],
    )
    runtime = _Runtime()
    result = run_cot_swap(
        _config(source, tmp_path / "output", limit=1),
        runtime=runtime,
    )
    assert result.executed_pairs == 1
    assert runtime.calls == ["a"]
    statuses = _read_jsonl(result.pair_status_records_path)
    assert [row["selected_for_execution"] for row in statuses] == [True, False, False]
    no_edit = next(row for row in statuses if row["sample_id"] == "excluded")
    assert no_edit["edit_valid"] is False
    assert no_edit["template_eligible"] is True
    assert no_edit["execution_status"] == "no-applied-edit"
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["counts"]["source_records"] == 3
    assert run["counts"]["no_applied_edit_records"] == 1
    assert run["counts"]["template_excluded_records"] == 0
    assert run["counts"]["eligible_pairs"] == 2
    assert run["counts"]["selected_pairs"] == 1
    assert run["protocol"]["edit_validity"] == {
        "policy": "stored-prompts-differ-and-positive-target-attempts/v1",
        "requires_prompt_difference": True,
        "requires_positive_target_attempts": True,
        "zero_edit_restoration": "undefined-excluded-before-template-filter",
    }
    assert run["protocol"]["change_denominator"] == (
        "edit-valid-template-eligible-successfully-executed-regenerated-a-correct"
    )
    assert (
        "fresh-edit-validity-gate-can-differ-from-legacy-historical-cohort"
        in run["comparability"]["limitations"]
    )
    assert run["comparability"]["status"] == "partial-smoke-run"
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["edit_validity"] == {
        "policy": "stored-prompts-differ-and-positive-target-attempts/v1",
        "excluded_records": 1,
        "exclusion_reason_counts": {"no-applied-edit": 1},
    }


def test_limited_or_incomplete_preparation_source_is_rejected_before_runtime(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(
        tmp_path / "prepared",
        [_pair("a")],
        limit=1,
    )
    runtime = _Runtime()
    with pytest.raises(ValueError, match="source pair preparation must not be limited"):
        run_cot_swap(_config(source, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda run, pair: run.update(paper_sha256="0" * 64), "paper SHA-256"),
        (lambda run, pair: run["arguments"].update(seed=7), "seed 42"),
        (lambda run, pair: run["arguments"].update(num_edits=3), "four edits"),
        (lambda run, pair: run["decoding"].update(max_new_tokens=511), "512"),
        (
            lambda run, pair: run["provenance"].update(dataset_cohort_rule="unreviewed-cohort"),
            "dataset cohort rule",
        ),
        (
            lambda run, pair: run["provenance"].update(
                random_seed_algorithm="process-random-python-hash"
            ),
            "seed algorithm",
        ),
        (
            lambda run, pair: run["provenance"].update(
                target_position="maximum-logit-before-first-cot-token"
            ),
            "target position",
        ),
        (
            lambda run, pair: run["provenance"].update(alignment="token-substring-coordinate"),
            "alignment protocol",
        ),
        (
            lambda run, pair: pair.update(
                num_target_attempts=5,
                target_attempts=[{} for _ in range(5)],
            ),
            "at most four target attempts",
        ),
        (
            lambda run, pair: pair["clean"].update(continuation_token_count=513),
            "at most 512 continuation tokens",
        ),
        (
            lambda run, pair: pair.update(
                num_aligned_words=2,
                aligned_words=[{}, {}],
            ),
            "aligned words cannot exceed target attempts",
        ),
        (lambda run, pair: pair.update(model="other/model"), "record model"),
    ),
)
def test_source_contract_mismatches_fail_before_runtime(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    run_path = source.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    pair = _read_jsonl(source)[0]
    mutate(run, pair)  # type: ignore[operator]
    _write_json(run_path, run)
    _write_jsonl(source, [pair])
    runtime = _Runtime()
    with pytest.raises(ValueError, match=message):
        run_cot_swap(_config(source, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_source_contract_rejects_wrong_model_benchmark_subset_cap(tmp_path: Path) -> None:
    source = _write_pair_source(
        tmp_path / "prepared",
        [_pair("a", benchmark="mmlu")],
        benchmark="mmlu",
    )
    run_path = source.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["provenance"]["dataset_samples_per_subset"] = 100
    _write_json(run_path, run)
    runtime = _Runtime()
    config = replace(_config(source, tmp_path / "output"), benchmark="mmlu")

    with pytest.raises(ValueError, match="samples-per-subset cap"):
        run_cot_swap(config, runtime=runtime)
    assert runtime.calls == []


def test_duplicate_json_key_or_unsorted_sample_ids_are_rejected(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a"), _pair("b")])
    first, second = source.read_text(encoding="utf-8").splitlines()
    source.write_text(second + "\n" + first + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly sorted"):
        run_cot_swap(_config(source, tmp_path / "unsorted"), runtime=_Runtime())

    source = _write_pair_source(tmp_path / "prepared-duplicate", [_pair("a")])
    line = source.read_text(encoding="utf-8").strip()
    source.write_text(line[:-1] + ',"sample_id":"duplicate"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        run_cot_swap(_config(source, tmp_path / "duplicate"), runtime=_Runtime())


def test_runtime_provenance_and_input_use_must_match_before_publication(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    bad_provenance = _Runtime(
        provenance_changes={
            "answer_extraction": "primary-only-asymmetric",
        }
    )
    with pytest.raises(ValueError, match="public protocol"):
        run_cot_swap(_config(source, tmp_path / "bad-provenance"), runtime=bad_provenance)
    assert bad_provenance.calls == []

    bad_input = _Runtime(input_mismatch_for=("a", "C"))
    with pytest.raises(CotSwapRunError, match="runtime fixed input does not match"):
        run_cot_swap(_config(source, tmp_path / "bad-input"), runtime=bad_input)
    assert bad_input.calls == ["a"]
    assert not (tmp_path / "bad-input" / "cot_swap_records.jsonl").exists()


def test_runtime_provenance_rejects_a_different_batch_shape_before_gpu_work(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    runtime = _Runtime(
        provenance_changes={
            "batching": {
                "policy": "two-pairs-eight-cells/v1",
                "batch_size": 8,
                "cell_order": list(CELL_ORDER) * 2,
            }
        }
    )

    with pytest.raises(ValueError, match="batching"):
        run_cot_swap(_config(source, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_runtime_provenance_rejects_a_different_answer_span_decode_policy(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    runtime = _Runtime(
        provenance_changes={
            "answer_span_decoding": {
                "source": "full-output-character-slice/v1",
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
        }
    )

    with pytest.raises(ValueError, match="answer-span decoding"):
        run_cot_swap(_config(source, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_runtime_cannot_return_more_than_the_paper_answer_span_cap(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    runtime = _Runtime()
    original_scan = runtime.scan_pair

    def overlong_scan(pair: dict[str, object], plan: object) -> CotSwapScan:
        scan = original_scan(pair, plan)
        generations = dict(scan.generations)
        generations["D"] = replace(generations["D"], token_ids=tuple(range(17)))
        return replace(scan, generations=generations)

    runtime.scan_pair = overlong_scan  # type: ignore[method-assign]
    with pytest.raises(CotSwapRunError, match="at most 16"):
        run_cot_swap(_config(source, tmp_path / "output"), runtime=runtime)


def test_failed_run_checkpoints_successes_and_resume_runs_only_failures(tmp_path: Path) -> None:
    source = _write_pair_source(
        tmp_path / "prepared",
        [_pair("a"), _pair("b"), _pair("c")],
    )
    output = tmp_path / "output"
    config = _config(source, output)
    first = _Runtime(failure_for="b")
    with pytest.raises(CotSwapRunError, match=r"1 pair.*failed"):
        run_cot_swap(config, runtime=first)
    assert first.calls == ["a", "b", "c"]
    assert not (output / "cot_swap_records.jsonl").exists()
    checkpoints = list((output / ".cot-swap-work" / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 2
    failed_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert failed_run["status"] == "failed"
    assert failed_run["counts"]["failed_pairs"] == 1

    resumed = _Runtime()
    result = run_cot_swap(replace(config, resume=True), runtime=resumed)
    assert result.executed_pairs == 3
    assert resumed.calls == ["b"]
    assert not (output / ".cot-swap-work").exists()


def test_registered_checkpoint_tampering_is_rejected_on_resume(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a"), _pair("b")])
    output = tmp_path / "output"
    config = _config(source, output)
    with pytest.raises(CotSwapRunError):
        run_cot_swap(config, runtime=_Runtime(failure_for="b"))
    checkpoint = next((output / ".cot-swap-work" / "checkpoints").glob("*.json"))
    checkpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises((ValueError, CotSwapRunError), match=r"checkpoint.*hash"):
        run_cot_swap(replace(config, resume=True), runtime=_Runtime())


def test_checkpoint_registry_identity_is_bound_to_targeting_and_sample_id(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a"), _pair("b")])
    output = tmp_path / "output"
    config = _config(source, output)
    with pytest.raises(CotSwapRunError):
        run_cot_swap(config, runtime=_Runtime(failure_for="b"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    ((identity, metadata),) = run["checkpoints"].items()
    assert identity != "tampered-registry-key"
    run["checkpoints"] = {"tampered-registry-key": metadata}
    _write_json(output / "run.json", run)

    with pytest.raises(ValueError, match="checkpoint registry identity"):
        run_cot_swap(replace(config, resume=True), runtime=_Runtime())


def test_resume_adopts_a_valid_checkpoint_left_before_manifest_update(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a"), _pair("b")])
    output = tmp_path / "output"
    config = _config(source, output)
    with pytest.raises(CotSwapRunError):
        run_cot_swap(config, runtime=_Runtime(failure_for="b"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert len(run["checkpoints"]) == 1
    run["checkpoints"] = {}
    _write_json(output / "run.json", run)

    resumed = _Runtime()
    result = run_cot_swap(replace(config, resume=True), runtime=resumed)
    assert result.executed_pairs == 2
    assert resumed.calls == ["b"]


def test_registered_checkpoint_is_read_once_for_hash_and_json_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a"), _pair("b")])
    output = tmp_path / "output"
    config = _config(source, output)
    with pytest.raises(CotSwapRunError):
        run_cot_swap(config, runtime=_Runtime(failure_for="b"))
    checkpoint = next((output / ".cot-swap-work" / "checkpoints").glob("*.json"))
    original_open = Path.open
    checkpoint_reads = 0

    def counting_open(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal checkpoint_reads
        if path == checkpoint:
            checkpoint_reads += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    result = run_cot_swap(replace(config, resume=True), runtime=_Runtime())

    assert result.executed_pairs == 2
    assert checkpoint_reads == 1


@pytest.mark.parametrize(
    "failure_stage",
    (
        "pair_status_records.jsonl",
        "cot_swap_summary.json",
        "completed-run-manifest",
    ),
)
def test_publication_failure_keeps_only_checkpoints_and_resumes_without_gpu_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    original_json = cot_runner._write_json_atomic
    original_jsonl = cot_runner._write_jsonl_atomic
    failed_once = False

    def fail_one_json_write(path: Path, payload: Mapping[str, object]) -> None:
        nonlocal failed_once
        completed_manifest = (
            failure_stage == "completed-run-manifest"
            and path.name == "run.json"
            and payload.get("status") == "completed"
        )
        summary = failure_stage == path.name == "cot_swap_summary.json"
        if not failed_once and (completed_manifest or summary):
            failed_once = True
            raise OSError(f"synthetic {failure_stage} publication failure")
        original_json(path, payload)

    def fail_one_jsonl_write(path: Path, rows: object) -> None:
        nonlocal failed_once
        if not failed_once and failure_stage == path.name:
            failed_once = True
            raise OSError(f"synthetic {failure_stage} publication failure")
        original_jsonl(path, rows)  # type: ignore[arg-type]

    monkeypatch.setattr(cot_runner, "_write_json_atomic", fail_one_json_write)
    monkeypatch.setattr(cot_runner, "_write_jsonl_atomic", fail_one_jsonl_write)
    with pytest.raises(CotSwapRunError, match="publication failed"):
        run_cot_swap(config, runtime=_Runtime())

    assert failed_once is True
    for name in (
        "cot_swap_records.jsonl",
        "pair_status_records.jsonl",
        "cot_swap_summary.json",
    ):
        assert not (output / name).exists()
    failed_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert failed_run["status"] == "failed"
    assert failed_run["counts"]["failed_pairs"] == 0
    assert failed_run["counts"]["publication_failures"] == 1
    assert failed_run["failures"][0]["stage"] == "publication"
    assert len(list((output / ".cot-swap-work" / "checkpoints").glob("*.json"))) == 1

    monkeypatch.setattr(cot_runner, "_write_json_atomic", original_json)
    monkeypatch.setattr(cot_runner, "_write_jsonl_atomic", original_jsonl)

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("all-checkpoint resume must not load model weights")

    monkeypatch.setattr(cot_runner, "HuggingFaceCotSwapRuntime", forbidden_factory)
    result = run_cot_swap(replace(config, resume=True))
    assert result.executed_pairs == 1


def test_resume_removes_crash_left_public_outputs_before_pending_gpu_work(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a"), _pair("b")])
    output = tmp_path / "output"
    config = _config(source, output)
    with pytest.raises(CotSwapRunError):
        run_cot_swap(config, runtime=_Runtime(failure_for="b"))
    stale_records = output / "cot_swap_records.jsonl"
    stale_records.write_text('{"partial":true}\n', encoding="utf-8")

    class GuardedRuntime(_Runtime):
        def scan_pair(self, pair: dict[str, object], plan: object) -> CotSwapScan:
            assert not stale_records.exists(), "partial public output survived until GPU work"
            return super().scan_pair(pair, plan)

    resumed = GuardedRuntime()
    result = run_cot_swap(replace(config, resume=True), runtime=resumed)
    assert result.executed_pairs == 2
    assert resumed.calls == ["b"]


def test_keyboard_interrupt_during_publication_cleans_outputs_and_is_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    original_jsonl = cot_runner._write_jsonl_atomic

    def interrupt_status_write(path: Path, rows: object) -> None:
        if path.name == "pair_status_records.jsonl":
            raise KeyboardInterrupt
        original_jsonl(path, rows)  # type: ignore[arg-type]

    monkeypatch.setattr(cot_runner, "_write_jsonl_atomic", interrupt_status_write)
    with pytest.raises(KeyboardInterrupt):
        run_cot_swap(_config(source, output), runtime=_Runtime())

    for name in (
        "cot_swap_records.jsonl",
        "pair_status_records.jsonl",
        "cot_swap_summary.json",
    ):
        assert not (output / name).exists()
    failed_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert failed_run["status"] == "failed"
    assert failed_run["failures"][0]["error_type"] == "KeyboardInterrupt"


def test_progress_manifest_flushes_at_power_of_two_checkpoint_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pair_source(
        tmp_path / "prepared",
        [_pair(f"sample-{index:02d}") for index in range(17)],
    )
    output = tmp_path / "output"
    original_write = cot_runner._write_json_atomic
    run_status_writes: list[str] = []

    def record_run_writes(path: Path, payload: object) -> None:
        if path.name == "run.json":
            assert isinstance(payload, Mapping)
            run_status_writes.append(str(payload["status"]))
        original_write(path, payload)

    monkeypatch.setattr(cot_runner, "_write_json_atomic", record_run_writes)
    result = run_cot_swap(_config(source, output), runtime=_Runtime())

    assert result.executed_pairs == 17
    assert run_status_writes == ["running"] * 6 + ["completed"]


def test_completed_resume_validates_sources_and_outputs_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    expected = run_cot_swap(config, runtime=_Runtime())

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed resume must not load model weights")

    monkeypatch.setattr(cot_runner, "HuggingFaceCotSwapRuntime", forbidden_factory)
    assert run_cot_swap(replace(config, resume=True)) == expected

    expected.records_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CotSwapRunError, match=r"output.*hash"):
        run_cot_swap(replace(config, resume=True))


def test_completed_output_registry_error_is_a_typed_integrity_failure(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    result = run_cot_swap(config, runtime=_Runtime())
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    del run["outputs"][result.records_path.name]
    _write_json(result.run_path, run)

    with pytest.raises(
        cot_runner._CompletedOutputIntegrityError,
        match="^completed output registry is incomplete$",
    ):
        run_cot_swap(replace(config, resume=True))


def test_completed_validation_does_not_dispatch_on_output_hash_message_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    run_cot_swap(config, runtime=_Runtime())

    def raise_non_integrity_error(*_args: object, **_kwargs: object) -> None:
        raise CotSwapRunError("synthetic output hash wording")

    monkeypatch.setattr(cot_runner, "_validate_output_hashes", raise_non_integrity_error)
    with pytest.raises(
        CotSwapRunError,
        match="^completed run validation failed: synthetic output hash wording$",
    ):
        run_cot_swap(replace(config, resume=True))


def test_completed_resume_rejects_semantic_tampering_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    result = run_cot_swap(config, runtime=_Runtime())
    rows = _read_jsonl(result.records_path)
    rows[0]["events"]["both_changed"] = False  # type: ignore[index]
    _write_jsonl(result.records_path, rows)
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    run["outputs"][result.records_path.name]["sha256"] = hashlib.sha256(
        result.records_path.read_bytes()
    ).hexdigest()
    _write_json(result.run_path, run)

    with pytest.raises(CotSwapRunError, match=r"completed.*validation"):
        run_cot_swap(replace(config, resume=True))


def test_completed_resume_reconstructs_and_validates_checkpoint_hashes(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    result = run_cot_swap(config, runtime=_Runtime())
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    ((_, checkpoint),) = run["checkpoints"].items()
    checkpoint["sha256"] = "0" * 64
    _write_json(result.run_path, run)

    with pytest.raises(CotSwapRunError, match=r"checkpoint.*hash"):
        run_cot_swap(replace(config, resume=True))


def test_completed_resume_reextracts_answers_from_generated_text(tmp_path: Path) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    result = run_cot_swap(config, runtime=_Runtime())
    rows = _read_jsonl(result.records_path)
    answer_b = rows[0]["cells"]["B"]["answer"]
    assert answer_b["text"] == "The answer is 3."
    answer_b["value"] = "4"
    _write_jsonl(result.records_path, rows)
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    run["outputs"][result.records_path.name]["sha256"] = hashlib.sha256(
        result.records_path.read_bytes()
    ).hexdigest()
    _write_json(result.run_path, run)

    with pytest.raises(CotSwapRunError, match="answer extraction does not match"):
        run_cot_swap(replace(config, resume=True))


@pytest.mark.parametrize(
    ("filename", "mutate"),
    (
        (
            "pair_status_records.jsonl",
            lambda payload: payload[0].update(execution_status="not-selected"),
        ),
        (
            "cot_swap_summary.json",
            lambda payload: payload.update(analysis_status="causal-effect-estimate"),
        ),
    ),
)
def test_completed_resume_recomputes_every_public_output_semantically(
    tmp_path: Path,
    filename: str,
    mutate: object,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"
    config = _config(source, output)
    run_cot_swap(config, runtime=_Runtime())
    path = output / filename
    payload = _read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
    mutate(payload)  # type: ignore[operator]
    if path.suffix == ".jsonl":
        _write_jsonl(path, payload)
    else:
        _write_json(path, payload)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    run["outputs"][filename]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(output / "run.json", run)

    with pytest.raises(CotSwapRunError, match=r"completed.*validation"):
        run_cot_swap(replace(config, resume=True))


def test_checkpoint_cleanup_failure_keeps_completed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pair_source(tmp_path / "prepared", [_pair("a")])
    output = tmp_path / "output"

    def fail_cleanup(_path: Path) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(cot_runner.shutil, "rmtree", fail_cleanup)
    result = run_cot_swap(_config(source, output), runtime=_Runtime())
    assert result.records_path.is_file()
    assert json.loads(result.run_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert (output / ".cot-swap-work").is_dir()


def test_runtime_answer_helper_uses_the_same_fallback_chain_for_every_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(HuggingFaceCotSwapRuntime)
    runtime.config = SimpleNamespace(benchmark="gsm8k")
    calls: list[str] = []

    def fake_extract(
        text: str,
        *,
        benchmark: str,
        correct_answer: str,
        allow_positional: bool,
    ) -> object:
        calls.append(text)
        assert allow_positional is True
        return SimpleNamespace(
            value="2",
            is_extracted=True,
            is_correct=True,
            method="fallback:fixture",
            primary_method="no_match",
        )

    monkeypatch.setattr(cot_runtime, "extract_with_fallback", fake_extract)
    for cell in CELL_ORDER:
        answer = runtime._answer_generation(
            (ord(cell),),
            f"span-{cell}",
            "2",
            stop_reason="eos_token",
            stop_token_id=ord(cell),
        )
        assert answer.method == "fallback:fixture"
    assert calls == [f"span-{cell}" for cell in CELL_ORDER]


def test_capped_answer_span_disables_positional_numeric_fallback_for_every_cell() -> None:
    terminal = cot_runtime.extract_with_fallback(
        "Total time = 1 + 1 + 1.5 + ",
        benchmark="gsm8k",
        correct_answer="1",
    )
    assert terminal.method == "fallback:N4_equals_tail"

    capped = cot_runtime.extract_with_fallback(
        "Total time = 1 + 1 + 1.5 + ",
        benchmark="gsm8k",
        correct_answer="1",
        allow_positional=False,
    )
    assert capped.value == ""
    assert capped.method == "unextractable"

    runtime = object.__new__(HuggingFaceCotSwapRuntime)
    runtime.config = SimpleNamespace(benchmark="gsm8k")
    for _cell in CELL_ORDER:
        answer = runtime._answer_generation(
            tuple(range(16)),
            "Total time = 1 + 1 + 1.5 + ",
            "1",
            stop_reason="max_new_tokens",
            stop_token_id=None,
        )
        assert answer.value == ""
        assert answer.method == "unextractable"
        assert answer.stop_reason == "max_new_tokens"
        assert answer.stop_token_id is None


def test_runtime_scan_builds_and_extracts_all_cells_in_paper_order() -> None:
    runtime = object.__new__(HuggingFaceCotSwapRuntime)
    runtime.config = SimpleNamespace(benchmark="gsm8k")
    pair = _pair("a")
    plan = build_cell_plan(pair)
    generated_order: list[str] = []

    def fake_generate(
        cell_plans: tuple[object, ...],
    ) -> dict[str, tuple[object, tuple[int, ...], str, str, int | None]]:
        payload: dict[str, tuple[object, tuple[int, ...], str, str, int | None]] = {}
        for cell in cell_plans:
            generated_order.append(cell.cell)  # type: ignore[attr-defined]
            use = CotSwapInputUse(
                cell=cell.cell,  # type: ignore[attr-defined]
                prompt_text_sha256=cell.prompt_sha256,  # type: ignore[attr-defined]
                pre_answer_text_sha256=cell.pre_answer_sha256,  # type: ignore[attr-defined]
                full_input_text_sha256=cell.full_input_sha256,  # type: ignore[attr-defined]
                prompt_char_count=len(cell.prompt),  # type: ignore[attr-defined]
                pre_answer_char_count=len(cell.pre_answer_text),  # type: ignore[attr-defined]
                full_input_char_count=len(cell.full_input),  # type: ignore[attr-defined]
                prompt_token_count=cell.prompt_token_count,  # type: ignore[attr-defined]
                full_input_token_count=20,
                full_input_ids_sha256="f" * 64,
                prompt_prefix_token_stable=True,
            )
            payload[cell.cell] = (  # type: ignore[attr-defined]
                use,
                (ord(cell.cell),),
                f"span-{cell.cell}",
                "eos_token",
                ord(cell.cell),
            )
        return payload

    runtime._generate_cell_batch = fake_generate
    runtime._answer_generation = (  # type: ignore[method-assign]
        lambda ids, text, gold, stop_reason, stop_token_id: _generation(
            "2", cell=chr(ids[0]), gold=gold
        )
    )
    scan = runtime.scan_pair(pair, plan)
    assert generated_order == list(CELL_ORDER)
    assert list(scan.input_uses) == list(CELL_ORDER)
    assert list(scan.generations) == list(CELL_ORDER)


def test_runtime_uses_all_requested_visible_devices_for_model_sharding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"gpu-{index}")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=(index + 1) * 1024),
    )

    class FakeModel:
        config = SimpleNamespace(_commit_hash="source-revision")
        generation_config = SimpleNamespace(
            stop_strings=None,
            forced_eos_token_id=None,
            eos_token_id=1,
        )
        hf_device_map = {"model.embed_tokens": 0, "lm_head": 1}

        def eval(self) -> None:
            return None

        def parameters(self) -> object:
            return iter((torch.tensor(0),))

    tokenizer = SimpleNamespace(
        init_kwargs={"_commit_hash": "source-revision"},
        eos_token_id=1,
        padding_side="right",
    )
    calls: list[dict[str, object]] = []

    def fake_wrapper(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return SimpleNamespace(model=FakeModel(), tokenizer=tokenizer)

    import typo_cot.models.wrapper as wrapper_module

    monkeypatch.setattr(wrapper_module, "create_model_wrapper", fake_wrapper)
    config = CotSwapConfig(
        model="test/model",
        benchmark="mmlu",
        pairs=tmp_path / "pairs.jsonl",
        targeting="attribution-4",
        output_dir=tmp_path / "out",
        gpu_id="0,1",
    )

    runtime = HuggingFaceCotSwapRuntime(config, revision="source-revision")
    provenance = runtime.provenance()

    assert calls[0]["gpu_id"] == "0,1"
    assert calls[0]["wrap_for_lxt"] is False
    assert provenance["cuda_visible_devices"] == "0,1"
    assert provenance["model_parallel"] is True
    assert provenance["gpu_names"] == ["gpu-0", "gpu-1"]
    assert provenance["gpu_total_memory_bytes"] == [1024, 2048]
    assert provenance["model_device_map"] == {
        "lm_head": 1,
        "model.embed_tokens": 0,
    }


def test_runtime_batch_trims_each_row_at_its_own_eos_and_preserves_a_16_token_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(cot_runtime._GENERATION, "top_k", 7)
    plan = build_cell_plan(_pair("a"))
    prompt_ids_by_text: dict[str, list[int]] = {}
    full_ids_by_text: dict[str, list[int]] = {}
    for index, cell in enumerate(plan.cells, 1):
        prompt_ids = list(range(index * 100, index * 100 + cell.prompt_token_count))
        prompt_ids_by_text.setdefault(cell.prompt, prompt_ids)
        prompt_ids = prompt_ids_by_text[cell.prompt]
        full_ids_by_text[cell.full_input] = [*prompt_ids, 500 + index]

    class FakeTokenizer:
        pad_token_id = 0

        def __init__(self) -> None:
            self.decoded: list[tuple[list[int], bool, bool]] = []

        def __call__(self, value: object, **kwargs: object) -> dict[str, object]:
            if isinstance(value, str):
                ids = prompt_ids_by_text.get(value, full_ids_by_text.get(value))
                assert ids is not None
                return {"input_ids": list(ids)}
            assert isinstance(value, list)
            rows = [full_ids_by_text[str(text)] for text in value]
            width = max(map(len, rows))
            padded = [[0] * (width - len(row)) + row for row in rows]
            masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            assert kwargs["padding"] is True
            assert kwargs["return_tensors"] == "pt"
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(masks),
            }

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            self.decoded.append((token_ids, skip_special_tokens, clean_up_tokenization_spaces))
            return " ".join(map(str, token_ids))

    tokenizer = FakeTokenizer()
    generate_kwargs: dict[str, object] = {}

    def generate(**kwargs: object) -> object:
        generate_kwargs.update(kwargs)
        inputs = kwargs["input_ids"]
        assert isinstance(inputs, torch.Tensor)
        continuations = torch.tensor(
            [
                [11, 98, *([0] * 14)],
                list(range(20, 36)),
                [12, 99, *([0] * 14)],
                [13, 98, *([0] * 14)],
            ]
        )
        return torch.cat((inputs, continuations), dim=1)

    runtime = object.__new__(HuggingFaceCotSwapRuntime)
    runtime._torch = torch
    runtime.tokenizer = tokenizer
    runtime.model = SimpleNamespace(generate=generate)
    runtime.device = torch.device("cpu")
    runtime.effective_eos_token_ids = (98, 99)
    generated = runtime._generate_cell_batch(plan.cells)

    assert generated["A"][1:] == ((11, 98), "11 98", "eos_token", 98)
    assert generated["B"][1:] == (
        tuple(range(20, 36)),
        " ".join(map(str, range(20, 36))),
        "max_new_tokens",
        None,
    )
    assert generated["C"][1:] == ((12, 99), "12 99", "eos_token", 99)
    assert generated["D"][1:] == ((13, 98), "13 98", "eos_token", 98)
    assert generate_kwargs["eos_token_id"] == [98, 99]
    assert generate_kwargs["max_new_tokens"] == 16
    assert generate_kwargs["do_sample"] is False
    assert generate_kwargs["top_k"] == 7
    assert all(skip is True and cleanup is False for _, skip, cleanup in tokenizer.decoded)


def test_docs_present_the_gpu0_command_and_final_pdf_conflict() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    doc = (root / "docs" / "cot-swap.md").read_text(encoding="utf-8")
    for text in (readme, doc):
        assert "CUDA_VISIBLE_DEVICES=0" in text
        assert "typo-cot cot-swap" in text
        assert "--targeting attribution-4" in text
        assert "--gpu-id 0" in text
        assert "--resume" in text
        assert "19,550" in text
        assert "4,634" in text
        assert "3,539" in text
        assert "symmetr" in text.lower()
        assert "applied-edit" in text.lower()
        assert "model-scale-cot-swap" in text
    assert "one-pair, four-cell batch" in doc
    assert "not specified by the final PDF" in doc
    assert "direct, indirect, total" in readme
