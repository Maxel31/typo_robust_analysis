"""Final-PDF and legacy-backed contracts for the clean-prefix scan."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
import typo_cot.experiments.clean_prefix_scan as prefix_api
import typo_cot.experiments.clean_prefix_scan.runner as clean_prefix_runner
import typo_cot.experiments.clean_prefix_scan.source as clean_prefix_source
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.clean_prefix_scan.runtime import (
    CleanPrefixBoundaryInvalid,
    HuggingFaceCleanPrefixScanRuntime,
)
from typo_cot.experiments.clean_prefix_scan.source import (
    load_extension_source,
    validate_source_snapshot,
)


RELATIVE_BUDGETS = (0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.325, 0.4, 0.5, 0.65, 0.8, 1.0)
ABSOLUTE_BUDGETS = (1, 2, 4, 8, 16, 32, 64)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aligned_word() -> dict[str, object]:
    return {
        "word_index": 0,
        "clean_text": "clean",
        "edited_text": "clena",
        "clean_editable_span": {"start": 0, "end": 5},
        "edited_editable_span": {"start": 0, "end": 5},
        "clean_prompt_span": {"start": 0, "end": 5},
        "edited_prompt_span": {"start": 0, "end": 5},
        "target_ranks": [1],
        "target_token_indices": [1],
        "clean_token_indices": [1],
        "edited_token_indices": [1],
        "clean_final_token": 1,
        "edited_final_token": 1,
    }


def _pair(sample_id: str, *, targeting: str) -> dict[str, object]:
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": "test/model",
        "benchmark": "gsm8k",
        "targeting": targeting,
        "seed": 42,
        "num_edits_requested": 4,
        "num_target_attempts": 1,
        "num_aligned_words": 1,
        "gold_answer": "2",
        "clean": {
            "prompt": f"clean prompt {sample_id}",
            "prompt_token_count": 2,
            "continuation": ("First reason. Then reason. Finally calculate two. The answer is 2."),
            "answer": {
                "value": "2",
                "is_extracted": True,
                "is_correct": True,
                "method": "fixture",
                "primary_method": "fixture",
                "confidence": 1.0,
            },
        },
        "edited": {
            "prompt": f"edited prompt {sample_id}",
            "prompt_token_count": 2,
            "continuation": "Wrong reasoning. The answer is 3.",
            "answer": {
                "value": "3",
                "is_extracted": True,
                "is_correct": False,
                "method": "fixture",
                "primary_method": "fixture",
                "confidence": 1.0,
            },
        },
        "answer_changed": True,
        "target_attempts": [{"token_index": 1, "operation": "substitution"}],
        "aligned_words": [_aligned_word()],
    }


def _write_pair_source(root: Path, *, targeting: str, sample_id: str) -> Path:
    root.mkdir(parents=True)
    pair = _pair(sample_id, targeting=targeting)
    path = root / "pairs.jsonl"
    path.write_text(
        json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
                "targeting": targeting,
                "num_edits": 4,
                "seed": 42,
                "max_new_tokens": 512,
                "gpu_id": "0",
                "limit": None,
                "output_dir": str(root.resolve()),
            },
            "counts": {"discovered": 1, "written": 1, "failed": 0},
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
                "dataset_sample_count": 1,
                "dataset_records_sha256": "d" * 64,
                "generation_protocol": "explicit-greedy-generation/v1",
            },
        },
    )
    return path


class _Runtime:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.generated_k: list[int] = []
        self.fail_after = fail_after

    def provenance(self) -> Mapping[str, object]:
        return {
            "runtime": "clean-prefix-fixture",
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
                "max_new_tokens": 512,
                "use_cache": True,
                "return_dict_in_generate": False,
                "output_scores": False,
                "padding_side": "left",
                "batch_size": 1,
            },
            "effective_eos_token_ids": [99],
            "effective_eos_token_ids_source": "fixture",
            "answer_decoding": "generated-token-ids-only/v1",
            "answer_extraction": "primary-then-empty-only-fallback-cap-aware/v1",
            "pre_answer_boundary": "first-submitted-[Tt]he-answer-is/v1",
            "selection_alignment": "prompt-length-suffix-equality/v1",
            "scan_boundary": "edited-full-must-preserve-separately-tokenized-prompt/v1",
            "input_assembly": "edited-prompt-ids-plus-clean-cot-id-prefix/v1",
        }

    def prepare_pair(self, pair: dict[str, object]) -> object:
        del pair
        alignment = prefix_api.align_clean_cot_suffixes(
            clean_prompt_ids=(1, 2),
            clean_full_ids=(1, 2, *range(10, 18)),
            edited_prompt_ids=(3, 4),
            edited_full_ids=(3, 4, *range(10, 18)),
        )
        return prefix_api.build_prefix_input_plan(
            edited_prompt_ids=(3, 4),
            edited_full_ids=(3, 4, *range(10, 18)),
            alignment=alignment,
        )

    def generate_point(self, plan: object, k: int, *, gold_answer: str = "2") -> object:
        if self.fail_after is not None and len(self.generated_k) >= self.fail_after:
            raise RuntimeError("fixture interruption")
        assert gold_answer == "2"
        self.generated_k.append(k)
        input_ids = plan.input_ids(k)
        value = "3" if k == 0 else "2"
        extraction = extract_with_fallback(
            f"The answer is {value}.",
            benchmark="gsm8k",
            correct_answer="2",
        )
        return prefix_api.CleanPrefixPointScan(
            k=k,
            input_use=prefix_api.CleanPrefixInputUse(
                k=k,
                input_ids=input_ids,
                prompt_token_count=2,
                prefix_token_count=k,
            ),
            generation=prefix_api.CleanPrefixGeneration(
                token_ids=(99,),
                text=f"The answer is {value}.",
                value=extraction.value,
                is_extracted=extraction.is_extracted,
                is_correct=extraction.is_correct,
                method=extraction.method,
                primary_method=extraction.primary_method,
                stop_reason="eos_token",
                stop_token_id=99,
            ),
        )


class _NoRuntimeUse:
    def __getattribute__(self, name: str) -> object:
        if name.startswith("__"):
            return super().__getattribute__(name)
        raise AssertionError(f"completed resume touched runtime method {name}")


def _config(tmp_path: Path, *, cohort: str) -> object:
    common = {
        "model": "test/model",
        "benchmark": "gsm8k",
        "cohort": cohort,
        "relative_budgets": RELATIVE_BUDGETS,
        "absolute_budgets": ABSOLUTE_BUDGETS,
        "output_dir": tmp_path / "output",
        "gpu_id": "0",
        "limit": None,
        "resume": False,
    }
    if cohort == "primary":
        return prefix_api.CleanPrefixScanConfig(
            **common,
            fixed_window_run=tmp_path / "fixed-window",
            pairs=(),
            max_pairs=None,
        )
    return prefix_api.CleanPrefixScanConfig(
        **common,
        fixed_window_run=None,
        pairs=(tmp_path / "attribution" / "pairs.jsonl", tmp_path / "random" / "pairs.jsonl"),
        max_pairs=150,
    )


def _extension_config(tmp_path: Path, *, resume: bool = False) -> object:
    attribution = _write_pair_source(
        tmp_path / "attribution", targeting="attribution-4", sample_id="sample-a"
    )
    random = _write_pair_source(tmp_path / "random", targeting="random-4", sample_id="sample-r")
    return prefix_api.CleanPrefixScanConfig(
        model="test/model",
        benchmark="gsm8k",
        cohort="extension",
        fixed_window_run=None,
        pairs=(attribution, random),
        max_pairs=150,
        relative_budgets=RELATIVE_BUDGETS,
        absolute_budgets=ABSOLUTE_BUDGETS,
        output_dir=tmp_path / "output",
        gpu_id="0",
        limit=1,
        resume=resume,
    )


def test_catalog_marks_clean_prefix_scan_as_implemented_with_four_outputs() -> None:
    spec = get_experiment("clean-prefix-scan")

    assert spec.status == "implemented"
    assert spec.outputs == (
        "prefix_scan_records.jsonl",
        "pair_status_records.jsonl",
        "prefix_scan_summary.json",
        "run.json",
    )
    assert "--cohort" in spec.required_arguments
    assert "--target-set" not in spec.required_arguments


def test_public_api_exposes_the_point_resumable_runtime_contract() -> None:
    expected = {
        "RELATIVE_BUDGETS",
        "ABSOLUTE_BUDGETS",
        "PrefixBudget",
        "CleanCotAlignment",
        "PrefixInputPlan",
        "PrefixTrajectorySummary",
        "align_clean_cot_suffixes",
        "build_prefix_input_plan",
        "build_budget_grid",
        "select_extension_sample_ids",
        "summarize_prefix_correctness",
        "CleanPrefixScanConfig",
        "CleanPrefixGeneration",
        "CleanPrefixInputUse",
        "CleanPrefixPointScan",
        "CleanPrefixPairScan",
        "CleanPrefixScanResult",
        "CleanPrefixScanRunError",
        "CleanPrefixScanRuntime",
        "run_clean_prefix_scan",
    }

    assert expected.issubset(set(prefix_api.__all__))
    assert callable(HuggingFaceCleanPrefixScanRuntime.prepare_pair)
    assert callable(HuggingFaceCleanPrefixScanRuntime.generate_point)


def test_config_requires_the_primary_fixed_window_source_only(tmp_path: Path) -> None:
    config = _config(tmp_path, cohort="primary")
    assert config.cohort == "primary"
    assert config.fixed_window_run == tmp_path / "fixed-window"
    assert config.pairs == ()
    assert config.max_pairs is None

    with pytest.raises(ValueError, match="primary.*fixed-window"):
        prefix_api.CleanPrefixScanConfig(
            model="test/model",
            benchmark="gsm8k",
            cohort="primary",
            fixed_window_run=None,
            pairs=(tmp_path / "pairs.jsonl",),
            max_pairs=150,
            relative_budgets=RELATIVE_BUDGETS,
            absolute_budgets=ABSOLUTE_BUDGETS,
            output_dir=tmp_path / "bad-primary",
        )


def test_config_requires_two_extension_pair_arms_and_a_target_cap(tmp_path: Path) -> None:
    config = _config(tmp_path, cohort="extension")
    assert config.cohort == "extension"
    assert config.fixed_window_run is None
    assert len(config.pairs) == 2
    assert config.max_pairs == 150

    with pytest.raises(ValueError, match="extension.*pairs"):
        prefix_api.CleanPrefixScanConfig(
            model="test/model",
            benchmark="gsm8k",
            cohort="extension",
            fixed_window_run=tmp_path / "fixed-window",
            pairs=(),
            max_pairs=None,
            relative_budgets=RELATIVE_BUDGETS,
            absolute_budgets=ABSOLUTE_BUDGETS,
            output_dir=tmp_path / "bad-extension",
        )


@pytest.mark.parametrize("value", (-0.1, 1.1, math.nan, math.inf, -math.inf))
def test_relative_budget_validation_rejects_values_outside_zero_through_one(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="relative"):
        prefix_api.build_budget_grid(
            10,
            relative_budgets=(0.0, value, 1.0),
            absolute_budgets=(1,),
        )


def test_budget_grid_requires_both_relative_endpoints() -> None:
    with pytest.raises(ValueError, match="0.*1"):
        prefix_api.build_budget_grid(
            10,
            relative_budgets=(0.2, 0.5),
            absolute_budgets=(1, 2),
        )


@pytest.mark.parametrize(
    ("relative", "absolute", "message"),
    (
        ((0.0, 0.5, 0.5, 1.0), (1,), "relative.*duplicate"),
        ((0.0, 0.5, 1.0), (1, 1), "absolute.*duplicate"),
    ),
)
def test_budget_grid_rejects_duplicate_requested_origins(
    relative: tuple[float, ...],
    absolute: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prefix_api.build_budget_grid(
            10,
            relative_budgets=relative,
            absolute_budgets=absolute,
        )


def test_budget_grid_uses_legacy_python_round_and_preserves_deduplicated_origins() -> None:
    grid = prefix_api.build_budget_grid(
        5,
        relative_budgets=(0.0, 0.5, 1.0),
        absolute_budgets=(1, 2, 4, 8),
    )

    assert [point.k for point in grid] == [0, 1, 2, 4, 5]
    midpoint = next(point for point in grid if point.k == 2)
    assert midpoint.relative_sources == (0.5,)
    assert midpoint.absolute_sources == (2,)
    assert grid[0].is_no_prefix is True
    assert grid[-1].is_complete_prefix is True
    assert all(point.k <= 5 for point in grid)


def test_selection_alignment_keeps_the_old_two_stage_boundary_audit() -> None:
    alignment = prefix_api.align_clean_cot_suffixes(
        clean_prompt_ids=(1, 2),
        clean_full_ids=(1, 2, 7, 8),
        edited_prompt_ids=(3, 4),
        edited_full_ids=(9, 9, 7, 8),
    )

    # The submitted selection stage compared the two suffixes. The later scan
    # separately rejected a target whose edited full input did not preserve the
    # independently tokenized edited prompt, producing 2,094/2,100 valid scans.
    assert alignment.clean_cot_ids == (7, 8)
    with pytest.raises(ValueError, match="edited.*prompt.*prefix"):
        prefix_api.build_prefix_input_plan(
            edited_prompt_ids=(3, 4),
            edited_full_ids=(9, 9, 7, 8),
            alignment=alignment,
        )


def test_runtime_prefix_plan_concatenates_ids_without_decode_and_retokenize() -> None:
    alignment = prefix_api.align_clean_cot_suffixes(
        clean_prompt_ids=(1, 2),
        clean_full_ids=(1, 2, 7, 8, 9),
        edited_prompt_ids=(3, 4),
        edited_full_ids=(3, 4, 7, 8, 9),
    )
    plan = prefix_api.build_prefix_input_plan(
        edited_prompt_ids=(3, 4),
        edited_full_ids=(3, 4, 7, 8, 9),
        alignment=alignment,
    )

    assert plan.cot_token_count == 3
    assert plan.input_ids(0) == (3, 4)
    assert plan.input_ids(2) == (3, 4, 7, 8)
    assert plan.input_ids(3) == (3, 4, 7, 8, 9)
    with pytest.raises(ValueError, match="budget"):
        plan.input_ids(4)


def test_clean_cot_length_filter_is_inclusive_at_eight_and_512_tokens() -> None:
    assert prefix_api.CleanCotAlignment((7,) * 8).eligible_length is True
    assert prefix_api.CleanCotAlignment((7,) * 512).eligible_length is True
    assert prefix_api.CleanCotAlignment((7,) * 7).eligible_length is False
    assert prefix_api.CleanCotAlignment((7,) * 513).eligible_length is False


def test_extension_selection_caps_each_arm_then_uses_proportional_systematic_ids() -> None:
    selected = prefix_api.select_extension_sample_ids(
        {
            "attribution-4": tuple(f"a{index:03d}" for index in range(6)),
            "random-4": tuple(f"r{index:03d}" for index in range(4)),
        },
        max_pairs=5,
        cap_per_targeting=400,
    )

    assert selected == (
        ("attribution-4", "a000"),
        ("attribution-4", "a002"),
        ("attribution-4", "a004"),
        ("random-4", "r000"),
        ("random-4", "r002"),
    )


def test_extension_selection_is_order_independent_and_never_uses_ids_after_the_cap() -> None:
    ascending = tuple(f"a{index:03d}" for index in range(405))
    forward = prefix_api.select_extension_sample_ids(
        {"attribution-4": ascending, "random-4": ("r001", "r000")},
        max_pairs=20,
        cap_per_targeting=400,
    )
    reversed_input = prefix_api.select_extension_sample_ids(
        {
            "random-4": ("r000", "r001"),
            "attribution-4": tuple(reversed(ascending)),
        },
        max_pairs=20,
        cap_per_targeting=400,
    )

    assert forward == reversed_input
    assert all(sample_id < "a400" for arm, sample_id in forward if arm == "attribution-4")


def test_correctness_summary_uses_all_later_distinct_k_values() -> None:
    summary = prefix_api.summarize_prefix_correctness(
        ((0, False), (1, True), (2, False), (4, True), (5, True)),
        cot_token_count=5,
    )

    assert summary.k0_correct is False
    assert summary.full_correct is True
    assert summary.k_star == 4
    assert summary.r_star == pytest.approx(0.8)
    assert summary.correctness_transitions == 3
    assert summary.non_monotone is True
    assert summary.stable_at_k == {0: False, 1: False, 2: False, 4: True, 5: True}
    assert summary.stable_le_0_2 is False


def test_short_event_is_stable_k_star_fraction_not_first_success() -> None:
    summary = prefix_api.summarize_prefix_correctness(
        ((0, False), (1, True), (2, True), (10, True)),
        cot_token_count=10,
    )

    assert summary.k_star == 1
    assert summary.stable_le_0_2 is True


def test_extension_runner_publishes_four_outputs_from_one_fixed_denominator(
    tmp_path: Path,
) -> None:
    config = _extension_config(tmp_path)
    runtime = _Runtime()

    result = prefix_api.run_clean_prefix_scan(config, runtime=runtime)

    assert isinstance(result, prefix_api.CleanPrefixScanResult)
    output = tmp_path / "output"
    public_names = {
        "prefix_scan_records.jsonl",
        "pair_status_records.jsonl",
        "prefix_scan_summary.json",
        "run.json",
    }
    assert {path.name for path in output.iterdir()} == public_names
    rows = [
        json.loads(line) for line in (output / "prefix_scan_records.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert [point["k"] for point in rows[0]["points"]] == runtime.generated_k
    assert rows[0]["trajectory"]["k0_correct"] is False
    assert rows[0]["trajectory"]["full_correct"] is True
    summary = json.loads((output / "prefix_scan_summary.json").read_text())
    assert summary["counts"]["selected_targets"] == 1
    assert summary["counts"]["valid_scans"] == 1
    assert summary["counts"]["fresh_k0_wrong"] == 1
    assert summary["metrics"]["full_correct"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }


def test_partial_point_checkpoints_resume_without_regenerating_completed_budgets(
    tmp_path: Path,
) -> None:
    config = _extension_config(tmp_path)
    interrupted = _Runtime(fail_after=2)

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="fixture interruption"):
        prefix_api.run_clean_prefix_scan(config, runtime=interrupted)

    output = tmp_path / "output"
    for name in (
        "prefix_scan_records.jsonl",
        "pair_status_records.jsonl",
        "prefix_scan_summary.json",
    ):
        assert not (output / name).exists()
    assert interrupted.generated_k == [0, 1]
    crash_left_temporary = output / ".prefix_scan_records.jsonl.999999.tmp"
    crash_left_temporary.write_text("incomplete", encoding="utf-8")

    resumed = _Runtime()
    result = prefix_api.run_clean_prefix_scan(
        replace(config, resume=True),
        runtime=resumed,
    )

    assert isinstance(result, prefix_api.CleanPrefixScanResult)
    assert not {0, 1}.intersection(resumed.generated_k)
    assert not crash_left_temporary.exists()
    assert (output / "run.json").is_file()
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    prefix_api.run_clean_prefix_scan(
        replace(config, resume=True),
        runtime=_NoRuntimeUse(),
    )
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    assert after == before


def test_completed_resume_rejects_output_tampering_without_touching_runtime(
    tmp_path: Path,
) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    result.records_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="completed.*hash"):
        prefix_api.run_clean_prefix_scan(
            replace(config, resume=True),
            runtime=_NoRuntimeUse(),
        )


def test_checkpoint_resume_recomputes_extraction_even_if_hash_registry_is_updated(
    tmp_path: Path,
) -> None:
    config = _extension_config(tmp_path)
    with pytest.raises(prefix_api.CleanPrefixScanRunError):
        prefix_api.run_clean_prefix_scan(config, runtime=_Runtime(fail_after=2))

    output = tmp_path / "output"
    checkpoint_path = next((output / ".clean-prefix-scan-work" / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["points"][0]["generation"]["method"] = "forged:method"
    _write_json(checkpoint_path, checkpoint)
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    first = next(iter(run["checkpoints"].values()))
    first["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    _write_json(run_path, run)

    with pytest.raises(ValueError, match="extraction"):
        prefix_api.run_clean_prefix_scan(replace(config, resume=True), runtime=_Runtime())


def test_completed_resume_recomputes_extraction_after_registry_hash_is_updated(
    tmp_path: Path,
) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    rows = [json.loads(line) for line in result.records_path.read_text().splitlines()]
    rows[0]["points"][0]["generation"]["method"] = "forged:method"
    result.records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    run["outputs"][result.records_path.name]["sha256"] = hashlib.sha256(
        result.records_path.read_bytes()
    ).hexdigest()
    _write_json(result.run_path, run)

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="semantic.*extraction"):
        prefix_api.run_clean_prefix_scan(
            replace(config, resume=True),
            runtime=_NoRuntimeUse(),
        )


@pytest.mark.parametrize(
    ("output_name", "mutate"),
    (
        (
            "pair_status_records.jsonl",
            lambda payload: payload.__setitem__("headline_denominator", False),
        ),
        (
            "prefix_scan_summary.json",
            lambda payload: payload["metrics"]["full_correct"].__setitem__("numerator", 0),
        ),
    ),
)
def test_completed_resume_rejects_semantic_output_tampering_with_updated_hash(
    tmp_path: Path,
    output_name: str,
    mutate: object,
) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    output_path = config.output_dir / output_name
    if output_name.endswith(".jsonl"):
        rows = [json.loads(line) for line in output_path.read_text().splitlines()]
        mutate(rows[0])
        output_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        payload = json.loads(output_path.read_text())
        mutate(payload)
        _write_json(output_path, payload)
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    run["outputs"][output_name]["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    _write_json(result.run_path, run)

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="completed.*semantic"):
        prefix_api.run_clean_prefix_scan(
            replace(config, resume=True),
            runtime=_NoRuntimeUse(),
        )


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("runtime", {"dtype": "float32"}),
        ("counts", {"selected_targets": 999}),
        ("checkpoints", {"sha256": "f" * 64}),
    ),
)
def test_completed_resume_rejects_manifest_semantic_tampering(
    tmp_path: Path,
    field: str,
    mutated: dict[str, object],
) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    if field == "checkpoints":
        run[field]["forged"] = {
            "file": "forged.json",
            "targeting": "attribution-4",
            "sample_id": "sample-a",
            **mutated,
        }
    else:
        run[field].update(mutated)
    _write_json(result.run_path, run)

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="completed.*semantic"):
        prefix_api.run_clean_prefix_scan(
            replace(config, resume=True),
            runtime=_NoRuntimeUse(),
        )


def test_completed_resume_rejects_frozen_plan_cases_sha256_tampering(tmp_path: Path) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    run["plan"]["cases_sha256"] = "f" * 64
    _write_json(result.run_path, run)

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="frozen plan"):
        prefix_api.run_clean_prefix_scan(
            replace(config, resume=True),
            runtime=_NoRuntimeUse(),
        )


def test_completed_resume_binds_record_token_ids_to_the_frozen_plan(tmp_path: Path) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    rows = [json.loads(line) for line in result.records_path.read_text().splitlines()]
    row = rows[0]
    for point in row["points"]:
        point["input_use"]["input_ids"] = [
            token_id + 1000 for token_id in point["input_use"]["input_ids"]
        ]
    prompt_count = row["token_plan"]["edited_prompt_token_count"]
    cot_count = row["token_plan"]["clean_cot_token_count"]
    no_prefix = next(point for point in row["points"] if point["k"] == 0)
    full_prefix = next(point for point in row["points"] if point["k"] == cot_count)
    reconstructed = prefix_api.PrefixInputPlan(
        tuple(no_prefix["input_use"]["input_ids"][:prompt_count]),
        tuple(full_prefix["input_use"]["input_ids"][prompt_count:]),
    )
    row["token_plan"]["sha256"] = _canonical_sha256(reconstructed.to_dict())
    result.records_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
        encoding="utf-8",
    )
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    run["outputs"][result.records_path.name]["sha256"] = hashlib.sha256(
        result.records_path.read_bytes()
    ).hexdigest()
    _write_json(result.run_path, run)

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="frozen plan"):
        prefix_api.run_clean_prefix_scan(
            replace(config, resume=True),
            runtime=_NoRuntimeUse(),
        )


def test_completed_manifest_does_not_reference_deleted_checkpoints(tmp_path: Path) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    run = json.loads(result.run_path.read_text(encoding="utf-8"))

    assert run["checkpoints"] == {}
    assert run["counts"]["checkpointed_pairs"] == 0
    assert {path.name for path in config.output_dir.iterdir()} == {
        "prefix_scan_records.jsonl",
        "pair_status_records.jsonl",
        "prefix_scan_summary.json",
        "run.json",
    }


def test_runtime_api_passes_gold_explicitly_without_hidden_pair_state(tmp_path: Path) -> None:
    config = _extension_config(tmp_path)

    class ExplicitGoldRuntime(_Runtime):
        def activate_pair(self, _pair: Mapping[str, object]) -> None:
            raise AssertionError("runner must not use hidden mutable pair state")

        def generate_point(self, plan: object, k: int, *, gold_answer: str) -> object:
            assert gold_answer == "2"
            return super().generate_point(plan, k, gold_answer=gold_answer)

    result = prefix_api.run_clean_prefix_scan(config, runtime=ExplicitGoldRuntime())

    signature = inspect.signature(HuggingFaceCleanPrefixScanRuntime.generate_point)
    assert signature.parameters["gold_answer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert result.records == 1


def test_generation_rejects_internally_inconsistent_stop_metadata() -> None:
    common = {
        "text": "The answer is 2.",
        "value": "2",
        "is_extracted": True,
        "is_correct": True,
        "method": "fixture",
        "primary_method": "fixture",
    }
    with pytest.raises(ValueError, match="last generated token"):
        prefix_api.CleanPrefixGeneration(
            token_ids=(99,),
            stop_reason="eos_token",
            stop_token_id=100,
            **common,
        )
    with pytest.raises(ValueError, match="512"):
        prefix_api.CleanPrefixGeneration(
            token_ids=(1,),
            stop_reason="max_new_tokens",
            stop_token_id=None,
            **common,
        )


def test_runner_rejects_eos_stop_outside_runtime_provenance(tmp_path: Path) -> None:
    config = _extension_config(tmp_path)

    class WrongEosRuntime(_Runtime):
        def generate_point(self, plan: object, k: int, *, gold_answer: str) -> object:
            point = super().generate_point(plan, k, gold_answer=gold_answer)
            return replace(
                point,
                generation=replace(
                    point.generation,
                    token_ids=(100,),
                    stop_token_id=100,
                ),
            )

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="effective EOS"):
        prefix_api.run_clean_prefix_scan(config, runtime=WrongEosRuntime())


def test_selected_runtime_boundary_failure_is_not_replaced(tmp_path: Path) -> None:
    config = replace(_extension_config(tmp_path), limit=None, max_pairs=2)

    class BoundaryRuntime(_Runtime):
        def prepare_pair(self, pair: dict[str, object]) -> object:
            if pair["sample_id"] == "sample-a":
                alignment = prefix_api.CleanCotAlignment(tuple(range(10, 18)))
                raise CleanPrefixBoundaryInvalid(
                    "edited full input does not preserve the edited prompt prefix",
                    alignment=alignment,
                    edited_prompt_ids=(3, 4),
                    edited_full_ids=(9, 9, *alignment.clean_cot_ids),
                )
            return super().prepare_pair(pair)

    result = prefix_api.run_clean_prefix_scan(config, runtime=BoundaryRuntime())
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    statuses = [
        json.loads(line) for line in result.pair_status_records_path.read_text().splitlines()
    ]

    assert summary["counts"]["selected_targets"] == 2
    assert summary["counts"]["valid_scans"] == 1
    invalid = next(row for row in statuses if row["sample_id"] == "sample-a")
    assert invalid["selected_for_execution"] is True
    assert invalid["boundary_valid"] is False
    assert invalid["execution_status"] == "invalid-boundary"


def test_primary_length_ineligible_targets_are_retained_but_not_generated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_config = _extension_config(tmp_path)
    source = load_extension_source(
        extension_config.pairs,
        model=extension_config.model,
        benchmark=extension_config.benchmark,
    )
    monkeypatch.setattr(clean_prefix_runner, "load_source_bundle", lambda **_kwargs: source)
    config = prefix_api.CleanPrefixScanConfig(
        model="test/model",
        benchmark="gsm8k",
        cohort="primary",
        fixed_window_run=tmp_path / "fixed-window-source",
        pairs=(),
        max_pairs=None,
        relative_budgets=RELATIVE_BUDGETS,
        absolute_budgets=ABSOLUTE_BUDGETS,
        output_dir=tmp_path / "primary-output",
        gpu_id="0",
    )

    class TooShortRuntime(_Runtime):
        def prepare_pair(self, _pair: dict[str, object]) -> object:
            return prefix_api.PrefixInputPlan((3, 4), tuple(range(10, 17)))

    runtime = TooShortRuntime()
    result = prefix_api.run_clean_prefix_scan(config, runtime=runtime)
    statuses = [
        json.loads(line) for line in result.pair_status_records_path.read_text().splitlines()
    ]

    assert runtime.generated_k == []
    assert result.records == 0
    assert all(row["selected_for_execution"] is True for row in statuses)
    assert all(row["candidate_eligible"] is False for row in statuses)
    assert all(row["execution_status"] == "not-candidate" for row in statuses)


def test_all_headline_metrics_share_the_fresh_k0_wrong_denominator(tmp_path: Path) -> None:
    config = replace(_extension_config(tmp_path), limit=None, max_pairs=2)

    class PerPairRuntime(_Runtime):
        def prepare_pair(self, pair: dict[str, object]) -> object:
            prompt = (3, 4) if pair["sample_id"] == "sample-a" else (5, 6)
            return prefix_api.PrefixInputPlan(prompt, tuple(range(10, 18)))

        def generate_point(self, plan: object, k: int, *, gold_answer: str) -> object:
            point = super().generate_point(plan, k, gold_answer=gold_answer)
            if plan.edited_prompt_ids == (3, 4) and k == 0:
                extraction = extract_with_fallback(
                    "The answer is 2.", benchmark="gsm8k", correct_answer="2"
                )
                point = replace(
                    point,
                    generation=replace(
                        point.generation,
                        text="The answer is 2.",
                        value=extraction.value,
                        is_correct=True,
                        method=extraction.method,
                        primary_method=extraction.primary_method,
                    ),
                )
            return point

    result = prefix_api.run_clean_prefix_scan(config, runtime=PerPairRuntime())
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert summary["counts"]["valid_scans"] == 2
    assert summary["counts"]["fresh_k0_correct"] == 1
    assert summary["counts"]["fresh_k0_wrong"] == 1
    for name in ("full_correct", "stable_by_20_percent", "non_monotone"):
        assert summary["metrics"][name]["denominator"] == 1
    assert all(
        row["point_correct"]["denominator"] == 1 and row["stable_through_later"]["denominator"] == 1
        for row in summary["metrics"]["relative_budgets"]
    )


def test_absolute_stable_metric_keeps_short_cot_rows_in_the_common_denominator(
    tmp_path: Path,
) -> None:
    config = replace(_extension_config(tmp_path), limit=None, max_pairs=2)

    class VariableLengthRuntime(_Runtime):
        def prepare_pair(self, pair: dict[str, object]) -> object:
            cot_length = 8 if pair["sample_id"] == "sample-a" else 16
            return prefix_api.PrefixInputPlan((3, 4), tuple(range(10, 10 + cot_length)))

    result = prefix_api.run_clean_prefix_scan(config, runtime=VariableLengthRuntime())
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    budget_16 = next(
        row for row in summary["metrics"]["absolute_budgets"] if row["absolute_budget"] == 16
    )

    assert budget_16["point_correct"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert budget_16["stable_through_later"] == {
        "numerator": 2,
        "denominator": 2,
        "rate": 1.0,
    }
    assert budget_16["exact_point_applicable"] == 1
    assert budget_16["shorter_than_budget"] == 1


def test_summary_preserves_appendix_d_printed_references_without_pooling_primary(
    tmp_path: Path,
) -> None:
    config = _extension_config(tmp_path)
    result = prefix_api.run_clean_prefix_scan(config, runtime=_Runtime())
    reference = json.loads(result.summary_path.read_text())["historical_reference"]

    assert reference["appendix_d_primary_absolute_point_correct_percent"] == {
        "denominator": 172,
        "pdf_printed_percent_by_k": {
            "0": 0.0,
            "1": 8.7,
            "2": 23.3,
            "4": 24.4,
            "8": 36.6,
            "16": 44.8,
            "32": 59.9,
            "64": 75.9,
        },
    }
    assert reference["appendix_d_grid_sensitivity"] == {
        "denominator": 1858,
        "stable_by_20_percent": {
            "full": 537,
            "relative_only": 544,
            "absolute_only": 541,
            "half_relative_plus_absolute": 545,
            "sparse_relative_plus_absolute": 561,
        },
        "includes_primary": False,
    }


def test_source_drift_during_generation_prevents_publication(tmp_path: Path) -> None:
    config = _extension_config(tmp_path)

    class MutatingRuntime(_Runtime):
        changed = False

        def generate_point(self, plan: object, k: int, *, gold_answer: str) -> object:
            point = super().generate_point(plan, k, gold_answer=gold_answer)
            if not self.changed:
                config.pairs[0].write_text(
                    config.pairs[0].read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                self.changed = True
            return point

    with pytest.raises(prefix_api.CleanPrefixScanRunError, match="source snapshot changed"):
        prefix_api.run_clean_prefix_scan(config, runtime=MutatingRuntime())
    for name in (
        "prefix_scan_records.jsonl",
        "pair_status_records.jsonl",
        "prefix_scan_summary.json",
    ):
        assert not (config.output_dir / name).exists()


def test_extension_source_bundle_rejects_post_load_mutation(tmp_path: Path) -> None:
    config = _extension_config(tmp_path)
    source = load_extension_source(config.pairs, model=config.model, benchmark=config.benchmark)
    config.pairs[0].write_text(
        config.pairs[0].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source snapshot changed"):
        validate_source_snapshot(source)


def test_extension_source_hash_is_bound_to_exact_bytes_seen_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _extension_config(tmp_path)
    original = clean_prefix_source.fixed_runner._load_source
    swapped = False

    def replace_before_parse(*args: object, **kwargs: object) -> object:
        nonlocal swapped
        path = Path(args[0])
        if not swapped:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["sample_id"] = "atomic-replacement"
            path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            swapped = True
        return original(*args, **kwargs)

    monkeypatch.setattr(clean_prefix_source.fixed_runner, "_load_source", replace_before_parse)

    with pytest.raises(ValueError, match="exact parsed bytes"):
        load_extension_source(config.pairs, model=config.model, benchmark=config.benchmark)


def test_production_prepare_pair_uses_first_trigger_and_token_exact_suffix() -> None:
    class CharacterTokenizer:
        def __call__(self, text: str, **_kwargs: object) -> dict[str, object]:
            return {"input_ids": [1, *[ord(character) for character in text]]}

    runtime = object.__new__(HuggingFaceCleanPrefixScanRuntime)
    runtime.tokenizer = CharacterTokenizer()
    pair = _pair("runtime", targeting="attribution-4")
    pair["clean"]["prompt"] = "C"
    pair["clean"]["prompt_token_count"] = 2
    pair["edited"]["prompt"] = "E"
    pair["edited"]["prompt_token_count"] = 2

    plan = runtime.prepare_pair(pair)

    expected_text = "First reason. Then reason. Finally calculate two. "
    assert plan.edited_prompt_ids == (1, ord("E"))
    assert plan.clean_cot_ids == tuple(ord(character) for character in expected_text)
    assert plan.input_ids(plan.cot_token_count) == (
        1,
        ord("E"),
        *[ord(character) for character in expected_text],
    )


def test_production_generate_point_passes_direct_ids_and_decodes_only_new_tokens() -> None:
    import torch

    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> object:
        calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        return torch.cat((input_ids, torch.tensor([[99]], dtype=torch.long)), dim=1)

    runtime = object.__new__(HuggingFaceCleanPrefixScanRuntime)
    runtime._torch = torch
    runtime.device = torch.device("cpu")
    runtime.config = SimpleNamespace(benchmark="gsm8k")
    runtime.model = SimpleNamespace(generate=generate)
    runtime.tokenizer = SimpleNamespace(
        pad_token_id=0,
        decode=lambda token_ids, **_kwargs: "The answer is 2.",
    )
    runtime.effective_eos_token_ids = (99,)
    plan = prefix_api.PrefixInputPlan((1, 2), tuple(range(10, 18)))

    point = runtime.generate_point(plan, 3, gold_answer="2")

    assert point.input_use.input_ids == (1, 2, 10, 11, 12)
    assert point.generation.token_ids == (99,)
    assert point.generation.is_correct is True
    assert tuple(calls[0]["input_ids"][0].tolist()) == (1, 2, 10, 11, 12)
    assert calls[0]["max_new_tokens"] == 512
    assert calls[0]["do_sample"] is False


def test_cli_parses_both_auditable_source_modes() -> None:
    parser = cli_module._parser()
    common = [
        "--relative-budgets",
        "0",
        ".2",
        "1",
        "--absolute-budgets",
        "1",
        "2",
        "--output-dir",
        "output",
    ]
    primary = parser.parse_args(
        [
            "clean-prefix-scan",
            "--model",
            "test/model",
            "--benchmark",
            "gsm8k",
            "--cohort",
            "primary",
            "--fixed-window-run",
            "fixed",
            *common,
        ]
    )
    extension = parser.parse_args(
        [
            "clean-prefix-scan",
            "--model",
            "test/model",
            "--benchmark",
            "arc",
            "--cohort",
            "extension",
            "--pairs",
            "attribution/pairs.jsonl",
            "random/pairs.jsonl",
            "--max-pairs",
            "150",
            *common,
        ]
    )

    assert primary.cohort == "primary"
    assert primary.fixed_window_run == Path("fixed")
    assert extension.cohort == "extension"
    assert extension.pairs == [Path("attribution/pairs.jsonl"), Path("random/pairs.jsonl")]
    assert extension.max_pairs == 150


def test_documentation_separates_pdf_rules_from_legacy_backed_details() -> None:
    project = Path(__file__).resolve().parents[1]
    readme = (project / "README.md").read_text(encoding="utf-8")
    details = (project / "docs" / "clean-prefix-scan.md").read_text(encoding="utf-8")

    for text in (readme, details):
        assert "--cohort primary" in text
        assert "--fixed-window-run" in text
        assert "--cohort extension" in text
        assert "--pairs" in text
        assert "--max-pairs 150" in text
        assert "legacy-backed" in text
        assert "prefix_scan_records.jsonl" in text
        assert "pair_status_records.jsonl" in text
        assert "prefix_scan_summary.json" in text
        assert "run.json" in text

    assert "2,094" in details
    assert "1,858" in details
    assert "must not be included" in details
