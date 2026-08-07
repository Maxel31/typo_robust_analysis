"""Final-paper contract tests for answer-level coordinate controls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.fixed_window_answer_patching import LayerWindow
from typo_cot.experiments.fixed_window_answer_patching.runner import (
    AnswerGeneration,
    BaselineScan,
    DirectionWindowScan,
    FixedWindowAnswerPatchingConfig,
    PairWindowScan,
    run_fixed_window_answer_patching,
)
from typo_cot.experiments.patch_coordinate_controls.metrics import exact_mcnemar
from typo_cot.experiments.patch_coordinate_controls.planning import (
    DonorItem,
    assign_cross_item_donors,
    plan_offset_positions,
)
from typo_cot.experiments.patch_coordinate_controls.runner import (
    CONTROL_NAMES,
    ControlScan,
    CoordinateControlConfig,
    CoordinateControlResult,
    CoordinateControlRunError,
    run_patch_coordinate_controls,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _answer(value: str, *, token: int) -> AnswerGeneration:
    return AnswerGeneration(
        token_ids=(token,),
        text=f"The answer is {value}." if value else "No final answer.",
        value=value,
        is_extracted=bool(value),
        is_correct=value == "2",
        method="primary:fixture" if value else "unextractable",
        primary_method="fixture" if value else "no_match",
    )


def _aligned_word(index: int, *, clean_final: int, edited_final: int) -> dict[str, object]:
    return {
        "word_index": index,
        "clean_text": f"word{index}",
        "edited_text": f"wrod{index}",
        "clean_editable_span": {"start": index * 5, "end": index * 5 + 4},
        "edited_editable_span": {"start": index * 5, "end": index * 5 + 4},
        "clean_prompt_span": {"start": index * 5, "end": index * 5 + 4},
        "edited_prompt_span": {"start": index * 5, "end": index * 5 + 4},
        "target_ranks": [index + 1],
        "target_token_indices": [clean_final],
        "clean_token_indices": [clean_final],
        "edited_token_indices": [edited_final],
        "clean_final_token": clean_final,
        "edited_final_token": edited_final,
    }


def _pair(
    sample_id: str,
    *,
    targeting: str = "attribution-4",
    aligned_positions: tuple[tuple[int, int], ...] = ((2, 2),),
    prompt_token_count: int = 8,
) -> dict[str, object]:
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": "test/model",
        "benchmark": "gsm8k",
        "targeting": targeting,
        "seed": 42,
        "num_edits_requested": 4,
        "num_aligned_words": len(aligned_positions),
        "gold_answer": "2",
        "clean": {
            "prompt": f"clean {sample_id}",
            "prompt_token_count": prompt_token_count,
            "continuation": "The answer is 2.",
            "answer": {
                "value": "2",
                "is_extracted": True,
                "is_correct": True,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "edited": {
            "prompt": f"edited {sample_id}",
            "prompt_token_count": prompt_token_count,
            "continuation": "The answer is 3.",
            "answer": {
                "value": "3",
                "is_extracted": True,
                "is_correct": False,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "answer_changed": True,
        "aligned_words": [
            _aligned_word(index, clean_final=clean, edited_final=edited)
            for index, (clean, edited) in enumerate(aligned_positions)
        ],
    }


def _write_pair_source(root: Path, pairs: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True)
    path = root / "pairs.jsonl"
    path.write_text(
        "".join(
            json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n"
            for pair in sorted(pairs, key=lambda row: str(row["sample_id"]))
        ),
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
                "targeting": "attribution-4",
                "num_edits": 4,
                "seed": 42,
                "max_new_tokens": 512,
                "gpu_id": "0",
                "limit": None,
                "output_dir": str(root.resolve()),
            },
            "counts": {"discovered": len(pairs), "written": len(pairs), "failed": 0},
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
                "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
                "dataset_sample_count": len(pairs),
                "dataset_records_sha256": "dataset-sha256",
                "random_seed_algorithm": "sha256-first-64-bits/v1",
                "generation_protocol": "explicit-greedy-generation/v1",
            },
        },
    )
    return path


class _FixedRuntime:
    num_layers = 12

    def __init__(self, correct: Mapping[str, bool]) -> None:
        self.correct = dict(correct)

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "fixed-fixture",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "decoder_adapter": "fixture.layers",
            "num_decoder_layers": 12,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "generation": {
                "do_sample": False,
                "max_new_tokens": 512,
                "use_cache": True,
                "padding_side": "left",
            },
            "answer_extraction": "primary-then-empty-only-fallback/v1",
        }

    def regenerate_baseline(self, pair: dict[str, object]) -> BaselineScan:
        return BaselineScan(
            sample_id=str(pair["sample_id"]),
            clean=_answer("2", token=20),
            edited=_answer("3", token=30),
        )

    def scan_pair(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        windows: tuple[LayerWindow, ...],
        directions: tuple[str, ...],
    ) -> PairWindowScan:
        value = "2" if self.correct[str(pair["sample_id"])] else "3"
        return PairWindowScan(
            sample_id=str(pair["sample_id"]),
            directions={
                "clean-to-edited": DirectionWindowScan(
                    patched_by_window={windows[0].label: _answer(value, token=100)}
                )
            },
        )


def _fixed_reference(
    tmp_path: Path,
    *,
    sample_ids: tuple[str, ...] = ("a", "b", "c", "d"),
    correct_events: tuple[bool, ...] = (True, True, False, False),
    pairs: list[dict[str, object]] | None = None,
) -> Path:
    pairs = pairs or [_pair(sample_id) for sample_id in sample_ids]
    source = _write_pair_source(tmp_path / "prepared", pairs)
    output = tmp_path / "fixed"
    run_fixed_window_answer_patching(
        FixedWindowAnswerPatchingConfig(
            model="test/model",
            benchmark="gsm8k",
            pairs=(source,),
            layers=(LayerWindow(0, 6),),
            directions=("clean-to-edited",),
            output_dir=output,
        ),
        runtime=_FixedRuntime(dict(zip(sample_ids, correct_events, strict=True))),
    )
    return output


class _ControlRuntime:
    num_layers = 12

    def __init__(
        self,
        *,
        answers: Mapping[str, Mapping[str, str]] | None = None,
        baseline_mismatch_for: str | None = None,
        failure_for: str | None = None,
    ) -> None:
        self.answers = {sample_id: dict(values) for sample_id, values in (answers or {}).items()}
        self.baseline_mismatch_for = baseline_mismatch_for
        self.failure_for = failure_for
        self.baseline_calls: list[str] = []
        self.control_calls: list[tuple[str, str | None, tuple[str, ...]]] = []

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "coordinate-fixture",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "decoder_adapter": "fixture.layers",
            "num_decoder_layers": 12,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "generation": {
                "do_sample": False,
                "max_new_tokens": 512,
                "use_cache": True,
                "padding_side": "left",
            },
            "answer_extraction": "primary-then-empty-only-fallback/v1",
        }

    def regenerate_baseline(self, pair: dict[str, object]) -> BaselineScan:
        sample_id = str(pair["sample_id"])
        self.baseline_calls.append(sample_id)
        edited_token = 31 if sample_id == self.baseline_mismatch_for else 30
        return BaselineScan(
            sample_id=sample_id,
            clean=_answer("2", token=20),
            edited=_answer("3", token=edited_token),
        )

    def scan_controls(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        donor_pair: dict[str, object] | None,
        window: LayerWindow,
        controls: tuple[str, ...],
    ) -> ControlScan:
        sample_id = str(pair["sample_id"])
        donor_id = None if donor_pair is None else str(donor_pair["sample_id"])
        self.control_calls.append((sample_id, donor_id, controls))
        if sample_id == self.failure_for:
            raise RuntimeError("synthetic coordinate GPU failure")
        defaults = {"offset-2": "3", "cross-item": "3", "self-copy": "3"}
        defaults.update(self.answers.get(sample_id, {}))
        generations = {
            control: baseline.edited
            if control == "self-copy" and defaults[control] == "3"
            else _answer(defaults[control], token=40 + CONTROL_NAMES.index(control))
            for control in controls
        }
        return ControlScan(sample_id=sample_id, generations=generations)


def _config(reference: Path, output: Path, **changes: object) -> CoordinateControlConfig:
    config = CoordinateControlConfig(
        model="test/model",
        benchmark="gsm8k",
        fixed_window_run=reference,
        layers=(LayerWindow(0, 6),),
        controls=CONTROL_NAMES,
        output_dir=output,
    )
    return replace(config, **changes)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_catalog_and_cli_expose_the_completed_coordinate_control_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = get_experiment("patch-coordinate-controls")
    assert spec.status == "implemented"
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--fixed-window-run",
        "--layers",
        "--controls",
        "--output-dir",
    )
    assert spec.outputs == (
        "coordinate_control_records.jsonl",
        "pair_status_records.jsonl",
        "coordinate_control_summary.json",
        "run.json",
    )

    captured: list[CoordinateControlConfig] = []

    def fake_run(config: CoordinateControlConfig) -> CoordinateControlResult:
        captured.append(config)
        return CoordinateControlResult(
            records_path=config.output_dir / "coordinate_control_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "coordinate_control_summary.json",
            run_path=config.output_dir / "run.json",
            pairs=1,
            records=4,
        )

    monkeypatch.setattr(cli_module, "run_patch_coordinate_controls", fake_run)
    assert (
        main(
            [
                "patch-coordinate-controls",
                "--model",
                "test/model",
                "--benchmark",
                "gsm8k",
                "--fixed-window-run",
                "results/fixed",
                "--layers",
                "0:6",
                "--controls",
                "cross-item",
                "correct",
                "self-copy",
                "offset-2",
                "--gpu-id",
                "0",
                "--limit",
                "1",
                "--output-dir",
                "results/controls",
                "--resume",
            ]
        )
        == 0
    )
    assert captured == [
        CoordinateControlConfig(
            model="test/model",
            benchmark="gsm8k",
            fixed_window_run=Path("results/fixed"),
            layers=(LayerWindow(0, 6),),
            controls=CONTROL_NAMES,
            output_dir=Path("results/controls"),
            gpu_id="0",
            limit=1,
            resume=True,
        )
    ]
    assert "wrote 4 coordinate-control record(s) for 1 pair(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"benchmark": "mmlu"}, "benchmark must be gsm8k"),
        ({"layers": ()}, "exactly one"),
        ({"layers": (LayerWindow(0, 6), LayerWindow(6, 12))}, "exactly one"),
        ({"layers": (LayerWindow(1, 7),)}, "paper window 0:6"),
        ({"controls": ()}, "at least one"),
        ({"controls": ("correct", "correct")}, "duplicates"),
        ({"controls": ("correct", "unknown")}, "unsupported control"),
        ({"controls": ("offset-2",)}, "must include correct"),
        ({"controls": ("correct",)}, "at least one diagnostic"),
        ({"gpu_id": "0,1"}, "single non-negative integer"),
        ({"limit": 0}, "positive integer"),
    ),
)
def test_config_rejects_non_paper_or_ambiguous_arguments(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "fixed", tmp_path / "out", **changes)


def test_offset_plan_moves_both_sides_and_drops_invalid_or_edited_endpoints() -> None:
    plan = plan_offset_positions(
        clean_positions=(2, 3, 6, 8),
        edited_positions=(3, 4, 7, 8),
        clean_prompt_length=12,
        edited_prompt_length=12,
        offset=2,
    )
    assert plan.source_positions == (4, 5, 10)
    assert plan.destination_positions == (5, 6, 10)
    assert plan.input_pairs == 4
    assert plan.dropped_pairs == 1

    with pytest.raises(ValueError, match="same cardinality"):
        plan_offset_positions((1,), (1, 2), 8, 8)
    with pytest.raises(ValueError, match="non-zero"):
        plan_offset_positions((1,), (1,), 8, 8, offset=0)


def test_donor_assignment_is_order_invariant_nonself_and_stratum_matched() -> None:
    items = (
        DonorItem("random-4", "r2", 2),
        DonorItem("attribution-4", "a2", 1),
        DonorItem("random-4", "r1", 2),
        DonorItem("attribution-4", "a1", 1),
    )
    donors = assign_cross_item_donors(items)
    assert donors == {
        ("attribution-4", "a1"): "a2",
        ("attribution-4", "a2"): "a1",
        ("random-4", "r1"): "r2",
        ("random-4", "r2"): "r1",
    }
    assert assign_cross_item_donors(reversed(items)) == donors
    with pytest.raises(ValueError, match="singleton"):
        assign_cross_item_donors((DonorItem("attribution-4", "only", 1),))


def test_exact_mcnemar_reproduces_the_published_discordant_tables() -> None:
    offset = exact_mcnemar(
        (True,) * 37 + (True,) * 92 + (False,) * 7 + (False,) * 36,
        (True,) * 37 + (False,) * 92 + (True,) * 7 + (False,) * 36,
    )
    assert offset["left_only"] == 92
    assert offset["right_only"] == 7
    assert offset["discordant"] == 99
    assert offset["p_value"] == pytest.approx(5.0749031687767575e-20)

    cross = exact_mcnemar(
        (True,) * 31 + (True,) * 98 + (False,) * 11 + (False,) * 32,
        (True,) * 31 + (False,) * 98 + (True,) * 11 + (False,) * 32,
    )
    assert cross["left_only"] == 98
    assert cross["right_only"] == 11
    assert cross["p_value"] == pytest.approx(1.3281749873159826e-18)


def test_runner_preserves_one_denominator_and_reports_paired_events(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _ControlRuntime(
        answers={
            "a": {"offset-2": "2", "cross-item": "3"},
            "b": {"offset-2": "3", "cross-item": "2"},
            "c": {"offset-2": "3", "cross-item": "3"},
            "d": {"offset-2": "3", "cross-item": "3"},
        }
    )
    result = run_patch_coordinate_controls(
        _config(reference, tmp_path / "controls"),
        runtime=runtime,
    )

    assert result.pairs == 4
    assert result.records == 16
    assert runtime.baseline_calls == ["a", "b", "c", "d"]
    assert [call[0] for call in runtime.control_calls] == ["a", "b", "c", "d"]
    rows = _read_jsonl(result.records_path)
    assert [(row["sample_id"], row["control"]) for row in rows] == [
        (sample_id, control) for sample_id in ("a", "b", "c", "d") for control in CONTROL_NAMES
    ]
    assert [row["event"] for row in rows if row["control"] == "correct"] == [
        True,
        True,
        False,
        False,
    ]
    assert all(row["denominator"] == "fixed-window-clean-to-edited" for row in rows)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["population"]["reference_pairs"] == 4
    assert summary["population"]["executed_pairs"] == 4
    assert summary["controls"]["correct"]["successes"] == 2
    assert summary["controls"]["offset-2"]["successes"] == 1
    assert summary["controls"]["cross-item"]["successes"] == 1
    assert summary["self_copy_integrity"]["token_identical"] == 4
    assert summary["paired_exact_mcnemar"]["correct-vs-offset-2"]["pairs"] == 4
    assert summary["historical_reference"]["correct"] == {
        "successes": 129,
        "total": 172,
        "rate": 0.75,
    }
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["paper_sha256"] == PAPER_SHA256
    assert run["reference"]["historical_cohort_identity"] is False


def test_limit_is_applied_after_the_full_reference_donor_map(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _ControlRuntime()
    result = run_patch_coordinate_controls(
        _config(reference, tmp_path / "controls", limit=1),
        runtime=runtime,
    )
    assert result.pairs == 1
    assert runtime.baseline_calls == ["a"]
    assert runtime.control_calls == [("a", "b", ("offset-2", "cross-item", "self-copy"))]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["comparability"]["status"] == "partial-smoke-run"
    assert run["counts"]["reference_pairs"] == 4


@pytest.mark.parametrize("target", ("fixed-output", "prepared-source"))
def test_reference_or_source_tampering_fails_before_runtime(
    tmp_path: Path,
    target: str,
) -> None:
    reference = _fixed_reference(tmp_path)
    if target == "fixed-output":
        (reference / "fixed_window_records.jsonl").write_text("tampered\n", encoding="utf-8")
    else:
        source_path = Path(
            json.loads((reference / "run.json").read_text(encoding="utf-8"))["arguments"]["pairs"][
                0
            ]
        )
        source_path.write_text(source_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    runtime = _ControlRuntime()
    with pytest.raises((ValueError, CoordinateControlRunError), match="hash|SHA-256"):
        run_patch_coordinate_controls(_config(reference, tmp_path / "controls"), runtime=runtime)
    assert runtime.baseline_calls == []
    assert runtime.control_calls == []


def test_all_baselines_must_match_before_any_control_generation(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _ControlRuntime(baseline_mismatch_for="c")
    with pytest.raises(CoordinateControlRunError, match="baseline replay"):
        run_patch_coordinate_controls(_config(reference, tmp_path / "controls"), runtime=runtime)
    assert runtime.baseline_calls == ["a", "b", "c", "d"]
    assert runtime.control_calls == []
    assert not (tmp_path / "controls" / "coordinate_control_records.jsonl").exists()


def test_self_copy_token_mismatch_is_a_hard_integrity_failure(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _ControlRuntime(answers={"b": {"self-copy": "2"}})
    with pytest.raises(CoordinateControlRunError, match="self-copy.*token-identical"):
        run_patch_coordinate_controls(_config(reference, tmp_path / "controls"), runtime=runtime)
    assert not (tmp_path / "controls" / "coordinate_control_summary.json").exists()


def test_unextractable_control_answer_is_a_retained_failure(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _ControlRuntime(answers={"a": {"offset-2": ""}})
    result = run_patch_coordinate_controls(
        _config(reference, tmp_path / "controls", controls=("correct", "offset-2")),
        runtime=runtime,
    )
    rows = _read_jsonl(result.records_path)
    row = next(row for row in rows if row["sample_id"] == "a" and row["control"] == "offset-2")
    assert row["patched_answer"]["is_extracted"] is False
    assert row["event"] is False
    assert len(rows) == 8


def test_failed_run_keeps_checkpoints_and_resume_reuses_verified_baselines(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "controls"
    config = _config(reference, output)
    first = _ControlRuntime(failure_for="c")
    with pytest.raises(CoordinateControlRunError, match="synthetic coordinate GPU failure"):
        run_patch_coordinate_controls(config, runtime=first)
    assert (output / "run.json").is_file()
    assert (output / "checkpoints" / "baselines").is_dir()
    assert not (output / "coordinate_control_records.jsonl").exists()

    resumed_runtime = _ControlRuntime()
    result = run_patch_coordinate_controls(
        replace(config, resume=True),
        runtime=resumed_runtime,
    )
    assert result.records == 16
    assert resumed_runtime.baseline_calls == []
    assert [call[0] for call in resumed_runtime.control_calls] == ["c", "d"]


def test_completed_resume_validates_hashes_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "controls"
    config = _config(reference, output)
    expected = run_patch_coordinate_controls(config, runtime=_ControlRuntime())

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed resume must not load model weights")

    monkeypatch.setattr(
        "typo_cot.experiments.patch_coordinate_controls.runner.HuggingFaceCoordinateControlRuntime",
        forbidden_factory,
    )
    assert run_patch_coordinate_controls(replace(config, resume=True)) == expected
    expected.records_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CoordinateControlRunError, match="output.*hash"):
        run_patch_coordinate_controls(replace(config, resume=True))
