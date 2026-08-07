"""Final-paper contract tests for the Table 2 patch/text crossing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
import typo_cot.experiments.patch_text_combination.runner as combination_runner
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
from typo_cot.experiments.patch_text_combination import (
    CELL_ORDER,
    CompleteTextInputUse,
    CompleteTextScan,
    PatchTextCombinationConfig,
    PatchTextCombinationResult,
    PatchTextCombinationRunError,
    locate_complete_pre_answer,
    run_patch_text_combination,
)
from typo_cot.experiments.patch_text_combination.runtime import (
    HuggingFacePatchTextCombinationRuntime,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
    start = index * 5
    return {
        "word_index": index,
        "clean_text": f"word{index}",
        "edited_text": f"wrod{index}",
        "clean_editable_span": {"start": start, "end": start + 4},
        "edited_editable_span": {"start": start, "end": start + 4},
        "clean_prompt_span": {"start": start, "end": start + 4},
        "edited_prompt_span": {"start": start, "end": start + 4},
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
    clean_continuation: str = "Reasoning gives 2.\nThe answer is 2.",
) -> dict[str, object]:
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": "test/model",
        "benchmark": "gsm8k",
        "targeting": targeting,
        "seed": 42,
        "num_edits_requested": 4,
        "num_aligned_words": 1,
        "gold_answer": "2",
        "clean": {
            "prompt": f"clean {sample_id}",
            "prompt_token_count": 8,
            "continuation": clean_continuation,
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
            "prompt_token_count": 8,
            "continuation": "Wrong reasoning.\nThe answer is 3.",
            "answer": {
                "value": "3",
                "is_extracted": True,
                "is_correct": False,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "answer_changed": True,
        "aligned_words": [_aligned_word(0, clean_final=2, edited_final=2)],
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
            clean=replace(
                _answer("2", token=20),
                text=str(pair["clean"]["continuation"]),  # type: ignore[index]
            ),
            edited=replace(
                _answer("3", token=30),
                text=str(pair["edited"]["continuation"]),  # type: ignore[index]
            ),
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
    pairs: list[dict[str, object]] | None = None,
    correct_events: Mapping[str, bool] | None = None,
) -> Path:
    pairs = pairs or [_pair(sample_id) for sample_id in ("a", "b", "c")]
    correct_events = correct_events or {"a": True, "b": False, "c": True}
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
        runtime=_FixedRuntime(correct_events),
    )
    return output


class _CombinationRuntime:
    num_layers = 12

    def __init__(
        self,
        *,
        answers: Mapping[str, tuple[str, str]] | None = None,
        baseline_mismatch_for: str | None = None,
        failure_for: str | None = None,
        num_layers: int = 12,
        provenance_changes: Mapping[str, object] | None = None,
        coordinate_mismatch_for: str | None = None,
    ) -> None:
        self.answers = dict(answers or {})
        self.baseline_mismatch_for = baseline_mismatch_for
        self.failure_for = failure_for
        self.num_layers = num_layers
        self.provenance_changes = dict(provenance_changes or {})
        self.coordinate_mismatch_for = coordinate_mismatch_for
        self.baseline_calls: list[str] = []
        self.complete_calls: list[tuple[str, str, str]] = []

    def provenance(self) -> dict[str, object]:
        payload = {
            "runtime": "patch-text-fixture",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "decoder_adapter": "fixture.layers",
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "generation": {
                "do_sample": False,
                "max_new_tokens": 512,
                "use_cache": True,
                "padding_side": "left",
            },
            "answer_extraction": "primary-then-empty-only-fallback/v1",
            "text_intervention": {
                "source": "prepared-clean-continuation",
                "boundary": "first-[Tt]he-answer-is-or-entire-continuation/v1",
                "recipient": "edited-prompt-plus-complete-clean-pre-answer-text",
                "donor": "clean-question-only-prompt",
                "tokenization": "single-concatenated-text-call-with-prompt-prefix-check",
            },
        }
        payload.update(self.provenance_changes)
        return payload

    def regenerate_baseline(self, pair: dict[str, object]) -> BaselineScan:
        sample_id = str(pair["sample_id"])
        self.baseline_calls.append(sample_id)
        edited_token = 31 if sample_id == self.baseline_mismatch_for else 30
        return BaselineScan(
            sample_id=sample_id,
            clean=replace(
                _answer("2", token=20),
                text=str(pair["clean"]["continuation"]),  # type: ignore[index]
            ),
            edited=replace(
                _answer("3", token=edited_token),
                text=str(pair["edited"]["continuation"]),  # type: ignore[index]
            ),
        )

    def scan_complete_text(
        self,
        pair: dict[str, object],
        pre_answer_text: str,
        window: LayerWindow,
    ) -> CompleteTextScan:
        sample_id = str(pair["sample_id"])
        self.complete_calls.append((sample_id, pre_answer_text, window.label))
        if sample_id == self.failure_for:
            raise RuntimeError("synthetic complete-text GPU failure")
        no_patch_value, patch_value = self.answers.get(sample_id, ("2", "2"))
        destination = (999,) if sample_id == self.coordinate_mismatch_for else (2,)
        return CompleteTextScan(
            sample_id=sample_id,
            input_use=CompleteTextInputUse(
                pre_answer_text_sha256=hashlib.sha256(pre_answer_text.encode("utf-8")).hexdigest(),
                pre_answer_char_count=len(pre_answer_text),
                pre_answer_token_count=4,
                edited_prompt_token_count=8,
                full_input_token_count=12,
                full_input_ids_sha256="f" * 64,
                clean_positions=(2,),
                edited_positions=destination,
                boundary_stable=True,
            ),
            no_patch=_answer(no_patch_value, token=40),
            fixed_window_patch=_answer(patch_value, token=50),
        )


def _config(reference: Path, output: Path, **changes: object) -> PatchTextCombinationConfig:
    config = PatchTextCombinationConfig(
        model="test/model",
        benchmark="gsm8k",
        fixed_window_run=reference,
        layers=(LayerWindow(0, 6),),
        output_dir=output,
    )
    return replace(config, **changes)


def test_catalog_and_cli_expose_the_completed_patch_text_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = get_experiment("patch-text-combination")
    assert spec.status == "implemented"
    assert spec.paper_question == "RQ1/RQ3 descriptive comparison"
    assert spec.paper_sections == ("§3.5", "§4.1", "Table 2", "Appendix D")
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--fixed-window-run",
        "--layers",
        "--output-dir",
    )
    assert spec.outputs == (
        "patch_text_records.jsonl",
        "pair_status_records.jsonl",
        "patch_text_summary.json",
        "run.json",
    )

    captured: list[PatchTextCombinationConfig] = []

    def fake_run(config: PatchTextCombinationConfig) -> PatchTextCombinationResult:
        captured.append(config)
        return PatchTextCombinationResult(
            records_path=config.output_dir / "patch_text_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "patch_text_summary.json",
            run_path=config.output_dir / "run.json",
            pairs=1,
            records=4,
        )

    monkeypatch.setattr(cli_module, "run_patch_text_combination", fake_run)
    assert (
        main(
            [
                "patch-text-combination",
                "--model",
                "test/model",
                "--benchmark",
                "gsm8k",
                "--fixed-window-run",
                "results/fixed",
                "--layers",
                "0:6",
                "--gpu-id",
                "0",
                "--limit",
                "1",
                "--output-dir",
                "results/patch-text",
                "--resume",
            ]
        )
        == 0
    )
    assert captured == [
        PatchTextCombinationConfig(
            model="test/model",
            benchmark="gsm8k",
            fixed_window_run=Path("results/fixed"),
            layers=(LayerWindow(0, 6),),
            output_dir=Path("results/patch-text"),
            gpu_id="0",
            limit=1,
            resume=True,
        )
    ]
    assert "wrote 4 patch/text cell record(s) for 1 pair(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"model": ""}, "model must not be empty"),
        ({"benchmark": "mmlu"}, "benchmark must be gsm8k"),
        ({"layers": ()}, "exactly one"),
        ({"layers": (LayerWindow(0, 6), LayerWindow(6, 12))}, "exactly one"),
        ({"layers": (LayerWindow(1, 7),)}, "paper window 0:6"),
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


@pytest.mark.parametrize(
    ("continuation", "text", "found", "count", "start", "residual"),
    (
        ("Steps.\nThe answer is 2.", "Steps.\n", True, 1, 7, False),
        ("Steps. the answer is 2.", "Steps. ", True, 1, 7, False),
        ("The answer is 1. The answer is 2.", "", True, 2, 0, False),
        ("No canonical trigger.", "No canonical trigger.", False, 0, None, False),
        ("Answer: maybe. The answer is 2.", "Answer: maybe. ", True, 1, 15, True),
    ),
)
def test_complete_pre_answer_locator_reproduces_the_legacy_boundary_without_exclusion(
    continuation: str,
    text: str,
    found: bool,
    count: int,
    start: int | None,
    residual: bool,
) -> None:
    result = locate_complete_pre_answer(continuation)
    assert result.text == text
    assert result.trigger_found is found
    assert result.trigger_count == count
    assert result.trigger_char_start == start
    assert result.residual_fragment is residual
    assert result.method == "first-[Tt]he-answer-is-or-entire-continuation/v1"


def test_runner_emits_four_ordered_gold_correctness_cells_on_one_denominator(
    tmp_path: Path,
) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _CombinationRuntime(answers={"a": ("2", "2"), "b": ("2", "3"), "c": ("", "2")})
    result = run_patch_text_combination(
        _config(reference, tmp_path / "combined"),
        runtime=runtime,
    )

    assert result.pairs == 3
    assert result.records == 12
    assert runtime.baseline_calls == ["a", "b", "c"]
    assert [call[0] for call in runtime.complete_calls] == ["a", "b", "c"]
    rows = _read_jsonl(result.records_path)
    assert [(row["sample_id"], row["cell"]) for row in rows] == [
        (sample_id, cell) for sample_id in ("a", "b", "c") for cell in CELL_ORDER
    ]
    assert all(row["denominator"] == "fixed-window-clean-to-edited" for row in rows)
    assert [row["event"] for row in rows if row["cell"] == CELL_ORDER[0]] == [
        False,
        False,
        False,
    ]
    assert [row["event"] for row in rows if row["cell"] == CELL_ORDER[1]] == [
        True,
        False,
        True,
    ]
    assert [row["event"] for row in rows if row["cell"] == CELL_ORDER[2]] == [
        True,
        True,
        False,
    ]
    assert [row["event"] for row in rows if row["cell"] == CELL_ORDER[3]] == [
        True,
        False,
        True,
    ]
    assert {row["result_source"] for row in rows[:2]} == {"fixed-window-reference"}
    assert {row["result_source"] for row in rows[2:4]} == {"patch-text-runtime"}

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert [cell["cell"] for cell in summary["cells"]] == list(CELL_ORDER)
    assert [cell["successes"] for cell in summary["cells"]] == [0, 2, 2, 2]
    assert all(cell["total"] == 3 for cell in summary["cells"])
    assert summary["historical_reference"]["total"] == 172
    assert [cell["successes"] for cell in summary["historical_reference"]["cells"]] == [
        0,
        129,
        168,
        171,
    ]
    serialized = json.dumps(summary, sort_keys=True).lower()
    for forbidden in ("p_value", "confidence_interval", "mcnemar", "mediation", "interaction"):
        assert forbidden not in serialized

    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["paper_sha256"] == PAPER_SHA256
    assert run["status"] == "completed"
    assert run["reference"]["historical_cohort_identity"] is False
    assert not (result.run_path.parent / ".patch-text-combination-work").exists()


def test_no_trigger_and_unextractable_complete_text_are_retained_failures(tmp_path: Path) -> None:
    pairs = [_pair("a", clean_continuation="Reasoning without the canonical phrase.\n2")]
    reference = _fixed_reference(
        tmp_path,
        pairs=pairs,
        correct_events={"a": False},
    )
    result = run_patch_text_combination(
        _config(reference, tmp_path / "combined"),
        runtime=_CombinationRuntime(answers={"a": ("", "")}),
    )
    rows = _read_jsonl(result.records_path)
    assert len(rows) == 4
    complete_rows = [row for row in rows if row["clean_text"] == "complete"]
    assert all(row["event"] is False for row in complete_rows)
    assert all(row["answer"]["is_extracted"] is False for row in complete_rows)
    statuses = _read_jsonl(result.pair_status_records_path)
    assert statuses[0]["complete_text"]["trigger_found"] is False
    assert statuses[0]["complete_text"]["text"] == pairs[0]["clean"]["continuation"]


def test_limit_is_applied_after_full_reference_planning_and_statuses_keep_the_denominator(
    tmp_path: Path,
) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _CombinationRuntime()
    result = run_patch_text_combination(
        _config(reference, tmp_path / "combined", limit=1),
        runtime=runtime,
    )
    assert result.pairs == 1
    assert result.records == 4
    assert runtime.baseline_calls == ["a"]
    statuses = _read_jsonl(result.pair_status_records_path)
    assert len(statuses) == 3
    assert [row["selected_for_execution"] for row in statuses] == [True, False, False]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["counts"]["reference_pairs"] == 3
    assert run["counts"]["executed_pairs"] == 1
    assert run["comparability"]["status"] == "partial-smoke-run"


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
    runtime = _CombinationRuntime()
    with pytest.raises((ValueError, PatchTextCombinationRunError), match="hash|SHA-256"):
        run_patch_text_combination(
            _config(reference, tmp_path / "combined"),
            runtime=runtime,
        )
    assert runtime.baseline_calls == []
    assert runtime.complete_calls == []


@pytest.mark.parametrize(
    ("runtime", "message"),
    (
        (_CombinationRuntime(provenance_changes={"torch": "different"}), "provenance differs"),
        (_CombinationRuntime(num_layers=13), "decoder depth"),
        (
            _CombinationRuntime(
                provenance_changes={
                    "text_intervention": {
                        "source": "prepared-clean-continuation",
                        "boundary": "first-[Tt]he-answer-is-or-entire-continuation/v1",
                        "recipient": "edited-prompt-plus-complete-clean-pre-answer-text",
                        "donor": "clean-question-plus-answer-text",
                        "tokenization": ("single-concatenated-text-call-with-prompt-prefix-check"),
                    }
                }
            ),
            "public protocol",
        ),
    ),
)
def test_runtime_must_match_the_reference_before_any_generation(
    tmp_path: Path,
    runtime: _CombinationRuntime,
    message: str,
) -> None:
    reference = _fixed_reference(tmp_path)
    with pytest.raises(ValueError, match=message):
        run_patch_text_combination(
            _config(reference, tmp_path / "combined"),
            runtime=runtime,
        )
    assert runtime.baseline_calls == []
    assert runtime.complete_calls == []


def test_every_baseline_must_match_before_any_complete_text_generation(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _CombinationRuntime(baseline_mismatch_for="b")
    with pytest.raises(PatchTextCombinationRunError, match="baseline replay"):
        run_patch_text_combination(
            _config(reference, tmp_path / "combined"),
            runtime=runtime,
        )
    assert runtime.baseline_calls == ["a", "b", "c"]
    assert runtime.complete_calls == []
    assert not (tmp_path / "combined" / "patch_text_records.jsonl").exists()


def test_runtime_coordinates_and_text_fingerprint_must_match_the_pure_plan(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    runtime = _CombinationRuntime(coordinate_mismatch_for="a")
    with pytest.raises(PatchTextCombinationRunError, match="coordinates do not match"):
        run_patch_text_combination(
            _config(reference, tmp_path / "combined"),
            runtime=runtime,
        )
    assert [call[0] for call in runtime.complete_calls] == ["a"]
    assert not (tmp_path / "combined" / "patch_text_records.jsonl").exists()


def test_failed_run_keeps_pair_atomic_checkpoints_and_resume_runs_only_outstanding_pairs(
    tmp_path: Path,
) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "combined"
    config = _config(reference, output)
    first = _CombinationRuntime(failure_for="b")
    with pytest.raises(PatchTextCombinationRunError, match="synthetic complete-text GPU failure"):
        run_patch_text_combination(config, runtime=first)
    assert first.baseline_calls == ["a", "b", "c"]
    assert [call[0] for call in first.complete_calls] == ["a", "b"]
    assert not (output / "patch_text_records.jsonl").exists()
    assert len(list((output / ".patch-text-combination-work" / "baselines").glob("*.json"))) == 3
    assert (
        len(list((output / ".patch-text-combination-work" / "complete-text").glob("*.json"))) == 1
    )

    resumed = _CombinationRuntime()
    result = run_patch_text_combination(replace(config, resume=True), runtime=resumed)
    assert result.records == 12
    assert resumed.baseline_calls == []
    assert [call[0] for call in resumed.complete_calls] == ["b", "c"]


def test_registered_checkpoint_tampering_is_rejected_on_resume(tmp_path: Path) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "combined"
    config = _config(reference, output)
    with pytest.raises(PatchTextCombinationRunError, match="synthetic complete-text GPU failure"):
        run_patch_text_combination(config, runtime=_CombinationRuntime(failure_for="b"))
    checkpoint = next((output / ".patch-text-combination-work" / "complete-text").glob("*.json"))
    checkpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises((ValueError, PatchTextCombinationRunError), match="checkpoint.*hash"):
        run_patch_text_combination(
            replace(config, resume=True),
            runtime=_CombinationRuntime(),
        )


def test_completed_resume_validates_outputs_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "combined"
    config = _config(reference, output)
    expected = run_patch_text_combination(config, runtime=_CombinationRuntime())

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed resume must not load model weights")

    monkeypatch.setattr(
        "typo_cot.experiments.patch_text_combination.runner.HuggingFacePatchTextCombinationRuntime",
        forbidden_factory,
    )
    assert run_patch_text_combination(replace(config, resume=True)) == expected
    expected.records_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(PatchTextCombinationRunError, match="output.*hash"):
        run_patch_text_combination(replace(config, resume=True))


@pytest.mark.parametrize(
    "tamper",
    ("runtime", "checkpoints", "checkpoint-sha", "counts"),
)
def test_completed_resume_validates_manifest_state_without_loading_model(
    tmp_path: Path,
    tamper: str,
) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "combined"
    config = _config(reference, output)
    run_patch_text_combination(config, runtime=_CombinationRuntime())
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if tamper == "runtime":
        run["runtime"] = {}
    elif tamper == "checkpoints":
        run["checkpoints"] = {"baselines": {}, "complete_text": {}}
    elif tamper == "checkpoint-sha":
        first = next(iter(run["checkpoints"]["complete_text"].values()))
        first["sha256"] = "0" * 64
    else:
        run["counts"]["failed_pairs"] = 1
    _write_json(run_path, run)

    with pytest.raises(PatchTextCombinationRunError, match="completed.*validation"):
        run_patch_text_combination(replace(config, resume=True))


def test_completed_resume_rejects_semantic_output_tampering_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "combined"
    config = _config(reference, output)
    result = run_patch_text_combination(config, runtime=_CombinationRuntime())
    rows = _read_jsonl(result.records_path)
    rows[0]["event"] = not rows[0]["event"]
    result.records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    run["outputs"][result.records_path.name]["sha256"] = hashlib.sha256(
        result.records_path.read_bytes()
    ).hexdigest()
    _write_json(result.run_path, run)

    with pytest.raises(PatchTextCombinationRunError, match="completed.*validation"):
        run_patch_text_combination(replace(config, resume=True))


def test_checkpoint_cleanup_failure_does_not_destroy_completed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _fixed_reference(tmp_path)
    output = tmp_path / "combined"

    def fail_cleanup(_path: Path) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(combination_runner.shutil, "rmtree", fail_cleanup)
    result = run_patch_text_combination(
        _config(reference, output),
        runtime=_CombinationRuntime(),
    )

    assert result.records_path.is_file()
    assert json.loads(result.run_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert (output / ".patch-text-combination-work").is_dir()


def test_boundary_diagnostics_retain_all_pairs_but_prevent_a_fresh_label(tmp_path: Path) -> None:
    pairs = [
        _pair("no-trigger", clean_continuation="Reasoning reaches 2."),
        _pair(
            "multi-trigger",
            clean_continuation="Reasoning. The answer is discussed. The answer is 2.",
        ),
        _pair("empty", clean_continuation="The answer is 2."),
        _pair("residual", clean_continuation="Long answer: 2.\nThe answer is 2."),
    ]
    reference = _fixed_reference(
        tmp_path,
        pairs=pairs,
        correct_events={pair["sample_id"]: True for pair in pairs},
    )
    result = run_patch_text_combination(
        _config(reference, tmp_path / "combined"),
        runtime=_CombinationRuntime(),
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    diagnostics = summary["complete_text_boundary"]["executed_diagnostics"]
    assert diagnostics == {
        "total_pairs": 4,
        "no_trigger_pairs": 1,
        "multiple_trigger_pairs": 1,
        "empty_text_pairs": 1,
        "residual_fragment_pairs": 1,
        "anomalous_pairs": 4,
    }
    assert len(_read_jsonl(result.pair_status_records_path)) == 4
    assert (
        "legacy-backed-complete-text-boundary-has-diagnostic-anomalies"
        in summary["comparability"]["limitations"]
    )


def _comparability_fixture(
    tmp_path: Path,
    *,
    source_status: str = "fresh-paper-protocol-run",
    source_limit: int | None = None,
    targetings: tuple[str, ...] | None = None,
    continuation: str = "Reasoning.\nThe answer is 2.",
    baseline_text: str | None = None,
) -> tuple[PatchTextCombinationConfig, object, tuple[object, ...]]:
    if targetings is None:
        targetings = tuple(
            "attribution-4" if index % 2 == 0 else "random-4" for index in range(172)
        )
    config = _config(
        tmp_path / "fixed",
        tmp_path / "combined",
        model="google/gemma-3-4b-it",
    )
    reference = SimpleNamespace(
        manifest={
            "comparability": {"status": source_status},
            "arguments": {"limit": source_limit},
        }
    )
    plans = tuple(
        SimpleNamespace(
            key=(targeting, f"sample-{index}"),
            complete_text=locate_complete_pre_answer(continuation),
            reference=SimpleNamespace(
                source=SimpleNamespace(record={"clean": {"continuation": continuation}}),
                baseline=SimpleNamespace(
                    clean=SimpleNamespace(
                        text=continuation if baseline_text is None else baseline_text
                    )
                ),
            ),
        )
        for index, targeting in enumerate(targetings)
    )
    return config, reference, plans


def test_comparability_fresh_label_requires_the_entire_paper_contract(tmp_path: Path) -> None:
    config, reference, plans = _comparability_fixture(tmp_path)
    result = combination_runner._comparability(config, reference, plans)
    assert result["status"] == "fresh-paper-protocol-run"
    assert result["limitations"] == []
    assert all(result["requirements"].values()) is False  # Historical identity is never claimed.
    assert result["requirements"]["complete_text_boundary_unambiguous"] is True
    assert result["requirements"]["prepared_clean_continuations_match_reference_baselines"] is True


@pytest.mark.parametrize(
    ("changes", "limitation"),
    (
        (
            {"source_status": "partial-paper-protocol"},
            "fixed-window-reference-is-not-a-full-paper-protocol-run",
        ),
        ({"source_limit": 1}, "fixed-window-reference-is-limit-truncated"),
        (
            {"targetings": ("attribution-4",)},
            "reference-denominator-does-not-contain-both-targeting-arms",
        ),
        (
            {"targetings": ("attribution-4", "random-4")},
            "reference-denominator-is-not-172-pairs",
        ),
        (
            {"continuation": "Reasoning without the literal trigger."},
            "legacy-backed-complete-text-boundary-has-diagnostic-anomalies",
        ),
        (
            {"baseline_text": "Different replayed clean text."},
            "prepared-clean-continuation-differs-from-reference-baseline",
        ),
    ),
)
def test_comparability_downgrades_each_incomplete_contract(
    tmp_path: Path,
    changes: dict[str, object],
    limitation: str,
) -> None:
    config, reference, plans = _comparability_fixture(tmp_path, **changes)  # type: ignore[arg-type]
    result = combination_runner._comparability(config, reference, plans)
    assert result["status"] == "partial-paper-protocol"
    assert limitation in result["limitations"]


def test_production_runtime_uses_one_complete_input_and_the_original_aligned_positions() -> None:
    runtime = object.__new__(HuggingFacePatchTextCombinationRuntime)
    runtime.num_layers = 6
    runtime.layers = [object()] * 6
    runtime._torch = SimpleNamespace(inference_mode=nullcontext)
    runtime._sample_id = lambda pair: str(pair["sample_id"])
    runtime._gold_answer = lambda _pair: "2"

    clean_ids = SimpleNamespace(shape=(1, 5))
    edited_ids = SimpleNamespace(shape=(1, 6))
    full_ids = SimpleNamespace(shape=(1, 10))
    clean_mask = object()
    edited_mask = object()
    full_mask = object()
    captures: list[tuple[object, tuple[int, ...]]] = []
    generated_inputs: list[tuple[object, object | None]] = []

    def tokenize_pair(
        pair: Mapping[str, object], *, side: str
    ) -> tuple[object, object, tuple[int, ...]]:
        if side == "clean":
            return clean_ids, clean_mask, (2,)
        return edited_ids, edited_mask, (3,)

    runtime._tokenize_and_validate = tokenize_pair
    runtime._tokenize_complete_input = lambda _pair, _text, _edited_ids: (
        full_ids,
        full_mask,
        CompleteTextInputUse(
            pre_answer_text_sha256=hashlib.sha256(b"Reasoning").hexdigest(),
            pre_answer_char_count=9,
            pre_answer_token_count=4,
            edited_prompt_token_count=6,
            full_input_token_count=10,
            full_input_ids_sha256="f" * 64,
            clean_positions=(2,),
            edited_positions=(3,),
            boundary_stable=True,
        ),
    )

    def capture(
        *, input_ids: object, attention_mask: object, positions: tuple[int, ...]
    ) -> list[str]:
        captures.append((input_ids, positions))
        return [f"clean-{layer}" for layer in range(6)]

    runtime._capture = capture

    class _Patch:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    runtime._window_patch = lambda **kwargs: (
        generated_inputs.append((kwargs["recipient_ids"], kwargs["positions"])),
        _Patch(),
    )[1]

    def generate(**kwargs: object) -> AnswerGeneration:
        generated_inputs.append((kwargs["input_ids"], kwargs.get("patch")))
        return _answer("2", token=40 + len(generated_inputs))

    runtime._generate = generate
    scan = runtime.scan_complete_text(_pair("a"), "Reasoning", LayerWindow(0, 6))

    assert scan.input_use.clean_positions == (2,)
    assert scan.input_use.edited_positions == (3,)
    assert captures == [(clean_ids, (2,))]
    assert generated_inputs[0] == (full_ids, None)
    assert generated_inputs[1] == (full_ids, (3,))
    assert generated_inputs[2][0] is full_ids


def test_production_runtime_rejects_a_changed_edited_prompt_token_prefix() -> None:
    import torch

    runtime = object.__new__(HuggingFacePatchTextCombinationRuntime)
    runtime._torch = torch
    runtime.device = torch.device("cpu")
    runtime.tokenizer = lambda *_args, **_kwargs: {
        "input_ids": torch.tensor([[1, 99, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    edited_prompt_ids = torch.tensor([[1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match="changed the edited prompt-token prefix"):
        runtime._tokenize_complete_input(_pair("a"), "Reasoning", edited_prompt_ids)
