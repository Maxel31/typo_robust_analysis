"""Final-paper contract tests for the layerwise free-answer scan."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import typo_cot.cli as cli_module
from typo_cot.experiments.layerwise_answer_patching import runner as answer_runner_module
from typo_cot.cli import main
from typo_cot.evaluation.fallback import (
    answers_equal,
    extract_with_fallback,
    fallback_answer,
)
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.layerwise_answer_patching.metrics import (
    summarize_binary_layers,
    wilson_interval,
)
from typo_cot.experiments.layerwise_answer_patching.patching import (
    PrefillBlockOutputPatch,
)
from typo_cot.experiments.layerwise_answer_patching.runner import (
    AnswerGeneration,
    BaselineScan,
    DirectionAnswerScan,
    LayerwiseAnswerPatchingConfig,
    LayerwiseAnswerPatchingResult,
    LayerwiseAnswerPatchingRunError,
    PairAnswerScan,
    run_layerwise_answer_patching,
)
from typo_cot.experiments.layerwise_answer_patching.runtime import (
    HuggingFaceLayerwiseAnswerPatchingRuntime,
    continuation_token_ids,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aligned_word(*, clean_final: int = 1, edited_final: int = 1) -> dict[str, object]:
    return {
        "word_index": 0,
        "clean_text": "alpha",
        "edited_text": "alhpa",
        "clean_editable_span": {"start": 0, "end": 5},
        "edited_editable_span": {"start": 0, "end": 5},
        "clean_prompt_span": {"start": 0, "end": 5},
        "edited_prompt_span": {"start": 0, "end": 5},
        "target_ranks": [1],
        "target_token_indices": [1],
        "clean_token_indices": [clean_final],
        "edited_token_indices": [edited_final],
        "clean_final_token": clean_final,
        "edited_final_token": edited_final,
    }


def _pair(
    sample_id: str,
    *,
    targeting: str,
    clean_correct: bool = True,
    edited_correct: bool = False,
    aligned: bool = True,
) -> dict[str, object]:
    aligned_words = [_aligned_word()] if aligned else []
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": "test/model",
        "benchmark": "gsm8k",
        "targeting": targeting,
        "seed": 42,
        "num_edits_requested": 4,
        "num_aligned_words": len(aligned_words),
        "gold_answer": "2",
        "clean": {
            "prompt": "alpha question",
            "prompt_token_count": 3,
            "continuation": "clean reasoning. The answer is 2.",
            "answer": {
                "value": "2",
                "is_extracted": True,
                "is_correct": clean_correct,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "edited": {
            "prompt": "alhpa question",
            "prompt_token_count": 3,
            "continuation": "edited reasoning. The answer is 3.",
            "answer": {
                "value": "2" if edited_correct else "3",
                "is_extracted": True,
                "is_correct": edited_correct,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "answer_changed": clean_correct and not edited_correct,
        "aligned_words": aligned_words,
    }


def _write_pair_source(
    root: Path,
    pairs: list[dict[str, object]],
    *,
    targeting: str,
    model: str = "test/model",
    benchmark: str = "gsm8k",
    revision: str = "source-revision",
    dataset_sha256: str = "dataset-sha256",
    seed: int = 42,
    num_edits: int = 4,
    limit: int | None = None,
    max_new_tokens: int = 512,
) -> Path:
    root.mkdir(parents=True)
    pairs_path = root / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for pair in sorted(pairs, key=lambda item: str(item["sample_id"])):
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
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
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "gpu_id": "0",
                "limit": limit,
                "output_dir": str(root.resolve()),
            },
            "counts": {
                "discovered": len(pairs),
                "written": len(pairs),
                "failed": 0,
            },
            "failures": [],
            "provenance": {
                "model": model,
                "model_revision": revision,
                "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
                "dataset_sample_count": len(pairs),
                "dataset_records_sha256": dataset_sha256,
                "random_seed_algorithm": "sha256-first-64-bits/v1",
            },
        },
    )
    return pairs_path


def _sources(
    tmp_path: Path,
    *,
    attribution_pairs: list[dict[str, object]] | None = None,
    random_pairs: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    attribution_pairs = attribution_pairs or [_pair("a", targeting="attribution-4")]
    random_pairs = random_pairs or [_pair("b", targeting="random-4")]
    return (
        _write_pair_source(
            tmp_path / "attribution",
            attribution_pairs,
            targeting="attribution-4",
        ),
        _write_pair_source(
            tmp_path / "random",
            random_pairs,
            targeting="random-4",
        ),
    )


def _answer(
    value: str,
    *,
    correct: bool,
    token: int,
    method: str = "primary:fixture",
) -> AnswerGeneration:
    return AnswerGeneration(
        token_ids=(token,),
        text=f"answer {value}" if value else "unextractable",
        value=value,
        is_extracted=bool(value),
        is_correct=correct,
        method=method if value else "unextractable",
        primary_method="fixture" if value else "no_match",
    )


class FakeRuntime:
    def __init__(
        self,
        *,
        n_layers: int = 3,
        baselines: dict[tuple[str, str], BaselineScan] | None = None,
        scans: dict[tuple[str, str], PairAnswerScan] | None = None,
        error_for: tuple[str, str] | None = None,
        requested_revision: str = "source-revision",
        model_revision: str = "source-revision",
        tokenizer_revision: str = "source-revision",
    ) -> None:
        self.num_layers = n_layers
        self.baselines = baselines or {}
        self.scans = scans or {}
        self.error_for = error_for
        self.requested_revision = requested_revision
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.baseline_calls: list[tuple[str, str]] = []
        self.scan_calls: list[tuple[str, str, tuple[str, ...]]] = []

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "fake",
            "requested_revision": self.requested_revision,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "decoder_adapter": "fake.layers",
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
        }

    def regenerate_baseline(self, pair: dict[str, object]) -> BaselineScan:
        key = (str(pair["targeting"]), str(pair["sample_id"]))
        self.baseline_calls.append(key)
        default = BaselineScan(
            sample_id=key[1],
            clean=_answer("2", correct=True, token=20),
            edited=_answer("3", correct=False, token=30),
        )
        return self.baselines.get(key, default)

    def scan_pair(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        directions: tuple[str, ...],
    ) -> PairAnswerScan:
        key = (str(pair["targeting"]), str(pair["sample_id"]))
        self.scan_calls.append((*key, directions))
        if key == self.error_for:
            raise RuntimeError("synthetic GPU failure")
        if key in self.scans:
            return self.scans[key]
        defaults = {
            "clean-to-edited": DirectionAnswerScan(
                patched_by_layer=(
                    _answer("2", correct=True, token=21),
                    _answer("", correct=False, token=22),
                    baseline.edited,
                )[: self.num_layers]
            ),
            "edited-to-clean": DirectionAnswerScan(
                patched_by_layer=(
                    _answer("3", correct=False, token=31),
                    _answer("", correct=False, token=32),
                    baseline.clean,
                )[: self.num_layers]
            ),
        }
        return PairAnswerScan(
            sample_id=key[1],
            directions={name: defaults[name] for name in directions},
        )


def _config(
    attribution_pairs: Path,
    random_pairs: Path,
    output_dir: Path,
    **changes: object,
) -> LayerwiseAnswerPatchingConfig:
    config = LayerwiseAnswerPatchingConfig(
        model="test/model",
        benchmark="gsm8k",
        attribution_pairs=attribution_pairs,
        random_pairs=random_pairs,
        directions=("clean-to-edited", "edited-to-clean"),
        max_pairs=300,
        output_dir=output_dir,
    )
    return replace(config, **changes)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_catalog_marks_layerwise_answer_patching_as_implemented() -> None:
    spec = get_experiment("layerwise-answer-patching")
    assert spec.status == "implemented"
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--attribution-pairs",
        "--random-pairs",
        "--directions",
        "--max-pairs",
        "--output-dir",
    )
    assert spec.outputs == (
        "answer_layer_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
        "run.json",
    )


def test_cli_dispatches_complete_operation_specific_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[LayerwiseAnswerPatchingConfig] = []

    def fake_run(config: LayerwiseAnswerPatchingConfig) -> LayerwiseAnswerPatchingResult:
        captured.append(config)
        return LayerwiseAnswerPatchingResult(
            answer_layer_records_path=config.output_dir / "answer_layer_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "setting_summary.json",
            run_path=config.output_dir / "run.json",
            fixed_pairs=1,
            layer_records=6,
        )

    monkeypatch.setattr(cli_module, "run_layerwise_answer_patching", fake_run)
    assert (
        main(
            [
                "layerwise-answer-patching",
                "--model",
                "test/model",
                "--benchmark",
                "gsm8k",
                "--attribution-pairs",
                "results/a/pairs.jsonl",
                "--random-pairs",
                "results/r/pairs.jsonl",
                "--directions",
                "edited-to-clean",
                "clean-to-edited",
                "--max-pairs",
                "2",
                "--output-dir",
                "results/answer",
                "--gpu-id",
                "0",
                "--resume",
            ]
        )
        == 0
    )
    assert captured == [
        LayerwiseAnswerPatchingConfig(
            model="test/model",
            benchmark="gsm8k",
            attribution_pairs=Path("results/a/pairs.jsonl"),
            random_pairs=Path("results/r/pairs.jsonl"),
            directions=("clean-to-edited", "edited-to-clean"),
            max_pairs=2,
            output_dir=Path("results/answer"),
            gpu_id="0",
            resume=True,
        )
    ]
    assert capsys.readouterr().out.splitlines() == [
        "wrote 6 layer record(s) for 1 fixed pair(s): results/answer/answer_layer_records.jsonl",
        "setting summary: results/answer/setting_summary.json",
        "run manifest: results/answer/run.json",
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"benchmark": "mmlu-pro"}, "supports only gsm8k and mmlu"),
        ({"directions": ()}, "at least one direction"),
        (
            {"directions": ("clean-to-edited", "clean-to-edited")},
            "must not contain duplicates",
        ),
        ({"directions": ("clean-to-typo",)}, "unsupported direction"),
        ({"max_pairs": 0}, "positive even integer"),
        ({"max_pairs": 3}, "positive even integer"),
        ({"max_pairs": 302}, "at most 300"),
        ({"gpu_id": "0,1"}, "single non-negative integer"),
    ),
)
def test_config_rejects_nonpaper_or_ambiguous_arguments(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "a", tmp_path / "r", tmp_path / "out", **changes)


def test_primary_extractor_wins_and_fallback_runs_only_for_empty_results() -> None:
    primary = extract_with_fallback(
        "The answer is 7.\nFinal answer: the total is $40.",
        benchmark="gsm8k",
        correct_answer="7",
    )
    assert primary.value == "7"
    assert primary.method == "primary:pattern_1"
    assert primary.is_correct is True

    rescued = extract_with_fallback(
        "Reasoning complete.\nFinal answer: the total is $26.00.",
        benchmark="gsm8k",
        correct_answer="26",
    )
    assert rescued.value == "26"
    assert rescued.method == "fallback:N2_answer_line"
    assert rescued.primary_method == "no_match"
    assert rescued.is_correct is True

    missing = extract_with_fallback(
        "There is no formatted result here.",
        benchmark="gsm8k",
        correct_answer="26",
    )
    assert missing.value == ""
    assert missing.method == "unextractable"
    assert missing.is_correct is False


def test_fallback_rules_and_answer_equality_match_the_canonical_answer_scan() -> None:
    assert fallback_answer("Work.\\boxed{1,200}", benchmark="gsm8k") == (
        "1200",
        "N1_boxed",
    )
    assert fallback_answer("Work.\n**Therefore $26.50**", benchmark="gsm8k") == (
        "26.5",
        "N3_bold",
    )
    assert fallback_answer("work\nx = $40", benchmark="gsm8k") == (
        "40",
        "N4_equals_tail",
    )
    assert answers_equal("$40.00", "40", benchmark="gsm8k") is True
    assert answers_equal("(b).", "B", benchmark="mmlu") is True
    assert answers_equal("", "B", benchmark="mmlu") is False
    huge_integer = "9" * 400
    assert fallback_answer(f"Final answer: {huge_integer}", benchmark="gsm8k") == (
        huge_integer,
        "N2_answer_line",
    )
    assert answers_equal(huge_integer, huge_integer, benchmark="gsm8k") is True
    finite_large_integer = "1234567890123456789"
    assert fallback_answer(
        f"Final answer: {finite_large_integer}", benchmark="gsm8k"
    ) == (
        finite_large_integer,
        "N2_answer_line",
    )
    assert answers_equal(
        f"{finite_large_integer}.000",
        finite_large_integer,
        benchmark="gsm8k",
    ) is True


def test_wilson_interval_and_binary_layer_summary_use_fixed_pair_denominator() -> None:
    low, high = wilson_interval(2, 4)
    assert low == pytest.approx(0.150039, abs=1e-6)
    assert high == pytest.approx(0.849961, abs=1e-6)

    events = (
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, False),
        (False, False, False),
    )
    summary = summarize_binary_layers(events, bootstrap_resamples=200, seed=42)
    assert [(row["successes"], row["total"], row["rate"]) for row in summary["layer_profile"]] == [
        (3, 6, 0.5),
        (3, 6, 0.5),
        (0, 6, 0.0),
    ]
    assert summary["peak"] == {
        "layer_index": 0,
        "tied_layer_indices": [0, 1],
        "relative_depth": 0.0,
        "rate": 0.5,
    }
    assert summary["mcb"]["member_layer_indices"] == [0, 1]
    assert summary["mcb"]["bootstrap_resamples"] == 200
    assert summary["mcb"]["seed"] == 42


class _AddBlock(nn.Module):
    def __init__(self, amount: float, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.amount = amount
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, str]:
        result = hidden + self.amount
        return (result, "cache") if self.tuple_output else result


def test_prefill_patch_applies_once_skips_cached_decode_and_removes_its_hook() -> None:
    layer = _AddBlock(1.0, tuple_output=True)
    patch = PrefillBlockOutputPatch(
        [layer],
        layer_index=0,
        positions=(1,),
        donor_values=torch.tensor([[7.0]]),
    )
    with patch:
        prefill = layer(torch.zeros(1, 3, 1))
        decode = layer(torch.zeros(1, 1, 1))
    assert isinstance(prefill, tuple)
    assert prefill[0][:, :, 0].tolist() == [[1.0, 7.0, 1.0]]
    assert isinstance(decode, tuple)
    assert decode[0][:, :, 0].tolist() == [[1.0]]
    assert patch.applications == 1
    assert layer(torch.zeros(1, 3, 1))[0][:, :, 0].tolist() == [[1.0, 1.0, 1.0]]

    first_token_patch = PrefillBlockOutputPatch(
        [layer],
        layer_index=0,
        positions=(0,),
        donor_values=torch.tensor([[8.0]]),
    )
    with first_token_patch:
        first_token_prefill = layer(torch.zeros(1, 3, 1))
        first_token_decode = layer(torch.zeros(1, 1, 1))
    assert first_token_prefill[0][:, :, 0].tolist() == [[8.0, 1.0, 1.0]]
    assert first_token_decode[0][:, :, 0].tolist() == [[1.0]]
    assert first_token_patch.applications == 1


def test_prefill_patch_fails_closed_on_repeated_or_missing_prefill() -> None:
    layer = _AddBlock(1.0)
    with pytest.raises(RuntimeError, match="more than once"):
        with PrefillBlockOutputPatch(
            [layer],
            layer_index=0,
            positions=(1,),
            donor_values=torch.tensor([[7.0]]),
        ):
            layer(torch.zeros(1, 3, 1))
            layer(torch.zeros(1, 4, 1))
    assert layer(torch.zeros(1, 3, 1))[:, :, 0].tolist() == [[1.0, 1.0, 1.0]]

    with pytest.raises(RuntimeError, match="did not run during prompt prefill"):
        with PrefillBlockOutputPatch(
            [layer],
            layer_index=0,
            positions=(1,),
            donor_values=torch.tensor([[7.0]]),
        ):
            layer(torch.zeros(1, 1, 1))


def test_continuation_slicing_rejects_empty_or_malformed_generation() -> None:
    output = torch.tensor([[10, 11, 20, 21]])
    assert continuation_token_ids(output, prompt_length=2) == (20, 21)
    with pytest.raises(ValueError, match="no continuation"):
        continuation_token_ids(torch.tensor([[10, 11]]), prompt_length=2)
    with pytest.raises(ValueError, match="batch size 1"):
        continuation_token_ids(torch.tensor([[1], [2]]), prompt_length=0)


def test_runtime_generation_freezes_greedy_tensor_output_and_extraction() -> None:
    runtime = object.__new__(HuggingFaceLayerwiseAnswerPatchingRuntime)
    calls: list[dict[str, object]] = []
    runtime.config = SimpleNamespace(benchmark="gsm8k")
    runtime.model = SimpleNamespace(
        generate=lambda **kwargs: calls.append(kwargs) or torch.tensor([[10, 11, 20]])
    )
    runtime.tokenizer = SimpleNamespace(
        pad_token_id=0,
        decode=lambda token_ids, **kwargs: "The answer is 2.",
    )

    generation = runtime._generate(
        input_ids=torch.tensor([[10, 11]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        correct_answer="2",
    )

    assert generation.token_ids == (20,)
    assert generation.value == "2"
    assert generation.is_correct is True
    assert calls[0]["max_new_tokens"] == 512
    assert calls[0]["do_sample"] is False
    assert calls[0]["use_cache"] is True
    assert calls[0]["return_dict_in_generate"] is False
    assert calls[0]["output_scores"] is False


def test_runner_rechecks_baselines_and_uses_one_fixed_denominator_for_both_directions(
    tmp_path: Path,
) -> None:
    attribution, random_pairs = _sources(
        tmp_path,
        attribution_pairs=[_pair("same-id", targeting="attribution-4")],
        random_pairs=[_pair("same-id", targeting="random-4")],
    )
    random_key = ("random-4", "same-id")
    runtime = FakeRuntime(
        baselines={
            random_key: BaselineScan(
                sample_id="same-id",
                clean=_answer("9", correct=False, token=90),
                edited=_answer("3", correct=False, token=30),
            )
        }
    )
    result = run_layerwise_answer_patching(
        _config(attribution, random_pairs, tmp_path / "out"),
        runtime=runtime,
    )

    assert result.fixed_pairs == 1
    assert result.layer_records == 6
    assert set(runtime.baseline_calls) == {
        ("attribution-4", "same-id"),
        ("random-4", "same-id"),
    }
    assert runtime.scan_calls == [
        ("attribution-4", "same-id", ("clean-to-edited", "edited-to-clean"))
    ]

    rows = _read_jsonl(result.answer_layer_records_path)
    assert [(row["direction"], row["layer_index"], row["event"]) for row in rows] == [
        ("clean-to-edited", 0, True),
        ("clean-to-edited", 1, False),
        ("clean-to-edited", 2, False),
        ("edited-to-clean", 0, True),
        ("edited-to-clean", 1, False),
        ("edited-to-clean", 2, False),
    ]
    assert rows[1]["patched_answer"]["is_extracted"] is False
    assert rows[4]["patched_answer"]["is_extracted"] is False

    statuses = _read_jsonl(result.pair_status_records_path)
    assert [(row["targeting"], row["status"], row["exclusion_reason"]) for row in statuses] == [
        ("attribution-4", "included", None),
        ("random-4", "excluded", "regenerated_clean_not_correct"),
    ]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["population"]["selected_anchors"] == 2
    assert summary["population"]["fixed_current_flips"] == 1
    assert summary["population"]["selected_by_targeting"] == {
        "attribution-4": 1,
        "random-4": 1,
    }
    assert summary["population"]["fixed_by_targeting"] == {
        "attribution-4": 1,
        "random-4": 0,
    }
    assert summary["directions"]["clean-to-edited"]["included_pairs"] == 1
    assert summary["directions"]["edited-to-clean"]["included_pairs"] == 1

    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["paper_sha256"] == PAPER_SHA256
    assert run["comparability"]["status"] == "non-paper-setting"
    assert run["comparability"]["limitations"] == [
        "model-not-in-paper-eight-settings",
        "no-fixed-random-4-anchors",
    ]
    assert run["comparability"]["exact_historical_figure2_ids"] is False
    assert run["comparability"]["historical_qwen_targeting_discrepancy"] is True
    assert run["comparability"]["historical_unextractable_induction_discrepancy"] is True
    assert set(run["outputs"]) == {
        "answer_layer_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    }


def test_paper_comparability_requires_both_directions_and_both_targeting_arms(
    tmp_path: Path,
) -> None:
    paper_model = "Qwen/Qwen2.5-3B-Instruct"

    def paper_pair(
        sample_id: str,
        *,
        targeting: str,
        edited_correct: bool = False,
    ) -> dict[str, object]:
        record = _pair(sample_id, targeting=targeting, edited_correct=edited_correct)
        record["model"] = paper_model
        return record

    full_attribution = _write_pair_source(
        tmp_path / "full" / "attribution",
        [
            paper_pair(f"a-{index:03d}", targeting="attribution-4")
            for index in range(150)
        ],
        targeting="attribution-4",
        model=paper_model,
    )
    full_random = _write_pair_source(
        tmp_path / "full" / "random",
        [paper_pair(f"r-{index:03d}", targeting="random-4") for index in range(150)],
        targeting="random-4",
        model=paper_model,
    )
    full_config = _config(
        full_attribution,
        full_random,
        tmp_path / "full" / "out",
        model=paper_model,
    )
    full_result = run_layerwise_answer_patching(full_config, runtime=FakeRuntime())
    full_run = json.loads(full_result.run_path.read_text(encoding="utf-8"))
    assert full_run["comparability"]["status"] == "fresh-paper-protocol-reproduction"
    assert full_run["comparability"]["limitations"] == []
    assert full_run["comparability"]["targeting_counts"] == {
        "paper_cap_per_targeting": 150,
        "selected": {"attribution-4": 150, "random-4": 150},
        "fixed": {"attribution-4": 150, "random-4": 150},
    }

    under_cap_attribution = _write_pair_source(
        tmp_path / "under-cap" / "attribution",
        [paper_pair("a", targeting="attribution-4")],
        targeting="attribution-4",
        model=paper_model,
    )
    under_cap_random = _write_pair_source(
        tmp_path / "under-cap" / "random",
        [paper_pair("r", targeting="random-4")],
        targeting="random-4",
        model=paper_model,
    )
    under_cap_config = replace(
        full_config,
        attribution_pairs=under_cap_attribution,
        random_pairs=under_cap_random,
        output_dir=tmp_path / "under-cap" / "out",
    )
    under_cap_result = run_layerwise_answer_patching(
        under_cap_config,
        runtime=FakeRuntime(),
    )
    under_cap_run = json.loads(under_cap_result.run_path.read_text(encoding="utf-8"))
    assert under_cap_run["comparability"]["status"] == "partial-paper-protocol"
    assert under_cap_run["comparability"]["limitations"] == [
        "selected-attribution-4-below-paper-cap-1-of-150",
        "selected-random-4-below-paper-cap-1-of-150",
    ]

    partial_result = run_layerwise_answer_patching(
        replace(
            under_cap_config,
            directions=("clean-to-edited",),
            output_dir=tmp_path / "partial-directions",
        ),
        runtime=FakeRuntime(),
    )
    partial_run = json.loads(partial_result.run_path.read_text(encoding="utf-8"))
    assert partial_run["comparability"]["status"] == "partial-paper-protocol"
    assert "directions-not-both" in partial_run["comparability"]["limitations"]

    missing_random = _write_pair_source(
        tmp_path / "missing-arm" / "random",
        [paper_pair("r", targeting="random-4", edited_correct=True)],
        targeting="random-4",
        model=paper_model,
    )
    missing_result = run_layerwise_answer_patching(
        replace(
            under_cap_config,
            random_pairs=missing_random,
            output_dir=tmp_path / "missing-arm" / "out",
        ),
        runtime=FakeRuntime(),
    )
    missing_run = json.loads(missing_result.run_path.read_text(encoding="utf-8"))
    assert missing_run["comparability"]["status"] == "partial-paper-protocol"
    assert "no-selected-random-4-anchors" in missing_run["comparability"]["limitations"]


def test_prompt_final_aligned_anchors_are_excluded_from_the_structural_control(
    tmp_path: Path,
) -> None:
    prompt_final = _pair("a-final", targeting="attribution-4")
    prompt_final["clean"]["prompt_token_count"] = 2
    prompt_final["edited"]["prompt_token_count"] = 2
    attribution, random_pairs = _sources(
        tmp_path,
        attribution_pairs=[
            _pair("a-safe", targeting="attribution-4"),
            prompt_final,
        ],
        random_pairs=[
            _pair("b", targeting="random-4"),
            _pair("r-ineligible", targeting="random-4", edited_correct=True),
        ],
    )
    runtime = FakeRuntime()

    result = run_layerwise_answer_patching(
        _config(attribution, random_pairs, tmp_path / "out"),
        runtime=runtime,
    )

    assert set(runtime.baseline_calls) == {
        ("attribution-4", "a-safe"),
        ("random-4", "b"),
    }
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["population"]["upstream_exclusions"]["attribution-4"] == {
        "aligned_word_at_prompt_final_position": 1,
    }


def test_empty_regenerated_fixed_cohort_is_recorded_as_a_failed_run(
    tmp_path: Path,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    excluded = {
        key: BaselineScan(
            sample_id=key[1],
            clean=_answer("9", correct=False, token=90),
            edited=_answer("3", correct=False, token=30),
        )
        for key in (("attribution-4", "a"), ("random-4", "b"))
    }
    output = tmp_path / "out"

    with pytest.raises(LayerwiseAnswerPatchingRunError, match="no selected anchor"):
        run_layerwise_answer_patching(
            _config(attribution, random_pairs, output),
            runtime=FakeRuntime(baselines=excluded),
        )

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["counts"]["fixed_current_flips"] == 0
    assert run["failures"] == [
        {
            "error_type": "EmptyFixedCohort",
            "message": "no selected anchor remained a regenerated clean-correct, edited-wrong pair",
            "sample_id": "*",
            "targeting": "all",
        }
    ]
    assert not (output / "answer_layer_records.jsonl").exists()
    assert len(list((output / ".layerwise-answer-patching-work/checkpoints").glob("*.json"))) == 2


def test_balanced_seed42_anchor_selection_is_per_arm_and_input_order_independent(
    tmp_path: Path,
) -> None:
    attribution_records = [_pair(f"a-{index}", targeting="attribution-4") for index in range(6)]
    random_records = [_pair(f"r-{index}", targeting="random-4") for index in range(6)]
    attribution, random_pairs = _sources(
        tmp_path,
        attribution_pairs=attribution_records,
        random_pairs=random_records,
    )
    runtime = FakeRuntime()
    run_layerwise_answer_patching(
        _config(attribution, random_pairs, tmp_path / "out", max_pairs=4),
        runtime=runtime,
    )

    expected_a = sorted(record["sample_id"] for record in attribution_records)
    expected_r = sorted(record["sample_id"] for record in random_records)
    random.Random(42).shuffle(expected_a)
    random.Random(42).shuffle(expected_r)
    assert set(runtime.baseline_calls) == {
        *(("attribution-4", str(sample_id)) for sample_id in expected_a[:2]),
        *(("random-4", str(sample_id)) for sample_id in expected_r[:2]),
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda a, r, ap, rp: a.update(status="failed"), "source run is not completed"),
        (lambda a, r, ap, rp: a.update(paper_sha256="bad"), "paper SHA-256"),
        (
            lambda a, r, ap, rp: a["arguments"].update(seed=7),
            "seed 42",
        ),
        (
            lambda a, r, ap, rp: r["arguments"].update(num_edits=2),
            "four edits",
        ),
        (
            lambda a, r, ap, rp: r["arguments"].update(limit=1),
            "must not be limited",
        ),
        (
            lambda a, r, ap, rp: r["arguments"].update(max_new_tokens=64),
            "generation cap 512",
        ),
        (
            lambda a, r, ap, rp: a["counts"].update(discovered=2),
            "complete dataset",
        ),
        (
            lambda a, r, ap, rp: r["provenance"].update(model_revision="other"),
            "model revisions differ",
        ),
        (
            lambda a, r, ap, rp: r["provenance"].update(dataset_records_sha256="other"),
            "dataset fingerprints differ",
        ),
        (
            lambda a, r, ap, rp: rp.update(targeting="attribution-4"),
            "record targeting",
        ),
        (
            lambda a, r, ap, rp: ap["aligned_words"][0].update(clean_final_token=2),
            "clean_final_token",
        ),
    ),
)
def test_two_source_contract_errors_fail_before_runtime_scanning(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    a_manifest = json.loads((attribution.parent / "run.json").read_text(encoding="utf-8"))
    r_manifest = json.loads((random_pairs.parent / "run.json").read_text(encoding="utf-8"))
    a_pair = json.loads(attribution.read_text(encoding="utf-8"))
    r_pair = json.loads(random_pairs.read_text(encoding="utf-8"))
    mutate(a_manifest, r_manifest, a_pair, r_pair)  # type: ignore[operator]
    _write_json(attribution.parent / "run.json", a_manifest)
    _write_json(random_pairs.parent / "run.json", r_manifest)
    attribution.write_text(json.dumps(a_pair) + "\n", encoding="utf-8")
    random_pairs.write_text(json.dumps(r_pair) + "\n", encoding="utf-8")
    runtime = FakeRuntime()

    with pytest.raises(ValueError, match=message):
        run_layerwise_answer_patching(
            _config(attribution, random_pairs, tmp_path / "out"),
            runtime=runtime,
        )
    assert runtime.baseline_calls == []
    assert runtime.scan_calls == []


@pytest.mark.parametrize(
    ("runtime", "message"),
    (
        (FakeRuntime(requested_revision="other"), "requested revision"),
        (FakeRuntime(model_revision="other"), "model revision"),
        (FakeRuntime(tokenizer_revision="other"), "tokenizer revision"),
    ),
)
def test_runtime_revision_must_match_both_pair_sources_before_generation(
    tmp_path: Path,
    runtime: FakeRuntime,
    message: str,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    with pytest.raises(ValueError, match=message):
        run_layerwise_answer_patching(
            _config(attribution, random_pairs, tmp_path / "out"),
            runtime=runtime,
        )
    assert runtime.baseline_calls == []


def test_default_runtime_receives_the_common_source_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typo_cot.experiments.layerwise_answer_patching import runtime as runtime_module

    attribution, random_pairs = _sources(tmp_path)
    received: dict[str, object] = {}

    class RecordingRuntime(FakeRuntime):
        def __init__(self, config: object, *, revision: str) -> None:
            super().__init__(
                requested_revision=revision,
                model_revision=revision,
                tokenizer_revision=revision,
            )
            received.update(config=config, revision=revision)

    monkeypatch.setattr(
        runtime_module,
        "HuggingFaceLayerwiseAnswerPatchingRuntime",
        RecordingRuntime,
    )
    run_layerwise_answer_patching(_config(attribution, random_pairs, tmp_path / "out"))
    assert received["revision"] == "source-revision"


def test_runtime_provenance_uses_explicit_revision_when_tokenizer_metadata_is_empty() -> None:
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    runtime = object.__new__(HuggingFaceLayerwiseAnswerPatchingRuntime)
    runtime.config = SimpleNamespace(model="test/model")
    runtime.revision = revision
    runtime.model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash=revision),
        get_decoder=lambda: SimpleNamespace(layers=[]),
    )
    runtime.tokenizer = SimpleNamespace(init_kwargs={})
    runtime.num_layers = 3
    runtime.device = "cuda:0"
    runtime._torch = SimpleNamespace(
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(
            get_device_name=lambda index: "test-gpu",
            get_device_properties=lambda index: SimpleNamespace(total_memory=1024),
        ),
    )
    provenance = runtime.provenance()
    assert provenance["requested_revision"] == revision
    assert provenance["model_revision"] == revision
    assert provenance["tokenizer_revision"] == revision
    assert provenance["tokenizer_revision_source"] == "explicit-load-revision"
    assert provenance["generation"]["max_new_tokens"] == 512
    assert provenance["generation"]["use_cache"] is True


def test_runtime_failure_keeps_completed_targeting_keyed_checkpoints_for_resume(
    tmp_path: Path,
) -> None:
    attribution, random_pairs = _sources(
        tmp_path,
        attribution_pairs=[_pair("same-id", targeting="attribution-4")],
        random_pairs=[_pair("same-id", targeting="random-4")],
    )
    output = tmp_path / "out"
    config = _config(attribution, random_pairs, output)
    first = FakeRuntime(error_for=("random-4", "same-id"))
    with pytest.raises(LayerwiseAnswerPatchingRunError, match="1 pair"):
        run_layerwise_answer_patching(config, runtime=first)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert not (output / "answer_layer_records.jsonl").exists()
    checkpoint_files = list((output / ".layerwise-answer-patching-work/checkpoints").glob("*.json"))
    assert len(checkpoint_files) == 1
    checkpoint = json.loads(checkpoint_files[0].read_text(encoding="utf-8"))
    checkpoint["directions"]["clean-to-edited"]["layers"][0]["event"] = False
    _write_json(checkpoint_files[0], checkpoint)

    resumed_runtime = FakeRuntime()
    result = run_layerwise_answer_patching(
        replace(config, resume=True),
        runtime=resumed_runtime,
    )
    assert resumed_runtime.baseline_calls == [
        ("attribution-4", "same-id"),
        ("random-4", "same-id"),
    ]
    assert resumed_runtime.scan_calls == [
        ("attribution-4", "same-id", ("clean-to-edited", "edited-to-clean")),
        ("random-4", "same-id", ("clean-to-edited", "edited-to-clean")),
    ]
    assert result.fixed_pairs == 2


def test_resume_recomputes_schema_invalid_checkpoint_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    attribution, random_pairs = _sources(
        tmp_path,
        attribution_pairs=[_pair("same-id", targeting="attribution-4")],
        random_pairs=[_pair("same-id", targeting="random-4")],
    )
    output = tmp_path / "out"
    config = _config(attribution, random_pairs, output)
    with pytest.raises(LayerwiseAnswerPatchingRunError):
        run_layerwise_answer_patching(
            config,
            runtime=FakeRuntime(error_for=("random-4", "same-id")),
        )

    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    checkpoint_path = next((output / ".layerwise-answer-patching-work/checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["directions"]["clean-to-edited"]["layers"][0]["layer_index"] = 99
    _write_json(checkpoint_path, checkpoint)
    run["checkpoints"][checkpoint_path.name]["sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    _write_json(run_path, run)

    resumed_runtime = FakeRuntime()
    result = run_layerwise_answer_patching(
        replace(config, resume=True),
        runtime=resumed_runtime,
    )
    assert resumed_runtime.baseline_calls == [
        ("attribution-4", "same-id"),
        ("random-4", "same-id"),
    ]
    assert result.fixed_pairs == 2


def test_output_compilation_failure_marks_manifest_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    output = tmp_path / "out"

    def fail_compilation(**kwargs: object) -> None:
        raise ValueError("synthetic compilation failure")

    monkeypatch.setattr(answer_runner_module, "_compile_outputs", fail_compilation)
    with pytest.raises(LayerwiseAnswerPatchingRunError, match="output compilation failed"):
        run_layerwise_answer_patching(
            _config(attribution, random_pairs, output),
            runtime=FakeRuntime(),
        )

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["failures"][0]["error_type"] == "OutputCompilationError"
    assert not (output / "answer_layer_records.jsonl").exists()
    assert (output / ".layerwise-answer-patching-work").is_dir()


def test_output_write_failure_removes_partial_public_outputs_and_marks_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    output = tmp_path / "out"
    original_write = answer_runner_module._write_jsonl_atomic

    def fail_second_output(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
        if path.name == "pair_status_records.jsonl":
            raise OSError("synthetic output failure")
        original_write(path, rows)

    monkeypatch.setattr(answer_runner_module, "_write_jsonl_atomic", fail_second_output)
    with pytest.raises(LayerwiseAnswerPatchingRunError, match="output finalization failed"):
        run_layerwise_answer_patching(
            _config(attribution, random_pairs, output),
            runtime=FakeRuntime(),
        )

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["failures"][0]["error_type"] == "OutputFinalizationError"
    for name in (
        "answer_layer_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    ):
        assert not (output / name).exists()


def test_completed_resume_verifies_outputs_before_default_runtime_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typo_cot.experiments.layerwise_answer_patching import runtime as runtime_module

    attribution, random_pairs = _sources(tmp_path)
    output = tmp_path / "out"
    config = _config(attribution, random_pairs, output)
    result = run_layerwise_answer_patching(config, runtime=FakeRuntime())

    class MustNotLoad:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("completed resume loaded model weights")

    monkeypatch.setattr(runtime_module, "HuggingFaceLayerwiseAnswerPatchingRuntime", MustNotLoad)
    assert run_layerwise_answer_patching(replace(config, resume=True)) == result

    result.answer_layer_records_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or has changed"):
        run_layerwise_answer_patching(replace(config, resume=True))


def test_nonempty_output_and_changed_resume_arguments_fail_closed(tmp_path: Path) -> None:
    attribution, random_pairs = _sources(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "unrelated").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output directory is not empty"):
        run_layerwise_answer_patching(
            _config(attribution, random_pairs, output),
            runtime=FakeRuntime(),
        )

    completed = tmp_path / "completed"
    config = _config(attribution, random_pairs, completed)
    run_layerwise_answer_patching(config, runtime=FakeRuntime())
    with pytest.raises(ValueError, match="resume arguments do not match"):
        run_layerwise_answer_patching(
            replace(config, directions=("clean-to-edited",), resume=True),
            runtime=FakeRuntime(),
        )
