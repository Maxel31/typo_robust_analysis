"""Final-paper contract tests for fixed-window free-answer patching."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.fixed_window_answer_patching.metrics import (
    paired_binary_difference,
)
from typo_cot.experiments.fixed_window_answer_patching.patching import (
    PrefillBlockOutputWindowPatch,
)
from typo_cot.experiments.fixed_window_answer_patching.runner import (
    AnswerGeneration,
    BaselineScan,
    DirectionWindowScan,
    FixedWindowAnswerPatchingConfig,
    FixedWindowAnswerPatchingResult,
    FixedWindowAnswerPatchingRunError,
    LayerWindow,
    PairWindowScan,
    parse_layer_window,
    run_fixed_window_answer_patching,
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
    model: str = "test/model",
    benchmark: str = "gsm8k",
    clean_correct: bool = True,
    edited_correct: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": model,
        "benchmark": benchmark,
        "targeting": targeting,
        "seed": 42,
        "num_edits_requested": 4,
        "num_aligned_words": 1,
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
        "aligned_words": [_aligned_word()],
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
) -> Path:
    root.mkdir(parents=True)
    pairs_path = root / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for pair in sorted(pairs, key=lambda item: str(item["sample_id"])):
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
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
    model: str = "test/model",
    benchmark: str = "gsm8k",
) -> tuple[Path, Path]:
    attribution_pairs = attribution_pairs or [
        _pair("a", targeting="attribution-4", model=model, benchmark=benchmark)
    ]
    random_pairs = random_pairs or [
        _pair("b", targeting="random-4", model=model, benchmark=benchmark)
    ]
    return (
        _write_pair_source(
            tmp_path / "attribution",
            attribution_pairs,
            targeting="attribution-4",
            model=model,
            benchmark=benchmark,
        ),
        _write_pair_source(
            tmp_path / "random",
            random_pairs,
            targeting="random-4",
            model=model,
            benchmark=benchmark,
        ),
    )


def _answer(value: str, *, correct: bool, token: int) -> AnswerGeneration:
    return AnswerGeneration(
        token_ids=(token,),
        text=f"answer {value}" if value else "unextractable",
        value=value,
        is_extracted=bool(value),
        is_correct=correct,
        method="primary:fixture" if value else "unextractable",
        primary_method="fixture" if value else "no_match",
    )


class FakeRuntime:
    def __init__(
        self,
        *,
        n_layers: int = 12,
        baselines: Mapping[tuple[str, str], BaselineScan] | None = None,
        scans: Mapping[tuple[str, str], PairWindowScan] | None = None,
        error_for: tuple[str, str] | None = None,
    ) -> None:
        self.num_layers = n_layers
        self.baselines = dict(baselines or {})
        self.scans = dict(scans or {})
        self.error_for = error_for
        self.baseline_calls: list[tuple[str, str]] = []
        self.scan_calls: list[
            tuple[str, str, tuple[LayerWindow, ...], tuple[str, ...]]
        ] = []

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "fake",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
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
        return self.baselines.get(
            key,
            BaselineScan(
                sample_id=key[1],
                clean=_answer("2", correct=True, token=20),
                edited=_answer("3", correct=False, token=30),
            ),
        )

    def scan_pair(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        windows: tuple[LayerWindow, ...],
        directions: tuple[str, ...],
    ) -> PairWindowScan:
        key = (str(pair["targeting"]), str(pair["sample_id"]))
        self.scan_calls.append((*key, windows, directions))
        if key == self.error_for:
            raise RuntimeError("synthetic GPU failure")
        if key in self.scans:
            return self.scans[key]
        generated = {
            "clean-to-edited": {
                window.label: _answer("2", correct=True, token=100 + window.start)
                for window in windows
            },
            "edited-to-clean": {
                window.label: _answer("3", correct=False, token=200 + window.start)
                for window in windows
            },
        }
        return PairWindowScan(
            sample_id=key[1],
            directions={
                direction: DirectionWindowScan(patched_by_window=generated[direction])
                for direction in directions
            },
        )


def _config(
    pairs: tuple[Path, ...],
    output_dir: Path,
    **changes: object,
) -> FixedWindowAnswerPatchingConfig:
    config = FixedWindowAnswerPatchingConfig(
        model="test/model",
        benchmark="gsm8k",
        pairs=pairs,
        layers=(LayerWindow(0, 6),),
        directions=("clean-to-edited", "edited-to-clean"),
        output_dir=output_dir,
    )
    return replace(config, **changes)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_catalog_marks_fixed_window_operation_as_implemented() -> None:
    spec = get_experiment("fixed-window-answer-patching")
    assert spec.status == "implemented"
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--pairs",
        "--layers",
        "--directions",
        "--output-dir",
    )
    assert spec.outputs == (
        "fixed_window_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
        "run.json",
    )


def test_cli_dispatches_paths_windows_and_runtime_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[FixedWindowAnswerPatchingConfig] = []

    def fake_run(config: FixedWindowAnswerPatchingConfig) -> FixedWindowAnswerPatchingResult:
        captured.append(config)
        return FixedWindowAnswerPatchingResult(
            fixed_window_records_path=config.output_dir / "fixed_window_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "setting_summary.json",
            run_path=config.output_dir / "run.json",
            included_direction_pairs=3,
            window_records=6,
        )

    monkeypatch.setattr(cli_module, "run_fixed_window_answer_patching", fake_run)
    assert main(
        [
            "fixed-window-answer-patching",
            "--model",
            "test/model",
            "--benchmark",
            "mmlu-pro",
            "--pairs",
            "results/a/pairs.jsonl",
            "--layers",
            "0:6",
            "6:12",
            "--directions",
            "clean-to-edited",
            "--gpu-id",
            "0",
            "--limit",
            "2",
            "--output-dir",
            "results/fixed",
            "--resume",
        ]
    ) == 0
    assert captured == [
        FixedWindowAnswerPatchingConfig(
            model="test/model",
            benchmark="mmlu-pro",
            pairs=(Path("results/a/pairs.jsonl"),),
            layers=(LayerWindow(0, 6), LayerWindow(6, 12)),
            directions=("clean-to-edited",),
            output_dir=Path("results/fixed"),
            gpu_id="0",
            limit=2,
            resume=True,
        )
    ]
    assert capsys.readouterr().out.splitlines() == [
        "wrote 6 fixed-window record(s) from 3 included pair-direction(s): "
        "results/fixed/fixed_window_records.jsonl",
        "setting summary: results/fixed/setting_summary.json",
        "run manifest: results/fixed/run.json",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    (("0:6", LayerWindow(0, 6)), ("6:12", LayerWindow(6, 12)), ("01:07", LayerWindow(1, 7))),
)
def test_parse_layer_window_uses_half_open_decoder_indices(
    value: str,
    expected: LayerWindow,
) -> None:
    assert parse_layer_window(value) == expected


@pytest.mark.parametrize("value", ("", "0", "0-6", "-1:6", "6:6", "7:6", "0:6:12"))
def test_parse_layer_window_rejects_ambiguous_or_empty_ranges(value: str) -> None:
    with pytest.raises((ValueError, TypeError), match="layer window"):
        parse_layer_window(value)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"benchmark": "math500"}, "gsm8k, mmlu, or mmlu-pro"),
        ({"pairs": ()}, "one or two"),
        ({"pairs": (Path("a"), Path("b"), Path("c"))}, "one or two"),
        ({"layers": ()}, "at least one layer window"),
        ({"layers": (LayerWindow(0, 6), LayerWindow(5, 11))}, "overlap"),
        ({"directions": ()}, "at least one direction"),
        ({"directions": ("clean-to-edited", "clean-to-edited")}, "duplicates"),
        ({"directions": ("sideways",)}, "unsupported direction"),
        ({"gpu_id": "0,1"}, "single non-negative integer"),
        ({"limit": 0}, "positive integer"),
    ),
)
def test_config_rejects_invalid_operation_arguments(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config((tmp_path / "pairs",), tmp_path / "out", **changes)


class _AddBlock(nn.Module):
    def __init__(self, amount: float, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.amount = amount
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, str]:
        result = hidden + self.amount
        return (result, "cache") if self.tuple_output else result


def test_window_patch_writes_every_layer_in_one_prefill_and_skips_decode() -> None:
    layers = [_AddBlock(1.0), _AddBlock(2.0, tuple_output=True)]
    patch = PrefillBlockOutputWindowPatch(
        layers,
        layer_indices=(0, 1),
        positions=(1,),
        donor_values=(torch.tensor([[10.0]]), torch.tensor([[20.0]])),
    )
    with patch:
        hidden: torch.Tensor | tuple[torch.Tensor, str] = torch.zeros(1, 3, 1)
        for layer in layers:
            hidden = layer(hidden[0] if isinstance(hidden, tuple) else hidden)
        decode: torch.Tensor | tuple[torch.Tensor, str] = torch.zeros(1, 1, 1)
        for layer in layers:
            decode = layer(decode[0] if isinstance(decode, tuple) else decode)
    assert isinstance(hidden, tuple)
    assert hidden[0][:, :, 0].tolist() == [[3.0, 20.0, 3.0]]
    assert isinstance(decode, tuple)
    assert decode[0][:, :, 0].tolist() == [[3.0]]
    assert patch.applications == (1, 1)
    assert len(layers[0]._forward_hooks) == len(layers[1]._forward_hooks) == 0


def test_window_patch_rejects_mismatched_layers_and_donor_cache() -> None:
    layers = [_AddBlock(1.0), _AddBlock(2.0)]
    with pytest.raises(ValueError, match="donor cache"):
        PrefillBlockOutputWindowPatch(
            layers,
            layer_indices=(0, 1),
            positions=(1,),
            donor_values=(torch.tensor([[10.0]]),),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        PrefillBlockOutputWindowPatch(
            layers,
            layer_indices=(1, 0),
            positions=(1,),
            donor_values=(torch.tensor([[10.0]]), torch.tensor([[20.0]])),
        )


def test_paired_binary_difference_is_deterministic_and_pair_resampled() -> None:
    result = paired_binary_difference(
        (True, True, False, False),
        (False, True, True, False),
        resamples=10_000,
        seed=42,
    )
    assert result["pairs"] == 4
    assert result["left_successes"] == 2
    assert result["right_successes"] == 2
    assert result["difference"] == 0.0
    assert result["bootstrap_resamples"] == 10_000
    assert result["seed"] == 42
    assert result["confidence_interval"] == [-0.5, 0.5]


def test_runner_uses_direction_specific_denominators_and_unextractable_is_failure(
    tmp_path: Path,
) -> None:
    attribution, random_pairs = _sources(
        tmp_path,
        attribution_pairs=[_pair("flip", targeting="attribution-4")],
        random_pairs=[_pair("still-correct", targeting="random-4")],
    )
    still_correct = ("random-4", "still-correct")
    baseline_correct = BaselineScan(
        sample_id="still-correct",
        clean=_answer("2", correct=True, token=20),
        edited=_answer("2", correct=True, token=21),
    )
    scans = {
        ("attribution-4", "flip"): PairWindowScan(
            sample_id="flip",
            directions={
                "clean-to-edited": DirectionWindowScan(
                    patched_by_window={"0:6": _answer("", correct=False, token=31)}
                ),
                "edited-to-clean": DirectionWindowScan(
                    patched_by_window={"0:6": _answer("", correct=False, token=32)}
                ),
            },
        ),
        still_correct: PairWindowScan(
            sample_id="still-correct",
            directions={
                "edited-to-clean": DirectionWindowScan(
                    patched_by_window={"0:6": _answer("3", correct=False, token=33)}
                )
            },
        ),
    }
    runtime = FakeRuntime(baselines={still_correct: baseline_correct}, scans=scans)
    result = run_fixed_window_answer_patching(
        _config((attribution, random_pairs), tmp_path / "out"),
        runtime=runtime,
    )

    assert result.included_direction_pairs == 3
    assert result.window_records == 3
    assert runtime.scan_calls == [
        (
            "attribution-4",
            "flip",
            (LayerWindow(0, 6),),
            ("clean-to-edited", "edited-to-clean"),
        ),
        (
            "random-4",
            "still-correct",
            (LayerWindow(0, 6),),
            ("edited-to-clean",),
        ),
    ]
    rows = _read_jsonl(result.fixed_window_records_path)
    assert [(row["sample_id"], row["direction"], row["event"]) for row in rows] == [
        ("flip", "clean-to-edited", False),
        ("flip", "edited-to-clean", False),
        ("still-correct", "edited-to-clean", True),
    ]
    statuses = _read_jsonl(result.pair_status_records_path)
    assert statuses[0]["direction_status"]["clean-to-edited"] == {
        "included": True,
        "exclusion_reason": None,
    }
    assert statuses[1]["direction_status"]["clean-to-edited"] == {
        "included": False,
        "exclusion_reason": "regenerated_edited_not_wrong",
    }
    assert statuses[1]["direction_status"]["edited-to-clean"]["included"] is True

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["directions"]["clean-to-edited"]["included_pairs"] == 1
    assert summary["directions"]["edited-to-clean"]["included_pairs"] == 2
    assert summary["directions"]["clean-to-edited"]["windows"]["0:6"]["successes"] == 0
    assert summary["directions"]["edited-to-clean"]["windows"]["0:6"]["successes"] == 1
    assert summary["historical_reference"]["induction_unextractable_discrepancy"] is True


def test_clean_incorrect_pair_is_excluded_from_both_directions_without_scan(
    tmp_path: Path,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    runtime = FakeRuntime(
        baselines={
            ("attribution-4", "a"): BaselineScan(
                sample_id="a",
                clean=_answer("9", correct=False, token=90),
                edited=_answer("3", correct=False, token=30),
            )
        }
    )
    result = run_fixed_window_answer_patching(
        _config((attribution, random_pairs), tmp_path / "out"),
        runtime=runtime,
    )
    assert ("attribution-4", "a") not in [call[:2] for call in runtime.scan_calls]
    statuses = _read_jsonl(result.pair_status_records_path)
    first = statuses[0]["direction_status"]
    assert first["clean-to-edited"]["exclusion_reason"] == "regenerated_clean_not_correct"
    assert first["edited-to-clean"]["exclusion_reason"] == "regenerated_clean_not_correct"


def test_mmlu_pro_summary_compares_windows_on_the_same_restoration_pairs(
    tmp_path: Path,
) -> None:
    model = "Qwen/Qwen2.5-3B-Instruct"
    attribution = _write_pair_source(
        tmp_path / "attribution",
        [_pair("a", targeting="attribution-4", model=model, benchmark="mmlu-pro")],
        targeting="attribution-4",
        model=model,
        benchmark="mmlu-pro",
    )
    scan = PairWindowScan(
        sample_id="a",
        directions={
            "clean-to-edited": DirectionWindowScan(
                patched_by_window={
                    "0:6": _answer("2", correct=True, token=31),
                    "6:12": _answer("3", correct=False, token=32),
                }
            )
        },
    )
    result = run_fixed_window_answer_patching(
        _config(
            (attribution,),
            tmp_path / "out",
            model=model,
            benchmark="mmlu-pro",
            layers=(LayerWindow(0, 6), LayerWindow(6, 12)),
            directions=("clean-to-edited",),
        ),
        runtime=FakeRuntime(scans={("attribution-4", "a"): scan}),
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    comparison = summary["prespecified_mmlu_pro_window_comparison"]
    assert comparison["left_window"] == "0:6"
    assert comparison["right_window"] == "6:12"
    assert comparison["pairs"] == 1
    assert comparison["difference"] == 1.0
    assert comparison["bootstrap_resamples"] == 10_000
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["comparability"]["status"] == "prespecified-mmlu-pro-window-comparison"


def test_source_contract_fails_before_runtime_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    manifest_path = attribution.parent / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paper_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    created = False

    def forbidden_factory(*args: object, **kwargs: object) -> FakeRuntime:
        nonlocal created
        created = True
        raise AssertionError("runtime must not be created")

    monkeypatch.setattr(
        "typo_cot.experiments.fixed_window_answer_patching.runner."
        "HuggingFaceFixedWindowAnswerPatchingRuntime",
        forbidden_factory,
    )
    with pytest.raises(ValueError, match="paper SHA-256"):
        run_fixed_window_answer_patching(
            _config((attribution, random_pairs), tmp_path / "out")
        )
    assert created is False


def test_gpu_failure_keeps_checkpoints_but_publishes_no_partial_tables(
    tmp_path: Path,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    output = tmp_path / "out"
    with pytest.raises(FixedWindowAnswerPatchingRunError, match="synthetic GPU failure"):
        run_fixed_window_answer_patching(
            _config((attribution, random_pairs), output),
            runtime=FakeRuntime(error_for=("random-4", "b")),
        )
    assert (output / "run.json").is_file()
    assert (output / "checkpoints").is_dir()
    for name in (
        "fixed_window_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    ):
        assert not (output / name).exists()
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failures"][0]["sample_id"] == "b"


def test_completed_resume_validates_outputs_without_loading_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attribution, random_pairs = _sources(tmp_path)
    output = tmp_path / "out"
    config = _config((attribution, random_pairs), output)
    expected = run_fixed_window_answer_patching(config, runtime=FakeRuntime())

    def forbidden_factory(*args: object, **kwargs: object) -> FakeRuntime:
        raise AssertionError("completed resume must not load model weights")

    monkeypatch.setattr(
        "typo_cot.experiments.fixed_window_answer_patching.runner."
        "HuggingFaceFixedWindowAnswerPatchingRuntime",
        forbidden_factory,
    )
    resumed = run_fixed_window_answer_patching(replace(config, resume=True))
    assert resumed == expected

