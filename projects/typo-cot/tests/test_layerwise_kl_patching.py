"""Paper-contract tests for the layerwise first-CoT-token KL scan."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.layerwise_kl_patching.metrics import (
    KL_DENOMINATOR_EPSILON,
    kl_from_logits,
    normalized_kl_score,
    summarize_direction,
)
from typo_cot.experiments.layerwise_kl_patching.patching import (
    BlockOutputPatch,
    capture_block_outputs,
    find_decoder_layers,
)
from typo_cot.experiments.layerwise_kl_patching.runner import (
    DirectionScan,
    LayerwiseKLPatchingConfig,
    LayerwiseKLPatchingResult,
    LayerwiseKLPatchingRunError,
    PairScan,
    run_layerwise_kl_patching,
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
        "targeting": "attribution-4",
        "seed": 42,
        "num_edits_requested": 4,
        "num_aligned_words": len(aligned_words),
        "clean": {
            "prompt": "alpha question",
            "prompt_token_count": 3,
            "continuation": "clean reasoning",
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
            "continuation": "edited reasoning",
            "answer": {
                "value": "3" if not edited_correct else "2",
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
    model: str = "test/model",
    benchmark: str = "gsm8k",
    targeting: str = "attribution-4",
    seed: int = 42,
    num_edits: int = 4,
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
                "model_revision": "source-revision",
                "random_seed_algorithm": "sha256-first-64-bits/v1",
            },
        },
    )
    return pairs_path


class FakeRuntime:
    def __init__(
        self,
        *,
        n_layers: int = 3,
        scans: dict[str, dict[str, DirectionScan]] | None = None,
        error_for: str | None = None,
    ) -> None:
        self.num_layers = n_layers
        self.scans = scans or {}
        self.error_for = error_for
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "fake",
            "model_revision": "runtime-revision",
            "tokenizer_revision": "runtime-revision",
            "decoder_adapter": "fake.layers",
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "device": "cuda:0",
        }

    def scan_pair(self, pair: dict[str, object], directions: tuple[str, ...]) -> PairScan:
        sample_id = str(pair["sample_id"])
        self.calls.append((sample_id, directions))
        if sample_id == self.error_for:
            raise RuntimeError("synthetic GPU failure")
        default = {
            "clean-to-edited": DirectionScan(
                denominator_kl=2.0,
                patched_kl_by_layer=tuple(1.0 - 0.2 * layer for layer in range(self.num_layers)),
            ),
            "edited-to-clean": DirectionScan(
                denominator_kl=4.0,
                patched_kl_by_layer=tuple(3.0 - 0.2 * layer for layer in range(self.num_layers)),
            ),
        }
        default.update(self.scans.get(sample_id, {}))
        return PairScan(sample_id=sample_id, directions={name: default[name] for name in directions})


def _config(pairs_path: Path, output_dir: Path, **changes: object) -> LayerwiseKLPatchingConfig:
    config = LayerwiseKLPatchingConfig(
        model="test/model",
        benchmark="gsm8k",
        pairs=pairs_path,
        targeting="attribution-4",
        directions=("clean-to-edited", "edited-to-clean"),
        output_dir=output_dir,
    )
    return replace(config, **changes)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_catalog_marks_layerwise_kl_patching_as_implemented() -> None:
    spec = get_experiment("layerwise-kl-patching")
    assert spec.status == "implemented"
    assert spec.outputs == (
        "layer_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
        "run.json",
    )


def test_cli_dispatches_operation_specific_layerwise_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[LayerwiseKLPatchingConfig] = []

    def fake_run(config: LayerwiseKLPatchingConfig) -> LayerwiseKLPatchingResult:
        captured.append(config)
        return LayerwiseKLPatchingResult(
            layer_records_path=config.output_dir / "layer_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "setting_summary.json",
            run_path=config.output_dir / "run.json",
            included_grids=2,
            layer_records=6,
        )

    monkeypatch.setattr(cli_module, "run_layerwise_kl_patching", fake_run)

    assert (
        main(
            [
                "layerwise-kl-patching",
                "--model",
                "test/model",
                "--benchmark",
                "gsm8k",
                "--pairs",
                "results/pairs.jsonl",
                "--targeting",
                "attribution-4",
                "--directions",
                "edited-to-clean",
                "clean-to-edited",
                "--output-dir",
                "results/kl",
                "--gpu-id",
                "0",
                "--limit",
                "1",
                "--resume",
            ]
        )
        == 0
    )

    assert captured == [
        LayerwiseKLPatchingConfig(
            model="test/model",
            benchmark="gsm8k",
            pairs=Path("results/pairs.jsonl"),
            targeting="attribution-4",
            directions=("clean-to-edited", "edited-to-clean"),
            output_dir=Path("results/kl"),
            gpu_id="0",
            limit=1,
            resume=True,
        )
    ]
    assert capsys.readouterr().out.splitlines() == [
        "wrote 6 layer record(s) from 2 complete grid(s): results/kl/layer_records.jsonl",
        "setting summary: results/kl/setting_summary.json",
        "run manifest: results/kl/run.json",
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"directions": ()}, "at least one direction"),
        (
            {"directions": ("clean-to-edited", "clean-to-edited")},
            "must not contain duplicates",
        ),
        ({"directions": ("clean-to-typo",)}, "unsupported direction"),
        ({"gpu_id": "0,1"}, "single non-negative integer"),
        ({"limit": 0}, "limit must be positive"),
    ),
)
def test_config_rejects_non_paper_or_ambiguous_arguments(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "pairs.jsonl", tmp_path / "out", **changes)


def test_normalized_kl_score_keeps_negative_values_and_exact_endpoints() -> None:
    assert normalized_kl_score(denominator_kl=2.0, patched_kl=0.0) == 1.0
    assert normalized_kl_score(denominator_kl=2.0, patched_kl=2.0) == 0.0
    assert normalized_kl_score(denominator_kl=2.0, patched_kl=3.0) == -0.5


@pytest.mark.parametrize(
    ("denominator", "reason"),
    (
        (0.0, "denominator_le_1e-9"),
        (KL_DENOMINATOR_EPSILON, "denominator_le_1e-9"),
        (float("inf"), "nonfinite_denominator"),
        (float("nan"), "nonfinite_denominator"),
    ),
)
def test_normalized_kl_score_rejects_invalid_untreated_denominators(
    denominator: float, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        normalized_kl_score(denominator_kl=denominator, patched_kl=0.5)


def test_kl_from_logits_matches_an_analytic_binary_distribution_and_is_asymmetric() -> None:
    p = torch.log(torch.tensor([0.5, 0.5]))
    q = torch.log(torch.tensor([0.75, 0.25]))

    expected_pq = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    expected_qp = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert kl_from_logits(p, q) == pytest.approx(expected_pq, abs=1e-7)
    assert kl_from_logits(q, p) == pytest.approx(expected_qp, abs=1e-7)
    assert kl_from_logits(p, q) != pytest.approx(kl_from_logits(q, p), abs=1e-4)


def test_setting_summary_uses_pair_medians_layer_centers_and_pair_first_thirds() -> None:
    scores = (
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        (0.2, 0.4, 0.6, 0.8, 1.0, 1.2),
        (0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
    )

    summary = summarize_direction(scores, bootstrap_resamples=100, seed=42)

    assert [row["relative_depth"] for row in summary["layer_profile"]] == pytest.approx(
        [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    )
    assert [row["layer_center_relative_depth"] for row in summary["layer_profile"]] == (
        pytest.approx([1 / 12, 3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12])
    )
    assert [row["median_normalized_kl"] for row in summary["layer_profile"]] == (
        pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    )
    thirds = {row["name"]: row for row in summary["depth_thirds"]}
    assert thirds["early"]["layer_indices"] == [0, 1]
    assert thirds["middle"]["layer_indices"] == [2, 3]
    assert thirds["late"]["layer_indices"] == [4, 5]
    assert thirds["early"]["setting_median_pair_mean"] == pytest.approx(0.3)
    assert thirds["middle"]["setting_median_pair_mean"] == pytest.approx(0.7)
    assert thirds["late"]["setting_median_pair_mean"] == pytest.approx(1.1)


def test_hsu_mcb_uses_paired_differences_and_retains_all_exact_ties() -> None:
    tied = ((0.5, 0.5, 0.1),) * 5

    summary = summarize_direction(tied, bootstrap_resamples=100, seed=42)

    assert summary["peak"] == {
        "layer_index": 0,
        "tied_layer_indices": [0, 1],
        "relative_depth": 0.0,
        "median_normalized_kl": 0.5,
    }
    assert summary["mcb"]["member_layer_indices"] == [0, 1]
    assert summary["mcb"]["bootstrap_resamples"] == 100
    assert summary["mcb"]["seed"] == 42


class _AddBlock(nn.Module):
    def __init__(self, amount: float, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.amount = amount
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, str]:
        value = hidden + self.amount
        return (value, "cache") if self.tuple_output else value


class _ToyDecoder(nn.Module):
    def __init__(self, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_AddBlock(1.0, tuple_output=tuple_output), _AddBlock(10.0)]
        )


class _ToyModel(nn.Module):
    def __init__(self, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.decoder = _ToyDecoder(tuple_output=tuple_output)
        self.config = SimpleNamespace(num_hidden_layers=2)

    def get_decoder(self) -> _ToyDecoder:
        return self.decoder

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor | tuple[torch.Tensor, str] = hidden
        for layer in self.decoder.layers:
            tensor = value[0] if isinstance(value, tuple) else value
            value = layer(tensor)
        return value[0] if isinstance(value, tuple) else value


def test_block_patch_changes_only_selected_position_and_recomputes_later_layers() -> None:
    model = _ToyModel()
    layers = find_decoder_layers(model)
    recipient = torch.zeros(1, 3, 1)
    donor = torch.tensor([[7.0]])

    untreated = model(recipient)
    with BlockOutputPatch(layers, layer_index=0, positions=(1,), donor_values=donor):
        patched = model(recipient)

    assert untreated[:, :, 0].tolist() == [[11.0, 11.0, 11.0]]
    # The layer-0 block output at token 1 is overwritten with 7, then layer 1
    # still adds 10. Other token positions retain their untreated values.
    assert patched[:, :, 0].tolist() == [[11.0, 17.0, 11.0]]
    assert recipient.tolist() == [[[0.0], [0.0], [0.0]]]


def test_capture_and_patch_preserve_tuple_outputs_and_remove_hooks_on_error() -> None:
    model = _ToyModel(tuple_output=True)
    layers = find_decoder_layers(model)
    hidden = torch.zeros(1, 3, 1)

    captured = capture_block_outputs(
        layers,
        positions=(0, 2),
        forward=lambda: model(hidden),
    )
    assert captured[0][:, 0].tolist() == [1.0, 1.0]
    assert captured[1][:, 0].tolist() == [11.0, 11.0]

    with pytest.raises(RuntimeError, match="stop"):
        with BlockOutputPatch(
            layers,
            layer_index=0,
            positions=(1,),
            donor_values=torch.tensor([[4.0]]),
        ):
            raise RuntimeError("stop")

    assert model(hidden)[:, :, 0].tolist() == [[11.0, 11.0, 11.0]]


def test_find_decoder_layers_rejects_an_ambiguous_or_mismatched_stack() -> None:
    model = _ToyModel()
    model.config.num_hidden_layers = 3
    with pytest.raises(ValueError, match="num_hidden_layers"):
        find_decoder_layers(model)

    ambiguous = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1),
        named_modules=lambda: iter((('vision.layers', nn.ModuleList([_AddBlock(1.0)])),)),
    )
    with pytest.raises(ValueError, match="text decoder"):
        find_decoder_layers(ambiguous)


def test_runner_selects_clean_correct_edited_wrong_and_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    pairs_path = _write_pair_source(
        tmp_path / "source",
        [
            _pair("c-not-failure", edited_correct=True),
            _pair("a-included"),
            _pair("b-no-alignment", aligned=False),
            _pair("d-clean-wrong", clean_correct=False),
        ],
    )
    output_dir = tmp_path / "output"
    runtime = FakeRuntime(n_layers=3)

    result = run_layerwise_kl_patching(_config(pairs_path, output_dir), runtime=runtime)

    assert result.included_grids == 2
    assert result.layer_records == 6
    assert runtime.calls == [
        ("a-included", ("clean-to-edited", "edited-to-clean")),
    ]
    rows = _read_jsonl(result.layer_records_path)
    assert [(row["sample_id"], row["direction"], row["layer_index"]) for row in rows] == [
        ("a-included", "clean-to-edited", 0),
        ("a-included", "clean-to-edited", 1),
        ("a-included", "clean-to-edited", 2),
        ("a-included", "edited-to-clean", 0),
        ("a-included", "edited-to-clean", 1),
        ("a-included", "edited-to-clean", 2),
    ]
    assert [row["normalized_kl"] for row in rows[:3]] == pytest.approx([0.5, 0.6, 0.7])
    assert all(row["schema_version"] == "layerwise-kl-patching-layer/v1" for row in rows)

    statuses = _read_jsonl(result.pair_status_records_path)
    assert {(row["sample_id"], row["direction"], row["status"]) for row in statuses} == {
        ("a-included", "clean-to-edited", "included"),
        ("a-included", "edited-to-clean", "included"),
    }
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["population"]["input_pairs"] == 4
    assert summary["population"]["selected_failures"] == 2
    assert summary["population"]["aligned_selected_failures"] == 1
    assert summary["population"]["upstream_exclusions"] == {
        "clean_not_correct": 1,
        "edited_not_wrong": 1,
        "no_aligned_words": 1,
    }
    assert summary["directions"]["clean-to-edited"]["included_pairs"] == 1

    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["paper_sha256"] == PAPER_SHA256
    assert run["comparability"]["status"] == "fresh-paper-protocol-reproduction"
    assert set(run["outputs"]) == {
        "layer_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    }


def test_direction_validity_is_independent_and_nonfinite_layer_excludes_complete_grid(
    tmp_path: Path,
) -> None:
    pairs_path = _write_pair_source(tmp_path / "source", [_pair("sample")])
    runtime = FakeRuntime(
        scans={
            "sample": {
                "clean-to-edited": DirectionScan(
                    denominator_kl=KL_DENOMINATOR_EPSILON,
                    patched_kl_by_layer=(),
                ),
                "edited-to-clean": DirectionScan(
                    denominator_kl=2.0,
                    patched_kl_by_layer=(1.0, float("nan"), 0.5),
                ),
            }
        }
    )

    result = run_layerwise_kl_patching(
        _config(pairs_path, tmp_path / "output"), runtime=runtime
    )

    assert result.included_grids == 0
    assert result.layer_records == 0
    statuses = _read_jsonl(result.pair_status_records_path)
    assert [(row["direction"], row["exclusion_reason"]) for row in statuses] == [
        ("clean-to-edited", "denominator_le_1e-9"),
        ("edited-to-clean", "nonfinite_layer_value"),
    ]
    assert "NaN" not in result.pair_status_records_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda manifest, pair: manifest.update(status="failed"), "source run is not completed"),
        (lambda manifest, pair: manifest.update(paper_sha256="bad"), "paper SHA-256"),
        (
            lambda manifest, pair: manifest["arguments"].update(num_edits=2),
            "four edits",
        ),
        (lambda manifest, pair: pair.update(model="other/model"), "record model"),
        (
            lambda manifest, pair: pair["aligned_words"][0].update(clean_final_token=2),
            "clean_final_token",
        ),
    ),
)
def test_source_contract_errors_fail_before_runtime_loading(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    pairs_path = _write_pair_source(tmp_path / "source", [_pair("sample")])
    manifest = json.loads((pairs_path.parent / "run.json").read_text(encoding="utf-8"))
    pair = json.loads(pairs_path.read_text(encoding="utf-8"))
    mutation(manifest, pair)  # type: ignore[operator]
    _write_json(pairs_path.parent / "run.json", manifest)
    pairs_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    runtime = FakeRuntime()

    with pytest.raises(ValueError, match=message):
        run_layerwise_kl_patching(_config(pairs_path, tmp_path / "out"), runtime=runtime)
    assert runtime.calls == []


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(tmp_path: Path) -> None:
    pairs_path = _write_pair_source(tmp_path / "source", [_pair("sample")])
    original = pairs_path.read_text(encoding="utf-8").strip()

    pairs_path.write_text(original[:-1] + ',"sample_id":"duplicate"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        run_layerwise_kl_patching(
            _config(pairs_path, tmp_path / "duplicate-out"), runtime=FakeRuntime()
        )

    pairs_path.write_text(original[:-1] + ',"unexpected":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        run_layerwise_kl_patching(
            _config(pairs_path, tmp_path / "nan-out"), runtime=FakeRuntime()
        )


def test_runtime_failure_keeps_checkpoints_and_resume_skips_completed_pairs(
    tmp_path: Path,
) -> None:
    pairs_path = _write_pair_source(
        tmp_path / "source", [_pair("a-complete"), _pair("b-fails")]
    )
    output_dir = tmp_path / "output"
    config = _config(pairs_path, output_dir)
    first_runtime = FakeRuntime(error_for="b-fails")

    with pytest.raises(LayerwiseKLPatchingRunError, match="1 pair"):
        run_layerwise_kl_patching(config, runtime=first_runtime)

    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert not (output_dir / "layer_records.jsonl").exists()
    assert (output_dir / ".layerwise-kl-patching-work").is_dir()

    resumed_runtime = FakeRuntime()
    result = run_layerwise_kl_patching(
        replace(config, resume=True),
        runtime=resumed_runtime,
    )
    assert resumed_runtime.calls == [
        ("b-fails", ("clean-to-edited", "edited-to-clean")),
    ]
    assert result.included_grids == 4


def test_completed_resume_verifies_outputs_without_calling_runtime(tmp_path: Path) -> None:
    pairs_path = _write_pair_source(tmp_path / "source", [_pair("sample")])
    output_dir = tmp_path / "output"
    config = _config(pairs_path, output_dir)
    result = run_layerwise_kl_patching(config, runtime=FakeRuntime())
    runtime = FakeRuntime(error_for="sample")

    resumed = run_layerwise_kl_patching(replace(config, resume=True), runtime=runtime)

    assert resumed == result
    assert runtime.calls == []


def test_limit_is_applied_after_failure_cohort_selection_and_marks_partial(tmp_path: Path) -> None:
    pairs_path = _write_pair_source(
        tmp_path / "source",
        [_pair("a-not-selected", edited_correct=True), _pair("b-first"), _pair("c-second")],
    )
    runtime = FakeRuntime()

    result = run_layerwise_kl_patching(
        _config(pairs_path, tmp_path / "output", limit=1),
        runtime=runtime,
    )

    assert runtime.calls == [("b-first", ("clean-to-edited", "edited-to-clean"))]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["comparability"]["status"] == "partial-smoke-run"
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["population"]["eligible_before_limit"] == 2
    assert summary["population"]["selected_by_limit"] == 1


def test_nonempty_output_requires_resume_and_resume_arguments_are_frozen(tmp_path: Path) -> None:
    pairs_path = _write_pair_source(tmp_path / "source", [_pair("sample")])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "unrelated").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        run_layerwise_kl_patching(_config(pairs_path, output_dir), runtime=FakeRuntime())

    output_dir = tmp_path / "completed"
    config = _config(pairs_path, output_dir)
    run_layerwise_kl_patching(config, runtime=FakeRuntime())
    with pytest.raises(ValueError, match="resume arguments do not match"):
        run_layerwise_kl_patching(
            replace(config, directions=("clean-to-edited",), resume=True),
            runtime=FakeRuntime(),
        )
