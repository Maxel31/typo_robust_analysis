"""Paper-contract tests for the alternative patch-position reachability scan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.layerwise_kl_patching.runner import (
    DirectionScan,
    LayerwiseKLPatchingConfig,
    PairScan,
    run_layerwise_kl_patching,
)
from typo_cot.experiments.patch_position_controls import (
    POSITION_NAMES,
    AlternativePositionScan,
    HuggingFacePositionControlRuntime,
    PositionControlConfig,
    PositionControlPairScan,
    PositionControlResult,
    PositionControlRunError,
    PositionCoordinates,
    locate_position_coordinates,
    run_patch_position_controls,
)
from typo_cot.experiments.patch_position_controls.runtime import (
    POSITION_RUNTIME_PROTOCOL,
)


MODEL = "google/gemma-3-4b-it"
OFFSETS = ((0, 0), (0, 2), (3, 9), (9, 10), (10, 12))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pair(sample_id: str) -> dict[str, object]:
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": MODEL,
        "benchmark": "gsm8k",
        "targeting": "attribution-4",
        "seed": 42,
        "num_edits_requested": 4,
        "num_aligned_words": 1,
        "clean": {
            "prompt": "Q: alpha?\nA:",
            "prompt_token_count": 5,
            "editable_text": "alpha?",
            "editable_prompt_span": {"start": 3, "end": 9},
            "continuation": "reasoning",
            "answer": {
                "value": "2",
                "is_extracted": True,
                "is_correct": True,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "edited": {
            "prompt": "Q: alhpa?\nA:",
            "prompt_token_count": 5,
            "editable_text": "alhpa?",
            "editable_prompt_span": {"start": 3, "end": 9},
            "continuation": "reasoning",
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
            {
                "word_index": 0,
                "clean_text": "alpha",
                "edited_text": "alhpa",
                "clean_editable_span": {"start": 0, "end": 5},
                "edited_editable_span": {"start": 0, "end": 5},
                "clean_prompt_span": {"start": 3, "end": 8},
                "edited_prompt_span": {"start": 3, "end": 8},
                "target_ranks": [1],
                "target_token_indices": [1],
                "clean_token_indices": [2],
                "edited_token_indices": [2],
                "clean_final_token": 2,
                "edited_final_token": 2,
            }
        ],
    }


def _write_pair_source(root: Path, pairs: list[dict[str, object]]) -> Path:
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
                "model": MODEL,
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
            "provenance": {
                "model": MODEL,
                "model_revision": "source-revision",
                "random_seed_algorithm": "sha256-first-64-bits/v1",
            },
        },
    )
    return pairs_path


class _LayerwiseRuntime:
    def __init__(
        self,
        *,
        n_layers: int = 3,
        denominators: dict[str, float] | None = None,
    ) -> None:
        self.num_layers = n_layers
        self.denominators = denominators or {}

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "layerwise-fixture",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "decoder_adapter": "fixture.layers",
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "device": "cuda:0",
        }

    def scan_pair(self, pair: dict[str, object], directions: tuple[str, ...]) -> PairScan:
        index = 0 if pair["sample_id"] == "a" else 1
        denominator = self.denominators.get(str(pair["sample_id"]), 2.0 + 2.0 * index)
        patched = tuple(denominator * ratio for ratio in (0.75, 0.25, 0.5))
        return PairScan(
            sample_id=str(pair["sample_id"]),
            directions={
                name: DirectionScan(
                    denominator_kl=denominator,
                    patched_kl_by_layer=patched,
                )
                for name in directions
            },
        )


def _reference_run(
    root: Path,
    *,
    sample_ids: tuple[str, ...] = ("a", "b"),
    denominators: dict[str, float] | None = None,
) -> Path:
    pairs = _write_pair_source(root / "pairs", [_pair(name) for name in sample_ids])
    result = run_layerwise_kl_patching(
        LayerwiseKLPatchingConfig(
            model=MODEL,
            benchmark="gsm8k",
            pairs=pairs,
            targeting="attribution-4",
            directions=("clean-to-edited",),
            output_dir=root / "layerwise",
        ),
        runtime=_LayerwiseRuntime(denominators=denominators),
    )
    return result.run_path.parent


class _PositionRuntime:
    def __init__(
        self,
        *,
        n_layers: int = 3,
        mismatch_for: str | None = None,
        fail_for: str | None = None,
        nonfinite_for: str | None = None,
        denominators: dict[str, float] | None = None,
    ) -> None:
        self.num_layers = n_layers
        self.mismatch_for = mismatch_for
        self.fail_for = fail_for
        self.nonfinite_for = nonfinite_for
        self.denominators = denominators or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "position-fixture",
            "operation": "patch-position-controls",
            "model": MODEL,
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "decoder_adapter": "fixture.layers",
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "cuda_visible_devices": "0",
            "position_controls": POSITION_RUNTIME_PROTOCOL,
        }

    def scan_pair(
        self, pair: dict[str, object], positions: tuple[str, ...]
    ) -> PositionControlPairScan:
        sample_id = str(pair["sample_id"])
        self.calls.append((sample_id, positions))
        if sample_id == self.fail_for:
            raise RuntimeError("synthetic GPU failure")
        denominator = self.denominators.get(sample_id, 2.0 if sample_id == "a" else 4.0)
        if sample_id == self.mismatch_for:
            denominator += 0.1
        scans: dict[str, AlternativePositionScan] = {}
        for position in positions:
            if position == "prompt-final":
                ratios = (1.0, 0.5, 0.0)
                coordinates = PositionCoordinates((4,), (4,), "prompt-final-token/v1")
            elif position == "question-final":
                ratios = (1.1, 1.0, 0.9)
                coordinates = PositionCoordinates(
                    (2,), (2,), "editable-prompt-span-final-overlap/v1"
                )
            else:
                raise AssertionError(position)
            values = [denominator * ratio for ratio in ratios]
            if sample_id == self.nonfinite_for:
                values[1] = float("nan")
            scans[position] = AlternativePositionScan(
                coordinates=coordinates,
                patched_kl_by_layer=tuple(values),
            )
        return PositionControlPairScan(
            sample_id=sample_id,
            denominator_kl=denominator,
            positions=scans,
        )


def _config(reference: Path, output: Path, **changes: object) -> PositionControlConfig:
    config = PositionControlConfig(
        model=MODEL,
        benchmark="gsm8k",
        layerwise_kl_run=reference,
        positions=("edited-word", "prompt-final", "question-final"),
        output_dir=output,
    )
    return replace(config, **changes)


def test_catalog_and_cli_expose_the_completed_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = get_experiment("patch-position-controls")
    assert spec.status == "implemented"
    assert spec.cohort.startswith("Gemma-3-4B/GSM8K Attribution-4 layerwise-KL")
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--layerwise-kl-run",
        "--positions",
        "--gpu-id",
        "--output-dir",
    )
    assert spec.outputs == (
        "position_control_records.jsonl",
        "pair_status_records.jsonl",
        "position_control_summary.json",
        "run.json",
    )

    captured: list[PositionControlConfig] = []

    def fake_run(config: PositionControlConfig) -> PositionControlResult:
        captured.append(config)
        return PositionControlResult(
            records_path=config.output_dir / "position_control_records.jsonl",
            pair_status_records_path=config.output_dir / "pair_status_records.jsonl",
            summary_path=config.output_dir / "position_control_summary.json",
            run_path=config.output_dir / "run.json",
            pairs=1,
            records=9,
        )

    monkeypatch.setattr(cli_module, "run_patch_position_controls", fake_run)
    assert (
        main(
            [
                "patch-position-controls",
                "--model",
                MODEL,
                "--benchmark",
                "gsm8k",
                "--layerwise-kl-run",
                str(tmp_path / "kl"),
                "--positions",
                "question-final",
                "edited-word",
                "prompt-final",
                "--gpu-id",
                "0",
                "--limit",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--resume",
            ]
        )
        == 0
    )
    assert captured == [
        PositionControlConfig(
            model=MODEL,
            benchmark="gsm8k",
            layerwise_kl_run=tmp_path / "kl",
            positions=POSITION_NAMES,
            gpu_id="0",
            limit=1,
            output_dir=tmp_path / "out",
            resume=True,
        )
    ]
    assert (
        capsys.readouterr()
        .out.splitlines()[0]
        .startswith("wrote 9 position-control record(s) for 1 pair(s):")
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"positions": ()}, "at least one position"),
        ({"positions": ("prompt-final", "prompt-final")}, "must not contain duplicates"),
        ({"positions": ("answer-final",)}, "unsupported position"),
        ({"benchmark": "mmlu"}, "only supports gsm8k"),
        ({"gpu_id": "0,1"}, "single non-negative integer"),
        ({"limit": 0}, "limit must be positive"),
    ),
)
def test_config_rejects_non_paper_or_ambiguous_arguments(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "source", tmp_path / "out", **changes)


def test_position_locator_uses_recorded_spans_and_validates_alignment() -> None:
    coordinates = locate_position_coordinates(_pair("a"), OFFSETS, OFFSETS)

    assert coordinates == {
        "edited-word": PositionCoordinates((2,), (2,), "aligned-edited-word-final-tokens/v1"),
        "prompt-final": PositionCoordinates((4,), (4,), "prompt-final-token/v1"),
        "question-final": PositionCoordinates((2,), (2,), "editable-prompt-span-final-overlap/v1"),
    }

    malformed = _pair("bad")
    malformed["edited"]["editable_prompt_span"] = {"start": 20, "end": 21}  # type: ignore[index]
    with pytest.raises(ValueError, match="edited.editable_prompt_span"):
        locate_position_coordinates(malformed, OFFSETS, OFFSETS)

    misaligned = _pair("bad")
    misaligned["aligned_words"][0]["clean_final_token"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="clean_final_token"):
        locate_position_coordinates(misaligned, OFFSETS, OFFSETS)


def test_runner_reuses_reference_arm_and_writes_common_complete_grids(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source")
    runtime = _PositionRuntime()

    result = run_patch_position_controls(_config(reference, tmp_path / "output"), runtime=runtime)

    assert result.pairs == 2
    assert result.records == 18
    assert runtime.calls == [
        ("a", ("prompt-final", "question-final")),
        ("b", ("prompt-final", "question-final")),
    ]
    rows = _read_jsonl(result.records_path)
    assert [(row["sample_id"], row["position"], row["layer_index"]) for row in rows] == [
        (sample_id, position, layer)
        for sample_id in ("a", "b")
        for position in POSITION_NAMES
        for layer in range(3)
    ]
    assert {row["denominator_kl"] for row in rows if row["sample_id"] == "a"} == {2.0}
    assert {row["denominator_kl"] for row in rows if row["sample_id"] == "b"} == {4.0}
    assert {row["execution"] for row in rows if row["position"] == "edited-word"} == {
        "verified-reference"
    }
    assert {row["execution"] for row in rows if row["position"] != "edited-word"} == {"executed"}
    assert [
        row["normalized_kl"]
        for row in rows
        if row["sample_id"] == "a" and row["position"] == "edited-word"
    ] == pytest.approx([0.25, 0.75, 0.5])

    statuses = _read_jsonl(result.pair_status_records_path)
    assert [row["sample_id"] for row in statuses] == ["a", "b"]
    assert all(row["status"] == "included" for row in statuses)
    assert statuses[0]["runtime_denominator_delta"] == pytest.approx(0.0)
    assert set(statuses[0]["positions"]) == set(POSITION_NAMES)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["design_status"] == "post_hoc_exploratory_descriptive"
    assert summary["population"]["source_common_pairs"] == 2
    assert summary["population"]["executed_common_pairs"] == 2
    assert (
        summary["population"]["source_id_fingerprint"]
        == summary["population"]["executed_id_fingerprint"]
    )
    assert [
        row["median_normalized_kl"] for row in summary["positions"]["prompt-final"]["layer_profile"]
    ] == pytest.approx([0.0, 0.5, 1.0])
    assert summary["positions"]["prompt-final"]["peak"] == {
        "layer_index": 2,
        "tied_layer_indices": [2],
        "relative_depth": pytest.approx(2 / 3),
        "median_normalized_kl": pytest.approx(1.0),
    }
    assert "mcb" not in summary["positions"]["prompt-final"]
    assert summary["published_reference"]["edited-word"]["table5_mcb_source_metadata"] == "8*"
    assert summary["interpretation"]["new_three_position_mcb"] is False
    assert summary["interpretation"]["edited_word_table5_mcb_preserved_as_source_metadata"] is True
    assert summary["interpretation"]["claim"] == "intervention-reachability"
    assert summary["published_reference"] == {
        "cohort_pairs": 109,
        "edited-word": {
            "peak_layer": 2,
            "peak_normalized_kl_rounded": 0.751,
            "relative_depth_rounded": 0.059,
            "table5_mcb_source_metadata": "8*",
        },
        "prompt-final": {"layer_values_rounded": {"0": 0.001, "16": 0.25, "32": 0.98}},
        "question-final": {
            "all_layer_peak_normalized_kl_rounded": 0.011,
            "peak_layer_published": False,
        },
    }
    assert {
        key: summary["protocol"][key]
        for key in (
            "source_direction",
            "kl_orientation",
            "score",
            "layer_grid",
            "layer_profile",
        )
    } == {
        "source_direction": "clean-to-edited",
        "kl_orientation": "clean-to-patched-edited",
        "score": "1-KL(clean||patched-edited)/KL(clean||edited)",
        "layer_grid": "all-decoder-layers-0-through-L-minus-1",
        "layer_profile": "across-pair-median",
    }
    assert {row["direction"] for row in rows} == {"clean-to-edited"}

    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["paper_sha256"] == PAPER_SHA256
    assert run["comparability"]["status"] == "fresh-paper-protocol-reproduction"
    assert set(run["outputs"]) == {
        "position_control_records.jsonl",
        "pair_status_records.jsonl",
        "position_control_summary.json",
    }


def test_edited_word_only_uses_verified_source_without_loading_runtime(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))

    result = run_patch_position_controls(
        _config(reference, tmp_path / "output", positions=("edited-word",)),
        runtime=None,
    )

    assert result.records == 3
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["runtime"] is None
    assert run["comparability"]["status"] == "partial-smoke-run"


def test_source_hash_and_grid_contract_fail_before_runtime_calls(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))
    layer_path = reference / "layer_records.jsonl"
    rows = _read_jsonl(layer_path)
    rows.pop()
    layer_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    manifest["outputs"]["layer_records.jsonl"]["sha256"] = _sha256(layer_path)
    manifest["outputs"]["layer_records.jsonl"]["records"] = len(rows)
    manifest["counts"]["layer_records"] = len(rows)
    _write_json(reference / "run.json", manifest)
    runtime = _PositionRuntime()

    with pytest.raises(ValueError, match="complete layer grid"):
        run_patch_position_controls(_config(reference, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_unupdated_source_output_hash_fails_before_runtime_calls(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))
    with (reference / "layer_records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    runtime = _PositionRuntime()

    with pytest.raises(ValueError, match="source output is missing or has changed"):
        run_patch_position_controls(_config(reference, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_source_pair_discovery_count_must_match_failure_free_output(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))
    source_run_path = reference / "run.json"
    source_run = json.loads(source_run_path.read_text(encoding="utf-8"))
    pair_run_path = Path(source_run["input"]["pairs_path"]).parent / "run.json"
    pair_run = json.loads(pair_run_path.read_text(encoding="utf-8"))
    pair_run["counts"]["discovered"] = 2
    _write_json(pair_run_path, pair_run)
    source_run["input"]["source_run_sha256"] = _sha256(pair_run_path)
    _write_json(source_run_path, source_run)
    runtime = _PositionRuntime()

    with pytest.raises(ValueError, match="discovered count"):
        run_patch_position_controls(_config(reference, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_source_cannot_upgrade_fresh_outputs_to_exact_historical_provenance(
    tmp_path: Path,
) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))
    run_path = reference / "run.json"
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["comparability"].update(
        status="exact-historical-reproduction",
        exact_historical_table5_ids=True,
    )
    _write_json(run_path, manifest)
    runtime = _PositionRuntime()

    with pytest.raises(ValueError, match="fresh protocol reproduction"):
        run_patch_position_controls(_config(reference, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_source_cannot_silently_drop_an_eligible_pair_from_every_grid(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source")
    status_path = reference / "pair_status_records.jsonl"
    layer_path = reference / "layer_records.jsonl"
    summary_path = reference / "setting_summary.json"
    _write_jsonl(
        status_path,
        [row for row in _read_jsonl(status_path) if row["sample_id"] == "a"],
    )
    _write_jsonl(
        layer_path,
        [row for row in _read_jsonl(layer_path) if row["sample_id"] == "a"],
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["population"].update(
        selected_failures=1,
        aligned_selected_failures=1,
        eligible_before_limit=1,
        selected_by_limit=1,
    )
    summary["directions"]["clean-to-edited"].update(
        included_pairs=1,
        excluded_pairs=0,
    )
    _write_json(summary_path, summary)
    manifest = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    manifest["counts"].update(
        eligible_pairs=1,
        selected_pairs=1,
        checkpointed_pairs=1,
        included_grids=1,
        layer_records=3,
    )
    for name, records in (
        ("pair_status_records.jsonl", 1),
        ("layer_records.jsonl", 3),
        ("setting_summary.json", None),
    ):
        manifest["outputs"][name]["sha256"] = _sha256(reference / name)
        if records is not None:
            manifest["outputs"][name]["records"] = records
    _write_json(reference / "run.json", manifest)
    runtime = _PositionRuntime()

    with pytest.raises(ValueError, match="source status coverage"):
        run_patch_position_controls(_config(reference, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda pair: pair["clean"]["answer"].__setitem__("is_correct", False),
            "source status coverage",
        ),
        (
            lambda pair: pair["aligned_words"][0].__setitem__("clean_token_indices", [999, 2]),
            "valid token indices",
        ),
        (
            lambda pair: pair["edited"].__setitem__("editable_text", "wrong"),
            "editable_text does not match",
        ),
    ),
)
def test_source_included_pair_semantics_are_revalidated_after_hashes(
    tmp_path: Path, mutation: object, message: str
) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))
    source_run = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    pairs_path = Path(source_run["input"]["pairs_path"])
    pair = _read_jsonl(pairs_path)[0]
    mutation(pair)  # type: ignore[operator]
    line = json.dumps(pair, ensure_ascii=False, sort_keys=True)
    pairs_path.write_text(line + "\n", encoding="utf-8")
    pair_fingerprint = hashlib.sha256(line.encode("utf-8")).hexdigest()
    source_run["input"]["pairs_sha256"] = _sha256(pairs_path)

    status_path = reference / "pair_status_records.jsonl"
    statuses = _read_jsonl(status_path)
    statuses[0]["source_record_sha256"] = pair_fingerprint
    _write_jsonl(status_path, statuses)
    layer_path = reference / "layer_records.jsonl"
    layers = _read_jsonl(layer_path)
    for row in layers:
        row["source_record_sha256"] = pair_fingerprint
    _write_jsonl(layer_path, layers)
    source_run["outputs"]["pair_status_records.jsonl"]["sha256"] = _sha256(status_path)
    source_run["outputs"]["layer_records.jsonl"]["sha256"] = _sha256(layer_path)
    _write_json(reference / "run.json", source_run)
    runtime = _PositionRuntime()

    with pytest.raises(ValueError, match=message):
        run_patch_position_controls(_config(reference, tmp_path / "output"), runtime=runtime)
    assert runtime.calls == []


def test_pair_atomic_failure_keeps_prior_checkpoint_and_resume_skips_it(
    tmp_path: Path,
) -> None:
    reference = _reference_run(tmp_path / "source")
    output = tmp_path / "output"
    config = _config(reference, output)
    first_runtime = _PositionRuntime(mismatch_for="b")

    with pytest.raises(PositionControlRunError, match="pair b failed"):
        run_patch_position_controls(config, runtime=first_runtime)

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert not (output / "position_control_records.jsonl").exists()
    checkpoints = list((output / ".patch-position-controls-work" / "checkpoints").iterdir())
    assert len(checkpoints) == 1

    resumed_runtime = _PositionRuntime()
    result = run_patch_position_controls(replace(config, resume=True), runtime=resumed_runtime)
    assert result.records == 18
    assert resumed_runtime.calls == [("b", ("prompt-final", "question-final"))]


def test_resume_rejects_a_changed_registered_checkpoint(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source")
    output = tmp_path / "output"
    config = _config(reference, output)

    with pytest.raises(PositionControlRunError):
        run_patch_position_controls(config, runtime=_PositionRuntime(fail_for="b"))
    checkpoint = next((output / ".patch-position-controls-work" / "checkpoints").iterdir())
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    runtime = _PositionRuntime()
    with pytest.raises(ValueError, match="registered checkpoint is missing or has changed"):
        run_patch_position_controls(replace(config, resume=True), runtime=runtime)
    assert runtime.calls == []


def test_nonfinite_grid_fails_pair_atomically(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))

    with pytest.raises(PositionControlRunError, match="nonfinite"):
        run_patch_position_controls(
            _config(reference, tmp_path / "output"),
            runtime=_PositionRuntime(nonfinite_for="a"),
        )
    assert not (tmp_path / "output" / "position_control_records.jsonl").exists()


def test_near_threshold_denominator_requires_a_tight_runtime_match(tmp_path: Path) -> None:
    source_denominator = 1.1e-9
    reference = _reference_run(
        tmp_path / "source",
        sample_ids=("a",),
        denominators={"a": source_denominator},
    )

    with pytest.raises(PositionControlRunError, match="does not match the fixed source"):
        run_patch_position_controls(
            _config(reference, tmp_path / "output"),
            runtime=_PositionRuntime(denominators={"a": source_denominator + 2e-12}),
        )


def test_completed_resume_verifies_outputs_without_consulting_runtime(tmp_path: Path) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))
    output = tmp_path / "output"
    config = _config(reference, output)
    expected = run_patch_position_controls(config, runtime=_PositionRuntime())
    runtime = _PositionRuntime(fail_for="a")

    resumed = run_patch_position_controls(replace(config, resume=True), runtime=runtime)

    assert resumed == expected
    assert runtime.calls == []

    with expected.records_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="completed output is missing or has changed"):
        run_patch_position_controls(replace(config, resume=True), runtime=runtime)
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda run: run.__setitem__("operation", "wrong-operation"), "wrong operation"),
        (
            lambda run: run["counts"].__setitem__("records", -999),
            "non-negative integer",
        ),
        (
            lambda run: run["runtime"].__setitem__("dtype", "float32"),
            "provenance field 'dtype'",
        ),
    ),
)
def test_completed_resume_rejects_semantically_modified_manifest(
    tmp_path: Path, mutation: object, message: str
) -> None:
    reference = _reference_run(tmp_path / "source", sample_ids=("a",))
    output = tmp_path / "output"
    config = _config(reference, output)
    run_patch_position_controls(config, runtime=_PositionRuntime())
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    mutation(run)  # type: ignore[operator]
    _write_json(run_path, run)
    runtime = _PositionRuntime(fail_for="a")

    with pytest.raises(ValueError, match=message):
        run_patch_position_controls(replace(config, resume=True), runtime=runtime)
    assert runtime.calls == []


class _AddBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1.0


class _LogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _AddBlock()

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.layer(input_ids.float().unsqueeze(-1))
        return SimpleNamespace(logits=torch.cat((hidden, -hidden), dim=-1))


def test_runtime_reads_logits_after_the_last_block_patch() -> None:
    runtime = object.__new__(HuggingFacePositionControlRuntime)
    runtime.layers = [_AddBlock()]
    runtime.model = _LogitModel()
    # Patch the actual block owned by the model, not the independent fixture above.
    runtime.layers = [runtime.model.layer]
    ids = torch.zeros((1, 3), dtype=torch.long)
    mask = torch.ones_like(ids)

    prompt_final = runtime._patched_logits(
        input_ids=ids,
        attention_mask=mask,
        layer_index=0,
        positions=(2,),
        donor_values=torch.tensor([[7.0]]),
    )
    question_final = runtime._patched_logits(
        input_ids=ids,
        attention_mask=mask,
        layer_index=0,
        positions=(1,),
        donor_values=torch.tensor([[7.0]]),
    )

    assert prompt_final.tolist() == [7.0, -7.0]
    assert question_final.tolist() == [1.0, -1.0]


def test_runtime_scans_both_arms_from_one_union_capture_and_one_recipient_baseline() -> None:
    from typo_cot.experiments.layerwise_kl_patching.metrics import kl_from_logits

    runtime = object.__new__(HuggingFacePositionControlRuntime)
    runtime.num_layers = 2
    runtime._torch = torch
    events: list[object] = []
    clean_logits = torch.tensor([2.0, 0.0])
    edited_logits = torch.tensor([0.0, 2.0])

    def tokenize(
        self: object, pair: object, *, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
        del self, pair
        events.append(("tokenize", side))
        ids = torch.zeros((1, len(OFFSETS)), dtype=torch.long)
        return ids, torch.ones_like(ids), OFFSETS

    def untreated(
        self: object,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: tuple[int, ...],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del self, input_ids, attention_mask
        events.append(("clean-untreated", positions))
        return clean_logits, [torch.tensor([[40.0 + layer], [20.0 + layer]]) for layer in range(2)]

    def plain(
        self: object, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        del self, input_ids, attention_mask
        events.append("edited-untreated")
        return edited_logits

    def patched(
        self: object,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_index: int,
        positions: tuple[int, ...],
        donor_values: torch.Tensor,
    ) -> torch.Tensor:
        del self, input_ids, attention_mask
        events.append(("patched", layer_index, positions, donor_values.flatten().tolist()))
        if positions == (4,):
            return torch.tensor([1.0 + layer_index, 1.0 - layer_index])
        return torch.tensor([0.5 * layer_index, 2.0 - 0.5 * layer_index])

    runtime._tokenize_for_positions = MethodType(tokenize, runtime)
    runtime._untreated = MethodType(untreated, runtime)
    runtime._plain_logits = MethodType(plain, runtime)
    runtime._patched_logits = MethodType(patched, runtime)

    scan = runtime.scan_pair(_pair("a"), ("prompt-final", "question-final"))

    assert scan.denominator_kl == pytest.approx(
        kl_from_logits(clean_logits, edited_logits), abs=1e-12
    )
    assert [event for event in events if event == "edited-untreated"] == ["edited-untreated"]
    assert ("clean-untreated", (4, 2)) in events
    assert [event for event in events if isinstance(event, tuple) and event[0] == "patched"] == [
        ("patched", 0, (4,), [40.0]),
        ("patched", 1, (4,), [41.0]),
        ("patched", 0, (2,), [20.0]),
        ("patched", 1, (2,), [21.0]),
    ]
    assert set(scan.positions) == {"prompt-final", "question-final"}
    assert all(len(result.patched_kl_by_layer) == 2 for result in scan.positions.values())
