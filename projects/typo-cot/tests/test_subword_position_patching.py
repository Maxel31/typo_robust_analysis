"""Contracts for first/final/all-subword answer patching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
from typo_cot.experiments.build_rebuttal_manifest import REBUTTAL_SETTINGS
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.subword_position_patching.planning import (
    plan_subword_patch,
)
from typo_cot.experiments.subword_position_patching.protocol import (
    load_subword_position_patching_protocol,
)
from typo_cot.experiments.subword_position_patching.runner import (
    SubwordGeneration,
    SubwordModeScan,
    SubwordPairScan,
    SubwordPositionPatchingConfig,
    SubwordPositionPatchingRunError,
    run_subword_position_patching,
)
from typo_cot.experiments.subword_position_patching.runtime import (
    HuggingFaceSubwordPositionPatchingRuntime,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "subword-position-patching.yaml"
PRIMARY = REBUTTAL_SETTINGS[0]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _edit(clean: list[int], typo: list[int]) -> dict[str, object]:
    return {
        "clean_char_span": [0, 1],
        "typo_char_span": [0, 1],
        "clean_token_indices": clean,
        "typo_token_indices": typo,
        "clean_word_final_token": clean[-1],
        "typo_word_final_token": typo[-1],
    }


def _record(index: int) -> dict[str, object]:
    edits = (
        [_edit([5, 6], [7, 8]), _edit([10], [12])]
        if index < 100
        else [_edit([5, 6], [7, 8, 9]), _edit([10, 11, 12], [13])]
    )
    pair_id = _digest(f"primary\0{index}")
    return {
        "pair_id": pair_id,
        "sample_id": f"sample-{index:04d}",
        "model": PRIMARY.model,
        "task": PRIMARY.task,
        "target_rule": "attribution-4",
        "gold_answer": "42",
        "clean_answer": "42",
        "typo_answer": "99",
        "clean_correct": True,
        "typo_correct": False,
        "clean_text": "clean prompt",
        "typo_text": "typo prompt",
        "clean_prompt_token_count": 20,
        "typo_prompt_token_count": 21,
        "number_of_aligned_words": len(edits),
        "edits": edits,
        "cohorts": {"restoration": True},
        "fixed_window": {"event": index % 4 == 0},
        "source": {
            "source_record_sha256": _digest(pair_id),
            "model_revision": _digest(PRIMARY.model),
        },
    }


def _records() -> tuple[dict[str, object], ...]:
    return tuple(_record(index) for index in range(PRIMARY.paper_denominator))


class _Runtime:
    num_layers = 12
    calls = 0

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        self.model = model
        self.task = task
        self.revision = revision
        self.gpu_id = gpu_id

    def provenance(self) -> Mapping[str, object]:
        return {
            "operation": "subword-position-patching",
            "runtime": "subword-position-fixture",
            "model": self.model,
            "task": self.task,
            "requested_revision": self.revision,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "cuda_visible_devices": self.gpu_id,
            "direction": "clean-to-typo",
            "coordinate_source": "manifest-edit-token-indices-retokenized/v1",
            "layer_window": [0, 6],
            "modes": ["first", "final", "all"],
            "token_count_policy": "equal-count-primary",
            "secondary_alignment": "nearest-normalized-position-half-up-endpoints/v1",
            "effective_eos_token_ids": [1, 107],
            "effective_eos_token_ids_source": "fixture",
            "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
            "answer_extraction": "primary-then-empty-only-positional/v1",
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
                "patch_application": "layers-0-5-on-typo-prompt-prefill-exactly-once/v1",
            },
        }

    def scan_pair(
        self,
        pair: Mapping[str, object],
        *,
        modes: tuple[str, ...],
    ) -> SubwordPairScan:
        type(self).calls += 1
        index = int(str(pair["sample_id"]).rsplit("-", 1)[1])
        scans: dict[str, SubwordModeScan] = {}
        for mode, divisor in (("first", 2), ("final", 3), ("all", 5)):
            if mode not in modes:
                continue
            plan = plan_subword_patch(pair["edits"], mode=mode)  # type: ignore[arg-type]
            correct = index % divisor == 0
            scans[mode] = SubwordModeScan(
                generation=SubwordGeneration(
                    token_ids=(10 + index, 1),
                    text=f"{mode}-{index}",
                    termination="eos",
                    value="42" if correct else "99",
                    is_extracted=True,
                    is_correct=correct,
                    method="fixture",
                    primary_method="fixture-primary",
                ),
                source_positions=plan.source_positions,
                destination_positions=plan.destination_positions,
            )
        return SubwordPairScan(scans=scans)


def _config(
    tmp_path: Path,
    *,
    limit: int | None = None,
    resume: bool = False,
) -> SubwordPositionPatchingConfig:
    manifest = tmp_path / "manifest" / "pair_manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("fixture\n", encoding="utf-8")
    return SubwordPositionPatchingConfig(
        protocol_path=DEFAULT_CONFIG,
        manifest_path=manifest,
        modes=("first", "final", "all"),
        token_count_policy="equal-count-primary",
        gpu_id="3",
        output_dir=tmp_path / "output",
        limit=limit,
        resume=resume,
    )


def test_default_protocol_catalog_and_cli_match_the_frozen_readme() -> None:
    protocol = load_subword_position_patching_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "subword-position-patching-config/v1"
    assert protocol.model == PRIMARY.model
    assert protocol.task == PRIMARY.task
    assert protocol.cohort_pairs == 172
    assert protocol.direction == "clean-to-typo"
    assert protocol.coordinate_source == "manifest-edit-token-indices-retokenized/v1"
    assert protocol.window == (0, 6)
    assert protocol.modes == ("first", "final", "all")
    assert protocol.token_count_policy == "equal-count-primary"
    assert protocol.subset_scope == "pair-level-shared-across-all-modes/v1"
    assert protocol.secondary_alignment == ("nearest-normalized-position-half-up-endpoints/v1")
    assert protocol.answer_target == "manifest-stored-clean-answer/v1"
    assert protocol.empty_rate_policy == "json-null-csv-blank-with-defined-flag/v1"
    assert protocol.pair_bootstrap_replicates == 10_000
    assert protocol.bootstrap_seed == 42
    assert protocol.confidence_level == 0.95
    assert protocol.multiplicity == "holm-3-primary-mode-contrasts/v1"

    spec = get_experiment("subword-position-patching")
    assert spec.status == "implemented"
    assert spec.compute == "gpu"
    assert spec.required_arguments == (
        "--config",
        "--manifest",
        "--modes",
        "--token-count-policy",
        "--gpu-id",
        "--output-dir",
    )
    assert spec.outputs == (
        "subword_patch_records.jsonl",
        "subword_patch_table.csv",
        "subword_patch_contrasts.csv",
        "subword_alignment_flow.csv",
        "subword_patch_summary.json",
        "run.json",
    )
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "subword-position-patching" in subparsers.choices


def test_protocol_rejects_mode_policy_or_alignment_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["modes"].reverse()
    changed = tmp_path / "changed.yaml"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mode"):
        load_subword_position_patching_protocol(changed)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["alignment"]["secondary"] = "adaptive"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="alignment"):
        load_subword_position_patching_protocol(changed)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["intervention"]["coordinates"] = "stored-without-retokenization"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinates"):
        load_subword_position_patching_protocol(changed)


def test_equal_count_plans_first_final_and_exact_all_subwords() -> None:
    edits = [_edit([1, 2], [3, 4]), _edit([5], [7])]

    first = plan_subword_patch(edits, mode="first")
    final = plan_subword_patch(edits, mode="final")
    all_tokens = plan_subword_patch(edits, mode="all")

    assert first.analysis_subset == "equal-count-primary"
    assert first.source_positions == (1, 5)
    assert first.destination_positions == (3, 7)
    assert final.source_positions == (2, 5)
    assert final.destination_positions == (4, 7)
    assert all_tokens.source_positions == (1, 2, 5)
    assert all_tokens.destination_positions == (3, 4, 7)
    assert all_tokens.alignment == "exact-within-word-ordinal/v1"


def test_mismatch_all_subwords_uses_frozen_monotone_endpoint_mapping() -> None:
    edits = [_edit([10, 11], [20, 21, 22, 23]), _edit([30, 31, 32], [40])]

    first = plan_subword_patch(edits, mode="first")
    final = plan_subword_patch(edits, mode="final")
    plan = plan_subword_patch(edits, mode="all")

    assert first.analysis_subset == "mismatch-monotone-secondary"
    assert final.analysis_subset == "mismatch-monotone-secondary"
    assert plan.analysis_subset == "mismatch-monotone-secondary"
    assert plan.source_positions == (10, 10, 11, 11, 32)
    assert plan.destination_positions == (20, 21, 22, 23, 40)
    assert plan.alignment == "nearest-normalized-position-half-up-endpoints/v1"


def test_mismatch_singletons_always_use_the_clean_word_final_state() -> None:
    edits = [_edit([1], [3, 4, 5]), _edit([10, 11], [20])]

    plan = plan_subword_patch(edits, mode="all")

    assert plan.source_positions == (1, 1, 1, 11)
    assert plan.destination_positions == (3, 4, 5, 20)


def test_generation_and_scan_reject_inconsistent_payloads() -> None:
    with pytest.raises(ValueError, match="extraction"):
        SubwordGeneration(
            token_ids=(1,),
            text="answer",
            termination="eos",
            value="42",
            is_extracted=False,
            is_correct=True,
            method="fixture",
            primary_method="fixture",
        )
    generation = SubwordGeneration(
        token_ids=(1,),
        text="answer",
        termination="eos",
        value="42",
        is_extracted=True,
        is_correct=True,
        method="fixture",
        primary_method="fixture",
    )
    with pytest.raises(ValueError, match="cardinality"):
        SubwordModeScan(generation, source_positions=(1,), destination_positions=(2, 3))


def test_runtime_reindexes_duplicate_donors_and_patches_layers_zero_through_five() -> None:
    pair = _record(120)
    runtime = object.__new__(HuggingFaceSubwordPositionPatchingRuntime)
    runtime.num_layers = 12
    runtime.layers = tuple(SimpleNamespace() for _ in range(12))
    runtime._torch = SimpleNamespace(inference_mode=nullcontext)

    def tokenize(_pair: Mapping[str, object], *, side: str) -> tuple[str, str, tuple[int, ...]]:
        positions = (6, 12) if side == "clean" else (9, 13)
        return f"{side}-ids", f"{side}-mask", positions

    runtime._tokenize_and_validate = tokenize
    runtime._validated_word_tokens = lambda _pair, side: (  # type: ignore[method-assign]
        ((5, 6), (10, 11, 12)) if side == "clean" else ((7, 8, 9), (13,))
    )
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> list[list[str]]:
        captured["capture"] = kwargs
        return [[f"layer-{layer}-row-{row}" for row in range(5)] for layer in range(12)]

    runtime._capture = capture

    class Patch:
        def __init__(self, _layers: object, **kwargs: object) -> None:
            self.layer_indices = kwargs["layer_indices"]
            captured.setdefault("patches", []).append(kwargs)  # type: ignore[union-attr]

    runtime._patch_type = Patch

    def generate(**kwargs: object) -> SubwordGeneration:
        return SubwordGeneration(
            token_ids=(1,),
            text="answer",
            termination="eos",
            value="42",
            is_extracted=True,
            is_correct=True,
            method="fixture",
            primary_method="fixture",
        )

    runtime._generate_subword = generate
    runtime._gold_answer = lambda _pair: "42"

    scan = runtime.scan_pair(pair, modes=("all",))

    assert scan.scans["all"].source_positions == (5, 6, 6, 12)
    assert scan.scans["all"].destination_positions == (7, 8, 9, 13)
    patch = captured["patches"][0]  # type: ignore[index]
    assert patch["layer_indices"] == tuple(range(6))  # type: ignore[index]
    assert patch["positions"] == (7, 8, 9, 13)  # type: ignore[index]
    assert patch["donor_values"][0] == [  # type: ignore[index]
        "layer-0-row-0",
        "layer-0-row-1",
        "layer-0-row-1",
        "layer-0-row-4",
    ]


def test_runner_compiles_primary_and_secondary_tables_then_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typo_cot.experiments.subword_position_patching import runner

    records = _records()
    load_calls: list[Path] = []

    def load_manifest(path: Path) -> tuple[dict[str, object], ...]:
        load_calls.append(path)
        return records

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", load_manifest)
    _Runtime.calls = 0
    config = _config(tmp_path)

    result = run_subword_position_patching(config, runtime_factory=_Runtime)

    assert result.pairs == 172
    assert result.evaluated_pairs == 172
    assert result.primary_pairs == 100
    assert result.secondary_pairs == 72
    assert result.table_rows == 6
    assert result.contrast_rows == 6
    assert _Runtime.calls == 172
    assert load_calls == [config.manifest_path]
    with result.table_path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    assert {(row["analysis_subset"], row["mode"]) for row in rows} == {
        (subset, mode)
        for subset in ("equal-count-primary", "mismatch-monotone-secondary")
        for mode in ("first", "final", "all")
    }
    primary_first = next(
        row
        for row in rows
        if row["analysis_subset"] == "equal-count-primary" and row["mode"] == "first"
    )
    assert int(primary_first["n_pairs"]) == 100
    assert int(primary_first["successes"]) == 50
    assert float(primary_first["restoration_rate"]) == pytest.approx(0.5)
    assert primary_first["restoration_rate_defined"] == "True"
    with result.contrasts_path.open(encoding="utf-8", newline="") as handle:
        contrasts = tuple(csv.DictReader(handle))
    assert len(contrasts) == 6
    assert {
        (row["left_mode"], row["right_mode"])
        for row in contrasts
        if row["analysis_subset"] == "equal-count-primary"
    } == {("first", "final"), ("first", "all"), ("final", "all")}
    assert all(
        row["mcnemar_p_holm_defined"] == "True"
        for row in contrasts
        if row["analysis_subset"] == "equal-count-primary"
    )
    assert all(
        row["mcnemar_p_holm_defined"] == "False"
        for row in contrasts
        if row["analysis_subset"] == "mismatch-monotone-secondary"
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["confirmatory"] is True
    assert summary["primary_inference"]["cochran_q"]["pairs"] == 100
    assert summary["primary_inference"]["holm_family_size"] == 3
    assert summary["historical_final_audit"]["compared"] == 172
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"

    rejected_revalidation: list[Path] = []

    def reject_manifest(path: Path) -> tuple[dict[str, object], ...]:
        rejected_revalidation.append(path)
        raise ValueError("manifest artifact set is incomplete")

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", reject_manifest)
    with pytest.raises(ValueError, match="artifact set is incomplete"):
        run_subword_position_patching(
            replace(config, resume=True),
            runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("source revalidation must precede runtime loading")
            ),
        )
    assert rejected_revalidation == [config.manifest_path]

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", load_manifest)
    resumed = run_subword_position_patching(
        replace(config, resume=True),
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume must not load a runtime")
        ),
    )
    assert resumed == result
    assert load_calls == [config.manifest_path, config.manifest_path]


def test_empty_analysis_cells_use_explicit_undefined_rate_metadata() -> None:
    from typo_cot.experiments.subword_position_patching import runner

    rows = runner._analysis_rows([], modes=("first", "final", "all"))

    assert len(rows) == 6
    assert all(row["n_pairs"] == 0 for row in rows)
    assert all(row["restoration_rate"] is None for row in rows)
    assert all(row["restoration_rate_defined"] is False for row in rows)


def test_limit_is_nonconfirmatory_and_checkpoint_resume_is_content_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typo_cot.experiments.subword_position_patching import runner

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", lambda _path: _records())

    class FailingRuntime(_Runtime):
        attempts = 0

        def scan_pair(self, *args: object, **kwargs: object) -> SubwordPairScan:
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise RuntimeError("injected interruption")
            return super().scan_pair(*args, **kwargs)  # type: ignore[arg-type]

    config = _config(tmp_path, limit=3)
    with pytest.raises(SubwordPositionPatchingRunError):
        run_subword_position_patching(config, runtime_factory=FailingRuntime)
    assert len(tuple((config.output_dir / "checkpoints").glob("*.json"))) == 1
    failed = json.loads((config.output_dir / "run.json").read_text(encoding="utf-8"))
    assert failed["failures"][0]["pair_id"] == _record(1)["pair_id"]
    assert failed["failures"][0]["sample_id"] == "sample-0001"
    assert failed["failures"][0]["stage"] == "scan-or-checkpoint"

    result = run_subword_position_patching(replace(config, resume=True), runtime_factory=_Runtime)
    assert result.evaluated_pairs == 3
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["confirmatory"] is False
    with result.contrasts_path.open(encoding="utf-8", newline="") as handle:
        smoke_contrasts = tuple(csv.DictReader(handle))
    assert all(row["confirmatory"] == "False" for row in smoke_contrasts)
    assert all(row["mcnemar_p_holm_defined"] == "False" for row in smoke_contrasts)

    result.summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output SHA-256 differs"):
        run_subword_position_patching(replace(config, resume=True), runtime_factory=_Runtime)
