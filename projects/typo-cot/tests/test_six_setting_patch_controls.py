"""Contracts for the six-setting answer-level specificity controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
)
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.six_setting_patch_controls import runner as control_runner
from typo_cot.experiments.six_setting_patch_controls.protocol import (
    load_six_setting_patch_controls_protocol,
)
from typo_cot.experiments.six_setting_patch_controls.runner import (
    ControlArmResult,
    ControlGeneration,
    SixSettingPatchControlsConfig,
    SixSettingPatchControlsRunError,
    run_six_setting_patch_controls,
)
from typo_cot.experiments.six_setting_patch_controls.source import (
    load_fixed_references,
)
from typo_cot.experiments.six_setting_patch_controls.statistics import (
    holm_adjust,
    nested_macro_bootstrap_risk_difference,
    paired_bootstrap_risk_difference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "six-setting-patch-controls.yaml"


def _write_protocol(path: Path, *, replicates: int = 200) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["statistics"]["pair_bootstrap_replicates"] = replicates
    payload["statistics"]["nested_bootstrap_replicates"] = replicates
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pair_id(setting_index: int, pair_index: int) -> str:
    return hashlib.sha256(f"{setting_index}\0{pair_index}".encode()).hexdigest()


def _records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for setting_index, setting in enumerate(REBUTTAL_SETTINGS):
        pair_ids = [
            _pair_id(setting_index, pair_index) for pair_index in range(setting.paper_denominator)
        ]
        for pair_index, pair_id in enumerate(pair_ids):
            donor_id = pair_ids[(pair_index + 1) % len(pair_ids)]
            correct_event = pair_index < setting.paper_successes
            records.append(
                {
                    "pair_id": pair_id,
                    "sample_id": f"sample-{pair_index:04d}",
                    "task": setting.task,
                    "model": setting.model,
                    "target_rule": "attribution-4",
                    "gold_answer": "2" if setting.task == "gsm8k" else "A",
                    "clean_text": f"clean prompt {pair_id}",
                    "typo_text": f"typo prompt {pair_id}",
                    "clean_prompt_token_count": 8,
                    "typo_prompt_token_count": 8,
                    "clean_answer": "2" if setting.task == "gsm8k" else "A",
                    "typo_answer": "3" if setting.task == "gsm8k" else "B",
                    "edits": [
                        {
                            "clean_char_span": [0, 5],
                            "typo_char_span": [0, 4],
                            "clean_word_final_token": 2,
                            "typo_word_final_token": 3,
                        }
                    ],
                    "cohorts": {"restoration": True},
                    "controls": {
                        "correct": {
                            "valid": True,
                            "source_positions": [2],
                            "destination_positions": [3],
                        },
                        "offset_2": {
                            "valid": True,
                            "source_positions": [4],
                            "destination_positions": [5],
                            "invalid_reason": None,
                        },
                        "cross_item": {
                            "valid": True,
                            "donor_pair_id": donor_id,
                            "invalid_reason": None,
                        },
                        "common_valid": True,
                    },
                    "fixed_window": {
                        "event": correct_event,
                        "run_path": f"setting-{setting_index}",
                        "run_sha256": "a" * 64,
                    },
                    "source": {
                        "source_record_sha256": hashlib.sha256(pair_id.encode()).hexdigest(),
                        "model_revision": hashlib.sha256(setting.model.encode()).hexdigest(),
                    },
                }
            )
    records.sort(key=lambda row: (str(row["task"]), str(row["model"]), str(row["pair_id"])))
    return tuple(records)


class _Runtime:
    num_layers = 12

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        self.model = model
        self.task = task
        self.revision = revision
        self.gpu_id = gpu_id

    def provenance(self) -> Mapping[str, object]:
        return {
            "operation": "six-setting-patch-controls",
            "runtime": "six-setting-fixture",
            "model": self.model,
            "task": self.task,
            "requested_revision": self.revision,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "cuda_visible_devices": self.gpu_id,
            "effective_eos_token_ids": [1],
            "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
            "coordinate_source": "rebuttal-pair-manifest/v1",
            "layer_window": [0, 6],
            "diagnostic_controls": ["offset-2", "cross-item"],
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "max_new_tokens": 512,
            },
            "answer_extraction": "primary-then-empty-only-positional/v1",
        }

    def scan_controls(
        self,
        pair: Mapping[str, object],
        donor_pair: Mapping[str, object] | None,
        controls: tuple[str, ...],
    ) -> Mapping[str, ControlArmResult]:
        pair_index = int(str(pair["sample_id"]).rsplit("-", 1)[1])
        clean_value = str(pair["clean_answer"])
        wrong_value = "3" if pair["task"] == "gsm8k" else "B"
        plans = pair["controls"]
        assert isinstance(plans, Mapping)
        output: dict[str, ControlArmResult] = {}
        for control in controls:
            restored = pair_index % (4 if control == "offset-2" else 5) == 0
            value = clean_value if restored else wrong_value
            generation = ControlGeneration(
                token_ids=(100 + pair_index,),
                text=f"The answer is {value}.",
                termination="eos",
                value=value,
                is_extracted=True,
                is_correct=restored,
                method="primary:fixture",
                primary_method="fixture",
            )
            if control == "offset-2":
                plan = plans["offset_2"]
                assert isinstance(plan, Mapping)
                source_positions = tuple(plan["source_positions"])
                destination_positions = tuple(plan["destination_positions"])
            else:
                assert donor_pair is not None
                donor_edits = donor_pair["edits"]
                recipient_edits = pair["edits"]
                assert isinstance(donor_edits, list) and isinstance(recipient_edits, list)
                source_positions = tuple(
                    int(edit["clean_word_final_token"]) for edit in donor_edits
                )
                destination_positions = tuple(
                    int(edit["typo_word_final_token"]) for edit in recipient_edits
                )
            output[control] = ControlArmResult(
                generation=generation,
                source_positions=source_positions,
                destination_positions=destination_positions,
            )
        return output


def test_default_protocol_and_catalog_match_the_frozen_readme() -> None:
    protocol = load_six_setting_patch_controls_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "six-setting-patch-controls-config/v1"
    assert protocol.window == (0, 6)
    assert protocol.controls == ("correct", "offset-2", "cross-item")
    assert protocol.primary_denominator == "common-valid"
    assert protocol.pair_bootstrap_replicates == 10_000
    assert protocol.nested_bootstrap_replicates == 10_000
    assert protocol.bootstrap_seed == 42
    assert protocol.confidence_level == 0.95
    assert protocol.multiplicity == "holm-12-setting-level-tests/v1"

    spec = get_experiment("six-setting-patch-controls")
    assert spec.status == "implemented"
    assert spec.compute == "gpu"
    assert spec.required_arguments == (
        "--config",
        "--manifest",
        "--fixed-window-root",
        "--gpu-id",
        "--output-dir",
    )
    assert spec.outputs == (
        "control_records.jsonl",
        "pair_status_records.jsonl",
        "six_setting_control_table.csv",
        "common_denominator_flow.csv",
        "multiplicity_table.csv",
        "macro_average.json",
        "risk_difference_forest.svg",
        "run.json",
    )

    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "six-setting-patch-controls" in subparsers.choices


def test_paired_bootstrap_holm_and_nested_macro_are_deterministic() -> None:
    left = (True, True, False, False, True, False)
    right = (False, True, True, False, False, False)

    first = paired_bootstrap_risk_difference(
        left,
        right,
        replicates=500,
        confidence_level=0.95,
        seed=42,
    )
    second = paired_bootstrap_risk_difference(
        left,
        right,
        replicates=500,
        confidence_level=0.95,
        seed=42,
    )
    assert first == second
    assert first["estimate"] == pytest.approx(1 / 6)
    assert first["lower"] <= first["estimate"] <= first["upper"]

    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02, "d": 0.5})
    assert adjusted == {"a": 0.04, "b": 0.06, "c": 0.06, "d": 0.5}

    setting_pairs = {
        "s1": (left, right),
        "s2": ((True, False, True), (False, False, False)),
    }
    macro = nested_macro_bootstrap_risk_difference(
        setting_pairs,
        replicates=500,
        confidence_level=0.95,
        seed=42,
    )
    assert macro == nested_macro_bootstrap_risk_difference(
        setting_pairs,
        replicates=500,
        confidence_level=0.95,
        seed=42,
    )
    assert macro["estimate"] == pytest.approx((1 / 6 + 2 / 3) / 2)


def test_runner_compiles_six_settings_common_denominators_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    assert len(records) == REBUTTAL_MANIFEST_PROTOCOL.restoration_pairs
    protocol_path = tmp_path / "protocol.yaml"
    _write_protocol(protocol_path)
    manifest_path = tmp_path / "manifest" / "pair_manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text("fixture\n", encoding="utf-8")
    fixed_root = tmp_path / "fixed"
    fixed_root.mkdir()
    output_dir = tmp_path / "output"

    monkeypatch.setattr(control_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    monkeypatch.setattr(
        control_runner,
        "_load_fixed_references",
        lambda **_kwargs: {
            str(record["pair_id"]): SimpleNamespace(
                clean_answer=str(record["clean_answer"]),
                correct_event=bool(record["fixed_window"]["event"]),
            )
            for record in records
        },
    )

    config = SixSettingPatchControlsConfig(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        fixed_window_root=fixed_root,
        output_dir=output_dir,
        gpu_id="3",
        limit_per_setting=2,
    )
    result = run_six_setting_patch_controls(config, runtime_factory=_Runtime)

    assert result.pairs == 12
    assert result.control_records == 36
    assert result.settings == 6
    for path in (
        result.control_records_path,
        result.pair_status_records_path,
        result.control_table_path,
        result.flow_table_path,
        result.multiplicity_table_path,
        result.macro_average_path,
        result.forest_plot_path,
        result.run_path,
    ):
        assert path.is_file()

    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["confirmatory"] is False
    assert run["counts"] == {
        "common_valid_pairs": 12,
        "control_records": 36,
        "pair_status_records": 12,
        "selected_pairs": 12,
        "settings": 6,
    }
    multiplicity_rows = result.multiplicity_table_path.read_text(encoding="utf-8").splitlines()
    assert len(multiplicity_rows) == 13
    assert result.forest_plot_path.read_text(encoding="utf-8").startswith("<svg")

    resumed = run_six_setting_patch_controls(
        replace(config, resume=True),
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume must not load a runtime")
        ),
    )
    assert resumed == result

    run["counts"]["selected_pairs"] = 999
    result.run_path.write_text(json.dumps(run) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        run_six_setting_patch_controls(replace(config, resume=True), runtime_factory=_Runtime)


def test_resume_rejects_an_unbound_nonempty_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    protocol_path = tmp_path / "protocol.yaml"
    _write_protocol(protocol_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("fixture\n", encoding="utf-8")
    fixed_root = tmp_path / "fixed"
    fixed_root.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "unbound.txt").write_text("not a checkpoint\n", encoding="utf-8")
    monkeypatch.setattr(control_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    monkeypatch.setattr(
        control_runner,
        "_load_fixed_references",
        lambda **_kwargs: {
            str(record["pair_id"]): SimpleNamespace(
                clean_answer=str(record["clean_answer"]),
                correct_event=bool(record["fixed_window"]["event"]),
            )
            for record in records
        },
    )

    with pytest.raises(ValueError, match="run.json"):
        run_six_setting_patch_controls(
            SixSettingPatchControlsConfig(
                protocol_path=protocol_path,
                manifest_path=manifest_path,
                fixed_window_root=fixed_root,
                output_dir=output_dir,
                gpu_id="3",
                limit_per_setting=1,
                resume=True,
            ),
            runtime_factory=_Runtime,
        )


def test_failed_run_retains_verified_checkpoints_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    protocol_path = tmp_path / "protocol.yaml"
    _write_protocol(protocol_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("fixture\n", encoding="utf-8")
    fixed_root = tmp_path / "fixed"
    fixed_root.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(control_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    monkeypatch.setattr(
        control_runner,
        "_load_fixed_references",
        lambda **_kwargs: {
            str(record["pair_id"]): SimpleNamespace(
                clean_answer=str(record["clean_answer"]),
                correct_event=bool(record["fixed_window"]["event"]),
            )
            for record in records
        },
    )

    class FailingRuntime(_Runtime):
        scans = 0

        def scan_controls(
            self,
            pair: Mapping[str, object],
            donor_pair: Mapping[str, object] | None,
            controls: tuple[str, ...],
        ) -> Mapping[str, ControlArmResult]:
            type(self).scans += 1
            if type(self).scans == 2:
                raise RuntimeError("injected interruption")
            return super().scan_controls(pair, donor_pair, controls)

    config = SixSettingPatchControlsConfig(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        fixed_window_root=fixed_root,
        output_dir=output_dir,
        gpu_id="3",
        limit_per_setting=2,
    )
    with pytest.raises(SixSettingPatchControlsRunError):
        run_six_setting_patch_controls(config, runtime_factory=FailingRuntime)
    assert len(tuple((output_dir / "checkpoints").glob("*.json"))) == 1

    resumed = run_six_setting_patch_controls(
        replace(config, resume=True),
        runtime_factory=_Runtime,
    )
    assert resumed.pairs == 12
    assert resumed.control_records == 36


def test_invalid_arm_is_reported_and_excluded_without_partial_patching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = list(deepcopy(_records()))
    first_setting = REBUTTAL_SETTINGS[0]
    first = next(
        record
        for record in records
        if (record["model"], record["task"], record["sample_id"])
        == (*first_setting.key, "sample-0000")
    )
    first["controls"]["offset_2"] = {
        "valid": False,
        "source_positions": [],
        "destination_positions": [],
        "invalid_reason": "fixture-invalid-offset",
    }
    first["controls"]["common_valid"] = False
    protocol_path = tmp_path / "protocol.yaml"
    _write_protocol(protocol_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("fixture\n", encoding="utf-8")
    fixed_root = tmp_path / "fixed"
    fixed_root.mkdir()
    monkeypatch.setattr(control_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    monkeypatch.setattr(
        control_runner,
        "_load_fixed_references",
        lambda **_kwargs: {
            str(record["pair_id"]): SimpleNamespace(
                clean_answer=str(record["clean_answer"]),
                correct_event=bool(record["fixed_window"]["event"]),
            )
            for record in records
        },
    )

    result = run_six_setting_patch_controls(
        SixSettingPatchControlsConfig(
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            fixed_window_root=fixed_root,
            output_dir=tmp_path / "output",
            gpu_id="3",
            limit_per_setting=2,
        ),
        runtime_factory=_Runtime,
    )
    assert result.control_records == 35
    status = [
        json.loads(line)
        for line in result.pair_status_records_path.read_text(encoding="utf-8").splitlines()
    ]
    invalid = next(row for row in status if row["pair_id"] == first["pair_id"])
    assert invalid["validity"]["offset-2"] == {
        "valid": False,
        "reason": "fixture-invalid-offset",
    }
    assert invalid["events"]["offset-2"] is None


def test_fixed_reference_loader_verifies_hashes_and_exact_pair_identity(tmp_path: Path) -> None:
    fixed_root = tmp_path / "fixed"
    run_dir = fixed_root / "setting"
    run_dir.mkdir(parents=True)
    source_sha = "a" * 64
    revision = "b" * 40
    model = "fixture/model"
    task = "gsm8k"
    sample_id = "sample-1"
    target_rule = "attribution-4"
    status = {
        "schema_version": "fixed-window-answer-patching-pair-status/v1",
        "paper_sha256": control_runner.PAPER_SHA256,
        "model": model,
        "benchmark": task,
        "targeting": target_rule,
        "sample_id": sample_id,
        "source_record_sha256": source_sha,
        "direction_status": {"clean-to-edited": {"included": True, "exclusion_reason": None}},
        "baseline": {
            "clean": {"value": "2", "is_extracted": True, "is_correct": True},
            "edited": {"value": "3", "is_extracted": True, "is_correct": False},
        },
    }
    fixed_record = {
        "schema_version": "fixed-window-answer-patching-window/v1",
        "paper_sha256": control_runner.PAPER_SHA256,
        "model": model,
        "benchmark": task,
        "targeting": target_rule,
        "sample_id": sample_id,
        "source_record_sha256": source_sha,
        "direction": "clean-to-edited",
        "window": "0:6",
        "window_start": 0,
        "window_stop": 6,
        "num_layers": 12,
        "event": True,
        "patched_answer": {
            "value": "2",
            "is_extracted": True,
            "is_correct": True,
        },
    }
    artifacts = {
        "fixed_window_records.jsonl": json.dumps(fixed_record) + "\n",
        "pair_status_records.jsonl": json.dumps(status) + "\n",
        "setting_summary.json": "{}\n",
    }
    for name, contents in artifacts.items():
        (run_dir / name).write_text(contents, encoding="utf-8")
    outputs = {
        name: {
            "sha256": hashlib.sha256(contents.encode()).hexdigest(),
            "records": 1,
        }
        for name, contents in artifacts.items()
    }
    run = {
        "schema_version": "fixed-window-answer-patching-run/v1",
        "paper_sha256": control_runner.PAPER_SHA256,
        "operation": "fixed-window-answer-patching",
        "status": "completed",
        "failures": [],
        "arguments": {
            "model": model,
            "benchmark": task,
            "layers": ["0:6"],
            "directions": ["clean-to-edited"],
        },
        "protocol": {"schema_version": "fixed-window-answer-patching-protocol/v1"},
        "runtime": {
            "requested_revision": revision,
            "model_revision": revision,
            "tokenizer_revision": revision,
        },
        "outputs": outputs,
    }
    run_path = run_dir / "run.json"
    run_path.write_text(json.dumps(run) + "\n", encoding="utf-8")
    manifest_record = {
        "pair_id": "pair-1",
        "sample_id": sample_id,
        "model": model,
        "task": task,
        "target_rule": target_rule,
        "cohorts": {"restoration": True},
        "source": {"source_record_sha256": source_sha, "model_revision": revision},
        "fixed_window": {
            "event": True,
            "run_path": "setting",
            "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        },
    }

    references = load_fixed_references((manifest_record,), fixed_root)
    assert references["pair-1"].clean_answer == "2"
    assert references["pair-1"].correct_event is True

    (run_dir / "fixed_window_records.jsonl").write_text(
        json.dumps(fixed_record) + "\n ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_fixed_references((manifest_record,), fixed_root)

    escaped = deepcopy(manifest_record)
    escaped["fixed_window"]["run_path"] = "../escaped"
    with pytest.raises(ValueError, match="relative|escaped"):
        load_fixed_references((escaped,), fixed_root)
