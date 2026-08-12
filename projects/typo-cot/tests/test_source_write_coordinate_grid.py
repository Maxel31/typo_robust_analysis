"""Contracts for the donor-source by write-position coordinate grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
from typo_cot.experiments.build_rebuttal_manifest import REBUTTAL_SETTINGS
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.six_setting_patch_controls import (
    ControlArmResult,
    ControlGeneration,
)
from typo_cot.experiments.source_write_coordinate_grid import runner as grid_runner
from typo_cot.experiments.source_write_coordinate_grid.protocol import (
    load_source_write_coordinate_grid_protocol,
)
from typo_cot.experiments.source_write_coordinate_grid.runner import (
    SourceWriteCoordinateGridConfig,
    SourceWriteCoordinateGridRunError,
    run_source_write_coordinate_grid,
)
from typo_cot.experiments.source_write_coordinate_grid.statistics import cochran_q


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "source-write-coordinate-grid.yaml"
COHORT_SETTINGS = {
    "primary": REBUTTAL_SETTINGS[0],
    "replication": REBUTTAL_SETTINGS[-1],
}


def _write_protocol(path: Path) -> None:
    path.write_bytes(DEFAULT_CONFIG.read_bytes())


@pytest.fixture(autouse=True)
def _fast_runner_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    def bootstrap(
        left: tuple[bool, ...],
        right: tuple[bool, ...],
        **_kwargs: object,
    ) -> dict[str, float]:
        estimate = sum(left) / len(left) - sum(right) / len(right)
        return {"estimate": estimate, "lower": estimate, "upper": estimate}

    monkeypatch.setattr(grid_runner, "paired_bootstrap_risk_difference", bootstrap)


def _records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for cohort_index, setting in enumerate(COHORT_SETTINGS.values()):
        for pair_index in range(setting.paper_denominator):
            pair_id = hashlib.sha256(f"grid\0{cohort_index}\0{pair_index}".encode()).hexdigest()
            correct_event = pair_index < setting.paper_successes
            clean_answer = "2" if setting.task == "gsm8k" else "A"
            typo_answer = "3" if setting.task == "gsm8k" else "B"
            records.append(
                {
                    "pair_id": pair_id,
                    "sample_id": f"sample-{pair_index:04d}",
                    "task": setting.task,
                    "model": setting.model,
                    "target_rule": "attribution-4",
                    "gold_answer": clean_answer,
                    "clean_text": f"clean prompt {pair_id}",
                    "typo_text": f"typo prompt {pair_id}",
                    "clean_prompt_token_count": 12,
                    "typo_prompt_token_count": 12,
                    "clean_answer": clean_answer,
                    "typo_answer": typo_answer,
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
                    },
                    "fixed_window": {
                        "event": correct_event,
                        "run_path": f"cohort-{cohort_index}",
                        "run_sha256": "a" * 64,
                    },
                    "source": {
                        "source_record_sha256": hashlib.sha256(pair_id.encode()).hexdigest(),
                        "model_revision": hashlib.sha256(setting.model.encode()).hexdigest(),
                    },
                }
            )
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
            "operation": "source-write-coordinate-grid",
            "runtime": "source-write-grid-fixture",
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
            "generated_arms": ["E->O", "O->E", "O->O"],
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "max_new_tokens": 512,
            },
            "answer_extraction": "primary-then-empty-only-positional-by-termination/v1",
        }

    def scan_grid(
        self,
        pair: Mapping[str, object],
    ) -> Mapping[str, ControlArmResult]:
        pair_index = int(str(pair["sample_id"]).rsplit("-", 1)[1])
        clean_value = str(pair["clean_answer"])
        wrong_value = "3" if pair["task"] == "gsm8k" else "B"
        plans = pair["controls"]
        assert isinstance(plans, Mapping)
        correct = plans["correct"]
        offset = plans["offset_2"]
        assert isinstance(correct, Mapping) and isinstance(offset, Mapping)
        coordinates = {
            "E->O": (tuple(correct["source_positions"]), tuple(offset["destination_positions"])),
            "O->E": (tuple(offset["source_positions"]), tuple(correct["destination_positions"])),
            "O->O": (tuple(offset["source_positions"]), tuple(offset["destination_positions"])),
        }
        results: dict[str, ControlArmResult] = {}
        divisors = {"E->O": 3, "O->E": 4, "O->O": 5}
        for arm, (source_positions, destination_positions) in coordinates.items():
            restored = pair_index % divisors[arm] == 0
            value = clean_value if restored else wrong_value
            results[arm] = ControlArmResult(
                generation=ControlGeneration(
                    token_ids=(100 + pair_index,),
                    text=f"The answer is {value}.",
                    termination="eos",
                    value=value,
                    is_extracted=True,
                    is_correct=restored,
                    method="primary:fixture",
                    primary_method="fixture",
                ),
                source_positions=source_positions,
                destination_positions=destination_positions,
            )
        return results


def test_default_protocol_catalog_and_cli_match_the_frozen_readme() -> None:
    protocol = load_source_write_coordinate_grid_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "source-write-coordinate-grid-config/v1"
    assert protocol.window == (0, 6)
    assert protocol.arms == ("E->E", "E->O", "O->E", "O->O")
    assert protocol.cohorts == {
        "primary": ("google/gemma-3-4b-it", "gsm8k"),
        "replication": ("mistralai/Mistral-7B-Instruct-v0.3", "mmlu"),
    }
    assert protocol.common_validity == "correct-and-offset-valid/v1"
    assert protocol.contrasts == (("E->E", "E->O"), ("E->E", "O->E"))
    assert protocol.pair_bootstrap_replicates == 10_000
    assert protocol.bootstrap_seed == 42
    assert protocol.confidence_level == 0.95
    assert protocol.multiplicity == "holm-4-cohort-contrast-tests/v1"

    spec = get_experiment("source-write-coordinate-grid")
    assert spec.status == "implemented"
    assert spec.compute == "gpu"
    assert spec.required_arguments == (
        "--config",
        "--manifest",
        "--fixed-window-root",
        "--cohorts",
        "--gpu-id",
        "--output-dir",
    )
    assert spec.outputs == (
        "source_write_grid_records.jsonl",
        "pair_status_records.jsonl",
        "source_write_grid_table.csv",
        "source_write_contrasts.csv",
        "run.json",
    )

    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "source-write-coordinate-grid" in subparsers.choices


def test_protocol_rejects_bootstrap_replicate_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["statistics"]["pair_bootstrap_replicates"] = 9_999
    changed = tmp_path / "changed.yaml"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pair_bootstrap_replicates.*10,000"):
        load_source_write_coordinate_grid_protocol(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("bootstrap_seed", 43, "bootstrap_seed.*42"),
        ("confidence_level", 0.9, "confidence_level.*0.95"),
    ),
)
def test_protocol_rejects_other_statistical_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["statistics"][field] = value
    changed = tmp_path / f"changed-{field}.yaml"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_source_write_coordinate_grid_protocol(changed)


def test_grid_runtime_requests_the_fixed_reference_extraction_contract() -> None:
    from typo_cot.experiments.source_write_coordinate_grid.runtime import (
        HuggingFaceSourceWriteCoordinateGridRuntime,
    )

    runtime = object.__new__(HuggingFaceSourceWriteCoordinateGridRuntime)
    runtime.num_layers = 6
    runtime._torch = SimpleNamespace(inference_mode=nullcontext)
    runtime._tokenize_and_validate = lambda _pair, *, side: (  # type: ignore[method-assign]
        object(),
        object(),
        (2,) if side == "clean" else (3,),
    )
    runtime._gold_answer = lambda _pair: "2"  # type: ignore[method-assign]
    runtime._capture = lambda **_kwargs: [object()] * 6  # type: ignore[method-assign]
    runtime._window_patch = lambda **_kwargs: nullcontext()  # type: ignore[method-assign]
    requested_contracts: list[bool] = []

    def generate(**kwargs: object) -> ControlGeneration:
        requested_contracts.append("allow_positional_after_length_cap" in kwargs)
        return ControlGeneration(
            token_ids=(1,),
            text="unfinished reasoning\n2",
            termination="length-cap",
            value="2",
            is_extracted=True,
            is_correct=True,
            method="fallback:N5_tail_number",
            primary_method="fixture",
        )

    runtime._generate_control = generate  # type: ignore[method-assign]
    record = {
        "pair_id": "grid-runtime-fixture",
        "gold_answer": "2",
        "target_rule": "attribution-4",
        "clean_text": "clean prompt",
        "typo_text": "typo prompt",
        "clean_prompt_token_count": 8,
        "typo_prompt_token_count": 8,
        "edits": [
            {
                "clean_char_span": [0, 5],
                "typo_char_span": [0, 4],
                "clean_word_final_token": 2,
                "typo_word_final_token": 3,
            }
        ],
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
            },
        },
    }

    assert set(runtime.scan_grid(record)) == {"E->O", "O->E", "O->O"}
    assert requested_contracts == [False, False, False]


def test_runner_rejects_duplicate_manifest_coordinates() -> None:
    record = {
        "controls": {
            "correct": {
                "source_positions": [2, 2],
                "destination_positions": [3, 4],
            },
            "offset_2": {
                "source_positions": [4, 5],
                "destination_positions": [5, 6],
            },
        }
    }

    with pytest.raises(ValueError, match="duplicate token positions"):
        grid_runner._coordinates(record, "E->E")


def test_cochran_q_handles_signal_and_the_all_equal_boundary() -> None:
    signal = cochran_q(
        {
            "E->E": (True, True, True, False, True),
            "E->O": (False, True, False, False, False),
            "O->E": (False, False, True, False, False),
            "O->O": (False, False, False, False, False),
        }
    )
    assert signal["pairs"] == 5
    assert signal["arms"] == 4
    assert signal["degrees_of_freedom"] == 3
    assert signal["statistic"] > 0
    assert 0 <= signal["p_value"] <= 1

    boundary = cochran_q(
        {
            "E->E": (True, False),
            "E->O": (True, False),
            "O->E": (True, False),
            "O->O": (True, False),
        }
    )
    assert boundary["statistic"] == 0.0
    assert boundary["p_value"] == 1.0


def test_partial_cohort_labels_the_executed_holm_family_nonconfirmatory() -> None:
    protocol = load_source_write_coordinate_grid_protocol(DEFAULT_CONFIG)
    model, task = protocol.cohorts["primary"]
    statuses = [
        {
            "model": model,
            "task": task,
            "validity": {
                "O->O": {"valid": True},
                "common-valid": True,
            },
            "events": {
                "E->E": True,
                "E->O": index == 0,
                "O->E": False,
                "O->O": False,
            },
        }
        for index in range(2)
    ]

    _tables, contrasts = grid_runner._analysis_rows(
        statuses,
        protocol=protocol,
        cohorts=("primary",),
        confirmatory=False,
    )

    assert len(contrasts) == 2
    assert {row["holm_family_size"] for row in contrasts} == {2}
    assert {row["holm_family"] for row in contrasts} == {
        "holm-2-executed-tests/nonconfirmatory/v1"
    }
    assert {row["planned_holm_family"] for row in contrasts} == {
        "holm-4-cohort-contrast-tests/v1"
    }
    assert {row["confirmatory"] for row in contrasts} == {False}


def test_runner_compiles_both_cohorts_and_resumes(
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
    monkeypatch.setattr(grid_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    monkeypatch.setattr(
        grid_runner,
        "_load_fixed_references",
        lambda **_kwargs: {
            str(record["pair_id"]): SimpleNamespace(
                clean_answer=str(record["clean_answer"]),
                correct_event=bool(record["fixed_window"]["event"]),
            )
            for record in records
        },
    )
    config = SourceWriteCoordinateGridConfig(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        fixed_window_root=fixed_root,
        cohorts=("primary", "replication"),
        gpu_id="3",
        output_dir=output_dir,
        limit_per_cohort=2,
    )

    result = run_source_write_coordinate_grid(config, runtime_factory=_Runtime)

    assert result.pairs == 4
    assert result.grid_records == 16
    assert result.cohorts == 2
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["confirmatory"] is False
    assert run["counts"] == {
        "cohorts": 2,
        "common_valid_pairs": 4,
        "grid_records": 16,
        "pair_status_records": 4,
        "selected_pairs": 4,
    }
    assert len(result.grid_table_path.read_text(encoding="utf-8").splitlines()) == 3
    assert len(result.contrasts_path.read_text(encoding="utf-8").splitlines()) == 5

    resumed = run_source_write_coordinate_grid(
        replace(config, resume=True),
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume must not load a runtime")
        ),
    )
    assert resumed == result


def test_invalid_offset_excludes_all_three_offset_dependent_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = list(deepcopy(_records()))
    first_setting = COHORT_SETTINGS["primary"]
    invalid = next(
        record
        for record in records
        if (record["model"], record["task"], record["sample_id"])
        == (*first_setting.key, "sample-0000")
    )
    invalid["controls"]["offset_2"] = {
        "valid": False,
        "source_positions": [],
        "destination_positions": [],
        "invalid_reason": "fixture-invalid-offset",
    }
    protocol_path = tmp_path / "protocol.yaml"
    _write_protocol(protocol_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("fixture\n", encoding="utf-8")
    fixed_root = tmp_path / "fixed"
    fixed_root.mkdir()
    monkeypatch.setattr(grid_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    monkeypatch.setattr(
        grid_runner,
        "_load_fixed_references",
        lambda **_kwargs: {
            str(record["pair_id"]): SimpleNamespace(
                clean_answer=str(record["clean_answer"]),
                correct_event=bool(record["fixed_window"]["event"]),
            )
            for record in records
        },
    )

    result = run_source_write_coordinate_grid(
        SourceWriteCoordinateGridConfig(
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            fixed_window_root=fixed_root,
            cohorts=("replication", "primary"),
            gpu_id="3",
            output_dir=tmp_path / "output",
        ),
        runtime_factory=_Runtime,
    )

    assert result.pairs == 377
    assert result.grid_records == 1505
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["confirmatory"] is True
    assert run["counts"]["common_valid_pairs"] == 376
    statuses = [
        json.loads(line)
        for line in result.pair_status_records_path.read_text(encoding="utf-8").splitlines()
    ]
    status = next(row for row in statuses if row["pair_id"] == invalid["pair_id"])
    assert status["validity"]["common-valid"] is False
    for arm in ("E->O", "O->E", "O->O"):
        assert status["validity"][arm] == {
            "valid": False,
            "reason": "fixture-invalid-offset",
        }
        assert status["events"][arm] is None


def test_failed_grid_run_retains_verified_checkpoint_and_resumes(
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
    monkeypatch.setattr(grid_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    monkeypatch.setattr(
        grid_runner,
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

        def scan_grid(
            self,
            pair: Mapping[str, object],
        ) -> Mapping[str, ControlArmResult]:
            type(self).scans += 1
            if type(self).scans == 2:
                raise RuntimeError("injected interruption")
            return super().scan_grid(pair)

    config = SourceWriteCoordinateGridConfig(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        fixed_window_root=fixed_root,
        cohorts=("primary", "replication"),
        gpu_id="3",
        output_dir=output_dir,
        limit_per_cohort=2,
    )
    with pytest.raises(SourceWriteCoordinateGridRunError):
        run_source_write_coordinate_grid(config, runtime_factory=FailingRuntime)
    assert len(tuple((output_dir / "checkpoints").glob("*.json"))) == 1

    resumed = run_source_write_coordinate_grid(
        replace(config, resume=True),
        runtime_factory=_Runtime,
    )
    assert resumed.pairs == 4
    assert resumed.grid_records == 16
