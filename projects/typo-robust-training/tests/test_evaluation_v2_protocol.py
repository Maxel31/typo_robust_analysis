"""Falsification tests for the preregistered Base-only evaluation v2."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from typo_robust_training.evaluation.calibration_v2 import (
    BaseCalibrationObservation,
    load_evaluation_v2_protocol,
    run_base_only_severity_calibration,
    score_base_only_severity_calibration,
)
from typo_robust_training.evaluation.statistics_v2 import clustered_paired_macro_contrast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs/robustness-evaluation-v2.yaml"
REGISTRY_TEMPLATE = PROJECT_ROOT / "configs/robustness-evaluation-v2-registry.template.json"


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _calibration_rows(*, selected_at: int | None) -> tuple[BaseCalibrationObservation, ...]:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)
    rows: list[BaseCalibrationObservation] = []
    task_offsets = {task: index * 1000 for index, task in enumerate(protocol.tasks)}
    for model in protocol.models:
        for task in protocol.tasks:
            for index in range(protocol.calibration_records_per_task):
                record_id = f"{task_offsets[task] + index:064x}"
                for severity in protocol.severity_edit_counts:
                    if selected_at is None:
                        incorrect = False
                    elif severity < selected_at:
                        incorrect = index < 8  # 4 points: below the per-model floor.
                    elif severity == selected_at:
                        incorrect = index < 20  # 10 points: first eligible severity.
                    else:
                        incorrect = index < 80  # 40 points, still above the accuracy floor.
                    for variant in range(protocol.calibration_variants_per_item):
                        rows.append(
                            BaseCalibrationObservation(
                                condition="base",
                                model_id=model.model_id,
                                model_revision=model.revision,
                                adapter_checkpoint_sha256=None,
                                training_run_sha256=None,
                                task=task,
                                record_id=record_id,
                                source_text_sha256=_digest(f"source:{task}:{record_id}"),
                                severity_edit_count=severity,
                                variant=variant,
                                realized_typo_sha256=_digest(
                                    f"typo:{task}:{record_id}:{severity}:{variant}"
                                ),
                                clean_correct=True,
                                typo_correct=not incorrect,
                            )
                        )
    return tuple(rows)


def test_v2_protocol_freezes_models_calibration_population_and_legacy_random2() -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)

    assert tuple((model.model_id, model.revision) for model in protocol.models) == (
        (
            "google/gemma-3-4b-it",
            "093f9f388b31de276ce2de164bdc2081324b9767",
        ),
        (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "c170c708c41dac9275d15a8fff4eca08d52bab71",
        ),
    )
    assert protocol.tasks == ("gsm8k", "mmlu", "arc", "mmlu_pro", "commonsense_qa")
    assert protocol.calibration_records_per_task == 200
    assert protocol.calibration_variants_per_item == 3
    assert protocol.severity_edit_counts == (2, 4, 8)
    assert protocol.confirmatory_records_per_task == 1000
    assert protocol.confirmatory_typo_variants_per_item == 2
    assert "legacy-random-2" in protocol.secondary_conditions
    assert protocol.arms == (
        "base",
        "output-matching-all-layers",
        "probe-boundary-output-matching",
        "random-freeze-output-matching",
    )

    registry = json.loads(REGISTRY_TEMPLATE.read_text(encoding="utf-8"))
    assert registry["state"] == "pending-calibration"
    assert registry["governance_attestation"] == {
        "adapter_outputs_used_for_calibration": False,
        "severity_grid_extended": False,
        "model_inventory_changed_after_calibration": False,
    }
    assert registry["legacy_random_2"]["role"] == "secondary-continuity-only"


def test_calibration_selects_the_smallest_eligible_frozen_severity() -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)

    status, selected, summaries = score_base_only_severity_calibration(
        _calibration_rows(selected_at=4), protocol=protocol
    )

    assert status == "selected"
    assert selected == 4
    assert summaries["2"]["eligible"] is False
    assert summaries["4"]["eligible"] is True
    assert summaries["8"]["eligible"] is True


def test_failed_calibration_stops_instead_of_inventing_a_new_severity() -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)

    status, selected, summaries = score_base_only_severity_calibration(
        _calibration_rows(selected_at=None), protocol=protocol
    )

    assert status == "stopped-no-eligible-severity"
    assert selected is None
    assert set(summaries) == {"2", "4", "8"}
    assert not any(row["eligible"] for row in summaries.values())


def test_calibration_rejects_adapter_outputs_even_if_outcomes_look_useful() -> None:
    malicious = {
        "schema_version": "robustness-evaluation-v2-calibration-observation/v1",
        "condition": "probe-boundary-output-matching",
        "model_id": "google/gemma-3-4b-it",
        "model_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "adapter_checkpoint_sha256": "a" * 64,
        "training_run_sha256": "b" * 64,
        "task": "gsm8k",
        "record_id": "1" * 64,
        "source_text_sha256": "2" * 64,
        "severity_edit_count": 4,
        "variant": 0,
        "realized_typo_sha256": "3" * 64,
        "clean_correct": True,
        "typo_correct": True,
    }

    with pytest.raises(ValueError, match="forbids adapter or trained-model outputs"):
        BaseCalibrationObservation.from_mapping(malicious)


def test_calibration_rejects_post_failure_severity_grid_extension() -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)
    rows = list(_calibration_rows(selected_at=None))
    rows[0] = replace(rows[0], severity_edit_count=16)

    with pytest.raises(ValueError, match="outside the frozen candidate grid"):
        score_base_only_severity_calibration(rows, protocol=protocol)


def test_calibration_runner_freezes_input_hashes_and_result(tmp_path: Path) -> None:
    observations = tmp_path / "base-observations.jsonl"
    with observations.open("w", encoding="utf-8") as handle:
        for row in _calibration_rows(selected_at=4):
            handle.write(json.dumps(row.as_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    items = tmp_path / "items.jsonl"
    typos = tmp_path / "typos.jsonl"
    items.write_text('{"frozen":"items"}\n', encoding="utf-8")
    typos.write_text('{"frozen":"typos"}\n', encoding="utf-8")

    result = run_base_only_severity_calibration(
        config_path=PROTOCOL_PATH,
        observations_path=observations,
        item_manifest_path=items,
        realized_typo_manifest_path=typos,
        output_dir=tmp_path / "output",
    )
    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    run = json.loads(result.run_path.read_text(encoding="utf-8"))

    assert result.selected_edit_count == 4
    assert artifact["provenance"]["adapter_outputs_used"] is False
    assert artifact["provenance"]["item_manifest_sha256"] == _digest(
        items.read_text(encoding="utf-8")
    )
    assert artifact["provenance"]["realized_typo_manifest_sha256"] == _digest(
        typos.read_text(encoding="utf-8")
    )
    assert run["status"] == "completed"


def _confirmatory_rows() -> list[dict[str, object]]:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=100,
    )
    rows: list[dict[str, object]] = []
    for model in protocol.models:
        for task_index, task in enumerate(protocol.tasks):
            for item in range(protocol.confirmatory_records_per_task):
                record_id = f"{task_index * 1000 + item:064x}"
                for condition in (
                    "output-matching-all-layers",
                    "probe-boundary-output-matching",
                ):
                    for variant in range(protocol.confirmatory_typo_variants_per_item):
                        for seed in protocol.training_seeds:
                            proposal_gain = condition == "probe-boundary-output-matching" and (
                                item == 0 and seed == 42
                            )
                            rows.append(
                                {
                                    "model_id": model.model_id,
                                    "task": task,
                                    "record_id": record_id,
                                    "variant": variant,
                                    "condition": condition,
                                    "seed": seed,
                                    "clean_correct": True,
                                    "typo_correct": proposal_gain,
                                }
                            )
    return rows


def test_clustered_bootstrap_averages_variants_and_seeds_before_item_resampling() -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=100,
    )
    rows = _confirmatory_rows()

    result = clustered_paired_macro_contrast(
        rows,
        protocol=protocol,
        left_condition="output-matching-all-layers",
        right_condition="probe-boundary-output-matching",
        outcome="typo_correct",
    )
    reversed_result = clustered_paired_macro_contrast(
        list(reversed(rows)),
        protocol=protocol,
        left_condition="output-matching-all-layers",
        right_condition="probe-boundary-output-matching",
        outcome="typo_correct",
    )

    # One of three seeds improves one of two source items: (1/3) / 2.
    assert result.point_difference_points == pytest.approx(100.0 / 6.0)
    assert result.source_items == 10
    assert reversed_result == result


def test_clustered_bootstrap_rejects_duplicate_seed_rows_as_false_replication() -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _confirmatory_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="contain duplicates"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="output-matching-all-layers",
            right_condition="probe-boundary-output-matching",
            outcome="typo_correct",
        )
