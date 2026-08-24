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
    validate_calibration_semantic_bindings,
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


def _write_calibration_manifests(
    root: Path, rows: tuple[BaseCalibrationObservation, ...]
) -> tuple[Path, Path]:
    item_rows: dict[tuple[str, str], dict[str, object]] = {}
    typo_rows: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for row in rows:
        source_text = f"source:{row.task}:{row.record_id}"
        typo_text = f"typo:{row.task}:{row.record_id}:{row.severity_edit_count}:{row.variant}"
        item_rows[(row.task, row.record_id)] = {
            "schema_version": "robustness-evaluation-v2-calibration-item/v1",
            "task": row.task,
            "record_id": row.record_id,
            "source_text": source_text,
            "source_text_sha256": _digest(source_text),
        }
        typo_rows[(row.task, row.record_id, row.severity_edit_count, row.variant)] = {
            "schema_version": "robustness-evaluation-v2-calibration-typo/v1",
            "task": row.task,
            "record_id": row.record_id,
            "source_text_sha256": _digest(source_text),
            "severity_edit_count": row.severity_edit_count,
            "variant": row.variant,
            "realized_typo_text": typo_text,
            "realized_typo_sha256": _digest(typo_text),
        }
    items = root / "items.jsonl"
    typos = root / "typos.jsonl"
    items.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for _key, value in sorted(item_rows.items())
        ),
        encoding="utf-8",
    )
    typos.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for _key, value in sorted(typo_rows.items())
        ),
        encoding="utf-8",
    )
    return items, typos


def test_v2_protocol_freezes_models_calibration_population_and_legacy_random2() -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)

    assert tuple((model.model_id, model.revision) for model in protocol.models) == (
        (
            "google/gemma-3-4b-it",
            "093f9f388b31de276ce2de164bdc2081324b9767",
        ),
        (
            "mistralai/Mistral-7B-v0.1",
            "7231864981174d9bee8c7687c24c8344414eae6b",
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
        "factorial-all-layers-all-tokens",
        "factorial-all-layers-downstream-horizon",
        "factorial-probe-suffix-all-tokens",
        "factorial-probe-suffix-downstream-horizon",
        "factorial-random-layers-downstream-horizon",
    )
    assert dict(protocol.training_contract_identities) == {
        "factorial_config_schema": "robustness-adapter-training-config/v7",
        "factorial_method_identity": "probe-output-factorial/v1",
        "factorial_evidence_schema": "probe-output-factorial-evidence-binding/v1",
        "kojima_faithful_config_schema": "robustness-adapter-training-config/v7",
        "kojima_faithful_method_identity": "kojima-faithful-output-matching/v1",
    }
    assert tuple(dict(arm) for arm in protocol.model_specific_arms) == (
        {
            "condition": "kojima-faithful-output-matching",
            "model_id": "mistralai/Mistral-7B-v0.1",
            "model_revision": "7231864981174d9bee8c7687c24c8344414eae6b",
            "pooling": "separate-mistral-only",
            "training_seeds": [42, 43, 44],
        },
    )
    assert tuple(dict(anchor) for anchor in protocol.descriptive_anchors) == (
        {
            "condition": "kojima-faithful-output-matching-public-seed1-anchor",
            "recipe_condition": "kojima-faithful-output-matching",
            "model_id": "mistralai/Mistral-7B-v0.1",
            "model_revision": "7231864981174d9bee8c7687c24c8344414eae6b",
            "training_seed": 1,
            "role": "reproducibility-only-not-pooled",
        },
    )
    assert tuple(dict(value) for value in protocol.primary_novelty_contrasts) == (
        {
            "left": "factorial-all-layers-all-tokens",
            "right": "factorial-probe-suffix-downstream-horizon",
            "difference": "right-minus-left",
            "pooling": "equal-gemma-mistral",
        },
        {
            "left": "factorial-all-layers-downstream-horizon",
            "right": "factorial-probe-suffix-downstream-horizon",
            "difference": "right-minus-left",
            "pooling": "equal-gemma-mistral",
        },
        {
            "left": "factorial-probe-suffix-all-tokens",
            "right": "factorial-probe-suffix-downstream-horizon",
            "difference": "right-minus-left",
            "pooling": "equal-gemma-mistral",
        },
        {
            "left": "factorial-random-layers-downstream-horizon",
            "right": "factorial-probe-suffix-downstream-horizon",
            "difference": "right-minus-left",
            "pooling": "equal-gemma-mistral",
        },
        {
            "left": "kojima-faithful-output-matching",
            "right": "factorial-probe-suffix-downstream-horizon",
            "difference": "right-minus-left",
            "pooling": "mistral-only",
        },
    )

    registry = json.loads(REGISTRY_TEMPLATE.read_text(encoding="utf-8"))
    assert registry["state"] == "pending-calibration"
    assert registry["governance_attestation"] == {
        "adapter_outputs_used_for_calibration": False,
        "severity_grid_extended": False,
        "model_inventory_changed_after_calibration": False,
    }
    assert registry["legacy_random_2"]["role"] == "secondary-continuity-only"
    assert registry["confirmatory"]["random_layer_mask_policy"] == (
        "sha256-seed42-count-matched-random-freeze/v1"
    )
    assert registry["confirmatory"]["mistral_only_direct_comparison_seeds"] == [42, 43, 44]
    assert registry["confirmatory"]["public_seed_1_anchor_role"] == (
        "reproducibility-only-not-pooled"
    )
    assert registry["confirmatory"]["training_contract_identities"] == dict(
        protocol.training_contract_identities
    )
    assert registry["confirmatory"]["final_merged_implementation_commit"] is None
    assert registry["confirmatory"]["final_merged_source_tree_sha256"] is None
    assert registry["confirmatory"]["source_tree_hash_policy"] == (
        "sha256-of-git-ls-tree-r-full-tree-head-lf/v1"
    )


def test_v2_protocol_rejects_random_mask_redrawn_per_learning_seed(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["confirmatory"]["random_freeze_control"] = (
        "same-frozen-layer-count-hash-selected-per-training-seed/v1"
    )
    drifted = tmp_path / "protocol.json"
    drifted.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="confirmatory population differs"):
        load_evaluation_v2_protocol(drifted)


def test_v2_protocol_rejects_feature_commit_as_scientific_training_identity(
    tmp_path: Path,
) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["confirmatory"]["training_contract_identities"] = {
        "factorial_v7": "a" * 40,
        "kojima_faithful_v7": "b" * 40,
    }
    drifted = tmp_path / "protocol.json"
    drifted.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="training_contract_identities fields differ"):
        load_evaluation_v2_protocol(drifted)


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
        "condition": "factorial-probe-suffix-downstream-horizon",
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
    rows = _calibration_rows(selected_at=4)
    observations = tmp_path / "base-observations.jsonl"
    with observations.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    items, typos = _write_calibration_manifests(tmp_path, rows)

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


def test_calibration_rejects_rehashed_stale_source_manifest(tmp_path: Path) -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)
    rows = _calibration_rows(selected_at=4)
    items, typos = _write_calibration_manifests(tmp_path, rows)
    item_values = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines()]
    typo_values = [json.loads(line) for line in typos.read_text(encoding="utf-8").splitlines()]
    stale = item_values[0]
    stale["source_text"] = "internally-valid-but-stale-source"
    stale["source_text_sha256"] = _digest(stale["source_text"])
    for value in typo_values:
        if value["task"] == stale["task"] and value["record_id"] == stale["record_id"]:
            value["source_text_sha256"] = stale["source_text_sha256"]
    items.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in item_values),
        encoding="utf-8",
    )
    typos.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in typo_values),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="observation/source manifest binding differs"):
        validate_calibration_semantic_bindings(
            rows,
            protocol=protocol,
            item_manifest_path=items,
            realized_typo_manifest_path=typos,
        )


def test_calibration_rejects_rehashed_mismatched_typo_manifest(tmp_path: Path) -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)
    rows = _calibration_rows(selected_at=4)
    items, typos = _write_calibration_manifests(tmp_path, rows)
    values = [json.loads(line) for line in typos.read_text(encoding="utf-8").splitlines()]
    values[0]["realized_typo_text"] = "internally-valid-but-wrong-realized-typo"
    values[0]["realized_typo_sha256"] = _digest(values[0]["realized_typo_text"])
    typos.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="observation/typo manifest binding differs"):
        validate_calibration_semantic_bindings(
            rows,
            protocol=protocol,
            item_manifest_path=items,
            realized_typo_manifest_path=typos,
        )


def test_calibration_rejects_arbitrary_files_with_only_a_digest(tmp_path: Path) -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)
    rows = _calibration_rows(selected_at=4)
    items = tmp_path / "items.jsonl"
    typos = tmp_path / "typos.jsonl"
    items.write_text(json.dumps({"sha256": "a" * 64}) + "\n", encoding="utf-8")
    typos.write_text(json.dumps({"sha256": "b" * 64}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="item manifest fields differ"):
        validate_calibration_semantic_bindings(
            rows,
            protocol=protocol,
            item_manifest_path=items,
            realized_typo_manifest_path=typos,
        )


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
                    "factorial-all-layers-all-tokens",
                    "factorial-probe-suffix-downstream-horizon",
                ):
                    for variant in range(protocol.confirmatory_typo_variants_per_item):
                        for seed in protocol.training_seeds:
                            proposal_gain = (
                                condition == "factorial-probe-suffix-downstream-horizon"
                                and item == 0
                                and seed == 42
                            )
                            rows.append(
                                {
                                    "model_id": model.model_id,
                                    "model_revision": model.revision,
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
        left_condition="factorial-all-layers-all-tokens",
        right_condition="factorial-probe-suffix-downstream-horizon",
        outcome="typo_correct",
    )
    reversed_result = clustered_paired_macro_contrast(
        list(reversed(rows)),
        protocol=protocol,
        left_condition="factorial-all-layers-all-tokens",
        right_condition="factorial-probe-suffix-downstream-horizon",
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
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
        )


def test_clustered_bootstrap_rejects_model_revision_drift() -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _confirmatory_rows()
    rows[0]["model_revision"] = "0" * 40

    with pytest.raises(ValueError, match="model revision differs"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
        )


def test_clustered_bootstrap_rejects_clean_outcomes_that_change_by_typo_variant() -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _confirmatory_rows()
    target = rows[0]
    conflicting = next(
        row
        for row in rows
        if row["model_id"] == target["model_id"]
        and row["task"] == target["task"]
        and row["record_id"] == target["record_id"]
        and row["condition"] == target["condition"]
        and row["seed"] == target["seed"]
        and row["variant"] != target["variant"]
    )
    conflicting["clean_correct"] = False

    with pytest.raises(ValueError, match="clean outcome differs across typo variants"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
        )


def _mistral_direct_rows() -> list[dict[str, object]]:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=50,
    )
    model_id = "mistralai/Mistral-7B-v0.1"
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(protocol.tasks):
        for item in range(protocol.confirmatory_records_per_task):
            record_id = f"{task_index * 1000 + item:064x}"
            for condition in (
                "factorial-probe-suffix-downstream-horizon",
                "kojima-faithful-output-matching",
            ):
                for variant in range(protocol.confirmatory_typo_variants_per_item):
                    for seed in protocol.training_seeds:
                        rows.append(
                            {
                                "model_id": model_id,
                                "model_revision": "7231864981174d9bee8c7687c24c8344414eae6b",
                                "task": task,
                                "record_id": record_id,
                                "variant": variant,
                                "condition": condition,
                                "seed": seed,
                                "clean_correct": True,
                                "typo_correct": condition
                                == "factorial-probe-suffix-downstream-horizon",
                            }
                        )
    return rows


def test_mistral_faithful_primary_contrast_uses_matched_replication_seeds() -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=50,
    )

    result = clustered_paired_macro_contrast(
        _mistral_direct_rows(),
        protocol=protocol,
        left_condition="kojima-faithful-output-matching",
        right_condition="factorial-probe-suffix-downstream-horizon",
        outcome="typo_correct",
    )

    assert result.point_difference_points == 100.0
    assert result.source_items == 10


def test_public_seed1_anchor_cannot_enter_primary_mistral_contrast() -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _mistral_direct_rows()
    faithful = next(row for row in rows if row["condition"] == "kojima-faithful-output-matching")
    faithful["seed"] = 1

    with pytest.raises(ValueError, match="adapter seed inventory differs"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="kojima-faithful-output-matching",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
        )

    with pytest.raises(ValueError, match="condition is not preregistered"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="kojima-faithful-output-matching-public-seed1-anchor",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
        )


def test_kojima_faithful_arm_is_rejected_on_gemma() -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _mistral_direct_rows()
    faithful = next(row for row in rows if row["condition"] == "kojima-faithful-output-matching")
    faithful["model_id"] = "google/gemma-3-4b-it"
    faithful["model_revision"] = "093f9f388b31de276ce2de164bdc2081324b9767"

    with pytest.raises(ValueError, match="condition is unavailable for this model"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="kojima-faithful-output-matching",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
        )
