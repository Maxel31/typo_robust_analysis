"""Falsification tests for the preregistered Base-only evaluation v2."""

from __future__ import annotations

import json
import subprocess
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
from typo_robust_training.evaluation.registry_v2 import (
    canonical_source_tree_sha256,
    load_confirmatory_semantic_binding,
    load_evaluation_opening_sealed_evaluation_v2_registry,
    load_ready_evaluation_v2_registry,
    load_training_preregistered_evaluation_v2_registry,
    validate_outcomes_against_confirmatory_binding,
    validate_tier_id_disjointness,
)
from typo_robust_training.evaluation.statistics_v2 import clustered_paired_macro_contrast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs/robustness-evaluation-v2.yaml"
REGISTRY_TEMPLATE = PROJECT_ROOT / "configs/robustness-evaluation-v2-registry.template.json"


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _calibration_rows(
    *, selected_at: int | None, protocol=None
) -> tuple[BaseCalibrationObservation, ...]:
    protocol = protocol or load_evaluation_v2_protocol(PROTOCOL_PATH)
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
                        incorrect = index < int(protocol.calibration_records_per_task * 0.04)
                    elif severity == selected_at:
                        incorrect = index < max(
                            1, int(protocol.calibration_records_per_task * 0.10)
                        )
                    else:
                        incorrect = index < min(
                            protocol.calibration_records_per_task - 1,
                            max(1, int(protocol.calibration_records_per_task * 0.40)),
                        )
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
                                reference_answer_sha256=_digest(f"answer:{task}:{record_id}"),
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
        reference_answer = f"answer:{row.task}:{row.record_id}"
        typo_text = f"typo:{row.task}:{row.record_id}:{row.severity_edit_count}:{row.variant}"
        item_rows[(row.task, row.record_id)] = {
            "schema_version": "robustness-evaluation-v2-calibration-item/v1",
            "task": row.task,
            "record_id": row.record_id,
            "source_text": source_text,
            "source_text_sha256": _digest(source_text),
            "reference_answer": reference_answer,
            "reference_answer_sha256": _digest(reference_answer),
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
    assert registry["legacy_random_2"]["inherited_runtime_contracts"] == [
        "generation",
        "typos.eligibility",
        "corpus_runtime",
    ]
    assert registry["legacy_random_2"]["runtime_contract_sha256"] == dict(
        protocol.runtime_contract_sha256
    )
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


def test_v2_protocol_rejects_unfrozen_generation_contract(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["legacy_v1"]["inherited_runtime_contracts"] = ["generation"]
    drifted = tmp_path / "protocol.json"
    drifted.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="legacy random-2 contract differs"):
        load_evaluation_v2_protocol(drifted)


def test_v2_protocol_rejects_rehashed_runtime_subcontract(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["legacy_v1"]["runtime_contract_sha256"]["extractor"] = "0" * 64
    drifted = tmp_path / "protocol.json"
    drifted.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="legacy random-2 contract differs"):
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
        "reference_answer_sha256": "4" * 64,
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


def test_calibration_rejects_self_rehashed_reference_answer_manifest(
    tmp_path: Path,
) -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)
    rows = _calibration_rows(selected_at=4)
    items, typos = _write_calibration_manifests(tmp_path, rows)
    values = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines()]
    values[0]["reference_answer"] = "self-rehashed-but-not-the-frozen-answer"
    values[0]["reference_answer_sha256"] = _digest(values[0]["reference_answer"])
    items.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference answer manifest binding differs"):
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


def test_calibration_rejects_same_source_relabelled_into_another_task(tmp_path: Path) -> None:
    protocol = load_evaluation_v2_protocol(PROTOCOL_PATH)
    rows = _calibration_rows(selected_at=4)
    items, typos = _write_calibration_manifests(tmp_path, rows)
    item_rows = [json.loads(line) for line in items.read_text().splitlines()]
    typo_rows = [json.loads(line) for line in typos.read_text().splitlines()]
    source = item_rows[0]
    relabelled = next(row for row in item_rows if row["task"] != source["task"])
    old_hash = relabelled["source_text_sha256"]
    relabelled["source_text"] = source["source_text"]
    relabelled["source_text_sha256"] = source["source_text_sha256"]
    for row in typo_rows:
        if row["source_text_sha256"] == old_hash:
            row["source_text_sha256"] = source["source_text_sha256"]
    items.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in item_rows),
        encoding="utf-8",
    )
    typos.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in typo_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not globally unique"):
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
                                    "source_text_sha256": _digest(f"source:{task}:{record_id}"),
                                    "reference_answer_sha256": _digest(
                                        f"answer:{task}:{record_id}"
                                    ),
                                    "variant": variant,
                                    "realized_typo_sha256": _digest(
                                        f"typo:{task}:{record_id}:{variant}"
                                    ),
                                    "condition": condition,
                                    "seed": seed,
                                    "clean_correct": True,
                                    "typo_correct": proposal_gain,
                                }
                            )
    return rows


def _write_confirmatory_binding(
    tmp_path: Path,
    protocol,
    *,
    selected_edit_count: int = 2,
    record_offset: int = 0,
):
    items = tmp_path / "confirmatory-items.jsonl"
    typos = tmp_path / "confirmatory-typos.jsonl"
    item_rows: list[dict[str, object]] = []
    typo_rows: list[dict[str, object]] = []
    for task_index, task in enumerate(protocol.tasks):
        for item in range(protocol.confirmatory_records_per_task):
            record_id = f"{record_offset + task_index * 1000 + item:064x}"
            source_text = f"source:{task}:{record_id}"
            source_hash = _digest(source_text)
            reference_answer = f"answer:{task}:{record_id}"
            item_rows.append(
                {
                    "schema_version": "robustness-evaluation-v2-confirmatory-item/v1",
                    "task": task,
                    "record_id": record_id,
                    "source_text": source_text,
                    "source_text_sha256": source_hash,
                    "reference_answer": reference_answer,
                    "reference_answer_sha256": _digest(reference_answer),
                }
            )
            for variant in range(protocol.confirmatory_typo_variants_per_item):
                typo_text = f"typo:{task}:{record_id}:{variant}"
                typo_rows.append(
                    {
                        "schema_version": "robustness-evaluation-v2-confirmatory-typo/v1",
                        "task": task,
                        "record_id": record_id,
                        "source_text_sha256": source_hash,
                        "severity_edit_count": selected_edit_count,
                        "variant": variant,
                        "realized_typo_text": typo_text,
                        "realized_typo_sha256": _digest(typo_text),
                    }
                )
    items.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in item_rows),
        encoding="utf-8",
    )
    typos.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in typo_rows),
        encoding="utf-8",
    )
    return load_confirmatory_semantic_binding(
        protocol=protocol,
        selected_edit_count=selected_edit_count,
        item_manifest_path=items,
        realized_typo_manifest_path=typos,
    )


def test_clustered_bootstrap_averages_variants_and_seeds_before_item_resampling(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=100,
    )
    rows = _confirmatory_rows()
    binding = _write_confirmatory_binding(tmp_path, protocol)

    result = clustered_paired_macro_contrast(
        rows,
        protocol=protocol,
        left_condition="factorial-all-layers-all-tokens",
        right_condition="factorial-probe-suffix-downstream-horizon",
        outcome="typo_correct",
        semantic_binding=binding,
    )
    reversed_result = clustered_paired_macro_contrast(
        list(reversed(rows)),
        protocol=protocol,
        left_condition="factorial-all-layers-all-tokens",
        right_condition="factorial-probe-suffix-downstream-horizon",
        outcome="typo_correct",
        semantic_binding=binding,
    )

    # One of three seeds improves one of two source items: (1/3) / 2.
    assert result.point_difference_points == pytest.approx(100.0 / 6.0)
    assert result.source_items == 10
    assert reversed_result == result


def test_clustered_bootstrap_rejects_duplicate_seed_rows_as_false_replication(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _confirmatory_rows()
    rows.append(dict(rows[0]))
    binding = _write_confirmatory_binding(tmp_path, protocol)

    with pytest.raises(ValueError, match="contain duplicates"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )


def test_clustered_bootstrap_rejects_model_revision_drift(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _confirmatory_rows()
    rows[0]["model_revision"] = "0" * 40
    binding = _write_confirmatory_binding(tmp_path, protocol)

    with pytest.raises(ValueError, match="model revision differs"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )


def test_clustered_bootstrap_rejects_clean_outcomes_that_change_by_typo_variant(
    tmp_path: Path,
) -> None:
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
    binding = _write_confirmatory_binding(tmp_path, protocol)

    with pytest.raises(ValueError, match="clean outcome differs across typo variants"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )


def test_clustered_bootstrap_rejects_arm_specific_realized_typo_text(tmp_path: Path) -> None:
    """A variant label is not a semantic identity for a frozen typo."""

    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _confirmatory_rows()
    target = next(
        row for row in rows if row["condition"] == "factorial-probe-suffix-downstream-horizon"
    )
    target["realized_typo_sha256"] = _digest("arm-specific-rehashed-typo")
    binding = _write_confirmatory_binding(tmp_path, protocol)

    with pytest.raises(ValueError, match="outcome/typo manifest binding differs"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )


def test_clustered_bootstrap_rejects_arm_specific_source_text(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _confirmatory_rows()
    target = next(
        row for row in rows if row["condition"] == "factorial-probe-suffix-downstream-horizon"
    )
    target["source_text_sha256"] = _digest("arm-specific-rehashed-source")
    binding = _write_confirmatory_binding(tmp_path, protocol)

    with pytest.raises(ValueError, match="outcome/source manifest binding differs"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="factorial-all-layers-all-tokens",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )


def test_self_rehashed_confirmatory_manifest_cannot_rebind_frozen_outcomes(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    original = _write_confirmatory_binding(tmp_path, protocol)
    items = tmp_path / "confirmatory-items.jsonl"
    typos = tmp_path / "confirmatory-typos.jsonl"
    item_rows = [json.loads(line) for line in items.read_text().splitlines()]
    typo_rows = [json.loads(line) for line in typos.read_text().splitlines()]
    changed = item_rows[0]
    old_hash = changed["source_text_sha256"]
    changed["source_text"] = "self-rehashed-but-not-the-frozen-source"
    changed["source_text_sha256"] = _digest(changed["source_text"])
    for row in typo_rows:
        if row["source_text_sha256"] == old_hash:
            row["source_text_sha256"] = changed["source_text_sha256"]
    items.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in item_rows),
        encoding="utf-8",
    )
    typos.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in typo_rows),
        encoding="utf-8",
    )
    rebound = load_confirmatory_semantic_binding(
        protocol=protocol,
        selected_edit_count=2,
        item_manifest_path=items,
        realized_typo_manifest_path=typos,
    )
    assert rebound.item_manifest_sha256 != original.item_manifest_sha256

    with pytest.raises(ValueError, match="outcome/source manifest binding differs"):
        validate_outcomes_against_confirmatory_binding(
            _confirmatory_rows(),
            protocol=protocol,
            binding=rebound,
        )


def test_self_rehashed_reference_answer_cannot_rebind_frozen_outcomes(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    frozen = _write_confirmatory_binding(tmp_path, protocol)
    items = tmp_path / "confirmatory-items.jsonl"
    item_rows = [json.loads(line) for line in items.read_text().splitlines()]
    changed = item_rows[0]
    changed["reference_answer"] = "self-rehashed-but-not-the-frozen-answer"
    changed["reference_answer_sha256"] = _digest(changed["reference_answer"])
    items.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in item_rows),
        encoding="utf-8",
    )
    rebound = load_confirmatory_semantic_binding(
        protocol=protocol,
        selected_edit_count=2,
        item_manifest_path=items,
        realized_typo_manifest_path=tmp_path / "confirmatory-typos.jsonl",
    )
    assert rebound.item_manifest_sha256 != frozen.item_manifest_sha256
    outcomes = _confirmatory_rows()
    for row in outcomes:
        if row["task"] == changed["task"] and row["record_id"] == changed["record_id"]:
            row["reference_answer_sha256"] = changed["reference_answer_sha256"]

    with pytest.raises(ValueError, match="reference answer manifest binding differs"):
        validate_outcomes_against_confirmatory_binding(
            outcomes,
            protocol=protocol,
            binding=frozen,
        )


def _write_tier_role_manifest(
    tmp_path: Path,
    *,
    binding,
    calibration_sources: dict[tuple[str, str], str],
) -> Path:
    rows: list[dict[str, object]] = []

    def add(role: str, record_id: str, source_hash: str) -> None:
        rows.append(
            {
                "schema_version": "robustness-evaluation-v2-tier-role-id/v1",
                "role": role,
                "record_id": record_id,
                "source_text_sha256": source_hash,
            }
        )

    for index, role in enumerate(
        (
            "training",
            "linear-probe-selection",
            "linear-probe-validation",
            "tune",
            "pre-pr",
        ),
        start=1,
    ):
        add(role, f"{900_000 + index:064x}", _digest(f"role-source:{role}"))
    for (_task, record_id), source_hash in calibration_sources.items():
        add("calibration", record_id, source_hash)
    for (_task, record_id), source_hash in binding.source_hashes.items():
        add("confirmatory", record_id, source_hash)
    path = tmp_path / "tier-role-ids.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_tier_manifest_rejects_rehashed_id_with_confirmatory_source_leak(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
    )
    binding = _write_confirmatory_binding(tmp_path, protocol)
    calibration_paths, calibration_sources = _write_calibration_registry_inputs(
        tmp_path, protocol=protocol, selected_edit_count=binding.selected_edit_count
    )
    role_path = _write_tier_role_manifest(
        tmp_path,
        binding=binding,
        calibration_sources=calibration_sources,
    )
    rows = [json.loads(line) for line in role_path.read_text().splitlines()]
    leaked_hash = next(iter(binding.source_hashes.values()))
    training = next(row for row in rows if row["role"] == "training")
    training["record_id"] = _digest("new-id-for-leaked-confirmatory-source")
    training["source_text_sha256"] = leaked_hash
    role_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source text leaks across data tiers"):
        validate_tier_id_disjointness(
            role_manifest_path=role_path,
            calibration_sources=calibration_sources,
            confirmatory_binding=binding,
        )


def _initialize_registry_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True
    )
    subprocess.run(["git", "config", "user.name", "Evaluation Test"], cwd=repository, check=True)
    (repository / "implementation.py").write_text("FROZEN = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "implementation.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "frozen implementation"], cwd=repository, check=True)
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/develop", "HEAD"],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit, canonical_source_tree_sha256(repository, commit=commit)


def _write_ready_registry(
    tmp_path: Path,
    *,
    protocol,
    binding,
    tier_hash: str,
    commit: str,
    tree_hash: str,
    calibration_paths: tuple[Path, Path, Path, Path],
    phase: str = "evaluation-opening-sealed",
) -> Path:
    artifact_names = {
        "factorial_arm_registry_sha256": "factorial-arm-registry.json",
        "probe_artifact_registry_sha256": "probe-artifact-registry.json",
        "training_config_registry_sha256": "training-config-registry.json",
        "training_data_registry_sha256": "training-data-registry.json",
        "mistral_kojima_faithful_matched_seed_42_43_44_registry_sha256": (
            "mistral-matched-seed-registry.json"
        ),
        "mistral_kojima_faithful_public_seed_1_anchor_checkpoint_sha256": (
            "mistral-public-seed1-checkpoint.json"
        ),
        "arm_checkpoint_registry_sha256": "arm-checkpoint-registry.json",
        "opening_log_sha256": "opening-log.json",
    }
    artifact_hashes: dict[str, str] = {}
    for field, name in artifact_names.items():
        artifact_path = tmp_path / name
        artifact_path.write_text(
            json.dumps({"artifact": field}, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_hashes[field] = _digest(artifact_path.read_text())
    legacy_path = tmp_path / "legacy-random-2-registry.json"
    legacy_path.write_text('{"registry":"legacy-random-2"}\n', encoding="utf-8")

    registry = json.loads(REGISTRY_TEMPLATE.read_text(encoding="utf-8"))
    registry["state"] = phase
    registry["protocol_sha256"] = protocol.config_sha256
    registry["calibration"].update(
        {
            "status": "selected",
            "item_manifest_sha256": _digest(calibration_paths[1].read_text()),
            "realized_typo_manifest_sha256": _digest(calibration_paths[2].read_text()),
            "base_observations_sha256": _digest(calibration_paths[0].read_text()),
            "result_sha256": _digest(calibration_paths[3].read_text()),
            "selected_primary_edit_count": binding.selected_edit_count,
        }
    )
    registry["confirmatory"].update(
        {
            "status": phase,
            "final_merged_implementation_commit": commit,
            "final_merged_source_tree_sha256": tree_hash,
            "item_manifest_sha256": binding.item_manifest_sha256,
            "realized_typo_variant_manifest_sha256": (binding.realized_typo_manifest_sha256),
            "tier_id_role_manifest_sha256": tier_hash,
            "factorial_arm_registry_sha256": artifact_hashes["factorial_arm_registry_sha256"],
            "probe_artifact_registry_sha256": artifact_hashes["probe_artifact_registry_sha256"],
            "training_config_registry_sha256": artifact_hashes["training_config_registry_sha256"],
            "training_data_registry_sha256": artifact_hashes["training_data_registry_sha256"],
        }
    )
    if phase == "evaluation-opening-sealed":
        training_preregistered = json.loads(json.dumps(registry))
        training_preregistered["state"] = "training-preregistered"
        training_preregistered["confirmatory"]["status"] = "training-preregistered"
        training_preregistered_path = tmp_path / "training-preregistered-registry.json"
        training_preregistered_path.write_text(
            json.dumps(training_preregistered, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry["confirmatory"].update(
            {
                "training_preregistered_registry_sha256": _digest(
                    training_preregistered_path.read_text()
                ),
                "mistral_kojima_faithful_matched_seed_42_43_44_registry_sha256": (
                    artifact_hashes["mistral_kojima_faithful_matched_seed_42_43_44_registry_sha256"]
                ),
                "mistral_kojima_faithful_public_seed_1_anchor_checkpoint_sha256": (
                    artifact_hashes[
                        "mistral_kojima_faithful_public_seed_1_anchor_checkpoint_sha256"
                    ]
                ),
                "arm_checkpoint_registry_sha256": artifact_hashes["arm_checkpoint_registry_sha256"],
                "opening_log_sha256": artifact_hashes["opening_log_sha256"],
            }
        )
    registry["legacy_random_2"]["frozen_registry_sha256"] = _digest(legacy_path.read_text())
    if phase == "evaluation-opening-sealed":
        training_preregistered["legacy_random_2"]["frozen_registry_sha256"] = _digest(
            legacy_path.read_text()
        )
        training_preregistered_path.write_text(
            json.dumps(training_preregistered, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry["confirmatory"]["training_preregistered_registry_sha256"] = _digest(
            training_preregistered_path.read_text()
        )
    path = tmp_path / "sealed-registry.json"
    path.write_text(json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_calibration_registry_inputs(
    tmp_path: Path, *, protocol, selected_edit_count: int
) -> tuple[tuple[Path, Path, Path, Path], dict[tuple[str, str], str]]:
    rows = _calibration_rows(selected_at=selected_edit_count, protocol=protocol)
    observations = tmp_path / "base-observations.jsonl"
    observations.write_text(
        "".join(
            json.dumps(row.as_dict(), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    items, typos = _write_calibration_manifests(tmp_path, rows)
    result = run_base_only_severity_calibration(
        config_path=PROTOCOL_PATH,
        observations_path=observations,
        item_manifest_path=items,
        realized_typo_manifest_path=typos,
        output_dir=tmp_path / "calibration-output",
    )
    sources = {(row.task, row.record_id): row.source_text_sha256 for row in rows}
    return (observations, items, typos, result.artifact_path), sources


def _preregistered_registry_artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "confirmatory_item_manifest_path": root / "confirmatory-items.jsonl",
        "confirmatory_typo_manifest_path": root / "confirmatory-typos.jsonl",
        "factorial_arm_registry_path": root / "factorial-arm-registry.json",
        "probe_artifact_registry_path": root / "probe-artifact-registry.json",
        "training_config_registry_path": root / "training-config-registry.json",
        "training_data_registry_path": root / "training-data-registry.json",
        "legacy_random_2_registry_path": root / "legacy-random-2-registry.json",
    }


def _opening_registry_artifact_paths(root: Path) -> dict[str, Path]:
    return {
        **_preregistered_registry_artifact_paths(root),
        "training_preregistered_registry_path": (root / "training-preregistered-registry.json"),
        "mistral_matched_seed_registry_path": root / "mistral-matched-seed-registry.json",
        "mistral_public_seed_1_checkpoint_path": (root / "mistral-public-seed1-checkpoint.json"),
        "arm_checkpoint_registry_path": root / "arm-checkpoint-registry.json",
        "opening_log_path": root / "opening-log.json",
    }


def test_ready_registry_binds_manifests_and_clean_merged_source_tree(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
    )
    binding = _write_confirmatory_binding(tmp_path, protocol, record_offset=100_000)
    calibration_paths, calibration_sources = _write_calibration_registry_inputs(
        tmp_path, protocol=protocol, selected_edit_count=binding.selected_edit_count
    )
    role_path = _write_tier_role_manifest(
        tmp_path,
        binding=binding,
        calibration_sources=calibration_sources,
    )
    tier_hash = validate_tier_id_disjointness(
        role_manifest_path=role_path,
        calibration_sources=calibration_sources,
        confirmatory_binding=binding,
    )
    repository, commit, tree_hash = _initialize_registry_repository(tmp_path)
    registry_path = _write_ready_registry(
        tmp_path,
        protocol=protocol,
        binding=binding,
        tier_hash=tier_hash,
        commit=commit,
        tree_hash=tree_hash,
        calibration_paths=calibration_paths,
    )

    loaded = load_ready_evaluation_v2_registry(
        registry_path=registry_path,
        protocol=protocol,
        repository_path=repository,
        calibration_observations_path=calibration_paths[0],
        calibration_item_manifest_path=calibration_paths[1],
        calibration_typo_manifest_path=calibration_paths[2],
        calibration_result_path=calibration_paths[3],
        tier_role_manifest_path=role_path,
        **_opening_registry_artifact_paths(tmp_path),
    )
    assert loaded["state"] == "evaluation-opening-sealed"

    opening_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    opening_payload["confirmatory"]["training_preregistered_registry_sha256"] = "f" * 64
    wrong_parent_path = tmp_path / "wrong-parent-opening-registry.json"
    wrong_parent_path.write_text(
        json.dumps(opening_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="training-preregistered phase binding differs"):
        load_ready_evaluation_v2_registry(
            registry_path=wrong_parent_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_opening_registry_artifact_paths(tmp_path),
        )

    probe_registry_path = tmp_path / "probe-artifact-registry.json"
    frozen_probe_registry = probe_registry_path.read_text(encoding="utf-8")
    probe_registry_path.write_text('{"artifact":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="probe_artifact_registry_sha256 binding differs"):
        load_training_preregistered_evaluation_v2_registry(
            registry_path=tmp_path / "training-preregistered-registry.json",
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_preregistered_registry_artifact_paths(tmp_path),
        )
    probe_registry_path.write_text(frozen_probe_registry, encoding="utf-8")

    with pytest.raises(ValueError, match="not training-preregistered"):
        load_training_preregistered_evaluation_v2_registry(
            registry_path=registry_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_preregistered_registry_artifact_paths(tmp_path),
        )

    preregistered_path = tmp_path / "training-preregistered-registry.json"
    preregistered = json.loads(preregistered_path.read_text(encoding="utf-8"))
    training_loaded = load_training_preregistered_evaluation_v2_registry(
        registry_path=preregistered_path,
        protocol=protocol,
        repository_path=repository,
        calibration_observations_path=calibration_paths[0],
        calibration_item_manifest_path=calibration_paths[1],
        calibration_typo_manifest_path=calibration_paths[2],
        calibration_result_path=calibration_paths[3],
        tier_role_manifest_path=role_path,
        **_preregistered_registry_artifact_paths(tmp_path),
    )
    assert training_loaded["state"] == "training-preregistered"

    with pytest.raises(ValueError, match="not evaluation-opening-sealed"):
        load_evaluation_opening_sealed_evaluation_v2_registry(
            registry_path=preregistered_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_opening_registry_artifact_paths(tmp_path),
        )

    incomplete_opening = json.loads(preregistered_path.read_text(encoding="utf-8"))
    incomplete_opening["state"] = "evaluation-opening-sealed"
    incomplete_opening["confirmatory"]["status"] = "evaluation-opening-sealed"
    incomplete_opening_path = tmp_path / "incomplete-opening-registry.json"
    incomplete_opening_path.write_text(
        json.dumps(incomplete_opening, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be a lowercase SHA-256"):
        load_evaluation_opening_sealed_evaluation_v2_registry(
            registry_path=incomplete_opening_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_opening_registry_artifact_paths(tmp_path),
        )

    preregistered["confirmatory"]["arm_checkpoint_registry_sha256"] = "e" * 64
    preregistered_path.write_text(
        json.dumps(preregistered, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contains post-training artifacts"):
        load_training_preregistered_evaluation_v2_registry(
            registry_path=preregistered_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_preregistered_registry_artifact_paths(tmp_path),
        )
    preregistered["confirmatory"]["arm_checkpoint_registry_sha256"] = None
    preregistered_path.write_text(
        json.dumps(preregistered, sort_keys=True) + "\n", encoding="utf-8"
    )

    (repository / "implementation.py").write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree is not clean"):
        load_ready_evaluation_v2_registry(
            registry_path=registry_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_opening_registry_artifact_paths(tmp_path),
        )


def test_ready_registry_rejects_runtime_contract_hash_drift(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
    )
    binding = _write_confirmatory_binding(tmp_path, protocol, record_offset=100_000)
    calibration_paths, calibration_sources = _write_calibration_registry_inputs(
        tmp_path, protocol=protocol, selected_edit_count=binding.selected_edit_count
    )
    role_path = _write_tier_role_manifest(
        tmp_path,
        binding=binding,
        calibration_sources=calibration_sources,
    )
    tier_hash = validate_tier_id_disjointness(
        role_manifest_path=role_path,
        calibration_sources=calibration_sources,
        confirmatory_binding=binding,
    )
    repository, commit, tree_hash = _initialize_registry_repository(tmp_path)
    registry_path = _write_ready_registry(
        tmp_path,
        protocol=protocol,
        binding=binding,
        tier_hash=tier_hash,
        commit=commit,
        tree_hash=tree_hash,
        calibration_paths=calibration_paths,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["legacy_random_2"]["runtime_contract_sha256"]["prompt"] = "0" * 64
    registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="legacy random-2 role differs"):
        load_ready_evaluation_v2_registry(
            registry_path=registry_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_opening_registry_artifact_paths(tmp_path),
        )


def test_ready_registry_recomputes_base_only_calibration_result(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
    )
    binding = _write_confirmatory_binding(tmp_path, protocol, record_offset=100_000)
    calibration_paths, calibration_sources = _write_calibration_registry_inputs(
        tmp_path, protocol=protocol, selected_edit_count=binding.selected_edit_count
    )
    role_path = _write_tier_role_manifest(
        tmp_path,
        binding=binding,
        calibration_sources=calibration_sources,
    )
    tier_hash = validate_tier_id_disjointness(
        role_manifest_path=role_path,
        calibration_sources=calibration_sources,
        confirmatory_binding=binding,
    )
    repository, commit, tree_hash = _initialize_registry_repository(tmp_path)
    registry_path = _write_ready_registry(
        tmp_path,
        protocol=protocol,
        binding=binding,
        tier_hash=tier_hash,
        commit=commit,
        tree_hash=tree_hash,
        calibration_paths=calibration_paths,
    )
    artifact = json.loads(calibration_paths[3].read_text(encoding="utf-8"))
    artifact["summaries"]["2"]["model_equal_macro_gap_points"] = 99.0
    calibration_paths[3].write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["calibration"]["result_sha256"] = _digest(calibration_paths[3].read_text())
    registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="calibration result artifact differs"):
        load_ready_evaluation_v2_registry(
            registry_path=registry_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_opening_registry_artifact_paths(tmp_path),
        )


def test_ready_registry_reparses_self_rehashed_tier_manifest(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
    )
    binding = _write_confirmatory_binding(tmp_path, protocol, record_offset=100_000)
    calibration_paths, calibration_sources = _write_calibration_registry_inputs(
        tmp_path, protocol=protocol, selected_edit_count=binding.selected_edit_count
    )
    role_path = _write_tier_role_manifest(
        tmp_path,
        binding=binding,
        calibration_sources=calibration_sources,
    )
    tier_hash = validate_tier_id_disjointness(
        role_manifest_path=role_path,
        calibration_sources=calibration_sources,
        confirmatory_binding=binding,
    )
    repository, commit, tree_hash = _initialize_registry_repository(tmp_path)
    registry_path = _write_ready_registry(
        tmp_path,
        protocol=protocol,
        binding=binding,
        tier_hash=tier_hash,
        commit=commit,
        tree_hash=tree_hash,
        calibration_paths=calibration_paths,
    )
    rows = [json.loads(line) for line in role_path.read_text().splitlines()]
    training = next(row for row in rows if row["role"] == "training")
    training["record_id"] = _digest("new-training-record-id")
    training["source_text_sha256"] = next(iter(binding.source_hashes.values()))
    role_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["confirmatory"]["tier_id_role_manifest_sha256"] = _digest(role_path.read_text())
    registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source text leaks across data tiers"):
        load_ready_evaluation_v2_registry(
            registry_path=registry_path,
            protocol=protocol,
            repository_path=repository,
            calibration_observations_path=calibration_paths[0],
            calibration_item_manifest_path=calibration_paths[1],
            calibration_typo_manifest_path=calibration_paths[2],
            calibration_result_path=calibration_paths[3],
            tier_role_manifest_path=role_path,
            **_opening_registry_artifact_paths(tmp_path),
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
                                "source_text_sha256": _digest(f"source:{task}:{record_id}"),
                                "reference_answer_sha256": _digest(f"answer:{task}:{record_id}"),
                                "variant": variant,
                                "realized_typo_sha256": _digest(
                                    f"typo:{task}:{record_id}:{variant}"
                                ),
                                "condition": condition,
                                "seed": seed,
                                "clean_correct": True,
                                "typo_correct": condition
                                == "factorial-probe-suffix-downstream-horizon",
                            }
                        )
    return rows


def test_mistral_faithful_primary_contrast_uses_matched_replication_seeds(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=50,
    )
    binding = _write_confirmatory_binding(tmp_path, protocol)

    result = clustered_paired_macro_contrast(
        _mistral_direct_rows(),
        protocol=protocol,
        left_condition="kojima-faithful-output-matching",
        right_condition="factorial-probe-suffix-downstream-horizon",
        outcome="typo_correct",
        semantic_binding=binding,
    )

    assert result.point_difference_points == 100.0
    assert result.source_items == 10


def test_public_seed1_anchor_cannot_enter_primary_mistral_contrast(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _mistral_direct_rows()
    faithful = next(row for row in rows if row["condition"] == "kojima-faithful-output-matching")
    faithful["seed"] = 1
    binding = _write_confirmatory_binding(tmp_path, protocol)

    with pytest.raises(ValueError, match="adapter seed inventory differs"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="kojima-faithful-output-matching",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )

    with pytest.raises(ValueError, match="condition is not preregistered"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="kojima-faithful-output-matching-public-seed1-anchor",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )


def test_kojima_faithful_arm_is_rejected_on_gemma(tmp_path: Path) -> None:
    protocol = replace(
        load_evaluation_v2_protocol(PROTOCOL_PATH),
        confirmatory_records_per_task=2,
        bootstrap_replicates=10,
    )
    rows = _mistral_direct_rows()
    faithful = next(row for row in rows if row["condition"] == "kojima-faithful-output-matching")
    faithful["model_id"] = "google/gemma-3-4b-it"
    faithful["model_revision"] = "093f9f388b31de276ce2de164bdc2081324b9767"
    binding = _write_confirmatory_binding(tmp_path, protocol)

    with pytest.raises(ValueError, match="condition is unavailable for this model"):
        clustered_paired_macro_contrast(
            rows,
            protocol=protocol,
            left_condition="kojima-faithful-output-matching",
            right_condition="factorial-probe-suffix-downstream-horizon",
            outcome="typo_correct",
            semantic_binding=binding,
        )
