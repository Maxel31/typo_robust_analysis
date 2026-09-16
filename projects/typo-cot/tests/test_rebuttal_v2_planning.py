"""Pure global planning over verified public intake/audit fixtures."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_cot.experiments.rebuttal_v2.donor_artifacts import load_donor_bank
from typo_cot.experiments.rebuttal_v2.identity import identity_hash
from typo_cot.experiments.rebuttal_v2.input_audit import load_input_audit, run_input_audit
from typo_cot.experiments.rebuttal_v2.planning import build_run_plan
from typo_cot.experiments.rebuttal_v2.runtime_lock import RuntimeLockBundle
from typo_cot.experiments.rebuttal_v2.schemas import (
    IntakeError,
    canonical_sha256,
    sha256_text,
)
from typo_cot.experiments.rebuttal_v2.semantic_labels import SemanticLabelsBundle


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((PROJECT / "configs/rebuttal_v2/protocol.json").read_text())


def _test_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INPUT = _test_module("test_rebuttal_v2_input_audit.py", "_planning_input_fixture")
_DONOR = _test_module("test_rebuttal_v2_donor_artifacts.py", "_planning_donor_fixture")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(root: Path, *, pair_changes=None):
    manifest_path, tokenizer_lock = _INPUT._fixture(
        root / "recipient", pair_changes={"target_rule": "rule-a", **(pair_changes or {})}
    )
    audit_dir = root / "audit"
    run_input_audit(manifest_path, tokenizer_lock, audit_dir)
    audit = load_input_audit(audit_dir / "input_audit_records.jsonl")
    return SimpleNamespace(manifest=audit.manifest, audit=audit)


def _labels(root: Path, manifest, values: dict[str, str]) -> SemanticLabelsBundle:
    rows = [
        {
            "pair_id": pair_id,
            "label": label,
            "source_ref": next(pair for pair in manifest.pairs if pair["pair_id"] == pair_id)[
                "source_ref"
            ],
            "manifest_record_ref": {
                "artifact": manifest.manifest_ref,
                "record_id": pair_id,
            },
        }
        for pair_id, label in sorted(values.items())
    ]
    path = root / "semantic_labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    run = root / "run.json"
    run.write_text("{}\n", encoding="utf-8")
    return SemanticLabelsBundle(
        path.resolve(),
        run.resolve(),
        rows,
        dict(values),
        {path.resolve(): _sha(path), run.resolve(): _sha(run)},
    )


def _runtime(
    root: Path,
    audit,
    experiments: list[str],
    *,
    donor_bank=None,
) -> RuntimeLockBundle:
    descriptor_by_model = audit.tokenizer_lock["models"]
    audit_rows = {row["pair_id"]: row for row in audit.rows}
    settings = {}
    protocol_settings = {row["id"]: row for row in PROTOCOL["settings"]}
    for setting in experiments:
        selected = [
            pair
            for pair in audit.pairs
            if pair["model_id"] == protocol_settings[setting]["model"]
            and pair["task"] == protocol_settings[setting]["task"]
        ]
        model = protocol_settings[setting]["model"]
        descriptor = copy.deepcopy(descriptor_by_model[model])
        policy = {
            key: copy.deepcopy(value)
            for key, value in descriptor.items()
            if key not in {"tokenizer_id", "revision"}
        }
        entry = {
            "model_id": model,
            "model_revision": "1" * 40,
            "model_files": {"model.safetensors": "2" * 64},
            "tokenizer_id": descriptor["tokenizer_id"],
            "tokenizer_revision": descriptor["revision"],
            "tokenizer_policy": policy,
            "generation": copy.deepcopy(PROTOCOL["generation"]),
            "attention_backend": "sdpa",
            "cache_behavior": {"use_cache": True, "implementation": "dynamic"},
            "effective_eos_ids": [1, 2],
            "library_versions": {
                "tokenizers": policy["library"]["version"],
                "transformers": "4.57.6",
                "torch": "2.10.0",
            },
            "code_commit": "3" * 40,
            "code_tree_sha256": "4" * 64,
            "device_identity": {
                "type": "cuda",
                "model": "Fixture GPU",
                "compute_capability": "8.0",
            },
            "compute_settings": {
                "cuda_version": "13.0",
                "driver_version": "999.0",
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "matmul_precision": "highest",
            },
            "prompt_sha256": {},
            "prompt_token_ids_sha256": {},
            "donor_prompt_sha256": {},
            "donor_prompt_token_ids_sha256": {},
        }
        for pair in selected:
            row = audit_rows[pair["pair_id"]]
            entry["prompt_sha256"][pair["pair_id"]] = {
                side: None
                if pair[f"{side}_prompt"] is None
                else sha256_text(pair[f"{side}_prompt"])
                for side in ("clean", "typo")
            }
            entry["prompt_token_ids_sha256"][pair["pair_id"]] = {
                side: None
                if pair[f"{side}_prompt"] is None
                else canonical_sha256(row["alignment"][side]["prompt_token_ids"])
                for side in ("clean", "typo")
            }
        if donor_bank is not None:
            for donor, donor_pair in zip(donor_bank.rows, donor_bank.pairs, strict=True):
                if (
                    donor["model_id"] != model
                    or donor["task"] != protocol_settings[setting]["task"]
                ):
                    continue
                donor_audit = donor_bank.audit_rows_by_pair_id[donor["pair_id"]]
                entry["donor_prompt_sha256"][donor["pair_id"]] = sha256_text(
                    donor_pair["clean_prompt"]
                )
                entry["donor_prompt_token_ids_sha256"][donor["pair_id"]] = canonical_sha256(
                    donor_audit["alignment"]["clean"]["prompt_token_ids"]
                )
        settings[setting] = entry
    path = root / "runtime_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return RuntimeLockBundle(path.resolve(), {}, settings, {path.resolve(): _sha(path)})


def _build(root: Path, case, *, label="unique_preserved", donor_bank=None, experiments=None):
    pair_id = case.audit.pairs[0]["pair_id"]
    labels = _labels(root / "labels", case.manifest, {} if label is None else {pair_id: label})
    experiments = experiments or ["R3"]
    runtime = _runtime(root / "runtime", case.audit, experiments, donor_bank=donor_bank)
    return build_run_plan(
        case.manifest,
        labels,
        case.audit,
        PROTOCOL,
        runtime_lock=runtime,
        experiments=experiments,
        donor_bank=donor_bank,
    )


def test_actual_partial_r3_fixture_plans_recovered_member_without_refill(tmp_path):
    case = _case(tmp_path)
    plan = _build(tmp_path, case)
    assert plan.preflight["status"] == "passed"
    assert len(plan.rows) == 6
    assert plan.coverage["source_cohort_coverage"][0]["coverage_status"] == "partial"
    assert plan.coverage["source_cohort_coverage"][0]["missing_source_pair_keys"] == [
        "missing-original"
    ]
    assert plan.coverage["counts"]["selected_pair_count"] == 1
    assert len(plan.expected_generation_ids) == sum(
        row["eligibility"] == "valid" for row in plan.rows
    )


def test_baselines_ignore_invalid_alignment_and_semantic_selection(tmp_path):
    changes = _INPUT._text_pair_changes("cat", "Xcat-Y")
    case = _case(tmp_path, pair_changes=changes)
    assert case.audit.rows[0]["alignment"]["eligibility"] == "invalid"
    plan = _build(tmp_path, case, label="task_changed")
    baselines = [row for row in plan.rows if row["window"] is None]
    patches = [row for row in plan.rows if row["window"] is not None]
    assert [row["eligibility"] for row in baselines] == ["valid", "valid"]
    assert all(row["eligibility"] == "invalid" for row in patches[:3])
    assert all(row["semantic_label"] == "task_changed" for row in plan.rows)


@pytest.mark.parametrize(
    "label", ["unique_preserved", "ambiguous", "task_changed", "unassessable", None]
)
def test_semantics_are_analysis_only_and_do_not_change_scientific_identity(tmp_path, label):
    case = _case(tmp_path)
    reference = _build(tmp_path / "reference", case, label="unique_preserved")
    compared = _build(tmp_path / "compared", case, label=label)
    assert [row["plan_id"] for row in compared.rows] == [row["plan_id"] for row in reference.rows]
    assert [row["eligibility"] for row in compared.rows] == [
        row["eligibility"] for row in reference.rows
    ]
    assert compared.expected_generation_ids == reference.expected_generation_ids
    expected_status = "unknown" if label is None else "human_labeled"
    assert {row["semantic_status"] for row in compared.rows} == {expected_status}


def test_missing_prompt_is_unknown_but_blank_or_empty_tokens_are_invalid(tmp_path):
    missing = _case(
        tmp_path / "missing",
        pair_changes={"clean_prompt": None, "clean_query_span": None},
    )
    missing_plan = _build(tmp_path / "missing", missing)
    clean = next(row for row in missing_plan.rows if row["arm"] == "clean")
    typo = next(row for row in missing_plan.rows if row["arm"] == "typo")
    assert clean["eligibility"] == "unknown"
    assert clean["recipient_prompt_ref"] is None
    assert typo["eligibility"] == "valid"

    blank = _case(
        tmp_path / "blank",
        pair_changes={"clean_prompt": " ", "clean_query_span": None},
    )
    blank_plan = _build(tmp_path / "blank", blank)
    clean = next(row for row in blank_plan.rows if row["arm"] == "clean")
    assert clean["eligibility"] == "invalid"
    assert {"clean_prompt_blank", "clean_prompt_token_sequence_empty"}.intersection(
        clean["reason_codes"]
    )


def test_null_group_is_a_global_blocker_and_returns_no_partial_plan(tmp_path):
    case = _case(tmp_path, pair_changes={"original_problem_id": None})
    plan = _build(tmp_path, case)
    pair_id = case.audit.pairs[0]["pair_id"]
    assert plan.preflight["status"] == "blocked"
    assert plan.preflight["blocking_pair_ids"] == [pair_id]
    assert "original_problem_group_id_unknown" in plan.preflight["blockers"][0]["reason_codes"]
    assert plan.rows == plan.expected_generation_ids == []
    assert plan.donor_assignment is None
    assert plan.coverage["selected_pair_ids"] == [pair_id]


def test_runtime_prompt_mismatch_is_global_not_a_row_level_skip(tmp_path):
    case = _case(tmp_path)
    pair_id = case.audit.pairs[0]["pair_id"]
    labels = _labels(tmp_path / "labels", case.manifest, {})
    runtime = _runtime(tmp_path / "runtime", case.audit, ["R3"])
    runtime.settings["R3"]["prompt_sha256"][pair_id]["clean"] = "0" * 64
    plan = build_run_plan(
        case.manifest,
        labels,
        case.audit,
        PROTOCOL,
        runtime_lock=runtime,
        experiments=["R3"],
    )
    assert plan.rows == []
    assert plan.preflight["blocking_pair_ids"] == [pair_id]
    assert "clean_runtime_prompt_sha256_mismatch" in plan.preflight["blockers"][0]["reason_codes"]


def test_missing_bank_retains_cross_rows_as_not_available(tmp_path):
    case = _case(tmp_path)
    plan = _build(tmp_path, case)
    cross = [row for row in plan.rows if row["arm"] == "cross"]
    assert len(cross) == 1
    assert cross[0]["eligibility"] == "not_available"
    assert "donor_bank_not_available" in cross[0]["reason_codes"]
    assert cross[0]["donor_bank_sha256"] is None
    assert plan.donor_assignment["donor_bank_available"] is False


def test_verified_donor_uses_clean_endpoints_and_manifest_prompt_ref(tmp_path):
    case = _case(tmp_path / "recipient")
    fixture = _DONOR._public_audit_fixture(tmp_path / "donor")
    bank = load_donor_bank(fixture["bank"], PROTOCOL)
    plan = _build(tmp_path / "plan", case, donor_bank=bank)
    cross = next(row for row in plan.rows if row["arm"] == "cross")
    assignment = plan.donor_assignment["assignments"][0]
    donor_id = assignment["donor_pair_id"]
    donor_row = next(row for row in bank.rows if row["pair_id"] == donor_id)
    donor_audit = bank.audit_rows_by_pair_id[donor_id]
    assert cross["eligibility"] == "valid"
    assert cross["source_pair_id"] == donor_id
    assert cross["source_positions"] == donor_row["clean_endpoints"]
    assert cross["write_positions"] == [
        edit["typo_endpoint"] for edit in case.audit.rows[0]["alignment"]["aligned_edits"]
    ]
    assert cross["source_prompt_ref"] == {
        **donor_audit["manifest_ref"],
        "field": "clean_prompt",
    }


def test_correct_self_and_offset_retain_complete_endpoint_arrays(tmp_path):
    case = _case(tmp_path)
    plan = _build(tmp_path, case)
    edits = case.audit.rows[0]["alignment"]["aligned_edits"]
    by_arm = {row["arm"]: row for row in plan.rows}
    clean = [edit["clean_endpoint"] for edit in edits]
    typo = [edit["typo_endpoint"] for edit in edits]
    assert by_arm["correct"]["source_positions"] == clean
    assert by_arm["correct"]["write_positions"] == typo
    assert by_arm["self"]["source_positions"] == typo
    assert by_arm["self"]["write_positions"] == typo
    assert by_arm["offset"]["source_positions"] == [position + 2 for position in clean]
    assert by_arm["offset"]["write_positions"] == [position + 2 for position in typo]
    assert by_arm["offset"]["expected_applications"] == {
        "prefill": {str(layer): 1 for layer in range(6)},
        "decode": {str(layer): 0 for layer in range(6)},
    }


def test_supplied_empty_bank_is_shortage_invalid_not_unavailable(tmp_path):
    case = _case(tmp_path / "recipient")
    bank_path = tmp_path / "empty-bank.jsonl"
    bank_path.write_bytes(b"")
    bank = load_donor_bank(bank_path, PROTOCOL)
    plan = _build(tmp_path / "plan", case, donor_bank=bank)
    cross = next(row for row in plan.rows if row["arm"] == "cross")
    assert cross["eligibility"] == "invalid"
    assert "donor_bank_insufficient" in cross["reason_codes"]
    assert cross["donor_bank_sha256"] == _sha(bank_path)


def test_donor_group_overlap_is_collected_as_global_preflight_blocker(tmp_path):
    case = _case(tmp_path / "recipient")
    fixture = _DONOR._public_audit_fixture(tmp_path / "donor")
    bank = load_donor_bank(fixture["bank"], PROTOCOL)
    bank.rows[0]["original_problem_group_id"] = case.audit.pairs[0]["original_problem_group_id"]
    plan = _build(tmp_path / "plan", case, donor_bank=bank)
    assert plan.rows == []
    assert plan.donor_assignment is None
    assert plan.preflight["blocking_pair_ids"] == [case.audit.pairs[0]["pair_id"]]
    assert (
        "donor_original_problem_group_overlaps_selected_recipients"
        in plan.preflight["blockers"][0]["reason_codes"]
    )


def _as_complete_r4(case) -> None:
    r4 = next(row for row in PROTOCOL["settings"] if row["id"] == "R4")
    for pair in case.manifest.pairs:
        pair.update(model_id=r4["model"], task=r4["task"])
    for pair in case.audit.pairs:
        pair.update(model_id=r4["model"], task=r4["task"])
    for row in case.audit.rows:
        row.update(model_id=r4["model"], task=r4["task"])
    descriptor = case.audit.tokenizer_lock["models"].pop("google/gemma-3-4b-it")
    case.audit.tokenizer_lock["models"][r4["model"]] = descriptor
    cohort = case.manifest.source_audit["cohorts"][0]
    cohort.update(
        setting="R4",
        coverage_status="complete",
        expected_count=1,
        recovered_count=1,
        missing_count=0,
        missing_source_pair_keys=[],
    )


def test_complete_r4_has_exact_ten_rows_and_two_windows(tmp_path):
    case = _case(tmp_path)
    _as_complete_r4(case)
    plan = _build(tmp_path, case, experiments=["R4"])
    assert plan.preflight["status"] == "passed"
    assert len(plan.rows) == 10
    assert sum(row["window"] is None for row in plan.rows) == 2
    assert {tuple(row["window"]) for row in plan.rows if row["window"] is not None} == {
        (0, 6),
        (6, 12),
    }


def test_r4_incomplete_declared_ids_or_missing_full_prompt_blocks_globally(tmp_path):
    incomplete = _case(tmp_path / "incomplete")
    _as_complete_r4(incomplete)
    incomplete.manifest.source_audit["cohorts"][0].update(
        coverage_status="partial", missing_count=1, missing_source_pair_keys=["missing"]
    )
    plan = _build(tmp_path / "incomplete", incomplete, experiments=["R4"])
    assert plan.rows == []
    assert any(
        "r4_declared_original_ids_not_fully_recovered" in row["reason_codes"]
        for row in plan.preflight["blockers"]
    )

    missing = _case(tmp_path / "missing")
    _as_complete_r4(missing)
    missing.manifest.pairs[0]["clean_prompt"] = None
    missing.audit.pairs[0]["clean_prompt"] = None
    plan = _build(tmp_path / "missing", missing, experiments=["R4"])
    assert plan.rows == []
    assert any(
        "r4_clean_full_prompt_unavailable" in row["reason_codes"]
        for row in plan.preflight["blockers"]
    )

    missing_text = _case(tmp_path / "missing-text")
    _as_complete_r4(missing_text)
    missing_text.manifest.pairs[0]["typo_text"] = " "
    missing_text.audit.pairs[0]["typo_text"] = " "
    plan = _build(tmp_path / "missing-text", missing_text, experiments=["R4"])
    assert plan.rows == []
    assert any(
        "r4_typo_text_unavailable" in row["reason_codes"] for row in plan.preflight["blockers"]
    )


def test_plan_and_generation_identity_oracles_and_typed_config(tmp_path):
    case = _case(tmp_path)
    plan = _build(tmp_path, case)
    row = plan.rows[0]
    assert set(row) == {
        "schema_version",
        "plan_id",
        "pair_id",
        "original_problem_group_id",
        "setting",
        "window",
        "arm",
        "source_pair_id",
        "source_prompt_ref",
        "recipient_prompt_ref",
        "source_positions",
        "write_positions",
        "source_tensor_site",
        "write_tensor_site",
        "application_phase",
        "expected_applications",
        "generation_config_sha256",
        "scientific_config_sha256",
        "donor_bank_sha256",
        "protocol_sha256",
        "manifest_sha256",
        "input_audit_sha256",
        "eligibility",
        "reason_codes",
        "semantic_label",
        "semantic_status",
        "semantic_reason_codes",
    }

    def strip_refs(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                return {"sha256": value["sha256"]}
            return {key: strip_refs(item) for key, item in value.items()}
        if isinstance(value, list):
            return [strip_refs(item) for item in value]
        return value

    payload = {
        key: value
        for key, value in row.items()
        if key != "plan_id" and not key.startswith("semantic_")
    }
    assert row["plan_id"] == identity_hash("rebuttal-plan-row/v2", strip_refs(payload))
    expected_id = identity_hash(
        "rebuttal-generation/v2",
        {
            "plan_id": row["plan_id"],
            "scientific_config_sha256": plan.scientific_config_sha256,
        },
    )
    if row["eligibility"] == "valid":
        assert expected_id in plan.expected_generation_ids
    runtime = _runtime(tmp_path / "typed-runtime", case.audit, ["R3"])
    assert row["generation_config_sha256"] == canonical_sha256(
        {
            "generation": runtime.settings["R3"]["generation"],
            "effective_eos_ids": runtime.settings["R3"]["effective_eos_ids"],
        }
    )


def test_input_order_does_not_affect_rows_or_assignment(tmp_path):
    case = _case(tmp_path)
    forward = _build(tmp_path / "forward", case)
    case.manifest.pairs.reverse()
    case.audit.pairs.reverse()
    case.audit.rows.reverse()
    case.manifest.source_audit["cohorts"].reverse()
    reverse = _build(tmp_path / "reverse", case)
    assert reverse.rows == forward.rows
    assert reverse.donor_assignment == forward.donor_assignment
    assert reverse.expected_generation_ids == forward.expected_generation_ids


def test_experiment_order_is_canonicalized(tmp_path):
    case = _case(tmp_path)
    r4 = next(row for row in PROTOCOL["settings"] if row["id"] == "R4")
    case.audit.tokenizer_lock["models"][r4["model"]] = copy.deepcopy(
        case.audit.tokenizer_lock["models"]["google/gemma-3-4b-it"]
    )
    labels = _labels(tmp_path / "labels", case.manifest, {})
    runtime = _runtime(tmp_path / "runtime", case.audit, ["R3", "R4"])
    forward = build_run_plan(
        case.manifest,
        labels,
        case.audit,
        PROTOCOL,
        runtime_lock=runtime,
        experiments=["R3", "R4"],
    )
    reverse = build_run_plan(
        case.manifest,
        labels,
        case.audit,
        PROTOCOL,
        runtime_lock=runtime,
        experiments=["R4", "R3"],
    )
    assert reverse == forward
    assert reverse.preflight["experiments"] == ["R3", "R4"]


def test_invalid_experiment_selection_and_protocol_are_rejected(tmp_path):
    case = _case(tmp_path)
    labels = _labels(tmp_path / "labels", case.manifest, {})
    runtime = _runtime(tmp_path / "runtime", case.audit, ["R3"])
    for experiments in ([], ["R5"], ["R3", "R3"]):
        with pytest.raises(IntakeError):
            build_run_plan(
                case.manifest,
                labels,
                case.audit,
                PROTOCOL,
                runtime_lock=runtime,
                experiments=experiments,
            )
    changed = copy.deepcopy(PROTOCOL)
    changed["seed"] = 43
    with pytest.raises(IntakeError, match="protocol content"):
        build_run_plan(
            case.manifest,
            labels,
            case.audit,
            changed,
            runtime_lock=runtime,
            experiments=["R3"],
        )
