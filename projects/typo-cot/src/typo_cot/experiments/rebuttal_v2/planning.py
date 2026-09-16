"""Pure, outcome-independent construction of the frozen global run plan."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ManifestBundle
from .coordinates import assign_shards, plan_offset
from .donor_artifacts import DonorBankBundle
from .donors import assign_donors
from .identity import identity_hash
from .input_audit import InputAuditBundle
from .runtime_lock import (
    RuntimeLockBundle,
    scientific_config_sha256,
    validate_runtime_inputs,
)
from .schemas import (
    PROTOCOL_SHA256,
    IntakeError,
    canonical_json,
    canonical_sha256,
    require_string_list,
)
from .semantic_labels import SemanticLabelsBundle


PLAN_SCHEMA = "rebuttal-plan/v2"
PREFLIGHT_SCHEMA = "rebuttal-plan-preflight/v2"
COVERAGE_SCHEMA = "rebuttal-plan-coverage/v2"
PLAN_ID_DOMAIN = "rebuttal-plan-row/v2"
GENERATION_ID_DOMAIN = "rebuttal-generation/v2"
SEMANTIC_LABELS = {"unique_preserved", "ambiguous", "task_changed", "unassessable"}
SELECTABLE_EXPERIMENTS = {"R3", "R4"}


@dataclass(frozen=True)
class GlobalPlan:
    rows: list[dict[str, Any]]
    preflight: dict[str, Any]
    coverage: dict[str, Any]
    donor_assignment: dict[str, Any] | None
    scientific_config_sha256: str | None
    expected_generation_ids: list[str]


def _bundle_digest(bundle: Any, path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return bundle.snapshots[resolved]
    except (AttributeError, KeyError) as exc:
        raise IntakeError(f"bundle does not bind its primary artifact: {resolved}") from exc


def _validate_bindings(
    manifest: ManifestBundle,
    labels: SemanticLabelsBundle,
    audit: InputAuditBundle,
    protocol: dict,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    if canonical_sha256(protocol) != PROTOCOL_SHA256:
        raise IntakeError("protocol content does not match frozen revision 2.0.0")
    if canonical_json(protocol) != canonical_json(manifest.protocol):
        raise IntakeError("planner protocol differs from the verified manifest protocol")
    if audit.manifest.path.resolve() != manifest.path.resolve() or _bundle_digest(
        audit.manifest, audit.manifest.path
    ) != _bundle_digest(manifest, manifest.path):
        raise IntakeError("input audit binds a different manifest")

    manifest_pairs = {row["pair_id"]: row for row in manifest.pairs}
    resolved_pairs = {row["pair_id"]: row for row in audit.pairs}
    audit_rows = {row["pair_id"]: row for row in audit.rows}
    if (
        len(manifest_pairs) != len(manifest.pairs)
        or len(resolved_pairs) != len(audit.pairs)
        or len(audit_rows) != len(audit.rows)
        or set(manifest_pairs) != set(resolved_pairs)
        or set(manifest_pairs) != set(audit_rows)
    ):
        raise IntakeError("manifest/input-audit pair inventories differ")

    manifest_ref = manifest.manifest_ref
    for pair_id, pair in manifest_pairs.items():
        resolved = resolved_pairs[pair_id]
        row = audit_rows[pair_id]
        if any(
            resolved.get(field) != pair.get(field)
            for field in ("pair_id", "model_id", "task", "target_rule")
        ) or any(row.get(field) != pair.get(field) for field in ("pair_id", "model_id", "task")):
            raise IntakeError(f"manifest/input-audit identity mismatch for pair {pair_id}")
        expected_record_ref = {"artifact": manifest_ref, "record_id": pair_id}
        if canonical_json(row.get("manifest_ref")) != canonical_json(expected_record_ref):
            raise IntakeError(f"input audit manifest binding mismatch for pair {pair_id}")

    if len(labels.rows) != len(labels.labels_by_pair_id):
        raise IntakeError("semantic labels contain duplicate pair IDs")
    for row in labels.rows:
        pair_id = row.get("pair_id")
        if pair_id not in manifest_pairs:
            raise IntakeError("semantic labels contain an unknown pair ID")
        label = row.get("label")
        if label not in SEMANTIC_LABELS or labels.labels_by_pair_id.get(pair_id) != label:
            raise IntakeError(f"semantic label inventory mismatch for pair {pair_id}")
        expected_record_ref = {"artifact": manifest_ref, "record_id": pair_id}
        if canonical_json(row.get("manifest_record_ref")) != canonical_json(
            expected_record_ref
        ) or canonical_json(row.get("source_ref")) != canonical_json(
            manifest_pairs[pair_id]["source_ref"]
        ):
            raise IntakeError(f"semantic label manifest/source binding mismatch for pair {pair_id}")
    return manifest_pairs, resolved_pairs, audit_rows


def _experiments(value: list[str]) -> list[str]:
    result = require_string_list(value, "experiments", unique=True)
    if not result:
        raise IntakeError("experiments must not be empty")
    if any(item not in SELECTABLE_EXPERIMENTS for item in result):
        raise IntakeError("experiments must be a subset of ('R3', 'R4')")
    return sorted(result)


def _add_blocker(
    blockers: dict[tuple[str | None, str | None], set[str]],
    pair_id: str | None,
    setting: str | None,
    *reasons: str,
) -> None:
    blockers.setdefault((pair_id, setting), set()).update(reasons)


def _selection(
    manifest: ManifestBundle,
    resolved_pairs: dict[str, dict],
    experiments: list[str],
    settings: dict[str, dict],
    blockers: dict[tuple[str | None, str | None], set[str]],
) -> tuple[list[tuple[str, dict]], list[dict], list[dict]]:
    cohorts = [row for row in manifest.source_audit["cohorts"] if row.get("setting") in experiments]
    by_setting = {setting: [] for setting in experiments}
    for row in cohorts:
        by_setting[row["setting"]].append(row)
    for setting, declared in by_setting.items():
        if not declared:
            _add_blocker(blockers, None, setting, "source_cohort_not_declared")
            continue
        if any(row.get("expected_count") is None for row in declared):
            _add_blocker(blockers, None, setting, "source_cohort_expected_ids_unknown")
        if setting == "R4" and any(
            row.get("coverage_status") != "complete" or row.get("missing_count") != 0
            for row in declared
        ):
            _add_blocker(blockers, None, setting, "r4_declared_original_ids_not_fully_recovered")

    selected_cohort_ids = {
        row["cohort_id"]: row["setting"] for row in cohorts if row.get("cohort_id") is not None
    }
    selected: list[tuple[str, dict]] = []
    pair_coverage: list[dict] = []
    for pair_id in sorted(resolved_pairs):
        pair = resolved_pairs[pair_id]
        memberships = [
            member
            for member in pair["cohort_membership"]
            if member["cohort_id"] in selected_cohort_ids
        ]
        selected_settings = sorted(
            {
                selected_cohort_ids[member["cohort_id"]]
                for member in memberships
                if member["membership"] == "member"
            }
        )
        reasons: list[str] = []
        if not selected_settings:
            reasons.append("not_a_verified_member_of_selected_cohort")
            for membership in memberships:
                if membership["membership"] == "extra":
                    reasons.append("selected_cohort_extra_not_selected")
                elif membership["membership"] == "unknown":
                    reasons.append("selected_cohort_membership_unknown")
        elif len(selected_settings) > 1:
            for setting in selected_settings:
                _add_blocker(blockers, pair_id, setting, "pair_selected_for_multiple_settings")
        for setting in selected_settings:
            expected = settings[setting]
            mismatch = []
            if pair["model_id"] != expected["model"]:
                mismatch.append("selected_pair_model_mismatch")
            if pair["task"] != expected["task"]:
                mismatch.append("selected_pair_task_mismatch")
            if mismatch:
                _add_blocker(blockers, pair_id, setting, *mismatch)
            selected.append((setting, pair))
        pair_coverage.append(
            {
                "pair_id": pair_id,
                "selected": bool(selected_settings),
                "settings": selected_settings,
                "cohort_membership": deepcopy(memberships),
                "reason_codes": sorted(set(reasons)),
            }
        )
    selected.sort(key=lambda item: (item[0], item[1]["pair_id"]))
    return selected, pair_coverage, deepcopy(cohorts)


def _prompt_state(pair: dict, audit_row: dict, side: str) -> tuple[str, list[str]]:
    prompt = pair[f"{side}_prompt"]
    if prompt is None:
        return "unknown", [f"{side}_prompt_unknown"]
    if not prompt.strip():
        return "invalid", [f"{side}_prompt_blank"]
    token_ids = audit_row["alignment"][side]["prompt_token_ids"]
    if token_ids == []:
        return "invalid", [f"{side}_prompt_token_sequence_empty"]
    return "valid", []


def _prompt_ref(audit_row: dict, pair: dict, side: str) -> dict[str, Any] | None:
    if pair[f"{side}_prompt"] is None:
        return None
    record_ref = audit_row["manifest_ref"]
    return {
        "artifact": deepcopy(record_ref["artifact"]),
        "record_id": record_ref["record_id"],
        "field": f"{side}_prompt",
    }


def _combine_status(*states: str) -> str:
    if "invalid" in states:
        return "invalid"
    if "unknown" in states:
        return "unknown"
    return "valid"


def _alignment_state(audit_row: dict) -> tuple[str, list[str]]:
    state = audit_row["alignment"]["eligibility"]
    if state == "valid":
        if not audit_row["alignment"]["aligned_edits"]:
            return "invalid", ["no_aligned_edits"]
        return "valid", []
    if state == "unknown":
        return "unknown", ["input_alignment_unknown"]
    return "invalid", ["input_alignment_invalid"]


def _applications(window: list[int]) -> dict[str, dict[str, int]]:
    return {
        "prefill": {str(layer): 1 for layer in range(window[0], window[1])},
        "decode": {str(layer): 0 for layer in range(window[0], window[1])},
    }


def _strip_artifact_paths(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return {"sha256": value["sha256"]}
        return {key: _strip_artifact_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_artifact_paths(item) for item in value]
    return value


def _finish_row(row: dict[str, Any]) -> dict[str, Any]:
    identity_payload = {
        key: value
        for key, value in row.items()
        if key != "plan_id" and not key.startswith("semantic_")
    }
    row["plan_id"] = identity_hash(PLAN_ID_DOMAIN, _strip_artifact_paths(identity_payload))
    return row


def _semantic(pair_id: str, labels: SemanticLabelsBundle) -> dict[str, Any]:
    label = labels.labels_by_pair_id.get(pair_id)
    return {
        "semantic_label": label,
        "semantic_status": "human_labeled" if label is not None else "unknown",
        "semantic_reason_codes": [] if label is not None else ["semantic_label_not_available"],
    }


def _row_base(
    *,
    pair: dict,
    setting: str,
    arm: str,
    window: list[int] | None,
    labels: SemanticLabelsBundle,
    protocol_sha256: str,
    manifest_sha256: str,
    audit_sha256: str,
    generation_hash: str,
    scientific_hash: str,
    donor_bank_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA,
        "plan_id": None,
        "pair_id": pair["pair_id"],
        "original_problem_group_id": pair["original_problem_group_id"],
        "setting": setting,
        "window": deepcopy(window),
        "arm": arm,
        "source_pair_id": None,
        "source_prompt_ref": None,
        "recipient_prompt_ref": None,
        "source_positions": [],
        "write_positions": [],
        "source_tensor_site": None,
        "write_tensor_site": None,
        "application_phase": None,
        "expected_applications": None,
        "generation_config_sha256": generation_hash,
        "scientific_config_sha256": scientific_hash,
        "donor_bank_sha256": donor_bank_sha256,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "input_audit_sha256": audit_sha256,
        "eligibility": "valid",
        "reason_codes": [],
        **_semantic(pair["pair_id"], labels),
    }


def _baseline_row(
    *,
    pair: dict,
    audit_row: dict,
    setting: str,
    arm: str,
    labels: SemanticLabelsBundle,
    common: dict[str, Any],
) -> dict[str, Any]:
    state, reasons = _prompt_state(pair, audit_row, arm)
    row = _row_base(pair=pair, setting=setting, arm=arm, window=None, labels=labels, **common)
    row["recipient_prompt_ref"] = _prompt_ref(audit_row, pair, arm)
    row["eligibility"] = state
    row["reason_codes"] = reasons
    return _finish_row(row)


def _patch_row(
    *,
    pair: dict,
    audit_row: dict,
    setting: str,
    arm: str,
    window: list[int],
    labels: SemanticLabelsBundle,
    common: dict[str, Any],
    protocol: dict,
    assignment: dict[str, Any],
    donor_bank: DonorBankBundle | None,
) -> dict[str, Any]:
    row = _row_base(pair=pair, setting=setting, arm=arm, window=window, labels=labels, **common)
    row.update(
        {
            "source_tensor_site": protocol["patch"]["site"],
            "write_tensor_site": protocol["patch"]["site"],
            "application_phase": "prefill-only",
            "expected_applications": _applications(window),
            "recipient_prompt_ref": _prompt_ref(audit_row, pair, "typo"),
            "source_positions": None,
            "write_positions": None,
        }
    )
    states: list[str] = []
    reasons: list[str] = []
    recipient_state, recipient_reasons = _prompt_state(pair, audit_row, "typo")
    states.append(recipient_state)
    reasons.extend(recipient_reasons)
    alignment_state, alignment_reasons = _alignment_state(audit_row)
    states.append(alignment_state)
    reasons.extend(alignment_reasons)
    edits = audit_row["alignment"]["aligned_edits"]
    write_positions = (
        [edit["typo_endpoint"] for edit in edits] if alignment_state == "valid" else None
    )
    row["write_positions"] = write_positions

    if arm in {"self", "correct", "offset"}:
        source_side = "typo" if arm == "self" else "clean"
        source_state, source_reasons = _prompt_state(pair, audit_row, source_side)
        states.append(source_state)
        reasons.extend(source_reasons)
        row["source_pair_id"] = pair["pair_id"]
        row["source_prompt_ref"] = _prompt_ref(audit_row, pair, source_side)
        if alignment_state == "valid":
            if arm == "self":
                row["source_positions"] = [edit["typo_endpoint"] for edit in edits]
            elif arm == "correct":
                row["source_positions"] = [edit["clean_endpoint"] for edit in edits]
            else:
                offset = plan_offset(
                    edits,
                    len(audit_row["alignment"]["clean"]["prompt_token_ids"]),
                    len(audit_row["alignment"]["typo"]["prompt_token_ids"]),
                )
                row["source_positions"] = offset["source_positions"]
                row["write_positions"] = offset["write_positions"]
                states.append(offset["eligibility"])
                reasons.extend(offset["reason_codes"])
    else:
        if donor_bank is None:
            reasons.extend(assignment["reason_codes"])
            states.append("not_available")
        else:
            reasons.extend(assignment["reason_codes"])
            donor_id = assignment["donor_pair_id"]
            if donor_id is None:
                if "donor_bank_insufficient" in assignment["reason_codes"]:
                    states.append("invalid")
                elif alignment_state == "invalid":
                    states.append("invalid")
                else:
                    states.append("unknown")
            else:
                donor_rows = {item["pair_id"]: item for item in donor_bank.rows}
                donor_pairs = {item["pair_id"]: item for item in donor_bank.pairs}
                donor = donor_rows[donor_id]
                donor_pair = donor_pairs[donor_id]
                donor_audit = donor_bank.audit_rows_by_pair_id[donor_id]
                donor_state, donor_reasons = _prompt_state(donor_pair, donor_audit, "clean")
                states.append(donor_state)
                reasons.extend(donor_reasons)
                row["source_pair_id"] = donor_id
                row["source_prompt_ref"] = _prompt_ref(donor_audit, donor_pair, "clean")
                row["source_positions"] = deepcopy(donor["clean_endpoints"])

    if row["source_pair_id"] is None:
        reasons.append("source_pair_id_unresolved")
    if row["source_prompt_ref"] is None:
        reasons.append("source_prompt_ref_unresolved")
    if row["recipient_prompt_ref"] is None:
        reasons.append("recipient_prompt_ref_unresolved")
    if row["source_positions"] is None:
        reasons.append("source_positions_unresolved")
    if row["write_positions"] is None:
        reasons.append("write_positions_unresolved")
    row["eligibility"] = (
        "not_available" if arm == "cross" and donor_bank is None else _combine_status(*states)
    )
    row["reason_codes"] = sorted(set(reasons))
    return _finish_row(row)


def _format_blockers(
    blockers: dict[tuple[str | None, str | None], set[str]],
) -> list[dict[str, Any]]:
    return [
        {"pair_id": pair_id, "setting": setting, "reason_codes": sorted(reasons)}
        for (pair_id, setting), reasons in sorted(
            blockers.items(), key=lambda item: (item[0][1] or "", item[0][0] or "")
        )
    ]


def _coverage(
    *,
    manifest_pairs: dict[str, dict],
    selected: list[tuple[str, dict]],
    pair_coverage: list[dict],
    source_cohorts: list[dict],
    labels: SemanticLabelsBundle,
    rows: list[dict],
    blockers: list[dict],
    experiments: list[str],
    settings: dict[str, dict],
) -> dict[str, Any]:
    selected_ids = sorted({pair["pair_id"] for _, pair in selected})
    by_setting = {
        setting: sum(pair_setting == setting for pair_setting, _ in selected)
        for setting in experiments
    }
    intended = sum(2 + 4 * len(settings[setting]["windows"]) for setting, _ in selected)
    status_counts = {
        status: sum(row["eligibility"] == status for row in rows)
        for status in ("valid", "invalid", "unknown", "not_available")
    }
    return {
        "schema_version": COVERAGE_SCHEMA,
        "selected_pair_ids": selected_ids,
        "pair_coverage": pair_coverage,
        "source_cohort_coverage": source_cohorts,
        "counts": {
            "manifest_pair_count": len(manifest_pairs),
            "selected_pair_count": len(selected_ids),
            "selected_pair_count_by_setting": by_setting,
            "human_labeled_selected_pair_count": sum(
                pair_id in labels.labels_by_pair_id for pair_id in selected_ids
            ),
            "semantic_unknown_selected_pair_count": sum(
                pair_id not in labels.labels_by_pair_id for pair_id in selected_ids
            ),
            "intended_plan_row_count": intended,
            "planned_row_count": len(rows),
            "plan_eligibility_counts": status_counts,
            "expected_generation_count": status_counts["valid"],
            "global_blocker_count": len(blockers),
            "global_blocking_pair_count": len(
                {row["pair_id"] for row in blockers if row["pair_id"] is not None}
            ),
        },
    }


def build_run_plan(
    manifest: ManifestBundle,
    labels: SemanticLabelsBundle,
    alignments: InputAuditBundle,
    protocol: dict,
    *,
    runtime_lock: RuntimeLockBundle,
    experiments: list[str],
    donor_bank: DonorBankBundle | None = None,
) -> GlobalPlan:
    """Build the complete deterministic plan without publishing artifacts."""
    selected_experiments = _experiments(experiments)
    manifest_pairs, resolved_pairs, audit_rows = _validate_bindings(
        manifest, labels, alignments, protocol
    )
    settings = {row["id"]: row for row in protocol["settings"]}
    blockers: dict[tuple[str | None, str | None], set[str]] = {}
    selected, pair_coverage, source_cohorts = _selection(
        manifest, resolved_pairs, selected_experiments, settings, blockers
    )

    for setting, pair in selected:
        pair_id = pair["pair_id"]
        if pair["original_problem_group_id"] is None:
            _add_blocker(blockers, pair_id, setting, "original_problem_group_id_unknown")
        if setting == "R4":
            for side in ("clean", "typo"):
                for field in ("text", "prompt"):
                    value = pair[f"{side}_{field}"]
                    if value is None or not value.strip():
                        name = "full_prompt" if field == "prompt" else "text"
                        _add_blocker(blockers, pair_id, setting, f"r4_{side}_{name}_unavailable")

    selected_groups = {
        pair["original_problem_group_id"]
        for _, pair in selected
        if pair["original_problem_group_id"] is not None
    }
    if donor_bank is not None:
        selected_ids = {pair["pair_id"] for _, pair in selected}
        for donor in donor_bank.rows:
            if donor["pair_id"] in selected_ids:
                setting = next(s for s, pair in selected if pair["pair_id"] == donor["pair_id"])
                _add_blocker(
                    blockers,
                    donor["pair_id"],
                    setting,
                    "donor_pair_id_overlaps_selected_recipient",
                )
            if donor["original_problem_group_id"] in selected_groups:
                for setting, pair in selected:
                    if pair["original_problem_group_id"] == donor["original_problem_group_id"]:
                        _add_blocker(
                            blockers,
                            pair["pair_id"],
                            setting,
                            "donor_original_problem_group_overlaps_selected_recipients",
                        )

    runtime_blockers = validate_runtime_inputs(
        runtime_lock,
        alignments,
        [pair for _, pair in selected],
        donor_bank=donor_bank,
    )
    for row in runtime_blockers:
        _add_blocker(
            blockers,
            row["pair_id"],
            row["setting"],
            *row["reason_codes"],
        )

    formatted_blockers = _format_blockers(blockers)
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "blocked" if formatted_blockers else "passed",
        "blockers": formatted_blockers,
        "blocking_pair_ids": sorted(
            {row["pair_id"] for row in formatted_blockers if row["pair_id"] is not None}
        ),
        "experiments": selected_experiments,
    }
    if formatted_blockers:
        coverage = _coverage(
            manifest_pairs=manifest_pairs,
            selected=selected,
            pair_coverage=pair_coverage,
            source_cohorts=source_cohorts,
            labels=labels,
            rows=[],
            blockers=formatted_blockers,
            experiments=selected_experiments,
            settings=settings,
        )
        return GlobalPlan([], preflight, coverage, None, None, [])

    scientific_hash = scientific_config_sha256(runtime_lock, protocol, selected_experiments)
    recipients = [
        {
            "pair_id": pair["pair_id"],
            "original_problem_group_id": pair["original_problem_group_id"],
            "model_id": pair["model_id"],
            "task": pair["task"],
            "target_rule": pair["target_rule"],
            "aligned_word_count": len(audit_rows[pair["pair_id"]]["alignment"]["aligned_edits"])
            if audit_rows[pair["pair_id"]]["alignment"]["eligibility"] == "valid"
            else 0,
        }
        for _, pair in selected
    ]
    donor_assignment = assign_donors(
        recipients,
        None if donor_bank is None else donor_bank.rows,
        protocol,
        recipient_groups=selected_groups,
    )
    assignments = {row["pair_id"]: row for row in donor_assignment["assignments"]}
    manifest_hash = _bundle_digest(manifest, manifest.path)
    audit_hash = _bundle_digest(alignments, alignments.path)
    donor_hash = None if donor_bank is None else donor_bank.reference["sha256"]
    rows: list[dict[str, Any]] = []
    for setting, pair in selected:
        audit_row = audit_rows[pair["pair_id"]]
        runtime_entry = runtime_lock.settings[setting]
        generation_hash = canonical_sha256(
            {
                "generation": runtime_entry["generation"],
                "effective_eos_ids": runtime_entry["effective_eos_ids"],
            }
        )
        common = {
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": manifest_hash,
            "audit_sha256": audit_hash,
            "generation_hash": generation_hash,
            "scientific_hash": scientific_hash,
            "donor_bank_sha256": donor_hash,
        }
        for arm in ("clean", "typo"):
            rows.append(
                _baseline_row(
                    pair=pair,
                    audit_row=audit_row,
                    setting=setting,
                    arm=arm,
                    labels=labels,
                    common=common,
                )
            )
        for window in settings[setting]["windows"]:
            for arm in ("self", "correct", "offset", "cross"):
                rows.append(
                    _patch_row(
                        pair=pair,
                        audit_row=audit_row,
                        setting=setting,
                        arm=arm,
                        window=window,
                        labels=labels,
                        common=common,
                        protocol=protocol,
                        assignment=assignments[pair["pair_id"]],
                        donor_bank=donor_bank,
                    )
                )
    arm_order = {arm: index for index, arm in enumerate(protocol["arms"])}
    rows.sort(
        key=lambda row: (
            row["setting"],
            row["pair_id"],
            row["window"] is not None,
            row["window"] or [],
            arm_order[row["arm"]],
        )
    )
    expected_generation_ids = sorted(
        identity_hash(
            GENERATION_ID_DOMAIN,
            {
                "plan_id": row["plan_id"],
                "scientific_config_sha256": scientific_hash,
            },
        )
        for row in rows
        if row["eligibility"] == "valid"
    )
    coverage = _coverage(
        manifest_pairs=manifest_pairs,
        selected=selected,
        pair_coverage=pair_coverage,
        source_cohorts=source_cohorts,
        labels=labels,
        rows=rows,
        blockers=formatted_blockers,
        experiments=selected_experiments,
        settings=settings,
    )
    return GlobalPlan(
        rows,
        preflight,
        coverage,
        donor_assignment,
        scientific_hash,
        expected_generation_ids,
    )


__all__ = ["GlobalPlan", "assign_shards", "build_run_plan", "plan_offset"]
