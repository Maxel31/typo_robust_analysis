"""Pure, outcome-independent donor assignment for the rebuttal v2 plan."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .identity import identity_hash
from .schemas import (
    PROTOCOL_SHA256,
    IntakeError,
    canonical_sha256,
    require_int,
    require_nullable_string,
    require_object,
    require_string,
)

ASSIGNMENT_SCHEMA_VERSION = "rebuttal-donor-assignment/v2"
DONOR_ORDER_DOMAIN = "rebuttal-donor-order/v2"

_ROW_FIELDS = (
    "pair_id",
    "original_problem_group_id",
    "model_id",
    "task",
    "target_rule",
    "aligned_word_count",
)


def _validate_protocol(protocol: object) -> dict[str, object]:
    value = require_object(protocol, "protocol")
    if canonical_sha256(value) != PROTOCOL_SHA256:
        raise IntakeError("protocol content does not match frozen revision 2.0.0")
    return value


def _normalize_row(row: object, *, role: str, index: int) -> dict[str, Any]:
    context = f"{role}[{index}]"
    value = require_object(row, context)
    missing = set(_ROW_FIELDS).difference(value)
    if missing:
        raise IntakeError(f"{context} missing required fields: {', '.join(sorted(missing))}")

    # Deliberately read only assignment inputs.  Gold, outcomes, labels, shard
    # metadata, and every other extension must be causally inert here.
    return {
        "pair_id": require_string(value["pair_id"], f"{context}.pair_id"),
        "original_problem_group_id": require_string(
            value["original_problem_group_id"], f"{context}.original_problem_group_id"
        ),
        "model_id": require_string(value["model_id"], f"{context}.model_id"),
        "task": require_string(value["task"], f"{context}.task"),
        "target_rule": require_nullable_string(value["target_rule"], f"{context}.target_rule"),
        "aligned_word_count": require_int(
            value["aligned_word_count"], f"{context}.aligned_word_count", minimum=0
        ),
    }


def _normalize_rows(rows: object, *, role: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise IntakeError(f"{role} must be a JSON array")
    normalized = [_normalize_row(row, role=role, index=index) for index, row in enumerate(rows)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in normalized:
        pair_id = row["pair_id"]
        if pair_id in seen:
            duplicates.add(pair_id)
        seen.add(pair_id)
    if duplicates:
        raise IntakeError(f"{role} contains duplicate pair IDs: {', '.join(sorted(duplicates))}")
    return normalized


def _normalize_recipient_groups(recipient_groups: object) -> set[str]:
    if not isinstance(recipient_groups, set):
        raise IntakeError("recipient_groups must be a set of non-empty strings")
    return {require_string(value, "recipient_groups member") for value in recipient_groups}


def _stratum(row: dict[str, Any]) -> tuple[str, str, str, int]:
    target_rule = row["target_rule"]
    if target_rule is None or row["aligned_word_count"] == 0:
        raise AssertionError("an unknown stratum cannot be ordered")
    return (
        row["model_id"],
        row["task"],
        target_rule,
        row["aligned_word_count"],
    )


def _order_key(row: dict[str, Any], *, role: str) -> tuple[str, str]:
    model_id, task, target_rule, aligned_word_count = _stratum(row)
    digest = identity_hash(
        DONOR_ORDER_DOMAIN,
        {
            "seed": 42,
            "role": role,
            "stratum": {
                "model_id": model_id,
                "task": task,
                "target_rule": target_rule,
                "aligned_word_count": aligned_word_count,
            },
            "pair_id": row["pair_id"],
        },
    )
    return digest, row["pair_id"]


def _recipient_reason_codes(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row["target_rule"] is None:
        reasons.append("target_rule_unknown")
    if row["aligned_word_count"] == 0:
        reasons.append("aligned_word_count_zero")
    return reasons


def assign_donors(
    recipients: list[dict],
    donor_bank: list[dict] | None,
    protocol: dict,
    *,
    recipient_groups: set[str] | None = None,
) -> dict[str, object]:
    """Assign a fixed donor bank one-to-one within exact frozen strata.

    The function is pure: it neither mutates its arguments nor consults runtime,
    outcome, label, gold, permutation, or shard metadata.
    """

    _validate_protocol(protocol)
    normalized_recipients = _normalize_rows(recipients, role="recipients")
    recipient_ids = {row["pair_id"] for row in normalized_recipients}

    selected_groups = {row["original_problem_group_id"] for row in normalized_recipients}
    if recipient_groups is not None:
        # The explicit set can include selected variants that are not among the
        # rows assigned in this call.  Current rows remain protected as well.
        selected_groups.update(_normalize_recipient_groups(recipient_groups))

    if donor_bank is None:
        assignments = []
        for row in normalized_recipients:
            reasons = _recipient_reason_codes(row)
            reasons.append("donor_bank_not_available")
            assignments.append(
                {
                    "pair_id": row["pair_id"],
                    "donor_pair_id": None,
                    "eligibility": "not_available",
                    "reason_codes": sorted(reasons),
                }
            )
        return {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "assignments": sorted(assignments, key=lambda row: row["pair_id"]),
            "unused_donor_pair_ids": [],
            "donor_bank_available": False,
        }

    normalized_donors = _normalize_rows(donor_bank, role="donor_bank")
    donor_ids = {row["pair_id"] for row in normalized_donors}
    colliding_ids = recipient_ids.intersection(donor_ids)
    if colliding_ids:
        raise IntakeError(
            "donor pair IDs overlap recipient pair IDs: " + ", ".join(sorted(colliding_ids))
        )

    overlapping_groups = selected_groups.intersection(
        row["original_problem_group_id"] for row in normalized_donors
    )
    if overlapping_groups:
        raise IntakeError(
            "donor original-problem groups overlap selected recipient groups: "
            + ", ".join(sorted(overlapping_groups))
        )

    for row in normalized_donors:
        if row["target_rule"] is None or row["aligned_word_count"] == 0:
            raise IntakeError(f"donor {row['pair_id']} has unknown assignment stratum metadata")

    recipients_by_stratum: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    donors_by_stratum: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    reasons_by_id: dict[str, list[str]] = {}
    donor_by_recipient: dict[str, str] = {}

    for row in normalized_recipients:
        reasons = _recipient_reason_codes(row)
        reasons_by_id[row["pair_id"]] = reasons
        if not reasons:
            recipients_by_stratum[_stratum(row)].append(row)
    for row in normalized_donors:
        donors_by_stratum[_stratum(row)].append(row)

    used_donor_ids: set[str] = set()
    for stratum, stratum_recipients in recipients_by_stratum.items():
        ordered_recipients = sorted(
            stratum_recipients, key=lambda row: _order_key(row, role="recipient")
        )
        ordered_donors = sorted(
            donors_by_stratum.get(stratum, []), key=lambda row: _order_key(row, role="donor")
        )
        for recipient, donor in zip(ordered_recipients, ordered_donors, strict=False):
            donor_by_recipient[recipient["pair_id"]] = donor["pair_id"]
            used_donor_ids.add(donor["pair_id"])
        for recipient in ordered_recipients[len(ordered_donors) :]:
            reasons_by_id[recipient["pair_id"]].append("donor_bank_insufficient")

    assignments: list[dict[str, object]] = []
    for row in normalized_recipients:
        pair_id = row["pair_id"]
        reasons = sorted(reasons_by_id[pair_id])
        donor_pair_id = donor_by_recipient.get(pair_id)
        assignments.append(
            {
                "pair_id": pair_id,
                "donor_pair_id": donor_pair_id,
                "eligibility": "valid" if donor_pair_id is not None else "invalid",
                "reason_codes": reasons,
            }
        )

    return {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "assignments": sorted(assignments, key=lambda row: row["pair_id"]),
        "unused_donor_pair_ids": sorted(donor_ids.difference(used_donor_ids)),
        "donor_bank_available": True,
    }
