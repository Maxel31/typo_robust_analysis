"""Pure coordinate planning and group-preserving shard assignment."""

from __future__ import annotations

import hashlib
from itertools import pairwise

from typo_cot.experiments.rebuttal_v2.schemas import (
    IntakeError,
    require_int,
    require_object,
    require_string,
)

SHARD_ASSIGNMENT_SCHEMA = "rebuttal-shard-assignment/v2"
_SHARD_DOMAIN = "rebuttal-shard/v2\n"
_OFFSET = 2


def _token_indices(value: object, context: str, length: int) -> list[int]:
    if not isinstance(value, list):
        raise IntakeError(f"{context} must be a JSON array")
    if not value:
        raise IntakeError(f"{context} must not be empty")
    result = [
        require_int(item, f"{context}[{index}]", minimum=0) for index, item in enumerate(value)
    ]
    if any(left >= right for left, right in pairwise(result)):
        raise IntakeError(f"{context} must contain strictly increasing token indices")
    if result[-1] >= length:
        raise IntakeError(f"{context} contains an index outside its prompt")
    return result


def _alignment_side(
    edit: dict[str, object],
    side: str,
    length: int,
    context: str,
) -> tuple[int, list[int]]:
    endpoint_key = f"{side}_endpoint"
    indices_key = f"{side}_token_indices"
    if endpoint_key not in edit or indices_key not in edit:
        missing = [key for key in (endpoint_key, indices_key) if key not in edit]
        raise IntakeError(f"{context} missing required fields: {', '.join(missing)}")
    endpoint = require_int(edit[endpoint_key], f"{context}.{endpoint_key}", minimum=0)
    indices = _token_indices(edit[indices_key], f"{context}.{indices_key}", length)
    if endpoint >= length:
        raise IntakeError(f"{context}.{endpoint_key} is outside its prompt")
    if endpoint != indices[-1]:
        raise IntakeError(f"{context}.{endpoint_key} must equal the final {indices_key} position")
    return endpoint, indices


def plan_offset(
    aligned_edits: list[dict],
    clean_length: int,
    typo_length: int,
) -> dict:
    """Plan the all-or-nothing +2 offset arm from audited word alignments.

    Structural defects raise :class:`IntakeError`. Scientific disqualifiers
    instead return an invalid plan while retaining the complete shifted arrays
    for audit diagnostics.
    """

    if not isinstance(aligned_edits, list):
        raise IntakeError("aligned_edits must be a JSON array")
    clean_prompt_length = require_int(clean_length, "clean_length", minimum=0)
    typo_prompt_length = require_int(typo_length, "typo_length", minimum=0)

    clean_endpoints: list[int] = []
    typo_endpoints: list[int] = []
    clean_edited_tokens: set[int] = set()
    typo_edited_tokens: set[int] = set()
    for index, raw_edit in enumerate(aligned_edits):
        context = f"aligned_edits[{index}]"
        edit = require_object(raw_edit, context)
        clean_endpoint, clean_indices = _alignment_side(edit, "clean", clean_prompt_length, context)
        typo_endpoint, typo_indices = _alignment_side(edit, "typo", typo_prompt_length, context)
        clean_endpoints.append(clean_endpoint)
        typo_endpoints.append(typo_endpoint)
        clean_edited_tokens.update(clean_indices)
        typo_edited_tokens.update(typo_indices)

    source_positions = [position + _OFFSET for position in clean_endpoints]
    write_positions = [position + _OFFSET for position in typo_endpoints]
    reasons: set[str] = set()
    if not aligned_edits:
        reasons.add("no_aligned_edits")
    if any(position <= 0 or position >= clean_prompt_length - 1 for position in source_positions):
        reasons.add("clean_offset_outside_prompt_interior")
    if any(position <= 0 or position >= typo_prompt_length - 1 for position in write_positions):
        reasons.add("typo_offset_outside_prompt_interior")
    if any(position in clean_edited_tokens for position in source_positions):
        reasons.add("clean_offset_overlaps_edited_token")
    if any(position in typo_edited_tokens for position in write_positions):
        reasons.add("typo_offset_overlaps_edited_token")
    if len(source_positions) != len(set(source_positions)):
        reasons.add("duplicate_source_positions")
    if len(write_positions) != len(set(write_positions)):
        reasons.add("duplicate_write_positions")

    return {
        "eligibility": "invalid" if reasons else "valid",
        "source_positions": source_positions,
        "write_positions": write_positions,
        "reason_codes": sorted(reasons),
    }


def _shard_index(group_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(bytes(f"{_SHARD_DOMAIN}{group_id}", encoding="utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def assign_shards(plan_rows: list[dict], num_shards: int) -> dict:
    """Assign complete original-problem groups to deterministic worker shards."""

    if not isinstance(plan_rows, list):
        raise IntakeError("plan_rows must be a JSON array")
    shard_count = require_int(num_shards, "num_shards", minimum=1)
    if shard_count > 6:
        raise IntakeError("num_shards must be at most 6")

    normalized: list[tuple[str, str, str | None]] = []
    seen_plan_ids: set[str] = set()
    missing_group_pair_ids: set[str] = set()
    pair_groups: dict[str, str] = {}
    for index, raw_row in enumerate(plan_rows):
        context = f"plan_rows[{index}]"
        row = require_object(raw_row, context)
        missing = [
            key for key in ("plan_id", "pair_id", "original_problem_group_id") if key not in row
        ]
        if missing:
            raise IntakeError(f"{context} missing required fields: {', '.join(missing)}")
        plan_id = require_string(row["plan_id"], f"{context}.plan_id")
        pair_id = require_string(row["pair_id"], f"{context}.pair_id")
        if plan_id in seen_plan_ids:
            raise IntakeError(f"duplicate plan_id: {plan_id}")
        seen_plan_ids.add(plan_id)

        raw_group_id = row["original_problem_group_id"]
        if raw_group_id is None:
            group_id = None
            missing_group_pair_ids.add(pair_id)
        else:
            group_id = require_string(raw_group_id, f"{context}.original_problem_group_id")
            previous = pair_groups.setdefault(pair_id, group_id)
            if previous != group_id:
                raise IntakeError(
                    f"pair_id {pair_id} has conflicting original_problem_group_id values"
                )
        normalized.append((plan_id, pair_id, group_id))

    if missing_group_pair_ids:
        pair_ids = ", ".join(sorted(missing_group_pair_ids))
        raise IntakeError(f"original_problem_group_id is null for pair_ids: {pair_ids}")

    grouped: dict[str, dict[str, set[str]]] = {}
    for plan_id, pair_id, group_id in normalized:
        assert group_id is not None
        member_ids = grouped.setdefault(group_id, {"pair_ids": set(), "plan_ids": set()})
        member_ids["pair_ids"].add(pair_id)
        member_ids["plan_ids"].add(plan_id)

    groups = [
        {
            "original_problem_group_id": group_id,
            "shard_index": _shard_index(group_id, shard_count),
            "pair_ids": sorted(member_ids["pair_ids"]),
            "plan_ids": sorted(member_ids["plan_ids"]),
        }
        for group_id, member_ids in sorted(grouped.items())
    ]
    return {
        "schema_version": SHARD_ASSIGNMENT_SCHEMA,
        "num_shards": shard_count,
        "groups": groups,
    }
