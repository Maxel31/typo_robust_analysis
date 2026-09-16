"""Outcome-independent, CPU-only intake of explicitly identified archive sources.

Only the two versioned JSONL adapters in this module normalize records. Inventory
of an unsupported or absent input is useful evidence, never fabricated pair data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .identity import (
    identity_hash,
    load_identity_aliases,
    original_problem_group_id_for,
    parse_identity_aliases,
    pair_id_for,
)
from .schemas import (
    IntakeError,
    canonical_json,
    canonical_sha256,
    load_protocol,
    require_enum,
    require_bool,
    require_int,
    require_keys,
    require_nullable_string,
    require_object,
    require_sha256,
    require_string,
    require_string_list,
    require_timestamp,
    sha256_file,
    sha256_text,
    strict_loads,
)


SOURCE_KINDS = {"submitted-archive", "public-regeneration", "fresh-audit"}
SOURCE_ROLES = {
    "pairs",
    "generations",
    "prompts",
    "cohort-ids",
    "runtime-metadata",
    "donor-bank",
    "rq2-prefixes",
}
AVAILABILITIES = {"available", "missing", "unknown", "not_applicable"}
PAIR_SCHEMA = "rebuttal-source-pair/v2"
GENERATION_SCHEMA = "rebuttal-source-generation/v2"
PAIR_REQUIRED = {"schema_version", "source_record_id", "source_pair_key", "task", "model_id"}
PAIR_OPTIONAL = {
    "historical_pair_id",
    "source_pair_key_kind",
    "cohort_ids",
    "dataset_id",
    "split",
    "original_problem_id",
    "dataset_revision",
    "target_rule",
    "perturbation_id",
    "model_revision",
    "tokenizer_revision",
    "archived_runtime",
    "clean_text",
    "typo_text",
    "clean_text_ref",
    "typo_text_ref",
    "clean_text_sha256",
    "typo_text_sha256",
    "clean_prompt",
    "typo_prompt",
    "clean_prompt_ref",
    "typo_prompt_ref",
    "clean_prompt_sha256",
    "typo_prompt_sha256",
    "prompt_template_revision",
    "clean_query_span",
    "typo_query_span",
    "prompt_regions",
    "gold",
    "canonical_gold",
    "gold_canonicalization_version",
    "valid_labels",
    "intended_edits",
    "archived_actual_edits",
    "archived_token_positions",
    "expected_generations",
    "archived_scoring",
}
GENERATION_REQUIRED = {
    "schema_version",
    "source_record_id",
    "source_generation_key",
    "source_pair_key",
    "arm",
    "window",
}
GENERATION_OPTIONAL = {
    "raw_text",
    "raw_text_ref",
    "raw_text_sha256",
    "generated_token_ids",
    "stopping_metadata",
    "termination",
    "archived_runtime",
    "archived_parser",
    "archived_scoring",
}
SOURCE_REQUIRED = {
    "source_id",
    "source_kind",
    "role",
    "path",
    "expected_sha256",
    "format",
    "adapter_version",
    "source_commit",
    "model_revision",
    "tokenizer_revision",
    "prompt_template_revision",
    "generation_config",
    "availability",
    "reason_codes",
}
COHORT_REQUIRED = {
    "cohort_id",
    "setting",
    "historical_n_reference",
    "expected_ids_ref",
    "id_namespace",
    "membership_provenance_ref",
}
_BLOCKS = {
    "pairs": ["answer-audit", "input-audit", "plan", "T1", "T2", "T3", "T4"],
    "generations": ["answer-audit", "T1"],
    "prompts": ["input-audit", "plan", "T2", "T4"],
    "cohort-ids": ["exact-cohort-coverage", "formal-cohort-inference"],
    "runtime-metadata": ["historical-runtime-replication"],
    "donor-bank": ["cross-donor-planning", "cross-control-estimate"],
    "rq2-prefixes": ["R5-prefix-control"],
}


@dataclass
class ArchiveIndex:
    """Validated index and exact input bytes used for this acquisition."""

    path: Path
    data: dict[str, Any]
    snapshots: dict[Path, str] = field(default_factory=dict)
    aliases: Any = None


@dataclass
class SourceAudit:
    index: ArchiveIndex
    inventory: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    source_records: list[tuple[dict[str, Any], dict[str, Any]]]
    cohorts: list[dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _snapshot(path: Path, snapshots: dict[Path, str], expected: str | None = None) -> bytes:
    path = path.resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IntakeError(f"Cannot read source {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise IntakeError(f"SHA256 mismatch for {path}: expected {expected}, observed {digest}")
    if path in snapshots and snapshots[path] != digest:
        raise IntakeError(f"Source changed during intake: {path}")
    snapshots[path] = digest
    return raw


def _decode(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntakeError(f"Source is not UTF-8: {path}") from exc


def _read_ref(ref: Any, containing_file: Path, snapshots: dict[Path, str]) -> tuple[Path, bytes]:
    obj = require_object(ref, "artifact reference")
    require_keys(obj, {"path", "sha256"}, context="artifact reference")
    raw_path = require_string(obj["path"], "artifact reference path")
    digest = require_sha256(obj["sha256"], "artifact reference sha256")
    path = (containing_file.parent / raw_path).resolve()
    return path, _snapshot(path, snapshots, digest)


def _absolute_ref(ref: Any, containing_file: Path, snapshots: dict[Path, str]) -> dict[str, str]:
    path, _ = _read_ref(ref, containing_file, snapshots)
    return {"path": str(path), "sha256": snapshots[path]}


def read_archive_index(path: str | Path) -> ArchiveIndex:
    """Read an owner-supplied inventory without selecting records by outcomes."""
    index_path = Path(path).resolve()
    snapshots: dict[Path, str] = {}
    data = require_object(
        strict_loads(_decode(_snapshot(index_path, snapshots), index_path)), "archive index"
    )
    require_keys(
        data,
        {
            "schema_version",
            "created_at",
            "source_namespace",
            "sources",
            "cohorts",
            "identity_aliases_ref",
            "notes",
        },
        context="archive index",
    )
    require_enum(
        data["schema_version"], {"rebuttal-archive-index/v2"}, "archive index schema_version"
    )
    require_timestamp(data["created_at"], "archive index created_at")
    require_string(data["source_namespace"], "archive index source_namespace")
    require_string(data["notes"], "archive index notes", allow_empty=True)
    for name in ("sources", "cohorts"):
        if not isinstance(data[name], list):
            raise IntakeError(f"archive index {name} must be an array")
    seen_sources: set[str] = set()
    for source in data["sources"]:
        require_object(source, "source")
        require_keys(source, SOURCE_REQUIRED, context="source")
        sid = require_string(source["source_id"], "source_id")
        if sid in seen_sources:
            raise IntakeError(f"Duplicate source_id: {sid}")
        seen_sources.add(sid)
        require_enum(source["source_kind"], SOURCE_KINDS, "source_kind")
        require_enum(source["role"], SOURCE_ROLES, "source role")
        for name in (
            "path",
            "format",
            "adapter_version",
            "source_commit",
            "model_revision",
            "tokenizer_revision",
            "prompt_template_revision",
        ):
            require_nullable_string(source[name], f"source {name}")
        if source["expected_sha256"] is not None:
            require_sha256(source["expected_sha256"], "source expected_sha256")
        if source["generation_config"] is not None:
            require_object(source["generation_config"], "source generation_config")
        require_enum(source["availability"], AVAILABILITIES, "source availability")
        require_string_list(source["reason_codes"], "source reason_codes", unique=True)
        if source["availability"] == "not_applicable" and (
            source["path"] is not None or source["expected_sha256"] is not None
        ):
            raise IntakeError(
                "Structurally unused not_applicable sources must have null path and expected_sha256"
            )
    seen_cohorts: set[str] = set()
    for cohort in data["cohorts"]:
        require_object(cohort, "cohort")
        require_keys(cohort, COHORT_REQUIRED, {"historical_counts"}, context="cohort")
        cid = require_string(cohort["cohort_id"], "cohort_id")
        if cid in seen_cohorts:
            raise IntakeError(f"Duplicate cohort_id: {cid}")
        seen_cohorts.add(cid)
        require_string(cohort["setting"], "cohort setting")
        require_string(cohort["id_namespace"], "cohort id_namespace")
        require_int(cohort["historical_n_reference"], "historical_n_reference", minimum=0)
        if "historical_counts" in cohort:
            require_object(cohort["historical_counts"], "historical_counts")
    aliases = load_identity_aliases(None)
    if data["identity_aliases_ref"] is not None:
        aliases_path, alias_bytes = _read_ref(data["identity_aliases_ref"], index_path, snapshots)
        alias_data = require_object(
            strict_loads(_decode(alias_bytes, aliases_path)),
            "identity aliases",
        )
        for rows in (alias_data.get("pair_aliases", []), alias_data.get("group_aliases", [])):
            if not isinstance(rows, list):
                raise IntakeError("Identity alias collections must be arrays")
            for row in rows:
                if isinstance(row, dict) and "evidence_ref" in row:
                    _read_ref(row["evidence_ref"], aliases_path, snapshots)
        aliases = parse_identity_aliases(alias_data, source_path=aliases_path)
    return ArchiveIndex(index_path, data, snapshots, aliases)


def _finding(
    source_id: str | None, component: str, reason: str, detail: str, blocks: list[str]
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "component": component,
        "reason_codes": [reason],
        "explanation": detail,
        "blocks": blocks,
    }


def audit_sources(index: ArchiveIndex) -> SourceAudit:
    """Inventory every requested source, including missing/unsupported layouts."""
    inventory: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    source_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for source in index.data["sources"]:
        if source["availability"] == "not_applicable":
            inventory.append(
                {
                    "schema_version": "rebuttal-archive-inventory/v2",
                    **source,
                    "requested_availability": "not_applicable",
                    "observed_sha256": None,
                    "historical_hash_status": "not_applicable",
                    "adapter_status": "not_applicable",
                    "record_count": None,
                }
            )
            continue
        path = (
            (index.path.parent / source["path"]).resolve() if source["path"] is not None else None
        )
        exists = path is not None and path.is_file()
        availability = "available" if exists else ("missing" if path is not None else "unknown")
        observed = None
        raw = None
        reasons = list(source["reason_codes"])
        if exists:
            raw = _snapshot(path, index.snapshots, source["expected_sha256"])
            observed = index.snapshots[path]
        else:
            reason = "source_missing" if path is not None else "source_location_unknown"
            reasons.append(reason)
            findings.append(
                _finding(
                    source["source_id"],
                    source["role"],
                    reason,
                    f"No source bytes recovered for {source['source_id']}.",
                    _BLOCKS[source["role"]],
                )
            )
        if source["expected_sha256"] is None:
            reasons.append("historical_hash_unknown")
        for name in (
            "source_commit",
            "model_revision",
            "tokenizer_revision",
            "prompt_template_revision",
            "generation_config",
        ):
            if source[name] is None:
                reasons.append(f"archived_{name}_unknown")
        adapter = {"pairs": PAIR_SCHEMA, "generations": GENERATION_SCHEMA}.get(source["role"])
        supported = (
            adapter is not None
            and source["format"] == "jsonl"
            and source["adapter_version"] == adapter
        )
        adapter_status = "supported" if supported else "unsupported"
        if not supported:
            reasons.append("unsupported_adapter")
            findings.append(
                _finding(
                    source["source_id"],
                    source["role"],
                    "unsupported_adapter",
                    f"No normalization adapter for role={source['role']}, format={source['format']}, adapter={source['adapter_version']}; bytes inventoried only.",
                    _BLOCKS[source["role"]],
                )
            )
        entry = {
            "schema_version": "rebuttal-archive-inventory/v2",
            **source,
            "requested_availability": source["availability"],
            "path": str(path) if path else None,
            "availability": availability,
            "observed_sha256": observed,
            "historical_hash_status": "verified"
            if source["expected_sha256"] is not None and exists
            else "unknown",
            "adapter_status": adapter_status,
            "reason_codes": sorted(set(reasons)),
            "record_count": None,
        }
        if supported and raw is not None:
            count = 0
            seen_records: set[str] = set()
            for lineno, line in enumerate(_decode(raw, path).split("\n"), start=1):
                if not line.strip():
                    continue
                record = require_object(
                    strict_loads(line, context=f"{path}:{lineno}"), "source record"
                )
                rid = require_string(record.get("source_record_id"), "source_record_id")
                if rid in seen_records:
                    raise IntakeError(f"Duplicate source_record_id {rid!r} in {path}")
                seen_records.add(rid)
                source_records.append((entry, record))
                count += 1
            entry["record_count"] = count
        inventory.append(entry)
    cohorts: list[dict[str, Any]] = []
    for cohort in index.data["cohorts"]:
        expected_keys = None
        expected_ref = None
        if cohort["expected_ids_ref"] is not None:
            ids_path, raw = _read_ref(cohort["expected_ids_ref"], index.path, index.snapshots)
            ids = require_object(strict_loads(_decode(raw, ids_path)), "cohort IDs")
            require_keys(
                ids,
                {"schema_version", "cohort_id", "source_namespace", "source_pair_keys"},
                context="cohort IDs",
            )
            require_enum(
                ids["schema_version"], {"rebuttal-cohort-ids/v2"}, "cohort IDs schema_version"
            )
            if (
                ids["cohort_id"] != cohort["cohort_id"]
                or ids["source_namespace"] != cohort["id_namespace"]
            ):
                raise IntakeError("Cohort ID artifact namespace/cohort_id does not match index")
            expected_keys = require_string_list(
                ids["source_pair_keys"], "cohort source_pair_keys", unique=True
            )
            expected_ref = {"path": str(ids_path), "sha256": index.snapshots[ids_path]}
        else:
            findings.append(
                _finding(
                    None,
                    cohort["cohort_id"],
                    "expected_ids_unknown",
                    "No frozen source ID enumeration; count agreement cannot establish cohort coverage.",
                    _BLOCKS["cohort-ids"],
                )
            )
        membership_ref = None
        if cohort["membership_provenance_ref"] is not None:
            membership_ref = _absolute_ref(
                cohort["membership_provenance_ref"], index.path, index.snapshots
            )
        cohorts.append(
            {
                **cohort,
                "expected_ids_ref": expected_ref,
                "membership_provenance_ref": membership_ref,
                "expected_source_pair_keys": expected_keys,
            }
        )
    return SourceAudit(index, inventory, findings, source_records, cohorts)


def _window(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise IntakeError("window must be null or [start,end]")
    start, end = (
        require_int(value[0], "window start", minimum=0),
        require_int(value[1], "window end", minimum=0),
    )
    if end <= start:
        raise IntakeError("window end must exceed start")
    return [start, end]


def _slot_fields(arm: Any, window: Any) -> tuple[str, list[int] | None]:
    arm = require_enum(arm, {"clean", "typo", "self", "correct", "offset", "cross"}, "arm")
    window = _window(window)
    if arm in {"clean", "typo"} and window is not None:
        raise IntakeError("Baseline clean/typo generation window must be null")
    if arm not in {"clean", "typo"} and window is None:
        raise IntakeError("Patch generation window must be an explicit [start,end]")
    return arm, window


def _text_payload(
    record: dict[str, Any], name: str, context: dict[str, Any]
) -> tuple[str | None, dict[str, str] | None, str | None]:
    inline = record.get(name)
    ref = record.get(f"{name}_ref")
    if inline is not None and ref is not None:
        raise IntakeError(f"{name} must use exactly one of inline text or artifact reference")
    if inline is not None:
        require_string(inline, name, allow_empty=True)
        text_hash = sha256_text(inline)
    elif ref is not None:
        path, raw = _read_ref(ref, context["source_path"], context["snapshots"])
        text_hash = sha256_text(_decode(raw, path))
        ref = {"path": str(path), "sha256": context["snapshots"][path]}
    else:
        text_hash = None
    supplied = record.get(f"{name}_sha256")
    if supplied is not None:
        require_sha256(supplied, f"{name}_sha256")
        if text_hash != supplied:
            raise IntakeError(f"{name} text hash mismatch or hash supplied without text")
    return inline, ref, text_hash


def _validate_pair_metadata(record: dict[str, Any]) -> None:
    for name in ("source_record_id", "source_pair_key", "task", "model_id"):
        require_string(record[name], name)
    for name in (
        "historical_pair_id",
        "dataset_id",
        "split",
        "original_problem_id",
        "dataset_revision",
        "target_rule",
        "perturbation_id",
        "model_revision",
        "tokenizer_revision",
        "prompt_template_revision",
        "gold_canonicalization_version",
    ):
        require_nullable_string(record.get(name), name)
    require_string_list(record.get("cohort_ids", []), "cohort_ids", unique=True)
    if record.get("valid_labels") is not None:
        require_string_list(record["valid_labels"], "valid_labels", unique=True)
    for name in ("archived_runtime", "archived_scoring"):
        if record.get(name) is not None:
            require_object(record[name], name)
    for name in ("intended_edits", "archived_actual_edits", "archived_token_positions"):
        if record.get(name) is not None and not isinstance(record[name], list):
            raise IntakeError(f"{name} must be an array or null")
    for name in ("clean_query_span", "typo_query_span"):
        if record.get(name) is not None:
            span = record[name]
            if not isinstance(span, list) or len(span) != 2:
                raise IntakeError(f"{name} must be [start,end] or null")
            start = require_int(span[0], name, minimum=0)
            end = require_int(span[1], name, minimum=0)
            if end < start:
                raise IntakeError(f"{name} end precedes start")
    if record.get("prompt_regions") is not None:
        require_object(record["prompt_regions"], "prompt_regions")
    canonical_gold = record.get("canonical_gold")
    if canonical_gold is not None:
        require_string(record.get("gold_canonicalization_version"), "gold_canonicalization_version")
        if record["task"] == "gsm8k":
            require_object(canonical_gold, "canonical_gold")
            require_keys(canonical_gold, {"numerator", "denominator"}, context="canonical_gold")
            numerator = require_string(canonical_gold["numerator"], "gold numerator")
            denominator = require_string(canonical_gold["denominator"], "gold denominator")
            if not re.fullmatch(r"(?:0|-?[1-9][0-9]*)", numerator) or not re.fullmatch(
                r"[1-9][0-9]*", denominator
            ):
                raise IntakeError(
                    "Canonical numeric gold requires normalized integer strings and a positive denominator"
                )
            try:
                reduced = math.gcd(int(numerator), int(denominator)) == 1
            except ValueError as exc:
                raise IntakeError("Canonical numeric gold integer is too large") from exc
            if not reduced:
                raise IntakeError("Canonical numeric gold must be a reduced rational")
        elif record["task"] == "mmlu-pro":
            label = require_string(canonical_gold, "canonical_gold")
            if record.get("valid_labels") is not None and label not in record["valid_labels"]:
                raise IntakeError("Canonical choice gold is not an item-valid label")
        else:
            raise IntakeError("No canonical gold schema for this task")


def _resolved_text(row: dict[str, Any], name: str, context: dict[str, Any]) -> str | None:
    if row[name] is not None:
        return row[name]
    if row[f"{name}_ref"] is not None:
        path, raw = _read_ref(row[f"{name}_ref"], context["source_path"], context["snapshots"])
        return _decode(raw, path)
    return None


def normalize_record(record: dict[str, Any], source_identity: dict[str, Any]) -> dict[str, Any]:
    """Normalize an explicit source-pair/v2 record, retaining archived unknowns.

    ``source_identity`` supplies acquisition namespace and source provenance;
    run_intake supplies its path/snapshot/alias context for referenced texts.
    """
    require_object(record, "source pair")
    require_keys(record, PAIR_REQUIRED, PAIR_OPTIONAL, context="source pair")
    require_enum(record["schema_version"], {PAIR_SCHEMA}, "source pair schema_version")
    _validate_pair_metadata(record)
    namespace = require_string(source_identity["source_namespace"], "source_namespace")
    aliases = source_identity.get("aliases") or load_identity_aliases(None)
    pair_alias = aliases.resolve_pair(namespace, record["source_pair_key"])
    group_alias = aliases.resolve_group(
        record.get("dataset_id"), record.get("split"), record.get("original_problem_id")
    )
    canonical_pair = pair_alias.canonical
    canonical_group = group_alias.canonical
    group_id = original_problem_group_id_for(
        canonical_group["dataset_id"],
        canonical_group["split"],
        canonical_group["original_problem_id"],
    )
    result: dict[str, Any] = {
        "schema_version": "rebuttal-pair-manifest/v2",
        "pair_id": pair_id_for(
            canonical_pair["source_namespace"], canonical_pair["source_pair_key"]
        ),
        **canonical_pair,
        "pair_identity": {
            "domain": "rebuttal-pair/v2",
            "payload": canonical_pair,
            "original_payload": pair_alias.original,
            "alias_evidence_refs": list(pair_alias.evidence_refs),
        },
        "original_problem_group_id": group_id,
        "group_identity": {
            "domain": "rebuttal-original-problem/v2",
            "payload": canonical_group,
            "original_payload": group_alias.original,
            "alias_evidence_refs": list(group_alias.evidence_refs),
        },
        "source_pair_key_kind": record.get(
            "source_pair_key_kind",
            "historical-id" if record.get("historical_pair_id") is not None else "producer-id",
        ),
        "source_kind": source_identity.get("source_kind"),
        "source_ref": source_identity.get("source_ref"),
        "additional_source_refs": [],
        "source_sha256": source_identity.get("source_sha256"),
        "source_record_id": record["source_record_id"],
        "task": record["task"],
        "model_id": record["model_id"],
        "cohort_ids": list(record.get("cohort_ids", [])),
        "archived_generations": [],
        "availability": {},
        "reason_codes": [],
    }
    require_enum(
        result["source_pair_key_kind"], {"historical-id", "producer-id"}, "source_pair_key_kind"
    )
    nullable_fields = {
        "historical_pair_id",
        "dataset_revision",
        "target_rule",
        "perturbation_id",
        "clean_query_span",
        "typo_query_span",
        "prompt_regions",
        "gold",
        "canonical_gold",
        "gold_canonicalization_version",
        "valid_labels",
        "intended_edits",
        "archived_actual_edits",
        "archived_token_positions",
        "archived_runtime",
        "archived_scoring",
    }
    for name in nullable_fields:
        result[name] = record.get(name)
    result.update(canonical_group)
    for name in ("model_revision", "tokenizer_revision", "prompt_template_revision"):
        result[name] = record.get(name, source_identity.get(name))
    if "archived_runtime" not in record:
        result["archived_runtime"] = source_identity.get("generation_config")
    for name in ("clean_text", "typo_text", "clean_prompt", "typo_prompt"):
        text, ref, digest = _text_payload(record, name, source_identity)
        result[name], result[f"{name}_ref"], result[f"{name}_sha256"] = text, ref, digest
        result["availability"][name] = "available" if digest is not None else "unknown"
        if digest is None:
            result["reason_codes"].append(f"{name}_unknown")
    for side in ("clean", "typo"):
        span = result[f"{side}_query_span"]
        if span is None:
            continue
        prompt = _resolved_text(result, f"{side}_prompt", source_identity)
        query = _resolved_text(result, f"{side}_text", source_identity)
        if prompt is None or query is None:
            result["reason_codes"].append(f"{side}_query_span_unverifiable")
        elif span[1] > len(prompt) or prompt[span[0] : span[1]] != query:
            raise IntakeError(
                f"{side}_query_span does not identify the exact query in the full prompt"
            )
    for name in sorted(
        nullable_fields
        | {
            "model_revision",
            "tokenizer_revision",
            "prompt_template_revision",
            "dataset_id",
            "split",
            "original_problem_id",
        }
    ):
        if result[name] is None:
            result["reason_codes"].append(f"{name}_unknown")
    if group_id is None:
        result["reason_codes"].append("original_problem_identity_unknown")
    expected = record.get("expected_generations")
    if expected is not None and not isinstance(expected, list):
        raise IntakeError("expected_generations must be an array or null")
    slots: set[str] = set()
    for entry in expected or []:
        require_object(entry, "expected generation")
        require_keys(
            entry, {"arm", "window"}, {"source_generation_key"}, context="expected generation"
        )
        _slot_fields(entry["arm"], entry["window"])
        if entry.get("source_generation_key") is not None:
            require_string(entry["source_generation_key"], "expected source_generation_key")
        slot = canonical_json({"arm": entry["arm"], "window": entry["window"]})
        if slot in slots:
            raise IntakeError("Duplicate expected arm/window")
        slots.add(slot)
    result["expected_generations"] = expected
    result["source_identity_provenance"] = [
        {
            "source_ref": result["source_ref"],
            "cohort_ids": sorted(result["cohort_ids"]),
            "pair_identity": result["pair_identity"],
            "group_identity": result["group_identity"],
            "historical_pair_id": result["historical_pair_id"],
            "source_pair_key_kind": result["source_pair_key_kind"],
            "source_pair_key_kind_explicit": "source_pair_key_kind" in record,
        }
    ]
    return result


def _source_context(
    audit: SourceAudit, entry: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    return {
        **entry,
        "source_namespace": audit.index.data["source_namespace"],
        "source_path": Path(entry["path"]),
        "snapshots": audit.index.snapshots,
        "aliases": audit.index.aliases,
        "source_sha256": entry["observed_sha256"],
        "source_ref": {
            "artifact": {"path": entry["path"], "sha256": entry["observed_sha256"]},
            "record_id": record["source_record_id"],
        },
    }


def _normalize_generation(
    record: dict[str, Any], context: dict[str, Any], pair_id: str
) -> dict[str, Any]:
    require_keys(record, GENERATION_REQUIRED, GENERATION_OPTIONAL, context="source generation")
    require_enum(record["schema_version"], {GENERATION_SCHEMA}, "source generation schema_version")
    for name in ("source_record_id", "source_generation_key", "source_pair_key", "arm"):
        require_string(record[name], name)
    _, window = _slot_fields(record["arm"], record["window"])
    text, ref, text_hash = _text_payload(record, "raw_text", context)
    tokens = record.get("generated_token_ids")
    if tokens is not None:
        if not isinstance(tokens, list):
            raise IntakeError("generated_token_ids must be an array or null")
        for token in tokens:
            require_int(token, "generated token ID", minimum=0)
    for name in ("stopping_metadata", "archived_runtime", "archived_parser", "archived_scoring"):
        if record.get(name) is not None:
            require_object(record[name], name)
    if record.get("archived_scoring") is not None:
        scoring = record["archived_scoring"]
        require_keys(
            scoring,
            set(),
            {"parser_id", "parser_version", "gold_match"},
            context="archived_scoring",
        )
        for name in ("parser_id", "parser_version"):
            require_nullable_string(scoring.get(name), f"archived_scoring {name}")
        if scoring.get("gold_match") is not None:
            require_bool(scoring["gold_match"], "archived_scoring gold_match")
    supplied_termination = (
        require_nullable_string(record.get("termination"), "termination") or "unknown"
    )
    # Only adapter-defined observed flags substantiate a stopping classification;
    # arbitrary historical metadata and a bare declaration remain provenance.
    stopping = record.get("stopping_metadata") or {}
    for name in ("eos_observed", "length_cap_reached"):
        if name in stopping:
            require_bool(stopping[name], f"stopping_metadata {name}")
    if stopping.get("eos_observed") is True:
        termination = "eos"
    elif stopping.get("eos_observed") is False and stopping.get("length_cap_reached") is True:
        termination = "length-cap"
    else:
        termination = "unknown"
    if (
        (supplied_termination == "eos" and stopping.get("eos_observed") is False)
        or (supplied_termination == "length-cap" and stopping.get("length_cap_reached") is False)
        or (
            supplied_termination in {"eos", "length-cap"}
            and termination != "unknown"
            and supplied_termination != termination
        )
    ):
        raise IntakeError("Archived termination conflicts with observed stopping metadata")
    payload = {
        "source_namespace": context["source_namespace"],
        "source_generation_key": record["source_generation_key"],
    }
    reasons = []
    if supplied_termination not in {"eos", "length-cap", "unknown"}:
        reasons.append("archived_termination_unsupported")
    if text_hash is None:
        reasons.append("raw_generation_missing")
    if termination == "unknown":
        reasons.append("termination_unknown")
    return {
        "schema_version": "rebuttal-archive-generation/v2",
        "generation_id": identity_hash("rebuttal-generation/v2", payload),
        "generation_identity": {"domain": "rebuttal-generation/v2", "payload": payload},
        "pair_id": pair_id,
        "source_ref": context["source_ref"],
        "source_kind": context["source_kind"],
        "source_record_id": record["source_record_id"],
        "source_generation_key": record["source_generation_key"],
        "arm": record["arm"],
        "window": window,
        "raw_text": text,
        "raw_text_ref": ref,
        "raw_text_sha256": text_hash,
        "generated_token_ids": tokens,
        "stopping_metadata": record.get("stopping_metadata"),
        "termination": termination,
        "archived_termination": record.get("termination"),
        "archived_runtime": record.get("archived_runtime", context.get("generation_config")),
        "archived_parser": record.get("archived_parser"),
        "archived_scoring": record.get("archived_scoring"),
        "availability": "available" if text_hash is not None else "missing",
        "reason_codes": reasons,
    }


def _merge_pair_identity_provenance(old: dict[str, Any], pair: dict[str, Any]) -> None:
    """Merge only acquisition identity metadata, preserving every declared claim.

    Scientific missingness is not repaired here. Distinct original identities
    may share a normalized pair only through the already-verified alias map.
    """
    provenance = sorted(
        old["source_identity_provenance"] + pair["source_identity_provenance"],
        key=canonical_json,
    )
    by_original: dict[str, list[dict[str, Any]]] = {}
    for item in provenance:
        original = item["pair_identity"]["original_payload"]
        canonical = item["pair_identity"]["payload"]
        if canonical != old["pair_identity"]["payload"]:
            raise IntakeError("Conflicting canonical pair identities in acquisition provenance")
        if original != canonical and not item["pair_identity"]["alias_evidence_refs"]:
            raise IntakeError("Distinct original pair identities require explicit alias evidence")
        by_original.setdefault(canonical_json(original), []).append(item)
    resolved: dict[str, tuple[str | None, str]] = {}
    for original, items in by_original.items():
        known_ids = {
            item["historical_pair_id"] for item in items if item["historical_pair_id"] is not None
        }
        if len(known_ids) > 1:
            raise IntakeError(
                "Conflicting historical_pair_id values for one original pair identity"
            )
        historical_id = next(iter(known_ids), None)
        explicit_kinds = {
            item["source_pair_key_kind"] for item in items if item["source_pair_key_kind_explicit"]
        }
        if len(explicit_kinds) > 1:
            raise IntakeError(
                "Conflicting explicit source_pair_key_kind values for one original pair identity"
            )
        kind = next(
            iter(explicit_kinds), "historical-id" if historical_id is not None else "producer-id"
        )
        resolved[original] = historical_id, kind

    canonical_original = canonical_json(old["pair_identity"]["payload"])
    selected_original = min(by_original, key=lambda key: (key != canonical_original, key))
    # Prefer an acquisition actually carrying the known historical value, then
    # canonical group identity, with deterministic provenance-only tie breaks.
    representative = min(
        by_original[selected_original],
        key=lambda item: (
            item["historical_pair_id"] is None,
            item["group_identity"]["original_payload"] != item["group_identity"]["payload"],
            canonical_json(item),
        ),
    )
    old["historical_pair_id"], old["source_pair_key_kind"] = resolved[selected_original]
    old["source_identity_provenance"] = provenance
    old["pair_identity"] = representative["pair_identity"]
    old["group_identity"] = representative["group_identity"]
    old["source_ref"] = representative["source_ref"]
    old["source_sha256"] = representative["source_ref"]["artifact"]["sha256"]
    old["source_record_id"] = representative["source_ref"]["record_id"]
    representative_ref = canonical_json(representative["source_ref"])
    unique_refs = {canonical_json(item["source_ref"]): item["source_ref"] for item in provenance}
    old["additional_source_refs"] = [
        unique_refs[key] for key in sorted(unique_refs) if key != representative_ref
    ]
    reasons = set(old["reason_codes"]) | set(pair["reason_codes"])
    reasons.discard("historical_pair_id_unknown")
    if old["historical_pair_id"] is None:
        reasons.add("historical_pair_id_unknown")
    old["reason_codes"] = sorted(reasons)


def _merge_pair_text_representations(old: dict[str, Any], pair: dict[str, Any]) -> None:
    """Select exact-text storage independently of acquisition iteration order.

    Call only after equal text hashes have been established. The minimum rule
    is associative: inline wins, otherwise canonical reference ordering wins.
    Inline and reference fields move together, preserving their exclusivity.
    """
    for name in ("clean_text", "typo_text", "clean_prompt", "typo_prompt"):
        choices = [(row[name], row[f"{name}_ref"]) for row in (old, pair)]
        inline, reference = min(
            choices, key=lambda item: (item[0] is None, canonical_json(item[1]))
        )
        old[name], old[f"{name}_ref"] = inline, reference


def _join_records(audit: SourceAudit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: dict[str, dict[str, Any]] = {}
    for entry, raw in audit.source_records:
        if entry["role"] != "pairs":
            continue
        pair = normalize_record(raw, _source_context(audit, entry, raw))
        pid = pair["pair_id"]
        if pid in pairs:
            old = pairs[pid]
            provenance_only = {
                "source_ref",
                "source_sha256",
                "source_record_id",
                "cohort_ids",
                "pair_identity",
                "group_identity",
                "source_pair_key_kind",
                "historical_pair_id",
                "additional_source_refs",
                "source_identity_provenance",
                "reason_codes",
                "availability",
            }
            for name in ("clean_text", "typo_text", "clean_prompt", "typo_prompt"):
                provenance_only.update({name, f"{name}_ref"})
            consistency = set(old) | set(pair)
            conflicts = sorted(
                key for key in consistency - provenance_only if old.get(key) != pair.get(key)
            )
            if conflicts:
                raise IntakeError(
                    f"Conflicting exact texts or identity metadata for pair_id {pid}; fields: {', '.join(conflicts)}"
                )
            # The manifest has unique IDs, but repeated acquisition records retain
            # all source provenance and are explicitly reported as duplicates.
            _merge_pair_identity_provenance(old, pair)
            _merge_pair_text_representations(old, pair)
            old["cohort_ids"] = sorted(set(old["cohort_ids"]) | set(pair["cohort_ids"]))
            audit.findings.append(
                _finding(
                    entry["source_id"],
                    pid,
                    "duplicate_pair_acquisition",
                    "Repeated consistent acquisition identity was deduplicated; source references retained.",
                    [],
                )
            )
        else:
            pairs[pid] = pair
    generations: list[dict[str, Any]] = []
    gen_ids: set[str] = set()
    slots: set[str] = set()
    for entry, raw in audit.source_records:
        if entry["role"] != "generations":
            continue
        context = _source_context(audit, entry, raw)
        source_pair_key = require_string(raw.get("source_pair_key"), "generation source_pair_key")
        alias = audit.index.aliases.resolve_pair(context["source_namespace"], source_pair_key)
        pid = pair_id_for(alias.canonical["source_namespace"], alias.canonical["source_pair_key"])
        generation = _normalize_generation(raw, context, pid)
        gid = generation["generation_id"]
        if gid in gen_ids:
            raise IntakeError(f"Duplicate generation_id {gid}")
        gen_ids.add(gid)
        slot = canonical_json(
            {"pair_id": pid, "arm": generation["arm"], "window": generation["window"]}
        )
        if slot in slots:
            raise IntakeError(f"Duplicate generation pair/arm/window slot: {slot}")
        slots.add(slot)
        if pid not in pairs:
            audit.findings.append(
                _finding(
                    entry["source_id"],
                    gid,
                    "generation_pair_missing",
                    "Raw generation retained but its pair record was not recovered.",
                    ["answer-audit", "input-audit"],
                )
            )
        elif generation["source_kind"] != pairs[pid]["source_kind"]:
            raise IntakeError(
                "Generation source_kind differs from its pair acquisition source_kind"
            )
        generations.append(generation)
    return sorted(pairs.values(), key=lambda row: row["pair_id"]), sorted(
        generations, key=lambda row: row["generation_id"]
    )


def _cohort_coverage(
    audit: SourceAudit, pairs: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    cohorts: list[dict[str, Any]] = []
    settings = {setting["id"]: setting for setting in protocol["settings"]}
    declared = {cohort["cohort_id"] for cohort in audit.cohorts}
    for pair in pairs:
        unknown = set(pair["cohort_ids"]) - declared
        if unknown:
            raise IntakeError(f"Pair references undeclared cohorts: {sorted(unknown)}")
        pair["cohort_membership"] = []
    for cohort in audit.cohorts:
        setting = settings.get(cohort["setting"])
        if (
            setting is not None
            and cohort["historical_n_reference"] != setting["historical_n_reference"]
        ):
            raise IntakeError(
                f"historical_n_reference does not match frozen setting {cohort['setting']}"
            )
        expected = cohort["expected_source_pair_keys"]
        expected_ids: dict[str, str] | None = None
        if expected is not None:
            expected_ids = {}
            for key in expected:
                alias = audit.index.aliases.resolve_pair(cohort["id_namespace"], key)
                pid = pair_id_for(
                    alias.canonical["source_namespace"], alias.canonical["source_pair_key"]
                )
                if pid in expected_ids:
                    raise IntakeError(
                        "Cohort expected IDs collapse onto a duplicate canonical pair identity"
                    )
                expected_ids[pid] = key
        available = {pair["pair_id"] for pair in pairs}
        claimed = {pair["pair_id"] for pair in pairs if cohort["cohort_id"] in pair["cohort_ids"]}
        recovered = available & set(expected_ids) if expected_ids is not None else claimed
        missing = set(expected_ids) - available if expected_ids is not None else None
        extras = claimed - set(expected_ids) if expected_ids is not None else set()
        coverage = (
            "unknown"
            if expected_ids is None
            else ("complete" if not missing and not extras else "partial")
        )
        claim_setting_mismatch = False
        for pair in pairs:
            if pair["pair_id"] not in recovered | extras:
                continue
            if (
                pair["pair_id"] in recovered
                and setting is not None
                and (pair["task"] != setting["task"] or pair["model_id"] != setting["model"])
            ):
                if expected_ids is not None:
                    raise IntakeError(
                        f"Pair task/model does not match frozen setting {cohort['setting']}"
                    )
                claim_setting_mismatch = True
                reason = "claimed_cohort_setting_mismatch"
                if reason not in pair["reason_codes"]:
                    pair["reason_codes"].append(reason)
                audit.findings.append(
                    _finding(
                        None,
                        pair["pair_id"],
                        reason,
                        f"Unverified membership claim for cohort {cohort['cohort_id']} has a task/model inconsistent with setting {cohort['setting']}; no frozen ID list establishes membership.",
                        ["exact-cohort-recovery", "formal-setting-inference"],
                    )
                )
            membership = (
                "unknown"
                if expected_ids is None
                else ("member" if pair["pair_id"] in recovered else "extra")
            )
            membership_source = cohort["expected_ids_ref"] or cohort["membership_provenance_ref"]
            if membership_source is None:
                claimant_refs = [
                    item["source_ref"]["artifact"]
                    for item in pair["source_identity_provenance"]
                    if cohort["cohort_id"] in item["cohort_ids"]
                ]
                if not claimant_refs:
                    raise IntakeError(
                        "Cohort membership claim has no supporting source acquisition"
                    )
                membership_source = min(claimant_refs, key=canonical_json)
            pair["cohort_membership"].append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "membership": membership,
                    "source_ref": membership_source,
                }
            )
        result = {
            **cohort,
            "coverage_status": coverage,
            "recovered_count": len(recovered),
            "expected_count": len(expected_ids) if expected_ids is not None else None,
            "missing_count": len(missing) if missing is not None else None,
            "extra_count": len(extras),
            "missing_source_pair_keys": sorted(expected_ids[pid] for pid in missing)
            if missing is not None
            else None,
            "extra_pair_ids": sorted(extras),
            "recovered_pair_ids": sorted(recovered),
            "historical_count_shortfall": max(0, cohort["historical_n_reference"] - len(recovered)),
            "reason_codes": ["claimed_cohort_setting_mismatch"] if claim_setting_mismatch else [],
        }
        if coverage == "unknown":
            result["reason_codes"].append("expected_ids_unknown")
        if cohort["setting"] not in settings:
            result["reason_codes"].append("setting_unvalidated")
            audit.findings.append(
                _finding(
                    None,
                    cohort["cohort_id"],
                    "setting_unvalidated",
                    "This historical setting is inventoried but has no task/model contract in the selected frozen protocol.",
                    ["formal-setting-inference"],
                )
            )
        if missing:
            result["reason_codes"].append("expected_pairs_missing")
            for pid in sorted(missing):
                audit.findings.append(
                    _finding(
                        None,
                        expected_ids[pid],
                        "expected_pair_missing",
                        f"Archived cohort {cohort['cohort_id']} explicitly expects this source_pair_key; it was not recovered or replaced.",
                        ["exact-cohort-coverage", "formal-cohort-inference"],
                    )
                )
        if extras:
            result["reason_codes"].append("extra_pairs_excluded")
            for pid in sorted(extras):
                audit.findings.append(
                    _finding(
                        None,
                        pid,
                        "extra_pair_excluded",
                        f"Claimed membership in {cohort['cohort_id']} is not in the frozen expected-ID list.",
                        ["archived-cohort-membership"],
                    )
                )
        cohorts.append(result)
    return cohorts


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _observed_scoring_counts(
    cohort: dict[str, Any], generations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Describe saved score metadata, never rerun a parser or infer an output."""
    members = set(cohort["recovered_pair_ids"])
    groups: dict[str, dict[str, Any]] = {}
    for generation in generations:
        if generation["pair_id"] not in members or generation["availability"] != "available":
            continue
        scoring = generation["archived_scoring"] or {}
        group = {
            "arm": generation["arm"],
            "window": generation["window"],
            "parser_id": scoring.get("parser_id"),
            "parser_version": scoring.get("parser_version"),
            "source_kind": generation["source_kind"],
        }
        key = canonical_json(group)
        if key not in groups:
            groups[key] = {
                **group,
                "observed_n": 0,
                "scored_n": 0,
                "correct_n": 0,
                "incorrect_n": 0,
                "unscored_n": 0,
                "generation_ids": [],
            }
        counts = groups[key]
        counts["observed_n"] += 1
        counts["generation_ids"].append(generation["generation_id"])
        match = scoring.get("gold_match")
        if match is None:
            counts["unscored_n"] += 1
        else:
            counts["scored_n"] += 1
            counts["correct_n" if match else "incorrect_n"] += 1
    return [groups[key] for key in sorted(groups)]


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _bytes_ref(name: str, raw: bytes) -> dict[str, str]:
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def _code_identity() -> dict[str, Any]:
    code_refs = []
    for name in ("source.py", "identity.py", "schemas.py"):
        code_path = Path(__file__).with_name(name).resolve()
        code_refs.append(
            {"path": str(code_path), "sha256": hashlib.sha256(code_path.read_bytes()).hexdigest()}
        )
    commit = None
    dirty = None
    reasons = []
    try:
        base = Path(__file__).resolve().parent
        result = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        candidate = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", candidate):
            commit = candidate
        else:
            reasons.append("code_commit_unknown")
        status = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        dirty = bool(status.stdout)
    except (OSError, subprocess.SubprocessError):
        reasons.append("code_git_identity_unavailable")
    return {
        "source_refs": code_refs,
        "git_commit": commit,
        "git_dirty": dirty,
        "reason_codes": reasons,
    }


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        cells = {}
        for name in fields:
            value = "NA" if row.get(name) is None else row[name]
            # Escaping is confined to presentation CSV, never scientific JSON or
            # typed numeric cells. Quoting alone does not neutralize formulas.
            if isinstance(value, str) and re.match(r"^[\s\x00-\x20]*[=+@-]", value):
                value = "'" + value
            cells[name] = value
        writer.writerow(cells)
    return buffer.getvalue().encode("utf-8")


def _validate_generation_bindings(
    records: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    source_namespace: str,
) -> None:
    """Bind each known generation identity to exactly one scientific source slot.

    Expected-but-missing outputs participate: absence does not make reuse of the
    same source generation key across jobs an acceptable identity ambiguity.
    """
    bindings: dict[str, str] = {}

    def bind(generation_id: str, pair_id: str, arm: str, window: list[int] | None) -> None:
        slot = canonical_json({"pair_id": pair_id, "arm": arm, "window": window})
        if generation_id in bindings and bindings[generation_id] != slot:
            raise IntakeError(
                f"Conflicting generation identity {generation_id}: reused across pair/arm/window slots"
            )
        bindings[generation_id] = slot

    for generation in generations:
        bind(
            generation["generation_id"],
            generation["pair_id"],
            generation["arm"],
            generation["window"],
        )
    for pair in records:
        for expected in pair["expected_generations"] or []:
            key = expected.get("source_generation_key")
            if key is None:
                continue
            generation_id = identity_hash(
                "rebuttal-generation/v2",
                {"source_namespace": source_namespace, "source_generation_key": key},
            )
            bind(generation_id, pair["pair_id"], expected["arm"], expected["window"])


def write_manifest(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    audit: SourceAudit,
    generations: list[dict[str, Any]],
    protocol_path: str | Path,
    protocol: dict[str, Any],
    started_at: str | None = None,
) -> dict[str, Any]:
    """Publish a complete, linked intake artifact set without overwriting a run."""
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IntakeError(f"Output directory must be absent or empty: {output}")
    started_at = started_at or _utc_now()
    protocol_path = Path(protocol_path).resolve()
    protocol_bytes = _snapshot(protocol_path, audit.index.snapshots)
    if canonical_sha256(strict_loads(_decode(protocol_bytes, protocol_path))) != canonical_sha256(
        protocol
    ):
        raise IntakeError("Protocol bytes changed between validation and publication")
    protocol_ref = _bytes_ref("protocol.json", protocol_bytes)
    protocol_hash = canonical_sha256(protocol)
    _validate_generation_bindings(records, generations, audit.index.data["source_namespace"])
    files: dict[str, bytes] = {"protocol.json": protocol_bytes}
    files["archive_generation_records.jsonl"] = _jsonl_bytes(generations)
    generation_ref = _bytes_ref(
        "archive_generation_records.jsonl", files["archive_generation_records.jsonl"]
    )
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for generation in generations:
        by_pair.setdefault(generation["pair_id"], []).append(generation)
    for pair in records:
        pair["protocol_ref"], pair["protocol_sha256"] = protocol_ref, protocol_hash
        links: dict[str, dict[str, Any]] = {}
        for generation in by_pair.get(pair["pair_id"], []):
            slot = canonical_json({"arm": generation["arm"], "window": generation["window"]})
            links[slot] = {
                "generation_id": generation["generation_id"],
                "arm": generation["arm"],
                "window": generation["window"],
                "availability": generation["availability"],
                "record_ref": {
                    "artifact": generation_ref,
                    "record_id": generation["generation_id"],
                },
                "reason_codes": list(generation["reason_codes"]),
            }
            if generation["availability"] != "available":
                audit.findings.append(
                    _finding(
                        None,
                        generation["generation_id"],
                        "raw_generation_missing",
                        "A source generation record exists but no raw text was recovered.",
                        ["answer-audit", "T1"],
                    )
                )
        known_expected_slots = {
            canonical_json({"arm": entry["arm"], "window": entry["window"]})
            for entry in pair["expected_generations"] or []
        }
        if pair["expected_generations"] is not None:
            for unexpected_slot in sorted(set(links) - known_expected_slots):
                links[unexpected_slot]["reason_codes"].append("unexpected_generation_slot")
                audit.findings.append(
                    _finding(
                        None,
                        links[unexpected_slot]["generation_id"],
                        "unexpected_generation_slot",
                        "Observed generation falls outside the explicitly enumerated archived grid; retained as an extra, not silently accepted as expected coverage.",
                        ["exact-generation-grid-coverage"],
                    )
                )
        for expected in pair["expected_generations"] or []:
            slot = canonical_json({"arm": expected["arm"], "window": expected["window"]})
            source_key = expected.get("source_generation_key")
            gid = (
                identity_hash(
                    "rebuttal-generation/v2",
                    {
                        "source_namespace": audit.index.data["source_namespace"],
                        "source_generation_key": source_key,
                    },
                )
                if source_key
                else None
            )
            if slot in links:
                if gid is not None and links[slot]["generation_id"] != gid:
                    raise IntakeError(
                        "Available generation does not match the frozen expected generation key"
                    )
                continue
            links[slot] = {
                "generation_id": gid,
                "arm": expected["arm"],
                "window": expected["window"],
                "availability": "missing",
                "record_ref": None,
                "reason_codes": ["expected_generation_missing"],
            }
            audit.findings.append(
                _finding(
                    None,
                    pair["pair_id"],
                    "expected_generation_missing",
                    f"No raw generation record for arm={expected['arm']}, window={expected['window']}.",
                    ["answer-audit", "T1"],
                )
            )
        if pair["expected_generations"] is None:
            pair["reason_codes"].append("expected_generation_grid_unknown")
        pair["archived_generations"] = [links[key] for key in sorted(links)]
    cohorts = _cohort_coverage(audit, records, protocol)
    for pair in records:
        for reason in pair["reason_codes"]:
            if reason.endswith("_unknown"):
                audit.findings.append(
                    _finding(
                        None,
                        pair["pair_id"],
                        reason,
                        "Archived evidence is unavailable; no contemporary value or derived reconstruction was substituted.",
                        ["historical-replication"]
                        if "revision" in reason
                        else ["downstream-preflight"],
                    )
                )
    source_audit = {
        "schema_version": "rebuttal-source-audit/v2",
        "source_namespace": audit.index.data["source_namespace"],
        "archive_index_ref": {
            "path": str(audit.index.path),
            "sha256": audit.index.snapshots[audit.index.path],
        },
        "cohorts": cohorts,
        "findings": audit.findings,
        "recovered_pair_count": len(records),
        "raw_generation_record_count": len(generations),
        "available_raw_generation_count": sum(
            row["availability"] == "available" for row in generations
        ),
        "run_status": "complete",
        "experiment_status": "complete",
        "experiment_id": "R0",
        "model_execution_performed": False,
        "scientific_readiness": "not_evaluated",
        "reason_codes": ["intake_only_no_model_execution"],
    }
    files["archive_inventory.jsonl"] = _jsonl_bytes(audit.inventory)
    files["source_audit.json"] = _json_bytes(source_audit)
    files["pair_manifest.jsonl"] = _jsonl_bytes(records)
    meta = {
        "schema_version": "rebuttal-pair-manifest-metadata/v2",
        "manifest_ref": _bytes_ref("pair_manifest.jsonl", files["pair_manifest.jsonl"]),
        "protocol_ref": protocol_ref,
        "protocol_sha256": protocol_hash,
        "source_audit_ref": _bytes_ref("source_audit.json", files["source_audit.json"]),
        "inventory_ref": _bytes_ref("archive_inventory.jsonl", files["archive_inventory.jsonl"]),
    }
    files["pair_manifest.meta.json"] = _json_bytes(meta)
    count_rows = [
        {
            **row,
            "historical_counts_json": canonical_json(row["historical_counts"])
            if "historical_counts" in row
            else None,
            "observed_scoring_counts_json": canonical_json(
                _observed_scoring_counts(row, generations)
            ),
        }
        for row in cohorts
    ]
    files["historical_count_comparison.csv"] = _csv_bytes(
        count_rows,
        [
            "cohort_id",
            "setting",
            "historical_n_reference",
            "recovered_count",
            "expected_count",
            "missing_count",
            "extra_count",
            "coverage_status",
            "historical_counts_json",
            "observed_scoring_counts_json",
        ],
    )
    missing_rows = [
        {**row, "reason_codes": ";".join(row["reason_codes"]), "blocks": ";".join(row["blocks"])}
        for row in audit.findings
    ]
    files["missing_inputs.csv"] = _csv_bytes(
        missing_rows, ["source_id", "component", "reason_codes", "explanation", "blocks"]
    )
    run = {
        "schema_version": "rebuttal-run/v2",
        "command": "intake",
        "mode": "cpu-audit",
        "started_at": started_at,
        "ended_at": _utc_now(),
        "invocation": {
            "archive_index": str(audit.index.path),
            "protocol": str(protocol_path),
            "output_dir": str(output),
        },
        "code_identity": _code_identity(),
        "protocol_revision": protocol["protocol_revision"],
        "input_refs": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(audit.index.snapshots.items(), key=lambda item: str(item[0]))
        ],
        "config_refs": [protocol_ref],
        "protocol_sha256": protocol_hash,
        "output_refs": {name: _bytes_ref(name, raw) for name, raw in sorted(files.items())},
        "run_status": "complete",
        "expected_ids": {row["cohort_id"]: row["expected_source_pair_keys"] for row in cohorts},
        "completed_ids": [row["pair_id"] for row in records],
        "failed_ids": [],
        "missing_ids": {row["cohort_id"]: row["missing_source_pair_keys"] for row in cohorts},
        "coverage_status": {row["cohort_id"]: row["coverage_status"] for row in cohorts},
        "reason_codes": sorted(
            {reason for row in audit.findings for reason in row["reason_codes"]}
        ),
        "explanation": "Intake completed; missing source evidence and cohort coverage remain separate from command execution status. No model execution or success-rate acceptance was performed.",
    }
    files["run.json"] = _json_bytes(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.intake-", dir=output.parent))
    try:
        for name, raw in files.items():
            (staging / name).write_bytes(raw)
        # Validate all original input snapshots immediately before publication.
        for path, digest in audit.index.snapshots.items():
            if sha256_file(path) != digest:
                raise IntakeError(f"Input artifact changed before publication: {path}")
        if output.exists():
            output.rmdir()  # Refuses a concurrent nonempty directory, no overwrite.
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return run


def run_intake(
    archive_index_path: str | Path, protocol_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Validate, audit and atomically publish R0; never compare success quotas."""
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IntakeError(f"Output directory must be absent or empty: {output}")
    started = _utc_now()
    index = read_archive_index(archive_index_path)
    protocol_path = Path(protocol_path).resolve()
    _snapshot(protocol_path, index.snapshots)
    protocol = load_protocol(protocol_path)
    audit = audit_sources(index)
    records, generations = _join_records(audit)
    return write_manifest(
        records,
        output,
        audit=audit,
        generations=generations,
        protocol_path=protocol_path,
        protocol=protocol,
        started_at=started,
    )
