"""Strict, CPU-only consumers of immutable rebuttal-v2 intake artifacts.

Every parsed value comes from the same captured bytes whose digest was checked.
The input snapshot is revalidated before publication by downstream commands.
Unknown inputs listed by intake remain unknown; a broken ArtifactRef is an error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from .identity import (
    GROUP_IDENTITY_FIELDS,
    PAIR_IDENTITY_FIELDS,
    IdentityAliases,
    identity_hash,
    original_problem_group_id_for,
    parse_identity_aliases,
)
from .schemas import (
    PROTOCOL_REVISION,
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_SHA256,
    IntakeError,
    canonical_json,
    canonical_sha256,
    require_bool,
    require_enum,
    require_int,
    require_keys,
    require_nullable_string,
    require_object,
    require_sha256,
    require_string,
    require_string_list,
    require_timestamp,
    sha256_text,
    strict_loads,
    validate_artifact_ref,
    validate_record_ref,
)
from .source import (
    AVAILABILITIES,
    COHORT_REQUIRED,
    GENERATION_OPTIONAL,
    GENERATION_REQUIRED,
    GENERATION_SCHEMA,
    PAIR_OPTIONAL,
    PAIR_REQUIRED,
    PAIR_SCHEMA,
    SOURCE_KINDS,
    SOURCE_REQUIRED,
    SOURCE_ROLES,
)


_PAIR_FIELDS = {
    "schema_version",
    "protocol_ref",
    "protocol_sha256",
    "pair_id",
    "source_namespace",
    "source_pair_key",
    "source_pair_key_kind",
    "historical_pair_id",
    "original_problem_group_id",
    "dataset_id",
    "split",
    "original_problem_id",
    "dataset_revision",
    "cohort_ids",
    "cohort_membership",
    "source_kind",
    "source_ref",
    "source_sha256",
    "source_record_id",
    "pair_identity",
    "group_identity",
    "source_identity_provenance",
    "additional_source_refs",
    "task",
    "model_id",
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
    "archived_generations",
    "availability",
    "reason_codes",
    "expected_generations",
    "archived_scoring",
}
_GENERATION_FIELDS = {
    "schema_version",
    "generation_id",
    "generation_identity",
    "pair_id",
    "source_ref",
    "source_kind",
    "source_record_id",
    "source_generation_key",
    "arm",
    "window",
    "raw_text",
    "raw_text_ref",
    "raw_text_sha256",
    "generated_token_ids",
    "stopping_metadata",
    "termination",
    "archived_termination",
    "archived_runtime",
    "archived_parser",
    "archived_scoring",
    "availability",
    "reason_codes",
}
_COHORT_EXTRA_FIELDS = {
    "expected_source_pair_keys",
    "coverage_status",
    "recovered_count",
    "expected_count",
    "missing_count",
    "extra_count",
    "missing_source_pair_keys",
    "extra_pair_ids",
    "recovered_pair_ids",
    "historical_count_shortfall",
    "reason_codes",
}


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise IntakeError(f"{context} must be a JSON array")
    return value


def _decode(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise IntakeError(f"artifact is not UTF-8: {path}") from exc


class ArtifactSnapshots:
    """Capture once, verify every reference, and parse only captured bytes."""

    def __init__(self) -> None:
        self.snapshots: dict[Path, str] = {}
        self._bytes: dict[Path, bytes] = {}
        self._rows: dict[tuple[Path, str], dict[str, dict[str, Any]]] = {}

    def capture(self, path: Path, expected: str | None = None) -> bytes:
        path = path.resolve()
        if path not in self._bytes:
            try:
                raw = path.read_bytes()
            except (OSError, ValueError) as exc:
                raise IntakeError(f"cannot read referenced artifact {path}: {exc}") from exc
            self._bytes[path] = raw
            self.snapshots[path] = hashlib.sha256(raw).hexdigest()
        if expected is not None and self.snapshots[path] != expected:
            raise IntakeError(f"artifact SHA-256 mismatch at {path}")
        return self._bytes[path]

    def reference(self, ref: object, containing_file: Path) -> tuple[Path, bytes]:
        parsed = validate_artifact_ref(ref)
        path = (containing_file.parent / parsed["path"]).resolve()
        return path, self.capture(path, parsed["sha256"])

    def object(self, path: Path) -> dict[str, Any]:
        return require_object(strict_loads(_decode(self.capture(path), path), str(path)), str(path))

    def jsonl(self, path: Path) -> list[dict[str, Any]]:
        # splitlines() would split U+0085/U+2028/U+2029 inside exact JSON strings.
        return [
            require_object(strict_loads(line, f"{path}:{number}"), f"{path}:{number}")
            for number, line in enumerate(_decode(self.capture(path), path).split("\n"), 1)
            if line.strip()
        ]

    def record(
        self, ref: object, containing_file: Path, *, id_field: str, schema: str
    ) -> tuple[Path, dict[str, Any]]:
        parsed = validate_record_ref(ref)
        path, _ = self.reference(parsed["artifact"], containing_file)
        key = (path, id_field)
        if key not in self._rows:
            rows: dict[str, dict[str, Any]] = {}
            for row in self.jsonl(path):
                require_enum(row.get("schema_version"), {schema}, "referenced row schema_version")
                rid = require_string(row.get(id_field), id_field)
                if rid in rows:
                    raise IntakeError(f"duplicate {id_field} in {path}: {rid}")
                rows[rid] = row
            self._rows[key] = rows
        rows = self._rows[key]
        rid = parsed["record_id"]
        if rid not in rows:
            raise IntakeError(f"RecordRef record_id {rid!r} not found in {path}")
        row = rows[rid]
        require_enum(row.get("schema_version"), {schema}, "referenced row schema_version")
        return path, row

    def references_in(self, payload: object, containing_file: Path) -> None:
        """Verify explicit reference values, never reinterpret free-form strings."""
        if isinstance(payload, list):
            for item in payload:
                self.references_in(item, containing_file)
        elif isinstance(payload, dict):
            if set(payload) == {"path", "sha256"}:
                self.reference(payload, containing_file)
            else:
                for key, value in payload.items():
                    if key.endswith("_ref") and value is not None:
                        if isinstance(value, dict) and set(value) == {"artifact", "record_id"}:
                            validate_record_ref(value)
                            self.reference(value["artifact"], containing_file)
                        else:
                            self.reference(value, containing_file)
                    elif isinstance(value, (dict, list)):
                        self.references_in(value, containing_file)


@dataclass(frozen=True)
class GenerationSlot:
    pair: dict[str, Any]
    link: dict[str, Any]
    generation: dict[str, Any] | None
    raw_text: str | None
    generation_path: Path | None


@dataclass(frozen=True)
class ManifestBundle:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    protocol: dict[str, Any]
    protocol_path: Path
    protocol_bytes: bytes
    source_audit: dict[str, Any]
    inventory: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    slots: list[GenerationSlot]
    snapshots: dict[Path, str]

    @property
    def manifest_ref(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.snapshots[self.path]}

    @property
    def metadata_ref(self) -> dict[str, str]:
        return {"path": str(self.metadata_path), "sha256": self.snapshots[self.metadata_path]}


def revalidate_inputs(bundle: ManifestBundle) -> None:
    """Fail publication if any consumed file no longer has its captured bytes."""
    for path, expected in bundle.snapshots.items():
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise IntakeError(f"cannot revalidate input {path}: {exc}") from exc
        if actual != expected:
            raise IntakeError(f"input changed after capture: {path}")


def _slot(row: dict[str, Any]) -> str:
    arm = require_enum(row["arm"], {"clean", "typo", "self", "correct", "offset", "cross"}, "arm")
    window = row["window"]
    if arm in {"clean", "typo"}:
        if window is not None:
            raise IntakeError("baseline generation window must be null")
    else:
        if not isinstance(window, list) or len(window) != 2:
            raise IntakeError("patch window must be [start,end]")
        start = require_int(window[0], "window start", minimum=0)
        end = require_int(window[1], "window end", minimum=0)
        if end <= start:
            raise IntakeError("window end must exceed start")
    return canonical_json({"arm": arm, "window": window})


def _text(
    row: dict[str, Any],
    name: str,
    path: Path,
    snapshots: ArtifactSnapshots,
    *,
    normalized: bool = True,
) -> str | None:
    inline, ref, digest = row.get(name), row.get(f"{name}_ref"), row.get(f"{name}_sha256")
    if inline is not None and ref is not None:
        raise IntakeError(f"{name} requires exactly one inline/reference representation")
    if inline is not None:
        text = require_string(inline, name, allow_empty=True)
    elif ref is not None:
        text_path, raw = snapshots.reference(ref, path)
        text = _decode(raw, text_path)
    else:
        text = None
    if digest is not None:
        require_sha256(digest, f"{name}_sha256")
    if (normalized or digest is not None) and digest != (
        sha256_text(text) if text is not None else None
    ):
        raise IntakeError(f"{name} text hash mismatch or hash without text")
    return text


def _pair_metadata(row: dict[str, Any]) -> None:
    for key in ("source_record_id", "source_pair_key", "task", "model_id"):
        require_string(row[key], key)
    for key in (
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
        require_nullable_string(row.get(key), key)
    require_string_list(row.get("cohort_ids", []), "cohort_ids", unique=True)
    for key in ("archived_runtime", "archived_scoring", "prompt_regions"):
        if row.get(key) is not None:
            require_object(row[key], key)
    for key in ("intended_edits", "archived_actual_edits", "archived_token_positions"):
        if row.get(key) is not None:
            _array(row[key], key)
    for key in ("clean_query_span", "typo_query_span"):
        span = row.get(key)
        if span is not None:
            if not isinstance(span, list) or len(span) != 2:
                raise IntakeError(f"{key} must be [start,end]")
            start = require_int(span[0], key, minimum=0)
            end = require_int(span[1], key, minimum=0)
            if end < start:
                raise IntakeError(f"{key} end precedes start")
    labels = row.get("valid_labels")
    if labels is not None:
        require_string_list(labels, "valid_labels", unique=True)
    gold = row.get("canonical_gold")
    if gold is not None:
        require_string(row.get("gold_canonicalization_version"), "gold_canonicalization_version")
        if row["task"] == "gsm8k":
            require_object(gold, "canonical_gold")
            require_keys(gold, {"numerator", "denominator"}, context="canonical_gold")
            numerator = require_string(gold["numerator"], "gold numerator")
            denominator = require_string(gold["denominator"], "gold denominator")
            if not re.fullmatch(r"(?:0|-?[1-9][0-9]*)", numerator) or not re.fullmatch(
                r"[1-9][0-9]*", denominator
            ):
                raise IntakeError("canonical_gold requires normalized integer strings")
            try:
                reduced = math.gcd(int(numerator), int(denominator)) == 1
            except ValueError as exc:
                raise IntakeError("canonical_gold integer is too large") from exc
            if not reduced:
                raise IntakeError("canonical_gold must be a reduced rational")
        elif row["task"] == "mmlu-pro":
            label = require_string(gold, "canonical_gold")
            if labels is not None and label not in labels:
                raise IntakeError("canonical_gold is not an item-valid label")
        else:
            raise IntakeError("unsupported canonical_gold task")


def _identity(
    value: object,
    *,
    pair: bool,
    aliases: IdentityAliases,
    path: Path,
    snapshots: ArtifactSnapshots,
) -> dict[str, Any]:
    identity = require_object(value, "identity")
    require_keys(
        identity,
        {"domain", "payload", "original_payload", "alias_evidence_refs"},
        context="identity",
    )
    fields = PAIR_IDENTITY_FIELDS if pair else GROUP_IDENTITY_FIELDS
    domain = "rebuttal-pair/v2" if pair else "rebuttal-original-problem/v2"
    require_enum(identity["domain"], {domain}, "identity domain")
    for key in ("payload", "original_payload"):
        payload = require_object(identity[key], f"identity {key}")
        require_keys(payload, fields, context=f"identity {key}")
        for field in fields:
            (require_string if pair else require_nullable_string)(payload[field], field)
    original = identity["original_payload"]
    resolution = aliases.resolve_pair(**original) if pair else aliases.resolve_group(**original)
    if identity["payload"] != resolution.canonical:
        raise IntakeError("identity canonical payload does not match verified aliases")
    supplied = []
    for ref in _array(identity["alias_evidence_refs"], "alias_evidence_refs"):
        ref_path, _ = snapshots.reference(ref, path)
        supplied.append({"path": str(ref_path), "sha256": snapshots.snapshots[ref_path]})
    if supplied != list(resolution.evidence_refs):
        raise IntakeError("identity alias evidence does not match verified aliases")
    return identity


def _source_fields(row: dict[str, Any]) -> None:
    require_string(row["source_id"], "source_id")
    require_enum(row["source_kind"], SOURCE_KINDS, "source_kind")
    require_enum(row["role"], SOURCE_ROLES, "source role")
    require_enum(row["availability"], AVAILABILITIES, "source availability")
    require_string_list(row["reason_codes"], "source reason_codes", unique=True)
    for key in (
        "path",
        "format",
        "adapter_version",
        "source_commit",
        "model_revision",
        "tokenizer_revision",
        "prompt_template_revision",
    ):
        require_nullable_string(row[key], key)
    if row["expected_sha256"] is not None:
        require_sha256(row["expected_sha256"], "expected_sha256")
    if row["generation_config"] is not None:
        require_object(row["generation_config"], "generation_config")


def _index_and_inventory(
    audit: dict[str, Any],
    audit_path: Path,
    inventory: list[dict[str, Any]],
    inventory_path: Path,
    snapshots: ArtifactSnapshots,
) -> tuple[dict[str, Any], IdentityAliases, dict[Path, dict[str, Any]]]:
    index_path, _ = snapshots.reference(audit["archive_index_ref"], audit_path)
    index = snapshots.object(index_path)
    require_keys(
        index,
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
        index["schema_version"], {"rebuttal-archive-index/v2"}, "archive index schema_version"
    )
    require_timestamp(index["created_at"], "archive index created_at")
    require_string(index["source_namespace"], "source_namespace")
    require_string(index["notes"], "notes", allow_empty=True)
    if index["source_namespace"] != audit["source_namespace"]:
        raise IntakeError("source audit namespace differs from archive index")
    sources = {}
    for row in _array(index["sources"], "sources"):
        require_object(row, "source")
        require_keys(row, SOURCE_REQUIRED, context="source")
        _source_fields(row)
        if row["source_id"] in sources:
            raise IntakeError("duplicate source_id in archive index")
        sources[row["source_id"]] = row
    cohorts = {}
    for cohort in _array(index["cohorts"], "cohorts"):
        require_object(cohort, "cohort")
        require_keys(cohort, COHORT_REQUIRED, {"historical_counts"}, context="cohort")
        cid = require_string(cohort["cohort_id"], "cohort_id")
        if cid in cohorts:
            raise IntakeError("duplicate cohort_id in archive index")
        cohorts[cid] = cohort
        require_string(cohort["setting"], "cohort setting")
        require_string(cohort["id_namespace"], "cohort id_namespace")
        require_int(cohort["historical_n_reference"], "historical_n_reference", minimum=0)
        if "historical_counts" in cohort:
            require_object(cohort["historical_counts"], "historical_counts")
        snapshots.references_in(cohort, index_path)
        if cohort["expected_ids_ref"] is not None:
            ids_path, _ = snapshots.reference(cohort["expected_ids_ref"], index_path)
            ids = snapshots.object(ids_path)
            require_keys(
                ids,
                {"schema_version", "cohort_id", "source_namespace", "source_pair_keys"},
                context="cohort IDs",
            )
            require_enum(
                ids["schema_version"], {"rebuttal-cohort-ids/v2"}, "cohort IDs schema_version"
            )
            require_string_list(ids["source_pair_keys"], "source_pair_keys", unique=True)
            if ids["cohort_id"] != cid or ids["source_namespace"] != cohort["id_namespace"]:
                raise IntakeError("cohort ID reference binding mismatch")
        matching = [row for row in audit["cohorts"] if row["cohort_id"] == cid]
        if len(matching) != 1:
            raise IntakeError("source audit does not contain the indexed cohort")
        normalized = matching[0]
        for name in ("expected_ids_ref", "membership_provenance_ref"):
            if cohort[name] is None:
                if normalized[name] is not None:
                    raise IntakeError("source audit invented a cohort input reference")
            else:
                original_path, _ = snapshots.reference(cohort[name], index_path)
                normalized_path, _ = snapshots.reference(normalized[name], audit_path)
                if original_path != normalized_path:
                    raise IntakeError("source audit cohort reference differs from archive index")
        expected_keys = ids["source_pair_keys"] if cohort["expected_ids_ref"] is not None else None
        if normalized["expected_source_pair_keys"] != expected_keys:
            raise IntakeError("source audit cohort ID list differs from frozen source artifact")
    aliases = IdentityAliases()
    if index["identity_aliases_ref"] is not None:
        alias_path, _ = snapshots.reference(index["identity_aliases_ref"], index_path)
        payload = snapshots.object(alias_path)
        snapshots.references_in(payload, alias_path)
        aliases = parse_identity_aliases(payload, alias_path)
    seen = set()
    available: dict[Path, dict[str, Any]] = {}
    for row in inventory:
        require_keys(
            row,
            SOURCE_REQUIRED
            | {
                "schema_version",
                "requested_availability",
                "observed_sha256",
                "historical_hash_status",
                "adapter_status",
                "record_count",
            },
            context="inventory row",
        )
        require_enum(
            row["schema_version"], {"rebuttal-archive-inventory/v2"}, "inventory schema_version"
        )
        _source_fields(row)
        sid = row["source_id"]
        if sid in seen or sid not in sources:
            raise IntakeError("duplicate or undeclared inventory source_id")
        seen.add(sid)
        original = sources[sid]
        for key in SOURCE_REQUIRED - {"path", "availability", "reason_codes"}:
            if row[key] != original[key]:
                raise IntakeError(f"inventory source field mismatch: {key}")
        if row["requested_availability"] != original["availability"]:
            raise IntakeError("inventory requested_availability mismatch")
        require_enum(
            row["historical_hash_status"],
            {"verified", "unknown", "not_applicable"},
            "historical_hash_status",
        )
        require_enum(
            row["adapter_status"], {"supported", "unsupported", "not_applicable"}, "adapter_status"
        )
        if row["record_count"] is not None:
            require_int(row["record_count"], "record_count", minimum=0)
        expected_path = (
            (index_path.parent / original["path"]).resolve()
            if original["path"] is not None
            else None
        )
        actual_path = (
            (inventory_path.parent / row["path"]).resolve() if row["path"] is not None else None
        )
        if actual_path != expected_path:
            raise IntakeError("inventory path differs from archive index")
        if row["availability"] == "available":
            digest = require_sha256(row["observed_sha256"], "observed_sha256")
            if actual_path is None:
                raise IntakeError("available inventory source has no path")
            snapshots.capture(actual_path, digest)
            if original["expected_sha256"] is not None and digest != original["expected_sha256"]:
                raise IntakeError("inventory observed and historical SHA-256 mismatch")
            available[actual_path] = row
            if row["adapter_status"] == "supported":
                expected_schema = {"pairs": PAIR_SCHEMA, "generations": GENERATION_SCHEMA}.get(
                    row["role"]
                )
                if (
                    row["format"] != "jsonl"
                    or row["adapter_version"] != expected_schema
                    or expected_schema is None
                ):
                    raise IntakeError("unsupported source incorrectly marked supported")
                source_rows = snapshots.jsonl(actual_path)
                ids = set()
                for source_row in source_rows:
                    schema = source_row.get("schema_version")
                    require_enum(schema, {expected_schema}, "source row schema_version")
                    required, optional = (
                        (PAIR_REQUIRED, PAIR_OPTIONAL)
                        if schema == PAIR_SCHEMA
                        else (GENERATION_REQUIRED, GENERATION_OPTIONAL)
                    )
                    require_keys(source_row, required, optional, context="source row")
                    rid = require_string(source_row["source_record_id"], "source_record_id")
                    if rid in ids:
                        raise IntakeError("duplicate source_record_id in inventory source")
                    ids.add(rid)
                    snapshots.references_in(source_row, actual_path)
                    if schema == PAIR_SCHEMA:
                        _pair_metadata(source_row)
                        for name in ("clean_text", "typo_text", "clean_prompt", "typo_prompt"):
                            _text(source_row, name, actual_path, snapshots, normalized=False)
                    else:
                        _slot(source_row)
                        _text(source_row, "raw_text", actual_path, snapshots, normalized=False)
                if row["record_count"] != len(source_rows):
                    raise IntakeError("inventory record_count mismatch")
        elif row["observed_sha256"] is not None:
            raise IntakeError("unavailable inventory source has an observed hash")
    if seen != set(sources):
        raise IntakeError("inventory does not cover every indexed source")
    return index, aliases, available


def _audit(payload: dict[str, Any], path: Path, snapshots: ArtifactSnapshots) -> None:
    require_keys(
        payload,
        {
            "schema_version",
            "source_namespace",
            "archive_index_ref",
            "cohorts",
            "findings",
            "recovered_pair_count",
            "raw_generation_record_count",
            "available_raw_generation_count",
            "run_status",
            "experiment_status",
            "experiment_id",
            "model_execution_performed",
            "scientific_readiness",
            "reason_codes",
        },
        context="source audit",
    )
    require_enum(
        payload["schema_version"], {"rebuttal-source-audit/v2"}, "source audit schema_version"
    )
    require_string(payload["source_namespace"], "source_namespace")
    for name in (
        "recovered_pair_count",
        "raw_generation_record_count",
        "available_raw_generation_count",
    ):
        require_int(payload[name], name, minimum=0)
    for name, expected in (
        ("run_status", "complete"),
        ("experiment_status", "complete"),
        ("experiment_id", "R0"),
        ("scientific_readiness", "not_evaluated"),
    ):
        require_enum(payload[name], {expected}, name)
    if require_bool(payload["model_execution_performed"], "model_execution_performed"):
        raise IntakeError("intake source audit cannot claim model execution")
    require_string_list(payload["reason_codes"], "source audit reason_codes", unique=True)
    for finding in _array(payload["findings"], "findings"):
        require_object(finding, "finding")
        require_keys(
            finding,
            {"source_id", "component", "reason_codes", "explanation", "blocks"},
            context="finding",
        )
        require_nullable_string(finding["source_id"], "finding source_id")
        require_string(finding["component"], "finding component")
        require_string(finding["explanation"], "finding explanation")
        for key in ("reason_codes", "blocks"):
            require_string_list(finding[key], key, unique=True)
    ids = set()
    for cohort in _array(payload["cohorts"], "source audit cohorts"):
        require_object(cohort, "cohort")
        require_keys(
            cohort,
            COHORT_REQUIRED | _COHORT_EXTRA_FIELDS,
            {"historical_counts"},
            context="source audit cohort",
        )
        cid = require_string(cohort["cohort_id"], "cohort_id")
        if cid in ids:
            raise IntakeError("duplicate cohort_id in source audit")
        ids.add(cid)
        require_enum(
            cohort["coverage_status"], {"complete", "partial", "unknown"}, "coverage_status"
        )
        for key in (
            "historical_n_reference",
            "recovered_count",
            "extra_count",
            "historical_count_shortfall",
        ):
            require_int(cohort[key], key, minimum=0)
        for key in ("expected_count", "missing_count"):
            if cohort[key] is not None:
                require_int(cohort[key], key, minimum=0)
        for key in ("extra_pair_ids", "recovered_pair_ids", "reason_codes"):
            require_string_list(cohort[key], key, unique=True)
        for key in ("expected_source_pair_keys", "missing_source_pair_keys"):
            if cohort[key] is not None:
                require_string_list(cohort[key], key, unique=True)
    snapshots.references_in(payload, path)


def _source_record(
    ref: object,
    path: Path,
    snapshots: ArtifactSnapshots,
    available: dict[Path, dict[str, Any]],
    *,
    pair: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source_path, raw = snapshots.record(
        ref, path, id_field="source_record_id", schema=PAIR_SCHEMA if pair else GENERATION_SCHEMA
    )
    inventory = available.get(source_path)
    if (
        inventory is None
        or inventory["role"] != ("pairs" if pair else "generations")
        or inventory["adapter_status"] != "supported"
    ):
        raise IntakeError("source RecordRef is not a matching available inventory source")
    return source_path, raw, inventory


def _pair(
    row: dict[str, Any],
    path: Path,
    snapshots: ArtifactSnapshots,
    aliases: IdentityAliases,
    available: dict[Path, dict[str, Any]],
    protocol: dict[str, Any],
    namespace: str,
) -> None:
    require_keys(row, _PAIR_FIELDS, context="pair manifest row")
    require_enum(row["schema_version"], {"rebuttal-pair-manifest/v2"}, "pair schema_version")
    _pair_metadata(row)
    require_sha256(row["pair_id"], "pair_id")
    if row["protocol_sha256"] != PROTOCOL_SHA256:
        raise IntakeError("pair protocol_sha256 differs from frozen protocol")
    protocol_path, _ = snapshots.reference(row["protocol_ref"], path)
    if snapshots.object(protocol_path) != protocol:
        raise IntakeError("pair protocol reference differs from manifest protocol")
    pair_identity = _identity(
        row["pair_identity"], pair=True, aliases=aliases, path=path, snapshots=snapshots
    )
    group_identity = _identity(
        row["group_identity"], pair=False, aliases=aliases, path=path, snapshots=snapshots
    )
    if (
        pair_identity["payload"] != {key: row[key] for key in PAIR_IDENTITY_FIELDS}
        or identity_hash(pair_identity["domain"], pair_identity["payload"]) != row["pair_id"]
    ):
        raise IntakeError("pair identity binding mismatch")
    if (
        group_identity["payload"] != {key: row[key] for key in GROUP_IDENTITY_FIELDS}
        or original_problem_group_id_for(**group_identity["payload"])
        != row["original_problem_group_id"]
    ):
        raise IntakeError("original-problem group identity binding mismatch")
    require_enum(
        row["source_pair_key_kind"], {"historical-id", "producer-id"}, "source_pair_key_kind"
    )
    require_enum(row["source_kind"], SOURCE_KINDS, "source_kind")
    require_string_list(row["reason_codes"], "reason_codes", unique=True)
    source_path, raw, inventory = _source_record(
        row["source_ref"], path, snapshots, available, pair=True
    )
    if (
        row["source_sha256"] != snapshots.snapshots[source_path]
        or row["source_record_id"] != raw["source_record_id"]
        or row["source_kind"] != inventory["source_kind"]
    ):
        raise IntakeError("pair source provenance binding mismatch")
    provenance = _array(row["source_identity_provenance"], "source_identity_provenance")
    if not provenance:
        raise IntakeError("source_identity_provenance must be nonempty")
    claimed_refs = set()
    for claim in provenance:
        require_object(claim, "identity provenance")
        require_keys(
            claim,
            {
                "source_ref",
                "pair_identity",
                "group_identity",
                "cohort_ids",
                "historical_pair_id",
                "source_pair_key_kind",
                "source_pair_key_kind_explicit",
            },
            context="identity provenance",
        )
        claim_path, claim_raw, claim_inventory = _source_record(
            claim["source_ref"], path, snapshots, available, pair=True
        )
        claim_pair = _identity(
            claim["pair_identity"], pair=True, aliases=aliases, path=path, snapshots=snapshots
        )
        claim_group = _identity(
            claim["group_identity"], pair=False, aliases=aliases, path=path, snapshots=snapshots
        )
        if (
            claim_pair["payload"] != pair_identity["payload"]
            or claim_group["payload"] != group_identity["payload"]
        ):
            raise IntakeError("source provenance canonical identity mismatch")
        if claim_pair["original_payload"] != {
            "source_namespace": namespace,
            "source_pair_key": claim_raw["source_pair_key"],
        } or claim_group["original_payload"] != {
            key: claim_raw.get(key) for key in GROUP_IDENTITY_FIELDS
        }:
            raise IntakeError("source provenance original identity mismatch")
        require_string_list(claim["cohort_ids"], "provenance cohort_ids", unique=True)
        if claim["cohort_ids"] != sorted(claim_raw.get("cohort_ids", [])) or claim[
            "historical_pair_id"
        ] != claim_raw.get("historical_pair_id"):
            raise IntakeError("source provenance membership/history mismatch")
        explicit = require_bool(
            claim["source_pair_key_kind_explicit"], "source_pair_key_kind_explicit"
        )
        kind = claim_raw.get(
            "source_pair_key_kind",
            "historical-id" if claim_raw.get("historical_pair_id") is not None else "producer-id",
        )
        if (
            explicit != ("source_pair_key_kind" in claim_raw)
            or claim["source_pair_key_kind"] != kind
        ):
            raise IntakeError("source provenance key-kind mismatch")
        if claim_inventory["source_kind"] != row["source_kind"]:
            raise IntakeError("source provenance source_kind mismatch")
        for key in (
            "task",
            "model_id",
            "target_rule",
            "perturbation_id",
            "gold",
            "canonical_gold",
            "gold_canonicalization_version",
            "valid_labels",
            "dataset_revision",
            "clean_query_span",
            "typo_query_span",
            "prompt_regions",
            "intended_edits",
            "archived_actual_edits",
            "archived_token_positions",
            "archived_scoring",
            "expected_generations",
        ):
            if claim_raw.get(key) != row[key]:
                raise IntakeError(f"pair field differs from acquisition source: {key}")
        for key in (
            "model_revision",
            "tokenizer_revision",
            "prompt_template_revision",
            "archived_runtime",
        ):
            fallback = (
                claim_inventory["generation_config"]
                if key == "archived_runtime"
                else claim_inventory[key]
            )
            if claim_raw.get(key, fallback) != row[key]:
                raise IntakeError(
                    f"pair archived provenance differs from acquisition source: {key}"
                )
        for name in ("clean_text", "typo_text", "clean_prompt", "typo_prompt"):
            text = _text(claim_raw, name, claim_path, snapshots, normalized=False)
            if (sha256_text(text) if text is not None else None) != row[f"{name}_sha256"]:
                raise IntakeError(f"pair {name} differs from acquisition source")
        ref_key = canonical_json(
            {
                "artifact": {"path": str(claim_path), "sha256": snapshots.snapshots[claim_path]},
                "record_id": claim_raw["source_record_id"],
            }
        )
        if ref_key in claimed_refs:
            raise IntakeError("duplicate source identity provenance reference")
        claimed_refs.add(ref_key)
    declared_refs = set()
    for ref in [
        row["source_ref"],
        *_array(row["additional_source_refs"], "additional_source_refs"),
    ]:
        ref_path, ref_raw, _ = _source_record(ref, path, snapshots, available, pair=True)
        key = canonical_json(
            {
                "artifact": {"path": str(ref_path), "sha256": snapshots.snapshots[ref_path]},
                "record_id": ref_raw["source_record_id"],
            }
        )
        if key in declared_refs:
            raise IntakeError("duplicate pair acquisition RecordRef")
        declared_refs.add(key)
    if declared_refs != claimed_refs:
        raise IntakeError("pair source refs differ from identity provenance")
    availability = require_object(row["availability"], "pair availability")
    require_keys(
        availability,
        {"clean_text", "typo_text", "clean_prompt", "typo_prompt"},
        context="pair availability",
    )
    texts = {}
    for name in availability:
        texts[name] = _text(row, name, path, snapshots)
        expected = "unknown" if texts[name] is None else "available"
        if availability[name] != expected:
            raise IntakeError(f"{name} availability contradicts payload")
    for side in ("clean", "typo"):
        span = row[f"{side}_query_span"]
        prompt, text = texts[f"{side}_prompt"], texts[f"{side}_text"]
        if (
            span is not None
            and prompt is not None
            and text is not None
            and (span[1] > len(prompt) or prompt[span[0] : span[1]] != text)
        ):
            raise IntakeError("query span does not identify exact query text")
    cohort_ids = set()
    for membership in _array(row["cohort_membership"], "cohort_membership"):
        require_object(membership, "cohort membership")
        require_keys(
            membership, {"cohort_id", "membership", "source_ref"}, context="cohort membership"
        )
        cid = require_string(membership["cohort_id"], "cohort_id")
        if cid in cohort_ids:
            raise IntakeError("duplicate cohort membership")
        cohort_ids.add(cid)
        require_enum(membership["membership"], {"member", "extra", "unknown"}, "membership")
        snapshots.reference(membership["source_ref"], path)
    if row["expected_generations"] is not None:
        seen = set()
        for expected in _array(row["expected_generations"], "expected_generations"):
            require_object(expected, "expected generation")
            require_keys(
                expected,
                {"arm", "window"},
                {"source_generation_key"},
                context="expected generation",
            )
            require_nullable_string(expected.get("source_generation_key"), "source_generation_key")
            key = _slot(expected)
            if key in seen:
                raise IntakeError("duplicate expected arm/window")
            seen.add(key)


def _generation(
    row: dict[str, Any],
    path: Path,
    snapshots: ArtifactSnapshots,
    aliases: IdentityAliases,
    available: dict[Path, dict[str, Any]],
    namespace: str,
) -> str | None:
    require_keys(row, _GENERATION_FIELDS, context="archive generation")
    require_enum(
        row["schema_version"], {"rebuttal-archive-generation/v2"}, "generation schema_version"
    )
    _slot(row)
    for name in ("generation_id", "pair_id"):
        require_sha256(row[name], name)
    require_string_list(row["reason_codes"], "generation reason_codes", unique=True)
    identity = require_object(row["generation_identity"], "generation_identity")
    require_keys(identity, {"domain", "payload"}, context="generation_identity")
    require_enum(identity["domain"], {"rebuttal-generation/v2"}, "generation identity domain")
    payload = require_object(identity["payload"], "generation identity payload")
    require_keys(
        payload,
        {"source_namespace", "source_generation_key"},
        context="generation identity payload",
    )
    require_string(row["source_generation_key"], "source_generation_key")
    if (
        payload
        != {"source_namespace": namespace, "source_generation_key": row["source_generation_key"]}
        or identity_hash(identity["domain"], payload) != row["generation_id"]
    ):
        raise IntakeError("generation identity binding mismatch")
    source_path, raw, inventory = _source_record(
        row["source_ref"], path, snapshots, available, pair=False
    )
    pair = aliases.resolve_pair(namespace, raw["source_pair_key"]).canonical
    if (
        row["pair_id"] != identity_hash("rebuttal-pair/v2", pair)
        or row["source_record_id"] != raw["source_record_id"]
        or row["source_generation_key"] != raw["source_generation_key"]
        or _slot(raw) != _slot(row)
        or row["source_kind"] != inventory["source_kind"]
    ):
        raise IntakeError("generation source binding mismatch")
    text = _text(row, "raw_text", path, snapshots)
    if text != _text(raw, "raw_text", source_path, snapshots, normalized=False):
        raise IntakeError("generation raw text differs from original source")
    if row["availability"] != ("available" if text is not None else "missing"):
        raise IntakeError("generation availability contradicts raw payload")
    for name in ("stopping_metadata", "archived_runtime", "archived_parser", "archived_scoring"):
        if row[name] is not None:
            require_object(row[name], name)
        original = raw.get(
            name, inventory["generation_config"] if name == "archived_runtime" else None
        )
        if row[name] != original:
            raise IntakeError(f"generation metadata differs from source: {name}")
    if row["generated_token_ids"] is not None:
        for token in _array(row["generated_token_ids"], "generated_token_ids"):
            require_int(token, "generated token ID", minimum=0)
    if row["generated_token_ids"] != raw.get("generated_token_ids") or row[
        "archived_termination"
    ] != raw.get("termination"):
        raise IntakeError("generation stopping/token provenance mismatch")
    require_nullable_string(row["archived_termination"], "archived_termination")
    stopping = row["stopping_metadata"] or {}
    for name in ("eos_observed", "length_cap_reached"):
        if name in stopping:
            require_bool(stopping[name], name)
    termination = (
        "eos"
        if stopping.get("eos_observed") is True
        else "length-cap"
        if stopping.get("eos_observed") is False and stopping.get("length_cap_reached") is True
        else "unknown"
    )
    if row["termination"] != termination:
        raise IntakeError("generation termination contradicts observed stopping metadata")
    scoring = row["archived_scoring"]
    if scoring is not None:
        require_keys(
            scoring,
            set(),
            {"parser_id", "parser_version", "gold_match"},
            context="archived_scoring",
        )
        for key in ("parser_id", "parser_version"):
            require_nullable_string(scoring.get(key), key)
        if scoring.get("gold_match") is not None:
            require_bool(scoring["gold_match"], "gold_match")
    return text


def _source_coverage(
    pairs: list[dict[str, Any]],
    slots: list[GenerationSlot],
    audit: dict[str, Any],
    index: dict[str, Any],
    aliases: IdentityAliases,
    available: dict[Path, dict[str, Any]],
    snapshots: ArtifactSnapshots,
) -> None:
    """Cross-check reported coverage against verified acquisition records."""
    namespace = index["source_namespace"]
    source_pairs = set()
    source_generations: dict[str, tuple[str, str]] = {}
    source_slots = set()
    available_raw = 0
    for source_path, inventory in available.items():
        if inventory["adapter_status"] != "supported":
            continue
        for row in snapshots.jsonl(source_path):
            pair = aliases.resolve_pair(namespace, row["source_pair_key"]).canonical
            pid = identity_hash("rebuttal-pair/v2", pair)
            if inventory["role"] == "pairs":
                source_pairs.add(pid)
            elif inventory["role"] == "generations":
                gid = identity_hash(
                    "rebuttal-generation/v2",
                    {
                        "source_namespace": namespace,
                        "source_generation_key": row["source_generation_key"],
                    },
                )
                binding = (pid, _slot(row))
                if gid in source_generations or binding in source_slots:
                    raise IntakeError("duplicate source generation identity or slot")
                source_generations[gid] = binding
                source_slots.add(binding)
                available_raw += (
                    _text(row, "raw_text", source_path, snapshots, normalized=False) is not None
                )
    if source_pairs != {pair["pair_id"] for pair in pairs}:
        raise IntakeError("manifest pair coverage differs from verified source records")
    linked = {slot.generation["generation_id"] for slot in slots if slot.generation is not None}
    required = {gid for gid, (pid, _) in source_generations.items() if pid in source_pairs}
    if linked != required:
        raise IntakeError("manifest omits or invents an observed generation source slot")
    if (
        audit["raw_generation_record_count"] != len(source_generations)
        or audit["available_raw_generation_count"] != available_raw
    ):
        raise IntakeError("source audit generation coverage differs from verified source records")
    orphan_ids = {gid for gid, (pid, _) in source_generations.items() if pid not in source_pairs}
    recorded_orphans = {
        finding["component"]
        for finding in audit["findings"]
        if "generation_pair_missing" in finding["reason_codes"]
    }
    if orphan_ids != recorded_orphans:
        raise IntakeError("source audit orphan generation findings differ from source records")
    audit_cohorts = {cohort["cohort_id"]: cohort for cohort in audit["cohorts"]}
    if set(audit_cohorts) != {cohort["cohort_id"] for cohort in index["cohorts"]}:
        raise IntakeError("source audit cohorts differ from archive index")
    for cohort in index["cohorts"]:
        actual = audit_cohorts[cohort["cohort_id"]]
        for key in ("setting", "historical_n_reference", "id_namespace"):
            if actual[key] != cohort[key]:
                raise IntakeError(f"source audit cohort metadata mismatch: {key}")
        expected = actual["expected_source_pair_keys"]
        expected_ids = (
            None
            if expected is None
            else {
                identity_hash(
                    "rebuttal-pair/v2", aliases.resolve_pair(cohort["id_namespace"], key).canonical
                )
                for key in expected
            }
        )
        claimed = {pair["pair_id"] for pair in pairs if cohort["cohort_id"] in pair["cohort_ids"]}
        recovered = claimed if expected_ids is None else source_pairs & expected_ids
        extras = set() if expected_ids is None else claimed - expected_ids
        missing = None if expected_ids is None else expected_ids - source_pairs
        expected_count = None if expected_ids is None else len(expected_ids)
        missing_count = None if missing is None else len(missing)
        coverage = (
            "unknown" if expected_ids is None else "partial" if missing or extras else "complete"
        )
        checks = {
            "recovered_count": len(recovered),
            "recovered_pair_ids": sorted(recovered),
            "expected_count": expected_count,
            "extra_count": len(extras),
            "extra_pair_ids": sorted(extras),
            "missing_count": missing_count,
            "coverage_status": coverage,
            "historical_count_shortfall": max(0, cohort["historical_n_reference"] - len(recovered)),
        }
        for key, value in checks.items():
            if actual[key] != value:
                raise IntakeError(f"source audit cohort coverage mismatch: {key}")
        for pair in pairs:
            memberships = {entry["cohort_id"]: entry for entry in pair["cohort_membership"]}
            if pair["pair_id"] not in recovered | extras:
                if cohort["cohort_id"] in memberships:
                    raise IntakeError("pair declares unsupported cohort membership")
                continue
            membership = memberships.get(cohort["cohort_id"])
            expected_membership = (
                "unknown"
                if expected_ids is None
                else "member"
                if pair["pair_id"] in recovered
                else "extra"
            )
            if membership is None or membership["membership"] != expected_membership:
                raise IntakeError("pair cohort membership contradicts source coverage")
    declared = set(audit_cohorts)
    for pair in pairs:
        if (
            set(pair["cohort_ids"]) - declared
            or {entry["cohort_id"] for entry in pair["cohort_membership"]} - declared
        ):
            raise IntakeError("pair references an undeclared cohort")


def load_manifest(path: str | Path) -> ManifestBundle:
    """Load a manifest through its mandatory metadata, including the empty case."""
    path = Path(path).resolve()
    metadata_path = path.with_name("pair_manifest.meta.json")
    snapshots = ArtifactSnapshots()
    metadata = snapshots.object(metadata_path)
    require_keys(
        metadata,
        {
            "schema_version",
            "manifest_ref",
            "protocol_ref",
            "protocol_sha256",
            "source_audit_ref",
            "inventory_ref",
        },
        context="manifest metadata",
    )
    require_enum(
        metadata["schema_version"],
        {"rebuttal-pair-manifest-metadata/v2"},
        "manifest metadata schema_version",
    )
    bound_manifest, _ = snapshots.reference(metadata["manifest_ref"], metadata_path)
    if bound_manifest != path:
        raise IntakeError("manifest metadata refers to a different manifest path")
    protocol_path, _ = snapshots.reference(metadata["protocol_ref"], metadata_path)
    protocol = snapshots.object(protocol_path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or protocol.get("protocol_revision") != PROTOCOL_REVISION
        or canonical_sha256(protocol) != PROTOCOL_SHA256
        or metadata["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise IntakeError("manifest does not bind the frozen protocol revision")
    audit_path, _ = snapshots.reference(metadata["source_audit_ref"], metadata_path)
    inventory_path, _ = snapshots.reference(metadata["inventory_ref"], metadata_path)
    audit = snapshots.object(audit_path)
    inventory = snapshots.jsonl(inventory_path)
    _audit(audit, audit_path, snapshots)
    index, aliases, available = _index_and_inventory(
        audit, audit_path, inventory, inventory_path, snapshots
    )
    pairs = snapshots.jsonl(path)
    ids = set()
    slots: list[GenerationSlot] = []
    generation_bindings: dict[str, str] = {}
    generation_files: set[Path] = set()
    for pair in pairs:
        _pair(pair, path, snapshots, aliases, available, protocol, index["source_namespace"])
        if pair["pair_id"] in ids:
            raise IntakeError("duplicate pair_id in manifest")
        ids.add(pair["pair_id"])
        seen_slots = set()
        expected = {_slot(row): row for row in pair["expected_generations"] or []}
        for link in _array(pair["archived_generations"], "archived_generations"):
            require_object(link, "generation link")
            require_keys(
                link,
                {"generation_id", "arm", "window", "availability", "record_ref", "reason_codes"},
                context="generation link",
            )
            key = _slot(link)
            if key in seen_slots:
                raise IntakeError("duplicate generation arm/window slot")
            seen_slots.add(key)
            require_enum(
                link["availability"], {"available", "missing"}, "generation link availability"
            )
            require_string_list(link["reason_codes"], "generation link reason_codes", unique=True)
            gid = link["generation_id"]
            if gid is not None:
                require_sha256(gid, "generation_id")
                binding = canonical_json({"pair_id": pair["pair_id"], "slot": key})
                if gid in generation_bindings and generation_bindings[gid] != binding:
                    raise IntakeError("generation_id reused across pair/arm/window slots")
                generation_bindings[gid] = binding
            expected_key = expected.get(key, {}).get("source_generation_key")
            if expected_key is not None and gid != identity_hash(
                "rebuttal-generation/v2",
                {
                    "source_namespace": index["source_namespace"],
                    "source_generation_key": expected_key,
                },
            ):
                raise IntakeError("generation link disagrees with expected source generation key")
            generation = None
            text = None
            generation_path = None
            if link["record_ref"] is None:
                if link["availability"] != "missing":
                    raise IntakeError("available generation link requires RecordRef")
            else:
                generation_path, generation = snapshots.record(
                    link["record_ref"],
                    path,
                    id_field="generation_id",
                    schema="rebuttal-archive-generation/v2",
                )
                generation_files.add(generation_path)
                text = _generation(
                    generation,
                    generation_path,
                    snapshots,
                    aliases,
                    available,
                    index["source_namespace"],
                )
                if (
                    generation["generation_id"] != gid
                    or generation["pair_id"] != pair["pair_id"]
                    or _slot(generation) != key
                    or generation["availability"] != link["availability"]
                    or generation["source_kind"] != pair["source_kind"]
                ):
                    raise IntakeError("generation RecordRef pair/arm/window binding mismatch")
            slots.append(GenerationSlot(pair, link, generation, text, generation_path))
        if not set(expected).issubset(seen_slots):
            raise IntakeError("manifest omits an expected generation slot")
    # Validate every row in discovered generation artifacts, including orphans.
    # With zero links, source_audit still records orphan counts/findings, and the
    # verified source inventory retains raw orphan evidence without invented pairs.
    all_generation_ids = set()
    all_generation_slots = set()
    for generation_path in generation_files:
        for generation in snapshots.jsonl(generation_path):
            _generation(
                generation,
                generation_path,
                snapshots,
                aliases,
                available,
                index["source_namespace"],
            )
            gid = generation["generation_id"]
            key = canonical_json({"pair_id": generation["pair_id"], "slot": _slot(generation)})
            if gid in all_generation_ids or key in all_generation_slots:
                raise IntakeError("duplicate generation identity or pair/arm/window slot")
            all_generation_ids.add(gid)
            all_generation_slots.add(key)
    if audit["recovered_pair_count"] != len(pairs):
        raise IntakeError("source audit recovered_pair_count differs from manifest")
    if generation_files and audit["raw_generation_record_count"] != len(all_generation_ids):
        raise IntakeError("source audit raw generation count differs from linked artifact")
    _source_coverage(pairs, slots, audit, index, aliases, available, snapshots)
    bundle = ManifestBundle(
        path,
        metadata_path,
        metadata,
        protocol,
        protocol_path,
        snapshots.capture(protocol_path),
        audit,
        inventory,
        pairs,
        slots,
        dict(snapshots.snapshots),
    )
    revalidate_inputs(bundle)
    return bundle
