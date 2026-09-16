"""Strict, read-only validation of independently audited rebuttal donor banks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import artifact_io
from .artifacts import ArtifactSnapshots
from .input_audit import InputAuditBundle, load_input_audit
from .schemas import (
    PROTOCOL_SHA256,
    IntakeError,
    canonical_json,
    canonical_sha256,
    require_enum,
    require_int,
    require_keys,
    require_object,
    require_sha256,
    require_string,
    validate_artifact_ref,
    validate_record_ref,
)

BANK_SCHEMA_VERSION = "rebuttal-donor-bank/v2"

_ROW_FIELDS = {
    "schema_version",
    "pair_id",
    "original_problem_group_id",
    "task",
    "model_id",
    "target_rule",
    "clean_prompt_ref",
    "clean_prompt_sha256",
    "aligned_word_count",
    "clean_endpoints",
    "tokenizer_lock_ref",
    "input_audit_ref",
    "provenance",
}


@dataclass(frozen=True)
class DonorBankBundle:
    path: Path
    rows: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    audit_rows_by_pair_id: dict[str, dict[str, Any]]
    tokenizer_descriptors_by_pair_id: dict[str, dict[str, Any]]
    snapshots: dict[Path, str]

    @property
    def reference(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.snapshots[self.path]}


def _normalize_row(value: object, index: int) -> dict[str, Any]:
    context = f"donor bank row {index}"
    row = require_object(value, context)
    require_keys(row, _ROW_FIELDS, context=context)
    require_enum(row["schema_version"], {BANK_SCHEMA_VERSION}, f"{context}.schema_version")

    endpoints_value = row["clean_endpoints"]
    if not isinstance(endpoints_value, list):
        raise IntakeError(f"{context}.clean_endpoints must be a JSON array")
    endpoints = [
        require_int(endpoint, f"{context}.clean_endpoints[{position}]", minimum=0)
        for position, endpoint in enumerate(endpoints_value)
    ]
    if any(left >= right for left, right in zip(endpoints, endpoints[1:], strict=False)):
        raise IntakeError(f"{context}.clean_endpoints must be strictly increasing")
    count = require_int(row["aligned_word_count"], f"{context}.aligned_word_count", minimum=1)
    if len(endpoints) != count:
        raise IntakeError(f"{context}.clean_endpoints count does not match aligned_word_count")

    provenance = require_object(row["provenance"], f"{context}.provenance")
    require_keys(
        provenance,
        {"manifest_ref", "source_ref"},
        context=f"{context}.provenance",
    )
    return {
        "schema_version": BANK_SCHEMA_VERSION,
        "pair_id": require_string(row["pair_id"], f"{context}.pair_id"),
        "original_problem_group_id": require_string(
            row["original_problem_group_id"], f"{context}.original_problem_group_id"
        ),
        "task": require_string(row["task"], f"{context}.task"),
        "model_id": require_string(row["model_id"], f"{context}.model_id"),
        "target_rule": require_string(row["target_rule"], f"{context}.target_rule"),
        "clean_prompt_ref": validate_artifact_ref(
            row["clean_prompt_ref"], f"{context}.clean_prompt_ref"
        ),
        "clean_prompt_sha256": require_sha256(
            row["clean_prompt_sha256"], f"{context}.clean_prompt_sha256"
        ),
        "aligned_word_count": count,
        "clean_endpoints": endpoints,
        "tokenizer_lock_ref": validate_artifact_ref(
            row["tokenizer_lock_ref"], f"{context}.tokenizer_lock_ref"
        ),
        "input_audit_ref": validate_record_ref(
            row["input_audit_ref"], f"{context}.input_audit_ref"
        ),
        "provenance": {
            "manifest_ref": validate_record_ref(
                provenance["manifest_ref"], f"{context}.provenance.manifest_ref"
            ),
            "source_ref": validate_record_ref(
                provenance["source_ref"], f"{context}.provenance.source_ref"
            ),
        },
    }


def _merge_snapshots(target: dict[Path, str], source: dict[Path, str]) -> None:
    for path, digest in source.items():
        resolved = Path(path).resolve()
        if resolved in target and target[resolved] != digest:
            raise IntakeError(f"scientific input snapshot mismatch at {resolved}")
        target[resolved] = digest


def _decode_prompt(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise IntakeError(f"clean donor prompt is not UTF-8: {path}") from exc


def _index_audit(bundle: InputAuditBundle) -> tuple[dict[str, dict], dict[str, dict]]:
    rows: dict[str, dict] = {}
    for row in bundle.rows:
        audit_id = require_string(row.get("input_audit_id"), "input_audit_id")
        if audit_id in rows:
            raise IntakeError(f"duplicate input_audit_id in {bundle.path}: {audit_id}")
        rows[audit_id] = row
    pairs: dict[str, dict] = {}
    for pair in bundle.pairs:
        pair_id = require_string(pair.get("pair_id"), "input audit pair_id")
        if pair_id in pairs:
            raise IntakeError(f"duplicate pair_id in verified input audit: {pair_id}")
        pairs[pair_id] = pair
    return rows, pairs


def _verify_record_binding(
    provided: dict[str, Any],
    expected: object,
    *,
    containing_file: Path,
    captured: ArtifactSnapshots,
    context: str,
) -> None:
    expected_ref = validate_record_ref(expected, f"verified {context}")
    captured.reference(provided["artifact"], containing_file)
    if (
        provided["record_id"] != expected_ref["record_id"]
        or provided["artifact"]["sha256"] != expected_ref["artifact"]["sha256"]
    ):
        raise IntakeError(f"{context} differs from verified provenance")


def _require_typed_equal(actual: object, expected: object, context: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise IntakeError(f"{context} differs from verified input audit pair")


def load_donor_bank(path: str | Path, protocol: dict) -> DonorBankBundle:
    """Load a bank only through its frozen protocol and verified PR-03 audit chain."""

    protocol = require_object(protocol, "protocol")
    if canonical_sha256(protocol) != PROTOCOL_SHA256:
        raise IntakeError("protocol content does not match frozen revision 2.0.0")

    path = Path(path).resolve()
    captured = ArtifactSnapshots()
    raw_rows = captured.jsonl(path)
    rows = [_normalize_row(row, index) for index, row in enumerate(raw_rows)]
    seen: set[str] = set()
    for row in rows:
        if row["pair_id"] in seen:
            raise IntakeError(f"duplicate donor pair_id: {row['pair_id']}")
        seen.add(row["pair_id"])

    audit_cache: dict[Path, tuple[InputAuditBundle, dict[str, dict], dict[str, dict]]] = {}
    snapshots: dict[Path, str] = {}
    resolved_pairs: list[dict[str, Any]] = []
    audit_rows_by_pair_id: dict[str, dict[str, Any]] = {}
    descriptors_by_pair_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        audit_path, _ = captured.reference(row["input_audit_ref"]["artifact"], path)
        if audit_path not in audit_cache:
            audit_bundle = load_input_audit(audit_path)
            if canonical_json(audit_bundle.protocol) != canonical_json(protocol):
                raise IntakeError("donor input audit protocol differs from requested protocol")
            indexed_rows, indexed_pairs = _index_audit(audit_bundle)
            audit_cache[audit_path] = (audit_bundle, indexed_rows, indexed_pairs)
            _merge_snapshots(snapshots, audit_bundle.snapshots)
        audit_bundle, indexed_rows, indexed_pairs = audit_cache[audit_path]

        audit_id = row["input_audit_ref"]["record_id"]
        if audit_id not in indexed_rows:
            raise IntakeError(f"input_audit_ref record_id {audit_id!r} not found in {audit_path}")
        audit_row = indexed_rows[audit_id]
        pair_id = row["pair_id"]
        if audit_row["pair_id"] != pair_id or pair_id not in indexed_pairs:
            raise IntakeError("input_audit_ref does not identify the declared donor pair")
        pair = indexed_pairs[pair_id]

        for field in (
            "pair_id",
            "original_problem_group_id",
            "task",
            "model_id",
            "target_rule",
        ):
            if field in {"original_problem_group_id", "target_rule"}:
                require_string(pair[field], f"verified donor pair {field}")
            _require_typed_equal(row[field], pair[field], f"donor {field}")

        prompt_path, prompt_raw = captured.reference(row["clean_prompt_ref"], path)
        prompt_digest = hashlib.sha256(prompt_raw).hexdigest()
        if (
            prompt_digest != row["clean_prompt_sha256"]
            or row["clean_prompt_ref"]["sha256"] != row["clean_prompt_sha256"]
        ):
            raise IntakeError("clean donor prompt reference/hash mismatch")
        prompt = _decode_prompt(prompt_raw, prompt_path)
        if pair["clean_prompt"] is None:
            raise IntakeError("verified donor clean prompt is unavailable")
        _require_typed_equal(prompt, pair["clean_prompt"], "clean donor prompt")
        _require_typed_equal(
            row["clean_prompt_sha256"], pair["clean_prompt_sha256"], "clean donor prompt hash"
        )

        alignment = require_object(audit_row.get("alignment"), "verified donor alignment")
        if alignment.get("eligibility") != "valid":
            raise IntakeError("verified donor alignment is not valid")
        aligned_edits = alignment.get("aligned_edits")
        if not isinstance(aligned_edits, list):
            raise IntakeError("verified donor aligned_edits must be a JSON array")
        expected_endpoints = [
            require_int(
                require_object(edit, "verified aligned edit").get("clean_endpoint"),
                "verified clean endpoint",
                minimum=0,
            )
            for edit in aligned_edits
        ]
        _require_typed_equal(row["clean_endpoints"], expected_endpoints, "clean endpoints")
        _require_typed_equal(row["aligned_word_count"], len(aligned_edits), "aligned word count")

        models = require_object(audit_bundle.tokenizer_lock.get("models"), "tokenizer models")
        descriptor = models.get(row["model_id"])
        if descriptor is None:
            raise IntakeError("verified donor tokenizer descriptor is unavailable")
        descriptor = require_object(descriptor, "verified donor tokenizer descriptor")

        provided_lock_path, provided_lock_raw = captured.reference(row["tokenizer_lock_ref"], path)
        audit_lock_ref = validate_artifact_ref(
            audit_bundle.metadata["tokenizer_lock_ref"], "verified tokenizer_lock_ref"
        )
        audit_lock_path = artifact_io.resolve_artifact(
            audit_lock_ref, audit_bundle.metadata_path, snapshots
        )
        audit_lock_raw = artifact_io.capture_file(
            audit_lock_path, snapshots, audit_lock_ref["sha256"]
        )
        if (
            row["tokenizer_lock_ref"]["sha256"] != audit_lock_ref["sha256"]
            or provided_lock_raw != audit_lock_raw
        ):
            raise IntakeError(
                f"tokenizer lock differs from verified input audit: {provided_lock_path}"
            )

        _verify_record_binding(
            row["provenance"]["manifest_ref"],
            audit_row["manifest_ref"],
            containing_file=path,
            captured=captured,
            context="donor manifest_ref",
        )
        _verify_record_binding(
            row["provenance"]["source_ref"],
            pair["source_ref"],
            containing_file=path,
            captured=captured,
            context="donor source_ref",
        )

        resolved_pairs.append(pair)
        audit_rows_by_pair_id[pair_id] = audit_row
        descriptors_by_pair_id[pair_id] = descriptor

    _merge_snapshots(snapshots, captured.snapshots)
    artifact_io.revalidate_snapshots(snapshots)
    return DonorBankBundle(
        path=path,
        rows=rows,
        pairs=resolved_pairs,
        audit_rows_by_pair_id=audit_rows_by_pair_id,
        tokenizer_descriptors_by_pair_id=descriptors_by_pair_id,
        snapshots=snapshots,
    )
