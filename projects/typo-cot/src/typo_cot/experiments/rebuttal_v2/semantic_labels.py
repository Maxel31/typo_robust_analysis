"""Strict, read-only loading of imported human semantic labels."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

from . import artifact_io
from .annotations import _tables, validate_semantic_judgments
from .artifacts import ArtifactSnapshots
from .input_audit import InputAuditBundle
from .schemas import (
    PROTOCOL_REVISION,
    PROTOCOL_SHA256,
    IntakeError,
    canonical_json,
    require_bool,
    require_enum,
    require_int,
    require_keys,
    require_object,
    require_string,
    require_string_list,
    require_timestamp,
    validate_artifact_ref,
)


_RUN_FIELDS = {
    "schema_version",
    "command",
    "mode",
    "experiment_id",
    "experiment_status",
    "started_at",
    "ended_at",
    "invocation",
    "code_identity",
    "protocol_revision",
    "protocol_sha256",
    "input_refs",
    "config_refs",
    "output_refs",
    "run_status",
    "expected_ids",
    "completed_ids",
    "failed_ids",
    "missing_ids",
    "reason_codes",
    "model_execution_performed",
    "human_judgments_generated",
    "annotation_stage_status",
}
_INVOCATION_FIELDS = {
    "audit",
    "output_dir",
    "annotation_batch",
    "labels",
    "annotation_batch_ref",
    "returned_labels_ref",
    "completed_count",
    "adjudicated_count",
}
_OUTPUT_NAMES = {
    "semantic_labels.jsonl",
    "annotation_agreement.json",
    "annotation_coverage.json",
    "cohort_flow.csv",
    "semantic_strata_table.csv",
    "protocol.json",
}


@dataclass(frozen=True)
class SemanticLabelsBundle:
    path: Path
    run_path: Path
    rows: list[dict[str, Any]]
    labels_by_pair_id: dict[str, str]
    snapshots: dict[Path, str]

    @property
    def reference(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.snapshots[self.path]}

    @property
    def run_ref(self) -> dict[str, str]:
        return {"path": str(self.run_path), "sha256": self.snapshots[self.run_path]}


def _absolute_path(value: object, context: str) -> Path:
    raw = require_string(value, context)
    path = Path(raw)
    if not path.is_absolute() or str(path.resolve()) != raw:
        raise IntakeError(f"{context} must be a normalized absolute path")
    return path


def _artifact_refs(value: object, context: str) -> dict[Path, str]:
    if not isinstance(value, list):
        raise IntakeError(f"{context} must be a JSON array")
    result: dict[Path, str] = {}
    for index, item in enumerate(value):
        ref = validate_artifact_ref(item, f"{context}[{index}]")
        path = _absolute_path(ref["path"], f"{context}[{index}].path")
        if path in result:
            raise IntakeError(f"{context} contains duplicate paths")
        result[path] = ref["sha256"]
    return result


def _validate_code_identity(value: object) -> dict[Path, str]:
    identity = require_object(value, "run code_identity")
    require_keys(
        identity,
        {"source_refs", "git_commit", "git_dirty", "reason_codes"},
        context="run code_identity",
    )
    # These refs describe the historical producer. Validate their declarations,
    # but deliberately do not compare them with upgraded source files on disk.
    source_refs = _artifact_refs(identity["source_refs"], "run code_identity.source_refs")
    commit = identity["git_commit"]
    if commit is not None and (
        not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None
    ):
        raise IntakeError("run code_identity.git_commit must be null or a Git object ID")
    if identity["git_dirty"] is not None:
        require_bool(identity["git_dirty"], "run code_identity.git_dirty")
    require_string_list(identity["reason_codes"], "run code_identity.reason_codes", unique=True)
    return source_refs


def _merge_snapshots(target: dict[Path, str], additions: dict[Path, str]) -> None:
    for path, digest in additions.items():
        if path in target and target[path] != digest:
            raise IntakeError(f"conflicting captured artifact digest at {path}")
        target[path] = digest


def load_semantic_labels(path: str | Path, audit_bundle: InputAuditBundle) -> SemanticLabelsBundle:
    """Load labels only after reconstructing their complete annotation provenance."""
    path = Path(path).resolve()
    run_path = path.with_name("run.json")
    observed = ArtifactSnapshots()
    run = observed.object(run_path)
    require_keys(run, _RUN_FIELDS, context="semantic-label run")
    require_enum(run["schema_version"], {"rebuttal-run/v2"}, "run schema_version")
    require_enum(run["command"], {"annotation-import"}, "run command")
    require_enum(run["mode"], {"cpu-audit"}, "run mode")
    require_enum(run["experiment_id"], {"R2"}, "run experiment_id")
    require_enum(run["experiment_status"], {"partial"}, "run experiment_status")
    require_timestamp(run["started_at"], "run started_at")
    require_timestamp(run["ended_at"], "run ended_at")
    require_enum(run["run_status"], {"complete"}, "run status")
    require_enum(run["annotation_stage_status"], {"complete"}, "annotation stage status")
    if run["protocol_revision"] != PROTOCOL_REVISION or run["protocol_sha256"] != PROTOCOL_SHA256:
        raise IntakeError("semantic-label run protocol binding mismatch")
    if require_bool(run["model_execution_performed"], "model_execution_performed"):
        raise IntakeError("annotation import must not claim model execution")
    if require_bool(run["human_judgments_generated"], "human_judgments_generated"):
        raise IntakeError("annotation import must not claim generated human judgments")
    code_refs = _validate_code_identity(run["code_identity"])

    invocation = require_object(run["invocation"], "annotation-import invocation")
    require_keys(invocation, _INVOCATION_FIELDS, context="annotation-import invocation")
    if _absolute_path(invocation["audit"], "invocation.audit") != audit_bundle.path:
        raise IntakeError("semantic-label run binds a different input audit")
    if _absolute_path(invocation["output_dir"], "invocation.output_dir") != path.parent:
        raise IntakeError("semantic-label run output directory mismatch")
    batch_path = _absolute_path(invocation["annotation_batch"], "invocation.annotation_batch")
    labels_path = _absolute_path(invocation["labels"], "invocation.labels")
    bound_batch, _ = observed.reference(invocation["annotation_batch_ref"], run_path)
    bound_labels, _ = observed.reference(invocation["returned_labels_ref"], run_path)
    if bound_batch != batch_path or bound_labels != labels_path:
        raise IntakeError("annotation-import invocation path/reference mismatch")
    completed_count = require_int(invocation["completed_count"], "completed_count", minimum=0)
    adjudicated_count = require_int(invocation["adjudicated_count"], "adjudicated_count", minimum=0)

    output_refs = require_object(run["output_refs"], "run output_refs")
    require_keys(output_refs, _OUTPUT_NAMES, context="run output_refs")
    output_paths: dict[str, Path] = {}
    for name in sorted(_OUTPUT_NAMES):
        ref = validate_artifact_ref(output_refs[name], f"run output_refs[{name!r}]")
        if ref["path"] != name:
            raise IntakeError(f"run output reference path mismatch for {name}")
        output_path, _ = observed.reference(ref, run_path)
        if output_path != run_path.parent / name:
            raise IntakeError(f"run output reference escapes its artifact directory: {name}")
        output_paths[name] = output_path
    if output_paths["semantic_labels.jsonl"] != path:
        raise IntakeError("run output_refs does not bind the requested semantic-label path")

    expected, agreement, coverage, source_snapshots = validate_semantic_judgments(
        audit_bundle, batch_path, labels_path
    )
    actual = observed.jsonl(path)
    if canonical_json(actual) != canonical_json(expected):
        raise IntakeError("semantic labels differ from verified annotation judgments")
    if canonical_json(observed.object(output_paths["annotation_agreement.json"])) != canonical_json(
        agreement
    ):
        raise IntakeError("annotation agreement differs from verified judgments")
    if canonical_json(observed.object(output_paths["annotation_coverage.json"])) != canonical_json(
        coverage
    ):
        raise IntakeError("annotation coverage differs from verified input")
    flow, strata = _tables(audit_bundle, expected)
    if observed.capture(output_paths["cohort_flow.csv"]) != flow:
        raise IntakeError("cohort flow differs from verified semantic labels")
    if observed.capture(output_paths["semantic_strata_table.csv"]) != strata:
        raise IntakeError("semantic strata table differs from verified semantic labels")
    if observed.capture(output_paths["protocol.json"]) != audit_bundle.manifest.protocol_bytes:
        raise IntakeError("annotation import protocol artifact differs from the bound protocol")

    pair_ids = [row["pair_id"] for row in expected]
    if completed_count != len(expected) or adjudicated_count != agreement["adjudicated_count"]:
        raise IntakeError("annotation-import invocation counts differ from verified judgments")
    if require_string_list(run["expected_ids"], "run expected_ids", unique=True) != pair_ids:
        raise IntakeError("semantic-label run expected ID inventory mismatch")
    if require_string_list(run["completed_ids"], "run completed_ids", unique=True) != pair_ids:
        raise IntakeError("semantic-label run completed ID inventory mismatch")
    if require_string_list(run["failed_ids"], "run failed_ids", unique=True):
        raise IntakeError("completed annotation import must not contain failed IDs")
    if require_string_list(run["missing_ids"], "run missing_ids", unique=True):
        raise IntakeError("completed annotation import must not contain missing IDs")
    expected_reasons = (
        ["unavailable_or_unverified_pairs_not_annotated"]
        if any(row["annotation_status"] != "completed" for row in coverage["pair_coverage"])
        else []
    )
    if (
        require_string_list(run["reason_codes"], "run reason_codes", unique=True)
        != expected_reasons
    ):
        raise IntakeError("semantic-label run reason codes differ from verified coverage")
    if run["config_refs"] != []:
        raise IntakeError("annotation import must not declare configuration references")

    captured_inputs = _artifact_refs(run["input_refs"], "run input_refs")
    for code_path, digest in code_refs.items():
        if captured_inputs.get(code_path) != digest:
            raise IntakeError(f"semantic-label run code/input binding mismatch at {code_path}")
    for source_path, digest in source_snapshots.snapshots.items():
        if captured_inputs.get(source_path) != digest:
            raise IntakeError(f"semantic-label run input binding mismatch at {source_path}")

    snapshots = dict(observed.snapshots)
    _merge_snapshots(snapshots, source_snapshots.snapshots)
    artifact_io.revalidate_snapshots(snapshots)
    return SemanticLabelsBundle(
        path=path,
        run_path=run_path,
        rows=actual,
        labels_by_pair_id={row["pair_id"]: row["label"] for row in actual},
        snapshots=snapshots,
    )
