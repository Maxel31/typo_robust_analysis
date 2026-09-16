"""Atomic publication and strict reconstruction of rebuttal-v2 global plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import artifact_io
from .artifacts import ManifestBundle, load_manifest
from .coordinates import assign_shards
from .donor_artifacts import DonorBankBundle, load_donor_bank
from .identity import identity_hash
from .input_audit import InputAuditBundle, load_input_audit
from .planning import GENERATION_ID_DOMAIN, GlobalPlan, build_run_plan
from .runtime_lock import RuntimeLockBundle, load_runtime_lock
from .schemas import (
    PROTOCOL_REVISION,
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_SHA256,
    IntakeError,
    canonical_json,
    canonical_sha256,
    require_enum,
    require_int,
    require_keys,
    require_object,
    require_sha256,
    require_string_list,
    require_timestamp,
    strict_loads,
    validate_artifact_ref,
)
from .semantic_labels import SemanticLabelsBundle, load_semantic_labels

PLAN_METADATA_SCHEMA = "rebuttal-plan-metadata/v2"
EXPECTED_GENERATION_SCHEMA = "rebuttal-expected-generation/v2"
_METADATA_FIELDS = {
    "schema_version",
    "freeze_timestamp",
    "plan_ref",
    "protocol_ref",
    "protocol_sha256",
    "manifest_ref",
    "manifest_metadata_ref",
    "input_audit_ref",
    "input_audit_metadata_ref",
    "semantic_labels_ref",
    "semantic_labels_run_ref",
    "runtime_lock_ref",
    "donor_bank_ref",
    "donor_bank_sha256",
    "experiments",
    "scientific_config_sha256",
    "intended_row_count",
    "row_counts",
    "expected_generation_ids",
    "ineligible_plan_rows",
    "preflight_ref",
    "coverage_ref",
    "donor_assignment_ref",
    "expected_generations_ref",
    "shard_assignment_ref",
}


@dataclass(frozen=True)
class PlanBundle:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    rows: list[dict[str, Any]]
    manifest: ManifestBundle
    input_audit: InputAuditBundle
    labels: SemanticLabelsBundle
    runtime_lock: RuntimeLockBundle
    donor_bank: DonorBankBundle | None
    shard_assignment: dict[str, Any]
    expected_generations: list[dict[str, Any]]
    snapshots: dict[Path, str]

    @property
    def reference(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.snapshots[self.path]}

    @property
    def metadata_ref(self) -> dict[str, str]:
        return {"path": str(self.metadata_path), "sha256": self.snapshots[self.metadata_path]}

    @property
    def generation_rows(self) -> list[dict[str, Any]]:
        return self.expected_generations


def _merge_snapshots(target: dict[Path, str], source: dict[Path, str]) -> None:
    for raw_path, digest in source.items():
        path = Path(raw_path).resolve()
        if path in target and target[path] != digest:
            raise IntakeError(f"conflicting captured artifact digest at {path}")
        target[path] = digest


def _output_ready(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IntakeError(f"Output directory must be absent or empty: {output}")
    return output


def _experiments(value: object) -> list[str]:
    result = require_string_list(value, "experiments", unique=True)
    if not result or any(item not in {"R3", "R4"} for item in result):
        raise IntakeError("experiments must be a nonempty subset of ('R3', 'R4')")
    return sorted(result)


def _shard_count(value: object) -> int:
    result = require_int(value, "num_shards", minimum=1)
    if result > 6:
        raise IntakeError("num_shards must be at most 6")
    return result


def _expected_rows(plan: GlobalPlan) -> list[dict[str, Any]]:
    if plan.scientific_config_sha256 is None:
        return []
    rows = []
    for row in plan.rows:
        if row["eligibility"] != "valid":
            continue
        generation_id = identity_hash(
            GENERATION_ID_DOMAIN,
            {
                "plan_id": row["plan_id"],
                "scientific_config_sha256": plan.scientific_config_sha256,
            },
        )
        rows.append(
            {
                "schema_version": EXPECTED_GENERATION_SCHEMA,
                "generation_id": generation_id,
                "plan_id": row["plan_id"],
                "pair_id": row["pair_id"],
                "setting": row["setting"],
                "arm": row["arm"],
                "window": row["window"],
                "scientific_config_sha256": plan.scientific_config_sha256,
            }
        )
    rows.sort(key=lambda row: row["generation_id"])
    if [row["generation_id"] for row in rows] != plan.expected_generation_ids:
        raise IntakeError("core plan expected generation inventory is inconsistent")
    return rows


def _row_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["setting"], row["arm"], row["eligibility"]) for row in rows)
    return [
        {"setting": setting, "arm": arm, "eligibility": eligibility, "count": count}
        for (setting, arm, eligibility), count in sorted(counts.items())
    ]


def _ineligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "plan_id": row["plan_id"],
                "eligibility": row["eligibility"],
                "reason_codes": row["reason_codes"],
            }
            for row in rows
            if row["eligibility"] != "valid"
        ],
        key=lambda row: row["plan_id"],
    )


def _load_inputs(
    manifest_path: str | Path,
    input_audit_path: str | Path,
    labels_path: str | Path,
    runtime_lock_path: str | Path,
    protocol_path: str | Path,
    experiments: list[str],
    donor_bank_path: str | Path | None,
) -> tuple[
    dict[str, Any],
    bytes,
    ManifestBundle,
    InputAuditBundle,
    SemanticLabelsBundle,
    RuntimeLockBundle,
    DonorBankBundle | None,
    dict[Path, str],
]:
    protocol_path = Path(protocol_path).resolve()
    snapshots: dict[Path, str] = {}
    protocol, protocol_bytes = _captured_protocol(protocol_path, snapshots)
    manifest = load_manifest(Path(manifest_path).resolve())
    audit = load_input_audit(Path(input_audit_path).resolve())
    if (
        audit.manifest.path != manifest.path
        or audit.manifest.metadata_path != manifest.metadata_path
        or canonical_json(audit.manifest.manifest_ref) != canonical_json(manifest.manifest_ref)
        or canonical_json(audit.manifest.metadata_ref) != canonical_json(manifest.metadata_ref)
    ):
        raise IntakeError("input audit does not bind the explicitly supplied manifest")
    if canonical_json(protocol) != canonical_json(manifest.protocol):
        raise IntakeError("explicit protocol differs from verified manifest protocol")
    labels = load_semantic_labels(Path(labels_path).resolve(), audit)
    runtime = load_runtime_lock(Path(runtime_lock_path).resolve(), protocol, experiments)
    donor = (
        None
        if donor_bank_path is None
        else load_donor_bank(Path(donor_bank_path).resolve(), protocol)
    )
    for bundle in (manifest, audit, labels, runtime, donor):
        if bundle is not None:
            _merge_snapshots(snapshots, bundle.snapshots)
    return protocol, protocol_bytes, manifest, audit, labels, runtime, donor, snapshots


def _code_identity(snapshots: dict[Path, str]) -> dict[str, Any]:
    base = Path(__file__).resolve().parent
    names = (
        "plan_artifacts.py",
        "planning.py",
        "coordinates.py",
        "donors.py",
        "donor_artifacts.py",
        "runtime_lock.py",
        "cli.py",
        "annotations.py",
        "semantic_labels.py",
        "input_audit.py",
        "input_spans.py",
        "input_tokenization.py",
        "artifacts.py",
        "artifact_io.py",
        "identity.py",
        "schemas.py",
        "source.py",
    )
    return artifact_io.capture_code((base / name for name in names), snapshots)


def _captured_protocol(path: Path, snapshots: dict[Path, str]) -> tuple[dict[str, Any], bytes]:
    try:
        raw = artifact_io.capture_file(path, snapshots)
    except (OSError, ValueError) as exc:
        raise IntakeError(f"cannot read protocol artifact {path}: {exc}") from exc
    try:
        payload = require_object(strict_loads(raw.decode("utf-8"), str(path)), str(path))
    except UnicodeError as exc:
        raise IntakeError(f"protocol is not UTF-8: {path}") from exc
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise IntakeError(f"unsupported protocol schema_version at {path}")
    if payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise IntakeError(f"unsupported protocol_revision at {path}")
    if canonical_sha256(payload) != PROTOCOL_SHA256:
        raise IntakeError(
            f"protocol content does not match frozen revision {PROTOCOL_REVISION}: {path}"
        )
    return payload, raw


def run_plan(
    manifest_path: str | Path,
    input_audit_path: str | Path,
    labels_path: str | Path,
    runtime_lock_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    experiments: list[str],
    donor_bank_path: str | Path | None = None,
    num_shards: int = 1,
) -> dict[str, Any]:
    """Build and atomically freeze a complete global plan or blocker report."""

    output = _output_ready(output_dir)
    selected = _experiments(experiments)
    shard_count = _shard_count(num_shards)
    started = artifact_io.utc_now()
    snapshots: dict[Path, str] = {}
    code_identity = _code_identity(snapshots)
    (
        protocol,
        protocol_bytes,
        manifest,
        audit,
        labels,
        runtime,
        donor,
        input_snapshots,
    ) = _load_inputs(
        manifest_path,
        input_audit_path,
        labels_path,
        runtime_lock_path,
        protocol_path,
        selected,
        donor_bank_path,
    )
    _merge_snapshots(snapshots, input_snapshots)
    plan = build_run_plan(
        manifest,
        labels,
        audit,
        protocol,
        runtime_lock=runtime,
        experiments=selected,
        donor_bank=donor,
    )
    invocation = {
        "manifest": str(Path(manifest_path).resolve()),
        "input_audit": str(Path(input_audit_path).resolve()),
        "labels": str(Path(labels_path).resolve()),
        "runtime_lock": str(Path(runtime_lock_path).resolve()),
        "protocol": str(Path(protocol_path).resolve()),
        "donor_bank": None if donor_bank_path is None else str(Path(donor_bank_path).resolve()),
        "experiments": selected,
        "num_shards": shard_count,
        "output_dir": str(output),
    }
    preflight_bytes = artifact_io.json_bytes(plan.preflight)
    coverage_bytes = artifact_io.json_bytes(plan.coverage)
    files = {
        "preflight.json": preflight_bytes,
        "planning_coverage.json": coverage_bytes,
        "protocol.json": protocol_bytes,
    }
    blocked = plan.preflight["status"] == "blocked"
    if not blocked:
        if plan.donor_assignment is None or plan.scientific_config_sha256 is None:
            raise IntakeError("passed core plan is missing frozen scientific outputs")
        expected = _expected_rows(plan)
        shards = assign_shards(plan.rows, shard_count)
        plan_bytes = artifact_io.jsonl_bytes(plan.rows)
        donor_bytes = artifact_io.json_bytes(plan.donor_assignment)
        expected_bytes = artifact_io.jsonl_bytes(expected)
        shard_bytes = artifact_io.json_bytes(shards)
        files.update(
            {
                "plan.jsonl": plan_bytes,
                "donor_assignment.json": donor_bytes,
                "expected_generations.jsonl": expected_bytes,
                "shard_assignment.json": shard_bytes,
            }
        )
        metadata = {
            "schema_version": PLAN_METADATA_SCHEMA,
            "freeze_timestamp": artifact_io.utc_now(),
            "plan_ref": artifact_io.artifact_ref("plan.jsonl", plan_bytes),
            "protocol_ref": artifact_io.artifact_ref("protocol.json", protocol_bytes),
            "protocol_sha256": canonical_sha256(protocol),
            "manifest_ref": manifest.manifest_ref,
            "manifest_metadata_ref": manifest.metadata_ref,
            "input_audit_ref": audit.audit_ref,
            "input_audit_metadata_ref": audit.metadata_ref,
            "semantic_labels_ref": labels.reference,
            "semantic_labels_run_ref": labels.run_ref,
            "runtime_lock_ref": runtime.reference,
            "donor_bank_ref": None if donor is None else donor.reference,
            "donor_bank_sha256": None if donor is None else donor.reference["sha256"],
            "experiments": selected,
            "scientific_config_sha256": plan.scientific_config_sha256,
            "intended_row_count": plan.coverage["counts"]["intended_plan_row_count"],
            "row_counts": _row_counts(plan.rows),
            "expected_generation_ids": plan.expected_generation_ids,
            "ineligible_plan_rows": _ineligible(plan.rows),
            "preflight_ref": artifact_io.artifact_ref("preflight.json", preflight_bytes),
            "coverage_ref": artifact_io.artifact_ref("planning_coverage.json", coverage_bytes),
            "donor_assignment_ref": artifact_io.artifact_ref("donor_assignment.json", donor_bytes),
            "expected_generations_ref": artifact_io.artifact_ref(
                "expected_generations.jsonl", expected_bytes
            ),
            "shard_assignment_ref": artifact_io.artifact_ref("shard_assignment.json", shard_bytes),
        }
        files["plan.meta.json"] = artifact_io.json_bytes(metadata)

    plan_ids = [row["plan_id"] for row in plan.rows]
    reasons = [reason for row in plan.preflight["blockers"] for reason in row["reason_codes"]]
    run = artifact_io.make_run(
        command="plan",
        started_at=started,
        invocation=invocation,
        snapshots=snapshots,
        code_identity=code_identity,
        protocol=protocol,
        files=files,
        expected_ids=plan_ids,
        completed_ids=plan_ids,
        missing_ids=[],
        reason_codes=reasons,
        config_refs=[runtime.reference] + ([] if donor is None else [donor.reference]),
        experiment_status="not_run",
    )
    run.update(
        {
            "mode": "cpu-planning",
            "experiment_id": None,
            "experiment_ids": selected,
            "planning_status": "blocked" if blocked else "complete",
            "run_status": "failed" if blocked else "complete",
        }
    )
    files["run.json"] = artifact_io.json_bytes(run)
    artifact_io.publish_artifacts(output, files, snapshots)
    return {
        "status": "blocked" if blocked else "complete",
        "model_execution_performed": False,
        "output_dir": str(output),
        "plan_count": len(plan.rows),
        "expected_generation_count": len(plan.expected_generation_ids),
        "blocking_pair_ids": plan.preflight["blocking_pair_ids"],
        "preflight_path": str(output / "preflight.json"),
        "plan_path": None if blocked else str(output / "plan.jsonl"),
    }


def _resolve_named_ref(
    metadata: dict[str, Any],
    field: str,
    name: str,
    owner: Path,
    snapshots: dict[Path, str],
) -> Path:
    ref = validate_artifact_ref(metadata[field], f"plan metadata.{field}")
    if ref["path"] != name:
        raise IntakeError(f"plan metadata.{field} must reference {name}")
    path = artifact_io.resolve_artifact(ref, owner, snapshots)
    if path != owner.parent / name:
        raise IntakeError(f"plan metadata.{field} escapes the plan directory")
    return path


def _upstream_path(
    metadata: dict[str, Any], field: str, owner: Path, snapshots: dict[Path, str]
) -> Path:
    return artifact_io.resolve_artifact(metadata[field], owner, snapshots)


def load_plan(path: str | Path) -> PlanBundle:
    """Reconstruct a frozen global plan and reject any changed inventory."""

    path = Path(path).resolve()
    metadata_path = path.with_name("plan.meta.json")
    snapshots: dict[Path, str] = {}
    metadata = artifact_io.read_json(metadata_path, snapshots)
    require_keys(metadata, _METADATA_FIELDS, context="plan metadata")
    require_enum(metadata["schema_version"], {PLAN_METADATA_SCHEMA}, "plan metadata schema")
    require_timestamp(metadata["freeze_timestamp"], "plan metadata freeze_timestamp")
    experiments = _experiments(metadata["experiments"])
    if metadata["experiments"] != experiments:
        raise IntakeError("plan metadata experiments must be sorted")
    require_sha256(metadata["protocol_sha256"], "plan metadata protocol_sha256")
    scientific_hash = require_sha256(
        metadata["scientific_config_sha256"], "plan metadata scientific_config_sha256"
    )
    plan_path = _resolve_named_ref(metadata, "plan_ref", "plan.jsonl", metadata_path, snapshots)
    if plan_path != path:
        raise IntakeError("plan metadata binds a different plan path")
    protocol_path = _resolve_named_ref(
        metadata, "protocol_ref", "protocol.json", metadata_path, snapshots
    )
    preflight_path = _resolve_named_ref(
        metadata, "preflight_ref", "preflight.json", metadata_path, snapshots
    )
    coverage_path = _resolve_named_ref(
        metadata, "coverage_ref", "planning_coverage.json", metadata_path, snapshots
    )
    donor_assignment_path = _resolve_named_ref(
        metadata,
        "donor_assignment_ref",
        "donor_assignment.json",
        metadata_path,
        snapshots,
    )
    expected_path = _resolve_named_ref(
        metadata,
        "expected_generations_ref",
        "expected_generations.jsonl",
        metadata_path,
        snapshots,
    )
    shard_path = _resolve_named_ref(
        metadata,
        "shard_assignment_ref",
        "shard_assignment.json",
        metadata_path,
        snapshots,
    )
    protocol, _ = _captured_protocol(protocol_path, snapshots)
    if canonical_sha256(protocol) != metadata["protocol_sha256"]:
        raise IntakeError("plan metadata protocol hash mismatch")

    manifest_path = _upstream_path(metadata, "manifest_ref", metadata_path, snapshots)
    manifest_metadata_path = _upstream_path(
        metadata, "manifest_metadata_ref", metadata_path, snapshots
    )
    manifest = load_manifest(manifest_path)
    if manifest.metadata_path != manifest_metadata_path:
        raise IntakeError("plan metadata binds a different manifest metadata artifact")
    audit_path = _upstream_path(metadata, "input_audit_ref", metadata_path, snapshots)
    audit_metadata_path = _upstream_path(
        metadata, "input_audit_metadata_ref", metadata_path, snapshots
    )
    audit = load_input_audit(audit_path)
    if audit.metadata_path != audit_metadata_path or audit.manifest.path != manifest.path:
        raise IntakeError("plan metadata input-audit/manifest binding mismatch")
    labels_path = _upstream_path(metadata, "semantic_labels_ref", metadata_path, snapshots)
    labels_run_path = _upstream_path(metadata, "semantic_labels_run_ref", metadata_path, snapshots)
    labels = load_semantic_labels(labels_path, audit)
    if labels.run_path != labels_run_path:
        raise IntakeError("plan metadata binds a different semantic-label run")
    runtime_path = _upstream_path(metadata, "runtime_lock_ref", metadata_path, snapshots)
    runtime = load_runtime_lock(runtime_path, protocol, experiments)

    donor_ref = metadata["donor_bank_ref"]
    donor_hash = metadata["donor_bank_sha256"]
    if donor_ref is None:
        if donor_hash is not None:
            raise IntakeError("null donor bank reference requires null donor bank hash")
        donor = None
    else:
        ref = validate_artifact_ref(donor_ref, "plan metadata.donor_bank_ref")
        if donor_hash != ref["sha256"]:
            raise IntakeError("donor bank reference/hash mismatch")
        donor_path = artifact_io.resolve_artifact(ref, metadata_path, snapshots)
        donor = load_donor_bank(donor_path, protocol)

    for bundle in (manifest, audit, labels, runtime, donor):
        if bundle is not None:
            _merge_snapshots(snapshots, bundle.snapshots)
    rows = artifact_io.read_jsonl(plan_path, snapshots)
    preflight = artifact_io.read_json(preflight_path, snapshots)
    coverage = artifact_io.read_json(coverage_path, snapshots)
    donor_assignment = artifact_io.read_json(donor_assignment_path, snapshots)
    expected = artifact_io.read_jsonl(expected_path, snapshots)
    shards = artifact_io.read_json(shard_path, snapshots)
    shard_count = _shard_count(shards.get("num_shards"))

    rebuilt = build_run_plan(
        manifest,
        labels,
        audit,
        protocol,
        runtime_lock=runtime,
        experiments=experiments,
        donor_bank=donor,
    )
    if rebuilt.preflight["status"] != "passed":
        raise IntakeError("frozen runnable plan now fails global preflight reconstruction")
    rebuilt_expected = _expected_rows(rebuilt)
    rebuilt_shards = assign_shards(rebuilt.rows, shard_count)
    comparisons = (
        (rows, rebuilt.rows, "plan rows"),
        (preflight, rebuilt.preflight, "plan preflight"),
        (coverage, rebuilt.coverage, "planning coverage"),
        (donor_assignment, rebuilt.donor_assignment, "donor assignment"),
        (expected, rebuilt_expected, "expected generation inventory"),
        (shards, rebuilt_shards, "shard assignment"),
    )
    for actual, wanted, context in comparisons:
        if canonical_json(actual) != canonical_json(wanted):
            raise IntakeError(f"{context} differs from reconstructed global plan")

    expected_ids = require_string_list(
        metadata["expected_generation_ids"],
        "plan metadata expected_generation_ids",
        unique=True,
    )
    if expected_ids != rebuilt.expected_generation_ids:
        raise IntakeError("plan metadata expected generation IDs differ from reconstruction")
    intended = require_int(metadata["intended_row_count"], "intended_row_count", minimum=0)
    if intended != rebuilt.coverage["counts"]["intended_plan_row_count"]:
        raise IntakeError("plan metadata intended row count differs from reconstruction")
    if canonical_json(metadata["row_counts"]) != canonical_json(_row_counts(rebuilt.rows)):
        raise IntakeError("plan metadata row counts differ from reconstruction")
    if canonical_json(metadata["ineligible_plan_rows"]) != canonical_json(
        _ineligible(rebuilt.rows)
    ):
        raise IntakeError("plan metadata ineligible inventory differs from reconstruction")
    if scientific_hash != rebuilt.scientific_config_sha256:
        raise IntakeError("plan metadata scientific hash differs from reconstruction")
    if donor_hash != (None if donor is None else donor.reference["sha256"]):
        raise IntakeError("plan metadata donor bank hash differs from verified input")
    artifact_io.revalidate_snapshots(snapshots)
    return PlanBundle(
        path,
        metadata_path,
        metadata,
        rows,
        manifest,
        audit,
        labels,
        runtime,
        donor,
        shards,
        expected,
        snapshots,
    )
