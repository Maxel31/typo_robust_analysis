"""Outcome-independent, two-stage human semantic annotation artifacts.

Reviewer files are explicit allowlists. Private mappings, full prompts, audit
coordinates and outcome metadata never become reviewer fields. Incomplete
returned batches fail closed; missing judgments are not human unassessability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from .artifacts import ArtifactSnapshots
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
    require_string_list,
    require_timestamp,
)


LABELS = frozenset({"unique_preserved", "ambiguous", "task_changed", "unassessable"})
INSTRUCTIONS_VERSION = "semantic-blinding/v2.0.0"
A_SCHEMA = "rebuttal-annotation-stage-a-rating/v2"
B_SCHEMA = "rebuttal-annotation-stage-b-rating/v2"
ADJ_SCHEMA = "rebuttal-annotation-adjudication/v2"
_BATCH_FIELDS = {
    "schema_version",
    "batch_id",
    "stage",
    "created_at",
    "audit_ref",
    "audit_metadata_ref",
    "manifest_ref",
    "protocol_sha256",
    "selection_sha256",
    "required_raters",
    "single_rater_reason",
    "selected_blind_ids",
    "instructions_version",
    "mapping_ref",
    "coverage_ref",
    "export_ref",
    "prior_batch_ref",
    "stage_a_labels_ref",
    "stage_a_lock_ref",
}
_A_FIELDS = {
    "schema_version",
    "blind_id",
    "rater_id",
    "timestamp",
    "interpretations",
    "ambiguity",
    "rationale",
}
_B_FIELDS = {
    "schema_version",
    "blind_id",
    "rater_id",
    "timestamp",
    "stage_a_lock_sha256",
    "label",
    "rationale",
}
_ADJ_FIELDS = {
    "schema_version",
    "blind_id",
    "adjudicator_id",
    "timestamp",
    "stage_a_lock_sha256",
    "disagreement_reason",
    "final_label",
}
_REGION_KINDS = frozenset(
    {
        "stem",
        "option-content",
        "option-label",
        "instructions",
        "context",
        "entity",
    }
)


def _region_layout_known(pair: dict[str, Any]) -> bool:
    layout = pair["prompt_regions"]
    return (
        isinstance(layout, dict)
        and set(layout) == {"clean", "typo"}
        and all(isinstance(layout[side], list) for side in ("clean", "typo"))
        and all(
            isinstance(region, dict)
            and isinstance(region.get("kind"), str)
            and region["kind"] in _REGION_KINDS
            and {"kind", "start", "end"} <= set(region)
            and set(region) <= {"kind", "start", "end", "option_label"}
            for side in ("clean", "typo")
            for region in layout[side]
        )
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonblank(value: object, context: str) -> str:
    result = require_string(value, context)
    if not result.strip():
        raise IntakeError(f"{context} must not be whitespace-only")
    return result


def _raters(value: object, reason: object) -> list[str]:
    raters = require_string_list(value, "required_raters", unique=True)
    if not raters:
        raise IntakeError("at least one required rater is needed")
    for rater in raters:
        _nonblank(rater, "rater_id")
    if len(raters) == 1:
        _nonblank(reason, "single_rater_reason")
    elif reason is not None:
        raise IntakeError("single_rater_reason is only valid with one rater")
    return raters


def _snapshot(bundle: Any) -> ArtifactSnapshots:
    snapshots = ArtifactSnapshots()
    for path, expected in bundle.snapshots.items():
        snapshots.capture(path, expected)
    return snapshots


def _ref(path: Path, snapshots: ArtifactSnapshots) -> dict[str, str]:
    path = path.resolve()
    snapshots.capture(path)
    return {"path": str(path), "sha256": snapshots.snapshots[path]}


def _text(pair: dict[str, Any], name: str, bundle: Any, snapshots: ArtifactSnapshots):
    if pair[name] is not None:
        return pair[name]
    if pair[f"{name}_ref"] is None:
        return None
    _, raw = snapshots.reference(pair[f"{name}_ref"], bundle.manifest.path)
    return raw.decode("utf-8")


def _selection(bundle: Any, snapshots: ArtifactSnapshots):
    """Freeze original members without consulting alignment, labels or outcomes."""
    cohort_ids = {
        row["cohort_id"]
        for row in bundle.manifest.source_audit["cohorts"]
        if row["setting"] in {"R3", "R4"}
    }
    selected, coverage = [], []
    for pair in bundle.manifest.pairs:
        memberships = [m for m in pair["cohort_membership"] if m["cohort_id"] in cohort_ids]
        original = [m["cohort_id"] for m in memberships if m["membership"] == "member"]
        reasons = []
        if not original:
            reasons.append("original_cohort_membership_not_verified")
        typo = _text(pair, "typo_text", bundle, snapshots)
        if typo is None or not typo.strip():
            reasons.append("typo_query_unavailable")
        if not _region_layout_known(pair):
            reasons.append("ancillary_prompt_regions_unknown")
        exported = bool(original) and typo is not None and bool(typo.strip())
        coverage.append(
            {
                "pair_id": pair["pair_id"],
                "cohort_membership": memberships,
                "annotation_status": "selected" if exported else "not_exported",
                "semantic_status": "unknown",
                "reason_codes": reasons,
            }
        )
        if exported:
            selected.append(pair)
    selected.sort(key=lambda row: row["pair_id"])
    coverage.sort(key=lambda row: row["pair_id"])
    missing = {
        row["cohort_id"]: row["missing_source_pair_keys"]
        for row in bundle.manifest.source_audit["cohorts"]
        if row["cohort_id"] in cohort_ids
    }
    return selected, {
        "schema_version": "rebuttal-annotation-coverage/v2",
        "selection_policy": "all_verified_original_R3_R4_members_with_available_typo_query",
        "pair_coverage": coverage,
        "missing_original_source_pair_keys": missing,
    }


def _regions(pair: dict[str, Any], side: str, bundle: Any, snapshots: ArtifactSnapshots):
    """Only producer-declared actual-item regions may add reviewer context."""
    result: dict[str, Any] = {"context": [], "instructions": [], "options": []}
    regions = pair["prompt_regions"]
    if not _region_layout_known(pair):
        return result
    require_keys(regions, {"clean", "typo"}, context="prompt_regions")
    entries = regions[side]
    if not isinstance(entries, list):
        raise IntakeError("prompt regions must be arrays")
    prompt = _text(pair, f"{side}_prompt", bundle, snapshots)
    if prompt is None:
        if entries:
            raise IntakeError("prompt regions require the corresponding exact prompt")
        return result
    for region in entries:
        require_object(region, "prompt region")
        require_keys(region, {"kind", "start", "end"}, {"option_label"}, context="prompt region")
        kind = require_enum(
            region["kind"],
            {
                "stem",
                "option-content",
                "option-label",
                "instructions",
                "context",
                "entity",
            },
            "prompt region kind",
        )
        start = require_int(region["start"], "region start", minimum=0)
        end = require_int(region["end"], "region end", minimum=0)
        if start >= end or end > len(prompt):
            raise IntakeError("prompt region is empty or outside exact prompt")
        label = region.get("option_label")
        if label is not None:
            require_enum(label, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "option_label")
        value = prompt[start:end]
        if kind in {"context", "instructions"}:
            result[kind].append(value)
        elif kind in {"option-content", "option-label"}:
            result["options"].append({"kind": kind, "label": label, "text": value})
    return result


def _stage_a_row(blind_id: str, pair: dict[str, Any], bundle: Any, snapshots: ArtifactSnapshots):
    regions = _regions(pair, "typo", bundle, snapshots)
    return {
        "schema_version": "rebuttal-annotation-stage-a/v2",
        "blind_id": blind_id,
        "stage": "a",
        "typo_query": _text(pair, "typo_text", bundle, snapshots),
        "typo_context": regions["context"],
        "task_instructions": regions["instructions"],
        "presented_options": regions["options"],
    }


def _selection_hash(mapping: list[dict[str, Any]]) -> str:
    return canonical_sha256(sorted(row["pair_id"] for row in mapping))


def _batch(
    bundle: Any, snapshots: ArtifactSnapshots, mapping: list[dict[str, Any]], raters, reason
):
    return {
        "schema_version": "rebuttal-annotation-batch/v2",
        "batch_id": uuid.uuid4().hex,
        "stage": "a",
        "created_at": _now(),
        "audit_ref": _ref(bundle.path, snapshots),
        "audit_metadata_ref": _ref(bundle.metadata_path, snapshots),
        "manifest_ref": bundle.manifest.manifest_ref,
        "protocol_sha256": PROTOCOL_SHA256,
        "selection_sha256": _selection_hash(mapping),
        "required_raters": raters,
        "single_rater_reason": reason,
        "selected_blind_ids": sorted(row["blind_id"] for row in mapping),
        "instructions_version": INSTRUCTIONS_VERSION,
        "mapping_ref": None,
        "coverage_ref": None,
        "export_ref": None,
        "prior_batch_ref": None,
        "stage_a_labels_ref": None,
        "stage_a_lock_ref": None,
    }


def _publish(bundle: Any, snapshots: ArtifactSnapshots, output_dir, files, operation, details):
    from .artifact_io import capture_code, json_bytes, make_run, publish_artifacts

    files["protocol.json"] = bundle.manifest.protocol_bytes
    base = Path(__file__).parent
    code = capture_code(
        [
            base / name
            for name in (
                "annotations.py",
                "artifact_io.py",
                "input_audit.py",
                "input_spans.py",
                "input_tokenization.py",
                "artifacts.py",
                "identity.py",
                "source.py",
                "schemas.py",
                "cli.py",
            )
        ],
        snapshots.snapshots,
    )
    selected, coverage = _selection(bundle, snapshots)
    expected = [pair["pair_id"] for pair in selected]
    run = make_run(
        command=operation,
        started_at=details.pop("started_at"),
        invocation={
            "audit": str(bundle.path),
            "output_dir": str(Path(output_dir).resolve()),
            **details,
        },
        snapshots=snapshots.snapshots,
        code_identity=code,
        protocol=bundle.manifest.protocol,
        files=files,
        expected_ids=expected,
        completed_ids=expected,
        missing_ids=[],
        reason_codes=["unavailable_or_unverified_pairs_not_annotated"]
        if any(row["annotation_status"] != "selected" for row in coverage["pair_coverage"])
        else [],
    )
    run["human_judgments_generated"] = False
    run["annotation_stage_status"] = "complete"
    files["run.json"] = json_bytes(run)
    publish_artifacts(output_dir, files, snapshots.snapshots)
    return run


def export_stage_a(
    audit_path: str | Path,
    output_dir: str | Path,
    raters=("rater-1", "rater-2"),
    single_rater_reason: str | None = None,
) -> dict[str, Any]:
    """Export typo-only forms for every available verified original member."""
    from .artifact_io import artifact_ref, json_bytes, jsonl_bytes
    from .input_audit import load_input_audit

    started_at = _now()
    required_raters = _raters(list(raters), single_rater_reason)
    bundle = load_input_audit(audit_path)
    snapshots = _snapshot(bundle)
    selected, coverage = _selection(bundle, snapshots)
    mapping = [
        {
            "schema_version": "rebuttal-annotation-mapping/v2",
            "blind_id": uuid.uuid4().hex,
            "pair_id": pair["pair_id"],
            "source_ref": pair["source_ref"],
            "manifest_record_ref": {
                "artifact": bundle.manifest.manifest_ref,
                "record_id": pair["pair_id"],
            },
        }
        for pair in selected
    ]
    by_id = {pair["pair_id"]: pair for pair in selected}
    # Random blind IDs also randomize presentation; source order is not a side channel.
    mapping.sort(key=lambda row: row["blind_id"])
    forms = [
        _stage_a_row(row["blind_id"], by_id[row["pair_id"]], bundle, snapshots) for row in mapping
    ]
    files = {
        "annotation_mapping.jsonl": jsonl_bytes(mapping),
        "annotation_coverage.json": json_bytes(coverage),
        "annotation_stage_a.jsonl": jsonl_bytes(forms),
    }
    batch = _batch(bundle, snapshots, mapping, required_raters, single_rater_reason)
    for field, name in (
        ("mapping_ref", "annotation_mapping.jsonl"),
        ("coverage_ref", "annotation_coverage.json"),
        ("export_ref", "annotation_stage_a.jsonl"),
    ):
        batch[field] = artifact_ref(name, files[name])
    files["annotation_batch.json"] = json_bytes(batch)
    return _publish(
        bundle,
        snapshots,
        output_dir,
        files,
        "annotation-export-a",
        {
            "started_at": started_at,
            "selected_count": len(mapping),
            "raters": required_raters,
            "single_rater_reason": single_rater_reason,
            "rating_mode": "single" if len(required_raters) == 1 else "independent_multiple",
        },
    )


def _read_batch(path: Path, bundle: Any, snapshots: ArtifactSnapshots, stage: str):
    path = path.resolve()
    batch = snapshots.object(path)
    require_keys(batch, _BATCH_FIELDS, context="annotation batch")
    require_enum(batch["schema_version"], {"rebuttal-annotation-batch/v2"}, "batch schema")
    require_enum(batch["stage"], {stage}, "batch stage")
    _nonblank(batch["batch_id"], "batch_id")
    require_timestamp(batch["created_at"], "batch created_at")
    if (
        batch["protocol_sha256"] != PROTOCOL_SHA256
        or batch["instructions_version"] != INSTRUCTIONS_VERSION
    ):
        raise IntakeError("annotation batch protocol/instructions mismatch")
    for name, expected in (
        ("audit_ref", bundle.path),
        ("audit_metadata_ref", bundle.metadata_path),
        ("manifest_ref", bundle.manifest.path),
    ):
        resolved, _ = snapshots.reference(batch[name], path)
        if (
            resolved != expected.resolve()
            or snapshots.snapshots[resolved] != bundle.snapshots[resolved]
        ):
            raise IntakeError(f"annotation batch {name} binding mismatch")
    _raters(batch["required_raters"], batch["single_rater_reason"])
    ids = require_string_list(batch["selected_blind_ids"], "selected_blind_ids", unique=True)
    if ids != sorted(ids):
        raise IntakeError("selected blind IDs must be sorted")
    mapping_path, _ = snapshots.reference(batch["mapping_ref"], path)
    mapping = snapshots.jsonl(mapping_path)
    selected, coverage = _selection(bundle, snapshots)
    pairs = {pair["pair_id"]: pair for pair in selected}
    seen = set()
    for row in mapping:
        require_keys(
            row,
            {"schema_version", "blind_id", "pair_id", "source_ref", "manifest_record_ref"},
            context="annotation mapping",
        )
        require_enum(row["schema_version"], {"rebuttal-annotation-mapping/v2"}, "mapping schema")
        blind_id = _nonblank(row["blind_id"], "blind_id")
        try:
            parsed_uuid = uuid.UUID(hex=blind_id)
            if parsed_uuid.hex != blind_id or parsed_uuid.version != 4:
                raise ValueError("noncanonical UUID")
        except (ValueError, AttributeError) as exc:
            raise IntakeError("blind IDs must be random opaque UUID hex values") from exc
        pid = _nonblank(row["pair_id"], "mapping pair_id")
        if pid not in pairs or pid in seen:
            raise IntakeError("mapping pair inventory differs from frozen selection")
        seen.add(pid)
        if canonical_json(row["source_ref"]) != canonical_json(
            pairs[pid]["source_ref"]
        ) or canonical_json(row["manifest_record_ref"]) != canonical_json(
            {
                "artifact": bundle.manifest.manifest_ref,
                "record_id": pid,
            }
        ):
            raise IntakeError("annotation mapping source binding mismatch")
    if seen != set(pairs) or [row["blind_id"] for row in mapping] != ids:
        raise IntakeError("annotation mapping/selected blind ID inventory mismatch")
    if batch["selection_sha256"] != _selection_hash(mapping):
        raise IntakeError("annotation frozen selection hash mismatch")
    coverage_path, _ = snapshots.reference(batch["coverage_ref"], path)
    if canonical_json(snapshots.object(coverage_path)) != canonical_json(coverage):
        raise IntakeError("annotation coverage differs from verified input")
    export_path, _ = snapshots.reference(batch["export_ref"], path)
    if stage == "a":
        if any(
            batch[key] is not None
            for key in ("prior_batch_ref", "stage_a_labels_ref", "stage_a_lock_ref")
        ):
            raise IntakeError("Stage A batch must not claim a prior lock")
        expected = [
            _stage_a_row(row["blind_id"], pairs[row["pair_id"]], bundle, snapshots)
            for row in mapping
        ]
        if canonical_json(snapshots.jsonl(export_path)) != canonical_json(expected):
            raise IntakeError("Stage A export differs from safe allowlisted rendering")
    return batch, mapping, export_path


def _keys(batch):
    return {
        (blind, rater)
        for blind in batch["selected_blind_ids"]
        for rater in batch["required_raters"]
    }


def _stage_a_ratings(path: Path, batch: dict[str, Any], snapshots: ArtifactSnapshots):
    ratings = {}
    expected = _keys(batch)
    for row in snapshots.jsonl(path):
        require_keys(row, _A_FIELDS, context="Stage A returned rating")
        require_enum(row["schema_version"], {A_SCHEMA}, "Stage A rating schema")
        key = (_nonblank(row["blind_id"], "blind_id"), _nonblank(row["rater_id"], "rater_id"))
        if key not in expected or key in ratings:
            raise IntakeError(f"unknown or duplicate Stage A rating key: {key}")
        require_timestamp(row["timestamp"], "Stage A timestamp")
        interpretations = require_string_list(
            row["interpretations"], "interpretations", unique=True
        )
        for interpretation in interpretations:
            _nonblank(interpretation, "interpretation")
        ambiguity = require_enum(
            row["ambiguity"], {"unique", "ambiguous", "unassessable"}, "ambiguity"
        )
        if ambiguity != "unassessable" and not interpretations:
            raise IntakeError("assessable Stage A judgments require an interpretation")
        _nonblank(row["rationale"], "Stage A rationale")
        ratings[key] = row
    missing = sorted(expected - set(ratings))
    if missing:
        raise IntakeError(f"missing Stage A ratings: {missing}")
    return ratings


def _lock(batch, ratings_path, snapshots, mapping_ref):
    return {
        "schema_version": "rebuttal-stage-a-lock/v2",
        "batch_id": batch["batch_id"],
        "locked_at": _now(),
        "completed_keys": [list(key) for key in sorted(_keys(batch))],
        "input_labels_ref": _ref(ratings_path, snapshots),
        "mapping_sha256": mapping_ref["sha256"],
        "selection_sha256": batch["selection_sha256"],
    }


def _stage_b_rows(bundle, snapshots, mapping, ratings, lock_hash, raters):
    pairs = {pair["pair_id"]: pair for pair in bundle.manifest.pairs}
    rows = []
    for item in mapping:
        pair = pairs[item["pair_id"]]
        regions = _regions(pair, "clean", bundle, snapshots)
        for rater in raters:
            own_a = ratings[(item["blind_id"], rater)]
            rows.append(
                {
                    **_stage_a_row(item["blind_id"], pair, bundle, snapshots),
                    "schema_version": "rebuttal-annotation-stage-b/v2",
                    "stage": "b",
                    "rater_id": rater,
                    "stage_a_lock_sha256": lock_hash,
                    "stage_a_response_sha256": canonical_sha256(own_a),
                    "clean_query": _text(pair, "clean_text", bundle, snapshots),
                    "clean_context": regions["context"],
                    "clean_task_instructions": regions["instructions"],
                    "clean_presented_options": regions["options"],
                    "gold": pair["canonical_gold"],
                    "allowed_labels": sorted(LABELS),
                }
            )
    return rows


def export_stage_b(
    audit_path: str | Path,
    stage_a_labels: str | Path,
    annotation_batch: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Lock all independent A judgments before disclosing any clean material."""
    from .artifact_io import artifact_ref, json_bytes, jsonl_bytes
    from .input_audit import load_input_audit

    started_at = _now()
    bundle = load_input_audit(audit_path)
    snapshots = _snapshot(bundle)
    prior_path, labels_path = Path(annotation_batch).resolve(), Path(stage_a_labels).resolve()
    prior, mapping, _ = _read_batch(prior_path, bundle, snapshots, "a")
    ratings = _stage_a_ratings(labels_path, prior, snapshots)
    _, mapping_raw = snapshots.reference(prior["mapping_ref"], prior_path)
    _, coverage_raw = snapshots.reference(prior["coverage_ref"], prior_path)
    lock = _lock(prior, labels_path, snapshots, prior["mapping_ref"])
    lock_raw = json_bytes(lock)
    lock_ref = artifact_ref("stage_a_lock.json", lock_raw)
    rows = _stage_b_rows(
        bundle, snapshots, mapping, ratings, lock_ref["sha256"], prior["required_raters"]
    )
    files = {
        "annotation_mapping.jsonl": mapping_raw,
        "annotation_coverage.json": coverage_raw,
        "annotation_stage_b.jsonl": jsonl_bytes(rows),
        "stage_a_lock.json": lock_raw,
    }
    batch = {
        **prior,
        "batch_id": uuid.uuid4().hex,
        "stage": "b",
        "created_at": _now(),
        "mapping_ref": artifact_ref("annotation_mapping.jsonl", mapping_raw),
        "coverage_ref": artifact_ref("annotation_coverage.json", coverage_raw),
        "export_ref": artifact_ref("annotation_stage_b.jsonl", files["annotation_stage_b.jsonl"]),
        "prior_batch_ref": _ref(prior_path, snapshots),
        "stage_a_labels_ref": _ref(labels_path, snapshots),
        "stage_a_lock_ref": lock_ref,
    }
    files["annotation_batch.json"] = json_bytes(batch)
    return _publish(
        bundle,
        snapshots,
        output_dir,
        files,
        "annotation-export-b",
        {
            "started_at": started_at,
            "selected_count": len(mapping),
            "locked_rating_count": len(ratings),
            "annotation_batch": str(prior_path),
            "stage_a_labels": str(labels_path),
        },
    )


def _read_stage_b(path, bundle, snapshots):
    batch, mapping, export_path = _read_batch(path, bundle, snapshots, "b")
    prior_path, _ = snapshots.reference(batch["prior_batch_ref"], path)
    prior, prior_mapping, _ = _read_batch(prior_path, bundle, snapshots, "a")
    for field in (
        "required_raters",
        "single_rater_reason",
        "selected_blind_ids",
        "selection_sha256",
    ):
        if canonical_json(batch[field]) != canonical_json(prior[field]):
            raise IntakeError(f"Stage B altered prior frozen {field}")
    if (
        canonical_json(mapping) != canonical_json(prior_mapping)
        or batch["mapping_ref"]["sha256"] != prior["mapping_ref"]["sha256"]
    ):
        raise IntakeError("Stage B mapping differs from locked Stage A mapping")
    labels_path, _ = snapshots.reference(batch["stage_a_labels_ref"], path)
    ratings = _stage_a_ratings(labels_path, prior, snapshots)
    lock_path, _ = snapshots.reference(batch["stage_a_lock_ref"], path)
    lock = snapshots.object(lock_path)
    require_keys(
        lock,
        {
            "schema_version",
            "batch_id",
            "locked_at",
            "completed_keys",
            "input_labels_ref",
            "mapping_sha256",
            "selection_sha256",
        },
        context="Stage A lock",
    )
    require_enum(lock["schema_version"], {"rebuttal-stage-a-lock/v2"}, "Stage A lock schema")
    require_timestamp(lock["locked_at"], "Stage A lock timestamp")
    bound_labels, _ = snapshots.reference(lock["input_labels_ref"], lock_path)
    if (
        lock["batch_id"] != prior["batch_id"]
        or bound_labels != labels_path
        or lock["mapping_sha256"] != prior["mapping_ref"]["sha256"]
        or lock["selection_sha256"] != prior["selection_sha256"]
        or canonical_json(lock["completed_keys"])
        != canonical_json([list(key) for key in sorted(_keys(prior))])
    ):
        raise IntakeError("Stage A lock binding mismatch")
    expected = _stage_b_rows(
        bundle,
        snapshots,
        mapping,
        ratings,
        batch["stage_a_lock_ref"]["sha256"],
        batch["required_raters"],
    )
    if canonical_json(snapshots.jsonl(export_path)) != canonical_json(expected):
        raise IntakeError("Stage B export differs from locked safe rendering")
    return batch, mapping, ratings


def _stage_b_ratings(path, batch, snapshots):
    ratings, adjudications = {}, {}
    expected = _keys(batch)
    for row in snapshots.jsonl(path):
        schema = row.get("schema_version")
        if schema == B_SCHEMA:
            require_keys(row, _B_FIELDS, context="Stage B returned rating")
            key = (_nonblank(row["blind_id"], "blind_id"), _nonblank(row["rater_id"], "rater_id"))
            if key not in expected or key in ratings:
                raise IntakeError(f"unknown or duplicate Stage B rating key: {key}")
            require_enum(row["label"], LABELS, "semantic label")
            _nonblank(row["rationale"], "Stage B rationale")
            ratings[key] = row
        elif schema == ADJ_SCHEMA:
            require_keys(row, _ADJ_FIELDS, context="adjudication")
            blind = _nonblank(row["blind_id"], "blind_id")
            if blind not in batch["selected_blind_ids"] or blind in adjudications:
                raise IntakeError("unknown or duplicate adjudication blind ID")
            _nonblank(row["adjudicator_id"], "adjudicator_id")
            _nonblank(row["disagreement_reason"], "disagreement_reason")
            require_enum(row["final_label"], LABELS, "adjudication final label")
            adjudications[blind] = row
        else:
            raise IntakeError("unknown returned annotation schema_version")
        require_timestamp(row["timestamp"], "Stage B/adjudication timestamp")
        require_sha256(row["stage_a_lock_sha256"], "stage_a_lock_sha256")
        if row["stage_a_lock_sha256"] != batch["stage_a_lock_ref"]["sha256"]:
            raise IntakeError("returned Stage B/adjudication lock hash mismatch")
    missing = sorted(expected - set(ratings))
    if missing:
        raise IntakeError(f"missing Stage B ratings: {missing}")
    for blind in batch["selected_blind_ids"]:
        labels = {ratings[(blind, rater)]["label"] for rater in batch["required_raters"]}
        if len(labels) > 1 and blind not in adjudications:
            raise IntakeError(f"unresolved annotation disagreement: {blind}")
    return ratings, adjudications


def _tables(bundle, semantic_rows):
    from .artifact_io import csv_bytes

    labels = {row["pair_id"]: row["label"] for row in semantic_rows}
    audits = {row["pair_id"]: row for row in bundle.rows}
    strata = {}
    flow = []
    for cohort in bundle.manifest.source_audit["cohorts"]:
        if cohort["setting"] not in {"R3", "R4"}:
            continue
        members = [
            pair
            for pair in bundle.manifest.pairs
            if any(
                m["cohort_id"] == cohort["cohort_id"] and m["membership"] == "member"
                for m in pair["cohort_membership"]
            )
        ]
        aligned = {
            pair["pair_id"]
            for pair in members
            if audits[pair["pair_id"]]["alignment"]["eligibility"] == "valid"
        }
        preserved = {
            pair["pair_id"] for pair in members if labels.get(pair["pair_id"]) == "unique_preserved"
        }
        counts = {
            "C_archive_recovered": len(members),
            "C_semantic": len(preserved),
            "C_aligned": len(aligned),
            "C_semantic_and_aligned": len(preserved & aligned),
            "semantic_unknown": sum(pair["pair_id"] not in labels for pair in members),
        }
        for name, count in counts.items():
            flow.append(
                {"cohort_id": cohort["cohort_id"], "set": name, "n": count, "status": "observed"}
            )
        flow.append(
            {
                "cohort_id": cohort["cohort_id"],
                "set": "original_source_missing",
                "n": "NA" if cohort["missing_count"] is None else cohort["missing_count"],
                "status": "unknown" if cohort["missing_count"] is None else "observed",
            }
        )
        for name in ("C_correct", "C_offset", "C_cross", "C_three", "C_fresh_failure"):
            flow.append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "set": name,
                    "n": "NA",
                    "status": "not_evaluated",
                }
            )
        for pair in members:
            pid = pair["pair_id"]
            label = labels.get(pid, "unknown")
            flags = audits[pid]["risk_flags"]["flags"]
            option_values = [flags.get("option_content"), flags.get("option_label")]
            option_state = (
                "edited"
                if any(value is True for value in option_values)
                else "unchanged"
                if all(value is False for value in option_values)
                else "unknown"
            )
            for dimension, value in (("semantic_label", label), ("option_edit", option_state)):
                key = (cohort["cohort_id"], dimension, value, label)
                strata[key] = strata.get(key, 0) + 1
    rows = [
        {"cohort_id": c, "dimension": d, "stratum": s, "semantic_label": label, "n": count}
        for (c, d, s, label), count in sorted(strata.items())
    ]
    return csv_bytes(flow, ["cohort_id", "set", "n", "status"]), csv_bytes(
        rows, ["cohort_id", "dimension", "stratum", "semantic_label", "n"]
    )


def import_adjudicated_labels(
    audit_path: str | Path,
    labels: str | Path,
    annotation_batch: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Import independent B judgments and explicit adjudication, never outcomes."""
    from .artifact_io import json_bytes, jsonl_bytes
    from .input_audit import load_input_audit

    started_at = _now()
    bundle = load_input_audit(audit_path)
    snapshots = _snapshot(bundle)
    batch_path, labels_path = Path(annotation_batch).resolve(), Path(labels).resolve()
    batch, mapping, a_ratings = _read_stage_b(batch_path, bundle, snapshots)
    b_ratings, adjudications = _stage_b_ratings(labels_path, batch, snapshots)
    labels_ref = _ref(labels_path, snapshots)
    independent_count = len(batch["required_raters"])
    agreement_count = 0
    semantic = []
    for item in mapping:
        blind = item["blind_id"]
        rows = [b_ratings[(blind, rater)] for rater in batch["required_raters"]]
        unique = {row["label"] for row in rows}
        agreement_count += len(unique) == 1
        adjudication = adjudications.get(blind)
        final = adjudication["final_label"] if adjudication is not None else rows[0]["label"]
        semantic.append(
            {
                "schema_version": "rebuttal-semantic-label/v2",
                "pair_id": item["pair_id"],
                "label": final,
                "rating_mode": "single" if independent_count == 1 else "independent_multiple",
                "single_rater_reason": batch["single_rater_reason"],
                "adjudication_status": "adjudicated"
                if adjudication is not None
                else "single_rating"
                if independent_count == 1
                else "unanimous",
                "source_ref": item["source_ref"],
                "manifest_record_ref": item["manifest_record_ref"],
                "annotation_batch_ref": _ref(batch_path, snapshots),
                "stage_a_labels_ref": batch["stage_a_labels_ref"],
                "returned_labels_ref": labels_ref,
                "stage_a_lock_sha256": batch["stage_a_lock_ref"]["sha256"],
                "stage_a_judgments": [
                    a_ratings[(blind, rater)] for rater in batch["required_raters"]
                ],
                "stage_b_judgments": rows,
                "adjudication": adjudication,
            }
        )
    semantic.sort(key=lambda row: row["pair_id"])
    agreement = {
        "schema_version": "rebuttal-annotation-agreement/v2",
        "rater_count": independent_count,
        "required_raters": batch["required_raters"],
        "single_rater_reason": batch["single_rater_reason"],
        "selected_pair_count": len(mapping),
        "required_rating_count": len(_keys(batch)),
        "missing_judgments": [],
        "pre_adjudication_agreement_n": agreement_count if independent_count > 1 else None,
        "pre_adjudication_agreement_rate": agreement_count / len(mapping)
        if independent_count > 1 and mapping
        else None,
        "agreement_status": "not_applicable_single_rater"
        if independent_count == 1
        else "empty_batch"
        if not mapping
        else "complete",
        "adjudicated_count": len(adjudications),
        "disagreement_count": len(mapping) - agreement_count if independent_count > 1 else None,
    }
    _, coverage = _selection(bundle, snapshots)
    imported_ids = {row["pair_id"] for row in semantic}
    for row in coverage["pair_coverage"]:
        if row["pair_id"] in imported_ids:
            row["annotation_status"] = "completed"
            row["semantic_status"] = "human_labeled"
    flow, strata = _tables(bundle, semantic)
    files = {
        "semantic_labels.jsonl": jsonl_bytes(semantic),
        "annotation_agreement.json": json_bytes(agreement),
        "annotation_coverage.json": json_bytes(coverage),
        "cohort_flow.csv": flow,
        "semantic_strata_table.csv": strata,
    }
    return _publish(
        bundle,
        snapshots,
        output_dir,
        files,
        "annotation-import",
        {
            "started_at": started_at,
            "annotation_batch": str(batch_path),
            "labels": str(labels_path),
            "annotation_batch_ref": _ref(batch_path, snapshots),
            "returned_labels_ref": labels_ref,
            "completed_count": len(semantic),
            "adjudicated_count": len(adjudications),
        },
    )
