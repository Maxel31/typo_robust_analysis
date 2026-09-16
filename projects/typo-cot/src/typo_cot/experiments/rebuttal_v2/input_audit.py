"""Independent text/token audit without selecting on answers or patch effects."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import artifact_io
from .artifacts import ManifestBundle, load_manifest
from .identity import identity_hash
from .input_spans import RISK_NAMES, classify_edit_regions, diff_actual_spans
from .input_tokenization import TokenizerLock, load_tokenizer_lock, reconstruct_token_alignment
from .schemas import IntakeError, canonical_json, canonical_sha256, require_keys


SCHEMA = "rebuttal-input-audit/v2"
METADATA_SCHEMA = "rebuttal-input-audit-metadata/v2"
ALGORITHM_VERSION = "independent-input-audit/2.0.0"


@dataclass(frozen=True)
class InputAuditBundle:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    rows: list[dict[str, Any]]
    manifest: ManifestBundle
    pairs: list[dict[str, Any]]
    tokenizer_lock: dict[str, Any]
    snapshots: dict[Path, str]

    @property
    def protocol(self) -> dict[str, Any]:
        return self.manifest.protocol

    @property
    def audit_ref(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.snapshots[self.path]}

    @property
    def metadata_ref(self) -> dict[str, str]:
        return {"path": str(self.metadata_path), "sha256": self.snapshots[self.metadata_path]}


def _resolved_pairs(bundle: ManifestBundle, snapshots: dict[Path, str]) -> list[dict[str, Any]]:
    pairs = []
    for pair in bundle.pairs:
        row = dict(pair)
        for field in ("clean_text", "typo_text", "clean_prompt", "typo_prompt"):
            if row[field] is None and row[f"{field}_ref"] is not None:
                path = artifact_io.resolve_artifact(row[f"{field}_ref"], bundle.path, snapshots)
                row[field] = artifact_io.capture_file(path, snapshots).decode("utf-8")
        pairs.append(row)
    return pairs


def _span(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(index) is int and index >= 0 for index in value)
        and value[0] <= value[1]
    )


def _intended_hit(pair: dict, span_audit: dict, alignment: dict) -> dict:
    intended = pair["intended_edits"]
    if intended is None:
        return {
            "status": "unknown",
            "records": [],
            "reason_codes": ["intended_metadata_unavailable"],
        }
    if not intended:
        return {
            "status": "not_applicable",
            "records": [],
            "reason_codes": ["no_recorded_intended_edits"],
        }
    records = []
    for index, entry in enumerate(intended):
        row = {"source_entry_index": index, "hit": None, "reason_codes": []}
        records.append(row)
        if (
            not isinstance(entry, dict)
            or entry.get("audit_adapter") != "rebuttal-intended-token/v2"
            or type(entry.get("clean_token_index")) is not int
            or not _span(entry.get("actual_clean_span"))
            or not _span(entry.get("actual_typo_span"))
        ):
            row["reason_codes"] = ["intended_coordinate_adapter_unavailable"]
            continue
        row["clean_token_index"] = entry["clean_token_index"]
        if alignment["eligibility"] != "valid":
            row["reason_codes"] = ["independent_alignment_unavailable"]
            continue
        target = entry["clean_token_index"]
        if not 0 <= target < len(alignment["clean"]["prompt_token_ids"]):
            row["reason_codes"] = ["intended_token_out_of_bounds"]
            continue
        matching = [
            edit_index
            for edit_index, edit in enumerate(span_audit["edit_spans"])
            if [edit["clean_start"], edit["clean_end"]] == entry["actual_clean_span"]
            and [edit["typo_start"], edit["typo_end"]] == entry["actual_typo_span"]
        ]
        words = [
            word
            for word in alignment["aligned_edits"]
            if any(i in word["edit_indices"] for i in matching)
        ]
        if len(words) != 1:
            row["reason_codes"] = ["intended_edit_not_uniquely_bound_to_independent_diff"]
            continue
        row["word_hit"] = target in words[0]["clean_token_indices"]
        token_start, token_end = alignment["clean"]["offset_mapping"][target]
        query_start = pair["clean_query_span"][0]
        edit_start, edit_end = [query_start + value for value in entry["actual_clean_span"]]
        if token_start == token_end:
            row["reason_codes"] = ["intended_token_has_no_character_span"]
        elif edit_start == edit_end and edit_start in {token_start, token_end}:
            row["reason_codes"] = ["insertion_at_intended_token_boundary"]
        else:
            row["hit"] = (
                token_start < edit_start < token_end
                if edit_start == edit_end
                else token_start < edit_end and edit_start < token_end
            )
            row["reason_codes"] = [] if row["hit"] else ["intended_token_misses_actual_edit"]
    hits = [row["hit"] for row in records]
    status = (
        "unknown" if None in hits else "hit" if all(hits) else "miss" if not any(hits) else "mixed"
    )
    return {
        "status": status,
        "records": records,
        "reason_codes": sorted({reason for row in records for reason in row["reason_codes"]}),
    }


def _archived_agreement(pair: dict, alignment: dict) -> dict:
    saved = pair["archived_token_positions"]
    if saved is None:
        return {"status": "unknown", "reason_codes": ["archived_coordinates_unavailable"]}
    if alignment["eligibility"] != "valid":
        return {"status": "unknown", "reason_codes": ["independent_alignment_unavailable"]}
    positions = []
    for entry in saved:
        if (
            not isinstance(entry, dict)
            or entry.get("audit_adapter") != "rebuttal-word-endpoints/v2"
            or any(
                type(entry.get(key)) is not int or entry[key] < 0
                for key in ("clean_endpoint", "typo_endpoint")
            )
        ):
            return {
                "status": "unknown",
                "reason_codes": ["archived_coordinate_adapter_unavailable"],
            }
        positions.append((entry["clean_endpoint"], entry["typo_endpoint"]))
    expected = [
        (word["clean_endpoint"], word["typo_endpoint"]) for word in alignment["aligned_edits"]
    ]
    equal = len(set(positions)) == len(positions) and sorted(positions) == sorted(expected)
    return {
        "status": "matches" if equal else "differs",
        "reason_codes": [] if equal else ["archived_endpoints_differ_from_independent_alignment"],
        "archived_endpoint_pairs": [list(value) for value in positions],
        "independent_endpoint_pairs": [list(value) for value in expected],
    }


def _audit_pair(pair: dict, manifest: ManifestBundle, lock: TokenizerLock) -> dict:
    clean, typo = pair["clean_text"], pair["typo_text"]
    if clean is None or typo is None:
        span = {
            "diff_status": "unavailable",
            "edit_distance": None,
            "optimal_path_count": 0,
            "edit_spans": [],
            "changed_words": {"clean": [], "typo": []},
            "aligned_words": [],
            "alignment_status": "unavailable",
            "reasons": ["exact_query_text_unavailable"],
            "algorithm_versions": {},
        }
        risk = {
            "flags": dict.fromkeys(RISK_NAMES),
            "reasons": {name: ["exact_query_text_unavailable"] for name in RISK_NAMES},
            "algorithm_versions": {},
        }
    else:
        result = diff_actual_spans(clean, typo)
        span = asdict(result)
        risk = classify_edit_regions(pair, result)
    alignment: dict[str, Any] = {"aligned_edits": []}
    entry = lock.entries.get(pair["model_id"])
    archived_runtime = pair["archived_runtime"] or {}
    for side in ("clean", "typo"):
        changed_word_spans = (
            [word[f"{side}_span"] for word in span["aligned_words"]]
            if span["alignment_status"] == "aligned"
            else []
        )
        alignment[side] = asdict(
            reconstruct_token_alignment(
                pair[f"{side}_prompt"],
                entry,
                changed_word_spans,
                query_text=pair[f"{side}_text"],
                query_span=pair[f"{side}_query_span"],
                archived_tokenizer_revision=pair["tokenizer_revision"],
                archived_tokenizer_id=archived_runtime.get("tokenizer_id"),
                archived_tokenizer_sha256=archived_runtime.get("tokenizer_sha256"),
            )
        )
    reasons = list(span["reasons"])
    for side in ("clean", "typo"):
        reasons.extend(alignment[side]["reason_codes"])
    if span["alignment_status"] in {"ambiguous", "unsupported", "no_change"}:
        eligibility = "invalid"
    elif span["alignment_status"] == "unavailable":
        eligibility = "unknown"
    elif all(alignment[side]["eligibility"] == "valid" for side in ("clean", "typo")):
        eligibility = "valid"
    else:
        eligibility = (
            "invalid"
            if any(alignment[side]["eligibility"] == "invalid" for side in ("clean", "typo"))
            else "unknown"
        )
    alignment["eligibility"] = eligibility
    alignment["reason_codes"] = sorted(set(reasons))
    if eligibility == "valid":
        for index, word in enumerate(span["aligned_words"]):
            left = alignment["clean"]["word_alignments"][index]
            right = alignment["typo"]["word_alignments"][index]
            alignment["aligned_edits"].append(
                {
                    "edit_indices": word["edit_indices"],
                    "clean_query_span": word["clean_span"],
                    "typo_query_span": word["typo_span"],
                    "clean_prompt_span": left["full_prompt_span"],
                    "typo_prompt_span": right["full_prompt_span"],
                    "clean_token_indices": left["token_indices"],
                    "typo_token_indices": right["token_indices"],
                    "clean_endpoint": left["endpoint"],
                    "typo_endpoint": right["endpoint"],
                }
            )
    if span["diff_status"] == "no_change" and pair["intended_edits"]:
        span["reasons"].append("recorded_attempts_with_no_net_change_cancellation_possible")
    scientific = {
        "algorithm_version": ALGORITHM_VERSION,
        "span_audit": span,
        "alignment": alignment,
        "risk_flags": risk,
        "intended_token_hit": _intended_hit(pair, span, alignment),
        "archived_coordinate_agreement": _archived_agreement(pair, alignment),
    }
    return {
        "schema_version": SCHEMA,
        "input_audit_id": identity_hash(
            SCHEMA,
            {
                "pair_id": pair["pair_id"],
                "protocol_sha256": pair["protocol_sha256"],
                "tokenizer_lock_sha256": canonical_sha256(lock.payload),
                "scientific_audit": scientific,
            },
        ),
        "pair_id": pair["pair_id"],
        "manifest_ref": {"artifact": manifest.manifest_ref, "record_id": pair["pair_id"]},
        "tokenizer_lock_ref": lock.reference,
        "protocol_sha256": pair["protocol_sha256"],
        "cohort_membership": pair["cohort_membership"],
        "source_kind": pair["source_kind"],
        "task": pair["task"],
        "model_id": pair["model_id"],
        "exact_text_sha256": {
            name: pair[f"{name}_sha256"]
            for name in ("clean_text", "typo_text", "clean_prompt", "typo_prompt")
        },
        **scientific,
    }


def _flow(pairs: list[dict], rows: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for pair, row in zip(pairs, rows, strict=True):
        for member in pair["cohort_membership"] or [{"cohort_id": None, "membership": "unknown"}]:
            key_fields = {
                "cohort_id": member["cohort_id"],
                "membership": member["membership"],
                "task": pair["task"],
                "model_id": pair["model_id"],
                "source_kind": pair["source_kind"],
            }
            key = canonical_json(key_fields)
            group = groups.setdefault(
                key,
                {
                    **key_fields,
                    "pair_count": 0,
                    "C_archive_count": 0,
                    "C_aligned_count": 0,
                    "alignment_unknown_count": 0,
                    "alignment_invalid_count": 0,
                    "semantic_status": "not_run",
                },
            )
            group["pair_count"] += 1
            group["C_archive_count"] += int(member["membership"] == "member")
            state = row["alignment"]["eligibility"]
            group["C_aligned_count"] += int(state == "valid")
            group["alignment_unknown_count"] += int(state == "unknown")
            group["alignment_invalid_count"] += int(state == "invalid")
    return [groups[key] for key in sorted(groups)]


def run_input_audit(
    manifest_path: str | Path, tokenizer_lock_path: str | Path, output_dir: str | Path
) -> dict:
    started = artifact_io.utc_now()
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IntakeError(f"Output directory must be absent or empty: {output}")
    manifest = load_manifest(manifest_path)
    snapshots = dict(manifest.snapshots)
    lock = load_tokenizer_lock(tokenizer_lock_path, snapshots)
    snapshots.update(lock.snapshots)
    pairs = _resolved_pairs(manifest, snapshots)
    rows = [_audit_pair(pair, manifest, lock) for pair in pairs]
    flow = _flow(pairs, rows)
    files = {
        "input_audit_records.jsonl": artifact_io.jsonl_bytes(rows),
        "protocol.json": manifest.protocol_bytes,
    }
    metadata = {
        "schema_version": METADATA_SCHEMA,
        "audit_ref": artifact_io.artifact_ref(
            "input_audit_records.jsonl", files["input_audit_records.jsonl"]
        ),
        "manifest_ref": manifest.manifest_ref,
        "manifest_metadata_ref": manifest.metadata_ref,
        "tokenizer_lock_ref": lock.reference,
        "protocol_ref": artifact_io.artifact_ref("protocol.json", files["protocol.json"]),
        "protocol_sha256": canonical_sha256(manifest.protocol),
        "algorithm_version": ALGORITHM_VERSION,
    }
    files["input_audit_records.meta.json"] = artifact_io.json_bytes(metadata)
    summary = {
        "schema_version": "rebuttal-input-audit-summary/v2",
        "experiment_id": "R2",
        "pair_count": len(rows),
        "source_cohort_coverage": manifest.source_audit["cohorts"],
        "alignment_counts": dict(Counter(row["alignment"]["eligibility"] for row in rows)),
        "diff_counts": dict(Counter(row["span_audit"]["diff_status"] for row in rows)),
        "intended_token_hit_counts": dict(
            Counter(row["intended_token_hit"]["status"] for row in rows)
        ),
        "archived_coordinate_agreement_counts": dict(
            Counter(row["archived_coordinate_agreement"]["status"] for row in rows)
        ),
        "risk_flag_counts": {
            name: dict(
                Counter(
                    "unknown"
                    if row["risk_flags"]["flags"][name] is None
                    else str(row["risk_flags"]["flags"][name]).lower()
                    for row in rows
                )
            )
            for name in RISK_NAMES
        },
        "cohort_flow": flow,
        "semantic_status": "not_run",
        "model_execution_performed": False,
        "reason_codes": ["risk_flags_are_not_semantic_labels", "no_human_judgments_imported"],
    }
    files["input_audit_summary.json"] = artifact_io.json_bytes(summary)
    files["cohort_flow.csv"] = artifact_io.csv_bytes(
        flow,
        [
            "cohort_id",
            "membership",
            "task",
            "model_id",
            "source_kind",
            "pair_count",
            "C_archive_count",
            "C_aligned_count",
            "alignment_unknown_count",
            "alignment_invalid_count",
            "semantic_status",
        ],
    )
    base = Path(__file__).resolve().parent
    code = artifact_io.capture_code(
        [
            base / name
            for name in (
                "input_audit.py",
                "input_spans.py",
                "input_tokenization.py",
                "artifact_io.py",
                "artifacts.py",
                "identity.py",
                "schemas.py",
                "source.py",
                "cli.py",
            )
        ],
        snapshots,
    )
    run = artifact_io.make_run(
        command="input-audit",
        started_at=started,
        invocation={
            "manifest": str(manifest.path),
            "tokenizer_lock": str(lock.path),
            "output_dir": str(output),
        },
        snapshots=snapshots,
        code_identity=code,
        protocol=manifest.protocol,
        files=files,
        expected_ids=[pair["pair_id"] for pair in pairs],
        completed_ids=[pair["pair_id"] for pair in pairs],
        missing_ids=[],
        reason_codes=summary["reason_codes"],
        config_refs=[lock.reference],
        experiment_status="partial",
    )
    run["input_audit_stage_status"] = "complete"
    files["run.json"] = artifact_io.json_bytes(run)
    artifact_io.publish_artifacts(output, files, snapshots)
    return run


def load_input_audit(path: str | Path) -> InputAuditBundle:
    """Validate the complete input chain and deterministic scientific projection."""
    path = Path(path).resolve()
    metadata_path = path.with_name("input_audit_records.meta.json")
    snapshots: dict[Path, str] = {}
    metadata = artifact_io.read_json(metadata_path, snapshots)
    require_keys(
        metadata,
        {
            "schema_version",
            "audit_ref",
            "manifest_ref",
            "manifest_metadata_ref",
            "tokenizer_lock_ref",
            "protocol_ref",
            "protocol_sha256",
            "algorithm_version",
        },
        context="input audit metadata",
    )
    if (
        metadata["schema_version"] != METADATA_SCHEMA
        or metadata["algorithm_version"] != ALGORITHM_VERSION
    ):
        raise IntakeError("unsupported input audit metadata/algorithm version")
    if artifact_io.resolve_artifact(metadata["audit_ref"], metadata_path, snapshots) != path:
        raise IntakeError("input audit metadata binds a different audit path")
    manifest_path = artifact_io.resolve_artifact(metadata["manifest_ref"], metadata_path, snapshots)
    bound_metadata = artifact_io.resolve_artifact(
        metadata["manifest_metadata_ref"], metadata_path, snapshots
    )
    manifest = load_manifest(manifest_path)
    if bound_metadata != manifest.metadata_path:
        raise IntakeError("input audit binds a different manifest metadata path")
    for ref_path, digest in manifest.snapshots.items():
        if ref_path in snapshots and snapshots[ref_path] != digest:
            raise IntakeError("input audit and manifest snapshot mismatch")
        snapshots[ref_path] = digest
    protocol_path = artifact_io.resolve_artifact(metadata["protocol_ref"], metadata_path, snapshots)
    if canonical_json(artifact_io.read_json(protocol_path, snapshots)) != canonical_json(
        manifest.protocol
    ) or metadata["protocol_sha256"] != canonical_sha256(manifest.protocol):
        raise IntakeError("input audit protocol binding mismatch")
    lock_path = artifact_io.resolve_artifact(
        metadata["tokenizer_lock_ref"], metadata_path, snapshots
    )
    lock = load_tokenizer_lock(lock_path, snapshots)
    snapshots.update(lock.snapshots)
    pairs = _resolved_pairs(manifest, snapshots)
    rows = artifact_io.read_jsonl(path, snapshots)
    expected = [_audit_pair(pair, manifest, lock) for pair in pairs]
    if canonical_json(rows) != canonical_json(expected):
        raise IntakeError("input audit differs from the verified independent audit of its manifest")
    artifact_io.revalidate_snapshots(snapshots)
    return InputAuditBundle(
        path, metadata_path, metadata, rows, manifest, pairs, lock.payload, snapshots
    )
