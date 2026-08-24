"""Fail-closed semantic bindings for confirmatory evaluation v2."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.evaluation.calibration_v2 import (
    EvaluationV2Protocol,
    load_base_calibration_observations,
    score_base_only_severity_calibration,
    validate_calibration_semantic_bindings,
)
from typo_robust_training.integrity import sha256_file


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_ITEM_FIELDS = {
    "schema_version",
    "task",
    "record_id",
    "source_text",
    "source_text_sha256",
    "reference_answer",
    "reference_answer_sha256",
}
_TYPO_FIELDS = {
    "schema_version",
    "task",
    "record_id",
    "source_text_sha256",
    "severity_edit_count",
    "variant",
    "realized_typo_text",
    "realized_typo_sha256",
}
_ROLE_FIELDS = {"schema_version", "role", "record_id", "source_text_sha256"}
_REQUIRED_ROLES = (
    "training",
    "linear-probe-selection",
    "linear-probe-validation",
    "tune",
    "pre-pr",
    "calibration",
    "confirmatory",
)


def _sha64(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA64.fullmatch(value) is None:
        raise ValueError(f"evaluation v2 {field} must be a lowercase SHA-256")
    return value


def _strict_row(value: object, *, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"evaluation v2 {label} fields differ")
    return value


def _exact_text_hash(text: object, digest: object, *, label: str) -> tuple[str, str]:
    if not isinstance(text, str) or not text:
        raise ValueError(f"evaluation v2 {label} text is invalid")
    frozen_digest = _sha64(digest, field=f"{label} text hash")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != frozen_digest:
        raise ValueError(f"evaluation v2 {label} text/hash binding differs")
    return text, frozen_digest


def _actual_file_sha256(path: Path, *, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"evaluation v2 {label} must be an existing file")
    return sha256_file(resolved)


@dataclass(frozen=True, slots=True)
class ConfirmatorySemanticBinding:
    protocol_sha256: str
    selected_edit_count: int
    item_manifest_sha256: str
    realized_typo_manifest_sha256: str
    source_hashes: Mapping[tuple[str, str], str]
    reference_answer_hashes: Mapping[tuple[str, str], str]
    typo_hashes: Mapping[tuple[str, str, int], str]


def load_confirmatory_semantic_binding(
    *,
    protocol: EvaluationV2Protocol,
    selected_edit_count: int,
    item_manifest_path: Path,
    realized_typo_manifest_path: Path,
) -> ConfirmatorySemanticBinding:
    """Parse and bind the exact 5-task item x 2-variant confirmatory corpus."""

    if selected_edit_count not in protocol.severity_edit_counts:
        raise ValueError("evaluation v2 selected severity is outside the frozen grid")
    item_path = Path(item_manifest_path).resolve()
    typo_path = Path(realized_typo_manifest_path).resolve()
    if not item_path.is_file() or not typo_path.is_file():
        raise ValueError("evaluation v2 confirmatory manifests must be files")

    source_hashes: dict[tuple[str, str], str] = {}
    reference_answer_hashes: dict[tuple[str, str], str] = {}
    global_record_ids: set[str] = set()
    global_source_hashes: set[str] = set()
    task_counts: Counter[str] = Counter()
    for line_number, line in read_lf_jsonl_lines(
        item_path, context="evaluation v2 confirmatory item manifest"
    ):
        value = strict_loads(line, context=f"{item_path}:{line_number}")
        row = _strict_row(value, fields=_ITEM_FIELDS, label="confirmatory item manifest")
        if row["schema_version"] != "robustness-evaluation-v2-confirmatory-item/v1":
            raise ValueError("evaluation v2 confirmatory item schema differs")
        task = row["task"]
        if not isinstance(task, str) or task not in protocol.tasks:
            raise ValueError("evaluation v2 confirmatory item task differs")
        record_id = _sha64(row["record_id"], field="confirmatory record ID")
        _text, source_hash = _exact_text_hash(
            row["source_text"], row["source_text_sha256"], label="confirmatory source"
        )
        _answer, reference_answer_hash = _exact_text_hash(
            row["reference_answer"],
            row["reference_answer_sha256"],
            label="confirmatory reference answer",
        )
        key = (task, record_id)
        if (
            key in source_hashes
            or record_id in global_record_ids
            or source_hash in global_source_hashes
        ):
            raise ValueError("evaluation v2 confirmatory source items are not unique")
        source_hashes[key] = source_hash
        reference_answer_hashes[key] = reference_answer_hash
        global_record_ids.add(record_id)
        global_source_hashes.add(source_hash)
        task_counts[task] += 1
    if set(task_counts) != set(protocol.tasks) or any(
        task_counts[task] != protocol.confirmatory_records_per_task for task in protocol.tasks
    ):
        raise ValueError("evaluation v2 confirmatory item sample size differs")

    typo_hashes: dict[tuple[str, str, int], str] = {}
    coverage: dict[tuple[str, str], set[int]] = {key: set() for key in source_hashes}
    for line_number, line in read_lf_jsonl_lines(
        typo_path, context="evaluation v2 confirmatory typo manifest"
    ):
        value = strict_loads(line, context=f"{typo_path}:{line_number}")
        row = _strict_row(value, fields=_TYPO_FIELDS, label="confirmatory typo manifest")
        if row["schema_version"] != "robustness-evaluation-v2-confirmatory-typo/v1":
            raise ValueError("evaluation v2 confirmatory typo schema differs")
        task, record_id = row["task"], row["record_id"]
        if not isinstance(task, str) or task not in protocol.tasks:
            raise ValueError("evaluation v2 confirmatory typo task differs")
        record_id = _sha64(record_id, field="confirmatory typo record ID")
        item_key = (task, record_id)
        if item_key not in source_hashes:
            raise ValueError("evaluation v2 confirmatory typo has no source item")
        source_hash = _sha64(row["source_text_sha256"], field="confirmatory typo source hash")
        if source_hash != source_hashes[item_key]:
            raise ValueError("evaluation v2 confirmatory typo/source binding differs")
        severity = row["severity_edit_count"]
        variant = row["variant"]
        if type(severity) is not int or severity != selected_edit_count:
            raise ValueError("evaluation v2 confirmatory typo severity differs")
        if type(variant) is not int or variant not in range(
            protocol.confirmatory_typo_variants_per_item
        ):
            raise ValueError("evaluation v2 confirmatory typo variant differs")
        _text, typo_hash = _exact_text_hash(
            row["realized_typo_text"],
            row["realized_typo_sha256"],
            label="confirmatory realized typo",
        )
        if typo_hash == source_hash:
            raise ValueError("evaluation v2 confirmatory typo is identical to clean source")
        key = (task, record_id, variant)
        if key in typo_hashes:
            raise ValueError("evaluation v2 confirmatory typo manifest contains duplicates")
        typo_hashes[key] = typo_hash
        coverage[item_key].add(variant)
    expected_variants = set(range(protocol.confirmatory_typo_variants_per_item))
    if any(variants != expected_variants for variants in coverage.values()):
        raise ValueError("evaluation v2 confirmatory typo coverage differs")

    return ConfirmatorySemanticBinding(
        protocol_sha256=protocol.config_sha256,
        selected_edit_count=selected_edit_count,
        item_manifest_sha256=sha256_file(item_path),
        realized_typo_manifest_sha256=sha256_file(typo_path),
        source_hashes=MappingProxyType(source_hashes),
        reference_answer_hashes=MappingProxyType(reference_answer_hashes),
        typo_hashes=MappingProxyType(typo_hashes),
    )


def validate_outcomes_against_confirmatory_binding(
    rows: Sequence[Mapping[str, object]],
    *,
    protocol: EvaluationV2Protocol,
    binding: ConfirmatorySemanticBinding,
) -> None:
    if binding.protocol_sha256 != protocol.config_sha256:
        raise ValueError("evaluation v2 confirmatory binding protocol differs")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("evaluation v2 confirmatory outcome must be an object")
        task, record_id, variant = row.get("task"), row.get("record_id"), row.get("variant")
        if not isinstance(task, str) or not isinstance(record_id, str) or type(variant) is not int:
            raise ValueError("evaluation v2 confirmatory outcome semantic key differs")
        source = binding.source_hashes.get((task, record_id))
        typo = binding.typo_hashes.get((task, record_id, variant))
        if source is None or typo is None:
            raise ValueError("evaluation v2 confirmatory outcome is outside frozen manifests")
        if row.get("source_text_sha256") != source:
            raise ValueError("evaluation v2 confirmatory outcome/source manifest binding differs")
        if row.get("reference_answer_sha256") != binding.reference_answer_hashes.get(
            (task, record_id)
        ):
            raise ValueError(
                "evaluation v2 confirmatory outcome/reference answer manifest binding differs"
            )
        if row.get("realized_typo_sha256") != typo:
            raise ValueError("evaluation v2 confirmatory outcome/typo manifest binding differs")


def validate_tier_id_disjointness(
    *,
    role_manifest_path: Path,
    calibration_sources: Mapping[tuple[str, str], str],
    confirmatory_binding: ConfirmatorySemanticBinding,
) -> str:
    """Reject ID or exact-source reuse across every preregistered data role."""

    resolved = Path(role_manifest_path).resolve()
    by_role: dict[str, dict[str, str]] = {role: {} for role in _REQUIRED_ROLES}
    source_hashes_by_role: dict[str, set[str]] = {role: set() for role in _REQUIRED_ROLES}
    id_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    for line_number, line in read_lf_jsonl_lines(
        resolved, context="evaluation v2 tier role manifest"
    ):
        value = strict_loads(line, context=f"{resolved}:{line_number}")
        row = _strict_row(value, fields=_ROLE_FIELDS, label="tier role manifest")
        if row["schema_version"] != "robustness-evaluation-v2-tier-role-id/v1":
            raise ValueError("evaluation v2 tier role schema differs")
        role = row["role"]
        if not isinstance(role, str) or role not in by_role:
            raise ValueError("evaluation v2 tier role differs")
        record_id = _sha64(row["record_id"], field="tier role record ID")
        source_hash = _sha64(row["source_text_sha256"], field="tier role source hash")
        if record_id in by_role[role] or source_hash in source_hashes_by_role[role]:
            raise ValueError("evaluation v2 tier role manifest contains duplicates")
        if record_id in id_owner and id_owner[record_id] != role:
            raise ValueError("evaluation v2 record ID leaks across data tiers")
        if source_hash in hash_owner and hash_owner[source_hash] != role:
            raise ValueError("evaluation v2 source text leaks across data tiers")
        by_role[role][record_id] = source_hash
        source_hashes_by_role[role].add(source_hash)
        id_owner[record_id] = role
        hash_owner[source_hash] = role
    if any(not values for values in by_role.values()):
        raise ValueError("evaluation v2 tier role inventory is incomplete")
    expected_calibration = {
        record_id: source for (_task, record_id), source in calibration_sources.items()
    }
    expected_confirmatory = {
        record_id: source
        for (_task, record_id), source in confirmatory_binding.source_hashes.items()
    }
    if by_role["calibration"] != expected_calibration:
        raise ValueError("evaluation v2 calibration tier semantic binding differs")
    if by_role["confirmatory"] != expected_confirmatory:
        raise ValueError("evaluation v2 confirmatory tier semantic binding differs")
    return sha256_file(resolved)


def canonical_source_tree_sha256(repository_path: Path, *, commit: str) -> str:
    """Hash the exact raw output of the preregistered git tree command."""

    if _SHA40.fullmatch(commit) is None:
        raise ValueError("evaluation v2 final merged commit is invalid")
    repository = Path(repository_path).resolve()
    process = subprocess.run(
        ["git", "ls-tree", "-r", "--full-tree", commit],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise ValueError("evaluation v2 final merged commit is unavailable")
    return hashlib.sha256(process.stdout).hexdigest()


def _git_output(repository: Path, arguments: Sequence[str]) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise ValueError("evaluation v2 repository provenance is unavailable")
    return process.stdout


def _load_evaluation_v2_registry_phase(
    *,
    required_phase: str,
    registry_path: Path,
    protocol: EvaluationV2Protocol,
    repository_path: Path,
    calibration_observations_path: Path,
    calibration_item_manifest_path: Path,
    calibration_typo_manifest_path: Path,
    calibration_result_path: Path,
    confirmatory_item_manifest_path: Path,
    confirmatory_typo_manifest_path: Path,
    tier_role_manifest_path: Path,
    factorial_arm_registry_path: Path,
    probe_artifact_registry_path: Path,
    training_config_registry_path: Path,
    training_data_registry_path: Path,
    legacy_random_2_registry_path: Path,
    post_training_artifact_paths: Mapping[str, Path] | None,
) -> Mapping[str, object]:
    """Load one exact registry phase and verify executable code/data provenance."""

    if required_phase not in {"training-preregistered", "evaluation-opening-sealed"}:
        raise ValueError("evaluation v2 requested registry phase is invalid")

    resolved = Path(registry_path).resolve()
    calibration_paths = tuple(
        Path(path).resolve()
        for path in (
            calibration_observations_path,
            calibration_item_manifest_path,
            calibration_typo_manifest_path,
            calibration_result_path,
        )
    )
    if any(not path.is_file() for path in calibration_paths):
        raise ValueError("evaluation v2 calibration registry inputs must be files")
    calibration_observations = load_base_calibration_observations(calibration_paths[0])
    calibration_sources = validate_calibration_semantic_bindings(
        calibration_observations,
        protocol=protocol,
        item_manifest_path=calibration_paths[1],
        realized_typo_manifest_path=calibration_paths[2],
    )
    calibration_status, calibration_selected, calibration_summaries = (
        score_base_only_severity_calibration(calibration_observations, protocol=protocol)
    )
    if calibration_status != "selected" or calibration_selected is None:
        raise ValueError("evaluation v2 calibration did not select a frozen severity")
    confirmatory_binding = load_confirmatory_semantic_binding(
        protocol=protocol,
        selected_edit_count=calibration_selected,
        item_manifest_path=confirmatory_item_manifest_path,
        realized_typo_manifest_path=confirmatory_typo_manifest_path,
    )
    tier_role_manifest_sha256 = validate_tier_id_disjointness(
        role_manifest_path=tier_role_manifest_path,
        calibration_sources=calibration_sources,
        confirmatory_binding=confirmatory_binding,
    )
    payload = strict_loads(resolved.read_text(encoding="utf-8"), context=str(resolved))
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation v2 registry must be an object")
    if payload.get("schema_version") != "robustness-evaluation-v2-registry/v1":
        raise ValueError("evaluation v2 registry schema differs")
    if (
        payload.get("protocol_id") != protocol.protocol_id
        or payload.get("protocol_sha256") != protocol.config_sha256
    ):
        raise ValueError("evaluation v2 registry protocol binding differs")
    if payload.get("state") != required_phase:
        raise ValueError(f"evaluation v2 registry is not {required_phase}")
    if set(payload) != {
        "schema_version",
        "protocol_id",
        "state",
        "protocol_sha256",
        "model_inventory",
        "governance_attestation",
        "calibration",
        "confirmatory",
        "legacy_random_2",
    }:
        raise ValueError("evaluation v2 registry top-level fields differ")
    expected_models = [
        {"id": model.model_id, "revision": model.revision} for model in protocol.models
    ]
    if payload.get("model_inventory") != expected_models:
        raise ValueError("evaluation v2 registry model inventory differs")
    if payload.get("governance_attestation") != {
        "adapter_outputs_used_for_calibration": False,
        "severity_grid_extended": False,
        "model_inventory_changed_after_calibration": False,
    }:
        raise ValueError("evaluation v2 registry governance differs")
    calibration = payload.get("calibration")
    confirmatory = payload.get("confirmatory")
    if not isinstance(calibration, Mapping) or not isinstance(confirmatory, Mapping):
        raise ValueError("evaluation v2 registry sections differ")
    if set(calibration) != {
        "status",
        "item_manifest_sha256",
        "realized_typo_manifest_sha256",
        "base_observations_sha256",
        "result_sha256",
        "selected_primary_edit_count",
    }:
        raise ValueError("evaluation v2 registry calibration fields differ")
    if set(confirmatory) != {
        "status",
        "factorial_arm_inventory",
        "mistral_only_direct_comparison_arm",
        "mistral_only_direct_comparison_seeds",
        "training_contract_identities",
        "final_merged_implementation_commit",
        "final_merged_source_tree_sha256",
        "source_tree_hash_policy",
        "item_manifest_sha256",
        "realized_typo_variant_manifest_sha256",
        "tier_id_role_manifest_sha256",
        "factorial_arm_registry_sha256",
        "probe_artifact_registry_sha256",
        "training_config_registry_sha256",
        "training_data_registry_sha256",
        "training_preregistered_registry_sha256",
        "mistral_kojima_faithful_matched_seed_42_43_44_registry_sha256",
        "mistral_kojima_faithful_public_seed_1_anchor_checkpoint_sha256",
        "public_seed_1_anchor_role",
        "random_layer_mask_policy",
        "arm_checkpoint_registry_sha256",
        "opening_log_sha256",
    }:
        raise ValueError("evaluation v2 registry confirmatory fields differ")
    if (
        calibration.get("status") != "selected"
        or calibration.get("selected_primary_edit_count")
        != confirmatory_binding.selected_edit_count
        or calibration_status != "selected"
        or calibration_selected != confirmatory_binding.selected_edit_count
    ):
        raise ValueError("evaluation v2 registry calibration result differs")
    expected_calibration_hashes = {
        "base_observations_sha256": sha256_file(calibration_paths[0]),
        "item_manifest_sha256": sha256_file(calibration_paths[1]),
        "realized_typo_manifest_sha256": sha256_file(calibration_paths[2]),
        "result_sha256": sha256_file(calibration_paths[3]),
    }
    if any(
        calibration.get(field) != digest for field, digest in expected_calibration_hashes.items()
    ):
        raise ValueError("evaluation v2 registry calibration file binding differs")
    result_artifact = strict_loads(
        calibration_paths[3].read_text(encoding="utf-8"),
        context=str(calibration_paths[3]),
    )
    if not isinstance(result_artifact, Mapping) or set(result_artifact) != {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "model_inventory",
        "candidate_edit_counts",
        "status",
        "selected_primary_edit_count",
        "stop_policy",
        "provenance",
        "summaries",
    }:
        raise ValueError("evaluation v2 calibration result artifact fields differ")
    expected_calibration_provenance = {
        "adapter_outputs_used": False,
        "item_manifest_sha256": expected_calibration_hashes["item_manifest_sha256"],
        "realized_typo_manifest_sha256": expected_calibration_hashes[
            "realized_typo_manifest_sha256"
        ],
        "base_observations_sha256": expected_calibration_hashes["base_observations_sha256"],
    }
    if (
        result_artifact.get("schema_version") != "robustness-evaluation-v2-severity-calibration/v1"
        or result_artifact.get("protocol_id") != protocol.protocol_id
        or result_artifact.get("protocol_sha256") != protocol.config_sha256
        or result_artifact.get("model_inventory")
        != [{"id": model.model_id, "revision": model.revision} for model in protocol.models]
        or result_artifact.get("candidate_edit_counts") != list(protocol.severity_edit_counts)
        or result_artifact.get("status") != calibration_status
        or result_artifact.get("selected_primary_edit_count") != calibration_selected
        or result_artifact.get("stop_policy") != "do-not-extend-grid-or-replace-models"
        or result_artifact.get("provenance") != expected_calibration_provenance
        or result_artifact.get("summaries") != dict(calibration_summaries)
    ):
        raise ValueError("evaluation v2 calibration result artifact differs")
    if confirmatory.get("status") != required_phase:
        raise ValueError(f"evaluation v2 confirmatory registry is not {required_phase}")
    if confirmatory.get("factorial_arm_inventory") != list(protocol.arms):
        raise ValueError("evaluation v2 factorial arm registry inventory differs")
    if confirmatory.get("mistral_only_direct_comparison_arm") != (
        "kojima-faithful-output-matching"
    ):
        raise ValueError("evaluation v2 faithful comparison arm differs")
    if confirmatory.get("training_contract_identities") != dict(
        protocol.training_contract_identities
    ):
        raise ValueError("evaluation v2 training contract identity differs")
    commit = confirmatory.get("final_merged_implementation_commit")
    if not isinstance(commit, str) or _SHA40.fullmatch(commit) is None:
        raise ValueError("evaluation v2 final merged commit is missing")
    repository = Path(repository_path).resolve()
    current_head = _git_output(repository, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if current_head != commit:
        raise ValueError("evaluation v2 execution HEAD differs from frozen merged commit")
    merged = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/develop"],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if merged.returncode != 0:
        raise ValueError("evaluation v2 implementation commit is not merged into origin/develop")
    if _git_output(repository, ["status", "--porcelain=v1", "--untracked-files=all"]):
        raise ValueError("evaluation v2 execution worktree is not clean")
    expected_tree = _sha64(
        confirmatory.get("final_merged_source_tree_sha256"),
        field="registry source-tree hash",
    )
    if canonical_source_tree_sha256(repository, commit=commit) != expected_tree:
        raise ValueError("evaluation v2 final merged source-tree binding differs")
    if confirmatory.get("source_tree_hash_policy") != (
        "sha256-of-git-ls-tree-r-full-tree-head-lf/v1"
    ):
        raise ValueError("evaluation v2 source-tree policy differs")
    if confirmatory.get("item_manifest_sha256") != confirmatory_binding.item_manifest_sha256:
        raise ValueError("evaluation v2 confirmatory item manifest registry binding differs")
    if confirmatory.get("realized_typo_variant_manifest_sha256") != (
        confirmatory_binding.realized_typo_manifest_sha256
    ):
        raise ValueError("evaluation v2 confirmatory typo manifest registry binding differs")
    if confirmatory.get("tier_id_role_manifest_sha256") != tier_role_manifest_sha256:
        raise ValueError("evaluation v2 tier role manifest registry binding differs")
    for field in (
        "factorial_arm_registry_sha256",
        "probe_artifact_registry_sha256",
        "training_config_registry_sha256",
        "training_data_registry_sha256",
    ):
        path = {
            "factorial_arm_registry_sha256": factorial_arm_registry_path,
            "probe_artifact_registry_sha256": probe_artifact_registry_path,
            "training_config_registry_sha256": training_config_registry_path,
            "training_data_registry_sha256": training_data_registry_path,
        }[field]
        if confirmatory.get(field) != _actual_file_sha256(path, label=field):
            raise ValueError(f"evaluation v2 registry confirmatory {field} binding differs")
    post_training_fields = (
        "mistral_kojima_faithful_matched_seed_42_43_44_registry_sha256",
        "mistral_kojima_faithful_public_seed_1_anchor_checkpoint_sha256",
        "arm_checkpoint_registry_sha256",
        "opening_log_sha256",
    )
    if required_phase == "training-preregistered":
        if (
            confirmatory.get("training_preregistered_registry_sha256") is not None
            or any(confirmatory.get(field) is not None for field in post_training_fields)
            or post_training_artifact_paths is not None
        ):
            raise ValueError(
                "evaluation v2 training-preregistered registry contains post-training artifacts"
            )
    else:
        if post_training_artifact_paths is None or set(post_training_artifact_paths) != set(
            post_training_fields
        ):
            raise ValueError("evaluation v2 post-training artifact inventory differs")
        _sha64(
            confirmatory.get("training_preregistered_registry_sha256"),
            field="training-preregistered registry hash",
        )
        for field, path in post_training_artifact_paths.items():
            if confirmatory.get(field) != _actual_file_sha256(path, label=field):
                raise ValueError(f"evaluation v2 registry confirmatory {field} binding differs")
    if confirmatory.get("public_seed_1_anchor_role") != "reproducibility-only-not-pooled":
        raise ValueError("evaluation v2 public seed-1 anchor role differs")
    if confirmatory.get("random_layer_mask_policy") != (
        "sha256-seed42-count-matched-random-freeze/v1"
    ):
        raise ValueError("evaluation v2 random layer control differs")
    if confirmatory.get("mistral_only_direct_comparison_seeds") != [42, 43, 44]:
        raise ValueError("evaluation v2 faithful matched seed inventory differs")
    legacy = payload.get("legacy_random_2")
    if (
        not isinstance(legacy, Mapping)
        or set(legacy)
        != {
            "role",
            "v1_protocol_sha256",
            "inherited_runtime_contracts",
            "runtime_contract_sha256",
            "frozen_registry_sha256",
        }
        or legacy.get("role") != "secondary-continuity-only"
        or legacy.get("v1_protocol_sha256") != protocol.legacy_v1_protocol_sha256
        or legacy.get("inherited_runtime_contracts")
        != ["generation", "typos.eligibility", "corpus_runtime"]
        or legacy.get("runtime_contract_sha256") != dict(protocol.runtime_contract_sha256)
    ):
        raise ValueError("evaluation v2 legacy random-2 role differs")
    if legacy.get("frozen_registry_sha256") != _actual_file_sha256(
        legacy_random_2_registry_path,
        label="legacy random-2 registry",
    ):
        raise ValueError("evaluation v2 legacy random-2 registry binding differs")
    return MappingProxyType(dict(payload))


def load_training_preregistered_evaluation_v2_registry(
    *,
    registry_path: Path,
    protocol: EvaluationV2Protocol,
    repository_path: Path,
    calibration_observations_path: Path,
    calibration_item_manifest_path: Path,
    calibration_typo_manifest_path: Path,
    calibration_result_path: Path,
    confirmatory_item_manifest_path: Path,
    confirmatory_typo_manifest_path: Path,
    tier_role_manifest_path: Path,
    factorial_arm_registry_path: Path,
    probe_artifact_registry_path: Path,
    training_config_registry_path: Path,
    training_data_registry_path: Path,
    legacy_random_2_registry_path: Path,
) -> Mapping[str, object]:
    """Require the immutable pre-training phase and prohibit checkpoint artifacts."""

    return _load_evaluation_v2_registry_phase(
        required_phase="training-preregistered",
        registry_path=registry_path,
        protocol=protocol,
        repository_path=repository_path,
        calibration_observations_path=calibration_observations_path,
        calibration_item_manifest_path=calibration_item_manifest_path,
        calibration_typo_manifest_path=calibration_typo_manifest_path,
        calibration_result_path=calibration_result_path,
        confirmatory_item_manifest_path=confirmatory_item_manifest_path,
        confirmatory_typo_manifest_path=confirmatory_typo_manifest_path,
        tier_role_manifest_path=tier_role_manifest_path,
        factorial_arm_registry_path=factorial_arm_registry_path,
        probe_artifact_registry_path=probe_artifact_registry_path,
        training_config_registry_path=training_config_registry_path,
        training_data_registry_path=training_data_registry_path,
        legacy_random_2_registry_path=legacy_random_2_registry_path,
        post_training_artifact_paths=None,
    )


def load_evaluation_opening_sealed_evaluation_v2_registry(
    *,
    registry_path: Path,
    training_preregistered_registry_path: Path,
    protocol: EvaluationV2Protocol,
    repository_path: Path,
    calibration_observations_path: Path,
    calibration_item_manifest_path: Path,
    calibration_typo_manifest_path: Path,
    calibration_result_path: Path,
    confirmatory_item_manifest_path: Path,
    confirmatory_typo_manifest_path: Path,
    tier_role_manifest_path: Path,
    factorial_arm_registry_path: Path,
    probe_artifact_registry_path: Path,
    training_config_registry_path: Path,
    training_data_registry_path: Path,
    legacy_random_2_registry_path: Path,
    mistral_matched_seed_registry_path: Path,
    mistral_public_seed_1_checkpoint_path: Path,
    arm_checkpoint_registry_path: Path,
    opening_log_path: Path,
) -> Mapping[str, object]:
    """Require the post-training checkpoint inventory and opening log."""

    common = {
        "protocol": protocol,
        "repository_path": repository_path,
        "calibration_observations_path": calibration_observations_path,
        "calibration_item_manifest_path": calibration_item_manifest_path,
        "calibration_typo_manifest_path": calibration_typo_manifest_path,
        "calibration_result_path": calibration_result_path,
        "confirmatory_item_manifest_path": confirmatory_item_manifest_path,
        "confirmatory_typo_manifest_path": confirmatory_typo_manifest_path,
        "tier_role_manifest_path": tier_role_manifest_path,
        "factorial_arm_registry_path": factorial_arm_registry_path,
        "probe_artifact_registry_path": probe_artifact_registry_path,
        "training_config_registry_path": training_config_registry_path,
        "training_data_registry_path": training_data_registry_path,
        "legacy_random_2_registry_path": legacy_random_2_registry_path,
    }
    opening = _load_evaluation_v2_registry_phase(
        required_phase="evaluation-opening-sealed",
        registry_path=registry_path,
        post_training_artifact_paths={
            "mistral_kojima_faithful_matched_seed_42_43_44_registry_sha256": (
                mistral_matched_seed_registry_path
            ),
            "mistral_kojima_faithful_public_seed_1_anchor_checkpoint_sha256": (
                mistral_public_seed_1_checkpoint_path
            ),
            "arm_checkpoint_registry_sha256": arm_checkpoint_registry_path,
            "opening_log_sha256": opening_log_path,
        },
        **common,
    )
    training = load_training_preregistered_evaluation_v2_registry(
        registry_path=training_preregistered_registry_path,
        **common,
    )
    opening_confirmatory = opening["confirmatory"]
    training_confirmatory = training["confirmatory"]
    if not isinstance(opening_confirmatory, Mapping) or not isinstance(
        training_confirmatory, Mapping
    ):
        raise ValueError("evaluation v2 phase transition sections differ")
    if opening_confirmatory.get("training_preregistered_registry_sha256") != sha256_file(
        Path(training_preregistered_registry_path).resolve()
    ):
        raise ValueError("evaluation v2 training-preregistered phase binding differs")
    for field in (
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "model_inventory",
        "governance_attestation",
        "calibration",
        "legacy_random_2",
    ):
        if opening.get(field) != training.get(field):
            raise ValueError("evaluation v2 immutable phase binding differs")
    post_only_fields = {
        "status",
        "training_preregistered_registry_sha256",
        "mistral_kojima_faithful_matched_seed_42_43_44_registry_sha256",
        "mistral_kojima_faithful_public_seed_1_anchor_checkpoint_sha256",
        "arm_checkpoint_registry_sha256",
        "opening_log_sha256",
    }
    for field in set(training_confirmatory) - post_only_fields:
        if opening_confirmatory.get(field) != training_confirmatory.get(field):
            raise ValueError("evaluation v2 immutable confirmatory phase binding differs")
    return opening


def load_ready_evaluation_v2_registry(
    *,
    registry_path: Path,
    training_preregistered_registry_path: Path,
    protocol: EvaluationV2Protocol,
    repository_path: Path,
    calibration_observations_path: Path,
    calibration_item_manifest_path: Path,
    calibration_typo_manifest_path: Path,
    calibration_result_path: Path,
    confirmatory_item_manifest_path: Path,
    confirmatory_typo_manifest_path: Path,
    tier_role_manifest_path: Path,
    factorial_arm_registry_path: Path,
    probe_artifact_registry_path: Path,
    training_config_registry_path: Path,
    training_data_registry_path: Path,
    legacy_random_2_registry_path: Path,
    mistral_matched_seed_registry_path: Path,
    mistral_public_seed_1_checkpoint_path: Path,
    arm_checkpoint_registry_path: Path,
    opening_log_path: Path,
) -> Mapping[str, object]:
    """Backward-compatible alias for the evaluation-opening-sealed loader."""

    return load_evaluation_opening_sealed_evaluation_v2_registry(
        registry_path=registry_path,
        training_preregistered_registry_path=training_preregistered_registry_path,
        protocol=protocol,
        repository_path=repository_path,
        calibration_observations_path=calibration_observations_path,
        calibration_item_manifest_path=calibration_item_manifest_path,
        calibration_typo_manifest_path=calibration_typo_manifest_path,
        calibration_result_path=calibration_result_path,
        confirmatory_item_manifest_path=confirmatory_item_manifest_path,
        confirmatory_typo_manifest_path=confirmatory_typo_manifest_path,
        tier_role_manifest_path=tier_role_manifest_path,
        factorial_arm_registry_path=factorial_arm_registry_path,
        probe_artifact_registry_path=probe_artifact_registry_path,
        training_config_registry_path=training_config_registry_path,
        training_data_registry_path=training_data_registry_path,
        legacy_random_2_registry_path=legacy_random_2_registry_path,
        mistral_matched_seed_registry_path=mistral_matched_seed_registry_path,
        mistral_public_seed_1_checkpoint_path=mistral_public_seed_1_checkpoint_path,
        arm_checkpoint_registry_path=arm_checkpoint_registry_path,
        opening_log_path=opening_log_path,
    )


__all__ = [
    "ConfirmatorySemanticBinding",
    "canonical_source_tree_sha256",
    "load_confirmatory_semantic_binding",
    "load_evaluation_opening_sealed_evaluation_v2_registry",
    "load_ready_evaluation_v2_registry",
    "load_training_preregistered_evaluation_v2_registry",
    "validate_outcomes_against_confirmatory_binding",
    "validate_tier_id_disjointness",
]
