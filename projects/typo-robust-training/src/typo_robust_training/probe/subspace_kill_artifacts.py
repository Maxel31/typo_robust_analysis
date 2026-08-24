"""Strict content-addressed evidence for semantic-subspace causal sufficiency."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.perturb import (
    classify_character_edit,
    eligible_word_spans,
    is_keyboard_neighbor_substitution,
)
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.probe.artifacts import (
    ProbeTransitionArtifact,
    load_probe_transition_artifact,
    require_probe_artifact_child_eligibility,
)
from typo_robust_training.probe.attestation import (
    RuntimeCheckoutAttestation,
    attest_runtime_checkout,
)
from typo_robust_training.probe.subspace import (
    SemanticProbeSubspace,
    derive_artifact_semantic_subspace,
    derive_pca_basis,
    deterministic_complement_basis,
    deterministic_haar_basis,
)
from typo_robust_training.probe.subspace_kill_config import (
    SemanticSubspaceKillProtocol,
    load_semantic_subspace_kill_config,
)
from typo_robust_training.probe.subspace_kill_scoring import (
    SemanticSubspaceKillSummary,
    SubspaceKillScoreRow,
    score_semantic_subspace_kill,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REFERENCE_FIELDS = {"relative_path", "sha256"}
_COHORT_FIELDS = {
    "record_id",
    "pair_id",
    "source_group_sha256",
    "parent_source_sha256",
    "normalized_clean_sha256",
    "normalized_noisy_sha256",
    "clean_text",
    "typo_text",
    "clean_word_char_span",
    "typo_word_char_span",
    "clean_word_final_token",
    "typo_word_final_token",
    "edit_type",
    "edit_count",
    "source",
    "split",
}
_PCA_MANIFEST_FIELDS = {
    "record_id",
    "source_group_sha256",
    "parent_source_sha256",
    "normalized_clean_sha256",
    "activation_row",
}
_SCORE_ROW_FIELDS = {
    "pair_id",
    "source_group_sha256",
    "transition_layer",
    "clean_word_final_token",
    "typo_word_final_token",
    "untreated_kl_2_16",
    "patched_kl_2_16",
    "invalid_reason",
}


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _json(path: Path, *, field: str) -> Mapping[str, object]:
    try:
        value = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} must be UTF-8") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must contain one JSON object")
    return value


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _fit_records_sha256(parent: ProbeTransitionArtifact) -> str:
    payload = [
        {
            "record_id": row.record_id,
            "source_group_sha256": row.source_group_sha256,
            "parent_source_sha256": row.parent_source_sha256,
            "normalized_clean_sha256": row.normalized_clean_sha256,
            "clean_text": row.clean_text,
            "clean_word_char_span": list(row.clean_word_char_span),
        }
        for row in parent.fit_records
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reference(value: object, *, root: Path, field: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != _REFERENCE_FIELDS:
        raise ValueError(f"{field} reference fields differ")
    relative_raw = value["relative_path"]
    if not isinstance(relative_raw, str) or not relative_raw:
        raise ValueError(f"{field} relative path must be non-empty")
    relative = PurePosixPath(relative_raw)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != relative_raw:
        raise ValueError(f"{field} reference path differs")
    supplied = root / Path(*relative.parts)
    if supplied.is_symlink():
        raise ValueError(f"{field} reference must not be a symlink")
    path = supplied.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} reference escapes its bundle") from exc
    expected = _sha(value["sha256"], field=f"{field} hash")
    if not path.is_file() or _digest(path) != expected:
        raise ValueError(f"{field} reference is missing or its hash differs")
    return path, expected


def _span(value: object, *, text: str, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or not 0 <= value[0] < value[1] <= len(text)
        or any(character.isspace() for character in text[value[0] : value[1]])
        or (value[0] > 0 and text[value[0] - 1].isalnum())
        or (value[1] < len(text) and text[value[1]].isalnum())
    ):
        raise ValueError(f"{field} is not exactly one word span")
    return value[0], value[1]


@dataclass(frozen=True, slots=True)
class _KillCohortRow:
    pair_id: str
    source_group_sha256: str
    parent_source_sha256: str
    normalized_clean_sha256: str
    normalized_noisy_sha256: str
    clean_word_final_token: int
    typo_word_final_token: int


def _load_cohort(path: Path, *, parent: ProbeTransitionArtifact) -> tuple[_KillCohortRow, ...]:
    payload = _json(path, field="semantic kill cohort")
    if set(payload) != {"schema_version", "records"} or payload["schema_version"] != (
        "probe-semantic-subspace-kill-cohort/v1"
    ):
        raise ValueError("semantic kill cohort identity differs")
    raw_rows = payload["records"]
    if not isinstance(raw_rows, list) or len(raw_rows) != 200:
        raise ValueError("semantic kill cohort must contain exactly 200 pairs")
    rows: list[_KillCohortRow] = []
    identities: set[str] = set()
    record_ids: set[str] = set()
    pair_ids: set[str] = set()
    parent_to_group: dict[str, str] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != _COHORT_FIELDS:
            raise ValueError("semantic kill cohort record fields differ")
        record_id = _text(raw["record_id"], field="semantic kill record id")
        pair_id = _text(raw["pair_id"], field="semantic kill pair id")
        if record_id in record_ids or pair_id in pair_ids:
            raise ValueError("semantic kill record and pair ids must be unique")
        record_ids.add(record_id)
        pair_ids.add(pair_id)
        if raw["source"] != "fineweb-edu" or raw["split"] != "subspace-kill-test":
            raise ValueError("semantic kill cohort source or split differs")
        clean = _text(raw["clean_text"], field="semantic kill clean text")
        typo = _text(raw["typo_text"], field="semantic kill typo text")
        clean_hash = _sha(raw["normalized_clean_sha256"], field="semantic kill clean hash")
        noisy_hash = _sha(raw["normalized_noisy_sha256"], field="semantic kill typo hash")
        if (
            normalized_content_sha256(clean) != clean_hash
            or normalized_content_sha256(typo) != noisy_hash
            or clean == typo
        ):
            raise ValueError("semantic kill resolved text hashes differ")
        clean_span = _span(raw["clean_word_char_span"], text=clean, field="clean word span")
        typo_span = _span(raw["typo_word_char_span"], text=typo, field="typo word span")
        if clean_span not in eligible_word_spans(clean):
            raise ValueError("semantic kill clean word is outside generator eligibility")
        edit_type = _text(raw["edit_type"], field="semantic kill edit type")
        observed = classify_character_edit(
            clean=clean[slice(*clean_span)], typo=typo[slice(*typo_span)]
        )
        expected = (
            "natural-statistics-substitution"
            if edit_type == "keyboard-neighbor-substitution"
            else edit_type
        )
        if (
            edit_type not in {"keyboard-neighbor-substitution", "deletion", "duplication"}
            or observed != expected
            or _integer(raw["edit_count"], field="semantic kill edit count", minimum=1) != 1
            or clean[: clean_span[0]] != typo[: typo_span[0]]
            or clean[clean_span[1] :] != typo[typo_span[1] :]
        ):
            raise ValueError("semantic kill pair does not isolate one registered typo")
        if edit_type == "keyboard-neighbor-substitution" and not (
            is_keyboard_neighbor_substitution(
                clean=clean[slice(*clean_span)],
                typo=typo[slice(*typo_span)],
            )
        ):
            raise ValueError(
                "semantic kill keyboard substitution is not a case-preserving neighbor"
            )
        group = _sha(raw["source_group_sha256"], field="semantic kill source group")
        parent_hash = _sha(raw["parent_source_sha256"], field="semantic kill parent source")
        previous = parent_to_group.setdefault(parent_hash, group)
        if previous != group:
            raise ValueError("semantic kill parent maps to multiple source groups")
        row_identities = {group, parent_hash, clean_hash, noisy_hash}
        if identities & row_identities:
            raise ValueError("semantic kill cohort contains transitive duplicate identities")
        identities.update(row_identities)
        rows.append(
            _KillCohortRow(
                pair_id=pair_id,
                source_group_sha256=group,
                parent_source_sha256=parent_hash,
                normalized_clean_sha256=clean_hash,
                normalized_noisy_sha256=noisy_hash,
                clean_word_final_token=_integer(
                    raw["clean_word_final_token"], field="clean final token"
                ),
                typo_word_final_token=_integer(
                    raw["typo_word_final_token"], field="typo final token"
                ),
            )
        )
    if identities & parent.all_reserved_identities:
        raise ValueError("semantic kill cohort overlaps parent or protected tiers")
    return tuple(rows)


def _load_pca_manifest(
    path: Path,
    *,
    parent: ProbeTransitionArtifact,
) -> tuple[dict[str, object], ...]:
    payload = _json(path, field="semantic kill PCA manifest")
    if set(payload) != {"schema_version", "records"} or payload["schema_version"] != (
        "probe-semantic-pca-fit-manifest/v1"
    ):
        raise ValueError("semantic kill PCA manifest identity differs")
    raw_rows = payload["records"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("semantic kill PCA manifest must contain records")
    rows: list[dict[str, object]] = []
    identities: set[str] = set()
    for expected_row, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != _PCA_MANIFEST_FIELDS:
            raise ValueError("semantic kill PCA manifest record fields differ")
        if _integer(raw["activation_row"], field="PCA activation row") != expected_row:
            raise ValueError("semantic kill PCA activation rows must be contiguous")
        identity = {
            _sha(raw["source_group_sha256"], field="PCA source group"),
            _sha(raw["parent_source_sha256"], field="PCA parent source"),
            _sha(raw["normalized_clean_sha256"], field="PCA clean content"),
        }
        if identities & identity:
            raise ValueError("semantic kill PCA manifest identities repeat")
        identities.update(identity)
        rows.append(dict(raw))
    if identities != parent.cohort_identities_by_role["fit"]:
        raise ValueError("semantic kill PCA input is not exactly the parent fit cohort")
    expected_rows = parent.fit_records
    if len(rows) != len(expected_rows) or any(
        row["record_id"] != expected.record_id
        or row["source_group_sha256"] != expected.source_group_sha256
        or row["parent_source_sha256"] != expected.parent_source_sha256
        or row["normalized_clean_sha256"] != expected.normalized_clean_sha256
        for row, expected in zip(rows, expected_rows, strict=True)
    ):
        raise ValueError("semantic kill PCA rows do not exactly replay the parent fit manifest")
    return tuple(rows)


def _load_pca_activations(
    path: Path,
    *,
    protocol: SemanticSubspaceKillProtocol,
    parent: ProbeTransitionArtifact,
    rows: Sequence[Mapping[str, object]],
    checkout_attestation: RuntimeCheckoutAttestation,
) -> np.ndarray:
    from safetensors import safe_open

    expected_metadata = {
        "schema_version": "probe-semantic-pca-fit-activations/v2",
        "parent_artifact_sha256": parent.artifact_sha256,
        "pca_manifest_sha256": protocol.pca_manifest_sha256,
        "model": protocol.model,
        "loaded_model_revision": protocol.model_revision,
        "loaded_tokenizer_revision": protocol.model_revision,
        "parent_probe_code_revision": protocol.parent_probe_code_revision,
        "kill_runtime_code_revision": protocol.kill_runtime_code_revision,
        "typo_robust_training_tree": checkout_attestation.typo_robust_training_tree,
        "typo_cot_tree": checkout_attestation.typo_cot_tree,
        "transition_layer": str(parent.selected_transition_layer),
        "coordinate": protocol.coordinate,
        "hidden_size": str(parent.hidden_size),
        "rows": str(len(parent.fit_records)),
        "fit_records_sha256": _fit_records_sha256(parent),
    }
    with safe_open(path, framework="np") as handle:
        if dict(handle.metadata() or {}) != expected_metadata or set(handle.keys()) != {
            "clean_fit_activations"
        }:
            raise ValueError("semantic kill PCA activation provenance differs")
        activations = handle.get_tensor("clean_fit_activations")
    if (
        activations.shape != (len(rows), parent.hidden_size)
        or activations.dtype != np.float32
        or not np.isfinite(activations).all()
    ):
        raise ValueError("semantic kill PCA activation tensor differs")
    return np.ascontiguousarray(activations, dtype=np.float64)


def _load_subspaces(
    path: Path,
    *,
    protocol: SemanticSubspaceKillProtocol,
    parent: ProbeTransitionArtifact,
    pca_activations: np.ndarray,
    pca_activations_sha256: str,
    checkout_attestation: RuntimeCheckoutAttestation,
) -> tuple[Mapping[int, SemanticProbeSubspace], np.ndarray, np.ndarray, Mapping[int, np.ndarray]]:
    from safetensors import safe_open

    semantic = {
        seed: derive_artifact_semantic_subspace(parent, seed=seed, rank=protocol.rank)
        for seed in parent.probe_seeds
    }
    pca = derive_pca_basis(pca_activations, rank=protocol.rank)
    random = deterministic_haar_basis(
        parent.hidden_size, rank=protocol.rank, seed=protocol.random_basis_seed
    )
    complement = {
        seed: deterministic_complement_basis(
            semantic[seed].basis, seed=protocol.complement_basis_seed
        )
        for seed in parent.probe_seeds
    }
    expected_keys = {"pca_basis", "random_basis"}
    for seed in parent.probe_seeds:
        expected_keys.update(
            {
                f"seed.{seed}.semantic_basis",
                f"seed.{seed}.projected_class_weights",
                f"seed.{seed}.classifier_bias",
                f"seed.{seed}.semantic_complement_basis",
            }
        )
    expected_metadata = {
        "schema_version": "probe-semantic-subspaces/v1",
        "config_sha256": protocol.config_sha256,
        "parent_artifact_sha256": parent.artifact_sha256,
        "cohort_sha256": protocol.cohort_sha256,
        "pca_manifest_sha256": protocol.pca_manifest_sha256,
        "pca_activations_sha256": pca_activations_sha256,
        "model": protocol.model,
        "model_revision": protocol.model_revision,
        "parent_probe_code_revision": protocol.parent_probe_code_revision,
        "kill_runtime_code_revision": protocol.kill_runtime_code_revision,
        "typo_robust_training_tree": checkout_attestation.typo_robust_training_tree,
        "typo_cot_tree": checkout_attestation.typo_cot_tree,
        "transition_layer": str(parent.selected_transition_layer),
        "rank": str(protocol.rank),
        "random_basis_seed": str(protocol.random_basis_seed),
        "complement_basis_seed": str(protocol.complement_basis_seed),
    }
    with safe_open(path, framework="np") as handle:
        if (
            dict(handle.metadata() or {}) != expected_metadata
            or set(handle.keys()) != expected_keys
        ):
            raise ValueError("semantic kill stored subspace provenance differs")
        stored = {key: handle.get_tensor(key) for key in expected_keys}
    expected_arrays: dict[str, np.ndarray] = {
        "pca_basis": pca,
        "random_basis": random,
    }
    for seed in parent.probe_seeds:
        expected_arrays.update(
            {
                f"seed.{seed}.semantic_basis": semantic[seed].basis,
                f"seed.{seed}.projected_class_weights": semantic[seed].projected_class_weights,
                f"seed.{seed}.classifier_bias": semantic[seed].classifier_bias,
                f"seed.{seed}.semantic_complement_basis": complement[seed],
            }
        )
    for key, expected in expected_arrays.items():
        actual = stored[key]
        if (
            actual.dtype != np.float64
            or actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("semantic kill stored subspace differs from parent re-derivation")
    return MappingProxyType(semantic), pca, random, MappingProxyType(complement)


def _validate_runtime(
    path: Path,
    *,
    protocol: SemanticSubspaceKillProtocol,
    parent: ProbeTransitionArtifact,
    checkout_attestation: RuntimeCheckoutAttestation,
    pca_activations_sha256: str,
) -> Mapping[str, object]:
    payload = _json(path, field="semantic kill runtime provenance")
    from typo_cot.models.tokenizer_attestation import (
        validate_tokenizer_attestation_provenance,
    )

    tokenizer_attestation = payload.get("tokenizer_snapshot_attestation")
    validated_attestation = validate_tokenizer_attestation_provenance(
        tokenizer_attestation,
        expected_model=protocol.model,
        expected_revision=protocol.model_revision,
    )
    if (
        parent.tokenizer_snapshot_attestation is not None
        and dict(parent.tokenizer_snapshot_attestation) != validated_attestation.provenance_dict()
    ):
        raise ValueError("semantic kill tokenizer provenance differs from parent probe")
    expected = {
        "schema_version": "probe-semantic-subspace-kill-runtime/v2",
        "runtime": "HuggingFaceSemanticSubspaceKillRuntime/v2",
        "model": protocol.model,
        "loaded_model_revision": protocol.model_revision,
        "loaded_tokenizer_revision": protocol.model_revision,
        "tokenizer_snapshot_attestation": tokenizer_attestation,
        "parent_probe_code_revision": protocol.parent_probe_code_revision,
        "kill_runtime_code_revision": protocol.kill_runtime_code_revision,
        "checkout_attestation": checkout_attestation.as_dict(),
        "pca_fit_activations_sha256": pca_activations_sha256,
        "transition_layer": parent.selected_transition_layer,
        "hook_site": protocol.hook_site,
        "coordinate": protocol.coordinate,
        "operators": list(protocol.operators),
        "random_basis_seed": protocol.random_basis_seed,
        "complement_basis_seed": protocol.complement_basis_seed,
        "teacher_forced_offsets": list(protocol.readout_offsets),
    }
    if dict(payload) != expected:
        raise ValueError("semantic kill runtime provenance differs")
    return MappingProxyType(validated_attestation.provenance_dict())


def _load_scores(
    path: Path,
    *,
    seed: int,
    protocol: SemanticSubspaceKillProtocol,
    parent: ProbeTransitionArtifact,
    cohort: Sequence[_KillCohortRow],
    bindings: Mapping[str, str],
) -> tuple[SubspaceKillScoreRow, ...]:
    payload = _json(path, field=f"semantic kill seed {seed} scores")
    if set(payload) != {"schema_version", "seed", "bindings", "records"} or (
        payload["schema_version"] != "probe-semantic-subspace-kill-scores/v1"
        or _integer(payload["seed"], field="semantic kill score seed") != seed
        or not isinstance(payload["bindings"], Mapping)
        or dict(payload["bindings"]) != dict(bindings)
    ):
        raise ValueError("semantic kill score identity or bindings differ")
    raw_rows = payload["records"]
    if not isinstance(raw_rows, list) or len(raw_rows) != len(cohort):
        raise ValueError("semantic kill score coverage differs")
    expected = {row.pair_id: row for row in cohort}
    rows: list[SubspaceKillScoreRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != _SCORE_ROW_FIELDS:
            raise ValueError("semantic kill score record fields differ")
        pair_id = _text(raw["pair_id"], field="semantic kill score pair id")
        cohort_row = expected.get(pair_id)
        if cohort_row is None:
            raise ValueError("semantic kill score pair is outside the cohort")
        group = _sha(raw["source_group_sha256"], field="semantic kill score source group")
        clean_token = _integer(raw["clean_word_final_token"], field="score clean token")
        typo_token = _integer(raw["typo_word_final_token"], field="score typo token")
        if (
            group != cohort_row.source_group_sha256
            or clean_token != cohort_row.clean_word_final_token
            or typo_token != cohort_row.typo_word_final_token
        ):
            raise ValueError("semantic kill score uses the wrong token coordinate")
        patched_raw = raw["patched_kl_2_16"]
        if not isinstance(patched_raw, Mapping):
            raise ValueError("semantic kill patched scores must be an object")
        invalid = raw["invalid_reason"]
        if invalid is not None and not isinstance(invalid, str):
            raise ValueError("semantic kill invalid reason differs")
        rows.append(
            SubspaceKillScoreRow(
                pair_id=pair_id,
                source_group_sha256=group,
                transition_layer=_integer(raw["transition_layer"], field="score layer"),
                clean_word_final_token=clean_token,
                typo_word_final_token=typo_token,
                untreated_kl_2_16=tuple(raw["untreated_kl_2_16"]),  # type: ignore[arg-type]
                patched_kl_2_16={
                    str(operator): tuple(values)  # type: ignore[arg-type]
                    for operator, values in patched_raw.items()
                },
                invalid_reason=invalid,
            )
        )
    if len({row.pair_id for row in rows}) != len(cohort):
        raise ValueError("semantic kill scores do not cover each cohort pair exactly once")
    if any(row.transition_layer != parent.selected_transition_layer for row in rows):
        raise ValueError("semantic kill scores use the wrong transition layer")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class SemanticSubspaceKillArtifact:
    model: str
    model_revision: str
    parent_probe_code_revision: str
    kill_runtime_code_revision: str
    decoder_layers: int
    hidden_size: int
    transition_layer: int
    rank: int
    primary_probe_seed: int
    parent_probe_artifact: ProbeTransitionArtifact
    semantic_subspace: SemanticProbeSubspace
    summary_by_seed: Mapping[int, SemanticSubspaceKillSummary]
    artifact_sha256: str
    tokenizer_snapshot_attestation: Mapping[str, object]

    @property
    def suffix_layers(self) -> tuple[int, ...]:
        return tuple(range(self.transition_layer, self.decoder_layers))


def load_semantic_subspace_kill_artifact(
    path: Path,
    *,
    checkout_attestor: Callable[[str], RuntimeCheckoutAttestation] = attest_runtime_checkout,
) -> SemanticSubspaceKillArtifact:
    """Resolve every input and recompute both seed gates from raw KL outputs."""

    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("semantic kill artifact must be one regular file")
    resolved = supplied.resolve()
    raw = resolved.read_bytes()
    payload = _json(resolved, field="semantic kill artifact")
    expected_top = {
        "schema_version",
        "operation",
        "model",
        "model_revision",
        "parent_probe_code_revision",
        "kill_runtime_code_revision",
        "decoder_layers",
        "hidden_size",
        "transition_layer",
        "rank",
        "probe_seeds",
        "primary_probe_seed",
        "references",
        "kill_test_passed",
    }
    if set(payload) != expected_top or (
        payload["schema_version"] != "probe-semantic-subspace-kill-evidence/v2"
        or payload["operation"] != "validate-causal-probe-semantic-subspace"
    ):
        raise ValueError("semantic kill artifact identity differs")
    root = resolved.parent.resolve()
    references = payload["references"]
    if not isinstance(references, Mapping) or set(references) != {
        "config",
        "parent_probe_artifact",
        "cohort_manifest",
        "pca_fit_manifest",
        "pca_fit_activations",
        "subspaces",
        "runtime_provenance",
        "scores_by_seed",
    }:
        raise ValueError("semantic kill artifact references differ")
    config_path, config_hash = _reference(references["config"], root=root, field="kill config")
    protocol = load_semantic_subspace_kill_config(config_path)
    if config_hash != protocol.config_sha256:
        raise ValueError("semantic kill config hash differs")
    checkout = checkout_attestor(protocol.kill_runtime_code_revision)
    parent_path, parent_hash = _reference(
        references["parent_probe_artifact"], root=root, field="parent probe artifact"
    )
    parent = load_probe_transition_artifact(parent_path)
    require_probe_artifact_child_eligibility(parent)
    if parent_hash != parent.artifact_sha256 or parent_hash != protocol.parent_artifact_sha256:
        raise ValueError("semantic kill parent artifact binding differs")
    identity = {
        "model": protocol.model,
        "model_revision": protocol.model_revision,
        "parent_probe_code_revision": protocol.parent_probe_code_revision,
        "kill_runtime_code_revision": protocol.kill_runtime_code_revision,
        "decoder_layers": protocol.decoder_layers,
        "hidden_size": protocol.hidden_size,
        "transition_layer": parent.selected_transition_layer,
        "rank": protocol.rank,
        "probe_seeds": list(protocol.reproducibility_probe_seeds),
        "primary_probe_seed": protocol.primary_probe_seed,
    }
    if any(payload[field] != expected for field, expected in identity.items()) or (
        parent.model != protocol.model
        or parent.model_revision != protocol.model_revision
        or parent.code_revision != protocol.parent_probe_code_revision
        or parent.decoder_layers != protocol.decoder_layers
        or parent.hidden_size != protocol.hidden_size
        or parent.probe_seeds != protocol.reproducibility_probe_seeds
    ):
        raise ValueError("semantic kill model, parent, or subspace identity differs")
    cohort_path, cohort_hash = _reference(
        references["cohort_manifest"], root=root, field="kill cohort"
    )
    if cohort_hash != protocol.cohort_sha256:
        raise ValueError("semantic kill cohort hash differs")
    cohort = _load_cohort(cohort_path, parent=parent)
    pca_manifest_path, pca_manifest_hash = _reference(
        references["pca_fit_manifest"], root=root, field="PCA fit manifest"
    )
    if pca_manifest_hash != protocol.pca_manifest_sha256:
        raise ValueError("semantic kill PCA manifest hash differs")
    pca_rows = _load_pca_manifest(pca_manifest_path, parent=parent)
    pca_path, pca_hash = _reference(
        references["pca_fit_activations"], root=root, field="PCA fit activations"
    )
    pca_activations = _load_pca_activations(
        pca_path,
        protocol=protocol,
        parent=parent,
        rows=pca_rows,
        checkout_attestation=checkout,
    )
    subspaces_path, subspaces_hash = _reference(
        references["subspaces"], root=root, field="semantic subspaces"
    )
    semantic, _pca, _random, _complement = _load_subspaces(
        subspaces_path,
        protocol=protocol,
        parent=parent,
        pca_activations=pca_activations,
        pca_activations_sha256=pca_hash,
        checkout_attestation=checkout,
    )
    runtime_path, runtime_hash = _reference(
        references["runtime_provenance"], root=root, field="kill runtime"
    )
    tokenizer_snapshot_attestation = _validate_runtime(
        runtime_path,
        protocol=protocol,
        parent=parent,
        checkout_attestation=checkout,
        pca_activations_sha256=pca_hash,
    )
    score_refs = references["scores_by_seed"]
    expected_seed_keys = {str(seed) for seed in parent.probe_seeds}
    if not isinstance(score_refs, Mapping) or set(score_refs) != expected_seed_keys:
        raise ValueError("semantic kill score seed inventory differs")
    summaries: dict[int, SemanticSubspaceKillSummary] = {}
    rows_by_seed: dict[int, tuple[SubspaceKillScoreRow, ...]] = {}
    for seed in parent.probe_seeds:
        score_path, _score_hash = _reference(
            score_refs[str(seed)], root=root, field=f"kill seed {seed} scores"
        )
        rows_by_seed[seed] = _load_scores(
            score_path,
            seed=seed,
            protocol=protocol,
            parent=parent,
            cohort=cohort,
            bindings={
                "config_sha256": protocol.config_sha256,
                "parent_probe_artifact_sha256": parent.artifact_sha256,
                "probe_weights_sha256": _digest(parent.probe_weights_by_seed[seed]),
                "cohort_sha256": cohort_hash,
                "pca_manifest_sha256": pca_manifest_hash,
                "pca_activations_sha256": pca_hash,
                "subspaces_sha256": subspaces_hash,
                "runtime_provenance_sha256": runtime_hash,
            },
        )
        summaries[seed] = score_semantic_subspace_kill(
            rows_by_seed[seed], protocol=protocol, transition_layer=parent.selected_transition_layer
        )
    first, second = parent.probe_seeds
    invariant_operators = (
        "full-state",
        "clean-fit-pca-rank16",
        "deterministic-haar-random-rank16",
    )
    first_rows = {row.pair_id: row for row in rows_by_seed[first]}
    second_rows = {row.pair_id: row for row in rows_by_seed[second]}
    for pair_id, left in first_rows.items():
        right = second_rows[pair_id]
        if (
            left.invalid_reason != right.invalid_reason
            or left.untreated_kl_2_16 != right.untreated_kl_2_16
            or any(
                left.patched_kl_2_16.get(operator) != right.patched_kl_2_16.get(operator)
                for operator in invariant_operators
            )
        ):
            raise ValueError("seed-invariant semantic kill controls differ")
    if payload["kill_test_passed"] is not True or not all(
        summary.passed for summary in summaries.values()
    ):
        raise ValueError("semantic kill artifact did not pass both probe replications")
    return SemanticSubspaceKillArtifact(
        model=parent.model,
        model_revision=parent.model_revision,
        parent_probe_code_revision=parent.code_revision,
        kill_runtime_code_revision=protocol.kill_runtime_code_revision,
        decoder_layers=parent.decoder_layers,
        hidden_size=parent.hidden_size,
        transition_layer=parent.selected_transition_layer,
        rank=protocol.rank,
        primary_probe_seed=protocol.primary_probe_seed,
        parent_probe_artifact=parent,
        semantic_subspace=semantic[protocol.primary_probe_seed],
        summary_by_seed=MappingProxyType(summaries),
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        tokenizer_snapshot_attestation=tokenizer_snapshot_attestation,
    )


__all__ = ["SemanticSubspaceKillArtifact", "load_semantic_subspace_kill_artifact"]
