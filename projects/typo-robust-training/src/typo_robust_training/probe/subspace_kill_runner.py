"""Produce a self-contained semantic-subspace causal kill-test bundle."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np

from typo_robust_training.data.config import strict_loads
from typo_robust_training.integrity import sha256_file
from typo_robust_training.probe.artifacts import (
    ProbeTransitionArtifact,
    load_probe_transition_artifact,
)
from typo_robust_training.probe.subspace import (
    SemanticProbeSubspace,
    derive_artifact_semantic_subspace,
    derive_pca_basis,
    deterministic_complement_basis,
    deterministic_haar_basis,
)
from typo_robust_training.probe.subspace_kill_artifacts import (
    _load_cohort,
    _load_pca_activations,
    _load_pca_manifest,
    load_semantic_subspace_kill_artifact,
)
from typo_robust_training.probe.subspace_kill_config import (
    SemanticSubspaceKillProtocol,
    load_semantic_subspace_kill_config,
)
from typo_robust_training.probe.subspace_kill_runtime import (
    HuggingFaceSemanticSubspaceKillRuntime,
)
from typo_robust_training.probe.subspace_kill_scoring import (
    SemanticSubspaceKillSummary,
    SubspaceKillScoreRow,
    score_semantic_subspace_kill,
)
from typo_robust_training.training.json_io import write_json_durable


class _KillRuntime(Protocol):
    def scan_pair_all_seeds(
        self, record: Mapping[str, object]
    ) -> Mapping[int, SubspaceKillScoreRow]: ...

    def provenance(self) -> Mapping[str, object]: ...


RuntimeFactory = Callable[..., _KillRuntime]


@dataclass(frozen=True, slots=True)
class SemanticSubspaceKillRunConfig:
    config_path: Path
    parent_probe_artifact_path: Path
    cohort_manifest_path: Path
    pca_fit_manifest_path: Path
    pca_fit_activations_path: Path
    gpu_id: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class SemanticSubspaceKillRunResult:
    passed: bool
    artifact_path: Path
    subspaces_path: Path
    scores_by_seed: Mapping[int, Path]
    run_path: Path
    summaries: Mapping[int, SemanticSubspaceKillSummary]


def _json(path: Path, *, label: str) -> Mapping[str, object]:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be one regular file")
    try:
        value = strict_loads(supplied.read_text(encoding="utf-8"), context=str(supplied))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _copy_regular(source: Path, destination: Path, *, label: str) -> Path:
    supplied = Path(source)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be one regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(supplied, destination)
    if sha256_file(supplied) != sha256_file(destination):
        raise RuntimeError(f"{label} copy hash differs")
    return destination


def _reference(root: Path, path: Path) -> dict[str, str]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("semantic kill output reference escapes bundle") from exc
    return {"relative_path": relative.as_posix(), "sha256": sha256_file(resolved)}


def _parent_reference_paths(parent_artifact: Path) -> tuple[Path, ...]:
    """Resolve the direct content-addressed graph without trusting its filenames."""

    payload = _json(parent_artifact, label="parent probe artifact")
    references = payload.get("references")
    if not isinstance(references, Mapping):
        raise ValueError("parent probe artifact references differ")
    root = parent_artifact.resolve().parent
    result: list[Path] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping) and set(value) == {"relative_path", "sha256"}:
            relative_raw = value["relative_path"]
            expected = value["sha256"]
            if not isinstance(relative_raw, str) or not isinstance(expected, str):
                raise ValueError("parent probe reference differs")
            relative = PurePosixPath(relative_raw)
            if relative.is_absolute() or ".." in relative.parts or str(relative) != relative_raw:
                raise ValueError("parent probe reference path differs")
            supplied = root / Path(*relative.parts)
            if supplied.is_symlink() or not supplied.is_file():
                raise ValueError("parent probe reference is not one regular file")
            resolved = supplied.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError("parent probe reference escapes its bundle") from exc
            if sha256_file(resolved) != expected:
                raise ValueError("parent probe reference hash differs")
            result.append(resolved)
            return
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)

    visit(references)
    return tuple(dict.fromkeys(result))


def _copy_parent_bundle(parent_artifact: Path, destination: Path) -> Path:
    source_root = parent_artifact.resolve().parent
    destination.mkdir(parents=True, exist_ok=False)
    for source in _parent_reference_paths(parent_artifact):
        relative = source.relative_to(source_root)
        _copy_regular(source, destination / relative, label="parent probe reference")
    copied_artifact = destination / parent_artifact.name
    return _copy_regular(parent_artifact, copied_artifact, label="parent probe artifact")


def _cohort_records(path: Path) -> tuple[Mapping[str, object], ...]:
    payload = _json(path, label="semantic kill cohort")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("semantic kill cohort records differ")
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("semantic kill cohort record must be an object")
    return tuple(records)  # type: ignore[return-value]


def _write_subspaces(
    path: Path,
    *,
    protocol: SemanticSubspaceKillProtocol,
    parent: ProbeTransitionArtifact,
    semantic: Mapping[int, SemanticProbeSubspace],
    pca: np.ndarray,
    random: np.ndarray,
    complement: Mapping[int, np.ndarray],
) -> None:
    from safetensors.numpy import save_file

    tensors: dict[str, np.ndarray] = {
        "pca_basis": np.ascontiguousarray(pca, dtype=np.float64),
        "random_basis": np.ascontiguousarray(random, dtype=np.float64),
    }
    for seed in parent.probe_seeds:
        tensors.update(
            {
                f"seed.{seed}.semantic_basis": semantic[seed].basis,
                f"seed.{seed}.projected_class_weights": semantic[
                    seed
                ].projected_class_weights,
                f"seed.{seed}.classifier_bias": semantic[seed].classifier_bias,
                f"seed.{seed}.semantic_complement_basis": complement[seed],
            }
        )
    metadata = {
        "schema_version": "probe-semantic-subspaces/v1",
        "config_sha256": protocol.config_sha256,
        "parent_artifact_sha256": parent.artifact_sha256,
        "cohort_sha256": protocol.cohort_sha256,
        "pca_manifest_sha256": protocol.pca_manifest_sha256,
        "pca_activations_sha256": protocol.pca_activations_sha256,
        "model": protocol.model,
        "model_revision": protocol.model_revision,
        "code_revision": protocol.code_revision,
        "transition_layer": str(parent.selected_transition_layer),
        "rank": str(protocol.rank),
        "random_basis_seed": str(protocol.random_basis_seed),
        "complement_basis_seed": str(protocol.complement_basis_seed),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _score_payload(
    rows: Sequence[SubspaceKillScoreRow],
    *,
    seed: int,
    bindings: Mapping[str, str],
) -> Mapping[str, object]:
    return {
        "schema_version": "probe-semantic-subspace-kill-scores/v1",
        "seed": seed,
        "bindings": dict(bindings),
        "records": [
            {
                "pair_id": row.pair_id,
                "source_group_sha256": row.source_group_sha256,
                "transition_layer": row.transition_layer,
                "clean_word_final_token": row.clean_word_final_token,
                "typo_word_final_token": row.typo_word_final_token,
                "untreated_kl_2_16": list(row.untreated_kl_2_16),
                "patched_kl_2_16": {
                    operator: list(values)
                    for operator, values in row.patched_kl_2_16.items()
                },
                "invalid_reason": row.invalid_reason,
            }
            for row in rows
        ],
    }


def _summary_payload(summary: SemanticSubspaceKillSummary) -> Mapping[str, object]:
    return {
        "records": summary.records,
        "valid_records": summary.valid_records,
        "restoration": dict(summary.restoration),
        "ci_lower": dict(summary.ci_lower),
        "semantic_full_ratio_ci_lower": summary.semantic_full_ratio_ci_lower,
        "semantic_minus_control_ci_lower": dict(summary.semantic_minus_control_ci_lower),
        "passed": summary.passed,
    }


def run_semantic_subspace_kill_test(
    config: SemanticSubspaceKillRunConfig,
    *,
    runtime_factory: RuntimeFactory = HuggingFaceSemanticSubspaceKillRuntime,
) -> SemanticSubspaceKillRunResult:
    """Run once from preregistered inputs and write raw, independently reloadable evidence."""

    protocol = load_semantic_subspace_kill_config(config.config_path)
    parent = load_probe_transition_artifact(config.parent_probe_artifact_path)
    if parent.artifact_sha256 != protocol.parent_artifact_sha256:
        raise ValueError("semantic kill parent artifact differs from preregistration")
    if sha256_file(config.cohort_manifest_path) != protocol.cohort_sha256:
        raise ValueError("semantic kill cohort differs from preregistration")
    if sha256_file(config.pca_fit_manifest_path) != protocol.pca_manifest_sha256:
        raise ValueError("semantic kill PCA manifest differs from preregistration")
    if sha256_file(config.pca_fit_activations_path) != protocol.pca_activations_sha256:
        raise ValueError("semantic kill PCA activations differ from preregistration")

    # These parsers perform every disjointness, coordinate, and PCA provenance check
    # before the runtime constructor is allowed to initialize CUDA.
    cohort = _load_cohort(config.cohort_manifest_path, parent=parent)
    pca_rows = _load_pca_manifest(config.pca_fit_manifest_path, parent=parent)
    pca_activations = _load_pca_activations(
        config.pca_fit_activations_path,
        protocol=protocol,
        parent=parent,
        rows=pca_rows,
    )
    records = _cohort_records(config.cohort_manifest_path)
    if len(cohort) != len(records):
        raise ValueError("semantic kill parsed cohort coverage differs")

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

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        inputs = output / "inputs"
        copied_config = _copy_regular(
            config.config_path, inputs / "kill-config.json", label="semantic kill config"
        )
        copied_cohort = _copy_regular(
            config.cohort_manifest_path,
            inputs / "cohort.json",
            label="semantic kill cohort",
        )
        copied_pca_manifest = _copy_regular(
            config.pca_fit_manifest_path,
            inputs / "pca-fit-manifest.json",
            label="semantic kill PCA manifest",
        )
        copied_pca_activations = _copy_regular(
            config.pca_fit_activations_path,
            inputs / "pca-fit-activations.safetensors",
            label="semantic kill PCA activations",
        )
        copied_parent = _copy_parent_bundle(
            config.parent_probe_artifact_path, output / "parent-probe"
        )
        subspaces_path = output / "subspaces.safetensors"
        _write_subspaces(
            subspaces_path,
            protocol=protocol,
            parent=parent,
            semantic=semantic,
            pca=pca,
            random=random,
            complement=complement,
        )

        runtime = runtime_factory(
            protocol=protocol,
            parent=parent,
            semantic_by_seed=semantic,
            pca_basis=pca,
            random_basis=random,
            complement_by_seed=complement,
            gpu_id=config.gpu_id,
        )
        runtime_path = output / "runtime.json"
        write_json_durable(runtime_path, dict(runtime.provenance()))
        rows_by_seed: dict[int, list[SubspaceKillScoreRow]] = {
            seed: [] for seed in parent.probe_seeds
        }
        for record in records:
            scanned = runtime.scan_pair_all_seeds(record)
            if set(scanned) != set(parent.probe_seeds):
                raise RuntimeError("semantic kill runtime seed inventory differs")
            for seed in parent.probe_seeds:
                rows_by_seed[seed].append(scanned[seed])

        common_bindings = {
            "config_sha256": protocol.config_sha256,
            "parent_probe_artifact_sha256": parent.artifact_sha256,
            "cohort_sha256": protocol.cohort_sha256,
            "pca_manifest_sha256": protocol.pca_manifest_sha256,
            "pca_activations_sha256": protocol.pca_activations_sha256,
            "subspaces_sha256": sha256_file(subspaces_path),
            "runtime_provenance_sha256": sha256_file(runtime_path),
        }
        score_paths: dict[int, Path] = {}
        summaries: dict[int, SemanticSubspaceKillSummary] = {}
        for seed in parent.probe_seeds:
            score_path = output / f"scores-seed-{seed}.json"
            bindings = {
                **common_bindings,
                "probe_weights_sha256": sha256_file(parent.probe_weights_by_seed[seed]),
            }
            write_json_durable(
                score_path,
                _score_payload(rows_by_seed[seed], seed=seed, bindings=bindings),
            )
            score_paths[seed] = score_path
            summaries[seed] = score_semantic_subspace_kill(
                rows_by_seed[seed],
                protocol=protocol,
                transition_layer=parent.selected_transition_layer,
            )
        passed = all(summary.passed for summary in summaries.values())
        artifact_path = output / "semantic-subspace-kill-evidence.json"
        write_json_durable(
            artifact_path,
            {
                "schema_version": "probe-semantic-subspace-kill-evidence/v1",
                "operation": "validate-causal-probe-semantic-subspace",
                "model": parent.model,
                "model_revision": parent.model_revision,
                "code_revision": parent.code_revision,
                "decoder_layers": parent.decoder_layers,
                "hidden_size": parent.hidden_size,
                "transition_layer": parent.selected_transition_layer,
                "rank": protocol.rank,
                "probe_seeds": list(parent.probe_seeds),
                "primary_probe_seed": protocol.primary_probe_seed,
                "references": {
                    "config": _reference(output, copied_config),
                    "parent_probe_artifact": _reference(output, copied_parent),
                    "cohort_manifest": _reference(output, copied_cohort),
                    "pca_fit_manifest": _reference(output, copied_pca_manifest),
                    "pca_fit_activations": _reference(output, copied_pca_activations),
                    "subspaces": _reference(output, subspaces_path),
                    "runtime_provenance": _reference(output, runtime_path),
                    "scores_by_seed": {
                        str(seed): _reference(output, score_paths[seed])
                        for seed in parent.probe_seeds
                    },
                },
                "kill_test_passed": passed,
            },
        )
        run_path = output / "run.json"
        write_json_durable(
            run_path,
            {
                "schema_version": "probe-semantic-subspace-kill-run/v1",
                "operation": "run-semantic-subspace-kill-test",
                "inputs": {
                    "config_sha256": protocol.config_sha256,
                    "parent_probe_artifact_sha256": parent.artifact_sha256,
                    "cohort_sha256": protocol.cohort_sha256,
                    "pca_manifest_sha256": protocol.pca_manifest_sha256,
                    "pca_activations_sha256": protocol.pca_activations_sha256,
                },
                "outputs": {
                    "artifact_sha256": sha256_file(artifact_path),
                    "subspaces_sha256": sha256_file(subspaces_path),
                    "runtime_provenance_sha256": sha256_file(runtime_path),
                    "score_sha256_by_seed": {
                        str(seed): sha256_file(score_paths[seed]) for seed in parent.probe_seeds
                    },
                },
                "summaries": {
                    str(seed): _summary_payload(summaries[seed]) for seed in parent.probe_seeds
                },
                "passed": passed,
            },
        )
        if passed:
            # The consumer parser is intentionally stricter than this producer and
            # re-derives every gate from raw KL before evidence can reach training.
            verified = load_semantic_subspace_kill_artifact(artifact_path)
            if verified.artifact_sha256 != sha256_file(artifact_path):
                raise RuntimeError("semantic kill produced evidence failed round-trip identity")
        return SemanticSubspaceKillRunResult(
            passed=passed,
            artifact_path=artifact_path,
            subspaces_path=subspaces_path,
            scores_by_seed=dict(score_paths),
            run_path=run_path,
            summaries=dict(summaries),
        )
    except Exception:
        shutil.rmtree(output)
        raise


__all__ = [
    "SemanticSubspaceKillRunConfig",
    "SemanticSubspaceKillRunResult",
    "run_semantic_subspace_kill_test",
]
