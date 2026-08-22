from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.probe import subspace_kill_artifacts as kill_artifacts
from typo_robust_training.probe.artifacts import load_probe_transition_artifact
from typo_robust_training.probe.attestation import RuntimeCheckoutAttestation
from typo_robust_training.probe.subspace import (
    derive_artifact_semantic_subspace,
    derive_pca_basis,
    deterministic_complement_basis,
    deterministic_haar_basis,
)
from typo_robust_training.probe.subspace_kill_artifacts import (
    load_semantic_subspace_kill_artifact,
)
from typo_robust_training.probe.subspace_kill_config import (
    load_semantic_subspace_kill_config,
)
from typo_robust_training.probe.subspace_kill_runner import (
    SemanticSubspaceKillRunConfig,
    run_semantic_subspace_kill_test,
)
from typo_robust_training.probe.subspace_kill_scoring import SubspaceKillScoreRow


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ref(path: Path) -> dict[str, str]:
    return {"relative_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


_KILL_REVISION = "c" * 40
_ATTESTATION = RuntimeCheckoutAttestation(
    revision=_KILL_REVISION,
    typo_robust_training_tree="d" * 40,
    typo_cot_tree="e" * 40,
)


def _attest(revision: str) -> RuntimeCheckoutAttestation:
    if revision != _KILL_REVISION:
        raise ValueError("wrong test runtime revision")
    return _ATTESTATION


def _load_kill(path: Path):
    return load_semantic_subspace_kill_artifact(path, checkout_attestor=_attest)


def _captured_fit_activations(records: tuple[object, ...], hidden: int) -> np.ndarray:
    """Deterministic fake-runtime output; never accepted as a caller input."""

    rows: list[np.ndarray] = []
    for record in records:
        seed = int(getattr(record, "normalized_clean_sha256")[:16], 16)
        rows.append(np.random.default_rng(seed).normal(size=hidden).astype(np.float32))
    return np.stack(rows)


def _parent_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    layers, hidden, class_count = 5, 32, 17
    labels = tuple(f"aidentity{index:02d}" for index in range(class_count))
    files: dict[str, Path] = {}
    files["classes"] = _write(
        tmp_path / "classes.json",
        {
            "schema_version": "typo-word-identity-classes/v1",
            "classes": [
                {"class_id": index, "label": label} for index, label in enumerate(labels)
            ],
        },
    )
    manifests: dict[str, dict[str, object]] = {}
    for role in ("fit", "selection", "validation"):
        rows: list[dict[str, object]] = []
        for class_id, word in enumerate(labels):
            for repetition in range(3):
                record = f"{role}-{class_id}-{repetition}"
                clean = f"Generic context {record} contains {word} for testing."
                start = clean.index(word)
                row: dict[str, object] = {
                    "record_id": record,
                    "source_group_sha256": _sha(f"group-{record}"),
                    "parent_source_sha256": _sha(f"parent-{record}"),
                    "normalized_clean_sha256": normalized_content_sha256(clean),
                    "clean_text": clean,
                    "clean_word_char_span": [start, start + len(word)],
                    "class_id": class_id,
                }
                if role != "fit":
                    typo_word = (
                        "s" + word[1:]
                        if repetition == 0
                        else word[1:]
                        if repetition == 1
                        else word[0] + word
                    )
                    typo = clean[:start] + typo_word + clean[start + len(word) :]
                    row.update(
                        {
                            "pair_id": f"pair-{record}",
                            "normalized_noisy_sha256": normalized_content_sha256(typo),
                            "typo_text": typo,
                            "typo_word_char_span": [start, start + len(typo_word)],
                            "edit_type": (
                                "keyboard-neighbor-substitution",
                                "deletion",
                                "duplication",
                            )[repetition],
                            "edit_count": 1,
                            "token_inflation_bucket": ("same", "minus-one", "plus-one")[
                                repetition
                            ],
                        }
                    )
                rows.append(row)
        manifests[role] = {
            "schema_version": "typo-probe-cohort/v2",
            "role": role,
            "records": rows,
        }
        files[role] = _write(tmp_path / f"{role}.json", manifests[role])
    files["protected"] = _write(
        tmp_path / "protected.json",
        {
            "schema_version": "typo-protected-split-registry/v1",
            "registries": [
                {
                    "tier": tier,
                    "source_group_sha256": [_sha(f"protected-group-{tier}")],
                    "parent_source_sha256": [_sha(f"protected-parent-{tier}")],
                    "normalized_content_sha256": [_sha(f"protected-content-{tier}")],
                }
                for tier in ("training", "localization", "tune", "pre-pr", "sealed")
            ],
        },
    )
    input_hashes = {
        "class_inventory": _ref(files["classes"])["sha256"],
        "fit_manifest": _ref(files["fit"])["sha256"],
        "selection_manifest": _ref(files["selection"])["sha256"],
        "validation_manifest": _ref(files["validation"])["sha256"],
        "protected_split_registry": _ref(files["protected"])["sha256"],
    }
    files["config"] = _write(
        tmp_path / "probe-config.json",
        {
            "schema_version": "typo-linear-probe-producer-config/v2",
            "model": {
                "id": "google/gemma-3-4b-it",
                "revision": "a" * 40,
                "code_revision": "b" * 40,
                "decoder_layers": layers,
                "hidden_size": hidden,
                "dtype": "bfloat16",
            },
            "inputs": {
                "class_inventory_sha256": input_hashes["class_inventory"],
                "fit_manifest_sha256": input_hashes["fit_manifest"],
                "selection_manifest_sha256": input_hashes["selection_manifest"],
                "validation_manifest_sha256": input_hashes["validation_manifest"],
                "protected_registry_sha256": input_hashes["protected_split_registry"],
            },
            "cohorts": {
                "records_per_class": {role: 3 for role in ("fit", "selection", "validation")},
                "min_source_groups_per_class": {
                    role: 2 for role in ("fit", "selection", "validation")
                },
                "stratum_counts": {
                    role: {
                        "keyboard-neighbor-substitution|1|same": class_count,
                        "deletion|1|minus-one": class_count,
                        "duplication|1|plus-one": class_count,
                    }
                    for role in ("selection", "validation")
                },
            },
            "probe": {
                "seeds": [42, 43],
                "optimizer": "adamw",
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "epochs": 10,
                "batch_size": 51,
                "hook_site": "complete-decoder-block-residual-output",
                "coordinate": "edited-word-final-token/v1",
            },
            "selection": {
                "metric": "largest-group-mean-paired-noise-penalty-drop/v2",
                "rule": "min-argmax-over-layers-one-through-last/v1",
                "tie_break": "smallest-layer/v1",
                "stability_rule": (
                    "selection-exact-and-validation-within-one-layer-for-both-seeds/v1"
                ),
                "validation_rule": "group-bootstrap-95pct-lower-positive-for-both-seeds/v1",
                "bootstrap": {
                    "resamples": 10_000,
                    "seed": 1729,
                    "confidence": 0.95,
                    "unit": "source-group",
                },
            },
        },
    )
    config_hash = _ref(files["config"])["sha256"]
    for seed in (42, 43):
        rng = np.random.default_rng(seed)
        tensors: dict[str, np.ndarray] = {}
        for layer in range(layers):
            tensors[f"decoder_layer.{layer}.weight"] = rng.normal(
                size=(class_count, hidden)
            ).astype(np.float32)
            tensors[f"decoder_layer.{layer}.bias"] = rng.normal(size=(class_count,)).astype(
                np.float32
            )
        files[f"weights-{seed}"] = tmp_path / f"weights-{seed}.safetensors"
        save_file(
            tensors,
            files[f"weights-{seed}"],
            metadata={
                "schema_version": "typo-linear-probe-weights/v1",
                "seed": str(seed),
                "config_sha256": config_hash,
                "fit_manifest_sha256": input_hashes["fit_manifest"],
                "class_inventory_sha256": input_hashes["class_inventory"],
                "model": "google/gemma-3-4b-it",
                "model_revision": "a" * 40,
                "code_revision": "b" * 40,
                "decoder_layers": str(layers),
                "hidden_size": str(hidden),
                "class_count": str(class_count),
            },
        )
        weight_hash = _ref(files[f"weights-{seed}"])["sha256"]
        for role in ("selection", "validation"):
            score_rows = [
                {
                    "pair_id": row["pair_id"],
                    "source_group_sha256": row["source_group_sha256"],
                    "class_id": row["class_id"],
                    "edit_type": row["edit_type"],
                    "edit_count": row["edit_count"],
                    "token_inflation_bucket": row["token_inflation_bucket"],
                    "clean_cross_entropy": [1.0] * layers,
                    "noisy_cross_entropy": [2.0, 1.9, 1.2, 1.1, 1.0],
                }
                for row in manifests[role]["records"]  # type: ignore[index]
            ]
            files[f"{role}-{seed}"] = _write(
                tmp_path / f"{role}-{seed}.json",
                {
                    "schema_version": "typo-paired-probe-scores/v1",
                    "role": role,
                    "seed": seed,
                    "decoder_layers": layers,
                    "bindings": {
                        "model": "google/gemma-3-4b-it",
                        "model_revision": "a" * 40,
                        "code_revision": "b" * 40,
                        "config_sha256": config_hash,
                        "class_inventory_sha256": input_hashes["class_inventory"],
                        "fit_manifest_sha256": input_hashes["fit_manifest"],
                        "role_manifest_sha256": input_hashes[f"{role}_manifest"],
                        "probe_weights_sha256": weight_hash,
                    },
                    "records": score_rows,
                },
            )
    files["artifact"] = _write(
        tmp_path / "parent-probe.json",
        {
            "schema_version": "typo-denoising-probe-selection/v2",
            "operation": "select-linear-probe-denoising-transition",
            "model": "google/gemma-3-4b-it",
            "model_revision": "a" * 40,
            "decoder_layers": layers,
            "hook_site": "complete-decoder-block-residual-output",
            "coordinate": "edited-word-final-token/v1",
            "probe_seeds": [42, 43],
            "references": {
                "config": _ref(files["config"]),
                "class_inventory": _ref(files["classes"]),
                "fit_manifest": _ref(files["fit"]),
                "selection_manifest": _ref(files["selection"]),
                "validation_manifest": _ref(files["validation"]),
                "protected_split_registry": _ref(files["protected"]),
                "probe_weights_by_seed": {
                    str(seed): _ref(files[f"weights-{seed}"]) for seed in (42, 43)
                },
                "selection_scores_by_seed": {
                    str(seed): _ref(files[f"selection-{seed}"]) for seed in (42, 43)
                },
                "validation_scores_by_seed": {
                    str(seed): _ref(files[f"validation-{seed}"]) for seed in (42, 43)
                },
            },
            "selection_metric": "largest-group-mean-paired-noise-penalty-drop/v2",
            "selection_rule": "min-argmax-over-layers-one-through-last/v1",
            "tie_break": "smallest-layer/v1",
            "stability_rule": (
                "selection-exact-and-validation-within-one-layer-for-both-seeds/v1"
            ),
            "validation_rule": "group-bootstrap-95pct-lower-positive-for-both-seeds/v1",
            "bootstrap": {
                "resamples": 10_000,
                "seed": 1729,
                "confidence": 0.95,
                "unit": "source-group",
            },
            "selected_transition_layer": 2,
            "validation_passed": True,
        },
    )
    return files["artifact"], files


def _child_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, object]]:
    parent_path, parent_files = _parent_bundle(tmp_path)
    parent = load_probe_transition_artifact(parent_path)
    files = dict(parent_files)
    cohort_rows: list[dict[str, object]] = []
    for index in range(200):
        word = f"novelword{index:03d}"
        clean = f"Independent diagnostic {index} contains {word} and enough continuation tokens."
        start = clean.index(word)
        operation = index % 3
        typo_word = "m" + word[1:] if operation == 0 else word[1:] if operation == 1 else word[0] + word
        typo = clean[:start] + typo_word + clean[start + len(word) :]
        cohort_rows.append(
            {
                "record_id": f"kill-{index}",
                "pair_id": f"kill-pair-{index}",
                "source_group_sha256": _sha(f"kill-group-{index}"),
                "parent_source_sha256": _sha(f"kill-parent-{index}"),
                "normalized_clean_sha256": normalized_content_sha256(clean),
                "normalized_noisy_sha256": normalized_content_sha256(typo),
                "clean_text": clean,
                "typo_text": typo,
                "clean_word_char_span": [start, start + len(word)],
                "typo_word_char_span": [start, start + len(typo_word)],
                "clean_word_final_token": 8,
                "typo_word_final_token": 9,
                "edit_type": (
                    "keyboard-neighbor-substitution",
                    "deletion",
                    "duplication",
                )[operation],
                "edit_count": 1,
                "source": "fineweb-edu",
                "split": "subspace-kill-test",
            }
        )
    files["kill-cohort"] = _write(
        tmp_path / "kill-cohort.json",
        {"schema_version": "probe-semantic-subspace-kill-cohort/v1", "records": cohort_rows},
    )
    fit = json.loads(files["fit"].read_text())
    pca_manifest_rows = [
        {
            "record_id": row["record_id"],
            "source_group_sha256": row["source_group_sha256"],
            "parent_source_sha256": row["parent_source_sha256"],
            "normalized_clean_sha256": row["normalized_clean_sha256"],
            "activation_row": index,
        }
        for index, row in enumerate(fit["records"])
    ]
    files["pca-manifest"] = _write(
        tmp_path / "pca-manifest.json",
        {"schema_version": "probe-semantic-pca-fit-manifest/v1", "records": pca_manifest_rows},
    )
    activations = _captured_fit_activations(parent.fit_records, parent.hidden_size)
    files["pca"] = tmp_path / "pca.safetensors"
    save_file(
        {"clean_fit_activations": activations},
        files["pca"],
        metadata={
            "schema_version": "probe-semantic-pca-fit-activations/v2",
            "parent_artifact_sha256": parent.artifact_sha256,
            "pca_manifest_sha256": _ref(files["pca-manifest"])["sha256"],
            "model": parent.model,
            "loaded_model_revision": parent.model_revision,
            "loaded_tokenizer_revision": parent.model_revision,
            "parent_probe_code_revision": parent.code_revision,
            "kill_runtime_code_revision": _KILL_REVISION,
            "typo_robust_training_tree": _ATTESTATION.typo_robust_training_tree,
            "typo_cot_tree": _ATTESTATION.typo_cot_tree,
            "transition_layer": str(parent.selected_transition_layer),
            "coordinate": "edited-word-final-token/v1",
            "hidden_size": str(parent.hidden_size),
            "rows": str(len(parent.fit_records)),
            "fit_records_sha256": kill_artifacts._fit_records_sha256(parent),
        },
    )
    files["kill-config"] = _write(
        tmp_path / "kill-config.json",
        {
            "schema_version": "probe-semantic-subspace-kill-config/v2",
            "model": {
                "id": parent.model,
                "revision": parent.model_revision,
                "parent_probe_code_revision": parent.code_revision,
                "kill_runtime_code_revision": _KILL_REVISION,
                "decoder_layers": parent.decoder_layers,
                "hidden_size": parent.hidden_size,
                "dtype": "bfloat16",
            },
            "inputs": {
                "parent_probe_artifact_sha256": parent.artifact_sha256,
                "cohort_manifest_sha256": _ref(files["kill-cohort"])["sha256"],
                "pca_fit_manifest_sha256": _ref(files["pca-manifest"])["sha256"],
            },
            "subspace": {
                "rank": 16,
                "primary_probe_seed": 42,
                "reproducibility_probe_seeds": [42, 43],
                "centering": "subtract-class-mean-weight/v1",
                "svd_dtype": "float64",
                "basis_sign_rule": "largest-absolute-coordinate-positive/v1",
                "random_basis_seed": 101,
                "complement_basis_seed": 202,
            },
            "intervention": {
                "transition_layer_source": "parent-probe-selected-transition/v1",
                "hook_site": "complete-decoder-block-residual-output",
                "coordinate": "edited-word-final-token/v1",
                "patch_direction": "clean-to-typo",
                "operators": [
                    "untreated",
                    "full-state",
                    "semantic-rank16",
                    "clean-fit-pca-rank16",
                    "deterministic-haar-random-rank16",
                    "semantic-complement-rank16",
                ],
            },
            "readout": {
                "teacher_forced_tokens": 16,
                "offsets": list(range(2, 17)),
                "metric": "forward-kl-restoration-r2-through-r16/v1",
                "denominator_min_exclusive": 1e-9,
                "minimum_valid": 160,
                "minimum_valid_fraction": 0.8,
            },
            "bootstrap": {
                "resamples": 10_000,
                "seed": 1729,
                "confidence": 0.95,
                "unit": "source-group",
            },
            "gates": {
                "full_ci_lower_strictly_positive": True,
                "semantic_ci_lower_strictly_positive": True,
                "semantic_full_ratio_ci_lower": 0.5,
                "semantic_minus_each_control_ci_lower": 0.0,
                "both_probe_seeds_must_pass": True,
            },
        },
    )
    config_hash = _ref(files["kill-config"])["sha256"]
    semantic = {
        seed: derive_artifact_semantic_subspace(parent, seed=seed) for seed in parent.probe_seeds
    }
    subspace_tensors: dict[str, np.ndarray] = {
        "pca_basis": derive_pca_basis(activations),
        "random_basis": deterministic_haar_basis(parent.hidden_size, seed=101),
    }
    for seed in parent.probe_seeds:
        subspace_tensors.update(
            {
                f"seed.{seed}.semantic_basis": semantic[seed].basis,
                f"seed.{seed}.projected_class_weights": semantic[
                    seed
                ].projected_class_weights,
                f"seed.{seed}.classifier_bias": semantic[seed].classifier_bias,
                f"seed.{seed}.semantic_complement_basis": deterministic_complement_basis(
                    semantic[seed].basis, seed=202
                ),
            }
        )
    files["subspaces"] = tmp_path / "subspaces.safetensors"
    save_file(
        subspace_tensors,
        files["subspaces"],
        metadata={
            "schema_version": "probe-semantic-subspaces/v1",
            "config_sha256": config_hash,
            "parent_artifact_sha256": parent.artifact_sha256,
            "cohort_sha256": _ref(files["kill-cohort"])["sha256"],
            "pca_manifest_sha256": _ref(files["pca-manifest"])["sha256"],
            "pca_activations_sha256": _ref(files["pca"])["sha256"],
            "model": parent.model,
            "model_revision": parent.model_revision,
            "parent_probe_code_revision": parent.code_revision,
            "kill_runtime_code_revision": _KILL_REVISION,
            "typo_robust_training_tree": _ATTESTATION.typo_robust_training_tree,
            "typo_cot_tree": _ATTESTATION.typo_cot_tree,
            "transition_layer": str(parent.selected_transition_layer),
            "rank": "16",
            "random_basis_seed": "101",
            "complement_basis_seed": "202",
        },
    )
    files["runtime"] = _write(
        tmp_path / "runtime.json",
        {
            "schema_version": "probe-semantic-subspace-kill-runtime/v2",
            "runtime": "HuggingFaceSemanticSubspaceKillRuntime/v2",
            "model": parent.model,
            "loaded_model_revision": parent.model_revision,
            "loaded_tokenizer_revision": parent.model_revision,
            "parent_probe_code_revision": parent.code_revision,
            "kill_runtime_code_revision": _KILL_REVISION,
            "checkout_attestation": _ATTESTATION.as_dict(),
            "pca_fit_activations_sha256": _ref(files["pca"])["sha256"],
            "transition_layer": parent.selected_transition_layer,
            "hook_site": "complete-decoder-block-residual-output",
            "coordinate": "edited-word-final-token/v1",
            "operators": [
                "untreated",
                "full-state",
                "semantic-rank16",
                "clean-fit-pca-rank16",
                "deterministic-haar-random-rank16",
                "semantic-complement-rank16",
            ],
            "random_basis_seed": 101,
            "complement_basis_seed": 202,
            "teacher_forced_offsets": list(range(2, 17)),
        },
    )
    score_bindings = {
        "config_sha256": config_hash,
        "parent_probe_artifact_sha256": parent.artifact_sha256,
        "cohort_sha256": _ref(files["kill-cohort"])["sha256"],
        "pca_manifest_sha256": _ref(files["pca-manifest"])["sha256"],
        "pca_activations_sha256": _ref(files["pca"])["sha256"],
        "subspaces_sha256": _ref(files["subspaces"])["sha256"],
        "runtime_provenance_sha256": _ref(files["runtime"])["sha256"],
    }
    for seed in parent.probe_seeds:
        rows = []
        for cohort in cohort_rows:
            rows.append(
                {
                    "pair_id": cohort["pair_id"],
                    "source_group_sha256": cohort["source_group_sha256"],
                    "transition_layer": parent.selected_transition_layer,
                    "clean_word_final_token": cohort["clean_word_final_token"],
                    "typo_word_final_token": cohort["typo_word_final_token"],
                    "untreated_kl_2_16": [1.0] * 15,
                    "patched_kl_2_16": {
                        "full-state": [0.2] * 15,
                        "semantic-rank16": [0.4] * 15,
                        "clean-fit-pca-rank16": [0.8] * 15,
                        "deterministic-haar-random-rank16": [0.9] * 15,
                        "semantic-complement-rank16": [0.85] * 15,
                    },
                    "invalid_reason": None,
                }
            )
        files[f"kill-scores-{seed}"] = _write(
            tmp_path / f"kill-scores-{seed}.json",
            {
                "schema_version": "probe-semantic-subspace-kill-scores/v1",
                "seed": seed,
                "bindings": {
                    **score_bindings,
                    "probe_weights_sha256": _ref(parent.probe_weights_by_seed[seed])["sha256"],
                },
                "records": rows,
            },
        )
    artifact_payload: dict[str, object] = {
        "schema_version": "probe-semantic-subspace-kill-evidence/v2",
        "operation": "validate-causal-probe-semantic-subspace",
        "model": parent.model,
        "model_revision": parent.model_revision,
        "parent_probe_code_revision": parent.code_revision,
        "kill_runtime_code_revision": _KILL_REVISION,
        "decoder_layers": parent.decoder_layers,
        "hidden_size": parent.hidden_size,
        "transition_layer": parent.selected_transition_layer,
        "rank": 16,
        "probe_seeds": [42, 43],
        "primary_probe_seed": 42,
        "references": {
            "config": _ref(files["kill-config"]),
            "parent_probe_artifact": _ref(parent_path),
            "cohort_manifest": _ref(files["kill-cohort"]),
            "pca_fit_manifest": _ref(files["pca-manifest"]),
            "pca_fit_activations": _ref(files["pca"]),
            "subspaces": _ref(files["subspaces"]),
            "runtime_provenance": _ref(files["runtime"]),
            "scores_by_seed": {
                str(seed): _ref(files[f"kill-scores-{seed}"]) for seed in parent.probe_seeds
            },
        },
        "kill_test_passed": True,
    }
    files["kill-artifact"] = _write(tmp_path / "kill-artifact.json", artifact_payload)
    return files["kill-artifact"], files, artifact_payload


def test_loader_rederives_bases_and_both_seed_gates_from_raw_kl(tmp_path: Path) -> None:
    path, _files, _payload = _child_bundle(tmp_path)

    artifact = _load_kill(path)

    assert artifact.transition_layer == 2
    assert artifact.suffix_layers == (2, 3, 4)
    assert artifact.semantic_subspace.rank == 16
    assert artifact.parent_probe_code_revision == "b" * 40
    assert artifact.kill_runtime_code_revision == _KILL_REVISION
    assert artifact.parent_probe_code_revision != artifact.kill_runtime_code_revision
    assert all(summary.passed for summary in artifact.summary_by_seed.values())
    assert artifact.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_rejects_random_basis_disguised_as_parent_semantics(tmp_path: Path) -> None:
    path, files, payload = _child_bundle(tmp_path)
    from safetensors import safe_open

    with safe_open(files["subspaces"], framework="np") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = dict(handle.metadata() or {})
    tensors["seed.42.semantic_basis"] = tensors["random_basis"].copy()
    save_file(tensors, files["subspaces"], metadata=metadata)
    payload["references"]["subspaces"] = _ref(files["subspaces"])  # type: ignore[index]
    _write(path, payload)

    with pytest.raises(ValueError, match="differs from parent re-derivation"):
        _load_kill(path)


def test_loader_rejects_random_gaussian_substituted_for_runtime_pca_source(
    tmp_path: Path,
) -> None:
    path, files, payload = _child_bundle(tmp_path)
    from safetensors import safe_open

    with safe_open(files["pca"], framework="np") as handle:
        metadata = dict(handle.metadata() or {})
        original = handle.get_tensor("clean_fit_activations")
    gaussian = np.random.default_rng(999).normal(size=original.shape).astype(np.float32)
    save_file({"clean_fit_activations": gaussian}, files["pca"], metadata=metadata)
    payload["references"]["pca_fit_activations"] = _ref(files["pca"])  # type: ignore[index]
    _write(path, payload)

    with pytest.raises(ValueError, match="stored subspace"):
        _load_kill(path)


def test_loader_rejects_wrong_token_coordinate_before_accepting_stored_pass(tmp_path: Path) -> None:
    path, files, payload = _child_bundle(tmp_path)
    score = json.loads(files["kill-scores-42"].read_text())
    score["records"][0]["typo_word_final_token"] += 1
    _write(files["kill-scores-42"], score)
    payload["references"]["scores_by_seed"]["42"] = _ref(files["kill-scores-42"])  # type: ignore[index]
    _write(path, payload)

    with pytest.raises(ValueError, match="wrong token coordinate"):
        _load_kill(path)


def test_loader_rejects_one_seed_only_pass(tmp_path: Path) -> None:
    path, files, payload = _child_bundle(tmp_path)
    score = json.loads(files["kill-scores-43"].read_text())
    for row in score["records"]:
        row["patched_kl_2_16"]["semantic-rank16"] = [0.9] * 15
    _write(files["kill-scores-43"], score)
    payload["references"]["scores_by_seed"]["43"] = _ref(files["kill-scores-43"])  # type: ignore[index]
    _write(path, payload)

    with pytest.raises(ValueError, match="did not pass both"):
        _load_kill(path)


def test_kill_config_rejects_substituted_probe_seeds(tmp_path: Path) -> None:
    _path, files, _payload = _child_bundle(tmp_path)
    config = json.loads(files["kill-config"].read_text())
    config["subspace"]["primary_probe_seed"] = 1
    config["subspace"]["reproducibility_probe_seeds"] = [1, 2]
    _write(files["kill-config"], config)

    with pytest.raises(ValueError, match="frozen probe seeds 42 and 43"):
        load_semantic_subspace_kill_config(files["kill-config"])


def test_cohort_overlap_with_any_parent_identity_is_rejected(tmp_path: Path) -> None:
    _path, files, _payload = _child_bundle(tmp_path)
    parent = load_probe_transition_artifact(files["artifact"])
    cohort = json.loads(files["kill-cohort"].read_text())
    cohort["records"][0]["source_group_sha256"] = next(
        iter(parent.cohort_identities_by_role["fit"])
    )
    _write(files["kill-cohort"], cohort)

    with pytest.raises(ValueError, match="overlaps parent"):
        kill_artifacts._load_cohort(files["kill-cohort"], parent=parent)


class _PassingKillRuntime:
    def __init__(self, **kwargs: object) -> None:
        self.protocol = kwargs["protocol"]
        self.parent = kwargs["parent"]
        self.checkout = kwargs["checkout_attestation"]
        self.pca_basis = None

    def collect_clean_fit_activations(self, records):
        assert records == self.parent.fit_records
        return _captured_fit_activations(records, self.parent.hidden_size)

    def bind_pca_basis(self, basis: np.ndarray) -> None:
        self.pca_basis = basis.copy()

    def provenance(self, *, pca_fit_activations_sha256: str) -> dict[str, object]:
        protocol = self.protocol
        parent = self.parent
        return {
            "schema_version": "probe-semantic-subspace-kill-runtime/v2",
            "runtime": "HuggingFaceSemanticSubspaceKillRuntime/v2",
            "model": protocol.model,
            "loaded_model_revision": protocol.model_revision,
            "loaded_tokenizer_revision": protocol.model_revision,
            "parent_probe_code_revision": protocol.parent_probe_code_revision,
            "kill_runtime_code_revision": protocol.kill_runtime_code_revision,
            "checkout_attestation": self.checkout.as_dict(),
            "pca_fit_activations_sha256": pca_fit_activations_sha256,
            "transition_layer": parent.selected_transition_layer,
            "hook_site": protocol.hook_site,
            "coordinate": protocol.coordinate,
            "operators": list(protocol.operators),
            "random_basis_seed": protocol.random_basis_seed,
            "complement_basis_seed": protocol.complement_basis_seed,
            "teacher_forced_offsets": list(protocol.readout_offsets),
        }

    def scan_pair_all_seeds(
        self, record: dict[str, object]
    ) -> dict[int, SubspaceKillScoreRow]:
        row = {
            "pair_id": record["pair_id"],
            "source_group_sha256": record["source_group_sha256"],
            "transition_layer": self.parent.selected_transition_layer,
            "clean_word_final_token": record["clean_word_final_token"],
            "typo_word_final_token": record["typo_word_final_token"],
            "untreated_kl_2_16": (1.0,) * 15,
            "patched_kl_2_16": {
                "full-state": (0.2,) * 15,
                "semantic-rank16": (0.4,) * 15,
                "clean-fit-pca-rank16": (0.8,) * 15,
                "deterministic-haar-random-rank16": (0.9,) * 15,
                "semantic-complement-rank16": (0.85,) * 15,
            },
        }
        return {seed: SubspaceKillScoreRow(**row) for seed in self.parent.probe_seeds}


def test_runner_produces_self_contained_revalidated_evidence(tmp_path: Path) -> None:
    _child, files, _payload = _child_bundle(tmp_path)
    output = tmp_path / "produced"
    # A pre-existing arbitrary activation tensor is not an input to the runner.
    files["pca"].unlink()

    result = run_semantic_subspace_kill_test(
        SemanticSubspaceKillRunConfig(
            config_path=files["kill-config"],
            parent_probe_artifact_path=files["artifact"],
            cohort_manifest_path=files["kill-cohort"],
            pca_fit_manifest_path=files["pca-manifest"],
            gpu_id="7",
            output_dir=output,
        ),
        runtime_factory=_PassingKillRuntime,
        checkout_attestor=_attest,
    )

    assert result.passed is True
    reloaded = _load_kill(result.artifact_path)
    assert reloaded.transition_layer == 2
    assert all(summary.passed for summary in reloaded.summary_by_seed.values())
    shutil_source = files["weights-42"]
    shutil_source.unlink()
    assert _load_kill(result.artifact_path).artifact_sha256 == (
        reloaded.artifact_sha256
    )


def test_runner_api_makes_caller_supplied_pca_activations_impossible(tmp_path: Path) -> None:
    _child, files, _payload = _child_bundle(tmp_path)

    with pytest.raises(TypeError, match="unexpected keyword"):
        SemanticSubspaceKillRunConfig(
            config_path=files["kill-config"],
            parent_probe_artifact_path=files["artifact"],
            cohort_manifest_path=files["kill-cohort"],
            pca_fit_manifest_path=files["pca-manifest"],
            pca_fit_activations_path=files["pca"],  # type: ignore[call-arg]
            gpu_id="7",
            output_dir=tmp_path / "must-not-run",
        )
