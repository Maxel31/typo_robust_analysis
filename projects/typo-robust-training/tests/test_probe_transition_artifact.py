from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from typo_robust_training.probe import artifacts as probe_artifacts
from typo_robust_training.probe.artifacts import load_probe_transition_artifact
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.training.methods import load_probe_transition_training_evidence


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ref(path: Path) -> dict[str, str]:
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest(role: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for class_id in range(2):
        for repetition in range(3):
            label = f"{role}-{class_id}-{repetition}"
            clean_word = ("alpha", "beta")[class_id]
            clean_text = f"Context {label} contains {clean_word} today."
            clean_start = clean_text.index(clean_word)
            row: dict[str, object] = {
                "record_id": label,
                "source_group_sha256": _sha(f"group-{label}"),
                "parent_source_sha256": _sha(f"parent-{label}"),
                "normalized_clean_sha256": normalized_content_sha256(clean_text),
                "clean_text": clean_text,
                "clean_word_char_span": [clean_start, clean_start + len(clean_word)],
                "class_id": class_id,
            }
            if role != "fit":
                typo_word = (
                    ("slpha", "neta")[class_id]
                    if repetition == 0
                    else ("apha", "eta")[class_id]
                    if repetition == 1
                    else ("allpha", "betaa")[class_id]
                )
                typo_text = (
                    clean_text[:clean_start]
                    + typo_word
                    + clean_text[clean_start + len(clean_word) :]
                )
                row.update(
                    {
                        "pair_id": f"pair-{label}",
                        "normalized_noisy_sha256": normalized_content_sha256(typo_text),
                        "typo_text": typo_text,
                        "typo_word_char_span": [clean_start, clean_start + len(typo_word)],
                        "edit_type": (
                            "keyboard-neighbor-substitution",
                            "deletion",
                            "duplication",
                        )[repetition],
                        "edit_count": 1,
                        "token_inflation_bucket": ("same", "minus-one", "plus-one")[repetition],
                    }
                )
            rows.append(row)
    return {"schema_version": "typo-probe-cohort/v2", "role": role, "records": rows}


def _scores(
    role: str,
    seed: int,
    manifest: dict[str, object],
    *,
    bindings: dict[str, str],
    noisy: list[float] | None = None,
) -> dict[str, object]:
    noisy = noisy or [2.0, 1.9, 1.2, 1.1, 1.0]
    rows = []
    for cohort in manifest["records"]:  # type: ignore[index]
        rows.append(
            {
                "pair_id": cohort["pair_id"],
                "source_group_sha256": cohort["source_group_sha256"],
                "class_id": cohort["class_id"],
                "edit_type": cohort["edit_type"],
                "edit_count": cohort["edit_count"],
                "token_inflation_bucket": cohort["token_inflation_bucket"],
                "clean_cross_entropy": [1.0] * 5,
                "noisy_cross_entropy": noisy,
            }
        )
    return {
        "schema_version": "typo-paired-probe-scores/v1",
        "role": role,
        "seed": seed,
        "decoder_layers": 5,
        "bindings": bindings,
        "records": rows,
    }


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, Path]]:
    files: dict[str, Path] = {}
    files["classes"] = _write_json(
        tmp_path / "classes.json",
        {
            "schema_version": "typo-word-identity-classes/v1",
            "classes": [
                {"class_id": 0, "label": "alpha"},
                {"class_id": 1, "label": "beta"},
            ],
        },
    )
    manifests = {role: _manifest(role) for role in ("fit", "selection", "validation")}
    for role, manifest in manifests.items():
        files[role] = _write_json(tmp_path / f"{role}.json", manifest)
    files["protected"] = _write_json(
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
        "class_inventory": hashlib.sha256(files["classes"].read_bytes()).hexdigest(),
        "fit_manifest": hashlib.sha256(files["fit"].read_bytes()).hexdigest(),
        "selection_manifest": hashlib.sha256(files["selection"].read_bytes()).hexdigest(),
        "validation_manifest": hashlib.sha256(files["validation"].read_bytes()).hexdigest(),
        "protected_split_registry": hashlib.sha256(files["protected"].read_bytes()).hexdigest(),
    }
    files["config"] = _write_json(
        tmp_path / "config.json",
        {
            "schema_version": "typo-linear-probe-producer-config/v2",
            "model": {
                "id": "google/gemma-3-4b-it",
                "revision": "a" * 40,
                "code_revision": "b" * 40,
                "decoder_layers": 5,
                "hidden_size": 3,
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
                "records_per_class": {"fit": 3, "selection": 3, "validation": 3},
                "min_source_groups_per_class": {
                    "fit": 2,
                    "selection": 2,
                    "validation": 2,
                },
                "stratum_counts": {
                    role: {
                        "keyboard-neighbor-substitution|1|same": 2,
                        "deletion|1|minus-one": 2,
                        "duplication|1|plus-one": 2,
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
                "batch_size": 6,
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
                "validation_rule": (
                    "group-bootstrap-95pct-lower-positive-for-both-seeds/v1"
                ),
                "bootstrap": {
                    "resamples": 10_000,
                    "seed": 1729,
                    "confidence": 0.95,
                    "unit": "source-group",
                },
            },
        },
    )
    config_sha256 = hashlib.sha256(files["config"].read_bytes()).hexdigest()
    for seed in (42, 43):
        files[f"weights-{seed}"] = tmp_path / f"probe-{seed}.safetensors"
        save_file(
            {
                **{
                    f"decoder_layer.{layer}.weight": np.full(
                        (2, 3), seed + layer, dtype=np.float32
                    )
                    for layer in range(5)
                },
                **{
                    f"decoder_layer.{layer}.bias": np.full(
                        (2,), seed - layer, dtype=np.float32
                    )
                    for layer in range(5)
                },
            },
            files[f"weights-{seed}"],
            metadata={
                "schema_version": "typo-linear-probe-weights/v1",
                "seed": str(seed),
                "config_sha256": config_sha256,
                "fit_manifest_sha256": input_hashes["fit_manifest"],
                "class_inventory_sha256": input_hashes["class_inventory"],
                "model": "google/gemma-3-4b-it",
                "model_revision": "a" * 40,
                "code_revision": "b" * 40,
                "decoder_layers": "5",
                "hidden_size": "3",
                "class_count": "2",
            },
        )
        weight_sha256 = hashlib.sha256(files[f"weights-{seed}"].read_bytes()).hexdigest()
        for role in ("selection", "validation"):
            files[f"{role}-{seed}"] = _write_json(
                tmp_path / f"{role}-{seed}.json",
                _scores(
                    role,
                    seed,
                    manifests[role],
                    bindings={
                        "model": "google/gemma-3-4b-it",
                        "model_revision": "a" * 40,
                        "code_revision": "b" * 40,
                        "config_sha256": config_sha256,
                        "class_inventory_sha256": input_hashes["class_inventory"],
                        "fit_manifest_sha256": input_hashes["fit_manifest"],
                        "role_manifest_sha256": input_hashes[f"{role}_manifest"],
                        "probe_weights_sha256": weight_sha256,
                    },
                ),
            )
    payload: dict[str, object] = {
        "schema_version": "typo-denoising-probe-selection/v2",
        "operation": "select-linear-probe-denoising-transition",
        "model": "google/gemma-3-4b-it",
        "model_revision": "a" * 40,
        "decoder_layers": 5,
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
        "stability_rule": "selection-exact-and-validation-within-one-layer-for-both-seeds/v1",
        "validation_rule": "group-bootstrap-95pct-lower-positive-for-both-seeds/v1",
        "bootstrap": {
            "resamples": 10_000,
            "seed": 1729,
            "confidence": 0.95,
            "unit": "source-group",
        },
        "selected_transition_layer": 2,
        "validation_passed": True,
    }
    artifact = _write_json(tmp_path / "probe-transition.json", payload)
    return artifact, payload, files


def _refresh_ref(payload: dict[str, object], name: str, path: Path) -> None:
    payload["references"][name] = _ref(path)  # type: ignore[index]


def test_artifact_recomputes_selection_and_resolves_suffix(tmp_path: Path) -> None:
    path, _payload, _files = _bundle(tmp_path)

    result = load_probe_transition_artifact(path)

    assert result.selected_transition_layer == 2
    assert result.suffix_layers == (2, 3, 4)
    assert result.probe_seeds == (42, 43)
    assert all(value > 0.0 for value in result.validation_ci_lower_by_seed.values())
    assert result.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(result.cohort_identities_by_role) == {"fit", "selection", "validation"}
    assert result.protected_identities
    assert result.all_reserved_identities == frozenset().union(
        result.protected_identities,
        *result.cohort_identities_by_role.values(),
    )


def test_training_consumer_loads_the_real_validated_artifact_bundle(tmp_path: Path) -> None:
    artifact_path, _payload, _files = _bundle(tmp_path)
    validated = load_probe_transition_artifact(artifact_path)

    evidence = load_probe_transition_training_evidence(
        artifact_path,
        model="google/gemma-3-4b-it",
        model_revision="a" * 40,
        decoder_layers=5,
    )

    assert evidence.selected_transition_layer == validated.selected_transition_layer
    assert evidence.evidence_sha256 == validated.artifact_sha256


def test_artifact_rejects_unresolved_or_mutated_reference(tmp_path: Path) -> None:
    path, _payload, files = _bundle(tmp_path)
    files["selection"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_transitive_source_group_overlap(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    fit = json.loads(files["fit"].read_text())
    validation = json.loads(files["validation"].read_text())
    validation["records"][0]["source_group_sha256"] = fit["records"][0]["source_group_sha256"]
    _write_json(files["validation"], validation)
    _refresh_ref(payload, "validation_manifest", files["validation"])
    _write_json(path, payload)

    with pytest.raises(ValueError, match="overlap transitively"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_duplicate_content_under_distinct_groups(tmp_path: Path) -> None:
    _path, _payload, files = _bundle(tmp_path)
    selection = json.loads(files["selection"].read_text())
    first = selection["records"][0]
    duplicate = selection["records"][1]
    for field in (
        "normalized_clean_sha256",
        "clean_text",
        "clean_word_char_span",
        "normalized_noisy_sha256",
        "typo_text",
        "typo_word_char_span",
        "edit_type",
        "token_inflation_bucket",
    ):
        duplicate[field] = first[field]
    _write_json(files["selection"], selection)

    with pytest.raises(ValueError, match="unique within role"):
        probe_artifacts._load_manifest(  # noqa: SLF001 - loader boundary falsification
            files["selection"],
            expected_role="selection",
            class_labels=("alpha", "beta"),
        )


def test_artifact_rejects_identical_probe_tensors_with_distinct_seed_metadata(
    tmp_path: Path,
) -> None:
    path, payload, files = _bundle(tmp_path)
    with safe_open(files["weights-42"], framework="np") as source:
        tensors = {name: source.get_tensor(name) for name in source.keys()}
    with safe_open(files["weights-43"], framework="np") as target:
        metadata = dict(target.metadata())
    save_file(tensors, files["weights-43"], metadata=metadata)
    payload["references"]["probe_weights_by_seed"]["43"] = _ref(  # type: ignore[index]
        files["weights-43"]
    )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="identical numerical tensors"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_numerically_identical_signed_zero_tensors(
    tmp_path: Path,
) -> None:
    path, payload, files = _bundle(tmp_path)
    with safe_open(files["weights-42"], framework="np") as source:
        tensors_42 = {name: source.get_tensor(name) for name in source.keys()}
        metadata_42 = dict(source.metadata())
    with safe_open(files["weights-43"], framework="np") as target:
        metadata_43 = dict(target.metadata())
    tensors_42["decoder_layer.0.weight"].flat[0] = 0.0
    tensors_43 = {name: value.copy() for name, value in tensors_42.items()}
    tensors_43["decoder_layer.0.weight"].flat[0] = -0.0
    save_file(tensors_42, files["weights-42"], metadata=metadata_42)
    save_file(tensors_43, files["weights-43"], metadata=metadata_43)
    for seed in (42, 43):
        payload["references"]["probe_weights_by_seed"][str(seed)] = _ref(  # type: ignore[index]
            files[f"weights-{seed}"]
        )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="identical numerical tensors"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_protected_split_overlap(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    fit = json.loads(files["fit"].read_text())
    protected = json.loads(files["protected"].read_text())
    protected["registries"][0]["parent_source_sha256"] = [fit["records"][0]["parent_source_sha256"]]
    _write_json(files["protected"], protected)
    _refresh_ref(payload, "protected_split_registry", files["protected"])
    _write_json(path, payload)

    with pytest.raises(ValueError, match="protected training or evaluation split"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_seed_unstable_selection_boundary(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    manifest = json.loads(files["selection"].read_text())
    _write_json(
        files["selection-43"],
        _scores(
            "selection",
            43,
            manifest,
            bindings=json.loads(files["selection-43"].read_text())["bindings"],
            noisy=[2.0, 1.4, 1.2, 1.1, 1.0],
        ),
    )
    payload["references"]["selection_scores_by_seed"]["43"] = _ref(  # type: ignore[index]
        files["selection-43"]
    )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="stable across selection probe seeds"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_validation_boundary_far_from_selection(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    manifest = json.loads(files["validation"].read_text())
    for seed in (42, 43):
        _write_json(
            files[f"validation-{seed}"],
            _scores(
                "validation",
                seed,
                manifest,
                bindings=json.loads(files[f"validation-{seed}"].read_text())[
                    "bindings"
                ],
                noisy=[2.0, 1.9, 1.8, 1.7, 1.0],
            ),
        )
        payload["references"]["validation_scores_by_seed"][str(seed)] = _ref(  # type: ignore[index]
            files[f"validation-{seed}"]
        )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="stable on independent validation"):
        load_probe_transition_artifact(path)


def test_artifact_requires_group_paired_scores_and_stratum_counts(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    scores = json.loads(files["selection-42"].read_text())
    scores["records"][0]["edit_type"] = "unregistered-confound"
    _write_json(files["selection-42"], scores)
    payload["references"]["selection_scores_by_seed"]["42"] = _ref(  # type: ignore[index]
        files["selection-42"]
    )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="paired cohort manifest"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_boolean_selected_layer(tmp_path: Path) -> None:
    path, payload, _files = _bundle(tmp_path)
    payload["selected_transition_layer"] = True
    _write_json(path, payload)

    with pytest.raises(ValueError, match="must be an integer"):
        load_probe_transition_artifact(path)


@pytest.mark.parametrize("invalid", [True, "1.0"])
def test_probe_score_rejects_string_or_boolean_losses(tmp_path: Path, invalid: object) -> None:
    path, payload, files = _bundle(tmp_path)
    scores = json.loads(files["selection-42"].read_text())
    scores["records"][0]["clean_cross_entropy"][0] = invalid
    _write_json(files["selection-42"], scores)
    payload["references"]["selection_scores_by_seed"]["42"] = _ref(  # type: ignore[index]
        files["selection-42"]
    )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="only JSON numbers"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_class_imbalance(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    fit = json.loads(files["fit"].read_text())
    fit["records"].pop()
    _write_json(files["fit"], fit)
    _refresh_ref(payload, "fit_manifest", files["fit"])
    _write_json(path, payload)

    with pytest.raises(ValueError, match="exactly class balanced"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_selection_without_positive_denoising_drop(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    manifest = json.loads(files["selection"].read_text())
    for seed in (42, 43):
        _write_json(
            files[f"selection-{seed}"],
            _scores(
                "selection",
                seed,
                manifest,
                bindings=json.loads(files[f"selection-{seed}"].read_text())[
                    "bindings"
                ],
                noisy=[1.0, 2.0, 3.0, 4.0, 5.0],
            ),
        )
        payload["references"]["selection_scores_by_seed"][str(seed)] = _ref(  # type: ignore[index]
            files[f"selection-{seed}"]
        )
    payload["selected_transition_layer"] = 1
    _write_json(path, payload)

    with pytest.raises(ValueError, match="positive selection denoising drop"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_reused_probe_weights_across_seeds(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    payload["references"]["probe_weights_by_seed"]["43"] = _ref(  # type: ignore[index]
        files["weights-42"]
    )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="distinct independently fitted"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_score_provenance_binding_tampering(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    scores = json.loads(files["selection-42"].read_text())
    scores["bindings"]["probe_weights_sha256"] = "f" * 64
    _write_json(files["selection-42"], scores)
    payload["references"]["selection_scores_by_seed"]["42"] = _ref(  # type: ignore[index]
        files["selection-42"]
    )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="score provenance bindings differ"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_probe_weight_metadata_tampering(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    original = files["weights-42"]
    with safe_open(original, framework="np") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = dict(handle.metadata())
    metadata["code_revision"] = "c" * 40
    tampered = tmp_path / "probe-42-wrong-metadata.safetensors"
    save_file(tensors, tampered, metadata=metadata)
    payload["references"]["probe_weights_by_seed"]["42"] = _ref(tampered)  # type: ignore[index]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="weight provenance metadata differs"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_probe_weight_tensor_inventory_tampering(
    tmp_path: Path,
) -> None:
    path, payload, files = _bundle(tmp_path)
    original = files["weights-42"]
    with safe_open(original, framework="np") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = dict(handle.metadata())
    tensors.pop("decoder_layer.4.bias")
    tampered = tmp_path / "probe-42-missing-tensor.safetensors"
    save_file(tensors, tampered, metadata=metadata)
    payload["references"]["probe_weights_by_seed"]["42"] = _ref(tampered)  # type: ignore[index]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="weight tensor inventory differs"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_cross_kind_transitive_identity_overlap(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    fit = json.loads(files["fit"].read_text())
    validation = json.loads(files["validation"].read_text())
    fit["records"][0]["parent_source_sha256"] = validation["records"][0][
        "normalized_clean_sha256"
    ]
    _write_json(files["fit"], fit)
    _refresh_ref(payload, "fit_manifest", files["fit"])
    _write_json(path, payload)

    with pytest.raises(ValueError, match="overlap transitively"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_single_observation_per_class(tmp_path: Path) -> None:
    path, payload, files = _bundle(tmp_path)
    fit = json.loads(files["fit"].read_text())
    fit["records"] = [fit["records"][0], fit["records"][3]]
    _write_json(files["fit"], fit)
    _refresh_ref(payload, "fit_manifest", files["fit"])
    _write_json(path, payload)

    with pytest.raises(ValueError, match="repeated independent sources"):
        load_probe_transition_artifact(path)


def test_artifact_rejects_unregistered_edit_stratum_even_if_scores_agree(
    tmp_path: Path,
) -> None:
    path, payload, files = _bundle(tmp_path)
    selection = json.loads(files["selection"].read_text())
    selection["records"][0]["edit_type"] = "unregistered-confound"
    _write_json(files["selection"], selection)
    _refresh_ref(payload, "selection_manifest", files["selection"])
    for seed in (42, 43):
        scores = json.loads(files[f"selection-{seed}"].read_text())
        scores["records"][0]["edit_type"] = "unregistered-confound"
        _write_json(files[f"selection-{seed}"], scores)
        payload["references"]["selection_scores_by_seed"][str(seed)] = _ref(  # type: ignore[index]
            files[f"selection-{seed}"]
        )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="frozen operation inventory"):
        load_probe_transition_artifact(path)
