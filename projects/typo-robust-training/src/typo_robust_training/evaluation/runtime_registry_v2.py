"""Runtime preflights that connect evaluation-v2 phases to real entrypoints.

The phase registry deliberately stores hashes rather than executable paths.  A
small, local bundle supplies those paths at execution time; this module resolves
the bundle and proves that the concrete training/evaluation arguments are the
artifacts preregistered by the phase.  Merely presenting a self-consistent phase
file is therefore not enough to substitute another config, corpus, probe, or
checkpoint at the runner boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from typo_robust_training.data.config import strict_loads
from typo_robust_training.evaluation.calibration_v2 import (
    EvaluationV2Protocol,
    load_evaluation_v2_protocol,
)
from typo_robust_training.evaluation.registry_v2 import (
    load_evaluation_opening_sealed_evaluation_v2_registry,
    load_training_preregistered_evaluation_v2_registry,
)
from typo_robust_training.integrity import sha256_file, sha256_tree


_FACTORIAL_CONDITIONS = (
    "factorial-all-layers-all-tokens",
    "factorial-all-layers-downstream-horizon",
    "factorial-probe-suffix-all-tokens",
    "factorial-probe-suffix-downstream-horizon",
    "factorial-random-layers-downstream-horizon",
)
_FAITHFUL_CONDITION = "kojima-faithful-output-matching"
CONFIRMATORY_TRAINING_CONDITIONS = frozenset((*_FACTORIAL_CONDITIONS, _FAITHFUL_CONDITION))

_COMMON_ARTIFACT_FIELDS = {
    "registry_path",
    "protocol_path",
    "repository_path",
    "calibration_observations_path",
    "calibration_item_manifest_path",
    "calibration_typo_manifest_path",
    "calibration_result_path",
    "confirmatory_item_manifest_path",
    "confirmatory_typo_manifest_path",
    "tier_role_manifest_path",
    "factorial_arm_registry_path",
    "probe_artifact_registry_path",
    "training_config_registry_path",
    "training_data_registry_path",
    "legacy_random_2_registry_path",
}
_POST_TRAINING_ARTIFACT_FIELDS = {
    "training_preregistered_registry_path",
    "mistral_matched_seed_registry_path",
    "mistral_public_seed_1_checkpoint_path",
    "arm_checkpoint_registry_path",
    "opening_log_path",
}
_ALL_ARTIFACT_FIELDS = _COMMON_ARTIFACT_FIELDS | _POST_TRAINING_ARTIFACT_FIELDS


@dataclass(frozen=True, slots=True)
class EvaluationV2RuntimeRegistryBundle:
    phase: str
    paths: Mapping[str, Path | None]
    bundle_sha256: str

    def required_path(self, field: str) -> Path:
        value = self.paths.get(field)
        if not isinstance(value, Path):
            raise ValueError(f"evaluation v2 runtime bundle is missing {field}")
        return value


@dataclass(frozen=True, slots=True)
class EvaluationV2TrainingRuntimeBinding:
    bundle_sha256: str
    registry_sha256: str
    protocol_sha256: str
    config_sha256: str
    training_data_tree_sha256: str
    probe_artifact_sha256: str | None


@dataclass(frozen=True, slots=True)
class EvaluationV2OpeningRuntimeBinding:
    bundle_sha256: str
    registry_sha256: str
    protocol_sha256: str
    checkpoint_tree_sha256: tuple[str, ...]


def _object(path: Path, *, label: str) -> Mapping[str, object]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"evaluation v2 {label} cannot be a symbolic link")
    resolved = supplied.resolve()
    if not resolved.is_file():
        raise ValueError(f"evaluation v2 {label} must be one regular file")
    value = strict_loads(resolved.read_text(encoding="utf-8"), context=str(resolved))
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation v2 {label} must contain an object")
    return value


def _entry_list(
    path: Path,
    *,
    schema: str,
    fields: set[str],
    label: str,
) -> tuple[Mapping[str, object], ...]:
    value = _object(path, label=label)
    if set(value) != {"schema_version", "entries"} or value.get("schema_version") != schema:
        raise ValueError(f"evaluation v2 {label} schema differs")
    raw = value.get("entries")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"evaluation v2 {label} entries differ")
    entries: list[Mapping[str, object]] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != fields:
            raise ValueError(f"evaluation v2 {label} entry fields differ")
        entries.append(entry)
    return tuple(entries)


def _resolve_bundle_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation v2 runtime bundle {field} must be a path")
    candidate = Path(value)
    supplied = candidate if candidate.is_absolute() else root / candidate
    if supplied.is_symlink():
        raise ValueError(f"evaluation v2 runtime bundle {field} cannot be a symbolic link")
    return supplied.resolve()


def _regular_file(path: Path, *, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"evaluation v2 concrete {label} cannot be a symbolic link")
    resolved = supplied.resolve()
    if not resolved.is_file():
        raise ValueError(f"evaluation v2 concrete {label} must be one regular file")
    return resolved


def _regular_directory(path: Path, *, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"evaluation v2 concrete {label} cannot be a symbolic link")
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ValueError(f"evaluation v2 concrete {label} must be one regular directory")
    return resolved


def load_evaluation_v2_runtime_registry_bundle(
    path: Path,
    *,
    required_phase: str,
) -> EvaluationV2RuntimeRegistryBundle:
    """Resolve one closed-world phase bundle without trusting ambient cwd paths."""

    if required_phase not in {"training-preregistered", "evaluation-opening-sealed"}:
        raise ValueError("evaluation v2 runtime bundle requested phase is invalid")
    supplied = Path(path)
    payload = _object(supplied, label="runtime registry bundle")
    resolved = supplied.resolve()
    if (
        set(payload) != {"schema_version", "phase", "artifacts"}
        or payload.get("schema_version") != "robustness-evaluation-v2-runtime-registry-bundle/v1"
    ):
        raise ValueError("evaluation v2 runtime registry bundle schema differs")
    if payload.get("phase") != required_phase:
        raise ValueError(f"evaluation v2 runtime registry bundle is not {required_phase}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _ALL_ARTIFACT_FIELDS:
        raise ValueError("evaluation v2 runtime registry bundle artifact fields differ")
    root = resolved.parent
    paths: dict[str, Path | None] = {}
    for field in sorted(_COMMON_ARTIFACT_FIELDS):
        paths[field] = _resolve_bundle_path(root, artifacts.get(field), field=field)
    for field in sorted(_POST_TRAINING_ARTIFACT_FIELDS):
        value = artifacts.get(field)
        if required_phase == "training-preregistered":
            if value is not None:
                raise ValueError(
                    "evaluation v2 training runtime bundle contains post-training artifacts"
                )
            paths[field] = None
        else:
            paths[field] = _resolve_bundle_path(root, value, field=field)
    return EvaluationV2RuntimeRegistryBundle(
        phase=required_phase,
        paths=MappingProxyType(paths),
        bundle_sha256=sha256_file(resolved),
    )


def _load_phase(
    bundle: EvaluationV2RuntimeRegistryBundle,
) -> tuple[EvaluationV2Protocol, Mapping[str, object]]:
    protocol = load_evaluation_v2_protocol(bundle.required_path("protocol_path"))
    common = {
        field: bundle.required_path(field)
        for field in (
            "repository_path",
            "calibration_observations_path",
            "calibration_item_manifest_path",
            "calibration_typo_manifest_path",
            "calibration_result_path",
            "confirmatory_item_manifest_path",
            "confirmatory_typo_manifest_path",
            "tier_role_manifest_path",
            "factorial_arm_registry_path",
            "probe_artifact_registry_path",
            "training_config_registry_path",
            "training_data_registry_path",
            "legacy_random_2_registry_path",
        )
    }
    if bundle.phase == "training-preregistered":
        phase = load_training_preregistered_evaluation_v2_registry(
            registry_path=bundle.required_path("registry_path"),
            protocol=protocol,
            **common,
        )
    else:
        phase = load_evaluation_opening_sealed_evaluation_v2_registry(
            registry_path=bundle.required_path("registry_path"),
            training_preregistered_registry_path=bundle.required_path(
                "training_preregistered_registry_path"
            ),
            protocol=protocol,
            mistral_matched_seed_registry_path=bundle.required_path(
                "mistral_matched_seed_registry_path"
            ),
            mistral_public_seed_1_checkpoint_path=bundle.required_path(
                "mistral_public_seed_1_checkpoint_path"
            ),
            arm_checkpoint_registry_path=bundle.required_path("arm_checkpoint_registry_path"),
            opening_log_path=bundle.required_path("opening_log_path"),
            **common,
        )
    return protocol, phase


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation v2 {field} must be non-empty text")
    return value


def _sha(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"evaluation v2 {field} must be a lowercase SHA-256")
    return text


def _training_registry_entries(
    *, bundle: EvaluationV2RuntimeRegistryBundle, protocol: EvaluationV2Protocol
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    configs = _entry_list(
        bundle.required_path("training_config_registry_path"),
        schema="robustness-evaluation-v2-training-config-registry/v1",
        fields={"model_id", "model_revision", "condition", "config_sha256"},
        label="training config registry",
    )
    data = _entry_list(
        bundle.required_path("training_data_registry_path"),
        schema="robustness-evaluation-v2-training-data-registry/v1",
        fields={
            "model_id",
            "model_revision",
            "condition",
            "seed",
            "training_data_tree_sha256",
        },
        label="training data registry",
    )
    probes = _entry_list(
        bundle.required_path("probe_artifact_registry_path"),
        schema="robustness-evaluation-v2-probe-artifact-registry/v1",
        fields={"model_id", "model_revision", "artifact_sha256"},
        label="probe artifact registry",
    )
    arms = _entry_list(
        bundle.required_path("factorial_arm_registry_path"),
        schema="robustness-evaluation-v2-factorial-arm-registry/v1",
        fields={
            "model_id",
            "model_revision",
            "condition",
            "config_sha256",
            "probe_artifact_sha256",
        },
        label="factorial arm registry",
    )

    frozen_models = {(model.model_id, model.revision) for model in protocol.models}
    factorial_keys = {
        (model, revision, condition)
        for model, revision in frozen_models
        for condition in _FACTORIAL_CONDITIONS
    }
    mistral = ("mistralai/Mistral-7B-v0.1", "7231864981174d9bee8c7687c24c8344414eae6b")
    expected_config_keys = factorial_keys | {(*mistral, _FAITHFUL_CONDITION)}
    config_keys = {
        (
            _text(entry.get("model_id"), field="training config model"),
            _text(entry.get("model_revision"), field="training config revision"),
            _text(entry.get("condition"), field="training config condition"),
        )
        for entry in configs
    }
    if config_keys != expected_config_keys or len(config_keys) != len(configs):
        raise ValueError("evaluation v2 training config registry inventory differs")
    probe_keys = {
        (
            _text(entry.get("model_id"), field="probe model"),
            _text(entry.get("model_revision"), field="probe revision"),
        )
        for entry in probes
    }
    if probe_keys != frozen_models or len(probe_keys) != len(probes):
        raise ValueError("evaluation v2 probe artifact registry inventory differs")
    arm_keys = {
        (
            _text(entry.get("model_id"), field="factorial arm model"),
            _text(entry.get("model_revision"), field="factorial arm revision"),
            _text(entry.get("condition"), field="factorial arm condition"),
        )
        for entry in arms
    }
    if arm_keys != factorial_keys or len(arm_keys) != len(arms):
        raise ValueError("evaluation v2 factorial arm registry inventory differs")
    expected_data_keys = {
        (*key, seed) for key in factorial_keys for seed in protocol.training_seeds
    } | {(*mistral, _FAITHFUL_CONDITION, seed) for seed in (1, *protocol.training_seeds)}
    data_keys: set[tuple[str, str, str, int]] = set()
    for entry in data:
        seed = entry.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("evaluation v2 training data registry seed differs")
        data_keys.add(
            (
                _text(entry.get("model_id"), field="training data model"),
                _text(entry.get("model_revision"), field="training data revision"),
                _text(entry.get("condition"), field="training data condition"),
                seed,
            )
        )
    if data_keys != expected_data_keys or len(data_keys) != len(data):
        raise ValueError("evaluation v2 training data registry inventory differs")
    return configs, data, probes, arms


def _one(entries: Sequence[Mapping[str, object]], *, key: Mapping[str, object], label: str):
    matches = tuple(
        entry for entry in entries if all(entry.get(field) == value for field, value in key.items())
    )
    if len(matches) != 1:
        raise ValueError(f"evaluation v2 {label} runtime identity is not unique")
    return matches[0]


def validate_confirmatory_training_runtime(
    *,
    bundle_path: Path | None,
    condition: str,
    seed: int,
    config_path: Path,
    training_data_dir: Path,
    probe_selection_path: Path | None,
) -> EvaluationV2TrainingRuntimeBinding:
    """Fail before data/model construction unless the concrete run was frozen."""

    if condition not in CONFIRMATORY_TRAINING_CONDITIONS:
        raise ValueError("evaluation v2 training preflight received a legacy condition")
    if bundle_path is None:
        raise ValueError("confirmatory training requires --evaluation-v2-registry-bundle")
    bundle = load_evaluation_v2_runtime_registry_bundle(
        bundle_path, required_phase="training-preregistered"
    )
    protocol, _phase = _load_phase(bundle)

    from typo_robust_training.training.config import load_adapter_training_config

    resolved_config = _regular_file(config_path, label="training config")
    training = load_adapter_training_config(resolved_config)
    identity = {
        "model_id": training.model,
        "model_revision": training.model_revision,
        "condition": condition,
    }
    if training.condition != condition or seed not in training.seed_inventory:
        raise ValueError("evaluation v2 concrete training config identity differs")
    if (training.model, training.model_revision) not in {
        (model.model_id, model.revision) for model in protocol.models
    }:
        raise ValueError("evaluation v2 concrete training model is outside the frozen inventory")
    configs, data, probes, arms = _training_registry_entries(bundle=bundle, protocol=protocol)
    config_sha = sha256_file(resolved_config)
    config_entry = _one(configs, key=identity, label="training config registry")
    if _sha(config_entry.get("config_sha256"), field="training config hash") != config_sha:
        raise ValueError("evaluation v2 concrete training config hash differs")
    data_sha = sha256_tree(_regular_directory(training_data_dir, label="training data directory"))
    data_entry = _one(data, key={**identity, "seed": seed}, label="training data registry")
    if (
        _sha(data_entry.get("training_data_tree_sha256"), field="training data tree hash")
        != data_sha
    ):
        raise ValueError("evaluation v2 concrete training data hash differs")

    probe_sha: str | None = None
    if condition in _FACTORIAL_CONDITIONS:
        if probe_selection_path is None:
            raise ValueError("evaluation v2 factorial training requires the frozen probe artifact")
        probe_sha = sha256_file(_regular_file(probe_selection_path, label="probe artifact"))
        probe_entry = _one(
            probes,
            key={"model_id": training.model, "model_revision": training.model_revision},
            label="probe artifact registry",
        )
        arm_entry = _one(arms, key=identity, label="factorial arm registry")
        if (
            _sha(probe_entry.get("artifact_sha256"), field="probe artifact hash") != probe_sha
            or _sha(arm_entry.get("probe_artifact_sha256"), field="factorial probe artifact hash")
            != probe_sha
            or _sha(arm_entry.get("config_sha256"), field="factorial config hash") != config_sha
            or training.expected_method_evidence_sha256 != probe_sha
        ):
            raise ValueError("evaluation v2 concrete factorial config/probe binding differs")
    elif probe_selection_path is not None:
        raise ValueError("evaluation v2 faithful Kojima run cannot consume a probe artifact")

    return EvaluationV2TrainingRuntimeBinding(
        bundle_sha256=bundle.bundle_sha256,
        registry_sha256=sha256_file(bundle.required_path("registry_path")),
        protocol_sha256=protocol.config_sha256,
        config_sha256=config_sha,
        training_data_tree_sha256=data_sha,
        probe_artifact_sha256=probe_sha,
    )


def validate_confirmatory_evaluation_opening(
    *,
    bundle_path: Path | None,
    checkpoint_paths: Sequence[Path],
    evaluation_data_dir: Path,
) -> EvaluationV2OpeningRuntimeBinding:
    """Bind a confirmatory opening to sealed manifests and checkpoint trees."""

    if bundle_path is None:
        raise ValueError("confirmatory evaluation requires --evaluation-v2-registry-bundle")
    bundle = load_evaluation_v2_runtime_registry_bundle(
        bundle_path, required_phase="evaluation-opening-sealed"
    )
    protocol, _phase = _load_phase(bundle)
    item_path = bundle.required_path("confirmatory_item_manifest_path")
    typo_path = bundle.required_path("confirmatory_typo_manifest_path")
    data_root = _regular_directory(evaluation_data_dir, label="evaluation data directory")
    if item_path.parent != data_root or typo_path.parent != data_root:
        raise ValueError("evaluation v2 concrete evaluation data directory differs")
    entries = _entry_list(
        bundle.required_path("arm_checkpoint_registry_path"),
        schema="robustness-evaluation-v2-arm-checkpoint-registry/v1",
        fields={
            "model_id",
            "model_revision",
            "condition",
            "seed",
            "adapter_tree_sha256",
        },
        label="arm checkpoint registry",
    )
    frozen_models = {(model.model_id, model.revision) for model in protocol.models}
    expected_keys = {
        (model, revision, condition, seed)
        for model, revision in frozen_models
        for condition in _FACTORIAL_CONDITIONS
        for seed in protocol.training_seeds
    } | {
        (
            "mistralai/Mistral-7B-v0.1",
            "7231864981174d9bee8c7687c24c8344414eae6b",
            _FAITHFUL_CONDITION,
            seed,
        )
        for seed in protocol.training_seeds
    }
    actual_keys: set[tuple[str, str, str, int]] = set()
    indexed: dict[tuple[str, str, str, int], str] = {}
    for entry in entries:
        seed = entry.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("evaluation v2 arm checkpoint seed differs")
        key = (
            _text(entry.get("model_id"), field="arm checkpoint model"),
            _text(entry.get("model_revision"), field="arm checkpoint revision"),
            _text(entry.get("condition"), field="arm checkpoint condition"),
            seed,
        )
        actual_keys.add(key)
        indexed[key] = _sha(entry.get("adapter_tree_sha256"), field="adapter tree hash")
    if actual_keys != expected_keys or len(actual_keys) != len(entries):
        raise ValueError("evaluation v2 arm checkpoint registry inventory differs")

    checkpoint_hashes: list[str] = []
    seen: set[tuple[str, str, str, int]] = set()
    for raw in checkpoint_paths:
        checkpoint = _regular_directory(Path(raw), label="checkpoint directory")
        runtime = _object(checkpoint / "training_runtime.json", label="checkpoint runtime")
        run = _object(checkpoint.parent / "run.json", label="checkpoint training run")
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("evaluation v2 concrete checkpoint seed differs")
        key = (
            _text(runtime.get("model"), field="concrete checkpoint model"),
            _text(runtime.get("requested_revision"), field="concrete checkpoint revision"),
            _text(run.get("condition"), field="concrete checkpoint condition"),
            seed,
        )
        if runtime.get("condition") != key[2] or runtime.get("seed") != key[3]:
            raise ValueError("evaluation v2 concrete checkpoint runtime/run identity differs")
        if key in seen or key not in indexed:
            raise ValueError("evaluation v2 concrete checkpoint identity differs")
        tree_sha = sha256_tree(checkpoint)
        if indexed[key] != tree_sha:
            raise ValueError("evaluation v2 concrete checkpoint hash differs")
        seen.add(key)
        checkpoint_hashes.append(tree_sha)
    selected_models = {(model, revision) for model, revision, _condition, _seed in seen}
    if len(selected_models) != 1:
        raise ValueError(
            "evaluation v2 confirmatory evaluation must contain one complete model batch"
        )
    selected_model = next(iter(selected_models))
    expected_model_batch = {key for key in expected_keys if key[:2] == selected_model}
    if seen != expected_model_batch:
        raise ValueError("evaluation v2 confirmatory evaluation checkpoint batch is incomplete")
    return EvaluationV2OpeningRuntimeBinding(
        bundle_sha256=bundle.bundle_sha256,
        registry_sha256=sha256_file(bundle.required_path("registry_path")),
        protocol_sha256=protocol.config_sha256,
        checkpoint_tree_sha256=tuple(checkpoint_hashes),
    )


def confirmatory_evaluation_is_required(checkpoint_paths: Sequence[Path]) -> bool:
    """Identify v2-only checkpoint conditions without constructing a model runtime."""

    conditions: set[str] = set()
    for raw in checkpoint_paths:
        supplied = Path(raw)
        if supplied.is_symlink():
            raise ValueError(
                "evaluation v2 concrete checkpoint directory cannot be a symbolic link"
            )
        checkpoint = supplied.resolve()
        if not checkpoint.is_dir():
            raise ValueError("evaluation checkpoint directory is unavailable")
        checkpoint_conditions: set[str] = set()
        for path, label in (
            (checkpoint.parent / "run.json", "checkpoint training run"),
            (checkpoint / "training_runtime.json", "checkpoint runtime"),
        ):
            if path.is_symlink():
                raise ValueError(f"evaluation {label} cannot be a symbolic link")
            if not path.is_file():
                raise ValueError(f"evaluation {label} is unavailable")
            value = _object(path, label=label)
            condition = value.get("condition")
            if not isinstance(condition, str) or not condition:
                raise ValueError(f"evaluation {label} condition is unavailable")
            checkpoint_conditions.add(condition)
        if len(checkpoint_conditions) != 1:
            raise ValueError("evaluation checkpoint runtime/run condition differs")
        conditions.update(checkpoint_conditions)
    confirmatory = conditions & CONFIRMATORY_TRAINING_CONDITIONS
    if confirmatory and confirmatory != conditions:
        raise ValueError("evaluation cannot mix v2 confirmatory and legacy checkpoints")
    return bool(confirmatory)


__all__ = [
    "CONFIRMATORY_TRAINING_CONDITIONS",
    "EvaluationV2OpeningRuntimeBinding",
    "EvaluationV2RuntimeRegistryBundle",
    "EvaluationV2TrainingRuntimeBinding",
    "confirmatory_evaluation_is_required",
    "load_evaluation_v2_runtime_registry_bundle",
    "validate_confirmatory_evaluation_opening",
    "validate_confirmatory_training_runtime",
]
