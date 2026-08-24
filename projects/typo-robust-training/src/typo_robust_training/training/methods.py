"""Pure resolution of method evidence into trainable and supervised scopes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from typo_robust_training.data.config import strict_loads
from typo_robust_training.training.config import AdapterTrainingProtocol
from typo_robust_training.training.config import load_adapter_training_config


_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
PROBE_FACTORIAL_CONDITIONS = (
    "factorial-all-layers-all-tokens",
    "factorial-all-layers-downstream-horizon",
    "factorial-probe-suffix-all-tokens",
    "factorial-probe-suffix-downstream-horizon",
    "factorial-random-layers-downstream-horizon",
)
_FACTORIAL_ARM_FIELDS: Mapping[str, tuple[str, str, str]] = {
    "factorial-all-layers-all-tokens": (
        "all-decoder-layers",
        "all-decoder-layers/v1",
        "aligned-non-edited-next-token/v1",
    ),
    "factorial-all-layers-downstream-horizon": (
        "all-decoder-layers",
        "all-decoder-layers/v1",
        "clean-all-noisy-edited-word-downstream-offsets-2-16/v1",
    ),
    "factorial-probe-suffix-all-tokens": (
        "probe-transition-suffix",
        "validated-linear-probe-transition-suffix/v1",
        "aligned-non-edited-next-token/v1",
    ),
    "factorial-probe-suffix-downstream-horizon": (
        "probe-transition-suffix",
        "validated-linear-probe-transition-suffix/v1",
        "clean-all-noisy-edited-word-downstream-offsets-2-16/v1",
    ),
    "factorial-random-layers-downstream-horizon": (
        "probe-count-matched-random-layers",
        "sha256-seed42-count-matched-random-freeze/v1",
        "clean-all-noisy-edited-word-downstream-offsets-2-16/v1",
    ),
}


@dataclass(frozen=True, slots=True)
class ProbeTransitionTrainingEvidence:
    """Minimal, hash-bound view of independently validated probe evidence."""

    model: str
    model_revision: str
    decoder_layers: int
    selected_transition_layer: int
    evidence_sha256: str
    tokenizer_snapshot_attestation: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("probe evidence model must be non-empty")
        if _REVISION.fullmatch(self.model_revision) is None:
            raise ValueError("probe evidence model revision must be a pinned commit SHA")
        if (
            isinstance(self.decoder_layers, bool)
            or not isinstance(self.decoder_layers, int)
            or self.decoder_layers < 2
        ):
            raise ValueError("probe evidence decoder layers must be at least two")
        if (
            isinstance(self.selected_transition_layer, bool)
            or not isinstance(self.selected_transition_layer, int)
            or not 1 <= self.selected_transition_layer < self.decoder_layers
        ):
            raise ValueError("probe evidence transition layer is outside the decoder")
        if _SHA256.fullmatch(self.evidence_sha256) is None:
            raise ValueError("probe evidence hash must be a SHA-256 digest")
        if self.tokenizer_snapshot_attestation is not None and not isinstance(
            self.tokenizer_snapshot_attestation, Mapping
        ):
            raise ValueError("probe tokenizer attestation must be an object")

    @property
    def suffix_layers(self) -> tuple[int, ...]:
        return tuple(range(self.selected_transition_layer, self.decoder_layers))


@dataclass(frozen=True, slots=True)
class ProbeTransitionStateTrainingEvidence:
    """Hash-bound view of a passed transition-layer causal gate."""

    model: str
    model_revision: str
    decoder_layers: int
    selected_transition_layer: int
    parent_probe_artifact_sha256: str
    evidence_sha256: str
    tokenizer_snapshot_attestation: Mapping[str, object]

    def __post_init__(self) -> None:
        ProbeTransitionTrainingEvidence(
            model=self.model,
            model_revision=self.model_revision,
            decoder_layers=self.decoder_layers,
            selected_transition_layer=self.selected_transition_layer,
            evidence_sha256=self.evidence_sha256,
        )
        if _SHA256.fullmatch(self.parent_probe_artifact_sha256) is None:
            raise ValueError("state gate parent probe hash must be a SHA-256 digest")
        if not isinstance(self.tokenizer_snapshot_attestation, Mapping):
            raise ValueError("state gate tokenizer attestation must be an object")

    @property
    def suffix_layers(self) -> tuple[int, ...]:
        return tuple(range(self.selected_transition_layer, self.decoder_layers))


@dataclass(frozen=True, slots=True)
class ProbeSemanticSubspaceTrainingEvidence:
    """Training-only view of a kill-tested rank-16 probe semantic subspace."""

    model: str
    model_revision: str
    decoder_layers: int
    transition_layer: int
    primary_probe_seed: int
    basis: np.ndarray
    projected_class_weights: np.ndarray
    classifier_bias: np.ndarray
    evidence_sha256: str
    tokenizer_snapshot_attestation: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("semantic evidence model must be non-empty")
        if _REVISION.fullmatch(self.model_revision) is None:
            raise ValueError("semantic evidence model revision must be pinned")
        if (
            isinstance(self.decoder_layers, bool)
            or not isinstance(self.decoder_layers, int)
            or self.decoder_layers < 2
            or isinstance(self.transition_layer, bool)
            or not isinstance(self.transition_layer, int)
            or not 1 <= self.transition_layer < self.decoder_layers
            or self.primary_probe_seed != 42
            or _SHA256.fullmatch(self.evidence_sha256) is None
        ):
            raise ValueError("semantic evidence identity differs")
        if not isinstance(self.tokenizer_snapshot_attestation, Mapping):
            raise ValueError("semantic evidence tokenizer attestation must be an object")
        basis = np.asarray(self.basis, dtype=np.float64)
        weights = np.asarray(self.projected_class_weights, dtype=np.float64)
        bias = np.asarray(self.classifier_bias, dtype=np.float64)
        if (
            basis.ndim != 2
            or basis.shape[0] != 16
            or weights.ndim != 2
            or weights.shape[1] != 16
            or bias.shape != (weights.shape[0],)
            or basis.shape[1] <= 16
            or weights.shape[0] <= 16
            or not np.isfinite(basis).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(bias).all()
            or not np.allclose(basis @ basis.T, np.eye(16), atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("semantic evidence tensors differ")
        frozen: list[np.ndarray] = []
        for value in (basis, weights, bias):
            copied = np.ascontiguousarray(value.copy())
            copied.flags.writeable = False
            frozen.append(copied)
        object.__setattr__(self, "basis", frozen[0])
        object.__setattr__(self, "projected_class_weights", frozen[1])
        object.__setattr__(self, "classifier_bias", frozen[2])

    @property
    def suffix_layers(self) -> tuple[int, ...]:
        return tuple(range(self.transition_layer, self.decoder_layers))


@dataclass(frozen=True, slots=True)
class ResolvedTrainingMethod:
    adapter_layers: tuple[int, ...]
    state_layers: tuple[int, ...]
    state_target: str
    method_evidence_sha256: str


def count_matched_random_layers(
    *,
    decoder_layers: int,
    selected_transition_layer: int,
) -> tuple[int, ...]:
    """Return the preregistered same-count random-freeze control.

    The fixed seed is part of the policy name, not a runtime/tuning seed.  The
    selected count exactly equals the transition suffix, while layer identity
    is derived only from the model depth and transition boundary.
    """

    if (
        isinstance(decoder_layers, bool)
        or not isinstance(decoder_layers, int)
        or decoder_layers < 2
        or isinstance(selected_transition_layer, bool)
        or not isinstance(selected_transition_layer, int)
        or not 1 <= selected_transition_layer < decoder_layers
    ):
        raise ValueError("random-freeze layer inventory is invalid")
    count = decoder_layers - selected_transition_layer
    ranked = sorted(
        range(decoder_layers),
        key=lambda layer: hashlib.sha256(
            (
                "probe-count-matched-random-layers/v1"
                f"\0seed=42\0layers={decoder_layers}"
                f"\0transition={selected_transition_layer}\0layer={layer}"
            ).encode("utf-8")
        ).digest(),
    )
    selected = set(ranked[:count])
    suffix = set(range(selected_transition_layer, decoder_layers))
    if selected == suffix:
        # A deterministic anti-degeneracy clause ensures this is an actual
        # location control, not an accidental duplicate of the proposal arm.
        selected.remove(min(selected))
        selected.add(min(set(range(decoder_layers)) - selected - suffix))
    result = tuple(sorted(selected))
    if len(result) != count or set(result) == suffix:
        raise RuntimeError("random-freeze control did not produce a distinct matched scope")
    return result


def load_probe_transition_training_evidence(
    path: Path,
    *,
    model: str,
    model_revision: str,
    decoder_layers: int,
) -> ProbeTransitionTrainingEvidence:
    """Load the validated producer bundle and bind it to the training identity."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("probe transition evidence must not be a symlink")
    from typo_robust_training.probe import load_probe_transition_artifact
    from typo_robust_training.probe.artifacts import (
        require_probe_artifact_child_eligibility,
    )

    artifact = load_probe_transition_artifact(supplied)
    require_probe_artifact_child_eligibility(artifact)
    if (
        artifact.model != model
        or artifact.model_revision != model_revision
        or artifact.decoder_layers != decoder_layers
    ):
        raise ValueError("probe transition artifact identity differs from training")
    return ProbeTransitionTrainingEvidence(
        model=artifact.model,
        model_revision=artifact.model_revision,
        decoder_layers=artifact.decoder_layers,
        selected_transition_layer=artifact.selected_transition_layer,
        evidence_sha256=artifact.artifact_sha256,
        tokenizer_snapshot_attestation=artifact.tokenizer_snapshot_attestation,
    )


def load_probe_transition_state_training_evidence(
    path: Path,
    *,
    model: str,
    model_revision: str,
    decoder_layers: int,
) -> ProbeTransitionStateTrainingEvidence:
    """Load and recompute the causal gate before exposing it to training."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("single-layer gate evidence must not be a symlink")
    from typo_robust_training.state_gate import load_single_layer_gate_artifact

    artifact = load_single_layer_gate_artifact(supplied)
    if (
        artifact.model != model
        or artifact.model_revision != model_revision
        or artifact.decoder_layers != decoder_layers
    ):
        raise ValueError("single-layer gate artifact identity differs from training")
    return ProbeTransitionStateTrainingEvidence(
        model=artifact.model,
        model_revision=artifact.model_revision,
        decoder_layers=artifact.decoder_layers,
        selected_transition_layer=artifact.selected_transition_layer,
        parent_probe_artifact_sha256=artifact.parent_probe_artifact_sha256,
        evidence_sha256=artifact.artifact_sha256,
        tokenizer_snapshot_attestation=artifact.tokenizer_snapshot_attestation,
    )


def load_probe_semantic_subspace_training_evidence(
    path: Path,
    *,
    model: str,
    model_revision: str,
    decoder_layers: int,
) -> ProbeSemanticSubspaceTrainingEvidence:
    """Load and independently revalidate the two-seed kill-test evidence."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("semantic subspace evidence must not be a symlink")
    from typo_robust_training.probe.subspace_kill_artifacts import (
        load_semantic_subspace_kill_artifact,
    )

    artifact = load_semantic_subspace_kill_artifact(supplied)
    if (
        artifact.model != model
        or artifact.model_revision != model_revision
        or artifact.decoder_layers != decoder_layers
        or artifact.rank != 16
        or artifact.primary_probe_seed != 42
    ):
        raise ValueError("semantic subspace artifact identity differs from training")
    subspace = artifact.semantic_subspace
    return ProbeSemanticSubspaceTrainingEvidence(
        model=artifact.model,
        model_revision=artifact.model_revision,
        decoder_layers=artifact.decoder_layers,
        transition_layer=artifact.transition_layer,
        primary_probe_seed=artifact.primary_probe_seed,
        basis=subspace.basis,
        projected_class_weights=subspace.projected_class_weights,
        classifier_bias=subspace.classifier_bias,
        evidence_sha256=artifact.artifact_sha256,
        tokenizer_snapshot_attestation=artifact.tokenizer_snapshot_attestation,
    )


def materialize_probe_transition_training_config(
    template_path: Path,
    *,
    evidence_path: Path,
    output_path: Path,
) -> AdapterTrainingProtocol:
    """Atomically bind one validated probe artifact into a non-runnable template."""

    template = Path(template_path)
    evidence = Path(evidence_path)
    output = Path(output_path)
    if template.is_symlink() or not template.is_file():
        raise ValueError("probe transition training template must be one regular file")
    if evidence.is_symlink():
        raise ValueError("probe transition evidence must not be a symlink")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"materialized training config already exists: {output}")
    try:
        payload = strict_loads(
            template.read_text(encoding="utf-8"),
            context=str(template.resolve()),
        )
    except UnicodeDecodeError as exc:
        raise ValueError("probe transition training template must be UTF-8") from exc
    expected_top = {
        "schema_version",
        "condition",
        "method_evidence",
        "model",
        "sequence",
        "adapter",
        "optimization",
        "objective",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("probe transition training template fields differ")
    if payload["schema_version"] != "robustness-adapter-training-config/v4-template":
        raise ValueError("probe transition training template schema differs")
    binding = payload["method_evidence"]
    if not isinstance(binding, Mapping) or dict(binding) != {
        "schema_version": "probe-transition-evidence-binding/v1",
        "artifact_sha256": None,
    }:
        raise ValueError("probe transition training template must contain one null binding")
    model_fields = payload["model"]
    if not isinstance(model_fields, Mapping):
        raise ValueError("probe transition training template model must be an object")
    model_id = model_fields.get("id")
    model_revision = model_fields.get("revision")
    decoder_layers = model_fields.get("decoder_layers")
    if (
        not isinstance(model_id, str)
        or not isinstance(model_revision, str)
        or isinstance(decoder_layers, bool)
        or not isinstance(decoder_layers, int)
    ):
        raise ValueError("probe transition training template model identity differs")

    artifact_evidence = load_probe_transition_training_evidence(
        evidence,
        model=model_id,
        model_revision=model_revision,
        decoder_layers=decoder_layers,
    )
    materialized = copy.deepcopy(dict(payload))
    materialized["schema_version"] = "robustness-adapter-training-config/v4"
    materialized["method_evidence"]["artifact_sha256"] = (  # type: ignore[index]
        artifact_evidence.evidence_sha256
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(materialized, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        protocol = load_adapter_training_config(temporary)
        resolved = resolve_training_method(protocol, evidence=artifact_evidence)
        if resolved.method_evidence_sha256 != artifact_evidence.evidence_sha256:
            raise ValueError("materialized training config lost its evidence binding")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return load_adapter_training_config(output)


def materialize_probe_output_factorial_configs(
    template_path: Path,
    *,
    evidence_path: Path,
    output_dir: Path,
) -> Mapping[str, AdapterTrainingProtocol]:
    """Bind one probe artifact and atomically emit the frozen 2x2 plus control."""

    template = Path(template_path)
    evidence = Path(evidence_path)
    destination = Path(output_dir)
    if template.is_symlink() or not template.is_file():
        raise ValueError("probe-factorial template must be one regular file")
    if evidence.is_symlink():
        raise ValueError("probe-factorial evidence must not be a symlink")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("probe-factorial output directory must be absent or empty")
    payload = strict_loads(
        template.read_text(encoding="utf-8"),
        context=str(template.resolve()),
    )
    expected_top = {
        "schema_version", "condition", "method_evidence", "model", "sequence",
        "adapter", "optimization", "objective",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("probe-factorial template fields differ")
    if (
        payload["schema_version"] != "robustness-adapter-training-config/v7-template"
        or payload["condition"] is not None
        or payload["method_evidence"]
        != {
            "schema_version": "probe-output-factorial-evidence-binding/v1",
            "artifact_sha256": None,
        }
    ):
        raise ValueError("probe-factorial template binding differs")
    model_fields = payload["model"]
    adapter_fields = payload["adapter"]
    objective_fields = payload["objective"]
    if (
        not isinstance(model_fields, Mapping)
        or not isinstance(adapter_fields, Mapping)
        or not isinstance(objective_fields, Mapping)
        or adapter_fields.get("layer_scope") is not None
        or adapter_fields.get("layer_policy") is not None
        or objective_fields.get("output_scope") is not None
    ):
        raise ValueError("probe-factorial template must leave only arm axes null")
    loaded_evidence = load_probe_transition_training_evidence(
        evidence,
        model=str(model_fields.get("id")),
        model_revision=str(model_fields.get("revision")),
        decoder_layers=int(model_fields.get("decoder_layers", 0)),
    )
    destination.mkdir(parents=True, exist_ok=True)
    temporary_dir = destination / f".materializing.{os.getpid()}"
    temporary_dir.mkdir()
    protocols: dict[str, AdapterTrainingProtocol] = {}
    try:
        for condition in PROBE_FACTORIAL_CONDITIONS:
            layer_scope, layer_policy, output_scope = _FACTORIAL_ARM_FIELDS[condition]
            arm = copy.deepcopy(dict(payload))
            arm["schema_version"] = "robustness-adapter-training-config/v7"
            arm["condition"] = condition
            arm["method_evidence"]["artifact_sha256"] = (  # type: ignore[index]
                loaded_evidence.evidence_sha256
            )
            arm["adapter"]["layer_scope"] = layer_scope  # type: ignore[index]
            arm["adapter"]["layer_policy"] = layer_policy  # type: ignore[index]
            arm["objective"]["output_scope"] = output_scope  # type: ignore[index]
            temporary = temporary_dir / f"{condition}.json"
            temporary.write_text(
                json.dumps(arm, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            protocol = load_adapter_training_config(temporary)
            resolve_training_method(protocol, evidence=loaded_evidence)
            protocols[condition] = protocol
        manifest = {
            "schema_version": "probe-output-factorial-manifest/v1",
            "method_evidence_sha256": loaded_evidence.evidence_sha256,
            "arms": {
                condition: {
                    "config": f"{condition}.json",
                    "config_sha256": protocols[condition].config_sha256,
                    "adapter_layers": list(
                        resolve_training_method(
                            protocols[condition], evidence=loaded_evidence
                        ).adapter_layers
                    ),
                    "output_scope": protocols[condition].output_scope,
                    "initialization_policy": protocols[
                        condition
                    ].adapter_initialization_policy,
                }
                for condition in PROBE_FACTORIAL_CONDITIONS
            },
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for path in sorted(temporary_dir.iterdir()):
            os.replace(path, destination / path.name)
    finally:
        if temporary_dir.exists():
            temporary_dir.rmdir()
    return MappingProxyType(protocols)


def materialize_probe_semantic_subspace_training_config(
    template_path: Path,
    *,
    evidence_path: Path,
    output_path: Path,
) -> AdapterTrainingProtocol:
    """Bind one freshly revalidated kill-test artifact into a v6 template."""

    template = Path(template_path)
    evidence = Path(evidence_path)
    output = Path(output_path)
    if template.is_symlink() or not template.is_file():
        raise ValueError("semantic training template must be one regular file")
    if evidence.is_symlink():
        raise ValueError("semantic subspace evidence must not be a symlink")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"materialized training config already exists: {output}")
    payload = strict_loads(template.read_text(encoding="utf-8"), context=str(template.resolve()))
    expected_top = {
        "schema_version",
        "condition",
        "method_evidence",
        "model",
        "sequence",
        "adapter",
        "optimization",
        "objective",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("semantic training template fields differ")
    if payload["schema_version"] != "robustness-adapter-training-config/v6-template":
        raise ValueError("semantic training template schema differs")
    if payload["condition"] != "probe-semantic-subspace-distillation":
        raise ValueError("semantic training template condition differs")
    if payload["method_evidence"] != {
        "schema_version": "probe-semantic-subspace-evidence-binding/v1",
        "artifact_sha256": None,
    }:
        raise ValueError("semantic training template must contain one null binding")
    model_fields = payload.get("model")
    if not isinstance(model_fields, Mapping):
        raise ValueError("semantic training template model differs")
    model = model_fields.get("id")
    revision = model_fields.get("revision")
    decoder_layers = model_fields.get("decoder_layers")
    if (
        not isinstance(model, str)
        or not isinstance(revision, str)
        or isinstance(decoder_layers, bool)
        or not isinstance(decoder_layers, int)
    ):
        raise ValueError("semantic training template model identity differs")
    loaded = load_probe_semantic_subspace_training_evidence(
        evidence,
        model=model,
        model_revision=revision,
        decoder_layers=decoder_layers,
    )
    materialized = copy.deepcopy(dict(payload))
    materialized["schema_version"] = "robustness-adapter-training-config/v6"
    materialized["method_evidence"]["artifact_sha256"] = loaded.evidence_sha256  # type: ignore[index]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(materialized, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        protocol = load_adapter_training_config(temporary)
        resolved = resolve_training_method(protocol, evidence=loaded)
        if resolved.method_evidence_sha256 != loaded.evidence_sha256:
            raise ValueError("materialized semantic config lost its evidence binding")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return load_adapter_training_config(output)


def resolve_training_method(
    protocol: AdapterTrainingProtocol,
    *,
    evidence: (
        ProbeTransitionTrainingEvidence
        | ProbeTransitionStateTrainingEvidence
        | ProbeSemanticSubspaceTrainingEvidence
    ),
) -> ResolvedTrainingMethod:
    """Resolve an attested method scope without inspecting evaluation outcomes."""

    if not isinstance(protocol, AdapterTrainingProtocol):
        raise TypeError("training protocol must be AdapterTrainingProtocol")
    if protocol.condition in PROBE_FACTORIAL_CONDITIONS:
        if not isinstance(evidence, ProbeTransitionTrainingEvidence) or isinstance(
            evidence, ProbeTransitionStateTrainingEvidence
        ):
            raise ValueError("probe-factorial conditions require transition evidence")
        if protocol.decoder_layers is None or (
            evidence.model != protocol.model
            or evidence.model_revision != protocol.model_revision
            or evidence.decoder_layers != protocol.decoder_layers
            or protocol.expected_method_evidence_sha256 != evidence.evidence_sha256
        ):
            raise ValueError("probe-factorial evidence identity or hash differs")
        if "all-layers" in protocol.condition:
            adapter_layers = tuple(range(evidence.decoder_layers))
        elif "probe-suffix" in protocol.condition:
            adapter_layers = evidence.suffix_layers
        else:
            adapter_layers = count_matched_random_layers(
                decoder_layers=evidence.decoder_layers,
                selected_transition_layer=evidence.selected_transition_layer,
            )
        return ResolvedTrainingMethod(
            adapter_layers=adapter_layers,
            state_layers=(),
            state_target="none",
            method_evidence_sha256=evidence.evidence_sha256,
        )
    if isinstance(evidence, ProbeSemanticSubspaceTrainingEvidence):
        if protocol.condition != "probe-semantic-subspace-distillation":
            raise ValueError("semantic subspace evidence cannot configure this condition")
        if protocol.decoder_layers is None or (
            evidence.model != protocol.model
            or evidence.model_revision != protocol.model_revision
            or evidence.decoder_layers != protocol.decoder_layers
        ):
            raise ValueError("semantic evidence identity differs from training")
        if protocol.expected_method_evidence_sha256 != evidence.evidence_sha256:
            raise ValueError("semantic evidence hash differs from the training config")
        if (
            protocol.layer_scope != "probe-transition-suffix"
            or protocol.layer_policy != "validated-probe-semantic-subspace-suffix/v1"
            or protocol.state_scope != "probe-semantic-subspace-edited-word-final-token"
            or protocol.state_distance != "frozen-probe-classifier-forward-kl/v1"
        ):
            raise ValueError("semantic training scope differs")
        return ResolvedTrainingMethod(
            adapter_layers=evidence.suffix_layers,
            state_layers=(evidence.transition_layer,),
            state_target="probe-semantic-subspace-rank16",
            method_evidence_sha256=evidence.evidence_sha256,
        )
    if not isinstance(
        evidence,
        (ProbeTransitionTrainingEvidence, ProbeTransitionStateTrainingEvidence),
    ):
        raise TypeError("probe transition evidence has the wrong type")
    valid_pair = (
        protocol.condition == "probe-transition-output-matching"
        and isinstance(evidence, ProbeTransitionTrainingEvidence)
        and not isinstance(evidence, ProbeTransitionStateTrainingEvidence)
    ) or (
        protocol.condition == "probe-transition-single-layer-state-distillation"
        and isinstance(evidence, ProbeTransitionStateTrainingEvidence)
    )
    if not valid_pair:
        raise ValueError("probe transition evidence cannot configure this condition")
    if protocol.decoder_layers is None or (
        evidence.model != protocol.model
        or evidence.model_revision != protocol.model_revision
        or evidence.decoder_layers != protocol.decoder_layers
    ):
        raise ValueError("probe evidence identity differs from training")
    if protocol.expected_method_evidence_sha256 != evidence.evidence_sha256:
        raise ValueError("probe evidence hash differs from the preregistered training config")
    if (
        protocol.layer_scope != "probe-transition-suffix"
        or protocol.layer_policy != "validated-linear-probe-transition-suffix/v1"
    ):
        raise ValueError("probe transition adapter policy differs")
    state_active = isinstance(evidence, ProbeTransitionStateTrainingEvidence)
    return ResolvedTrainingMethod(
        adapter_layers=evidence.suffix_layers,
        state_layers=(evidence.selected_transition_layer,) if state_active else (),
        state_target=(
            "complete-decoder-block-residual-output-at-edited-word-final/v1"
            if state_active
            else "none"
        ),
        method_evidence_sha256=evidence.evidence_sha256,
    )


def materialize_probe_transition_state_training_config(
    template_path: Path,
    *,
    evidence_path: Path,
    output_path: Path,
) -> AdapterTrainingProtocol:
    """Bind one passed causal-gate artifact into the v5 template."""

    template = Path(template_path)
    evidence = Path(evidence_path)
    output = Path(output_path)
    if template.is_symlink() or not template.is_file():
        raise ValueError("state training template must be one regular file")
    if evidence.is_symlink():
        raise ValueError("state gate evidence must not be a symlink")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"materialized state training config already exists: {output}")
    payload = strict_loads(template.read_text(encoding="utf-8"), context=str(template.resolve()))
    expected_top = {
        "schema_version",
        "condition",
        "method_evidence",
        "model",
        "sequence",
        "adapter",
        "optimization",
        "objective",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_top
        or payload["schema_version"] != "robustness-adapter-training-config/v5-template"
    ):
        raise ValueError("state training template fields or schema differ")
    if payload["method_evidence"] != {
        "schema_version": "probe-transition-state-gate-binding/v1",
        "artifact_sha256": None,
    }:
        raise ValueError("state training template must contain one null gate binding")
    model_fields = payload["model"]
    if not isinstance(model_fields, Mapping):
        raise ValueError("state training template model must be an object")
    artifact_evidence = load_probe_transition_state_training_evidence(
        evidence,
        model=str(model_fields.get("id")),
        model_revision=str(model_fields.get("revision")),
        decoder_layers=int(model_fields.get("decoder_layers", 0)),
    )
    materialized = copy.deepcopy(dict(payload))
    materialized["schema_version"] = "robustness-adapter-training-config/v5"
    materialized["method_evidence"]["artifact_sha256"] = artifact_evidence.evidence_sha256  # type: ignore[index]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(materialized, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        protocol = load_adapter_training_config(temporary)
        resolve_training_method(protocol, evidence=artifact_evidence)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return load_adapter_training_config(output)


__all__ = [
    "PROBE_FACTORIAL_CONDITIONS",
    "ProbeSemanticSubspaceTrainingEvidence",
    "ProbeTransitionTrainingEvidence",
    "ProbeTransitionStateTrainingEvidence",
    "ResolvedTrainingMethod",
    "count_matched_random_layers",
    "load_probe_semantic_subspace_training_evidence",
    "load_probe_transition_training_evidence",
    "load_probe_transition_state_training_evidence",
    "materialize_probe_transition_state_training_config",
    "materialize_probe_output_factorial_configs",
    "materialize_probe_semantic_subspace_training_config",
    "materialize_probe_transition_training_config",
    "resolve_training_method",
]
