"""Pure resolution of method evidence into trainable and supervised scopes."""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.data.config import strict_loads
from typo_robust_training.training.config import AdapterTrainingProtocol
from typo_robust_training.training.config import load_adapter_training_config


_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ProbeTransitionTrainingEvidence:
    """Minimal, hash-bound view of independently validated probe evidence."""

    model: str
    model_revision: str
    decoder_layers: int
    selected_transition_layer: int
    evidence_sha256: str

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

    @property
    def suffix_layers(self) -> tuple[int, ...]:
        return tuple(range(self.selected_transition_layer, self.decoder_layers))


@dataclass(frozen=True, slots=True)
class ResolvedTrainingMethod:
    adapter_layers: tuple[int, ...]
    state_layers: tuple[int, ...]
    state_target: str
    method_evidence_sha256: str


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

    artifact = load_probe_transition_artifact(supplied)
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


def resolve_training_method(
    protocol: AdapterTrainingProtocol,
    *,
    evidence: ProbeTransitionTrainingEvidence,
) -> ResolvedTrainingMethod:
    """Resolve v4 method scope without inspecting evaluation outcomes."""

    if not isinstance(protocol, AdapterTrainingProtocol):
        raise TypeError("training protocol must be AdapterTrainingProtocol")
    if not isinstance(evidence, ProbeTransitionTrainingEvidence):
        raise TypeError("probe transition evidence has the wrong type")
    if protocol.condition != "probe-transition-output-matching":
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
    return ResolvedTrainingMethod(
        adapter_layers=evidence.suffix_layers,
        state_layers=(),
        state_target="none",
        method_evidence_sha256=evidence.evidence_sha256,
    )


__all__ = [
    "ProbeTransitionTrainingEvidence",
    "ResolvedTrainingMethod",
    "load_probe_transition_training_evidence",
    "materialize_probe_transition_training_config",
    "resolve_training_method",
]
