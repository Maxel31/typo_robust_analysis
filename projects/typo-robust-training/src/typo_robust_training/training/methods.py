"""Pure resolution of method evidence into trainable and supervised scopes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.training.config import AdapterTrainingProtocol


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
    """Consumer interface supplied by the independently reviewed probe loader.

    The artifact parser is intentionally implemented in the probe-evidence branch.
    Keeping the signature here lets the training consumer be reviewed independently
    and fail closed until that producer is integrated.
    """

    del path, model, model_revision, decoder_layers
    raise NotImplementedError("the probe-transition evidence loader is not integrated")


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
    "resolve_training_method",
]
