"""Shared condition-specific evidence contracts for training and evaluation."""

from __future__ import annotations

import re


_SHA256 = re.compile(r"[0-9a-f]{64}")

LEGACY_ADAPTER_CONDITIONS = frozenset(
    {
        "noisy-language-model",
        "output-matching",
        "global-state-alignment",
        "localized-state-distillation",
        "random-window-state-distillation",
    }
)
LOCALIZATION_EVIDENCE_CONDITIONS = frozenset(
    {
        "localized-state-distillation",
        "random-window-state-distillation",
    }
)
METHOD_EVIDENCE_CONDITIONS = frozenset(
    {
        "probe-transition-output-matching",
        "probe-transition-state-distillation",
        "causal-probe-subspace-distillation",
    }
)
SUPPORTED_ADAPTER_CONDITIONS = LEGACY_ADAPTER_CONDITIONS | METHOD_EVIDENCE_CONDITIONS


def optional_sha256(value: object, *, field: str) -> str | None:
    """Return a canonical optional digest or reject ambiguous provenance."""

    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256 digest or null")
    return value


def validate_condition_evidence(
    *,
    condition: object,
    localization_sha256: object,
    method_evidence_sha256: object,
) -> tuple[str | None, str | None]:
    """Validate the one evidence namespace owned by an adapter condition.

    Historical conditions retain their exact ``localization_sha256`` contract.
    The v4 methods use a separate generic evidence digest so a probe or kill-test
    artifact can never be misrepresented as Activation Patching localization.
    """

    if condition not in SUPPORTED_ADAPTER_CONDITIONS:
        raise ValueError("adapter condition is unsupported")
    localization = optional_sha256(
        localization_sha256,
        field="localization_sha256",
    )
    method = optional_sha256(
        method_evidence_sha256,
        field="method_evidence_sha256",
    )
    if condition in LOCALIZATION_EVIDENCE_CONDITIONS:
        if localization is None or method is not None:
            raise ValueError("adapter localization provenance differs")
    elif condition in METHOD_EVIDENCE_CONDITIONS:
        if method is None or localization is not None:
            raise ValueError("adapter method evidence provenance differs")
    elif localization is not None or method is not None:
        raise ValueError("adapter condition cannot consume evidence provenance")
    return localization, method


__all__ = [
    "LEGACY_ADAPTER_CONDITIONS",
    "LOCALIZATION_EVIDENCE_CONDITIONS",
    "METHOD_EVIDENCE_CONDITIONS",
    "SUPPORTED_ADAPTER_CONDITIONS",
    "optional_sha256",
    "validate_condition_evidence",
]
