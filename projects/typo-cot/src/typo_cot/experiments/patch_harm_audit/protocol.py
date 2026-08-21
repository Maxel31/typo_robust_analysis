"""Strict, result-independent protocol for the correct-answer harm audit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest.records import strict_loads

_SCHEMA = "patch-harm-audit-config/v1"
_COHORT = "clean-correct-typo-correct"
_SELECTION = "all-prepared-clean-correct-typo-correct-aligned-uncapped/v1"
_BASELINE_SOURCE = "manifest-stored-deterministically-reextracted-typo-answer/v1"
_WINDOW = (0, 6)
_DIRECTION = "clean-to-typo"
_SITE = "complete-decoder-block-residual-output"
_COORDINATES = "rebuttal-pair-manifest-correct/v1"
_DTYPE = "bfloat16"
_DECODING = "greedy"
_MAX_NEW_TOKENS = 512
_TERMINATION = "effective-eos-vs-length-cap/v1"
_PRESERVE = "patched-correct/v1"
_HARM = "patched-incorrect-including-unextractable/v1"
_ANSWER_CHANGED = "canonical-patched-value-differs-from-stored-typo/v1"
_UNEXTRACTABLE = "retained-in-harm-and-denominator/v1"
_COMPOSITE_LABEL = "repair-harm-conditional-composite"
_REPAIR_SOURCE = "manifest-fixed-window-0:6-event/v1"
_RESTORATION_PAIRS = 1_241
_RESTORATION_SUCCESSES = 800
_UNCOVERED_POLICY = "report-and-withhold-population-net-accuracy/v1"


@dataclass(frozen=True, slots=True)
class PatchHarmAuditProtocol:
    """Complete prospective protocol loaded from the public config."""

    schema_version: str
    cohort: str
    selection: str
    baseline_source: str
    window: tuple[int, int]
    direction: str
    site: str
    coordinate_source: str
    dtype: str
    decoding: str
    max_new_tokens: int
    termination_protocol: str
    preserve_definition: str
    harm_definition: str
    answer_changed_definition: str
    unextractable_policy: str
    composite_label: str
    repair_source: str
    restoration_pairs: int
    restoration_successes: int
    uncovered_policy: str
    config_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cohort": {
                "name": self.cohort,
                "selection": self.selection,
                "baseline_source": self.baseline_source,
            },
            "intervention": {
                "direction": self.direction,
                "window": {"start": self.window[0], "stop": self.window[1]},
                "site": self.site,
                "coordinates": self.coordinate_source,
            },
            "generation": {
                "dtype": self.dtype,
                "decoding": self.decoding,
                "max_new_tokens": self.max_new_tokens,
                "termination_protocol": self.termination_protocol,
            },
            "outcomes": {
                "preserve": self.preserve_definition,
                "harm": self.harm_definition,
                "answer_changed": self.answer_changed_definition,
                "unextractable": self.unextractable_policy,
            },
            "composite": {
                "label": self.composite_label,
                "repair_source": self.repair_source,
                "restoration_pairs": self.restoration_pairs,
                "restoration_successes": self.restoration_successes,
                "uncovered_policy": self.uncovered_policy,
            },
        }


def _mapping(value: object, *, field: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} fields differ from the frozen contract")
    return value


def load_patch_harm_audit_protocol(path: Path) -> PatchHarmAuditProtocol:
    """Load JSON-compatible YAML and reject every frozen-field drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"patch harm config is not a file: {resolved}")
    raw = resolved.read_bytes()
    root = _mapping(
        strict_loads(raw.decode("utf-8"), context=str(resolved)),
        field="patch harm config",
        keys={"schema_version", "cohort", "intervention", "generation", "outcomes", "composite"},
    )
    if root.get("schema_version") != _SCHEMA:
        raise ValueError("patch harm config schema differs")

    cohort = _mapping(
        root.get("cohort"),
        field="cohort",
        keys={"name", "selection", "baseline_source"},
    )
    if cohort != {
        "name": _COHORT,
        "selection": _SELECTION,
        "baseline_source": _BASELINE_SOURCE,
    }:
        raise ValueError("patch harm cohort contract differs")

    intervention = _mapping(
        root.get("intervention"),
        field="intervention",
        keys={"direction", "window", "site", "coordinates"},
    )
    window = _mapping(
        intervention.get("window"),
        field="intervention.window",
        keys={"start", "stop"},
    )
    if window != {"start": _WINDOW[0], "stop": _WINDOW[1]}:
        raise ValueError("patch harm audit requires the frozen [0,6) window")
    if intervention != {
        "direction": _DIRECTION,
        "window": window,
        "site": _SITE,
        "coordinates": _COORDINATES,
    }:
        raise ValueError("patch harm intervention contract differs")

    generation = _mapping(
        root.get("generation"),
        field="generation",
        keys={"dtype", "decoding", "max_new_tokens", "termination_protocol"},
    )
    if generation != {
        "dtype": _DTYPE,
        "decoding": _DECODING,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "termination_protocol": _TERMINATION,
    }:
        raise ValueError("patch harm generation contract differs")

    outcomes = _mapping(
        root.get("outcomes"),
        field="outcomes",
        keys={"preserve", "harm", "answer_changed", "unextractable"},
    )
    if outcomes != {
        "preserve": _PRESERVE,
        "harm": _HARM,
        "answer_changed": _ANSWER_CHANGED,
        "unextractable": _UNEXTRACTABLE,
    }:
        raise ValueError("patch harm outcome contract differs")

    composite = _mapping(
        root.get("composite"),
        field="composite",
        keys={
            "label",
            "repair_source",
            "restoration_pairs",
            "restoration_successes",
            "uncovered_policy",
        },
    )
    if composite != {
        "label": _COMPOSITE_LABEL,
        "repair_source": _REPAIR_SOURCE,
        "restoration_pairs": _RESTORATION_PAIRS,
        "restoration_successes": _RESTORATION_SUCCESSES,
        "uncovered_policy": _UNCOVERED_POLICY,
    }:
        raise ValueError("patch harm composite contract differs")

    return PatchHarmAuditProtocol(
        schema_version=_SCHEMA,
        cohort=_COHORT,
        selection=_SELECTION,
        baseline_source=_BASELINE_SOURCE,
        window=_WINDOW,
        direction=_DIRECTION,
        site=_SITE,
        coordinate_source=_COORDINATES,
        dtype=_DTYPE,
        decoding=_DECODING,
        max_new_tokens=_MAX_NEW_TOKENS,
        termination_protocol=_TERMINATION,
        preserve_definition=_PRESERVE,
        harm_definition=_HARM,
        answer_changed_definition=_ANSWER_CHANGED,
        unextractable_policy=_UNEXTRACTABLE,
        composite_label=_COMPOSITE_LABEL,
        repair_source=_REPAIR_SOURCE,
        restoration_pairs=_RESTORATION_PAIRS,
        restoration_successes=_RESTORATION_SUCCESSES,
        uncovered_policy=_UNCOVERED_POLICY,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["PatchHarmAuditProtocol", "load_patch_harm_audit_protocol"]
