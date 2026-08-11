"""Strict, result-independent protocol for subword-position patching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest.records import strict_loads

_SCHEMA = "subword-position-patching-config/v1"
_MODEL = "google/gemma-3-4b-it"
_TASK = "gsm8k"
_MODES = ("first", "final", "all")
_POLICY = "equal-count-primary"
_SECONDARY = "nearest-normalized-position-half-up-endpoints/v1"


@dataclass(frozen=True, slots=True)
class SubwordPositionPatchingProtocol:
    schema_version: str
    model: str
    task: str
    cohort_pairs: int
    cohort_selection: str
    direction: str
    window: tuple[int, int]
    site: str
    coordinate_source: str
    modes: tuple[str, ...]
    token_count_policy: str
    subset_scope: str
    primary_subset: str
    primary_alignment: str
    secondary_subset: str
    secondary_alignment: str
    singleton_alignment: str
    dtype: str
    max_new_tokens: int
    termination_protocol: str
    answer_target: str
    answer_extraction: str
    historical_final_audit: bool
    empty_rate_policy: str
    config_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cohort": {
                "model": self.model,
                "task": self.task,
                "restoration_pairs": self.cohort_pairs,
                "selection": self.cohort_selection,
            },
            "intervention": {
                "direction": self.direction,
                "window": {"start": self.window[0], "stop": self.window[1]},
                "site": self.site,
                "coordinates": self.coordinate_source,
            },
            "modes": list(self.modes),
            "alignment": {
                "token_count_policy": self.token_count_policy,
                "subset_scope": self.subset_scope,
                "primary_subset": self.primary_subset,
                "primary_all": self.primary_alignment,
                "secondary_subset": self.secondary_subset,
                "secondary": self.secondary_alignment,
                "singleton": self.singleton_alignment,
            },
            "generation": {
                "dtype": self.dtype,
                "decoding": "greedy",
                "max_new_tokens": self.max_new_tokens,
                "termination_protocol": self.termination_protocol,
                "answer_target": self.answer_target,
                "answer_extraction": self.answer_extraction,
            },
            "reporting": {
                "subsets": [self.primary_subset, self.secondary_subset],
                "historical_final_audit": self.historical_final_audit,
                "empty_rate_policy": self.empty_rate_policy,
            },
        }


def _mapping(value: object, *, field: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"subword-position {field} fields differ from the frozen contract")
    return value


def load_subword_position_patching_protocol(
    path: Path,
) -> SubwordPositionPatchingProtocol:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"subword-position config is not a file: {resolved}")
    raw = resolved.read_bytes()
    payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    payload = _mapping(
        payload,
        field="config",
        fields={
            "schema_version",
            "cohort",
            "intervention",
            "modes",
            "alignment",
            "generation",
            "reporting",
        },
    )
    if payload["schema_version"] != _SCHEMA:
        raise ValueError("subword-position schema differs")
    cohort = _mapping(
        payload["cohort"],
        field="cohort",
        fields={"model", "task", "restoration_pairs", "selection"},
    )
    if cohort != {
        "model": _MODEL,
        "task": _TASK,
        "restoration_pairs": 172,
        "selection": "primary-restoration-cohort/v1",
    }:
        raise ValueError("subword-position cohort differs")
    intervention = _mapping(
        payload["intervention"],
        field="intervention",
        fields={"direction", "window", "site", "coordinates"},
    )
    window = _mapping(intervention["window"], field="window", fields={"start", "stop"})
    if window != {"start": 0, "stop": 6}:
        raise ValueError("subword-position window must be [0,6)")
    if intervention["site"] != "complete-decoder-block-residual-output":
        raise ValueError("subword-position intervention site differs")
    if (
        intervention["direction"] != "clean-to-typo"
        or intervention["coordinates"] != "manifest-edit-token-indices-retokenized/v1"
    ):
        raise ValueError("subword-position intervention direction or coordinates differ")
    if not isinstance(payload["modes"], list) or tuple(payload["modes"]) != _MODES:
        raise ValueError("subword-position mode inventory or order differs")
    alignment = _mapping(
        payload["alignment"],
        field="alignment",
        fields={
            "token_count_policy",
            "subset_scope",
            "primary_subset",
            "primary_all",
            "secondary_subset",
            "secondary",
            "singleton",
        },
    )
    expected_alignment = {
        "token_count_policy": _POLICY,
        "subset_scope": "pair-level-shared-across-all-modes/v1",
        "primary_subset": "all-edited-words-equal-subtoken-count/v1",
        "primary_all": "exact-within-word-ordinal/v1",
        "secondary_subset": "any-edited-word-subtoken-count-mismatch/v1",
        "secondary": _SECONDARY,
        "singleton": "clean-word-final/v1",
    }
    if alignment != expected_alignment:
        raise ValueError("subword-position alignment contract differs")
    generation = _mapping(
        payload["generation"],
        field="generation",
        fields={
            "dtype",
            "decoding",
            "max_new_tokens",
            "termination_protocol",
            "answer_target",
            "answer_extraction",
        },
    )
    expected_generation = {
        "dtype": "bfloat16",
        "decoding": "greedy",
        "max_new_tokens": 512,
        "termination_protocol": "effective-eos-vs-length-cap/v1",
        "answer_target": "manifest-stored-clean-answer/v1",
        "answer_extraction": "primary-then-empty-only-positional/v1",
    }
    if generation != expected_generation:
        raise ValueError("subword-position generation contract differs")
    reporting = _mapping(
        payload["reporting"],
        field="reporting",
        fields={"subsets", "historical_final_audit", "empty_rate_policy"},
    )
    if reporting != {
        "subsets": ["equal-count-primary", "mismatch-monotone-secondary"],
        "historical_final_audit": True,
        "empty_rate_policy": "json-null-csv-blank-with-defined-flag/v1",
    }:
        raise ValueError("subword-position reporting contract differs")
    return SubwordPositionPatchingProtocol(
        schema_version=_SCHEMA,
        model=_MODEL,
        task=_TASK,
        cohort_pairs=172,
        cohort_selection="primary-restoration-cohort/v1",
        direction="clean-to-typo",
        window=(0, 6),
        site="complete-decoder-block-residual-output",
        coordinate_source="manifest-edit-token-indices-retokenized/v1",
        modes=_MODES,
        token_count_policy=_POLICY,
        subset_scope="pair-level-shared-across-all-modes/v1",
        primary_subset="equal-count-primary",
        primary_alignment="exact-within-word-ordinal/v1",
        secondary_subset="mismatch-monotone-secondary",
        secondary_alignment=_SECONDARY,
        singleton_alignment="clean-word-final/v1",
        dtype="bfloat16",
        max_new_tokens=512,
        termination_protocol="effective-eos-vs-length-cap/v1",
        answer_target="manifest-stored-clean-answer/v1",
        answer_extraction="primary-then-empty-only-positional/v1",
        historical_final_audit=True,
        empty_rate_policy="json-null-csv-blank-with-defined-flag/v1",
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "SubwordPositionPatchingProtocol",
    "load_subword_position_patching_protocol",
]
