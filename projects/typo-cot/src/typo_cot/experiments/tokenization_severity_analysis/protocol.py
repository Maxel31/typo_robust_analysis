"""Strict result-independent protocol for tokenization-severity strata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest.records import strict_loads

_SCHEMA = "tokenization-severity-analysis-config/v1"
_SOURCE_SCHEMA = "six-setting-patch-controls-run/v1"
_CONTROLS = ("correct", "offset-2", "cross-item")
_REQUIRED_PAIRS = 1_241
_DIMENSIONS = (
    (
        "subtoken-count-change",
        ("unchanged-all-edits", "changed-any-edit"),
        "per-edit-token-index-cardinality-all-equal-vs-any-different/v1",
    ),
    (
        "typo-fragmentation",
        ("increased-any-edit", "not-increased"),
        "any-typo-cardinality-greater-than-clean/v1",
    ),
    (
        "edit-count",
        ("1", "2", "3-4"),
        "aligned-edit-cardinality/v1",
    ),
    (
        "clean-edited-word-tokenization",
        ("all-single-token", "any-multi-token"),
        "all-clean-cardinality-one-vs-any-greater-than-one/v1",
    ),
)
_DENOMINATORS = ("arm-valid", "common-valid")
_SCOPES = ("overall", "setting")
_EMPTY_CELLS = "emit-zero-denominator/v1"


@dataclass(frozen=True, slots=True)
class TokenizationSeverityProtocol:
    """The complete, prospective CPU analysis contract."""

    schema_version: str
    controls_run_schema: str
    require_confirmatory: bool
    required_pairs: int
    controls: tuple[str, ...]
    dimensions: dict[str, tuple[str, ...]]
    dimension_rules: dict[str, str]
    denominators: tuple[str, ...]
    scopes: tuple[str, ...]
    empty_cells: str
    config_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "controls_run_schema": self.controls_run_schema,
                "require_confirmatory": self.require_confirmatory,
                "required_pairs": self.required_pairs,
                "required_arms": list(self.controls),
            },
            "dimensions": [
                {
                    "name": name,
                    "bins": list(bins),
                    "rule": self.dimension_rules[name],
                }
                for name, bins in self.dimensions.items()
            ],
            "denominators": list(self.denominators),
            "scopes": list(self.scopes),
            "empty_cells": self.empty_cells,
        }


def _mapping(value: object, *, field: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} fields differ from the frozen contract")
    return value


def load_tokenization_severity_protocol(path: Path) -> TokenizationSeverityProtocol:
    """Load JSON-compatible YAML and reject any analysis-protocol drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"tokenization severity config is not a file: {resolved}")
    raw = resolved.read_bytes()
    root = _mapping(
        strict_loads(raw.decode("utf-8"), context=str(resolved)),
        field="tokenization severity config",
        keys={
            "schema_version",
            "source",
            "dimensions",
            "denominators",
            "scopes",
            "empty_cells",
        },
    )
    if root.get("schema_version") != _SCHEMA:
        raise ValueError("tokenization severity schema differs")
    source = _mapping(
        root.get("source"),
        field="source",
        keys={
            "controls_run_schema",
            "require_confirmatory",
            "required_pairs",
            "required_arms",
        },
    )
    if source != {
        "controls_run_schema": _SOURCE_SCHEMA,
        "require_confirmatory": True,
        "required_pairs": _REQUIRED_PAIRS,
        "required_arms": list(_CONTROLS),
    }:
        raise ValueError("tokenization severity source contract differs")
    raw_dimensions = root.get("dimensions")
    expected_dimensions = [
        {"name": name, "bins": list(bins), "rule": rule} for name, bins, rule in _DIMENSIONS
    ]
    if raw_dimensions != expected_dimensions:
        raise ValueError("tokenization severity dimension contract differs")
    if root.get("denominators") != list(_DENOMINATORS):
        raise ValueError("tokenization severity denominator contract differs")
    if root.get("scopes") != list(_SCOPES):
        raise ValueError("tokenization severity scope contract differs")
    if root.get("empty_cells") != _EMPTY_CELLS:
        raise ValueError("tokenization severity empty-cell policy differs")
    return TokenizationSeverityProtocol(
        schema_version=_SCHEMA,
        controls_run_schema=_SOURCE_SCHEMA,
        require_confirmatory=True,
        required_pairs=_REQUIRED_PAIRS,
        controls=_CONTROLS,
        dimensions={name: bins for name, bins, _rule in _DIMENSIONS},
        dimension_rules={name: rule for name, _bins, rule in _DIMENSIONS},
        denominators=_DENOMINATORS,
        scopes=_SCOPES,
        empty_cells=_EMPTY_CELLS,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["TokenizationSeverityProtocol", "load_tokenization_severity_protocol"]
