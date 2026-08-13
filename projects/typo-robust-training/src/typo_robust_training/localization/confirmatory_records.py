"""Per-pair observations for confirmatory joint-window patching."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _trajectory(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError(f"{field} must contain finite non-negative values")
    return result


@dataclass(frozen=True, slots=True)
class JointWindowScan:
    record_id: str
    role: str
    decoder_layers: int
    window_width: int
    target_token_ids: tuple[int, ...]
    untreated_kl_2_16: tuple[float, ...]
    patched_kl_2_16_by_window: Mapping[int, tuple[float, ...]]
    invalid_reason: str | None
    audit: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("joint-window scan record_id must be non-empty")
        if self.role not in {"selection", "validation"}:
            raise ValueError("joint-window scan role differs")
        if (
            isinstance(self.decoder_layers, bool)
            or not isinstance(self.decoder_layers, int)
            or self.decoder_layers <= 0
            or isinstance(self.window_width, bool)
            or not isinstance(self.window_width, int)
            or not 0 < self.window_width <= self.decoder_layers
        ):
            raise ValueError("joint-window scan layer dimensions are invalid")
        targets = tuple(self.target_token_ids)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in targets
        ):
            raise ValueError("joint-window target token IDs are invalid")
        untreated = _trajectory(self.untreated_kl_2_16, field="untreated KL")
        if not isinstance(self.patched_kl_2_16_by_window, Mapping):
            raise TypeError("patched window KL must be an object")
        patched: dict[int, tuple[float, ...]] = {}
        for start, values in self.patched_kl_2_16_by_window.items():
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or not 0 <= start <= self.decoder_layers - self.window_width
                or start in patched
            ):
                raise ValueError("patched window start is invalid or duplicated")
            patched[start] = _trajectory(values, field=f"patched window {start} KL")
        if untreated:
            if len(targets) != 16 or len(untreated) != 15:
                raise ValueError("valid joint-window scans require 16 targets and 15 KL values")
            if any(len(values) != 15 for values in patched.values()):
                raise ValueError("valid patched window trajectories require 15 KL values")
            if self.role == "selection" and set(patched) != set(
                range(self.decoder_layers - self.window_width + 1)
            ):
                raise ValueError("selection scan must preserve every candidate window")
            if self.role == "validation" and len(patched) != 1:
                raise ValueError("validation scan must contain only the frozen window")
            if self.invalid_reason is not None:
                raise ValueError("valid joint-window scan cannot have an invalid reason")
        else:
            if targets or patched:
                raise ValueError("invalid joint-window scan cannot contain readout values")
            if not isinstance(self.invalid_reason, str) or not self.invalid_reason:
                raise ValueError("invalid joint-window scan requires an explicit reason")
        if not isinstance(self.audit, Mapping):
            raise TypeError("joint-window scan audit must be an object")
        audit = dict(self.audit)
        try:
            json.dumps(audit, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("joint-window scan audit must be canonical JSON") from exc
        object.__setattr__(self, "target_token_ids", targets)
        object.__setattr__(self, "untreated_kl_2_16", untreated)
        object.__setattr__(self, "patched_kl_2_16_by_window", MappingProxyType(patched))
        object.__setattr__(self, "audit", MappingProxyType(audit))

    @property
    def untreated_mean_kl(self) -> float | None:
        if not self.untreated_kl_2_16:
            return None
        return sum(self.untreated_kl_2_16) / len(self.untreated_kl_2_16)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-joint-window-scan/v1",
            "record_id": self.record_id,
            "role": self.role,
            "decoder_layers": self.decoder_layers,
            "window_width": self.window_width,
            "target_token_ids": list(self.target_token_ids),
            "untreated_kl_2_16": list(self.untreated_kl_2_16),
            "patched_kl_2_16_by_window": [
                {"start": start, "kl_2_16": list(values)}
                for start, values in sorted(self.patched_kl_2_16_by_window.items())
            ],
            "invalid_reason": self.invalid_reason,
            "audit": dict(self.audit),
        }

    @classmethod
    def from_dict(cls, value: object) -> JointWindowScan:
        expected = {
            "schema_version",
            "record_id",
            "role",
            "decoder_layers",
            "window_width",
            "target_token_ids",
            "untreated_kl_2_16",
            "patched_kl_2_16_by_window",
            "invalid_reason",
            "audit",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema_version") != "robustness-joint-window-scan/v1"
        ):
            raise ValueError("joint-window scan payload fields or schema differ")
        rows = value["patched_kl_2_16_by_window"]
        if not isinstance(rows, list):
            raise ValueError("patched window payload must be a list")
        patched: dict[int, tuple[float, ...]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"start", "kl_2_16"}:
                raise ValueError("patched window row fields differ")
            start = row["start"]
            if isinstance(start, bool) or not isinstance(start, int) or start in patched:
                raise ValueError("patched window starts are invalid or duplicated")
            if not isinstance(row["kl_2_16"], list):
                raise ValueError("patched window KL must be a list")
            patched[start] = tuple(row["kl_2_16"])
        return cls(
            record_id=value["record_id"],  # type: ignore[arg-type]
            role=value["role"],  # type: ignore[arg-type]
            decoder_layers=value["decoder_layers"],  # type: ignore[arg-type]
            window_width=value["window_width"],  # type: ignore[arg-type]
            target_token_ids=tuple(value["target_token_ids"]),  # type: ignore[arg-type]
            untreated_kl_2_16=tuple(value["untreated_kl_2_16"]),  # type: ignore[arg-type]
            patched_kl_2_16_by_window=patched,
            invalid_reason=value["invalid_reason"],  # type: ignore[arg-type]
            audit=value["audit"],  # type: ignore[arg-type]
        )


__all__ = ["JointWindowScan"]
