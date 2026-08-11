"""Hash-bound, resumable runner for multi-token teacher-forced KL readout."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Protocol

from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
    load_rebuttal_pair_manifest,
)
from typo_cot.experiments.build_rebuttal_manifest.records import (
    iter_jsonl_objects,
    load_json_object,
    sha256_file,
)
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.multitoken_kl_readout.metrics import summarize_restoration
from typo_cot.experiments.multitoken_kl_readout.protocol import (
    MultiTokenKLReadoutProtocol,
    load_multitoken_kl_readout_protocol,
)
from typo_cot.experiments.multitoken_kl_readout.statistics import bootstrap_median
from typo_cot.experiments.six_setting_patch_controls.statistics import derived_seed

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RUN_SCHEMA = "multitoken-kl-readout-run/v1"
_CHECKPOINT_SCHEMA = "multitoken-kl-readout-checkpoint/v1"
_RECORD_SCHEMA = "multitoken-kl-readout-record/v1"
_SUMMARY_SCHEMA = "multitoken-kl-readout-summary/v1"
_PUBLIC_OUTPUTS = (
    "multitoken_kl_records.jsonl",
    "setting_metrics.csv",
    "token_position_trajectory.csv",
    "token_position_trajectory.svg",
    "multitoken_summary.json",
)
_METRIC_RANGES = {
    "R_1": (1, 1),
    "R_2:4": (2, 4),
    "R_2:8": (2, 8),
    "R_2:16": (2, 16),
}
_METRIC_COLUMNS = {
    "R_1": "r_1",
    "R_2:4": "r_2_4",
    "R_2:8": "r_2_8",
    "R_2:16": "r_2_16",
    "R_1_minus_R_2:16": "r_1_minus_r_2_16",
}


class MultiTokenKLReadoutRunError(RuntimeError):
    """Raised after retaining verified checkpoints for a failed GPU run."""


def _finite_nonnegative(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not normalized or any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError(f"{field} must contain finite non-negative values")
    return normalized


@dataclass(frozen=True, slots=True)
class MultiTokenKLScan:
    """One pair's common targets and untreated/patched KL trajectories."""

    target_token_ids: tuple[int, ...]
    target_token_text: tuple[str, ...]
    untreated_kl: tuple[float, ...]
    patched_kl: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_token_ids", tuple(self.target_token_ids))
        object.__setattr__(self, "target_token_text", tuple(self.target_token_text))
        object.__setattr__(
            self, "untreated_kl", _finite_nonnegative(self.untreated_kl, field="untreated_kl")
        )
        object.__setattr__(
            self, "patched_kl", _finite_nonnegative(self.patched_kl, field="patched_kl")
        )
        lengths = {
            len(self.target_token_ids),
            len(self.target_token_text),
            len(self.untreated_kl),
            len(self.patched_kl),
        }
        if len(lengths) != 1:
            raise ValueError("multi-token scan fields must have equal length")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.target_token_ids
        ):
            raise ValueError("target_token_ids must contain non-negative integers")
        if any(not isinstance(text, str) for text in self.target_token_text):
            raise ValueError("target_token_text must contain strings")


@dataclass(frozen=True, slots=True)
class MultiTokenKLReadoutConfig:
    """Public command arguments, including explicit non-confirmatory limits."""

    protocol_path: Path
    manifest_path: Path
    teacher_forced_tokens: int
    primary_token_range: tuple[int, int]
    gpu_id: str
    output_dir: Path
    limit_per_setting: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("protocol_path", "manifest_path", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        object.__setattr__(self, "primary_token_range", tuple(self.primary_token_range))
        if (
            isinstance(self.teacher_forced_tokens, bool)
            or not isinstance(self.teacher_forced_tokens, int)
            or self.teacher_forced_tokens <= 0
        ):
            raise ValueError("teacher_forced_tokens must be a positive integer")
        if (
            len(self.primary_token_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.primary_token_range
            )
            or not 1
            <= self.primary_token_range[0]
            <= self.primary_token_range[1]
            <= self.teacher_forced_tokens
        ):
            raise ValueError("primary_token_range must be within teacher-forced tokens")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit_per_setting is not None and (
            isinstance(self.limit_per_setting, bool)
            or not isinstance(self.limit_per_setting, int)
            or self.limit_per_setting <= 0
        ):
            raise ValueError("limit_per_setting must be a positive integer")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be boolean")
        output = self.output_dir.resolve()
        manifest_directory = self.manifest_path.resolve().parent
        if (
            output == manifest_directory
            or output in manifest_directory.parents
            or manifest_directory in output.parents
        ):
            raise ValueError("output and manifest directories must not overlap")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("protocol_path", "manifest_path", "output_dir"):
            payload[field] = str(Path(payload[field]).resolve())
        payload["primary_token_range"] = list(self.primary_token_range)
        payload.pop("resume")
        return payload


class MultiTokenKLRuntime(Protocol):
    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_pair(
        self,
        pair: Mapping[str, object],
        *,
        teacher_forced_tokens: int,
        layer_window: tuple[int, int],
    ) -> MultiTokenKLScan: ...


RuntimeFactory = Callable[..., MultiTokenKLRuntime]


@dataclass(frozen=True, slots=True)
class MultiTokenKLReadoutResult:
    records_path: Path
    setting_metrics_path: Path
    trajectory_path: Path
    trajectory_plot_path: Path
    summary_path: Path
    run_path: Path
    pairs: int
    primary_valid_pairs: int
    settings: int
    trajectory_rows: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_factory() -> RuntimeFactory:
    from typo_cot.experiments.multitoken_kl_readout.runtime import (
        HuggingFaceMultiTokenKLReadoutRuntime,
    )

    return HuggingFaceMultiTokenKLReadoutRuntime


def _runtime_provenance(
    runtime: MultiTokenKLRuntime,
    *,
    model: str,
    task: str,
    revision: str,
    gpu_id: str,
    protocol: MultiTokenKLReadoutProtocol,
) -> dict[str, object]:
    provenance = dict(runtime.provenance())
    if (
        provenance.get("operation") != "multitoken-kl-readout"
        or provenance.get("model") != model
        or provenance.get("task") != task
        or provenance.get("requested_revision") != revision
        or provenance.get("model_revision") != revision
        or provenance.get("tokenizer_revision") != revision
        or provenance.get("dtype") != "bfloat16"
        or provenance.get("cuda_visible_devices") != gpu_id
        or provenance.get("coordinate_source") != "rebuttal-pair-manifest/v1"
        or provenance.get("layer_window") != list(protocol.window)
        or provenance.get("teacher_forced_tokens") != protocol.teacher_forced_tokens
        or provenance.get("target_source") != protocol.target_source
        or provenance.get("prompt_prefix_validation") != protocol.prompt_prefix_validation
        or provenance.get("model_inputs") != protocol.model_inputs
        or provenance.get("divergence") != protocol.divergence
        or provenance.get("negative_kl_roundoff_tolerance")
        != protocol.negative_kl_roundoff_tolerance
    ):
        raise ValueError("multi-token runtime provenance differs from the frozen protocol")
    forward = _mapping(provenance.get("forward"), field="runtime forward")
    if dict(forward) != {
        "use_cache": False,
        "logits_materialized_dtype": protocol.logits_materialized_dtype,
        "kl_dtype": protocol.kl_dtype,
    }:
        raise ValueError("multi-token runtime forward provenance differs")
    layers = provenance.get("num_decoder_layers")
    if (
        isinstance(layers, bool)
        or not isinstance(layers, int)
        or layers < protocol.window[1]
        or runtime.num_layers != layers
    ):
        raise ValueError("multi-token runtime decoder-layer count differs")
    return provenance


def _select_records(
    records: Sequence[Mapping[str, object]],
    *,
    limit_per_setting: int | None,
) -> tuple[Mapping[str, object], ...]:
    selected: list[Mapping[str, object]] = []
    for setting in REBUTTAL_SETTINGS:
        candidates = [
            record
            for record in records
            if record["cohorts"]["restoration"] is True
            and (record["model"], record["task"]) == setting.key
        ]
        if len(candidates) != setting.paper_denominator:
            raise ValueError(f"multi-token paper denominator differs for {setting.slug}")
        candidates.sort(
            key=lambda record: (
                str(record["target_rule"]),
                str(record["sample_id"]),
                str(record["pair_id"]),
            )
        )
        if limit_per_setting is not None:
            if limit_per_setting > len(candidates):
                raise ValueError(f"multi-token smoke limit exceeds {setting.slug} denominator")
            candidates = candidates[:limit_per_setting]
        selected.extend(candidates)
    pair_ids = [str(record["pair_id"]) for record in selected]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("multi-token selected pair IDs are duplicated")
    return tuple(selected)


def _scan_payload(scan: MultiTokenKLScan) -> dict[str, object]:
    return {
        "target_token_ids": list(scan.target_token_ids),
        "target_token_text": list(scan.target_token_text),
        "untreated_kl": list(scan.untreated_kl),
        "patched_kl": list(scan.patched_kl),
    }


def _scan_from_payload(value: object, *, field: str) -> MultiTokenKLScan:
    payload = _mapping(value, field=field)
    if set(payload) != {"target_token_ids", "target_token_text", "untreated_kl", "patched_kl"}:
        raise ValueError(f"{field} fields differ from the checkpoint contract")
    for name in payload:
        if not isinstance(payload[name], list):
            raise ValueError(f"{field}.{name} must be a list")
    return MultiTokenKLScan(
        target_token_ids=tuple(payload["target_token_ids"]),  # type: ignore[arg-type]
        target_token_text=tuple(payload["target_token_text"]),  # type: ignore[arg-type]
        untreated_kl=tuple(payload["untreated_kl"]),  # type: ignore[arg-type]
        patched_kl=tuple(payload["patched_kl"]),  # type: ignore[arg-type]
    )


def _validate_scan(scan: object, *, protocol: MultiTokenKLReadoutProtocol) -> MultiTokenKLScan:
    if not isinstance(scan, MultiTokenKLScan):
        raise ValueError("runtime result must be MultiTokenKLScan")
    if len(scan.target_token_ids) != protocol.teacher_forced_tokens:
        raise ValueError("runtime result does not contain exactly 16 target tokens")
    return scan


def _checkpoint_path(
    directory: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol_sha256: str,
    manifest_sha256: str,
) -> Path:
    """Return a content address bound to every reusable checkpoint input."""

    source = _mapping(record.get("source"), field="record source")
    address = _canonical_sha256(
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
            "pair_id": record.get("pair_id"),
            "source_record_sha256": source.get("source_record_sha256"),
            "runtime_fingerprint": runtime_fingerprint,
            "protocol_sha256": protocol_sha256,
            "manifest_sha256": manifest_sha256,
        }
    )
    return directory / f"{address}.json"


def _write_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    scan: MultiTokenKLScan,
    runtime_fingerprint: str,
    protocol: MultiTokenKLReadoutProtocol,
    manifest_sha256: str,
) -> None:
    source = _mapping(record.get("source"), field="record source")
    scan_payload = _scan_payload(scan)
    _write_json_atomic(
        path,
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
            "readout_protocol_sha256": protocol.config_sha256,
            "pair_manifest_sha256": manifest_sha256,
            "pair_id": record["pair_id"],
            "source_record_sha256": source.get("source_record_sha256"),
            "runtime_fingerprint": runtime_fingerprint,
            "scan_sha256": _canonical_sha256(scan_payload),
            "scan": scan_payload,
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: MultiTokenKLReadoutProtocol,
    manifest_sha256: str,
) -> MultiTokenKLScan:
    payload = load_json_object(path)
    source = _mapping(record.get("source"), field="record source")
    expected_fields = {
        "schema_version",
        "paper_sha256",
        "manifest_protocol_sha256",
        "readout_protocol_sha256",
        "pair_manifest_sha256",
        "pair_id",
        "source_record_sha256",
        "runtime_fingerprint",
        "scan_sha256",
        "scan",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema_version") != _CHECKPOINT_SCHEMA
        or payload.get("paper_sha256") != PAPER_SHA256
        or payload.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or payload.get("readout_protocol_sha256") != protocol.config_sha256
        or payload.get("pair_manifest_sha256") != manifest_sha256
        or payload.get("pair_id") != record.get("pair_id")
        or payload.get("source_record_sha256") != source.get("source_record_sha256")
        or payload.get("runtime_fingerprint") != runtime_fingerprint
        or payload.get("scan_sha256") != _canonical_sha256(payload.get("scan"))
    ):
        raise ValueError(f"multi-token checkpoint provenance differs: {path}")
    return _validate_scan(
        _scan_from_payload(payload.get("scan"), field="checkpoint scan"),
        protocol=protocol,
    )


def _metric_payload(
    untreated: tuple[float, ...],
    patched: tuple[float, ...],
    *,
    token_range: tuple[int, int],
    epsilon: float,
) -> dict[str, object]:
    summary = summarize_restoration(
        untreated_kl=untreated,
        patched_kl=patched,
        token_range=token_range,
        denominator_epsilon=epsilon,
    )
    if summary.invalid_reason is not None:
        return {
            "valid": False,
            "value": None,
            "untreated_mean_kl": summary.untreated_mean_kl,
            "patched_mean_kl": summary.patched_mean_kl,
            "invalid_reason": summary.invalid_reason,
        }
    return {
        "valid": True,
        "value": summary.value,
        "untreated_mean_kl": summary.untreated_mean_kl,
        "patched_mean_kl": summary.patched_mean_kl,
        "invalid_reason": None,
    }


def _compile_records(
    *,
    selected: Sequence[Mapping[str, object]],
    checkpoints_dir: Path,
    runtime_fingerprints: Mapping[tuple[str, str], str],
    protocol: MultiTokenKLReadoutProtocol,
    manifest_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in selected:
        setting = str(record["model"]), str(record["task"])
        source = _mapping(record.get("source"), field="record source")
        scan = _load_checkpoint(
            _checkpoint_path(
                checkpoints_dir,
                record=record,
                runtime_fingerprint=runtime_fingerprints[setting],
                protocol_sha256=protocol.config_sha256,
                manifest_sha256=manifest_sha256,
            ),
            record=record,
            runtime_fingerprint=runtime_fingerprints[setting],
            protocol=protocol,
            manifest_sha256=manifest_sha256,
        )
        metrics = {
            name: _metric_payload(
                scan.untreated_kl,
                scan.patched_kl,
                token_range=token_range,
                epsilon=protocol.denominator_epsilon,
            )
            for name, token_range in _METRIC_RANGES.items()
        }
        first, primary = metrics["R_1"], metrics["R_2:16"]
        if first["valid"] is True and primary["valid"] is True:
            first_minus_primary: dict[str, object] = {
                "valid": True,
                "value": float(first["value"]) - float(primary["value"]),
                "invalid_reason": None,
            }
        else:
            first_minus_primary = {
                "valid": False,
                "value": None,
                "invalid_reason": "first-or-primary-invalid",
            }
        metrics["R_1_minus_R_2:16"] = first_minus_primary
        rows.append(
            {
                "schema_version": _RECORD_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "readout_protocol_sha256": protocol.config_sha256,
                "pair_manifest_sha256": manifest_sha256,
                "source_record_sha256": source.get("source_record_sha256"),
                "runtime_fingerprint": runtime_fingerprints[setting],
                "pair_id": record["pair_id"],
                "sample_id": record["sample_id"],
                "model": record["model"],
                "task": record["task"],
                "target_rule": record["target_rule"],
                "target_token_ids": list(scan.target_token_ids),
                "target_token_text": list(scan.target_token_text),
                "untreated_kl": list(scan.untreated_kl),
                "patched_kl": list(scan.patched_kl),
                "raw_kl_reduction": [
                    untreated - patched
                    for untreated, patched in zip(
                        scan.untreated_kl, scan.patched_kl, strict=True
                    )
                ],
                "metrics": metrics,
            }
        )
    return rows


def _summary_or_null(
    values: Sequence[float],
    *,
    protocol: MultiTokenKLReadoutProtocol,
    label: str,
) -> dict[str, float | None]:
    if not values:
        return {"estimate": None, "lower": None, "upper": None}
    return bootstrap_median(
        tuple(values),
        replicates=protocol.pair_bootstrap_replicates,
        confidence_level=protocol.confidence_level,
        seed=derived_seed(protocol.bootstrap_seed, label),
    )


def _analysis_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    protocol: MultiTokenKLReadoutProtocol,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    setting_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    for setting in REBUTTAL_SETTINGS:
        selected = [row for row in rows if (row["model"], row["task"]) == setting.key]
        if not selected:
            raise ValueError(f"multi-token output has no rows for {setting.slug}")
        setting_row: dict[str, object] = {
            "model": setting.model,
            "task": setting.task,
            "n_original": setting.paper_denominator,
            "n_selected": len(selected),
        }
        for metric, column in _METRIC_COLUMNS.items():
            values = tuple(
                float(row["metrics"][metric]["value"])
                for row in selected
                if row["metrics"][metric]["valid"] is True
            )
            summary = _summary_or_null(
                values,
                protocol=protocol,
                label=f"{setting.model}|{setting.task}|{metric}",
            )
            setting_row[f"{column}_n"] = len(values)
            setting_row[f"{column}_median"] = summary["estimate"]
            setting_row[f"{column}_ci_lower"] = summary["lower"]
            setting_row[f"{column}_ci_upper"] = summary["upper"]
        setting_rows.append(setting_row)

        for token_index in range(protocol.teacher_forced_tokens):
            untreated = tuple(float(row["untreated_kl"][token_index]) for row in selected)
            patched = tuple(float(row["patched_kl"][token_index]) for row in selected)
            reduction = tuple(
                left - right for left, right in zip(untreated, patched, strict=True)
            )
            reduction_summary = _summary_or_null(
                reduction,
                protocol=protocol,
                label=f"{setting.model}|{setting.task}|token-{token_index + 1}|raw",
            )
            normalized = tuple(
                1.0 - right / left
                for left, right in zip(untreated, patched, strict=True)
                if left > protocol.denominator_epsilon
            )
            normalized_summary = _summary_or_null(
                normalized,
                protocol=protocol,
                label=f"{setting.model}|{setting.task}|token-{token_index + 1}|normalized",
            )
            trajectory_rows.append(
                {
                    "model": setting.model,
                    "task": setting.task,
                    "token_index": token_index + 1,
                    "n": len(selected),
                    "median_untreated_kl": statistics.median(untreated),
                    "median_patched_kl": statistics.median(patched),
                    "median_raw_kl_reduction": reduction_summary["estimate"],
                    "raw_reduction_ci_lower": reduction_summary["lower"],
                    "raw_reduction_ci_upper": reduction_summary["upper"],
                    "normalized_n": len(normalized),
                    "median_normalized_restoration": normalized_summary["estimate"],
                    "normalized_ci_lower": normalized_summary["lower"],
                    "normalized_ci_upper": normalized_summary["upper"],
                }
            )
    return setting_rows, trajectory_rows


def _trajectory_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 960, 520
    left, right, top, bottom = 80.0, 710.0, 55.0, 455.0
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    values = [float(row["median_raw_kl_reduction"]) for row in rows]
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.1, 1e-9)
        y_min, y_max = y_min - padding, y_max + padding
    else:
        padding = 0.08 * (y_max - y_min)
        y_min, y_max = y_min - padding, y_max + padding

    def x(token: int) -> float:
        return left + (token - 1) / 15.0 * (right - left)

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="27" text-anchor="middle" font-family="sans-serif" '
        'font-size="18">Token-wise raw KL reduction</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    if y_min <= 0.0 <= y_max:
        zero = y(0.0)
        lines.append(
            f'<line x1="{left}" y1="{zero:.2f}" x2="{right}" y2="{zero:.2f}" '
            'stroke="#999" stroke-dasharray="4 4"/>'
        )
    for token in (1, 4, 8, 12, 16):
        position = x(token)
        lines.extend(
            (
                f'<line x1="{position:.2f}" y1="{bottom}" x2="{position:.2f}" '
                f'y2="{bottom + 6}" stroke="#333"/>',
                f'<text x="{position:.2f}" y="{bottom + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">{token}</text>',
            )
        )
    for tick in range(5):
        value = y_min + tick / 4.0 * (y_max - y_min)
        position = y(value)
        lines.extend(
            (
                f'<line x1="{left - 6}" y1="{position:.2f}" x2="{left}" '
                f'y2="{position:.2f}" stroke="#333"/>',
                f'<text x="{left - 10}" y="{position + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11">{value:.3g}</text>',
            )
        )
    lines.append(
        f'<text x="{(left + right) / 2:.2f}" y="{height - 18}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">Teacher-forced clean token position</text>'
    )
    settings: list[tuple[str, str]] = []
    for row in rows:
        setting = str(row["model"]), str(row["task"])
        if setting not in settings:
            settings.append(setting)
    for index, setting in enumerate(settings):
        setting_rows = [row for row in rows if (row["model"], row["task"]) == setting]
        points = " ".join(
            f'{x(int(row["token_index"])):.2f},{y(float(row["median_raw_kl_reduction"])):.2f}'
            for row in setting_rows
        )
        color = colors[index % len(colors)]
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
        label = escape(f"{setting[0].rsplit('/', 1)[-1]} / {setting[1]}")
        legend_y = top + 19 * index
        lines.extend(
            (
                f'<line x1="735" y1="{legend_y}" x2="755" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="2"/>',
                f'<text x="762" y="{legend_y + 4}" font-family="sans-serif" '
                f'font-size="11">{label}</text>',
            )
        )
    lines.append("</svg>\n")
    return "\n".join(lines)


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_public_record(
    row: Mapping[str, object],
    *,
    protocol: MultiTokenKLReadoutProtocol,
    manifest_sha256: str,
    runtime_fingerprints: Mapping[tuple[str, str], str],
) -> None:
    expected_fields = {
        "schema_version",
        "paper_sha256",
        "readout_protocol_sha256",
        "pair_manifest_sha256",
        "source_record_sha256",
        "runtime_fingerprint",
        "pair_id",
        "sample_id",
        "model",
        "task",
        "target_rule",
        "target_token_ids",
        "target_token_text",
        "untreated_kl",
        "patched_kl",
        "raw_kl_reduction",
        "metrics",
    }
    if set(row) != expected_fields:
        raise ValueError("completed multi-token record fields differ")
    if (
        row.get("schema_version") != _RECORD_SCHEMA
        or row.get("paper_sha256") != PAPER_SHA256
        or row.get("readout_protocol_sha256") != protocol.config_sha256
        or row.get("pair_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("completed multi-token record provenance differs")
    _digest(row.get("pair_id"), field="record pair_id")
    _digest(row.get("source_record_sha256"), field="record source_record_sha256")
    setting = row.get("model"), row.get("task")
    if setting not in runtime_fingerprints:
        raise ValueError("completed multi-token record setting differs")
    if row.get("runtime_fingerprint") != runtime_fingerprints[setting]:
        raise ValueError("completed multi-token record runtime fingerprint differs")
    for field in ("sample_id", "model", "task", "target_rule"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"completed multi-token record {field} differs")
    scan = _validate_scan(
        _scan_from_payload(
            {
                "target_token_ids": row.get("target_token_ids"),
                "target_token_text": row.get("target_token_text"),
                "untreated_kl": row.get("untreated_kl"),
                "patched_kl": row.get("patched_kl"),
            },
            field="completed record scan",
        ),
        protocol=protocol,
    )
    expected_reduction = [
        untreated - patched
        for untreated, patched in zip(scan.untreated_kl, scan.patched_kl, strict=True)
    ]
    if row.get("raw_kl_reduction") != expected_reduction:
        raise ValueError("completed multi-token raw KL reduction differs")
    metrics = _mapping(row.get("metrics"), field="completed record metrics")
    if set(metrics) != set(_METRIC_COLUMNS):
        raise ValueError("completed multi-token metric inventory differs")
    expected_metrics = {
        name: _metric_payload(
            scan.untreated_kl,
            scan.patched_kl,
            token_range=token_range,
            epsilon=protocol.denominator_epsilon,
        )
        for name, token_range in _METRIC_RANGES.items()
    }
    first, primary = expected_metrics["R_1"], expected_metrics["R_2:16"]
    expected_metrics["R_1_minus_R_2:16"] = (
        {
            "valid": True,
            "value": float(first["value"]) - float(primary["value"]),
            "invalid_reason": None,
        }
        if first["valid"] is True and primary["valid"] is True
        else {
            "valid": False,
            "value": None,
            "invalid_reason": "first-or-primary-invalid",
        }
    )
    if dict(metrics) != expected_metrics:
        raise ValueError("completed multi-token metrics differ from raw trajectories")


def _result_from_run(
    output_dir: Path,
    run: Mapping[str, object],
    protocol: MultiTokenKLReadoutProtocol,
) -> MultiTokenKLReadoutResult:
    outputs = _mapping(run.get("outputs"), field="completed run outputs")
    if set(outputs) != set(_PUBLIC_OUTPUTS):
        raise ValueError("completed multi-token output inventory differs")
    for name in _PUBLIC_OUTPUTS:
        path = output_dir / name
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"completed multi-token output SHA-256 differs: {path}")

    record_rows = tuple(
        row for _number, _line, row in iter_jsonl_objects(output_dir / _PUBLIC_OUTPUTS[0])
    )
    manifest_sha256 = _digest(
        run.get("pair_manifest_sha256"), field="completed run pair_manifest_sha256"
    )
    runtime_payloads = _mapping(
        run.get("runtime_by_setting"), field="completed run runtime_by_setting"
    )
    runtime_fingerprints: dict[tuple[str, str], str] = {}
    for setting in REBUTTAL_SETTINGS:
        payload = _mapping(
            runtime_payloads.get(setting.slug),
            field=f"completed runtime {setting.slug}",
        )
        runtime_fingerprints[setting.key] = _canonical_sha256(payload)
    for row in record_rows:
        _validate_public_record(
            row,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
            runtime_fingerprints=runtime_fingerprints,
        )
    pair_ids = [str(row["pair_id"]) for row in record_rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("completed multi-token pair IDs are duplicated")

    def csv_count(name: str) -> int:
        with (output_dir / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = tuple(reader)
        if reader.fieldnames is None:
            raise ValueError(f"completed multi-token CSV has no header: {name}")
        return len(rows)

    actual_outputs = {
        _PUBLIC_OUTPUTS[0]: len(record_rows),
        _PUBLIC_OUTPUTS[1]: csv_count(_PUBLIC_OUTPUTS[1]),
        _PUBLIC_OUTPUTS[2]: csv_count(_PUBLIC_OUTPUTS[2]),
        _PUBLIC_OUTPUTS[3]: 1,
        _PUBLIC_OUTPUTS[4]: 1,
    }
    load_json_object(output_dir / _PUBLIC_OUTPUTS[4])
    if not (output_dir / _PUBLIC_OUTPUTS[3]).read_text(encoding="utf-8").startswith("<svg"):
        raise ValueError("completed multi-token SVG contract differs")
    for name, count in actual_outputs.items():
        if _mapping(outputs[name], field=f"completed output {name}").get("records") != count:
            raise ValueError(f"completed multi-token output count differs: {name}")
    settings = {(str(row["model"]), str(row["task"])) for row in record_rows}
    primary_valid = sum(
        int(row["metrics"]["R_2:16"]["valid"] is True) for row in record_rows
    )
    expected_counts = {
        "pairs": len(record_rows),
        "primary_valid_pairs": primary_valid,
        "records": len(record_rows),
        "setting_metrics": len(settings),
        "settings": len(settings),
        "trajectory_rows": 16 * len(settings),
    }
    counts = _mapping(run.get("counts"), field="completed run counts")
    if dict(counts) != expected_counts:
        raise ValueError("completed multi-token run counts differ from its outputs")
    if actual_outputs[_PUBLIC_OUTPUTS[1]] != len(settings) or actual_outputs[
        _PUBLIC_OUTPUTS[2]
    ] != 16 * len(settings):
        raise ValueError("completed multi-token analysis grid is incomplete")
    return MultiTokenKLReadoutResult(
        records_path=output_dir / _PUBLIC_OUTPUTS[0],
        setting_metrics_path=output_dir / _PUBLIC_OUTPUTS[1],
        trajectory_path=output_dir / _PUBLIC_OUTPUTS[2],
        trajectory_plot_path=output_dir / _PUBLIC_OUTPUTS[3],
        summary_path=output_dir / _PUBLIC_OUTPUTS[4],
        run_path=output_dir / "run.json",
        pairs=int(counts["pairs"]),
        primary_valid_pairs=int(counts["primary_valid_pairs"]),
        settings=int(counts["settings"]),
        trajectory_rows=int(counts["trajectory_rows"]),
    )


def _completed_resume(
    config: MultiTokenKLReadoutConfig,
    protocol: MultiTokenKLReadoutProtocol,
) -> MultiTokenKLReadoutResult | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "multitoken-kl-readout"
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("pair_manifest_sha256") != sha256_file(config.manifest_path)
    ):
        raise ValueError("completed multi-token resume contract differs")
    return _result_from_run(config.output_dir, run, protocol)


def run_multitoken_kl_readout(
    config: MultiTokenKLReadoutConfig,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> MultiTokenKLReadoutResult:
    """Run the frozen 16-token readout and compile setting-level estimates."""

    if not isinstance(config, MultiTokenKLReadoutConfig):
        raise TypeError("config must be MultiTokenKLReadoutConfig")
    protocol = load_multitoken_kl_readout_protocol(config.protocol_path)
    if config.teacher_forced_tokens != protocol.teacher_forced_tokens:
        raise ValueError("--teacher-forced-tokens differs from the frozen config")
    if config.primary_token_range != protocol.primary_token_range:
        raise ValueError("--primary-token-range differs from the frozen config")
    completed = _completed_resume(config, protocol)
    if completed is not None:
        return completed
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.resume:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        if not (output_dir / "run.json").is_file():
            raise ValueError("non-empty resume directory is missing its bound run.json")
    if not config.manifest_path.is_file():
        raise ValueError(f"rebuttal manifest is not a file: {config.manifest_path}")
    manifest_sha256 = sha256_file(config.manifest_path)
    records = load_rebuttal_pair_manifest(config.manifest_path)
    selected = _select_records(records, limit_per_setting=config.limit_per_setting)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    run_path = output_dir / "run.json"
    started_at = _now()
    if config.resume and run_path.is_file():
        previous = load_json_object(run_path)
        if (
            previous.get("schema_version") != _RUN_SCHEMA
            or previous.get("paper_sha256") != PAPER_SHA256
            or previous.get("operation") != "multitoken-kl-readout"
            or previous.get("status") not in {"running", "failed"}
            or previous.get("arguments") != config.public_arguments()
            or previous.get("protocol") != protocol.as_dict()
            or previous.get("protocol_sha256") != protocol.config_sha256
            or previous.get("manifest_protocol_sha256")
            != REBUTTAL_MANIFEST_PROTOCOL.sha256()
            or previous.get("pair_manifest_sha256") != manifest_sha256
        ):
            raise ValueError("multi-token resume contract differs")
        if isinstance(previous.get("started_at"), str):
            started_at = str(previous["started_at"])
    confirmatory = (
        config.limit_per_setting is None
        and len(selected) == REBUTTAL_MANIFEST_PROTOCOL.restoration_pairs
    )
    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "multitoken-kl-readout",
        "status": "running",
        "confirmatory": confirmatory,
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "pair_manifest_sha256": manifest_sha256,
        "started_at": started_at,
        "updated_at": _now(),
        "runtime_by_setting": {},
        "counts": {"pairs": len(selected), "settings": len(REBUTTAL_SETTINGS)},
        "failures": [],
    }
    _write_json_atomic(run_path, base_run)
    factory = runtime_factory or _runtime_factory()
    runtime_fingerprints: dict[tuple[str, str], str] = {}
    runtime_payloads: dict[str, object] = {}
    try:
        for setting in REBUTTAL_SETTINGS:
            setting_records = [
                record
                for record in selected
                if (record["model"], record["task"]) == setting.key
            ]
            revisions = {
                str(_mapping(record["source"], field="record source")["model_revision"])
                for record in setting_records
            }
            if len(revisions) != 1:
                raise ValueError(f"multi-token model revisions differ for {setting.slug}")
            revision = next(iter(revisions))
            runtime = factory(
                model=setting.model,
                task=setting.task,
                revision=revision,
                gpu_id=config.gpu_id,
            )
            provenance = _runtime_provenance(
                runtime,
                model=setting.model,
                task=setting.task,
                revision=revision,
                gpu_id=config.gpu_id,
                protocol=protocol,
            )
            fingerprint = _canonical_sha256(provenance)
            runtime_fingerprints[setting.key] = fingerprint
            runtime_payloads[setting.slug] = provenance
            for record in setting_records:
                checkpoint = _checkpoint_path(
                    checkpoints_dir,
                    record=record,
                    runtime_fingerprint=fingerprint,
                    protocol_sha256=protocol.config_sha256,
                    manifest_sha256=manifest_sha256,
                )
                if checkpoint.is_file():
                    _load_checkpoint(
                        checkpoint,
                        record=record,
                        runtime_fingerprint=fingerprint,
                        protocol=protocol,
                        manifest_sha256=manifest_sha256,
                    )
                    continue
                scan = _validate_scan(
                    runtime.scan_pair(
                        record,
                        teacher_forced_tokens=protocol.teacher_forced_tokens,
                        layer_window=protocol.window,
                    ),
                    protocol=protocol,
                )
                _write_checkpoint(
                    checkpoint,
                    record=record,
                    scan=scan,
                    runtime_fingerprint=fingerprint,
                    protocol=protocol,
                    manifest_sha256=manifest_sha256,
                )
            del runtime

        output_rows = _compile_records(
            selected=selected,
            checkpoints_dir=checkpoints_dir,
            runtime_fingerprints=runtime_fingerprints,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
        )
        setting_rows, trajectory_rows = _analysis_rows(output_rows, protocol=protocol)
        summary = {
            "schema_version": _SUMMARY_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "protocol_sha256": protocol.config_sha256,
            "primary_metric": {
                "name": "R_2:16",
                "token_range": [2, 16],
                "estimator": protocol.setting_estimator,
                "denominator_epsilon": protocol.denominator_epsilon,
                "first_token_excluded": True,
            },
            "counts": {
                "pairs": len(output_rows),
                "primary_valid_pairs": sum(
                    int(row["metrics"]["R_2:16"]["valid"] is True) for row in output_rows
                ),
                "settings": len(setting_rows),
            },
            "settings": setting_rows,
            "positive_primary_setting_estimates": sum(
                int(
                    row["r_2_16_median"] is not None
                    and float(row["r_2_16_median"]) > 0.0
                )
                for row in setting_rows
            ),
        }
        paths = {name: output_dir / name for name in _PUBLIC_OUTPUTS}
        _write_jsonl_atomic(paths[_PUBLIC_OUTPUTS[0]], output_rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[1]], setting_rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[2]], trajectory_rows)
        _write_text_atomic(paths[_PUBLIC_OUTPUTS[3]], _trajectory_svg(trajectory_rows))
        _write_json_atomic(paths[_PUBLIC_OUTPUTS[4]], summary)
        counts = {
            "pairs": len(output_rows),
            "primary_valid_pairs": int(summary["counts"]["primary_valid_pairs"]),
            "records": len(output_rows),
            "setting_metrics": len(setting_rows),
            "settings": len(setting_rows),
            "trajectory_rows": len(trajectory_rows),
        }
        row_counts = {
            _PUBLIC_OUTPUTS[0]: len(output_rows),
            _PUBLIC_OUTPUTS[1]: len(setting_rows),
            _PUBLIC_OUTPUTS[2]: len(trajectory_rows),
            _PUBLIC_OUTPUTS[3]: 1,
            _PUBLIC_OUTPUTS[4]: 1,
        }
        outputs = {
            name: {"sha256": sha256_file(path), "records": row_counts[name]}
            for name, path in paths.items()
        }
        completed_run = {
            **base_run,
            "status": "completed",
            "runtime_by_setting": runtime_payloads,
            "counts": counts,
            "outputs": outputs,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "gpu_required": True,
            },
            "updated_at": _now(),
        }
        _write_json_atomic(run_path, completed_run)
        return _result_from_run(output_dir, completed_run, protocol)
    except Exception as exc:
        for name in _PUBLIC_OUTPUTS:
            (output_dir / name).unlink(missing_ok=True)
            (output_dir / f".{name}.tmp").unlink(missing_ok=True)
        failed = {
            **base_run,
            "status": "failed",
            "runtime_by_setting": runtime_payloads,
            "failures": [{"type": type(exc).__name__, "message": str(exc)}],
            "updated_at": _now(),
        }
        _write_json_atomic(run_path, failed)
        raise MultiTokenKLReadoutRunError(
            "multi-token KL readout failed; verified checkpoints were retained"
        ) from exc


__all__ = [
    "MultiTokenKLReadoutConfig",
    "MultiTokenKLReadoutResult",
    "MultiTokenKLReadoutRunError",
    "MultiTokenKLRuntime",
    "MultiTokenKLScan",
    "RuntimeFactory",
    "run_multitoken_kl_readout",
]
