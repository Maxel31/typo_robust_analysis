"""Hash-bound and resumable runner for diagnostic all-layer localization."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typo_robust_training.data.config import strict_loads
from typo_robust_training.localization.config import (
    LayerSelectionProtocol,
    load_layer_selection_config,
)
from typo_robust_training.localization.records import LayerScan
from typo_robust_training.localization.scoring import compute_layer_selection


_PAIR_FIELDS = {
    "schema_version",
    "kind",
    "record_id",
    "source",
    "source_revision",
    "source_split",
    "source_id",
    "group_id",
    "split",
    "clean_text",
    "typo_text",
    "task",
    "answer",
    "metadata",
    "operation",
    "operations",
    "edit_count",
    "generator_seed",
    "generator_variant",
    "edits",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _write_atomic(path, _canonical_bytes(value))


def _load_json(path: Path) -> Mapping[str, object]:
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


@dataclass(frozen=True, slots=True)
class LayerSelectionRunConfig:
    config_path: Path
    diagnostic_manifest_path: Path
    tasks: tuple[str, ...]
    gpu_id: str
    output_dir: Path
    resume: bool = False


class LayerSelectionRuntime(Protocol):
    def scan_pair(self, record: dict[str, object]) -> LayerScan: ...

    def provenance(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class LayerSelectionRunResult:
    records: int
    selected_window: tuple[int, int]
    scans_path: Path
    selection_path: Path
    run_path: Path


def _validate_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _load_manifest(
    path: Path, *, tasks: tuple[str, ...]
) -> tuple[tuple[dict[str, object], ...], str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"diagnostic manifest is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("diagnostic manifest is not UTF-8") from exc
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        payload = strict_loads(line, context=f"{resolved}:{line_number}")
        if not isinstance(payload, Mapping) or set(payload) != _PAIR_FIELDS:
            raise ValueError(f"diagnostic manifest fields differ at line {line_number}")
        if payload.get("schema_version") != "robustness-fixed-typo-pair/v1":
            raise ValueError(f"diagnostic schema differs at line {line_number}")
        if payload.get("kind") != "synthetic":
            raise ValueError(f"diagnostic kind differs at line {line_number}")
        if payload.get("split") != "diagnostic":
            raise ValueError(f"diagnostic split differs at line {line_number}")
        record_id = _validate_text(payload.get("record_id"), field="record_id")
        task = _validate_text(payload.get("task"), field=f"{record_id}.task")
        if task not in tasks:
            raise ValueError(f"diagnostic task {task!r} is outside requested tasks")
        _validate_text(payload.get("answer"), field=f"{record_id}.answer")
        if not isinstance(payload.get("metadata"), Mapping):
            raise ValueError(f"{record_id}.metadata must be an object")
        if not isinstance(payload.get("edits"), list) or not payload["edits"]:
            raise ValueError(f"{record_id}.edits must be non-empty")
        records.append(dict(payload))
    if not records:
        raise ValueError("diagnostic manifest is empty")
    ids = [str(record["record_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("diagnostic manifest record IDs are duplicated")
    observed = {str(record["task"]) for record in records}
    if observed != set(tasks):
        raise ValueError("diagnostic manifest does not contain every requested task")
    records.sort(key=lambda record: str(record["record_id"]))
    return tuple(records), _sha256_bytes(raw)


def _checkpoint_path(work_dir: Path, record_id: str) -> Path:
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return work_dir / f"{digest}.json"


def _record_sha(record: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_bytes(record))


def _checkpoint_common(
    record: Mapping[str, object],
    *,
    protocol: LayerSelectionProtocol,
    manifest_sha256: str,
    runtime_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "robustness-layer-selection-checkpoint/v1",
        "record_id": record["record_id"],
        "record_sha256": _record_sha(record),
        "config_sha256": protocol.config_sha256,
        "diagnostic_manifest_sha256": manifest_sha256,
        "runtime_sha256": runtime_sha256,
    }


def _load_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    protocol: LayerSelectionProtocol,
    manifest_sha256: str,
    runtime_sha256: str,
) -> LayerScan:
    payload = _load_json(path)
    expected = _checkpoint_common(
        record,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
        runtime_sha256=runtime_sha256,
    )
    if set(payload) != set(expected) | {"scan"} or any(
        payload.get(field) != value for field, value in expected.items()
    ):
        raise ValueError(f"layer selection checkpoint binding differs: {path}")
    scan = LayerScan.from_dict(payload["scan"])
    if scan.record_id != record["record_id"] or scan.task != record["task"]:
        raise ValueError(f"layer selection checkpoint identity differs: {path}")
    return scan


def _write_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    scan: LayerScan,
    protocol: LayerSelectionProtocol,
    manifest_sha256: str,
    runtime_sha256: str,
) -> None:
    _write_json(
        path,
        {
            **_checkpoint_common(
                record,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
                runtime_sha256=runtime_sha256,
            ),
            "scan": scan.as_dict(),
        },
    )


def _base_run_payload(
    config: LayerSelectionRunConfig,
    *,
    protocol: LayerSelectionProtocol,
    manifest_sha256: str,
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "robustness-layer-selection-run/v1",
        "operation": "select-distillation-layers",
        "config_path": str(Path(config.config_path).resolve()),
        "config_sha256": protocol.config_sha256,
        "diagnostic_manifest_path": str(Path(config.diagnostic_manifest_path).resolve()),
        "diagnostic_manifest_sha256": manifest_sha256,
        "tasks": list(config.tasks),
        "gpu_id": config.gpu_id,
        "resume": config.resume,
        "python": platform.python_version(),
        "runtime": dict(runtime_provenance),
    }


def _existing_result(
    output_dir: Path,
    *,
    protocol: LayerSelectionProtocol,
    manifest_sha256: str,
) -> LayerSelectionRunResult | None:
    run_path = output_dir / "run.json"
    selection_path = output_dir / "layer_selection.json"
    scans_path = output_dir / "layer_scans.jsonl"
    if not all(path.is_file() for path in (run_path, selection_path, scans_path)):
        return None
    run = _load_json(run_path)
    selection = _load_json(selection_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("config_sha256") != protocol.config_sha256
        or run.get("diagnostic_manifest_sha256") != manifest_sha256
        or run.get("outputs", {}).get("layer_selection.json", {}).get("sha256")
        != _sha256_file(selection_path)
        or run.get("outputs", {}).get("layer_scans.jsonl", {}).get("sha256")
        != _sha256_file(scans_path)
    ):
        raise ValueError("completed layer-selection artifacts fail hash validation")
    selected = selection.get("selected_window")
    if not isinstance(selected, Mapping):
        raise ValueError("completed layer selection has no selected window")
    return LayerSelectionRunResult(
        records=int(selection["records"]),
        selected_window=(int(selected["start"]), int(selected["stop"])),
        scans_path=scans_path,
        selection_path=selection_path,
        run_path=run_path,
    )


def run_select_distillation_layers(
    config: LayerSelectionRunConfig,
    *,
    runtime: LayerSelectionRuntime | None = None,
) -> LayerSelectionRunResult:
    """Run or resume one complete diagnostic all-layer scan."""

    if not isinstance(config, LayerSelectionRunConfig):
        raise TypeError("config must be LayerSelectionRunConfig")
    protocol = load_layer_selection_config(config.config_path)
    if config.tasks != protocol.tasks:
        raise ValueError("--tasks must exactly match the frozen config task order")
    if not isinstance(config.gpu_id, str) or not config.gpu_id or "," in config.gpu_id:
        raise ValueError("--gpu-id must name exactly one physical GPU")
    records, manifest_sha256 = _load_manifest(
        config.diagnostic_manifest_path,
        tasks=config.tasks,
    )
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume:
        raise FileExistsError(f"layer selection output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.resume:
        existing = _existing_result(
            output_dir,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
        )
        if existing is not None:
            return existing
    if runtime is None:
        from typo_robust_training.localization.runtime import (
            HuggingFaceLayerSelectionRuntime,
        )

        runtime = HuggingFaceLayerSelectionRuntime(protocol=protocol, gpu_id=config.gpu_id)
    runtime_provenance = dict(runtime.provenance())
    try:
        runtime_bytes = _canonical_bytes(runtime_provenance)
    except (TypeError, ValueError) as exc:
        raise ValueError("layer selection runtime provenance is not canonical JSON") from exc
    runtime_sha256 = _sha256_bytes(runtime_bytes)
    run_path = output_dir / "run.json"
    run_base = _base_run_payload(
        config,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
        runtime_provenance=runtime_provenance,
    )
    _write_json(run_path, {**run_base, "status": "running", "started_at": _now()})
    work_dir = output_dir / ".select-distillation-layers-work"
    work_dir.mkdir(exist_ok=True)
    scans: list[LayerScan] = []
    try:
        for record in records:
            checkpoint = _checkpoint_path(work_dir, str(record["record_id"]))
            if config.resume and checkpoint.is_file():
                scan = _load_checkpoint(
                    checkpoint,
                    record=record,
                    protocol=protocol,
                    manifest_sha256=manifest_sha256,
                    runtime_sha256=runtime_sha256,
                )
            else:
                scan = runtime.scan_pair(record)
                if not isinstance(scan, LayerScan):
                    raise TypeError("layer selection runtime must return LayerScan")
                if scan.record_id != record["record_id"] or scan.task != record["task"]:
                    raise ValueError("layer selection runtime returned a different record identity")
                _write_checkpoint(
                    checkpoint,
                    record=record,
                    scan=scan,
                    protocol=protocol,
                    manifest_sha256=manifest_sha256,
                    runtime_sha256=runtime_sha256,
                )
            scans.append(scan)
        selection = compute_layer_selection(scans, protocol=protocol)
        scans_path = output_dir / "layer_scans.jsonl"
        _write_atomic(scans_path, b"".join(_canonical_bytes(scan.as_dict()) for scan in scans))
        selection_path = output_dir / "layer_selection.json"
        selection_payload = {
            **selection.as_dict(),
            "operation": "select-distillation-layers",
            "records": len(scans),
            "model": protocol.model,
            "model_revision": protocol.model_revision,
            "config_sha256": protocol.config_sha256,
            "diagnostic_manifest_sha256": manifest_sha256,
            "layer_scans_sha256": _sha256_file(scans_path),
            "runtime": runtime_provenance,
        }
        _write_json(selection_path, selection_payload)
        outputs = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (scans_path, selection_path)
        }
        _write_json(
            run_path,
            {
                **run_base,
                "status": "completed",
                "completed_at": _now(),
                "records": len(scans),
                "selected_window": list(selection.selected_window),
                "outputs": outputs,
            },
        )
    except BaseException as exc:
        _write_json(
            run_path,
            {
                **run_base,
                "status": "failed",
                "failed_at": _now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "completed_checkpoints": len(scans),
            },
        )
        raise
    return LayerSelectionRunResult(
        records=len(scans),
        selected_window=selection.selected_window,
        scans_path=scans_path,
        selection_path=selection_path,
        run_path=run_path,
    )


__all__ = [
    "LayerSelectionRunConfig",
    "LayerSelectionRunResult",
    "LayerSelectionRuntime",
    "run_select_distillation_layers",
]
