"""Hash-bound runners for confirmatory generic joint-window patching."""

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
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.localization.confirmatory_config import (
    ConfirmatoryLocalizationProtocol,
    load_confirmatory_localization_config,
)
from typo_robust_training.localization.confirmatory_records import JointWindowScan
from typo_robust_training.localization.confirmatory_scoring import (
    select_joint_patch_window,
    validate_joint_patch_window,
)


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


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_object(path: Path) -> Mapping[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"JSON artifact is not a file: {resolved}")
    value = strict_loads(resolved.read_text(encoding="utf-8"), context=str(resolved))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {resolved}")
    return value


def _load_manifest(
    path: Path,
    *,
    role: str,
    expected_records: int,
    protocol: ConfirmatoryLocalizationProtocol,
) -> tuple[tuple[dict[str, object], ...], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{role} manifest is not a file: {resolved}")
    raw = resolved.read_bytes()
    rows: list[dict[str, object]] = []
    expected_split = f"localization-{role}"
    for line_number, line in read_lf_jsonl_lines(resolved, context=f"confirmatory {role} manifest"):
        value = strict_loads(line, context=f"{resolved}:{line_number}")
        if not isinstance(value, Mapping) or set(value) != _PAIR_FIELDS:
            raise ValueError(f"{role} manifest fields differ at line {line_number}")
        if (
            value.get("schema_version") != "robustness-fixed-typo-pair/v1"
            or value.get("kind") != "synthetic"
            or value.get("source") != protocol.selection_source
            or value.get("source_revision") != protocol.source_revision
            or value.get("split") != expected_split
            or value.get("task") is not None
            or value.get("answer") is not None
            or value.get("edit_count") != 1
            or value.get("operation") not in protocol.typo_operations
            or value.get("operations") != [value.get("operation")]
            or not isinstance(value.get("edits"), list)
            or len(value["edits"]) != 1
        ):
            raise ValueError(f"{role} manifest protocol differs at line {line_number}")
        for field in ("record_id", "source_id", "group_id", "clean_text", "typo_text"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f"{role} manifest {field} differs at line {line_number}")
        rows.append(dict(value))
    if len(rows) != expected_records:
        raise ValueError(
            f"{role} manifest count differs: expected={expected_records}, observed={len(rows)}"
        )
    for field in ("record_id", "source_id", "group_id"):
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"{role} manifest {field} values are duplicated")
    return tuple(rows), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class JointWindowSelectionRunConfig:
    config_path: Path
    selection_manifest_path: Path
    gpu_id: str
    output_dir: Path
    resume: bool = False


@dataclass(frozen=True, slots=True)
class JointWindowValidationRunConfig:
    config_path: Path
    validation_manifest_path: Path
    window_selection_path: Path
    gpu_id: str
    output_dir: Path
    resume: bool = False


class ConfirmatoryJointWindowRuntime(Protocol):
    def scan_selection_pair(self, record: dict[str, object]) -> JointWindowScan: ...

    def scan_validation_pair(
        self, record: dict[str, object], selected_window: tuple[int, int]
    ) -> JointWindowScan: ...

    def provenance(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class JointWindowSelectionRunResult:
    records: int
    selected_window: tuple[int, int]
    scans_path: Path
    selection_path: Path
    run_path: Path


@dataclass(frozen=True, slots=True)
class JointWindowValidationRunResult:
    records: int
    passed: bool
    scans_path: Path
    validation_path: Path
    run_path: Path


def _validate_gpu_id(gpu_id: str) -> None:
    if not isinstance(gpu_id, str) or not gpu_id or "," in gpu_id or not gpu_id.isdecimal():
        raise ValueError("--gpu-id must name exactly one physical GPU")


def _runtime(
    protocol: ConfirmatoryLocalizationProtocol,
    gpu_id: str,
    runtime: ConfirmatoryJointWindowRuntime | None,
) -> ConfirmatoryJointWindowRuntime:
    if runtime is not None:
        return runtime
    from typo_robust_training.localization.confirmatory_runtime import (
        HuggingFaceConfirmatoryJointWindowRuntime,
    )

    return HuggingFaceConfirmatoryJointWindowRuntime(protocol=protocol, gpu_id=gpu_id)


def _checkpoint_path(work_dir: Path, record_id: str) -> Path:
    return work_dir / f"{hashlib.sha256(record_id.encode()).hexdigest()}.json"


def _checkpoint_common(
    record: Mapping[str, object],
    *,
    role: str,
    config_sha256: str,
    manifest_sha256: str,
    runtime_sha256: str,
    window_selection_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "robustness-joint-window-checkpoint/v1",
        "role": role,
        "record_id": record["record_id"],
        "record_sha256": _sha256_value(record),
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "runtime_sha256": runtime_sha256,
        "window_selection_sha256": window_selection_sha256,
    }


def _load_or_scan(
    *,
    records: tuple[dict[str, object], ...],
    role: str,
    protocol: ConfirmatoryLocalizationProtocol,
    manifest_sha256: str,
    runtime: ConfirmatoryJointWindowRuntime,
    runtime_sha256: str,
    work_dir: Path,
    resume: bool,
    selected_window: tuple[int, int] | None,
    window_selection_sha256: str | None,
) -> list[JointWindowScan]:
    scans: list[JointWindowScan] = []
    for record in records:
        checkpoint = _checkpoint_path(work_dir, str(record["record_id"]))
        expected = _checkpoint_common(
            record,
            role=role,
            config_sha256=protocol.config_sha256,
            manifest_sha256=manifest_sha256,
            runtime_sha256=runtime_sha256,
            window_selection_sha256=window_selection_sha256,
        )
        if resume and checkpoint.is_file():
            payload = _load_object(checkpoint)
            if set(payload) != set(expected) | {"scan"} or any(
                payload.get(field) != value for field, value in expected.items()
            ):
                raise ValueError(f"joint-window checkpoint binding differs: {checkpoint}")
            scan = JointWindowScan.from_dict(payload["scan"])
        else:
            scan = (
                runtime.scan_selection_pair(record)
                if role == "selection"
                else runtime.scan_validation_pair(record, selected_window)  # type: ignore[arg-type]
            )
            if not isinstance(scan, JointWindowScan) or scan.record_id != record["record_id"]:
                raise ValueError("joint-window runtime returned a different record identity")
            _write_json(checkpoint, {**expected, "scan": scan.as_dict()})
        if (
            scan.role != role
            or scan.decoder_layers != protocol.decoder_layers
            or scan.window_width != protocol.window_width
        ):
            raise ValueError("joint-window runtime dimensions or role differ")
        scans.append(scan)
    return scans


def _window_from_evidence(
    path: Path, protocol: ConfirmatoryLocalizationProtocol
) -> tuple[tuple[int, int], str]:
    payload = _load_object(path)
    if (
        payload.get("schema_version") != "robustness-joint-window-selection/v1"
        or payload.get("operation") != "select-generic-joint-patch-window"
        or payload.get("config_sha256") != protocol.config_sha256
    ):
        raise ValueError("window selection config binding differs")
    if (
        payload.get("model") != protocol.model
        or payload.get("model_revision") != protocol.model_revision
    ):
        raise ValueError("window selection model binding differs")
    selected = payload.get("selected_window")
    if not isinstance(selected, Mapping):
        raise ValueError("window selection coordinate is missing")
    start, stop = selected.get("start"), selected.get("stop")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or stop - start != protocol.window_width
        or not 0 <= start < stop <= protocol.decoder_layers
    ):
        raise ValueError("window selection coordinate differs")
    return (start, stop), _sha256_file(path.resolve())


def _existing_selection(
    output_dir: Path,
    *,
    protocol: ConfirmatoryLocalizationProtocol,
    manifest_sha256: str,
) -> JointWindowSelectionRunResult | None:
    run_path = output_dir / "run.json"
    scans_path = output_dir / "joint_window_scans.jsonl"
    selection_path = output_dir / "window_selection.json"
    if not all(path.is_file() for path in (run_path, scans_path, selection_path)):
        return None
    run = _load_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("config_sha256") != protocol.config_sha256
        or run.get("selection_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("completed joint-window selection binding differs")
    for path in (scans_path, selection_path):
        if not isinstance(run.get("outputs"), Mapping) or run["outputs"].get(path.name, {}).get(
            "sha256"
        ) != _sha256_file(path):
            raise ValueError("completed joint-window selection output hash differs")
    payload = _load_object(selection_path)
    selected = payload.get("selected_window")
    if not isinstance(selected, Mapping):
        raise ValueError("completed selection coordinate is missing")
    window = (int(selected["start"]), int(selected["stop"]))
    return JointWindowSelectionRunResult(
        records=int(run["records"]),
        selected_window=window,
        scans_path=scans_path,
        selection_path=selection_path,
        run_path=run_path,
    )


def run_select_generic_joint_patch_window(
    config: JointWindowSelectionRunConfig,
    *,
    runtime: ConfirmatoryJointWindowRuntime | None = None,
) -> JointWindowSelectionRunResult:
    if not isinstance(config, JointWindowSelectionRunConfig):
        raise TypeError("config must be JointWindowSelectionRunConfig")
    _validate_gpu_id(config.gpu_id)
    protocol = load_confirmatory_localization_config(config.config_path)
    records, manifest_sha = _load_manifest(
        config.selection_manifest_path,
        role="selection",
        expected_records=protocol.selection_records,
        protocol=protocol,
    )
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume:
        raise FileExistsError(f"joint-window selection output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.resume:
        existing = _existing_selection(output_dir, protocol=protocol, manifest_sha256=manifest_sha)
        if existing is not None:
            return existing
    runtime = _runtime(protocol, config.gpu_id, runtime)
    provenance = dict(runtime.provenance())
    runtime_sha = _sha256_value(provenance)
    run_path = output_dir / "run.json"
    run_base = {
        "schema_version": "select-generic-joint-patch-window-run/v1",
        "operation": "select-generic-joint-patch-window",
        "config_sha256": protocol.config_sha256,
        "selection_manifest_sha256": manifest_sha,
        "gpu_id": config.gpu_id,
        "runtime": provenance,
        "python": platform.python_version(),
    }
    _write_json(run_path, {**run_base, "status": "running", "started_at": _now()})
    work_dir = output_dir / ".selection-work"
    work_dir.mkdir(exist_ok=True)
    scans: list[JointWindowScan] = []
    try:
        scans = _load_or_scan(
            records=records,
            role="selection",
            protocol=protocol,
            manifest_sha256=manifest_sha,
            runtime=runtime,
            runtime_sha256=runtime_sha,
            work_dir=work_dir,
            resume=config.resume,
            selected_window=None,
            window_selection_sha256=None,
        )
        summary = select_joint_patch_window(
            scans,
            denominator_min_exclusive=protocol.denominator_min_exclusive,
            minimum_eligible=protocol.minimum_eligible,
            minimum_eligible_fraction=protocol.minimum_eligible_fraction,
            bootstrap_replicates=protocol.bootstrap_replicates,
            bootstrap_seed=protocol.bootstrap_seed,
            confidence_level=protocol.confidence_level,
            random_control_seed=protocol.random_control_seed,
        )
        scans_path = output_dir / "joint_window_scans.jsonl"
        _write_atomic(scans_path, b"".join(_canonical_bytes(scan.as_dict()) for scan in scans))
        selection_path = output_dir / "window_selection.json"
        _write_json(
            selection_path,
            {
                **summary.as_dict(),
                "model": protocol.model,
                "model_revision": protocol.model_revision,
                "config_sha256": protocol.config_sha256,
                "selection_manifest_sha256": manifest_sha,
                "joint_window_scans_sha256": _sha256_file(scans_path),
                "runtime": provenance,
            },
        )
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
                "eligible_records": summary.eligible_records,
                "selected_window": list(summary.selected_window),
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
                "completed_checkpoints": len(scans),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    return JointWindowSelectionRunResult(
        records=len(scans),
        selected_window=summary.selected_window,
        scans_path=scans_path,
        selection_path=selection_path,
        run_path=run_path,
    )


def _existing_validation(
    output_dir: Path,
    *,
    protocol: ConfirmatoryLocalizationProtocol,
    manifest_sha256: str,
    selection_sha256: str,
) -> JointWindowValidationRunResult | None:
    run_path = output_dir / "run.json"
    scans_path = output_dir / "joint_window_validation_scans.jsonl"
    validation_path = output_dir / "window_validation.json"
    if not all(path.is_file() for path in (run_path, scans_path, validation_path)):
        return None
    run = _load_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("config_sha256") != protocol.config_sha256
        or run.get("validation_manifest_sha256") != manifest_sha256
        or run.get("window_selection_sha256") != selection_sha256
    ):
        raise ValueError("completed joint-window validation binding differs")
    for path in (scans_path, validation_path):
        if not isinstance(run.get("outputs"), Mapping) or run["outputs"].get(path.name, {}).get(
            "sha256"
        ) != _sha256_file(path):
            raise ValueError("completed joint-window validation output hash differs")
    payload = _load_object(validation_path)
    return JointWindowValidationRunResult(
        records=int(run["records"]),
        passed=payload.get("passed") is True,
        scans_path=scans_path,
        validation_path=validation_path,
        run_path=run_path,
    )


def run_validate_generic_joint_patch_window(
    config: JointWindowValidationRunConfig,
    *,
    runtime: ConfirmatoryJointWindowRuntime | None = None,
) -> JointWindowValidationRunResult:
    if not isinstance(config, JointWindowValidationRunConfig):
        raise TypeError("config must be JointWindowValidationRunConfig")
    _validate_gpu_id(config.gpu_id)
    protocol = load_confirmatory_localization_config(config.config_path)
    records, manifest_sha = _load_manifest(
        config.validation_manifest_path,
        role="validation",
        expected_records=protocol.validation_records,
        protocol=protocol,
    )
    selected_window, selection_sha = _window_from_evidence(
        Path(config.window_selection_path), protocol
    )
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume:
        raise FileExistsError(f"joint-window validation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.resume:
        existing = _existing_validation(
            output_dir,
            protocol=protocol,
            manifest_sha256=manifest_sha,
            selection_sha256=selection_sha,
        )
        if existing is not None:
            return existing
    runtime = _runtime(protocol, config.gpu_id, runtime)
    provenance = dict(runtime.provenance())
    runtime_sha = _sha256_value(provenance)
    run_path = output_dir / "run.json"
    run_base = {
        "schema_version": "validate-generic-joint-patch-window-run/v1",
        "operation": "validate-generic-joint-patch-window",
        "config_sha256": protocol.config_sha256,
        "validation_manifest_sha256": manifest_sha,
        "window_selection_sha256": selection_sha,
        "selected_window": list(selected_window),
        "gpu_id": config.gpu_id,
        "runtime": provenance,
        "python": platform.python_version(),
    }
    _write_json(run_path, {**run_base, "status": "running", "started_at": _now()})
    work_dir = output_dir / ".validation-work"
    work_dir.mkdir(exist_ok=True)
    scans: list[JointWindowScan] = []
    try:
        scans = _load_or_scan(
            records=records,
            role="validation",
            protocol=protocol,
            manifest_sha256=manifest_sha,
            runtime=runtime,
            runtime_sha256=runtime_sha,
            work_dir=work_dir,
            resume=config.resume,
            selected_window=selected_window,
            window_selection_sha256=selection_sha,
        )
        summary = validate_joint_patch_window(
            scans,
            selected_window=selected_window,
            denominator_min_exclusive=protocol.denominator_min_exclusive,
            minimum_eligible=protocol.minimum_eligible,
            minimum_eligible_fraction=protocol.minimum_eligible_fraction,
            bootstrap_replicates=protocol.bootstrap_replicates,
            bootstrap_seed=protocol.bootstrap_seed,
            confidence_level=protocol.confidence_level,
        )
        scans_path = output_dir / "joint_window_validation_scans.jsonl"
        _write_atomic(scans_path, b"".join(_canonical_bytes(scan.as_dict()) for scan in scans))
        validation_path = output_dir / "window_validation.json"
        _write_json(
            validation_path,
            {
                **summary.as_dict(),
                "model": protocol.model,
                "model_revision": protocol.model_revision,
                "config_sha256": protocol.config_sha256,
                "validation_manifest_sha256": manifest_sha,
                "window_selection_sha256": selection_sha,
                "joint_window_validation_scans_sha256": _sha256_file(scans_path),
                "runtime": provenance,
            },
        )
        outputs = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (scans_path, validation_path)
        }
        _write_json(
            run_path,
            {
                **run_base,
                "status": "completed",
                "completed_at": _now(),
                "records": len(scans),
                "eligible_records": summary.eligible_records,
                "passed": summary.passed,
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
                "completed_checkpoints": len(scans),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    return JointWindowValidationRunResult(
        records=len(scans),
        passed=summary.passed,
        scans_path=scans_path,
        validation_path=validation_path,
        run_path=run_path,
    )


__all__ = [
    "ConfirmatoryJointWindowRuntime",
    "JointWindowSelectionRunConfig",
    "JointWindowSelectionRunResult",
    "JointWindowValidationRunConfig",
    "JointWindowValidationRunResult",
    "run_select_generic_joint_patch_window",
    "run_validate_generic_joint_patch_window",
]
