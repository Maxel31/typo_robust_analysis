"""Shared captured-byte I/O and immutable publication for CPU audit stages."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schemas import (
    IntakeError,
    canonical_json,
    canonical_sha256,
    require_object,
    sha256_file,
    strict_loads,
    validate_artifact_ref,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(json_bytes(row) for row in rows)


def artifact_ref(path: str | Path, raw: bytes) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def capture_file(
    path: str | Path, snapshots: dict[Path, str], expected: str | None = None
) -> bytes:
    """Hash the exact bytes returned for parsing, binding previous observations."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise IntakeError(f"artifact SHA-256 mismatch at {path}")
    if path in snapshots and snapshots[path] != digest:
        raise IntakeError(f"artifact changed after capture: {path}")
    snapshots[path] = digest
    return raw


def resolve_artifact(ref: object, owner: str | Path, snapshots: dict[Path, str]) -> Path:
    value = validate_artifact_ref(ref)
    path = (Path(owner).resolve().parent / value["path"]).resolve()
    capture_file(path, snapshots, value["sha256"])
    return path


def read_json(path: str | Path, snapshots: dict[Path, str]) -> dict[str, Any]:
    path = Path(path).resolve()
    return require_object(
        strict_loads(capture_file(path, snapshots).decode("utf-8"), str(path)), str(path)
    )


def read_jsonl(path: str | Path, snapshots: dict[Path, str]) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    return [
        require_object(strict_loads(line, f"{path}:{index}"), f"{path}:{index}")
        for index, line in enumerate(capture_file(path, snapshots).decode("utf-8").split("\n"), 1)
        if line.strip()
    ]


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        cells = {}
        for field in fields:
            value = row.get(field)
            if isinstance(value, (dict, list)):
                value = canonical_json(value)
            if value is None:
                value = "NA"
            if isinstance(value, str) and re.match(r"^[\s\x00-\x20]*[=+@-]", value):
                value = "'" + value
            cells[field] = value
        writer.writerow(cells)
    return output.getvalue().encode("utf-8")


def capture_code(paths: Iterable[Path], snapshots: dict[Path, str]) -> dict[str, Any]:
    refs = []
    for path in sorted({Path(path).resolve() for path in paths}):
        refs.append(artifact_ref(path, capture_file(path, snapshots)))
    commit, dirty, reasons = None, None, []
    base = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        value = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
            commit = value
        else:
            reasons.append("code_commit_unknown")
        result = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        dirty = bool(result.stdout)
    except (OSError, subprocess.SubprocessError):
        reasons.append("code_git_identity_unavailable")
    return {"source_refs": refs, "git_commit": commit, "git_dirty": dirty, "reason_codes": reasons}


def revalidate_snapshots(snapshots: dict[Path, str]) -> None:
    for path, expected in snapshots.items():
        if sha256_file(path) != expected:
            raise IntakeError(f"artifact or code changed after capture: {path}")


def publish_artifacts(
    output_dir: str | Path, files: dict[str, bytes], snapshots: dict[Path, str]
) -> None:
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IntakeError(f"Output directory must be absent or empty: {output}")
    if any(Path(name).name != name or name in {"", ".", ".."} for name in files):
        raise IntakeError("Published artifacts require flat file names")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.audit-", dir=output.parent))
    try:
        for name, raw in files.items():
            (staging / name).write_bytes(raw)
        revalidate_snapshots(snapshots)
        if output.exists():
            output.rmdir()
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise


def make_run(
    *,
    command: str,
    started_at: str,
    invocation: dict[str, Any],
    snapshots: dict[Path, str],
    code_identity: dict[str, Any],
    protocol: dict[str, Any],
    files: dict[str, bytes],
    expected_ids: list[str],
    completed_ids: list[str],
    missing_ids: list[str],
    reason_codes: list[str],
    config_refs: list[dict[str, str]] | None = None,
    experiment_status: str = "partial",
) -> dict[str, Any]:
    return {
        "schema_version": "rebuttal-run/v2",
        "command": command,
        "mode": "cpu-audit",
        "experiment_id": "R2",
        "experiment_status": experiment_status,
        "started_at": started_at,
        "ended_at": utc_now(),
        "invocation": invocation,
        "code_identity": code_identity,
        "protocol_revision": protocol["protocol_revision"],
        "protocol_sha256": canonical_sha256(protocol),
        "input_refs": [
            {"path": str(path), "sha256": digest} for path, digest in sorted(snapshots.items())
        ],
        "config_refs": config_refs or [],
        "output_refs": {name: artifact_ref(name, raw) for name, raw in sorted(files.items())},
        "run_status": "complete",
        "expected_ids": expected_ids,
        "completed_ids": completed_ids,
        "failed_ids": [],
        "missing_ids": missing_ids,
        "reason_codes": sorted(set(reason_codes)),
        "model_execution_performed": False,
    }
