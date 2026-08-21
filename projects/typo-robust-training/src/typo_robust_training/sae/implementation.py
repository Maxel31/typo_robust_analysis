"""Canonical identity of the reviewed code used by one WP-2 retry claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def retry_implementation_sha256(repo_root: Path | None = None) -> str:
    """Hash the explicit editable runtime closure, independent of its location."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[5]
    )
    source_roots = (
        root / "projects" / "typo-robust-training" / "src",
        root / "projects" / "typo-cot" / "src",
    )
    fixed_files = (
        root / "projects" / "typo-robust-training" / "pyproject.toml",
        root / "projects" / "typo-cot" / "pyproject.toml",
        root / "projects" / "typo-robust-training" / "uv.lock",
    )
    included: list[Path] = []
    for source_root in source_roots:
        if not source_root.is_dir() or source_root.is_symlink():
            raise ValueError(f"retry implementation source root is invalid: {source_root}")
        included.extend(source_root.rglob("*.py"))
    included.extend(fixed_files)

    entries: list[dict[str, object]] = []
    for path in sorted(included, key=lambda candidate: candidate.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"retry implementation input is not a regular file: {path}")
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not entries:
        raise ValueError("retry implementation identity contains no files")
    manifest = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"robustness-sae-retry-implementation/v1\0" + manifest
    ).hexdigest()


__all__ = ["retry_implementation_sha256"]
