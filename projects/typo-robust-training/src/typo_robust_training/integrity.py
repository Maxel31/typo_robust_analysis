"""Canonical file and directory digests shared by training and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading it wholly into memory."""

    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"hash input must be one regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash a directory from canonical relative paths, sizes, and file digests."""

    resolved = Path(root)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"hash input must be one regular directory: {resolved}")
    entries = tuple(sorted(resolved.rglob("*")))
    if any(path.is_symlink() for path in entries):
        raise ValueError("hash tree cannot contain symbolic links")
    files = tuple(path for path in entries if path.is_file())
    if not files:
        raise ValueError("hash tree must contain at least one regular file")
    inventory = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = ["sha256_file", "sha256_tree"]
