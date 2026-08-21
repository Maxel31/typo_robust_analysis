"""Canonical schemas, hashes, and atomic writes for localization artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path


FIXED_TYPO_PAIR_FIELDS = frozenset(
    {
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
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, value: bytes) -> None:
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


def write_json(path: Path, value: object) -> None:
    write_atomic(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    write_atomic(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


__all__ = [
    "FIXED_TYPO_PAIR_FIELDS",
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_file",
    "sha256_value",
    "utc_now",
    "write_atomic",
    "write_json",
    "write_jsonl",
]
