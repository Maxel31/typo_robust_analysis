"""Backward-compatible atomic JSON writer for training metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_json_atomic(path: Path, payload: object) -> None:
    """Replace one UTF-8 JSON artifact only after serialization succeeds."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_durable(path: Path, payload: object) -> None:
    """Atomically replace JSON and fsync both content and directory entry."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["write_json_atomic", "write_json_durable"]
