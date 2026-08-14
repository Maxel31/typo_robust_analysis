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


__all__ = ["write_json_atomic"]
