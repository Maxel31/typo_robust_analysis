"""Canonical LF-delimited JSONL input shared by scientific artifacts."""

from __future__ import annotations

from pathlib import Path


def read_lf_jsonl_lines(path: Path, *, context: str) -> tuple[tuple[int, str], ...]:
    """Read records using ASCII LF only and reject partial/blank artifacts."""

    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"{context} is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context} is not UTF-8: {resolved}") from exc
    if not text.endswith("\n"):
        raise ValueError(f"{context} must end with a final LF: {resolved}")
    lines = text[:-1].split("\n")
    if any(not line for line in lines):
        line_number = next(index for index, line in enumerate(lines, 1) if not line)
        raise ValueError(f"{context} has a blank line: {resolved}:{line_number}")
    return tuple(enumerate(lines, 1))


__all__ = ["read_lf_jsonl_lines"]
