"""Atomic training metadata writes preserve the prior valid artifact on failure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from typo_robust_training.training.json_io import write_json_atomic, write_json_durable


def test_atomic_json_write_is_canonical_and_failure_preserves_prior_file(tmp_path: Path) -> None:
    target = tmp_path / "metadata.json"
    write_json_atomic(target, {"z": 2, "a": 1})
    assert target.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "z": 2\n}\n'

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_atomic(target, {"invalid": float("nan")})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "z": 2}
    assert list(tmp_path.iterdir()) == [target]


def test_durable_json_write_ignores_an_abandoned_same_pid_temporary(
    tmp_path: Path,
) -> None:
    target = tmp_path / "metadata.json"
    abandoned = tmp_path / f".metadata.json.{os.getpid()}.tmp"
    abandoned.write_bytes(b"abandoned-but-non-authoritative")

    write_json_durable(target, {"current": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"current": True}
    assert abandoned.read_bytes() == b"abandoned-but-non-authoritative"
