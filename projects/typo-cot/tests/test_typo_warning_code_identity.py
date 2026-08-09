"""Regression tests for typo-warning executable-source provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import typo_cot.experiments.typo_warning_prompt.integrity as warning_integrity


def _aggregate_sha256(files: dict[str, str]) -> str:
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_warning_code_identity_returns_stable_relative_file_hashes() -> None:
    package_root = Path(warning_integrity.__file__).resolve().parents[2]

    first = warning_integrity.implementation_code_identity()
    second = warning_integrity.implementation_code_identity()

    assert first == second
    assert first["algorithm"] == "typo-warning-executable-python-files-sha256/v1"
    files = first["files"]
    assert isinstance(files, dict)
    assert list(files) == sorted(files)
    assert set(files) >= {
        "evaluation/extractor.py",
        "experiments/typo_warning_prompt/__init__.py",
        "experiments/typo_warning_prompt/aggregation.py",
        "experiments/typo_warning_prompt/integrity.py",
        "experiments/typo_warning_prompt/planning.py",
        "experiments/typo_warning_prompt/protocol.py",
        "experiments/typo_warning_prompt/runner.py",
        "experiments/typo_warning_prompt/runtime.py",
        "experiments/typo_warning_prompt/scoring.py",
        "experiments/typo_warning_prompt/source.py",
        "experiments/typo_warning_prompt/statistics.py",
        "experiments/typo_warning_prompt/submitted_inputs.py",
        "models/prompts.py",
        "models/wrapper.py",
    }
    assert first["python_file_count"] == len(files)
    assert first["sha256"] == _aggregate_sha256(files)
    for relative, digest in files.items():
        assert not Path(relative).is_absolute()
        assert digest == hashlib.sha256((package_root / relative).read_bytes()).hexdigest()


def test_warning_code_identity_fails_closed_when_bundle_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(warning_integrity, "_PACKAGE_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="incomplete"):
        warning_integrity.implementation_code_identity()
