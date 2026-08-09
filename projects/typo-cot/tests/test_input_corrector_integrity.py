"""Executable-source and runtime-environment contracts for input correction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import typo_cot.experiments.input_corrector_audit.integrity as corrector_integrity


_INPUT_PACKAGE = "experiments/input_corrector_audit/"
_EXTERNAL_PRODUCER_FILES = {
    "evaluation/extractor.py",
    "evaluation/fallback.py",
    "models/prompts.py",
    "models/wrapper.py",
}
_REQUIRED_PRODUCER_FILES = _EXTERNAL_PRODUCER_FILES | {
    f"{_INPUT_PACKAGE}correctors.py",
    f"{_INPUT_PACKAGE}integrity.py",
    f"{_INPUT_PACKAGE}protocol.py",
    f"{_INPUT_PACKAGE}restoration.py",
    f"{_INPUT_PACKAGE}runner.py",
    f"{_INPUT_PACKAGE}runtime.py",
    f"{_INPUT_PACKAGE}source.py",
}


def _aggregate_sha256(files: dict[str, str]) -> str:
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_file_identity(identity: dict[str, object]) -> dict[str, str]:
    files = identity["files"]
    assert isinstance(files, dict)
    assert files
    assert list(files) == sorted(files)
    assert identity["python_file_count"] == len(files)
    assert identity["sha256"] == _aggregate_sha256(files)

    package_root = Path(corrector_integrity.__file__).resolve().parents[2]
    for relative, digest in files.items():
        assert isinstance(relative, str)
        assert not Path(relative).is_absolute()
        assert isinstance(digest, str)
        assert digest == hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
    return files


def test_integrity_module_exposes_only_the_small_public_boundary() -> None:
    assert set(corrector_integrity.__all__) == {
        "analysis_code_identity",
        "implementation_code_identity",
        "validate_paper_runtime_environment",
    }


def test_producer_identity_hashes_the_executable_closure_without_analysis_files() -> None:
    first = corrector_integrity.implementation_code_identity()
    second = corrector_integrity.implementation_code_identity()

    assert first == second
    assert first["algorithm"] == "input-corrector-producer-python-files-sha256/v1"
    files = _assert_file_identity(first)
    assert _REQUIRED_PRODUCER_FILES <= set(files)
    assert f"{_INPUT_PACKAGE}aggregation.py" not in files
    assert f"{_INPUT_PACKAGE}__init__.py" not in files
    assert "experiments/catalog.py" not in files
    assert all(
        relative.startswith(_INPUT_PACKAGE) or relative in _EXTERNAL_PRODUCER_FILES
        for relative in files
    )


def test_analysis_identity_contains_the_producer_and_cpu_builder_closures() -> None:
    producer = corrector_integrity.implementation_code_identity()
    analysis = corrector_integrity.analysis_code_identity()

    assert analysis["algorithm"] == "input-corrector-analysis-python-files-sha256/v1"
    producer_files = _assert_file_identity(producer)
    analysis_files = _assert_file_identity(analysis)
    assert producer_files.items() <= analysis_files.items()
    assert {
        f"{_INPUT_PACKAGE}__init__.py",
        f"{_INPUT_PACKAGE}aggregation.py",
    } <= set(analysis_files)
    assert "experiments/catalog.py" not in analysis_files
    assert all(
        relative.startswith(_INPUT_PACKAGE) or relative in _EXTERNAL_PRODUCER_FILES
        for relative in analysis_files
    )


@pytest.mark.parametrize(
    "identity_function",
    (
        pytest.param(
            lambda: corrector_integrity.implementation_code_identity(),
            id="producer",
        ),
        pytest.param(
            lambda: corrector_integrity.analysis_code_identity(),
            id="analysis",
        ),
    ),
)
def test_code_identity_fails_closed_when_the_source_bundle_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_function: Callable[[], dict[str, object]],
) -> None:
    monkeypatch.setattr(corrector_integrity, "_PACKAGE_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="incomplete|missing"):
        identity_function()


def _neural_environment() -> dict[str, object]:
    return {
        "python": "3.12.7",
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
        "device": "cuda:0",
        "cuda": "12.8",
        "cuda_visible_devices": "1",
        "gpu_name": "Synthetic CUDA GPU",
        "gpu_total_memory_bytes": 1024,
    }


def _pyspellchecker_environment() -> dict[str, object]:
    return {
        "python": "3.12.7",
        "pyspellchecker": "0.9.0",
        "device": "cpu",
        "dictionary_language": "en",
        "dictionary_sha256": "1" * 64,
    }


def test_environment_validation_accepts_the_two_execution_profiles() -> None:
    corrector_integrity.validate_paper_runtime_environment(
        _neural_environment(),
        profile="neural",
        field_prefix="generation environment",
    )
    corrector_integrity.validate_paper_runtime_environment(
        _pyspellchecker_environment(),
        profile="pyspellchecker",
        field_prefix="correction environment",
    )


@pytest.mark.parametrize(
    ("profile", "field", "value"),
    (
        ("neural", "python", "3.11.9"),
        ("neural", "torch", "2.9.0"),
        ("neural", "device", "cpu"),
        ("neural", "cuda_visible_devices", "0,1"),
        ("neural", "gpu_total_memory_bytes", True),
        ("pyspellchecker", "pyspellchecker", "0.8.0"),
        ("pyspellchecker", "device", "cuda:0"),
        ("pyspellchecker", "dictionary_sha256", "not-a-sha256"),
    ),
)
def test_environment_validation_rejects_incompatible_or_unverifiable_values(
    profile: str,
    field: str,
    value: object,
) -> None:
    provenance = _neural_environment() if profile == "neural" else _pyspellchecker_environment()
    provenance[field] = value

    with pytest.raises(ValueError, match=rf"input-corrector runtime.*{field}"):
        corrector_integrity.validate_paper_runtime_environment(
            provenance,
            profile=profile,
            field_prefix="input-corrector runtime",
        )


def test_environment_validation_rejects_an_unknown_profile() -> None:
    with pytest.raises(ValueError, match="profile"):
        corrector_integrity.validate_paper_runtime_environment(
            _neural_environment(),
            profile="unknown",
            field_prefix="input-corrector runtime",
        )
