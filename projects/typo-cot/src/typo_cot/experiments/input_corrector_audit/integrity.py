"""Executable-source and runtime identities for the input-corrector audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PACKAGE = Path("experiments/input_corrector_audit")
_PRODUCER_FILES = frozenset(
    {
        "correctors.py",
        "integrity.py",
        "protocol.py",
        "restoration.py",
        "runner.py",
        "runtime.py",
        "source.py",
    }
)
_ANALYSIS_FILES = _PRODUCER_FILES | {"__init__.py", "aggregation.py"}
_EXTERNAL_FILES = (
    Path("evaluation/extractor.py"),
    Path("evaluation/fallback.py"),
    Path("models/prompts.py"),
    Path("models/wrapper.py"),
)
_PYTHON_VERSION = re.compile(r"([0-9]+)\.([0-9]+)(?:\.[0-9]+)?(?:[-+].*)?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SINGLE_GPU = re.compile(r"0|[1-9][0-9]*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot read input-corrector executable source: {path}") from exc
    return digest.hexdigest()


def _source_paths(file_names: frozenset[str] | set[str]) -> tuple[Path, ...]:
    package = _PACKAGE_ROOT / _INPUT_PACKAGE
    if not package.is_dir():
        raise RuntimeError(f"input-corrector code bundle is incomplete: missing {package}")
    package_files = tuple(package / name for name in file_names)
    external_files = tuple(_PACKAGE_ROOT / path for path in _EXTERNAL_FILES)
    files = (*package_files, *external_files)
    missing = sorted(
        path.relative_to(_PACKAGE_ROOT).as_posix() for path in files if not path.is_file()
    )
    if missing:
        raise RuntimeError(
            "input-corrector code bundle is incomplete: missing " + ", ".join(missing)
        )
    return tuple(sorted(files, key=lambda path: path.relative_to(_PACKAGE_ROOT).as_posix()))


def _identity(file_names: frozenset[str] | set[str], *, algorithm: str) -> dict[str, object]:
    files = {
        path.relative_to(_PACKAGE_ROOT).as_posix(): _sha256(path)
        for path in _source_paths(file_names)
    }
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "algorithm": algorithm,
        "python_file_count": len(files),
        "files": files,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def implementation_code_identity() -> dict[str, object]:
    """Hash every Python source that affects producer records and scoring."""
    return _identity(
        _PRODUCER_FILES,
        algorithm="input-corrector-producer-python-files-sha256/v1",
    )


def analysis_code_identity() -> dict[str, object]:
    """Hash the producer closure plus CPU validation and rendering sources."""
    return _identity(
        _ANALYSIS_FILES,
        algorithm="input-corrector-analysis-python-files-sha256/v1",
    )


def _validate_python(provenance: Mapping[str, object], *, field_prefix: str) -> None:
    value = provenance.get("python")
    match = _PYTHON_VERSION.fullmatch(value) if isinstance(value, str) else None
    if match is None or tuple(int(part) for part in match.groups()[:2]) < (3, 12):
        raise ValueError(f"{field_prefix} python must be version 3.12 or newer")


def validate_paper_runtime_environment(
    provenance: Mapping[str, object],
    *,
    profile: str,
    field_prefix: str,
) -> None:
    """Validate one neural or dictionary runtime against the public lock."""
    _validate_python(provenance, field_prefix=field_prefix)
    if profile == "neural":
        exact = {
            "torch": "2.10.0",
            "transformers": "4.57.6",
            "accelerate": "1.12.0",
        }
        for field, expected in exact.items():
            if provenance.get(field) != expected:
                raise ValueError(f"{field_prefix} {field} must be {expected!r}")
        device = provenance.get("device")
        if not isinstance(device, str) or not device.startswith("cuda:"):
            raise ValueError(f"{field_prefix} device must identify a CUDA device")
        visible = provenance.get("cuda_visible_devices")
        if not isinstance(visible, str) or _SINGLE_GPU.fullmatch(visible) is None:
            raise ValueError(f"{field_prefix} cuda_visible_devices must identify one GPU")
        for field in ("cuda", "gpu_name"):
            value = provenance.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_prefix} {field} must be non-empty")
        total_memory = provenance.get("gpu_total_memory_bytes")
        if not isinstance(total_memory, int) or isinstance(total_memory, bool) or total_memory <= 0:
            raise ValueError(f"{field_prefix} gpu_total_memory_bytes must be positive")
        return
    if profile == "pyspellchecker":
        if provenance.get("pyspellchecker") != "0.9.0":
            raise ValueError(f"{field_prefix} pyspellchecker must be '0.9.0'")
        if provenance.get("device") != "cpu":
            raise ValueError(f"{field_prefix} device must be 'cpu'")
        if provenance.get("dictionary_language") != "en":
            raise ValueError(f"{field_prefix} dictionary_language must be 'en'")
        digest = provenance.get("dictionary_sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{field_prefix} dictionary_sha256 must be a SHA-256")
        return
    raise ValueError(f"unsupported input-corrector runtime profile: {profile!r}")


__all__ = [
    "analysis_code_identity",
    "implementation_code_identity",
    "validate_paper_runtime_environment",
]
