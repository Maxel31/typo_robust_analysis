"""Executable Python-source identity for typo-warning checkpoint compatibility."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from typo_cot.experiments.typo_warning_prompt.protocol import RUNTIME_ENVIRONMENT


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_WARNING_PACKAGE = Path("experiments/typo_warning_prompt")
_REQUIRED_WARNING_FILES = frozenset(
    {
        "__init__.py",
        "aggregation.py",
        "integrity.py",
        "planning.py",
        "protocol.py",
        "runner.py",
        "runtime.py",
        "scoring.py",
        "source.py",
        "statistics.py",
        "submitted_inputs.py",
    }
)
_REQUIRED_EXTERNAL_FILES = (
    Path("evaluation/extractor.py"),
    Path("models/prompts.py"),
    Path("models/wrapper.py"),
)
_PYTHON_VERSION = re.compile(r"([0-9]+)\.([0-9]+)(?:\.[0-9]+)?(?:[-+].*)?")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot read typo-warning executable source: {path}") from exc
    return digest.hexdigest()


def _source_paths() -> tuple[Path, ...]:
    warning_root = _PACKAGE_ROOT / _WARNING_PACKAGE
    if not warning_root.is_dir():
        raise RuntimeError(
            f"typo-warning executable code bundle is incomplete: missing {warning_root}"
        )

    warning_files = tuple(path for path in warning_root.rglob("*.py") if path.is_file())
    present_warning_files = {path.relative_to(warning_root).as_posix() for path in warning_files}
    missing_warning_files = sorted(_REQUIRED_WARNING_FILES - present_warning_files)
    external_files = tuple(_PACKAGE_ROOT / relative for relative in _REQUIRED_EXTERNAL_FILES)
    missing_external_files = sorted(
        path.relative_to(_PACKAGE_ROOT).as_posix() for path in external_files if not path.is_file()
    )
    if missing_warning_files or missing_external_files:
        missing = [
            *((_WARNING_PACKAGE / relative).as_posix() for relative in missing_warning_files),
            *missing_external_files,
        ]
        raise RuntimeError(
            "typo-warning executable code bundle is incomplete: " + ", ".join(missing)
        )
    return tuple(
        sorted(
            {*warning_files, *external_files},
            key=lambda path: path.relative_to(_PACKAGE_ROOT).as_posix(),
        )
    )


def implementation_file_sha256s() -> dict[str, str]:
    """Return deterministic package-relative SHA-256s for result-affecting sources."""

    return {path.relative_to(_PACKAGE_ROOT).as_posix(): _sha256(path) for path in _source_paths()}


def implementation_code_identity() -> dict[str, object]:
    """Return a provenance payload suitable for runtime and checkpoint manifests."""

    files = implementation_file_sha256s()
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "algorithm": "typo-warning-executable-python-files-sha256/v1",
        "python_file_count": len(files),
        "files": files,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_paper_runtime_environment(
    provenance: Mapping[str, object],
    *,
    field_prefix: str,
) -> None:
    """Validate the recorded execution environment against the paper lock."""

    exact = {
        "runtime": RUNTIME_ENVIRONMENT["runtime"],
        **RUNTIME_ENVIRONMENT["package_versions"],
    }
    for field, expected in exact.items():
        if provenance.get(field) != expected:
            raise ValueError(f"{field_prefix} {field} must be {expected!r}")

    python = provenance.get("python")
    match = _PYTHON_VERSION.fullmatch(python) if isinstance(python, str) else None
    minimum_python = tuple(RUNTIME_ENVIRONMENT["minimum_python"])
    if match is None or tuple(int(part) for part in match.groups()[:2]) < minimum_python:
        required = ".".join(str(part) for part in minimum_python)
        raise ValueError(f"{field_prefix} python must be version {required} or newer")

    revision_sources = {
        "model_revision_source": {"model-config-metadata", "explicit-load-revision"},
        "tokenizer_revision_source": {
            "tokenizer-init-metadata",
            "explicit-load-revision",
        },
    }
    for field, permitted in revision_sources.items():
        value = provenance.get(field)
        if value not in permitted:
            raise ValueError(f"{field_prefix} {field} is invalid")
    device = provenance.get("device")
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError(f"{field_prefix} device must identify a CUDA device")
    for field in ("cuda", "gpu_name"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_prefix} {field} must be non-empty")
    total_memory = provenance.get("gpu_total_memory_bytes")
    if not isinstance(total_memory, int) or isinstance(total_memory, bool) or total_memory <= 0:
        raise ValueError(f"{field_prefix} gpu_total_memory_bytes must be positive")


__all__ = [
    "implementation_code_identity",
    "implementation_file_sha256s",
    "validate_paper_runtime_environment",
]
