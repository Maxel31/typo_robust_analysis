"""Executable-source and runtime identities for the Table 13 operation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_PACKAGE = Path("experiments/restoration_order_accuracy")
_PRODUCER_FILES = frozenset(
    {
        "integrity.py",
        "planning.py",
        "protocol.py",
        "publication.py",
        "runner.py",
        "runtime.py",
        "source.py",
    }
)
_EXTERNAL_FILES = (
    Path("evaluation/extractor.py"),
    Path("evaluation/fallback.py"),
    Path("evaluation/generation.py"),
    Path("experiments/input_corrector_audit/protocol.py"),
    Path("experiments/input_corrector_audit/restoration.py"),
    Path("experiments/input_corrector_audit/source.py"),
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
        raise RuntimeError(f"cannot read restoration-order executable source: {path}") from exc
    return digest.hexdigest()


def _paths(names: frozenset[str] | set[str]) -> tuple[Path, ...]:
    package = _PACKAGE_ROOT / _EXPERIMENT_PACKAGE
    files = (*(package / name for name in names), *(_PACKAGE_ROOT / path for path in _EXTERNAL_FILES))
    missing = sorted(
        path.relative_to(_PACKAGE_ROOT).as_posix() for path in files if not path.is_file()
    )
    if missing:
        raise RuntimeError(
            "restoration-order code bundle is incomplete: missing " + ", ".join(missing)
        )
    return tuple(sorted(files, key=lambda path: path.relative_to(_PACKAGE_ROOT).as_posix()))


def _identity(names: frozenset[str] | set[str], *, algorithm: str) -> dict[str, object]:
    files = {
        path.relative_to(_PACKAGE_ROOT).as_posix(): _sha256(path) for path in _paths(names)
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
    """Hash every Python source that affects producer prompts and scoring."""
    return _identity(
        _PRODUCER_FILES,
        algorithm="restoration-order-producer-python-files-sha256/v1",
    )


def validate_paper_runtime_environment(provenance: Mapping[str, object]) -> None:
    """Fail closed unless one runtime matches the final-paper environment."""
    python = provenance.get("python")
    match = _PYTHON_VERSION.fullmatch(python) if isinstance(python, str) else None
    if match is None or tuple(int(part) for part in match.groups()[:2]) < (3, 12):
        raise ValueError("restoration-order runtime python must be 3.12 or newer")
    for field, expected in {
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
    }.items():
        if provenance.get(field) != expected:
            raise ValueError(f"restoration-order runtime {field} must be {expected!r}")
    visible = provenance.get("cuda_visible_devices")
    if not isinstance(visible, str) or _SINGLE_GPU.fullmatch(visible) is None:
        raise ValueError("restoration-order runtime must expose exactly one physical GPU")
    if provenance.get("device") != "cuda:0":
        raise ValueError("restoration-order runtime device must be 'cuda:0'")
    for field in ("cuda", "gpu_name"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError(f"restoration-order runtime {field} must be non-empty")
    memory = provenance.get("gpu_total_memory_bytes")
    if not isinstance(memory, int) or isinstance(memory, bool) or memory <= 0:
        raise ValueError("restoration-order runtime GPU memory must be positive")
    for field in ("model_revision", "tokenizer_revision"):
        value = provenance.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"restoration-order runtime {field} must be a pinned commit")
    code = provenance.get("implementation_code")
    if not isinstance(code, Mapping) or _SHA256.fullmatch(str(code.get("sha256", ""))) is None:
        raise ValueError("restoration-order runtime implementation identity is invalid")


__all__ = [
    "implementation_code_identity",
    "validate_paper_runtime_environment",
]
