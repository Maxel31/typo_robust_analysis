"""CPU-analysis code identity kept separate from the GPU producer closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typo_cot.experiments.restoration_order_accuracy.integrity import (
    implementation_code_identity,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_PACKAGE = Path("experiments/restoration_order_accuracy")
_ANALYSIS_ONLY_FILES = (
    "__init__.py",
    "aggregation.py",
    "analysis_integrity.py",
    "reference.py",
    "statistics.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot read restoration-order analysis source: {path}") from exc
    return digest.hexdigest()


def analysis_code_identity() -> dict[str, object]:
    """Hash the producer identity and every analysis-only Python source."""
    producer = implementation_code_identity()
    producer_files = producer.get("files")
    if not isinstance(producer_files, dict):
        raise RuntimeError("restoration-order producer identity lacks its file map")
    files = dict(producer_files)
    for name in _ANALYSIS_ONLY_FILES:
        path = _PACKAGE_ROOT / _EXPERIMENT_PACKAGE / name
        if not path.is_file():
            raise RuntimeError(f"restoration-order analysis source is missing: {path}")
        files[path.relative_to(_PACKAGE_ROOT).as_posix()] = _sha256(path)
    files = dict(sorted(files.items()))
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "algorithm": "restoration-order-analysis-python-files-sha256/v1",
        "producer_sha256": producer.get("sha256"),
        "python_file_count": len(files),
        "files": files,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


__all__ = ["analysis_code_identity"]
