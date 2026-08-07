"""Self-contained executable-code identity for checkpoint compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_code_identity() -> dict[str, object]:
    """Fingerprint the installed sources that can affect this producer.

    The deliberately conservative transitive bundle prevents a resumed run
    from combining checkpoint outcomes produced by different executable code,
    even when a behavior change forgot to bump a protocol constant. Unrelated
    experiment packages are excluded so later features do not invalidate a
    completed one-token run.
    """

    package_root = Path(__file__).resolve().parents[2]
    directory_scopes = (
        "evaluation",
        "experiments/fixed_window_answer_patching",
        "experiments/one_token_prefix_replacement",
        "experiments/patch_coordinate_controls",
    )
    file_scopes = (
        "experiments/catalog.py",
        "experiments/clean_prefix_scan/planning.py",
        "experiments/clean_prefix_scan/source.py",
        "models/__init__.py",
        "models/wrapper.py",
    )
    files = {
        path
        for relative in directory_scopes
        for path in (package_root / relative).rglob("*.py")
        if path.is_file()
    }
    files.update(package_root / relative for relative in file_scopes)
    if any(not path.is_file() for path in files):
        raise RuntimeError("one-token executable code bundle is incomplete")
    entries = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(files)
    ]
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "algorithm": "one-token-executable-code-bundle-sha256/v1",
        "python_file_count": len(entries),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


__all__ = ["implementation_code_identity"]
