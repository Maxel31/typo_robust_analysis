"""The retry claim binds the complete editable implementation closure."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from typo_robust_training.sae.implementation import retry_implementation_sha256


def _implementation_tree(root: Path) -> Path:
    files = {
        "projects/typo-robust-training/src/typo_robust_training/main.py": b"TRAINING = 1\n",
        "projects/typo-cot/src/typo_cot/main.py": b"COT = 1\n",
        "projects/typo-robust-training/pyproject.toml": b"[project]\nname='training'\n",
        "projects/typo-cot/pyproject.toml": b"[project]\nname='cot'\n",
        "projects/typo-robust-training/uv.lock": b"version = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.mark.parametrize(
    "relative",
    (
        "projects/typo-robust-training/src/typo_robust_training/main.py",
        "projects/typo-cot/src/typo_cot/main.py",
        "projects/typo-robust-training/pyproject.toml",
        "projects/typo-cot/pyproject.toml",
        "projects/typo-robust-training/uv.lock",
    ),
)
def test_retry_implementation_identity_changes_for_every_runtime_input(
    tmp_path: Path,
    relative: str,
) -> None:
    root = _implementation_tree(tmp_path / "repo")
    before = retry_implementation_sha256(root)
    with (root / relative).open("ab") as handle:
        handle.write(b"# changed\n")

    assert retry_implementation_sha256(root) != before


def test_retry_implementation_identity_binds_add_remove_and_rename(tmp_path: Path) -> None:
    root = _implementation_tree(tmp_path / "repo")
    before = retry_implementation_sha256(root)
    added = root / "projects/typo-cot/src/typo_cot/added.py"
    added.write_text("ADDED = 1\n", encoding="utf-8")
    after_add = retry_implementation_sha256(root)
    assert after_add != before

    renamed = added.with_name("renamed.py")
    added.rename(renamed)
    after_rename = retry_implementation_sha256(root)
    assert after_rename not in {before, after_add}

    renamed.unlink()
    assert retry_implementation_sha256(root) == before


def test_retry_implementation_identity_ignores_location_metadata_and_caches(
    tmp_path: Path,
) -> None:
    first = _implementation_tree(tmp_path / "first")
    second = tmp_path / "second"
    shutil.copytree(first, second)
    target = second / "projects/typo-robust-training/src/typo_robust_training/main.py"
    os.chmod(target, 0o600)
    target.touch()
    cache = target.parent / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-312.pyc").write_bytes(b"ignored")

    assert retry_implementation_sha256(first) == retry_implementation_sha256(second)


def test_retry_implementation_identity_rejects_an_included_symlink(tmp_path: Path) -> None:
    root = _implementation_tree(tmp_path / "repo")
    target = root / "outside.py"
    target.write_text("OUTSIDE = 1\n", encoding="utf-8")
    link = root / "projects/typo-cot/src/typo_cot/link.py"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        retry_implementation_sha256(root)
