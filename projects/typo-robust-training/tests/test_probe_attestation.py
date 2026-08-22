from __future__ import annotations

from pathlib import Path

import pytest

from typo_robust_training.probe import attestation


def _fake_git(monkeypatch: pytest.MonkeyPatch, *, revision: str, status: str = "") -> None:
    root = Path(__file__).resolve().parents[3]

    def fake(*arguments: str, cwd: Path) -> str:
        del cwd
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(root)
        if arguments == ("rev-parse", "HEAD"):
            return revision
        if arguments[:3] == ("status", "--porcelain=v1", "--untracked-files=all"):
            return status
        if arguments[:2] == ("ls-files", "--"):
            package = arguments[2]
            return f"{package}/__init__.py"
        if arguments[0] == "rev-parse" and arguments[1].startswith("HEAD:"):
            return "f" * 40 if "typo-robust-training" in arguments[1] else "e" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(attestation, "_git", fake)


def test_runtime_attestation_rejects_wrong_current_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_git(monkeypatch, revision="a" * 40)

    with pytest.raises(ValueError, match="revision differs"):
        attestation.attest_runtime_checkout("b" * 40)


def test_runtime_attestation_rejects_dirty_or_untracked_runtime_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_git(
        monkeypatch,
        revision="a" * 40,
        status="?? projects/typo-cot/src/typo_cot/injected.py",
    )

    with pytest.raises(ValueError, match="dirty or contain untracked"):
        attestation.attest_runtime_checkout("a" * 40)


def test_runtime_attestation_binds_both_clean_tracked_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_git(monkeypatch, revision="a" * 40)

    observed = attestation.attest_runtime_checkout("a" * 40)

    assert observed.revision == "a" * 40
    assert observed.typo_robust_training_tree == "f" * 40
    assert observed.typo_cot_tree == "e" * 40
