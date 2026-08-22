from __future__ import annotations

import importlib
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

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
            return "\n".join(
                path.relative_to(root).as_posix()
                for path in sorted((root / package).rglob("*.py"))
            )
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
    assert observed.typo_cot_runtime_sources == tuple(
        sorted(attestation._RUNTIME_DEPENDENCY_SOURCES.values())
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "typo_cot.experiments.layerwise_kl_patching.patching",
        "typo_cot.models.wrapper",
    ],
)
def test_runtime_attestation_rejects_rogue_preloaded_runtime_submodule(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    """A legitimate top package cannot hide a rogue preloaded runtime leaf."""

    _fake_git(monkeypatch, revision="a" * 40)
    importlib.import_module("typo_cot")
    rogue = ModuleType(module_name)
    rogue.__file__ = f"/tmp/rogue/{module_name.rsplit('.', 1)[-1]}.py"
    rogue.__spec__ = importlib.machinery.ModuleSpec(
        module_name,
        loader=None,
        origin=rogue.__file__,
    )
    monkeypatch.setitem(sys.modules, module_name, rogue)

    with pytest.raises(
        ValueError,
        match="dependency source is outside the required checkout",
    ):
        attestation.attest_runtime_checkout("a" * 40)


def test_runtime_attestation_rejects_unknown_preloaded_runtime_submodule_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_git(monkeypatch, revision="a" * 40)
    importlib.import_module("typo_cot")
    module_name = "typo_cot.models.wrapper"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    with pytest.raises(ValueError, match="source identity is unavailable"):
        attestation.attest_runtime_checkout("a" * 40)


def test_runtime_attestation_rejects_untracked_actual_dependency_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[3]

    def fake(*arguments: str, cwd: Path) -> str:
        del cwd
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(root)
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments[:3] == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments[:2] == ("ls-files", "--"):
            package = arguments[2]
            if "typo-cot" in package:
                return f"{package}/models/wrapper.py"
            return f"{package}/__init__.py"
        if arguments[0] == "rev-parse" and arguments[1].startswith("HEAD:"):
            return "f" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(attestation, "_git", fake)
    with pytest.raises(ValueError, match="dependency source is not tracked"):
        attestation.attest_runtime_checkout("a" * 40)


def test_runtime_attestation_rejects_loaded_dependency_file_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing __file__ after import cannot preserve a valid source attestation."""

    _fake_git(monkeypatch, revision="a" * 40)
    loaded = importlib.import_module("typo_cot")
    monkeypatch.setattr(loaded, "__file__", "/tmp/rogue/typo_cot/__init__.py")

    with pytest.raises(ValueError, match="loaded typo_cot source differs"):
        attestation.attest_runtime_checkout("a" * 40)
