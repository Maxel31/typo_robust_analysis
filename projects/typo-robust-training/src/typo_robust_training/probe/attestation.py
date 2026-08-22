"""Fail-closed source and model identity attestation for GPU diagnostics."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REVISION = re.compile(r"[0-9a-f]{40}")
_TRACKED_TREES = (
    "projects/typo-robust-training/src/typo_robust_training",
    "projects/typo-cot/src/typo_cot",
)
_RUNTIME_DEPENDENCY_MODULES = (
    "typo_cot",
    "typo_cot.experiments.layerwise_kl_patching.patching",
    "typo_cot.models.wrapper",
)
_RUNTIME_DEPENDENCY_SOURCES = {
    "typo_cot": "projects/typo-cot/src/typo_cot/__init__.py",
    "typo_cot.experiments": "projects/typo-cot/src/typo_cot/experiments/__init__.py",
    "typo_cot.experiments.layerwise_kl_patching": (
        "projects/typo-cot/src/typo_cot/experiments/layerwise_kl_patching/__init__.py"
    ),
    "typo_cot.experiments.layerwise_kl_patching.patching": (
        "projects/typo-cot/src/typo_cot/experiments/layerwise_kl_patching/patching.py"
    ),
    "typo_cot.models": "projects/typo-cot/src/typo_cot/models/__init__.py",
    "typo_cot.models.wrapper": "projects/typo-cot/src/typo_cot/models/wrapper.py",
}


def _git(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("runtime checkout cannot be attested by git")
    return result.stdout.strip()


@dataclass(frozen=True, slots=True)
class RuntimeCheckoutAttestation:
    revision: str
    typo_robust_training_tree: str
    typo_cot_tree: str
    typo_cot_runtime_sources: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "typo_robust_training_tree": self.typo_robust_training_tree,
            "typo_cot_tree": self.typo_cot_tree,
            "typo_cot_runtime_sources": list(self.typo_cot_runtime_sources),
        }


def attest_runtime_checkout(expected_revision: str) -> RuntimeCheckoutAttestation:
    """Attest HEAD and both complete runtime source trees before CUDA/output.

    Staged, unstaged, and untracked files under either package are rejected.
    The git tree object IDs bind every tracked path and byte in the two trees.
    """

    if not isinstance(expected_revision, str) or _REVISION.fullmatch(expected_revision) is None:
        raise ValueError("kill runtime revision must be one pinned git revision")
    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel", cwd=module.parent)).resolve()
    try:
        training_root_relative = module.parents[1].relative_to(root)
    except ValueError as exc:
        raise ValueError("kill runtime source is outside the required checkout layout") from exc
    if training_root_relative != Path(_TRACKED_TREES[0]):
        raise ValueError("kill runtime source is outside the required checkout layout")
    revision = _git("rev-parse", "HEAD", cwd=root)
    if revision != expected_revision:
        raise ValueError("kill runtime checkout revision differs from preregistration")
    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_TRACKED_TREES,
        cwd=root,
    )
    if status:
        raise ValueError("kill runtime source trees are dirty or contain untracked files")
    trees: list[str] = []
    tracked_by_tree: list[set[str]] = []
    for relative in _TRACKED_TREES:
        tracked = _git("ls-files", "--", relative, cwd=root).splitlines()
        if not tracked or any(not (root / item).is_file() for item in tracked):
            raise ValueError("kill runtime source tree is not fully tracked")
        tracked_by_tree.append(set(tracked))
        tree = _git("rev-parse", f"HEAD:{relative}", cwd=root)
        if _REVISION.fullmatch(tree) is None:
            raise ValueError("kill runtime source tree identity is unavailable")
        trees.append(tree)

    try:
        for module_name in _RUNTIME_DEPENDENCY_MODULES:
            importlib.import_module(module_name)
    except (ImportError, ValueError) as exc:
        raise ValueError("kill runtime cannot import its typo_cot dependency source") from exc
    dependency_sources: set[str] = set()
    required_sources = {
        module_name: Path(relative)
        for module_name, relative in _RUNTIME_DEPENDENCY_SOURCES.items()
    }
    for module_name, loaded in tuple(sys.modules.items()):
        if module_name != "typo_cot" and not module_name.startswith("typo_cot."):
            continue
        loaded_origin = getattr(loaded, "__file__", None)
        spec_origin = getattr(getattr(loaded, "__spec__", None), "origin", None)
        if not isinstance(loaded_origin, str) or not isinstance(spec_origin, str):
            raise ValueError("loaded typo_cot module source identity is unavailable")
        loaded_path = Path(loaded_origin).resolve()
        if loaded_path != Path(spec_origin).resolve():
            raise ValueError("loaded typo_cot source differs from its import specification")
        try:
            relative = loaded_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "typo_cot dependency source is outside the required checkout layout"
            ) from exc
        if relative != required_sources.get(module_name, relative):
            raise ValueError("required typo_cot dependency source path differs")
        relative_text = relative.as_posix()
        if (
            not relative.is_relative_to(Path(_TRACKED_TREES[1]))
            or relative_text not in tracked_by_tree[1]
        ):
            raise ValueError("typo_cot dependency source is not tracked by the attested checkout")
        if module_name in required_sources:
            dependency_sources.add(relative_text)
    if set(required_sources.values()) != {Path(value) for value in dependency_sources}:
        raise ValueError("kill runtime dependency module inventory is incomplete")
    return RuntimeCheckoutAttestation(
        revision=revision,
        typo_robust_training_tree=trees[0],
        typo_cot_tree=trees[1],
        typo_cot_runtime_sources=tuple(sorted(dependency_sources)),
    )


def _revision_candidates(value: Any) -> set[str]:
    result: set[str] = set()
    for candidate in (getattr(value, "_commit_hash", None),):
        if isinstance(candidate, str) and _REVISION.fullmatch(candidate):
            result.add(candidate)
    kwargs = getattr(value, "init_kwargs", None)
    if isinstance(kwargs, dict):
        candidate = kwargs.get("_commit_hash")
        if isinstance(candidate, str) and _REVISION.fullmatch(candidate):
            result.add(candidate)
    return result


def attest_loaded_revisions(model: Any, tokenizer: Any, expected: str) -> tuple[str, str]:
    """Independently bind the actually loaded model and tokenizer revisions."""

    model_candidates = _revision_candidates(getattr(model, "config", None))
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    model_candidates.update(_revision_candidates(text_config))
    tokenizer_candidates = _revision_candidates(tokenizer)
    if model_candidates != {expected}:
        raise ValueError("loaded model revision cannot be proven exact")
    if tokenizer_candidates != {expected}:
        raise ValueError("loaded tokenizer revision cannot be proven exact")
    return expected, expected


__all__ = [
    "RuntimeCheckoutAttestation",
    "attest_loaded_revisions",
    "attest_runtime_checkout",
]
