"""Fail-closed source and model identity attestation for GPU diagnostics."""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_REVISION = re.compile(r"[0-9a-f]{40}")
_TRACKED_TREES = (
    "projects/typo-robust-training/src/typo_robust_training",
    "projects/typo-cot/src/typo_cot",
)


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

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def attest_runtime_checkout(expected_revision: str) -> RuntimeCheckoutAttestation:
    """Attest HEAD and both complete runtime source trees before CUDA/output.

    Staged, unstaged, and untracked files under either package are rejected.
    The git tree object IDs bind every tracked path and byte in the two trees.
    """

    if not isinstance(expected_revision, str) or _REVISION.fullmatch(expected_revision) is None:
        raise ValueError("kill runtime revision must be one pinned git revision")
    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel", cwd=module.parent)).resolve()
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
    for relative in _TRACKED_TREES:
        tracked = _git("ls-files", "--", relative, cwd=root).splitlines()
        if not tracked or any(not (root / item).is_file() for item in tracked):
            raise ValueError("kill runtime source tree is not fully tracked")
        tree = _git("rev-parse", f"HEAD:{relative}", cwd=root)
        if _REVISION.fullmatch(tree) is None:
            raise ValueError("kill runtime source tree identity is unavailable")
        trees.append(tree)
    return RuntimeCheckoutAttestation(
        revision=revision,
        typo_robust_training_tree=trees[0],
        typo_cot_tree=trees[1],
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
