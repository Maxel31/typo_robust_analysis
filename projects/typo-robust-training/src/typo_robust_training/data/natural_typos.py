"""GitHub Typo Corpus parsing with repository-level approval and orientation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from typo_robust_training.data.perturb import TRAINING_OPERATIONS, classify_character_edit
from typo_robust_training.data.records import NaturalTypoRecord


GITHUB_TYPO_CORPUS_REVISION = "240c4c0716fde62934edc2445c326e0896d8d109"


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def parse_github_typo_commit(
    payload: Mapping[str, object],
    *,
    approved_repositories: Mapping[str, str],
    source_revision: str = GITHUB_TYPO_CORPUS_REVISION,
) -> tuple[NaturalTypoRecord, ...]:
    """Parse English typo corrections without redistributing unapproved repositories.

    Corpus `src` is the text before a fixing commit (the typo) and `tgt` is the
    corrected text (the clean counterpart). Only repositories with an explicit
    local license decision are emitted.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("GitHub typo commit must be an object")
    if not isinstance(approved_repositories, Mapping):
        raise TypeError("approved_repositories must be an object")
    repository = payload.get("repo")
    commit = payload.get("commit")
    edits = payload.get("edits")
    if (
        not isinstance(repository, str)
        or not repository
        or not isinstance(commit, str)
        or not commit
        or not isinstance(edits, list)
    ):
        return ()
    license_name = approved_repositories.get(repository)
    if not isinstance(license_name, str) or not license_name:
        return ()
    records: list[NaturalTypoRecord] = []
    for index, raw_edit in enumerate(edits):
        edit = _mapping(raw_edit)
        if edit is None or edit.get("is_typo") is not True:
            continue
        source = _mapping(edit.get("src"))
        target = _mapping(edit.get("tgt"))
        if source is None or target is None:
            continue
        typo_text, clean_text = source.get("text"), target.get("text")
        if (
            source.get("lang") != "eng"
            or target.get("lang") != "eng"
            or not isinstance(typo_text, str)
            or not typo_text
            or not isinstance(clean_text, str)
            or not clean_text
        ):
            continue
        operation = classify_character_edit(clean=clean_text, typo=typo_text)
        if operation is None:
            continue
        path = target.get("path") if isinstance(target.get("path"), str) else source.get("path")
        source_id = f"{commit}:{index}"
        digest = hashlib.sha256(
            f"{repository}\0{source_id}\0{clean_text}\0{typo_text}".encode("utf-8")
        ).hexdigest()
        probability = edit.get("prob_typo")
        records.append(
            NaturalTypoRecord(
                source="github_typo_corpus",
                source_revision=source_revision,
                source_split="v1.0.0",
                source_id=f"{source_id}:{digest[:16]}",
                group_id=repository,
                clean_text=clean_text,
                typo_text=typo_text,
                repository=repository,
                repository_license=license_name,
                operation=operation,
                training_eligible=operation in TRAINING_OPERATIONS,
                metadata={
                    "commit": commit,
                    "path": path,
                    "prob_typo": probability,
                    "corpus_orientation": "src-typo-to-tgt-clean/v1",
                },
            )
        )
    return tuple(records)


__all__ = ["GITHUB_TYPO_CORPUS_REVISION", "parse_github_typo_commit"]
