"""Clean-only source selection and identity binding for SAE training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.training.pairs import TrainingSource, stable_epoch_sources


@dataclass(frozen=True, slots=True)
class PreparedSaeSources:
    sources: tuple[TrainingSource, ...]
    reserved: tuple[TrainingSource, ...]
    input_paths: tuple[Path, ...]
    input_sha256: tuple[str, ...]
    input_record_ids: frozenset[str]
    input_source_ids: frozenset[str]
    input_group_ids: frozenset[str]
    record_id_sha256: str
    source_stream_sha256: str
    source_tokens: int
    protected_eligible_records: int
    protected_eligible_source_tokens: int
    protected_eligible_record_ids_sha256: str
    protected_normalized_duplicates_removed: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clean_fineweb_sources(path: Path) -> tuple[TrainingSource, ...]:
    rows: list[TrainingSource] = []
    for line_number, line in read_lf_jsonl_lines(Path(path), context="SAE training data"):
        payload = strict_loads(line, context=f"{path}:{line_number}")
        if not isinstance(payload, dict):
            raise ValueError("SAE training source must be an object")
        if payload.get("kind") != "clean":
            raise ValueError("SAE training data must contain clean records only")
        if payload.get("source") != "fineweb_edu":
            raise ValueError("SAE training data must contain FineWeb-Edu only")
        if (
            payload.get("split") != "train"
            or payload.get("source_split") != "train"
            or payload.get("task") is not None
            or payload.get("answer") is not None
        ):
            raise ValueError("SAE training data must be task-free train records")
        rows.append(TrainingSource.from_dict(payload))
    if not rows:
        raise ValueError("SAE training data contains no records")
    if len({row.record_id for row in rows}) != len(rows):
        raise ValueError("SAE training record IDs are duplicated")
    return tuple(rows)


def reserve_confirmatory_training_prefix(
    sources: Sequence[TrainingSource],
    *,
    seed: int,
    epoch: int,
    reserved_records: int,
) -> tuple[tuple[TrainingSource, ...], tuple[TrainingSource, ...]]:
    ordered = stable_epoch_sources(tuple(sources), seed=seed, epoch=epoch)
    if (
        isinstance(reserved_records, bool)
        or not isinstance(reserved_records, int)
        or reserved_records <= 0
        or reserved_records >= len(ordered)
    ):
        raise ValueError("SAE reserved record count is outside the source manifest")
    return ordered[:reserved_records], ordered[reserved_records:]


def record_id_sha256(sources: Sequence[TrainingSource]) -> str:
    rows = tuple(sources)
    if not rows or len({row.record_id for row in rows}) != len(rows):
        raise ValueError("SAE record-ID digest requires unique non-empty sources")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.record_id.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_stream_sha256(
    *,
    reserved: Sequence[TrainingSource],
    eligible: Sequence[TrainingSource],
) -> str:
    """Hash the exact in-memory source values, roles, and training order.

    Raw manifest hashes identify the bytes that were opened.  This second
    digest identifies what parsing actually placed in memory, so a retry claim
    cannot silently bind stale file hashes after a hash/read race.
    """

    digest = hashlib.sha256(b"robustness-sae-loaded-source-stream/v1\0")
    for role, sources in (("reserved", reserved), ("eligible", eligible)):
        for index, source in enumerate(sources):
            if not isinstance(source, TrainingSource):
                raise TypeError("SAE source-stream digest requires TrainingSource values")
            values = {field.name: getattr(source, field.name) for field in fields(source)}
            values["metadata"] = dict(source.metadata)
            encoded = json.dumps(
                {"index": index, "role": role, "source": values},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _sae_order(sources: Sequence[TrainingSource]) -> tuple[TrainingSource, ...]:
    def key(source: TrainingSource) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"sae-clean-source-order/v1\0{source.record_id}".encode()
        ).hexdigest()
        return digest, source.record_id

    return tuple(sorted(sources, key=key))


def _manifest_identities(
    path: Path,
    *,
    allow_duplicate_normalized_content: bool = False,
) -> tuple[set[str], set[str], set[str], set[str]]:
    record_ids: set[str] = set()
    source_ids: set[str] = set()
    group_ids: set[str] = set()
    normalized_hashes: set[str] = set()
    for line_number, line in read_lf_jsonl_lines(path, context="SAE identity manifest"):
        payload = strict_loads(line, context=f"{path}:{line_number}")
        if not isinstance(payload, dict):
            raise ValueError("SAE identity record must be an object")
        values = (
            payload.get("record_id"),
            payload.get("source_id"),
            payload.get("group_id"),
            payload.get("normalized_content_sha256"),
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("SAE identity record is incomplete")
        for value, inventory, name in zip(
            values,
            (record_ids, source_ids, group_ids, normalized_hashes),
            ("record ID", "source ID", "group ID", "normalized content"),
            strict=True,
        ):
            if value in inventory:
                if name == "normalized content" and allow_duplicate_normalized_content:
                    continue
                raise ValueError(f"SAE manifest duplicates {name}")
            inventory.add(value)
    return record_ids, source_ids, group_ids, normalized_hashes


def _deduplicate_protected_eligible(
    reserved: Sequence[TrainingSource],
    eligible: Sequence[TrainingSource],
) -> tuple[TrainingSource, ...]:
    """Keep one deterministic eligible row per normalized clean document.

    Every reserved row remains an exclusion anchor, including duplicate text.  An
    eligible row is retained only when its normalized content occurs in neither
    the protected prefix nor an earlier row in the SAE-specific stable order.
    """

    seen = {normalized_content_sha256(source.clean_text) for source in reserved}
    retained: list[TrainingSource] = []
    for source in _sae_order(eligible):
        normalized_hash = normalized_content_sha256(source.clean_text)
        if normalized_hash in seen:
            continue
        seen.add(normalized_hash)
        retained.append(source)
    return tuple(retained)


def prepare_sae_sources(
    paths: Sequence[Path],
    *,
    protected_manifest_sha256: str,
    reserved_seed: int,
    reserved_epoch: int,
    reserved_records: int,
) -> PreparedSaeSources:
    resolved_paths = tuple(Path(path).resolve() for path in paths)
    if not resolved_paths:
        raise ValueError("SAE training requires at least one clean manifest")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("SAE training manifest paths are duplicated")
    hashes = tuple(sha256_file(path) for path in resolved_paths)
    if len(set(hashes)) != len(hashes):
        raise ValueError("SAE training manifest contents are duplicated")
    if hashes.count(protected_manifest_sha256) != 1:
        raise ValueError("SAE inputs must contain the one preregistered protected-run manifest")

    all_sources: list[TrainingSource] = []
    reserved: tuple[TrainingSource, ...] = ()
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    seen_group_ids: set[str] = set()
    seen_normalized: set[str] = set()
    protected_eligible: tuple[TrainingSource, ...] | None = None
    protected_duplicates_removed: int | None = None
    for path, digest in zip(resolved_paths, hashes, strict=True):
        rows = load_clean_fineweb_sources(path)
        is_protected = digest == protected_manifest_sha256
        identities = _manifest_identities(
            path,
            allow_duplicate_normalized_content=is_protected,
        )
        for observed, prior, name in zip(
            identities,
            (seen_record_ids, seen_source_ids, seen_group_ids, seen_normalized),
            ("record ID", "source ID", "group ID", "normalized content"),
            strict=True,
        ):
            overlap = observed & prior
            if overlap:
                raise ValueError(f"SAE manifests overlap by {name}")
            prior.update(observed)
        if is_protected:
            reserved, eligible = reserve_confirmatory_training_prefix(
                rows,
                seed=reserved_seed,
                epoch=reserved_epoch,
                reserved_records=reserved_records,
            )
            protected_eligible = _deduplicate_protected_eligible(reserved, eligible)
            protected_duplicates_removed = len(eligible) - len(protected_eligible)
            all_sources.extend(protected_eligible)
        else:
            all_sources.extend(rows)
    if protected_eligible is None or protected_duplicates_removed is None:
        raise AssertionError("protected SAE manifest was not prepared")
    ordered = _sae_order(all_sources)
    return PreparedSaeSources(
        sources=ordered,
        reserved=reserved,
        input_paths=resolved_paths,
        input_sha256=hashes,
        input_record_ids=frozenset(seen_record_ids),
        input_source_ids=frozenset(seen_source_ids),
        input_group_ids=frozenset(seen_group_ids),
        record_id_sha256=record_id_sha256(ordered),
        source_stream_sha256=source_stream_sha256(
            reserved=reserved,
            eligible=ordered,
        ),
        source_tokens=sum(row.token_count for row in ordered),
        protected_eligible_records=len(protected_eligible),
        protected_eligible_source_tokens=sum(row.token_count for row in protected_eligible),
        protected_eligible_record_ids_sha256=record_id_sha256(protected_eligible),
        protected_normalized_duplicates_removed=protected_duplicates_removed,
    )


def canonical_record_registry(
    *,
    input_paths: Sequence[Path],
    input_sha256: Sequence[str],
    reserved: Sequence[TrainingSource],
    eligible: Sequence[TrainingSource],
    protected_normalized_duplicates_removed: int,
) -> dict[str, object]:
    if len(input_paths) != len(input_sha256):
        raise ValueError("SAE input paths and hashes differ")
    return {
        "schema_version": "robustness-sae-source-registry/v1",
        "inputs": [
            {"path": str(Path(path).resolve()), "sha256": digest}
            for path, digest in zip(input_paths, input_sha256, strict=True)
        ],
        "reserved_records": len(reserved),
        "reserved_record_ids_sha256": record_id_sha256(reserved),
        "eligible_records": len(eligible),
        "eligible_source_tokens": sum(row.token_count for row in eligible),
        "eligible_record_ids_sha256": record_id_sha256(eligible),
        "protected_normalized_duplicates_removed": (protected_normalized_duplicates_removed),
    }


def write_json_atomic(path: Path, payload: object) -> None:
    import os

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PreparedSaeSources",
    "canonical_record_registry",
    "load_clean_fineweb_sources",
    "record_id_sha256",
    "source_stream_sha256",
    "prepare_sae_sources",
    "reserve_confirmatory_training_prefix",
    "sha256_file",
    "write_json_atomic",
]
