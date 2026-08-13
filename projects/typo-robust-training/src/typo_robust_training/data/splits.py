"""Stable group-level splits and near-duplicate clustering."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cache

from typo_robust_training.data.records import CleanRecord


_WHITESPACE = re.compile(r"\s+")
_HASH_MASK = (1 << 64) - 1
_SHINGLE_BASE = 257


@cache
def _minhash_coefficients(components: int) -> tuple[tuple[int, int], ...]:
    coefficients: list[tuple[int, int]] = []
    for component in range(components):
        digest = hashlib.blake2b(component.to_bytes(2, "big"), digest_size=16).digest()
        multiplier = int.from_bytes(digest[:8], "big") | 1
        offset = int.from_bytes(digest[8:], "big")
        coefficients.append((multiplier, offset))
    return tuple(coefficients)


def normalize_for_duplicate_detection(text: str) -> str:
    """Normalize case/space only; source text itself is never rewritten."""

    if not isinstance(text, str):
        raise TypeError("duplicate-detection text must be a string")
    return _WHITESPACE.sub(" ", text.casefold()).strip()


def normalized_content_sha256(text: str) -> str:
    return hashlib.sha256(normalize_for_duplicate_detection(text).encode("utf-8")).hexdigest()


def _shingles(text: str, size: int) -> frozenset[int]:
    normalized = normalize_for_duplicate_detection(text)
    if not normalized:
        return frozenset()
    width = min(size, len(normalized))
    high = pow(_SHINGLE_BASE, width - 1, 1 << 64)
    value = 0
    for character in normalized[:width]:
        value = (value * _SHINGLE_BASE + ord(character) + 1) & _HASH_MASK
    hashes = {value}
    for index in range(width, len(normalized)):
        outgoing = ord(normalized[index - width]) + 1
        incoming = ord(normalized[index]) + 1
        value = ((value - outgoing * high) * _SHINGLE_BASE + incoming) & _HASH_MASK
        hashes.add(value)
    return frozenset(hashes)


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _minhash_signature(shingles: frozenset[int], *, components: int = 32) -> tuple[int, ...]:
    if not shingles:
        return (0,) * components
    return tuple(
        min((multiplier * shingle + offset) & _HASH_MASK for shingle in shingles)
        for multiplier, offset in _minhash_coefficients(components)
    )


def cluster_near_duplicates(
    records: Sequence[CleanRecord],
    *,
    shingle_size: int = 5,
    threshold: float = 0.90,
) -> dict[str, str]:
    """Cluster exact/near duplicates with deterministic MinHash candidate bands."""

    if isinstance(shingle_size, bool) or not isinstance(shingle_size, int) or shingle_size <= 0:
        raise ValueError("shingle_size must be a positive integer")
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    ordered = sorted(records, key=lambda record: record.source_id)
    identifiers = [record.source_id for record in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("near-duplicate source IDs must be unique")
    union = _UnionFind(identifiers)
    text_by_id = {record.source_id: record.text for record in ordered}

    exact: dict[str, str] = {}
    for record in ordered:
        digest = normalized_content_sha256(record.text)
        previous = exact.setdefault(digest, record.source_id)
        union.union(previous, record.source_id)

    candidate_buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for record in ordered:
        signature = _minhash_signature(_shingles(record.text, shingle_size))
        for band_index in range(8):
            start = band_index * 4
            candidate_buckets[(band_index, signature[start : start + 4])].append(record.source_id)

    candidates: set[tuple[str, str]] = set()
    for bucket in candidate_buckets.values():
        unique = sorted(set(bucket))
        for left_index, left in enumerate(unique):
            for right in unique[left_index + 1 :]:
                candidates.add((left, right))
    candidate_shingles: dict[str, frozenset[int]] = {}
    for left, right in sorted(candidates):
        if union.find(left) == union.find(right):
            continue
        if left not in candidate_shingles:
            candidate_shingles[left] = _shingles(text_by_id[left], shingle_size)
        if right not in candidate_shingles:
            candidate_shingles[right] = _shingles(text_by_id[right], shingle_size)
        left_shingles = candidate_shingles[left]
        right_shingles = candidate_shingles[right]
        if _jaccard(left_shingles, right_shingles) >= float(threshold):
            union.union(left, right)
    return {identifier: union.find(identifier) for identifier in sorted(identifiers)}


def _validate_weights(weights: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("split weights must be a non-empty object")
    result: list[tuple[str, float]] = []
    total = 0.0
    for name, weight in weights.items():
        if not isinstance(name, str) or not name:
            raise ValueError("split names must be non-empty strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or float(weight) < 0.0:
            raise ValueError("split weights must be non-negative numbers")
        total += float(weight)
        result.append((name, float(weight)))
    if abs(total - 1.0) > 1e-12:
        raise ValueError("split weights must sum to one")
    return tuple(result)


def stable_weighted_split(
    key: str,
    *,
    seed: int,
    namespace: str,
    weights: Mapping[str, float],
) -> str:
    """Map a stable group key to a split without consuming shared RNG state."""

    if not isinstance(key, str) or not key:
        raise ValueError("split key must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a non-negative integer")
    ordered = _validate_weights(weights)
    digest = hashlib.sha256(f"{namespace}\0{seed}\0{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest, "big") / (1 << 256)
    cumulative = 0.0
    for name, weight in ordered:
        cumulative += weight
        if value < cumulative:
            return name
    return ordered[-1][0]


def assign_content_splits(
    records: Sequence[CleanRecord],
    *,
    clusters: Mapping[str, str],
    seed: int,
    weights: Mapping[str, float],
) -> dict[str, str]:
    """Assign source groups and near-duplicate clusters atomically."""

    ordered = sorted(records, key=lambda record: record.source_id)
    identifiers = [record.source_id for record in ordered]
    if set(clusters) != set(identifiers):
        raise ValueError("near-duplicate cluster inventory differs from records")
    union = _UnionFind(identifiers)
    by_group: dict[str, list[str]] = defaultdict(list)
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for record in ordered:
        by_group[record.group_id].append(record.source_id)
        by_cluster[clusters[record.source_id]].append(record.source_id)
    for members in (*by_group.values(), *by_cluster.values()):
        anchor = min(members)
        for member in members:
            union.union(anchor, member)
    atomic_members: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        atomic_members[union.find(identifier)].append(identifier)
    assignments: dict[str, str] = {}
    for root, members in sorted(atomic_members.items()):
        stable_key = "\0".join(sorted(members))
        split = stable_weighted_split(
            stable_key,
            seed=seed,
            namespace="robustness-content-split/v1",
            weights=weights,
        )
        for member in members:
            assignments[member] = split
    return dict(sorted(assignments.items()))


def validate_group_disjointness(
    records: Sequence[CleanRecord],
    assignments: Mapping[str, str],
    *,
    clusters: Mapping[str, str],
) -> None:
    if set(assignments) != {record.source_id for record in records}:
        raise ValueError("content split assignment inventory differs")
    groups: dict[str, set[str]] = defaultdict(set)
    near_duplicates: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record.group_id].add(assignments[record.source_id])
        near_duplicates[clusters[record.source_id]].add(assignments[record.source_id])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("source group crosses content splits")
    if any(len(splits) != 1 for splits in near_duplicates.values()):
        raise ValueError("near-duplicate cluster crosses content splits")


def assign_repository_split(
    repository: str,
    *,
    seed: int,
    weights: Mapping[str, float] | None = None,
) -> str:
    return stable_weighted_split(
        repository,
        seed=seed,
        namespace="github-typo-repository-split/v1",
        weights=weights or {"train": 0.70, "tune": 0.10, "held_out": 0.20},
    )


__all__ = [
    "assign_content_splits",
    "assign_repository_split",
    "cluster_near_duplicates",
    "normalize_for_duplicate_detection",
    "normalized_content_sha256",
    "stable_weighted_split",
    "validate_group_disjointness",
]
