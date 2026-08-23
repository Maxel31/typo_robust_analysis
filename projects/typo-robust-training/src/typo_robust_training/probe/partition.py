"""Deterministic, disjoint scientific fit partitions for linear probes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


FIT_PARTITION_RULE = "class-stratified-record-id-sha256-balanced-halves/v1"


class FitPartitionRecord(Protocol):
    """Minimal immutable record identity required by the partition rule."""

    record_id: str
    class_id: int


@dataclass(frozen=True, slots=True)
class ProbeFitPartition:
    """One disjoint half of every word-identity class."""

    seed: int
    indices: tuple[int, ...]
    record_ids: tuple[str, ...]
    class_counts: tuple[tuple[int, int], ...]
    identity_sha256: str


def _ordering_key(record_id: str) -> tuple[bytes, str]:
    material = f"{FIT_PARTITION_RULE}\0{record_id}".encode()
    return hashlib.sha256(material).digest(), record_id


def _identity_digest(
    *,
    seed: int,
    members: Sequence[tuple[int, str]],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{FIT_PARTITION_RULE}\0seed={seed}\0".encode())
    for class_id, record_id in sorted(members):
        digest.update(f"class={class_id}\0record={record_id}\0".encode())
    return digest.hexdigest()


def build_probe_fit_partitions(
    records: Sequence[FitPartitionRecord],
    *,
    seeds: tuple[int, int],
) -> dict[int, ProbeFitPartition]:
    """Split each class into exact hash-ordered halves, one per scientific fit.

    The rule is independent of manifest row order.  It deliberately uses no
    model output, task accuracy, or validation statistic.
    """

    if len(seeds) != 2 or len(set(seeds)) != 2 or tuple(sorted(seeds)) != seeds:
        raise ValueError("probe fit partitions require two sorted distinct seeds")
    by_class: dict[int, list[tuple[int, FitPartitionRecord]]] = defaultdict(list)
    record_ids: set[str] = set()
    for index, record in enumerate(records):
        if record.record_id in record_ids:
            raise ValueError("probe fit partition record ids must be unique")
        record_ids.add(record.record_id)
        by_class[record.class_id].append((index, record))
    if not by_class:
        raise ValueError("probe fit partitions require at least one class")

    members_by_seed: dict[int, list[tuple[int, FitPartitionRecord]]] = {seed: [] for seed in seeds}
    for class_id, members in sorted(by_class.items()):
        if len(members) < 2 or len(members) % 2 != 0:
            raise ValueError(
                f"probe fit class {class_id} must contain an even number of at least two records"
            )
        ordered = sorted(members, key=lambda item: _ordering_key(item[1].record_id))
        half = len(ordered) // 2
        members_by_seed[seeds[0]].extend(ordered[:half])
        members_by_seed[seeds[1]].extend(ordered[half:])

    partitions: dict[int, ProbeFitPartition] = {}
    for seed in seeds:
        members = sorted(members_by_seed[seed], key=lambda item: item[0])
        class_counts: dict[int, int] = defaultdict(int)
        identities: list[tuple[int, str]] = []
        for _index, record in members:
            class_counts[record.class_id] += 1
            identities.append((record.class_id, record.record_id))
        partitions[seed] = ProbeFitPartition(
            seed=seed,
            indices=tuple(index for index, _record in members),
            record_ids=tuple(record.record_id for _index, record in members),
            class_counts=tuple(sorted(class_counts.items())),
            identity_sha256=_identity_digest(seed=seed, members=identities),
        )

    left_ids = set(partitions[seeds[0]].record_ids)
    right_ids = set(partitions[seeds[1]].record_ids)
    if left_ids & right_ids or left_ids | right_ids != record_ids:
        raise AssertionError("probe fit partition construction violated exact disjoint coverage")
    if partitions[seeds[0]].class_counts != partitions[seeds[1]].class_counts:
        raise AssertionError("probe fit partition construction violated class balance")
    if partitions[seeds[0]].identity_sha256 == partitions[seeds[1]].identity_sha256:
        raise AssertionError("probe fit partitions must have distinct identities")
    return partitions


__all__ = [
    "FIT_PARTITION_RULE",
    "ProbeFitPartition",
    "build_probe_fit_partitions",
]
