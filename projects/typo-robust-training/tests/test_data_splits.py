"""Leakage-resistant content and natural-repository split contracts."""

from __future__ import annotations

from collections import defaultdict

import pytest

from typo_robust_training.data.records import CleanRecord
from typo_robust_training.data.splits import (
    NearDuplicateTextIndex,
    _minhash_coefficients,
    _minhash_signature,
    assign_balanced_group_roles,
    assign_content_splits,
    assign_repository_split,
    cluster_near_duplicates,
    normalized_content_sha256,
    validate_group_disjointness,
)


def test_vectorized_minhash_matches_the_scalar_definition() -> None:
    shingles = frozenset({0, 1, 17, 2**32, 2**64 - 1})
    expected = tuple(
        min((multiplier * shingle + offset) & (2**64 - 1) for shingle in shingles)
        for multiplier, offset in _minhash_coefficients(32)
    )

    assert _minhash_signature(shingles) == expected


def _record(index: int, text: str, *, group: str | None = None) -> CleanRecord:
    return CleanRecord(
        source="fixture",
        source_revision="a" * 40,
        source_split="train",
        source_id=f"row-{index}",
        group_id=group or f"document-{index}",
        text=text,
        task=None,
        answer=None,
        metadata={"fixture": True},
    )


def test_content_hash_normalizes_only_for_duplicate_detection() -> None:
    assert normalized_content_sha256("  Alpha\n beta  ") == normalized_content_sha256("alpha beta")
    assert normalized_content_sha256("alpha beta") != normalized_content_sha256("alpha gamma")


def test_near_duplicate_clusters_are_order_independent_and_split_atomically() -> None:
    records = (
        _record(0, "The airport is located in Chicago and serves many passengers."),
        _record(1, "The airport is located in Chicago and serves many passenger."),
        _record(2, "Photosynthesis converts light energy into chemical energy."),
        _record(3, "A completely unrelated discussion of medieval history."),
    )
    forward = cluster_near_duplicates(records, shingle_size=3, threshold=0.80)
    reverse = cluster_near_duplicates(tuple(reversed(records)), shingle_size=3, threshold=0.80)
    assert forward == reverse
    assert forward["row-0"] == forward["row-1"]
    assert forward["row-0"] != forward["row-2"]

    assignments = assign_content_splits(
        records,
        clusters=forward,
        seed=42,
        weights={"train": 0.8, "tune": 0.1, "pre_pr_gate": 0.05, "final_test": 0.05},
    )
    assert assignments["row-0"] == assignments["row-1"]
    validate_group_disjointness(records, assignments, clusters=forward)

    leaked = dict(assignments)
    leaked["row-1"] = "final_test" if leaked["row-0"] != "final_test" else "train"
    with pytest.raises(ValueError, match="near-duplicate cluster"):
        validate_group_disjointness(records, leaked, clusters=forward)


def test_near_duplicate_text_index_reuses_the_clustering_rule() -> None:
    index = NearDuplicateTextIndex(
        ("The airport is located in Chicago and serves many passengers.",),
        shingle_size=3,
        threshold=0.80,
    )
    assert index.contains_near_duplicate(
        "The airport is located in Chicago and serves many passenger."
    )
    assert not index.contains_near_duplicate("A completely unrelated medieval discussion.")


def test_near_duplicate_text_index_does_not_retain_uncompressed_corpus_or_cache() -> None:
    text = "The airport is located in Chicago. " * 1_000
    index = NearDuplicateTextIndex((text,))

    assert not hasattr(index, "_texts")
    assert not hasattr(index, "_cached_shingles")
    assert len(index._compressed_normalized_texts[0]) < len(text.encode("utf-8")) // 10
    assert index.contains_near_duplicate(text)


def test_content_assignment_does_not_depend_on_input_order() -> None:
    records = tuple(
        _record(index, f"Document number {index} contains enough prose.") for index in range(40)
    )
    clusters = cluster_near_duplicates(records, shingle_size=3, threshold=0.95)
    expected = assign_content_splits(
        records,
        clusters=clusters,
        seed=42,
        weights={"train": 0.8, "tune": 0.1, "pre_pr_gate": 0.05, "final_test": 0.05},
    )
    observed = assign_content_splits(
        tuple(reversed(records)),
        clusters=clusters,
        seed=42,
        weights={"train": 0.8, "tune": 0.1, "pre_pr_gate": 0.05, "final_test": 0.05},
    )
    assert observed == expected


def test_natural_typo_repository_split_is_stable_and_repository_disjoint() -> None:
    repositories = tuple(f"https://github.com/example/repository-{index}" for index in range(500))
    assignments = {
        repository: assign_repository_split(repository, seed=42) for repository in repositories
    }
    assert assignments == {
        repository: assign_repository_split(repository, seed=42)
        for repository in reversed(repositories)
    }
    assert set(assignments.values()) == {"train", "tune", "held_out"}

    counts: dict[str, int] = defaultdict(int)
    for split in assignments.values():
        counts[split] += 1
    assert 0.60 <= counts["train"] / len(repositories) <= 0.80
    assert 0.04 <= counts["tune"] / len(repositories) <= 0.16
    assert 0.12 <= counts["held_out"] / len(repositories) <= 0.28


def test_balanced_group_roles_are_order_independent_and_keep_groups_atomic() -> None:
    sizes = {
        "repository-a": 51,
        "repository-b": 23,
        "repository-c": 17,
        "repository-d": 9,
        "repository-e": 5,
    }
    weights = {"train": 0.6, "tune": 0.1, "held_out": 0.3}
    expected = assign_balanced_group_roles(
        sizes,
        seed=42,
        namespace="fixture",
        weights=weights,
    )
    observed = assign_balanced_group_roles(
        dict(reversed(tuple(sizes.items()))),
        seed=42,
        namespace="fixture",
        weights=weights,
    )

    assert observed == expected
    assert set(observed) == set(sizes)
    assert set(observed.values()) == set(weights)


def test_balanced_group_role_does_not_fix_the_largest_group_to_one_role() -> None:
    sizes = {"largest": 100, "a": 20, "b": 19, "c": 18, "d": 17, "e": 16}
    roles = {
        assign_balanced_group_roles(
            sizes,
            seed=seed,
            namespace="size-independence-fixture",
            weights={"train": 0.55, "tune": 0.10, "held_out": 0.35},
        )["largest"]
        for seed in range(16)
    }
    assert len(roles) >= 2


def test_balanced_group_role_coverage_cannot_pin_a_giant_group_to_a_tiny_role() -> None:
    sizes = {"giant": 9_000, **{f"small-{index}": 10 for index in range(40)}}
    assignments = assign_balanced_group_roles(
        sizes,
        seed=42,
        namespace="skewed-coverage-fixture",
        weights={"train": 0.98, "tune": 0.01, "held_out": 0.01},
    )
    counts: dict[str, int] = defaultdict(int)
    for group, role in assignments.items():
        counts[role] += sizes[group]

    assert assignments["giant"] == "train"
    assert counts["tune"] < 0.05 * sum(sizes.values())
    assert counts["held_out"] < 0.05 * sum(sizes.values())


def test_balanced_group_role_coverage_uses_weights_when_all_groups_are_required() -> None:
    sizes = {"big": 1_000, "mid": 10, "small": 5}
    assignments = assign_balanced_group_roles(
        sizes,
        seed=42,
        namespace="small-inventory-fixture",
        weights={"train": 0.70, "tune": 0.10, "held_out": 0.20},
    )

    assert assignments["big"] == "train"
    assert set(assignments.values()) == {"train", "tune", "held_out"}
