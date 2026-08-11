"""Deterministic matched non-self donor assignment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def _string_tuple(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field} must be a non-empty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must contain only non-empty strings")
    return values


@dataclass(frozen=True, slots=True)
class MatchedDonorCandidate:
    """One recipient identity and the exact stratum its donor must match."""

    stratum: tuple[str, ...]
    identity: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stratum", _string_tuple(self.stratum, field="stratum"))
        object.__setattr__(self, "identity", _string_tuple(self.identity, field="identity"))


@dataclass(frozen=True, slots=True)
class CyclicDonorPlan:
    """Canonical cyclic assignments plus recipients in singleton strata."""

    assignments: tuple[
        tuple[tuple[str, ...], tuple[str, ...]],
        ...,
    ]
    singleton_identities: tuple[tuple[str, ...], ...]

    def as_dict(self) -> Mapping[tuple[str, ...], tuple[str, ...]]:
        """Materialize assignments for lookup without changing canonical order."""

        return dict(self.assignments)


def plan_cyclic_derangement(
    candidates: Iterable[MatchedDonorCandidate],
    *,
    reject_singletons: bool,
) -> CyclicDonorPlan:
    """Map sorted identities to the next member of their exact stratum.

    The result is independent of input iteration order. Recipient identities
    must be globally unique, which prevents an accidental assignment from being
    overwritten when two source collections are combined.
    """

    groups: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if not isinstance(candidate, MatchedDonorCandidate):
            raise TypeError("donor candidates must be MatchedDonorCandidate instances")
        if candidate.identity in seen:
            raise ValueError(f"duplicate donor recipient identity: {candidate.identity!r}")
        seen.add(candidate.identity)
        groups[candidate.stratum].append(candidate.identity)

    assignments: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    singletons: list[tuple[str, ...]] = []
    for stratum, identities in sorted(groups.items()):
        identities.sort()
        if len(identities) == 1:
            identity = identities[0]
            if reject_singletons:
                raise ValueError(
                    "matched donor stratum is singleton: "
                    f"stratum={stratum!r}, identity={identity!r}"
                )
            singletons.append(identity)
            continue
        for index, identity in enumerate(identities):
            assignments.append((identity, identities[(index + 1) % len(identities)]))

    assignments.sort()
    singletons.sort()
    return CyclicDonorPlan(tuple(assignments), tuple(singletons))


__all__ = [
    "CyclicDonorPlan",
    "MatchedDonorCandidate",
    "plan_cyclic_derangement",
]
