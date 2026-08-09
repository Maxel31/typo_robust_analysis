"""Deterministic restoration plans for the eleven Table 13 conditions."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass

from typo_cot.experiments.restoration_order_accuracy.protocol import (
    ALL_CONDITION_IDS,
    INTERMEDIATE_BUDGETS,
    PAPER_ORDERS,
    EditGroup,
    canonical_sha256,
    restore_edit_groups,
)


@dataclass(frozen=True, slots=True)
class RestorationCondition:
    """One exact editable string at one order/budget arm."""

    condition_id: str
    order: str | None
    budget: int
    text: str
    restored_group_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "order": self.order,
            "budget": self.budget,
            "text": self.text,
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "restored_group_indices": list(self.restored_group_indices),
        }


@dataclass(frozen=True, slots=True)
class RestorationPlan:
    """One sample's frozen edit groups, order permutations, and arm texts."""

    sample_id: str
    clean_text: str
    edited_text: str
    groups: tuple[EditGroup, ...]
    order_indices: tuple[tuple[str, tuple[int, ...]], ...]
    conditions: tuple[RestorationCondition, ...]

    @property
    def realized_edit_count(self) -> int:
        return len(self.groups)

    @property
    def edited_endpoint(self) -> RestorationCondition:
        return self.conditions[0]

    @property
    def clean_endpoint(self) -> RestorationCondition:
        return self.conditions[-1]

    def condition_for(self, order: str, budget: int) -> RestorationCondition:
        if order not in PAPER_ORDERS:
            raise ValueError(f"unsupported restoration order: {order!r}")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError("restoration budget must be a non-negative integer")
        if budget == 0:
            return self.edited_endpoint
        if budget >= 4:
            return self.clean_endpoint
        condition_id = f"{order}:k{budget}"
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        raise ValueError(f"unsupported restoration budget: {budget}")

    def order_for(self, order: str) -> tuple[int, ...]:
        try:
            return dict(self.order_indices)[order]
        except KeyError:
            raise ValueError(f"unsupported restoration order: {order!r}") from None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "sample_id": self.sample_id,
            "realized_edit_count": self.realized_edit_count,
            "clean_text_sha256": hashlib.sha256(self.clean_text.encode()).hexdigest(),
            "edited_text_sha256": hashlib.sha256(self.edited_text.encode()).hexdigest(),
            "groups": [group.to_dict() for group in self.groups],
            "order_group_indices": {
                name: list(indices) for name, indices in self.order_indices
            },
            "conditions": [condition.to_dict() for condition in self.conditions],
        }
        return {**payload, "sha256": canonical_sha256(payload)}


def order_group_indices(
    groups: Sequence[EditGroup],
    *,
    order: str,
    sample_id: str,
    seed: int,
) -> tuple[int, ...]:
    """Return one fixed permutation under a named submitted-compatible order."""
    groups_tuple = tuple(groups)
    if not groups_tuple:
        raise ValueError("restoration ordering requires at least one edit group")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if order == "high-relevance-first":
        return tuple(
            group.index
            for group in sorted(groups_tuple, key=lambda group: (-abs(group.relevance), group.index))
        )
    if order == "low-relevance-first":
        return tuple(
            group.index
            for group in sorted(groups_tuple, key=lambda group: (abs(group.relevance), group.index))
        )
    if order == "seeded-random":
        digest = hashlib.md5(  # noqa: S324 - frozen non-security paper compatibility
            f"{seed}:{sample_id}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest, 16))
        indices = [group.index for group in groups_tuple]
        rng.shuffle(indices)
        return tuple(indices)
    raise ValueError(f"unsupported restoration order: {order!r}")


def build_restoration_plan(
    *,
    sample_id: str,
    clean_text: str,
    edited_text: str,
    groups: Sequence[EditGroup],
    seed: int,
) -> RestorationPlan:
    """Build two shared endpoints and three-by-three intermediate arms."""
    groups_tuple = tuple(groups)
    if not 1 <= len(groups_tuple) <= 4:
        raise ValueError("restoration plan requires between one and four edit groups")
    if tuple(group.index for group in groups_tuple) != tuple(range(len(groups_tuple))):
        raise ValueError("edit group indices must be contiguous and left-to-right")
    orders = tuple(
        (
            order,
            order_group_indices(
                groups_tuple,
                order=order,
                sample_id=sample_id,
                seed=seed,
            ),
        )
        for order in PAPER_ORDERS
    )
    edited_endpoint = RestorationCondition(
        condition_id="edited:k0",
        order=None,
        budget=0,
        text=restore_edit_groups(clean_text, edited_text, groups_tuple, ()),
        restored_group_indices=(),
    )
    intermediates: list[RestorationCondition] = []
    for order, permutation in orders:
        for budget in INTERMEDIATE_BUDGETS:
            restored = (
                tuple(range(len(groups_tuple)))
                if budget >= len(groups_tuple)
                else permutation[:budget]
            )
            intermediates.append(
                RestorationCondition(
                    condition_id=f"{order}:k{budget}",
                    order=order,
                    budget=budget,
                    text=restore_edit_groups(
                        clean_text,
                        edited_text,
                        groups_tuple,
                        restored,
                    ),
                    restored_group_indices=restored,
                )
            )
    clean_endpoint = RestorationCondition(
        condition_id="clean:k4",
        order=None,
        budget=4,
        text=restore_edit_groups(
            clean_text,
            edited_text,
            groups_tuple,
            tuple(range(len(groups_tuple))),
        ),
        restored_group_indices=tuple(range(len(groups_tuple))),
    )
    conditions = (edited_endpoint, *intermediates, clean_endpoint)
    if tuple(condition.condition_id for condition in conditions) != ALL_CONDITION_IDS:
        raise AssertionError("restoration condition order differs from the protocol")
    if edited_endpoint.text != edited_text or clean_endpoint.text != clean_text:
        raise ValueError("restoration plan does not reproduce exact endpoints")
    return RestorationPlan(
        sample_id=sample_id,
        clean_text=clean_text,
        edited_text=edited_text,
        groups=groups_tuple,
        order_indices=orders,
        conditions=conditions,
    )


__all__ = [
    "RestorationCondition",
    "RestorationPlan",
    "build_restoration_plan",
    "order_group_indices",
]
