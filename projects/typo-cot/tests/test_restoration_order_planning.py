"""Frozen ordering and shared-endpoint plans for the Table 13 producer."""

from __future__ import annotations

import pytest

from typo_cot.experiments.restoration_order_accuracy.planning import (
    build_restoration_plan,
    order_group_indices,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    ALL_CONDITION_IDS,
    PAPER_ORDERS,
    build_edit_groups,
)


def _attempt(token_index: int, relevance: float) -> dict[str, object]:
    return {
        "selection_rank": token_index,
        "target_token_index": token_index,
        "relevance": relevance,
    }


def _four_groups(relevances: tuple[float, float, float, float]):
    return build_edit_groups(
        "aa bb cc dd ee",
        "ax bx cx dx ee",
        tuple(
            _attempt(token_index, relevance)
            for token_index, relevance in zip((10, 20, 30, 40), relevances, strict=True)
        ),
    )


def test_high_and_low_orders_use_absolute_relevance_with_left_to_right_ties() -> None:
    groups = _four_groups((3.0, -5.0, 5.0, -1.0))

    high = order_group_indices(
        groups,
        order="high-relevance-first",
        sample_id="sample-42",
        seed=42,
    )
    low = order_group_indices(
        groups,
        order="low-relevance-first",
        sample_id="sample-42",
        seed=42,
    )

    assert high == (1, 2, 0, 3)
    assert low == (3, 0, 1, 2)


@pytest.mark.parametrize(
    ("seed", "sample_id", "expected"),
    [
        (42, "sample-42", (1, 3, 0, 2)),
        (7, "sample-42", (0, 2, 1, 3)),
        (42, "mmlu_test_x", (0, 2, 3, 1)),
    ],
)
def test_seeded_random_order_is_exactly_legacy_md5_python_random_compatible(
    seed: int,
    sample_id: str,
    expected: tuple[int, ...],
) -> None:
    groups = _four_groups((4.0, 3.0, 2.0, 1.0))

    actual = order_group_indices(
        groups,
        order="seeded-random",
        sample_id=sample_id,
        seed=seed,
    )

    # Frozen from random.Random(int(md5(f"{seed}:{sample_id}"), 16)).shuffle.
    assert actual == expected


def test_four_edit_plan_has_shared_endpoints_and_nine_intermediate_conditions() -> None:
    clean = "aa bb cc dd ee"
    edited = "ax bx cx dx ee"
    groups = _four_groups((4.0, 3.0, 2.0, 1.0))

    plan = build_restoration_plan(
        sample_id="sample-42",
        clean_text=clean,
        edited_text=edited,
        groups=groups,
        seed=42,
    )

    assert PAPER_ORDERS == (
        "high-relevance-first",
        "seeded-random",
        "low-relevance-first",
    )
    assert plan.realized_edit_count == 4
    assert len(plan.conditions) == 11
    assert tuple(condition.condition_id for condition in plan.conditions) == ALL_CONDITION_IDS
    for order in PAPER_ORDERS:
        assert plan.condition_for(order, 0) is plan.edited_endpoint
        assert plan.condition_for(order, 4) is plan.clean_endpoint
        assert plan.condition_for(order, 5) is plan.clean_endpoint
    assert plan.edited_endpoint.text == edited
    assert plan.edited_endpoint.restored_group_indices == ()
    assert plan.clean_endpoint.text == clean
    assert plan.clean_endpoint.restored_group_indices == (0, 1, 2, 3)
    assert plan.condition_for("high-relevance-first", 1).text == "aa bx cx dx ee"
    assert plan.condition_for("high-relevance-first", 2).text == "aa bb cx dx ee"
    assert plan.condition_for("low-relevance-first", 1).text == "ax bx cx dd ee"


def test_budget_at_or_above_realized_count_is_the_one_shared_clean_endpoint() -> None:
    clean = "aa bb cc"
    edited = "ax bx cc"
    groups = build_edit_groups(
        clean,
        edited,
        (_attempt(10, 2.0), _attempt(20, 1.0)),
    )
    plan = build_restoration_plan(
        sample_id="archived-two-edits",
        clean_text=clean,
        edited_text=edited,
        groups=groups,
        seed=42,
    )

    assert plan.realized_edit_count == 2
    assert len(plan.conditions) == 11
    assert tuple(condition.condition_id for condition in plan.conditions) == ALL_CONDITION_IDS
    for order in PAPER_ORDERS:
        for budget in (2, 3):
            condition = plan.condition_for(order, budget)
            assert condition.text == clean
            assert condition.restored_group_indices == (0, 1)
        assert plan.condition_for(order, 4) is plan.clean_endpoint
        assert plan.condition_for(order, 100) is plan.clean_endpoint


def test_seeded_random_budget_sets_are_nested_prefixes_of_one_permutation() -> None:
    groups = _four_groups((4.0, 3.0, 2.0, 1.0))
    plan = build_restoration_plan(
        sample_id="sample-42",
        clean_text="aa bb cc dd ee",
        edited_text="ax bx cx dx ee",
        groups=groups,
        seed=42,
    )

    assert plan.condition_for("seeded-random", 1).restored_group_indices == (1,)
    assert plan.condition_for("seeded-random", 2).restored_group_indices == (1, 3)
    assert plan.condition_for("seeded-random", 3).restored_group_indices == (1, 3, 0)


def test_plan_rejects_negative_budgets_and_unknown_orders() -> None:
    groups = _four_groups((4.0, 3.0, 2.0, 1.0))
    plan = build_restoration_plan(
        sample_id="sample-42",
        clean_text="aa bb cc dd ee",
        edited_text="ax bx cx dx ee",
        groups=groups,
        seed=42,
    )

    with pytest.raises(ValueError, match="budget|non-negative"):
        plan.condition_for("high-relevance-first", -1)
    with pytest.raises(ValueError, match="order|unsupported"):
        plan.condition_for("not-an-order", 1)
