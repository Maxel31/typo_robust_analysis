"""Pure planning contracts shared by the rebuttal manifest and controls."""

from __future__ import annotations

import pytest

from typo_cot.data.matched_donors import (
    MatchedDonorCandidate,
    plan_cyclic_derangement,
)
from typo_cot.experiments.build_rebuttal_manifest.planning import (
    StrictOffsetPlan,
    WindowSplitCandidate,
    plan_strict_offset_control,
    plan_window_split,
)


def test_cyclic_derangement_is_order_invariant_matched_and_nonself() -> None:
    candidates = (
        MatchedDonorCandidate(("m1", "gsm8k", "random-4", "2"), ("pair-d",)),
        MatchedDonorCandidate(("m1", "gsm8k", "attribution-4", "1"), ("pair-b",)),
        MatchedDonorCandidate(("m1", "gsm8k", "random-4", "2"), ("pair-c",)),
        MatchedDonorCandidate(("m1", "gsm8k", "attribution-4", "1"), ("pair-a",)),
    )

    plan = plan_cyclic_derangement(candidates, reject_singletons=False)
    reversed_plan = plan_cyclic_derangement(reversed(candidates), reject_singletons=False)

    assert plan == reversed_plan
    assert plan.singleton_identities == ()
    donors = plan.as_dict()
    assert donors == {
        ("pair-a",): ("pair-b",),
        ("pair-b",): ("pair-a",),
        ("pair-c",): ("pair-d",),
        ("pair-d",): ("pair-c",),
    }
    assert all(recipient != donor for recipient, donor in donors.items())


def test_cyclic_derangement_reports_or_rejects_singleton_strata() -> None:
    singleton = MatchedDonorCandidate(
        ("m1", "gsm8k", "attribution-4", "3"),
        ("only",),
    )

    plan = plan_cyclic_derangement((singleton,), reject_singletons=False)
    assert plan.assignments == ()
    assert plan.singleton_identities == (("only",),)

    with pytest.raises(ValueError, match="singleton"):
        plan_cyclic_derangement((singleton,), reject_singletons=True)


def test_cyclic_derangement_rejects_duplicate_recipient_identity() -> None:
    duplicate = (
        MatchedDonorCandidate(("m1", "gsm8k", "attribution-4", "1"), ("same",)),
        MatchedDonorCandidate(("m2", "mmlu", "random-4", "2"), ("same",)),
    )

    with pytest.raises(ValueError, match="duplicate.*identity"):
        plan_cyclic_derangement(duplicate, reject_singletons=False)


def test_strict_offset_control_keeps_the_full_coordinate_cardinality() -> None:
    plan = plan_strict_offset_control(
        clean_positions=(2, 5),
        typo_positions=(3, 6),
        clean_edited_tokens=(2, 5),
        typo_edited_tokens=(3, 6),
        clean_prompt_length=12,
        typo_prompt_length=13,
        offset=2,
    )

    assert plan.valid is True
    assert plan.source_positions == (4, 7)
    assert plan.destination_positions == (5, 8)
    assert plan.invalid_reason is None
    assert plan.input_pairs == 2


def test_strict_offset_control_marks_an_unedited_pair_invalid() -> None:
    plan = plan_strict_offset_control([], [], [], [], 10, 10, offset=2)

    assert plan == StrictOffsetPlan(
        valid=False,
        source_positions=(),
        destination_positions=(),
        input_pairs=0,
        offset_tokens=2,
        invalid_reason="no-aligned-edited-words",
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    (
        ({"clean_positions": (2, 9)}, "clean-offset-outside-prompt-interior"),
        ({"typo_positions": (3, 10)}, "typo-offset-outside-prompt-interior"),
        ({"clean_edited_tokens": (2, 4, 5)}, "clean-offset-overlaps-edited-token"),
        ({"typo_edited_tokens": (3, 5, 6)}, "typo-offset-overlaps-edited-token"),
    ),
)
def test_strict_offset_control_invalidates_the_whole_arm(
    changes: dict[str, tuple[int, ...]],
    reason: str,
) -> None:
    arguments: dict[str, object] = {
        "clean_positions": (2, 5),
        "typo_positions": (3, 6),
        "clean_edited_tokens": (2, 5),
        "typo_edited_tokens": (3, 6),
        "clean_prompt_length": 12,
        "typo_prompt_length": 13,
        "offset": 2,
    }
    arguments.update(changes)

    plan = plan_strict_offset_control(**arguments)  # type: ignore[arg-type]

    assert plan.valid is False
    assert plan.source_positions == ()
    assert plan.destination_positions == ()
    assert plan.invalid_reason == reason
    assert plan.input_pairs == 2


def test_window_split_is_stratified_disjoint_complete_and_order_invariant() -> None:
    candidates = tuple(
        WindowSplitCandidate(
            pair_id=f"{targeting}-{index}",
            stratum=("m1", "gsm8k", targeting),
            sample_group=f"sample-{index}",
        )
        for targeting in ("attribution-4", "random-4")
        for index in range(5)
    )

    plan = plan_window_split(candidates, seed=42)
    reversed_plan = plan_window_split(reversed(candidates), seed=42)

    assert plan == reversed_plan
    selection = set(plan.selection_pair_ids)
    evaluation = set(plan.evaluation_pair_ids)
    assert selection.isdisjoint(evaluation)
    assert selection | evaluation == {candidate.pair_id for candidate in candidates}
    for targeting in ("attribution-4", "random-4"):
        assert sum(pair_id.startswith(targeting) for pair_id in selection) == 2
        assert sum(pair_id.startswith(targeting) for pair_id in evaluation) == 3


def test_window_split_keeps_the_same_task_sample_group_across_models_and_targets() -> None:
    candidates = tuple(
        WindowSplitCandidate(
            pair_id=f"{model}-{target}-{sample}",
            stratum=(model, "gsm8k", target),
            sample_group=sample,
        )
        for model in ("gemma", "llama", "mistral")
        for target in ("attribution-4", "random-4")
        for sample in ("sample-0", "sample-1", "sample-2", "sample-3")
    )

    plan = plan_window_split(candidates, seed=42)
    selection = set(plan.selection_pair_ids)

    for sample in ("sample-0", "sample-1", "sample-2", "sample-3"):
        phases = {
            "selection" if candidate.pair_id in selection else "evaluation"
            for candidate in candidates
            if candidate.sample_group == sample
        }
        assert len(phases) == 1


def test_window_split_rejects_duplicate_ids_and_singleton_strata() -> None:
    with pytest.raises(ValueError, match="duplicate pair_id"):
        plan_window_split(
            (
                WindowSplitCandidate("same", ("m1", "gsm8k", "attribution-4"), "s1"),
                WindowSplitCandidate("same", ("m1", "gsm8k", "random-4"), "s1"),
            ),
            seed=42,
        )

    with pytest.raises(ValueError, match="at least two"):
        plan_window_split(
            (WindowSplitCandidate("only", ("m1", "gsm8k", "attribution-4"), "s1"),),
            seed=42,
        )
