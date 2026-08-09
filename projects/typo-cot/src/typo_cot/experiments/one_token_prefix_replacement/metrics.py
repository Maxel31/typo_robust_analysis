"""Pure event classification and aggregation for the one-token diagnostic."""

# External arm payload shape violations are schema-value errors, not API type errors.
# ruff: noqa: TRY004

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


def _arm_state(arms: Mapping[str, object], name: str) -> tuple[bool, bool]:
    value = arms.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing one-token arm {name}")
    generation = value.get("generation")
    if not isinstance(generation, Mapping) or type(generation.get("is_correct")) is not bool:
        raise ValueError(f"arm {name} has no boolean generation correctness")
    if type(value.get("is_noop")) is not bool:
        raise ValueError(f"arm {name} has no boolean is_noop")
    return bool(generation["is_correct"]), bool(value["is_noop"])


def _replacement_token_state(
    arms: Mapping[str, object],
    *,
    same_source_arms: tuple[str, str],
) -> tuple[int, bool]:
    states: list[tuple[int, bool]] = []
    for name in same_source_arms:
        value = arms.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"missing one-token arm {name}")
        token_id = value.get("forced_token_id")
        admissible = value.get("token_is_admissible")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(f"arm {name} has no non-negative forced_token_id")
        if type(admissible) is not bool:
            raise ValueError(f"arm {name} has no boolean token_is_admissible")
        states.append((token_id, admissible))
    if states[0] != states[1]:
        raise ValueError("same-source replacement arms disagree on token identity or admissibility")
    return states[0]


def classify_one_token_events(
    arms: Mapping[str, object],
    *,
    selected_before_control: bool,
    adjacent_requested: bool,
) -> dict[str, object]:
    """Classify one record under the three final-PDF denominators."""

    selected_keep, _ = _arm_state(arms, "selected_keep")
    selected_selected, selected_selected_noop = _arm_state(arms, "selected_from_selected")
    selected_distant, selected_distant_noop = _arm_state(arms, "selected_from_distant")
    distant_keep, _ = _arm_state(arms, "distant_keep")
    distant_selected, distant_selected_noop = _arm_state(arms, "distant_from_selected")
    distant_distant, distant_distant_noop = _arm_state(arms, "distant_from_distant")

    table10_eligible = (
        selected_keep and distant_keep and not selected_selected_noop and not distant_distant_noop
    )
    table10 = {
        "eligible": table10_eligible,
        "selected_correct_to_incorrect": (table10_eligible and not selected_selected),
        "control_correct_to_incorrect": table10_eligible and not distant_distant,
    }

    factorial_eligible = (
        selected_keep
        and distant_keep
        and not selected_selected_noop
        and not selected_distant_noop
        and not distant_selected_noop
        and not distant_distant_noop
    )
    factorial = {
        "eligible": factorial_eligible,
        "selected_loss_events": (
            int(not selected_selected) + int(not selected_distant) if factorial_eligible else 0
        ),
        "control_loss_events": (
            int(not distant_selected) + int(not distant_distant) if factorial_eligible else 0
        ),
        "event_opportunities_per_position": 2,
        "selected_before_control": selected_before_control,
    }

    selected_token, selected_token_admissible = _replacement_token_state(
        arms,
        same_source_arms=("selected_from_selected", "distant_from_selected"),
    )
    control_token, control_token_admissible = _replacement_token_state(
        arms,
        same_source_arms=("selected_from_distant", "distant_from_distant"),
    )
    distinct_tokens = selected_token != control_token
    additional_guard = {
        "selected_and_control_tokens_distinct": distinct_tokens,
        "selected_token_admissible": selected_token_admissible,
        "control_token_admissible": control_token_admissible,
    }
    exclusion_reasons = []
    if not distinct_tokens:
        exclusion_reasons.append("selected-and-control-tokens-identical")
    if not selected_token_admissible:
        exclusion_reasons.append("selected-token-not-admissible")
    if not control_token_admissible:
        exclusion_reasons.append("control-token-not-admissible")
    submitted_eligible = factorial_eligible and not exclusion_reasons
    submitted_factorial = {
        "eligible": submitted_eligible,
        "selected_loss_events": (
            int(not selected_selected) + int(not selected_distant) if submitted_eligible else 0
        ),
        "control_loss_events": (
            int(not distant_selected) + int(not distant_distant) if submitted_eligible else 0
        ),
        "event_opportunities_per_position": 2,
        "selected_before_control": selected_before_control,
        "additional_guard": additional_guard,
        "exclusion_reasons": exclusion_reasons,
    }

    adjacent: dict[str, object] | None = None
    if adjacent_requested:
        adjacent_keep, _ = _arm_state(arms, "adjacent_keep")
        adjacent_selected, adjacent_selected_noop = _arm_state(arms, "adjacent_from_selected")
        adjacent_eligible = (
            selected_keep
            and adjacent_keep
            and not selected_selected_noop
            and not adjacent_selected_noop
        )
        adjacent = {
            "eligible": adjacent_eligible,
            "selected_correct_to_incorrect": (adjacent_eligible and not selected_selected),
            "control_correct_to_incorrect": adjacent_eligible and not adjacent_selected,
        }
    return {
        "table10": table10,
        "distant_factorial": factorial,
        "distant_factorial_submitted_producer": submitted_factorial,
        "adjacent": adjacent,
    }


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def aggregate_one_token_events(
    rows: Sequence[Mapping[str, object]],
    *,
    adjacent_requested: bool,
) -> dict[str, object]:
    """Aggregate record events without mixing their denominators."""

    table_rows = [row["table10"] for row in rows if isinstance(row.get("table10"), Mapping)]
    table_eligible = [row for row in table_rows if row.get("eligible") is True]
    factorial_rows = [
        row["distant_factorial"]
        for row in rows
        if isinstance(row.get("distant_factorial"), Mapping)
    ]
    submitted_rows = [
        row["distant_factorial_submitted_producer"]
        for row in rows
        if isinstance(row.get("distant_factorial_submitted_producer"), Mapping)
    ]

    def factorial_summary(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
        eligible = [row for row in values if row.get("eligible") is True]
        opportunities = 2 * len(eligible)
        strata: dict[str, object] = {}
        for label, before in (
            ("selected_before_control", True),
            ("selected_after_control", False),
        ):
            selected = [row for row in eligible if row.get("selected_before_control") is before]
            denominator = 2 * len(selected)
            strata[label] = {
                "paired_eligible": len(selected),
                "selected": _rate(
                    sum(int(row["selected_loss_events"]) for row in selected), denominator
                ),
                "control": _rate(
                    sum(int(row["control_loss_events"]) for row in selected), denominator
                ),
            }
        return {
            "paired_eligible": len(eligible),
            "event_opportunities_per_position": opportunities,
            "selected": _rate(
                sum(int(row["selected_loss_events"]) for row in eligible), opportunities
            ),
            "control": _rate(
                sum(int(row["control_loss_events"]) for row in eligible), opportunities
            ),
            "position_order_strata": strata,
        }

    paper_factorial = factorial_summary(factorial_rows)
    submitted_factorial = factorial_summary(submitted_rows)
    attrition_reasons: Counter[str] = Counter()
    for paper, submitted in zip(factorial_rows, submitted_rows, strict=True):
        if paper.get("eligible") is not True or submitted.get("eligible") is True:
            continue
        reasons = submitted.get("exclusion_reasons")
        if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            raise ValueError("submitted-producer exclusion reasons are invalid")
        attrition_reasons.update(reasons)
    submitted_factorial["attrition_from_paper_literal"] = {
        "count": int(paper_factorial["paired_eligible"])
        - int(submitted_factorial["paired_eligible"]),
        "reasons": dict(sorted(attrition_reasons.items())),
    }

    adjacent_rows: list[Mapping[str, object]] = []
    if adjacent_requested:
        adjacent_rows = [
            row["adjacent"]
            for row in rows
            if isinstance(row.get("adjacent"), Mapping) and row["adjacent"].get("eligible") is True
        ]

    return {
        "table10": {
            "paired_eligible": len(table_eligible),
            "selected": _rate(
                sum(row.get("selected_correct_to_incorrect") is True for row in table_eligible),
                len(table_eligible),
            ),
            "control": _rate(
                sum(row.get("control_correct_to_incorrect") is True for row in table_eligible),
                len(table_eligible),
            ),
        },
        "distant_factorial": paper_factorial,
        "distant_factorial_submitted_producer": submitted_factorial,
        "adjacent": (
            {
                "requested": True,
                "paired_eligible": len(adjacent_rows),
                "selected": _rate(
                    sum(row.get("selected_correct_to_incorrect") is True for row in adjacent_rows),
                    len(adjacent_rows),
                ),
                "control": _rate(
                    sum(row.get("control_correct_to_incorrect") is True for row in adjacent_rows),
                    len(adjacent_rows),
                ),
            }
            if adjacent_requested
            else {"requested": False, "paired_eligible": 0, "selected": None, "control": None}
        ),
    }


__all__ = ["aggregate_one_token_events", "classify_one_token_events"]
