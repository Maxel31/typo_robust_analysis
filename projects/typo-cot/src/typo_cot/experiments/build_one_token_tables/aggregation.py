"""Pure cross-setting aggregation for Appendix D Tables 10 and 11."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from typo_cot.experiments.build_one_token_tables.protocol import (
    InferenceProtocol,
    PUBLIC_INFERENCE,
)
from typo_cot.experiments.build_one_token_tables.source import ValidatedOneTokenRun
from typo_cot.experiments.build_one_token_tables.statistics import (
    cluster_robust_mean_ci,
    cluster_sign_flip,
    percentile_cluster_bootstrap,
)
from typo_cot.experiments.one_token_prefix_replacement.protocol import (
    ADJACENT_SETTINGS,
    EXTENSION_SETTINGS,
    HISTORICAL_REFERENCE,
    LEGACY_SETTING_IDS,
    PRIMARY_SETTING,
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def _record_events(record: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(record.get("events"), field="record events")


def _table10(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    eligible = []
    for record in records:
        event = _mapping(_record_events(record).get("table10"), field="table10 event")
        if event.get("eligible") is True:
            eligible.append(event)
    denominator = len(eligible)
    return {
        "paired_eligible": denominator,
        "selected": _rate(
            sum(event.get("selected_correct_to_incorrect") is True for event in eligible),
            denominator,
        ),
        "control": _rate(
            sum(event.get("control_correct_to_incorrect") is True for event in eligible),
            denominator,
        ),
    }


def _cluster_key(row: Mapping[str, object]) -> tuple[str, str]:
    benchmark = row.get("benchmark")
    sample_id = row.get("sample_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("analysis record has no benchmark")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("analysis record has no sample_id")
    return benchmark, sample_id


def _factorial_rows(
    records: Sequence[Mapping[str, object]],
    *,
    event_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        event = _mapping(_record_events(record).get(event_name), field=event_name)
        if event.get("eligible") is not True:
            continue
        selected = event.get("selected_loss_events")
        control = event.get("control_loss_events")
        if (
            not isinstance(selected, int)
            or isinstance(selected, bool)
            or not 0 <= selected <= 2
            or not isinstance(control, int)
            or isinstance(control, bool)
            or not 0 <= control <= 2
            or event.get("event_opportunities_per_position") != 2
            or type(event.get("selected_before_control")) is not bool
        ):
            raise ValueError(f"{event_name} has invalid integer events")
        rows.append(
            {
                "benchmark": record.get("benchmark"),
                "sample_id": record.get("sample_id"),
                "selected_loss_events": selected,
                "control_loss_events": control,
                "difference_percentage_points": 50.0 * (selected - control),
                "selected_before_control": event["selected_before_control"],
            }
        )
    return rows


def _factorial_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    inference: InferenceProtocol,
    include_strata: bool = True,
) -> dict[str, object]:
    denominator = 2 * len(rows)
    selected = sum(int(row["selected_loss_events"]) for row in rows)
    control = sum(int(row["control_loss_events"]) for row in rows)
    summary: dict[str, object] = {
        "paired_eligible": len(rows),
        "event_opportunities_per_position": denominator,
        "selected": _rate(selected, denominator),
        "control": _rate(control, denominator),
        "difference_percentage_points": (
            None if denominator == 0 else 100.0 * (selected - control) / denominator
        ),
    }
    if len({_cluster_key(row) for row in rows}) >= 2:
        summary["cluster_robust_interval"] = cluster_robust_mean_ci(
            rows,
            value=lambda row: float(row["difference_percentage_points"]),
            cluster_key=_cluster_key,
        )
        summary["cluster_sign_flip"] = cluster_sign_flip(
            rows,
            value=lambda row: float(row["difference_percentage_points"]),
            cluster_key=_cluster_key,
            draws=inference.sign_flip_draws,
            seed=inference.seed,
        )
    else:
        summary["cluster_robust_interval"] = None
        summary["cluster_sign_flip"] = None
    if include_strata:
        summary["position_order_strata"] = {
            label: _factorial_summary(
                [row for row in rows if row["selected_before_control"] is before],
                inference=inference,
                include_strata=False,
            )
            for label, before in (
                ("selected_before_control", True),
                ("selected_after_control", False),
            )
        }
    return summary


def _submitted_attrition(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = 0
    reasons: Counter[str] = Counter()
    for record in records:
        events = _record_events(record)
        literal = _mapping(events.get("distant_factorial"), field="distant factorial")
        submitted = _mapping(
            events.get("distant_factorial_submitted_producer"),
            field="submitted distant factorial",
        )
        if literal.get("eligible") is not True or submitted.get("eligible") is True:
            continue
        count += 1
        raw_reasons = submitted.get("exclusion_reasons")
        if not isinstance(raw_reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in raw_reasons
        ):
            raise ValueError("submitted factorial exclusion reasons are invalid")
        reasons.update(raw_reasons)
    return {"count": count, "reasons": dict(sorted(reasons.items()))}


def _adjacent_summary(
    records: Sequence[Mapping[str, object]],
    *,
    inference: InferenceProtocol,
    label: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for record in records:
        raw = _record_events(record).get("adjacent")
        if raw is None:
            continue
        event = _mapping(raw, field="adjacent event")
        if event.get("eligible") is not True:
            continue
        selected = event.get("selected_correct_to_incorrect") is True
        control = event.get("control_correct_to_incorrect") is True
        rows.append(
            {
                "benchmark": record.get("benchmark"),
                "sample_id": record.get("sample_id"),
                "difference_percentage_points": 100.0 * (int(selected) - int(control)),
                "selected": selected,
                "control": control,
            }
        )
    denominator = len(rows)
    summary: dict[str, object] = {
        "paired_eligible": denominator,
        "selected": _rate(sum(row["selected"] is True for row in rows), denominator),
        "control": _rate(sum(row["control"] is True for row in rows), denominator),
        "difference_percentage_points": (
            None
            if denominator == 0
            else math.fsum(float(row["difference_percentage_points"]) for row in rows) / denominator
        ),
    }
    if rows:
        summary["cluster_bootstrap_interval"] = percentile_cluster_bootstrap(
            rows,
            value=lambda row: float(row["difference_percentage_points"]),
            cluster_key=_cluster_key,
            draws=inference.bootstrap_draws,
            seed=inference.seed,
            label=label,
        )
    else:
        summary["cluster_bootstrap_interval"] = None
    return summary


def _answer(record: Mapping[str, object], arm_name: str) -> object:
    arms = record.get("arms")
    if not isinstance(arms, list):
        return None
    matching = [arm for arm in arms if isinstance(arm, Mapping) and arm.get("name") == arm_name]
    if len(matching) != 1:
        return None
    generation = matching[0].get("generation")
    if not isinstance(generation, Mapping):
        return None
    return generation.get("extracted_answer", generation.get("value"))


def _figure5(runs: Sequence[ValidatedOneTokenRun]) -> dict[str, object]:
    identity = {
        "setting_id": "gemma4b_gsm8k",
        "targeting": "attribution-4",
        "sample_id": "gsm8k_00556",
    }
    candidates = [
        record
        for run in runs
        if run.setting_id == identity["setting_id"]
        for record in run.records
        if record.get("targeting") == identity["targeting"]
        and record.get("sample_id") == identity["sample_id"]
    ]
    if len(candidates) != 1:
        return {
            "identity": identity,
            "status": "missing" if not candidates else "duplicate",
            "checks": {},
            "candidate_count": len(candidates),
        }
    record = candidates[0]
    plan = _mapping(record.get("input_plan"), field="Figure 5 input plan")
    profile = _mapping(record.get("profile"), field="Figure 5 profile")
    positions = _mapping(record.get("positions"), field="Figure 5 positions")
    clean_ids = plan.get("clean_cot_ids")
    kl = profile.get("clean_to_edited_kl")
    clean_ranks = profile.get("clean_token_rank_under_clean")
    edited_ranks = profile.get("clean_token_rank_under_edited")
    selected = positions.get("selected")
    if (
        not isinstance(clean_ids, list)
        or not isinstance(kl, list)
        or not isinstance(clean_ranks, list)
        or not isinstance(edited_ranks, list)
        or not isinstance(selected, int)
        or isinstance(selected, bool)
        or not 0 <= selected < len(kl)
        or selected >= len(clean_ranks)
        or selected >= len(edited_ranks)
    ):
        raise ValueError("Figure 5 record is missing indexed profile fields")
    observed = {
        "clean_cot_token_count": len(clean_ids),
        "selected_position": selected,
        "distant_position": positions.get("distant"),
        "selected_kl": kl[selected],
        "clean_token_rank_under_clean": clean_ranks[selected],
        "clean_token_rank_under_edited": edited_ranks[selected],
        "selected_keep_answer": _answer(record, "selected_keep"),
        "selected_replacement_answer": _answer(record, "selected_from_selected"),
        "distant_keep_answer": _answer(record, "distant_keep"),
        "distant_replacement_answer": _answer(record, "distant_from_distant"),
    }
    expected = {
        "clean_cot_token_count": 61,
        "selected_position": 23,
        "distant_position": 60,
        "selected_kl": 8.785974,
        "clean_token_rank_under_clean": 1,
        "clean_token_rank_under_edited": 8,
        "selected_keep_answer": "160",
        "selected_replacement_answer": "120",
        "distant_keep_answer": "160",
        "distant_replacement_answer": "160",
    }
    checks = {
        field: {
            "expected": expected_value,
            "observed": observed[field],
            "status": (
                "match"
                if (
                    math.isclose(float(observed[field]), float(expected_value), abs_tol=5e-7)
                    if field == "selected_kl" and observed[field] is not None
                    else observed[field] == expected_value
                )
                else "mismatch"
            ),
        }
        for field, expected_value in expected.items()
    }
    for field, text in (
        ("clean_token_text", " thrice"),
        ("selected_edited_top1_text", " twice"),
    ):
        checks[field] = {
            "expected": text,
            "observed": None,
            "status": "unverifiable-from-one-token-record",
        }
    return {
        "identity": identity,
        "status": (
            "stored-fields-match"
            if all(
                check["status"] in {"match", "unverifiable-from-one-token-record"}
                for check in checks.values()
            )
            else "stored-field-mismatch"
        ),
        "checks": checks,
        "candidate_count": 1,
    }


def _table10_reference_comparison(
    observed: Mapping[str, object] | None,
    reference: Mapping[str, object],
    *,
    missing_reason: str,
) -> dict[str, object]:
    if observed is None:
        return {"status": "not_comparable", "reason": missing_reason}
    selected = _mapping(observed.get("selected"), field="observed Table 10 selected")
    control = _mapping(observed.get("control"), field="observed Table 10 control")
    reference_selected = _mapping(reference.get("selected"), field="reference Table 10 selected")
    reference_control = _mapping(reference.get("control"), field="reference Table 10 control")
    matches = (
        observed.get("paired_eligible") == reference.get("paired_eligible")
        and selected.get("numerator") == reference_selected.get("numerator")
        and control.get("numerator") == reference_control.get("numerator")
    )
    return {
        "status": "match" if matches else "differs",
        "observed": dict(observed),
        "reference": dict(reference),
        "comparison_fields": [
            "paired_eligible",
            "selected.numerator",
            "control.numerator",
        ],
    }


def _rounded_interval(summary: Mapping[str, object], key: str) -> list[float] | None:
    interval = summary.get(key)
    if not isinstance(interval, Mapping):
        return None
    low = interval.get("ci95_low")
    high = interval.get("ci95_high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    return [round(float(low), 1), round(float(high), 1)]


def _distant_reference_comparison(
    observed: Mapping[str, object] | None,
    reference: Mapping[str, object],
    *,
    missing_reason: str,
    printed_interval: Sequence[float] | None,
) -> dict[str, object]:
    if observed is None:
        return {"status": "not_comparable", "reason": missing_reason}
    selected = _mapping(observed.get("selected"), field="observed distant selected")
    control = _mapping(observed.get("control"), field="observed distant control")
    reference_selected = _mapping(reference.get("selected"), field="reference distant selected")
    reference_control = _mapping(reference.get("control"), field="reference distant control")
    matches = (
        observed.get("paired_eligible") == reference.get("paired_eligible")
        and observed.get("event_opportunities_per_position")
        == reference.get("event_opportunities_per_position")
        and selected.get("numerator") == reference_selected.get("numerator")
        and control.get("numerator") == reference_control.get("numerator")
    )
    if printed_interval is not None:
        matches = matches and _rounded_interval(observed, "cluster_robust_interval") == [
            round(float(value), 1) for value in printed_interval
        ]
    return {
        "status": "match" if matches else "differs",
        "observed": dict(observed),
        "reference": dict(reference),
        "printed_confidence_interval_95": (
            None if printed_interval is None else list(printed_interval)
        ),
    }


def _adjacent_reference_comparison(
    observed: Mapping[str, object] | None,
    reference: Mapping[str, object],
    *,
    missing_reason: str,
) -> dict[str, object]:
    if observed is None:
        return {"status": "not_comparable", "reason": missing_reason}
    selected = _mapping(observed.get("selected"), field="observed adjacent selected")
    control = _mapping(observed.get("control"), field="observed adjacent control")
    observed_signature = {
        "paired_eligible": observed.get("paired_eligible"),
        "selected_percent": (
            None if selected.get("rate") is None else round(100.0 * float(selected["rate"]), 1)
        ),
        "control_percent": (
            None if control.get("rate") is None else round(100.0 * float(control["rate"]), 1)
        ),
        "difference_percentage_points": (
            None
            if observed.get("difference_percentage_points") is None
            else round(float(observed["difference_percentage_points"]), 1)
        ),
        "confidence_interval_95": _rounded_interval(observed, "cluster_bootstrap_interval"),
    }
    reference_signature = {
        key: reference.get(key)
        for key in (
            "paired_eligible",
            "selected_percent",
            "control_percent",
            "difference_percentage_points",
            "confidence_interval_95",
        )
    }
    return {
        "status": "match" if observed_signature == reference_signature else "differs",
        "observed": dict(observed),
        "observed_rounded_signature": observed_signature,
        "reference": dict(reference),
    }


def _distant_stratum_reference_comparison(
    observed: Mapping[str, object] | None,
    reference: Mapping[str, object],
) -> dict[str, object]:
    if observed is None:
        return {
            "status": "not_comparable",
            "reason": "complete-14-extension-grid-required",
        }
    selected = _mapping(observed.get("selected"), field="observed stratum selected")
    control = _mapping(observed.get("control"), field="observed stratum control")
    observed_signature = {
        "paired_eligible": observed.get("paired_eligible"),
        "selected_percent": (
            None if selected.get("rate") is None else round(100.0 * float(selected["rate"]), 1)
        ),
        "control_percent": (
            None if control.get("rate") is None else round(100.0 * float(control["rate"]), 1)
        ),
        "difference_percentage_points": (
            None
            if observed.get("difference_percentage_points") is None
            else round(float(observed["difference_percentage_points"]), 1)
        ),
        "confidence_interval_95": _rounded_interval(observed, "cluster_robust_interval"),
    }
    reference_signature = {
        key: reference.get(key)
        for key in (
            "paired_eligible",
            "selected_percent",
            "control_percent",
            "difference_percentage_points",
            "confidence_interval_95",
        )
    }
    return {
        "status": "match" if observed_signature == reference_signature else "differs",
        "observed": dict(observed),
        "observed_rounded_signature": observed_signature,
        "reference": dict(reference),
    }


def _historical_comparison(
    *,
    table10_cells: Mapping[str, Mapping[str, object]],
    table10_pool: Mapping[str, object] | None,
    factorial: Mapping[str, object],
    adjacent_cells: Mapping[str, Mapping[str, object]],
    adjacent_pool: Mapping[str, object] | None,
) -> dict[str, object]:
    submitted = _mapping(
        factorial.get("submitted_producer"), field="submitted factorial comparison"
    ).get("extension_pool")
    literal = _mapping(factorial.get("pdf_literal"), field="literal factorial comparison").get(
        "extension_pool"
    )
    submitted_reference = HISTORICAL_REFERENCE["table11_distant_submitted_producer_exact_counts"]
    literal_reference = HISTORICAL_REFERENCE["table11_distant_final_pdf_literal_reclassification"]
    printed_distant = HISTORICAL_REFERENCE["table11_distant_pooled"]
    table10_cell_reference = HISTORICAL_REFERENCE["table10_cells"]
    adjacent_cell_reference = HISTORICAL_REFERENCE["table11_adjacent_cells"]
    submitted_strata = (
        submitted.get("position_order_strata") if isinstance(submitted, Mapping) else None
    )
    historical_strata = _mapping(
        HISTORICAL_REFERENCE["table11_distant_strata"],
        field="historical distant strata",
    )
    return {
        "contract": "reference-only-not-runtime-acceptance",
        "table10": {
            "settings": {
                setting_id: _table10_reference_comparison(
                    table10_cells.get(setting_id),
                    _mapping(reference, field=f"historical Table 10 {setting_id}"),
                    missing_reason="setting-run-not-supplied",
                )
                for setting_id, reference in _mapping(
                    table10_cell_reference, field="historical Table 10 cells"
                ).items()
            },
            "primary": _table10_reference_comparison(
                table10_cells.get(LEGACY_SETTING_IDS[PRIMARY_SETTING]),
                _mapping(
                    HISTORICAL_REFERENCE["table10_primary_gemma4b_gsm8k"],
                    field="historical primary Table 10",
                ),
                missing_reason="primary-setting-run-required",
            ),
            "extension_pool": _table10_reference_comparison(
                table10_pool,
                _mapping(
                    HISTORICAL_REFERENCE["table10_extension_aggregate"],
                    field="historical extension Table 10",
                ),
                missing_reason="complete-14-extension-grid-required",
            ),
        },
        "table11": {
            "distant_pdf_literal": _distant_reference_comparison(
                literal if isinstance(literal, Mapping) else None,
                _mapping(literal_reference, field="historical literal distant"),
                missing_reason="complete-14-extension-grid-required",
                printed_interval=None,
            ),
            "distant_submitted_producer": _distant_reference_comparison(
                submitted if isinstance(submitted, Mapping) else None,
                _mapping(submitted_reference, field="historical submitted distant"),
                missing_reason="complete-14-extension-grid-required",
                printed_interval=printed_distant["confidence_interval_95"],  # type: ignore[arg-type]
            ),
            "distant_submitted_position_order_strata": {
                name: _distant_stratum_reference_comparison(
                    (
                        _mapping(submitted_strata, field="observed distant strata").get(name)
                        if isinstance(submitted_strata, Mapping)
                        else None
                    ),  # type: ignore[arg-type]
                    _mapping(reference, field=f"historical distant stratum {name}"),
                )
                for name, reference in historical_strata.items()
            },
            "adjacent": _adjacent_reference_comparison(
                adjacent_pool,
                _mapping(
                    HISTORICAL_REFERENCE["table11_adjacent_pooled"],
                    field="historical adjacent pool",
                ),
                missing_reason="complete-3-adjacent-grid-required",
            ),
            "adjacent_settings": {
                setting_id: _adjacent_reference_comparison(
                    adjacent_cells.get(setting_id),
                    _mapping(reference, field=f"historical adjacent {setting_id}"),
                    missing_reason="setting-run-not-supplied",
                )
                for setting_id, reference in _mapping(
                    adjacent_cell_reference, field="historical adjacent cells"
                ).items()
            },
        },
    }


def assemble_one_token_tables(
    runs: Sequence[ValidatedOneTokenRun],
    *,
    inference: InferenceProtocol = PUBLIC_INFERENCE,
) -> dict[str, object]:
    """Assemble cells and gated paper pools from already validated runs."""

    if not isinstance(inference, InferenceProtocol):
        raise TypeError("inference must be InferenceProtocol")
    by_setting: dict[str, ValidatedOneTokenRun] = {}
    for run in runs:
        if not isinstance(run, ValidatedOneTokenRun):
            raise TypeError("runs must contain ValidatedOneTokenRun values")
        if run.setting_id in by_setting:
            raise ValueError(f"duplicate validated setting {run.setting_id}")
        by_setting[run.setting_id] = run

    expected_ids = frozenset(LEGACY_SETTING_IDS.values())
    unknown = sorted(set(by_setting) - expected_ids)
    if unknown:
        raise ValueError(f"unexpected one-token settings: {', '.join(unknown)}")
    primary_id = LEGACY_SETTING_IDS[PRIMARY_SETTING]
    extension_ids = frozenset(LEGACY_SETTING_IDS[setting] for setting in EXTENSION_SETTINGS)
    adjacent_ids = frozenset(LEGACY_SETTING_IDS[setting] for setting in ADJACENT_SETTINGS)
    available = set(by_setting)
    complete_extensions = extension_ids <= available
    complete_adjacent = adjacent_ids <= available

    ordered_ids = [
        setting_id
        for _, setting_id in sorted(
            LEGACY_SETTING_IDS.items(),
            key=lambda item: item[1],
        )
    ]
    table10_cells = {
        setting_id: _table10(by_setting[setting_id].records)
        for setting_id in ordered_ids
        if setting_id in by_setting
    }
    extension_records = [
        record
        for setting_id in ordered_ids
        if setting_id in extension_ids and setting_id in by_setting
        for record in by_setting[setting_id].records
    ]
    table10_pool = _table10(extension_records) if complete_extensions else None

    factorial: dict[str, object] = {}
    for output_name, event_name in (
        ("pdf_literal", "distant_factorial"),
        ("submitted_producer", "distant_factorial_submitted_producer"),
    ):
        cells = {
            setting_id: _factorial_summary(
                _factorial_rows(by_setting[setting_id].records, event_name=event_name),
                inference=inference,
            )
            for setting_id in ordered_ids
            if setting_id in by_setting
        }
        pool = None
        if complete_extensions:
            pool = _factorial_summary(
                _factorial_rows(extension_records, event_name=event_name),
                inference=inference,
            )
            if output_name == "submitted_producer":
                pool["attrition_from_pdf_literal"] = _submitted_attrition(extension_records)
        factorial[output_name] = {"settings": cells, "extension_pool": pool}

    adjacent_cells = {
        setting_id: _adjacent_summary(
            by_setting[setting_id].records,
            inference=inference,
            # Exact stable labels from the submitted local-control aggregate.
            label=f"setting:{setting_id}",
        )
        for setting_id in sorted(adjacent_ids & available)
    }
    adjacent_pool = None
    if complete_adjacent:
        adjacent_records = [
            record
            for setting_id in sorted(adjacent_ids)
            for record in by_setting[setting_id].records
        ]
        adjacent_pool = _adjacent_summary(
            adjacent_records,
            inference=inference,
            label="pooled",
        )

    historical_comparison = _historical_comparison(
        table10_cells=table10_cells,
        table10_pool=table10_pool,
        factorial=factorial,
        adjacent_cells=adjacent_cells,
        adjacent_pool=adjacent_pool,
    )

    return {
        "schema_version": "one-token-tables/v1",
        "coverage": {
            "available_settings": sorted(available),
            "missing_settings": sorted(expected_ids - available),
            "complete_15_setting_grid": expected_ids <= available,
            "complete_14_extension_grid": complete_extensions,
            "complete_3_adjacent_grid": complete_adjacent,
            "extension_pool_setting_ids": sorted(extension_ids) if complete_extensions else None,
            "primary_in_extension_pool": False,
        },
        "table10": {
            "settings": table10_cells,
            "primary": table10_cells.get(primary_id),
            "extension_pool": table10_pool,
        },
        "table11": {
            "distant": factorial,
            "adjacent": {
                "settings": adjacent_cells,
                "extension_pool": adjacent_pool,
            },
        },
        "figure5_validation": _figure5(runs),
        "inference": inference.to_dict(),
        "historical_reference": HISTORICAL_REFERENCE,
        "historical_comparison": historical_comparison,
    }


__all__ = ["assemble_one_token_tables"]
