"""Integer-event aggregation for the final paper's Table 8."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.edit_count_sensitivity.protocol import (
    ANALYSIS_PROTOCOL,
    ANALYSIS_PROTOCOL_SHA256,
    EDIT_COUNTS,
    EXPECTED_ACCURACY_SETTING_COUNT,
    EXPECTED_ACCURACY_SETTINGS,
    EXPECTED_RESTORATION_SETTINGS,
    PUBLISHED_REFERENCE,
    restoration_setting_label,
)
from typo_cot.experiments.edit_count_sensitivity.source import (
    CotSwapEditCountRun,
    EditCountSensitivityInputError,
    PreparedEditCountRun,
)


def setting_id(model: str, benchmark: str) -> str:
    """Return an unambiguous machine-readable setting identity."""
    return f"{model}::{benchmark}"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EditCountSensitivityInputError(f"{context} must be a JSON object")
    return value


def _correct(record: Mapping[str, object], side: str) -> bool:
    side_payload = _mapping(record.get(side), context=f"record.{side}")
    answer = _mapping(side_payload.get("answer"), context=f"record.{side}.answer")
    value = answer.get("is_correct")
    if not isinstance(value, bool):
        raise EditCountSensitivityInputError(f"record.{side}.answer.is_correct must be boolean")
    return value


def _clean_identity(record: Mapping[str, object]) -> str:
    clean = _mapping(record.get("clean"), context="record.clean")
    identity = {
        "gold_answer": record.get("gold_answer"),
        "prompt": clean.get("prompt"),
        "prompt_token_count": clean.get("prompt_token_count"),
        "continuation": clean.get("continuation"),
        "continuation_token_count": clean.get("continuation_token_count"),
        "answer": clean.get("answer"),
    }
    return _canonical_sha256(identity)


def _rate(correct: int, denominator: int) -> dict[str, object]:
    return {
        "correct": correct,
        "denominator": denominator,
        "rate": correct / denominator if denominator else None,
    }


def _restoration_rate(restored: int, denominator: int) -> dict[str, object]:
    return {
        "denominator": denominator,
        "restored": restored,
        "rate": restored / denominator if denominator else None,
    }


def _source_payload(run: PreparedEditCountRun) -> dict[str, object]:
    return {
        "edit_count": run.edit_count,
        "pairs": str(run.pairs_path),
        "pairs_sha256": run.pairs_sha256,
        "run": str(run.run_path),
        "run_sha256": run.run_sha256,
        "records": len(run.records),
        "model_revision": run.model_revision,
        "dataset_records_sha256": run.dataset_records_sha256,
    }


def _accuracy_row(
    runs: Sequence[PreparedEditCountRun],
    *,
    edit_counts: tuple[int, ...],
) -> dict[str, object]:
    by_count = {run.edit_count: run for run in runs}
    model = runs[0].model
    benchmark = runs[0].benchmark
    if set(by_count) != set(edit_counts):
        raise AssertionError("accuracy aggregation received an incomplete setting")
    if any(run.setting != (model, benchmark) for run in runs):
        raise AssertionError("accuracy aggregation mixed settings")
    if len({run.model_revision for run in runs}) != 1:
        raise EditCountSensitivityInputError(
            f"{setting_id(model, benchmark)} uses different model revisions across edit counts"
        )
    dataset_identities = {(run.dataset_records_sha256, run.dataset_sample_count) for run in runs}
    if len(dataset_identities) != 1:
        raise EditCountSensitivityInputError(
            f"{setting_id(model, benchmark)} uses a different dataset cohort across edit counts"
        )

    baseline = by_count[edit_counts[0]]
    full_conditions: dict[str, object] = {
        "0": _rate(
            sum(_correct(record, "clean") for record in baseline.records),
            len(baseline.records),
        )
    }
    for count in edit_counts:
        run = by_count[count]
        full_conditions[str(count)] = _rate(
            sum(_correct(record, "edited") for record in run.records),
            len(run.records),
        )

    shared_ids = set.intersection(
        *(set(run.records_by_id) for run in (by_count[count] for count in edit_counts))
    )
    if not shared_ids:
        raise EditCountSensitivityInputError(
            f"{setting_id(model, benchmark)} has no sample IDs common to all edit counts"
        )
    for sample_id in sorted(shared_ids):
        identities = {
            _clean_identity(by_count[count].records_by_id[sample_id]) for count in edit_counts
        }
        if len(identities) != 1:
            raise EditCountSensitivityInputError(
                f"{setting_id(model, benchmark)} sample {sample_id}: clean condition "
                "differs across edit-count sources"
            )
    matched: dict[str, object] = {
        "sample_count": len(shared_ids),
        "conditions": {
            "0": _rate(
                sum(
                    _correct(baseline.records_by_id[sample_id], "clean") for sample_id in shared_ids
                ),
                len(shared_ids),
            )
        },
    }
    conditions = matched["conditions"]
    if not isinstance(conditions, dict):
        raise AssertionError("matched condition payload is not mutable")
    for count in edit_counts:
        conditions[str(count)] = _rate(
            sum(
                _correct(by_count[count].records_by_id[sample_id], "edited")
                for sample_id in shared_ids
            ),
            len(shared_ids),
        )
    clean_above_four: bool | None = None
    if 4 in by_count:
        clean_rate = _mapping(full_conditions["0"], context="full clean rate")["rate"]
        four_rate = _mapping(full_conditions["4"], context="full four-edit rate")["rate"]
        clean_above_four = bool(clean_rate > four_rate)  # type: ignore[operator]
    return {
        "schema_version": "edit-count-sensitivity-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "edit-count-sensitivity",
        "analysis": "accuracy",
        "setting_id": setting_id(model, benchmark),
        "model": model,
        "benchmark": benchmark,
        "targeting": "attribution-4",
        "edit_counts": list(edit_counts),
        "full_conditions": full_conditions,
        "matched_conditions": matched,
        "clean_above_four_edits": clean_above_four,
        "sources": [_source_payload(by_count[count]) for count in edit_counts],
    }


def _event_counts(run: CotSwapEditCountRun) -> tuple[int, int]:
    denominator = 0
    restored = 0
    for row in run.records:
        events = _mapping(row.get("events"), context="cot-swap record.events")
        denominator += events.get("restoration_denominator") is True
        restored += events.get("b_to_c_restored") is True
    return denominator, restored


def _restoration_row(
    runs: Sequence[CotSwapEditCountRun],
    *,
    edit_counts: tuple[int, ...],
) -> dict[str, object]:
    by_count = {run.edit_count: run for run in runs}
    model = runs[0].model
    benchmark = runs[0].benchmark
    if set(by_count) != set(edit_counts):
        raise AssertionError("restoration aggregation received an incomplete setting")
    by_edit_count: dict[str, object] = {}
    sources: list[dict[str, object]] = []
    for count in edit_counts:
        run = by_count[count]
        denominator, restored = _event_counts(run)
        by_edit_count[str(count)] = _restoration_rate(restored, denominator)
        sources.append(
            {
                "edit_count": count,
                "run": str(run.run_path),
                "run_sha256": run.run_sha256,
                "records": str(run.records_path),
                "records_sha256": run.records_sha256,
                "source_pairs_sha256": run.source_pairs_sha256,
                "source_run_sha256": run.source_run_sha256,
            }
        )
    return {
        "schema_version": "edit-count-sensitivity-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "edit-count-sensitivity",
        "analysis": "restoration",
        "setting_id": setting_id(model, benchmark),
        "label": restoration_setting_label(model, benchmark),
        "model": model,
        "benchmark": benchmark,
        "targeting": "attribution-4",
        "edit_counts": list(edit_counts),
        "zero_edit_restoration": None,
        "zero_edit_reason": "undefined-because-no-typo-induced-change",
        "by_edit_count": by_edit_count,
        "sources": sources,
    }


def _group_complete[T](
    runs: Sequence[T],
    *,
    edit_counts: tuple[int, ...],
) -> tuple[dict[tuple[str, str], list[T]], dict[str, list[int]]]:
    grouped: defaultdict[tuple[str, str], list[T]] = defaultdict(list)
    for run in runs:
        grouped[(run.model, run.benchmark)].append(run)  # type: ignore[attr-defined]
    complete: dict[tuple[str, str], list[T]] = {}
    incomplete: dict[str, list[int]] = {}
    for key, group in sorted(grouped.items()):
        available = {run.edit_count for run in group}  # type: ignore[attr-defined]
        missing = sorted(set(edit_counts) - available)
        if missing:
            incomplete[setting_id(*key)] = missing
        else:
            complete[key] = sorted(
                group,
                key=lambda run: run.edit_count,  # type: ignore[attr-defined]
            )
    return complete, incomplete


def _accuracy_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    condition_names = ("0", "1", "2", "4")
    available = tuple(
        name
        for name in condition_names
        if all(name in _mapping(row["full_conditions"], context="full_conditions") for row in rows)
    )
    equal_mean = {
        name: sum(
            float(
                _mapping(
                    _mapping(row["full_conditions"], context="full_conditions")[name],
                    context=f"full_conditions.{name}",
                )["rate"]
            )
            for row in rows
        )
        / len(rows)
        for name in available
    }
    matched_total = sum(
        int(_mapping(row["matched_conditions"], context="matched_conditions")["sample_count"])
        for row in rows
    )
    matched_conditions: dict[str, object] = {}
    for name in available:
        correct = sum(
            int(
                _mapping(
                    _mapping(row["matched_conditions"], context="matched_conditions")["conditions"],
                    context="matched_conditions.conditions",
                )[name]["correct"]  # type: ignore[index]
            )
            for row in rows
        )
        matched_conditions[name] = _rate(correct, matched_total)
    eligible_clean_four = [
        row["clean_above_four_edits"]
        for row in rows
        if row.get("clean_above_four_edits") is not None
    ]
    return {
        "complete_settings": len(rows),
        "equal_setting_mean": equal_mean,
        "matched_items": {
            "sample_count": matched_total,
            "conditions": matched_conditions,
        },
        "clean_above_four_settings": {
            "numerator": sum(value is True for value in eligible_clean_four),
            "denominator": len(eligible_clean_four),
        },
    }


def _restoration_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    pooled: dict[str, object] = {}
    for count in EDIT_COUNTS:
        key = str(count)
        applicable = [
            _mapping(row["by_edit_count"], context="by_edit_count")[key]
            for row in rows
            if key in _mapping(row["by_edit_count"], context="by_edit_count")
        ]
        denominator = sum(
            int(_mapping(cell, context=f"by_edit_count.{key}")["denominator"])
            for cell in applicable
        )
        restored = sum(
            int(_mapping(cell, context=f"by_edit_count.{key}")["restored"]) for cell in applicable
        )
        if applicable:
            pooled[key] = _restoration_rate(restored, denominator)
    return {
        "complete_settings": len(rows),
        "zero_edit_restoration": None,
        "zero_edit_reason": "undefined-because-no-typo-induced-change",
        "settings": [dict(row) for row in rows],
        "pooled": pooled,
    }


def _rounded_percent(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100, 1)


def _published_comparison(
    *,
    accuracy: Mapping[str, object],
    restoration: Mapping[str, object],
    complete_accuracy: bool,
    complete_restoration: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "acceptance_rule": "rounded-to-one-decimal-percentage-equality/v1",
        "historical_cohort_identity": False,
    }
    if complete_accuracy:
        accuracy_reference = _mapping(PUBLISHED_REFERENCE["accuracy"], context="published accuracy")
        equal = _mapping(accuracy["equal_setting_mean"], context="equal_setting_mean")
        matched = _mapping(
            _mapping(accuracy["matched_items"], context="matched_items")["conditions"],
            context="matched_items.conditions",
        )
        result["accuracy"] = {
            "equal_setting_mean_matches": {
                key: _rounded_percent(equal[key])
                == _rounded_percent(
                    _mapping(
                        accuracy_reference["equal_setting_mean"],
                        context="published equal setting",
                    )[key]
                )
                for key in ("0", "1", "2", "4")
            },
            "matched_items_matches": {
                key: _rounded_percent(_mapping(matched[key], context=f"matched.{key}")["rate"])
                == _rounded_percent(
                    _mapping(
                        accuracy_reference["matched_81812_items"],
                        context="published matched items",
                    )[key]
                )
                for key in ("0", "1", "2", "4")
            },
        }
    else:
        result["accuracy"] = {"status": "not-evaluated-incomplete-grid"}
    if complete_restoration:
        pooled = _mapping(restoration["pooled"], context="restoration.pooled")
        published = _mapping(
            PUBLISHED_REFERENCE["restoration_pooled"],
            context="published restoration pool",
        )
        result["restoration_pooled"] = {
            key: {
                "counts_match": (
                    _mapping(pooled[key], context=f"pooled.{key}")["restored"]
                    == _mapping(published[key], context=f"published.{key}")["restored"]
                    and _mapping(pooled[key], context=f"pooled.{key}")["denominator"]
                    == _mapping(published[key], context=f"published.{key}")["denominator"]
                ),
                "rounded_rate_matches": _rounded_percent(
                    _mapping(pooled[key], context=f"pooled.{key}")["rate"]
                )
                == _rounded_percent(_mapping(published[key], context=f"published.{key}")["rate"]),
            }
            for key in ("1", "2", "4")
        }
    else:
        result["restoration_pooled"] = {"status": "not-evaluated-incomplete-grid"}
    return result


def build_analysis(
    prepared_runs: Sequence[PreparedEditCountRun],
    cot_swap_runs: Sequence[CotSwapEditCountRun],
    *,
    edit_counts: tuple[int, ...],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Build deterministic per-setting records and the Table 8 summary."""
    prepared_groups, incomplete_prepared = _group_complete(prepared_runs, edit_counts=edit_counts)
    if not prepared_groups:
        raise EditCountSensitivityInputError(
            "no accuracy setting contains every requested edit count"
        )
    cot_groups, incomplete_cot = _group_complete(cot_swap_runs, edit_counts=edit_counts)
    if not cot_groups:
        raise EditCountSensitivityInputError(
            "no restoration setting contains every requested edit count"
        )
    accuracy_rows = tuple(
        _accuracy_row(group, edit_counts=edit_counts)
        for _, group in sorted(prepared_groups.items())
    )
    restoration_rows = tuple(
        _restoration_row(group, edit_counts=edit_counts) for _, group in sorted(cot_groups.items())
    )
    accuracy = _accuracy_summary(accuracy_rows)
    restoration = _restoration_summary(restoration_rows)
    available_accuracy = set(prepared_groups)
    expected_accuracy = set(EXPECTED_ACCURACY_SETTINGS)
    available_restoration = set(cot_groups)
    expected_restoration = set(EXPECTED_RESTORATION_SETTINGS)
    complete_counts = edit_counts == EDIT_COUNTS
    complete_accuracy = (
        complete_counts and available_accuracy == expected_accuracy and not incomplete_prepared
    )
    complete_restoration = (
        complete_counts and available_restoration == expected_restoration and not incomplete_cot
    )
    missing_restoration = [
        setting_id(*key) for key in sorted(expected_restoration - available_restoration)
    ]
    extra_restoration = [
        setting_id(*key) for key in sorted(available_restoration - expected_restoration)
    ]
    missing_accuracy = [setting_id(*key) for key in sorted(expected_accuracy - available_accuracy)]
    extra_accuracy = [setting_id(*key) for key in sorted(available_accuracy - expected_accuracy)]
    limitations: list[str] = [
        "fresh-source-identities-are-not-the-unpublished-historical-table8-identities",
        "fallback-regex-details-remain-legacy-backed-as-documented-by-cot-swap",
    ]
    if not complete_counts:
        limitations.append("requested-edit-count-grid-is-not-one-two-four")
    if not complete_accuracy:
        limitations.append("accuracy-grid-is-not-the-51-setting-final-pdf-grid")
    if not complete_restoration:
        limitations.append("restoration-grid-is-not-the-six-setting-final-pdf-grid")
    status = (
        "fresh-paper-protocol-analysis"
        if complete_accuracy and complete_restoration
        else "partial-valid-analysis"
    )
    coverage = {
        "requested_edit_counts": list(edit_counts),
        "accuracy": {
            "complete_settings": [row["setting_id"] for row in accuracy_rows],
            "complete_setting_count": len(accuracy_rows),
            "expected_setting_count": EXPECTED_ACCURACY_SETTING_COUNT,
            "missing_expected_settings": missing_accuracy,
            "unexpected_settings": extra_accuracy,
            "benchmarks": sorted({str(row["benchmark"]) for row in accuracy_rows}),
            "incomplete_settings_missing_counts": incomplete_prepared,
        },
        "restoration": {
            "complete_settings": [row["setting_id"] for row in restoration_rows],
            "complete_setting_count": len(restoration_rows),
            "expected_setting_count": len(EXPECTED_RESTORATION_SETTINGS),
            "missing_expected_settings": missing_restoration,
            "unexpected_settings": extra_restoration,
            "incomplete_settings_missing_counts": incomplete_cot,
        },
        "complete_accuracy_grid": complete_accuracy,
        "complete_restoration_grid": complete_restoration,
    }
    summary = {
        "schema_version": "edit-count-sensitivity-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "edit-count-sensitivity",
        "analysis_protocol": copy.deepcopy(ANALYSIS_PROTOCOL),
        "analysis_protocol_sha256": ANALYSIS_PROTOCOL_SHA256,
        "coverage": coverage,
        "accuracy": accuracy,
        "restoration": restoration,
        "published_reference": copy.deepcopy(PUBLISHED_REFERENCE),
        "published_comparison": _published_comparison(
            accuracy=accuracy,
            restoration=restoration,
            complete_accuracy=complete_accuracy,
            complete_restoration=complete_restoration,
        ),
        "comparability": {
            "status": status,
            "historical_cohort_identity": False,
            "requirements": {
                "edit_counts_one_two_four": complete_counts,
                "accuracy_51_complete_settings": complete_accuracy,
                "restoration_six_complete_settings": complete_restoration,
                "integer_events_recomputed": True,
                "separate_restoration_denominators": True,
            },
            "limitations": limitations,
        },
    }
    rows = tuple(
        sorted(
            (*accuracy_rows, *restoration_rows),
            key=lambda row: (str(row["setting_id"]), str(row["analysis"])),
        )
    )
    return rows, summary


__all__ = ["build_analysis", "setting_id"]
