"""Integer aggregation for Appendix C/Table 9."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.model_scale_cot_swap.protocol import (
    ANALYSIS_PROTOCOL,
    EXPECTED_MODELS,
    MODEL_LABELS,
    published_reference_payload,
)


def _metric(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _events(record: Mapping[str, object]) -> Mapping[str, object]:
    value = record.get("events")
    if not isinstance(value, Mapping):
        raise ValueError("validated CoT-swap record is missing events")
    return value


def _source_record_count(run: Any) -> int:
    explicit = getattr(run, "source_record_count", None)
    if not isinstance(explicit, int) or isinstance(explicit, bool) or explicit <= 0:
        raise ValueError("source_record_count must be a positive integer")
    return explicit


def _row(run: Any) -> dict[str, object]:
    clean_events = [event for record in run.records if (event := _events(record))["clean_correct"]]
    n_s = len(clean_events)
    both = sum(event["both_changed"] is True for event in clean_events)
    question = sum(event["question_only_changed"] is True for event in clean_events)
    cot = sum(event["cot_only_changed"] is True for event in clean_events)
    n_b = sum(event["restoration_denominator"] is True for event in clean_events)
    restored = sum(event.get("b_to_c_restored") is True for event in clean_events)
    return {
        "schema_version": "model-scale-cot-swap-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "model-scale-cot-swap",
        "model": run.model,
        "label": MODEL_LABELS[run.model],
        "source_records": _source_record_count(run),
        "executed_pairs": len(run.records),
        "n_s": n_s,
        "both": _metric(both, n_s),
        "question_only": _metric(question, n_s),
        "cot_only": _metric(cot, n_s),
        "restoration": _metric(restored, n_b),
    }


def _comparison(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    if tuple(str(row["model"]) for row in rows) != EXPECTED_MODELS:
        return None
    reference = published_reference_payload()
    fields = {
        "n_s": lambda row: row["n_s"],
        "both": lambda row: row["both"]["numerator"],  # type: ignore[index]
        "question_only": lambda row: row["question_only"]["numerator"],  # type: ignore[index]
        "cot_only": lambda row: row["cot_only"]["numerator"],  # type: ignore[index]
        "restored": lambda row: row["restoration"]["numerator"],  # type: ignore[index]
        "n_b": lambda row: row["restoration"]["denominator"],  # type: ignore[index]
    }
    per_model: dict[str, object] = {}
    all_equal = True
    for row in rows:
        model = str(row["model"])
        deltas = {
            field: int(getter(row)) - reference[model][field] for field, getter in fields.items()
        }
        exact = all(delta == 0 for delta in deltas.values())
        all_equal = all_equal and exact
        per_model[model] = {"exact_integer_match": exact, "deltas": deltas}
    return {
        "reference": "final-pdf-appendix-c-table-9",
        "all_integer_cells_match": all_equal,
        "per_model": per_model,
    }


def build_analysis(
    cot_swap_runs: Sequence[Any],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Build ordered model rows and an explicitly labelled coverage summary."""
    by_model: dict[str, Any] = {}
    for run in cot_swap_runs:
        if run.benchmark != "mmlu" or run.model not in EXPECTED_MODELS:
            raise ValueError(
                f"unexpected model-scale CoT-swap setting: {run.model} {run.benchmark}"
            )
        if run.model in by_model:
            raise ValueError(f"duplicate model-scale CoT-swap setting: {run.model}")
        by_model[run.model] = run
    if not by_model:
        raise ValueError("model-scale CoT-swap analysis has no validated settings")
    rows = tuple(_row(by_model[model]) for model in EXPECTED_MODELS if model in by_model)
    present = [str(row["model"]) for row in rows]
    missing = [model for model in EXPECTED_MODELS if model not in by_model]
    complete = not missing
    comparison = _comparison(rows)
    summary: dict[str, object] = {
        "schema_version": "model-scale-cot-swap-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "model-scale-cot-swap",
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "coverage": {
            "complete_grid": complete,
            "expected_models": list(EXPECTED_MODELS),
            "present_models": present,
            "missing_models": missing,
            "unexpected_models": [],
        },
        "models": {str(row["model"]): dict(row) for row in rows},
        "published_reference": published_reference_payload(),
        "published_comparison": comparison,
        "comparability": {
            "status": "complete-paper-grid-comparison" if complete else "partial-valid-analysis",
            "historical_cohort_identity": False,
            "reference_is_acceptance_target": False,
        },
        "limitations": [
            "one-task-mmlu-spot-check",
            "descriptive-counts-without-cross-model-inference",
            "qwen2.5-72b-directional-only-because-published-n-b-is-10",
        ],
    }
    return rows, summary


__all__ = ["build_analysis"]
