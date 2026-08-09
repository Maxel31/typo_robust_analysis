"""Strict source adapter and source-outcome cohort selection for Table 13."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.input_corrector_audit.source import (
    InputCorrectorSource,
    load_input_corrector_source,
)
from typo_cot.experiments.restoration_order_accuracy.planning import (
    RestorationPlan,
    build_restoration_plan,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    GENERATION,
    PAPER_SEED,
    EditGroupingError,
    build_edit_groups,
    canonical_sha256,
)


@dataclass(frozen=True, slots=True)
class SourceExclusion:
    """One source-selected record excluded before fresh generation."""

    sample_id: str
    reason: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RestorationOrderSource:
    """A strict prepared source plus one immutable runnable source cohort."""

    prepared: InputCorrectorSource | object
    records: tuple[dict[str, object], ...]
    plans: tuple[RestorationPlan, ...]
    source_record_count: int
    source_selected_count: int
    separable_count: int
    exclusions: tuple[SourceExclusion, ...]
    selected_sample_ids_sha256: str
    cohort_sha256: str
    limit: int | None

    @property
    def model(self) -> str:
        return str(getattr(self.prepared, "model"))

    @property
    def benchmark(self) -> str:
        return str(getattr(self.prepared, "benchmark"))

    @property
    def model_revision(self) -> str:
        return str(getattr(self.prepared, "model_revision"))

    @property
    def pairs_sha256(self) -> str:
        return str(getattr(self.prepared, "pairs_sha256"))

    @property
    def run_sha256(self) -> str:
        return str(getattr(self.prepared, "run_sha256"))

    @property
    def ordered_sample_ids_sha256(self) -> str:
        return str(getattr(self.prepared, "ordered_sample_ids_sha256"))

    @property
    def dataset_records_sha256(self) -> str:
        return str(getattr(self.prepared, "dataset_records_sha256"))

    def assert_unchanged(self) -> None:
        getattr(self.prepared, "assert_unchanged")()

    def to_dict(self) -> dict[str, object]:
        source_payload = dict(getattr(self.prepared, "to_dict")())
        source_payload.update(
            {
                "input_kind": "completed-prepare-edited-pairs/v1",
                "source_records": self.source_record_count,
                "source_selected": self.source_selected_count,
                "separable": self.separable_count,
                "selected": len(self.records),
                "limit": self.limit,
                "selected_sample_ids_sha256": self.selected_sample_ids_sha256,
                "cohort_sha256": self.cohort_sha256,
                "exclusions": [item.to_dict() for item in self.exclusions],
            }
        )
        return source_payload


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _source_outcome(
    record: Mapping[str, object],
    arm: str,
    *,
    benchmark: str,
) -> bool:
    """Recompute and verify one stored endpoint under the final-PDF rule."""
    sample_id = record.get("sample_id")
    payload = _object(record.get(arm), field=f"{arm} source arm")
    continuation = payload.get("continuation")
    token_count = payload.get("continuation_token_count")
    gold_answer = record.get("gold_answer")
    if not isinstance(continuation, str):
        raise ValueError(f"{sample_id} {arm}.continuation must be a string")
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or not 1 <= token_count <= int(GENERATION["max_new_tokens"])
    ):
        raise ValueError(f"{sample_id} {arm}.continuation_token_count is invalid")
    if not isinstance(gold_answer, str) or not gold_answer:
        raise ValueError(f"{sample_id} gold_answer must be a non-empty string")
    answer = _object(payload.get("answer"), field=f"{arm}.answer")
    recomputed = extract_with_fallback(
        continuation,
        benchmark=benchmark,
        correct_answer=gold_answer,
        allow_positional=token_count < int(GENERATION["max_new_tokens"]),
    )
    expected = {
        "value": recomputed.value,
        "is_extracted": recomputed.is_extracted,
        "is_correct": recomputed.is_correct,
        "method": recomputed.method,
        "primary_method": recomputed.primary_method,
    }
    if any(answer.get(field) != value for field, value in expected.items()):
        raise ValueError(
            f"{sample_id} {arm} source answer differs from final-PDF recomputation"
        )
    return recomputed.is_correct


def _text(record: Mapping[str, object], arm: str) -> str:
    payload = _object(record.get(arm), field=f"{arm} source arm")
    value = payload.get("editable_text")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{arm}.editable_text must be a non-empty string")
    return value


def load_restoration_order_source(
    pairs_path: Path,
    *,
    model: str,
    benchmark: str,
    limit: int | None = None,
    require_paper_cohort_size: bool = True,
) -> RestorationOrderSource:
    """Validate one prepared source and freeze its source-outcome cohort."""
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")
    if type(require_paper_cohort_size) is not bool:
        raise TypeError("require_paper_cohort_size must be boolean")
    prepared = load_input_corrector_source(
        Path(pairs_path),
        model=model,
        benchmark=benchmark,
        require_paper_cohort_size=require_paper_cohort_size,
    )
    source_selected: list[dict[str, object]] = []
    for raw_record in getattr(prepared, "records"):
        record = dict(_object(raw_record, field="prepared pair record"))
        if _source_outcome(record, "clean", benchmark=benchmark) and not _source_outcome(
            record,
            "edited",
            benchmark=benchmark,
        ):
            source_selected.append(record)

    separable_records: list[dict[str, object]] = []
    plans: list[RestorationPlan] = []
    exclusions: list[SourceExclusion] = []
    for record in source_selected:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("prepared pair sample_id must be a non-empty string")
        attempts = record.get("target_attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"{sample_id} target_attempts must be a list")
        try:
            groups = build_edit_groups(
                _text(record, "clean"),
                _text(record, "edited"),
                attempts,
            )
            if len(groups) > 4:
                raise EditGroupingError(
                    "Table 13 source must realize no more than four edit groups"
                )
            plan = build_restoration_plan(
                sample_id=sample_id,
                clean_text=_text(record, "clean"),
                edited_text=_text(record, "edited"),
                groups=groups,
                seed=PAPER_SEED,
            )
        except EditGroupingError as exc:
            exclusions.append(
                SourceExclusion(
                    sample_id=sample_id,
                    reason="inseparable-edit-groups",
                    detail=str(exc),
                )
            )
            continue
        separable_records.append(record)
        plans.append(plan)

    selected_records = separable_records[:limit] if limit is not None else separable_records
    selected_plans = plans[:limit] if limit is not None else plans
    selected_ids = [str(record["sample_id"]) for record in selected_records]
    plan_identities = [plan.to_dict()["sha256"] for plan in selected_plans]
    result = RestorationOrderSource(
        prepared=prepared,
        records=tuple(selected_records),
        plans=tuple(selected_plans),
        source_record_count=len(getattr(prepared, "records")),
        source_selected_count=len(source_selected),
        separable_count=len(separable_records),
        exclusions=tuple(exclusions),
        selected_sample_ids_sha256=canonical_sha256(selected_ids),
        cohort_sha256=canonical_sha256(
            {
                "model": model,
                "benchmark": benchmark,
                "sample_ids": selected_ids,
                "plans": plan_identities,
            }
        ),
        limit=limit,
    )
    result.assert_unchanged()
    return result


__all__ = [
    "RestorationOrderSource",
    "SourceExclusion",
    "load_restoration_order_source",
]
