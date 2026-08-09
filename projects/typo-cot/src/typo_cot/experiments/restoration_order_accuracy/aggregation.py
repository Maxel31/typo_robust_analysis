"""Fail-closed six-setting aggregation and Table 13 rendering."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.restoration_order_accuracy.analysis_integrity import (
    analysis_code_identity,
)
from typo_cot.experiments.restoration_order_accuracy.integrity import (
    implementation_code_identity,
    validate_paper_runtime_environment,
)
from typo_cot.experiments.restoration_order_accuracy.planning import (
    build_restoration_plan,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    ALL_CONDITION_IDS,
    GENERATION,
    INTERMEDIATE_BUDGETS,
    PAPER_BATCH_SIZE,
    PAPER_BENCHMARKS,
    PAPER_BUDGETS,
    PAPER_MODELS,
    PAPER_ORDERS,
    PAPER_SEED,
    PAPER_SOURCE_RECORD_COUNTS,
    PROTOCOL_SHA256,
    EditGroup,
    EditGroupingError,
    build_edit_groups,
    canonical_sha256,
)
from typo_cot.experiments.restoration_order_accuracy.reference import (
    PAPER_PUBLISHED_REFERENCE,
)
from typo_cot.experiments.restoration_order_accuracy.publication import (
    rename_directory_no_replace,
)
from typo_cot.experiments.restoration_order_accuracy.source import (
    load_restoration_order_source,
)
from typo_cot.experiments.restoration_order_accuracy.statistics import (
    exact_mcnemar_p_value,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_RECORDS_NAME = "restoration_order_records.jsonl"
_SUMMARY_NAME = "restoration_order_summary.json"
_TABLE_NAME = "restoration_order_table.json"
_CSV_NAME = "table13_restoration_order.csv"
_MARKDOWN_NAME = "table13_restoration_order.md"
_LATEX_NAME = "table13_restoration_order.tex"
_PRIVATE_DIRECTORY_SUFFIXES = (
    ".restoration-order-work",
    ".restoration-order-publish",
)


class RestorationOrderTableInputError(ValueError):
    """Raised when producer artifacts cannot form one complete Table 13 grid."""


@dataclass(frozen=True, slots=True)
class BuildRestorationOrderTableConfig:
    """CPU-only inputs for the complete Table 13 builder."""

    runs_root: Path
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs_root", Path(self.runs_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True, slots=True)
class RestorationOrderTableResult:
    """Published aggregate artifact paths."""

    table_path: Path
    csv_path: Path
    markdown_path: Path
    latex_path: Path
    run_path: Path
    settings: int


@dataclass(frozen=True, slots=True)
class _Setting:
    model: str
    benchmark: str
    source_model_revision: str
    source_dataset_records_sha256: str
    source_ordered_sample_ids_sha256: str
    run_path: Path
    run_sha256: str
    records_path: Path
    records_sha256: str
    summary_path: Path
    summary_sha256: str
    rows: tuple[dict[str, object], ...]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RestorationOrderTableInputError(f"cannot read artifact: {path}") from exc
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError) as exc:
        raise RestorationOrderTableInputError(f"invalid JSON at {context}: {exc}") from exc


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = _loads(path.read_text(encoding="utf-8"), context=str(path))
    except OSError as exc:
        raise RestorationOrderTableInputError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RestorationOrderTableInputError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RestorationOrderTableInputError(f"{field} must be a JSON object")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RestorationOrderTableInputError(f"{field} must be an integer >= {minimum}")
    return value


def _output_path(directory: Path, metadata: Mapping[str, object], expected: str) -> Path:
    if metadata.get("path") != expected:
        raise RestorationOrderTableInputError(f"output path must be {expected!r}")
    path = directory / expected
    if not path.is_file() or path.resolve().parent != directory.resolve():
        raise RestorationOrderTableInputError(f"required output is missing: {path}")
    digest = metadata.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise RestorationOrderTableInputError(f"output hash is invalid: {path}")
    if _sha256_file(path) != digest:
        raise RestorationOrderTableInputError(f"output hash differs: {path}")
    return path


def _validate_arm(value: object, *, field: str) -> dict[str, object]:
    arm = dict(_mapping(value, field=field))
    for flag in ("is_extracted", "is_correct"):
        if type(arm.get(flag)) is not bool:
            raise RestorationOrderTableInputError(f"{field}.{flag} must be boolean")
    prompt_hash = arm.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or _SHA256.fullmatch(prompt_hash) is None:
        raise RestorationOrderTableInputError(f"{field}.prompt_sha256 must be a SHA-256")
    prompt = arm.get("prompt")
    if not isinstance(prompt, str) or hashlib.sha256(prompt.encode()).hexdigest() != prompt_hash:
        raise RestorationOrderTableInputError(f"{field}.prompt bytes differ from its hash")
    editable_text = arm.get("editable_text")
    editable_hash = arm.get("editable_text_sha256")
    if (
        not isinstance(editable_text, str)
        or not isinstance(editable_hash, str)
        or _SHA256.fullmatch(editable_hash) is None
        or hashlib.sha256(editable_text.encode()).hexdigest() != editable_hash
    ):
        raise RestorationOrderTableInputError(
            f"{field}.editable_text bytes differ from its hash"
        )
    if not isinstance(arm.get("sample_id"), str):
        raise RestorationOrderTableInputError(f"{field}.sample_id must be a string")
    token_ids = arm.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in token_ids
    ):
        raise RestorationOrderTableInputError(f"{field}.token_ids are invalid")
    if arm.get("termination") not in {"eos", "length-cap"}:
        raise RestorationOrderTableInputError(f"{field}.termination is invalid")
    if type(arm.get("individual_retry")) is not bool:
        raise RestorationOrderTableInputError(f"{field}.individual_retry must be boolean")
    for name in ("text", "extracted_answer", "method", "primary_method"):
        if not isinstance(arm.get(name), str):
            raise RestorationOrderTableInputError(f"{field}.{name} must be a string")
    if not arm["method"] or not arm["primary_method"]:
        raise RestorationOrderTableInputError(
            f"{field} extraction methods must be non-empty"
        )
    if arm["is_extracted"] != bool(arm["extracted_answer"]):
        raise RestorationOrderTableInputError(
            f"{field} extraction flag differs from its extracted answer"
        )
    if arm["is_correct"] and not arm["is_extracted"]:
        raise RestorationOrderTableInputError(f"{field} cannot be correct and unextractable")
    return arm


def _validate_generation_score(
    arm: Mapping[str, object],
    *,
    field: str,
    benchmark: str,
    gold_answer: str,
    effective_eos_token_ids: frozenset[int],
) -> None:
    token_ids = tuple(arm["token_ids"])  # type: ignore[arg-type]
    termination = str(arm["termination"])
    if len(token_ids) > int(GENERATION["max_new_tokens"]):
        raise RestorationOrderTableInputError(
            f"{field} exceeds the generation token cap"
        )
    if termination == "eos":
        if (
            token_ids[-1] not in effective_eos_token_ids
            or any(token in effective_eos_token_ids for token in token_ids[:-1])
        ):
            raise RestorationOrderTableInputError(
                f"{field} EOS termination differs from its generated token IDs"
            )
        allow_positional = True
    else:
        if len(token_ids) != int(GENERATION["max_new_tokens"]) or any(
            token in effective_eos_token_ids for token in token_ids
        ):
            raise RestorationOrderTableInputError(
                f"{field} length-cap termination differs from its generated token IDs"
            )
        allow_positional = False
    extraction = extract_with_fallback(
        str(arm["text"]),
        benchmark=benchmark,
        correct_answer=gold_answer,
        allow_positional=allow_positional,
    )
    expected = {
        "extracted_answer": extraction.value,
        "is_extracted": extraction.is_extracted,
        "is_correct": extraction.is_correct,
        "method": extraction.method,
        "primary_method": extraction.primary_method,
    }
    if any(arm.get(name) != value for name, value in expected.items()):
        raise RestorationOrderTableInputError(
            f"{field} extracted answer or correctness differs from fresh scoring"
        )


def _group_span(
    value: object,
    *,
    field: str,
    text: str,
) -> tuple[int, int]:
    span = _mapping(value, field=field)
    start = _integer(span.get("start"), field=f"{field}.start")
    end = _integer(span.get("end"), field=f"{field}.end")
    if start > end or end > len(text):
        raise RestorationOrderTableInputError(f"{field} is outside its endpoint text")
    return start, end


def _edit_group(
    value: object,
    *,
    field: str,
    expected_index: int,
    clean_text: str,
    edited_text: str,
) -> EditGroup:
    payload = dict(_mapping(value, field=field))
    index = _integer(payload.get("index"), field=f"{field}.index")
    if index != expected_index:
        raise RestorationOrderTableInputError(
            f"{field}.index must preserve left-to-right contiguous order"
        )
    clean_start, clean_end = _group_span(
        payload.get("clean_span"), field=f"{field}.clean_span", text=clean_text
    )
    edited_start, edited_end = _group_span(
        payload.get("edited_span"), field=f"{field}.edited_span", text=edited_text
    )
    if clean_start == clean_end and edited_start == edited_end:
        raise RestorationOrderTableInputError(f"{field} cannot contain two empty spans")
    clean_group_text = payload.get("clean_text")
    edited_group_text = payload.get("edited_text")
    if (
        not isinstance(clean_group_text, str)
        or clean_group_text != clean_text[clean_start:clean_end]
        or not isinstance(edited_group_text, str)
        or edited_group_text != edited_text[edited_start:edited_end]
    ):
        raise RestorationOrderTableInputError(
            f"{field} text differs from its endpoint span"
        )
    selection_rank = _integer(
        payload.get("selection_rank"), field=f"{field}.selection_rank", minimum=1
    )
    target_token_index = _integer(
        payload.get("target_token_index"), field=f"{field}.target_token_index"
    )
    relevance = payload.get("relevance")
    if (
        not isinstance(relevance, (int, float))
        or isinstance(relevance, bool)
        or not math.isfinite(float(relevance))
    ):
        raise RestorationOrderTableInputError(f"{field}.relevance must be finite")
    group = EditGroup(
        index=index,
        clean_start=clean_start,
        clean_end=clean_end,
        edited_start=edited_start,
        edited_end=edited_end,
        clean_text=clean_group_text,
        edited_text=edited_group_text,
        selection_rank=selection_rank,
        target_token_index=target_token_index,
        relevance=float(relevance),
    )
    if payload != group.to_dict():
        raise RestorationOrderTableInputError(f"{field} schema or values differ")
    return group


def _validate_prompt_context(
    value: object,
    *,
    sample_id: str,
    arms: Mapping[str, Mapping[str, object]],
) -> None:
    context = _mapping(value, field=f"{sample_id}.prompt_context")
    prefix_length = _integer(
        context.get("prefix_length"), field=f"{sample_id}.prompt_context.prefix_length"
    )
    suffix_length = _integer(
        context.get("suffix_length"), field=f"{sample_id}.prompt_context.suffix_length"
    )
    prefix_hash = context.get("prefix_sha256")
    suffix_hash = context.get("suffix_sha256")
    if (
        not isinstance(prefix_hash, str)
        or _SHA256.fullmatch(prefix_hash) is None
        or not isinstance(suffix_hash, str)
        or _SHA256.fullmatch(suffix_hash) is None
    ):
        raise RestorationOrderTableInputError(
            f"{sample_id} prompt-context hashes must be SHA-256 values"
        )
    expected_prefix: str | None = None
    expected_suffix: str | None = None
    for condition_id in ALL_CONDITION_IDS:
        arm = arms[condition_id]
        prompt = str(arm["prompt"])
        editable_text = str(arm["editable_text"])
        if prefix_length + suffix_length > len(prompt):
            raise RestorationOrderTableInputError(
                f"{sample_id} prompt context exceeds {condition_id}"
            )
        editable_end = len(prompt) - suffix_length
        prefix = prompt[:prefix_length]
        suffix = prompt[editable_end:]
        if prompt[prefix_length:editable_end] != editable_text:
            raise RestorationOrderTableInputError(
                f"{sample_id} prompt does not contain the exact {condition_id} editable text"
            )
        if (
            hashlib.sha256(prefix.encode()).hexdigest() != prefix_hash
            or hashlib.sha256(suffix.encode()).hexdigest() != suffix_hash
        ):
            raise RestorationOrderTableInputError(
                f"{sample_id} prompt context differs from its hashes"
            )
        if expected_prefix is None:
            expected_prefix, expected_suffix = prefix, suffix
        elif prefix != expected_prefix or suffix != expected_suffix:
            raise RestorationOrderTableInputError(
                f"{sample_id} prompt context differs across conditions"
            )


def _validate_record_plan(
    row: Mapping[str, object],
    *,
    sample_id: str,
    realized: int,
    arms: Mapping[str, Mapping[str, object]],
) -> None:
    clean_text = str(arms["clean:k4"]["editable_text"])
    edited_text = str(arms["edited:k0"]["editable_text"])
    edit_groups = row.get("edit_groups")
    if not isinstance(edit_groups, list) or len(edit_groups) != realized:
        raise RestorationOrderTableInputError("record edit_groups count differs")
    groups = tuple(
        _edit_group(
            group,
            field=f"{sample_id}.edit_groups[{index}]",
            expected_index=index,
            clean_text=clean_text,
            edited_text=edited_text,
        )
        for index, group in enumerate(edit_groups)
    )
    try:
        canonical_groups = build_edit_groups(
            clean_text,
            edited_text,
            [
                {
                    "target_token_index": group.target_token_index,
                    "selection_rank": group.selection_rank,
                    "relevance": group.relevance,
                }
                for group in groups
            ],
        )
    except EditGroupingError as exc:
        raise RestorationOrderTableInputError(
            f"{sample_id} edit groups differ from canonical reconstruction: {exc}"
        ) from exc
    if canonical_groups != groups:
        raise RestorationOrderTableInputError(
            f"{sample_id} edit groups differ from canonical reconstruction"
        )
    try:
        plan = build_restoration_plan(
            sample_id=sample_id,
            clean_text=clean_text,
            edited_text=edited_text,
            groups=groups,
            seed=PAPER_SEED,
        )
    except (TypeError, ValueError) as exc:
        raise RestorationOrderTableInputError(
            f"{sample_id} restoration plan is invalid: {exc}"
        ) from exc
    expected_orders = {name: list(indices) for name, indices in plan.order_indices}
    if dict(_mapping(row.get("order_group_indices"), field="record order_group_indices")) != expected_orders:
        raise RestorationOrderTableInputError(
            f"{sample_id} restoration order permutations differ from the protocol"
        )
    conditions = {condition.condition_id: condition for condition in plan.conditions}
    for condition_id in ALL_CONDITION_IDS:
        condition = conditions[condition_id]
        arm = arms[condition_id]
        if (
            arm.get("order") != condition.order
            or arm.get("budget") != condition.budget
            or arm.get("restored_group_indices")
            != list(condition.restored_group_indices)
            or arm.get("editable_text") != condition.text
        ):
            raise RestorationOrderTableInputError(
                f"{sample_id} arm differs from its restoration plan for {condition_id}"
            )
    plan_sha256 = row.get("plan_sha256")
    if plan_sha256 != plan.to_dict()["sha256"]:
        raise RestorationOrderTableInputError(
            f"{sample_id} plan SHA-256 differs from its reconstructed plan"
        )
    _validate_prompt_context(
        row.get("prompt_context"), sample_id=sample_id, arms=arms
    )


def _load_records(
    path: Path,
    *,
    model: str,
    benchmark: str,
    expected_count: int,
    effective_eos_token_ids: frozenset[int],
) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RestorationOrderTableInputError(f"cannot read records: {path}") from exc
    if not lines or any(not line for line in lines):
        raise RestorationOrderTableInputError(f"records JSONL is empty or sparse: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        value = _loads(line, context=f"{path}:{line_number}")
        row = dict(_mapping(value, field=f"record {line_number}"))
        if row.get("schema_version") != "restoration-order-record/v1":
            raise RestorationOrderTableInputError("record schema is unsupported")
        if row.get("model") != model or row.get("benchmark") != benchmark:
            raise RestorationOrderTableInputError("record setting identity differs")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RestorationOrderTableInputError("record sample_id must be non-empty")
        realized = _integer(
            row.get("realized_edit_groups"), field="realized_edit_groups", minimum=1
        )
        if realized > 4:
            raise RestorationOrderTableInputError("realized_edit_groups must not exceed four")
        gold_answer = row.get("gold_answer")
        if not isinstance(gold_answer, str):
            raise RestorationOrderTableInputError("record gold_answer must be a string")
        source_record_sha256 = row.get("source_record_sha256")
        if (
            not isinstance(source_record_sha256, str)
            or _SHA256.fullmatch(source_record_sha256) is None
        ):
            raise RestorationOrderTableInputError(
                "record source_record_sha256 must be a SHA-256"
            )
        selection = _mapping(row.get("source_selection"), field="source_selection")
        if selection != {"clean_correct": True, "edited_wrong": True, "separable": True}:
            raise RestorationOrderTableInputError("record source selection evidence differs")
        arms = _mapping(row.get("arms"), field=f"{sample_id}.arms")
        if set(arms) != set(ALL_CONDITION_IDS) or len(arms) != len(ALL_CONDITION_IDS):
            raise RestorationOrderTableInputError(f"{sample_id} arm grid differs")
        validated_arms = {
            condition_id: _validate_arm(
                arms[condition_id], field=f"{sample_id}.arms.{condition_id}"
            )
            for condition_id in ALL_CONDITION_IDS
        }
        for condition_id, arm in validated_arms.items():
            if arm["sample_id"] != sample_id:
                raise RestorationOrderTableInputError(
                    f"{sample_id} arm sample identity differs"
                )
            _validate_generation_score(
                arm,
                field=f"{sample_id}.arms.{condition_id}",
                benchmark=benchmark,
                gold_answer=gold_answer,
                effective_eos_token_ids=effective_eos_token_ids,
            )
        row["arms"] = validated_arms
        _validate_record_plan(
            row,
            sample_id=sample_id,
            realized=realized,
            arms=validated_arms,
        )
        rows.append(row)
    if len(rows) != expected_count:
        raise RestorationOrderTableInputError("records count differs from the run manifest")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise RestorationOrderTableInputError("record sample IDs must be unique and sorted")
    return tuple(rows)


def _validate_rows_against_source(
    rows: Sequence[Mapping[str, object]],
    *,
    validated_source: object,
) -> None:
    source_records = tuple(getattr(validated_source, "records"))
    source_plans = tuple(getattr(validated_source, "plans"))
    if len(rows) != len(source_records) or len(rows) != len(source_plans):
        raise RestorationOrderTableInputError(
            "producer rows differ from the reloaded source cohort"
        )
    for row, source_record, source_plan in zip(
        rows,
        source_records,
        source_plans,
        strict=True,
    ):
        source_record = _mapping(source_record, field="reloaded source record")
        sample_id = str(row["sample_id"])
        if source_record.get("sample_id") != sample_id:
            raise RestorationOrderTableInputError(
                "producer sample order differs from the reloaded source cohort"
            )
        if row.get("source_record_sha256") != canonical_sha256(source_record):
            raise RestorationOrderTableInputError(
                f"{sample_id} source record identity differs from the reloaded source"
            )
        if row.get("gold_answer") != source_record.get("gold_answer"):
            raise RestorationOrderTableInputError(
                f"{sample_id} gold answer differs from the reloaded source"
            )
        plan_payload = getattr(source_plan, "to_dict")()
        if not isinstance(plan_payload, Mapping):
            raise RestorationOrderTableInputError(
                f"{sample_id} reloaded source plan is invalid"
            )
        if (
            row.get("plan_sha256") != plan_payload.get("sha256")
            or row.get("edit_groups") != plan_payload.get("groups")
            or row.get("order_group_indices")
            != plan_payload.get("order_group_indices")
        ):
            raise RestorationOrderTableInputError(
                f"{sample_id} restoration plan differs from the reloaded source"
            )
        arms = _mapping(row.get("arms"), field=f"{sample_id}.arms")
        for source_arm, condition_id in (
            ("clean", "clean:k4"),
            ("edited", "edited:k0"),
        ):
            endpoint = _mapping(
                source_record.get(source_arm),
                field=f"{sample_id} reloaded source {source_arm}",
            )
            arm = _mapping(arms.get(condition_id), field=f"{sample_id}.{condition_id}")
            if (
                arm.get("editable_text") != endpoint.get("editable_text")
                or arm.get("prompt") != endpoint.get("prompt")
            ):
                raise RestorationOrderTableInputError(
                    f"{sample_id} {condition_id} prompt differs from the reloaded source"
                )


def _validate_summary(
    path: Path,
    *,
    model: str,
    benchmark: str,
    rows: Sequence[Mapping[str, object]],
    source_records: int,
    source_flip: int,
    separable: int,
) -> None:
    summary = _load_json(path)
    if summary.get("schema_version") != "restoration-order-summary/v1":
        raise RestorationOrderTableInputError("summary schema is unsupported")
    if summary.get("model") != model or summary.get("benchmark") != benchmark:
        raise RestorationOrderTableInputError("summary setting identity differs")
    cohort = _mapping(summary.get("cohort"), field="summary cohort")
    if (
        cohort.get("limited") is not False
        or cohort.get("selected") != len(rows)
        or cohort.get("source_records") != source_records
        or cohort.get("source_flip") != source_flip
        or cohort.get("separable") != separable
    ):
        raise RestorationOrderTableInputError("summary cohort is limited or inconsistent")
    conditions = _mapping(summary.get("conditions"), field="summary conditions")
    if set(conditions) != set(ALL_CONDITION_IDS) or len(conditions) != len(ALL_CONDITION_IDS):
        raise RestorationOrderTableInputError("summary condition grid differs")
    for condition_id in ALL_CONDITION_IDS:
        stored = _mapping(conditions[condition_id], field=f"summary {condition_id}")
        correct = sum(
            int(bool(_mapping(row["arms"], field="row arms")[condition_id]["is_correct"]))  # type: ignore[index]
            for row in rows
        )
        extracted = sum(
            int(
                bool(
                    _mapping(row["arms"], field="row arms")[condition_id][
                        "is_extracted"
                    ]
                )
            )
            for row in rows
        )
        stored_correct = _integer(
            stored.get("correct"), field=f"summary {condition_id}.correct"
        )
        stored_extracted = _integer(
            stored.get("extracted"), field=f"summary {condition_id}.extracted"
        )
        stored_total = _integer(
            stored.get("total"), field=f"summary {condition_id}.total", minimum=1
        )
        if (
            stored_correct != correct
            or stored_extracted != extracted
            or stored_total != len(rows)
        ):
            raise RestorationOrderTableInputError(
                f"summary integer count differs for {condition_id}"
            )
        accuracy = stored.get("accuracy")
        expected = correct / len(rows)
        if (
            not isinstance(accuracy, (int, float))
            or isinstance(accuracy, bool)
            or not math.isfinite(float(accuracy))
            or float(accuracy) != expected
        ):
            raise RestorationOrderTableInputError(
                f"summary accuracy differs for {condition_id}"
            )


def _validate_retry_ledger(
    value: object,
    *,
    rows: Sequence[Mapping[str, object]],
) -> None:
    ledger = _mapping(value, field="producer generation retries")
    stored_batches = _integer(
        ledger.get("individual_retry_batches"), field="retry batch count"
    )
    stored_items = _integer(
        ledger.get("individual_retry_items"), field="retry item count"
    )
    batches = ledger.get("batches")
    if not isinstance(batches, list) or len(batches) != stored_batches:
        raise RestorationOrderTableInputError("producer retry batch ledger differs")
    ordered_ids = [str(row["sample_id"]) for row in rows]
    retried: set[tuple[str, str]] = set()
    seen_batches: set[tuple[str, int]] = set()
    for ledger_index, raw_batch in enumerate(batches):
        batch = _mapping(raw_batch, field=f"retry batches[{ledger_index}]")
        condition_id = batch.get("condition_id")
        if condition_id not in ALL_CONDITION_IDS:
            raise RestorationOrderTableInputError("producer retry condition differs")
        batch_index = _integer(
            batch.get("batch_index"), field=f"retry batches[{ledger_index}].batch_index"
        )
        batch_identity = (str(condition_id), batch_index)
        if batch_identity in seen_batches:
            raise RestorationOrderTableInputError("producer retry batch is duplicated")
        seen_batches.add(batch_identity)
        sample_ids = batch.get("sample_ids")
        start = batch_index * PAPER_BATCH_SIZE
        expected_ids = ordered_ids[start : start + PAPER_BATCH_SIZE]
        if not isinstance(sample_ids, list) or sample_ids != expected_ids or not sample_ids:
            raise RestorationOrderTableInputError(
                "producer retry batch sample identities differ"
            )
        retried.update((str(condition_id), str(sample_id)) for sample_id in sample_ids)
    flagged = {
        (condition_id, str(row["sample_id"]))
        for row in rows
        for condition_id, arm in _mapping(row["arms"], field="record arms").items()
        if _mapping(arm, field="record arm").get("individual_retry") is True
    }
    if retried != flagged or stored_items != len(retried):
        raise RestorationOrderTableInputError(
            "producer retry ledger differs from per-arm provenance"
        )


def _validate_run(run_path: Path) -> _Setting:
    initial_run_sha256 = _sha256_file(run_path)
    manifest = _load_json(run_path)
    if manifest.get("schema_version") != "restoration-order-run/v1":
        raise RestorationOrderTableInputError(f"producer schema is unsupported: {run_path}")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise RestorationOrderTableInputError("producer paper fingerprint differs")
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RestorationOrderTableInputError("producer protocol fingerprint differs")
    if manifest.get("operation") != "restoration-order-accuracy":
        raise RestorationOrderTableInputError("producer operation differs")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise RestorationOrderTableInputError("producer run ID is missing or invalid")
    if manifest.get("status") != "completed":
        raise RestorationOrderTableInputError("producer run must be completed")
    arguments = _mapping(manifest.get("arguments"), field="producer arguments")
    model = arguments.get("model")
    benchmark = arguments.get("benchmark")
    if model not in PAPER_MODELS or benchmark not in PAPER_BENCHMARKS:
        raise RestorationOrderTableInputError("producer is outside the six-setting paper grid")
    if arguments.get("orders") != list(PAPER_ORDERS):
        raise RestorationOrderTableInputError("producer order grid differs")
    if arguments.get("budgets") != list(PAPER_BUDGETS):
        raise RestorationOrderTableInputError("producer budget grid differs")
    if arguments.get("seed") != PAPER_SEED:
        raise RestorationOrderTableInputError("producer seed differs")
    if arguments.get("batch_size") != PAPER_BATCH_SIZE:
        raise RestorationOrderTableInputError("producer batch size differs")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise RestorationOrderTableInputError("producer limit must be null for Table 13")

    source = _mapping(manifest.get("source"), field="producer source")
    if source.get("model") != model or source.get("benchmark") != benchmark:
        raise RestorationOrderTableInputError("producer source setting differs")
    if source.get("input_kind") != "completed-prepare-edited-pairs/v1":
        raise RestorationOrderTableInputError("producer source kind differs")
    pairs_argument = arguments.get("pairs")
    if not isinstance(pairs_argument, str) or not Path(pairs_argument).is_absolute():
        raise RestorationOrderTableInputError(
            "producer pairs argument must be an absolute source path"
        )
    try:
        validated_source = load_restoration_order_source(
            Path(pairs_argument),
            model=str(model),
            benchmark=str(benchmark),
            limit=None,
        )
        reloaded_source_payload = getattr(validated_source, "to_dict")()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RestorationOrderTableInputError(
            f"cannot revalidate producer pairs source: {exc}"
        ) from exc
    if not isinstance(reloaded_source_payload, Mapping) or dict(
        reloaded_source_payload
    ) != dict(source):
        raise RestorationOrderTableInputError(
            "producer source manifest differs from the reloaded pairs source"
        )
    expected_source_records = PAPER_SOURCE_RECORD_COUNTS[str(benchmark)]
    if (
        source.get("records") != expected_source_records
        or source.get("source_records") != expected_source_records
    ):
        raise RestorationOrderTableInputError("producer source is not the complete paper cohort")
    if source.get("limit") is not None:
        raise RestorationOrderTableInputError("producer source limit must be null")
    source_selected = _integer(
        source.get("source_selected"), field="source source_selected"
    )
    separable = _integer(source.get("separable"), field="source separable")
    selected = _integer(source.get("selected"), field="source selected", minimum=1)
    if not source_selected >= separable >= selected:
        raise RestorationOrderTableInputError("producer source cohort counts are inconsistent")
    if selected != separable:
        raise RestorationOrderTableInputError(
            "complete producer must include every separable source item"
        )
    exclusions = source.get("exclusions")
    if not isinstance(exclusions, list) or len(exclusions) != source_selected - separable:
        raise RestorationOrderTableInputError("producer source exclusions are inconsistent")
    exclusion_ids: list[str] = []
    for index, raw_exclusion in enumerate(exclusions):
        exclusion = _mapping(raw_exclusion, field=f"source exclusions[{index}]")
        sample_id = exclusion.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RestorationOrderTableInputError(
                "producer source exclusion sample_id must be non-empty"
            )
        if exclusion.get("reason") != "inseparable-edit-groups" or not isinstance(
            exclusion.get("detail"), str
        ):
            raise RestorationOrderTableInputError(
                "producer source exclusion reason or detail differs"
            )
        exclusion_ids.append(sample_id)
    if len(exclusion_ids) != len(set(exclusion_ids)):
        raise RestorationOrderTableInputError(
            "producer source exclusion sample IDs must be unique"
        )
    for field in (
        "pairs_sha256",
        "run_sha256",
        "ordered_sample_ids_sha256",
        "dataset_records_sha256",
        "selected_sample_ids_sha256",
        "cohort_sha256",
    ):
        digest = source.get(field)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RestorationOrderTableInputError(f"producer source {field} is invalid")
    revision = source.get("model_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise RestorationOrderTableInputError("producer source revision is invalid")
    runtime = _mapping(manifest.get("runtime"), field="producer runtime")
    if (
        runtime.get("operation") != "restoration-order-accuracy"
        or runtime.get("runtime") != "HuggingFaceRestorationRuntime"
        or runtime.get("model") != model
    ):
        raise RestorationOrderTableInputError("producer runtime identity differs")
    if runtime.get("requested_revision") != revision:
        raise RestorationOrderTableInputError(
            "producer requested runtime revision differs from source"
        )
    if runtime.get("model_revision") != revision or runtime.get("tokenizer_revision") != revision:
        raise RestorationOrderTableInputError("producer runtime revision differs from source")
    if runtime.get("model_revision_source") not in {
        "model-config-metadata",
        "explicit-load-revision",
    } or runtime.get("tokenizer_revision_source") not in {
        "tokenizer-init-metadata",
        "explicit-load-revision",
    }:
        raise RestorationOrderTableInputError(
            "producer runtime revision provenance differs"
        )
    if runtime.get("dtype") != "bfloat16":
        raise RestorationOrderTableInputError("producer runtime dtype differs")
    if runtime.get("cuda_visible_devices") != arguments.get("gpu_id"):
        raise RestorationOrderTableInputError(
            "producer runtime GPU differs from its arguments"
        )
    if runtime.get("answer_extraction") != (
        "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1"
    ):
        raise RestorationOrderTableInputError("producer answer extraction differs")
    if runtime.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RestorationOrderTableInputError("producer runtime protocol differs")
    if runtime.get("benchmark_extractor") != benchmark:
        raise RestorationOrderTableInputError("producer runtime benchmark extractor differs")
    if runtime.get("historical_extraction_difference") != (
        "submitted-table13-primary-only"
    ):
        raise RestorationOrderTableInputError(
            "producer runtime historical extraction provenance differs"
        )
    eos_ids = runtime.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in eos_ids
        )
        or eos_ids != sorted(set(eos_ids))
        or runtime.get("effective_eos_token_ids_source")
        not in {"model-generation-config", "tokenizer-fallback"}
    ):
        raise RestorationOrderTableInputError("producer runtime EOS provenance differs")
    generation = _mapping(runtime.get("generation"), field="producer generation")
    if generation != GENERATION:
        raise RestorationOrderTableInputError("producer runtime generation protocol differs")
    for field in (
        "base_generation_config_sha256",
        "effective_generation_config_sha256",
    ):
        digest = runtime.get(field)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RestorationOrderTableInputError(
                f"producer runtime {field} is invalid"
            )
    implementation = implementation_code_identity()
    if manifest.get("implementation_code") != implementation:
        raise RestorationOrderTableInputError("producer implementation integrity differs")
    if runtime.get("implementation_code") != implementation:
        raise RestorationOrderTableInputError("producer runtime implementation differs")
    try:
        validate_paper_runtime_environment(runtime)
    except ValueError as exc:
        raise RestorationOrderTableInputError(str(exc)) from exc

    counts = _mapping(manifest.get("counts"), field="producer counts")
    record_count = _integer(counts.get("records"), field="producer records", minimum=1)
    if counts.get("selected") != record_count:
        raise RestorationOrderTableInputError("producer selected and record counts differ")
    if source.get("selected") != record_count:
        raise RestorationOrderTableInputError("producer source selection count differs")
    expected_batches = len(ALL_CONDITION_IDS) * (
        (record_count + PAPER_BATCH_SIZE - 1) // PAPER_BATCH_SIZE
    )
    progress = _mapping(manifest.get("progress"), field="producer progress")
    if (
        progress.get("completed_batches") != expected_batches
        or progress.get("total_batches") != expected_batches
        or manifest.get("failure") is not None
    ):
        raise RestorationOrderTableInputError(
            "producer completed progress or failure state differs"
        )
    outputs = _mapping(manifest.get("outputs"), field="producer outputs")
    records_meta = _mapping(outputs.get("records"), field="producer records output")
    summary_meta = _mapping(outputs.get("summary"), field="producer summary output")
    if records_meta.get("records") != record_count:
        raise RestorationOrderTableInputError("producer output record count differs")
    directory = run_path.parent
    records_path = _output_path(directory, records_meta, _RECORDS_NAME)
    summary_path = _output_path(directory, summary_meta, _SUMMARY_NAME)
    rows = _load_records(
        records_path,
        model=str(model),
        benchmark=str(benchmark),
        expected_count=record_count,
        effective_eos_token_ids=frozenset(eos_ids),
    )
    _validate_rows_against_source(rows, validated_source=validated_source)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if source.get("selected_sample_ids_sha256") != canonical_sha256(sample_ids):
        raise RestorationOrderTableInputError(
            "producer selected sample identity differs from its records"
        )
    expected_cohort_sha256 = canonical_sha256(
        {
            "model": model,
            "benchmark": benchmark,
            "sample_ids": sample_ids,
            "plans": [row["plan_sha256"] for row in rows],
        }
    )
    if source.get("cohort_sha256") != expected_cohort_sha256:
        raise RestorationOrderTableInputError(
            "producer cohort identity differs from its records"
        )
    _validate_retry_ledger(manifest.get("generation_retries"), rows=rows)
    _validate_summary(
        summary_path,
        model=str(model),
        benchmark=str(benchmark),
        rows=rows,
        source_records=expected_source_records,
        source_flip=source_selected,
        separable=separable,
    )
    if _sha256_file(run_path) != initial_run_sha256:
        raise RestorationOrderTableInputError("producer run changed during validation")
    if _sha256_file(records_path) != records_meta.get("sha256"):
        raise RestorationOrderTableInputError("producer records changed during validation")
    if _sha256_file(summary_path) != summary_meta.get("sha256"):
        raise RestorationOrderTableInputError("producer summary changed during validation")
    try:
        getattr(validated_source, "assert_unchanged")()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RestorationOrderTableInputError(
            f"producer pairs source changed during table validation: {exc}"
        ) from exc
    return _Setting(
        model=str(model),
        benchmark=str(benchmark),
        source_model_revision=str(source["model_revision"]),
        source_dataset_records_sha256=str(source["dataset_records_sha256"]),
        source_ordered_sample_ids_sha256=str(source["ordered_sample_ids_sha256"]),
        run_path=run_path,
        run_sha256=initial_run_sha256,
        records_path=records_path,
        records_sha256=_sha256_file(records_path),
        summary_path=summary_path,
        summary_sha256=_sha256_file(summary_path),
        rows=rows,
    )


def _validate_cross_setting_source_identities(settings: Sequence[_Setting]) -> None:
    for model in PAPER_MODELS:
        revisions = {
            setting.source_model_revision for setting in settings if setting.model == model
        }
        if len(revisions) != 1:
            raise RestorationOrderTableInputError(
                f"source model revision differs across benchmarks for {model}"
            )
    for benchmark in PAPER_BENCHMARKS:
        identities = {
            (
                setting.source_dataset_records_sha256,
                setting.source_ordered_sample_ids_sha256,
            )
            for setting in settings
            if setting.benchmark == benchmark
        }
        if len(identities) != 1:
            raise RestorationOrderTableInputError(
                f"source dataset/sample identity differs across models for {benchmark}"
            )


def _discover(root: Path) -> tuple[_Setting, ...]:
    if not root.is_dir():
        raise RestorationOrderTableInputError(f"runs root is not a directory: {root}")
    candidates: list[Path] = []
    for path in sorted(root.rglob("run.json")):
        relative_parts = path.relative_to(root).parts[:-1]
        if any(
            part.endswith(_PRIVATE_DIRECTORY_SUFFIXES) for part in relative_parts
        ):
            continue
        try:
            payload = _load_json(path)
        except RestorationOrderTableInputError:
            raise
        if payload.get("operation") == "restoration-order-accuracy":
            candidates.append(path)
    if len(candidates) != 6:
        raise RestorationOrderTableInputError(
            f"Table 13 requires exactly six producer runs; found {len(candidates)}"
        )
    settings = tuple(_validate_run(path) for path in candidates)
    identities = [(setting.model, setting.benchmark) for setting in settings]
    expected = {(model, benchmark) for model in PAPER_MODELS for benchmark in PAPER_BENCHMARKS}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        missing = sorted(expected - set(identities))
        extras = sorted(set(identities) - expected)
        raise RestorationOrderTableInputError(
            f"Table 13 six-setting grid differs; missing={missing}, extras={extras}"
        )
    _validate_cross_setting_source_identities(settings)
    return tuple(sorted(settings, key=lambda item: (item.model, item.benchmark)))


def _aggregate(settings: Sequence[_Setting]) -> dict[str, object]:
    paired_rows: list[tuple[str, str, str, Mapping[str, object]]] = []
    seen: set[tuple[str, str, str]] = set()
    per_setting: list[dict[str, object]] = []
    for setting in settings:
        setting_counts: dict[str, object] = {}
        for condition_id in ALL_CONDITION_IDS:
            correct = sum(
                int(bool(_mapping(row["arms"], field="row arms")[condition_id]["is_correct"]))  # type: ignore[index]
                for row in setting.rows
            )
            setting_counts[condition_id] = {
                "correct": correct,
                "total": len(setting.rows),
                "accuracy": correct / len(setting.rows),
            }
        per_setting.append(
            {
                "model": setting.model,
                "benchmark": setting.benchmark,
                "items": len(setting.rows),
                "conditions": setting_counts,
            }
        )
        for row in setting.rows:
            identity = (setting.model, setting.benchmark, str(row["sample_id"]))
            if identity in seen:
                raise RestorationOrderTableInputError(
                    f"duplicate model/task/sample identity: {identity!r}"
                )
            seen.add(identity)
            paired_rows.append((*identity, _mapping(row["arms"], field="row arms")))

    total = len(paired_rows)
    conditions: dict[str, object] = {}
    for condition_id in ALL_CONDITION_IDS:
        correct = sum(
            int(bool(arms[condition_id]["is_correct"]))  # type: ignore[index]
            for _model, _benchmark, _sample, arms in paired_rows
        )
        conditions[condition_id] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }
    paired_tests: dict[str, object] = {}
    for budget in INTERMEDIATE_BUDGETS:
        high_id = f"high-relevance-first:k{budget}"
        random_id = f"seeded-random:k{budget}"
        high_only = 0
        random_only = 0
        both_correct = 0
        both_wrong = 0
        for _model, _benchmark, _sample, arms in paired_rows:
            high = bool(arms[high_id]["is_correct"])  # type: ignore[index]
            random_value = bool(arms[random_id]["is_correct"])  # type: ignore[index]
            if high and not random_value:
                high_only += 1
            elif random_value and not high:
                random_only += 1
            elif high:
                both_correct += 1
            else:
                both_wrong += 1
        paired_tests[f"k{budget}"] = {
            "method": "two-sided-exact-mcnemar-binomial",
            "comparison": "high-relevance-first-vs-seeded-random",
            "pooling": "micro-by-model-task-sample-identity",
            "high_only": high_only,
            "random_only": random_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "discordant": high_only + random_only,
            "p_value": exact_mcnemar_p_value(high_only, random_only),
            "adjustment": None,
        }
    return {
        "schema_version": "restoration-order-table/v1",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "pooling": "micro-by-model-task-sample-identity",
        "cohort": {
            "kind": "fresh-final-pdf-protocol-replication",
            "settings": len(settings),
            "items": total,
        },
        "conditions": conditions,
        "paired_tests": paired_tests,
        "per_setting": per_setting,
        "paper_published_reference": PAPER_PUBLISHED_REFERENCE,
    }


def _percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _table_rows(table: Mapping[str, object]) -> tuple[tuple[str, ...], ...]:
    conditions = _mapping(table.get("conditions"), field="table conditions")
    rows: list[tuple[str, ...]] = []
    labels = {
        "high-relevance-first": "High relevance first",
        "seeded-random": "Seeded random",
        "low-relevance-first": "Low relevance first",
    }
    for order in PAPER_ORDERS:
        accuracies = [
            float(_mapping(conditions["edited:k0"], field="edited endpoint")["accuracy"]),
            *(
                float(
                    _mapping(
                        conditions[f"{order}:k{budget}"],
                        field=f"{order}:k{budget}",
                    )["accuracy"]
                )
                for budget in INTERMEDIATE_BUDGETS
            ),
            float(_mapping(conditions["clean:k4"], field="clean endpoint")["accuracy"]),
        ]
        rows.append((labels[order], *(_percentage(value) for value in accuracies)))
    return tuple(rows)


def _render_csv(table: Mapping[str, object]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    cohort = _mapping(table.get("cohort"), field="table cohort")
    writer.writerow(
        (
            "Result kind",
            "Cohort n",
            "Restoration order",
            "Zero restored",
            "One",
            "Two",
            "Three",
            "All restored",
        )
    )
    writer.writerows(
        (cohort["kind"], cohort["items"], *row) for row in _table_rows(table)
    )
    return stream.getvalue()


def _render_markdown(table: Mapping[str, object]) -> str:
    cohort = _mapping(table.get("cohort"), field="table cohort")
    paper_n = int(PAPER_PUBLISHED_REFERENCE["cohort_items"])
    lines = [
        "# Edited-word restoration order (Table 13 protocol replication)",
        "",
        f"Fresh final-PDF protocol replication; n={cohort['items']}.",
        f"For comparison only: historical final-PDF reference; n={paper_n:,}.",
        "",
        "| Restoration order | Zero restored | One | Two | Three | All restored |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in _table_rows(table))
    lines.extend(("", "Paired high-first versus seeded-random tests:"))
    tests = _mapping(table.get("paired_tests"), field="paired tests")
    for budget in INTERMEDIATE_BUDGETS:
        test = _mapping(tests[f"k{budget}"], field=f"paired k{budget}")
        lines.append(
            f"- k={budget}: high-only={test['high_only']}, "
            f"random-only={test['random_only']}, exact p={float(test['p_value']):.10g}"
        )
    return "\n".join(lines) + "\n"


def _render_latex(table: Mapping[str, object]) -> str:
    cohort = _mapping(table.get("cohort"), field="table cohort")
    paper_n = int(PAPER_PUBLISHED_REFERENCE["cohort_items"])
    lines = [
        "% Fresh final-PDF protocol replication; "
        f"n={cohort['items']}. Historical PDF reference n={paper_n}.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Restoration order & Zero restored & One & Two & Three & All restored \\",
        r"\midrule",
    ]
    for row in _table_rows(table):
        lines.append(" & ".join(value.replace("%", r"\%") for value in row) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _serialized_json(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    )


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _destination_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir.parent / f".{output_dir.name}.restoration-order-table.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _commit_table_directory(stage: Path, output_dir: Path) -> None:
    """Commit one completed directory without replacing an existing entry."""
    rename_directory_no_replace(stage, output_dir)


def _fsync_published_parent(parent: Path) -> None:
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_restoration_order_table(
    config: BuildRestorationOrderTableConfig,
) -> RestorationOrderTableResult:
    """Validate six completed settings, micro-pool them, and atomically publish."""
    runs_root = config.runs_root.resolve()
    output_dir = config.output_dir.resolve()
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with _destination_lock(output_dir):
        if os.path.lexists(output_dir):
            raise FileExistsError(f"output directory already exists: {output_dir}")
        analysis_code = analysis_code_identity()
        settings = _discover(runs_root)
        table = _aggregate(settings)
        stage = parent / f".{output_dir.name}.{os.getpid()}.staging"
        if os.path.lexists(stage):
            raise FileExistsError(f"owned staging directory already exists: {stage}")
        try:
            stage.mkdir()
            payloads = {
                _TABLE_NAME: _serialized_json(table),
                _CSV_NAME: _render_csv(table),
                _MARKDOWN_NAME: _render_markdown(table),
                _LATEX_NAME: _render_latex(table),
            }
            for name, value in payloads.items():
                _write(stage / name, value)
            run = {
                "schema_version": "build-restoration-order-table-run/v1",
                "paper_sha256": PAPER_SHA256,
                "protocol_sha256": PROTOCOL_SHA256,
                "operation": "build-restoration-order-table",
                "status": "completed",
                "completed_at": _now(),
                "arguments": {
                    "runs_root": str(runs_root),
                    "output_dir": str(output_dir),
                },
                "analysis_code": analysis_code,
                "inputs": [
                    {
                        "model": setting.model,
                        "benchmark": setting.benchmark,
                        "source_model_revision": setting.source_model_revision,
                        "source_dataset_records_sha256": (
                            setting.source_dataset_records_sha256
                        ),
                        "source_ordered_sample_ids_sha256": (
                            setting.source_ordered_sample_ids_sha256
                        ),
                        "run_path": str(setting.run_path),
                        "run_sha256": setting.run_sha256,
                        "records_sha256": setting.records_sha256,
                        "summary_sha256": setting.summary_sha256,
                        "records": len(setting.rows),
                    }
                    for setting in settings
                ],
                "counts": {
                    "settings": len(settings),
                    "items": table["cohort"]["items"],  # type: ignore[index]
                },
                "outputs": {
                    name: {"sha256": _sha256_file(stage / name)} for name in payloads
                },
            }
            _write(stage / "run.json", _serialized_json(run))
            descriptor = os.open(stage, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if analysis_code_identity() != analysis_code:
                raise RestorationOrderTableInputError(
                    "restoration-order analysis code changed during table construction"
                )
            _commit_table_directory(stage, output_dir)
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        try:
            _fsync_published_parent(parent)
        except OSError:
            # The directory rename is the public commit point. A durability hint
            # failure after that point must not report an already-published run as
            # unpublished or attempt to roll it back.
            pass
    return RestorationOrderTableResult(
        table_path=output_dir / _TABLE_NAME,
        csv_path=output_dir / _CSV_NAME,
        markdown_path=output_dir / _MARKDOWN_NAME,
        latex_path=output_dir / _LATEX_NAME,
        run_path=output_dir / "run.json",
        settings=len(settings),
    )


__all__ = [
    "BuildRestorationOrderTableConfig",
    "RestorationOrderTableInputError",
    "RestorationOrderTableResult",
    "build_restoration_order_table",
]
