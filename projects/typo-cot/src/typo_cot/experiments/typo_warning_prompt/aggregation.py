"""CPU builder for the six-setting Appendix E typo-warning headline."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.typo_warning_prompt.integrity import (
    analysis_code_identity,
    implementation_code_identity,
    validate_paper_runtime_environment,
)
from typo_cot.experiments.typo_warning_prompt.planning import (
    WarningPromptPlan,
    build_warning_prompt_plan,
)
from typo_cot.experiments.typo_warning_prompt.protocol import (
    ARM_ORDER,
    COMPARABILITY_LIMITATIONS,
    HISTORICAL_REFERENCE,
    PAPER_GRID,
    PROTOCOL,
    PROTOCOL_SHA256,
)
from typo_cot.experiments.typo_warning_prompt.source import (
    DEFAULT_SUBMITTED_INPUTS_SHA256,
    DEFAULT_SUBMITTED_INPUTS_CANONICAL_SHA256,
    WarningSourceCase,
    canonical_sha256,
    load_json,
    load_warning_sources,
    select_warning_cases,
    sha256_file,
    strict_loads,
)
from typo_cot.experiments.typo_warning_prompt.scoring import extract_submitted_answer
from typo_cot.experiments.typo_warning_prompt.statistics import exact_mcnemar

_SETTING_OUTPUTS = ("warning_prompt_records.jsonl", "warning_prompt_summary.json")
_BUILDER_OUTPUTS = (
    "typo_warning_summary.json",
    "typo_warning_summary.csv",
    "typo_warning_summary.md",
    "typo_warning_summary.tex",
)
_SOURCE_FILE_ROLES = {"submitted-input-edit-manifest"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class TypoWarningSummaryInputError(ValueError):
    """Producer artifacts cannot support the paper-labelled pooled summary."""


@dataclass(frozen=True, slots=True)
class BuildTypoWarningSummaryConfig:
    runs_root: Path
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs_root", Path(self.runs_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.runs_root.resolve() == self.output_dir.resolve():
            raise ValueError("runs_root and output_dir must be different directories")


@dataclass(frozen=True, slots=True)
class BuildTypoWarningSummaryResult:
    summary_path: Path
    csv_path: Path
    markdown_path: Path
    latex_path: Path
    run_path: Path
    settings: int


@dataclass(frozen=True, slots=True)
class _Setting:
    directory: Path
    run_path: Path
    run_sha256: str
    model: str
    benchmark: str
    records_path: Path
    records_sha256: str
    summary_path: Path
    summary_sha256: str
    source_files: tuple[tuple[Path, str], ...]
    records: tuple[dict[str, object], ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.model, self.benchmark


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _int(value: object, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append(payload)
    return tuple(rows)


def _events(record: Mapping[str, object], *, model: str, benchmark: str) -> dict[str, bool]:
    expected = {
        "schema_version": "typo-warning-prompt-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "typo-warning-prompt",
        "model": model,
        "benchmark": benchmark,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"record {field} must be {value!r}")
    _string(record.get("sample_id"), field="record sample_id")
    gold = _string(record.get("gold_answer"), field="record gold_answer")
    source = _mapping(record.get("source"), field="record source")
    if set(source) != {"submitted_record_sha256", "selector_record_sha256"}:
        raise ValueError("record source has an invalid shape")
    for field in source:
        value = source[field]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"record source {field} must be SHA-256")
    plan_sha256 = record.get("plan_sha256")
    if not isinstance(plan_sha256, str) or _SHA256.fullmatch(plan_sha256) is None:
        raise ValueError("record plan_sha256 must be SHA-256")
    arms = _mapping(record.get("arms"), field="record arms")
    if set(arms) != set(ARM_ORDER):
        raise ValueError("record must contain both protocol arms")
    correctness: dict[str, bool] = {}
    for arm in ARM_ORDER:
        arm_payload = _mapping(arms[arm], field=f"record arm {arm}")
        if arm_payload.get("warning_enabled") != (arm == "with-warning"):
            raise ValueError(f"record arm {arm} warning flag differs")
        answer = _mapping(arm_payload.get("answer"), field=f"record arm {arm} answer")
        text = answer.get("text")
        value = answer.get("value")
        extracted = answer.get("is_extracted")
        correct = answer.get("is_correct")
        method = answer.get("method")
        primary_method = answer.get("primary_method")
        stop_reason = answer.get("stop_reason")
        token_ids = answer.get("token_ids")
        if (
            not isinstance(text, str)
            or not isinstance(value, str)
            or type(extracted) is not bool
            or type(correct) is not bool
            or not isinstance(method, str)
            or not method
            or not isinstance(primary_method, str)
            or not primary_method
            or stop_reason not in {"eos_token", "max_new_tokens"}
            or not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
        ):
            raise ValueError(f"record arm {arm} answer is invalid")
        extraction = extract_submitted_answer(
            text,
            benchmark=benchmark,
            correct_answer=gold,
        )
        if (value, extracted, correct, method, primary_method) != (
            extraction.value,
            extraction.is_extracted,
            extraction.is_correct,
            extraction.method,
            extraction.primary_method,
        ):
            raise ValueError(f"record arm {arm} answer extraction differs from generated text")
        correctness[arm] = correct
    off = correctness["without-warning"]
    on = correctness["with-warning"]
    expected_events = {
        "without_warning_correct": off,
        "with_warning_correct": on,
        "without_warning_only": off and not on,
        "with_warning_only": on and not off,
    }
    if record.get("events") != expected_events:
        raise ValueError("record events differ from the paired arm outcomes")
    return expected_events


def _metrics(
    records: Sequence[Mapping[str, object]], *, model: str, benchmark: str
) -> dict[str, object]:
    if not records:
        raise ValueError("warning metrics require at least one record")
    events = [_events(record, model=model, benchmark=benchmark) for record in records]
    n = len(events)
    off = sum(event["without_warning_correct"] for event in events)
    on = sum(event["with_warning_correct"] for event in events)
    b10 = sum(event["without_warning_only"] for event in events)
    b01 = sum(event["with_warning_only"] for event in events)
    return {
        "n": n,
        "counts": {
            "without_warning_correct": off,
            "with_warning_correct": on,
        },
        "accuracy": {
            "without_warning": off / n,
            "with_warning": on / n,
            "with_minus_without": (on - off) / n,
        },
        "mcnemar": exact_mcnemar(b10=b10, b01=b01),
    }


def _validate_record_source_binding(
    record: Mapping[str, object],
    *,
    case: WarningSourceCase,
    plan: WarningPromptPlan,
) -> None:
    if record.get("gold_answer") != plan.gold_answer:
        raise ValueError("setting record gold answer differs from its prepared source")
    expected_source = {
        "submitted_record_sha256": case.submitted_record_sha256,
        "selector_record_sha256": case.selector_record_sha256,
    }
    if record.get("source") != expected_source:
        raise ValueError("setting record source fingerprint differs from its submitted source")
    if record.get("plan_sha256") != plan.fingerprint:
        raise ValueError("setting record plan fingerprint differs from its prepared source")
    arms = _mapping(record.get("arms"), field="setting record arms")
    if set(arms) != set(ARM_ORDER):
        raise ValueError("setting record does not contain both protocol arms")
    for arm_plan in plan.arms:
        arm_payload = _mapping(arms[arm_plan.arm], field=f"setting record arm {arm_plan.arm}")
        if arm_payload.get("fixed_input") != arm_plan.to_dict(include_text=False):
            raise ValueError(f"setting record fixed input differs for {arm_plan.arm}")
        input_use = _mapping(
            arm_payload.get("input_use"),
            field=f"setting record input use {arm_plan.arm}",
        )
        expected_input = {
            "arm": arm_plan.arm,
            "prompt_text_sha256": arm_plan.prompt_sha256,
            "prompt_char_count": len(arm_plan.prompt),
        }
        for field, expected in expected_input.items():
            if input_use.get(field) != expected:
                raise ValueError(f"setting record input use {field} differs for {arm_plan.arm}")
        _int(
            input_use.get("prompt_token_count"),
            field=f"setting record input use {arm_plan.arm}.prompt_token_count",
            minimum=1,
        )
        prompt_ids_sha256 = input_use.get("prompt_ids_sha256")
        if not isinstance(prompt_ids_sha256, str) or _SHA256.fullmatch(prompt_ids_sha256) is None:
            raise ValueError("setting record prompt_ids_sha256 must be SHA-256")


def _validate_setting_runtime(
    manifest: Mapping[str, object],
    *,
    model: str,
    model_revision: str,
    gpu_id: str,
) -> tuple[int, ...]:
    runtime = _mapping(manifest.get("runtime"), field="setting runtime")
    validate_paper_runtime_environment(runtime, field_prefix="setting runtime")
    required = {
        "operation": "typo-warning-prompt",
        "model": model,
        "requested_revision": model_revision,
        "model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "dtype": "bfloat16",
        "cuda_visible_devices": gpu_id,
        "implementation": PROTOCOL["implementation"],
        "answer_extraction": PROTOCOL["answer_extraction"],
        "prompt_intervention": PROTOCOL["prompt_intervention"],
        "implementation_code": implementation_code_identity(),
    }
    for field, expected in required.items():
        if runtime.get(field) != expected:
            raise ValueError(f"setting runtime {field} differs from the protocol")
    for field in ("generation", "batching", "answer_span_decoding"):
        if runtime.get(field) != PROTOCOL[field]:
            raise ValueError(f"setting runtime {field} differs from the protocol")
    eos = runtime.get("effective_eos_token_ids")
    if (
        not isinstance(eos, list)
        or not eos
        or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos)
        or len(eos) != len(set(eos))
    ):
        raise ValueError("setting runtime effective EOS token IDs are invalid")
    _string(
        runtime.get("effective_eos_token_ids_source"),
        field="setting runtime effective_eos_token_ids_source",
    )
    return tuple(eos)


def _validate_record_termination(
    record: Mapping[str, object], *, effective_eos_token_ids: Sequence[int]
) -> None:
    arms = _mapping(record.get("arms"), field="setting record arms")
    eos_ids = set(effective_eos_token_ids)
    for arm in ARM_ORDER:
        arm_payload = _mapping(arms[arm], field=f"setting record arm {arm}")
        answer = _mapping(arm_payload.get("answer"), field=f"setting record arm {arm} answer")
        token_ids = answer.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"setting record arm {arm} token IDs are invalid")
        stop_reason = answer.get("stop_reason")
        stop_token_id = answer.get("stop_token_id")
        if stop_reason == "eos_token":
            if (
                stop_token_id not in eos_ids
                or token_ids[-1] != stop_token_id
                or any(token in eos_ids for token in token_ids[:-1])
            ):
                raise ValueError(f"setting record arm {arm} EOS termination is inconsistent")
        elif (
            stop_reason != "max_new_tokens"
            or stop_token_id is not None
            or len(token_ids) != int(PROTOCOL["generation"]["max_new_tokens"])
            or any(token in eos_ids for token in token_ids)
        ):
            raise ValueError(f"setting record arm {arm} capped termination is inconsistent")


def _validate_output_metadata(metadata: Mapping[str, object], *, path: Path, records: int) -> None:
    if metadata.get("sha256") != sha256_file(path):
        raise ValueError(f"setting output SHA-256 mismatch: {path}")
    if metadata.get("bytes") != path.stat().st_size:
        raise ValueError(f"setting output byte count mismatch: {path}")
    if metadata.get("records") != records:
        raise ValueError(f"setting output record count mismatch: {path}")


def _validate_source_files(manifest: Mapping[str, object]) -> tuple[tuple[Path, str], ...]:
    source = _mapping(manifest.get("source"), field="setting source")
    arguments = _mapping(manifest.get("arguments"), field="setting arguments")
    if set(arguments) != {
        "model",
        "benchmark",
        "input_fixture",
        "output_dir",
        "gpu_id",
        "limit",
    }:
        raise ValueError("setting arguments have an invalid schema")
    input_fixture = Path(
        _string(arguments.get("input_fixture"), field="setting input_fixture")
    ).resolve()
    expected_paths = {"submitted-input-edit-manifest": input_fixture}
    files = source.get("source_files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("setting source must retain the submitted-input edit manifest")
    seen: set[Path] = set()
    seen_roles: set[str] = set()
    snapshots: list[tuple[Path, str]] = []
    for raw_file in files:
        item = _mapping(raw_file, field="setting source file")
        role = _string(item.get("role"), field="setting source file role")
        if role not in _SOURCE_FILE_ROLES or role in seen_roles:
            raise ValueError(f"setting source file role is invalid or duplicated: {role!r}")
        seen_roles.add(role)
        path = Path(_string(item.get("path"), field="setting source file path")).resolve()
        expected = _string(item.get("sha256"), field="setting source file sha256")
        if _SHA256.fullmatch(expected) is None:
            raise ValueError("setting source file sha256 must be a full SHA-256")
        if path != expected_paths[role].resolve():
            raise ValueError(f"setting source file path differs for role {role!r}")
        if path in seen:
            raise ValueError("setting source contains a duplicate file")
        seen.add(path)
        if not path.is_file():
            raise ValueError(f"setting source file is unavailable: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"setting source file SHA-256 mismatch: {path}")
        snapshots.append((path, expected))
    if seen_roles != _SOURCE_FILE_ROLES:
        raise ValueError("setting source file roles are incomplete")
    return tuple(snapshots)


def _load_setting(run_path: Path) -> _Setting | None:
    initial_run_sha256 = sha256_file(run_path)
    manifest = load_json(run_path)
    if manifest.get("operation") != "typo-warning-prompt":
        return None
    if manifest.get("schema_version") != "typo-warning-prompt-run/v2":
        raise ValueError(f"unknown setting schema: {run_path}")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise ValueError(f"setting paper SHA-256 mismatch: {run_path}")
    if manifest.get("status") != "completed":
        raise ValueError(f"setting must be completed: {run_path}")
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError(f"setting protocol SHA-256 mismatch: {run_path}")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"setting protocol payload mismatch: {run_path}")
    arguments = _mapping(manifest.get("arguments"), field="setting arguments")
    model = _string(arguments.get("model"), field="setting model")
    benchmark = _string(arguments.get("benchmark"), field="setting benchmark")
    if (model, benchmark) not in set(PAPER_GRID):
        raise ValueError(f"unexpected typo-warning setting: {model}/{benchmark}")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise ValueError("paper setting must be unlimited")
    directory = run_path.parent.resolve()
    if (
        Path(_string(arguments.get("output_dir"), field="setting output_dir")).resolve()
        != directory
    ):
        raise ValueError("setting output directory differs from its manifest path")
    validated_source = load_warning_sources(
        Path(_string(arguments.get("input_fixture"), field="setting input_fixture")),
        model=model,
        benchmark=benchmark,
    )
    selected_cases = select_warning_cases(validated_source)
    if (
        not validated_source.fixture_is_bundled_submitted_input
        or validated_source.fixture_sha256 != DEFAULT_SUBMITTED_INPUTS_SHA256
        or validated_source.fixture_canonical_sha256 != DEFAULT_SUBMITTED_INPUTS_CANONICAL_SHA256
        or validated_source.shared_sample_count != 300
    ):
        raise ValueError(
            "paper setting must use the exact bundled 300-item submitted-input fixture"
        )
    effective_eos_token_ids = _validate_setting_runtime(
        manifest,
        model=model,
        model_revision=validated_source.model_revision,
        gpu_id=_string(arguments.get("gpu_id"), field="setting gpu_id"),
    )
    plan = _mapping(manifest.get("plan"), field="setting plan")
    source = _mapping(manifest.get("source"), field="setting source")
    source_files = _validate_source_files(manifest)
    if dict(source) != validated_source.to_dict():
        raise ValueError("setting source payload differs from revalidated submitted sources")
    if source.get("model") != model or source.get("benchmark") != benchmark:
        raise ValueError("setting source model or benchmark differs")
    shared_pairs = _int(
        source.get("shared_sample_count"),
        field="setting source shared_sample_count",
        minimum=300,
    )
    if shared_pairs != 300:
        raise ValueError("paper setting source must contain exactly 300 submitted inputs")
    for field in ("dataset_records_sha256", "source_sha256"):
        value = source.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"setting source {field} must be SHA-256")
    _string(source.get("model_revision"), field="setting source model_revision")
    _int(
        source.get("dataset_sample_count"),
        field="setting source dataset_sample_count",
        minimum=shared_pairs,
    )
    if source.get("dataset_sample_count") != shared_pairs:
        raise ValueError("paper setting dataset sample count must equal its submitted cohort")
    if plan.get("selection") != PROTOCOL["selection"]:
        raise ValueError("paper setting selection protocol differs")
    if (
        plan.get("shared_source_pairs") != shared_pairs
        or plan.get("selected_pairs") != 300
        or plan.get("executed_pairs") != 300
    ):
        raise ValueError("paper setting plan must execute 300 pairs from its shared source")
    expected_plan = {
        "selection": PROTOCOL["selection"],
        "shared_source_pairs": validated_source.shared_sample_count,
        "selected_pairs": len(selected_cases),
        "executed_pairs": len(selected_cases),
        "selected_ids_sha256": canonical_sha256([case.sample_id for case in selected_cases]),
        "executed_ids_sha256": canonical_sha256([case.sample_id for case in selected_cases]),
        "selected_source_sha256": canonical_sha256([case.identity for case in selected_cases]),
        "executed_source_sha256": canonical_sha256([case.identity for case in selected_cases]),
    }
    if dict(plan) != expected_plan:
        raise ValueError("setting plan differs from the revalidated prepared-source selection")
    counts = _mapping(manifest.get("counts"), field="setting counts")
    if set(counts) != {
        "shared_source_pairs",
        "selected_pairs",
        "executed_pairs",
        "completed_arms",
        "completed_pairs",
        "failed",
    }:
        raise ValueError("setting counts have an invalid schema")
    if (
        counts.get("shared_source_pairs") != shared_pairs
        or counts.get("selected_pairs") != 300
        or counts.get("executed_pairs") != 300
        or counts.get("completed_arms") != 600
        or counts.get("completed_pairs") != 300
    ):
        raise ValueError("paper setting counts must contain 300 completed pairs")
    if counts.get("failed") != 0 or manifest.get("failures") not in ([], None):
        raise ValueError("paper setting contains failures")
    if manifest.get("checkpoints") != {}:
        raise ValueError("completed paper setting retains checkpoints")

    records_path = directory / _SETTING_OUTPUTS[0]
    summary_path = directory / _SETTING_OUTPUTS[1]
    if not records_path.is_file() or not summary_path.is_file():
        raise ValueError(f"setting outputs are missing: {directory}")
    initial_records_sha256 = sha256_file(records_path)
    initial_summary_sha256 = sha256_file(summary_path)
    outputs = _mapping(manifest.get("outputs"), field="setting outputs")
    if set(outputs) != set(_SETTING_OUTPUTS):
        raise ValueError("setting output registry has the wrong files")
    records = _load_jsonl(records_path)
    _validate_output_metadata(
        _mapping(outputs[records_path.name], field="setting records metadata"),
        path=records_path,
        records=len(records),
    )
    _validate_output_metadata(
        _mapping(outputs[summary_path.name], field="setting summary metadata"),
        path=summary_path,
        records=1,
    )
    if len(records) != 300:
        raise ValueError("paper setting records must contain exactly 300 rows")
    ids = [_string(record.get("sample_id"), field="setting record sample_id") for record in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("setting record sample IDs must be sorted and unique")
    selected_by_id = {case.sample_id: case for case in selected_cases}
    if ids != sorted(selected_by_id):
        raise ValueError("setting records differ from the revalidated selected sample IDs")
    for record in records:
        sample_id = str(record["sample_id"])
        case = selected_by_id[sample_id]
        record_plan = build_warning_prompt_plan(
            case.submitted_record,
            source_record_sha256=case.submitted_record_sha256,
        )
        _validate_record_source_binding(record, case=case, plan=record_plan)
        _validate_record_termination(
            record,
            effective_eos_token_ids=effective_eos_token_ids,
        )
    selected_ids_sha256 = plan.get("selected_ids_sha256")
    if selected_ids_sha256 != canonical_sha256(ids):
        raise ValueError("setting records differ from the selected ID fingerprint")
    if plan.get("executed_ids_sha256") != selected_ids_sha256:
        raise ValueError("setting executed ID fingerprint differs from its selected cohort")
    source_identities = [
        {
            "sample_id": record["sample_id"],
            **dict(_mapping(record.get("source"), field="setting record source")),
        }
        for record in records
    ]
    selected_source_sha256 = canonical_sha256(source_identities)
    if plan.get("selected_source_sha256") != selected_source_sha256:
        raise ValueError("setting records differ from the selected source fingerprint")
    if plan.get("executed_source_sha256") != selected_source_sha256:
        raise ValueError("setting executed source fingerprint differs from its selected cohort")
    metrics = _metrics(records, model=model, benchmark=benchmark)
    summary = _mapping(load_json(summary_path), field="setting summary")
    expected_comparability = {
        "status": "exact-submitted-input-paper-grid-setting",
        "requirements": {
            "final_paper_operation": True,
            "submitted_model": True,
            "submitted_benchmark": True,
            "exact_bundled_submitted_input_fixture": True,
            "not_limit_truncated": True,
        },
        "limitations": list(COMPARABILITY_LIMITATIONS),
        "historical_cohort_identity": True,
    }
    if manifest.get("comparability") != expected_comparability:
        raise ValueError("setting comparability metadata differs from the paper-grid contract")
    if manifest.get("historical_reference") != HISTORICAL_REFERENCE:
        raise ValueError("setting historical reference metadata differs")
    expected_summary_identity = {
        "schema_version": "typo-warning-prompt-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "typo-warning-prompt",
        "model": model,
        "benchmark": benchmark,
        "n": metrics["n"],
        "selection": dict(plan),
        "comparability": expected_comparability,
        "historical_reference": HISTORICAL_REFERENCE["pooled_by_benchmark"][benchmark],
    }
    expected_summary_fields = set(expected_summary_identity) | {
        "counts",
        "accuracy",
        "mcnemar",
    }
    if set(summary) != expected_summary_fields:
        raise ValueError("setting summary has an invalid schema")
    for field, expected in expected_summary_identity.items():
        if summary.get(field) != expected:
            raise ValueError(f"setting summary {field} differs")
    for field in ("counts", "accuracy"):
        if summary.get(field) != metrics[field]:
            raise ValueError(f"setting summary {field} differs from the paired records")
    _mapping(summary.get("mcnemar"), field="setting summary mcnemar")
    if sha256_file(run_path) != initial_run_sha256:
        raise ValueError(f"setting run snapshot changed while validating: {run_path}")
    if sha256_file(records_path) != initial_records_sha256:
        raise ValueError(f"setting records snapshot changed while validating: {records_path}")
    if sha256_file(summary_path) != initial_summary_sha256:
        raise ValueError(f"setting summary snapshot changed while validating: {summary_path}")
    return _Setting(
        directory=directory,
        run_path=run_path.resolve(),
        run_sha256=initial_run_sha256,
        model=model,
        benchmark=benchmark,
        records_path=records_path,
        records_sha256=initial_records_sha256,
        summary_path=summary_path,
        summary_sha256=initial_summary_sha256,
        source_files=source_files,
        records=records,
    )


def _validate_setting_snapshots(settings: Sequence[_Setting]) -> None:
    """Rehash every producer and upstream source used by the analysis."""

    for setting in settings:
        snapshots = (
            ("setting run", setting.run_path, setting.run_sha256),
            ("setting records", setting.records_path, setting.records_sha256),
            ("setting summary", setting.summary_path, setting.summary_sha256),
            *(("submitted input", path, expected) for path, expected in setting.source_files),
        )
        for role, path, expected in snapshots:
            try:
                actual = sha256_file(path)
            except OSError as exc:
                raise ValueError(f"input snapshot changed: {role} is unavailable: {path}") from exc
            if actual != expected:
                raise ValueError(f"input snapshot changed: {role} SHA-256 mismatch: {path}")


def _discover(runs_root: Path) -> tuple[_Setting, ...]:
    if not runs_root.is_dir():
        raise ValueError(f"runs_root is not a directory: {runs_root}")
    settings: list[_Setting] = []
    for run_path in sorted(runs_root.rglob("run.json")):
        setting = _load_setting(run_path)
        if setting is not None:
            settings.append(setting)
    by_key: dict[tuple[str, str], _Setting] = {}
    for setting in settings:
        if setting.key in by_key:
            raise ValueError(f"duplicate typo-warning setting: {setting.key}")
        by_key[setting.key] = setting
    expected = set(PAPER_GRID)
    actual = set(by_key)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        fragments = []
        if missing:
            fragments.append("missing " + ", ".join(f"{m}/{b}" for m, b in missing))
        if unexpected:
            fragments.append("unexpected " + ", ".join(f"{m}/{b}" for m, b in unexpected))
        raise ValueError("typo-warning setting grid is incomplete: " + "; ".join(fragments))
    return tuple(by_key[key] for key in PAPER_GRID)


def _payload(settings: Sequence[_Setting]) -> dict[str, object]:
    per_setting: list[dict[str, object]] = []
    for setting in settings:
        per_setting.append(
            {
                "model": setting.model,
                "benchmark": setting.benchmark,
                **_metrics(
                    setting.records,
                    model=setting.model,
                    benchmark=setting.benchmark,
                ),
            }
        )
    pooled: dict[str, object] = {}
    # Every setting row was already validated with its own model identity.
    # Pool only the resulting Boolean events so the three model labels remain
    # distinct inputs rather than pretending to be one synthetic model.
    for benchmark in ("gsm8k", "mmlu"):
        events = [
            _mapping(record["events"], field="pooled record events")
            for setting in settings
            if setting.benchmark == benchmark
            for record in setting.records
        ]
        n = len(events)
        off = sum(event.get("without_warning_correct") is True for event in events)
        on = sum(event.get("with_warning_correct") is True for event in events)
        b10 = sum(event.get("without_warning_only") is True for event in events)
        b01 = sum(event.get("with_warning_only") is True for event in events)
        pooled[benchmark] = {
            "n": n,
            "counts": {
                "without_warning_correct": off,
                "with_warning_correct": on,
            },
            "accuracy": {
                "without_warning": off / n,
                "with_warning": on / n,
                "with_minus_without": (on - off) / n,
            },
            "mcnemar": exact_mcnemar(b10=b10, b01=b01),
            "models": [setting.model for setting in settings if setting.benchmark == benchmark],
            "historical_reference": HISTORICAL_REFERENCE["pooled_by_benchmark"][benchmark],
        }
    return {
        "schema_version": "typo-warning-paper-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "build-typo-warning-summary",
        "complete_grid": True,
        "settings": per_setting,
        "pooled_by_benchmark": pooled,
        "analysis": {
            "pooling": "micro-average-three-300-item-settings-per-benchmark/v1",
            "paired_test": "exact-two-sided-mcnemar-binomial/v1",
            "result_role": "fresh-validated-public-reproduction",
            "historical_reference_role": "descriptive-only",
        },
        "interpretation_limits": [
            "one-warning-instruction-only",
            "two-tasks-only",
            "not-a-general-self-correction-evaluation",
            "not-a-performance-comparison-with-activation-patching",
        ],
    }


def _rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    settings = payload["settings"]
    assert isinstance(settings, list)
    for setting in settings:
        assert isinstance(setting, dict)
        rows.append({"scope": "setting", **setting})
    pooled = payload["pooled_by_benchmark"]
    assert isinstance(pooled, dict)
    for benchmark in ("gsm8k", "mmlu"):
        metric = pooled[benchmark]
        assert isinstance(metric, dict)
        rows.append(
            {"scope": "pooled", "model": "all-three-models", "benchmark": benchmark, **metric}
        )
    return rows


def _csv_text(payload: Mapping[str, object]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "scope",
            "model",
            "benchmark",
            "n",
            "without_warning_correct",
            "without_warning_accuracy",
            "with_warning_correct",
            "with_warning_accuracy",
            "with_minus_without",
            "b10",
            "b01",
            "p_value",
        )
    )
    for row in _rows(payload):
        counts = row["counts"]
        accuracy = row["accuracy"]
        test = row["mcnemar"]
        assert isinstance(counts, Mapping)
        assert isinstance(accuracy, Mapping)
        assert isinstance(test, Mapping)
        writer.writerow(
            (
                row["scope"],
                row["model"],
                row["benchmark"],
                row["n"],
                counts["without_warning_correct"],
                f"{float(accuracy['without_warning']):.12g}",
                counts["with_warning_correct"],
                f"{float(accuracy['with_warning']):.12g}",
                f"{float(accuracy['with_minus_without']):.12g}",
                test["b10"],
                test["b01"],
                f"{float(test['p_value']):.12g}",
            )
        )
    return stream.getvalue()


def _markdown_text(payload: Mapping[str, object]) -> str:
    lines = [
        "# Appendix E typo-warning accuracy",
        "",
        "Fresh validated results; final-PDF values are descriptive references, not acceptance targets.",
        "",
        "| scope | model | benchmark | n | no warning | warning | difference | b10/b01 | exact p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _rows(payload):
        accuracy = row["accuracy"]
        test = row["mcnemar"]
        assert isinstance(accuracy, Mapping) and isinstance(test, Mapping)
        lines.append(
            f"| {row['scope']} | {row['model']} | {row['benchmark']} | {row['n']} "
            f"| {100 * float(accuracy['without_warning']):.1f}% "
            f"| {100 * float(accuracy['with_warning']):.1f}% "
            f"| {100 * float(accuracy['with_minus_without']):+.1f} pp "
            f"| {test['b10']}/{test['b01']} | {float(test['p_value']):.6g} |"
        )
    return "\n".join(lines) + "\n"


def _latex_text(payload: Mapping[str, object]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Benchmark & $n$ & No warning & Warning & $p$ \\",
        r"\midrule",
    ]
    for row in _rows(payload):
        accuracy = row["accuracy"]
        test = row["mcnemar"]
        assert isinstance(accuracy, Mapping) and isinstance(test, Mapping)
        label = str(row["model"]).replace("_", r"\_")
        lines.append(
            f"{label} & {row['benchmark']} & {row['n']} & "
            f"{100 * float(accuracy['without_warning']):.1f} & "
            f"{100 * float(accuracy['with_warning']):.1f} & "
            f"{float(test['p_value']):.3g} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _metadata(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_staged_directory(stage: Path, output_dir: Path) -> None:
    """Rename one staged artifact set and clean it up if durability fails.

    The rename is atomic, but a successful rename is not reported as completed
    until the parent directory has also been synced. If that durability barrier
    fails, remove only the directory whose filesystem identity matches the stage
    we just renamed. An identity mismatch is left untouched to avoid deleting a
    concurrently replaced path.
    """

    staged = stage.stat()
    stage_identity = (staged.st_dev, staged.st_ino)
    os.replace(stage, output_dir)
    try:
        _fsync_directory(output_dir.parent)
    except OSError as publish_error:
        try:
            published = output_dir.stat()
            if (published.st_dev, published.st_ino) != stage_identity:
                raise RuntimeError(
                    "parent directory fsync failed after publication; cleanup was refused "
                    f"because the output identity changed: {output_dir}"
                )
            shutil.rmtree(output_dir)
        except Exception as cleanup_error:
            raise RuntimeError(
                "parent directory fsync failed after publication and the renamed output "
                f"could not be safely removed: {output_dir}: {cleanup_error}"
            ) from publish_error
        try:
            _fsync_directory(output_dir.parent)
        except OSError:
            # Preserve the original publication error. The owned output has
            # already been removed; a second failed durability barrier cannot
            # be recovered within this process.
            pass
        raise


def build_typo_warning_summary(
    config: BuildTypoWarningSummaryConfig,
) -> BuildTypoWarningSummaryResult:
    """Validate the exact six-setting grid and atomically build paper artifacts."""

    if config.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {config.output_dir}")
    try:
        settings = _discover(config.runs_root)
        payload = _payload(settings)
        _validate_setting_snapshots(settings)
    except (OSError, TypeError, ValueError) as exc:
        raise TypoWarningSummaryInputError(str(exc)) from exc

    parent = config.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=parent))
    try:
        summary_path = stage / _BUILDER_OUTPUTS[0]
        csv_path = stage / _BUILDER_OUTPUTS[1]
        markdown_path = stage / _BUILDER_OUTPUTS[2]
        latex_path = stage / _BUILDER_OUTPUTS[3]
        _write_json(summary_path, payload)
        _write_text(csv_path, _csv_text(payload))
        _write_text(markdown_path, _markdown_text(payload))
        _write_text(latex_path, _latex_text(payload))
        output_metadata = {
            path.name: _metadata(path)
            for path in (summary_path, csv_path, markdown_path, latex_path)
        }
        input_metadata = [
            {
                "model": setting.model,
                "benchmark": setting.benchmark,
                "run": str(setting.run_path),
                "run_sha256": setting.run_sha256,
                "records": str(setting.records_path),
                "records_sha256": setting.records_sha256,
                "summary": str(setting.summary_path),
                "summary_sha256": setting.summary_sha256,
            }
            for setting in settings
        ]
        run_path = stage / "run.json"
        _write_json(
            run_path,
            {
                "schema_version": "build-typo-warning-summary-run/v1",
                "paper_sha256": PAPER_SHA256,
                "operation": "build-typo-warning-summary",
                "status": "completed",
                "created_at": _now(),
                "arguments": {
                    "runs_root": str(config.runs_root.resolve()),
                    "output_dir": str(config.output_dir.resolve()),
                },
                "analysis": payload["analysis"],
                "analysis_code": analysis_code_identity(),
                "input_settings": input_metadata,
                "input_grid_sha256": canonical_sha256(input_metadata),
                "outputs": output_metadata,
            },
        )
        try:
            _validate_setting_snapshots(settings)
        except (OSError, ValueError) as exc:
            raise TypoWarningSummaryInputError(str(exc)) from exc
        _publish_staged_directory(stage, config.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return BuildTypoWarningSummaryResult(
        summary_path=config.output_dir / _BUILDER_OUTPUTS[0],
        csv_path=config.output_dir / _BUILDER_OUTPUTS[1],
        markdown_path=config.output_dir / _BUILDER_OUTPUTS[2],
        latex_path=config.output_dir / _BUILDER_OUTPUTS[3],
        run_path=config.output_dir / "run.json",
        settings=len(settings),
    )


__all__ = [
    "BuildTypoWarningSummaryConfig",
    "BuildTypoWarningSummaryResult",
    "TypoWarningSummaryInputError",
    "build_typo_warning_summary",
]
