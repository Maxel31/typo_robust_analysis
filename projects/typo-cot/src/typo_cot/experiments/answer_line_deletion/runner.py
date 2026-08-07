"""Validated, restartable runner for the final answer-line deletion control."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from tqdm.auto import tqdm

import typo_cot.experiments.answer_line_deletion.protocol as deletion_protocol
from typo_cot.evaluation.fallback import answers_equal, extract_with_fallback
from typo_cot.experiments.answer_line_deletion.planning import (
    AnswerLineArmPlan,
    AnswerLineDeletionPlan,
    build_answer_line_deletion_plan,
)
from typo_cot.experiments.answer_line_deletion.source import (
    CompletedCotSwapSource,
    CotSwapSourceCase,
    canonical_sha256,
    load_completed_cot_swap_source,
    load_json,
    sha256_file,
    strict_loads,
    validate_source_snapshot,
)
from typo_cot.experiments.catalog import PAPER_SHA256

ANSWER_LINE_DELETION_BENCHMARKS = tuple(deletion_protocol.BENCHMARK_DATASET_NAMES)
ARM_ORDER = deletion_protocol.ARM_ORDER
_ANSWER_EXTRACTION = deletion_protocol.ANSWER_EXTRACTION
_ANSWER_SPAN_DECODING = deletion_protocol.ANSWER_SPAN_DECODING
_BATCHING = deletion_protocol.BATCHING
_BENCHMARK_NAMES = deletion_protocol.BENCHMARK_DATASET_NAMES
_GENERATION = deletion_protocol.GENERATION
_IMPLEMENTATION = deletion_protocol.IMPLEMENTATION
_TEXT_INTERVENTION = deletion_protocol.TEXT_INTERVENTION
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PAPER_MODELS = {
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
}
_PROTOCOL = {
    "source": "completed-unlimited-random-4-cot-swap-run/v1",
    "cohort": "canonical-a-correct-and-b-differs-from-a/v1",
    "ordering": "sample-id-ascending-then-max-pairs-then-limit/v1",
    "arms": list(ARM_ORDER),
    "generation": dict(_GENERATION),
    "batching": dict(_BATCHING),
    "answer_span_decoding": dict(_ANSWER_SPAN_DECODING),
    "answer_extraction": _ANSWER_EXTRACTION,
    "text_intervention": dict(_TEXT_INTERVENTION),
    "unextractable": "retain-as-failed-restoration/v1",
    "checkpoint": "one-pair-two-arm-atomic/v1",
}
_PROTOCOL_SHA256 = canonical_sha256(_PROTOCOL)

_PUBLISHED_REFERENCE = {
    "scope": "three-model-task-micro-pool",
    "final_pdf_protocol": {
        "max_new_tokens": 16,
        "restoration": "canonical-answer-equality-to-source-A",
    },
    "printed_table_source": {
        "max_new_tokens": 256,
        "gsm8k": {
            "n": 333,
            "complete_restored": 317,
            "deleted_restored": 163,
            "deleted_unextractable": 29,
        },
        "mmlu": {
            "n": 450,
            "complete_restored": 370,
            "deleted_restored": 131,
            "deleted_unextractable": 7,
        },
    },
    "archived_16_token_control": {
        "max_new_tokens": 16,
        "gsm8k": {
            "n": 333,
            "complete_restored": 315,
            "deleted_restored": 14,
            "complete_unextractable": 13,
            "deleted_unextractable": 262,
        },
        "mmlu": {
            "n": 450,
            "complete_restored": 369,
            "deleted_restored": 54,
            "complete_unextractable": 7,
            "deleted_unextractable": 232,
        },
    },
    "historical_prefix_became_empty": {
        "gsm8k": {"numerator": 179, "denominator": 333},
        "mmlu": {"numerator": 334, "denominator": 450},
    },
    "historical_cohort_identity": False,
    "use_as_acceptance_target": False,
    "protocol_conflict": "printed-table-uses-256-while-final-pdf-specifies-16",
}


class AnswerLineDeletionRunError(RuntimeError):
    """Raised after failures or published-state corruption are recorded."""


class _CompletedOutputIntegrityError(AnswerLineDeletionRunError):
    """Raised when a completed public output set fails byte-level checks."""


@dataclass(frozen=True, slots=True)
class AnswerLineDeletionConfig:
    """Frozen public arguments for one model/task control setting."""

    model: str
    benchmark: Literal["gsm8k", "mmlu"]
    cot_swap_run: Path
    max_pairs: int
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cot_swap_run", Path(self.cot_swap_run))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in ANSWER_LINE_DELETION_BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {self.benchmark!r}")
        if not isinstance(self.gpu_id, str) or _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if (
            not isinstance(self.max_pairs, int)
            or isinstance(self.max_pairs, bool)
            or self.max_pairs <= 0
        ):
            raise ValueError("max_pairs must be a positive integer")
        if self.limit is not None and (
            not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        source = self.cot_swap_run.resolve()
        output = self.output_dir.resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("output_dir must be separate from the source CoT-swap run")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cot_swap_run"] = str(self.cot_swap_run.resolve())
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class AnswerLineDeletionInputUse:
    """Runtime proof of one exact fixed input and tokenization."""

    arm: str
    prompt_text_sha256: str
    pre_answer_text_sha256: str
    full_input_text_sha256: str
    prompt_char_count: int
    pre_answer_char_count: int
    full_input_char_count: int
    prompt_token_count: int
    full_input_token_count: int
    full_input_ids_sha256: str
    prompt_prefix_token_stable: bool

    def __post_init__(self) -> None:
        if self.arm not in ARM_ORDER:
            raise ValueError(f"unsupported input-use arm: {self.arm!r}")
        for field in (
            "prompt_text_sha256",
            "pre_answer_text_sha256",
            "full_input_text_sha256",
            "full_input_ids_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field} must be a full lowercase SHA-256")
        for field in (
            "prompt_char_count",
            "pre_answer_char_count",
            "full_input_char_count",
            "prompt_token_count",
            "full_input_token_count",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if not isinstance(self.prompt_prefix_token_stable, bool):
            raise TypeError("prompt_prefix_token_stable must be boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnswerLineDeletionGeneration:
    """One final-PDF 16-token-capped answer-span generation."""

    token_ids: tuple[int, ...]
    text: str
    value: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str
    stop_reason: Literal["eos_token", "max_new_tokens"]
    stop_token_id: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("token_ids must contain non-negative integers")
        for field in ("text", "value", "method", "primary_method"):
            if not isinstance(getattr(self, field), str):
                raise TypeError(f"{field} must be a string")
        if not isinstance(self.is_extracted, bool) or not isinstance(self.is_correct, bool):
            raise TypeError("generation extraction and correctness flags must be boolean")
        if self.stop_reason not in {"eos_token", "max_new_tokens"}:
            raise ValueError("stop_reason must be 'eos_token' or 'max_new_tokens'")
        if self.stop_token_id is not None and (
            not isinstance(self.stop_token_id, int)
            or isinstance(self.stop_token_id, bool)
            or self.stop_token_id < 0
        ):
            raise ValueError("stop_token_id must be null or a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        return payload


@dataclass(frozen=True, slots=True)
class AnswerLineDeletionScan:
    """Both control arms returned atomically for one source pair."""

    sample_id: str
    input_uses: Mapping[str, AnswerLineDeletionInputUse]
    generations: Mapping[str, AnswerLineDeletionGeneration]


class AnswerLineDeletionRuntime(Protocol):
    def provenance(self) -> Mapping[str, object]: ...

    def scan_pair(
        self, pair: dict[str, object], plan: AnswerLineDeletionPlan
    ) -> AnswerLineDeletionScan: ...


@dataclass(frozen=True, slots=True)
class AnswerLineDeletionResult:
    records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    source_cohort_pairs: int
    executed_pairs: int
    records: int


@dataclass(frozen=True, slots=True)
class _PlannedCase:
    source: CotSwapSourceCase
    plan: AnswerLineDeletionPlan


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _serialized_json(payload: object) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(path, _serialized_json(payload))


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _write_text_atomic(path, text)


def _remove_exact_files(paths: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def _load_json_with_sha256(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 at {path}: {exc}") from exc
    payload = strict_loads(text, context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
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
    return rows


def _build_plans(
    source: CompletedCotSwapSource,
    config: AnswerLineDeletionConfig,
) -> tuple[tuple[_PlannedCase, ...], tuple[_PlannedCase, ...], tuple[_PlannedCase, ...]]:
    cohort_sources = sorted(
        (case for case in source.cases if case.restoration_denominator),
        key=lambda case: case.sample_id,
    )
    planned = tuple(
        _PlannedCase(
            source=case,
            plan=build_answer_line_deletion_plan(
                case.pair,
                source_a_answer=case.source_a_answer,
                source_c_answer=case.source_c_answer,
                source_record_sha256=case.record_sha256,
                prepared_record_sha256=case.pair_record_sha256,
            ),
        )
        for case in cohort_sources
    )
    if not planned:
        raise ValueError("source CoT-swap run has no A-correct, B-changed control cases")
    capped = planned[: config.max_pairs]
    selected = capped[: config.limit] if config.limit is not None else capped
    if not selected:
        raise ValueError("answer-line deletion selected no cases")
    return planned, capped, selected


def _plan_row(item: _PlannedCase) -> dict[str, object]:
    return {
        "sample_id": item.source.sample_id,
        "source_record_sha256": item.source.record_sha256,
        "prepared_record_sha256": item.source.pair_record_sha256,
        "plan_sha256": item.plan.fingerprint,
        "prefix_became_empty": item.plan.deletion.prefix_became_empty,
    }


def _source_population_rows(source: CompletedCotSwapSource) -> list[dict[str, object]]:
    return [
        {
            "sample_id": case.sample_id,
            "source_record_sha256": case.record_sha256,
            "prepared_record_sha256": case.pair_record_sha256,
            "source_a_correct": case.source_a_correct,
            "source_b_changed_from_a": case.source_b_changed_from_a,
            "restoration_denominator": case.restoration_denominator,
        }
        for case in source.cases
    ]


def _plan_payload(
    source: CompletedCotSwapSource,
    cohort: Sequence[_PlannedCase],
    capped: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
) -> dict[str, object]:
    return {
        "arm_order": list(ARM_ORDER),
        "source_records": len(source.cases),
        "source_restoration_cohort_pairs": len(cohort),
        "capped_pairs": len(capped),
        "selected_pairs": len(selected),
        "source_population_fingerprint": canonical_sha256(_source_population_rows(source)),
        "cohort_fingerprint": canonical_sha256([_plan_row(item) for item in cohort]),
        "capped_fingerprint": canonical_sha256([_plan_row(item) for item in capped]),
        "selected_fingerprint": canonical_sha256([_plan_row(item) for item in selected]),
    }


def _comparability(config: AnswerLineDeletionConfig) -> dict[str, object]:
    requirements = {
        "final_pdf_declared_generation_and_estimand": True,
        "completed_unlimited_cot_swap_source": True,
        "random_4_source": True,
        "canonical_a_correct_b_changed_cohort": True,
        "answer_span_cap_16": True,
        "symmetric_fallback_both_arms": True,
        "one_pair_two_arm_batch": True,
        "max_pairs_150": config.max_pairs == 150,
        "not_limit_truncated": config.limit is None,
        "paper_model": config.model in _PAPER_MODELS,
    }
    limitations = [
        "final-line-locator-is-legacy-backed-and-not-defined-by-the-final-pdf",
        "single-line-prefixes-become-empty-so-the-intervention-is-not-local-in-those-cases",
        "historical-table-used-256-answer-tokens-while-final-pdf-specifies-16",
        "historical-gsm8k-b-flips-used-raw-string-rather-than-canonical-equality",
        "historical-source-lacks-raw-generation-token-and-model-revision-provenance",
        "fresh-complete-arm-is-a-diagnostic-rerun-and-can-differ-from-source-cell-c",
        "control-combines-content-removal-and-format-disruption",
        "two-arm-batch-is-a-public-implementation-choice-not-preserved-legacy-provenance",
    ]
    if config.limit is not None:
        limitations.append("run-is-limit-truncated")
        status = "partial-smoke-run"
    elif config.max_pairs != 150:
        limitations.append("max-pairs-differs-from-submitted-control-cap")
        status = "fresh-sensitivity-setting"
    elif config.model not in _PAPER_MODELS:
        limitations.append("model-is-outside-the-three-control-models")
        status = "fresh-protocol-setting"
    else:
        status = "fresh-final-pdf-priority-setting-with-legacy-backed-details"
    return {
        "status": status,
        "requirements": requirements,
        "limitations": limitations,
        "historical_cohort_identity": False,
    }


def _validate_runtime_provenance(
    provenance: Mapping[str, object],
    *,
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
) -> dict[str, object]:
    required = {
        "operation": "answer-line-deletion",
        "model": config.model,
        "requested_revision": source.model_revision,
        "model_revision": source.model_revision,
        "tokenizer_revision": source.model_revision,
        "dtype": "bfloat16",
        "cuda_visible_devices": config.gpu_id,
        "answer_extraction": _ANSWER_EXTRACTION,
        "implementation": _IMPLEMENTATION,
    }
    for field, expected in required.items():
        if provenance.get(field) != expected:
            raise ValueError(
                "runtime provenance does not match the public protocol: "
                f"{field} must be {expected!r}"
            )
    generation = _mapping(provenance.get("generation"), field="runtime generation")
    for field, expected in _GENERATION.items():
        actual = generation.get(field)
        if field not in generation or actual != expected or type(actual) is not type(expected):
            raise ValueError(
                "runtime generation does not match the public protocol: "
                f"{field} must be {expected!r}"
            )
    if provenance.get("text_intervention") != _TEXT_INTERVENTION:
        raise ValueError("runtime text intervention does not match the public protocol")
    if provenance.get("batching") != _BATCHING:
        raise ValueError("runtime batching does not match the one-pair two-arm protocol")
    if provenance.get("answer_span_decoding") != _ANSWER_SPAN_DECODING:
        raise ValueError("runtime answer-span decoding does not match the public protocol")
    eos_ids = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or eos_ids != sorted(set(eos_ids))
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos_ids
        )
    ):
        raise ValueError("runtime effective_eos_token_ids must be sorted unique token IDs")
    if provenance.get("effective_eos_token_ids_source") not in {
        "model-generation-config",
        "tokenizer-fallback",
    }:
        raise ValueError("runtime EOS token source is unsupported")
    if generation.get("eos_token_id") != eos_ids:
        raise ValueError("runtime generation eos_token_id must equal the effective EOS IDs")
    return dict(provenance)


def _validate_generation(
    generation: AnswerLineDeletionGeneration,
    *,
    field: str,
    benchmark: str,
    gold_answer: str,
    eos_ids: Sequence[int],
) -> None:
    if not isinstance(generation, AnswerLineDeletionGeneration):
        raise TypeError(f"{field} must be AnswerLineDeletionGeneration")
    if not 1 <= len(generation.token_ids) <= int(_GENERATION["max_new_tokens"]):
        raise ValueError(f"{field}.token_ids exceeds the final-PDF answer-span cap")
    eos_set = set(eos_ids)
    if generation.stop_reason == "eos_token":
        if (
            generation.stop_token_id not in eos_set
            or generation.token_ids[-1] != generation.stop_token_id
            or any(token in eos_set for token in generation.token_ids[:-1])
        ):
            raise ValueError(f"{field} EOS metadata does not match its token IDs")
    elif (
        len(generation.token_ids) != int(_GENERATION["max_new_tokens"])
        or generation.stop_token_id is not None
        or any(token in eos_set for token in generation.token_ids)
    ):
        raise ValueError(f"{field} max-token termination metadata is inconsistent")
    if generation.is_extracted is not bool(generation.value):
        raise ValueError(f"{field}.is_extracted does not match its value")
    expected = extract_with_fallback(
        generation.text,
        benchmark=_BENCHMARK_NAMES[benchmark],
        correct_answer=gold_answer,
        allow_positional=generation.stop_reason == "eos_token",
    )
    observed = (
        generation.value,
        generation.is_extracted,
        generation.is_correct,
        generation.method,
        generation.primary_method,
    )
    if observed != (
        expected.value,
        expected.is_extracted,
        expected.is_correct,
        expected.method,
        expected.primary_method,
    ):
        raise ValueError(f"{field} extraction does not match generated text")


def _validate_input_use(
    input_use: AnswerLineDeletionInputUse,
    arm: AnswerLineArmPlan,
) -> None:
    if not isinstance(input_use, AnswerLineDeletionInputUse):
        raise TypeError(f"runtime input use for {arm.arm} must be AnswerLineDeletionInputUse")
    expected = {
        "arm": arm.arm,
        "prompt_text_sha256": arm.prompt_sha256,
        "pre_answer_text_sha256": arm.pre_answer_sha256,
        "full_input_text_sha256": arm.full_input_sha256,
        "prompt_char_count": len(arm.prompt),
        "pre_answer_char_count": len(arm.pre_answer_text),
        "full_input_char_count": len(arm.full_input),
        "prompt_token_count": arm.prompt_token_count,
    }
    if any(getattr(input_use, field) != value for field, value in expected.items()):
        raise ValueError(f"runtime fixed input does not match the plan for {arm.arm}")
    if input_use.full_input_token_count <= 0:
        raise ValueError(f"runtime {arm.arm} full input contains no token")


def _validate_scan(
    scan: AnswerLineDeletionScan,
    *,
    item: _PlannedCase,
    config: AnswerLineDeletionConfig,
    eos_ids: Sequence[int],
) -> None:
    if not isinstance(scan, AnswerLineDeletionScan):
        raise TypeError("runtime must return AnswerLineDeletionScan")
    if scan.sample_id != item.source.sample_id:
        raise ValueError("runtime scan sample_id does not match its source pair")
    if tuple(scan.input_uses) != ARM_ORDER or tuple(scan.generations) != ARM_ORDER:
        raise ValueError("runtime scan must contain both arms in protocol order")
    gold_answer = item.source.pair.get("gold_answer")
    if not isinstance(gold_answer, str) or not gold_answer:
        raise ValueError("prepared pair gold_answer must be non-empty")
    for arm in item.plan.arms:
        _validate_input_use(scan.input_uses[arm.arm], arm)
        _validate_generation(
            scan.generations[arm.arm],
            field=arm.arm,
            benchmark=config.benchmark,
            gold_answer=gold_answer,
            eos_ids=eos_ids,
        )


def _input_use_from_dict(value: object) -> AnswerLineDeletionInputUse:
    return AnswerLineDeletionInputUse(**dict(_mapping(value, field="checkpoint input use")))  # type: ignore[arg-type]


def _generation_from_dict(value: object) -> AnswerLineDeletionGeneration:
    row = dict(_mapping(value, field="checkpoint generation"))
    token_ids = row.get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError("checkpoint generation token_ids must be a list")
    row["token_ids"] = tuple(token_ids)
    return AnswerLineDeletionGeneration(**row)  # type: ignore[arg-type]


def _checkpoint_identity(sample_id: str) -> str:
    return hashlib.sha256(f"answer-line-deletion\0{sample_id}".encode()).hexdigest()


def _checkpoint_path(directory: Path, sample_id: str) -> Path:
    return directory / f"{_checkpoint_identity(sample_id)}.json"


def _checkpoint_payload(
    scan: AnswerLineDeletionScan,
    *,
    item: _PlannedCase,
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
    runtime_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "answer-line-deletion-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": _PROTOCOL_SHA256,
        "model": config.model,
        "benchmark": config.benchmark,
        "sample_id": item.source.sample_id,
        "cot_swap_run_sha256": source.run_sha256,
        "cot_swap_records_sha256": source.records_sha256,
        "source_record_sha256": item.source.record_sha256,
        "prepared_pairs_sha256": source.pairs_sha256,
        "prepared_record_sha256": item.source.pair_record_sha256,
        "plan_sha256": item.plan.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "input_uses": {arm: scan.input_uses[arm].to_dict() for arm in ARM_ORDER},
        "generations": {arm: scan.generations[arm].to_dict() for arm in ARM_ORDER},
    }


def _scan_from_checkpoint(
    payload: Mapping[str, object],
    *,
    item: _PlannedCase,
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
    runtime_fingerprint: str,
    eos_ids: Sequence[int],
) -> AnswerLineDeletionScan:
    expected = {
        "schema_version": "answer-line-deletion-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": _PROTOCOL_SHA256,
        "model": config.model,
        "benchmark": config.benchmark,
        "sample_id": item.source.sample_id,
        "cot_swap_run_sha256": source.run_sha256,
        "cot_swap_records_sha256": source.records_sha256,
        "source_record_sha256": item.source.record_sha256,
        "prepared_pairs_sha256": source.pairs_sha256,
        "prepared_record_sha256": item.source.pair_record_sha256,
        "plan_sha256": item.plan.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ValueError(f"checkpoint provenance does not match pair {item.source.sample_id}")
    raw_uses = _mapping(payload.get("input_uses"), field="checkpoint input_uses")
    raw_generations = _mapping(payload.get("generations"), field="checkpoint generations")
    if set(raw_uses) != set(ARM_ORDER) or set(raw_generations) != set(ARM_ORDER):
        raise ValueError("checkpoint must contain both control arms exactly")
    scan = AnswerLineDeletionScan(
        sample_id=item.source.sample_id,
        input_uses={arm: _input_use_from_dict(raw_uses[arm]) for arm in ARM_ORDER},
        generations={arm: _generation_from_dict(raw_generations[arm]) for arm in ARM_ORDER},
    )
    _validate_scan(scan, item=item, config=config, eos_ids=eos_ids)
    reconstructed = _checkpoint_payload(
        scan,
        item=item,
        config=config,
        source=source,
        runtime_fingerprint=runtime_fingerprint,
    )
    if _serialized_json(dict(payload)) != _serialized_json(reconstructed):
        raise ValueError(
            f"checkpoint payload does not match reconstructed pair {item.source.sample_id}"
        )
    return scan


def _checkpoint_registry_entry(
    path: Path,
    *,
    item: _PlannedCase,
    sha256: str,
) -> dict[str, object]:
    return {
        "file": path.name,
        "sha256": sha256,
        "sample_id": item.source.sample_id,
    }


def _load_registered_checkpoints(
    registry: Mapping[str, object],
    *,
    checkpoints_dir: Path,
    selected: Sequence[_PlannedCase],
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
    runtime_fingerprint: str,
    eos_ids: Sequence[int],
) -> tuple[dict[str, AnswerLineDeletionScan], dict[str, object]]:
    by_id = {item.source.sample_id: item for item in selected}
    expected_files: set[str] = set()
    scans: dict[str, AnswerLineDeletionScan] = {}
    normalized: dict[str, object] = {}
    for identity, raw_metadata in registry.items():
        metadata = _mapping(raw_metadata, field=f"checkpoint registry {identity!r}")
        if set(metadata) != {"file", "sha256", "sample_id"}:
            raise ValueError("checkpoint registry has an invalid shape")
        sample_id = metadata.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in by_id:
            raise ValueError("checkpoint registry references an unknown selected pair")
        if identity != _checkpoint_identity(sample_id):
            raise ValueError("checkpoint registry identity does not match sample_id")
        item = by_id[sample_id]
        path = _checkpoint_path(checkpoints_dir, sample_id)
        expected_files.add(path.name)
        if metadata.get("file") != path.name or not path.is_file():
            raise ValueError(f"checkpoint file or hash mismatch for {sample_id}")
        payload, actual_sha = _load_json_with_sha256(path)
        if metadata.get("sha256") != actual_sha:
            raise ValueError(f"checkpoint file or hash mismatch for {sample_id}")
        scans[sample_id] = _scan_from_checkpoint(
            payload,
            item=item,
            config=config,
            source=source,
            runtime_fingerprint=runtime_fingerprint,
            eos_ids=eos_ids,
        )
        normalized[identity] = dict(metadata)

    for item in selected:
        sample_id = item.source.sample_id
        identity = _checkpoint_identity(sample_id)
        if identity in normalized:
            continue
        path = _checkpoint_path(checkpoints_dir, sample_id)
        if not path.is_file():
            continue
        expected_files.add(path.name)
        payload, actual_sha = _load_json_with_sha256(path)
        scans[sample_id] = _scan_from_checkpoint(
            payload,
            item=item,
            config=config,
            source=source,
            runtime_fingerprint=runtime_fingerprint,
            eos_ids=eos_ids,
        )
        normalized[identity] = _checkpoint_registry_entry(
            path,
            item=item,
            sha256=actual_sha,
        )
    actual_files = {path.name for path in checkpoints_dir.glob("*.json")}
    if actual_files != expected_files:
        raise ValueError("checkpoint directory contains unregistered or missing files")
    return scans, normalized


def _events(
    scan: AnswerLineDeletionScan,
    *,
    item: _PlannedCase,
    benchmark: str,
) -> dict[str, bool]:
    extraction_benchmark = _BENCHMARK_NAMES[benchmark]
    complete = scan.generations["complete"].value
    deleted = scan.generations["answer-line-deleted"].value
    source_a = item.plan.source_a_answer
    complete_restored = answers_equal(complete, source_a, benchmark=extraction_benchmark)
    deleted_restored = answers_equal(deleted, source_a, benchmark=extraction_benchmark)
    return {
        "complete_matches_source_c": answers_equal(
            complete,
            item.plan.source_c_answer,
            benchmark=extraction_benchmark,
        ),
        "complete_restored_to_a": complete_restored,
        "answer_line_deleted_restored_to_a": deleted_restored,
        "restoration_lost_after_deletion": complete_restored and not deleted_restored,
        "restoration_gained_after_deletion": not complete_restored and deleted_restored,
    }


def _record(
    scan: AnswerLineDeletionScan,
    *,
    item: _PlannedCase,
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
) -> dict[str, object]:
    events = _events(scan, item=item, benchmark=config.benchmark)
    arms: dict[str, object] = {}
    for arm_plan in item.plan.arms:
        arm = arm_plan.arm
        arms[arm] = {
            "fixed_input": arm_plan.to_dict(include_text=False),
            "input_use": scan.input_uses[arm].to_dict(),
            "answer": scan.generations[arm].to_dict(),
            "restored_to_source_a": events[
                "complete_restored_to_a"
                if arm == "complete"
                else "answer_line_deleted_restored_to_a"
            ],
        }
    return {
        "schema_version": "answer-line-deletion-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "answer-line-deletion",
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": "random-4",
        "sample_id": item.source.sample_id,
        "gold_answer": item.source.pair["gold_answer"],
        "source": {
            "cot_swap_run_sha256": source.run_sha256,
            "cot_swap_records_sha256": source.records_sha256,
            "cot_swap_record_sha256": item.source.record_sha256,
            "prepared_pairs_sha256": source.pairs_sha256,
            "prepared_record_sha256": item.source.pair_record_sha256,
            "source_a_answer": item.plan.source_a_answer,
            "source_c_answer": item.plan.source_c_answer,
        },
        "deletion": item.plan.deletion.to_dict(include_text=False),
        "arms": arms,
        "events": events,
    }


def _status_rows(
    source: CompletedCotSwapSource,
    cohort: Sequence[_PlannedCase],
    capped: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
    scans: Mapping[str, AnswerLineDeletionScan],
    *,
    config: AnswerLineDeletionConfig,
) -> list[dict[str, object]]:
    planned_by_id = {item.source.sample_id: item for item in cohort}
    capped_ids = {item.source.sample_id for item in capped}
    selected_ids = {item.source.sample_id for item in selected}
    rows: list[dict[str, object]] = []
    for case in source.cases:
        item = planned_by_id.get(case.sample_id)
        in_cohort = item is not None
        within_cap = case.sample_id in capped_ids
        selected_for_execution = case.sample_id in selected_ids
        if not in_cohort:
            status = "not-restoration-denominator"
        elif not within_cap:
            status = "beyond-max-pairs"
        elif not selected_for_execution:
            status = "limit-truncated"
        elif case.sample_id in scans:
            status = "completed"
        else:
            status = "failed"
        rows.append(
            {
                "schema_version": "answer-line-deletion-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "targeting": "random-4",
                "sample_id": case.sample_id,
                "source_record_sha256": case.record_sha256,
                "prepared_record_sha256": case.pair_record_sha256,
                "plan_sha256": item.plan.fingerprint if item is not None else None,
                "source_a_correct": case.source_a_correct,
                "source_b_changed_from_a": case.source_b_changed_from_a,
                "source_restoration_denominator": in_cohort,
                "cohort_exclusion_reason": (
                    None
                    if in_cohort
                    else (
                        "source-a-not-correct" if not case.source_a_correct else "source-b-equals-a"
                    )
                ),
                "within_max_pairs": within_cap,
                "selected_for_execution": selected_for_execution,
                "prefix_became_empty": (
                    item.plan.deletion.prefix_became_empty if item is not None else None
                ),
                "execution_status": status,
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _metrics(
    items: Sequence[_PlannedCase],
    scans: Mapping[str, AnswerLineDeletionScan],
    *,
    benchmark: str,
) -> dict[str, object]:
    events = [
        _events(scans[item.source.sample_id], item=item, benchmark=benchmark) for item in items
    ]
    denominator = len(events)
    return {
        "complete_restoration": _rate(
            sum(event["complete_restored_to_a"] for event in events), denominator
        ),
        "answer_line_deleted_restoration": _rate(
            sum(event["answer_line_deleted_restored_to_a"] for event in events), denominator
        ),
        "complete_matches_source_c": _rate(
            sum(event["complete_matches_source_c"] for event in events), denominator
        ),
        "restoration_lost_after_deletion": _rate(
            sum(event["restoration_lost_after_deletion"] for event in events), denominator
        ),
        "restoration_gained_after_deletion": _rate(
            sum(event["restoration_gained_after_deletion"] for event in events), denominator
        ),
        "restoration_rate_difference_deleted_minus_complete": (
            (
                sum(event["answer_line_deleted_restored_to_a"] for event in events)
                - sum(event["complete_restored_to_a"] for event in events)
            )
            / denominator
            if denominator
            else None
        ),
    }


def _diagnostics(
    items: Sequence[_PlannedCase],
    scans: Mapping[str, AnswerLineDeletionScan],
) -> dict[str, object]:
    return {
        "unextractable_by_arm": {
            arm: sum(
                not scans[item.source.sample_id].generations[arm].is_extracted for item in items
            )
            for arm in ARM_ORDER
        },
        "stop_reason_by_arm": {
            arm: {
                reason: sum(
                    scans[item.source.sample_id].generations[arm].stop_reason == reason
                    for item in items
                )
                for reason in ("eos_token", "max_new_tokens")
            }
            for arm in ARM_ORDER
        },
        "extraction_method_by_arm_and_stop_reason": {
            arm: {
                reason: dict(
                    sorted(
                        Counter(
                            scans[item.source.sample_id].generations[arm].method
                            for item in items
                            if scans[item.source.sample_id].generations[arm].stop_reason == reason
                        ).items()
                    )
                )
                for reason in ("eos_token", "max_new_tokens")
            }
            for arm in ARM_ORDER
        },
        "prompt_prefix_token_unstable_by_arm": {
            arm: sum(
                not scans[item.source.sample_id].input_uses[arm].prompt_prefix_token_stable
                for item in items
            )
            for arm in ARM_ORDER
        },
    }


def _summary(
    source: CompletedCotSwapSource,
    cohort: Sequence[_PlannedCase],
    capped: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
    scans: Mapping[str, AnswerLineDeletionScan],
    *,
    config: AnswerLineDeletionConfig,
) -> dict[str, object]:
    ordered = [item for item in selected if item.source.sample_id in scans]
    by_empty = {
        key: [item for item in ordered if item.plan.deletion.prefix_became_empty is value]
        for key, value in (("false", False), ("true", True))
    }
    diagnostics = _diagnostics(ordered, scans)
    return {
        "schema_version": "answer-line-deletion-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "answer-line-deletion",
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": "random-4",
        "source": source.to_dict(),
        "counts": {
            "source_cot_swap_records": len(source.cases),
            "source_restoration_cohort_pairs": len(cohort),
            "capped_pairs": len(capped),
            "selected_pairs": len(selected),
            "executed_pairs": len(ordered),
            "failed_pairs": 0,
            "prefix_became_empty_pairs": sum(
                item.plan.deletion.prefix_became_empty for item in ordered
            ),
        },
        "metrics": _metrics(ordered, scans, benchmark=config.benchmark),
        "metrics_by_prefix_became_empty": {
            key: _metrics(items, scans, benchmark=config.benchmark)
            for key, items in by_empty.items()
        },
        **diagnostics,
        "diagnostics_by_prefix_became_empty": {
            key: _diagnostics(items, scans) for key, items in by_empty.items()
        },
        "published_reference": json.loads(json.dumps(_PUBLISHED_REFERENCE)),
        "analysis_status": "paired-descriptive-content-and-format-control",
        "comparability": _comparability(config),
    }


def _counts(
    source: CompletedCotSwapSource,
    cohort: Sequence[_PlannedCase],
    capped: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
    checkpoints: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    selected_ids = {item.source.sample_id for item in selected}
    return {
        "source_cot_swap_records": len(source.cases),
        "source_restoration_cohort_pairs": len(cohort),
        "capped_pairs": len(capped),
        "selected_pairs": len(selected),
        "checkpointed_pairs": len(checkpoints),
        "failed_pairs": sum(failure.get("sample_id") in selected_ids for failure in failures),
        "publication_failures": sum(failure.get("stage") == "publication" for failure in failures),
    }


def _manifest(
    *,
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
    plan: Mapping[str, object],
    status: str,
    started_at: str,
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
    checkpoints: Mapping[str, object],
    runtime: Mapping[str, object] | None,
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "answer-line-deletion-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "answer-line-deletion",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "protocol_sha256": _PROTOCOL_SHA256,
        "source": source.to_dict(),
        "plan": dict(plan),
        "runtime": dict(runtime) if runtime is not None else None,
        "checkpoints": dict(checkpoints),
        "counts": dict(counts),
        "failures": list(failures),
        "comparability": _comparability(config),
        "started_at": started_at,
        "updated_at": _now(),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    return payload


def _validate_resume_core(
    previous: Mapping[str, object],
    *,
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
    plan: Mapping[str, object],
) -> str:
    if previous.get("schema_version") != "answer-line-deletion-run/v1":
        raise ValueError("cannot resume an unknown answer-line deletion schema")
    if previous.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("resume paper SHA-256 does not match")
    if previous.get("operation") != "answer-line-deletion":
        raise ValueError("resume operation does not match answer-line-deletion")
    if previous.get("status") not in {"running", "failed", "completed"}:
        raise ValueError("resume status is not recognized")
    if previous.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run")
    if previous.get("protocol") != _PROTOCOL or previous.get("protocol_sha256") != _PROTOCOL_SHA256:
        raise ValueError("resume protocol fingerprint does not match")
    if previous.get("source") != source.to_dict():
        raise ValueError("resume source fingerprint does not match")
    if previous.get("plan") != plan:
        raise ValueError("resume deterministic plan fingerprint does not match")
    started_at = previous.get("started_at")
    return started_at if isinstance(started_at, str) else _now()


def _output_metadata(path: Path, records: int) -> dict[str, object]:
    return {"sha256": sha256_file(path), "records": records}


def _validate_output_hashes(manifest: Mapping[str, object], output_dir: Path) -> None:
    outputs = _mapping(manifest.get("outputs"), field="completed outputs")
    expected = {
        "answer_line_deletion_records.jsonl",
        "pair_status_records.jsonl",
        "answer_line_deletion_summary.json",
    }
    if set(outputs) != expected:
        raise _CompletedOutputIntegrityError("completed output registry is incomplete")
    for name in expected:
        metadata = _mapping(outputs[name], field=f"completed output {name}")
        path = output_dir / name
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise _CompletedOutputIntegrityError(f"completed output hash mismatch: {name}")


def _validate_completed_checkpoint_registry(
    registry: Mapping[str, object],
    *,
    selected: Sequence[_PlannedCase],
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
    runtime_fingerprint: str,
    scans: Mapping[str, AnswerLineDeletionScan],
) -> None:
    by_id = {item.source.sample_id: item for item in selected}
    observed: set[str] = set()
    for identity, raw_metadata in registry.items():
        metadata = _mapping(raw_metadata, field=f"completed checkpoint {identity!r}")
        if set(metadata) != {"file", "sha256", "sample_id"}:
            raise ValueError("completed checkpoint registry has an invalid shape")
        sample_id = metadata.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in by_id:
            raise ValueError("completed checkpoint references an unknown selected pair")
        if identity != _checkpoint_identity(sample_id):
            raise ValueError("completed checkpoint identity does not match sample_id")
        if (
            metadata.get("file") != _checkpoint_path(Path("."), sample_id).name
            or not isinstance(metadata.get("sha256"), str)
            or _SHA256.fullmatch(str(metadata["sha256"])) is None
        ):
            raise ValueError("completed checkpoint metadata does not match its pair")
        if sample_id in observed:
            raise ValueError("completed checkpoint registry contains a duplicate pair")
        item = by_id[sample_id]
        payload = _checkpoint_payload(
            scans[sample_id],
            item=item,
            config=config,
            source=source,
            runtime_fingerprint=runtime_fingerprint,
        )
        expected_sha = hashlib.sha256(_serialized_json(payload).encode("utf-8")).hexdigest()
        if metadata.get("sha256") != expected_sha:
            raise ValueError("completed checkpoint hash does not match reconstructed payload")
        observed.add(sample_id)
    if observed != set(by_id):
        raise ValueError("completed checkpoint registry does not cover every selected pair")


def _validate_completed_semantics(
    manifest: Mapping[str, object],
    *,
    config: AnswerLineDeletionConfig,
    source: CompletedCotSwapSource,
    cohort: Sequence[_PlannedCase],
    capped: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
) -> AnswerLineDeletionResult:
    output_dir = config.output_dir
    records_path = output_dir / "answer_line_deletion_records.jsonl"
    statuses_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "answer_line_deletion_summary.json"
    run_path = output_dir / "run.json"
    try:
        _validate_output_hashes(manifest, output_dir)
        runtime = _validate_runtime_provenance(
            _mapping(manifest.get("runtime"), field="completed runtime"),
            config=config,
            source=source,
        )
        eos_ids = tuple(runtime["effective_eos_token_ids"])
        if manifest.get("comparability") != _comparability(config):
            raise ValueError("completed comparability metadata is inconsistent")
        if manifest.get("failures") != []:
            raise ValueError("completed run must not retain failures")
        records = _load_jsonl(records_path)
        statuses = _load_jsonl(statuses_path)
        summary = load_json(summary_path)
        expected_ids = [item.source.sample_id for item in selected]
        if [row.get("sample_id") for row in records] != expected_ids:
            raise ValueError("completed records do not match selected sample order")
        by_id = {item.source.sample_id: item for item in selected}
        scans: dict[str, AnswerLineDeletionScan] = {}
        for row in records:
            sample_id = str(row["sample_id"])
            item = by_id[sample_id]
            if (
                row.get("schema_version") != "answer-line-deletion-record/v1"
                or row.get("paper_sha256") != PAPER_SHA256
                or row.get("model") != config.model
                or row.get("benchmark") != config.benchmark
                or row.get("targeting") != "random-4"
            ):
                raise ValueError("completed record setting or schema mismatch")
            arms = _mapping(row.get("arms"), field="completed record arms")
            if set(arms) != set(ARM_ORDER):
                raise ValueError("completed record is missing a control arm")
            uses: dict[str, AnswerLineDeletionInputUse] = {}
            generations: dict[str, AnswerLineDeletionGeneration] = {}
            for arm in ARM_ORDER:
                arm_row = _mapping(arms[arm], field=f"completed arm {arm}")
                uses[arm] = _input_use_from_dict(arm_row.get("input_use"))
                generations[arm] = _generation_from_dict(arm_row.get("answer"))
            scan = AnswerLineDeletionScan(
                sample_id=sample_id,
                input_uses=uses,
                generations=generations,
            )
            _validate_scan(scan, item=item, config=config, eos_ids=eos_ids)
            if row != _record(scan, item=item, config=config, source=source):
                raise ValueError("completed record semantics do not match source and answers")
            scans[sample_id] = scan
        expected_statuses = _status_rows(
            source,
            cohort,
            capped,
            selected,
            scans,
            config=config,
        )
        if statuses != expected_statuses:
            raise ValueError("completed pair statuses do not match source and plan")
        expected_summary = _summary(
            source,
            cohort,
            capped,
            selected,
            scans,
            config=config,
        )
        if summary != expected_summary:
            raise ValueError("completed summary does not match records and plan")
        registry = _mapping(manifest.get("checkpoints"), field="completed checkpoints")
        _validate_completed_checkpoint_registry(
            registry,
            selected=selected,
            config=config,
            source=source,
            runtime_fingerprint=canonical_sha256(runtime),
            scans=scans,
        )
        if manifest.get("counts") != _counts(
            source,
            cohort,
            capped,
            selected,
            registry,
            [],
        ):
            raise ValueError("completed run counts are inconsistent")
        outputs = _mapping(manifest.get("outputs"), field="completed outputs")
        expected_outputs = {
            records_path.name: _output_metadata(records_path, len(records)),
            statuses_path.name: _output_metadata(statuses_path, len(statuses)),
            summary_path.name: _output_metadata(summary_path, 1),
        }
        if outputs != expected_outputs:
            raise ValueError("completed output registry metadata is inconsistent")
    except (KeyError, TypeError, ValueError, AnswerLineDeletionRunError) as exc:
        if isinstance(exc, _CompletedOutputIntegrityError):
            raise
        raise AnswerLineDeletionRunError(f"completed run validation failed: {exc}") from exc
    return AnswerLineDeletionResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        source_cohort_pairs=len(cohort),
        executed_pairs=len(records),
        records=len(records),
    )


def run_answer_line_deletion(
    config: AnswerLineDeletionConfig,
    *,
    runtime: AnswerLineDeletionRuntime | None = None,
) -> AnswerLineDeletionResult:
    """Validate, execute, checkpoint, and publish one paired control setting."""

    output_dir = config.output_dir
    run_path = output_dir / "run.json"
    records_path = output_dir / "answer_line_deletion_records.jsonl"
    statuses_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "answer_line_deletion_summary.json"
    public_paths = (records_path, statuses_path, summary_path)
    work_dir = output_dir / ".answer-line-deletion-work"
    checkpoints_dir = work_dir / "checkpoints"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    source = load_completed_cot_swap_source(
        config.cot_swap_run,
        model=config.model,
        benchmark=config.benchmark,
    )
    cohort, capped, selected = _build_plans(source, config)
    plan_payload = _plan_payload(source, cohort, capped, selected)

    previous: dict[str, object] | None = None
    if config.resume:
        previous = load_json(run_path)
        started_at = _validate_resume_core(
            previous,
            config=config,
            source=source,
            plan=plan_payload,
        )
        if previous.get("status") == "completed":
            return _validate_completed_semantics(
                previous,
                config=config,
                source=source,
                cohort=cohort,
                capped=capped,
                selected=selected,
            )
        cleanup_errors = _remove_exact_files(public_paths)
        if cleanup_errors:
            raise AnswerLineDeletionRunError(
                "could not remove stale public outputs before resume: " + "; ".join(cleanup_errors)
            )
    else:
        started_at = _now()

    if previous is None:
        if runtime is None:
            from typo_cot.experiments.answer_line_deletion.runtime import (
                HuggingFaceAnswerLineDeletionRuntime,
            )

            runtime = HuggingFaceAnswerLineDeletionRuntime(
                config,
                revision=source.model_revision,
            )
        runtime_provenance = _validate_runtime_provenance(
            runtime.provenance(),
            config=config,
            source=source,
        )
    else:
        runtime_provenance = _validate_runtime_provenance(
            _mapping(previous.get("runtime"), field="resume runtime"),
            config=config,
            source=source,
        )
    runtime_fingerprint = canonical_sha256(runtime_provenance)
    eos_ids = tuple(runtime_provenance["effective_eos_token_ids"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    previous_registry = (
        _mapping(previous.get("checkpoints"), field="resume checkpoints")
        if previous is not None
        else {}
    )
    scans, checkpoint_registry = _load_registered_checkpoints(
        previous_registry,
        checkpoints_dir=checkpoints_dir,
        selected=selected,
        config=config,
        source=source,
        runtime_fingerprint=runtime_fingerprint,
        eos_ids=eos_ids,
    )
    pending = tuple(item for item in selected if item.source.sample_id not in scans)
    if pending:
        if runtime is None:
            from typo_cot.experiments.answer_line_deletion.runtime import (
                HuggingFaceAnswerLineDeletionRuntime,
            )

            runtime = HuggingFaceAnswerLineDeletionRuntime(
                config,
                revision=source.model_revision,
            )
        active_provenance = _validate_runtime_provenance(
            runtime.provenance(),
            config=config,
            source=source,
        )
        if active_provenance != runtime_provenance:
            raise ValueError("resume runtime provenance differs from the original run")
    failures: list[dict[str, str]] = []

    def write_progress(status: str) -> None:
        _write_json_atomic(
            run_path,
            _manifest(
                config=config,
                source=source,
                plan=plan_payload,
                status=status,
                started_at=started_at,
                counts=_counts(
                    source,
                    cohort,
                    capped,
                    selected,
                    checkpoint_registry,
                    failures,
                ),
                failures=failures,
                checkpoints=checkpoint_registry,
                runtime=runtime_provenance,
            ),
        )

    def should_flush_progress() -> bool:
        count = len(checkpoint_registry)
        return count > 0 and count & (count - 1) == 0

    write_progress("running")
    for item in tqdm(
        pending,
        desc="answer-line-deletion",
        unit="pair",
        initial=len(scans),
        total=len(selected),
        disable=None,
    ):
        if runtime is None:
            raise AssertionError("pending answer-line deletion work has no runtime")
        checkpoint_added = False
        try:
            scan = runtime.scan_pair(item.source.pair, item.plan)
            _validate_scan(scan, item=item, config=config, eos_ids=eos_ids)
            path = _checkpoint_path(checkpoints_dir, item.source.sample_id)
            payload = _checkpoint_payload(
                scan,
                item=item,
                config=config,
                source=source,
                runtime_fingerprint=runtime_fingerprint,
            )
            _write_json_atomic(path, payload)
            checkpoint_sha = hashlib.sha256(_serialized_json(payload).encode("utf-8")).hexdigest()
            identity = _checkpoint_identity(item.source.sample_id)
            checkpoint_registry[identity] = _checkpoint_registry_entry(
                path,
                item=item,
                sha256=checkpoint_sha,
            )
            scans[item.source.sample_id] = scan
            checkpoint_added = True
        except Exception as exc:
            failures.append(
                {
                    "stage": "pair",
                    "sample_id": item.source.sample_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        if checkpoint_added and should_flush_progress():
            write_progress("running")

    if failures:
        write_progress("failed")
        raise AnswerLineDeletionRunError(
            f"{len(failures)} pair(s) failed; first failure: {failures[0]['message']}; "
            "inspect run.json and rerun with --resume"
        )
    if len(scans) != len(selected):
        write_progress("failed")
        raise AnswerLineDeletionRunError("not every selected pair has a valid checkpoint")

    ordered_scans = {item.source.sample_id: scans[item.source.sample_id] for item in selected}
    records = [
        _record(
            ordered_scans[item.source.sample_id],
            item=item,
            config=config,
            source=source,
        )
        for item in selected
    ]
    statuses = _status_rows(
        source,
        cohort,
        capped,
        selected,
        ordered_scans,
        config=config,
    )
    summary = _summary(
        source,
        cohort,
        capped,
        selected,
        ordered_scans,
        config=config,
    )

    try:
        validate_source_snapshot(source)
        cleanup_errors = _remove_exact_files(public_paths)
        if cleanup_errors:
            raise OSError("could not remove stale public outputs: " + "; ".join(cleanup_errors))
        _write_jsonl_atomic(records_path, records)
        _write_jsonl_atomic(statuses_path, statuses)
        _write_json_atomic(summary_path, summary)
        outputs = {
            records_path.name: _output_metadata(records_path, len(records)),
            statuses_path.name: _output_metadata(statuses_path, len(statuses)),
            summary_path.name: _output_metadata(summary_path, 1),
        }
        validate_source_snapshot(source)
        _write_json_atomic(
            run_path,
            _manifest(
                config=config,
                source=source,
                plan=plan_payload,
                status="completed",
                started_at=started_at,
                counts=_counts(
                    source,
                    cohort,
                    capped,
                    selected,
                    checkpoint_registry,
                    [],
                ),
                failures=[],
                checkpoints=checkpoint_registry,
                runtime=runtime_provenance,
                outputs=outputs,
            ),
        )
    except BaseException as exc:
        cleanup_errors = _remove_exact_files(public_paths)
        failures.append(
            {
                "stage": "publication",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        try:
            write_progress("failed")
        except BaseException as manifest_exc:
            detail = f"; additionally failed to record run status: {manifest_exc}"
        else:
            detail = ""
        if cleanup_errors:
            detail += "; public-output cleanup failed: " + "; ".join(cleanup_errors)
        if not isinstance(exc, Exception):
            raise
        raise AnswerLineDeletionRunError(
            f"publication failed: {type(exc).__name__}: {exc}{detail}; "
            "checkpoints were retained for --resume"
        ) from exc

    with suppress(OSError):
        shutil.rmtree(work_dir)
    return AnswerLineDeletionResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        source_cohort_pairs=len(cohort),
        executed_pairs=len(selected),
        records=len(records),
    )
