"""Restartable producer for one Appendix E input-corrector setting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.input_corrector_audit.protocol import (
    CORE_BENCHMARKS,
    CORRECTOR_IDS,
    MATH_BENCHMARK,
    MATH_DIAGNOSTIC_CORRECTORS,
    PAPER_MODELS,
    PROTOCOL_SHA256,
)
from typo_cot.experiments.input_corrector_audit.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.input_corrector_audit.restoration import (
    classify_restoration,
    is_byte_exact_equal,
)
from typo_cot.experiments.input_corrector_audit.source import (
    InputCorrectorSource,
    load_input_corrector_source,
)

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RECORDS_NAME = "corrector_records.jsonl"
_SUMMARY_NAME = "corrector_audit_summary.json"
_WORK_NAME = ".input-corrector-audit-work"


class InputCorrectorAuditRunError(RuntimeError):
    """A setting failed after retaining its validated checkpoints."""


@dataclass(frozen=True, slots=True)
class InputCorrectorAuditConfig:
    """Frozen public arguments for one model/task/corrector setting."""

    corrector: str
    model: str
    benchmark: str
    pairs: Path
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", Path(self.pairs))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.corrector not in CORRECTOR_IDS:
            raise ValueError(f"unsupported input corrector: {self.corrector!r}")
        if self.model not in PAPER_MODELS:
            raise ValueError(f"model is outside the paper grid: {self.model!r}")
        if self.benchmark not in (*CORE_BENCHMARKS, MATH_BENCHMARK):
            raise ValueError(f"benchmark is outside the paper contract: {self.benchmark!r}")
        if self.benchmark == MATH_BENCHMARK and self.corrector not in MATH_DIAGNOSTIC_CORRECTORS:
            raise ValueError("the MATH-500 diagnostic only covers the T5 and Qwen correctors")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        if type(self.resume) is not bool:
            raise TypeError("resume must be boolean")


@dataclass(frozen=True, slots=True)
class CorrectionOutcome:
    """One correction result, including fail-closed parser diagnostics."""

    corrected_text: str
    parse_failed: bool
    n_calls: int
    raw_response: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.corrected_text, str):
            raise TypeError("corrected_text must be a string")
        if type(self.parse_failed) is not bool:
            raise TypeError("parse_failed must be boolean")
        if not isinstance(self.n_calls, int) or isinstance(self.n_calls, bool) or self.n_calls <= 0:
            raise ValueError("n_calls must be a positive integer")
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be a string or null")

    def to_dict(self) -> dict[str, object]:
        return {
            "corrected_text": self.corrected_text,
            "parse_failed": self.parse_failed,
            "n_calls": self.n_calls,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True, slots=True)
class SamePromptGeneration:
    """One row from an adjacent duplicate-prompt generation call."""

    sample_id: str
    token_ids: tuple[int, ...]
    text: str
    extracted_answer: str
    is_extracted: bool
    is_correct: bool
    method: str = "unrecorded"
    primary_method: str = "unrecorded"

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not self.sample_id:
            raise ValueError("generation sample_id must not be empty")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("generation token IDs must be non-negative integers")
        if not isinstance(self.text, str) or not isinstance(self.extracted_answer, str):
            raise TypeError("generation text and extracted_answer must be strings")
        if type(self.is_extracted) is not bool or type(self.is_correct) is not bool:
            raise TypeError("generation extraction flags must be boolean")
        if not self.method or not self.primary_method:
            raise ValueError("generation extraction methods must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "extracted_answer": self.extracted_answer,
            "is_extracted": self.is_extracted,
            "is_correct": self.is_correct,
            "method": self.method,
            "primary_method": self.primary_method,
        }


class CorrectionRuntime(Protocol):
    def correct(self, text: str) -> CorrectionOutcome: ...

    def provenance(self) -> dict[str, object]: ...

    def close(self) -> None: ...


class GenerationRuntime(Protocol):
    def generate_duplicate_batch(
        self,
        prompts: Sequence[str],
        *,
        sample_ids: Sequence[str],
        gold_answers: Sequence[str],
    ) -> Sequence[SamePromptGeneration]: ...

    def provenance(self) -> dict[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class InputCorrectorAuditResult:
    """Published artifact paths for one completed setting."""

    records_path: Path
    summary_path: Path
    run_path: Path
    records: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(path, _serialized_json(payload))


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _arguments(config: InputCorrectorAuditConfig) -> dict[str, object]:
    return {
        "corrector": config.corrector,
        "model": config.model,
        "benchmark": config.benchmark,
        "pairs": str(config.pairs.resolve()),
        "gpu_id": config.gpu_id,
        "limit": config.limit,
        "output_dir": str(config.output_dir.resolve()),
    }


def _scope(config: InputCorrectorAuditConfig) -> str:
    return "custom-smoke" if config.limit is not None else "paper-setting"


def _source_record_sha256(record: Mapping[str, object]) -> str:
    encoded = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _correction_checkpoint_path(work_dir: Path, index: int) -> Path:
    return work_dir / "corrections" / f"{index:08d}.json"


def _generation_checkpoint_path(work_dir: Path, batch_index: int) -> Path:
    return work_dir / "same-batches" / f"{batch_index:08d}.json"


def _correction_checkpoint(
    record: Mapping[str, object],
    outcome: CorrectionOutcome,
) -> dict[str, object]:
    edited = _mapping(record.get("edited"), field="source edited")
    editable = edited.get("editable_text")
    if not isinstance(editable, str):
        raise ValueError("source edited editable_text must be a string")
    return {
        "schema_version": "input-corrector-correction-checkpoint/v1",
        "sample_id": record["sample_id"],
        "source_record_sha256": _source_record_sha256(record),
        "input_sha256": _sha256_text(editable),
        "outcome": outcome.to_dict(),
    }


def _outcome_from_checkpoint(
    path: Path,
    record: Mapping[str, object],
) -> CorrectionOutcome:
    payload = _load_json(path)
    if payload.get("schema_version") != "input-corrector-correction-checkpoint/v1":
        raise ValueError(f"correction checkpoint schema differs: {path}")
    if payload.get("sample_id") != record.get("sample_id"):
        raise ValueError(f"correction checkpoint sample differs: {path}")
    if payload.get("source_record_sha256") != _source_record_sha256(record):
        raise ValueError(f"correction checkpoint source hash differs: {path}")
    edited = _mapping(record.get("edited"), field="source edited")
    editable = edited.get("editable_text")
    if not isinstance(editable, str) or payload.get("input_sha256") != _sha256_text(editable):
        raise ValueError(f"correction checkpoint input hash differs: {path}")
    outcome = _mapping(payload.get("outcome"), field=f"{path}.outcome")
    return CorrectionOutcome(
        corrected_text=outcome.get("corrected_text"),  # type: ignore[arg-type]
        parse_failed=outcome.get("parse_failed"),  # type: ignore[arg-type]
        n_calls=outcome.get("n_calls"),  # type: ignore[arg-type]
        raw_response=outcome.get("raw_response"),  # type: ignore[arg-type]
    )


def _generation_from_dict(value: object, *, field: str) -> SamePromptGeneration:
    payload = _mapping(value, field=field)
    token_ids = payload.get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError(f"{field}.token_ids must be a list")
    return SamePromptGeneration(
        sample_id=payload.get("sample_id"),  # type: ignore[arg-type]
        token_ids=tuple(token_ids),  # type: ignore[arg-type]
        text=payload.get("text"),  # type: ignore[arg-type]
        extracted_answer=payload.get("extracted_answer"),  # type: ignore[arg-type]
        is_extracted=payload.get("is_extracted"),  # type: ignore[arg-type]
        is_correct=payload.get("is_correct"),  # type: ignore[arg-type]
        method=payload.get("method", "unrecorded"),  # type: ignore[arg-type]
        primary_method=payload.get("primary_method", "unrecorded"),  # type: ignore[arg-type]
    )


def _load_generation_checkpoint(
    path: Path,
    *,
    records: Sequence[Mapping[str, object]],
) -> tuple[SamePromptGeneration, ...]:
    payload = _load_json(path)
    if payload.get("schema_version") != "input-corrector-same-batch-checkpoint/v1":
        raise ValueError(f"same-batch checkpoint schema differs: {path}")
    expected_ids = [str(record["sample_id"]) for record in records for _ in range(2)]
    if payload.get("sample_ids") != expected_ids:
        raise ValueError(f"same-batch checkpoint sample order differs: {path}")
    expected_hashes = [_source_record_sha256(record) for record in records]
    if payload.get("source_record_sha256s") != expected_hashes:
        raise ValueError(f"same-batch checkpoint source hash differs: {path}")
    raw_generations = payload.get("generations")
    if not isinstance(raw_generations, list) or len(raw_generations) != len(expected_ids):
        raise ValueError(f"same-batch checkpoint row count differs: {path}")
    generations = tuple(
        _generation_from_dict(value, field=f"{path}.generations[{index}]")
        for index, value in enumerate(raw_generations)
    )
    if [generation.sample_id for generation in generations] != expected_ids:
        raise ValueError(f"same-batch checkpoint generation IDs differ: {path}")
    return generations


def _validate_resume_core(
    previous: Mapping[str, object],
    *,
    arguments: Mapping[str, object],
    source: InputCorrectorSource,
    implementation_code: Mapping[str, object],
) -> str:
    if previous.get("schema_version") != "input-corrector-audit-run/v1":
        raise ValueError("resume manifest has an unsupported schema")
    if previous.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("resume paper SHA-256 differs")
    if previous.get("operation") != "input-corrector-audit":
        raise ValueError("resume operation differs")
    if previous.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("resume protocol SHA-256 differs")
    if previous.get("arguments") != dict(arguments):
        raise ValueError("resume arguments differ from the original run")
    if previous.get("source") != source.to_dict():
        raise ValueError("resume source hash or identity changed")
    if previous.get("implementation_code") != dict(implementation_code):
        raise ValueError("resume executable implementation code identity differs")
    started_at = previous.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume started_at is missing")
    return started_at


def _result_from_completed(
    output_dir: Path,
    manifest: Mapping[str, object],
) -> InputCorrectorAuditResult:
    outputs = _mapping(manifest.get("outputs"), field="completed outputs")
    expected_names = {_RECORDS_NAME, _SUMMARY_NAME}
    if set(outputs) != expected_names:
        raise InputCorrectorAuditRunError("completed output inventory differs")
    for name in expected_names:
        metadata = _mapping(outputs[name], field=f"completed outputs.{name}")
        path = output_dir / name
        if not path.is_file():
            raise InputCorrectorAuditRunError(f"completed output is missing: {path}")
        if metadata.get("sha256") != _sha256_file(path):
            raise InputCorrectorAuditRunError(f"completed output SHA-256 differs: {path}")
        if metadata.get("bytes") != path.stat().st_size:
            raise InputCorrectorAuditRunError(f"completed output byte size differs: {path}")
    records_metadata = _mapping(outputs[_RECORDS_NAME], field="completed records")
    records = records_metadata.get("records")
    if not isinstance(records, int) or isinstance(records, bool) or records <= 0:
        raise InputCorrectorAuditRunError("completed record count is invalid")
    return InputCorrectorAuditResult(
        records_path=output_dir / _RECORDS_NAME,
        summary_path=output_dir / _SUMMARY_NAME,
        run_path=output_dir / "run.json",
        records=records,
    )


def _splice_corrected_prompt(
    record: Mapping[str, object],
    corrected_text: str,
) -> tuple[str, str, str, str]:
    clean = _mapping(record.get("clean"), field="source clean")
    edited = _mapping(record.get("edited"), field="source edited")
    clean_prompt = clean.get("prompt")
    clean_editable = clean.get("editable_text")
    edited_prompt = edited.get("prompt")
    edited_editable = edited.get("editable_text")
    span = _mapping(edited.get("editable_prompt_span"), field="edited editable prompt span")
    if not all(
        isinstance(value, str)
        for value in (clean_prompt, clean_editable, edited_prompt, edited_editable)
    ):
        raise ValueError("source prompt/editable values must be strings")
    start = span.get("start")
    end = span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("edited editable prompt span must contain integers")
    corrected_prompt = edited_prompt[:start] + corrected_text + edited_prompt[end:]
    return clean_prompt, clean_editable, edited_editable, corrected_prompt


def _record_payload(
    *,
    config: InputCorrectorAuditConfig,
    source_record: Mapping[str, object],
    outcome: CorrectionOutcome,
    same_pair: tuple[SamePromptGeneration, SamePromptGeneration] | None,
) -> dict[str, object]:
    clean_prompt, clean_editable, edited_editable, corrected_prompt = _splice_corrected_prompt(
        source_record,
        outcome.corrected_text,
    )
    restored = classify_restoration(clean_editable, edited_editable, outcome.corrected_text)
    exact = is_byte_exact_equal(clean_prompt, corrected_prompt)
    if exact != (same_pair is not None) and config.benchmark != MATH_BENCHMARK:
        raise ValueError("Same-prompt generation coverage disagrees with exact-byte selection")

    same_answers: dict[str, object] | None = None
    separate_answers: dict[str, object] | None = None
    if same_pair is not None:
        first, duplicate = same_pair
        same_answers = {
            "protocol": "adjacent-duplicate-prompt-pair/v1",
            "first_extracted_answer": first.extracted_answer,
            "duplicate_extracted_answer": duplicate.extracted_answer,
            "first": first.to_dict(),
            "duplicate": duplicate.to_dict(),
        }
        clean = _mapping(source_record.get("clean"), field="source clean")
        clean_answer = _mapping(clean.get("answer"), field="source clean answer")
        source_answer = clean_answer.get("value")
        if not isinstance(source_answer, str):
            raise ValueError("source clean answer value must be a string")
        separate_answers = {
            "comparison": "same_batch_corrected_vs_source_pair_clean",
            "same_batch_corrected_extracted_answer": duplicate.extracted_answer,
            "source_pair_clean_extracted_answer": source_answer,
        }

    return {
        "schema_version": "input-corrector-audit-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "input-corrector-audit",
        "model": config.model,
        "benchmark": config.benchmark,
        "corrector": config.corrector,
        "sample_id": source_record["sample_id"],
        "source_record_sha256": _source_record_sha256(source_record),
        "correction": {
            **outcome.to_dict(),
            "input_sha256": _sha256_text(edited_editable),
            "corrected_sha256": _sha256_text(outcome.corrected_text),
        },
        "edited_words": {
            "restored": restored.n_restored,
            "total": restored.n_perturbed_words,
            "unalignable": restored.n_unalignable,
        },
        "prompt_endpoints": {
            "clean_sha256": _sha256_text(clean_prompt),
            "corrected_sha256": _sha256_text(corrected_prompt),
            "exact_utf8": exact,
        },
        "same_batch_answers": same_answers,
        "separate_source_answers": separate_answers,
        "diagnostics": {
            "whitespace_normalized_full": restored.whitespace_normalized_full,
            "all_perturbed_restored": restored.all_perturbed_restored,
            "intact_word_changes": restored.n_collateral,
            "collateral_changes": [
                {"word_index": index, "clean": clean, "corrected": corrected}
                for index, clean, corrected in restored.collateral
            ],
        },
    }


def _metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    restored = sum(
        int(_mapping(row["edited_words"], field="edited_words")["restored"]) for row in records
    )
    total = sum(
        int(_mapping(row["edited_words"], field="edited_words")["total"]) for row in records
    )
    exact_records = [
        row
        for row in records
        if bool(_mapping(row["prompt_endpoints"], field="prompt_endpoints")["exact_utf8"])
    ]
    same_changed = 0
    separate_changed = 0
    for row in exact_records:
        same = row.get("same_batch_answers")
        separate = row.get("separate_source_answers")
        if isinstance(same, Mapping):
            same_changed += same.get("first_extracted_answer") != same.get(
                "duplicate_extracted_answer"
            )
        if isinstance(separate, Mapping):
            separate_changed += separate.get(
                "same_batch_corrected_extracted_answer"
            ) != separate.get("source_pair_clean_extracted_answer")
    collateral = sum(
        int(_mapping(row["diagnostics"], field="diagnostics")["intact_word_changes"])
        for row in records
    )
    return {
        "records": len(records),
        "word_restored": restored,
        "word_total": total,
        "word_restoration_rate": restored / total if total else None,
        "exact_clean": len(exact_records),
        "same_changed": same_changed,
        "separate_source_changed": separate_changed,
        "intact_word_changes": collateral,
    }


def _close(runtime: object | None) -> None:
    if runtime is not None:
        runtime.close()  # type: ignore[attr-defined]


def _default_correction_runtime(config: InputCorrectorAuditConfig) -> CorrectionRuntime:
    from typo_cot.experiments.input_corrector_audit.runtime import (
        ProductionCorrectionRuntime,
    )

    return ProductionCorrectionRuntime(config)


def _default_generation_runtime(
    config: InputCorrectorAuditConfig,
    *,
    model_revision: str,
) -> GenerationRuntime:
    from typo_cot.experiments.input_corrector_audit.runtime import (
        HuggingFaceSamePromptRuntime,
    )

    return HuggingFaceSamePromptRuntime(config, revision=model_revision)


def run_input_corrector_audit(
    config: InputCorrectorAuditConfig,
    *,
    correction_runtime: CorrectionRuntime | None = None,
    generation_runtime: GenerationRuntime | None = None,
) -> InputCorrectorAuditResult:
    """Run one validated corrector setting and publish atomically verifiable outputs."""
    source = load_input_corrector_source(
        config.pairs,
        model=config.model,
        benchmark=config.benchmark,
    )
    selected = source.records[: config.limit]
    if not selected:
        raise ValueError("the selected source cohort is empty")

    output_dir = config.output_dir.resolve()
    run_path = output_dir / "run.json"
    records_path = output_dir / _RECORDS_NAME
    summary_path = output_dir / _SUMMARY_NAME
    work_dir = output_dir / _WORK_NAME
    arguments = _arguments(config)
    implementation_code = implementation_code_identity()
    output_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if output_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    previous: dict[str, object] | None = None
    started_at = _now()
    previous_runtime: Mapping[str, object] = {}
    if config.resume:
        previous = _load_json(run_path)
        started_at = _validate_resume_core(
            previous,
            arguments=arguments,
            source=source,
            implementation_code=implementation_code,
        )
        status = previous.get("status")
        if status == "completed":
            source.assert_unchanged()
            return _result_from_completed(output_dir, previous)
        if status not in {"running", "failed"}:
            raise ValueError(f"resume manifest has unsupported status: {status!r}")
        previous_runtime = _mapping(previous.get("runtime", {}), field="resume runtime")

    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_provenance: dict[str, object] = dict(previous_runtime)
    outcomes: dict[str, CorrectionOutcome] = {}
    same_pairs: dict[str, tuple[SamePromptGeneration, SamePromptGeneration]] = {}
    correction_closed = False
    generation_closed = False

    def progress(status: str, *, failure: str | None = None) -> None:
        correction_files = len(list((work_dir / "corrections").glob("*.json")))
        generation_files = len(list((work_dir / "same-batches").glob("*.json")))
        payload: dict[str, object] = {
            "schema_version": "input-corrector-audit-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "input-corrector-audit",
            "status": status,
            "scope": _scope(config),
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_code": implementation_code,
            "arguments": arguments,
            "source": source.to_dict(),
            "selected_records": len(selected),
            "runtime": runtime_provenance,
            "checkpoints": {
                "corrections": correction_files,
                "complete_same_batches": generation_files,
            },
            "outputs": {},
            "started_at": started_at,
            "updated_at": _now(),
            "failure": failure,
        }
        _write_json_atomic(run_path, payload)

    if previous is None:
        progress("running")

    try:
        missing_corrections: list[tuple[int, Mapping[str, object]]] = []
        for index, record in enumerate(selected):
            checkpoint = _correction_checkpoint_path(work_dir, index)
            if checkpoint.is_file():
                outcomes[str(record["sample_id"])] = _outcome_from_checkpoint(
                    checkpoint,
                    record,
                )
            else:
                missing_corrections.append((index, record))

        previous_correction = runtime_provenance.get("correction")
        if correction_runtime is None and (missing_corrections or previous_correction is None):
            correction_runtime = _default_correction_runtime(config)

        if missing_corrections and correction_runtime is None:
            raise RuntimeError("a correction runtime is required for missing checkpoints")
        correction_provenance_captured = False
        for index, record in missing_corrections:
            edited = _mapping(record.get("edited"), field="source edited")
            editable = edited.get("editable_text")
            if not isinstance(editable, str):
                raise ValueError("source edited editable_text must be a string")
            assert correction_runtime is not None
            outcome = correction_runtime.correct(editable)
            if not isinstance(outcome, CorrectionOutcome):
                raise TypeError("correction runtime returned an unsupported result")
            if not correction_provenance_captured:
                current_provenance = correction_runtime.provenance()
                if previous_correction is not None and previous_correction != current_provenance:
                    raise ValueError("resume correction runtime provenance differs")
                runtime_provenance["correction"] = current_provenance
                correction_provenance_captured = True
                progress("running")
            checkpoint = _correction_checkpoint_path(work_dir, index)
            _write_json_atomic(checkpoint, _correction_checkpoint(record, outcome))
            outcomes[str(record["sample_id"])] = outcome
            progress("running")

        if correction_runtime is not None and not correction_provenance_captured:
            current_provenance = correction_runtime.provenance()
            if previous_correction is not None and previous_correction != current_provenance:
                raise ValueError("resume correction runtime provenance differs")
            runtime_provenance["correction"] = current_provenance
            progress("running")

        _close(correction_runtime)
        correction_closed = True
        source.assert_unchanged()

        exact_records: list[Mapping[str, object]] = []
        if config.benchmark != MATH_BENCHMARK:
            for record in selected:
                outcome = outcomes[str(record["sample_id"])]
                clean_prompt, _clean_editable, _edited_editable, corrected_prompt = (
                    _splice_corrected_prompt(record, outcome.corrected_text)
                )
                if is_byte_exact_equal(clean_prompt, corrected_prompt):
                    exact_records.append(record)

        batches = [exact_records[index : index + 2] for index in range(0, len(exact_records), 2)]
        missing_batches = [
            (index, batch)
            for index, batch in enumerate(batches)
            if not _generation_checkpoint_path(work_dir, index).is_file()
        ]
        previous_generation = runtime_provenance.get("generation")
        if generation_runtime is not None:
            current_provenance = generation_runtime.provenance()
            if previous_generation is not None and previous_generation != current_provenance:
                raise ValueError("resume generation runtime provenance differs")
            runtime_provenance["generation"] = current_provenance
            progress("running")
        elif missing_batches:
            generation_runtime = _default_generation_runtime(
                config,
                model_revision=source.model_revision,
            )
            current_provenance = generation_runtime.provenance()
            if previous_generation is not None and previous_generation != current_provenance:
                raise ValueError("resume generation runtime provenance differs")
            runtime_provenance["generation"] = current_provenance
            progress("running")
        elif batches and previous_generation is None:
            raise ValueError(
                "complete Same generation checkpoints lack runtime provenance"
            )

        for batch_index, batch in enumerate(batches):
            checkpoint = _generation_checkpoint_path(work_dir, batch_index)
            if checkpoint.is_file():
                generations = _load_generation_checkpoint(checkpoint, records=batch)
            else:
                if generation_runtime is None:
                    raise RuntimeError("a generation runtime is required for an exact-prompt batch")
                prompts: list[str] = []
                sample_ids: list[str] = []
                gold_answers: list[str] = []
                for record in batch:
                    clean = _mapping(record.get("clean"), field="source clean")
                    prompt = clean.get("prompt")
                    answer = record.get("gold_answer")
                    if not isinstance(prompt, str) or not isinstance(answer, str):
                        raise ValueError("source clean prompt/gold answer must be strings")
                    prompts.extend((prompt, prompt))
                    sample_ids.extend((str(record["sample_id"]), str(record["sample_id"])))
                    gold_answers.extend((answer, answer))
                generated = generation_runtime.generate_duplicate_batch(
                    prompts,
                    sample_ids=sample_ids,
                    gold_answers=gold_answers,
                )
                generations = tuple(generated)
                if len(generations) != len(prompts) or any(
                    not isinstance(generation, SamePromptGeneration) for generation in generations
                ):
                    raise RuntimeError("generation runtime returned an incomplete batch")
                if [generation.sample_id for generation in generations] != sample_ids:
                    raise RuntimeError("generation runtime changed the duplicate row order")
                _write_json_atomic(
                    checkpoint,
                    {
                        "schema_version": "input-corrector-same-batch-checkpoint/v1",
                        "sample_ids": sample_ids,
                        "source_record_sha256s": [
                            _source_record_sha256(record) for record in batch
                        ],
                        "generations": [generation.to_dict() for generation in generations],
                    },
                )
                progress("running")
            for pair_index, record in enumerate(batch):
                offset = pair_index * 2
                same_pairs[str(record["sample_id"])] = (
                    generations[offset],
                    generations[offset + 1],
                )

        _close(generation_runtime)
        generation_closed = True
        source.assert_unchanged()

        rows = [
            _record_payload(
                config=config,
                source_record=record,
                outcome=outcomes[str(record["sample_id"])],
                same_pair=same_pairs.get(str(record["sample_id"])),
            )
            for record in selected
        ]
        metrics = _metrics(rows)
        summary = {
            "schema_version": "input-corrector-audit-summary/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "input-corrector-audit",
            "status": "completed",
            "scope": _scope(config),
            "model": config.model,
            "benchmark": config.benchmark,
            "corrector": config.corrector,
            "metrics": metrics,
        }
        staged_records = work_dir / _RECORDS_NAME
        staged_summary = work_dir / _SUMMARY_NAME
        _write_jsonl_atomic(staged_records, rows)
        _write_json_atomic(staged_summary, summary)
        source.assert_unchanged()
        os.replace(staged_records, records_path)
        os.replace(staged_summary, summary_path)
        outputs = {
            _RECORDS_NAME: {
                "sha256": _sha256_file(records_path),
                "bytes": records_path.stat().st_size,
                "records": len(rows),
            },
            _SUMMARY_NAME: {
                "sha256": _sha256_file(summary_path),
                "bytes": summary_path.stat().st_size,
            },
        }
        completed = {
            "schema_version": "input-corrector-audit-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "input-corrector-audit",
            "status": "completed",
            "scope": _scope(config),
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_code": implementation_code,
            "arguments": arguments,
            "source": source.to_dict(),
            "selected_records": len(selected),
            "runtime": runtime_provenance,
            "checkpoints": {},
            "outputs": outputs,
            "started_at": started_at,
            "updated_at": _now(),
            "failure": None,
        }
        _write_json_atomic(run_path, completed)
        result = InputCorrectorAuditResult(
            records_path=records_path,
            summary_path=summary_path,
            run_path=run_path,
            records=len(rows),
        )
        try:
            shutil.rmtree(work_dir)
        except OSError:
            # Publication is already complete and hash-bound. A removable
            # checkpoint directory must not roll valid public outputs back to
            # a failed run.
            pass
        return result
    except Exception as exc:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        try:
            progress("failed", failure=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise InputCorrectorAuditRunError(
            f"input-corrector generation/publication failed: {exc}"
        ) from exc
    finally:
        if not correction_closed:
            try:
                _close(correction_runtime)
            except Exception:
                pass
        if not generation_closed:
            try:
                _close(generation_runtime)
            except Exception:
                pass
