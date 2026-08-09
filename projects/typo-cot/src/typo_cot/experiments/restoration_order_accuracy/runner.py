"""Restartable condition-major producer for Appendix E Table 13."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from contextlib import contextmanager
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import fcntl

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.restoration_order_accuracy.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.restoration_order_accuracy.planning import (
    RestorationCondition,
    RestorationPlan,
    build_restoration_plan,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    ALL_CONDITION_IDS,
    PAPER_BATCH_SIZE,
    PAPER_BENCHMARKS,
    PAPER_BUDGETS,
    PAPER_MODELS,
    PAPER_ORDERS,
    PAPER_SEED,
    PROTOCOL_SHA256,
    build_edit_groups,
    canonical_sha256,
)
from typo_cot.experiments.restoration_order_accuracy.publication import (
    rename_directory_no_replace,
)
from typo_cot.experiments.restoration_order_accuracy.source import (
    load_restoration_order_source,
)

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RECORDS_NAME = "restoration_order_records.jsonl"
_SUMMARY_NAME = "restoration_order_summary.json"
_WORK_SUFFIX = ".restoration-order-work"
_PUBLISH_SUFFIX = ".restoration-order-publish"
_LOCK_SUFFIX = ".restoration-order.lock"
_OWNER_NAME = ".restoration-order-owner.json"
_RUN_ID = re.compile(r"[0-9a-f]{32}")


class RestorationOrderRunError(RuntimeError):
    """A setting failed after retaining its validated complete-batch checkpoints."""


@dataclass(frozen=True, slots=True)
class RestorationOrderConfig:
    """Frozen public arguments for one model/task restoration-order setting."""

    model: str
    benchmark: str
    pairs: Path
    orders: tuple[str, ...]
    budgets: tuple[int, ...]
    output_dir: Path
    seed: int = PAPER_SEED
    batch_size: int = PAPER_BATCH_SIZE
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", Path(self.pairs))
        object.__setattr__(self, "orders", tuple(self.orders))
        object.__setattr__(self, "budgets", tuple(self.budgets))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.model not in PAPER_MODELS:
            raise ValueError(f"model is outside the Table 13 paper grid: {self.model!r}")
        if self.benchmark not in PAPER_BENCHMARKS:
            raise ValueError(
                f"benchmark is outside the Table 13 paper grid: {self.benchmark!r}"
            )
        if self.orders != PAPER_ORDERS:
            raise ValueError(f"orders must equal the paper orders: {PAPER_ORDERS!r}")
        if self.budgets != PAPER_BUDGETS:
            raise ValueError(f"budgets must equal the paper budgets: {PAPER_BUDGETS!r}")
        if type(self.seed) is not int or self.seed != PAPER_SEED:
            raise ValueError(f"seed must be the submitted paper seed {PAPER_SEED}")
        if type(self.batch_size) is not int or self.batch_size != PAPER_BATCH_SIZE:
            raise ValueError(f"batch_size must be the submitted batch size {PAPER_BATCH_SIZE}")
        if not isinstance(self.gpu_id, str) or _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must identify one non-negative physical GPU")
        if self.limit is not None and (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        if type(self.resume) is not bool:
            raise TypeError("resume must be boolean")


@dataclass(frozen=True, slots=True)
class RestorationGeneration:
    """One freshly generated and task-scored answer."""

    sample_id: str
    token_ids: tuple[int, ...]
    text: str
    termination: str
    extracted_answer: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("generation sample_id must be a non-empty string")
        if not self.token_ids or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("generation token IDs must be non-empty non-negative integers")
        if not isinstance(self.text, str) or not isinstance(self.extracted_answer, str):
            raise TypeError("generation text and extracted answer must be strings")
        if self.termination not in {"eos", "length-cap"}:
            raise ValueError("generation termination must be 'eos' or 'length-cap'")
        if type(self.is_extracted) is not bool or type(self.is_correct) is not bool:
            raise TypeError("generation extraction flags must be boolean")
        if self.is_extracted != bool(self.extracted_answer):
            raise ValueError("generation extraction flag differs from the extracted answer")
        if self.is_correct and not self.is_extracted:
            raise ValueError("a correct generation must have an extracted answer")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("generation method must be non-empty")
        if not isinstance(self.primary_method, str) or not self.primary_method:
            raise ValueError("generation primary_method must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "termination": self.termination,
            "extracted_answer": self.extracted_answer,
            "is_extracted": self.is_extracted,
            "is_correct": self.is_correct,
            "method": self.method,
            "primary_method": self.primary_method,
        }


class GenerationRuntime(Protocol):
    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        sample_ids: Sequence[str],
        gold_answers: Sequence[str],
    ) -> Sequence[RestorationGeneration]: ...

    def provenance(self) -> dict[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RestorationOrderResult:
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
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owned_sibling(output_dir: Path, suffix: str) -> Path:
    return output_dir.parent / f".{output_dir.name}{suffix}"


@contextmanager
def _exclusive_output_lock(output_dir: Path) -> Iterator[None]:
    """Serialize fresh and resumed invocations for one final output identity."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _owned_sibling(output_dir, _LOCK_SUFFIX)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RestorationOrderRunError(
                f"another restoration-order invocation owns {output_dir}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _commit_publish_directory(stage: Path, output_dir: Path) -> None:
    """Commit one complete staged setting while its output lock is held."""
    rename_directory_no_replace(stage, output_dir)
    try:
        _fsync_directory(output_dir.parent)
    except OSError:
        # The rename is already the commit point. Do not report a completed
        # directory as failed merely because the durability fsync was refused.
        pass


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


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(path, _serialized_json(payload))


def _arguments(config: RestorationOrderConfig) -> dict[str, object]:
    return {
        "model": config.model,
        "benchmark": config.benchmark,
        "pairs": str(config.pairs.resolve()),
        "orders": list(config.orders),
        "budgets": list(config.budgets),
        "seed": config.seed,
        "batch_size": config.batch_size,
        "gpu_id": config.gpu_id,
        "limit": config.limit,
        "output_dir": str(config.output_dir.resolve()),
    }


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _create_owned_private_directory(
    path: Path,
    *,
    output_dir: Path,
    run_id: str,
) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"private state already exists: {path}")
    path.mkdir()
    try:
        _write_json_atomic(
            path / _OWNER_NAME,
            {
                "schema_version": "restoration-order-private-owner/v1",
                "operation": "restoration-order-accuracy",
                "output_dir": str(output_dir),
                "run_id": run_id,
            },
        )
    except Exception:
        shutil.rmtree(path)
        raise


def _private_owner_id(path: Path, *, output_dir: Path) -> str:
    marker_path = path / _OWNER_NAME
    if marker_path.is_file():
        payload = _load_json(marker_path)
        valid = (
            payload.get("schema_version") == "restoration-order-private-owner/v1"
            and payload.get("operation") == "restoration-order-accuracy"
            and payload.get("output_dir") == str(output_dir)
        )
    else:
        run_path = path / "run.json"
        if not run_path.is_file():
            raise ValueError(f"private directory lacks an ownership record: {path}")
        payload = _load_json(run_path)
        arguments = _mapping(payload.get("arguments"), field="private owner arguments")
        valid = (
            payload.get("schema_version") == "restoration-order-run/v1"
            and payload.get("operation") == "restoration-order-accuracy"
            and arguments.get("output_dir") == str(output_dir)
        )
    run_id = payload.get("run_id")
    if not valid or not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError(f"private directory ownership differs: {path}")
    return run_id


def _remove_owned_private_directory(
    path: Path,
    *,
    output_dir: Path,
    run_id: str,
) -> None:
    if _private_owner_id(path, output_dir=output_dir) != run_id:
        raise ValueError(f"private directory belongs to another run: {path}")
    shutil.rmtree(path)


def _record_plan(record: Mapping[str, object], *, seed: int) -> RestorationPlan:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("source record sample_id must be non-empty")
    clean = _mapping(record.get("clean"), field=f"{sample_id}.clean")
    edited = _mapping(record.get("edited"), field=f"{sample_id}.edited")
    clean_text = clean.get("editable_text")
    edited_text = edited.get("editable_text")
    attempts = record.get("target_attempts")
    if not isinstance(clean_text, str) or not isinstance(edited_text, str):
        raise ValueError(f"{sample_id} source editable texts must be strings")
    if not isinstance(attempts, list):
        raise ValueError(f"{sample_id} target_attempts must be a list")
    return build_restoration_plan(
        sample_id=sample_id,
        clean_text=clean_text,
        edited_text=edited_text,
        groups=build_edit_groups(clean_text, edited_text, attempts),
        seed=seed,
    )


def _plans(source: object, records: tuple[dict[str, object], ...], *, seed: int) -> tuple[RestorationPlan, ...]:
    supplied = getattr(source, "plans", None)
    if supplied is None:
        return tuple(_record_plan(record, seed=seed) for record in records)
    plans = tuple(supplied)
    if len(plans) != len(records):
        raise ValueError("source plan count differs from selected record count")
    return plans


def _span(value: object, *, field: str, text: str) -> tuple[int, int]:
    payload = _mapping(value, field=field)
    start = payload.get("start")
    end = payload.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError(f"{field} is outside its prompt")
    return start, end


def _condition_prompts(
    record: Mapping[str, object], plan: RestorationPlan
) -> dict[str, str]:
    sample_id = plan.sample_id
    clean = _mapping(record.get("clean"), field=f"{sample_id}.clean")
    edited = _mapping(record.get("edited"), field=f"{sample_id}.edited")
    clean_prompt = clean.get("prompt")
    edited_prompt = edited.get("prompt")
    if not isinstance(clean_prompt, str) or not isinstance(edited_prompt, str):
        raise ValueError(f"{sample_id} source prompts must be strings")
    clean_start, clean_end = _span(
        clean.get("editable_prompt_span"),
        field=f"{sample_id}.clean.editable_prompt_span",
        text=clean_prompt,
    )
    edited_start, edited_end = _span(
        edited.get("editable_prompt_span"),
        field=f"{sample_id}.edited.editable_prompt_span",
        text=edited_prompt,
    )
    if clean_prompt[clean_start:clean_end] != plan.clean_text:
        raise ValueError(f"{sample_id} clean prompt span differs from the restoration plan")
    if edited_prompt[edited_start:edited_end] != plan.edited_text:
        raise ValueError(f"{sample_id} edited prompt span differs from the restoration plan")
    if (
        clean_prompt[:clean_start] != edited_prompt[:edited_start]
        or clean_prompt[clean_end:] != edited_prompt[edited_end:]
    ):
        raise ValueError(f"{sample_id} prompts differ outside the editable span")
    prefix = clean_prompt[:clean_start]
    suffix = clean_prompt[clean_end:]
    prompts = {
        condition.condition_id: prefix + condition.text + suffix
        for condition in plan.conditions
    }
    if prompts["edited:k0"].encode() != edited_prompt.encode():
        raise ValueError(f"{sample_id} edited endpoint prompt is not byte exact")
    if prompts["clean:k4"].encode() != clean_prompt.encode():
        raise ValueError(f"{sample_id} clean endpoint prompt is not byte exact")
    return prompts


def _gold_answer(record: Mapping[str, object]) -> str:
    value = record.get("gold_answer")
    if not isinstance(value, str):
        raise ValueError("source gold_answer must be a string")
    return value


def _condition(plan: RestorationPlan, condition_id: str) -> RestorationCondition:
    for condition in plan.conditions:
        if condition.condition_id == condition_id:
            return condition
    raise ValueError(f"plan lacks required condition {condition_id!r}")


def _checkpoint_path(work_dir: Path, condition_id: str, batch_index: int) -> Path:
    safe_condition = condition_id.replace(":", "_")
    return work_dir / "checkpoints" / safe_condition / f"{batch_index:06d}.json"


def _checkpoint_identity(
    *,
    condition_id: str,
    sample_ids: Sequence[str],
    prompts: Sequence[str],
    gold_answers: Sequence[str],
    source: Mapping[str, object],
    code: Mapping[str, object],
    runtime: Mapping[str, object],
) -> str:
    return canonical_sha256(
        {
            "condition_id": condition_id,
            "sample_ids": list(sample_ids),
            "prompt_sha256": [_sha256_text(prompt) for prompt in prompts],
            "gold_answers": list(gold_answers),
            "source_cohort_sha256": source.get("cohort_sha256"),
            "source_pairs_sha256": source.get("pairs_sha256"),
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_sha256": code.get("sha256"),
            "runtime_sha256": canonical_sha256(runtime),
        }
    )


def _generation_from_dict(value: object, *, field: str) -> RestorationGeneration:
    payload = _mapping(value, field=field)
    sample_id = payload.get("sample_id")
    text = payload.get("text")
    termination = payload.get("termination")
    extracted_answer = payload.get("extracted_answer")
    is_extracted = payload.get("is_extracted")
    is_correct = payload.get("is_correct")
    method = payload.get("method")
    primary_method = payload.get("primary_method")
    if not isinstance(sample_id, str):
        raise ValueError(f"{field}.sample_id must be a string")
    if not isinstance(text, str) or not isinstance(extracted_answer, str):
        raise ValueError(f"{field} text fields must be strings")
    if not isinstance(termination, str):
        raise ValueError(f"{field}.termination must be a string")
    if type(is_extracted) is not bool or type(is_correct) is not bool:
        raise ValueError(f"{field} extraction flags must be boolean")
    if not isinstance(method, str) or not isinstance(primary_method, str):
        raise ValueError(f"{field} extraction methods must be strings")
    token_ids = payload.get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError(f"{field}.token_ids must be a list")
    return RestorationGeneration(
        sample_id=sample_id,
        token_ids=tuple(token_ids),
        text=text,
        termination=termination,
        extracted_answer=extracted_answer,
        is_extracted=is_extracted,
        is_correct=is_correct,
        method=method,
        primary_method=primary_method,
    )


def _load_checkpoint(
    path: Path,
    *,
    identity: str,
    sample_ids: Sequence[str],
) -> tuple[tuple[RestorationGeneration, ...], bool]:
    payload = _load_json(path)
    if payload.get("schema_version") != "restoration-order-batch-checkpoint/v1":
        raise ValueError(f"checkpoint schema differs: {path}")
    if payload.get("input_sha256") != identity:
        raise ValueError(f"checkpoint input identity differs: {path}")
    rows = payload.get("generations")
    if not isinstance(rows, list):
        raise ValueError(f"checkpoint generations must be a list: {path}")
    generations = tuple(
        _generation_from_dict(row, field=f"{path}.generations[{index}]")
        for index, row in enumerate(rows)
    )
    if payload.get("generations_sha256") != canonical_sha256(
        [generation.to_dict() for generation in generations]
    ):
        raise ValueError(f"checkpoint generation integrity differs: {path}")
    if tuple(row.sample_id for row in generations) != tuple(sample_ids):
        raise ValueError(f"checkpoint sample order differs: {path}")
    individual_retry = payload.get("individual_retry")
    if type(individual_retry) is not bool:
        raise ValueError(f"checkpoint individual_retry flag is invalid: {path}")
    return generations, individual_retry


def _validate_generation_rows(
    rows: Sequence[RestorationGeneration], sample_ids: Sequence[str]
) -> tuple[RestorationGeneration, ...]:
    result = tuple(rows)
    if tuple(row.sample_id for row in result) != tuple(sample_ids):
        raise ValueError("runtime generation sample order differs from the requested batch")
    return result


def _generate_batch_with_fallback(
    runtime: GenerationRuntime,
    *,
    prompts: Sequence[str],
    sample_ids: Sequence[str],
    gold_answers: Sequence[str],
) -> tuple[tuple[RestorationGeneration, ...], bool]:
    batch_error_message: str | None = None
    try:
        batch_rows = runtime.generate_batch(
            prompts,
            sample_ids=sample_ids,
            gold_answers=gold_answers,
        )
    except RuntimeError as batch_error:
        batch_error_message = str(batch_error)
    if batch_error_message is not None:
        # Leaving the except block clears its traceback frames before CUDA
        # recovery, so tensors referenced only by the failed call can be freed.
        recovery = getattr(runtime, "recover_after_batch_failure", None)
        if recovery is not None:
            recovery()
        rows: list[RestorationGeneration] = []
        try:
            for prompt, sample_id, gold_answer in zip(
                prompts, sample_ids, gold_answers, strict=True
            ):
                generated = runtime.generate_batch(
                    (prompt,),
                    sample_ids=(sample_id,),
                    gold_answers=(gold_answer,),
                )
                rows.extend(_validate_generation_rows(generated, (sample_id,)))
        except RuntimeError as item_error:
            raise RuntimeError(
                "batch generation failed "
                f"({batch_error_message}); individual retry failed ({item_error})"
            ) from item_error
        return tuple(rows), True
    return _validate_generation_rows(batch_rows, sample_ids), False


def _core_manifest(
    *,
    config: RestorationOrderConfig,
    status: str,
    started_at: str,
    source: Mapping[str, object],
    code: Mapping[str, object],
    runtime: Mapping[str, object],
    completed_batches: int,
    total_batches: int,
    run_id: str,
    failure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "restoration-order-run/v1",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "operation": "restoration-order-accuracy",
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "updated_at": _now(),
        "arguments": _arguments(config),
        "source": dict(source),
        "implementation_code": dict(code),
        "runtime": dict(runtime),
        "progress": {
            "completed_batches": completed_batches,
            "total_batches": total_batches,
        },
        "failure": None if failure is None else dict(failure),
    }


def _validate_resume(
    previous: Mapping[str, object],
    *,
    config: RestorationOrderConfig,
    source: Mapping[str, object],
    code: Mapping[str, object],
) -> tuple[str, str]:
    if previous.get("schema_version") != "restoration-order-run/v1":
        raise ValueError("resume manifest has an unsupported schema")
    if previous.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("resume paper SHA-256 differs")
    if previous.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("resume protocol SHA-256 differs")
    if previous.get("operation") != "restoration-order-accuracy":
        raise ValueError("resume operation differs")
    if previous.get("arguments") != _arguments(config):
        raise ValueError("resume arguments differ from the original run")
    if previous.get("source") != source:
        raise ValueError("resume source or cohort identity changed")
    if previous.get("implementation_code") != code:
        raise ValueError("resume executable implementation identity changed")
    run_id = previous.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("resume run_id is missing or invalid")
    started_at = previous.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume started_at is missing")
    return started_at, run_id


def _validate_public_outputs(
    manifest: Mapping[str, object],
    *,
    records_path: Path,
    summary_path: Path,
) -> int:
    outputs = _mapping(manifest.get("outputs"), field="completed outputs")
    records_meta = _mapping(outputs.get("records"), field="outputs.records")
    summary_meta = _mapping(outputs.get("summary"), field="outputs.summary")
    if records_meta.get("path") != records_path.name or not records_path.is_file():
        raise ValueError("completed records output is missing")
    if summary_meta.get("path") != summary_path.name or not summary_path.is_file():
        raise ValueError("completed summary output is missing")
    if records_meta.get("sha256") != _sha256_file(records_path):
        raise ValueError("completed records output hash differs")
    if summary_meta.get("sha256") != _sha256_file(summary_path):
        raise ValueError("completed summary output hash differs")
    record_count = records_meta.get("records")
    if not isinstance(record_count, int) or isinstance(record_count, bool):
        raise ValueError("completed records count is invalid")
    lines = records_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != record_count or any(not line for line in lines):
        raise ValueError("completed records count differs from JSONL")
    for index, line in enumerate(lines):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except ValueError as exc:
            raise ValueError(f"invalid completed record {index}: {exc}") from exc
        if not isinstance(row, Mapping) or tuple(_mapping(row.get("arms"), field="arms")) != ALL_CONDITION_IDS:
            raise ValueError("completed record arm grid differs")
    summary = _load_json(summary_path)
    if tuple(_mapping(summary.get("conditions"), field="summary conditions")) != ALL_CONDITION_IDS:
        raise ValueError("completed summary condition grid differs")
    return record_count


def _build_records(
    *,
    config: RestorationOrderConfig,
    records: Sequence[Mapping[str, object]],
    plans: Sequence[RestorationPlan],
    prompts_by_sample: Mapping[str, Mapping[str, str]],
    generations: Mapping[str, Mapping[str, RestorationGeneration]],
    individual_retry: Mapping[str, Mapping[str, bool]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record, plan in zip(records, plans, strict=True):
        sample_id = plan.sample_id
        clean = _mapping(record.get("clean"), field=f"{sample_id}.clean")
        clean_prompt = clean.get("prompt")
        if not isinstance(clean_prompt, str):
            raise ValueError(f"{sample_id} clean prompt must be a string")
        clean_start, clean_end = _span(
            clean.get("editable_prompt_span"),
            field=f"{sample_id}.clean.editable_prompt_span",
            text=clean_prompt,
        )
        prompt_prefix = clean_prompt[:clean_start]
        prompt_suffix = clean_prompt[clean_end:]
        arms: dict[str, object] = {}
        for condition_id in ALL_CONDITION_IDS:
            condition = _condition(plan, condition_id)
            prompt = prompts_by_sample[sample_id][condition_id]
            generation = generations[condition_id][sample_id]
            arms[condition_id] = {
                "order": condition.order,
                "budget": condition.budget,
                "restored_group_indices": list(condition.restored_group_indices),
                "editable_text": condition.text,
                "editable_text_sha256": _sha256_text(condition.text),
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "individual_retry": individual_retry[condition_id][sample_id],
                **generation.to_dict(),
            }
        plan_payload = plan.to_dict()
        output.append(
            {
                "schema_version": "restoration-order-record/v1",
                "sample_id": sample_id,
                "model": config.model,
                "benchmark": config.benchmark,
                "gold_answer": _gold_answer(record),
                "source_record_sha256": canonical_sha256(record),
                "source_selection": {
                    "clean_correct": True,
                    "edited_wrong": True,
                    "separable": True,
                },
                "prompt_context": {
                    "prefix_length": len(prompt_prefix),
                    "prefix_sha256": _sha256_text(prompt_prefix),
                    "suffix_length": len(prompt_suffix),
                    "suffix_sha256": _sha256_text(prompt_suffix),
                },
                "realized_edit_groups": plan.realized_edit_count,
                "edit_groups": [group.to_dict() for group in plan.groups],
                "order_group_indices": {
                    name: list(indices) for name, indices in plan.order_indices
                },
                "plan_sha256": plan_payload["sha256"],
                "arms": arms,
            }
        )
    return output


def _summary(
    *,
    config: RestorationOrderConfig,
    source: object,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    conditions: dict[str, object] = {}
    for condition_id in ALL_CONDITION_IDS:
        correct = sum(
            int(
                bool(
                    _mapping(
                        _mapping(row.get("arms"), field="record arms").get(condition_id),
                        field=f"record arm {condition_id}",
                    ).get("is_correct")
                )
            )
            for row in rows
        )
        extracted = sum(
            int(
                bool(
                    _mapping(
                        _mapping(row.get("arms"), field="record arms").get(condition_id),
                        field=f"record arm {condition_id}",
                    ).get("is_extracted")
                )
            )
            for row in rows
        )
        total = len(rows)
        conditions[condition_id] = {
            "correct": correct,
            "extracted": extracted,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    source_records = int(getattr(source, "source_record_count", len(getattr(source, "records"))))
    source_selected = int(
        getattr(source, "source_selected_count", len(getattr(source, "records")))
    )
    separable = int(getattr(source, "separable_count", len(getattr(source, "records"))))
    return {
        "schema_version": "restoration-order-summary/v1",
        "model": config.model,
        "benchmark": config.benchmark,
        "cohort": {
            "source_records": source_records,
            "source_flip": source_selected,
            "separable": separable,
            "selected": len(rows),
            "limited": config.limit is not None,
        },
        "conditions": conditions,
    }


def _publish_completed_outputs(
    *,
    config: RestorationOrderConfig,
    source: object,
    source_payload: Mapping[str, object],
    code: Mapping[str, object],
    runtime_payload: Mapping[str, object],
    started_at: str,
    run_id: str,
    total_batches: int,
    records: Sequence[Mapping[str, object]],
    plans: Sequence[RestorationPlan],
    prompts_by_sample: Mapping[str, Mapping[str, str]],
    generated: Mapping[str, Mapping[str, RestorationGeneration]],
    individual_retry: Mapping[str, Mapping[str, bool]],
    retry_batches: Sequence[Mapping[str, object]],
    sample_ids: tuple[str, ...],
    output_dir: Path,
    publish_dir: Path,
    work_dir: Path,
) -> RestorationOrderResult:
    """Validate complete checkpoints, recheck the source, and publish outputs."""
    for condition_id in ALL_CONDITION_IDS:
        if tuple(generated[condition_id]) != sample_ids:
            raise ValueError(f"condition {condition_id} lacks the complete selected cohort")
        if tuple(individual_retry[condition_id]) != sample_ids:
            raise ValueError(
                f"condition {condition_id} lacks complete retry provenance"
            )
    if implementation_code_identity() != code:
        raise ValueError("restoration-order implementation changed during generation")
    source.assert_unchanged()  # type: ignore[attr-defined]
    public_rows = _build_records(
        config=config,
        records=records,
        plans=plans,
        prompts_by_sample=prompts_by_sample,
        generations=generated,
        individual_retry=individual_retry,
    )
    summary = _summary(config=config, source=source, rows=public_rows)
    records_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=False, allow_nan=False) + "\n"
        for row in public_rows
    )
    summary_text = _serialized_json(summary)
    retry_item_count = 0
    for batch in retry_batches:
        retry_sample_ids = batch.get("sample_ids")
        if not isinstance(retry_sample_ids, list):
            raise AssertionError("retry batch sample_ids must be a list")
        retry_item_count += len(retry_sample_ids)
    completed_manifest = _core_manifest(
        config=config,
        status="completed",
        started_at=started_at,
        source=source_payload,
        code=code,
        runtime=runtime_payload,
        completed_batches=total_batches,
        total_batches=total_batches,
        run_id=run_id,
    )
    completed_manifest.update(
        {
            "completed_at": _now(),
            "counts": {"selected": len(public_rows), "records": len(public_rows)},
            "generation_retries": {
                "individual_retry_batches": len(retry_batches),
                "individual_retry_items": retry_item_count,
                "batches": [dict(batch) for batch in retry_batches],
            },
            "outputs": {
                "records": {
                    "path": _RECORDS_NAME,
                    "sha256": "pending",
                    "records": len(public_rows),
                },
                "summary": {
                    "path": _SUMMARY_NAME,
                    "sha256": "pending",
                },
            },
        }
    )
    if publish_dir.exists():
        raise FileExistsError(f"owned publication stage already exists: {publish_dir}")
    try:
        _create_owned_private_directory(
            publish_dir,
            output_dir=output_dir,
            run_id=run_id,
        )
        staged_records = publish_dir / _RECORDS_NAME
        staged_summary = publish_dir / _SUMMARY_NAME
        staged_run = publish_dir / "run.json"
        _write_text_atomic(staged_records, records_text)
        _write_text_atomic(staged_summary, summary_text)
        completed_manifest["outputs"] = {
            "records": {
                "path": _RECORDS_NAME,
                "sha256": _sha256_file(staged_records),
                "records": len(public_rows),
            },
            "summary": {
                "path": _SUMMARY_NAME,
                "sha256": _sha256_file(staged_summary),
            },
        }
        _write_json_atomic(staged_run, completed_manifest)
        (publish_dir / _OWNER_NAME).unlink()
        _fsync_directory(publish_dir)
        _commit_publish_directory(publish_dir, output_dir)
    except Exception:
        if publish_dir.exists() and not output_dir.exists():
            _remove_owned_private_directory(
                publish_dir,
                output_dir=output_dir,
                run_id=run_id,
            )
        raise
    try:
        _remove_owned_private_directory(
            work_dir,
            output_dir=output_dir,
            run_id=run_id,
        )
    except OSError:
        # The final directory and completed run manifest are already committed.
        # A stale private checkpoint directory is safe to remove on --resume.
        pass
    return RestorationOrderResult(
        output_dir / _RECORDS_NAME,
        output_dir / _SUMMARY_NAME,
        output_dir / "run.json",
        len(public_rows),
    )


def _run_restoration_order_accuracy_locked(
    config: RestorationOrderConfig,
    *,
    runtime_factory: Callable[..., GenerationRuntime] | None = None,
) -> RestorationOrderResult:
    """Generate one strict setting and publish only its complete eleven-arm cohort."""
    source = load_restoration_order_source(
        config.pairs,
        model=config.model,
        benchmark=config.benchmark,
        limit=config.limit,
    )
    records = tuple(dict(record) for record in getattr(source, "records"))
    if not records:
        raise ValueError("restoration-order source selected no runnable records")
    sample_ids = tuple(str(record.get("sample_id", "")) for record in records)
    if sample_ids != tuple(sorted(sample_ids)) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("selected source sample IDs must be unique and sorted")
    plans = _plans(source, records, seed=config.seed)
    if tuple(plan.sample_id for plan in plans) != sample_ids:
        raise ValueError("source plans differ from selected sample order")
    prompts_by_sample = {
        plan.sample_id: _condition_prompts(record, plan)
        for record, plan in zip(records, plans, strict=True)
    }

    output_dir = config.output_dir.resolve()
    records_path = output_dir / _RECORDS_NAME
    summary_path = output_dir / _SUMMARY_NAME
    final_run_path = output_dir / "run.json"
    work_dir = _owned_sibling(output_dir, _WORK_SUFFIX)
    publish_dir = _owned_sibling(output_dir, _PUBLISH_SUFFIX)
    source_payload = dict(getattr(source, "to_dict")())
    code = implementation_code_identity()
    started_at = _now()
    run_id = ""
    previous_runtime: dict[str, object] = {}

    if output_dir.exists():
        if not config.resume:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; "
                "pass --resume only for this run"
            )
        if not final_run_path.is_file():
            raise ValueError(f"completed output lacks run.json: {output_dir}")
        previous = _load_json(final_run_path)
        _previous_started_at, run_id = _validate_resume(
            previous,
            config=config,
            source=source_payload,
            code=code,
        )
        if previous.get("status") != "completed":
            raise ValueError("final output directory does not contain a completed run")
        source.assert_unchanged()
        count = _validate_public_outputs(
            previous,
            records_path=records_path,
            summary_path=summary_path,
        )
        for stale in (work_dir, publish_dir):
            if stale.exists():
                if not stale.is_dir():
                    raise ValueError(f"owned private path is not a directory: {stale}")
                _remove_owned_private_directory(
                    stale,
                    output_dir=output_dir,
                    run_id=run_id,
                )
        return RestorationOrderResult(
            records_path, summary_path, final_run_path, count
        )

    if config.resume:
        run_path = work_dir / "run.json"
        if not run_path.is_file():
            raise ValueError(f"cannot resume without the private run manifest: {work_dir}")
        owner_run_id = _private_owner_id(work_dir, output_dir=output_dir)
        previous = _load_json(run_path)
        started_at, run_id = _validate_resume(
            previous,
            config=config,
            source=source_payload,
            code=code,
        )
        if owner_run_id != run_id:
            raise ValueError("private work directory owner differs from its run manifest")
        status = previous.get("status")
        if status not in {"running", "failed"}:
            raise ValueError(f"resume manifest has unsupported status: {status!r}")
        previous_runtime = dict(_mapping(previous.get("runtime"), field="resume runtime"))
        if publish_dir.exists():
            if not publish_dir.is_dir():
                raise ValueError(f"owned publication stage is not a directory: {publish_dir}")
            _remove_owned_private_directory(
                publish_dir,
                output_dir=output_dir,
                run_id=run_id,
            )
    else:
        if os.path.lexists(work_dir) or os.path.lexists(publish_dir):
            raise FileExistsError(
                f"private state already exists for {output_dir}; pass --resume or inspect it"
            )
        run_id = uuid.uuid4().hex
        _create_owned_private_directory(
            work_dir,
            output_dir=output_dir,
            run_id=run_id,
        )
        run_path = work_dir / "run.json"

    total_batches = len(ALL_CONDITION_IDS) * (
        (len(records) + config.batch_size - 1) // config.batch_size
    )
    runtime: GenerationRuntime | None = None
    runtime_payload: dict[str, object] = dict(previous_runtime)
    completed_batches = 0
    generated: dict[str, dict[str, RestorationGeneration]] = {
        condition_id: {} for condition_id in ALL_CONDITION_IDS
    }
    individual_retry: dict[str, dict[str, bool]] = {
        condition_id: {} for condition_id in ALL_CONDITION_IDS
    }
    retry_batches: list[dict[str, object]] = []
    _write_json_atomic(
        run_path,
        _core_manifest(
            config=config,
            status="running",
            started_at=started_at,
            source=source_payload,
            code=code,
            runtime=runtime_payload,
            completed_batches=0,
            total_batches=total_batches,
            run_id=run_id,
        ),
    )
    try:
        for condition_id in ALL_CONDITION_IDS:
            prompts = tuple(
                prompts_by_sample[sample_id][condition_id] for sample_id in sample_ids
            )
            gold_answers = tuple(_gold_answer(record) for record in records)
            for batch_index, start in enumerate(range(0, len(records), config.batch_size)):
                stop = min(start + config.batch_size, len(records))
                batch_prompts = prompts[start:stop]
                batch_ids = sample_ids[start:stop]
                batch_answers = gold_answers[start:stop]
                checkpoint = _checkpoint_path(work_dir, condition_id, batch_index)
                if checkpoint.is_file():
                    if not runtime_payload:
                        raise ValueError(
                            "existing checkpoints lack generation runtime provenance"
                        )
                    identity = _checkpoint_identity(
                        condition_id=condition_id,
                        sample_ids=batch_ids,
                        prompts=batch_prompts,
                        gold_answers=batch_answers,
                        source=source_payload,
                        code=code,
                        runtime=runtime_payload,
                    )
                    rows, used_individual_fallback = _load_checkpoint(
                        checkpoint,
                        identity=identity,
                        sample_ids=batch_ids,
                    )
                else:
                    if runtime is None:
                        if runtime_factory is None:
                            from typo_cot.experiments.restoration_order_accuracy.runtime import (
                                HuggingFaceRestorationRuntime,
                            )

                            runtime_factory = HuggingFaceRestorationRuntime
                        runtime = runtime_factory(config, revision=source.model_revision)
                        current_runtime = runtime.provenance()
                        if current_runtime.get("implementation_code") != code:
                            raise ValueError(
                                "generation runtime implementation code differs "
                                "from the runner snapshot"
                            )
                        if previous_runtime and current_runtime != previous_runtime:
                            raise ValueError("resume generation runtime provenance differs")
                        runtime_payload = current_runtime
                        _write_json_atomic(
                            run_path,
                            _core_manifest(
                                config=config,
                                status="running",
                                started_at=started_at,
                                source=source_payload,
                                code=code,
                                runtime=runtime_payload,
                                completed_batches=completed_batches,
                                total_batches=total_batches,
                                run_id=run_id,
                            ),
                        )
                    identity = _checkpoint_identity(
                        condition_id=condition_id,
                        sample_ids=batch_ids,
                        prompts=batch_prompts,
                        gold_answers=batch_answers,
                        source=source_payload,
                        code=code,
                        runtime=runtime_payload,
                    )
                    rows, used_individual_fallback = _generate_batch_with_fallback(
                        runtime,
                        prompts=batch_prompts,
                        sample_ids=batch_ids,
                        gold_answers=batch_answers,
                    )
                    _write_json_atomic(
                        checkpoint,
                        {
                            "schema_version": "restoration-order-batch-checkpoint/v1",
                            "condition_id": condition_id,
                            "batch_index": batch_index,
                            "input_sha256": identity,
                            "individual_retry": used_individual_fallback,
                            "generations_sha256": canonical_sha256(
                                [row.to_dict() for row in rows]
                            ),
                            "generations": [row.to_dict() for row in rows],
                        },
                    )
                if used_individual_fallback:
                    retry_batches.append(
                        {
                            "condition_id": condition_id,
                            "batch_index": batch_index,
                            "sample_ids": list(batch_ids),
                        }
                    )
                for row in rows:
                    if row.sample_id in generated[condition_id]:
                        raise ValueError("duplicate sample generation in one condition")
                    generated[condition_id][row.sample_id] = row
                    individual_retry[condition_id][row.sample_id] = (
                        used_individual_fallback
                    )
                completed_batches += 1
                _write_json_atomic(
                    run_path,
                    _core_manifest(
                        config=config,
                        status="running",
                        started_at=started_at,
                        source=source_payload,
                        code=code,
                        runtime=runtime_payload,
                        completed_batches=completed_batches,
                        total_batches=total_batches,
                        run_id=run_id,
                    ),
                )
        if not runtime_payload:
            raise ValueError("complete checkpoints lack generation runtime provenance")
    except Exception as exc:
        _write_json_atomic(
            run_path,
            _core_manifest(
                config=config,
                status="failed",
                started_at=started_at,
                source=source_payload,
                code=code,
                runtime=runtime_payload,
                completed_batches=completed_batches,
                total_batches=total_batches,
                run_id=run_id,
                failure={"error_type": type(exc).__name__, "message": str(exc)},
            ),
        )
        raise RestorationOrderRunError(
            f"restoration-order generation failed after {completed_batches} complete batches: {exc}"
        ) from exc
    finally:
        if runtime is not None:
            active_exception = sys.exc_info()[0] is not None
            try:
                runtime.close()
            except Exception as close_error:
                if not active_exception:
                    _write_json_atomic(
                        run_path,
                        _core_manifest(
                            config=config,
                            status="failed",
                            started_at=started_at,
                            source=source_payload,
                            code=code,
                            runtime=runtime_payload,
                            completed_batches=completed_batches,
                            total_batches=total_batches,
                            run_id=run_id,
                            failure={
                                "error_type": type(close_error).__name__,
                                "message": str(close_error),
                            },
                        ),
                    )
                    raise RestorationOrderRunError(
                        "restoration-order runtime cleanup failed after generation: "
                        f"{close_error}"
                    ) from close_error

    try:
        return _publish_completed_outputs(
            config=config,
            source=source,
            source_payload=source_payload,
            code=code,
            runtime_payload=runtime_payload,
            started_at=started_at,
            run_id=run_id,
            total_batches=total_batches,
            records=records,
            plans=plans,
            prompts_by_sample=prompts_by_sample,
            generated=generated,
            individual_retry=individual_retry,
            retry_batches=retry_batches,
            sample_ids=sample_ids,
            output_dir=output_dir,
            publish_dir=publish_dir,
            work_dir=work_dir,
        )
    except Exception as exc:
        _write_json_atomic(
            run_path,
            _core_manifest(
                config=config,
                status="failed",
                started_at=started_at,
                source=source_payload,
                code=code,
                runtime=runtime_payload,
                completed_batches=completed_batches,
                total_batches=total_batches,
                run_id=run_id,
                failure={"error_type": type(exc).__name__, "message": str(exc)},
            ),
        )
        raise RestorationOrderRunError(
            f"restoration-order publication failed after complete generation: {exc}"
        ) from exc


def run_restoration_order_accuracy(
    config: RestorationOrderConfig,
    *,
    runtime_factory: Callable[..., GenerationRuntime] | None = None,
) -> RestorationOrderResult:
    """Run one invocation under an inter-process output-identity lock."""
    output_dir = config.output_dir.resolve()
    with _exclusive_output_lock(output_dir):
        return _run_restoration_order_accuracy_locked(
            config,
            runtime_factory=runtime_factory,
        )


__all__ = [
    "RestorationGeneration",
    "RestorationOrderConfig",
    "RestorationOrderResult",
    "RestorationOrderRunError",
    "run_restoration_order_accuracy",
]
