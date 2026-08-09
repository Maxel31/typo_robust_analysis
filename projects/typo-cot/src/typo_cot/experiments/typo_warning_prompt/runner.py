"""Validated and restartable setting runner for the typo-warning audit."""

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
from typing import Any, Literal, Protocol

from tqdm.auto import tqdm

import typo_cot.experiments.typo_warning_prompt.protocol as warning_protocol
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.typo_warning_prompt.planning import (
    WarningPromptPlan,
    build_warning_prompt_plan,
)
from typo_cot.experiments.typo_warning_prompt.integrity import (
    implementation_code_identity,
    validate_paper_runtime_environment,
)
from typo_cot.experiments.typo_warning_prompt.source import (
    WarningSourceBundle,
    WarningSourceCase,
    canonical_sha256,
    load_json,
    load_warning_sources,
    select_warning_cases,
    sha256_file,
    strict_loads,
    validate_source_snapshot,
)
from typo_cot.experiments.typo_warning_prompt.scoring import extract_submitted_answer
from typo_cot.experiments.typo_warning_prompt.statistics import exact_mcnemar

ARM_ORDER = warning_protocol.ARM_ORDER
_ANSWER_EXTRACTION = warning_protocol.ANSWER_EXTRACTION
_ANSWER_SPAN_DECODING = warning_protocol.ANSWER_SPAN_DECODING
_BATCHING = warning_protocol.BATCHING
_GENERATION = warning_protocol.GENERATION
_IMPLEMENTATION = warning_protocol.IMPLEMENTATION
_PROMPT_INTERVENTION = warning_protocol.PROMPT_INTERVENTION
_PROTOCOL = warning_protocol.PROTOCOL
_PROTOCOL_SHA256 = warning_protocol.PROTOCOL_SHA256
_SELECTION = warning_protocol.SELECTION
_HISTORICAL_REFERENCE = warning_protocol.HISTORICAL_REFERENCE

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PUBLIC_OUTPUTS = ("warning_prompt_records.jsonl", "warning_prompt_summary.json")


class WarningPromptRunError(RuntimeError):
    """A setting failed after its valid checkpoints were retained."""


@dataclass(frozen=True, slots=True)
class WarningPromptConfig:
    """Frozen public arguments for one model-task warning comparison."""

    model: str
    benchmark: Literal["gsm8k", "mmlu"]
    output_dir: Path
    input_fixture: Path | None = None
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.input_fixture is not None:
            object.__setattr__(self, "input_fixture", Path(self.input_fixture))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in warning_protocol.PAPER_BENCHMARKS:
            raise ValueError("benchmark must be gsm8k or mmlu")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        if type(self.resume) is not bool:
            raise TypeError("resume must be boolean")


@dataclass(frozen=True, slots=True)
class WarningPromptInputUse:
    """Tokenized prompt identity actually passed to the model."""

    arm: str
    prompt_text_sha256: str
    prompt_char_count: int
    prompt_token_count: int
    prompt_ids_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in ARM_ORDER:
            raise ValueError(f"unsupported warning arm: {self.arm!r}")
        for field in ("prompt_text_sha256", "prompt_ids_sha256"):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                raise ValueError(f"{field} must be a full lowercase SHA-256")
        for field in ("prompt_char_count", "prompt_token_count"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "prompt_text_sha256": self.prompt_text_sha256,
            "prompt_char_count": self.prompt_char_count,
            "prompt_token_count": self.prompt_token_count,
            "prompt_ids_sha256": self.prompt_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class WarningPromptGeneration:
    """One generated answer span and its symmetric extraction result."""

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
        if not self.token_ids or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("generation token_ids must be non-empty non-negative integers")
        if not isinstance(self.text, str) or not isinstance(self.value, str):
            raise TypeError("generation text and value must be strings")
        if type(self.is_extracted) is not bool or type(self.is_correct) is not bool:
            raise TypeError("generation extraction flags must be boolean")
        if not self.method or not self.primary_method:
            raise ValueError("generation extraction methods must not be empty")
        if self.stop_reason not in {"eos_token", "max_new_tokens"}:
            raise ValueError("generation stop_reason is unsupported")
        if self.stop_reason == "eos_token":
            if self.stop_token_id != self.token_ids[-1]:
                raise ValueError("EOS stop_token_id must be the final retained token")
        elif self.stop_token_id is not None:
            raise ValueError("max_new_tokens generation must not record a stop token")

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "text": self.text,
            "value": self.value,
            "is_extracted": self.is_extracted,
            "is_correct": self.is_correct,
            "method": self.method,
            "primary_method": self.primary_method,
            "stop_reason": self.stop_reason,
            "stop_token_id": self.stop_token_id,
        }


@dataclass(frozen=True, slots=True)
class WarningPromptArmScan:
    """One sample-arm result returned from a submitted-compatible batch."""

    sample_id: str
    arm: str
    input_use: WarningPromptInputUse
    generation: WarningPromptGeneration

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("arm scan sample_id must not be empty")
        if self.arm not in ARM_ORDER:
            raise ValueError(f"unsupported warning arm: {self.arm!r}")
        if self.input_use.arm != self.arm:
            raise ValueError("arm scan input-use arm differs")


@dataclass(frozen=True, slots=True)
class WarningPromptScan:
    """Both paired warning outcomes for one source item."""

    sample_id: str
    input_uses: Mapping[str, WarningPromptInputUse]
    generations: Mapping[str, WarningPromptGeneration]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("scan sample_id must not be empty")
        if set(self.input_uses) != set(ARM_ORDER) or set(self.generations) != set(ARM_ORDER):
            raise ValueError("scan must contain both warning arms exactly once")
        for arm in ARM_ORDER:
            if self.input_uses[arm].arm != arm:
                raise ValueError("input-use mapping key and arm disagree")


class WarningPromptRuntime(Protocol):
    def provenance(self) -> dict[str, object]: ...

    def scan_arm_batch(
        self,
        pairs: Sequence[dict[str, object]],
        plans: Sequence[WarningPromptPlan],
        *,
        arm: str,
    ) -> Sequence[WarningPromptArmScan]: ...


@dataclass(frozen=True, slots=True)
class WarningPromptResult:
    records_path: Path
    summary_path: Path
    run_path: Path
    records: int


# Kept import-lazy so completed resumes and CPU validation never load torch.
HuggingFaceWarningPromptRuntime: Any = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(path, _serialized_json(payload))


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=False, allow_nan=False) + "\n"
            for row in rows
        ),
    )


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


def _arguments(config: WarningPromptConfig) -> dict[str, object]:
    from typo_cot.experiments.typo_warning_prompt.source import DEFAULT_SUBMITTED_INPUTS

    input_fixture = config.input_fixture or DEFAULT_SUBMITTED_INPUTS
    return {
        "model": config.model,
        "benchmark": config.benchmark,
        "input_fixture": str(input_fixture.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "gpu_id": config.gpu_id,
        "limit": config.limit,
    }


def _plan_payload(
    source: WarningSourceBundle,
    selected: Sequence[WarningSourceCase],
    executed: Sequence[WarningSourceCase],
) -> dict[str, object]:
    return {
        "selection": dict(_SELECTION),
        "shared_source_pairs": source.shared_sample_count,
        "selected_pairs": len(selected),
        "executed_pairs": len(executed),
        "selected_ids_sha256": canonical_sha256([case.sample_id for case in selected]),
        "executed_ids_sha256": canonical_sha256([case.sample_id for case in executed]),
        "selected_source_sha256": canonical_sha256([case.identity for case in selected]),
        "executed_source_sha256": canonical_sha256([case.identity for case in executed]),
    }


def _comparability(
    config: WarningPromptConfig,
    source: WarningSourceBundle,
) -> dict[str, object]:
    exact_submitted_inputs = (
        source.fixture_is_bundled_submitted_input
        and source.shared_sample_count == int(_SELECTION["pairs_per_setting"])
    )
    requirements = {
        "final_paper_operation": True,
        "submitted_model": config.model in warning_protocol.PAPER_MODELS,
        "submitted_benchmark": config.benchmark in warning_protocol.PAPER_BENCHMARKS,
        "exact_bundled_submitted_input_fixture": exact_submitted_inputs,
        "not_limit_truncated": config.limit is None,
    }
    limitations = list(warning_protocol.COMPARABILITY_LIMITATIONS)
    if not exact_submitted_inputs:
        limitations.append("input-fixture-is-not-the-bundled-submitted-artifact")
    if config.model not in warning_protocol.PAPER_MODELS:
        limitations.append("model-is-outside-submitted-grid")
    if config.limit is not None:
        limitations.append("run-is-limit-truncated")
        status = "partial-smoke-run"
    elif exact_submitted_inputs and config.model in warning_protocol.PAPER_MODELS:
        status = "exact-submitted-input-paper-grid-setting"
    else:
        status = "custom-input-protocol-setting"
    return {
        "status": status,
        "requirements": requirements,
        "limitations": limitations,
        "historical_cohort_identity": exact_submitted_inputs,
    }


def _validate_runtime_provenance(
    provenance: Mapping[str, object],
    *,
    config: WarningPromptConfig,
    source: WarningSourceBundle,
) -> dict[str, object]:
    validate_paper_runtime_environment(provenance, field_prefix="runtime provenance")
    required = {
        "operation": "typo-warning-prompt",
        "model": config.model,
        "requested_revision": source.model_revision,
        "model_revision": source.model_revision,
        "tokenizer_revision": source.model_revision,
        "dtype": "bfloat16",
        "cuda_visible_devices": config.gpu_id,
        "implementation": _IMPLEMENTATION,
        "answer_extraction": _ANSWER_EXTRACTION,
        "prompt_intervention": _PROMPT_INTERVENTION,
        "implementation_code": implementation_code_identity(),
    }
    for field, expected in required.items():
        if provenance.get(field) != expected:
            raise ValueError(f"runtime provenance {field} must be {expected!r}")
    for name, expected in (
        ("generation", _GENERATION),
        ("batching", _BATCHING),
        ("answer_span_decoding", _ANSWER_SPAN_DECODING),
    ):
        actual = _mapping(provenance.get(name), field=f"runtime {name}")
        if dict(actual) != expected:
            raise ValueError(f"runtime {name} does not match the public protocol")
    eos = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos, list)
        or not eos
        or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos)
    ):
        raise ValueError("runtime effective_eos_token_ids must be non-empty integers")
    _string(
        provenance.get("effective_eos_token_ids_source"),
        field="runtime effective_eos_token_ids_source",
    )
    return dict(provenance)


def _manifest(
    *,
    config: WarningPromptConfig,
    source: WarningSourceBundle,
    plan: Mapping[str, object],
    runtime: Mapping[str, object],
    status: str,
    started_at: str,
    completed_arms: int,
    completed_pairs: int,
    failures: Sequence[Mapping[str, str]],
    checkpoints: Mapping[str, object],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "typo-warning-prompt-run/v2",
        "paper_sha256": PAPER_SHA256,
        "operation": "typo-warning-prompt",
        "status": status,
        "started_at": started_at,
        "updated_at": _now(),
        "protocol": _PROTOCOL,
        "protocol_sha256": _PROTOCOL_SHA256,
        "arguments": _arguments(config),
        "source": source.to_dict(),
        "plan": dict(plan),
        "runtime": dict(runtime),
        "counts": {
            "shared_source_pairs": source.shared_sample_count,
            "selected_pairs": plan["selected_pairs"],
            "executed_pairs": plan["executed_pairs"],
            "completed_arms": completed_arms,
            "completed_pairs": completed_pairs,
            "failed": len(failures),
        },
        "failures": list(failures),
        "checkpoints": dict(checkpoints),
        "comparability": _comparability(config, source),
        "historical_reference": _HISTORICAL_REFERENCE,
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
        payload["completed_at"] = _now()
    return payload


def _checkpoint_key(sample_id: str, arm: str) -> str:
    return f"{sample_id}|{arm}"


def _checkpoint_name(sample_id: str, arm: str) -> str:
    return (
        hashlib.sha256(
            json.dumps([sample_id, arm], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        + ".json"
    )


def _checkpoint_payload(
    scan: WarningPromptArmScan,
    *,
    case: WarningSourceCase,
    plan: WarningPromptPlan,
    runtime_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "typo-warning-prompt-arm-checkpoint/v2",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": _PROTOCOL_SHA256,
        "sample_id": case.sample_id,
        "arm": scan.arm,
        "submitted_record_sha256": case.submitted_record_sha256,
        "selector_record_sha256": case.selector_record_sha256,
        "plan_sha256": plan.fingerprint,
        "runtime_sha256": runtime_sha256,
        "input_use": scan.input_use.to_dict(),
        "generation": scan.generation.to_dict(),
    }


def _input_use_from_dict(value: object, *, arm: str) -> WarningPromptInputUse:
    payload = _mapping(value, field=f"checkpoint input use {arm}")
    return WarningPromptInputUse(
        arm=_string(payload.get("arm"), field="input use arm"),
        prompt_text_sha256=_string(
            payload.get("prompt_text_sha256"), field="input use prompt_text_sha256"
        ),
        prompt_char_count=_int(
            payload.get("prompt_char_count"), field="input use prompt_char_count", minimum=1
        ),
        prompt_token_count=_int(
            payload.get("prompt_token_count"), field="input use prompt_token_count", minimum=1
        ),
        prompt_ids_sha256=_string(
            payload.get("prompt_ids_sha256"), field="input use prompt_ids_sha256"
        ),
    )


def _generation_from_dict(value: object, *, arm: str) -> WarningPromptGeneration:
    payload = _mapping(value, field=f"checkpoint generation {arm}")
    token_ids = payload.get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError("checkpoint generation token_ids must be a list")
    is_extracted = payload.get("is_extracted")
    is_correct = payload.get("is_correct")
    if type(is_extracted) is not bool or type(is_correct) is not bool:
        raise ValueError("checkpoint generation flags must be boolean")
    stop_reason = payload.get("stop_reason")
    if stop_reason not in {"eos_token", "max_new_tokens"}:
        raise ValueError("checkpoint generation stop_reason is invalid")
    stop_token_id = payload.get("stop_token_id")
    if stop_token_id is not None and (
        not isinstance(stop_token_id, int) or isinstance(stop_token_id, bool)
    ):
        raise ValueError("checkpoint generation stop_token_id is invalid")
    text = payload.get("text")
    answer_value = payload.get("value")
    if not isinstance(text, str) or not isinstance(answer_value, str):
        raise ValueError("checkpoint generation text and value must be strings")
    return WarningPromptGeneration(
        token_ids=tuple(token_ids),
        text=text,
        value=answer_value,
        is_extracted=is_extracted,
        is_correct=is_correct,
        method=_string(payload.get("method"), field="generation method"),
        primary_method=_string(payload.get("primary_method"), field="generation primary_method"),
        stop_reason=stop_reason,
        stop_token_id=stop_token_id,
    )


def _validate_arm_scan(
    scan: WarningPromptArmScan,
    *,
    case: WarningSourceCase,
    plan: WarningPromptPlan,
    benchmark: str,
    effective_eos_token_ids: Sequence[int],
) -> WarningPromptArmScan:
    if scan.sample_id != case.sample_id or scan.sample_id != plan.sample_id:
        raise ValueError("runtime scan sample_id differs from its source plan")
    edited = _mapping(case.submitted_record.get("edited"), field="submitted source edited")
    arm = scan.arm
    arm_plan = next((item for item in plan.arms if item.arm == arm), None)
    if arm_plan is None:
        raise ValueError(f"runtime arm is absent from the fixed plan: {arm}")
    use = scan.input_use
    generation = scan.generation
    if use.prompt_text_sha256 != arm_plan.prompt_sha256 or use.prompt_char_count != len(
        arm_plan.prompt
    ):
        raise ValueError(f"runtime input identity differs from plan for {arm}")
    if arm == "without-warning" and use.prompt_text_sha256 != edited.get("prompt_sha256"):
        raise ValueError("runtime without-warning prompt differs from submitted input")
    if generation.is_extracted != bool(generation.value):
        raise ValueError(f"runtime extraction flag differs from value for {arm}")
    eos_ids = set(effective_eos_token_ids)
    if generation.stop_reason == "eos_token":
        if generation.stop_token_id not in eos_ids:
            raise ValueError(f"runtime stop token is not an effective EOS token for {arm}")
        if any(token in eos_ids for token in generation.token_ids[:-1]):
            raise ValueError(f"runtime generation contains an early EOS token for {arm}")
    elif len(generation.token_ids) != int(_GENERATION["max_new_tokens"]) or any(
        token in eos_ids for token in generation.token_ids
    ):
        raise ValueError(f"runtime max-token generation is inconsistent for {arm}")
    extraction = extract_submitted_answer(
        generation.text,
        benchmark=benchmark,
        correct_answer=plan.gold_answer,
    )
    observed_extraction = (
        generation.value,
        generation.is_extracted,
        generation.is_correct,
        generation.method,
        generation.primary_method,
    )
    expected_extraction = (
        extraction.value,
        extraction.is_extracted,
        extraction.is_correct,
        extraction.method,
        extraction.primary_method,
    )
    if observed_extraction != expected_extraction:
        raise ValueError(f"runtime answer extraction differs from generated text for {arm}")
    return scan


def _validate_scan(
    scan: WarningPromptScan,
    *,
    case: WarningSourceCase,
    plan: WarningPromptPlan,
    benchmark: str,
    effective_eos_token_ids: Sequence[int],
) -> WarningPromptScan:
    if set(scan.input_uses) != set(ARM_ORDER) or set(scan.generations) != set(ARM_ORDER):
        raise ValueError("runtime pair scan must contain both warning arms")
    for arm in ARM_ORDER:
        _validate_arm_scan(
            WarningPromptArmScan(
                sample_id=scan.sample_id,
                arm=arm,
                input_use=scan.input_uses[arm],
                generation=scan.generations[arm],
            ),
            case=case,
            plan=plan,
            benchmark=benchmark,
            effective_eos_token_ids=effective_eos_token_ids,
        )
    return scan


def _arm_scan_from_checkpoint(
    payload: Mapping[str, object],
    *,
    case: WarningSourceCase,
    plan: WarningPromptPlan,
    runtime_sha256: str,
    benchmark: str,
    effective_eos_token_ids: Sequence[int],
    arm: str,
) -> WarningPromptArmScan:
    expected = {
        "schema_version": "typo-warning-prompt-arm-checkpoint/v2",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": _PROTOCOL_SHA256,
        "sample_id": case.sample_id,
        "arm": arm,
        "submitted_record_sha256": case.submitted_record_sha256,
        "selector_record_sha256": case.selector_record_sha256,
        "plan_sha256": plan.fingerprint,
        "runtime_sha256": runtime_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"checkpoint {field} does not match the current run")
    scan = WarningPromptArmScan(
        sample_id=case.sample_id,
        arm=arm,
        input_use=_input_use_from_dict(payload.get("input_use"), arm=arm),
        generation=_generation_from_dict(payload.get("generation"), arm=arm),
    )
    return _validate_arm_scan(
        scan,
        case=case,
        plan=plan,
        benchmark=benchmark,
        effective_eos_token_ids=effective_eos_token_ids,
    )


def _record(
    scan: WarningPromptScan,
    *,
    case: WarningSourceCase,
    plan: WarningPromptPlan,
    config: WarningPromptConfig,
) -> dict[str, object]:
    off = scan.generations["without-warning"].is_correct
    on = scan.generations["with-warning"].is_correct
    arms: dict[str, object] = {}
    for arm_plan in plan.arms:
        arm = arm_plan.arm
        arms[arm] = {
            "warning_enabled": arm_plan.warning_enabled,
            "fixed_input": arm_plan.to_dict(include_text=False),
            "input_use": scan.input_uses[arm].to_dict(),
            "answer": scan.generations[arm].to_dict(),
        }
    return {
        "schema_version": "typo-warning-prompt-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "typo-warning-prompt",
        "model": config.model,
        "benchmark": config.benchmark,
        "sample_id": case.sample_id,
        "gold_answer": plan.gold_answer,
        "source": {
            "submitted_record_sha256": case.submitted_record_sha256,
            "selector_record_sha256": case.selector_record_sha256,
        },
        "plan_sha256": plan.fingerprint,
        "arms": arms,
        "events": {
            "without_warning_correct": off,
            "with_warning_correct": on,
            "without_warning_only": off and not on,
            "with_warning_only": on and not off,
        },
    }


def _summary_core(
    records: Sequence[Mapping[str, object]],
    *,
    config: WarningPromptConfig,
    source: WarningSourceBundle,
    plan: Mapping[str, object],
) -> tuple[dict[str, object], int, int]:
    n = len(records)
    if n <= 0:
        raise ValueError("warning summary requires at least one record")
    events = [_mapping(record.get("events"), field="warning record events") for record in records]
    off = sum(event.get("without_warning_correct") is True for event in events)
    on = sum(event.get("with_warning_correct") is True for event in events)
    b10 = sum(event.get("without_warning_only") is True for event in events)
    b01 = sum(event.get("with_warning_only") is True for event in events)
    return (
        {
            "schema_version": "typo-warning-prompt-summary/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "typo-warning-prompt",
            "model": config.model,
            "benchmark": config.benchmark,
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
            "selection": dict(plan),
            "comparability": _comparability(config, source),
            "historical_reference": _HISTORICAL_REFERENCE["pooled_by_benchmark"][config.benchmark],
        },
        b10,
        b01,
    )


def _summary(
    records: Sequence[Mapping[str, object]],
    *,
    config: WarningPromptConfig,
    source: WarningSourceBundle,
    plan: Mapping[str, object],
) -> dict[str, object]:
    core, b10, b01 = _summary_core(
        records,
        config=config,
        source=source,
        plan=plan,
    )
    return {
        **core,
        "mcnemar": exact_mcnemar(b10=b10, b01=b01),
    }


def _output_metadata(path: Path, records: int) -> dict[str, object]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size, "records": records}


def _validate_resume_core(
    previous: Mapping[str, object],
    *,
    config: WarningPromptConfig,
    source: WarningSourceBundle,
    plan: Mapping[str, object],
) -> str:
    if previous.get("schema_version") != "typo-warning-prompt-run/v2":
        raise ValueError("resume manifest has an unknown schema")
    if previous.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("resume manifest paper SHA-256 differs")
    if previous.get("operation") != "typo-warning-prompt":
        raise ValueError("resume manifest operation differs")
    if previous.get("protocol_sha256") != _PROTOCOL_SHA256 or previous.get("protocol") != _PROTOCOL:
        raise ValueError("resume protocol differs")
    if previous.get("arguments") != _arguments(config):
        raise ValueError("resume arguments differ from the original run")
    if previous.get("source") != source.to_dict():
        raise ValueError("resume source identity differs from the original run")
    if previous.get("plan") != dict(plan):
        raise ValueError("resume selected cohort differs from the original run")
    return _string(previous.get("started_at"), field="resume started_at")


def _validate_record_semantics(
    record: Mapping[str, object],
    *,
    config: WarningPromptConfig,
    case: WarningSourceCase,
    plan: WarningPromptPlan,
    effective_eos_token_ids: Sequence[int],
) -> None:
    expected = {
        "schema_version": "typo-warning-prompt-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "typo-warning-prompt",
        "model": config.model,
        "benchmark": config.benchmark,
        "sample_id": case.sample_id,
        "gold_answer": plan.gold_answer,
        "plan_sha256": plan.fingerprint,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"completed record {field} differs")
    source = _mapping(record.get("source"), field="completed record source")
    if source != {
        "submitted_record_sha256": case.submitted_record_sha256,
        "selector_record_sha256": case.selector_record_sha256,
    }:
        raise ValueError("completed record source identity differs")
    arms = _mapping(record.get("arms"), field="completed record arms")
    if set(arms) != set(ARM_ORDER):
        raise ValueError("completed record must contain both protocol arms")
    input_uses: dict[str, WarningPromptInputUse] = {}
    generations: dict[str, WarningPromptGeneration] = {}
    for arm_plan in plan.arms:
        arm = arm_plan.arm
        arm_payload = _mapping(arms[arm], field=f"completed record arm {arm}")
        if arm_payload.get("warning_enabled") != arm_plan.warning_enabled:
            raise ValueError("completed arm warning flag differs")
        if arm_payload.get("fixed_input") != arm_plan.to_dict(include_text=False):
            raise ValueError("completed arm fixed input differs")
        input_uses[arm] = _input_use_from_dict(arm_payload.get("input_use"), arm=arm)
        generations[arm] = _generation_from_dict(arm_payload.get("answer"), arm=arm)
    scan = _validate_scan(
        WarningPromptScan(
            sample_id=case.sample_id,
            input_uses=input_uses,
            generations=generations,
        ),
        case=case,
        plan=plan,
        benchmark=config.benchmark,
        effective_eos_token_ids=effective_eos_token_ids,
    )
    correctness = {arm: scan.generations[arm].is_correct for arm in ARM_ORDER}
    off = correctness["without-warning"]
    on = correctness["with-warning"]
    expected_events = {
        "without_warning_correct": off,
        "with_warning_correct": on,
        "without_warning_only": off and not on,
        "with_warning_only": on and not off,
    }
    if record.get("events") != expected_events:
        raise ValueError("completed record events differ from arm outcomes")


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


def _validate_completed(
    manifest: Mapping[str, object],
    *,
    config: WarningPromptConfig,
    source: WarningSourceBundle,
    selected: Sequence[WarningSourceCase],
    executed: Sequence[WarningSourceCase],
    plan_payload: Mapping[str, object],
) -> WarningPromptResult:
    records_path = config.output_dir / _PUBLIC_OUTPUTS[0]
    summary_path = config.output_dir / _PUBLIC_OUTPUTS[1]
    run_path = config.output_dir / "run.json"
    try:
        if manifest.get("status") != "completed":
            raise ValueError("manifest is not completed")
        if manifest.get("checkpoints") != {} or manifest.get("failures") not in ([], None):
            raise ValueError("completed manifest retains checkpoints or failures")
        outputs = _mapping(manifest.get("outputs"), field="completed outputs")
        if set(outputs) != set(_PUBLIC_OUTPUTS):
            raise ValueError("completed output registry has the wrong files")
        for path in (records_path, summary_path):
            if not path.is_file():
                raise ValueError(f"completed public output is missing: {path.name}")
            metadata = _mapping(outputs[path.name], field=f"completed output {path.name}")
            if metadata.get("sha256") != sha256_file(path):
                raise ValueError(f"completed output SHA-256 mismatch: {path.name}")
            if metadata.get("bytes") != path.stat().st_size:
                raise ValueError(f"completed output byte count mismatch: {path.name}")
        records = _load_jsonl(records_path)
        if len(records) != len(executed):
            raise ValueError("completed record count differs from the selected execution cohort")
        case_by_id = {case.sample_id: case for case in executed}
        runtime = _validate_runtime_provenance(
            _mapping(manifest.get("runtime"), field="completed runtime"),
            config=config,
            source=source,
        )
        effective_eos_token_ids = tuple(runtime["effective_eos_token_ids"])
        ids = [record.get("sample_id") for record in records]
        if ids != sorted(case_by_id) or len(ids) != len(set(ids)):
            raise ValueError("completed records do not match sorted selected sample IDs")
        for record in records:
            sample_id = str(record["sample_id"])
            case = case_by_id[sample_id]
            plan = build_warning_prompt_plan(
                case.submitted_record,
                source_record_sha256=case.submitted_record_sha256,
            )
            _validate_record_semantics(
                record,
                config=config,
                case=case,
                plan=plan,
                effective_eos_token_ids=effective_eos_token_ids,
            )
        expected_summary, _, _ = _summary_core(
            records,
            config=config,
            source=source,
            plan=plan_payload,
        )
        stored_summary = dict(_mapping(load_json(summary_path), field="completed setting summary"))
        _mapping(
            stored_summary.pop("mcnemar", None),
            field="completed setting summary mcnemar",
        )
        if stored_summary != expected_summary:
            raise ValueError("completed summary core differs from recomputed integer events")
        if outputs[records_path.name] != _output_metadata(records_path, len(records)):
            raise ValueError("completed records metadata differs")
        if outputs[summary_path.name] != _output_metadata(summary_path, 1):
            raise ValueError("completed summary metadata differs")
        counts = _mapping(manifest.get("counts"), field="completed counts")
        if (
            counts.get("completed_arms") != len(records) * len(ARM_ORDER)
            or counts.get("completed_pairs") != len(records)
            or counts.get("failed") != 0
        ):
            raise ValueError("completed manifest counts differ")
        validate_source_snapshot(source)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise WarningPromptRunError(f"completed run validation failed: {exc}") from exc
    return WarningPromptResult(
        records_path=records_path,
        summary_path=summary_path,
        run_path=run_path,
        records=len(records),
    )


def _load_checkpoints(
    registry_value: object,
    *,
    checkpoints_dir: Path,
    executed: Sequence[WarningSourceCase],
    plans: Mapping[str, WarningPromptPlan],
    runtime_sha256: str,
    benchmark: str,
    effective_eos_token_ids: Sequence[int],
) -> tuple[dict[tuple[str, str], WarningPromptArmScan], dict[str, object]]:
    registry = _mapping(registry_value, field="checkpoint registry")
    cases = {case.sample_id: case for case in executed}
    expected_keys = {
        _checkpoint_key(sample_id, arm): (sample_id, arm)
        for sample_id in cases
        for arm in ARM_ORDER
    }
    scans: dict[tuple[str, str], WarningPromptArmScan] = {}
    normalized: dict[str, object] = {}
    expected_files: set[str] = set()
    for key, raw_metadata in registry.items():
        if key not in expected_keys:
            raise ValueError("checkpoint registry references an unknown selected sample-arm")
        sample_id, arm = expected_keys[key]
        metadata = _mapping(raw_metadata, field=f"checkpoint registry {key}")
        expected_name = _checkpoint_name(sample_id, arm)
        path = checkpoints_dir / expected_name
        expected_files.add(expected_name)
        if metadata.get("file") != expected_name or not path.is_file():
            raise ValueError(f"checkpoint file is missing for {sample_id} {arm}")
        payload = load_json(path)
        digest = sha256_file(path)
        if metadata.get("sha256") != digest:
            raise ValueError(f"checkpoint SHA-256 mismatch for {sample_id} {arm}")
        scans[(sample_id, arm)] = _arm_scan_from_checkpoint(
            payload,
            case=cases[sample_id],
            plan=plans[sample_id],
            runtime_sha256=runtime_sha256,
            benchmark=benchmark,
            effective_eos_token_ids=effective_eos_token_ids,
            arm=arm,
        )
        normalized[key] = {"file": expected_name, "sha256": digest}
    # Adopt a checkpoint whose rename completed immediately before a crash.
    for sample_id, case in cases.items():
        for arm in ARM_ORDER:
            scan_key = (sample_id, arm)
            if scan_key in scans:
                continue
            path = checkpoints_dir / _checkpoint_name(sample_id, arm)
            if not path.is_file():
                continue
            expected_files.add(path.name)
            payload = load_json(path)
            scans[scan_key] = _arm_scan_from_checkpoint(
                payload,
                case=case,
                plan=plans[sample_id],
                runtime_sha256=runtime_sha256,
                benchmark=benchmark,
                effective_eos_token_ids=effective_eos_token_ids,
                arm=arm,
            )
            digest = sha256_file(path)
            normalized[_checkpoint_key(sample_id, arm)] = {
                "file": path.name,
                "sha256": digest,
            }
    actual_files = {path.name for path in checkpoints_dir.glob("*.json")}
    if actual_files != expected_files:
        raise ValueError("checkpoint directory contains unregistered files")
    return scans, normalized


def run_typo_warning_prompt(
    config: WarningPromptConfig,
    *,
    runtime: WarningPromptRuntime | None = None,
) -> WarningPromptResult:
    """Validate sources, execute paired generations, and publish one setting."""

    output_dir = config.output_dir
    run_path = output_dir / "run.json"
    records_path = output_dir / _PUBLIC_OUTPUTS[0]
    summary_path = output_dir / _PUBLIC_OUTPUTS[1]
    work_dir = output_dir / ".typo-warning-work"
    checkpoints_dir = work_dir / "checkpoints"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    source = load_warning_sources(
        config.input_fixture,
        model=config.model,
        benchmark=config.benchmark,
    )
    selected = select_warning_cases(source)
    executed = selected[: config.limit] if config.limit is not None else selected
    if not executed:
        raise ValueError("no shared warning-prompt cases were selected")
    plan_payload = _plan_payload(source, selected, executed)
    plans = {
        case.sample_id: build_warning_prompt_plan(
            case.submitted_record,
            source_record_sha256=case.submitted_record_sha256,
        )
        for case in executed
    }

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
            completed_result = _validate_completed(
                previous,
                config=config,
                source=source,
                selected=selected,
                executed=executed,
                plan_payload=plan_payload,
            )
            if work_dir.exists():
                shutil.rmtree(work_dir)
                _fsync_directory(output_dir)
            return completed_result
        for path in (records_path, summary_path):
            path.unlink(missing_ok=True)
    else:
        started_at = _now()

    if previous is None:
        if runtime is None:
            global HuggingFaceWarningPromptRuntime
            if HuggingFaceWarningPromptRuntime is None:
                from typo_cot.experiments.typo_warning_prompt.runtime import (
                    HuggingFaceWarningPromptRuntime as runtime_class,
                )

                HuggingFaceWarningPromptRuntime = runtime_class
            runtime = HuggingFaceWarningPromptRuntime(config, revision=source.model_revision)
        runtime_provenance = _validate_runtime_provenance(
            runtime.provenance(), config=config, source=source
        )
    else:
        runtime_provenance = _validate_runtime_provenance(
            _mapping(previous.get("runtime"), field="resume runtime"),
            config=config,
            source=source,
        )
    runtime_sha256 = canonical_sha256(runtime_provenance)
    effective_eos_token_ids = tuple(runtime_provenance["effective_eos_token_ids"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    previous_registry: object = previous.get("checkpoints", {}) if previous is not None else {}
    arm_scans, checkpoint_registry = _load_checkpoints(
        previous_registry,
        checkpoints_dir=checkpoints_dir,
        executed=executed,
        plans=plans,
        runtime_sha256=runtime_sha256,
        benchmark=config.benchmark,
        effective_eos_token_ids=effective_eos_token_ids,
    )
    pending_arms = [
        (case, arm)
        for arm in ARM_ORDER
        for case in executed
        if (case.sample_id, arm) not in arm_scans
    ]
    if pending_arms:
        if runtime is None:
            if HuggingFaceWarningPromptRuntime is None:
                from typo_cot.experiments.typo_warning_prompt.runtime import (
                    HuggingFaceWarningPromptRuntime as runtime_class,
                )

                HuggingFaceWarningPromptRuntime = runtime_class
            runtime = HuggingFaceWarningPromptRuntime(config, revision=source.model_revision)
        active = _validate_runtime_provenance(runtime.provenance(), config=config, source=source)
        if active != runtime_provenance:
            raise ValueError("resume runtime provenance differs from the original run")

    failures: list[dict[str, str]] = []

    def completed_pair_count() -> int:
        return sum(
            all((case.sample_id, arm) in arm_scans for arm in ARM_ORDER) for case in executed
        )

    def write_progress(status: str) -> None:
        _write_json_atomic(
            run_path,
            _manifest(
                config=config,
                source=source,
                plan=plan_payload,
                runtime=runtime_provenance,
                status=status,
                started_at=started_at,
                completed_arms=len(arm_scans),
                completed_pairs=completed_pair_count(),
                failures=failures,
                checkpoints=checkpoint_registry,
            ),
        )

    write_progress("running")
    batch_size = int(_BATCHING["batch_size"])
    for arm in ARM_ORDER:
        pending_cases = [case for case in executed if (case.sample_id, arm) not in arm_scans]
        for start in tqdm(
            range(0, len(pending_cases), batch_size),
            desc=f"typo-warning {arm} batches",
            unit="batch",
        ):
            batch = pending_cases[start : start + batch_size]
            batch_ids = [case.sample_id for case in batch]
            batch_label = ",".join(batch_ids)
            try:
                if runtime is None:
                    raise RuntimeError("runtime is unavailable for pending warning arms")
                raw_scans = tuple(
                    runtime.scan_arm_batch(
                        [case.submitted_record for case in batch],
                        [plans[case.sample_id] for case in batch],
                        arm=arm,
                    )
                )
                if [scan.sample_id for scan in raw_scans] != batch_ids or any(
                    scan.arm != arm for scan in raw_scans
                ):
                    raise ValueError(
                        "runtime batch results differ from the requested sorted sample-arm order"
                    )
                validated = [
                    _validate_arm_scan(
                        scan,
                        case=case,
                        plan=plans[case.sample_id],
                        benchmark=config.benchmark,
                        effective_eos_token_ids=effective_eos_token_ids,
                    )
                    for case, scan in zip(batch, raw_scans, strict=True)
                ]
                for case, scan in zip(batch, validated, strict=True):
                    checkpoint_path = checkpoints_dir / _checkpoint_name(case.sample_id, arm)
                    _write_json_atomic(
                        checkpoint_path,
                        _checkpoint_payload(
                            scan,
                            case=case,
                            plan=plans[case.sample_id],
                            runtime_sha256=runtime_sha256,
                        ),
                    )
                    arm_scans[(case.sample_id, arm)] = scan
                    checkpoint_registry[_checkpoint_key(case.sample_id, arm)] = {
                        "file": checkpoint_path.name,
                        "sha256": sha256_file(checkpoint_path),
                    }
                write_progress("running")
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": batch_label,
                        "arm": arm,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                write_progress("failed")
                raise WarningPromptRunError(
                    f"typo-warning generation failed for {batch_label} {arm}: {exc}"
                ) from exc

    try:
        validate_source_snapshot(source)
        scans = {
            case.sample_id: WarningPromptScan(
                sample_id=case.sample_id,
                input_uses={arm: arm_scans[(case.sample_id, arm)].input_use for arm in ARM_ORDER},
                generations={arm: arm_scans[(case.sample_id, arm)].generation for arm in ARM_ORDER},
            )
            for case in executed
        }
        records = [
            _record(
                scans[case.sample_id],
                case=case,
                plan=plans[case.sample_id],
                config=config,
            )
            for case in executed
        ]
        records.sort(key=lambda row: str(row["sample_id"]))
        summary = _summary(
            records,
            config=config,
            source=source,
            plan=plan_payload,
        )
        _write_jsonl_atomic(records_path, records)
        _write_json_atomic(summary_path, summary)
        outputs = {
            records_path.name: _output_metadata(records_path, len(records)),
            summary_path.name: _output_metadata(summary_path, 1),
        }
        completed_manifest = _manifest(
            config=config,
            source=source,
            plan=plan_payload,
            runtime=runtime_provenance,
            status="completed",
            started_at=started_at,
            completed_arms=len(records) * len(ARM_ORDER),
            completed_pairs=len(records),
            failures=(),
            checkpoints={},
            outputs=outputs,
        )
        _write_json_atomic(run_path, completed_manifest)
        completed_result = _validate_completed(
            completed_manifest,
            config=config,
            source=source,
            selected=selected,
            executed=executed,
            plan_payload=plan_payload,
        )
    except Exception as exc:
        for public_path in (records_path, summary_path):
            public_path.unlink(missing_ok=True)
        failures.append(
            {
                "sample_id": "__publication__",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        write_progress("failed")
        if isinstance(exc, WarningPromptRunError):
            raise
        raise WarningPromptRunError(f"typo-warning publication failed: {exc}") from exc
    shutil.rmtree(work_dir)
    _fsync_directory(output_dir)
    return completed_result


__all__ = [
    "WarningPromptConfig",
    "WarningPromptArmScan",
    "WarningPromptGeneration",
    "WarningPromptInputUse",
    "WarningPromptResult",
    "WarningPromptRunError",
    "WarningPromptRuntime",
    "WarningPromptScan",
    "run_typo_warning_prompt",
]
