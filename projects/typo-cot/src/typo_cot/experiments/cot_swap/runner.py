"""Strict, resumable runner for the final-paper complete-text CoT swap."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.evaluation.fallback import answers_equal, extract_with_fallback
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.cot_swap.planning import (
    CELL_ORDER,
    CELL_SIDES,
    CellPlan,
    CotSwapPlan,
    build_cell_plan,
)
from typo_cot.experiments.cot_swap.runtime import HuggingFaceCotSwapRuntime
from typo_cot.experiments.prepare_edited_pairs.runtime import (
    paper_dataset_samples_per_subset,
)

COT_SWAP_BENCHMARKS = ("gsm8k", "mmlu", "mmlu-pro", "arc", "csqa")
TARGETING_CONDITIONS = ("attribution-4", "random-4")
_SOURCE_DATASETS = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_ANSWER_EXTRACTION = "primary-then-empty-only-fallback-symmetric-a-b-c-d-cap-aware/v2"
_TEXT_INTERVENTION = {
    "boundary": "submitted-first-[Tt]he-answer-is-filter/v1",
    "assembly": "recorded-prompt-plus-decoded-pre-answer-text-retokenized/v1",
}
_GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 16,
    "use_cache": True,
    "return_dict_in_generate": False,
    "output_scores": False,
    "padding_side": "left",
}
_IMPLEMENTATION = "huggingface-cot-swap-four-cell-batch/v1"
_BATCHING = {
    "policy": "one-pair-four-cells/v1",
    "batch_size": 4,
    "cell_order": list(CELL_ORDER),
}
_ANSWER_SPAN_DECODING = {
    "source": "generated-token-ids-only/v1",
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": False,
}
_EDIT_VALIDITY = {
    "policy": "stored-prompts-differ-and-positive-target-attempts/v1",
    "requires_prompt_difference": True,
    "requires_positive_target_attempts": True,
    "zero_edit_restoration": "undefined-excluded-before-template-filter",
}
_ANSWER_EXTRACTION_DETAIL = {
    "primary_precedence": "task-extractor-preserve-nonempty/v1",
    "fallback_invocation": "empty-primary-only-symmetric-a-b-c-d/v1",
    "max_token_cap_gate": "disable-positional-numeric-n4-n5-only/v1",
    "regex_and_cap_gate_source": "legacy-backed-detail-not-specified-by-final-pdf",
}
_PROTOCOL = {
    "schema_version": "cot-swap-protocol/v1",
    "source": "completed-unlimited-prepare-edited-pairs-v1",
    "source_generation": {
        "seed": 42,
        "num_edits_requested": 4,
        "max_new_tokens": 512,
    },
    "edit_validity": dict(_EDIT_VALIDITY),
    "template_filter": {
        "policy": "submitted-first-[Tt]he-answer-is-filter/v1",
        "implementation_source": "legacy-backed-detail-not-specified-by-final-pdf",
        "requires_both_sides": True,
        "requires_one_trigger": True,
        "early_trigger_ratio_exclusive": 0.25,
        "rejects_residual_answer_fragment": True,
    },
    "cells": {cell: list(CELL_SIDES[cell]) for cell in CELL_ORDER},
    "implementation": _IMPLEMENTATION,
    "batching": dict(_BATCHING),
    "answer_span_decoding": dict(_ANSWER_SPAN_DECODING),
    "generation": dict(_GENERATION),
    "answer_extraction": _ANSWER_EXTRACTION,
    "answer_extraction_detail": dict(_ANSWER_EXTRACTION_DETAIL),
    "change_denominator": (
        "edit-valid-template-eligible-successfully-executed-regenerated-a-correct"
    ),
    "both_changed": "canonical-b-does-not-equal-canonical-a",
    "question_only_changed": "canonical-c-does-not-equal-canonical-a",
    "cot_only_changed": "canonical-d-does-not-equal-canonical-a",
    "restoration": "canonical-c-equals-canonical-a-conditioned-on-b-not-equal-a",
    "unextractable": "non-equality-failure-retained-in-applicable-denominator",
    "analysis": "descriptive-four-cell-counts-only",
    "historical_conflict": (
        "final-pdf-requires-symmetric-fallback-but-printed-headline-kept-legacy-a-correct"
    ),
}
_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        _PROTOCOL,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

_MAIN_MODELS = frozenset(
    {
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
)
_TASK_REFERENCES: dict[str, dict[str, object]] = {
    "gsm8k": {
        "b_changed_denominator": 630,
        "restoration_rate": 0.978,
        "question_only_change_rate": 0.008,
        "cot_only_change_rate": 0.195,
    },
    "mmlu": {
        "b_changed_denominator": 1716,
        "restoration_rate": 0.768,
        "question_only_change_rate": 0.079,
        "cot_only_change_rate": 0.211,
    },
    "mmlu-pro": {
        "b_changed_denominator": 531,
        "restoration_rate": 0.793,
        "question_only_change_rate": 0.084,
        "cot_only_change_rate": 0.278,
    },
    "arc": {
        "b_changed_denominator": 735,
        "restoration_rate": 0.676,
        "question_only_change_rate": 0.080,
        "cot_only_change_rate": 0.138,
    },
    "csqa": {
        "b_changed_denominator": 1022,
        "restoration_rate": 0.672,
        "question_only_change_rate": 0.108,
        "cot_only_change_rate": 0.188,
    },
}
_POOLED_REFERENCE = {
    "a_correct_denominator": 19_550,
    "b_changed": 4_634,
    "both_changed_rate": 0.237,
    "restored": 3_539,
    "restoration_rate": 0.764,
    "question_only_change_rate": 0.074,
    "cot_only_change_rate": 0.196,
}


class CotSwapRunError(RuntimeError):
    """Raised after a run records failures or detects published-state corruption."""


@dataclass(frozen=True, slots=True)
class CotSwapConfig:
    """Frozen public arguments for one model/benchmark/targeting setting."""

    model: str
    benchmark: Literal["gsm8k", "mmlu", "mmlu-pro", "arc", "csqa"]
    pairs: Path
    targeting: Literal["attribution-4", "random-4"]
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", Path(self.pairs))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in COT_SWAP_BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {self.benchmark!r}")
        if self.targeting not in TARGETING_CONDITIONS:
            raise ValueError(f"unsupported targeting condition: {self.targeting!r}")
        if not isinstance(self.gpu_id, str) or _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        if self.output_dir.resolve() == self.pairs.parent.resolve():
            raise ValueError("output_dir must not overwrite the source pair directory")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pairs"] = str(self.pairs.resolve())
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class CotSwapInputUse:
    """Runtime proof of the exact fixed text and its tokenized representation."""

    cell: str
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
        if self.cell not in CELL_ORDER:
            raise ValueError(f"unsupported input-use cell: {self.cell!r}")
        for name in (
            "prompt_text_sha256",
            "pre_answer_text_sha256",
            "full_input_text_sha256",
            "full_input_ids_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a full lowercase SHA-256")
        for name in (
            "prompt_char_count",
            "pre_answer_char_count",
            "full_input_char_count",
            "prompt_token_count",
            "full_input_token_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.prompt_prefix_token_stable, bool):
            raise TypeError("prompt_prefix_token_stable must be boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CotSwapGeneration:
    """One 16-token-capped answer-span generation and extraction result."""

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
        for name in ("text", "value", "method", "primary_method"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if not isinstance(self.is_extracted, bool) or not isinstance(self.is_correct, bool):
            raise TypeError("generation extraction and correctness flags must be boolean")
        if self.stop_reason not in {"eos_token", "max_new_tokens"}:
            raise ValueError("stop_reason must be 'eos_token' or 'max_new_tokens'")
        if self.stop_token_id is not None and (
            isinstance(self.stop_token_id, bool)
            or not isinstance(self.stop_token_id, int)
            or self.stop_token_id < 0
        ):
            raise ValueError("stop_token_id must be null or a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        return payload


@dataclass(frozen=True, slots=True)
class CotSwapScan:
    """All four results returned atomically for one source pair."""

    sample_id: str
    input_uses: Mapping[str, CotSwapInputUse]
    generations: Mapping[str, CotSwapGeneration]


class CotSwapRuntime(Protocol):
    def provenance(self) -> Mapping[str, object]: ...

    def scan_pair(self, pair: dict[str, object], plan: CotSwapPlan) -> CotSwapScan: ...


@dataclass(frozen=True, slots=True)
class CotSwapResult:
    records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    executed_pairs: int
    records: int


@dataclass(frozen=True, slots=True)
class _SourcePair:
    sample_id: str
    record: dict[str, object]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    run_path: Path
    pairs: tuple[_SourcePair, ...]
    pairs_sha256: str
    run_sha256: str
    manifest: dict[str, object]
    model_revision: str
    dataset_records_sha256: str
    dataset_sample_count: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _strict_loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {context}: {exc}") from exc


def _load_json(path: Path) -> dict[str, object]:
    payload = _strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


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
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    _write_text_atomic(path, text)


def _remove_exact_files(paths: Sequence[Path]) -> list[str]:
    """Remove only named publication files and return any cleanup errors."""

    cleanup_errors: list[str] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_errors.append(f"{path.name}: {exc}")
    return cleanup_errors


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _evaluation_benchmark(benchmark: str) -> str:
    if benchmark == "mmlu-pro":
        return "mmlu_pro"
    if benchmark == "csqa":
        return "commonsense_qa"
    return benchmark


def _validate_pair_record(
    record: dict[str, object],
    *,
    config: CotSwapConfig,
    path: Path,
    line_number: int,
) -> str:
    context = f"{path}:{line_number}"
    if record.get("schema_version") != "prepare-edited-pairs/v1":
        raise ValueError(f"{context}: unknown pair schema")
    sample_id = _nonempty_string(record.get("sample_id"), field=f"{context}.sample_id")
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", config.targeting),
    ):
        if record.get(field) != expected:
            raise ValueError(f"{context}: record {field} does not match the requested source")
    if record.get("seed") != 42:
        raise ValueError(f"{context}: record seed must be paper seed 42")
    if record.get("num_edits_requested") != 4:
        raise ValueError(f"{context}: record must request four edits")
    _nonempty_string(record.get("gold_answer"), field=f"{context}.gold_answer")

    validated_counts: dict[str, int] = {}
    for count_field, list_field in (
        ("num_target_attempts", "target_attempts"),
        ("num_aligned_words", "aligned_words"),
    ):
        count = record.get(count_field)
        values = record.get(list_field)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or not isinstance(values, list)
            or count != len(values)
        ):
            raise ValueError(f"{context}: {count_field} must match {list_field}")
        validated_counts[count_field] = count
    if validated_counts["num_target_attempts"] > 4:
        raise ValueError(f"{context}: record must contain at most four target attempts")
    if validated_counts["num_aligned_words"] > validated_counts["num_target_attempts"]:
        raise ValueError(f"{context}: aligned words cannot exceed target attempts")

    for side in ("clean", "edited"):
        payload = _mapping(record.get(side), field=f"{context}.{side}")
        _nonempty_string(payload.get("prompt"), field=f"{context}.{side}.prompt")
        _positive_int(
            payload.get("prompt_token_count"),
            field=f"{context}.{side}.prompt_token_count",
        )
        continuation = payload.get("continuation")
        if not isinstance(continuation, str):
            raise ValueError(f"{context}.{side}.continuation must be a string")
        continuation_token_count = _positive_int(
            payload.get("continuation_token_count"),
            field=f"{context}.{side}.continuation_token_count",
        )
        if continuation_token_count > 512:
            raise ValueError(f"{context}: record must contain at most 512 continuation tokens")
        answer = _mapping(payload.get("answer"), field=f"{context}.{side}.answer")
        if not isinstance(answer.get("is_correct"), bool):
            raise ValueError(f"{context}.{side}.answer.is_correct must be boolean")
    return sample_id


def _load_source(config: CotSwapConfig) -> _Source:
    path = config.pairs
    if not path.is_file():
        raise ValueError(f"pairs input is not a file: {path}")
    run_path = path.parent / "run.json"
    if not run_path.is_file():
        raise ValueError(f"pairs input is missing sibling run.json: {run_path}")
    manifest = _load_json(run_path)
    if manifest.get("schema_version") != "prepare-edited-pairs-run/v1":
        raise ValueError("source run has an unknown schema")
    if manifest.get("operation") != "prepare-edited-pairs":
        raise ValueError("source run has the wrong operation")
    if manifest.get("status") != "completed":
        raise ValueError("source run is not completed")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("source run paper SHA-256 does not match the final paper")
    arguments = _mapping(manifest.get("arguments"), field="source run arguments")
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", config.targeting),
    ):
        if arguments.get(field) != expected:
            raise ValueError(f"source run {field} does not match the requested setting")
    if arguments.get("seed") != 42:
        raise ValueError("source run must use paper seed 42")
    if arguments.get("num_edits") != 4:
        raise ValueError("source run must request four edits")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise ValueError("source pair preparation must not be limited")
    if arguments.get("max_new_tokens") != 512:
        raise ValueError("source run generation cap must be 512")

    decoding = _mapping(manifest.get("decoding"), field="source run decoding")
    expected_decoding: dict[str, object] = {
        "strategy": "greedy",
        "dtype": "bfloat16",
        "padding_side": "left",
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
    }
    for field, expected in expected_decoding.items():
        actual = decoding.get(field)
        if field not in decoding or actual != expected or type(actual) is not type(expected):
            raise ValueError(f"source run decoding.{field} must be {expected!r}")

    provenance = _mapping(manifest.get("provenance"), field="source run provenance")
    expected_source_provenance = {
        "model": (config.model, "model"),
        "benchmark_dataset_loader": (
            _SOURCE_DATASETS[config.benchmark],
            "benchmark dataset loader",
        ),
        "dataset_cohort_rule": (
            "paper-model-benchmark-cohort/v1",
            "dataset cohort rule",
        ),
        "random_seed_algorithm": ("sha256-first-64-bits/v1", "seed algorithm"),
        "generation_protocol": (
            "explicit-greedy-generation/v1",
            "generation protocol",
        ),
        "target_position": (
            "maximum-logit-after-first-cot-token",
            "target position",
        ),
        "alignment": ("actual-edited-word-final-token", "alignment protocol"),
    }
    for field, (expected, label) in expected_source_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"source run {label} must be {expected!r}")
    expected_subset_cap = paper_dataset_samples_per_subset(
        model=config.model,
        benchmark=config.benchmark,
    )
    if provenance.get("dataset_samples_per_subset") != expected_subset_cap:
        raise ValueError(f"source run samples-per-subset cap must be {expected_subset_cap!r}")
    model_revision = _nonempty_string(
        provenance.get("model_revision"), field="source run provenance.model_revision"
    )
    dataset_records_sha256 = _nonempty_string(
        provenance.get("dataset_records_sha256"),
        field="source run provenance.dataset_records_sha256",
    )
    if _SHA256.fullmatch(dataset_records_sha256) is None:
        raise ValueError(
            "source run provenance.dataset_records_sha256 must be a full lowercase SHA-256"
        )
    dataset_sample_count = _positive_int(
        provenance.get("dataset_sample_count"),
        field="source run provenance.dataset_sample_count",
    )
    counts = _mapping(manifest.get("counts"), field="source run counts")
    discovered = _positive_int(counts.get("discovered"), field="source run counts.discovered")
    written = _positive_int(counts.get("written"), field="source run counts.written")
    if counts.get("failed") != 0 or manifest.get("failures") not in ([], None):
        raise ValueError("source run contains failures")
    if discovered != written or written != dataset_sample_count:
        raise ValueError(
            "source run must contain the complete dataset: discovered, written, and "
            "dataset_sample_count must match"
        )

    source_pairs: list[_SourcePair] = []
    previous_id: str | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = _strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"pair record must be an object at {path}:{line_number}")
            sample_id = _validate_pair_record(
                payload,
                config=config,
                path=path,
                line_number=line_number,
            )
            if previous_id is not None and sample_id <= previous_id:
                raise ValueError("pair sample IDs must be strictly sorted and unique")
            previous_id = sample_id
            source_pairs.append(
                _SourcePair(
                    sample_id=sample_id,
                    record=payload,
                    fingerprint=hashlib.sha256(line.encode("utf-8")).hexdigest(),
                )
            )
    if not source_pairs:
        raise ValueError("pairs input contains no records")
    if len(source_pairs) != written:
        raise ValueError("source run written count does not match pairs.jsonl")
    return _Source(
        path=path,
        run_path=run_path,
        pairs=tuple(source_pairs),
        pairs_sha256=_sha256(path),
        run_sha256=_sha256(run_path),
        manifest=manifest,
        model_revision=model_revision,
        dataset_records_sha256=dataset_records_sha256,
        dataset_sample_count=dataset_sample_count,
    )


def _source_payload(source: _Source) -> dict[str, object]:
    return {
        "pairs": str(source.path.resolve()),
        "pairs_sha256": source.pairs_sha256,
        "source_run": str(source.run_path.resolve()),
        "source_run_sha256": source.run_sha256,
        "source_schema": "prepare-edited-pairs/v1",
        "model_revision": source.model_revision,
        "dataset_records_sha256": source.dataset_records_sha256,
        "dataset_sample_count": source.dataset_sample_count,
        "record_count": len(source.pairs),
    }


def _build_plans(
    source: _Source,
    config: CotSwapConfig,
) -> tuple[
    tuple[tuple[_SourcePair, CotSwapPlan], ...], tuple[tuple[_SourcePair, CotSwapPlan], ...]
]:
    all_plans = tuple((pair, build_cell_plan(pair.record)) for pair in source.pairs)
    eligible = tuple(item for item in all_plans if item[1].eligible)
    selected = eligible[: config.limit] if config.limit is not None else eligible
    return all_plans, selected


def _plan_rows(plans: Sequence[tuple[_SourcePair, CotSwapPlan]]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": pair.sample_id,
            "source_record_sha256": pair.fingerprint,
            "plan_sha256": plan.fingerprint,
            "edit_valid": plan.edit_valid,
            "template_eligible": plan.template_eligible,
            "eligible": plan.eligible,
            "exclusion_reasons": list(plan.exclusion_reasons),
        }
        for pair, plan in plans
    ]


def _plan_payload(
    all_plans: Sequence[tuple[_SourcePair, CotSwapPlan]],
    selected: Sequence[tuple[_SourcePair, CotSwapPlan]],
) -> dict[str, object]:
    eligible = [item for item in all_plans if item[1].eligible]
    return {
        "cell_order": list(CELL_ORDER),
        "source_records": len(all_plans),
        "eligible_pairs": len(eligible),
        "selected_pairs": len(selected),
        "full_fingerprint": _canonical_sha256(_plan_rows(all_plans)),
        "eligible_fingerprint": _canonical_sha256(_plan_rows(eligible)),
        "selected_fingerprint": _canonical_sha256(_plan_rows(selected)),
    }


def _comparability(config: CotSwapConfig) -> dict[str, object]:
    requirements = {
        "final_paper_protocol": True,
        "unlimited_complete_preparation_source": True,
        "four_requested_edits": True,
        "seed_42": True,
        "answer_span_cap_16": True,
        "symmetric_fallback_all_cells": True,
        "one_pair_four_cell_batch": True,
        "generated_token_ids_only_decode": True,
        "main_model": config.model in _MAIN_MODELS,
        "main_benchmark": config.benchmark in COT_SWAP_BENCHMARKS,
        "not_limit_truncated": config.limit is None,
        "historical_cohort_identity": False,
    }
    limitations = [
        "historical-sample-identities-and-source-hashes-are-not-published",
        "fresh-edit-validity-gate-can-differ-from-legacy-historical-cohort",
        "template-boundary-is-legacy-backed-and-not-fully-specified-by-final-pdf",
        "fallback-regex-and-cap-gate-are-legacy-backed-and-not-specified-by-final-pdf",
        "submitted-historical-batch-size-eight-public-uses-one-pair-four-cell-batch",
        "submitted-historical-full-output-character-slice-public-decodes-generated-token-ids",
        "printed-headline-used-asymmetric-legacy-a-correct-while-final-pdf-requires-symmetry",
        "appendix-c-table-9-model-scale-grid-is-a-separate-operation",
    ]
    if config.limit is not None:
        limitations.append("run-is-limit-truncated")
        status = "partial-smoke-run"
    elif config.model in _MAIN_MODELS:
        status = "fresh-paper-protocol-setting"
    else:
        limitations.append("model-is-outside-the-five-model-headline-grid")
        status = "fresh-protocol-setting"
    return {
        "status": status,
        "requirements": requirements,
        "limitations": limitations,
        "historical_cohort_identity": False,
    }


def _validate_runtime_provenance(
    provenance: Mapping[str, object],
    *,
    config: CotSwapConfig,
    source: _Source,
) -> dict[str, object]:
    required = {
        "operation": "cot-swap",
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
                f"runtime provenance does not match the public protocol: {field} must be "
                f"{expected!r}"
            )
    generation = _mapping(provenance.get("generation"), field="runtime generation")
    for field, expected in _GENERATION.items():
        actual = generation.get(field)
        if field not in generation or actual != expected or type(actual) is not type(expected):
            raise ValueError(
                f"runtime generation does not match the public protocol: {field} must be "
                f"{expected!r}"
            )
    if provenance.get("text_intervention") != _TEXT_INTERVENTION:
        raise ValueError("runtime text intervention does not match the public protocol")
    if provenance.get("batching") != _BATCHING:
        raise ValueError("runtime batching does not match the one-pair four-cell protocol")
    if provenance.get("answer_span_decoding") != _ANSWER_SPAN_DECODING:
        raise ValueError("runtime answer-span decoding does not match the public protocol")
    eos_token_ids = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos_token_ids, list)
        or not eos_token_ids
        or eos_token_ids != sorted(set(eos_token_ids))
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in eos_token_ids
        )
    ):
        raise ValueError(
            "runtime provenance effective_eos_token_ids must be sorted unique token IDs"
        )
    if provenance.get("effective_eos_token_ids_source") not in {
        "model-generation-config",
        "tokenizer-fallback",
    }:
        raise ValueError("runtime provenance EOS token source is unsupported")
    if generation.get("eos_token_id") != eos_token_ids:
        raise ValueError("runtime generation eos_token_id must equal the effective EOS IDs")
    return dict(provenance)


def _validate_generation(
    generation: CotSwapGeneration,
    *,
    field: str,
    benchmark: str,
    gold_answer: str,
    effective_eos_token_ids: Sequence[int],
) -> None:
    if not isinstance(generation, CotSwapGeneration):
        raise TypeError(f"{field} must be CotSwapGeneration")
    if not 1 <= len(generation.token_ids) <= 16:
        raise ValueError(f"{field}.token_ids must contain at most 16 answer-span tokens")
    eos_ids = set(effective_eos_token_ids)
    if generation.stop_reason == "eos_token":
        if (
            generation.stop_token_id not in eos_ids
            or generation.token_ids[-1] != generation.stop_token_id
            or any(token in eos_ids for token in generation.token_ids[:-1])
        ):
            raise ValueError(f"{field} EOS termination metadata does not match its token IDs")
    elif (
        len(generation.token_ids) != 16
        or generation.stop_token_id is not None
        or any(token in eos_ids for token in generation.token_ids)
    ):
        raise ValueError(f"{field} max-token termination metadata is inconsistent")
    if generation.is_extracted is not bool(generation.value):
        raise ValueError(f"{field}.is_extracted does not match its extracted value")
    expected_correct = answers_equal(
        generation.value,
        gold_answer,
        benchmark=_evaluation_benchmark(benchmark),
    )
    if generation.is_correct is not expected_correct:
        raise ValueError(f"{field}.is_correct does not match canonical gold equality")
    if not generation.method or not generation.primary_method:
        raise ValueError(f"{field} extraction methods must not be empty")
    expected_extraction = extract_with_fallback(
        generation.text,
        benchmark=_evaluation_benchmark(benchmark),
        correct_answer=gold_answer,
        allow_positional=generation.stop_reason == "eos_token",
    )
    actual_extraction = (
        generation.value,
        generation.is_extracted,
        generation.is_correct,
        generation.method,
        generation.primary_method,
    )
    if actual_extraction != (
        expected_extraction.value,
        expected_extraction.is_extracted,
        expected_extraction.is_correct,
        expected_extraction.method,
        expected_extraction.primary_method,
    ):
        raise ValueError(f"{field} answer extraction does not match generated text")


def _validate_input_use(input_use: CotSwapInputUse, cell: CellPlan) -> None:
    if not isinstance(input_use, CotSwapInputUse):
        raise TypeError(f"runtime input use for cell {cell.cell} must be CotSwapInputUse")
    expected = {
        "cell": cell.cell,
        "prompt_text_sha256": cell.prompt_sha256,
        "pre_answer_text_sha256": cell.pre_answer_sha256,
        "full_input_text_sha256": cell.full_input_sha256,
        "prompt_char_count": len(cell.prompt),
        "pre_answer_char_count": len(cell.pre_answer_text),
        "full_input_char_count": len(cell.full_input),
        "prompt_token_count": cell.prompt_token_count,
    }
    if any(getattr(input_use, field) != value for field, value in expected.items()):
        raise ValueError(f"runtime fixed input does not match the pure plan for cell {cell.cell}")
    if input_use.full_input_token_count <= 0:
        raise ValueError(f"runtime cell {cell.cell} full input contains no token")


def _validate_scan(
    scan: CotSwapScan,
    *,
    pair: _SourcePair,
    plan: CotSwapPlan,
    config: CotSwapConfig,
    effective_eos_token_ids: Sequence[int],
) -> None:
    if not isinstance(scan, CotSwapScan):
        raise TypeError("runtime must return CotSwapScan")
    if scan.sample_id != pair.sample_id:
        raise ValueError("runtime scan sample_id does not match its source pair")
    if tuple(scan.input_uses) != CELL_ORDER or tuple(scan.generations) != CELL_ORDER:
        raise ValueError("runtime scan must contain A, B, C, and D exactly in paper order")
    gold_answer = _nonempty_string(pair.record.get("gold_answer"), field="gold_answer")
    for cell in plan.cells:
        _validate_input_use(scan.input_uses[cell.cell], cell)
        _validate_generation(
            scan.generations[cell.cell],
            field=f"cell {cell.cell}",
            benchmark=config.benchmark,
            gold_answer=gold_answer,
            effective_eos_token_ids=effective_eos_token_ids,
        )


def _input_use_from_dict(value: object) -> CotSwapInputUse:
    row = _mapping(value, field="checkpoint input use")
    return CotSwapInputUse(**row)  # type: ignore[arg-type]


def _generation_from_dict(value: object) -> CotSwapGeneration:
    row = dict(_mapping(value, field="checkpoint generation"))
    token_ids = row.get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError("checkpoint generation token_ids must be a list")
    row["token_ids"] = tuple(token_ids)
    return CotSwapGeneration(**row)  # type: ignore[arg-type]


def _checkpoint_identity(config: CotSwapConfig, sample_id: str) -> str:
    return f"{config.targeting}\0{sample_id}"


def _checkpoint_path(directory: Path, config: CotSwapConfig, sample_id: str) -> Path:
    identity = _checkpoint_identity(config, sample_id).encode()
    return directory / f"{hashlib.sha256(identity).hexdigest()}.json"


def _checkpoint_payload(
    scan: CotSwapScan,
    *,
    pair: _SourcePair,
    plan: CotSwapPlan,
    config: CotSwapConfig,
    source: _Source,
    runtime_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "cot-swap-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": _PROTOCOL_SHA256,
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": config.targeting,
        "sample_id": pair.sample_id,
        "source_pairs_sha256": source.pairs_sha256,
        "source_run_sha256": source.run_sha256,
        "source_record_sha256": pair.fingerprint,
        "plan_sha256": plan.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "input_uses": {cell: scan.input_uses[cell].to_dict() for cell in CELL_ORDER},
        "generations": {cell: scan.generations[cell].to_dict() for cell in CELL_ORDER},
    }


def _scan_from_checkpoint(
    payload: Mapping[str, object],
    *,
    pair: _SourcePair,
    plan: CotSwapPlan,
    config: CotSwapConfig,
    source: _Source,
    runtime_fingerprint: str,
    effective_eos_token_ids: Sequence[int],
) -> CotSwapScan:
    expected = {
        "schema_version": "cot-swap-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": _PROTOCOL_SHA256,
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": config.targeting,
        "sample_id": pair.sample_id,
        "source_pairs_sha256": source.pairs_sha256,
        "source_run_sha256": source.run_sha256,
        "source_record_sha256": pair.fingerprint,
        "plan_sha256": plan.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ValueError(f"checkpoint provenance does not match pair {pair.sample_id}")
    raw_uses = _mapping(payload.get("input_uses"), field="checkpoint input_uses")
    raw_generations = _mapping(payload.get("generations"), field="checkpoint generations")
    if tuple(raw_uses) != CELL_ORDER or tuple(raw_generations) != CELL_ORDER:
        raise ValueError("checkpoint must contain all cells in paper order")
    scan = CotSwapScan(
        sample_id=pair.sample_id,
        input_uses={cell: _input_use_from_dict(raw_uses[cell]) for cell in CELL_ORDER},
        generations={cell: _generation_from_dict(raw_generations[cell]) for cell in CELL_ORDER},
    )
    _validate_scan(
        scan,
        pair=pair,
        plan=plan,
        config=config,
        effective_eos_token_ids=effective_eos_token_ids,
    )
    return scan


def _checkpoint_registry_entry(
    path: Path,
    *,
    pair: _SourcePair,
    config: CotSwapConfig,
) -> dict[str, object]:
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "targeting": config.targeting,
        "sample_id": pair.sample_id,
    }


def _load_registered_checkpoints(
    registry: Mapping[str, object],
    *,
    checkpoints_dir: Path,
    selected: Sequence[tuple[_SourcePair, CotSwapPlan]],
    config: CotSwapConfig,
    source: _Source,
    runtime_fingerprint: str,
    effective_eos_token_ids: Sequence[int],
) -> tuple[dict[str, CotSwapScan], dict[str, object]]:
    by_id = {pair.sample_id: (pair, plan) for pair, plan in selected}
    expected_files: set[str] = set()
    scans: dict[str, CotSwapScan] = {}
    normalized_registry: dict[str, object] = {}
    for identity, raw_metadata in registry.items():
        metadata = _mapping(raw_metadata, field=f"checkpoint registry {identity!r}")
        if set(metadata) != {"file", "sha256", "targeting", "sample_id"}:
            raise ValueError("checkpoint registry has an invalid shape")
        sample_id = metadata.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in by_id:
            raise ValueError("checkpoint registry references an unknown selected pair")
        if identity != _checkpoint_identity(config, sample_id):
            raise ValueError("checkpoint registry identity does not match targeting/sample_id")
        pair, plan = by_id[sample_id]
        expected_path = _checkpoint_path(checkpoints_dir, config, sample_id)
        expected_files.add(expected_path.name)
        if (
            metadata.get("file") != expected_path.name
            or metadata.get("targeting") != config.targeting
            or not expected_path.is_file()
            or metadata.get("sha256") != _sha256(expected_path)
        ):
            raise ValueError(f"checkpoint file or checkpoint hash mismatch for {sample_id}")
        scans[sample_id] = _scan_from_checkpoint(
            _load_json(expected_path),
            pair=pair,
            plan=plan,
            config=config,
            source=source,
            runtime_fingerprint=runtime_fingerprint,
            effective_eos_token_ids=effective_eos_token_ids,
        )
        normalized_registry[identity] = dict(metadata)

    # A crash can occur after the atomic checkpoint rename and before the next
    # run.json update. Adopt only files whose deterministic identity and full
    # semantic payload validate against this exact source, plan, and runtime.
    for pair, plan in selected:
        identity = _checkpoint_identity(config, pair.sample_id)
        if identity in normalized_registry:
            continue
        expected_path = _checkpoint_path(checkpoints_dir, config, pair.sample_id)
        if not expected_path.is_file():
            continue
        expected_files.add(expected_path.name)
        scans[pair.sample_id] = _scan_from_checkpoint(
            _load_json(expected_path),
            pair=pair,
            plan=plan,
            config=config,
            source=source,
            runtime_fingerprint=runtime_fingerprint,
            effective_eos_token_ids=effective_eos_token_ids,
        )
        normalized_registry[identity] = _checkpoint_registry_entry(
            expected_path,
            pair=pair,
            config=config,
        )
    actual_files = {path.name for path in checkpoints_dir.glob("*.json")}
    if actual_files != expected_files:
        raise ValueError("checkpoint directory contains unregistered or missing files")
    return scans, normalized_registry


def _events(scan: CotSwapScan, *, benchmark: str) -> dict[str, object]:
    extraction_benchmark = _evaluation_benchmark(benchmark)
    answers = {cell: scan.generations[cell].value for cell in CELL_ORDER}

    def same(cell: str) -> bool:
        return answers_equal(answers[cell], answers["A"], benchmark=extraction_benchmark)

    clean_correct = scan.generations["A"].is_correct
    both_changed = not same("B")
    question_changed = not same("C")
    cot_changed = not same("D")
    restoration_denominator = clean_correct and both_changed
    return {
        "clean_correct": clean_correct,
        "both_changed": both_changed,
        "question_only_changed": question_changed,
        "cot_only_changed": cot_changed,
        "restoration_denominator": restoration_denominator,
        "b_to_c_restored": same("C") if restoration_denominator else None,
    }


def _record(
    scan: CotSwapScan,
    *,
    pair: _SourcePair,
    plan: CotSwapPlan,
    config: CotSwapConfig,
    source: _Source,
) -> dict[str, object]:
    extraction_benchmark = _evaluation_benchmark(config.benchmark)
    answer_a = scan.generations["A"].value
    cells: dict[str, object] = {}
    for cell_plan in plan.cells:
        cell = cell_plan.cell
        cells[cell] = {
            "question_side": cell_plan.question_side,
            "cot_side": cell_plan.cot_side,
            "fixed_input": cell_plan.to_dict(include_text=False),
            "input_use": scan.input_uses[cell].to_dict(),
            "answer": scan.generations[cell].to_dict(),
            "equal_to_a": (
                True
                if cell == "A"
                else answers_equal(
                    scan.generations[cell].value,
                    answer_a,
                    benchmark=extraction_benchmark,
                )
            ),
        }
    return {
        "schema_version": "cot-swap-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "cot-swap",
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": config.targeting,
        "sample_id": pair.sample_id,
        "gold_answer": pair.record["gold_answer"],
        "source": {
            "pairs_sha256": source.pairs_sha256,
            "source_run_sha256": source.run_sha256,
            "source_record_sha256": pair.fingerprint,
        },
        "template_filter": {
            "policy": _PROTOCOL["template_filter"],
            "clean": plan.clean_boundary.to_dict(include_text=True),
            "edited": plan.edited_boundary.to_dict(include_text=True),
        },
        "cells": cells,
        "events": _events(scan, benchmark=config.benchmark),
    }


def _status_rows(
    all_plans: Sequence[tuple[_SourcePair, CotSwapPlan]],
    selected: Sequence[tuple[_SourcePair, CotSwapPlan]],
    scans: Mapping[str, CotSwapScan],
    *,
    config: CotSwapConfig,
) -> list[dict[str, object]]:
    selected_ids = {pair.sample_id for pair, _ in selected}
    rows: list[dict[str, object]] = []
    for pair, plan in all_plans:
        selected_for_execution = pair.sample_id in selected_ids
        scan = scans.get(pair.sample_id)
        events = _events(scan, benchmark=config.benchmark) if scan is not None else None
        if not plan.edit_valid:
            execution_status = "no-applied-edit"
        elif not plan.template_eligible:
            execution_status = "template-excluded"
        elif not selected_for_execution:
            execution_status = "not-selected"
        else:
            execution_status = "completed"
        rows.append(
            {
                "schema_version": "cot-swap-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "targeting": config.targeting,
                "sample_id": pair.sample_id,
                "source_record_sha256": pair.fingerprint,
                "plan_sha256": plan.fingerprint,
                "edit_valid": plan.edit_valid,
                "template_eligible": plan.template_eligible,
                "exclusion_reasons": list(plan.exclusion_reasons),
                "selected_for_execution": selected_for_execution,
                "execution_status": execution_status,
                "included_in_change_denominator": (
                    events["clean_correct"] if events is not None else False
                ),
                "included_in_restoration_denominator": (
                    events["restoration_denominator"] if events is not None else False
                ),
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _metric_payload(scans: Sequence[CotSwapScan], *, benchmark: str) -> dict[str, object]:
    events = [_events(scan, benchmark=benchmark) for scan in scans]
    included = [event for event in events if event["clean_correct"]]
    restoration = [event for event in included if event["restoration_denominator"]]
    return {
        "both_changed": _rate(
            sum(event["both_changed"] is True for event in included), len(included)
        ),
        "question_only_changed": _rate(
            sum(event["question_only_changed"] is True for event in included), len(included)
        ),
        "cot_only_changed": _rate(
            sum(event["cot_only_changed"] is True for event in included), len(included)
        ),
        "restoration": _rate(
            sum(event["b_to_c_restored"] is True for event in restoration),
            len(restoration),
        ),
    }


def _published_reference(config: CotSwapConfig) -> dict[str, object]:
    return {
        "scope": "task-pooled-attribution-4",
        "applies_to_targeting": config.targeting == "attribution-4",
        "task": config.benchmark,
        "task_reference": dict(_TASK_REFERENCES[config.benchmark]),
        "pooled_attribution_4": dict(_POOLED_REFERENCE),
        "historical_cohort_identity": False,
        "use_as_acceptance_target": False,
        "protocol_conflict": (
            "printed-headline-kept-legacy-a-correct; final-pdf-requires-symmetric-fallback"
        ),
    }


def _summary(
    all_plans: Sequence[tuple[_SourcePair, CotSwapPlan]],
    selected: Sequence[tuple[_SourcePair, CotSwapPlan]],
    scans: Mapping[str, CotSwapScan],
    *,
    config: CotSwapConfig,
    source: _Source,
) -> dict[str, object]:
    ordered_scans = [scans[pair.sample_id] for pair, _ in selected]
    events = [_events(scan, benchmark=config.benchmark) for scan in ordered_scans]
    no_edit = [plan for _, plan in all_plans if not plan.edit_valid]
    template_excluded = [
        plan for _, plan in all_plans if plan.edit_valid and not plan.template_eligible
    ]
    exclusion_counts = Counter(
        reason
        for plan in template_excluded
        for reason in plan.exclusion_reasons
        if reason != "no-applied-edit"
    )
    return {
        "schema_version": "cot-swap-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "cot-swap",
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": config.targeting,
        "source": _source_payload(source),
        "counts": {
            "source_records": len(all_plans),
            "no_applied_edit_records": len(no_edit),
            "template_excluded_records": len(template_excluded),
            "eligible_pairs": sum(plan.eligible for _, plan in all_plans),
            "selected_pairs": len(selected),
            "executed_pairs": len(ordered_scans),
            "failed_pairs": 0,
            "a_correct_pairs": sum(event["clean_correct"] is True for event in events),
        },
        "metrics": _metric_payload(ordered_scans, benchmark=config.benchmark),
        "unextractable_by_cell": {
            cell: sum(not scan.generations[cell].is_extracted for scan in ordered_scans)
            for cell in CELL_ORDER
        },
        "stop_reason_by_cell": {
            cell: {
                reason: sum(scan.generations[cell].stop_reason == reason for scan in ordered_scans)
                for reason in ("eos_token", "max_new_tokens")
            }
            for cell in CELL_ORDER
        },
        "extraction_method_by_cell_and_stop_reason": {
            cell: {
                reason: dict(
                    sorted(
                        Counter(
                            scan.generations[cell].method
                            for scan in ordered_scans
                            if scan.generations[cell].stop_reason == reason
                        ).items()
                    )
                )
                for reason in ("eos_token", "max_new_tokens")
            }
            for cell in CELL_ORDER
        },
        "template_filter": {
            "policy": _PROTOCOL["template_filter"],
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        },
        "edit_validity": {
            "policy": _EDIT_VALIDITY["policy"],
            "excluded_records": len(no_edit),
            "exclusion_reason_counts": {"no-applied-edit": len(no_edit)},
        },
        "answer_extraction": {
            "policy": _ANSWER_EXTRACTION,
            "implementation_detail": dict(_ANSWER_EXTRACTION_DETAIL),
            "symmetric_cells": list(CELL_ORDER),
            "unextractable_policy": _PROTOCOL["unextractable"],
        },
        "published_reference": _published_reference(config),
        "analysis_status": "descriptive-four-cell-counts-only",
        "comparability": _comparability(config),
    }


def _counts(
    all_plans: Sequence[tuple[_SourcePair, CotSwapPlan]],
    selected: Sequence[tuple[_SourcePair, CotSwapPlan]],
    checkpoints: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    selected_ids = {pair.sample_id for pair, _ in selected}
    return {
        "source_records": len(all_plans),
        "no_applied_edit_records": sum(not plan.edit_valid for _, plan in all_plans),
        "template_excluded_records": sum(
            plan.edit_valid and not plan.template_eligible for _, plan in all_plans
        ),
        "eligible_pairs": sum(plan.eligible for _, plan in all_plans),
        "selected_pairs": len(selected),
        "checkpointed_pairs": len(checkpoints),
        "failed_pairs": sum(failure.get("sample_id") in selected_ids for failure in failures),
        "publication_failures": sum(failure.get("stage") == "publication" for failure in failures),
    }


def _manifest(
    *,
    config: CotSwapConfig,
    source: _Source,
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
        "schema_version": "cot-swap-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "cot-swap",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "protocol_sha256": _PROTOCOL_SHA256,
        "source": _source_payload(source),
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
    config: CotSwapConfig,
    source: _Source,
    plan: Mapping[str, object],
) -> str:
    if previous.get("schema_version") != "cot-swap-run/v1":
        raise ValueError("cannot resume an unknown cot-swap run schema")
    if previous.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("resume paper SHA-256 does not match")
    if previous.get("operation") != "cot-swap":
        raise ValueError("resume operation does not match cot-swap")
    if previous.get("status") not in {"running", "failed", "completed"}:
        raise ValueError("resume status is not recognized")
    if previous.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run")
    if previous.get("protocol") != _PROTOCOL or previous.get("protocol_sha256") != _PROTOCOL_SHA256:
        raise ValueError("resume protocol fingerprint does not match")
    if previous.get("source") != _source_payload(source):
        raise ValueError("resume source input fingerprint does not match")
    if previous.get("plan") != plan:
        raise ValueError("resume deterministic plan fingerprint does not match")
    started_at = previous.get("started_at")
    return started_at if isinstance(started_at, str) else _now()


def _output_metadata(path: Path, records: int) -> dict[str, object]:
    return {"sha256": _sha256(path), "records": records}


def _validate_output_hashes(manifest: Mapping[str, object], output_dir: Path) -> None:
    outputs = _mapping(manifest.get("outputs"), field="completed outputs")
    expected = {
        "cot_swap_records.jsonl",
        "pair_status_records.jsonl",
        "cot_swap_summary.json",
    }
    if set(outputs) != expected:
        raise CotSwapRunError("completed output registry is incomplete")
    for name in expected:
        metadata = _mapping(outputs[name], field=f"completed output {name}")
        path = output_dir / name
        if not path.is_file() or metadata.get("sha256") != _sha256(path):
            raise CotSwapRunError(f"completed output hash mismatch: {name}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = _strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _validate_completed_checkpoint_registry(
    registry: Mapping[str, object],
    *,
    selected: Sequence[tuple[_SourcePair, CotSwapPlan]],
    config: CotSwapConfig,
    source: _Source,
    runtime_fingerprint: str,
    scans: Mapping[str, CotSwapScan],
) -> None:
    selected_by_id = {pair.sample_id: (pair, plan) for pair, plan in selected}
    expected_ids = set(selected_by_id)
    observed_ids: set[str] = set()
    for identity, raw_metadata in registry.items():
        metadata = _mapping(raw_metadata, field=f"completed checkpoint {identity!r}")
        if set(metadata) != {"file", "sha256", "targeting", "sample_id"}:
            raise ValueError("completed checkpoint registry has an invalid shape")
        sample_id = metadata.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise ValueError("completed checkpoint references an unknown selected pair")
        if identity != _checkpoint_identity(config, sample_id):
            raise ValueError("completed checkpoint registry identity does not match its pair")
        expected_file = _checkpoint_path(Path("."), config, sample_id).name
        if (
            metadata.get("file") != expected_file
            or metadata.get("targeting") != config.targeting
            or not isinstance(metadata.get("sha256"), str)
            or _SHA256.fullmatch(str(metadata["sha256"])) is None
        ):
            raise ValueError("completed checkpoint metadata does not match its pair")
        if sample_id in observed_ids:
            raise ValueError("completed checkpoint registry contains a duplicate pair")
        pair, plan = selected_by_id[sample_id]
        expected_payload = _checkpoint_payload(
            scans[sample_id],
            pair=pair,
            plan=plan,
            config=config,
            source=source,
            runtime_fingerprint=runtime_fingerprint,
        )
        expected_sha256 = hashlib.sha256(
            _serialized_json(expected_payload).encode("utf-8")
        ).hexdigest()
        if metadata.get("sha256") != expected_sha256:
            raise ValueError("completed checkpoint hash does not match reconstructed payload")
        observed_ids.add(sample_id)
    if observed_ids != expected_ids:
        raise ValueError("completed checkpoint registry does not cover every selected pair")


def _validate_completed_semantics(
    manifest: Mapping[str, object],
    *,
    config: CotSwapConfig,
    all_plans: Sequence[tuple[_SourcePair, CotSwapPlan]],
    selected: Sequence[tuple[_SourcePair, CotSwapPlan]],
    source: _Source,
) -> CotSwapResult:
    output_dir = config.output_dir
    records_path = output_dir / "cot_swap_records.jsonl"
    statuses_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "cot_swap_summary.json"
    run_path = output_dir / "run.json"
    try:
        _validate_output_hashes(manifest, output_dir)
        runtime_provenance = _mapping(manifest.get("runtime"), field="completed runtime")
        validated_runtime = _validate_runtime_provenance(
            runtime_provenance,
            config=config,
            source=source,
        )
        effective_eos_token_ids = tuple(validated_runtime["effective_eos_token_ids"])
        if manifest.get("comparability") != _comparability(config):
            raise ValueError("completed run comparability metadata is inconsistent")
        if manifest.get("failures") != []:
            raise ValueError("completed run must not retain failures")
        records = _load_jsonl(records_path)
        statuses = _load_jsonl(statuses_path)
        summary = _load_json(summary_path)
        expected_ids = [pair.sample_id for pair, _ in selected]
        if [row.get("sample_id") for row in records] != expected_ids:
            raise ValueError("completed records do not match selected sample order")
        if len(statuses) != len(all_plans):
            raise ValueError("completed status count does not match source population")
        if [row.get("sample_id") for row in statuses] != [pair.sample_id for pair, _ in all_plans]:
            raise ValueError("completed statuses do not match source sample order")
        plan_by_id = {pair.sample_id: (pair, plan) for pair, plan in selected}
        scans: dict[str, CotSwapScan] = {}
        for row in records:
            sample_id = str(row["sample_id"])
            pair, plan = plan_by_id[sample_id]
            if (
                row.get("schema_version") != "cot-swap-record/v1"
                or row.get("paper_sha256") != PAPER_SHA256
                or row.get("model") != config.model
                or row.get("benchmark") != config.benchmark
                or row.get("targeting") != config.targeting
            ):
                raise ValueError("completed record setting or schema mismatch")
            source_row = _mapping(row.get("source"), field="completed record source")
            if source_row.get("source_record_sha256") != pair.fingerprint:
                raise ValueError("completed record source fingerprint mismatch")
            cells = _mapping(row.get("cells"), field="completed record cells")
            if tuple(cells) != CELL_ORDER:
                raise ValueError("completed record is missing a CoT-swap cell")
            generations: dict[str, CotSwapGeneration] = {}
            uses: dict[str, CotSwapInputUse] = {}
            for cell_plan in plan.cells:
                cell_row = _mapping(cells[cell_plan.cell], field=f"cell {cell_plan.cell}")
                uses[cell_plan.cell] = _input_use_from_dict(cell_row.get("input_use"))
                generations[cell_plan.cell] = _generation_from_dict(cell_row.get("answer"))
            scan = CotSwapScan(sample_id=sample_id, input_uses=uses, generations=generations)
            _validate_scan(
                scan,
                pair=pair,
                plan=plan,
                config=config,
                effective_eos_token_ids=effective_eos_token_ids,
            )
            if row != _record(
                scan,
                pair=pair,
                plan=plan,
                config=config,
                source=source,
            ):
                raise ValueError(
                    "completed record semantics do not match its source, plan, and answers"
                )
            scans[sample_id] = scan

        expected_statuses = _status_rows(
            all_plans,
            selected,
            scans,
            config=config,
        )
        if statuses != expected_statuses:
            raise ValueError("completed pair statuses do not match records and plan")
        expected_summary = _summary(
            all_plans,
            selected,
            scans,
            config=config,
            source=source,
        )
        if summary != expected_summary:
            raise ValueError("completed summary does not match records and plan")

        checkpoint_registry = _mapping(
            manifest.get("checkpoints"),
            field="completed checkpoints",
        )
        _validate_completed_checkpoint_registry(
            checkpoint_registry,
            selected=selected,
            config=config,
            source=source,
            runtime_fingerprint=_canonical_sha256(validated_runtime),
            scans=scans,
        )
        if manifest.get("counts") != _counts(
            all_plans,
            selected,
            checkpoint_registry,
            [],
        ):
            raise ValueError("completed run counts are inconsistent")
        outputs = _mapping(manifest.get("outputs"), field="completed outputs")
        expected_output_metadata = {
            records_path.name: _output_metadata(records_path, len(records)),
            statuses_path.name: _output_metadata(statuses_path, len(statuses)),
            summary_path.name: _output_metadata(summary_path, 1),
        }
        if outputs != expected_output_metadata:
            raise ValueError("completed output registry metadata does not match public outputs")
    except (KeyError, TypeError, ValueError, CotSwapRunError) as exc:
        if isinstance(exc, CotSwapRunError) and "output hash" in str(exc):
            raise
        raise CotSwapRunError(f"completed run validation failed: {exc}") from exc
    return CotSwapResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        executed_pairs=len(records),
        records=len(records),
    )


def run_cot_swap(
    config: CotSwapConfig,
    *,
    runtime: CotSwapRuntime | None = None,
) -> CotSwapResult:
    """Validate, execute, checkpoint, and atomically publish one CoT-swap setting."""

    output_dir = config.output_dir
    run_path = output_dir / "run.json"
    records_path = output_dir / "cot_swap_records.jsonl"
    statuses_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "cot_swap_summary.json"
    public_paths = (records_path, statuses_path, summary_path)
    work_dir = output_dir / ".cot-swap-work"
    checkpoints_dir = work_dir / "checkpoints"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    source = _load_source(config)
    all_plans, selected = _build_plans(source, config)
    if not selected:
        raise ValueError("no template-eligible edited pairs were selected")
    plan_payload = _plan_payload(all_plans, selected)

    previous: dict[str, object] | None = None
    if config.resume:
        previous = _load_json(run_path)
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
                all_plans=all_plans,
                selected=selected,
                source=source,
            )
        stale_cleanup_errors = _remove_exact_files(public_paths)
        if stale_cleanup_errors:
            raise CotSwapRunError(
                "could not remove stale public outputs before resume: "
                + "; ".join(stale_cleanup_errors)
            )
    else:
        started_at = _now()

    if previous is None:
        if runtime is None:
            runtime = HuggingFaceCotSwapRuntime(config, revision=source.model_revision)
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
    runtime_fingerprint = _canonical_sha256(runtime_provenance)
    effective_eos_token_ids = tuple(runtime_provenance["effective_eos_token_ids"])

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
        effective_eos_token_ids=effective_eos_token_ids,
    )
    pending = tuple(item for item in selected if item[0].sample_id not in scans)
    if pending:
        if runtime is None:
            runtime = HuggingFaceCotSwapRuntime(config, revision=source.model_revision)
        active_runtime_provenance = _validate_runtime_provenance(
            runtime.provenance(),
            config=config,
            source=source,
        )
        if active_runtime_provenance != runtime_provenance:
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
                counts=_counts(all_plans, selected, checkpoint_registry, failures),
                failures=failures,
                checkpoints=checkpoint_registry,
                runtime=runtime_provenance,
            ),
        )

    write_progress("running")
    for pair, plan in tqdm(
        pending,
        desc="cot-swap",
        unit="pair",
        initial=len(scans),
        total=len(selected),
        disable=None,
    ):
        if pair.sample_id in scans:
            continue
        if runtime is None:
            raise AssertionError("pending CoT-swap work has no runtime")
        try:
            scan = runtime.scan_pair(pair.record, plan)
            _validate_scan(
                scan,
                pair=pair,
                plan=plan,
                config=config,
                effective_eos_token_ids=effective_eos_token_ids,
            )
            checkpoint_path = _checkpoint_path(checkpoints_dir, config, pair.sample_id)
            _write_json_atomic(
                checkpoint_path,
                _checkpoint_payload(
                    scan,
                    pair=pair,
                    plan=plan,
                    config=config,
                    source=source,
                    runtime_fingerprint=runtime_fingerprint,
                ),
            )
            identity = _checkpoint_identity(config, pair.sample_id)
            checkpoint_registry[identity] = _checkpoint_registry_entry(
                checkpoint_path,
                pair=pair,
                config=config,
            )
            scans[pair.sample_id] = scan
        except Exception as exc:
            failures.append(
                {
                    "stage": "pair",
                    "sample_id": pair.sample_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        write_progress("running")

    if failures:
        write_progress("failed")
        first = failures[0]["message"]
        raise CotSwapRunError(
            f"{len(failures)} pair(s) failed; first failure: {first}; "
            "inspect run.json and rerun with --resume"
        )
    if len(scans) != len(selected):
        write_progress("failed")
        raise CotSwapRunError("not every selected pair has a valid four-cell checkpoint")

    ordered_scans = {pair.sample_id: scans[pair.sample_id] for pair, _ in selected}
    records = [
        _record(
            ordered_scans[pair.sample_id],
            pair=pair,
            plan=plan,
            config=config,
            source=source,
        )
        for pair, plan in selected
    ]
    statuses = _status_rows(
        all_plans,
        selected,
        ordered_scans,
        config=config,
    )
    summary = _summary(
        all_plans,
        selected,
        ordered_scans,
        config=config,
        source=source,
    )

    try:
        stale_cleanup_errors = _remove_exact_files(public_paths)
        if stale_cleanup_errors:
            raise OSError(
                "could not remove stale public outputs: " + "; ".join(stale_cleanup_errors)
            )
        _write_jsonl_atomic(records_path, records)
        _write_jsonl_atomic(statuses_path, statuses)
        _write_json_atomic(summary_path, summary)
        outputs = {
            records_path.name: _output_metadata(records_path, len(records)),
            statuses_path.name: _output_metadata(statuses_path, len(statuses)),
            summary_path.name: _output_metadata(summary_path, 1),
        }
        completed_manifest = _manifest(
            config=config,
            source=source,
            plan=plan_payload,
            status="completed",
            started_at=started_at,
            counts=_counts(all_plans, selected, checkpoint_registry, []),
            failures=[],
            checkpoints=checkpoint_registry,
            runtime=runtime_provenance,
            outputs=outputs,
        )
        # This atomic manifest replacement is the commit point for the public
        # output set. Until it succeeds, every checkpoint remains resumable.
        _write_json_atomic(run_path, completed_manifest)
    except BaseException as exc:
        cleanup_errors = _remove_exact_files(public_paths)
        failure = {
            "stage": "publication",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        failures.append(failure)
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
        raise CotSwapRunError(
            f"publication failed: {type(exc).__name__}: {exc}{detail}; "
            "checkpoints were retained for --resume"
        ) from exc

    with suppress(OSError):
        shutil.rmtree(work_dir)
    return CotSwapResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        executed_pairs=len(selected),
        records=len(records),
    )
