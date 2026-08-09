"""CPU-only validation and Table 12 aggregation for input-corrector runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.input_corrector_audit.integrity import (
    analysis_code_identity,
    implementation_code_identity,
    validate_paper_runtime_environment,
)
from typo_cot.experiments.input_corrector_audit.protocol import (
    CORRECTOR_MODELS,
    CORE_SETTINGS,
    GENERATION,
    MATH_DIAGNOSTIC_SETTINGS,
    PAPER_BENCHMARK_ITEM_COUNTS,
    PUBLISHED_REFERENCE,
    PROTOCOL_SHA256,
    canonical_sha256,
)

_RECORDS_NAME = "corrector_records.jsonl"
_SETTING_SUMMARY_NAME = "corrector_audit_summary.json"
_SUMMARY_NAME = "input_corrector_summary.json"
_CSV_NAME = "table12_input_correctors.csv"
_MARKDOWN_NAME = "table12_input_correctors.md"
_LATEX_NAME = "table12_input_correctors.tex"
_METHOD_ORDER = (
    "pyspellchecker",
    "t5-large-spell",
    "qwen2.5-7b-instruct",
)
_EXPECTED_PRODUCER_CODE = implementation_code_identity()
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_SOURCE_FIELDS = frozenset(
    {
        "input_kind",
        "model",
        "benchmark",
        "model_revision",
        "records",
        "pairs_sha256",
        "run_sha256",
        "ordered_sample_ids_sha256",
        "dataset_records_sha256",
    }
)
_BENCHMARK_EXTRACTORS = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}
_GENERATION_RUNTIME = {
    "padding_side": "left",
    "max_new_tokens": GENERATION["max_new_tokens"],
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
_ANSWER_EXTRACTION = "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1"
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "paper_sha256",
        "operation",
        "model",
        "benchmark",
        "corrector",
        "sample_id",
        "source_record_sha256",
        "correction",
        "edited_words",
        "prompt_endpoints",
        "same_batch_answers",
        "separate_source_answers",
        "diagnostics",
    }
)
_GENERATION_FIELDS = frozenset(
    {
        "sample_id",
        "token_ids",
        "text",
        "extracted_answer",
        "is_extracted",
        "is_correct",
        "method",
        "primary_method",
    }
)


class InputCorrectorSummaryInputError(ValueError):
    """Raised when producer outputs cannot support the paper summary."""


@dataclass(frozen=True, slots=True)
class BuildInputCorrectorSummaryConfig:
    """Inputs for the complete core grid and optional MATH diagnostic."""

    runs_root: Path
    output_dir: Path
    math_runs_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs_root", Path(self.runs_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.math_runs_root is not None:
            object.__setattr__(self, "math_runs_root", Path(self.math_runs_root))


@dataclass(frozen=True, slots=True)
class BuildInputCorrectorSummaryResult:
    """Published Table 12 artifact paths."""

    summary_path: Path
    csv_path: Path
    markdown_path: Path
    latex_path: Path
    run_path: Path
    settings: int
    math_settings: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError) as exc:
        raise InputCorrectorSummaryInputError(f"invalid JSON at {context}: {exc}") from exc


def _load_json(path: Path) -> dict[str, object]:
    payload = _loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict):
        raise InputCorrectorSummaryInputError(f"expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise InputCorrectorSummaryInputError(f"empty JSONL line at {path}:{line_number}")
        payload = _loads(line, context=f"{path}:{line_number}")
        if not isinstance(payload, dict):
            raise InputCorrectorSummaryInputError(f"expected a JSON object at {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise InputCorrectorSummaryInputError(f"producer records are empty: {path}")
    return rows


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InputCorrectorSummaryInputError(f"{field} must be a JSON object")
    return value


def _integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputCorrectorSummaryInputError(f"{field} must be a non-negative integer")
    return value


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "denominator": denominator,
        "changed": numerator,
        "rate": numerator / denominator if denominator else None,
    }


def _validate_generation_evidence(
    value: object,
    *,
    context: str,
    sample_id: str,
) -> str:
    generation = _mapping(value, field=context)
    if set(generation) != _GENERATION_FIELDS:
        raise InputCorrectorSummaryInputError(f"{context} generation fields differ")
    if generation.get("sample_id") != sample_id:
        raise InputCorrectorSummaryInputError(f"{context} generation sample ID differs")
    token_ids = generation.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in token_ids
        )
    ):
        raise InputCorrectorSummaryInputError(f"{context} generation token IDs are invalid")
    for field in ("text", "extracted_answer"):
        if not isinstance(generation.get(field), str):
            raise InputCorrectorSummaryInputError(f"{context}.{field} must be a string")
    for field in ("is_extracted", "is_correct"):
        if type(generation.get(field)) is not bool:
            raise InputCorrectorSummaryInputError(f"{context}.{field} must be boolean")
    for field in ("method", "primary_method"):
        if not isinstance(generation.get(field), str) or not generation.get(field):
            raise InputCorrectorSummaryInputError(f"{context}.{field} must be non-empty")
    return str(generation["extracted_answer"])


def _validate_record_evidence(
    record: Mapping[str, object],
    *,
    context: str,
    benchmark: str,
) -> None:
    if set(record) != _RECORD_FIELDS:
        raise InputCorrectorSummaryInputError(f"{context} record evidence fields differ")
    _sha256(record.get("source_record_sha256"), field=f"{context}.source_record_sha256")

    correction = _mapping(record.get("correction"), field=f"{context}.correction")
    if set(correction) != {
        "corrected_text",
        "parse_failed",
        "n_calls",
        "raw_response",
        "input_sha256",
        "corrected_sha256",
    }:
        raise InputCorrectorSummaryInputError(f"{context} correction evidence fields differ")
    corrected_text = correction.get("corrected_text")
    if not isinstance(corrected_text, str):
        raise InputCorrectorSummaryInputError(f"{context} corrected_text must be a string")
    if type(correction.get("parse_failed")) is not bool:
        raise InputCorrectorSummaryInputError(f"{context} parse_failed must be boolean")
    n_calls = correction.get("n_calls")
    if not isinstance(n_calls, int) or isinstance(n_calls, bool) or n_calls <= 0:
        raise InputCorrectorSummaryInputError(f"{context} n_calls must be positive")
    if correction.get("raw_response") is not None and not isinstance(
        correction.get("raw_response"), str
    ):
        raise InputCorrectorSummaryInputError(f"{context} raw_response is invalid")
    _sha256(correction.get("input_sha256"), field=f"{context}.correction.input_sha256")
    if correction.get("corrected_sha256") != hashlib.sha256(
        corrected_text.encode("utf-8")
    ).hexdigest():
        raise InputCorrectorSummaryInputError(
            f"{context} correction corrected_sha256 differs from corrected_text"
        )

    words = _mapping(record.get("edited_words"), field=f"{context}.edited_words")
    if set(words) != {"restored", "total", "unalignable"}:
        raise InputCorrectorSummaryInputError(f"{context} edited-word evidence fields differ")
    restored = _integer(words.get("restored"), field=f"{context}.edited_words.restored")
    total = _integer(words.get("total"), field=f"{context}.edited_words.total")
    _integer(words.get("unalignable"), field=f"{context}.edited_words.unalignable")
    if restored > total:
        raise InputCorrectorSummaryInputError(f"{context} restores more words than its total")

    endpoints = _mapping(
        record.get("prompt_endpoints"), field=f"{context}.prompt_endpoints"
    )
    if set(endpoints) != {"clean_sha256", "corrected_sha256", "exact_utf8"}:
        raise InputCorrectorSummaryInputError(f"{context} prompt endpoint fields differ")
    clean_sha = _sha256(
        endpoints.get("clean_sha256"), field=f"{context}.prompt_endpoints.clean_sha256"
    )
    corrected_sha = _sha256(
        endpoints.get("corrected_sha256"),
        field=f"{context}.prompt_endpoints.corrected_sha256",
    )
    exact = clean_sha == corrected_sha
    if type(endpoints.get("exact_utf8")) is not bool or endpoints.get("exact_utf8") is not exact:
        raise InputCorrectorSummaryInputError(
            f"{context} exact prompt endpoint flag disagrees with its SHA-256 values"
        )

    same = record.get("same_batch_answers")
    separate = record.get("separate_source_answers")
    if benchmark == "math-500" or not exact:
        if same is not None or separate is not None:
            qualifier = "MATH diagnostic" if benchmark == "math-500" else "non-exact prompt"
            raise InputCorrectorSummaryInputError(
                f"{context} {qualifier} must not contain Same generation fields"
            )
    else:
        same_payload = _mapping(same, field=f"{context}.same_batch_answers")
        if set(same_payload) != {
            "protocol",
            "first_extracted_answer",
            "duplicate_extracted_answer",
            "first",
            "duplicate",
        } or same_payload.get("protocol") != "adjacent-duplicate-prompt-pair/v1":
            raise InputCorrectorSummaryInputError(f"{context} Same protocol/evidence differs")
        first_answer = same_payload.get("first_extracted_answer")
        duplicate_answer = same_payload.get("duplicate_extracted_answer")
        if not isinstance(first_answer, str) or not isinstance(duplicate_answer, str):
            raise InputCorrectorSummaryInputError(f"{context} Same answers must be strings")
        sample_id = str(record["sample_id"])
        if _validate_generation_evidence(
            same_payload.get("first"),
            context=f"{context}.same_batch_answers.first",
            sample_id=sample_id,
        ) != first_answer or _validate_generation_evidence(
            same_payload.get("duplicate"),
            context=f"{context}.same_batch_answers.duplicate",
            sample_id=sample_id,
        ) != duplicate_answer:
            raise InputCorrectorSummaryInputError(
                f"{context} nested Same generation answers disagree with top-level answers"
            )
        separate_payload = _mapping(
            separate, field=f"{context}.separate_source_answers"
        )
        if set(separate_payload) != {
            "comparison",
            "same_batch_corrected_extracted_answer",
            "source_pair_clean_extracted_answer",
        } or separate_payload.get("comparison") != (
            "same_batch_corrected_vs_source_pair_clean"
        ):
            raise InputCorrectorSummaryInputError(
                f"{context} separate-source comparison evidence differs"
            )
        current = separate_payload.get("same_batch_corrected_extracted_answer")
        source = separate_payload.get("source_pair_clean_extracted_answer")
        if current != duplicate_answer or not isinstance(source, str):
            raise InputCorrectorSummaryInputError(
                f"{context} separate-source answer disagrees with the duplicate Same arm"
            )

    diagnostics = _mapping(record.get("diagnostics"), field=f"{context}.diagnostics")
    if set(diagnostics) != {
        "whitespace_normalized_full",
        "all_perturbed_restored",
        "intact_word_changes",
        "collateral_changes",
    }:
        raise InputCorrectorSummaryInputError(f"{context} diagnostic evidence fields differ")
    for field in ("whitespace_normalized_full", "all_perturbed_restored"):
        if type(diagnostics.get(field)) is not bool:
            raise InputCorrectorSummaryInputError(f"{context}.diagnostics.{field} must be boolean")
    if diagnostics.get("all_perturbed_restored") is not (total > 0 and restored == total):
        raise InputCorrectorSummaryInputError(
            f"{context} all_perturbed_restored disagrees with edited-word counts"
        )
    collateral_count = _integer(
        diagnostics.get("intact_word_changes"),
        field=f"{context}.diagnostics.intact_word_changes",
    )
    collateral = diagnostics.get("collateral_changes")
    if not isinstance(collateral, list) or len(collateral) != collateral_count:
        raise InputCorrectorSummaryInputError(
            f"{context} collateral diagnostics disagree with intact_word_changes"
        )
    seen_positions: set[int] = set()
    for index, value in enumerate(collateral):
        change = _mapping(value, field=f"{context}.diagnostics.collateral_changes[{index}]")
        if set(change) != {"word_index", "clean", "corrected"}:
            raise InputCorrectorSummaryInputError(f"{context} collateral change fields differ")
        position = _integer(
            change.get("word_index"),
            field=f"{context}.diagnostics.collateral_changes[{index}].word_index",
        )
        if position in seen_positions:
            raise InputCorrectorSummaryInputError(f"{context} collateral word index is duplicated")
        seen_positions.add(position)
        if (
            not isinstance(change.get("clean"), str)
            or not isinstance(change.get("corrected"), str)
            or change.get("clean") == change.get("corrected")
        ):
            raise InputCorrectorSummaryInputError(f"{context} collateral word change is invalid")
    if exact and (
        diagnostics.get("whitespace_normalized_full") is not True or collateral_count != 0
    ):
        raise InputCorrectorSummaryInputError(
            f"{context} exact prompt disagrees with correction diagnostics"
        )


def _derive_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    identity: tuple[str, str, str],
) -> dict[str, object]:
    model, benchmark, corrector = identity
    seen: set[str] = set()
    word_restored = 0
    word_total = 0
    exact_clean = 0
    same_changed = 0
    separate_changed = 0
    intact_word_changes = 0
    for index, record in enumerate(records):
        context = f"record {index} for {identity}"
        _validate_record_evidence(record, context=context, benchmark=benchmark)
        if record.get("schema_version") != "input-corrector-audit-record/v1":
            raise InputCorrectorSummaryInputError(f"{context} has an unsupported schema")
        if record.get("paper_sha256") != PAPER_SHA256:
            raise InputCorrectorSummaryInputError(f"{context} paper fingerprint differs")
        if record.get("operation") != "input-corrector-audit":
            raise InputCorrectorSummaryInputError(f"{context} operation differs")
        if (
            record.get("model") != model
            or record.get("benchmark") != benchmark
            or record.get("corrector") != corrector
        ):
            raise InputCorrectorSummaryInputError(f"{context} setting identity differs")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise InputCorrectorSummaryInputError(f"{context} has a duplicate or invalid sample ID")
        seen.add(sample_id)

        words = _mapping(record.get("edited_words"), field=f"{context}.edited_words")
        restored = _integer(words.get("restored"), field=f"{context}.edited_words.restored")
        total = _integer(words.get("total"), field=f"{context}.edited_words.total")
        if restored > total:
            raise InputCorrectorSummaryInputError(f"{context} restores more words than its total")
        word_restored += restored
        word_total += total

        endpoints = _mapping(record.get("prompt_endpoints"), field=f"{context}.prompt_endpoints")
        clean_sha = endpoints.get("clean_sha256")
        corrected_sha = endpoints.get("corrected_sha256")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (clean_sha, corrected_sha)
        ):
            raise InputCorrectorSummaryInputError(f"{context} prompt SHA-256 is invalid")
        exact = clean_sha == corrected_sha
        exact_flag = endpoints.get("exact_utf8")
        if type(exact_flag) is not bool or exact_flag is not exact:
            raise InputCorrectorSummaryInputError(
                f"{context} exact prompt endpoint flag disagrees with its SHA-256 values"
            )
        same = record.get("same_batch_answers")
        separate = record.get("separate_source_answers")
        if benchmark == "math-500":
            if same is not None or separate is not None:
                raise InputCorrectorSummaryInputError(
                    f"{context} MATH diagnostic must not contain Same generation fields"
                )
        elif not exact and (same is not None or separate is not None):
            raise InputCorrectorSummaryInputError(
                f"{context} non-exact prompt must not contain Same generation fields"
            )
        if exact:
            exact_clean += 1
            if isinstance(same, Mapping):
                first = same.get("first_extracted_answer")
                duplicate = same.get("duplicate_extracted_answer")
                if not isinstance(first, str) or not isinstance(duplicate, str):
                    raise InputCorrectorSummaryInputError(f"{context} Same answers are invalid")
                same_changed += first != duplicate
            elif benchmark != "math-500":
                raise InputCorrectorSummaryInputError(f"{context} exact prompt lacks Same answers")
            if isinstance(separate, Mapping):
                current = separate.get("same_batch_corrected_extracted_answer")
                source = separate.get("source_pair_clean_extracted_answer")
                if not isinstance(current, str) or not isinstance(source, str):
                    raise InputCorrectorSummaryInputError(
                        f"{context} separate-source answers are invalid"
                    )
                separate_changed += current != source
            elif benchmark != "math-500":
                raise InputCorrectorSummaryInputError(
                    f"{context} exact prompt lacks separate-source answers"
                )

        diagnostics = _mapping(record.get("diagnostics"), field=f"{context}.diagnostics")
        intact_word_changes += _integer(
            diagnostics.get("intact_word_changes"),
            field=f"{context}.diagnostics.intact_word_changes",
        )

    if word_total <= 0:
        raise InputCorrectorSummaryInputError(f"setting {identity} has no aligned edited words")
    return {
        "records": len(records),
        "word_restored": word_restored,
        "word_total": word_total,
        "word_restoration_rate": word_restored / word_total,
        "exact_clean": exact_clean,
        "same_changed": same_changed,
        "separate_source_changed": separate_changed,
        "intact_word_changes": intact_word_changes,
    }


@dataclass(frozen=True, slots=True)
class _Setting:
    identity: tuple[str, str, str]
    records: tuple[dict[str, object], ...]
    metrics: dict[str, object]
    source_identity: dict[str, object]
    source_prompt_identity: tuple[tuple[str, str, str], ...]
    dictionary_sha256: str | None
    run_path: Path
    run_sha256: str
    records_sha256: str
    summary_sha256: str


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise InputCorrectorSummaryInputError(f"{field} must be a lowercase SHA-256")
    return value


def _revision(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise InputCorrectorSummaryInputError(f"{field} must be a pinned commit revision")
    return value


def _validate_source_identity(
    value: object,
    *,
    run_path: Path,
    model: str,
    benchmark: str,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    source = _mapping(value, field=f"{run_path}.source")
    if set(source) != _SOURCE_FIELDS:
        raise InputCorrectorSummaryInputError(
            f"producer source identity fields differ: {run_path}"
        )
    if source.get("input_kind") != "completed-prepare-edited-pairs/v1":
        raise InputCorrectorSummaryInputError(f"producer source kind differs: {run_path}")
    if source.get("model") != model or source.get("benchmark") != benchmark:
        raise InputCorrectorSummaryInputError(f"producer source setting differs: {run_path}")
    if type(source.get("records")) is not int or source.get("records") != len(records):
        raise InputCorrectorSummaryInputError(f"producer source record count differs: {run_path}")
    expected_records = PAPER_BENCHMARK_ITEM_COUNTS.get(benchmark)
    if source.get("records") != expected_records:
        raise InputCorrectorSummaryInputError(
            f"producer source does not contain the paper {benchmark} cohort: {run_path}"
        )
    _revision(source.get("model_revision"), field=f"{run_path}.source.model_revision")
    for name in (
        "pairs_sha256",
        "run_sha256",
        "ordered_sample_ids_sha256",
        "dataset_records_sha256",
    ):
        _sha256(source.get(name), field=f"{run_path}.source.{name}")
    sample_ids = [str(record["sample_id"]) for record in records]
    if source.get("ordered_sample_ids_sha256") != canonical_sha256(sample_ids):
        raise InputCorrectorSummaryInputError(
            f"producer source sample cohort differs from output records: {run_path}"
        )
    return dict(source)


def _validate_runtime_environment(
    provenance: Mapping[str, object],
    *,
    profile: str,
    field: str,
) -> None:
    try:
        validate_paper_runtime_environment(
            provenance,
            profile=profile,
            field_prefix=field,
        )
    except ValueError as exc:
        raise InputCorrectorSummaryInputError(str(exc)) from exc


def _validate_correction_runtime(
    value: object,
    *,
    run_path: Path,
    corrector: str,
) -> None:
    field = f"{run_path}.runtime.correction"
    provenance = _mapping(value, field=field)
    if (
        provenance.get("operation") != "input-corrector-audit"
        or provenance.get("runtime") != "ProductionCorrectionRuntime"
        or provenance.get("corrector") != corrector
        or provenance.get("protocol_sha256") != PROTOCOL_SHA256
        or provenance.get("implementation_code") != _EXPECTED_PRODUCER_CODE
    ):
        raise InputCorrectorSummaryInputError(
            f"producer correction runtime/protocol identity differs: {run_path}"
        )
    specification = CORRECTOR_MODELS.get(corrector)
    if not isinstance(specification, Mapping):
        raise InputCorrectorSummaryInputError(
            f"producer correction runtime has an unknown corrector: {run_path}"
        )
    expected_revision = specification["revision"]
    if provenance.get("requested_revision") != expected_revision:
        raise InputCorrectorSummaryInputError(
            f"producer correction revision pin differs: {run_path}"
        )
    if corrector == "pyspellchecker":
        if provenance.get("profile") != "pyspellchecker" or any(
            provenance.get(name) is not None
            for name in ("model_revision", "tokenizer_revision")
        ):
            raise InputCorrectorSummaryInputError(
                f"producer dictionary correction provenance differs: {run_path}"
            )
        _validate_runtime_environment(
            provenance,
            profile="pyspellchecker",
            field=f"{field} environment",
        )
        return
    if provenance.get("profile") != "neural":
        raise InputCorrectorSummaryInputError(
            f"producer neural correction profile differs: {run_path}"
        )
    expected = _revision(expected_revision, field=f"{field}.requested_revision")
    for name in ("model_revision", "tokenizer_revision"):
        if provenance.get(name) != expected:
            raise InputCorrectorSummaryInputError(
                f"producer correction {name} differs from its public pin: {run_path}"
            )
    if provenance.get("model_revision_source") not in {
        "model-config-metadata",
        "explicit-load-revision",
    } or provenance.get("tokenizer_revision_source") not in {
        "tokenizer-init-metadata",
        "explicit-load-revision",
    }:
        raise InputCorrectorSummaryInputError(
            f"producer correction revision source differs: {run_path}"
        )
    _validate_runtime_environment(
        provenance,
        profile="neural",
        field=f"{field} environment",
    )


def _validate_generation_runtime(
    value: object,
    *,
    run_path: Path,
    model: str,
    benchmark: str,
    source_revision: str,
) -> None:
    field = f"{run_path}.runtime.generation"
    provenance = _mapping(value, field=field)
    if (
        provenance.get("operation") != "input-corrector-audit"
        or provenance.get("runtime") != "HuggingFaceSamePromptRuntime"
        or provenance.get("profile") != "neural"
        or provenance.get("model") != model
        or provenance.get("protocol_sha256") != PROTOCOL_SHA256
        or provenance.get("implementation_code") != _EXPECTED_PRODUCER_CODE
        or provenance.get("dtype") != "bfloat16"
    ):
        raise InputCorrectorSummaryInputError(
            f"producer Same generation runtime/protocol identity differs: {run_path}"
        )
    if any(
        provenance.get(name) != source_revision
        for name in ("requested_revision", "model_revision", "tokenizer_revision")
    ):
        raise InputCorrectorSummaryInputError(
            f"producer evaluator revision differs from the source revision: {run_path}"
        )
    if provenance.get("model_revision_source") not in {
        "model-config-metadata",
        "explicit-load-revision",
    } or provenance.get("tokenizer_revision_source") not in {
        "tokenizer-init-metadata",
        "explicit-load-revision",
    }:
        raise InputCorrectorSummaryInputError(
            f"producer evaluator revision source differs: {run_path}"
        )
    generation = _mapping(provenance.get("generation"), field=f"{field}.generation")
    if dict(generation) != _GENERATION_RUNTIME:
        raise InputCorrectorSummaryInputError(
            f"producer Same generation parameters differ: {run_path}"
        )
    eos_ids = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos_ids)
        or len(eos_ids) != len(set(eos_ids))
        or provenance.get("effective_eos_token_ids_source")
        not in {"model-generation-config", "tokenizer-fallback"}
    ):
        raise InputCorrectorSummaryInputError(
            f"producer evaluator EOS provenance differs: {run_path}"
        )
    if (
        provenance.get("answer_extraction") != _ANSWER_EXTRACTION
        or provenance.get("benchmark_extractor") != _BENCHMARK_EXTRACTORS.get(benchmark)
    ):
        raise InputCorrectorSummaryInputError(
            f"producer evaluator answer-extraction protocol differs: {run_path}"
        )
    _validate_runtime_environment(
        provenance,
        profile="neural",
        field=f"{field} environment",
    )


def _validate_runtime(
    value: object,
    *,
    run_path: Path,
    identity: tuple[str, str, str],
    source: Mapping[str, object],
    exact_clean: int,
    gpu_id: str,
) -> str | None:
    model, benchmark, corrector = identity
    runtime = _mapping(value, field=f"{run_path}.runtime")
    expected_fields = {"correction"}
    needs_generation = benchmark != "math-500" and exact_clean > 0
    if needs_generation:
        expected_fields.add("generation")
    if set(runtime) != expected_fields:
        qualifier = "MATH diagnostic" if benchmark == "math-500" else "producer"
        raise InputCorrectorSummaryInputError(
            f"{qualifier} runtime provenance fields differ: {run_path}"
        )
    _validate_correction_runtime(
        runtime.get("correction"),
        run_path=run_path,
        corrector=corrector,
    )
    correction = _mapping(runtime["correction"], field=f"{run_path}.runtime.correction")
    if correction.get("profile") == "neural" and correction.get(
        "cuda_visible_devices"
    ) != gpu_id:
        raise InputCorrectorSummaryInputError(
            f"producer correction visible GPU differs from its argument: {run_path}"
        )
    if needs_generation:
        _validate_generation_runtime(
            runtime.get("generation"),
            run_path=run_path,
            model=model,
            benchmark=benchmark,
            source_revision=str(source["model_revision"]),
        )
        generation = _mapping(runtime["generation"], field=f"{run_path}.runtime.generation")
        if generation.get("cuda_visible_devices") != gpu_id:
            raise InputCorrectorSummaryInputError(
                f"producer evaluator visible GPU differs from its argument: {run_path}"
            )
    dictionary_sha256 = correction.get("dictionary_sha256")
    return str(dictionary_sha256) if corrector == "pyspellchecker" else None


def _validate_output(
    manifest: Mapping[str, object],
    *,
    run_path: Path,
    name: str,
    expected_records: int | None = None,
) -> Path:
    outputs = _mapping(manifest.get("outputs"), field=f"{run_path}.outputs")
    metadata = _mapping(outputs.get(name), field=f"{run_path}.outputs.{name}")
    path = run_path.parent / name
    if not path.is_file():
        raise InputCorrectorSummaryInputError(f"producer output is missing: {path}")
    expected_sha = metadata.get("sha256")
    expected_bytes = metadata.get("bytes")
    if _sha256_file(path) != expected_sha or path.stat().st_size != expected_bytes:
        raise InputCorrectorSummaryInputError(f"producer output hash/size mismatch: {path}")
    if expected_records is not None and metadata.get("records") != expected_records:
        raise InputCorrectorSummaryInputError(f"producer output record count differs: {path}")
    return path


def _load_setting(run_path: Path) -> _Setting:
    manifest = _load_json(run_path)
    if manifest.get("schema_version") != "input-corrector-audit-run/v1":
        raise InputCorrectorSummaryInputError(f"unsupported producer run schema: {run_path}")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise InputCorrectorSummaryInputError(f"producer paper fingerprint differs: {run_path}")
    if manifest.get("operation") != "input-corrector-audit":
        raise InputCorrectorSummaryInputError(f"unexpected producer operation: {run_path}")
    if manifest.get("status") != "completed":
        raise InputCorrectorSummaryInputError(f"producer run is not completed: {run_path}")
    if manifest.get("scope") != "paper-setting":
        raise InputCorrectorSummaryInputError(
            f"producer scope must be paper-setting: {run_path}"
        )
    if manifest.get("failure") is not None:
        raise InputCorrectorSummaryInputError(
            f"completed producer run retains a failure: {run_path}"
        )
    checkpoints = _mapping(manifest.get("checkpoints"), field=f"{run_path}.checkpoints")
    if checkpoints:
        raise InputCorrectorSummaryInputError(
            f"completed producer run retains checkpoint inventory: {run_path}"
        )
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise InputCorrectorSummaryInputError(f"producer protocol SHA-256 differs: {run_path}")
    if manifest.get("implementation_code") != _EXPECTED_PRODUCER_CODE:
        raise InputCorrectorSummaryInputError(
            f"producer implementation code identity differs: {run_path}"
        )
    arguments = _mapping(manifest.get("arguments"), field=f"{run_path}.arguments")
    if set(arguments) != {
        "corrector",
        "model",
        "benchmark",
        "pairs",
        "gpu_id",
        "limit",
        "output_dir",
    }:
        raise InputCorrectorSummaryInputError(
            f"producer argument fields differ: {run_path}"
        )
    model = arguments.get("model")
    benchmark = arguments.get("benchmark")
    corrector = arguments.get("corrector")
    if not all(isinstance(value, str) and value for value in (model, benchmark, corrector)):
        raise InputCorrectorSummaryInputError(f"producer setting identity is invalid: {run_path}")
    if arguments.get("limit") is not None:
        raise InputCorrectorSummaryInputError(
            f"limited producer run is not paper-grid input: {run_path}"
        )
    pairs = arguments.get("pairs")
    output_dir = arguments.get("output_dir")
    gpu_id = arguments.get("gpu_id")
    if not isinstance(pairs, str) or not Path(pairs).is_absolute():
        raise InputCorrectorSummaryInputError(
            f"producer pairs argument must be an absolute path: {run_path}"
        )
    if not isinstance(output_dir, str) or not Path(output_dir).is_absolute():
        raise InputCorrectorSummaryInputError(
            f"producer output directory argument must be absolute: {run_path}"
        )
    if not isinstance(gpu_id, str) or _GPU_ID.fullmatch(gpu_id) is None:
        raise InputCorrectorSummaryInputError(
            f"producer GPU argument is invalid: {run_path}"
        )
    identity = (str(model), str(benchmark), str(corrector))

    outputs = _mapping(manifest.get("outputs"), field=f"{run_path}.outputs")
    if set(outputs) != {_RECORDS_NAME, _SETTING_SUMMARY_NAME}:
        raise InputCorrectorSummaryInputError(
            f"producer output inventory differs: {run_path}"
        )
    records_path = _validate_output(manifest, run_path=run_path, name=_RECORDS_NAME)
    records = _load_jsonl(records_path)
    records_metadata = _mapping(outputs[_RECORDS_NAME], field="records metadata")
    if records_metadata.get("records") != len(records):
        raise InputCorrectorSummaryInputError(f"producer record count differs: {records_path}")
    metrics = _derive_metrics(records, identity=identity)
    if type(manifest.get("selected_records")) is not int or manifest.get(
        "selected_records"
    ) != len(records):
        raise InputCorrectorSummaryInputError(
            f"producer selected_records differs from its outputs: {run_path}"
        )
    source_identity = _validate_source_identity(
        manifest.get("source"),
        run_path=run_path,
        model=str(model),
        benchmark=str(benchmark),
        records=records,
    )
    dictionary_sha256 = _validate_runtime(
        manifest.get("runtime"),
        run_path=run_path,
        identity=identity,
        source=source_identity,
        exact_clean=int(metrics["exact_clean"]),
        gpu_id=str(gpu_id),
    )

    summary_path = _validate_output(manifest, run_path=run_path, name=_SETTING_SUMMARY_NAME)
    summary = _load_json(summary_path)
    if summary.get("schema_version") != "input-corrector-audit-summary/v1":
        raise InputCorrectorSummaryInputError(f"producer summary schema differs: {summary_path}")
    if (
        summary.get("paper_sha256") != PAPER_SHA256
        or summary.get("operation") != "input-corrector-audit"
        or summary.get("status") != "completed"
        or summary.get("scope") != "paper-setting"
        or summary.get("model") != model
        or summary.get("benchmark") != benchmark
        or summary.get("corrector") != corrector
    ):
        raise InputCorrectorSummaryInputError(f"producer summary identity differs: {summary_path}")
    if summary.get("metrics") != metrics:
        raise InputCorrectorSummaryInputError(
            f"producer summary metrics do not match records: {summary_path}"
        )
    return _Setting(
        identity=identity,
        records=tuple(records),
        metrics=metrics,
        source_identity=source_identity,
        source_prompt_identity=tuple(
            (
                str(record["sample_id"]),
                str(record["source_record_sha256"]),
                str(_mapping(record["prompt_endpoints"], field="prompt_endpoints")["clean_sha256"]),
            )
            for record in records
        ),
        dictionary_sha256=dictionary_sha256,
        run_path=run_path,
        run_sha256=_sha256_file(run_path),
        records_sha256=_sha256_file(records_path),
        summary_sha256=_sha256_file(summary_path),
    )


def _discover(root: Path, *, role: str) -> list[_Setting]:
    path = root.resolve()
    if not path.is_dir():
        raise InputCorrectorSummaryInputError(f"{role} runs root is not a directory: {path}")
    manifests = sorted(path.rglob("run.json"))
    if not manifests:
        raise InputCorrectorSummaryInputError(f"{role} runs root contains no run.json files")
    return [_load_setting(run_path) for run_path in manifests]


def _index_grid(
    settings: Sequence[_Setting],
    *,
    expected: set[tuple[str, str, str]],
    role: str,
) -> dict[tuple[str, str, str], _Setting]:
    indexed: dict[tuple[str, str, str], _Setting] = {}
    duplicates: list[tuple[str, str, str]] = []
    for setting in settings:
        if setting.identity in indexed:
            duplicates.append(setting.identity)
        indexed[setting.identity] = setting
    present = set(indexed)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if duplicates or missing or unexpected:
        raise InputCorrectorSummaryInputError(
            f"{role} grid mismatch: missing={missing}, unexpected={unexpected}, "
            f"duplicates={sorted(set(duplicates))}"
        )
    return indexed


def _validate_shared_source_cohorts(
    indexed: Mapping[tuple[str, str, str], _Setting],
    *,
    role: str,
) -> None:
    grouped: dict[tuple[str, str], list[_Setting]] = {}
    for (model, benchmark, _corrector), setting in indexed.items():
        grouped.setdefault((model, benchmark), []).append(setting)
    for identity, settings in grouped.items():
        baseline = settings[0]
        for setting in settings[1:]:
            if (
                setting.source_identity != baseline.source_identity
                or setting.source_prompt_identity != baseline.source_prompt_identity
            ):
                raise InputCorrectorSummaryInputError(
                    f"{role} correctors do not share one source cohort for {identity}"
                )

    revisions_by_model: dict[str, set[str]] = {}
    for (model, _benchmark, _corrector), setting in indexed.items():
        revisions_by_model.setdefault(model, set()).add(
            str(setting.source_identity["model_revision"])
        )
    inconsistent_models = sorted(
        model for model, revisions in revisions_by_model.items() if len(revisions) != 1
    )
    if inconsistent_models:
        raise InputCorrectorSummaryInputError(
            f"{role} source model revision differs across benchmarks: {inconsistent_models}"
        )

    datasets_by_benchmark: dict[str, set[tuple[str, str]]] = {}
    for (_model, benchmark, _corrector), setting in indexed.items():
        datasets_by_benchmark.setdefault(benchmark, set()).add(
            (
                str(setting.source_identity["dataset_records_sha256"]),
                str(setting.source_identity["ordered_sample_ids_sha256"]),
            )
        )
    inconsistent_benchmarks = sorted(
        benchmark
        for benchmark, identities in datasets_by_benchmark.items()
        if len(identities) != 1
    )
    if inconsistent_benchmarks:
        raise InputCorrectorSummaryInputError(
            f"{role} dataset/sample cohort differs across models: {inconsistent_benchmarks}"
        )

    dictionary_hashes = {
        setting.dictionary_sha256
        for setting in indexed.values()
        if setting.dictionary_sha256 is not None
    }
    if len(dictionary_hashes) > 1:
        raise InputCorrectorSummaryInputError(
            f"{role} pyspellchecker dictionary identity differs across settings"
        )


def _validate_core_math_revisions(
    core: Mapping[tuple[str, str, str], _Setting],
    math: Mapping[tuple[str, str, str], _Setting],
) -> None:
    core_by_model = {
        model: str(setting.source_identity["model_revision"])
        for (model, _benchmark, _corrector), setting in core.items()
    }
    math_by_model = {
        model: str(setting.source_identity["model_revision"])
        for (model, _benchmark, _corrector), setting in math.items()
    }
    mismatched = sorted(
        model for model in core_by_model if core_by_model[model] != math_by_model.get(model)
    )
    if mismatched:
        raise InputCorrectorSummaryInputError(
            f"MATH source model revision differs from core: {mismatched}"
        )


def _method_summary(
    indexed: Mapping[tuple[str, str, str], _Setting],
    method: str,
) -> dict[str, object]:
    settings = sorted(
        (setting for identity, setting in indexed.items() if identity[2] == method),
        key=lambda setting: setting.identity,
    )
    rates = [float(setting.metrics["word_restoration_rate"]) for setting in settings]
    word_restored = sum(int(setting.metrics["word_restored"]) for setting in settings)
    word_total = sum(int(setting.metrics["word_total"]) for setting in settings)
    exact = sum(int(setting.metrics["exact_clean"]) for setting in settings)
    same = sum(int(setting.metrics["same_changed"]) for setting in settings)
    separate = sum(int(setting.metrics["separate_source_changed"]) for setting in settings)
    return {
        "settings": len(settings),
        "items": sum(int(setting.metrics["records"]) for setting in settings),
        "word": {
            "setting_mean_exact_restoration": sum(rates) / len(rates),
            "pooled_exact_restoration": word_restored / word_total,
            "restored": word_restored,
            "denominator": word_total,
        },
        "exact_clean": exact,
        "same": _rate(same, exact),
        "separate_source": _rate(separate, exact),
    }


def _math_summary(indexed: Mapping[tuple[str, str, str], _Setting]) -> dict[str, object]:
    methods: dict[str, object] = {}
    for method in ("t5-large-spell", "qwen2.5-7b-instruct"):
        settings = sorted(
            (setting for identity, setting in indexed.items() if identity[2] == method),
            key=lambda setting: setting.identity,
        )
        items = sum(int(setting.metrics["records"]) for setting in settings)
        changes = sum(int(setting.metrics["intact_word_changes"]) for setting in settings)
        methods[method] = {
            "settings": len(settings),
            "items": items,
            "intact_word_changes": changes,
            "changes_per_item": changes / items,
        }
    return {
        "status": "complete-diagnostic-grid",
        "expected_settings": len(MATH_DIAGNOSTIC_SETTINGS),
        "present_settings": len(indexed),
        "methods": methods,
    }


def _published_reference() -> dict[str, object]:
    methods = PUBLISHED_REFERENCE["methods"]
    assert isinstance(methods, Mapping)
    return {
        "role": PUBLISHED_REFERENCE["role"],
        "methods": {
            method: {
                "exact_clean": _mapping(methods[method], field=method)["exact_clean"],
                "same_changed": _mapping(methods[method], field=method)["same_changed"],
                "archive_changed": _mapping(methods[method], field=method)["archive_changed"],
            }
            for method in _METHOD_ORDER
        },
    }


def _summary_payload(
    indexed: Mapping[tuple[str, str, str], _Setting],
    math_indexed: Mapping[tuple[str, str, str], _Setting] | None,
) -> dict[str, object]:
    methods = {method: _method_summary(indexed, method) for method in _METHOD_ORDER}
    reference = _published_reference()
    reference_methods = _mapping(reference["methods"], field="published methods")
    comparison: dict[str, object] = {}
    all_match = True
    for method in _METHOD_ORDER:
        fresh = _mapping(methods[method], field=method)
        published = _mapping(reference_methods[method], field=method)
        cells = {
            "exact_clean": fresh["exact_clean"] == published["exact_clean"],
            "same_changed": _mapping(fresh["same"], field="same")["changed"]
            == published["same_changed"],
            "archive": {
                "comparable": False,
                "published_changed": published["archive_changed"],
                "fresh_endpoint": "separate_source",
            },
        }
        comparison[method] = cells
        all_match = all_match and bool(cells["exact_clean"]) and bool(cells["same_changed"])
    return {
        "schema_version": "input-corrector-paper-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "build-input-corrector-summary",
        "coverage": {
            "core": {
                "complete_grid": True,
                "expected_settings": len(CORE_SETTINGS),
                "present_settings": len(indexed),
                "missing_settings": [],
                "unexpected_settings": [],
            }
        },
        "methods": methods,
        "math_intact_word_changes": (
            {
                "status": "not-run",
                "expected_settings": len(MATH_DIAGNOSTIC_SETTINGS),
                "present_settings": 0,
                "methods": {},
            }
            if math_indexed is None
            else _math_summary(math_indexed)
        ),
        "published_reference": {"methods": reference["methods"]},
        "comparability": {
            "reference_is_acceptance_target": False,
            "fresh_separate_source_is_published_archive": False,
            "archive_interpretation": "separate archived-run provenance control only",
        },
        "published_comparison": {
            "role": "descriptive-comparable-cells-only",
            "all_comparable_integer_cells_match": all_match,
            "by_method": comparison,
        },
    }


def _csv_text(payload: Mapping[str, object]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "method",
            "word_setting_mean",
            "word_pooled_diagnostic",
            "exact_clean",
            "same_changed",
            "same_rate",
            "separate_source_changed",
            "separate_source_rate",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    methods = _mapping(payload["methods"], field="methods")
    for method in _METHOD_ORDER:
        row = _mapping(methods[method], field=method)
        word = _mapping(row["word"], field="word")
        same = _mapping(row["same"], field="same")
        separate = _mapping(row["separate_source"], field="separate_source")
        writer.writerow(
            {
                "method": method,
                "word_setting_mean": word["setting_mean_exact_restoration"],
                "word_pooled_diagnostic": word["pooled_exact_restoration"],
                "exact_clean": row["exact_clean"],
                "same_changed": same["changed"],
                "same_rate": same["rate"],
                "separate_source_changed": separate["changed"],
                "separate_source_rate": separate["rate"],
            }
        )
    return stream.getvalue()


def _markdown_text(payload: Mapping[str, object]) -> str:
    lines = [
        "# Fresh input-corrector summary",
        "",
        "| Method | Word setting mean | Exact clean | Same changed | Separate-source changed |",
        "|---|---:|---:|---:|---:|",
    ]
    methods = _mapping(payload["methods"], field="methods")
    for method in _METHOD_ORDER:
        row = _mapping(methods[method], field=method)
        word = _mapping(row["word"], field="word")
        same = _mapping(row["same"], field="same")
        separate = _mapping(row["separate_source"], field="separate_source")
        lines.append(
            f"| {method} | {float(word['setting_mean_exact_restoration']):.6f} | "
            f"{row['exact_clean']} | {same['changed']} | {separate['changed']} |"
        )
    lines.extend(
        (
            "",
            "The separate-source column is a fresh cross-run diagnostic, not the paper's archive column.",
            "",
        )
    )
    return "\n".join(lines)


def _latex_text(payload: Mapping[str, object]) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Word & Exact clean & Same & Separate source \\",
        r"\midrule",
    ]
    methods = _mapping(payload["methods"], field="methods")
    for method in _METHOD_ORDER:
        row = _mapping(methods[method], field=method)
        word = _mapping(row["word"], field="word")
        same = _mapping(row["same"], field="same")
        separate = _mapping(row["separate_source"], field="separate_source")
        label = method.replace("_", r"\_")
        lines.append(
            f"{label} & {float(word['setting_mean_exact_restoration']):.3f} & "
            f"{row['exact_clean']} & {same['changed']} & {separate['changed']} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_producer_inputs_unchanged(settings: Sequence[_Setting]) -> None:
    for setting in settings:
        try:
            current = (
                _sha256_file(setting.run_path),
                _sha256_file(setting.run_path.parent / _RECORDS_NAME),
                _sha256_file(setting.run_path.parent / _SETTING_SUMMARY_NAME),
            )
        except OSError as exc:
            raise InputCorrectorSummaryInputError(
                f"producer input disappeared during summary rendering: {setting.run_path}"
            ) from exc
        expected = (
            setting.run_sha256,
            setting.records_sha256,
            setting.summary_sha256,
        )
        if current != expected:
            raise InputCorrectorSummaryInputError(
                f"producer input changed during summary rendering: {setting.run_path}"
            )


def run_build_input_corrector_summary(
    config: BuildInputCorrectorSummaryConfig,
) -> BuildInputCorrectorSummaryResult:
    """Validate a complete producer grid and atomically publish Table 12 files."""
    if config.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {config.output_dir}")
    core_settings = _discover(config.runs_root, role="core")
    indexed = _index_grid(core_settings, expected=set(CORE_SETTINGS), role="core")
    _validate_shared_source_cohorts(indexed, role="core")
    math_indexed: dict[tuple[str, str, str], _Setting] | None = None
    if config.math_runs_root is not None:
        math_settings = _discover(config.math_runs_root, role="MATH-500")
        math_indexed = _index_grid(
            math_settings,
            expected=set(MATH_DIAGNOSTIC_SETTINGS),
            role="MATH-500",
        )
        _validate_shared_source_cohorts(math_indexed, role="MATH-500")
        _validate_core_math_revisions(indexed, math_indexed)

    payload = _summary_payload(indexed, math_indexed)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.stage-",
            dir=config.output_dir.parent,
        )
    )
    try:
        summary_path = stage / _SUMMARY_NAME
        csv_path = stage / _CSV_NAME
        markdown_path = stage / _MARKDOWN_NAME
        latex_path = stage / _LATEX_NAME
        _write_json(summary_path, payload)
        _write_text(csv_path, _csv_text(payload))
        _write_text(markdown_path, _markdown_text(payload))
        _write_text(latex_path, _latex_text(payload))
        outputs = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (summary_path, csv_path, markdown_path, latex_path)
        }
        source_runs = [
            {
                "model": setting.identity[0],
                "benchmark": setting.identity[1],
                "corrector": setting.identity[2],
                "run_sha256": setting.run_sha256,
                "records_sha256": setting.records_sha256,
                "summary_sha256": setting.summary_sha256,
            }
            for setting in sorted(
                (*indexed.values(), *(math_indexed or {}).values()),
                key=lambda value: value.identity,
            )
        ]
        run_path = stage / "run.json"
        _write_json(
            run_path,
            {
                "schema_version": "build-input-corrector-summary-run/v1",
                "paper_sha256": PAPER_SHA256,
                "operation": "build-input-corrector-summary",
                "status": "completed",
                "analysis_code": analysis_code_identity(),
                "arguments": {
                    "math_runs_root_provided": config.math_runs_root is not None,
                },
                "counts": {
                    "core_settings": len(indexed),
                    "math_settings": len(math_indexed or {}),
                },
                "source_runs": source_runs,
                "outputs": outputs,
            },
        )
        _assert_producer_inputs_unchanged(
            (*indexed.values(), *(math_indexed or {}).values())
        )
        _fsync_directory(stage)
        os.replace(stage, config.output_dir)
        _fsync_directory(config.output_dir.parent)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return BuildInputCorrectorSummaryResult(
        summary_path=config.output_dir / _SUMMARY_NAME,
        csv_path=config.output_dir / _CSV_NAME,
        markdown_path=config.output_dir / _MARKDOWN_NAME,
        latex_path=config.output_dir / _LATEX_NAME,
        run_path=config.output_dir / "run.json",
        settings=len(indexed),
        math_settings=len(math_indexed or {}),
    )
