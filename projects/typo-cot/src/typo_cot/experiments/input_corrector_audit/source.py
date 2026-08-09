"""Fail-closed adapter for completed ``prepare-edited-pairs`` sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.input_corrector_audit.protocol import (
    PAPER_BENCHMARK_ITEM_COUNTS,
    SOURCE_PROTOCOL,
    SUPPORTED_BENCHMARKS,
    canonical_sha256,
)
from typo_cot.experiments.input_corrector_audit.restoration import build_reference

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_DATASET_LOADERS = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
    "math-500": "math",
}
_DATASET_SAMPLES_PER_SUBSET = {
    "gsm8k": None,
    "mmlu": 50,
    "mmlu-pro": 100,
    "arc": None,
    "csqa": None,
    "math-500": None,
}
_PREPARATION_DECODING = {
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
_PREPARATION_PROVENANCE = {
    "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
    "random_seed_algorithm": "sha256-first-64-bits/v1",
    "generation_protocol": "explicit-greedy-generation/v1",
    "generation_termination_protocol": SOURCE_PROTOCOL[
        "generation_termination_protocol"
    ],
    "target_position": "maximum-logit-after-first-cot-token",
    "alignment": "actual-edited-word-final-token",
}


class InputCorrectorSourceError(ValueError):
    """Raised when prepared pairs cannot satisfy the frozen audit contract."""


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        raise InputCorrectorSourceError(f"invalid JSON at {context}: {exc}") from exc


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InputCorrectorSourceError(f"{field} must be a JSON object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise InputCorrectorSourceError(f"{field} must be a JSON list")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputCorrectorSourceError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise InputCorrectorSourceError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _span(value: object, *, field: str, text: str) -> tuple[int, int]:
    payload = _object(value, field=field)
    start = _integer(payload.get("start"), field=f"{field}.start")
    end = _integer(payload.get("end"), field=f"{field}.end", minimum=1)
    if end <= start or end > len(text):
        raise InputCorrectorSourceError(f"{field} is outside its text boundary")
    return start, end


def _validate_answer(value: object, *, field: str) -> None:
    payload = _object(value, field=field)
    if not isinstance(payload.get("value"), str):
        raise InputCorrectorSourceError(f"{field}.value must be a string")
    for name in ("is_extracted", "is_correct"):
        if not isinstance(payload.get(name), bool):
            raise InputCorrectorSourceError(f"{field}.{name} must be boolean")


def _validate_generation_arm(value: Mapping[str, object], *, field: str) -> None:
    continuation = value.get("continuation")
    if not isinstance(continuation, str):
        raise InputCorrectorSourceError(f"{field}.continuation must be a string")
    token_count = value.get("continuation_token_count")
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or not 1 <= token_count <= int(_PREPARATION_DECODING["max_new_tokens"])
    ):
        raise InputCorrectorSourceError(
            f"{field}.continuation_token_count must be between 1 and 512"
        )
    termination = value.get("termination")
    if termination not in {"eos", "length-cap"}:
        raise InputCorrectorSourceError(
            f"{field}.termination must be 'eos' or 'length-cap'"
        )
    if (
        termination == "length-cap"
        and token_count != _PREPARATION_DECODING["max_new_tokens"]
    ):
        raise InputCorrectorSourceError(
            f"{field}.length-cap termination requires exactly 512 tokens"
        )


def _validate_aligned_word(
    value: object,
    *,
    index: int,
    clean_editable: str,
    edited_editable: str,
    clean_prompt: str,
    edited_prompt: str,
) -> None:
    field = f"aligned_words[{index}]"
    word = _object(value, field=field)
    clean_text = _string(word.get("clean_text"), field=f"{field}.clean_text")
    edited_text = _string(word.get("edited_text"), field=f"{field}.edited_text")
    clean_start, clean_end = _span(
        word.get("clean_editable_span"),
        field=f"{field}.clean_editable_span",
        text=clean_editable,
    )
    edited_start, edited_end = _span(
        word.get("edited_editable_span"),
        field=f"{field}.edited_editable_span",
        text=edited_editable,
    )
    prompt_clean_start, prompt_clean_end = _span(
        word.get("clean_prompt_span"),
        field=f"{field}.clean_prompt_span",
        text=clean_prompt,
    )
    prompt_edited_start, prompt_edited_end = _span(
        word.get("edited_prompt_span"),
        field=f"{field}.edited_prompt_span",
        text=edited_prompt,
    )
    if clean_editable[clean_start:clean_end] != clean_text:
        raise InputCorrectorSourceError(f"{field} clean word span does not match clean_text")
    if edited_editable[edited_start:edited_end] != edited_text:
        raise InputCorrectorSourceError(f"{field} edited word span does not match edited_text")
    if clean_prompt[prompt_clean_start:prompt_clean_end] != clean_text:
        raise InputCorrectorSourceError(f"{field} clean prompt span does not match clean_text")
    if edited_prompt[prompt_edited_start:prompt_edited_end] != edited_text:
        raise InputCorrectorSourceError(f"{field} edited prompt span does not match edited_text")
    _integer(word.get("word_index"), field=f"{field}.word_index")
    for name in (
        "target_ranks",
        "target_token_indices",
        "clean_token_indices",
        "edited_token_indices",
    ):
        values = _list(word.get(name), field=f"{field}.{name}")
        if not values or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values
        ):
            raise InputCorrectorSourceError(f"{field}.{name} must contain token indices")
    _integer(word.get("clean_final_token"), field=f"{field}.clean_final_token")
    _integer(word.get("edited_final_token"), field=f"{field}.edited_final_token")


def _validate_record(
    value: object,
    *,
    model: str,
    benchmark: str,
) -> dict[str, object]:
    record = dict(_object(value, field="pair record"))
    if record.get("schema_version") != SOURCE_PROTOCOL["schema"]:
        raise InputCorrectorSourceError("pair record schema is not prepare-edited-pairs/v1")
    sample_id = _string(record.get("sample_id"), field="pair record sample_id")
    if record.get("model") != model:
        raise InputCorrectorSourceError(f"pair record {sample_id} model does not match")
    if record.get("benchmark") != benchmark:
        raise InputCorrectorSourceError(f"pair record {sample_id} benchmark does not match")
    if record.get("targeting") != SOURCE_PROTOCOL["targeting"]:
        raise InputCorrectorSourceError(f"pair record {sample_id} must use attribution-4 targeting")
    if type(record.get("seed")) is not int or record.get("seed") != SOURCE_PROTOCOL["seed"]:
        raise InputCorrectorSourceError(f"pair record {sample_id} seed must be 42")
    if (
        type(record.get("num_edits_requested")) is not int
        or record.get("num_edits_requested") != SOURCE_PROTOCOL["num_edits"]
    ):
        raise InputCorrectorSourceError(
            f"pair record {sample_id} num_edits_requested must describe four edits"
        )

    attempts = _list(
        record.get("target_attempts"), field=f"pair record {sample_id}.target_attempts"
    )
    attempt_count = _integer(
        record.get("num_target_attempts"),
        field=f"pair record {sample_id}.num_target_attempts",
    )
    if attempt_count != len(attempts):
        raise InputCorrectorSourceError(
            f"pair record {sample_id} num_target_attempts disagrees with target_attempts"
        )

    clean = _object(record.get("clean"), field=f"pair record {sample_id}.clean")
    edited = _object(record.get("edited"), field=f"pair record {sample_id}.edited")
    clean_prompt = _string(clean.get("prompt"), field=f"pair record {sample_id}.clean.prompt")
    edited_prompt = _string(edited.get("prompt"), field=f"pair record {sample_id}.edited.prompt")
    clean_editable = _string(
        clean.get("editable_text"), field=f"pair record {sample_id}.clean.editable_text"
    )
    edited_editable = _string(
        edited.get("editable_text"), field=f"pair record {sample_id}.edited.editable_text"
    )
    question = _string(clean.get("question"), field=f"pair record {sample_id}.clean.question")
    raw_choices = clean.get("choices")
    if raw_choices is None:
        choices = None
    elif isinstance(raw_choices, list) and all(
        isinstance(choice, str) and choice for choice in raw_choices
    ):
        choices = raw_choices
    else:
        raise InputCorrectorSourceError(
            f"pair record {sample_id}.clean.choices must be null or a list of strings"
        )
    if build_reference(question, choices) != clean_editable:
        raise InputCorrectorSourceError(
            f"pair record {sample_id} clean editable text disagrees with question/choices"
        )
    clean_start, clean_end = _span(
        clean.get("editable_prompt_span"),
        field=f"pair record {sample_id}.clean.editable_prompt_span",
        text=clean_prompt,
    )
    edited_start, edited_end = _span(
        edited.get("editable_prompt_span"),
        field=f"pair record {sample_id}.edited.editable_prompt_span",
        text=edited_prompt,
    )
    if clean_prompt[clean_start:clean_end] != clean_editable:
        raise InputCorrectorSourceError(
            f"pair record {sample_id} clean editable text does not match its prompt span"
        )
    if edited_prompt[edited_start:edited_end] != edited_editable:
        raise InputCorrectorSourceError(
            f"pair record {sample_id} edited editable text does not match its prompt span"
        )
    if (
        clean_prompt[:clean_start] != edited_prompt[:edited_start]
        or clean_prompt[clean_end:] != edited_prompt[edited_end:]
    ):
        raise InputCorrectorSourceError(
            f"pair record {sample_id} prompt differs outside its editable span"
        )
    _validate_answer(clean.get("answer"), field=f"pair record {sample_id}.clean.answer")
    _validate_answer(edited.get("answer"), field=f"pair record {sample_id}.edited.answer")
    _validate_generation_arm(clean, field=f"pair record {sample_id}.clean")
    _validate_generation_arm(edited, field=f"pair record {sample_id}.edited")

    aligned_words = _list(
        record.get("aligned_words"), field=f"pair record {sample_id}.aligned_words"
    )
    aligned_count = _integer(
        record.get("num_aligned_words"),
        field=f"pair record {sample_id}.num_aligned_words",
    )
    if aligned_count != len(aligned_words):
        raise InputCorrectorSourceError(
            f"pair record {sample_id} aligned word count disagrees with aligned_words"
        )
    for index, word in enumerate(aligned_words):
        _validate_aligned_word(
            word,
            index=index,
            clean_editable=clean_editable,
            edited_editable=edited_editable,
            clean_prompt=clean_prompt,
            edited_prompt=edited_prompt,
        )
    return record


def _validate_manifest(
    value: object,
    *,
    model: str,
    benchmark: str,
    record_count: int,
    records: list[dict[str, object]],
    pairs_sha256: str,
) -> dict[str, object]:
    manifest = dict(_object(value, field="source run manifest"))
    if manifest.get("schema_version") != SOURCE_PROTOCOL["run_schema"]:
        raise InputCorrectorSourceError("source run manifest schema is unsupported")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise InputCorrectorSourceError("source run paper fingerprint does not match")
    if manifest.get("operation") != "prepare-edited-pairs":
        raise InputCorrectorSourceError("source run operation must be prepare-edited-pairs")
    if manifest.get("status") != SOURCE_PROTOCOL["status"]:
        raise InputCorrectorSourceError("source run must be completed")

    arguments = _object(manifest.get("arguments"), field="source run arguments")
    if arguments.get("model") != model:
        raise InputCorrectorSourceError("source run model does not match")
    if arguments.get("benchmark") != benchmark:
        raise InputCorrectorSourceError("source run benchmark does not match")
    if arguments.get("targeting") != SOURCE_PROTOCOL["targeting"]:
        raise InputCorrectorSourceError("source run must use attribution-4 targeting")
    if type(arguments.get("seed")) is not int or arguments.get("seed") != 42:
        raise InputCorrectorSourceError("source run seed must be 42")
    if type(arguments.get("num_edits")) is not int or arguments.get("num_edits") != 4:
        raise InputCorrectorSourceError("source run num_edits must describe four edits")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise InputCorrectorSourceError("source run must be unlimited with limit=null")
    if "sample_ids" in arguments:
        raise InputCorrectorSourceError(
            "source run must use the paper cohort and must not set arguments.sample_ids"
        )
    if arguments.get("max_new_tokens") != 512:
        raise InputCorrectorSourceError("source run max_new_tokens must be 512")

    decoding = _object(manifest.get("decoding"), field="source run decoding")
    for field, expected in _PREPARATION_DECODING.items():
        value = decoding.get(field)
        if value != expected or type(value) is not type(expected):
            raise InputCorrectorSourceError(
                f"source decoding {field} must be {expected!r}"
            )

    counts = _object(manifest.get("counts"), field="source run counts")
    for name in ("discovered", "written"):
        if type(counts.get(name)) is not int or counts.get(name) != record_count:
            raise InputCorrectorSourceError(f"source run {name} count does not match records")
    if type(counts.get("failed")) is not int or counts.get("failed") != 0:
        raise InputCorrectorSourceError("source run failed count must be zero")
    failures = _list(manifest.get("failures"), field="source run failures")
    if failures:
        raise InputCorrectorSourceError("source run contains failures")

    outputs = _object(manifest.get("outputs"), field="source run outputs")
    if set(outputs) != {"pairs"}:
        raise InputCorrectorSourceError(
            "source run output inventory must contain only pairs; "
            "rerun prepare-edited-pairs"
        )
    pairs_output = _object(outputs.get("pairs"), field="source run outputs.pairs")
    if pairs_output.get("path") != "pairs.jsonl":
        raise InputCorrectorSourceError("source pairs output path must be pairs.jsonl")
    if (
        type(pairs_output.get("records")) is not int
        or pairs_output.get("records") != record_count
    ):
        raise InputCorrectorSourceError("source pairs output record count does not match")
    if pairs_output.get("sha256") != pairs_sha256:
        raise InputCorrectorSourceError(
            "source pairs output SHA-256 does not match pairs.jsonl"
        )

    provenance = _object(manifest.get("provenance"), field="source run provenance")
    if provenance.get("model") != model:
        raise InputCorrectorSourceError("source provenance model does not match")
    revision = provenance.get("model_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise InputCorrectorSourceError("source provenance model_revision is missing or invalid")
    if provenance.get("benchmark_dataset_loader") != _DATASET_LOADERS[benchmark]:
        raise InputCorrectorSourceError("source benchmark_dataset_loader does not match dataset")
    for field, expected in _PREPARATION_PROVENANCE.items():
        if provenance.get(field) != expected:
            raise InputCorrectorSourceError(
                f"source provenance {field} must be {expected!r}"
            )
    if "sample_id_cohort" in provenance:
        raise InputCorrectorSourceError(
            "source run must not contain explicit sample-ID cohort provenance"
        )
    if provenance.get("dataset_samples_per_subset") != _DATASET_SAMPLES_PER_SUBSET[benchmark]:
        raise InputCorrectorSourceError(
            "source dataset_samples_per_subset does not match the paper cohort"
        )
    if provenance.get("dataset_sample_count") != record_count:
        raise InputCorrectorSourceError("source dataset_sample_count does not match record count")
    dataset_hash = provenance.get("dataset_records_sha256")
    if not isinstance(dataset_hash, str) or _SHA256.fullmatch(dataset_hash) is None:
        raise InputCorrectorSourceError("source dataset_records_sha256 is not a SHA-256")
    dataset_identity = [
        {
            "sample_id": record["sample_id"],
            "question": _object(record["clean"], field="clean").get("question"),
            "choices": _object(record["clean"], field="clean").get("choices"),
            "correct_answer": record.get("gold_answer"),
            "subset": record.get("subset"),
        }
        for record in records
    ]
    if canonical_sha256(dataset_identity) != dataset_hash:
        raise InputCorrectorSourceError("source dataset_records_sha256 does not match records")
    return manifest


@dataclass(frozen=True, slots=True)
class InputCorrectorSource:
    """Validated prepared pairs and immutable source identities."""

    model: str
    benchmark: str
    model_revision: str
    records: tuple[dict[str, object], ...]
    pairs_path: Path
    run_path: Path
    pairs_sha256: str
    run_sha256: str
    ordered_sample_ids_sha256: str
    dataset_records_sha256: str

    def assert_unchanged(self) -> None:
        """Fail if either source file changed after the initial validation."""
        if sha256_file(self.pairs_path) != self.pairs_sha256:
            raise InputCorrectorSourceError("pairs source changed after validation")
        if sha256_file(self.run_path) != self.run_sha256:
            raise InputCorrectorSourceError("source run manifest changed after validation")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_kind": "completed-prepare-edited-pairs/v1",
            "model": self.model,
            "benchmark": self.benchmark,
            "model_revision": self.model_revision,
            "records": len(self.records),
            "pairs_sha256": self.pairs_sha256,
            "run_sha256": self.run_sha256,
            "ordered_sample_ids_sha256": self.ordered_sample_ids_sha256,
            "dataset_records_sha256": self.dataset_records_sha256,
        }


def load_input_corrector_source(
    pairs_path: Path,
    *,
    model: str,
    benchmark: str,
    require_paper_cohort_size: bool = True,
) -> InputCorrectorSource:
    """Load and validate one completed paper-aligned pair source."""
    if type(require_paper_cohort_size) is not bool:
        raise TypeError("require_paper_cohort_size must be boolean")
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise InputCorrectorSourceError(f"unsupported public benchmark: {benchmark!r}")
    path = Path(pairs_path).resolve()
    if not path.is_file():
        raise InputCorrectorSourceError(f"pairs file does not exist: {path}")
    run_path = path.with_name("run.json")
    if not run_path.is_file():
        raise InputCorrectorSourceError(f"sibling run.json manifest does not exist: {run_path}")

    initial_pairs_sha256 = sha256_file(path)
    initial_run_sha256 = sha256_file(run_path)
    manifest_payload = _loads(run_path.read_text(encoding="utf-8"), context=str(run_path))
    records: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise InputCorrectorSourceError(f"empty JSONL line at {path}:{line_number}")
        payload = _loads(line, context=f"{path}:{line_number}")
        records.append(_validate_record(payload, model=model, benchmark=benchmark))
    if not records:
        raise InputCorrectorSourceError("pairs file contains no records")
    expected_records = PAPER_BENCHMARK_ITEM_COUNTS[benchmark]
    if require_paper_cohort_size and len(records) != expected_records:
        raise InputCorrectorSourceError(
            f"paper source for {benchmark} must contain {expected_records} records; "
            f"found {len(records)}"
        )
    sample_ids = [str(record["sample_id"]) for record in records]
    if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise InputCorrectorSourceError("pair sample IDs must be unique and sorted")

    manifest = _validate_manifest(
        manifest_payload,
        model=model,
        benchmark=benchmark,
        record_count=len(records),
        records=records,
        pairs_sha256=initial_pairs_sha256,
    )
    if sha256_file(path) != initial_pairs_sha256:
        raise InputCorrectorSourceError("pairs source changed during hash validation")
    if sha256_file(run_path) != initial_run_sha256:
        raise InputCorrectorSourceError("source run manifest changed during hash validation")
    provenance = _object(manifest["provenance"], field="source run provenance")
    return InputCorrectorSource(
        model=model,
        benchmark=benchmark,
        model_revision=str(provenance["model_revision"]),
        records=tuple(records),
        pairs_path=path,
        run_path=run_path,
        pairs_sha256=initial_pairs_sha256,
        run_sha256=initial_run_sha256,
        ordered_sample_ids_sha256=canonical_sha256(sample_ids),
        dataset_records_sha256=str(provenance["dataset_records_sha256"]),
    )
