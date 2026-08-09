"""Fail-closed readers for Table 8 producer artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_cot.data.cohorts import SampleIdCohort, canonical_sample_id_set_sha256
from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.cot_swap.protocol import protocol_for as cot_swap_protocol_for
from typo_cot.experiments.edit_count_sensitivity.protocol import (
    EXPECTED_RESTORATION_SETTINGS,
)
from typo_cot.experiments.prepare_edited_pairs.runtime import (
    paper_dataset_samples_per_subset,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREPARE_DECODING: dict[str, object] = {
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
_DATASET_LOADERS = {
    "gsm8k": "gsm8k",
    "math-500": "math",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}
_COT_OUTPUTS = (
    "cot_swap_records.jsonl",
    "pair_status_records.jsonl",
    "cot_swap_summary.json",
)


class EditCountSensitivityInputError(ValueError):
    """An input cannot support an auditable Table 8 analysis."""


@dataclass(frozen=True, slots=True)
class PreparedEditCountRun:
    """One validated model/benchmark/edit-count accuracy source."""

    model: str
    benchmark: str
    edit_count: int
    records: tuple[dict[str, object], ...]
    records_by_id: Mapping[str, dict[str, object]]
    record_sha256_by_id: Mapping[str, str]
    pairs_path: Path
    run_path: Path
    pairs_sha256: str
    run_sha256: str
    model_revision: str
    dataset_records_sha256: str
    dataset_sample_count: int
    sample_id_cohort: Mapping[str, object] | None = None

    @property
    def setting(self) -> tuple[str, str]:
        return self.model, self.benchmark


@dataclass(frozen=True, slots=True)
class CotSwapEditCountRun:
    """One validated setting/edit-count CoT-swap event source."""

    model: str
    benchmark: str
    edit_count: int
    records: tuple[dict[str, object], ...]
    run_path: Path
    records_path: Path
    summary_path: Path
    run_sha256: str
    records_sha256: str
    source_pairs_sha256: str
    source_run_sha256: str
    source_record_count: int = 0

    @property
    def setting(self) -> tuple[str, str]:
        return self.model, self.benchmark


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_loads(text: str, *, context: str) -> object:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise EditCountSensitivityInputError(f"{context}: invalid JSON: {exc}") from exc
    return value


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise EditCountSensitivityInputError(f"required input file is missing: {path}")
    value = _strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(value, dict):
        raise EditCountSensitivityInputError(f"{path}: expected a JSON object")
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise EditCountSensitivityInputError(f"required input file is missing: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise EditCountSensitivityInputError(
                    f"{path}:{line_number}: blank JSONL records are forbidden"
                )
            value = _strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise EditCountSensitivityInputError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(value)
    return tuple(rows)


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EditCountSensitivityInputError(f"{context} must be a JSON object")
    return value


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise EditCountSensitivityInputError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, *, context: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise EditCountSensitivityInputError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _hash(value: object, *, context: str) -> str:
    text = _string(value, context=context)
    if _SHA256.fullmatch(text) is None:
        raise EditCountSensitivityInputError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _validate_answer(value: object, *, context: str) -> Mapping[str, object]:
    answer = _mapping(value, context=context)
    if not isinstance(answer.get("is_correct"), bool):
        raise EditCountSensitivityInputError(f"{context}.is_correct must be boolean")
    if not isinstance(answer.get("is_extracted"), bool):
        raise EditCountSensitivityInputError(f"{context}.is_extracted must be boolean")
    if not isinstance(answer.get("value"), str):
        raise EditCountSensitivityInputError(f"{context}.value must be a string")
    return answer


def _validate_pair_record(
    row: dict[str, object],
    *,
    path: Path,
    line_number: int,
    model: str,
    benchmark: str,
    edit_count: int,
) -> str:
    context = f"{path}:{line_number}"
    if row.get("schema_version") != "prepare-edited-pairs/v1":
        raise EditCountSensitivityInputError(f"{context}: unknown pair schema")
    for field, expected in (
        ("model", model),
        ("benchmark", benchmark),
        ("targeting", "attribution-4"),
        ("seed", 42),
        ("num_edits_requested", edit_count),
    ):
        if row.get(field) != expected or type(row.get(field)) is not type(expected):
            raise EditCountSensitivityInputError(
                f"{context}: {field} does not match the producer manifest"
            )
    sample_id = _string(row.get("sample_id"), context=f"{context}.sample_id")
    _string(row.get("gold_answer"), context=f"{context}.gold_answer")
    attempts = row.get("target_attempts")
    attempt_count = _integer(
        row.get("num_target_attempts"), context=f"{context}.num_target_attempts"
    )
    if not isinstance(attempts, list) or len(attempts) != attempt_count:
        raise EditCountSensitivityInputError(
            f"{context}: target_attempts must match num_target_attempts"
        )
    if attempt_count > edit_count:
        raise EditCountSensitivityInputError(
            f"{context}: applied edit attempts exceed the requested count"
        )
    aligned = row.get("aligned_words")
    aligned_count = _integer(row.get("num_aligned_words"), context=f"{context}.num_aligned_words")
    if not isinstance(aligned, list) or len(aligned) != aligned_count:
        raise EditCountSensitivityInputError(
            f"{context}: aligned_words must match num_aligned_words"
        )
    if aligned_count > attempt_count:
        raise EditCountSensitivityInputError(
            f"{context}: aligned words exceed applied edit attempts"
        )
    for side in ("clean", "edited"):
        payload = _mapping(row.get(side), context=f"{context}.{side}")
        _string(payload.get("prompt"), context=f"{context}.{side}.prompt")
        _integer(
            payload.get("prompt_token_count"),
            context=f"{context}.{side}.prompt_token_count",
            minimum=1,
        )
        if not isinstance(payload.get("continuation"), str):
            raise EditCountSensitivityInputError(f"{context}.{side}.continuation must be a string")
        continuation_tokens = _integer(
            payload.get("continuation_token_count"),
            context=f"{context}.{side}.continuation_token_count",
        )
        if continuation_tokens > 512:
            raise EditCountSensitivityInputError(
                f"{context}.{side} exceeds the 512-token generation cap"
            )
        _validate_answer(payload.get("answer"), context=f"{context}.{side}.answer")
    return sample_id


def _validate_prepare_manifest(
    path: Path,
    manifest: dict[str, object],
    *,
    edit_counts: frozenset[int],
    expected_settings: frozenset[tuple[str, str]] | None = None,
    sample_id_cohort: SampleIdCohort | None = None,
) -> PreparedEditCountRun | None:
    if manifest.get("operation") != "prepare-edited-pairs":
        return None
    arguments = _mapping(manifest.get("arguments"), context=f"{path}.arguments")
    if arguments.get("targeting") != "attribution-4":
        return None
    raw_count = arguments.get("num_edits")
    if raw_count not in edit_counts or isinstance(raw_count, bool):
        return None
    edit_count = int(raw_count)
    if manifest.get("schema_version") != "prepare-edited-pairs-run/v1":
        raise EditCountSensitivityInputError(f"{path}: unknown prepare run schema")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise EditCountSensitivityInputError(f"{path}: final-paper SHA-256 mismatch")
    if manifest.get("status") != "completed":
        raise EditCountSensitivityInputError(f"{path}: prepare run is not completed")
    model = _string(arguments.get("model"), context=f"{path}.arguments.model")
    benchmark = _string(arguments.get("benchmark"), context=f"{path}.arguments.benchmark")
    if benchmark not in _DATASET_LOADERS:
        raise EditCountSensitivityInputError(f"{path}: unsupported benchmark {benchmark!r}")
    if expected_settings is not None and (model, benchmark) not in expected_settings:
        return None
    if arguments.get("seed") != 42:
        raise EditCountSensitivityInputError(f"{path}: prepare seed must be 42")
    if arguments.get("max_new_tokens") != 512:
        raise EditCountSensitivityInputError(f"{path}: prepare generation cap must be 512")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise EditCountSensitivityInputError(f"{path}: limited prepare runs are not accepted")
    decoding = _mapping(manifest.get("decoding"), context=f"{path}.decoding")
    for field, expected in _PREPARE_DECODING.items():
        if decoding.get(field) != expected or type(decoding.get(field)) is not type(expected):
            raise EditCountSensitivityInputError(f"{path}: decoding.{field} must be {expected!r}")
    provenance = _mapping(manifest.get("provenance"), context=f"{path}.provenance")
    expected_cohort_rule = (
        "explicit-sample-id-cohort/v1"
        if sample_id_cohort is not None
        else "paper-model-benchmark-cohort/v1"
    )
    for field, expected in (
        ("model", model),
        ("benchmark_dataset_loader", _DATASET_LOADERS[benchmark]),
        ("dataset_cohort_rule", expected_cohort_rule),
        ("random_seed_algorithm", "sha256-first-64-bits/v1"),
        ("generation_protocol", "explicit-greedy-generation/v1"),
        ("target_position", "maximum-logit-after-first-cot-token"),
        ("alignment", "actual-edited-word-final-token"),
    ):
        if provenance.get(field) != expected:
            raise EditCountSensitivityInputError(f"{path}: provenance.{field} must be {expected!r}")
    expected_subset_cap = paper_dataset_samples_per_subset(model=model, benchmark=benchmark)
    if provenance.get("dataset_samples_per_subset") != expected_subset_cap:
        raise EditCountSensitivityInputError(
            f"{path}: dataset_samples_per_subset does not match the paper cohort"
        )
    cohort_provenance: Mapping[str, object] | None = None
    if sample_id_cohort is not None:
        if benchmark != sample_id_cohort.benchmark:
            raise EditCountSensitivityInputError(
                f"{path}: prepare benchmark does not match the sample-ID cohort"
            )
        if not isinstance(arguments.get("sample_ids"), str) or not arguments.get("sample_ids"):
            raise EditCountSensitivityInputError(
                f"{path}: explicit cohort prepare run is missing arguments.sample_ids"
            )
        cohort_provenance = _mapping(
            provenance.get("sample_id_cohort"),
            context=f"{path}.provenance.sample_id_cohort",
        )
        expected_cohort_provenance = sample_id_cohort.provenance_for(model)
        if dict(cohort_provenance) != expected_cohort_provenance:
            raise EditCountSensitivityInputError(
                f"{path}: sample-ID cohort provenance does not match the requested artifact"
            )
    elif "sample_ids" in arguments or "sample_id_cohort" in provenance:
        raise EditCountSensitivityInputError(
            f"{path}: default cohort run contains unexpected explicit sample-ID metadata"
        )
    model_revision = _string(
        provenance.get("model_revision"), context=f"{path}.provenance.model_revision"
    )
    dataset_hash = _hash(
        provenance.get("dataset_records_sha256"),
        context=f"{path}.provenance.dataset_records_sha256",
    )
    dataset_count = _integer(
        provenance.get("dataset_sample_count"),
        context=f"{path}.provenance.dataset_sample_count",
        minimum=1,
    )
    counts = _mapping(manifest.get("counts"), context=f"{path}.counts")
    discovered = _integer(counts.get("discovered"), context=f"{path}.counts.discovered")
    written = _integer(counts.get("written"), context=f"{path}.counts.written")
    if counts.get("failed") != 0 or manifest.get("failures") not in ([], None):
        raise EditCountSensitivityInputError(f"{path}: prepare run contains failures")
    if discovered != written or written != dataset_count:
        raise EditCountSensitivityInputError(
            f"{path}: prepare run is not a complete unlimited dataset cohort"
        )
    if sample_id_cohort is not None:
        expected_count = sample_id_cohort.expected_count_for(model)
        if expected_count is None:
            raise EditCountSensitivityInputError(
                f"{path}: sample-ID cohort has no coverage contract for {model}"
            )
        if written != expected_count:
            raise EditCountSensitivityInputError(
                f"{path}: prepare count does not match the sample-ID cohort"
            )
    pairs_path = path.parent / "pairs.jsonl"
    rows = _load_jsonl(pairs_path)
    if len(rows) != written:
        raise EditCountSensitivityInputError(f"{path}: written count does not match pairs.jsonl")
    records_by_id: dict[str, dict[str, object]] = {}
    record_sha256_by_id: dict[str, str] = {}
    with pairs_path.open(encoding="utf-8") as handle:
        raw_lines = [raw_line.rstrip("\n") for raw_line in handle]
    previous: str | None = None
    for line_number, row in enumerate(rows, 1):
        sample_id = _validate_pair_record(
            row,
            path=pairs_path,
            line_number=line_number,
            model=model,
            benchmark=benchmark,
            edit_count=edit_count,
        )
        if previous is not None and sample_id <= previous:
            raise EditCountSensitivityInputError(
                f"{pairs_path}: sample IDs must be strictly sorted and unique"
            )
        previous = sample_id
        records_by_id[sample_id] = row
        record_sha256_by_id[sample_id] = hashlib.sha256(
            raw_lines[line_number - 1].encode("utf-8")
        ).hexdigest()
    if sample_id_cohort is not None:
        if not set(records_by_id).issubset(sample_id_cohort.sample_ids):
            raise EditCountSensitivityInputError(
                f"{pairs_path}: pair sample IDs fall outside the requested cohort"
            )
        expected_ids_sha256 = sample_id_cohort.expected_sample_ids_sha256_for(model)
        if expected_ids_sha256 is None:
            raise EditCountSensitivityInputError(
                f"{pairs_path}: sample-ID cohort has no exact-ID contract for {model}"
            )
        if canonical_sample_id_set_sha256(tuple(records_by_id)) != expected_ids_sha256:
            raise EditCountSensitivityInputError(
                f"{pairs_path}: pair IDs do not match the exact model-specific sample-ID set"
            )
    return PreparedEditCountRun(
        model=model,
        benchmark=benchmark,
        edit_count=edit_count,
        records=rows,
        records_by_id=records_by_id,
        record_sha256_by_id=record_sha256_by_id,
        pairs_path=pairs_path.resolve(),
        run_path=path.resolve(),
        pairs_sha256=_sha256(pairs_path),
        run_sha256=_sha256(path),
        model_revision=model_revision,
        dataset_records_sha256=dataset_hash,
        dataset_sample_count=dataset_count,
        sample_id_cohort=cohort_provenance,
    )


def discover_prepared_runs(
    root: Path,
    *,
    edit_counts: Sequence[int],
    expected_settings: Sequence[tuple[str, str]] | None = None,
    sample_id_cohort: SampleIdCohort | None = None,
) -> tuple[PreparedEditCountRun, ...]:
    """Discover relevant completed pair preparations below an arbitrary layout."""
    root = Path(root)
    if not root.is_dir():
        raise EditCountSensitivityInputError(f"pairs_root is not a directory: {root}")
    requested = frozenset(edit_counts)
    settings = frozenset(expected_settings) if expected_settings is not None else None
    found: dict[tuple[str, str, int], PreparedEditCountRun] = {}
    for path in sorted(root.rglob("run.json")):
        manifest = _load_json(path)
        validated = _validate_prepare_manifest(
            path,
            manifest,
            edit_counts=requested,
            expected_settings=settings,
            sample_id_cohort=sample_id_cohort,
        )
        if validated is None:
            continue
        key = (validated.model, validated.benchmark, validated.edit_count)
        if key in found:
            raise EditCountSensitivityInputError(
                "duplicate prepare setting/edit-count input: "
                f"{validated.model} {validated.benchmark} k={validated.edit_count}"
            )
        found[key] = validated
    if not found:
        raise EditCountSensitivityInputError(
            "pairs_root contains no completed unlimited Attribution-4 prepare runs"
        )
    return tuple(found[key] for key in sorted(found))


def _source_num_edits(
    arguments: Mapping[str, object],
    protocol: Mapping[str, object],
    *,
    path: Path,
) -> int:
    generation = _mapping(
        protocol.get("source_generation"), context=f"{path}.protocol.source_generation"
    )
    protocol_count = _integer(
        generation.get("num_edits_requested"),
        context=f"{path}.protocol.source_generation.num_edits_requested",
        minimum=1,
    )
    argument_count = arguments.get("source_num_edits", protocol_count)
    if argument_count != protocol_count or isinstance(argument_count, bool):
        raise EditCountSensitivityInputError(
            f"{path}: source edit count differs between arguments and protocol"
        )
    return protocol_count


def _validate_cot_source(
    value: object,
    *,
    context: str,
    prepared: PreparedEditCountRun,
) -> tuple[str, str]:
    source = _mapping(value, context=context)
    _string(source.get("pairs"), context=f"{context}.pairs")
    _string(source.get("source_run"), context=f"{context}.source_run")
    pairs_hash = _hash(source.get("pairs_sha256"), context=f"{context}.pairs_sha256")
    run_hash = _hash(source.get("source_run_sha256"), context=f"{context}.source_run_sha256")
    expected = {
        "source_schema": "prepare-edited-pairs/v1",
        "model_revision": prepared.model_revision,
        "dataset_records_sha256": prepared.dataset_records_sha256,
        "dataset_sample_count": prepared.dataset_sample_count,
        "record_count": len(prepared.records),
    }
    if prepared.sample_id_cohort is not None:
        expected["sample_id_cohort"] = dict(prepared.sample_id_cohort)
    for field, expected_value in expected.items():
        if source.get(field) != expected_value or type(source.get(field)) is not type(
            expected_value
        ):
            raise EditCountSensitivityInputError(
                f"{context}.{field} does not match the validated prepare run"
            )
    if pairs_hash != prepared.pairs_sha256 or run_hash != prepared.run_sha256:
        raise EditCountSensitivityInputError(
            f"{context}: source SHA-256 does not match the validated prepare run"
        )
    return pairs_hash, run_hash


def _validate_output_registry(
    path: Path,
    manifest: Mapping[str, object],
) -> dict[str, Path]:
    outputs = _mapping(manifest.get("outputs"), context=f"{path}.outputs")
    if set(outputs) != set(_COT_OUTPUTS):
        raise EditCountSensitivityInputError(
            f"{path}: CoT-swap output registry must contain exactly {_COT_OUTPUTS!r}"
        )
    result: dict[str, Path] = {}
    for name in _COT_OUTPUTS:
        metadata = _mapping(outputs[name], context=f"{path}.outputs.{name}")
        expected_hash = _hash(metadata.get("sha256"), context=f"{path}.outputs.{name}.sha256")
        records = _integer(metadata.get("records"), context=f"{path}.outputs.{name}.records")
        output_path = path.parent / name
        if not output_path.is_file():
            raise EditCountSensitivityInputError(f"{path}: missing output {name}")
        if _sha256(output_path) != expected_hash:
            raise EditCountSensitivityInputError(
                f"{output_path}: SHA-256 does not match the producer registry"
            )
        if name.endswith(".jsonl"):
            actual_records = sum(1 for _ in output_path.open(encoding="utf-8"))
            if actual_records != records:
                raise EditCountSensitivityInputError(
                    f"{output_path}: record count does not match the producer registry"
                )
        elif records != 1:
            raise EditCountSensitivityInputError(
                f"{output_path}: JSON summary registry count must be one"
            )
        result[name] = output_path
    return result


def _validate_event_record(
    row: dict[str, object],
    *,
    path: Path,
    line_number: int,
    model: str,
    benchmark: str,
    edit_count: int,
    prepared: PreparedEditCountRun,
) -> str:
    context = f"{path}:{line_number}"
    for field, expected in (
        ("schema_version", "cot-swap-record/v1"),
        ("paper_sha256", PAPER_SHA256),
        ("operation", "cot-swap"),
        ("model", model),
        ("benchmark", benchmark),
        ("targeting", "attribution-4"),
    ):
        if row.get(field) != expected:
            raise EditCountSensitivityInputError(f"{context}: invalid {field}")
    if row.get("source_num_edits", edit_count) != edit_count:
        raise EditCountSensitivityInputError(f"{context}: source edit count mismatch")
    sample_id = _string(row.get("sample_id"), context=f"{context}.sample_id")
    expected_record_hash = prepared.record_sha256_by_id.get(sample_id)
    if expected_record_hash is None:
        raise EditCountSensitivityInputError(
            f"{context}: sample ID is absent from the validated prepare run"
        )
    source = _mapping(row.get("source"), context=f"{context}.source")
    for field, expected in (
        ("pairs_sha256", prepared.pairs_sha256),
        ("source_run_sha256", prepared.run_sha256),
        ("source_record_sha256", expected_record_hash),
    ):
        if source.get(field) != expected:
            raise EditCountSensitivityInputError(
                f"{context}.source.{field} does not match the validated prepare run"
            )
    events = _mapping(row.get("events"), context=f"{context}.events")
    for field in (
        "clean_correct",
        "both_changed",
        "question_only_changed",
        "cot_only_changed",
        "restoration_denominator",
    ):
        if not isinstance(events.get(field), bool):
            raise EditCountSensitivityInputError(f"{context}.events.{field} must be boolean")
    gold_answer = _string(row.get("gold_answer"), context=f"{context}.gold_answer")
    prepared_gold_answer = _string(
        prepared.records_by_id[sample_id].get("gold_answer"),
        context=f"{context}.prepared_pair.gold_answer",
    )
    if gold_answer != prepared_gold_answer:
        raise EditCountSensitivityInputError(
            f"{context}: gold answer does not match the prepared pair"
        )
    cells = _mapping(row.get("cells"), context=f"{context}.cells")
    if set(cells) != {"A", "B", "C", "D"}:
        raise EditCountSensitivityInputError(f"{context}.cells must contain exactly A, B, C, and D")
    answers: dict[str, Mapping[str, object]] = {}
    for cell in ("A", "B", "C", "D"):
        cell_payload = _mapping(cells[cell], context=f"{context}.cells.{cell}")
        answer = _validate_answer(
            cell_payload.get("answer"), context=f"{context}.cells.{cell}.answer"
        )
        if not isinstance(cell_payload.get("equal_to_a"), bool):
            raise EditCountSensitivityInputError(
                f"{context}.cells.{cell}.equal_to_a must be boolean"
            )
        answers[cell] = answer
    answer_benchmark = _DATASET_LOADERS[benchmark]
    for cell, answer in answers.items():
        expected_correct = answers_equal(
            str(answer["value"]),
            prepared_gold_answer,
            benchmark=answer_benchmark,
        )
        if answer["is_correct"] is not expected_correct:
            raise EditCountSensitivityInputError(
                f"{context}.cells.{cell}.answer.is_correct contradicts the extracted "
                "answer and gold answer"
            )
    answer_a = str(answers["A"]["value"])
    same_as_a = {
        "A": True,
        **{
            cell: answers_equal(
                str(answers[cell]["value"]),
                answer_a,
                benchmark=answer_benchmark,
            )
            for cell in ("B", "C", "D")
        },
    }
    for cell, expected in same_as_a.items():
        cell_payload = _mapping(cells[cell], context=f"{context}.cells.{cell}")
        if cell_payload["equal_to_a"] is not expected:
            raise EditCountSensitivityInputError(
                f"{context}.cells.{cell}.equal_to_a contradicts extracted answers"
            )
    expected_events = {
        "clean_correct": answers_equal(
            answer_a,
            prepared_gold_answer,
            benchmark=answer_benchmark,
        ),
        "both_changed": not same_as_a["B"],
        "question_only_changed": not same_as_a["C"],
        "cot_only_changed": not same_as_a["D"],
    }
    for field, expected in expected_events.items():
        if events[field] is not expected:
            raise EditCountSensitivityInputError(
                f"{context}.events.{field} contradicts the four cell answers"
            )
    denominator = events["restoration_denominator"] is True
    restored = events.get("b_to_c_restored")
    expected_denominator = events["clean_correct"] is True and events["both_changed"] is True
    if denominator is not expected_denominator:
        raise EditCountSensitivityInputError(
            f"{context}: restoration denominator contradicts A/B events"
        )
    if denominator:
        if restored not in {True, False} or not isinstance(restored, bool):
            raise EditCountSensitivityInputError(
                f"{context}: restoration denominator requires a boolean outcome"
            )
        if restored is not same_as_a["C"]:
            raise EditCountSensitivityInputError(
                f"{context}: restored outcome contradicts the A/C answers"
            )
    elif restored is not None:
        raise EditCountSensitivityInputError(
            f"{context}: restored outcome must be null outside its denominator"
        )
    return sample_id


def _metric_from_events(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    denominator = sum(
        _mapping(row["events"], context="record.events")["restoration_denominator"] is True
        for row in rows
    )
    numerator = sum(
        _mapping(row["events"], context="record.events").get("b_to_c_restored") is True
        for row in rows
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _validate_cot_manifest(
    path: Path,
    manifest: dict[str, object],
    *,
    edit_counts: frozenset[int],
    prepared_by_key: Mapping[tuple[str, str, int], PreparedEditCountRun],
    expected_settings: frozenset[tuple[str, str]],
) -> CotSwapEditCountRun | None:
    if manifest.get("operation") != "cot-swap":
        return None
    arguments = _mapping(manifest.get("arguments"), context=f"{path}.arguments")
    if arguments.get("targeting") != "attribution-4":
        return None
    model = _string(arguments.get("model"), context=f"{path}.arguments.model")
    benchmark = _string(arguments.get("benchmark"), context=f"{path}.arguments.benchmark")
    if (model, benchmark) not in expected_settings:
        return None
    protocol = _mapping(manifest.get("protocol"), context=f"{path}.protocol")
    edit_count = _source_num_edits(arguments, protocol, path=path)
    if edit_count not in edit_counts:
        return None
    if manifest.get("schema_version") != "cot-swap-run/v1":
        raise EditCountSensitivityInputError(f"{path}: unknown CoT-swap run schema")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise EditCountSensitivityInputError(f"{path}: final-paper SHA-256 mismatch")
    if manifest.get("status") != "completed":
        raise EditCountSensitivityInputError(f"{path}: CoT-swap run is not completed")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise EditCountSensitivityInputError(f"{path}: limited CoT-swap runs are not accepted")
    protocol_hash = _hash(manifest.get("protocol_sha256"), context=f"{path}.protocol_sha256")
    if protocol_hash != _canonical_sha256(dict(protocol)):
        raise EditCountSensitivityInputError(f"{path}: protocol SHA-256 does not match protocol")
    if dict(protocol) != cot_swap_protocol_for(edit_count):
        raise EditCountSensitivityInputError(
            f"{path}: protocol does not match the public CoT-swap protocol"
        )
    prepared = prepared_by_key.get((model, benchmark, edit_count))
    if prepared is None:
        raise EditCountSensitivityInputError(
            f"{path}: CoT-swap source has no matching validated prepare run"
        )
    _validate_cot_source(
        manifest.get("source"),
        context=f"{path}.source",
        prepared=prepared,
    )
    generation = _mapping(
        protocol.get("source_generation"), context=f"{path}.protocol.source_generation"
    )
    if generation.get("seed") != 42 or generation.get("max_new_tokens") != 512:
        raise EditCountSensitivityInputError(
            f"{path}: CoT-swap source generation protocol is not paper-compatible"
        )
    if protocol.get("restoration") != (
        "canonical-c-equals-canonical-a-conditioned-on-b-not-equal-a"
    ):
        raise EditCountSensitivityInputError(f"{path}: restoration definition mismatch")
    counts = _mapping(manifest.get("counts"), context=f"{path}.counts")
    if counts.get("failed_pairs") != 0 or manifest.get("failures") not in ([], None):
        raise EditCountSensitivityInputError(f"{path}: CoT-swap run contains failures")
    outputs = _validate_output_registry(path, manifest)
    records = _load_jsonl(outputs["cot_swap_records.jsonl"])
    if not records:
        raise EditCountSensitivityInputError(f"{path}: CoT-swap run has no records")
    previous: str | None = None
    records_by_id: dict[str, dict[str, object]] = {}
    for line_number, row in enumerate(records, 1):
        sample_id = _validate_event_record(
            row,
            path=outputs["cot_swap_records.jsonl"],
            line_number=line_number,
            model=model,
            benchmark=benchmark,
            edit_count=edit_count,
            prepared=prepared,
        )
        if previous is not None and sample_id <= previous:
            raise EditCountSensitivityInputError(
                f"{outputs['cot_swap_records.jsonl']}: sample IDs must be sorted and unique"
            )
        previous = sample_id
        records_by_id[sample_id] = row
    statuses = _load_jsonl(outputs["pair_status_records.jsonl"])
    status_ids: list[str] = []
    completed_status_ids: set[str] = set()
    statuses_by_id: dict[str, Mapping[str, object]] = {}
    for line_number, row in enumerate(statuses, 1):
        context = f"{outputs['pair_status_records.jsonl']}:{line_number}"
        for field, expected in (
            ("schema_version", "cot-swap-pair-status/v1"),
            ("paper_sha256", PAPER_SHA256),
            ("model", model),
            ("benchmark", benchmark),
            ("targeting", "attribution-4"),
        ):
            if row.get(field) != expected:
                raise EditCountSensitivityInputError(f"{context}: invalid {field}")
        sample_id = _string(row.get("sample_id"), context=f"{context}.sample_id")
        if sample_id in statuses_by_id:
            raise EditCountSensitivityInputError(f"{context}: duplicate status sample ID")
        expected_record_hash = prepared.record_sha256_by_id.get(sample_id)
        if expected_record_hash is None:
            raise EditCountSensitivityInputError(
                f"{context}: status sample ID is absent from the validated prepare run"
            )
        if row.get("source_record_sha256") != expected_record_hash:
            raise EditCountSensitivityInputError(
                f"{context}: status source record SHA-256 mismatch"
            )
        for field in (
            "edit_valid",
            "template_eligible",
            "selected_for_execution",
            "included_in_change_denominator",
            "included_in_restoration_denominator",
        ):
            if not isinstance(row.get(field), bool):
                raise EditCountSensitivityInputError(f"{context}.{field} must be boolean")
        exclusions = row.get("exclusion_reasons")
        if not isinstance(exclusions, list) or any(
            not isinstance(reason, str) for reason in exclusions
        ):
            raise EditCountSensitivityInputError(
                f"{context}.exclusion_reasons must be a list of strings"
            )
        execution_status = row.get("execution_status")
        if execution_status not in {
            "no-applied-edit",
            "template-excluded",
            "not-selected",
            "completed",
        }:
            raise EditCountSensitivityInputError(f"{context}: unknown execution_status")
        if execution_status == "completed":
            completed_status_ids.add(sample_id)
        status_ids.append(sample_id)
        statuses_by_id[sample_id] = row
    expected_status_ids = [str(row["sample_id"]) for row in prepared.records]
    if status_ids != expected_status_ids:
        raise EditCountSensitivityInputError(
            f"{outputs['pair_status_records.jsonl']}: statuses must exactly follow the "
            "validated prepare cohort"
        )
    if completed_status_ids != set(records_by_id):
        raise EditCountSensitivityInputError(
            f"{outputs['pair_status_records.jsonl']}: completed status IDs do not match "
            "CoT-swap record IDs"
        )
    for sample_id, status in statuses_by_id.items():
        execution_status = status["execution_status"]
        if execution_status != "completed":
            if (
                status["included_in_change_denominator"] is not False
                or status["included_in_restoration_denominator"] is not False
            ):
                raise EditCountSensitivityInputError(
                    f"{outputs['pair_status_records.jsonl']}: excluded statuses cannot enter "
                    "analysis denominators"
                )
            continue
        if not (
            status["edit_valid"] is True
            and status["template_eligible"] is True
            and status["selected_for_execution"] is True
        ):
            raise EditCountSensitivityInputError(
                f"{outputs['pair_status_records.jsonl']}: completed status flags contradict "
                "execution_status"
            )
        events = _mapping(records_by_id[sample_id]["events"], context="record.events")
        if (
            status["included_in_change_denominator"] is not events["clean_correct"]
            or status["included_in_restoration_denominator"]
            is not events["restoration_denominator"]
        ):
            raise EditCountSensitivityInputError(
                f"{outputs['pair_status_records.jsonl']}: status denominator flags contradict "
                "CoT-swap events"
            )
    expected_manifest_counts = {
        "source_records": len(prepared.records),
        "selected_pairs": len(records),
        "checkpointed_pairs": len(records),
        "failed_pairs": 0,
        "publication_failures": 0,
    }
    for field, expected in expected_manifest_counts.items():
        if counts.get(field) != expected or type(counts.get(field)) is not int:
            raise EditCountSensitivityInputError(
                f"{path}.counts.{field} does not match completed producer outputs"
            )
    summary = _load_json(outputs["cot_swap_summary.json"])
    for field, expected in (
        ("schema_version", "cot-swap-summary/v1"),
        ("paper_sha256", PAPER_SHA256),
        ("operation", "cot-swap"),
        ("model", model),
        ("benchmark", benchmark),
        ("targeting", "attribution-4"),
    ):
        if summary.get(field) != expected:
            raise EditCountSensitivityInputError(
                f"{outputs['cot_swap_summary.json']}: invalid {field}"
            )
    if summary.get("source_num_edits", edit_count) != edit_count:
        raise EditCountSensitivityInputError(
            f"{outputs['cot_swap_summary.json']}: source edit count mismatch"
        )
    summary_counts = _mapping(
        summary.get("counts"), context=f"{outputs['cot_swap_summary.json']}.counts"
    )
    for field, expected in (
        ("source_records", len(prepared.records)),
        ("executed_pairs", len(records)),
        ("failed_pairs", 0),
    ):
        if summary_counts.get(field) != expected or type(summary_counts.get(field)) is not int:
            raise EditCountSensitivityInputError(
                f"{outputs['cot_swap_summary.json']}.counts.{field} contradicts outputs"
            )
    if summary.get("comparability") != manifest.get("comparability"):
        raise EditCountSensitivityInputError(
            f"{outputs['cot_swap_summary.json']}: comparability differs from producer manifest"
        )
    source = _mapping(summary.get("source"), context="cot-swap summary.source")
    source_pairs_hash = _hash(
        source.get("pairs_sha256"), context="cot-swap summary.source.pairs_sha256"
    )
    source_run_hash = _hash(
        source.get("source_run_sha256"),
        context="cot-swap summary.source.source_run_sha256",
    )
    _validate_cot_source(
        source,
        context=f"{outputs['cot_swap_summary.json']}.source",
        prepared=prepared,
    )
    metrics = _mapping(summary.get("metrics"), context="cot-swap summary.metrics")
    restoration = _mapping(
        metrics.get("restoration"), context="cot-swap summary.metrics.restoration"
    )
    expected_metric = _metric_from_events(records)
    if dict(restoration) != expected_metric:
        raise EditCountSensitivityInputError(
            f"{outputs['cot_swap_summary.json']}: restoration metric contradicts records"
        )
    rate = restoration.get("rate")
    if rate is not None and (not isinstance(rate, (int, float)) or not math.isfinite(rate)):
        raise EditCountSensitivityInputError(
            f"{outputs['cot_swap_summary.json']}: restoration rate is not finite"
        )
    return CotSwapEditCountRun(
        model=model,
        benchmark=benchmark,
        edit_count=edit_count,
        records=records,
        run_path=path.resolve(),
        records_path=outputs["cot_swap_records.jsonl"].resolve(),
        summary_path=outputs["cot_swap_summary.json"].resolve(),
        run_sha256=_sha256(path),
        records_sha256=_sha256(outputs["cot_swap_records.jsonl"]),
        source_pairs_sha256=source_pairs_hash,
        source_run_sha256=source_run_hash,
        source_record_count=len(prepared.records),
    )


def discover_cot_swap_runs(
    root: Path,
    *,
    edit_counts: Sequence[int],
    prepared_runs: Sequence[PreparedEditCountRun],
    expected_settings: Sequence[tuple[str, str]] = EXPECTED_RESTORATION_SETTINGS,
) -> tuple[CotSwapEditCountRun, ...]:
    """Discover the six-setting restoration grid and bind every run to its source."""
    root = Path(root)
    if not root.is_dir():
        raise EditCountSensitivityInputError(f"cot_swap_runs_root is not a directory: {root}")
    requested = frozenset(edit_counts)
    prepared = {(run.model, run.benchmark, run.edit_count): run for run in prepared_runs}
    settings = frozenset(expected_settings)
    found: dict[tuple[str, str, int], CotSwapEditCountRun] = {}
    for path in sorted(root.rglob("run.json")):
        manifest = _load_json(path)
        validated = _validate_cot_manifest(
            path,
            manifest,
            edit_counts=requested,
            prepared_by_key=prepared,
            expected_settings=settings,
        )
        if validated is None:
            continue
        key = (validated.model, validated.benchmark, validated.edit_count)
        if key in found:
            raise EditCountSensitivityInputError(
                "duplicate CoT-swap setting/edit-count input: "
                f"{validated.model} {validated.benchmark} k={validated.edit_count}"
            )
        found[key] = validated
    if not found:
        raise EditCountSensitivityInputError(
            "cot_swap_runs_root contains no applicable completed Attribution-4 runs"
        )
    return tuple(found[key] for key in sorted(found))


__all__ = [
    "CotSwapEditCountRun",
    "EditCountSensitivityInputError",
    "PreparedEditCountRun",
    "discover_cot_swap_runs",
    "discover_prepared_runs",
]
