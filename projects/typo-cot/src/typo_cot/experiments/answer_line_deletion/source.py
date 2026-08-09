"""Read-only validation and joining of a completed CoT-swap source run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.cot_swap import (
    COT_SWAP_BENCHMARKS,
    CotSwapConfig,
    CotSwapRunError,
    run_cot_swap,
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def strict_loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid JSON at {context}: {exc}") from exc


def load_json(path: Path) -> dict[str, object]:
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    value = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _load_jsonl_with_hashes(path: Path) -> tuple[tuple[dict[str, object], str], ...]:
    rows: list[tuple[dict[str, object], str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append((payload, hashlib.sha256(line.encode("utf-8")).hexdigest()))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class CotSwapSourceCase:
    """One semantically validated CoT-swap record joined to its prepared pair."""

    sample_id: str
    record: dict[str, object]
    record_sha256: str
    pair: dict[str, object]
    pair_record_sha256: str
    source_a_answer: str
    source_c_answer: str
    source_a_correct: bool
    source_b_changed_from_a: bool
    restoration_denominator: bool


@dataclass(frozen=True, slots=True)
class CompletedCotSwapSource:
    """Immutable fingerprints and cases from one validated upstream run."""

    directory: Path
    run_path: Path
    run_sha256: str
    records_path: Path
    records_sha256: str
    statuses_path: Path
    statuses_sha256: str
    summary_path: Path
    summary_sha256: str
    pairs_path: Path
    pairs_sha256: str
    pairs_run_path: Path
    pairs_run_sha256: str
    model: str
    benchmark: str
    targeting: str
    model_revision: str
    cases: tuple[CotSwapSourceCase, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cot_swap_run": str(self.directory.resolve()),
            "cot_swap_run_manifest": str(self.run_path.resolve()),
            "cot_swap_run_sha256": self.run_sha256,
            "cot_swap_records": str(self.records_path.resolve()),
            "cot_swap_records_sha256": self.records_sha256,
            "cot_swap_pair_status_records": str(self.statuses_path.resolve()),
            "cot_swap_pair_status_records_sha256": self.statuses_sha256,
            "cot_swap_summary": str(self.summary_path.resolve()),
            "cot_swap_summary_sha256": self.summary_sha256,
            "prepared_pairs": str(self.pairs_path.resolve()),
            "prepared_pairs_sha256": self.pairs_sha256,
            "prepared_pairs_run": str(self.pairs_run_path.resolve()),
            "prepared_pairs_run_sha256": self.pairs_run_sha256,
            "source_schema": "cot-swap-record/v1",
            "prepared_pair_schema": "prepare-edited-pairs/v1",
            "model_revision": self.model_revision,
            "record_count": len(self.cases),
        }


def validate_source_snapshot(source: CompletedCotSwapSource) -> None:
    """Require every upstream file to retain the bytes validated at load time."""

    files = (
        ("CoT-swap run manifest", source.run_path, source.run_sha256),
        ("CoT-swap records", source.records_path, source.records_sha256),
        ("CoT-swap pair statuses", source.statuses_path, source.statuses_sha256),
        ("CoT-swap summary", source.summary_path, source.summary_sha256),
        ("prepared pairs", source.pairs_path, source.pairs_sha256),
        ("prepared-pair run manifest", source.pairs_run_path, source.pairs_run_sha256),
    )
    for label, path, expected_sha256 in files:
        try:
            actual_sha256 = sha256_file(path)
        except OSError as exc:
            raise ValueError(f"source snapshot changed: {label} is unavailable: {path}") from exc
        if actual_sha256 != expected_sha256:
            raise ValueError(f"source snapshot changed: {label} SHA-256 mismatch: {path}")


def _source_config(
    manifest: Mapping[str, object],
    directory: Path,
    *,
    expected_model: str,
    expected_benchmark: str,
) -> CotSwapConfig:
    if manifest.get("schema_version") != "cot-swap-run/v1":
        raise ValueError("source CoT-swap run has an unknown schema")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("source CoT-swap run paper SHA-256 does not match")
    if manifest.get("operation") != "cot-swap":
        raise ValueError("source CoT-swap run has the wrong operation")
    if manifest.get("status") != "completed":
        raise ValueError("source CoT-swap run must be completed")
    arguments = _mapping(manifest.get("arguments"), field="source CoT-swap arguments")
    if arguments.get("model") != expected_model:
        raise ValueError("source CoT-swap run model does not match")
    if arguments.get("benchmark") != expected_benchmark:
        raise ValueError("source CoT-swap run benchmark does not match")
    if arguments.get("targeting") != "random-4":
        raise ValueError("source CoT-swap run must use random-4")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise ValueError("source CoT-swap run must be unlimited")
    output_dir = Path(_string(arguments.get("output_dir"), field="source output_dir"))
    if output_dir.resolve() != directory.resolve():
        raise ValueError("source CoT-swap run directory does not match its recorded path")
    benchmark = _string(arguments.get("benchmark"), field="source benchmark")
    if benchmark not in COT_SWAP_BENCHMARKS:
        raise ValueError("source CoT-swap benchmark is unsupported")
    gpu_id = _string(arguments.get("gpu_id"), field="source gpu_id")
    return CotSwapConfig(
        model=_string(arguments.get("model"), field="source model"),
        benchmark=cast(Literal["gsm8k", "mmlu", "mmlu-pro", "arc", "csqa"], benchmark),
        pairs=Path(_string(arguments.get("pairs"), field="source pairs")),
        targeting="random-4",
        output_dir=directory,
        gpu_id=gpu_id,
        limit=None,
        resume=True,
    )


def load_completed_cot_swap_source(
    directory: Path,
    *,
    model: str,
    benchmark: str,
) -> CompletedCotSwapSource:
    """Validate all upstream bytes and semantics without loading model weights."""

    directory = Path(directory)
    run_path = directory / "run.json"
    if not run_path.is_file():
        raise ValueError(f"source CoT-swap run is missing run.json: {directory}")
    initial_run_sha256 = sha256_file(run_path)
    manifest = load_json(run_path)
    config = _source_config(
        manifest,
        directory,
        expected_model=model,
        expected_benchmark=benchmark,
    )
    try:
        result = run_cot_swap(config)
    except (CotSwapRunError, FileExistsError, RuntimeError, ValueError) as exc:
        raise ValueError(f"source CoT-swap completed run validation failed: {exc}") from exc

    # The completed validator above checks output hashes and reconstructs every
    # record, status, summary, and checkpoint. Read them only after that gate.
    if sha256_file(run_path) != initial_run_sha256:
        raise ValueError("source CoT-swap run changed during validation")
    manifest = load_json(run_path)
    if sha256_file(run_path) != initial_run_sha256:
        raise ValueError("source CoT-swap run changed while reloading its manifest")
    source_payload = _mapping(manifest.get("source"), field="source CoT-swap source")
    outputs = _mapping(manifest.get("outputs"), field="source CoT-swap outputs")
    pairs_path = config.pairs
    pairs_run_path = pairs_path.parent / "run.json"
    expected_pairs_sha = _string(
        source_payload.get("pairs_sha256"), field="source prepared pairs SHA-256"
    )
    if sha256_file(pairs_path) != expected_pairs_sha:
        raise ValueError("source prepared pairs hash changed after validation")
    expected_pairs_run_sha = _string(
        source_payload.get("source_run_sha256"),
        field="source prepared-pair run SHA-256",
    )
    actual_pairs_run_sha = sha256_file(pairs_run_path)
    if actual_pairs_run_sha != expected_pairs_run_sha:
        raise ValueError("source prepared-pair run changed during validation")

    def output_sha(name: str, path: Path) -> str:
        metadata = _mapping(outputs.get(name), field=f"source output {name}")
        expected = _string(metadata.get("sha256"), field=f"source output {name} SHA-256")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source output changed after validation: {name}")
        return actual

    records_sha256 = output_sha(result.records_path.name, result.records_path)
    statuses_sha256 = output_sha(
        result.pair_status_records_path.name, result.pair_status_records_path
    )
    summary_sha256 = output_sha(result.summary_path.name, result.summary_path)

    pair_rows = _load_jsonl_with_hashes(pairs_path)
    if sha256_file(pairs_path) != expected_pairs_sha:
        raise ValueError("source prepared pairs changed while parsing")
    pairs_by_id: dict[str, tuple[dict[str, object], str]] = {}
    for pair, fingerprint in pair_rows:
        sample_id = _string(pair.get("sample_id"), field="prepared pair sample_id")
        if sample_id in pairs_by_id:
            raise ValueError("source prepared pairs contain a duplicate sample ID")
        pairs_by_id[sample_id] = (pair, fingerprint)

    record_rows = _load_jsonl_with_hashes(result.records_path)
    if sha256_file(result.records_path) != records_sha256:
        raise ValueError("source CoT-swap records changed while parsing")
    cases: list[CotSwapSourceCase] = []
    previous_id: str | None = None
    for record, record_sha256 in record_rows:
        sample_id = _string(record.get("sample_id"), field="CoT-swap record sample_id")
        if previous_id is not None and sample_id <= previous_id:
            raise ValueError("source CoT-swap record sample IDs must be sorted and unique")
        previous_id = sample_id
        if sample_id not in pairs_by_id:
            raise ValueError("source CoT-swap record has no prepared pair")
        pair, pair_sha256 = pairs_by_id[sample_id]
        record_source = _mapping(record.get("source"), field="CoT-swap record source")
        if record_source.get("source_record_sha256") != pair_sha256:
            raise ValueError("source CoT-swap record/pair fingerprint mismatch")
        cells = _mapping(record.get("cells"), field="CoT-swap record cells")
        cell_a = _mapping(cells.get("A"), field="CoT-swap cell A")
        cell_c = _mapping(cells.get("C"), field="CoT-swap cell C")
        answer_a = _mapping(cell_a.get("answer"), field="CoT-swap cell A answer")
        answer_c = _mapping(cell_c.get("answer"), field="CoT-swap cell C answer")
        source_a_answer = _text(answer_a.get("value"), field="source A answer")
        source_c_answer = _text(answer_c.get("value"), field="source C answer")
        events = _mapping(record.get("events"), field="CoT-swap record events")
        source_a_correct = events.get("clean_correct")
        source_b_changed = events.get("both_changed")
        denominator = events.get("restoration_denominator")
        if not all(
            isinstance(value, bool) for value in (source_a_correct, source_b_changed, denominator)
        ):
            raise ValueError("CoT-swap denominator events must be boolean")
        if denominator is not (source_a_correct and source_b_changed):
            raise ValueError("CoT-swap restoration denominator events are inconsistent")
        if denominator and not source_a_answer:
            raise ValueError("CoT-swap denominator case must have an extracted source A answer")
        cases.append(
            CotSwapSourceCase(
                sample_id=sample_id,
                record=record,
                record_sha256=record_sha256,
                pair=pair,
                pair_record_sha256=pair_sha256,
                source_a_answer=source_a_answer,
                source_c_answer=source_c_answer,
                source_a_correct=source_a_correct,
                source_b_changed_from_a=source_b_changed,
                restoration_denominator=denominator,
            )
        )
    if not cases:
        raise ValueError("source CoT-swap run contains no records")

    runtime = _mapping(manifest.get("runtime"), field="source CoT-swap runtime")
    model_revision = _string(runtime.get("model_revision"), field="source model revision")

    source = CompletedCotSwapSource(
        directory=directory,
        run_path=run_path,
        run_sha256=initial_run_sha256,
        records_path=result.records_path,
        records_sha256=records_sha256,
        statuses_path=result.pair_status_records_path,
        statuses_sha256=statuses_sha256,
        summary_path=result.summary_path,
        summary_sha256=summary_sha256,
        pairs_path=pairs_path,
        pairs_sha256=expected_pairs_sha,
        pairs_run_path=pairs_run_path,
        pairs_run_sha256=actual_pairs_run_sha,
        model=config.model,
        benchmark=config.benchmark,
        targeting=config.targeting,
        model_revision=model_revision,
        cases=tuple(cases),
    )
    validate_source_snapshot(source)
    return source
