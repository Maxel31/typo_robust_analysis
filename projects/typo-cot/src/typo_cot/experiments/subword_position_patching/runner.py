"""Resumable runner for the frozen first/final/all-subword comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
    load_rebuttal_pair_manifest,
)
from typo_cot.experiments.build_rebuttal_manifest.records import (
    iter_jsonl_objects,
    load_json_object,
    sha256_file,
)
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.subword_position_patching.planning import plan_subword_patch
from typo_cot.experiments.subword_position_patching.protocol import (
    SubwordPositionPatchingProtocol,
    load_subword_position_patching_protocol,
)

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RUN_SCHEMA = "subword-position-patching-run/v1"
_CHECKPOINT_SCHEMA = "subword-position-patching-checkpoint/v1"
_RECORD_SCHEMA = "subword-position-patching-record/v1"
_SUMMARY_SCHEMA = "subword-position-patching-summary/v1"
_SUBSETS = ("equal-count-primary", "mismatch-monotone-secondary")
_PUBLIC_OUTPUTS = (
    "subword_patch_records.jsonl",
    "subword_patch_table.csv",
    "subword_alignment_flow.csv",
    "subword_patch_summary.json",
)


class SubwordPositionPatchingRunError(RuntimeError):
    """Raised after verified pair checkpoints have been retained."""


@dataclass(frozen=True, slots=True)
class SubwordPositionPatchingConfig:
    protocol_path: Path
    manifest_path: Path
    modes: tuple[str, ...]
    token_count_policy: str
    gpu_id: str
    output_dir: Path
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("protocol_path", "manifest_path", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        object.__setattr__(self, "modes", tuple(self.modes))
        if self.modes != ("first", "final", "all") or len(self.modes) != len(set(self.modes)):
            raise ValueError("modes must be ordered exactly as first, final, all")
        if self.token_count_policy != "equal-count-primary":
            raise ValueError("token_count_policy must be equal-count-primary")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be boolean")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("protocol_path", "manifest_path", "output_dir"):
            payload[field] = str(Path(payload[field]).resolve())
        payload["modes"] = list(self.modes)
        payload.pop("resume")
        return payload


def _positions(values: Sequence[int], *, field: str, allow_duplicates: bool) -> tuple[int, ...]:
    try:
        positions = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be an integer sequence") from exc
    if not positions or any(
        isinstance(position, bool) or not isinstance(position, int) or position < 0
        for position in positions
    ):
        raise ValueError(f"{field} must contain non-negative integers")
    if not allow_duplicates and len(positions) != len(set(positions)):
        raise ValueError(f"{field} contains duplicate positions")
    return positions


@dataclass(frozen=True, slots=True)
class SubwordGeneration:
    token_ids: tuple[int, ...]
    text: str
    termination: Literal["eos", "length-cap"]
    value: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not self.token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("subword generation token_ids must be non-empty integers")
        if not isinstance(self.text, str) or not isinstance(self.value, str):
            raise TypeError("subword generation text and value must be strings")
        if self.termination not in {"eos", "length-cap"}:
            raise ValueError("subword generation termination must be eos or length-cap")
        if type(self.is_extracted) is not bool or self.is_extracted is not bool(self.value):
            raise ValueError("subword generation extraction flag differs from its value")
        if type(self.is_correct) is not bool or (self.is_correct and not self.is_extracted):
            raise ValueError("subword generation correctness differs from extraction flag")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("subword generation method must be non-empty")
        if not isinstance(self.primary_method, str) or not self.primary_method:
            raise ValueError("subword generation primary_method must be non-empty")


@dataclass(frozen=True, slots=True)
class SubwordModeScan:
    generation: SubwordGeneration
    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.generation, SubwordGeneration):
            raise TypeError("generation must be SubwordGeneration")
        object.__setattr__(
            self,
            "source_positions",
            _positions(self.source_positions, field="source_positions", allow_duplicates=True),
        )
        object.__setattr__(
            self,
            "destination_positions",
            _positions(
                self.destination_positions,
                field="destination_positions",
                allow_duplicates=False,
            ),
        )
        if len(self.source_positions) != len(self.destination_positions):
            raise ValueError("subword coordinate cardinality differs")


@dataclass(frozen=True, slots=True)
class SubwordPairScan:
    scans: Mapping[str, SubwordModeScan]

    def __post_init__(self) -> None:
        if not isinstance(self.scans, Mapping) or not self.scans:
            raise ValueError("subword pair scan must contain one or more modes")
        normalized = dict(self.scans)
        if set(normalized) - {"first", "final", "all"} or any(
            not isinstance(scan, SubwordModeScan) for scan in normalized.values()
        ):
            raise ValueError("subword pair scan contains an invalid mode payload")
        object.__setattr__(self, "scans", normalized)


class SubwordPositionPatchingRuntime(Protocol):
    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_pair(
        self, pair: Mapping[str, object], *, modes: tuple[str, ...]
    ) -> SubwordPairScan: ...


RuntimeFactory = Callable[..., SubwordPositionPatchingRuntime]


@dataclass(frozen=True, slots=True)
class SubwordPositionPatchingResult:
    records_path: Path
    table_path: Path
    flow_path: Path
    summary_path: Path
    run_path: Path
    pairs: int
    evaluated_pairs: int
    primary_pairs: int
    secondary_pairs: int
    table_rows: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_input_sha256(record: Mapping[str, object]) -> str:
    return _canonical_sha256(record)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_factory() -> RuntimeFactory:
    from typo_cot.experiments.subword_position_patching.runtime import (
        HuggingFaceSubwordPositionPatchingRuntime,
    )

    return HuggingFaceSubwordPositionPatchingRuntime


def _runtime_provenance(
    runtime: SubwordPositionPatchingRuntime,
    *,
    revision: str,
    gpu_id: str,
    protocol: SubwordPositionPatchingProtocol,
) -> dict[str, object]:
    provenance = dict(runtime.provenance())
    if (
        provenance.get("operation") != "subword-position-patching"
        or provenance.get("model") != protocol.model
        or provenance.get("task") != protocol.task
        or provenance.get("requested_revision") != revision
        or provenance.get("model_revision") != revision
        or provenance.get("tokenizer_revision") != revision
        or provenance.get("num_decoder_layers") != runtime.num_layers
        or isinstance(runtime.num_layers, bool)
        or not isinstance(runtime.num_layers, int)
        or runtime.num_layers < protocol.window[1]
        or provenance.get("dtype") != protocol.dtype
        or provenance.get("cuda_visible_devices") != gpu_id
        or provenance.get("direction") != protocol.direction
        or provenance.get("coordinate_source") != protocol.coordinate_source
        or provenance.get("layer_window") != list(protocol.window)
        or provenance.get("modes") != list(protocol.modes)
        or provenance.get("token_count_policy") != protocol.token_count_policy
        or provenance.get("secondary_alignment") != protocol.secondary_alignment
        or provenance.get("generation_termination_protocol") != protocol.termination_protocol
        or provenance.get("answer_extraction") != protocol.answer_extraction
    ):
        raise ValueError("subword runtime provenance differs from the frozen protocol")
    eos_ids = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in eos_ids
        )
        or eos_ids != sorted(set(eos_ids))
        or not isinstance(provenance.get("effective_eos_token_ids_source"), str)
    ):
        raise ValueError("subword runtime effective EOS provenance differs")
    generation = _mapping(provenance.get("generation"), field="runtime generation")
    expected_generation = {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": protocol.max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
        "padding_side": "left",
        "patch_application": "layers-0-5-on-typo-prompt-prefill-exactly-once/v1",
    }
    if dict(generation) != expected_generation:
        raise ValueError("subword runtime generation provenance differs")
    return provenance


def _generation_payload(generation: SubwordGeneration) -> dict[str, object]:
    return {
        "token_ids": list(generation.token_ids),
        "text": generation.text,
        "termination": generation.termination,
        "value": generation.value,
        "is_extracted": generation.is_extracted,
        "is_correct": generation.is_correct,
        "method": generation.method,
        "primary_method": generation.primary_method,
    }


def _generation_from_payload(value: object, *, field: str) -> SubwordGeneration:
    payload = _mapping(value, field=field)
    if set(payload) != {
        "token_ids",
        "text",
        "termination",
        "value",
        "is_extracted",
        "is_correct",
        "method",
        "primary_method",
    } or not isinstance(payload.get("token_ids"), list):
        raise ValueError(f"{field} fields differ from the checkpoint contract")
    return SubwordGeneration(
        token_ids=tuple(payload["token_ids"]),  # type: ignore[arg-type]
        text=payload["text"],  # type: ignore[arg-type]
        termination=payload["termination"],  # type: ignore[arg-type]
        value=payload["value"],  # type: ignore[arg-type]
        is_extracted=payload["is_extracted"],  # type: ignore[arg-type]
        is_correct=payload["is_correct"],  # type: ignore[arg-type]
        method=payload["method"],  # type: ignore[arg-type]
        primary_method=payload["primary_method"],  # type: ignore[arg-type]
    )


def _scan_payload(scan: SubwordPairScan) -> dict[str, object]:
    return {
        mode: {
            "generation": _generation_payload(result.generation),
            "source_positions": list(result.source_positions),
            "destination_positions": list(result.destination_positions),
        }
        for mode, result in sorted(scan.scans.items())
    }


def _scan_from_payload(value: object, *, field: str) -> SubwordPairScan:
    payload = _mapping(value, field=field)
    scans: dict[str, SubwordModeScan] = {}
    for mode, raw in payload.items():
        item = _mapping(raw, field=f"{field}.{mode}")
        if set(item) != {"generation", "source_positions", "destination_positions"}:
            raise ValueError(f"{field}.{mode} fields differ")
        if not isinstance(item["source_positions"], list) or not isinstance(
            item["destination_positions"], list
        ):
            raise TypeError(f"{field}.{mode} positions must be lists")
        scans[mode] = SubwordModeScan(
            generation=_generation_from_payload(
                item["generation"], field=f"{field}.{mode}.generation"
            ),
            source_positions=tuple(item["source_positions"]),  # type: ignore[arg-type]
            destination_positions=tuple(item["destination_positions"]),  # type: ignore[arg-type]
        )
    return SubwordPairScan(scans=scans)


def _validate_scan(
    scan: object, *, record: Mapping[str, object], modes: tuple[str, ...]
) -> SubwordPairScan:
    if not isinstance(scan, SubwordPairScan) or set(scan.scans) != set(modes):
        raise ValueError("runtime did not return the ordered complete subword mode inventory")
    edits = record.get("edits")
    if not isinstance(edits, list):
        raise TypeError("record edits must be a list")
    for mode in modes:
        plan = plan_subword_patch(edits, mode=mode)
        result = scan.scans[mode]
        if (
            result.source_positions != plan.source_positions
            or result.destination_positions != plan.destination_positions
        ):
            raise ValueError(f"runtime {mode} coordinates differ from the frozen plan")
        expected_correct = answers_equal(
            result.generation.value,
            str(record.get("gold_answer")),
            benchmark=str(record.get("task")),
        )
        if result.generation.is_correct is not expected_correct:
            raise ValueError(f"runtime {mode} correctness differs from its answer")
    return scan


def _checkpoint_path(
    directory: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol_sha256: str,
    manifest_sha256: str,
) -> Path:
    identity = _canonical_sha256(
        {
            "pair_input_sha256": _record_input_sha256(record),
            "runtime_fingerprint": runtime_fingerprint,
            "protocol_sha256": protocol_sha256,
            "manifest_sha256": manifest_sha256,
        }
    )
    return directory / f"{identity}.json"


def _write_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    scan: SubwordPairScan,
    runtime_fingerprint: str,
    protocol: SubwordPositionPatchingProtocol,
    manifest_sha256: str,
) -> None:
    scan_payload = _scan_payload(scan)
    _write_json_atomic(
        path,
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
            "protocol_sha256": protocol.config_sha256,
            "pair_manifest_sha256": manifest_sha256,
            "pair_id": record.get("pair_id"),
            "pair_input_sha256": _record_input_sha256(record),
            "runtime_fingerprint": runtime_fingerprint,
            "modes": list(protocol.modes),
            "scan": scan_payload,
            "scan_sha256": _canonical_sha256(scan_payload),
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: SubwordPositionPatchingProtocol,
    manifest_sha256: str,
) -> SubwordPairScan:
    payload = load_json_object(path)
    scan_payload = payload.get("scan")
    if (
        payload.get("schema_version") != _CHECKPOINT_SCHEMA
        or payload.get("paper_sha256") != PAPER_SHA256
        or payload.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or payload.get("protocol_sha256") != protocol.config_sha256
        or payload.get("pair_manifest_sha256") != manifest_sha256
        or payload.get("pair_id") != record.get("pair_id")
        or payload.get("pair_input_sha256") != _record_input_sha256(record)
        or payload.get("runtime_fingerprint") != runtime_fingerprint
        or payload.get("modes") != list(protocol.modes)
        or payload.get("scan_sha256") != _canonical_sha256(scan_payload)
    ):
        raise ValueError(f"subword checkpoint provenance differs: {path}")
    return _validate_scan(
        _scan_from_payload(scan_payload, field=f"{path} scan"),
        record=record,
        modes=protocol.modes,
    )


def _select_records(
    records: Sequence[Mapping[str, object]],
    *,
    protocol: SubwordPositionPatchingProtocol,
    limit: int | None,
) -> tuple[Mapping[str, object], ...]:
    paper = next(
        setting for setting in REBUTTAL_SETTINGS if setting.key == (protocol.model, protocol.task)
    )
    selected = [
        record
        for record in records
        if (record.get("model"), record.get("task")) == paper.key
        and _mapping(record.get("cohorts"), field="record cohorts").get("restoration") is True
    ]
    if len(selected) != paper.paper_denominator or len(selected) != protocol.cohort_pairs:
        raise ValueError("subword-position primary paper denominator differs")
    selected.sort(
        key=lambda record: (
            str(record.get("target_rule")),
            str(record.get("sample_id")),
            str(record.get("pair_id")),
        )
    )
    selected = selected if limit is None else selected[:limit]
    if limit is not None and len(selected) != limit:
        raise ValueError("subword-position smoke limit exceeds the primary cohort")
    for record in selected:
        if record.get("clean_correct") is not True or record.get("typo_correct") is not False:
            raise ValueError("subword-position cohort escaped clean-correct/typo-wrong")
        if not isinstance(record.get("clean_answer"), str) or not record.get("clean_answer"):
            raise ValueError("subword-position clean target answer must be extracted")
        edits = record.get("edits")
        if not isinstance(edits, list):
            raise TypeError("subword-position record edits must be a list")
        for mode in protocol.modes:
            plan_subword_patch(edits, mode=mode)
    return tuple(selected)


def _compile_records(
    *,
    selected: Sequence[Mapping[str, object]],
    checkpoints_dir: Path,
    runtime_fingerprint: str,
    protocol: SubwordPositionPatchingProtocol,
    manifest_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in selected:
        checkpoint = _checkpoint_path(
            checkpoints_dir,
            record=record,
            runtime_fingerprint=runtime_fingerprint,
            protocol_sha256=protocol.config_sha256,
            manifest_sha256=manifest_sha256,
        )
        scan = _load_checkpoint(
            checkpoint,
            record=record,
            runtime_fingerprint=runtime_fingerprint,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
        )
        edits = record["edits"]
        source = _mapping(record.get("source"), field="record source")
        historical = _mapping(record.get("fixed_window"), field="record fixed_window").get("event")
        for mode in protocol.modes:
            plan = plan_subword_patch(edits, mode=mode)  # type: ignore[arg-type]
            result = scan.scans[mode]
            success = bool(
                result.generation.is_extracted
                and answers_equal(
                    result.generation.value,
                    str(record["clean_answer"]),
                    benchmark=str(record["task"]),
                )
            )
            rows.append(
                {
                    "schema_version": _RECORD_SCHEMA,
                    "paper_sha256": PAPER_SHA256,
                    "protocol_sha256": protocol.config_sha256,
                    "pair_manifest_sha256": manifest_sha256,
                    "runtime_fingerprint": runtime_fingerprint,
                    "pair_input_sha256": _record_input_sha256(record),
                    "source_record_sha256": source.get("source_record_sha256"),
                    "pair_id": record["pair_id"],
                    "sample_id": record["sample_id"],
                    "model": record["model"],
                    "task": record["task"],
                    "target_rule": record["target_rule"],
                    "analysis_subset": plan.analysis_subset,
                    "mode": mode,
                    "alignment": plan.alignment,
                    "source_positions": list(result.source_positions),
                    "destination_positions": list(result.destination_positions),
                    "source_to_destination": [
                        {"source": donor, "destination": recipient}
                        for donor, recipient in zip(
                            result.source_positions,
                            result.destination_positions,
                            strict=True,
                        )
                    ],
                    "clean_reference_answer": record["clean_answer"],
                    "generation": _generation_payload(result.generation),
                    "success": success,
                    "historical_final_event": historical,
                }
            )
    return rows


def _analysis_rows(
    rows: Sequence[Mapping[str, object]], *, modes: Sequence[str]
) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for subset in _SUBSETS:
        for mode in modes:
            cell = [row for row in rows if row["analysis_subset"] == subset and row["mode"] == mode]
            successes = sum(int(row["success"] is True) for row in cell)
            unextractable = sum(int(row["generation"]["is_extracted"] is False) for row in cell)
            compared = len(cell) if mode == "final" else 0
            agrees = (
                sum(int(row["success"] is row["historical_final_event"]) for row in cell)
                if mode == "final"
                else 0
            )
            table.append(
                {
                    "analysis_subset": subset,
                    "mode": mode,
                    "n_pairs": len(cell),
                    "successes": successes,
                    "restoration_rate": successes / len(cell) if cell else None,
                    "restoration_rate_defined": bool(cell),
                    "unextractable": unextractable,
                    "historical_final_compared": compared,
                    "historical_final_agrees": agrees,
                    "historical_final_disagrees": compared - agrees,
                }
            )
    return table


def _result_from_run(output_dir: Path, run: Mapping[str, object]) -> SubwordPositionPatchingResult:
    outputs = _mapping(run.get("outputs"), field="completed run outputs")
    if set(outputs) != set(_PUBLIC_OUTPUTS):
        raise ValueError("completed subword output inventory differs")
    for name in _PUBLIC_OUTPUTS:
        path = output_dir / name
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"completed subword output SHA-256 differs: {path}")
    records = tuple(
        row for _number, _line, row in iter_jsonl_objects(output_dir / _PUBLIC_OUTPUTS[0])
    )
    with (output_dir / _PUBLIC_OUTPUTS[1]).open(encoding="utf-8", newline="") as handle:
        table_rows = tuple(csv.DictReader(handle))
    with (output_dir / _PUBLIC_OUTPUTS[2]).open(encoding="utf-8", newline="") as handle:
        flow_rows = tuple(csv.DictReader(handle))
    summary = load_json_object(output_dir / _PUBLIC_OUTPUTS[3])
    metadata_counts = {
        _PUBLIC_OUTPUTS[0]: len(records),
        _PUBLIC_OUTPUTS[1]: len(table_rows),
        _PUBLIC_OUTPUTS[2]: len(flow_rows),
        _PUBLIC_OUTPUTS[3]: 1,
    }
    for name, count in metadata_counts.items():
        if _mapping(outputs[name], field=f"completed output {name}").get("records") != count:
            raise ValueError(f"completed subword output count differs: {name}")
    counts = _mapping(run.get("counts"), field="completed run counts")
    expected = {
        "cohort_pairs": int(summary["counts"]["cohort_pairs"]),
        "evaluated_pairs": int(summary["counts"]["evaluated_pairs"]),
        "primary_pairs": int(summary["counts"]["primary_pairs"]),
        "secondary_pairs": int(summary["counts"]["secondary_pairs"]),
        "records": len(records),
        "table_rows": len(table_rows),
        "flow_rows": len(flow_rows),
    }
    if dict(counts) != expected or len(table_rows) != 6 or len(flow_rows) != 1:
        raise ValueError("completed subword counts differ from verified outputs")
    return SubwordPositionPatchingResult(
        records_path=output_dir / _PUBLIC_OUTPUTS[0],
        table_path=output_dir / _PUBLIC_OUTPUTS[1],
        flow_path=output_dir / _PUBLIC_OUTPUTS[2],
        summary_path=output_dir / _PUBLIC_OUTPUTS[3],
        run_path=output_dir / "run.json",
        pairs=int(counts["cohort_pairs"]),
        evaluated_pairs=int(counts["evaluated_pairs"]),
        primary_pairs=int(counts["primary_pairs"]),
        secondary_pairs=int(counts["secondary_pairs"]),
        table_rows=int(counts["table_rows"]),
    )


def _completed_resume(
    config: SubwordPositionPatchingConfig,
    protocol: SubwordPositionPatchingProtocol,
) -> SubwordPositionPatchingResult | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "subword-position-patching"
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("pair_manifest_sha256") != sha256_file(config.manifest_path)
    ):
        raise ValueError("completed subword resume contract differs")
    return _result_from_run(config.output_dir, run)


def run_subword_position_patching(
    config: SubwordPositionPatchingConfig,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> SubwordPositionPatchingResult:
    """Generate all modes, checkpoint each pair, and compile separated subsets."""

    if not isinstance(config, SubwordPositionPatchingConfig):
        raise TypeError("config must be SubwordPositionPatchingConfig")
    protocol = load_subword_position_patching_protocol(config.protocol_path)
    if config.modes != protocol.modes:
        raise ValueError("--modes differs from the frozen protocol")
    if config.token_count_policy != protocol.token_count_policy:
        raise ValueError("--token-count-policy differs from the frozen protocol")
    completed = _completed_resume(config, protocol)
    if completed is not None:
        return completed
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.resume:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        if not (output_dir / "run.json").is_file():
            raise ValueError("non-empty resume directory is missing its bound run.json")
    if not config.manifest_path.is_file():
        raise ValueError(f"rebuttal manifest is not a file: {config.manifest_path}")
    manifest_sha256 = sha256_file(config.manifest_path)
    all_records = load_rebuttal_pair_manifest(config.manifest_path)
    selected = _select_records(all_records, protocol=protocol, limit=config.limit)
    confirmatory = config.limit is None and len(selected) == protocol.cohort_pairs
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    run_path = output_dir / "run.json"
    started_at = _now()
    if config.resume and run_path.is_file():
        previous = load_json_object(run_path)
        if (
            previous.get("schema_version") != _RUN_SCHEMA
            or previous.get("paper_sha256") != PAPER_SHA256
            or previous.get("operation") != "subword-position-patching"
            or previous.get("status") not in {"running", "failed"}
            or previous.get("arguments") != config.public_arguments()
            or previous.get("protocol") != protocol.as_dict()
            or previous.get("protocol_sha256") != protocol.config_sha256
            or previous.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
            or previous.get("pair_manifest_sha256") != manifest_sha256
        ):
            raise ValueError("subword resume contract differs")
        if isinstance(previous.get("started_at"), str):
            started_at = str(previous["started_at"])
    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "subword-position-patching",
        "status": "running",
        "confirmatory": confirmatory,
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "pair_manifest_sha256": manifest_sha256,
        "started_at": started_at,
        "updated_at": _now(),
        "runtime": None,
        "counts": {
            "cohort_pairs": protocol.cohort_pairs,
            "evaluated_pairs": len(selected),
        },
        "failures": [],
    }
    _write_json_atomic(run_path, base_run)
    factory = runtime_factory or _runtime_factory()
    provenance: dict[str, object] | None = None
    try:
        revisions = {
            str(_mapping(record.get("source"), field="record source").get("model_revision"))
            for record in selected
        }
        if len(revisions) != 1 or not next(iter(revisions)):
            raise ValueError("subword-position model revision is not unique and non-empty")
        revision = next(iter(revisions))
        runtime = factory(
            model=protocol.model,
            task=protocol.task,
            revision=revision,
            gpu_id=config.gpu_id,
        )
        provenance = _runtime_provenance(
            runtime,
            revision=revision,
            gpu_id=config.gpu_id,
            protocol=protocol,
        )
        fingerprint = _canonical_sha256(provenance)
        for record in selected:
            checkpoint = _checkpoint_path(
                checkpoints_dir,
                record=record,
                runtime_fingerprint=fingerprint,
                protocol_sha256=protocol.config_sha256,
                manifest_sha256=manifest_sha256,
            )
            if checkpoint.is_file():
                _load_checkpoint(
                    checkpoint,
                    record=record,
                    runtime_fingerprint=fingerprint,
                    protocol=protocol,
                    manifest_sha256=manifest_sha256,
                )
                continue
            scan = _validate_scan(
                runtime.scan_pair(record, modes=protocol.modes),
                record=record,
                modes=protocol.modes,
            )
            _write_checkpoint(
                checkpoint,
                record=record,
                scan=scan,
                runtime_fingerprint=fingerprint,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
            )
        del runtime
        rows = _compile_records(
            selected=selected,
            checkpoints_dir=checkpoints_dir,
            runtime_fingerprint=fingerprint,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
        )
        table_rows = _analysis_rows(rows, modes=protocol.modes)
        pair_subsets = {str(row["pair_id"]): str(row["analysis_subset"]) for row in rows}
        primary_pairs = sum(
            int(subset == "equal-count-primary") for subset in pair_subsets.values()
        )
        secondary_pairs = len(pair_subsets) - primary_pairs
        final_rows = [row for row in rows if row["mode"] == "final"]
        compared = len(final_rows)
        agrees = sum(int(row["success"] is row["historical_final_event"]) for row in final_rows)
        flow_rows = [
            {
                "model": protocol.model,
                "task": protocol.task,
                "n_original": protocol.cohort_pairs,
                "n_evaluated": len(selected),
                "n_equal_count_primary": primary_pairs,
                "n_mismatch_monotone_secondary": secondary_pairs,
                "confirmatory": confirmatory,
            }
        ]
        summary = {
            "schema_version": _SUMMARY_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "protocol_sha256": protocol.config_sha256,
            "confirmatory": confirmatory,
            "counts": {
                "cohort_pairs": protocol.cohort_pairs,
                "evaluated_pairs": len(selected),
                "primary_pairs": primary_pairs,
                "secondary_pairs": secondary_pairs,
            },
            "answer_target": protocol.answer_target,
            "subsets": {
                subset: [row for row in table_rows if row["analysis_subset"] == subset]
                for subset in _SUBSETS
            },
            "historical_final_audit": {
                "compared": compared,
                "agrees": agrees,
                "disagrees": compared - agrees,
                "new_generation_is_authoritative": True,
            },
        }
        paths = {name: output_dir / name for name in _PUBLIC_OUTPUTS}
        _write_jsonl_atomic(paths[_PUBLIC_OUTPUTS[0]], rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[1]], table_rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[2]], flow_rows)
        _write_json_atomic(paths[_PUBLIC_OUTPUTS[3]], summary)
        counts = {
            "cohort_pairs": protocol.cohort_pairs,
            "evaluated_pairs": len(selected),
            "primary_pairs": primary_pairs,
            "secondary_pairs": secondary_pairs,
            "records": len(rows),
            "table_rows": len(table_rows),
            "flow_rows": len(flow_rows),
        }
        row_counts = {
            _PUBLIC_OUTPUTS[0]: len(rows),
            _PUBLIC_OUTPUTS[1]: len(table_rows),
            _PUBLIC_OUTPUTS[2]: len(flow_rows),
            _PUBLIC_OUTPUTS[3]: 1,
        }
        outputs = {
            name: {"sha256": sha256_file(path), "records": row_counts[name]}
            for name, path in paths.items()
        }
        completed = {
            **base_run,
            "status": "completed",
            "runtime": provenance,
            "counts": counts,
            "outputs": outputs,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "gpu_required": True,
            },
            "updated_at": _now(),
        }
        _write_json_atomic(run_path, completed)
        return _result_from_run(output_dir, completed)
    except Exception as exc:
        for name in _PUBLIC_OUTPUTS:
            (output_dir / name).unlink(missing_ok=True)
            (output_dir / f".{name}.tmp").unlink(missing_ok=True)
        failed = {
            **base_run,
            "status": "failed",
            "runtime": provenance,
            "failures": [{"type": type(exc).__name__, "message": str(exc)}],
            "updated_at": _now(),
        }
        _write_json_atomic(run_path, failed)
        raise SubwordPositionPatchingRunError(
            "subword-position patching failed; verified checkpoints were retained"
        ) from exc


__all__ = [
    "RuntimeFactory",
    "SubwordGeneration",
    "SubwordModeScan",
    "SubwordPairScan",
    "SubwordPositionPatchingConfig",
    "SubwordPositionPatchingResult",
    "SubwordPositionPatchingRunError",
    "SubwordPositionPatchingRuntime",
    "run_subword_position_patching",
]
