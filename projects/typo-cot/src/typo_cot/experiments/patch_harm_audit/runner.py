"""Resumable runner for correct-answer clean-to-typo patch harm auditing."""

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
from typo_cot.experiments.patch_harm_audit.protocol import (
    PatchHarmAuditProtocol,
    load_patch_harm_audit_protocol,
)

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RUN_SCHEMA = "patch-harm-audit-run/v1"
_CHECKPOINT_SCHEMA = "patch-harm-audit-checkpoint/v1"
_RECORD_SCHEMA = "patch-harm-audit-record/v1"
_SUMMARY_SCHEMA = "patch-harm-audit-summary/v1"
_PUBLIC_OUTPUTS = (
    "patch_harm_records.jsonl",
    "setting_harm_table.csv",
    "repair_harm_composite.csv",
    "patch_harm_summary.json",
)


class PatchHarmAuditRunError(RuntimeError):
    """Raised after preserving verified checkpoints for a failed GPU run."""


@dataclass(frozen=True, slots=True)
class PatchHarmAuditConfig:
    """Public command arguments; ``resume`` is transport-only."""

    protocol_path: Path
    manifest_path: Path
    cohort: str
    gpu_id: str
    output_dir: Path
    limit_per_setting: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("protocol_path", "manifest_path", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        if self.cohort != "clean-correct-typo-correct":
            raise ValueError("cohort must be clean-correct-typo-correct")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit_per_setting is not None and (
            isinstance(self.limit_per_setting, bool)
            or not isinstance(self.limit_per_setting, int)
            or self.limit_per_setting <= 0
        ):
            raise ValueError("limit_per_setting must be a positive integer")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be boolean")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("protocol_path", "manifest_path", "output_dir"):
            payload[field] = str(Path(payload[field]).resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class PatchHarmGeneration:
    """One capped, termination-aware patched answer generation."""

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
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("patch harm generation token_ids must be non-empty integers")
        if not isinstance(self.text, str) or not isinstance(self.value, str):
            raise TypeError("patch harm generation text and value must be strings")
        if self.termination not in {"eos", "length-cap"}:
            raise ValueError("patch harm generation termination must be eos or length-cap")
        if type(self.is_extracted) is not bool or self.is_extracted is not bool(self.value):
            raise ValueError("patch harm generation extraction flag differs from its value")
        if type(self.is_correct) is not bool or (self.is_correct and not self.is_extracted):
            raise ValueError("patch harm generation correctness differs from extraction flag")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("patch harm generation method must be non-empty")
        if not isinstance(self.primary_method, str) or not self.primary_method:
            raise ValueError("patch harm generation primary_method must be non-empty")


def _position_tuple(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    try:
        positions = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be an integer sequence") from exc
    if not positions or any(
        not isinstance(position, int) or isinstance(position, bool) or position < 0
        for position in positions
    ):
        raise ValueError(f"{field} must contain non-negative integers")
    if len(positions) != len(set(positions)):
        raise ValueError(f"{field} contains duplicate positions")
    return positions


@dataclass(frozen=True, slots=True)
class PatchHarmScan:
    """Patched generation and the exact causal coordinates used."""

    generation: PatchHarmGeneration
    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.generation, PatchHarmGeneration):
            raise TypeError("generation must be PatchHarmGeneration")
        object.__setattr__(
            self,
            "source_positions",
            _position_tuple(self.source_positions, field="source_positions"),
        )
        object.__setattr__(
            self,
            "destination_positions",
            _position_tuple(self.destination_positions, field="destination_positions"),
        )
        if len(self.source_positions) != len(self.destination_positions):
            raise ValueError("patch harm coordinate cardinality differs")


class PatchHarmRuntime(Protocol):
    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_pair(self, pair: Mapping[str, object]) -> PatchHarmScan: ...


RuntimeFactory = Callable[..., PatchHarmRuntime]


@dataclass(frozen=True, slots=True)
class PatchHarmAuditResult:
    records_path: Path
    setting_table_path: Path
    composite_path: Path
    summary_path: Path
    run_path: Path
    harm_pairs: int
    evaluated_pairs: int
    preserve: int
    harm: int
    answer_changed: int
    unextractable: int
    settings: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


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
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames = tuple(rows[0])
    if any(tuple(row) != fieldnames for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_factory() -> RuntimeFactory:
    from typo_cot.experiments.patch_harm_audit.runtime import (
        HuggingFacePatchHarmAuditRuntime,
    )

    return HuggingFacePatchHarmAuditRuntime


def _runtime_provenance(
    runtime: PatchHarmRuntime,
    *,
    model: str,
    task: str,
    revision: str,
    gpu_id: str,
    protocol: PatchHarmAuditProtocol,
) -> dict[str, object]:
    provenance = dict(runtime.provenance())
    if (
        provenance.get("operation") != "patch-harm-audit"
        or provenance.get("model") != model
        or provenance.get("task") != task
        or provenance.get("requested_revision") != revision
        or provenance.get("model_revision") != revision
        or provenance.get("tokenizer_revision") != revision
        or provenance.get("num_decoder_layers") != runtime.num_layers
        or not isinstance(runtime.num_layers, int)
        or runtime.num_layers < protocol.window[1]
        or provenance.get("dtype") != protocol.dtype
        or provenance.get("cuda_visible_devices") != gpu_id
        or provenance.get("coordinate_source") != protocol.coordinate_source
        or provenance.get("layer_window") != list(protocol.window)
        or provenance.get("cohort") != protocol.cohort
        or provenance.get("generated_arm") != "correct-coordinate-clean-to-typo/v1"
        or provenance.get("baseline_source") != protocol.baseline_source
        or provenance.get("generation_termination_protocol") != protocol.termination_protocol
        or provenance.get("answer_extraction")
        != "primary-then-empty-only-positional-by-termination/v1"
    ):
        raise ValueError("patch harm runtime provenance differs from the frozen protocol")
    eos_ids = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos_ids
        )
        or eos_ids != sorted(set(eos_ids))
        or not isinstance(provenance.get("effective_eos_token_ids_source"), str)
    ):
        raise ValueError("patch harm runtime effective EOS provenance differs")
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
        raise ValueError("patch harm runtime generation provenance differs")
    return provenance


def _generation_payload(generation: PatchHarmGeneration) -> dict[str, object]:
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


def _generation_from_payload(value: object, *, field: str) -> PatchHarmGeneration:
    payload = _mapping(value, field=field)
    expected = {
        "token_ids",
        "text",
        "termination",
        "value",
        "is_extracted",
        "is_correct",
        "method",
        "primary_method",
    }
    if set(payload) != expected or not isinstance(payload.get("token_ids"), list):
        raise ValueError(f"{field} differs from the generation contract")
    return PatchHarmGeneration(
        token_ids=tuple(payload["token_ids"]),  # type: ignore[arg-type]
        text=payload.get("text"),  # type: ignore[arg-type]
        termination=payload.get("termination"),  # type: ignore[arg-type]
        value=payload.get("value"),  # type: ignore[arg-type]
        is_extracted=payload.get("is_extracted"),  # type: ignore[arg-type]
        is_correct=payload.get("is_correct"),  # type: ignore[arg-type]
        method=payload.get("method"),  # type: ignore[arg-type]
        primary_method=payload.get("primary_method"),  # type: ignore[arg-type]
    )


def _scan_payload(scan: PatchHarmScan) -> dict[str, object]:
    return {
        "generation": _generation_payload(scan.generation),
        "source_positions": list(scan.source_positions),
        "destination_positions": list(scan.destination_positions),
    }


def _scan_from_payload(value: object, *, field: str) -> PatchHarmScan:
    payload = _mapping(value, field=field)
    if set(payload) != {"generation", "source_positions", "destination_positions"}:
        raise ValueError(f"{field} fields differ from the scan contract")
    for name in ("source_positions", "destination_positions"):
        if not isinstance(payload.get(name), list):
            raise ValueError(f"{field}.{name} must be a list")
    return PatchHarmScan(
        generation=_generation_from_payload(payload.get("generation"), field=f"{field}.generation"),
        source_positions=tuple(payload["source_positions"]),  # type: ignore[arg-type]
        destination_positions=tuple(payload["destination_positions"]),  # type: ignore[arg-type]
    )


def _correct_plan(record: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    controls = _mapping(record.get("controls"), field="record controls")
    correct = _mapping(controls.get("correct"), field="record controls.correct")
    if correct.get("valid") is not True:
        raise ValueError("harm record has no valid correct-coordinate plan")
    source = correct.get("source_positions")
    destination = correct.get("destination_positions")
    if not isinstance(source, list) or not isinstance(destination, list):
        raise ValueError("harm correct-coordinate plan must contain position lists")
    return (
        _position_tuple(source, field="correct source positions"),
        _position_tuple(destination, field="correct destination positions"),
    )


def _record_input_sha256(record: Mapping[str, object]) -> str:
    source = _mapping(record.get("source"), field="record source")
    source_positions, destination_positions = _correct_plan(record)
    return _canonical_sha256(
        {
            "pair_id": record.get("pair_id"),
            "source_record_sha256": source.get("source_record_sha256"),
            "model_revision": source.get("model_revision"),
            "typo_answer": record.get("typo_answer"),
            "gold_answer": record.get("gold_answer"),
            "source_positions": source_positions,
            "destination_positions": destination_positions,
        }
    )


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


def _validate_scan(scan: object, *, record: Mapping[str, object]) -> PatchHarmScan:
    if not isinstance(scan, PatchHarmScan):
        raise ValueError("runtime result must be PatchHarmScan")
    expected_source, expected_destination = _correct_plan(record)
    if (
        scan.source_positions != expected_source
        or scan.destination_positions != expected_destination
    ):
        raise ValueError("runtime result differs from the manifest correct-coordinate plan")
    benchmark = str(record.get("task"))
    gold = str(record.get("gold_answer"))
    if scan.generation.is_correct != answers_equal(
        scan.generation.value,
        gold,
        benchmark=benchmark,
    ):
        raise ValueError("runtime correctness differs from its extracted patched answer")
    return scan


def _write_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    scan: PatchHarmScan,
    runtime_fingerprint: str,
    protocol: PatchHarmAuditProtocol,
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
            "scan": scan_payload,
            "scan_sha256": _canonical_sha256(scan_payload),
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: PatchHarmAuditProtocol,
    manifest_sha256: str,
) -> PatchHarmScan:
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
        or payload.get("scan_sha256") != _canonical_sha256(scan_payload)
    ):
        raise ValueError(f"patch harm checkpoint provenance differs: {path}")
    return _validate_scan(
        _scan_from_payload(scan_payload, field=f"{path} scan"),
        record=record,
    )


def _select_records(
    records: Sequence[Mapping[str, object]],
    *,
    limit_per_setting: int | None,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    harm_records: list[Mapping[str, object]] = []
    for record in records:
        cohorts = _mapping(record.get("cohorts"), field="record cohorts")
        if cohorts.get("harm") is not True:
            continue
        if record.get("clean_correct") is not True or record.get("typo_correct") is not True:
            raise ValueError("harm cohort escaped clean-correct/typo-correct outcomes")
        if not isinstance(record.get("typo_answer"), str) or not record.get("typo_answer"):
            raise ValueError("harm baseline answer must be extracted and non-empty")
        _digest(record.get("pair_id"), field="harm pair_id")
        _correct_plan(record)
        harm_records.append(record)
    selected: list[Mapping[str, object]] = []
    for setting in REBUTTAL_SETTINGS:
        setting_records = sorted(
            (
                record
                for record in harm_records
                if (record.get("model"), record.get("task")) == setting.key
            ),
            key=lambda record: (str(record.get("target_rule")), str(record.get("sample_id"))),
        )
        selected.extend(
            setting_records if limit_per_setting is None else setting_records[:limit_per_setting]
        )
    return harm_records, selected


def _compile_records(
    *,
    selected: Sequence[Mapping[str, object]],
    checkpoints_dir: Path,
    runtime_fingerprints: Mapping[tuple[str, str], str],
    protocol: PatchHarmAuditProtocol,
    manifest_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in selected:
        setting = str(record["model"]), str(record["task"])
        runtime_fingerprint = runtime_fingerprints[setting]
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
        generation = scan.generation
        baseline_value = str(record["typo_answer"])
        preserve = generation.is_correct
        harm = not preserve
        unextractable = not generation.is_extracted
        answer_changed = not answers_equal(
            generation.value,
            baseline_value,
            benchmark=str(record["task"]),
        )
        source = _mapping(record.get("source"), field="record source")
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
                "baseline": {
                    "source": protocol.baseline_source,
                    "value": baseline_value,
                    "is_extracted": True,
                    "is_correct": True,
                },
                "patched": _generation_payload(generation),
                "source_positions": list(scan.source_positions),
                "destination_positions": list(scan.destination_positions),
                "preserve": preserve,
                "harm": harm,
                "answer_changed": answer_changed,
                "unextractable": unextractable,
            }
        )
    return rows


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _setting_rows(
    output_rows: Sequence[Mapping[str, object]],
    harm_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for setting in REBUTTAL_SETTINGS:
        full = [
            record
            for record in harm_records
            if (record.get("model"), record.get("task")) == setting.key
        ]
        evaluated = [
            row for row in output_rows if (row.get("model"), row.get("task")) == setting.key
        ]
        denominator = len(evaluated)
        preserve = sum(int(row["preserve"] is True) for row in evaluated)
        harm = sum(int(row["harm"] is True) for row in evaluated)
        answer_changed = sum(int(row["answer_changed"] is True) for row in evaluated)
        unextractable = sum(int(row["unextractable"] is True) for row in evaluated)
        rows.append(
            {
                "model": setting.model,
                "task": setting.task,
                "n_typo_correct": len(full),
                "n_evaluated": denominator,
                "preserve": preserve,
                "preserve_rate": _rate(preserve, denominator),
                "harm": harm,
                "harm_rate": _rate(harm, denominator),
                "answer_changed": answer_changed,
                "answer_changed_rate": _rate(answer_changed, denominator),
                "unextractable": unextractable,
                "unextractable_rate": _rate(unextractable, denominator),
            }
        )
    return rows


def _composite_row(
    *,
    scope: str,
    model: str,
    task: str,
    restoration_n: int,
    wrong_to_right: int,
    harm_n: int,
    evaluated_harm_n: int,
    right_to_wrong: int,
    uncovered: int,
    complete: bool,
    protocol: PatchHarmAuditProtocol,
) -> dict[str, object]:
    composite_n = restoration_n + harm_n if complete else None
    transition_balance = wrong_to_right - right_to_wrong if complete else None
    return {
        "scope": scope,
        "model": model,
        "task": task,
        "label": protocol.composite_label,
        "complete": complete,
        "invalid_reason": None if complete else "non-confirmatory-limit",
        "restoration_n": restoration_n,
        "wrong_to_right": wrong_to_right,
        "wrong_to_right_rate": _rate(wrong_to_right, restoration_n),
        "harm_n": harm_n,
        "evaluated_harm_n": evaluated_harm_n,
        "right_to_wrong": right_to_wrong if complete else None,
        "right_to_wrong_rate": _rate(right_to_wrong, harm_n) if complete else None,
        "transition_balance": transition_balance,
        "composite_n": composite_n,
        "composite_baseline_correct": harm_n if complete else None,
        "composite_patched_correct": (
            wrong_to_right + harm_n - right_to_wrong if complete else None
        ),
        "composite_accuracy_change": (
            transition_balance / composite_n
            if complete and transition_balance is not None and composite_n
            else None
        ),
        "prepared_typo_wrong_outside_restoration": uncovered,
        "population_net_accuracy": False,
    }


def _composite_rows(
    *,
    records: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
    harm_records: Sequence[Mapping[str, object]],
    confirmatory: bool,
    protocol: PatchHarmAuditProtocol,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    totals = {
        "restoration_n": 0,
        "wrong_to_right": 0,
        "harm_n": 0,
        "evaluated_harm_n": 0,
        "right_to_wrong": 0,
        "uncovered": 0,
    }
    for setting in REBUTTAL_SETTINGS:
        setting_records = [
            record for record in records if (record.get("model"), record.get("task")) == setting.key
        ]
        restoration = [
            record
            for record in setting_records
            if _mapping(record.get("cohorts"), field="record cohorts").get("restoration") is True
        ]
        wrong_to_right = sum(
            int(_mapping(record.get("fixed_window"), field="fixed_window").get("event") is True)
            for record in restoration
        )
        if (
            len(restoration) != setting.paper_denominator
            or wrong_to_right != setting.paper_successes
        ):
            raise ValueError(f"manifest fixed-window reference differs for {setting.slug}")
        full_harm = [
            record
            for record in harm_records
            if (record.get("model"), record.get("task")) == setting.key
        ]
        evaluated = [
            row for row in output_rows if (row.get("model"), row.get("task")) == setting.key
        ]
        right_to_wrong = sum(int(row["harm"] is True) for row in evaluated)
        uncovered = sum(
            int(
                _mapping(record.get("cohorts"), field="record cohorts").get(
                    "prepared_typo_wrong_outside_restoration"
                )
                is True
            )
            for record in setting_records
        )
        complete = confirmatory and len(evaluated) == len(full_harm)
        rows.append(
            _composite_row(
                scope="setting",
                model=setting.model,
                task=setting.task,
                restoration_n=len(restoration),
                wrong_to_right=wrong_to_right,
                harm_n=len(full_harm),
                evaluated_harm_n=len(evaluated),
                right_to_wrong=right_to_wrong,
                uncovered=uncovered,
                complete=complete,
                protocol=protocol,
            )
        )
        totals["restoration_n"] += len(restoration)
        totals["wrong_to_right"] += wrong_to_right
        totals["harm_n"] += len(full_harm)
        totals["evaluated_harm_n"] += len(evaluated)
        totals["right_to_wrong"] += right_to_wrong
        totals["uncovered"] += uncovered
    if (
        totals["restoration_n"] != protocol.restoration_pairs
        or totals["wrong_to_right"] != protocol.restoration_successes
    ):
        raise ValueError("manifest fixed-window totals differ from the harm protocol")
    rows.append(
        _composite_row(
            scope="overall",
            model="all",
            task="all",
            restoration_n=totals["restoration_n"],
            wrong_to_right=totals["wrong_to_right"],
            harm_n=totals["harm_n"],
            evaluated_harm_n=totals["evaluated_harm_n"],
            right_to_wrong=totals["right_to_wrong"],
            uncovered=totals["uncovered"],
            complete=confirmatory and totals["evaluated_harm_n"] == totals["harm_n"],
            protocol=protocol,
        )
    )
    return rows


def _result_from_run(output_dir: Path, run: Mapping[str, object]) -> PatchHarmAuditResult:
    outputs = _mapping(run.get("outputs"), field="completed run outputs")
    if set(outputs) != set(_PUBLIC_OUTPUTS):
        raise ValueError("completed patch harm output inventory differs")
    for name in _PUBLIC_OUTPUTS:
        path = output_dir / name
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"completed patch harm output SHA-256 differs: {path}")
    record_rows = tuple(
        row for _number, _line, row in iter_jsonl_objects(output_dir / _PUBLIC_OUTPUTS[0])
    )
    for row in record_rows:
        if row.get("schema_version") != _RECORD_SCHEMA:
            raise ValueError("completed patch harm record schema differs")
        patched = _generation_from_payload(row.get("patched"), field="completed patched answer")
        baseline = _mapping(row.get("baseline"), field="completed baseline")
        benchmark = str(row.get("task"))
        if (
            row.get("preserve") is not patched.is_correct
            or row.get("harm") is patched.is_correct
            or row.get("unextractable") is patched.is_extracted
            or row.get("answer_changed")
            is answers_equal(patched.value, str(baseline.get("value")), benchmark=benchmark)
        ):
            raise ValueError("completed patch harm outcomes differ from raw answers")

    def csv_count(name: str) -> int:
        with (output_dir / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = tuple(reader)
        if reader.fieldnames is None:
            raise ValueError(f"completed patch harm CSV has no header: {name}")
        return len(rows)

    actual_counts = {
        _PUBLIC_OUTPUTS[0]: len(record_rows),
        _PUBLIC_OUTPUTS[1]: csv_count(_PUBLIC_OUTPUTS[1]),
        _PUBLIC_OUTPUTS[2]: csv_count(_PUBLIC_OUTPUTS[2]),
        _PUBLIC_OUTPUTS[3]: 1,
    }
    load_json_object(output_dir / _PUBLIC_OUTPUTS[3])
    for name, count in actual_counts.items():
        if _mapping(outputs[name], field=f"completed output {name}").get("records") != count:
            raise ValueError(f"completed patch harm output count differs: {name}")
    counts = _mapping(run.get("counts"), field="completed run counts")
    expected = {
        "harm_pairs": counts.get("harm_pairs"),
        "evaluated_pairs": len(record_rows),
        "preserve": sum(int(row["preserve"] is True) for row in record_rows),
        "harm": sum(int(row["harm"] is True) for row in record_rows),
        "answer_changed": sum(int(row["answer_changed"] is True) for row in record_rows),
        "unextractable": sum(int(row["unextractable"] is True) for row in record_rows),
        "settings": len(REBUTTAL_SETTINGS),
        "setting_rows": len(REBUTTAL_SETTINGS),
        "composite_rows": len(REBUTTAL_SETTINGS) + 1,
    }
    if dict(counts) != expected:
        raise ValueError("completed patch harm counts differ from its outputs")
    return PatchHarmAuditResult(
        records_path=output_dir / _PUBLIC_OUTPUTS[0],
        setting_table_path=output_dir / _PUBLIC_OUTPUTS[1],
        composite_path=output_dir / _PUBLIC_OUTPUTS[2],
        summary_path=output_dir / _PUBLIC_OUTPUTS[3],
        run_path=output_dir / "run.json",
        harm_pairs=int(counts["harm_pairs"]),
        evaluated_pairs=int(counts["evaluated_pairs"]),
        preserve=int(counts["preserve"]),
        harm=int(counts["harm"]),
        answer_changed=int(counts["answer_changed"]),
        unextractable=int(counts["unextractable"]),
        settings=int(counts["settings"]),
    )


def _completed_resume(
    config: PatchHarmAuditConfig,
    protocol: PatchHarmAuditProtocol,
) -> PatchHarmAuditResult | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "patch-harm-audit"
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("pair_manifest_sha256") != sha256_file(config.manifest_path)
    ):
        raise ValueError("completed patch harm resume contract differs")
    return _result_from_run(config.output_dir, run)


def run_patch_harm_audit(
    config: PatchHarmAuditConfig,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> PatchHarmAuditResult:
    """Run the frozen correct-answer harm audit and conditional composite."""

    if not isinstance(config, PatchHarmAuditConfig):
        raise TypeError("config must be PatchHarmAuditConfig")
    protocol = load_patch_harm_audit_protocol(config.protocol_path)
    if config.cohort != protocol.cohort:
        raise ValueError("--cohort differs from the frozen config")
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
    records = load_rebuttal_pair_manifest(config.manifest_path)
    harm_records, selected = _select_records(
        records,
        limit_per_setting=config.limit_per_setting,
    )
    confirmatory = config.limit_per_setting is None and len(selected) == len(harm_records)

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
            or previous.get("operation") != "patch-harm-audit"
            or previous.get("status") not in {"running", "failed"}
            or previous.get("arguments") != config.public_arguments()
            or previous.get("protocol") != protocol.as_dict()
            or previous.get("protocol_sha256") != protocol.config_sha256
            or previous.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
            or previous.get("pair_manifest_sha256") != manifest_sha256
        ):
            raise ValueError("patch harm resume contract differs")
        if isinstance(previous.get("started_at"), str):
            started_at = str(previous["started_at"])
    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-harm-audit",
        "status": "running",
        "confirmatory": confirmatory,
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "pair_manifest_sha256": manifest_sha256,
        "started_at": started_at,
        "updated_at": _now(),
        "runtime_by_setting": {},
        "counts": {"harm_pairs": len(harm_records), "evaluated_pairs": len(selected)},
        "failures": [],
    }
    _write_json_atomic(run_path, base_run)
    factory = runtime_factory or _runtime_factory()
    runtime_fingerprints: dict[tuple[str, str], str] = {}
    runtime_payloads: dict[str, object] = {}
    try:
        for setting in REBUTTAL_SETTINGS:
            setting_records = [
                record
                for record in selected
                if (record.get("model"), record.get("task")) == setting.key
            ]
            if not setting_records:
                continue
            revisions = {
                str(_mapping(record.get("source"), field="record source").get("model_revision"))
                for record in setting_records
            }
            if len(revisions) != 1:
                raise ValueError(f"patch harm model revisions differ for {setting.slug}")
            revision = _digest(next(iter(revisions)), field=f"{setting.slug} model revision")
            runtime = factory(
                model=setting.model,
                task=setting.task,
                revision=revision,
                gpu_id=config.gpu_id,
            )
            provenance = _runtime_provenance(
                runtime,
                model=setting.model,
                task=setting.task,
                revision=revision,
                gpu_id=config.gpu_id,
                protocol=protocol,
            )
            fingerprint = _canonical_sha256(provenance)
            runtime_fingerprints[setting.key] = fingerprint
            runtime_payloads[setting.slug] = provenance
            for record in setting_records:
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
                scan = _validate_scan(runtime.scan_pair(record), record=record)
                _write_checkpoint(
                    checkpoint,
                    record=record,
                    scan=scan,
                    runtime_fingerprint=fingerprint,
                    protocol=protocol,
                    manifest_sha256=manifest_sha256,
                )
            del runtime

        output_rows = _compile_records(
            selected=selected,
            checkpoints_dir=checkpoints_dir,
            runtime_fingerprints=runtime_fingerprints,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
        )
        setting_rows = _setting_rows(output_rows, harm_records)
        composite_rows = _composite_rows(
            records=records,
            output_rows=output_rows,
            harm_records=harm_records,
            confirmatory=confirmatory,
            protocol=protocol,
        )
        preserve = sum(int(row["preserve"] is True) for row in output_rows)
        harm = sum(int(row["harm"] is True) for row in output_rows)
        answer_changed = sum(int(row["answer_changed"] is True) for row in output_rows)
        unextractable = sum(int(row["unextractable"] is True) for row in output_rows)
        overall_composite = composite_rows[-1]
        summary = {
            "schema_version": _SUMMARY_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "protocol_sha256": protocol.config_sha256,
            "confirmatory": confirmatory,
            "counts": {
                "harm_pairs": len(harm_records),
                "evaluated_pairs": len(output_rows),
                "preserve": preserve,
                "right_to_wrong": harm,
                "answer_changed": answer_changed,
                "unextractable": unextractable,
            },
            "definitions": {
                "preserve": protocol.preserve_definition,
                "harm": protocol.harm_definition,
                "answer_changed": protocol.answer_changed_definition,
                "unextractable": protocol.unextractable_policy,
            },
            "repair_harm_composite": {
                "label": protocol.composite_label,
                "available": bool(overall_composite["complete"]),
                "population_net_accuracy": False,
                "population_net_accuracy_reason": protocol.uncovered_policy,
                "transition_balance": overall_composite["transition_balance"],
                "composite_accuracy_change": overall_composite["composite_accuracy_change"],
            },
            "settings": setting_rows,
        }
        paths = {name: output_dir / name for name in _PUBLIC_OUTPUTS}
        _write_jsonl_atomic(paths[_PUBLIC_OUTPUTS[0]], output_rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[1]], setting_rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[2]], composite_rows)
        _write_json_atomic(paths[_PUBLIC_OUTPUTS[3]], summary)
        counts = {
            "harm_pairs": len(harm_records),
            "evaluated_pairs": len(output_rows),
            "preserve": preserve,
            "harm": harm,
            "answer_changed": answer_changed,
            "unextractable": unextractable,
            "settings": len(setting_rows),
            "setting_rows": len(setting_rows),
            "composite_rows": len(composite_rows),
        }
        row_counts = {
            _PUBLIC_OUTPUTS[0]: len(output_rows),
            _PUBLIC_OUTPUTS[1]: len(setting_rows),
            _PUBLIC_OUTPUTS[2]: len(composite_rows),
            _PUBLIC_OUTPUTS[3]: 1,
        }
        outputs = {
            name: {"sha256": sha256_file(path), "records": row_counts[name]}
            for name, path in paths.items()
        }
        completed_run = {
            **base_run,
            "status": "completed",
            "runtime_by_setting": runtime_payloads,
            "counts": counts,
            "outputs": outputs,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "gpu_required": True,
            },
            "updated_at": _now(),
        }
        _write_json_atomic(run_path, completed_run)
        return _result_from_run(output_dir, completed_run)
    except Exception as exc:
        for name in _PUBLIC_OUTPUTS:
            (output_dir / name).unlink(missing_ok=True)
            (output_dir / f".{name}.tmp").unlink(missing_ok=True)
        failed = {
            **base_run,
            "status": "failed",
            "runtime_by_setting": runtime_payloads,
            "failures": [{"type": type(exc).__name__, "message": str(exc)}],
            "updated_at": _now(),
        }
        _write_json_atomic(run_path, failed)
        raise PatchHarmAuditRunError(
            "patch harm audit failed; verified checkpoints were retained"
        ) from exc


__all__ = [
    "PatchHarmAuditConfig",
    "PatchHarmAuditResult",
    "PatchHarmAuditRunError",
    "PatchHarmGeneration",
    "PatchHarmRuntime",
    "PatchHarmScan",
    "RuntimeFactory",
    "run_patch_harm_audit",
]
