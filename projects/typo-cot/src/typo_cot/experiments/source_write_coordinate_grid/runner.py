"""Resumable source-position by write-position 2x2 experiment runner."""

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
from typing import Protocol

from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
    RebuttalSetting,
    load_rebuttal_pair_manifest,
)
from typo_cot.experiments.build_rebuttal_manifest.records import (
    iter_jsonl_objects,
    load_json_object,
    sha256_file,
)
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.patch_coordinate_controls.metrics import exact_mcnemar
from typo_cot.experiments.six_setting_patch_controls import (
    ControlArmResult,
    ControlGeneration,
)
from typo_cot.experiments.six_setting_patch_controls.source import (
    FixedReferenceAnswer,
    load_fixed_references,
)
from typo_cot.experiments.six_setting_patch_controls.statistics import (
    derived_seed,
    holm_adjust,
    paired_bootstrap_risk_difference,
)
from typo_cot.experiments.source_write_coordinate_grid.protocol import (
    SourceWriteCoordinateGridProtocol,
    load_source_write_coordinate_grid_protocol,
)
from typo_cot.experiments.source_write_coordinate_grid.statistics import cochran_q

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RUN_SCHEMA = "source-write-coordinate-grid-run/v1"
_CHECKPOINT_SCHEMA = "source-write-coordinate-grid-checkpoint/v1"
_RECORD_SCHEMA = "source-write-coordinate-grid-record/v1"
_STATUS_SCHEMA = "source-write-coordinate-grid-pair-status/v1"
_GENERATED_ARMS = ("E->O", "O->E", "O->O")
_PUBLIC_OUTPUTS = (
    "source_write_grid_records.jsonl",
    "pair_status_records.jsonl",
    "source_write_grid_table.csv",
    "source_write_contrasts.csv",
)


class SourceWriteCoordinateGridRunError(RuntimeError):
    """Raised after preserving verified checkpoints for a failed GPU run."""


@dataclass(frozen=True, slots=True)
class SourceWriteCoordinateGridConfig:
    """Public command arguments; smoke limits and resume are explicit."""

    protocol_path: Path
    manifest_path: Path
    fixed_window_root: Path
    cohorts: tuple[str, ...]
    gpu_id: str
    output_dir: Path
    limit_per_cohort: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("protocol_path", "manifest_path", "fixed_window_root", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        object.__setattr__(self, "cohorts", tuple(self.cohorts))
        if (
            not self.cohorts
            or len(self.cohorts) != len(set(self.cohorts))
            or set(self.cohorts) - {"primary", "replication"}
        ):
            raise ValueError("cohorts must contain unique primary and/or replication labels")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit_per_cohort is not None and (
            isinstance(self.limit_per_cohort, bool)
            or not isinstance(self.limit_per_cohort, int)
            or self.limit_per_cohort <= 0
        ):
            raise ValueError("limit_per_cohort must be a positive integer")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be boolean")
        output = self.output_dir.resolve()
        fixed = self.fixed_window_root.resolve()
        if output == fixed or output in fixed.parents or fixed in output.parents:
            raise ValueError("fixed-window root and output directory must not overlap")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("protocol_path", "manifest_path", "fixed_window_root", "output_dir"):
            payload[field] = str(Path(payload[field]).resolve())
        payload["cohorts"] = list(self.cohorts)
        payload.pop("resume")
        return payload


class SourceWriteGridRuntime(Protocol):
    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_grid(
        self,
        pair: Mapping[str, object],
    ) -> Mapping[str, ControlArmResult]: ...


RuntimeFactory = Callable[..., SourceWriteGridRuntime]


@dataclass(frozen=True, slots=True)
class SourceWriteCoordinateGridResult:
    records_path: Path
    pair_status_records_path: Path
    grid_table_path: Path
    contrasts_path: Path
    run_path: Path
    pairs: int
    grid_records: int
    cohorts: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _positions(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty integer list")
    positions = tuple(value)
    if any(
        not isinstance(position, int) or isinstance(position, bool) or position < 0
        for position in positions
    ):
        raise ValueError(f"{field} contains an invalid token position")
    return positions


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generation_payload(generation: ControlGeneration) -> dict[str, object]:
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


def _generation_from_payload(value: object, *, field: str) -> ControlGeneration:
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
    }:
        raise ValueError(f"{field} fields differ from the checkpoint contract")
    token_ids = payload.get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError(f"{field}.token_ids must be a list")
    return ControlGeneration(
        token_ids=tuple(token_ids),  # type: ignore[arg-type]
        text=payload.get("text"),  # type: ignore[arg-type]
        termination=payload.get("termination"),  # type: ignore[arg-type]
        value=payload.get("value"),  # type: ignore[arg-type]
        is_extracted=payload.get("is_extracted"),  # type: ignore[arg-type]
        is_correct=payload.get("is_correct"),  # type: ignore[arg-type]
        method=payload.get("method"),  # type: ignore[arg-type]
        primary_method=payload.get("primary_method"),  # type: ignore[arg-type]
    )


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
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_fixed_references(
    *, records: Sequence[Mapping[str, object]], fixed_window_root: Path
) -> dict[str, FixedReferenceAnswer]:
    return load_fixed_references(records, fixed_window_root)


def _runtime_factory() -> RuntimeFactory:
    from typo_cot.experiments.source_write_coordinate_grid.runtime import (
        HuggingFaceSourceWriteCoordinateGridRuntime,
    )

    return HuggingFaceSourceWriteCoordinateGridRuntime


def _runtime_provenance(
    runtime: SourceWriteGridRuntime,
    *,
    revision: str,
    model: str,
    task: str,
    gpu_id: str,
) -> dict[str, object]:
    provenance = dict(runtime.provenance())
    if (
        provenance.get("operation") != "source-write-coordinate-grid"
        or provenance.get("model") != model
        or provenance.get("task") != task
        or provenance.get("requested_revision") != revision
        or provenance.get("model_revision") != revision
        or provenance.get("tokenizer_revision") != revision
        or provenance.get("dtype") != "bfloat16"
        or provenance.get("cuda_visible_devices") != gpu_id
        or provenance.get("generation_termination_protocol") != "effective-eos-vs-length-cap/v1"
        or provenance.get("coordinate_source") != "rebuttal-pair-manifest/v1"
        or provenance.get("layer_window") != [0, 6]
        or provenance.get("generated_arms") != list(_GENERATED_ARMS)
        or provenance.get("answer_extraction") != "primary-then-empty-only-positional/v1"
    ):
        raise ValueError("source/write runtime provenance differs from the frozen protocol")
    eos_ids = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos_ids
        )
        or len(eos_ids) != len(set(eos_ids))
    ):
        raise ValueError("source/write runtime effective EOS inventory differs")
    generation = _mapping(provenance.get("generation"), field="runtime generation")
    for field, expected in (
        ("do_sample", False),
        ("num_beams", 1),
        ("num_return_sequences", 1),
        ("max_new_tokens", 512),
    ):
        if generation.get(field) != expected:
            raise ValueError(f"source/write runtime generation.{field} differs")
    layers = provenance.get("num_decoder_layers")
    if (
        not isinstance(layers, int)
        or isinstance(layers, bool)
        or layers < 6
        or runtime.num_layers != layers
    ):
        raise ValueError("source/write runtime decoder-layer count differs")
    return provenance


def _coordinates(record: Mapping[str, object], arm: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    controls = _mapping(record.get("controls"), field="record controls")
    correct = _mapping(controls.get("correct"), field="record controls.correct")
    offset = _mapping(controls.get("offset_2"), field="record controls.offset_2")
    edited_source = _positions(correct.get("source_positions"), field="correct source_positions")
    edited_write = _positions(
        correct.get("destination_positions"), field="correct destination_positions"
    )
    if arm == "E->E":
        return edited_source, edited_write
    offset_source = _positions(offset.get("source_positions"), field="offset source_positions")
    offset_write = _positions(
        offset.get("destination_positions"), field="offset destination_positions"
    )
    if arm == "E->O":
        return edited_source, offset_write
    if arm == "O->E":
        return offset_source, edited_write
    if arm == "O->O":
        return offset_source, offset_write
    raise ValueError(f"unknown source/write arm: {arm}")


def _common_valid(record: Mapping[str, object]) -> bool:
    controls = _mapping(record.get("controls"), field="record controls")
    correct = _mapping(controls.get("correct"), field="record controls.correct")
    offset = _mapping(controls.get("offset_2"), field="record controls.offset_2")
    return correct.get("valid") is True and offset.get("valid") is True


def _validate_arm(
    result: object,
    *,
    record: Mapping[str, object],
    arm: str,
) -> ControlArmResult:
    if not isinstance(result, ControlArmResult):
        raise ValueError(f"runtime {arm} result must be ControlArmResult")
    expected_source, expected_destination = _coordinates(record, arm)
    if (
        result.source_positions != expected_source
        or result.destination_positions != expected_destination
    ):
        raise ValueError(f"runtime {arm} coordinates differ from the manifest")
    expected_correct = answers_equal(
        result.generation.value,
        str(record["gold_answer"]),
        benchmark=str(record["task"]),
    )
    if result.generation.is_correct is not expected_correct:
        raise ValueError(f"runtime {arm} correctness differs from its answer")
    return result


def _checkpoint_path(directory: Path, pair_id: str) -> Path:
    return directory / f"{pair_id}.json"


def _write_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    results: Mapping[str, ControlArmResult],
    runtime_fingerprint: str,
    protocol: SourceWriteCoordinateGridProtocol,
    manifest_sha256: str,
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
            "grid_protocol_sha256": protocol.config_sha256,
            "pair_manifest_sha256": manifest_sha256,
            "pair_id": record["pair_id"],
            "source_record_sha256": record["source"]["source_record_sha256"],
            "runtime_fingerprint": runtime_fingerprint,
            "generated_arms": list(_GENERATED_ARMS),
            "results": {
                arm: {
                    "source_positions": list(result.source_positions),
                    "destination_positions": list(result.destination_positions),
                    "generation": _generation_payload(result.generation),
                }
                for arm, result in sorted(results.items())
            },
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: SourceWriteCoordinateGridProtocol,
    manifest_sha256: str,
) -> dict[str, ControlArmResult]:
    payload = load_json_object(path)
    if (
        payload.get("schema_version") != _CHECKPOINT_SCHEMA
        or payload.get("paper_sha256") != PAPER_SHA256
        or payload.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or payload.get("grid_protocol_sha256") != protocol.config_sha256
        or payload.get("pair_manifest_sha256") != manifest_sha256
        or payload.get("pair_id") != record["pair_id"]
        or payload.get("source_record_sha256") != record["source"]["source_record_sha256"]
        or payload.get("runtime_fingerprint") != runtime_fingerprint
        or payload.get("generated_arms") != list(_GENERATED_ARMS)
    ):
        raise ValueError(f"source/write checkpoint provenance differs: {path}")
    stored = _mapping(payload.get("results"), field="checkpoint results")
    if set(stored) != set(_GENERATED_ARMS):
        raise ValueError(f"source/write checkpoint arm inventory differs: {path}")
    results: dict[str, ControlArmResult] = {}
    for arm in _GENERATED_ARMS:
        raw = _mapping(stored.get(arm), field=f"checkpoint results.{arm}")
        result = ControlArmResult(
            generation=_generation_from_payload(
                raw.get("generation"), field=f"checkpoint {arm}.generation"
            ),
            source_positions=_positions(
                raw.get("source_positions"), field=f"checkpoint {arm}.source_positions"
            ),
            destination_positions=_positions(
                raw.get("destination_positions"),
                field=f"checkpoint {arm}.destination_positions",
            ),
        )
        results[arm] = _validate_arm(result, record=record, arm=arm)
    return results


def _paper_setting(model: str, task: str) -> RebuttalSetting:
    matches = [setting for setting in REBUTTAL_SETTINGS if setting.key == (model, task)]
    if len(matches) != 1:
        raise ValueError(
            f"source/write cohort setting is not a frozen paper setting: {model}/{task}"
        )
    return matches[0]


def _select_records(
    records: Sequence[Mapping[str, object]],
    *,
    protocol: SourceWriteCoordinateGridProtocol,
    cohorts: Sequence[str],
    limit_per_cohort: int | None,
) -> tuple[Mapping[str, object], ...]:
    selected: list[Mapping[str, object]] = []
    for cohort in cohorts:
        model, task = protocol.cohorts[cohort]
        setting = _paper_setting(model, task)
        candidates = [
            record
            for record in records
            if record["cohorts"]["restoration"]
            and (record["model"], record["task"]) == (model, task)
        ]
        if len(candidates) != setting.paper_denominator:
            raise ValueError(f"source/write paper denominator differs for {cohort}")
        candidates.sort(
            key=lambda record: (
                str(record["target_rule"]),
                str(record["sample_id"]),
                str(record["pair_id"]),
            )
        )
        if limit_per_cohort is not None:
            candidates = [record for record in candidates if _common_valid(record)][
                :limit_per_cohort
            ]
            if len(candidates) != limit_per_cohort:
                raise ValueError(f"source/write smoke limit exceeds common-valid {cohort} pairs")
        selected.extend(candidates)
    return tuple(selected)


def _compile_rows(
    *,
    selected: Sequence[Mapping[str, object]],
    references: Mapping[str, FixedReferenceAnswer],
    checkpoints_dir: Path,
    runtime_fingerprints: Mapping[tuple[str, str], str],
    protocol: SourceWriteCoordinateGridProtocol,
    manifest_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for record in selected:
        pair_id = str(record["pair_id"])
        reference = references[pair_id]
        valid = _common_valid(record)
        results = (
            _load_checkpoint(
                _checkpoint_path(checkpoints_dir, pair_id),
                record=record,
                runtime_fingerprint=runtime_fingerprints[
                    (str(record["model"]), str(record["task"]))
                ],
                protocol=protocol,
                manifest_sha256=manifest_sha256,
            )
            if valid
            else {}
        )
        events: dict[str, bool | None] = {
            "E->E": reference.correct_event,
            "E->O": None,
            "O->E": None,
            "O->O": None,
        }
        source, destination = _coordinates(record, "E->E")
        grid_rows.append(
            {
                "schema_version": _RECORD_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "pair_id": pair_id,
                "sample_id": record["sample_id"],
                "model": record["model"],
                "task": record["task"],
                "target_rule": record["target_rule"],
                "arm": "E->E",
                "valid": True,
                "event": reference.correct_event,
                "source_positions": list(source),
                "destination_positions": list(destination),
                "clean_reference_answer": reference.clean_answer,
                "generation": None,
                "result_source": "fixed-window-reference",
            }
        )
        if valid:
            for arm in _GENERATED_ARMS:
                result = results[arm]
                event = bool(
                    result.generation.is_extracted
                    and answers_equal(
                        result.generation.value,
                        reference.clean_answer,
                        benchmark=str(record["task"]),
                    )
                )
                events[arm] = event
                grid_rows.append(
                    {
                        "schema_version": _RECORD_SCHEMA,
                        "paper_sha256": PAPER_SHA256,
                        "pair_id": pair_id,
                        "sample_id": record["sample_id"],
                        "model": record["model"],
                        "task": record["task"],
                        "target_rule": record["target_rule"],
                        "arm": arm,
                        "valid": True,
                        "event": event,
                        "source_positions": list(result.source_positions),
                        "destination_positions": list(result.destination_positions),
                        "clean_reference_answer": reference.clean_answer,
                        "generation": _generation_payload(result.generation),
                        "result_source": "new-generation",
                    }
                )
        controls = _mapping(record["controls"], field="record controls")
        offset = _mapping(controls["offset_2"], field="record offset")
        invalid_reason = None if valid else offset.get("invalid_reason")
        status_rows.append(
            {
                "schema_version": _STATUS_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "pair_id": pair_id,
                "sample_id": record["sample_id"],
                "model": record["model"],
                "task": record["task"],
                "target_rule": record["target_rule"],
                "validity": {
                    "E->E": {"valid": True, "reason": None},
                    "E->O": {"valid": valid, "reason": invalid_reason},
                    "O->E": {"valid": valid, "reason": invalid_reason},
                    "O->O": {"valid": valid, "reason": invalid_reason},
                    "common-valid": valid,
                },
                "events": events,
            }
        )
    return grid_rows, status_rows


def _analysis_rows(
    status_rows: Sequence[Mapping[str, object]],
    *,
    protocol: SourceWriteCoordinateGridProtocol,
    cohorts: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_rows: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    for cohort in cohorts:
        model, task = protocol.cohorts[cohort]
        rows = [row for row in status_rows if (row["model"], row["task"]) == (model, task)]
        common = [row for row in rows if row["validity"]["common-valid"] is True]
        if not common:
            raise ValueError(f"source/write common-valid denominator is empty for {cohort}")
        arm_events = {
            arm: tuple(bool(row["events"][arm]) for row in common) for arm in protocol.arms
        }
        omnibus = cochran_q(arm_events)
        table: dict[str, object] = {
            "cohort": cohort,
            "model": model,
            "task": task,
            "n_original": len(rows),
            "n_offset_valid": sum(int(row["validity"]["O->O"]["valid"] is True) for row in rows),
            "n_common_valid": len(common),
            "cochran_q": omnibus["statistic"],
            "cochran_q_df": omnibus["degrees_of_freedom"],
            "cochran_q_p": omnibus["p_value"],
        }
        for arm in protocol.arms:
            prefix = arm.replace("->", "_to_")
            table[f"{prefix}_successes"] = sum(arm_events[arm])
            table[f"{prefix}_rate"] = sum(arm_events[arm]) / len(common)
        table_rows.append(table)

        for left, right in protocol.contrasts:
            label = f"{cohort}|{left}-vs-{right}"
            mcnemar = exact_mcnemar(arm_events[left], arm_events[right])
            bootstrap = paired_bootstrap_risk_difference(
                arm_events[left],
                arm_events[right],
                replicates=protocol.pair_bootstrap_replicates,
                confidence_level=protocol.confidence_level,
                seed=derived_seed(protocol.bootstrap_seed, label),
            )
            contrasts.append(
                {
                    "comparison_id": label,
                    "cohort": cohort,
                    "model": model,
                    "task": task,
                    "left_arm": left,
                    "right_arm": right,
                    "n_common_valid": len(common),
                    "risk_difference": bootstrap["estimate"],
                    "ci_lower": bootstrap["lower"],
                    "ci_upper": bootstrap["upper"],
                    "mcnemar_p_raw": mcnemar["p_value"],
                }
            )
    adjusted = holm_adjust(
        {str(row["comparison_id"]): float(row["mcnemar_p_raw"]) for row in contrasts}
    )
    for row in contrasts:
        row["mcnemar_p_holm"] = adjusted[str(row["comparison_id"])]
        row["holm_family_size"] = len(contrasts)
        row["holm_family"] = protocol.multiplicity
    return table_rows, contrasts


def _result_from_run(
    output_dir: Path, run: Mapping[str, object]
) -> SourceWriteCoordinateGridResult:
    outputs = _mapping(run.get("outputs"), field="completed run outputs")
    if set(outputs) != set(_PUBLIC_OUTPUTS):
        raise ValueError("completed source/write output inventory differs")
    for name in _PUBLIC_OUTPUTS:
        path = output_dir / name
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"completed source/write output SHA-256 differs: {path}")
    record_rows = tuple(
        row for _number, _line, row in iter_jsonl_objects(output_dir / _PUBLIC_OUTPUTS[0])
    )
    status_rows = tuple(
        row for _number, _line, row in iter_jsonl_objects(output_dir / _PUBLIC_OUTPUTS[1])
    )

    def csv_count(name: str) -> int:
        with (output_dir / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = tuple(reader)
        if reader.fieldnames is None:
            raise ValueError(f"completed source/write CSV has no header: {name}")
        return len(rows)

    actual_outputs = {
        _PUBLIC_OUTPUTS[0]: len(record_rows),
        _PUBLIC_OUTPUTS[1]: len(status_rows),
        _PUBLIC_OUTPUTS[2]: csv_count(_PUBLIC_OUTPUTS[2]),
        _PUBLIC_OUTPUTS[3]: csv_count(_PUBLIC_OUTPUTS[3]),
    }
    for name, actual in actual_outputs.items():
        metadata = _mapping(outputs[name], field=f"completed output {name}")
        if metadata.get("records") != actual:
            raise ValueError(f"completed source/write output count differs: {name}")
    settings = {(str(row["model"]), str(row["task"])) for row in status_rows}
    common_valid = sum(int(row["validity"]["common-valid"] is True) for row in status_rows)
    expected_counts = {
        "cohorts": len(settings),
        "common_valid_pairs": common_valid,
        "grid_records": len(record_rows),
        "pair_status_records": len(status_rows),
        "selected_pairs": len(status_rows),
    }
    counts = _mapping(run.get("counts"), field="completed run counts")
    if dict(counts) != expected_counts:
        raise ValueError("completed source/write run counts differ from its outputs")
    if actual_outputs[_PUBLIC_OUTPUTS[2]] != len(settings) or actual_outputs[
        _PUBLIC_OUTPUTS[3]
    ] != 2 * len(settings):
        raise ValueError("completed source/write analysis grid is incomplete")
    return SourceWriteCoordinateGridResult(
        records_path=output_dir / _PUBLIC_OUTPUTS[0],
        pair_status_records_path=output_dir / _PUBLIC_OUTPUTS[1],
        grid_table_path=output_dir / _PUBLIC_OUTPUTS[2],
        contrasts_path=output_dir / _PUBLIC_OUTPUTS[3],
        run_path=output_dir / "run.json",
        pairs=int(counts["selected_pairs"]),
        grid_records=int(counts["grid_records"]),
        cohorts=int(counts["cohorts"]),
    )


def _completed_resume(
    config: SourceWriteCoordinateGridConfig,
    protocol: SourceWriteCoordinateGridProtocol,
) -> SourceWriteCoordinateGridResult | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "source-write-coordinate-grid"
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("pair_manifest_sha256") != sha256_file(config.manifest_path)
    ):
        raise ValueError("completed source/write resume contract differs")
    return _result_from_run(config.output_dir, run)


def run_source_write_coordinate_grid(
    config: SourceWriteCoordinateGridConfig,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> SourceWriteCoordinateGridResult:
    """Run the three missing cells and compile the prespecified paired grid."""

    if not isinstance(config, SourceWriteCoordinateGridConfig):
        raise TypeError("config must be SourceWriteCoordinateGridConfig")
    protocol = load_source_write_coordinate_grid_protocol(config.protocol_path)
    if any(cohort not in protocol.cohorts for cohort in config.cohorts):
        raise ValueError("requested cohort is absent from the frozen protocol")
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
    selected = _select_records(
        records,
        protocol=protocol,
        cohorts=config.cohorts,
        limit_per_cohort=config.limit_per_cohort,
    )
    cohort_settings = {protocol.cohorts[cohort] for cohort in config.cohorts}
    reference_source = tuple(
        record
        for record in records
        if record["cohorts"]["restoration"] and (record["model"], record["task"]) in cohort_settings
    )
    references = _load_fixed_references(
        records=reference_source,
        fixed_window_root=config.fixed_window_root,
    )

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
            or previous.get("operation") != "source-write-coordinate-grid"
            or previous.get("status") not in {"running", "failed"}
            or previous.get("arguments") != config.public_arguments()
            or previous.get("protocol") != protocol.as_dict()
            or previous.get("protocol_sha256") != protocol.config_sha256
            or previous.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
            or previous.get("pair_manifest_sha256") != manifest_sha256
        ):
            raise ValueError("source/write resume contract differs")
        if isinstance(previous.get("started_at"), str):
            started_at = str(previous["started_at"])
    confirmatory = (
        set(config.cohorts) == set(protocol.cohorts)
        and len(config.cohorts) == len(protocol.cohorts)
        and config.limit_per_cohort is None
    )
    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "source-write-coordinate-grid",
        "status": "running",
        "confirmatory": confirmatory,
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "pair_manifest_sha256": manifest_sha256,
        "started_at": started_at,
        "updated_at": _now(),
        "runtime_by_cohort": {},
        "counts": {"selected_pairs": len(selected), "cohorts": len(config.cohorts)},
        "failures": [],
    }
    _write_json_atomic(run_path, base_run)
    factory = runtime_factory or _runtime_factory()
    runtime_fingerprints: dict[tuple[str, str], str] = {}
    runtime_payloads: dict[str, object] = {}
    try:
        for cohort in config.cohorts:
            model, task = protocol.cohorts[cohort]
            cohort_records = [
                record for record in selected if (record["model"], record["task"]) == (model, task)
            ]
            revisions = {str(record["source"]["model_revision"]) for record in cohort_records}
            if len(revisions) != 1:
                raise ValueError(f"source/write model revisions differ for {cohort}")
            revision = next(iter(revisions))
            runtime = factory(model=model, task=task, revision=revision, gpu_id=config.gpu_id)
            provenance = _runtime_provenance(
                runtime,
                revision=revision,
                model=model,
                task=task,
                gpu_id=config.gpu_id,
            )
            fingerprint = _canonical_sha256(provenance)
            runtime_fingerprints[(model, task)] = fingerprint
            runtime_payloads[cohort] = provenance
            for record in cohort_records:
                if not _common_valid(record):
                    continue
                checkpoint = _checkpoint_path(checkpoints_dir, str(record["pair_id"]))
                if checkpoint.is_file():
                    _load_checkpoint(
                        checkpoint,
                        record=record,
                        runtime_fingerprint=fingerprint,
                        protocol=protocol,
                        manifest_sha256=manifest_sha256,
                    )
                    continue
                results = dict(runtime.scan_grid(record))
                if set(results) != set(_GENERATED_ARMS):
                    raise ValueError("runtime did not return the complete source/write grid")
                validated = {
                    arm: _validate_arm(results[arm], record=record, arm=arm)
                    for arm in _GENERATED_ARMS
                }
                _write_checkpoint(
                    checkpoint,
                    record=record,
                    results=validated,
                    runtime_fingerprint=fingerprint,
                    protocol=protocol,
                    manifest_sha256=manifest_sha256,
                )
            del runtime

        grid_rows, status_rows = _compile_rows(
            selected=selected,
            references=references,
            checkpoints_dir=checkpoints_dir,
            runtime_fingerprints=runtime_fingerprints,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
        )
        table_rows, contrast_rows = _analysis_rows(
            status_rows,
            protocol=protocol,
            cohorts=config.cohorts,
        )
        paths = {name: output_dir / name for name in _PUBLIC_OUTPUTS}
        _write_jsonl_atomic(paths[_PUBLIC_OUTPUTS[0]], grid_rows)
        _write_jsonl_atomic(paths[_PUBLIC_OUTPUTS[1]], status_rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[2]], table_rows)
        _write_csv_atomic(paths[_PUBLIC_OUTPUTS[3]], contrast_rows)
        counts = {
            "cohorts": len(config.cohorts),
            "common_valid_pairs": sum(
                int(row["validity"]["common-valid"] is True) for row in status_rows
            ),
            "grid_records": len(grid_rows),
            "pair_status_records": len(status_rows),
            "selected_pairs": len(selected),
        }
        row_counts = {
            _PUBLIC_OUTPUTS[0]: len(grid_rows),
            _PUBLIC_OUTPUTS[1]: len(status_rows),
            _PUBLIC_OUTPUTS[2]: len(table_rows),
            _PUBLIC_OUTPUTS[3]: len(contrast_rows),
        }
        outputs = {
            name: {"sha256": sha256_file(path), "records": row_counts[name]}
            for name, path in paths.items()
        }
        completed_run = {
            **base_run,
            "status": "completed",
            "runtime_by_cohort": runtime_payloads,
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
            "runtime_by_cohort": runtime_payloads,
            "failures": [{"type": type(exc).__name__, "message": str(exc)}],
            "updated_at": _now(),
        }
        _write_json_atomic(run_path, failed)
        raise SourceWriteCoordinateGridRunError(
            "source/write coordinate grid failed; verified checkpoints were retained"
        ) from exc


__all__ = [
    "RuntimeFactory",
    "SourceWriteCoordinateGridConfig",
    "SourceWriteCoordinateGridResult",
    "SourceWriteCoordinateGridRunError",
    "SourceWriteGridRuntime",
    "run_source_write_coordinate_grid",
]
