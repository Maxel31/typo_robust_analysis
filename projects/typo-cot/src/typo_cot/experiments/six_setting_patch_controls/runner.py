"""Resumable six-setting answer-level specificity-control runner."""

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
from html import escape
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
from typo_cot.experiments.patch_coordinate_controls.metrics import exact_mcnemar
from typo_cot.experiments.six_setting_patch_controls.protocol import (
    SixSettingPatchControlsProtocol,
    load_six_setting_patch_controls_protocol,
)
from typo_cot.experiments.six_setting_patch_controls.source import (
    FixedReferenceAnswer,
    load_fixed_references,
)
from typo_cot.experiments.six_setting_patch_controls.statistics import (
    derived_seed,
    holm_adjust,
    nested_macro_bootstrap_risk_difference,
    paired_bootstrap_risk_difference,
)

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RUN_SCHEMA = "six-setting-patch-controls-run/v1"
_CHECKPOINT_SCHEMA = "six-setting-patch-controls-checkpoint/v1"
_CONTROL_RECORD_SCHEMA = "six-setting-patch-controls-record/v1"
_STATUS_SCHEMA = "six-setting-patch-controls-pair-status/v1"
_DIAGNOSTIC_CONTROLS = ("offset-2", "cross-item")
_PUBLIC_OUTPUTS = (
    "control_records.jsonl",
    "pair_status_records.jsonl",
    "six_setting_control_table.csv",
    "common_denominator_flow.csv",
    "multiplicity_table.csv",
    "macro_average.json",
    "risk_difference_forest.svg",
)


class SixSettingPatchControlsRunError(RuntimeError):
    """Raised after preserving verified checkpoints for a failed GPU run."""


@dataclass(frozen=True, slots=True)
class SixSettingPatchControlsConfig:
    """Public command arguments; ``resume`` is transport-only."""

    protocol_path: Path
    manifest_path: Path
    fixed_window_root: Path
    output_dir: Path
    gpu_id: str = "0"
    limit_per_setting: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("protocol_path", "manifest_path", "fixed_window_root", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
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
        output = self.output_dir.resolve()
        fixed = self.fixed_window_root.resolve()
        if output == fixed or output in fixed.parents or fixed in output.parents:
            raise ValueError("fixed-window root and output directory must not overlap")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("protocol_path", "manifest_path", "fixed_window_root", "output_dir"):
            payload[field] = str(Path(payload[field]).resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class ControlGeneration:
    """One capped, termination-aware diagnostic generation."""

    token_ids: tuple[int, ...]
    text: str
    termination: Literal["eos", "length-cap"]
    value: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str

    def __post_init__(self) -> None:
        if not self.token_ids or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("control generation token_ids must be non-empty integers")
        if not isinstance(self.text, str) or not isinstance(self.value, str):
            raise TypeError("control generation text and value must be strings")
        if self.termination not in {"eos", "length-cap"}:
            raise ValueError("control generation termination must be eos or length-cap")
        if type(self.is_extracted) is not bool or self.is_extracted is not bool(self.value):
            raise ValueError("control generation extraction flag differs from its value")
        if type(self.is_correct) is not bool:
            raise TypeError("control generation correctness must be boolean")
        if not self.method or not self.primary_method:
            raise ValueError("control generation extraction methods must be non-empty")


@dataclass(frozen=True, slots=True)
class ControlArmResult:
    """Generation plus the exact source and write coordinates actually used."""

    generation: ControlGeneration
    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        for field, values in (
            ("source_positions", self.source_positions),
            ("destination_positions", self.destination_positions),
        ):
            if not values or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            ):
                raise ValueError(f"{field} must be a non-empty integer tuple")
        if len(self.source_positions) != len(self.destination_positions):
            raise ValueError("control arm coordinate cardinalities differ")


class SixSettingControlRuntime(Protocol):
    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_controls(
        self,
        pair: Mapping[str, object],
        donor_pair: Mapping[str, object] | None,
        controls: tuple[str, ...],
    ) -> Mapping[str, ControlArmResult]: ...


RuntimeFactory = Callable[..., SixSettingControlRuntime]


@dataclass(frozen=True, slots=True)
class SixSettingPatchControlsResult:
    control_records_path: Path
    pair_status_records_path: Path
    control_table_path: Path
    flow_table_path: Path
    multiplicity_table_path: Path
    macro_average_path: Path
    forest_plot_path: Path
    run_path: Path
    pairs: int
    control_records: int
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


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_fixed_references(
    *, records: Sequence[Mapping[str, object]], fixed_window_root: Path
) -> dict[str, FixedReferenceAnswer]:
    return load_fixed_references(records, fixed_window_root)


def _setting_slug(model: str, task: str) -> str:
    return f"{model.rsplit('/', 1)[-1]}__{task}"


def _runtime_factory() -> RuntimeFactory:
    from typo_cot.experiments.six_setting_patch_controls.runtime import (
        HuggingFaceSixSettingPatchControlsRuntime,
    )

    return HuggingFaceSixSettingPatchControlsRuntime


def _runtime_provenance(
    runtime: SixSettingControlRuntime,
    *,
    revision: str,
    model: str,
    task: str,
    gpu_id: str,
) -> dict[str, object]:
    provenance = dict(runtime.provenance())
    if (
        provenance.get("operation") != "six-setting-patch-controls"
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
        or provenance.get("diagnostic_controls") != ["offset-2", "cross-item"]
        or provenance.get("answer_extraction")
        != "primary-then-empty-only-positional-by-termination/v1"
    ):
        raise ValueError("six-setting runtime provenance differs from the frozen protocol")
    eos_ids = provenance.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos_ids
        )
        or len(eos_ids) != len(set(eos_ids))
    ):
        raise ValueError("six-setting runtime effective EOS inventory differs")
    generation = _mapping(provenance.get("generation"), field="runtime generation")
    for field, expected in (
        ("do_sample", False),
        ("num_beams", 1),
        ("num_return_sequences", 1),
        ("max_new_tokens", 512),
    ):
        if generation.get(field) != expected:
            raise ValueError(f"six-setting runtime generation.{field} differs")
    layers = provenance.get("num_decoder_layers")
    if (
        not isinstance(layers, int)
        or isinstance(layers, bool)
        or layers < 6
        or runtime.num_layers != layers
    ):
        raise ValueError("six-setting runtime decoder-layer count differs")
    return provenance


def _expected_coordinates(
    record: Mapping[str, object],
    donor: Mapping[str, object] | None,
    control: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    controls = _mapping(record.get("controls"), field="record controls")
    if control == "offset-2":
        offset = _mapping(controls.get("offset_2"), field="record controls.offset_2")
        return (
            _positions(offset.get("source_positions"), field="offset source_positions"),
            _positions(
                offset.get("destination_positions"),
                field="offset destination_positions",
            ),
        )
    if control != "cross-item" or donor is None:
        raise ValueError("cross-item requires its manifest donor")
    donor_edits = donor.get("edits")
    recipient_edits = record.get("edits")
    if not isinstance(donor_edits, list) or not isinstance(recipient_edits, list):
        raise ValueError("cross-item records must contain edit lists")
    source = tuple(
        int(_mapping(edit, field="donor edit").get("clean_word_final_token"))
        for edit in donor_edits
    )
    destination = tuple(
        int(_mapping(edit, field="recipient edit").get("typo_word_final_token"))
        for edit in recipient_edits
    )
    if not source or len(source) != len(destination):
        raise ValueError("cross-item manifest coordinates differ in cardinality")
    return source, destination


def _validate_arm_result(
    result: object,
    *,
    record: Mapping[str, object],
    donor: Mapping[str, object] | None,
    control: str,
) -> ControlArmResult:
    if not isinstance(result, ControlArmResult):
        raise ValueError(f"runtime {control} result must be ControlArmResult")
    expected_source, expected_destination = _expected_coordinates(record, donor, control)
    if (
        result.source_positions != expected_source
        or result.destination_positions != expected_destination
    ):
        raise ValueError(f"runtime {control} coordinates differ from the manifest")
    benchmark = str(record["task"])
    gold = str(record["gold_answer"])
    expected_correct = answers_equal(result.generation.value, gold, benchmark=benchmark)
    if result.generation.is_correct is not expected_correct:
        raise ValueError(f"runtime {control} correctness differs from its answer")
    return result


def _checkpoint_path(checkpoints_dir: Path, pair_id: str) -> Path:
    return checkpoints_dir / f"{pair_id}.json"


def _write_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    requested_controls: tuple[str, ...],
    results: Mapping[str, ControlArmResult],
    runtime_fingerprint: str,
    protocol: SixSettingPatchControlsProtocol,
    manifest_sha256: str,
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
            "control_protocol_sha256": protocol.config_sha256,
            "pair_manifest_sha256": manifest_sha256,
            "pair_id": record["pair_id"],
            "source_record_sha256": record["source"]["source_record_sha256"],
            "runtime_fingerprint": runtime_fingerprint,
            "requested_controls": list(requested_controls),
            "results": {
                control: {
                    "source_positions": list(result.source_positions),
                    "destination_positions": list(result.destination_positions),
                    "generation": _generation_payload(result.generation),
                }
                for control, result in sorted(results.items())
            },
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    donor_by_id: Mapping[str, Mapping[str, object]],
    requested_controls: tuple[str, ...],
    runtime_fingerprint: str,
    protocol: SixSettingPatchControlsProtocol,
    manifest_sha256: str,
) -> dict[str, ControlArmResult]:
    payload = load_json_object(path)
    if set(payload) != {
        "schema_version",
        "paper_sha256",
        "manifest_protocol_sha256",
        "control_protocol_sha256",
        "pair_manifest_sha256",
        "pair_id",
        "source_record_sha256",
        "runtime_fingerprint",
        "requested_controls",
        "results",
    }:
        raise ValueError(f"control checkpoint fields differ: {path}")
    if (
        payload.get("schema_version") != _CHECKPOINT_SCHEMA
        or payload.get("paper_sha256") != PAPER_SHA256
        or payload.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or payload.get("control_protocol_sha256") != protocol.config_sha256
        or payload.get("pair_manifest_sha256") != manifest_sha256
        or payload.get("pair_id") != record["pair_id"]
        or payload.get("source_record_sha256") != record["source"]["source_record_sha256"]
        or payload.get("runtime_fingerprint") != runtime_fingerprint
        or payload.get("requested_controls") != list(requested_controls)
    ):
        raise ValueError(f"control checkpoint provenance differs: {path}")
    stored = _mapping(payload.get("results"), field="checkpoint results")
    if set(stored) != set(requested_controls):
        raise ValueError(f"control checkpoint arm inventory differs: {path}")
    results: dict[str, ControlArmResult] = {}
    controls = _mapping(record.get("controls"), field="record controls")
    cross = _mapping(controls.get("cross_item"), field="record cross-item")
    donor = donor_by_id.get(str(cross.get("donor_pair_id")))
    for control in requested_controls:
        arm = _mapping(stored.get(control), field=f"checkpoint results.{control}")
        if set(arm) != {"source_positions", "destination_positions", "generation"}:
            raise ValueError(f"checkpoint {control} fields differ: {path}")
        result = ControlArmResult(
            generation=_generation_from_payload(
                arm.get("generation"), field=f"checkpoint {control}.generation"
            ),
            source_positions=_positions(
                arm.get("source_positions"), field=f"checkpoint {control}.source_positions"
            ),
            destination_positions=_positions(
                arm.get("destination_positions"),
                field=f"checkpoint {control}.destination_positions",
            ),
        )
        results[control] = _validate_arm_result(
            result,
            record=record,
            donor=donor,
            control=control,
        )
    return results


def _requested_controls(record: Mapping[str, object]) -> tuple[str, ...]:
    controls = _mapping(record.get("controls"), field="record controls")
    offset = _mapping(controls.get("offset_2"), field="record controls.offset_2")
    cross = _mapping(controls.get("cross_item"), field="record controls.cross_item")
    return tuple(
        control
        for control, valid in (
            ("offset-2", offset.get("valid")),
            ("cross-item", cross.get("valid")),
        )
        if valid is True
    )


def _selected_records(
    records: Sequence[Mapping[str, object]],
    *,
    limit_per_setting: int | None,
) -> tuple[Mapping[str, object], ...]:
    selected: list[Mapping[str, object]] = []
    for setting in REBUTTAL_SETTINGS:
        setting_records = [
            record
            for record in records
            if record["cohorts"]["restoration"] and (record["model"], record["task"]) == setting.key
        ]
        if len(setting_records) != setting.paper_denominator:
            raise ValueError(f"restoration denominator differs for {setting.slug}")
        setting_records.sort(
            key=lambda record: (
                str(record["target_rule"]),
                str(record["sample_id"]),
                str(record["pair_id"]),
            )
        )
        if limit_per_setting is None:
            selected.extend(setting_records)
            continue
        common_valid = [
            record
            for record in setting_records
            if _mapping(record.get("controls"), field="record controls").get("common_valid")
            is True
        ]
        if not common_valid:
            raise ValueError(f"no common-valid smoke pair exists for {setting.slug}")
        selected.extend(common_valid[:limit_per_setting])
    return tuple(selected)


def _compile_rows(
    *,
    selected: Sequence[Mapping[str, object]],
    references: Mapping[str, FixedReferenceAnswer],
    checkpoints_dir: Path,
    runtime_fingerprints: Mapping[tuple[str, str], str],
    protocol: SixSettingPatchControlsProtocol,
    manifest_sha256: str,
    by_id: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    control_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for record in selected:
        pair_id = str(record["pair_id"])
        reference = references[pair_id]
        requested = _requested_controls(record)
        controls = _mapping(record["controls"], field="record controls")
        cross = _mapping(controls["cross_item"], field="record cross-item")
        donor = by_id.get(str(cross.get("donor_pair_id")))
        results = (
            _load_checkpoint(
                _checkpoint_path(checkpoints_dir, pair_id),
                record=record,
                donor_by_id=by_id,
                requested_controls=requested,
                runtime_fingerprint=runtime_fingerprints[
                    (str(record["model"]), str(record["task"]))
                ],
                protocol=protocol,
                manifest_sha256=manifest_sha256,
            )
            if requested
            else {}
        )
        arm_events: dict[str, bool | None] = {
            "correct": reference.correct_event,
            "offset-2": None,
            "cross-item": None,
        }
        control_rows.append(
            {
                "schema_version": _CONTROL_RECORD_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "pair_id": pair_id,
                "sample_id": record["sample_id"],
                "model": record["model"],
                "task": record["task"],
                "target_rule": record["target_rule"],
                "control": "correct",
                "valid": True,
                "event": reference.correct_event,
                "source_positions": controls["correct"]["source_positions"],
                "destination_positions": controls["correct"]["destination_positions"],
                "donor_pair_id": pair_id,
                "clean_reference_answer": reference.clean_answer,
                "generation": None,
                "result_source": "fixed-window-reference",
            }
        )
        for control in requested:
            result = results[control]
            event = bool(
                result.generation.is_extracted
                and answers_equal(
                    result.generation.value,
                    reference.clean_answer,
                    benchmark=str(record["task"]),
                )
            )
            arm_events[control] = event
            control_rows.append(
                {
                    "schema_version": _CONTROL_RECORD_SCHEMA,
                    "paper_sha256": PAPER_SHA256,
                    "pair_id": pair_id,
                    "sample_id": record["sample_id"],
                    "model": record["model"],
                    "task": record["task"],
                    "target_rule": record["target_rule"],
                    "control": control,
                    "valid": True,
                    "event": event,
                    "source_positions": list(result.source_positions),
                    "destination_positions": list(result.destination_positions),
                    "donor_pair_id": (pair_id if control == "offset-2" else donor["pair_id"]),
                    "clean_reference_answer": reference.clean_answer,
                    "generation": _generation_payload(result.generation),
                    "result_source": "new-generation",
                }
            )
        offset = _mapping(controls["offset_2"], field="record offset")
        cross_plan = _mapping(controls["cross_item"], field="record cross")
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
                    "correct": {"valid": True, "reason": None},
                    "offset-2": {
                        "valid": offset.get("valid"),
                        "reason": offset.get("invalid_reason"),
                    },
                    "cross-item": {
                        "valid": cross_plan.get("valid"),
                        "reason": cross_plan.get("invalid_reason"),
                    },
                    "common-valid": controls.get("common_valid"),
                },
                "events": arm_events,
            }
        )
    return control_rows, status_rows


def _analysis_rows(
    status_rows: Sequence[Mapping[str, object]],
    protocol: SixSettingPatchControlsProtocol,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    table_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    raw_comparisons: list[dict[str, object]] = []
    macro_inputs: dict[str, dict[str, tuple[Sequence[bool], Sequence[bool]]]] = {
        control: {} for control in _DIAGNOSTIC_CONTROLS
    }
    for setting in REBUTTAL_SETTINGS:
        rows = [row for row in status_rows if (row["model"], row["task"]) == setting.key]
        setting_label = _setting_slug(setting.model, setting.task)
        n_original = len(rows)
        n_offset = sum(bool(row["validity"]["offset-2"]["valid"]) for row in rows)
        n_cross = sum(bool(row["validity"]["cross-item"]["valid"]) for row in rows)
        common = [row for row in rows if row["validity"]["common-valid"] is True]
        if not common:
            raise ValueError(f"common-valid denominator is empty for {setting.slug}")
        correct_events = tuple(bool(row["events"]["correct"]) for row in common)
        control_events = {
            control: tuple(bool(row["events"][control]) for row in common)
            for control in _DIAGNOSTIC_CONTROLS
        }
        flow_rows.append(
            {
                "model": setting.model,
                "task": setting.task,
                "n_original": n_original,
                "n_offset_valid": n_offset,
                "n_cross_item_valid": n_cross,
                "n_common_valid": len(common),
            }
        )
        table: dict[str, object] = {
            "model": setting.model,
            "task": setting.task,
            "n_common_valid": len(common),
            "correct_successes": sum(correct_events),
            "correct_rate": sum(correct_events) / len(common),
        }
        for control in _DIAGNOSTIC_CONTROLS:
            events = control_events[control]
            label = f"{setting_label}|correct-vs-{control}"
            mcnemar = exact_mcnemar(correct_events, events)
            bootstrap = paired_bootstrap_risk_difference(
                correct_events,
                events,
                replicates=protocol.pair_bootstrap_replicates,
                confidence_level=protocol.confidence_level,
                seed=derived_seed(protocol.bootstrap_seed, label),
            )
            prefix = control.replace("-", "_")
            table.update(
                {
                    f"{prefix}_successes": sum(events),
                    f"{prefix}_rate": sum(events) / len(events),
                    f"correct_minus_{prefix}": bootstrap["estimate"],
                    f"correct_minus_{prefix}_ci_lower": bootstrap["lower"],
                    f"correct_minus_{prefix}_ci_upper": bootstrap["upper"],
                    f"correct_vs_{prefix}_mcnemar_p": mcnemar["p_value"],
                }
            )
            raw_comparisons.append(
                {
                    "comparison_id": label,
                    "model": setting.model,
                    "task": setting.task,
                    "control": control,
                    "n_common_valid": len(common),
                    "risk_difference": bootstrap["estimate"],
                    "ci_lower": bootstrap["lower"],
                    "ci_upper": bootstrap["upper"],
                    "mcnemar_p_raw": mcnemar["p_value"],
                }
            )
            macro_inputs[control][setting_label] = (correct_events, events)
        table_rows.append(table)

    adjusted = holm_adjust(
        {str(row["comparison_id"]): float(row["mcnemar_p_raw"]) for row in raw_comparisons}
    )
    for row in raw_comparisons:
        row["mcnemar_p_holm"] = adjusted[str(row["comparison_id"])]
        row["holm_family_size"] = len(raw_comparisons)
        row["holm_family"] = protocol.multiplicity
    macro = {
        "schema_version": "six-setting-patch-controls-macro-average/v1",
        "paper_sha256": PAPER_SHA256,
        "equal_setting_weight": True,
        "comparisons": {
            control: nested_macro_bootstrap_risk_difference(
                macro_inputs[control],
                replicates=protocol.nested_bootstrap_replicates,
                confidence_level=protocol.confidence_level,
                seed=derived_seed(protocol.bootstrap_seed, f"macro|correct-vs-{control}"),
            )
            for control in _DIAGNOSTIC_CONTROLS
        },
    }
    return table_rows, flow_rows, raw_comparisons, macro


def _forest_svg(comparisons: Sequence[Mapping[str, object]]) -> str:
    width = 980
    row_height = 30
    margin_top = 52
    height = margin_top + row_height * len(comparisons) + 42
    plot_left, plot_right = 430, 940
    minimum, maximum = -1.0, 1.0

    def x(value: float) -> float:
        return plot_left + (value - minimum) / (maximum - minimum) * (plot_right - plot_left)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18">'
        "Correct-coordinate minus control restoration rate</text>",
        f'<line x1="{x(0):.2f}" y1="38" x2="{x(0):.2f}" y2="{height - 30}" '
        'stroke="#777" stroke-dasharray="4 4"/>',
    ]
    for index, row in enumerate(comparisons):
        y = margin_top + index * row_height
        label = f"{str(row['model']).rsplit('/', 1)[-1]} / {row['task']} / {row['control']}"
        estimate = float(row["risk_difference"])
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        lines.extend(
            [
                f'<text x="20" y="{y + 5}" font-family="sans-serif" font-size="12">'
                f"{escape(label)}</text>",
                f'<line x1="{x(max(minimum, lower)):.2f}" y1="{y}" '
                f'x2="{x(min(maximum, upper)):.2f}" y2="{y}" stroke="#315b7d" '
                'stroke-width="3"/>',
                f'<circle cx="{x(max(minimum, min(maximum, estimate))):.2f}" cy="{y}" '
                'r="4" fill="#d95f02"/>',
            ]
        )
    lines.append("</svg>\n")
    return "\n".join(lines)


def _result_from_run(output_dir: Path, run: Mapping[str, object]) -> SixSettingPatchControlsResult:
    outputs = _mapping(run.get("outputs"), field="completed run outputs")
    if set(outputs) != set(_PUBLIC_OUTPUTS):
        raise ValueError("completed six-setting output inventory differs")
    for name in _PUBLIC_OUTPUTS:
        path = output_dir / name
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"completed six-setting output SHA-256 differs: {path}")
    control_rows = tuple(
        row for _line_number, _line, row in iter_jsonl_objects(output_dir / _PUBLIC_OUTPUTS[0])
    )
    status_rows = tuple(
        row for _line_number, _line, row in iter_jsonl_objects(output_dir / _PUBLIC_OUTPUTS[1])
    )

    def csv_count(name: str) -> int:
        with (output_dir / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = tuple(reader)
        if reader.fieldnames is None:
            raise ValueError(f"completed six-setting CSV has no header: {name}")
        return len(rows)

    output_counts = {
        "control_records.jsonl": len(control_rows),
        "pair_status_records.jsonl": len(status_rows),
        "six_setting_control_table.csv": csv_count("six_setting_control_table.csv"),
        "common_denominator_flow.csv": csv_count("common_denominator_flow.csv"),
        "multiplicity_table.csv": csv_count("multiplicity_table.csv"),
        "macro_average.json": 1,
        "risk_difference_forest.svg": 1,
    }
    load_json_object(output_dir / "macro_average.json")
    if (
        not (output_dir / "risk_difference_forest.svg")
        .read_text(encoding="utf-8")
        .startswith("<svg")
    ):
        raise ValueError("completed six-setting forest plot is not SVG")
    for name, actual in output_counts.items():
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        if metadata.get("records") != actual:
            raise ValueError(f"completed six-setting output count differs: {name}")

    counts = _mapping(run.get("counts"), field="completed run counts")
    settings = {(str(row.get("model")), str(row.get("task"))) for row in status_rows}
    common_valid = 0
    for row in status_rows:
        validity = _mapping(row.get("validity"), field="completed status validity")
        common_valid += int(validity.get("common-valid") is True)
    expected_counts = {
        "common_valid_pairs": common_valid,
        "control_records": len(control_rows),
        "pair_status_records": len(status_rows),
        "selected_pairs": len(status_rows),
        "settings": len(settings),
    }
    if dict(counts) != expected_counts:
        raise ValueError("completed six-setting run counts differ from its outputs")
    if (
        expected_counts["settings"] != len(REBUTTAL_SETTINGS)
        or output_counts["six_setting_control_table.csv"] != len(REBUTTAL_SETTINGS)
        or output_counts["common_denominator_flow.csv"] != len(REBUTTAL_SETTINGS)
        or output_counts["multiplicity_table.csv"] != 2 * len(REBUTTAL_SETTINGS)
    ):
        raise ValueError("completed six-setting analysis grid is incomplete")
    return SixSettingPatchControlsResult(
        control_records_path=output_dir / _PUBLIC_OUTPUTS[0],
        pair_status_records_path=output_dir / _PUBLIC_OUTPUTS[1],
        control_table_path=output_dir / _PUBLIC_OUTPUTS[2],
        flow_table_path=output_dir / _PUBLIC_OUTPUTS[3],
        multiplicity_table_path=output_dir / _PUBLIC_OUTPUTS[4],
        macro_average_path=output_dir / _PUBLIC_OUTPUTS[5],
        forest_plot_path=output_dir / _PUBLIC_OUTPUTS[6],
        run_path=output_dir / "run.json",
        pairs=int(counts["selected_pairs"]),
        control_records=int(counts["control_records"]),
        settings=int(counts["settings"]),
    )


def _completed_resume(
    config: SixSettingPatchControlsConfig,
    protocol: SixSettingPatchControlsProtocol,
) -> SixSettingPatchControlsResult | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "six-setting-patch-controls"
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("pair_manifest_sha256") != sha256_file(config.manifest_path)
    ):
        raise ValueError("completed six-setting resume contract differs")
    return _result_from_run(config.output_dir, run)


def run_six_setting_patch_controls(
    config: SixSettingPatchControlsConfig,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> SixSettingPatchControlsResult:
    """Run all valid offset/cross arms and compile the frozen paired analysis."""

    if not isinstance(config, SixSettingPatchControlsConfig):
        raise TypeError("config must be SixSettingPatchControlsConfig")
    protocol = load_six_setting_patch_controls_protocol(config.protocol_path)
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
    references = _load_fixed_references(
        records=records,
        fixed_window_root=config.fixed_window_root,
    )
    selected = _selected_records(records, limit_per_setting=config.limit_per_setting)
    by_id = {str(record["pair_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("rebuttal manifest pair IDs are duplicated")

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
            or previous.get("operation") != "six-setting-patch-controls"
            or previous.get("status") not in {"running", "failed"}
            or previous.get("arguments") != config.public_arguments()
            or previous.get("protocol") != protocol.as_dict()
            or previous.get("protocol_sha256") != protocol.config_sha256
            or previous.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
            or previous.get("pair_manifest_sha256") != manifest_sha256
        ):
            raise ValueError("six-setting resume contract differs")
        previous_started = previous.get("started_at")
        if isinstance(previous_started, str):
            started_at = previous_started

    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "six-setting-patch-controls",
        "status": "running",
        "confirmatory": config.limit_per_setting is None,
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "pair_manifest_sha256": manifest_sha256,
        "started_at": started_at,
        "updated_at": _now(),
        "runtime_by_setting": {},
        "counts": {"selected_pairs": len(selected), "settings": len(REBUTTAL_SETTINGS)},
        "failures": [],
    }
    _write_json_atomic(run_path, base_run)
    factory = runtime_factory or _runtime_factory()
    runtime_fingerprints: dict[tuple[str, str], str] = {}
    runtime_payloads: dict[str, object] = {}
    try:
        for setting in REBUTTAL_SETTINGS:
            setting_records = [
                record for record in selected if (record["model"], record["task"]) == setting.key
            ]
            if not setting_records:
                raise ValueError(f"selected no pairs for {setting.slug}")
            revision_values = {
                str(record["source"]["model_revision"]) for record in setting_records
            }
            if len(revision_values) != 1:
                raise ValueError(f"model revisions differ for {setting.slug}")
            revision = next(iter(revision_values))
            runtime = factory(
                model=setting.model,
                task=setting.task,
                revision=revision,
                gpu_id=config.gpu_id,
            )
            provenance = _runtime_provenance(
                runtime,
                revision=revision,
                model=setting.model,
                task=setting.task,
                gpu_id=config.gpu_id,
            )
            fingerprint = _canonical_sha256(provenance)
            runtime_fingerprints[setting.key] = fingerprint
            runtime_payloads[_setting_slug(setting.model, setting.task)] = provenance
            for record in setting_records:
                requested = _requested_controls(record)
                if not requested:
                    continue
                checkpoint = _checkpoint_path(checkpoints_dir, str(record["pair_id"]))
                controls = _mapping(record["controls"], field="record controls")
                cross = _mapping(controls["cross_item"], field="record cross-item")
                donor = by_id.get(str(cross.get("donor_pair_id")))
                if checkpoint.is_file():
                    _load_checkpoint(
                        checkpoint,
                        record=record,
                        donor_by_id=by_id,
                        requested_controls=requested,
                        runtime_fingerprint=fingerprint,
                        protocol=protocol,
                        manifest_sha256=manifest_sha256,
                    )
                    continue
                results = dict(runtime.scan_controls(record, donor, requested))
                if set(results) != set(requested):
                    raise ValueError("runtime did not return the requested control inventory")
                validated = {
                    control: _validate_arm_result(
                        results[control],
                        record=record,
                        donor=donor,
                        control=control,
                    )
                    for control in requested
                }
                _write_checkpoint(
                    checkpoint,
                    record=record,
                    requested_controls=requested,
                    results=validated,
                    runtime_fingerprint=fingerprint,
                    protocol=protocol,
                    manifest_sha256=manifest_sha256,
                )
            del runtime

        control_rows, status_rows = _compile_rows(
            selected=selected,
            references=references,
            checkpoints_dir=checkpoints_dir,
            runtime_fingerprints=runtime_fingerprints,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
            by_id=by_id,
        )
        table_rows, flow_rows, multiplicity_rows, macro = _analysis_rows(
            status_rows,
            protocol,
        )
        paths = {name: output_dir / name for name in _PUBLIC_OUTPUTS}
        _write_jsonl_atomic(paths["control_records.jsonl"], control_rows)
        _write_jsonl_atomic(paths["pair_status_records.jsonl"], status_rows)
        table_fields = list(table_rows[0])
        _write_csv_atomic(paths["six_setting_control_table.csv"], table_rows, table_fields)
        _write_csv_atomic(
            paths["common_denominator_flow.csv"],
            flow_rows,
            list(flow_rows[0]),
        )
        _write_csv_atomic(
            paths["multiplicity_table.csv"],
            multiplicity_rows,
            list(multiplicity_rows[0]),
        )
        _write_json_atomic(paths["macro_average.json"], macro)
        _write_text_atomic(
            paths["risk_difference_forest.svg"],
            _forest_svg(multiplicity_rows),
        )
        counts = {
            "common_valid_pairs": sum(
                int(row["validity"]["common-valid"] is True) for row in status_rows
            ),
            "control_records": len(control_rows),
            "pair_status_records": len(status_rows),
            "selected_pairs": len(selected),
            "settings": len(REBUTTAL_SETTINGS),
        }
        outputs = {
            name: {
                "sha256": sha256_file(path),
                "records": (
                    len(control_rows)
                    if name == "control_records.jsonl"
                    else len(status_rows)
                    if name == "pair_status_records.jsonl"
                    else len(table_rows)
                    if name == "six_setting_control_table.csv"
                    else len(flow_rows)
                    if name == "common_denominator_flow.csv"
                    else len(multiplicity_rows)
                    if name == "multiplicity_table.csv"
                    else 1
                ),
            }
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
        raise SixSettingPatchControlsRunError(
            "six-setting patch controls failed; verified checkpoints were retained"
        ) from exc


__all__ = [
    "ControlArmResult",
    "ControlGeneration",
    "RuntimeFactory",
    "SixSettingControlRuntime",
    "SixSettingPatchControlsConfig",
    "SixSettingPatchControlsResult",
    "SixSettingPatchControlsRunError",
    "run_six_setting_patch_controls",
]
