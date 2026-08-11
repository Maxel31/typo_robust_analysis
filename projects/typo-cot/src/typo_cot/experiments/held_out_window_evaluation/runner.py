"""Two-phase runner for diagnostic selection and disjoint held-out evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Literal, Protocol

from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
)
from typo_cot.experiments.build_rebuttal_manifest.records import (
    REBUTTAL_COHORT_SCHEMA,
    iter_jsonl_objects,
    load_json_object,
    load_rebuttal_pair_manifest,
    sha256_file,
)
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.held_out_window_evaluation.protocol import (
    HeldOutWindowProtocol,
    load_held_out_window_protocol,
)
from typo_cot.experiments.patch_coordinate_controls.metrics import exact_mcnemar
from typo_cot.experiments.six_setting_patch_controls.statistics import (
    derived_seed,
    holm_adjust,
    nested_macro_bootstrap_risk_difference,
    paired_bootstrap_risk_difference,
)

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_RUN_SCHEMA = "held-out-window-evaluation-run/v1"
_SELECTION_SCHEMA = "held-out-window-selection/v1"
_CHECKPOINT_SCHEMA = "held-out-window-checkpoint/v1"
_STATUS_SCHEMA = "held-out-window-pair-status/v1"
_WINDOW_IDS = ("layers-0-6", "layers-6-12", "layers-12-18", "layers-18-24", "layers-22-28")
_PUBLIC_OUTPUTS = (
    "window_selection_records.jsonl",
    "window_selection.json",
    "held_out_window_records.jsonl",
    "held_out_window_table.csv",
    "held_out_window_contrasts.csv",
    "held_out_window_summary.json",
    "pair_status_records.jsonl",
)
_FINAL_OUTPUTS = _PUBLIC_OUTPUTS[2:6]


class HeldOutWindowRunError(RuntimeError):
    """Raised after valid phase checkpoints and status records are retained."""


@dataclass(frozen=True, slots=True)
class HeldOutWindowConfig:
    protocol_path: Path
    manifest_path: Path
    cohort_ids_path: Path
    gpu_id: str
    output_dir: Path
    limit_per_setting: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("protocol_path", "manifest_path", "cohort_ids_path", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit_per_setting is not None and (
            isinstance(self.limit_per_setting, bool)
            or not isinstance(self.limit_per_setting, int)
            or self.limit_per_setting < len(REBUTTAL_MANIFEST_PROTOCOL.target_rules)
        ):
            raise ValueError(
                "limit_per_setting must be at least 2 so both diagnostic target rules "
                "remain represented"
            )
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be boolean")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("protocol_path", "manifest_path", "cohort_ids_path", "output_dir"):
            payload[field] = str(Path(payload[field]).resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class WindowSelectionScan:
    available: bool
    invalid_reason: str | None
    target_token_id: int | None
    target_token_text: str | None
    untreated_kl: float | None
    patched_kl: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "patched_kl", tuple(self.patched_kl))
        if type(self.available) is not bool:
            raise TypeError("selection availability must be boolean")
        if not self.available:
            if self.invalid_reason not in {
                "clean_continuation_lt_1",
                "clean_prompt_not_exact_token_prefix",
            }:
                raise ValueError("unavailable selection scan must retain its frozen reason")
            if (
                self.target_token_id is not None
                or self.target_token_text is not None
                or self.untreated_kl is not None
                or self.patched_kl
            ):
                raise ValueError("unavailable selection scan must not contain KL payloads")
            return
        if self.invalid_reason is not None:
            raise ValueError("available selection scan must not have an invalid reason")
        if (
            isinstance(self.target_token_id, bool)
            or not isinstance(self.target_token_id, int)
            or self.target_token_id < 0
            or not isinstance(self.target_token_text, str)
            or not self.target_token_text
        ):
            raise ValueError("available selection target token is invalid")
        if (
            not isinstance(self.untreated_kl, (int, float))
            or isinstance(self.untreated_kl, bool)
            or not math.isfinite(float(self.untreated_kl))
            or float(self.untreated_kl) < 0.0
        ):
            raise ValueError("available selection untreated KL is invalid")
        if len(self.patched_kl) != 5 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in self.patched_kl
        ):
            raise ValueError("selection scan must contain one finite KL per candidate window")


@dataclass(frozen=True, slots=True)
class HeldOutGeneration:
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
            raise ValueError("held-out generation token_ids must be non-empty integers")
        if not isinstance(self.text, str) or not isinstance(self.value, str):
            raise TypeError("held-out generation text and value must be strings")
        if self.termination not in {"eos", "length-cap"}:
            raise ValueError("held-out generation termination must be eos or length-cap")
        if type(self.is_extracted) is not bool or self.is_extracted is not bool(self.value):
            raise ValueError("held-out generation extraction flag differs from its value")
        if type(self.is_correct) is not bool or (self.is_correct and not self.is_extracted):
            raise ValueError("held-out generation correctness differs from extraction")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("held-out generation method must be non-empty")
        if not isinstance(self.primary_method, str) or not self.primary_method:
            raise ValueError("held-out generation primary_method must be non-empty")


@dataclass(frozen=True, slots=True)
class WindowEvaluationScan:
    generations: Mapping[str, HeldOutGeneration]

    def __post_init__(self) -> None:
        if not isinstance(self.generations, Mapping):
            raise TypeError("held-out evaluation generations must be an object")
        normalized = dict(self.generations)
        if set(normalized) != {"selected", "runner-up"} or any(
            not isinstance(generation, HeldOutGeneration) for generation in normalized.values()
        ):
            raise ValueError("held-out evaluation scan must contain both frozen arms")
        object.__setattr__(self, "generations", normalized)


class HeldOutWindowRuntime(Protocol):
    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_selection(
        self,
        pair: Mapping[str, object],
        *,
        candidate_windows: tuple[tuple[int, int], ...],
    ) -> WindowSelectionScan: ...

    def scan_evaluation(
        self,
        pair: Mapping[str, object],
        *,
        windows: Mapping[str, tuple[int, int]],
    ) -> WindowEvaluationScan: ...


RuntimeFactory = Callable[..., HeldOutWindowRuntime]


@dataclass(frozen=True, slots=True)
class HeldOutWindowResult:
    selection_records_path: Path
    selection_path: Path
    evaluation_records_path: Path
    table_path: Path
    contrasts_path: Path
    summary_path: Path
    status_path: Path
    run_path: Path
    selection_pairs: int
    evaluation_pairs: int
    evaluation_records: int
    table_rows: int
    contrast_rows: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"held-out CSV cannot be empty: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _setting_key(record: Mapping[str, object]) -> tuple[str, str]:
    return str(record.get("model")), str(record.get("task"))


def _setting_id(key: tuple[str, str]) -> str:
    return f"{key[1]}|{key[0]}"


def _diagnostic_cell_id(record: Mapping[str, object]) -> str:
    return (
        f"{record.get('task')}|{record.get('model')}|{record.get('target_rule')}"
    )


def _window_id(window: tuple[int, int]) -> str:
    return f"layers-{window[0]}-{window[1]}"


def _window_payload(window: tuple[int, int]) -> dict[str, int]:
    return {"start": window[0], "stop": window[1]}


def _digest_list(values: Sequence[str]) -> str:
    return _canonical_sha256(list(values))


@dataclass(frozen=True, slots=True)
class _Partitions:
    selection: tuple[Mapping[str, object], ...]
    evaluation: tuple[Mapping[str, object], ...]
    manifest_sha256: str
    cohort_ids_sha256: str
    confirmatory: bool


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    values = tuple(value)
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and unique")
    return values


def _record_source_sha256(record: Mapping[str, object]) -> str:
    source = _mapping(record.get("source"), field="manifest record source")
    value = source.get("source_record_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("manifest source_record_sha256 differs")
    return value


def _validate_and_partition(
    config: HeldOutWindowConfig,
    protocol: HeldOutWindowProtocol,
    records: Sequence[Mapping[str, object]],
) -> _Partitions:
    manifest = config.manifest_path.resolve()
    cohort_path = config.cohort_ids_path.resolve()
    if cohort_path != manifest.with_name(protocol.cohort_ids_filename):
        raise ValueError("cohort-ids must be the canonical sibling of pair_manifest.jsonl")
    if not manifest.is_file() or not cohort_path.is_file():
        raise ValueError("held-out manifest artifact set is incomplete")
    manifest_sha256 = sha256_file(manifest)
    cohort_sha256 = sha256_file(cohort_path)
    cohort_payload = load_json_object(cohort_path)
    if (
        cohort_payload.get("schema_version") != REBUTTAL_COHORT_SCHEMA
        or cohort_payload.get("paper_sha256") != PAPER_SHA256
        or cohort_payload.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or cohort_payload.get("pair_manifest_sha256") != manifest_sha256
        or cohort_payload.get("identity") != "sha256-canonical-model-task-target-rule-sample-id/v1"
    ):
        raise ValueError("held-out cohort sidecar provenance differs")
    if _mapping(cohort_payload.get("window_split"), field="cohort window_split") != {
        "algorithm": protocol.split_algorithm,
        "seed": protocol.split_seed,
        "outcome_independent": True,
    }:
        raise ValueError("held-out cohort sidecar split contract differs")
    cohorts = _mapping(cohort_payload.get("cohorts"), field="cohort cohorts")
    restoration_ids = _string_list(cohorts.get("restoration"), field="cohorts.restoration")
    selection_ids = _string_list(
        cohorts.get(protocol.selection_cohort), field=f"cohorts.{protocol.selection_cohort}"
    )
    evaluation_ids = _string_list(
        cohorts.get(protocol.evaluation_cohort), field=f"cohorts.{protocol.evaluation_cohort}"
    )
    restoration_set = set(restoration_ids)
    selection_set = set(selection_ids)
    evaluation_set = set(evaluation_ids)
    if selection_set & evaluation_set:
        raise ValueError("held-out selection and evaluation cohorts must be disjoint")
    if selection_set | evaluation_set != restoration_set:
        raise ValueError("held-out cohorts do not partition the restoration cohort")

    by_id: dict[str, Mapping[str, object]] = {}
    setting_counts: dict[tuple[str, str], int] = defaultdict(int)
    sample_phases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("held-out manifest pair_id differs")
        if pair_id in by_id:
            raise ValueError(f"held-out manifest pair_id is duplicated: {pair_id}")
        by_id[pair_id] = record
        cohorts_for_record = _mapping(record.get("cohorts"), field=f"{pair_id} cohorts")
        restoration = cohorts_for_record.get("restoration")
        selection = cohorts_for_record.get(protocol.selection_cohort)
        evaluation = cohorts_for_record.get(protocol.evaluation_cohort)
        if any(type(value) is not bool for value in (restoration, selection, evaluation)):
            raise ValueError(f"held-out cohort flags differ for {pair_id}")
        if restoration is not (pair_id in restoration_set):
            raise ValueError(f"held-out restoration sidecar differs for {pair_id}")
        if selection is not (pair_id in selection_set) or evaluation is not (
            pair_id in evaluation_set
        ):
            raise ValueError(f"held-out split sidecar differs for {pair_id}")
        if not restoration:
            continue
        if (
            record.get("clean_correct") is not True
            or record.get("typo_correct") is not False
            or not isinstance(record.get("clean_answer"), str)
            or not record.get("clean_answer")
            or isinstance(record.get("number_of_aligned_words"), bool)
            or not isinstance(record.get("number_of_aligned_words"), int)
            or int(record["number_of_aligned_words"]) <= 0
        ):
            raise ValueError(f"held-out pair is outside the paper restoration cohort: {pair_id}")
        key = _setting_key(record)
        if key not in {setting.key for setting in REBUTTAL_SETTINGS}:
            raise ValueError(f"held-out pair has an unknown setting: {pair_id}")
        setting_counts[key] += 1
        sample_id = record.get("sample_id")
        target_rule = record.get("target_rule")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(target_rule, str)
            or not target_rule
        ):
            raise ValueError(f"held-out pair lacks its shared sample identity: {pair_id}")
        phase = "selection" if pair_id in selection_set else "evaluation"
        sample_phases[(str(record.get("task")), sample_id)].add(phase)
    if set(by_id) & restoration_set != restoration_set:
        raise ValueError("held-out cohort sidecar names absent manifest pairs")
    for setting in REBUTTAL_SETTINGS:
        if setting_counts[setting.key] != setting.paper_denominator:
            raise ValueError(f"held-out paper denominator differs for {setting.slug}")
    if any(len(phases) != 1 for phases in sample_phases.values()):
        raise ValueError("held-out sample groups are not disjoint across phases")

    def selected_records(pair_ids: set[str], *, phase: str) -> tuple[Mapping[str, object], ...]:
        selected: list[Mapping[str, object]] = []
        for setting in REBUTTAL_SETTINGS:
            setting_records = sorted(
                (
                    record
                    for pair_id, record in by_id.items()
                    if pair_id in pair_ids and _setting_key(record) == setting.key
                ),
                key=lambda record: str(record["pair_id"]),
            )
            if not setting_records:
                raise ValueError(f"held-out {phase} cohort is empty for {setting.slug}")
            if config.limit_per_setting is not None:
                if phase == "selection":
                    first_per_rule: list[Mapping[str, object]] = []
                    for target_rule in REBUTTAL_MANIFEST_PROTOCOL.target_rules:
                        matching = tuple(
                            record
                            for record in setting_records
                            if record.get("target_rule") == target_rule
                        )
                        if not matching:
                            raise ValueError(
                                f"held-out selection target-rule cell is empty for "
                                f"{setting.slug}/{target_rule}"
                            )
                        first_per_rule.append(matching[0])
                    first_ids = {str(record["pair_id"]) for record in first_per_rule}
                    remaining = [
                        record
                        for record in setting_records
                        if str(record["pair_id"]) not in first_ids
                    ]
                    setting_records = sorted(
                        first_per_rule
                        + remaining[
                            : config.limit_per_setting
                            - len(REBUTTAL_MANIFEST_PROTOCOL.target_rules)
                        ],
                        key=lambda record: str(record["pair_id"]),
                    )
                else:
                    setting_records = setting_records[: config.limit_per_setting]
            selected.extend(setting_records)
        return tuple(selected)

    return _Partitions(
        selection=selected_records(selection_set, phase="selection"),
        evaluation=selected_records(evaluation_set, phase="evaluation"),
        manifest_sha256=manifest_sha256,
        cohort_ids_sha256=cohort_sha256,
        confirmatory=config.limit_per_setting is None,
    )


def _runtime_factory() -> RuntimeFactory:
    from typo_cot.experiments.held_out_window_evaluation.runtime import (
        HuggingFaceHeldOutWindowRuntime,
    )

    return HuggingFaceHeldOutWindowRuntime


def _runtime_provenance(
    runtime: HeldOutWindowRuntime,
    *,
    protocol: HeldOutWindowProtocol,
    revision: str,
    model: str,
    task: str,
    gpu_id: str,
) -> dict[str, object]:
    payload = dict(runtime.provenance())
    required = {
        "operation": "held-out-window-evaluation",
        "model": model,
        "task": task,
        "requested_revision": revision,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "dtype": "bfloat16",
        "cuda_visible_devices": gpu_id,
        "direction": protocol.direction,
        "coordinate_source": protocol.coordinate_source,
        "diagnostic_readout": protocol.diagnostic_readout,
        "candidate_windows": [list(window) for window in protocol.candidate_windows],
        "generation_termination_protocol": protocol.generation["termination_protocol"],
        "answer_extraction": protocol.answer_extraction,
    }
    if any(payload.get(field) != value for field, value in required.items()):
        raise ValueError("held-out runtime provenance differs from the frozen protocol")
    layers = payload.get("num_decoder_layers")
    if (
        isinstance(layers, bool)
        or not isinstance(layers, int)
        or layers < 28
        or runtime.num_layers != layers
    ):
        raise ValueError("held-out runtime decoder-layer inventory differs")
    eos_ids = payload.get("effective_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in eos_ids
        )
        or len(eos_ids) != len(set(eos_ids))
    ):
        raise ValueError("held-out runtime effective EOS inventory differs")
    generation = _mapping(payload.get("generation"), field="held-out runtime generation")
    for field, value in protocol.generation.items():
        if field in {"strategy", "dtype", "termination_protocol"}:
            continue
        if generation.get(field) != value:
            raise ValueError(f"held-out runtime generation.{field} differs")
    if generation.get("patch_application") != ("selected-window-on-typo-prefill-exactly-once/v1"):
        raise ValueError("held-out runtime patch application differs")
    return payload


def _generation_payload(generation: HeldOutGeneration) -> dict[str, object]:
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


def _generation_from_payload(value: object, *, field: str) -> HeldOutGeneration:
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
    return HeldOutGeneration(
        token_ids=tuple(payload["token_ids"]),  # type: ignore[arg-type]
        text=payload.get("text"),  # type: ignore[arg-type]
        termination=payload.get("termination"),  # type: ignore[arg-type]
        value=payload.get("value"),  # type: ignore[arg-type]
        is_extracted=payload.get("is_extracted"),  # type: ignore[arg-type]
        is_correct=payload.get("is_correct"),  # type: ignore[arg-type]
        method=payload.get("method"),  # type: ignore[arg-type]
        primary_method=payload.get("primary_method"),  # type: ignore[arg-type]
    )


def _selection_scan_payload(scan: WindowSelectionScan) -> dict[str, object]:
    return {
        "available": scan.available,
        "invalid_reason": scan.invalid_reason,
        "target_token_id": scan.target_token_id,
        "target_token_text": scan.target_token_text,
        "untreated_kl": scan.untreated_kl,
        "patched_kl": list(scan.patched_kl),
    }


def _selection_scan_from_payload(value: object) -> WindowSelectionScan:
    payload = _mapping(value, field="selection checkpoint scan")
    if set(payload) != {
        "available",
        "invalid_reason",
        "target_token_id",
        "target_token_text",
        "untreated_kl",
        "patched_kl",
    } or not isinstance(payload.get("patched_kl"), list):
        raise ValueError("selection checkpoint scan fields differ")
    return WindowSelectionScan(
        available=payload.get("available"),  # type: ignore[arg-type]
        invalid_reason=payload.get("invalid_reason"),  # type: ignore[arg-type]
        target_token_id=payload.get("target_token_id"),  # type: ignore[arg-type]
        target_token_text=payload.get("target_token_text"),  # type: ignore[arg-type]
        untreated_kl=payload.get("untreated_kl"),  # type: ignore[arg-type]
        patched_kl=tuple(payload["patched_kl"]),  # type: ignore[arg-type]
    )


def _checkpoint_path(output_dir: Path, phase: str, pair_id: str) -> Path:
    return output_dir / "checkpoints" / phase / f"{pair_id}.json"


def _checkpoint_common(
    *,
    phase: str,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
) -> dict[str, object]:
    return {
        "schema_version": _CHECKPOINT_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "evaluation_protocol_sha256": protocol.config_sha256,
        "pair_manifest_sha256": partitions.manifest_sha256,
        "cohort_ids_sha256": partitions.cohort_ids_sha256,
        "phase": phase,
        "pair_id": record["pair_id"],
        "sample_id": record["sample_id"],
        "source_record_sha256": _record_source_sha256(record),
        "runtime_fingerprint": runtime_fingerprint,
    }


def _validate_checkpoint_common(
    payload: Mapping[str, object],
    *,
    phase: str,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
) -> None:
    common = _checkpoint_common(
        phase=phase,
        record=record,
        runtime_fingerprint=runtime_fingerprint,
        protocol=protocol,
        partitions=partitions,
    )
    if any(payload.get(field) != value for field, value in common.items()):
        raise ValueError(f"held-out {phase} checkpoint provenance differs")


def _write_selection_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    scan: WindowSelectionScan,
    runtime_fingerprint: str,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
) -> None:
    _write_json_atomic(
        path,
        {
            **_checkpoint_common(
                phase="selection",
                record=record,
                runtime_fingerprint=runtime_fingerprint,
                protocol=protocol,
                partitions=partitions,
            ),
            "candidate_windows": [list(window) for window in protocol.candidate_windows],
            "scan": _selection_scan_payload(scan),
        },
    )


def _load_selection_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
) -> WindowSelectionScan:
    payload = load_json_object(path)
    if set(payload) != set(
        _checkpoint_common(
            phase="selection",
            record=record,
            runtime_fingerprint=runtime_fingerprint,
            protocol=protocol,
            partitions=partitions,
        )
    ) | {"candidate_windows", "scan"}:
        raise ValueError(f"selection checkpoint fields differ: {path}")
    _validate_checkpoint_common(
        payload,
        phase="selection",
        record=record,
        runtime_fingerprint=runtime_fingerprint,
        protocol=protocol,
        partitions=partitions,
    )
    if payload.get("candidate_windows") != [list(window) for window in protocol.candidate_windows]:
        raise ValueError(f"selection checkpoint candidate windows differ: {path}")
    return _selection_scan_from_payload(payload.get("scan"))


def _validate_generation(
    generation: HeldOutGeneration,
    *,
    record: Mapping[str, object],
    arm: str,
) -> HeldOutGeneration:
    expected = answers_equal(
        generation.value,
        str(record["clean_answer"]),
        benchmark=str(record["task"]),
    )
    if generation.is_correct is not expected:
        raise ValueError(f"held-out {arm} correctness differs from deterministic scoring")
    return generation


def _write_evaluation_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    scan: WindowEvaluationScan,
    runtime_fingerprint: str,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
    selection_sha256: str,
    windows: Mapping[str, tuple[int, int]],
) -> None:
    _write_json_atomic(
        path,
        {
            **_checkpoint_common(
                phase="evaluation",
                record=record,
                runtime_fingerprint=runtime_fingerprint,
                protocol=protocol,
                partitions=partitions,
            ),
            "selection_sha256": selection_sha256,
            "windows": {arm: list(window) for arm, window in sorted(windows.items())},
            "generations": {
                arm: _generation_payload(generation)
                for arm, generation in sorted(scan.generations.items())
            },
        },
    )


def _load_evaluation_checkpoint(
    path: Path,
    *,
    record: Mapping[str, object],
    runtime_fingerprint: str,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
    selection_sha256: str,
    windows: Mapping[str, tuple[int, int]],
) -> WindowEvaluationScan:
    payload = load_json_object(path)
    common = _checkpoint_common(
        phase="evaluation",
        record=record,
        runtime_fingerprint=runtime_fingerprint,
        protocol=protocol,
        partitions=partitions,
    )
    if set(payload) != set(common) | {"selection_sha256", "windows", "generations"}:
        raise ValueError(f"evaluation checkpoint fields differ: {path}")
    _validate_checkpoint_common(
        payload,
        phase="evaluation",
        record=record,
        runtime_fingerprint=runtime_fingerprint,
        protocol=protocol,
        partitions=partitions,
    )
    if payload.get("selection_sha256") != selection_sha256 or payload.get("windows") != {
        arm: list(window) for arm, window in sorted(windows.items())
    }:
        raise ValueError(f"evaluation checkpoint selection binding differs: {path}")
    stored = _mapping(payload.get("generations"), field="evaluation checkpoint generations")
    if set(stored) != set(protocol.evaluation_arms):
        raise ValueError(f"evaluation checkpoint arm inventory differs: {path}")
    return WindowEvaluationScan(
        generations={
            arm: _validate_generation(
                _generation_from_payload(stored[arm], field=f"evaluation {arm}"),
                record=record,
                arm=arm,
            )
            for arm in protocol.evaluation_arms
        }
    )


def _selection_record(
    record: Mapping[str, object],
    scan: WindowSelectionScan,
    *,
    protocol: HeldOutWindowProtocol,
) -> dict[str, object]:
    scoring_available = bool(
        scan.available
        and scan.untreated_kl is not None
        and scan.untreated_kl > protocol.untreated_kl_min_exclusive
    )
    reason = scan.invalid_reason
    if scan.available and not scoring_available:
        reason = "untreated-kl-at-or-below-threshold"
    candidates: list[dict[str, object]] = []
    for index, window in enumerate(protocol.candidate_windows):
        patched = scan.patched_kl[index] if scan.available else None
        restoration = (
            1.0 - float(patched) / float(scan.untreated_kl)
            if scoring_available and patched is not None and scan.untreated_kl is not None
            else None
        )
        candidates.append(
            {
                "window_id": _window_id(window),
                **_window_payload(window),
                "patched_kl": patched,
                "normalized_restoration": restoration,
            }
        )
    return {
        "schema_version": "held-out-window-selection-record/v1",
        "paper_sha256": PAPER_SHA256,
        "pair_id": record["pair_id"],
        "sample_id": record["sample_id"],
        "model": record["model"],
        "task": record["task"],
        "target_rule": record["target_rule"],
        "target_token_id": scan.target_token_id,
        "target_token_text": scan.target_token_text,
        "scoring_available": scoring_available,
        "invalid_reason": reason,
        "untreated_kl": scan.untreated_kl,
        "candidates": candidates,
    }


def _rank_windows(
    rows: Sequence[Mapping[str, object]],
    *,
    protocol: HeldOutWindowProtocol,
) -> list[dict[str, object]]:
    scores: dict[str, dict[str, list[float]]] = {
        _window_id(window): defaultdict(list) for window in protocol.candidate_windows
    }
    for row in rows:
        if row.get("scoring_available") is not True:
            continue
        cell = _diagnostic_cell_id(row)
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(protocol.candidate_windows):
            raise ValueError("selection record candidate inventory differs")
        for candidate in candidates:
            payload = _mapping(candidate, field="selection record candidate")
            window_id = str(payload.get("window_id"))
            value = payload.get("normalized_restoration")
            if window_id not in scores or not isinstance(value, (int, float)):
                raise ValueError("selection record score differs")
            scores[window_id][cell].append(float(value))
    summaries: list[dict[str, object]] = []
    expected_cells = {
        f"{setting.task}|{setting.model}|{target_rule}"
        for setting in REBUTTAL_SETTINGS
        for target_rule in REBUTTAL_MANIFEST_PROTOCOL.target_rules
    }
    by_window = {_window_id(window): window for window in protocol.candidate_windows}
    for window_id, cell_values in scores.items():
        if set(cell_values) != expected_cells or any(
            not values for values in cell_values.values()
        ):
            raise ValueError(f"diagnostic scoring is unavailable for a cell: {window_id}")
        cell_medians = {
            cell: float(median(values)) for cell, values in sorted(cell_values.items())
        }
        window = by_window[window_id]
        summaries.append(
            {
                "window_id": window_id,
                **_window_payload(window),
                "macro_score": sum(cell_medians.values()) / len(cell_medians),
                "cell_medians": cell_medians,
                "cell_available_pairs": {
                    cell: len(values) for cell, values in sorted(cell_values.items())
                },
            }
        )
    summaries.sort(key=lambda row: (-float(row["macro_score"]), int(row["start"])))
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
    return summaries


def _window_from_object(value: object, *, field: str) -> tuple[int, int]:
    payload = _mapping(value, field=field)
    if set(payload) != {"start", "stop"}:
        raise ValueError(f"{field} fields differ")
    start, stop = payload.get("start"), payload.get("stop")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or stop - start != 6
    ):
        raise ValueError(f"{field} is not a six-layer window")
    return start, stop


def _selection_payload(
    *,
    rows_path: Path,
    summaries: Sequence[Mapping[str, object]],
    partitions: _Partitions,
    protocol: HeldOutWindowProtocol,
    runtime_by_setting: Mapping[str, object],
) -> dict[str, object]:
    if len(summaries) != len(protocol.candidate_windows):
        raise ValueError("held-out selection did not rank every candidate window")
    selected = summaries[0]
    runner_up = summaries[1]
    return {
        "schema_version": _SELECTION_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "evaluation_protocol_sha256": protocol.config_sha256,
        "pair_manifest_sha256": partitions.manifest_sha256,
        "cohort_ids_sha256": partitions.cohort_ids_sha256,
        "selection_metric": protocol.selection_metric,
        "cross_setting_score": protocol.cross_setting_score,
        "ranking_rule": protocol.ranking,
        "candidate_summaries": list(summaries),
        "selected_window": {"start": selected["start"], "stop": selected["stop"]},
        "runner_up_window": {
            "start": runner_up["start"],
            "stop": runner_up["stop"],
        },
        "selection_pairs": len(partitions.selection),
        "evaluation_pairs": len(partitions.evaluation),
        "selection_pair_ids_sha256": _digest_list(
            sorted(str(record["pair_id"]) for record in partitions.selection)
        ),
        "evaluation_pair_ids_sha256": _digest_list(
            sorted(str(record["pair_id"]) for record in partitions.evaluation)
        ),
        "selection_records_sha256": sha256_file(rows_path),
        "runtime_by_setting": dict(runtime_by_setting),
        "committed_at": _now(),
    }


def _locked_selection(
    output_dir: Path,
    *,
    commit: object,
    partitions: _Partitions,
    protocol: HeldOutWindowProtocol,
) -> tuple[dict[str, object], dict[str, tuple[int, int]]]:
    commit_payload = _mapping(commit, field="run selection_commit")
    if set(commit_payload) != {
        "sha256",
        "records_sha256",
        "selected_window",
        "runner_up_window",
    }:
        raise ValueError("held-out selection commit fields differ")
    selection_path = output_dir / "window_selection.json"
    rows_path = output_dir / "window_selection_records.jsonl"
    if not selection_path.is_file() or not rows_path.is_file():
        raise ValueError("held-out committed selection artifacts are missing")
    if commit_payload.get("sha256") != sha256_file(selection_path) or commit_payload.get(
        "records_sha256"
    ) != sha256_file(rows_path):
        raise ValueError("held-out committed selection SHA-256 differs")
    payload = load_json_object(selection_path)
    if (
        payload.get("schema_version") != _SELECTION_SCHEMA
        or payload.get("paper_sha256") != PAPER_SHA256
        or payload.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or payload.get("evaluation_protocol_sha256") != protocol.config_sha256
        or payload.get("pair_manifest_sha256") != partitions.manifest_sha256
        or payload.get("cohort_ids_sha256") != partitions.cohort_ids_sha256
        or payload.get("selection_pairs") != len(partitions.selection)
        or payload.get("evaluation_pairs") != len(partitions.evaluation)
        or payload.get("selection_pair_ids_sha256")
        != _digest_list(sorted(str(record["pair_id"]) for record in partitions.selection))
        or payload.get("evaluation_pair_ids_sha256")
        != _digest_list(sorted(str(record["pair_id"]) for record in partitions.evaluation))
        or payload.get("selection_records_sha256") != sha256_file(rows_path)
    ):
        raise ValueError("held-out committed selection provenance differs")
    rows = tuple(row for _number, _line, row in iter_jsonl_objects(rows_path))
    if len(rows) != len(partitions.selection):
        raise ValueError("held-out committed selection record count differs")
    windows = {
        "selected": _window_from_object(payload.get("selected_window"), field="selected window"),
        "runner-up": _window_from_object(payload.get("runner_up_window"), field="runner-up window"),
    }
    if (
        windows["selected"] == windows["runner-up"]
        or windows["selected"] not in protocol.candidate_windows
        or windows["runner-up"] not in protocol.candidate_windows
        or commit_payload.get("selected_window") != payload.get("selected_window")
        or commit_payload.get("runner_up_window") != payload.get("runner_up_window")
    ):
        raise ValueError("held-out committed windows differ from the frozen candidates")
    return payload, windows


def _evaluation_rows(
    records: Sequence[Mapping[str, object]],
    scans: Mapping[str, WindowEvaluationScan],
    *,
    windows: Mapping[str, tuple[int, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        pair_id = str(record["pair_id"])
        scan = scans.get(pair_id)
        if scan is None:
            raise ValueError(f"held-out evaluation scan is missing: {pair_id}")
        for arm in ("selected", "runner-up"):
            generation = scan.generations[arm]
            start, stop = windows[arm]
            rows.append(
                {
                    "schema_version": "held-out-window-evaluation-record/v1",
                    "paper_sha256": PAPER_SHA256,
                    "pair_id": pair_id,
                    "sample_id": record["sample_id"],
                    "model": record["model"],
                    "task": record["task"],
                    "target_rule": record["target_rule"],
                    "arm": arm,
                    "window_start": start,
                    "window_stop": stop,
                    "token_ids": list(generation.token_ids),
                    "text": generation.text,
                    "termination": generation.termination,
                    "value": generation.value,
                    "is_extracted": generation.is_extracted,
                    "is_correct": generation.is_correct,
                    "method": generation.method,
                    "primary_method": generation.primary_method,
                }
            )
    return rows


def _analysis(
    records: Sequence[Mapping[str, object]],
    scans: Mapping[str, WindowEvaluationScan],
    *,
    windows: Mapping[str, tuple[int, int]],
    protocol: HeldOutWindowProtocol,
    confirmatory: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    tables: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    macro_inputs: dict[str, tuple[tuple[bool, ...], tuple[bool, ...]]] = {}
    for setting in REBUTTAL_SETTINGS:
        setting_records = [record for record in records if _setting_key(record) == setting.key]
        if not setting_records:
            raise ValueError(f"held-out evaluation setting is empty: {setting.slug}")
        selected = tuple(
            scans[str(record["pair_id"])].generations["selected"].is_correct
            for record in setting_records
        )
        runner_up = tuple(
            scans[str(record["pair_id"])].generations["runner-up"].is_correct
            for record in setting_records
        )
        label = _setting_id(setting.key)
        mcnemar = exact_mcnemar(selected, runner_up)
        bootstrap = paired_bootstrap_risk_difference(
            selected,
            runner_up,
            replicates=protocol.pair_bootstrap_replicates,
            confidence_level=protocol.confidence_level,
            seed=derived_seed(protocol.bootstrap_seed, f"held-out-window|{label}"),
        )
        tables.append(
            {
                "model": setting.model,
                "task": setting.task,
                "n": len(setting_records),
                "selected_window": _window_id(windows["selected"]),
                "selected_successes": sum(selected),
                "selected_rate": sum(selected) / len(selected),
                "runner_up_window": _window_id(windows["runner-up"]),
                "runner_up_successes": sum(runner_up),
                "runner_up_rate": sum(runner_up) / len(runner_up),
            }
        )
        contrasts.append(
            {
                "comparison_id": f"{label}|selected-vs-runner-up",
                "model": setting.model,
                "task": setting.task,
                "n": len(setting_records),
                "selected_window": _window_id(windows["selected"]),
                "runner_up_window": _window_id(windows["runner-up"]),
                "risk_difference": bootstrap["estimate"],
                "ci_lower": bootstrap["lower"],
                "ci_upper": bootstrap["upper"],
                "both_success": mcnemar["both_success"],
                "selected_only": mcnemar["left_only"],
                "runner_up_only": mcnemar["right_only"],
                "both_failure": mcnemar["both_failure"],
                "mcnemar_p_raw": mcnemar["p_value"],
                "mcnemar_p_holm": None,
                "mcnemar_p_holm_defined": False,
                "holm_family": protocol.multiplicity,
                "holm_family_size": 0,
            }
        )
        macro_inputs[label] = (selected, runner_up)
    if confirmatory:
        adjusted = holm_adjust(
            {str(row["comparison_id"]): float(row["mcnemar_p_raw"]) for row in contrasts}
        )
        for row in contrasts:
            row["mcnemar_p_holm"] = adjusted[str(row["comparison_id"])]
            row["mcnemar_p_holm_defined"] = True
            row["holm_family_size"] = len(contrasts)
    macro = nested_macro_bootstrap_risk_difference(
        macro_inputs,
        replicates=protocol.nested_bootstrap_replicates,
        confidence_level=protocol.confidence_level,
        seed=derived_seed(protocol.bootstrap_seed, "held-out-window|macro"),
    )
    summary = {
        "schema_version": "held-out-window-evaluation-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "confirmatory": confirmatory,
        "selected_window": _window_payload(windows["selected"]),
        "runner_up_window": _window_payload(windows["runner-up"]),
        "macro_difference": macro,
        "initial_six_advantage_reproduced": bool(
            confirmatory and windows["selected"] == (0, 6) and float(macro["lower"]) > 0.0
        ),
        "reproduction_rule": protocol.reproduction_rule,
        "multiplicity": {
            "method": protocol.multiplicity,
            "defined": confirmatory,
            "family_size": len(contrasts) if confirmatory else 0,
        },
    }
    return tables, contrasts, summary


def _status_rows(
    partitions: _Partitions,
    *,
    selection_scans: Mapping[str, WindowSelectionScan],
    evaluation_scans: Mapping[str, WindowEvaluationScan],
    active: tuple[str, Mapping[str, object]] | None = None,
    failure: BaseException | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    active_phase = active[0] if active is not None else None
    active_id = str(active[1]["pair_id"]) if active is not None else None
    for phase, records, completed in (
        ("selection", partitions.selection, selection_scans),
        ("evaluation", partitions.evaluation, evaluation_scans),
    ):
        for record in records:
            pair_id = str(record["pair_id"])
            if pair_id in completed:
                status = "completed"
            elif phase == active_phase and pair_id == active_id and failure is not None:
                status = "failed"
            else:
                status = "pending"
            rows.append(
                {
                    "schema_version": _STATUS_SCHEMA,
                    "paper_sha256": PAPER_SHA256,
                    "phase": phase,
                    "pair_id": pair_id,
                    "sample_id": record["sample_id"],
                    "model": record["model"],
                    "task": record["task"],
                    "status": status,
                    "checkpoint": (
                        f"checkpoints/{phase}/{pair_id}.json" if status == "completed" else None
                    ),
                    "failure_type": type(failure).__name__ if status == "failed" else None,
                    "failure_message": str(failure) if status == "failed" else None,
                }
            )
    return rows


def _csv_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(reader)
    if reader.fieldnames is None:
        raise ValueError(f"held-out CSV has no header: {path}")
    return len(rows)


def _result_from_run(
    output_dir: Path,
    run: Mapping[str, object],
    *,
    partitions: _Partitions,
    protocol: HeldOutWindowProtocol,
) -> HeldOutWindowResult:
    outputs = _mapping(run.get("outputs"), field="completed held-out outputs")
    if set(outputs) != set(_PUBLIC_OUTPUTS):
        raise ValueError("completed held-out output inventory differs")
    counts_by_name = {
        "window_selection_records.jsonl": len(
            tuple(iter_jsonl_objects(output_dir / "window_selection_records.jsonl"))
        ),
        "window_selection.json": 1,
        "held_out_window_records.jsonl": len(
            tuple(iter_jsonl_objects(output_dir / "held_out_window_records.jsonl"))
        ),
        "held_out_window_table.csv": _csv_count(output_dir / "held_out_window_table.csv"),
        "held_out_window_contrasts.csv": _csv_count(output_dir / "held_out_window_contrasts.csv"),
        "held_out_window_summary.json": 1,
        "pair_status_records.jsonl": len(
            tuple(iter_jsonl_objects(output_dir / "pair_status_records.jsonl"))
        ),
    }
    load_json_object(output_dir / "window_selection.json")
    load_json_object(output_dir / "held_out_window_summary.json")
    for name, expected_count in counts_by_name.items():
        path = output_dir / name
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"completed held-out output SHA-256 differs: {path}")
        if metadata.get("records") != expected_count:
            raise ValueError(f"completed held-out output count differs: {name}")
    expected_counts = {
        "selection_pairs": len(partitions.selection),
        "evaluation_pairs": len(partitions.evaluation),
        "evaluation_records": 2 * len(partitions.evaluation),
        "table_rows": len(REBUTTAL_SETTINGS),
        "contrast_rows": len(REBUTTAL_SETTINGS),
        "pair_status_records": len(partitions.selection) + len(partitions.evaluation),
        "settings": len(REBUTTAL_SETTINGS),
    }
    if run.get("counts") != expected_counts:
        raise ValueError("completed held-out run counts differ")
    if counts_by_name != {
        "window_selection_records.jsonl": expected_counts["selection_pairs"],
        "window_selection.json": 1,
        "held_out_window_records.jsonl": expected_counts["evaluation_records"],
        "held_out_window_table.csv": expected_counts["table_rows"],
        "held_out_window_contrasts.csv": expected_counts["contrast_rows"],
        "held_out_window_summary.json": 1,
        "pair_status_records.jsonl": expected_counts["pair_status_records"],
    }:
        raise ValueError("completed held-out artifact grid is incomplete")
    _locked_selection(
        output_dir,
        commit=run.get("selection_commit"),
        partitions=partitions,
        protocol=protocol,
    )
    return HeldOutWindowResult(
        selection_records_path=output_dir / "window_selection_records.jsonl",
        selection_path=output_dir / "window_selection.json",
        evaluation_records_path=output_dir / "held_out_window_records.jsonl",
        table_path=output_dir / "held_out_window_table.csv",
        contrasts_path=output_dir / "held_out_window_contrasts.csv",
        summary_path=output_dir / "held_out_window_summary.json",
        status_path=output_dir / "pair_status_records.jsonl",
        run_path=output_dir / "run.json",
        selection_pairs=expected_counts["selection_pairs"],
        evaluation_pairs=expected_counts["evaluation_pairs"],
        evaluation_records=expected_counts["evaluation_records"],
        table_rows=expected_counts["table_rows"],
        contrast_rows=expected_counts["contrast_rows"],
    )


def _completed_resume(
    config: HeldOutWindowConfig,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
) -> HeldOutWindowResult | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "held-out-window-evaluation"
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or run.get("pair_manifest_sha256") != partitions.manifest_sha256
        or run.get("cohort_ids_sha256") != partitions.cohort_ids_sha256
        or run.get("confirmatory") is not partitions.confirmatory
    ):
        raise ValueError("completed held-out resume contract differs")
    return _result_from_run(
        config.output_dir,
        run,
        partitions=partitions,
        protocol=protocol,
    )


def _revision(records: Sequence[Mapping[str, object]], *, setting: str) -> str:
    values = {_record_source_sha256(record) for record in records}
    if not values:
        raise ValueError(f"held-out selected no records for {setting}")
    revisions = {
        str(_mapping(record.get("source"), field="manifest source").get("model_revision"))
        for record in records
    }
    if len(revisions) != 1 or not next(iter(revisions)):
        raise ValueError(f"held-out model revisions differ for {setting}")
    return next(iter(revisions))


def _resume_run(
    config: HeldOutWindowConfig,
    protocol: HeldOutWindowProtocol,
    partitions: _Partitions,
) -> dict[str, object] | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "held-out-window-evaluation"
        or run.get("status") not in {"running", "failed"}
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or run.get("pair_manifest_sha256") != partitions.manifest_sha256
        or run.get("cohort_ids_sha256") != partitions.cohort_ids_sha256
        or run.get("confirmatory") is not partitions.confirmatory
    ):
        raise ValueError("held-out resume contract differs")
    return run


def run_held_out_window_evaluation(
    config: HeldOutWindowConfig,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> HeldOutWindowResult:
    """Select on diagnostic IDs, commit, then evaluate on untouched IDs."""

    if not isinstance(config, HeldOutWindowConfig):
        raise TypeError("config must be HeldOutWindowConfig")
    protocol = load_held_out_window_protocol(config.protocol_path)
    if not config.manifest_path.is_file():
        raise ValueError(f"rebuttal manifest is not a file: {config.manifest_path}")
    records = load_rebuttal_pair_manifest(config.manifest_path)
    partitions = _validate_and_partition(config, protocol, records)
    completed = _completed_resume(config, protocol, partitions)
    if completed is not None:
        return completed
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.resume:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        if not (output_dir / "run.json").is_file():
            raise ValueError("non-empty held-out resume directory is missing run.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    for phase in ("selection", "evaluation"):
        (output_dir / "checkpoints" / phase).mkdir(parents=True, exist_ok=True)

    previous = _resume_run(config, protocol, partitions)
    started_at = (
        str(previous.get("started_at"))
        if previous is not None and isinstance(previous.get("started_at"), str)
        else _now()
    )
    previous_runtime = previous.get("runtime_by_setting") if previous is not None else None
    runtime_by_setting: dict[str, object] = (
        dict(previous_runtime) if isinstance(previous_runtime, Mapping) else {}
    )
    for phase in ("selection", "evaluation"):
        if not isinstance(runtime_by_setting.get(phase), Mapping):
            runtime_by_setting[phase] = {}
        else:
            runtime_by_setting[phase] = dict(runtime_by_setting[phase])  # type: ignore[arg-type]
    selection_commit = previous.get("selection_commit") if previous is not None else None
    phase = "evaluation" if selection_commit is not None else "selection"
    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "held-out-window-evaluation",
        "status": "running",
        "phase": phase,
        "confirmatory": partitions.confirmatory,
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "pair_manifest_sha256": partitions.manifest_sha256,
        "cohort_ids_sha256": partitions.cohort_ids_sha256,
        "started_at": started_at,
        "updated_at": _now(),
        "runtime_by_setting": runtime_by_setting,
        "selection_commit": selection_commit,
        "counts": {
            "selection_pairs": len(partitions.selection),
            "evaluation_pairs": len(partitions.evaluation),
            "settings": len(REBUTTAL_SETTINGS),
        },
        "failures": [],
    }
    _write_json_atomic(output_dir / "run.json", base_run)
    factory = runtime_factory or _runtime_factory()
    selection_scans: dict[str, WindowSelectionScan] = {}
    evaluation_scans: dict[str, WindowEvaluationScan] = {}
    active: tuple[str, Mapping[str, object]] | None = None
    try:
        if selection_commit is None:
            phase = "selection"
            for setting in REBUTTAL_SETTINGS:
                setting_records = tuple(
                    record for record in partitions.selection if _setting_key(record) == setting.key
                )
                active = (phase, setting_records[0])
                revision = _revision(setting_records, setting=setting.slug)
                runtime = factory(
                    model=setting.model,
                    task=setting.task,
                    revision=revision,
                    gpu_id=config.gpu_id,
                )
                provenance = _runtime_provenance(
                    runtime,
                    protocol=protocol,
                    revision=revision,
                    model=setting.model,
                    task=setting.task,
                    gpu_id=config.gpu_id,
                )
                fingerprint = _canonical_sha256(provenance)
                selection_runtime = runtime_by_setting["selection"]
                assert isinstance(selection_runtime, dict)
                selection_runtime[_setting_id(setting.key)] = provenance
                for record in setting_records:
                    active = (phase, record)
                    pair_id = str(record["pair_id"])
                    checkpoint = _checkpoint_path(output_dir, phase, pair_id)
                    if checkpoint.is_file():
                        scan = _load_selection_checkpoint(
                            checkpoint,
                            record=record,
                            runtime_fingerprint=fingerprint,
                            protocol=protocol,
                            partitions=partitions,
                        )
                    else:
                        scan = runtime.scan_selection(
                            record,
                            candidate_windows=protocol.candidate_windows,
                        )
                        if not isinstance(scan, WindowSelectionScan):
                            raise ValueError("held-out runtime returned an invalid selection scan")
                        _write_selection_checkpoint(
                            checkpoint,
                            record=record,
                            scan=scan,
                            runtime_fingerprint=fingerprint,
                            protocol=protocol,
                            partitions=partitions,
                        )
                    selection_scans[pair_id] = scan
                    active = None
                del runtime

            selection_rows = [
                _selection_record(
                    record,
                    selection_scans[str(record["pair_id"])],
                    protocol=protocol,
                )
                for record in partitions.selection
            ]
            selection_records_path = output_dir / "window_selection_records.jsonl"
            _write_jsonl_atomic(selection_records_path, selection_rows)
            summaries = _rank_windows(selection_rows, protocol=protocol)
            selection_payload = _selection_payload(
                rows_path=selection_records_path,
                summaries=summaries,
                partitions=partitions,
                protocol=protocol,
                runtime_by_setting=_mapping(
                    runtime_by_setting["selection"], field="selection runtime provenance"
                ),
            )
            selection_path = output_dir / "window_selection.json"
            _write_json_atomic(selection_path, selection_payload)
            selection_commit = {
                "sha256": sha256_file(selection_path),
                "records_sha256": sha256_file(selection_records_path),
                "selected_window": selection_payload["selected_window"],
                "runner_up_window": selection_payload["runner_up_window"],
            }
            base_run.update(
                {
                    "phase": "evaluation",
                    "selection_commit": selection_commit,
                    "runtime_by_setting": runtime_by_setting,
                    "updated_at": _now(),
                }
            )
            _write_json_atomic(output_dir / "run.json", base_run)

        selection_payload, windows = _locked_selection(
            output_dir,
            commit=selection_commit,
            partitions=partitions,
            protocol=protocol,
        )
        if not selection_scans:
            selection_scans.update(
                {
                    str(record["pair_id"]): WindowSelectionScan(
                        available=False,
                        invalid_reason="clean_continuation_lt_1",
                        target_token_id=None,
                        target_token_text=None,
                        untreated_kl=None,
                        patched_kl=(),
                    )
                    for record in partitions.selection
                }
            )
        selection_sha256 = sha256_file(output_dir / "window_selection.json")
        phase = "evaluation"
        for setting in REBUTTAL_SETTINGS:
            setting_records = tuple(
                record for record in partitions.evaluation if _setting_key(record) == setting.key
            )
            active = (phase, setting_records[0])
            revision = _revision(setting_records, setting=setting.slug)
            runtime = factory(
                model=setting.model,
                task=setting.task,
                revision=revision,
                gpu_id=config.gpu_id,
            )
            provenance = _runtime_provenance(
                runtime,
                protocol=protocol,
                revision=revision,
                model=setting.model,
                task=setting.task,
                gpu_id=config.gpu_id,
            )
            fingerprint = _canonical_sha256(provenance)
            evaluation_runtime = runtime_by_setting["evaluation"]
            assert isinstance(evaluation_runtime, dict)
            evaluation_runtime[_setting_id(setting.key)] = provenance
            for record in setting_records:
                active = (phase, record)
                pair_id = str(record["pair_id"])
                checkpoint = _checkpoint_path(output_dir, phase, pair_id)
                if checkpoint.is_file():
                    scan = _load_evaluation_checkpoint(
                        checkpoint,
                        record=record,
                        runtime_fingerprint=fingerprint,
                        protocol=protocol,
                        partitions=partitions,
                        selection_sha256=selection_sha256,
                        windows=windows,
                    )
                else:
                    scan = runtime.scan_evaluation(record, windows=windows)
                    if not isinstance(scan, WindowEvaluationScan):
                        raise ValueError("held-out runtime returned an invalid evaluation scan")
                    validated = WindowEvaluationScan(
                        generations={
                            arm: _validate_generation(scan.generations[arm], record=record, arm=arm)
                            for arm in protocol.evaluation_arms
                        }
                    )
                    _write_evaluation_checkpoint(
                        checkpoint,
                        record=record,
                        scan=validated,
                        runtime_fingerprint=fingerprint,
                        protocol=protocol,
                        partitions=partitions,
                        selection_sha256=selection_sha256,
                        windows=windows,
                    )
                    scan = validated
                evaluation_scans[pair_id] = scan
                active = None
            del runtime

        evaluation_rows = _evaluation_rows(
            partitions.evaluation,
            evaluation_scans,
            windows=windows,
        )
        table_rows, contrast_rows, summary = _analysis(
            partitions.evaluation,
            evaluation_scans,
            windows=windows,
            protocol=protocol,
            confirmatory=partitions.confirmatory,
        )
        summary.update(
            {
                "selection_sha256": selection_sha256,
                "selection_records_sha256": selection_payload["selection_records_sha256"],
                "pair_manifest_sha256": partitions.manifest_sha256,
                "cohort_ids_sha256": partitions.cohort_ids_sha256,
            }
        )
        status_rows = _status_rows(
            partitions,
            selection_scans=selection_scans,
            evaluation_scans=evaluation_scans,
        )
        paths = {name: output_dir / name for name in _PUBLIC_OUTPUTS}
        _write_jsonl_atomic(paths["held_out_window_records.jsonl"], evaluation_rows)
        _write_csv_atomic(paths["held_out_window_table.csv"], table_rows)
        _write_csv_atomic(paths["held_out_window_contrasts.csv"], contrast_rows)
        _write_json_atomic(paths["held_out_window_summary.json"], summary)
        _write_jsonl_atomic(paths["pair_status_records.jsonl"], status_rows)
        counts = {
            "selection_pairs": len(partitions.selection),
            "evaluation_pairs": len(partitions.evaluation),
            "evaluation_records": len(evaluation_rows),
            "table_rows": len(table_rows),
            "contrast_rows": len(contrast_rows),
            "pair_status_records": len(status_rows),
            "settings": len(REBUTTAL_SETTINGS),
        }
        record_counts = {
            "window_selection_records.jsonl": len(partitions.selection),
            "window_selection.json": 1,
            "held_out_window_records.jsonl": len(evaluation_rows),
            "held_out_window_table.csv": len(table_rows),
            "held_out_window_contrasts.csv": len(contrast_rows),
            "held_out_window_summary.json": 1,
            "pair_status_records.jsonl": len(status_rows),
        }
        outputs = {
            name: {"sha256": sha256_file(path), "records": record_counts[name]}
            for name, path in paths.items()
        }
        completed_run = {
            **base_run,
            "status": "completed",
            "phase": "completed",
            "runtime_by_setting": runtime_by_setting,
            "selection_commit": selection_commit,
            "counts": counts,
            "outputs": outputs,
            "failures": [],
            "updated_at": _now(),
        }
        _write_json_atomic(output_dir / "run.json", completed_run)
        return _result_from_run(
            output_dir,
            completed_run,
            partitions=partitions,
            protocol=protocol,
        )
    except Exception as exc:
        for name in _FINAL_OUTPUTS:
            (output_dir / name).unlink(missing_ok=True)
        if selection_commit is None:
            (output_dir / "window_selection_records.jsonl").unlink(missing_ok=True)
            (output_dir / "window_selection.json").unlink(missing_ok=True)
        status_rows = _status_rows(
            partitions,
            selection_scans=selection_scans,
            evaluation_scans=evaluation_scans,
            active=active,
            failure=exc,
        )
        _write_jsonl_atomic(output_dir / "pair_status_records.jsonl", status_rows)
        failure_payload: dict[str, object] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": phase,
        }
        if active is not None:
            failure_payload.update(
                {
                    "pair_id": active[1]["pair_id"],
                    "sample_id": active[1]["sample_id"],
                    "model": active[1]["model"],
                    "task": active[1]["task"],
                }
            )
        failed_run = {
            **base_run,
            "status": "failed",
            "phase": phase,
            "runtime_by_setting": runtime_by_setting,
            "selection_commit": selection_commit,
            "failures": [failure_payload],
            "updated_at": _now(),
        }
        _write_json_atomic(output_dir / "run.json", failed_run)
        raise HeldOutWindowRunError(
            "held-out window evaluation failed; verified checkpoints were retained"
        ) from exc


__all__ = [
    "HeldOutGeneration",
    "HeldOutWindowConfig",
    "HeldOutWindowResult",
    "HeldOutWindowRunError",
    "HeldOutWindowRuntime",
    "RuntimeFactory",
    "WindowEvaluationScan",
    "WindowSelectionScan",
    "run_held_out_window_evaluation",
]
