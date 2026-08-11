"""Build one hash-bound manifest for all six rebuttal settings."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from typo_cot.data.matched_donors import (
    MatchedDonorCandidate,
    plan_cyclic_derangement,
)
from typo_cot.experiments.build_rebuttal_manifest.planning import (
    WindowSplitCandidate,
    plan_strict_offset_control,
    plan_window_split,
)
from typo_cot.experiments.build_rebuttal_manifest.protocol import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
    RebuttalSetting,
)
from typo_cot.experiments.build_rebuttal_manifest.records import (
    REBUTTAL_AUDIT_SCHEMA,
    REBUTTAL_COHORT_SCHEMA,
    REBUTTAL_RUN_SCHEMA,
    iter_jsonl_objects,
    load_json_object,
    normalize_prepared_pair,
    sha256_file,
)
from typo_cot.experiments.catalog import PAPER_SHA256

_PROTOCOL = REBUTTAL_MANIFEST_PROTOCOL
_TARGET_RULES = _PROTOCOL.target_rules
_TARGET_RULE_ORDER = {name: index for index, name in enumerate(_TARGET_RULES)}
_PAIR_SCHEMA = "prepare-edited-pairs/v1"
_PAIR_RUN_SCHEMA = "prepare-edited-pairs-run/v1"
_FIXED_RUN_SCHEMA = "fixed-window-answer-patching-run/v1"
_FIXED_RECORD_SCHEMA = "fixed-window-answer-patching-window/v1"
_FIXED_STATUS_SCHEMA = "fixed-window-answer-patching-pair-status/v1"
_SPLIT_SEED = _PROTOCOL.window_split_seed


@dataclass(frozen=True, slots=True)
class BuildRebuttalManifestConfig:
    """Roots containing completed producers and the new artifact directory."""

    prepared_pairs_root: Path
    fixed_window_root: Path
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "prepared_pairs_root", Path(self.prepared_pairs_root))
        object.__setattr__(self, "fixed_window_root", Path(self.fixed_window_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        resolved = (
            self.prepared_pairs_root.resolve(),
            self.fixed_window_root.resolve(),
            self.output_dir.resolve(),
        )
        for index, left in enumerate(resolved):
            for right in resolved[index + 1 :]:
                if left == right or left in right.parents or right in left.parents:
                    raise ValueError("prepared, fixed-window, and output roots must not overlap")

    def public_arguments(self) -> dict[str, str]:
        return {
            "prepared_pairs_root": str(self.prepared_pairs_root.resolve()),
            "fixed_window_root": str(self.fixed_window_root.resolve()),
            "output_dir": str(self.output_dir.resolve()),
        }


@dataclass(frozen=True, slots=True)
class BuildRebuttalManifestResult:
    """Published artifact paths and audited paper totals."""

    pair_manifest_path: Path
    cohort_ids_path: Path
    source_audit_path: Path
    run_path: Path
    pair_records: int
    restoration_pairs: int
    fixed_window_successes: int
    harm_pairs: int
    fixed_selected_anchors: int
    fixed_excluded_anchors: int
    prepared_wrong_outside_restoration: int


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    setting: RebuttalSetting
    target_rule: str
    pairs_path: Path
    run_path: Path
    relative_pairs_path: str
    pairs_sha256: str
    run_sha256: str
    model_revision: str
    dataset_records_sha256: str
    dataset_sample_count: int
    records: tuple[dict[str, object], ...]
    record_sha256_by_sample_id: Mapping[str, str]

    @property
    def key(self) -> tuple[str, str, str]:
        return self.setting.model, self.setting.task, self.target_rule


@dataclass(frozen=True, slots=True)
class _FixedReference:
    setting: RebuttalSetting
    run_dir: Path
    relative_run_dir: str
    run_sha256: str
    records_sha256: str
    statuses_sha256: str
    summary_sha256: str
    model_revision: str
    tokenizer_revision: str
    selected_pair_ids: frozenset[str]
    exclusion_reason_by_pair_id: Mapping[str, str]
    event_by_pair_id: Mapping[str, bool]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a JSON string array")
    normalized = tuple(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError(f"{field} must contain only non-empty strings")
    return normalized


def _sha256_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"source path escaped its configured root: {path}") from exc


def _setting_map() -> dict[tuple[str, str], RebuttalSetting]:
    return {setting.key: setting for setting in REBUTTAL_SETTINGS}


def _validate_output_metadata(
    *,
    run: Mapping[str, object],
    output_key: str,
    path: Path,
    expected_records: int | None = None,
) -> int:
    outputs = _mapping(run.get("outputs"), field="run outputs")
    metadata = _mapping(outputs.get(output_key), field=f"run outputs.{output_key}")
    recorded_digest = _sha256_digest(
        metadata.get("sha256"),
        field=f"run outputs.{output_key}.sha256",
    )
    if not path.is_file():
        raise ValueError(f"declared output is not a file: {path}")
    if sha256_file(path) != recorded_digest:
        raise ValueError(f"output SHA-256 does not match: {path}")
    records = _positive_int(
        metadata.get("records"),
        field=f"run outputs.{output_key}.records",
    )
    if expected_records is not None and records != expected_records:
        raise ValueError(f"output record count differs for {path}: {records} != {expected_records}")
    return records


def _validate_prepared_run(
    *,
    pairs_path: Path,
    root: Path,
    run: dict[str, object],
    setting: RebuttalSetting,
    target_rule: str,
) -> _PreparedSource:
    run_path = pairs_path.parent / "run.json"
    if run.get("schema_version") != _PAIR_RUN_SCHEMA:
        raise ValueError(f"unknown prepared run schema: {run_path}")
    if run.get("paper_sha256") != PAPER_SHA256:
        raise ValueError(f"prepared run paper SHA-256 differs: {run_path}")
    if run.get("operation") != "prepare-edited-pairs" or run.get("status") != "completed":
        raise ValueError(f"prepared run is not a completed producer: {run_path}")
    arguments = _mapping(run.get("arguments"), field=f"{run_path} arguments")
    expected_arguments = {
        "model": setting.model,
        "benchmark": setting.task,
        "targeting": target_rule,
        "num_edits": _PROTOCOL.requested_edits,
        "seed": _PROTOCOL.source_seed,
        "max_new_tokens": _PROTOCOL.source_max_new_tokens,
        "limit": None,
    }
    for field, expected in expected_arguments.items():
        if field not in arguments or arguments.get(field) != expected:
            raise ValueError(f"prepared run {field} must be {expected!r}: {run_path}")
    decoding = _mapping(run.get("decoding"), field=f"{run_path} decoding")
    for field, expected in (
        ("strategy", "greedy"),
        ("dtype", "bfloat16"),
        ("padding_side", "left"),
        ("max_new_tokens", _PROTOCOL.source_max_new_tokens),
        ("do_sample", False),
        ("num_beams", 1),
        ("num_return_sequences", 1),
    ):
        if decoding.get(field) != expected:
            raise ValueError(f"prepared decoding {field} must be {expected!r}: {run_path}")
    failures = run.get("failures")
    if failures not in ([], None):
        raise ValueError(f"prepared run contains failures: {run_path}")
    counts = _mapping(run.get("counts"), field=f"{run_path} counts")
    written = _positive_int(counts.get("written"), field=f"{run_path} counts.written")
    failed = _nonnegative_int(counts.get("failed"), field=f"{run_path} counts.failed")
    discovered = _positive_int(
        counts.get("discovered"),
        field=f"{run_path} counts.discovered",
    )
    if failed != 0 or discovered != written:
        raise ValueError(f"prepared run is incomplete: {run_path}")
    if set(_mapping(run.get("outputs"), field=f"{run_path} outputs")) != {"pairs"}:
        raise ValueError(f"prepared run output inventory differs: {run_path}")
    output = _mapping(
        _mapping(run.get("outputs"), field=f"{run_path} outputs").get("pairs"),
        field=f"{run_path} outputs.pairs",
    )
    if output.get("path") != pairs_path.name:
        raise ValueError(f"prepared pairs output path differs: {run_path}")
    _validate_output_metadata(
        run=run,
        output_key="pairs",
        path=pairs_path,
        expected_records=written,
    )
    provenance = _mapping(run.get("provenance"), field=f"{run_path} provenance")
    model_revision = _nonempty_string(
        provenance.get("model_revision"),
        field=f"{run_path} provenance.model_revision",
    )
    dataset_records_sha256 = _sha256_digest(
        provenance.get("dataset_records_sha256"),
        field=f"{run_path} provenance.dataset_records_sha256",
    )
    dataset_sample_count = _positive_int(
        provenance.get("dataset_sample_count"),
        field=f"{run_path} provenance.dataset_sample_count",
    )
    if dataset_sample_count != written:
        raise ValueError(f"prepared dataset and output record counts differ: {run_path}")
    if provenance.get("generation_protocol") != "explicit-greedy-generation/v1":
        raise ValueError(f"prepared generation protocol differs: {run_path}")
    if provenance.get("generation_termination_protocol") != "effective-eos-vs-length-cap/v1":
        raise ValueError(f"prepared generation termination protocol differs: {run_path}")

    pairs_digest = sha256_file(pairs_path)
    run_digest = sha256_file(run_path)
    normalized: list[dict[str, object]] = []
    fingerprints: dict[str, str] = {}
    previous_sample_id: str | None = None
    for line_number, line, record in iter_jsonl_objects(pairs_path):
        context = f"{pairs_path}:{line_number}"
        for field, expected in (
            ("model", setting.model),
            ("benchmark", setting.task),
            ("targeting", target_rule),
        ):
            if record.get(field) != expected:
                raise ValueError(f"{context}: {field} differs from the prepared run")
        sample_id = _nonempty_string(record.get("sample_id"), field=f"{context}.sample_id")
        if previous_sample_id is not None and sample_id <= previous_sample_id:
            raise ValueError(f"prepared sample IDs are not strictly sorted: {pairs_path}")
        previous_sample_id = sample_id
        record_digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        fingerprints[sample_id] = record_digest
        normalized.append(
            normalize_prepared_pair(
                record,
                source_record_sha256=record_digest,
                prepared_pairs_path=_relative(pairs_path, root),
                prepared_pairs_sha256=pairs_digest,
                prepared_run_sha256=run_digest,
                model_revision=model_revision,
            )
        )
    if len(normalized) != written:
        raise ValueError(f"prepared run written count differs from JSONL: {pairs_path}")
    return _PreparedSource(
        setting=setting,
        target_rule=target_rule,
        pairs_path=pairs_path,
        run_path=run_path,
        relative_pairs_path=_relative(pairs_path, root),
        pairs_sha256=pairs_digest,
        run_sha256=run_digest,
        model_revision=model_revision,
        dataset_records_sha256=dataset_records_sha256,
        dataset_sample_count=dataset_sample_count,
        records=tuple(normalized),
        record_sha256_by_sample_id=fingerprints,
    )


def _discover_prepared_sources(
    root: Path,
) -> tuple[dict[tuple[str, str, str], _PreparedSource], list[str]]:
    if not root.is_dir():
        raise ValueError(f"prepared pairs root is not a directory: {root}")
    settings = _setting_map()
    sources: dict[tuple[str, str, str], _PreparedSource] = {}
    ignored: list[str] = []
    for pairs_path in sorted(root.rglob("pairs.jsonl")):
        run_path = pairs_path.parent / "run.json"
        if not run_path.is_file():
            raise ValueError(f"pairs.jsonl is missing sibling run.json: {pairs_path}")
        run = load_json_object(run_path)
        if run.get("schema_version") != _PAIR_RUN_SCHEMA:
            ignored.append(_relative(pairs_path, root))
            continue
        arguments = _mapping(run.get("arguments"), field=f"{run_path} arguments")
        model = arguments.get("model")
        task = arguments.get("benchmark")
        target_rule = arguments.get("targeting")
        setting = settings.get((str(model), str(task)))
        if setting is None:
            ignored.append(_relative(pairs_path, root))
            continue
        if target_rule not in _TARGET_RULE_ORDER:
            raise ValueError(f"rebuttal prepared source has unknown target rule: {run_path}")
        source = _validate_prepared_run(
            pairs_path=pairs_path,
            root=root,
            run=run,
            setting=setting,
            target_rule=str(target_rule),
        )
        if source.key in sources:
            raise ValueError(f"duplicate prepared source for {source.key!r}")
        sources[source.key] = source

    required = {
        (setting.model, setting.task, target_rule)
        for setting in REBUTTAL_SETTINGS
        for target_rule in _TARGET_RULES
    }
    missing = sorted(required - set(sources))
    if missing:
        raise ValueError(f"missing rebuttal prepared source(s): {missing!r}")
    for setting in REBUTTAL_SETTINGS:
        paired = [sources[(setting.model, setting.task, rule)] for rule in _TARGET_RULES]
        if len({source.model_revision for source in paired}) != 1:
            raise ValueError(f"prepared target-rule model revisions differ: {setting.slug}")
        if len({source.dataset_records_sha256 for source in paired}) != 1:
            raise ValueError(f"prepared target-rule dataset SHA-256 differs: {setting.slug}")
        if len({source.dataset_sample_count for source in paired}) != 1:
            raise ValueError(f"prepared target-rule dataset counts differ: {setting.slug}")
    return sources, ignored


def _fixed_output_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / "fixed_window_records.jsonl",
        run_dir / "pair_status_records.jsonl",
        run_dir / "setting_summary.json",
    )


def _validate_fixed_reference(
    *,
    run_dir: Path,
    root: Path,
    run: dict[str, object],
    setting: RebuttalSetting,
    sources: Mapping[tuple[str, str, str], _PreparedSource],
) -> _FixedReference:
    run_path = run_dir / "run.json"
    if run.get("schema_version") != _FIXED_RUN_SCHEMA:
        raise ValueError(f"unknown fixed-window run schema: {run_path}")
    if run.get("paper_sha256") != PAPER_SHA256:
        raise ValueError(f"fixed-window run paper SHA-256 differs: {run_path}")
    if run.get("operation") != "fixed-window-answer-patching" or run.get("status") != "completed":
        raise ValueError(f"fixed-window run is not a completed producer: {run_path}")
    producer_protocol = _mapping(
        run.get("protocol"),
        field=f"{run_path} protocol",
    )
    if producer_protocol.get("schema_version") != "fixed-window-answer-patching-protocol/v1":
        raise ValueError(f"fixed-window producer protocol differs: {run_path}")
    arguments = _mapping(run.get("arguments"), field=f"{run_path} arguments")
    for field, expected in (("model", setting.model), ("benchmark", setting.task)):
        if arguments.get(field) != expected:
            raise ValueError(f"fixed-window run {field} differs: {run_path}")
    if arguments.get("limit") is not None:
        raise ValueError(f"fixed-window reference must be unlimited: {run_path}")
    layers = _string_sequence(
        arguments.get("layers"),
        field=f"{run_path} arguments.layers",
    )
    directions = _string_sequence(
        arguments.get("directions"),
        field=f"{run_path} arguments.directions",
    )
    if _PROTOCOL.fixed_window not in layers:
        raise ValueError(
            f"fixed-window reference does not contain {_PROTOCOL.fixed_window}: {run_path}"
        )
    if _PROTOCOL.fixed_direction not in directions:
        raise ValueError(f"fixed-window reference lacks {_PROTOCOL.fixed_direction}: {run_path}")
    if run.get("failures") not in ([], None):
        raise ValueError(f"fixed-window reference contains failures: {run_path}")

    records_path, statuses_path, summary_path = _fixed_output_paths(run_dir)
    outputs = _mapping(run.get("outputs"), field=f"{run_path} outputs")
    if set(outputs) != {path.name for path in (records_path, statuses_path, summary_path)}:
        raise ValueError(f"fixed-window output inventory differs: {run_path}")
    records_count = _validate_output_metadata(
        run=run,
        output_key=records_path.name,
        path=records_path,
    )
    statuses_count = _validate_output_metadata(
        run=run,
        output_key=statuses_path.name,
        path=statuses_path,
    )
    _validate_output_metadata(
        run=run,
        output_key=summary_path.name,
        path=summary_path,
        expected_records=1,
    )
    summary = load_json_object(summary_path)
    if (
        summary.get("schema_version") != "fixed-window-answer-patching-setting-summary/v1"
        or summary.get("paper_sha256") != PAPER_SHA256
        or summary.get("operation") != "fixed-window-answer-patching"
    ):
        raise ValueError(f"fixed-window setting summary contract differs: {summary_path}")
    summary_setting = _mapping(
        summary.get("setting"),
        field=f"{summary_path} setting",
    )
    if (
        summary_setting.get("model") != setting.model
        or summary_setting.get("benchmark") != setting.task
        or _PROTOCOL.fixed_window
        not in _string_sequence(
            summary_setting.get("layer_windows"),
            field=f"{summary_path} setting.layer_windows",
        )
    ):
        raise ValueError(f"fixed-window setting summary differs: {summary_path}")
    if statuses_count < setting.paper_denominator:
        raise ValueError(
            f"fixed-window selected-anchor count is smaller than the paper denominator "
            f"for {setting.slug}: {statuses_count} < {setting.paper_denominator}"
        )
    run_counts = _mapping(run.get("counts"), field=f"{run_path} counts")
    selected_count = _positive_int(
        run_counts.get("selected_anchors"),
        field=f"{run_path} counts.selected_anchors",
    )
    failed_count = _nonnegative_int(
        run_counts.get("failed_pairs"),
        field=f"{run_path} counts.failed_pairs",
    )
    window_count = _positive_int(
        run_counts.get("window_records"),
        field=f"{run_path} counts.window_records",
    )
    if selected_count != statuses_count or failed_count != 0 or window_count != records_count:
        raise ValueError(f"fixed-window run counts differ from outputs: {run_path}")
    population = _mapping(summary.get("population"), field=f"{summary_path} population")
    if population.get("selected_anchors") != statuses_count:
        raise ValueError(f"fixed-window selected-anchor summary differs: {summary_path}")
    direction_denominators = _mapping(
        population.get("direction_denominators"),
        field=f"{summary_path} population.direction_denominators",
    )
    if direction_denominators.get(_PROTOCOL.fixed_direction) != setting.paper_denominator:
        raise ValueError(f"fixed-window direction denominator differs: {summary_path}")
    direction_summaries = _mapping(
        summary.get("directions"),
        field=f"{summary_path} directions",
    )
    fixed_direction_summary = _mapping(
        direction_summaries.get(_PROTOCOL.fixed_direction),
        field=f"{summary_path} directions.{_PROTOCOL.fixed_direction}",
    )
    fixed_windows = _mapping(
        fixed_direction_summary.get("windows"),
        field=f"{summary_path} directions.{_PROTOCOL.fixed_direction}.windows",
    )
    fixed_window_summary = _mapping(
        fixed_windows.get(_PROTOCOL.fixed_window),
        field=(
            f"{summary_path} directions.{_PROTOCOL.fixed_direction}."
            f"windows.{_PROTOCOL.fixed_window}"
        ),
    )
    if (
        fixed_direction_summary.get("included_pairs") != setting.paper_denominator
        or fixed_window_summary.get("total") != setting.paper_denominator
        or fixed_window_summary.get("successes") != setting.paper_successes
    ):
        raise ValueError(f"fixed-window summary outcome counts differ: {summary_path}")

    input_payload = _mapping(run.get("input"), field=f"{run_path} input")
    targeting_sources = _mapping(
        input_payload.get("targeting_sources"),
        field=f"{run_path} input.targeting_sources",
    )
    if set(targeting_sources) != set(_TARGET_RULES):
        raise ValueError(f"fixed-window reference must bind both target rules: {run_path}")
    for target_rule in _TARGET_RULES:
        expected_source = sources[(setting.model, setting.task, target_rule)]
        metadata = _mapping(
            targeting_sources.get(target_rule),
            field=f"{run_path} targeting source {target_rule}",
        )
        expected_metadata = {
            "pairs_sha256": expected_source.pairs_sha256,
            "source_run_sha256": expected_source.run_sha256,
            "source_record_count": len(expected_source.records),
            "source_schema": _PAIR_SCHEMA,
        }
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                raise ValueError(f"fixed-window source {target_rule}.{field} differs: {run_path}")
    revision = sources[(setting.model, setting.task, _TARGET_RULES[0])].model_revision
    reference_source = sources[(setting.model, setting.task, _TARGET_RULES[0])]
    for field, expected in (
        ("source_model_revision", revision),
        ("dataset_records_sha256", reference_source.dataset_records_sha256),
        ("dataset_sample_count", reference_source.dataset_sample_count),
    ):
        if input_payload.get(field) != expected:
            raise ValueError(f"fixed-window input {field} differs: {run_path}")
    runtime = _mapping(run.get("runtime"), field=f"{run_path} runtime")
    for field in ("requested_revision", "model_revision", "tokenizer_revision"):
        if runtime.get(field) != revision:
            raise ValueError(f"fixed-window runtime {field} differs: {run_path}")

    source_by_key: dict[tuple[str, str], dict[str, object]] = {}
    fingerprint_by_key: dict[tuple[str, str], str] = {}
    for target_rule in _TARGET_RULES:
        source = sources[(setting.model, setting.task, target_rule)]
        for record in source.records:
            key = target_rule, str(record["sample_id"])
            source_by_key[key] = record
            fingerprint_by_key[key] = str(record["source"]["source_record_sha256"])

    event_by_pair_id: dict[str, bool] = {}
    for line_number, _line, record in iter_jsonl_objects(records_path):
        context = f"{records_path}:{line_number}"
        if record.get("schema_version") != _FIXED_RECORD_SCHEMA:
            raise ValueError(f"{context}: unknown fixed-window record schema")
        if record.get("paper_sha256") != PAPER_SHA256:
            raise ValueError(f"{context}: paper SHA-256 differs")
        if record.get("model") != setting.model or record.get("benchmark") != setting.task:
            raise ValueError(f"{context}: fixed-window setting differs")
        if (
            record.get("direction") != _PROTOCOL.fixed_direction
            or record.get("window") != _PROTOCOL.fixed_window
        ):
            continue
        target_rule = str(record.get("targeting"))
        sample_id = str(record.get("sample_id"))
        key = target_rule, sample_id
        source_record = source_by_key.get(key)
        if source_record is None:
            raise ValueError(f"{context}: fixed-window pair is absent from prepared sources")
        if record.get("source_record_sha256") != fingerprint_by_key[key]:
            raise ValueError(f"{context}: source record SHA-256 differs")
        event = record.get("event")
        if not isinstance(event, bool):
            raise ValueError(f"{context}.event must be boolean")
        pair_id = str(source_record["pair_id"])
        if pair_id in event_by_pair_id:
            raise ValueError(f"{context}: duplicate fixed-window pair")
        if (
            source_record["clean_correct"] is not True
            or source_record["typo_correct"] is not False
            or int(source_record["number_of_aligned_words"]) <= 0
        ):
            raise ValueError(f"{context}: fixed-window pair is outside restoration cohort")
        event_by_pair_id[pair_id] = event
    if len(event_by_pair_id) != setting.paper_denominator:
        raise ValueError(
            f"fixed-window denominator differs from paper for {setting.slug}: "
            f"{len(event_by_pair_id)} != {setting.paper_denominator}"
        )
    if sum(event_by_pair_id.values()) != setting.paper_successes:
        raise ValueError(
            f"fixed-window successes differ from paper for {setting.slug}: "
            f"{sum(event_by_pair_id.values())} != {setting.paper_successes}"
        )
    if records_count < len(event_by_pair_id):
        raise ValueError(f"fixed-window output record count is impossible: {run_path}")

    selected_pair_ids: set[str] = set()
    included_status_pair_ids: set[str] = set()
    exclusion_reason_by_pair_id: dict[str, str] = {}
    for line_number, _line, status in iter_jsonl_objects(statuses_path):
        context = f"{statuses_path}:{line_number}"
        if status.get("schema_version") != _FIXED_STATUS_SCHEMA:
            raise ValueError(f"{context}: unknown fixed-window status schema")
        if status.get("paper_sha256") != PAPER_SHA256:
            raise ValueError(f"{context}: paper SHA-256 differs")
        if status.get("model") != setting.model or status.get("benchmark") != setting.task:
            raise ValueError(f"{context}: fixed-window status setting differs")
        key = str(status.get("targeting")), str(status.get("sample_id"))
        source_record = source_by_key.get(key)
        if source_record is None:
            raise ValueError(f"{context}: status pair is absent from prepared sources")
        if status.get("source_record_sha256") != fingerprint_by_key[key]:
            raise ValueError(f"{context}: status source record SHA-256 differs")
        if (
            source_record["clean_correct"] is not True
            or source_record["typo_correct"] is not False
            or int(source_record["number_of_aligned_words"]) <= 0
        ):
            raise ValueError(f"{context}: selected anchor is outside the prepared flip pool")
        direction_status = _mapping(
            status.get("direction_status"),
            field=f"{context}.direction_status",
        )
        clean_to_edited = _mapping(
            direction_status.get(_PROTOCOL.fixed_direction),
            field=f"{context}.direction_status.{_PROTOCOL.fixed_direction}",
        )
        included = clean_to_edited.get("included")
        if not isinstance(included, bool):
            raise ValueError(f"{context}: direction included flag must be boolean")
        pair_id = str(source_record["pair_id"])
        if pair_id in selected_pair_ids:
            raise ValueError(f"{context}: duplicate fixed-window status pair")
        selected_pair_ids.add(pair_id)
        if included:
            if clean_to_edited.get("exclusion_reason") is not None:
                raise ValueError(f"{context}: included status has an exclusion reason")
            included_status_pair_ids.add(pair_id)
        else:
            exclusion_reason_by_pair_id[pair_id] = _nonempty_string(
                clean_to_edited.get("exclusion_reason"),
                field=f"{context}.direction_status.{_PROTOCOL.fixed_direction}.exclusion_reason",
            )
    if len(selected_pair_ids) != statuses_count:
        raise ValueError(f"fixed-window status output count differs: {run_path}")
    if included_status_pair_ids != set(event_by_pair_id):
        raise ValueError(f"fixed-window included statuses and event denominator differ: {run_path}")

    return _FixedReference(
        setting=setting,
        run_dir=run_dir,
        relative_run_dir=_relative(run_dir, root),
        run_sha256=sha256_file(run_path),
        records_sha256=sha256_file(records_path),
        statuses_sha256=sha256_file(statuses_path),
        summary_sha256=sha256_file(summary_path),
        model_revision=revision,
        tokenizer_revision=str(runtime["tokenizer_revision"]),
        selected_pair_ids=frozenset(selected_pair_ids),
        exclusion_reason_by_pair_id=exclusion_reason_by_pair_id,
        event_by_pair_id=event_by_pair_id,
    )


def _discover_fixed_references(
    root: Path,
    *,
    sources: Mapping[tuple[str, str, str], _PreparedSource],
) -> tuple[dict[tuple[str, str], _FixedReference], list[str]]:
    if not root.is_dir():
        raise ValueError(f"fixed-window root is not a directory: {root}")
    settings = _setting_map()
    references: dict[tuple[str, str], _FixedReference] = {}
    ignored: list[str] = []
    for run_path in sorted(root.rglob("run.json")):
        run = load_json_object(run_path)
        if run.get("schema_version") != _FIXED_RUN_SCHEMA:
            if run.get("operation") == "fixed-window-answer-patching":
                raise ValueError(f"unknown fixed-window run schema: {run_path}")
            ignored.append(_relative(run_path.parent, root))
            continue
        arguments = _mapping(run.get("arguments"), field=f"{run_path} arguments")
        setting = settings.get((str(arguments.get("model")), str(arguments.get("benchmark"))))
        if setting is None:
            ignored.append(_relative(run_path.parent, root))
            continue
        layers = _string_sequence(
            arguments.get("layers"),
            field=f"{run_path} arguments.layers",
        )
        directions = _string_sequence(
            arguments.get("directions"),
            field=f"{run_path} arguments.directions",
        )
        if _PROTOCOL.fixed_window not in layers or _PROTOCOL.fixed_direction not in directions:
            ignored.append(_relative(run_path.parent, root))
            continue
        if setting.key in references:
            raise ValueError(f"duplicate fixed-window reference for {setting.slug}")
        references[setting.key] = _validate_fixed_reference(
            run_dir=run_path.parent,
            root=root,
            run=run,
            setting=setting,
            sources=sources,
        )
    missing = [setting.slug for setting in REBUTTAL_SETTINGS if setting.key not in references]
    if missing:
        raise ValueError(f"missing fixed-window reference(s): {missing!r}")
    return references, ignored


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
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


def _augment_records(
    *,
    sources: Mapping[tuple[str, str, str], _PreparedSource],
    references: Mapping[tuple[str, str], _FixedReference],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    records: list[dict[str, object]] = []
    source_by_pair_id: dict[str, dict[str, object]] = {}
    for source in sources.values():
        for source_record in source.records:
            pair_id = str(source_record["pair_id"])
            if pair_id in source_by_pair_id:
                raise ValueError(f"duplicate normalized pair identity: {pair_id}")
            source_by_pair_id[pair_id] = dict(source_record)

    restoration_ids = {
        pair_id for reference in references.values() for pair_id in reference.event_by_pair_id
    }
    fixed_selected_ids = {
        pair_id for reference in references.values() for pair_id in reference.selected_pair_ids
    }
    full_clean_correct_ids = {
        pair_id for pair_id, record in source_by_pair_id.items() if record["clean_correct"] is True
    }
    patch_eligible_clean_correct_ids = {
        pair_id
        for pair_id in full_clean_correct_ids
        if int(source_by_pair_id[pair_id]["number_of_aligned_words"]) > 0
    }
    alignment_ineligible_clean_correct_ids = (
        full_clean_correct_ids - patch_eligible_clean_correct_ids
    )
    harm_ids = {
        pair_id
        for pair_id in patch_eligible_clean_correct_ids
        if source_by_pair_id[pair_id]["typo_correct"] is True
    }
    wrong_ids = {
        pair_id
        for pair_id in patch_eligible_clean_correct_ids
        if source_by_pair_id[pair_id]["typo_correct"] is False
    }
    if not restoration_ids <= wrong_ids or not fixed_selected_ids <= wrong_ids:
        invalid_restoration = sorted(restoration_ids - wrong_ids)
        invalid_selected = sorted(fixed_selected_ids - wrong_ids)
        raise ValueError(
            "fixed-window identities escaped the prepared clean-correct/typo-wrong pool: "
            f"denominator={invalid_restoration[:3]!r}, selected={invalid_selected[:3]!r}"
        )
    fixed_excluded_ids = fixed_selected_ids - restoration_ids
    prepared_wrong_outside_restoration_ids = wrong_ids - restoration_ids
    repair_harm_composite_ids = restoration_ids | harm_ids

    donor_candidates = []
    for pair_id in sorted(restoration_ids):
        record = source_by_pair_id[pair_id]
        donor_candidates.append(
            MatchedDonorCandidate(
                stratum=(
                    str(record["task"]),
                    str(record["model"]),
                    str(record["target_rule"]),
                    str(record["number_of_aligned_words"]),
                ),
                identity=(pair_id,),
            )
        )
    donor_plan = plan_cyclic_derangement(donor_candidates, reject_singletons=False)
    donor_by_pair_id = {recipient[0]: donor[0] for recipient, donor in donor_plan.assignments}
    singleton_pair_ids = {identity[0] for identity in donor_plan.singleton_identities}

    split = plan_window_split(
        (
            WindowSplitCandidate(
                pair_id,
                (
                    str(source_by_pair_id[pair_id]["model"]),
                    str(source_by_pair_id[pair_id]["task"]),
                    str(source_by_pair_id[pair_id]["target_rule"]),
                ),
            )
            for pair_id in restoration_ids
        ),
        seed=_SPLIT_SEED,
    )
    selection_ids = set(split.selection_pair_ids)
    evaluation_ids = set(split.evaluation_pair_ids)

    setting_audit: dict[tuple[str, str], Counter[str]] = {
        setting.key: Counter() for setting in REBUTTAL_SETTINGS
    }
    for pair_id, raw_record in source_by_pair_id.items():
        record = dict(raw_record)
        planning = _mapping(record.pop("_planning"), field=f"{pair_id} planning")
        record["manifest_protocol_sha256"] = _PROTOCOL.sha256()
        offset = plan_strict_offset_control(
            planning.get("clean_final_tokens", []),  # type: ignore[arg-type]
            planning.get("typo_final_tokens", []),  # type: ignore[arg-type]
            planning.get("clean_edited_tokens", []),  # type: ignore[arg-type]
            planning.get("typo_edited_tokens", []),  # type: ignore[arg-type]
            int(record["clean_prompt_token_count"]),
            int(record["typo_prompt_token_count"]),
            offset=_PROTOCOL.offset_tokens,
        )
        restoration = pair_id in restoration_ids
        harm = pair_id in harm_ids
        donor_id = donor_by_pair_id.get(pair_id)
        cross_valid = donor_id is not None
        record["cohorts"] = {
            "full_clean_correct": pair_id in full_clean_correct_ids,
            "patch_eligible_clean_correct": pair_id in patch_eligible_clean_correct_ids,
            "alignment_ineligible_clean_correct": (
                pair_id in alignment_ineligible_clean_correct_ids
            ),
            "restoration": restoration,
            "harm": harm,
            "fixed_selected_anchor": pair_id in fixed_selected_ids,
            "fixed_excluded_anchor": pair_id in fixed_excluded_ids,
            "prepared_typo_wrong": pair_id in wrong_ids,
            "prepared_typo_wrong_outside_restoration": (
                pair_id in prepared_wrong_outside_restoration_ids
            ),
            "repair_harm_composite": pair_id in repair_harm_composite_ids,
            "window_selection": pair_id in selection_ids,
            "window_evaluation": pair_id in evaluation_ids,
        }
        record["controls"] = {
            "correct": {
                "valid": bool(record["number_of_aligned_words"]),
                "source_positions": list(planning.get("clean_final_tokens", [])),
                "destination_positions": list(planning.get("typo_final_tokens", [])),
            },
            "offset_2": {
                "valid": offset.valid,
                "source_positions": list(offset.source_positions),
                "destination_positions": list(offset.destination_positions),
                "input_pairs": offset.input_pairs,
                "offset_tokens": offset.offset_tokens,
                "invalid_reason": offset.invalid_reason,
                "validity_rule": "all-prompt-interior-non-edited-coordinates/v1",
            },
            "cross_item": {
                "valid": cross_valid,
                "donor_pair_id": donor_id,
                "invalid_reason": (
                    "singleton-matching-stratum" if pair_id in singleton_pair_ids else None
                ),
                "matching_rule": (
                    "task-model-target-rule-aligned-word-count-cyclic-derangement/v1"
                ),
            },
            "common_valid": restoration and offset.valid and cross_valid,
        }
        reference = references[(str(record["model"]), str(record["task"]))]
        record["fixed_window"] = {
            "in_reference_denominator": restoration,
            "in_selected_anchors": pair_id in reference.selected_pair_ids,
            "exclusion_reason": reference.exclusion_reason_by_pair_id.get(pair_id),
            "direction": _PROTOCOL.fixed_direction if restoration else None,
            "window": _PROTOCOL.fixed_window if restoration else None,
            "event": reference.event_by_pair_id.get(pair_id),
            "run_path": reference.relative_run_dir,
            "run_sha256": reference.run_sha256,
        }
        records.append(record)

        counter = setting_audit[(str(record["model"]), str(record["task"]))]
        counter["prepared_pairs"] += 1
        counter["full_clean_correct"] += int(pair_id in full_clean_correct_ids)
        counter["patch_eligible_clean_correct"] += int(pair_id in patch_eligible_clean_correct_ids)
        counter["alignment_ineligible_clean_correct"] += int(
            pair_id in alignment_ineligible_clean_correct_ids
        )
        if restoration:
            counter["restoration_pairs"] += 1
            counter["fixed_window_successes"] += int(bool(reference.event_by_pair_id[pair_id]))
            counter["offset_valid"] += int(offset.valid)
            counter["cross_item_valid"] += int(cross_valid)
            counter["common_valid"] += int(offset.valid and cross_valid)
        if pair_id in fixed_selected_ids:
            counter["fixed_selected_anchors"] += 1
        if pair_id in fixed_excluded_ids:
            counter["fixed_excluded_anchors"] += 1
        if pair_id in wrong_ids:
            counter["prepared_typo_wrong"] += 1
        if pair_id in prepared_wrong_outside_restoration_ids:
            counter["prepared_typo_wrong_outside_restoration"] += 1
        if harm:
            counter["harm_pairs"] += 1

    records.sort(
        key=lambda record: (
            str(record["task"]),
            str(record["model"]),
            _TARGET_RULE_ORDER[str(record["target_rule"])],
            str(record["sample_id"]),
        )
    )
    cohort_ids = {
        "schema_version": REBUTTAL_COHORT_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "manifest_protocol_sha256": _PROTOCOL.sha256(),
        "identity": "sha256-canonical-model-task-target-rule-sample-id/v1",
        "window_split": {
            "algorithm": "sha256-order-first-floor-half-per-model-task-target-rule/v1",
            "seed": _PROTOCOL.window_split_seed,
            "outcome_independent": True,
        },
        "cohorts": {
            "restoration": sorted(restoration_ids),
            "harm": sorted(harm_ids),
            "full_clean_correct": sorted(full_clean_correct_ids),
            "patch_eligible_clean_correct": sorted(patch_eligible_clean_correct_ids),
            "alignment_ineligible_clean_correct": sorted(alignment_ineligible_clean_correct_ids),
            "fixed_selected_anchor": sorted(fixed_selected_ids),
            "fixed_excluded_anchor": sorted(fixed_excluded_ids),
            "prepared_typo_wrong": sorted(wrong_ids),
            "prepared_typo_wrong_outside_restoration": sorted(
                prepared_wrong_outside_restoration_ids
            ),
            "repair_harm_composite": sorted(repair_harm_composite_ids),
            "window_selection": list(split.selection_pair_ids),
            "window_evaluation": list(split.evaluation_pair_ids),
            "offset_valid": sorted(
                str(record["pair_id"])
                for record in records
                if record["cohorts"]["restoration"] and record["controls"]["offset_2"]["valid"]
            ),
            "cross_item_valid": sorted(donor_by_pair_id),
            "common_valid": sorted(
                str(record["pair_id"]) for record in records if record["controls"]["common_valid"]
            ),
        },
        "cross_item_donors": [
            {"recipient_pair_id": recipient, "donor_pair_id": donor}
            for recipient, donor in sorted(donor_by_pair_id.items())
        ],
        "invalid_cross_item_recipients": sorted(singleton_pair_ids),
    }
    settings_payload = []
    for setting in REBUTTAL_SETTINGS:
        counts = setting_audit[setting.key]
        settings_payload.append(
            {
                "model": setting.model,
                "task": setting.task,
                "paper_denominator": setting.paper_denominator,
                "paper_successes": setting.paper_successes,
                **{
                    field: counts[field]
                    for field in (
                        "prepared_pairs",
                        "full_clean_correct",
                        "patch_eligible_clean_correct",
                        "alignment_ineligible_clean_correct",
                        "restoration_pairs",
                        "fixed_window_successes",
                        "harm_pairs",
                        "fixed_selected_anchors",
                        "fixed_excluded_anchors",
                        "prepared_typo_wrong",
                        "prepared_typo_wrong_outside_restoration",
                        "offset_valid",
                        "cross_item_valid",
                        "common_valid",
                    )
                },
            }
        )
    audit_counts = {
        "schema_version": REBUTTAL_AUDIT_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "settings": settings_payload,
        "paper_totals": {
            "restoration_pairs": sum(
                reference.setting.paper_denominator for reference in references.values()
            ),
            "fixed_window_successes": sum(
                reference.setting.paper_successes for reference in references.values()
            ),
        },
    }
    return records, cohort_ids, audit_counts


def run_build_rebuttal_manifest(
    config: BuildRebuttalManifestConfig,
) -> BuildRebuttalManifestResult:
    """Validate completed producers and freeze every downstream pair identity."""

    if not isinstance(config, BuildRebuttalManifestConfig):
        raise TypeError("config must be BuildRebuttalManifestConfig")
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    sources, ignored_prepared = _discover_prepared_sources(config.prepared_pairs_root)
    references, ignored_fixed = _discover_fixed_references(
        config.fixed_window_root,
        sources=sources,
    )
    records, cohort_ids, audit_counts = _augment_records(
        sources=sources,
        references=references,
    )
    restoration_pairs = sum(int(bool(record["cohorts"]["restoration"])) for record in records)
    fixed_window_successes = sum(int(record["fixed_window"]["event"] is True) for record in records)
    harm_pairs = sum(int(bool(record["cohorts"]["harm"])) for record in records)
    fixed_selected_anchors = sum(
        int(bool(record["cohorts"]["fixed_selected_anchor"])) for record in records
    )
    fixed_excluded_anchors = sum(
        int(bool(record["cohorts"]["fixed_excluded_anchor"])) for record in records
    )
    prepared_wrong_outside_restoration = sum(
        int(bool(record["cohorts"]["prepared_typo_wrong_outside_restoration"]))
        for record in records
    )
    if (
        restoration_pairs != _PROTOCOL.restoration_pairs
        or fixed_window_successes != _PROTOCOL.fixed_window_successes
    ):
        raise ValueError(
            "rebuttal paper totals differ after normalization: "
            f"restoration={restoration_pairs}, successes={fixed_window_successes}"
        )

    source_audit = {
        **audit_counts,
        "protocol": {
            "manifest_contract": _PROTOCOL.as_dict(),
            "manifest_contract_sha256": _PROTOCOL.sha256(),
            "prepared_schema": _PAIR_RUN_SCHEMA,
            "fixed_window_schema": _FIXED_RUN_SCHEMA,
            "fixed_window_direction": _PROTOCOL.fixed_direction,
            "fixed_window": _PROTOCOL.fixed_window,
            "target_rules": list(_TARGET_RULES),
            "offset_validity": "all-prompt-interior-non-edited-coordinates/v1",
            "cross_item_donor": ("task-model-target-rule-aligned-word-count-cyclic-derangement/v1"),
            "harm_selection": ("all-prepared-clean-correct-typo-correct-aligned-uncapped/v1"),
            "net_balance_scope": ("repair-harm-conditional-composite-not-full-population/v1"),
        },
        "sources": {
            "prepared": [
                {
                    "model": source.setting.model,
                    "task": source.setting.task,
                    "target_rule": source.target_rule,
                    "pairs_path": source.relative_pairs_path,
                    "pairs_sha256": source.pairs_sha256,
                    "run_sha256": source.run_sha256,
                    "records": len(source.records),
                    "model_revision": source.model_revision,
                    "dataset_records_sha256": source.dataset_records_sha256,
                }
                for source in sorted(
                    sources.values(),
                    key=lambda source: (
                        source.setting.task,
                        source.setting.model,
                        _TARGET_RULE_ORDER[source.target_rule],
                    ),
                )
            ],
            "fixed_window": [
                {
                    "model": reference.setting.model,
                    "task": reference.setting.task,
                    "run_path": reference.relative_run_dir,
                    "run_sha256": reference.run_sha256,
                    "fixed_window_records_sha256": reference.records_sha256,
                    "pair_status_records_sha256": reference.statuses_sha256,
                    "setting_summary_sha256": reference.summary_sha256,
                    "model_revision": reference.model_revision,
                    "tokenizer_revision": reference.tokenizer_revision,
                }
                for reference in (references[setting.key] for setting in REBUTTAL_SETTINGS)
            ],
            "ignored_prepared": ignored_prepared,
            "ignored_fixed_window": ignored_fixed,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_manifest_path = output_dir / "pair_manifest.jsonl"
    cohort_ids_path = output_dir / "cohort_ids.json"
    source_audit_path = output_dir / "source_audit.json"
    run_path = output_dir / "run.json"
    public_paths = (pair_manifest_path, cohort_ids_path, source_audit_path)
    try:
        _write_jsonl_atomic(pair_manifest_path, records)
        cohort_ids["pair_manifest_sha256"] = sha256_file(pair_manifest_path)
        _write_json_atomic(cohort_ids_path, cohort_ids)
        _write_json_atomic(source_audit_path, source_audit)
        outputs = {
            path.name: {
                "sha256": sha256_file(path),
                "records": len(records) if path == pair_manifest_path else 1,
            }
            for path in public_paths
        }
        timestamp = _now()
        _write_json_atomic(
            run_path,
            {
                "schema_version": REBUTTAL_RUN_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "operation": "build-rebuttal-manifest",
                "status": "completed",
                "arguments": config.public_arguments(),
                "protocol": source_audit["protocol"],
                "counts": {
                    "pair_records": len(records),
                    "restoration_pairs": restoration_pairs,
                    "fixed_window_successes": fixed_window_successes,
                    "harm_pairs": harm_pairs,
                    "fixed_selected_anchors": fixed_selected_anchors,
                    "fixed_excluded_anchors": fixed_excluded_anchors,
                    "prepared_wrong_outside_restoration": (prepared_wrong_outside_restoration),
                },
                "outputs": outputs,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "gpu_required": False,
                },
                "started_at": timestamp,
                "updated_at": timestamp,
            },
        )
    except Exception:
        for path in (*public_paths, run_path):
            path.unlink(missing_ok=True)
            path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
        raise

    return BuildRebuttalManifestResult(
        pair_manifest_path=pair_manifest_path,
        cohort_ids_path=cohort_ids_path,
        source_audit_path=source_audit_path,
        run_path=run_path,
        pair_records=len(records),
        restoration_pairs=restoration_pairs,
        fixed_window_successes=fixed_window_successes,
        harm_pairs=harm_pairs,
        fixed_selected_anchors=fixed_selected_anchors,
        fixed_excluded_anchors=fixed_excluded_anchors,
        prepared_wrong_outside_restoration=prepared_wrong_outside_restoration,
    )


__all__ = [
    "BuildRebuttalManifestConfig",
    "BuildRebuttalManifestResult",
    "run_build_rebuttal_manifest",
]
