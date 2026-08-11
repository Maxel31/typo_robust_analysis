"""Validated six-setting control outcomes for severity stratification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
)
from typo_cot.experiments.build_rebuttal_manifest.records import (
    iter_jsonl_objects,
    load_json_object,
    sha256_file,
)
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.tokenization_severity_analysis.protocol import (
    TokenizationSeverityProtocol,
)

_OUTPUTS = {
    "control_records.jsonl",
    "pair_status_records.jsonl",
    "six_setting_control_table.csv",
    "common_denominator_flow.csv",
    "multiplicity_table.csv",
    "macro_average.json",
    "risk_difference_forest.svg",
}


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    """Three-arm validity and events for one restoration pair."""

    correct_event: bool
    offset_valid: bool
    offset_event: bool | None
    cross_valid: bool
    cross_event: bool | None
    common_valid: bool

    def valid(self, arm: str) -> bool:
        if arm == "correct":
            return True
        if arm == "offset-2":
            return self.offset_valid
        if arm == "cross-item":
            return self.cross_valid
        raise ValueError(f"unknown control arm: {arm!r}")

    def event(self, arm: str) -> bool | None:
        if arm == "correct":
            return self.correct_event
        if arm == "offset-2":
            return self.offset_event
        if arm == "cross-item":
            return self.cross_event
        raise ValueError(f"unknown control arm: {arm!r}")


@dataclass(frozen=True, slots=True)
class CompletedControlRun:
    """Hash-bound outcomes and producer fingerprints."""

    outcomes: dict[str, ControlOutcome]
    run_sha256: str
    control_records_sha256: str
    status_records_sha256: str


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _identity_matches(row: Mapping[str, object], record: Mapping[str, object]) -> bool:
    return all(
        row.get(field) == record.get(field)
        for field in ("pair_id", "sample_id", "model", "task", "target_rule")
    )


def _output_paths(run_dir: Path, run: Mapping[str, object]) -> dict[str, Path]:
    outputs = _mapping(run.get("outputs"), field="controls outputs")
    if set(outputs) != _OUTPUTS:
        raise ValueError("six-setting controls output inventory differs")
    paths: dict[str, Path] = {}
    for name in sorted(_OUTPUTS):
        metadata = _mapping(outputs.get(name), field=f"controls outputs.{name}")
        path = run_dir / name
        digest = metadata.get("sha256")
        count = metadata.get("records")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError(f"six-setting controls output SHA-256 differs: {path}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"six-setting controls output count is invalid: {path}")
        paths[name] = path
    return paths


def _restoration_records(
    records: Sequence[Mapping[str, object]],
    *,
    required_pairs: int,
) -> tuple[Mapping[str, object], ...]:
    restoration = tuple(
        record
        for record in records
        if _mapping(record.get("cohorts"), field="manifest cohorts").get("restoration") is True
    )
    if len(restoration) != required_pairs:
        raise ValueError("tokenization severity manifest denominator differs")
    seen: set[str] = set()
    for record in restoration:
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or len(pair_id) != 64 or pair_id in seen:
            raise ValueError("tokenization severity manifest pair IDs differ")
        seen.add(pair_id)
    for setting in REBUTTAL_SETTINGS:
        count = sum(
            (record.get("model"), record.get("task")) == setting.key for record in restoration
        )
        if count != setting.paper_denominator:
            raise ValueError(f"tokenization severity denominator differs for {setting.slug}")
    return restoration


def load_completed_control_run(
    *,
    records: Sequence[Mapping[str, object]],
    manifest_path: Path,
    controls_run: Path,
    protocol: TokenizationSeverityProtocol,
) -> CompletedControlRun:
    """Validate the complete upstream run and return pairwise outcomes."""

    restoration = _restoration_records(records, required_pairs=protocol.required_pairs)
    by_id = {str(record["pair_id"]): record for record in restoration}
    manifest_path = Path(manifest_path).resolve()
    run_dir = Path(controls_run).resolve()
    run_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise ValueError(f"rebuttal manifest is not a file: {manifest_path}")
    if not run_path.is_file():
        raise ValueError(f"six-setting controls run is not a file: {run_path}")
    run = load_json_object(run_path)
    arguments = _mapping(run.get("arguments"), field="controls arguments")
    producer_protocol = _mapping(run.get("protocol"), field="controls protocol")
    if (
        run.get("schema_version") != protocol.controls_run_schema
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "six-setting-patch-controls"
        or run.get("status") != "completed"
        or run.get("confirmatory") is not protocol.require_confirmatory
        or arguments.get("limit_per_setting") is not None
        or run.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or run.get("pair_manifest_sha256") != sha256_file(manifest_path)
        or run.get("failures") not in ([], None)
        or producer_protocol.get("schema_version") != "six-setting-patch-controls-config/v1"
        or producer_protocol.get("controls") != list(protocol.controls)
        or producer_protocol.get("window") != {"start": 0, "stop": 6}
    ):
        raise ValueError("six-setting controls run is not complete and confirmatory")
    paths = _output_paths(run_dir, run)
    outputs = _mapping(run.get("outputs"), field="controls outputs")
    status_items = tuple(iter_jsonl_objects(paths["pair_status_records.jsonl"]))
    control_items = tuple(iter_jsonl_objects(paths["control_records.jsonl"]))
    if len(status_items) != _mapping(
        outputs["pair_status_records.jsonl"], field="status metadata"
    ).get("records") or len(control_items) != _mapping(
        outputs["control_records.jsonl"], field="control metadata"
    ).get("records"):
        raise ValueError("six-setting controls JSONL count differs from run metadata")

    outcomes: dict[str, ControlOutcome] = {}
    expected_control_keys: set[tuple[str, str]] = set()
    for line_number, _line, status in status_items:
        context = f"{paths['pair_status_records.jsonl']}:{line_number}"
        pair_id = status.get("pair_id")
        record = by_id.get(str(pair_id))
        if (
            record is None
            or status.get("schema_version") != "six-setting-patch-controls-pair-status/v1"
            or status.get("paper_sha256") != PAPER_SHA256
            or not _identity_matches(status, record)
            or str(pair_id) in outcomes
        ):
            raise ValueError(f"six-setting controls status identity differs: {context}")
        validity = _mapping(status.get("validity"), field=f"{context}.validity")
        events = _mapping(status.get("events"), field=f"{context}.events")
        if set(validity) != {*protocol.controls, "common-valid"} or set(events) != set(
            protocol.controls
        ):
            raise ValueError(f"six-setting controls status arms differ: {context}")
        correct_valid = _boolean(
            _mapping(validity.get("correct"), field=f"{context}.validity.correct").get("valid"),
            field=f"{context}.validity.correct.valid",
        )
        offset_valid = _boolean(
            _mapping(validity.get("offset-2"), field=f"{context}.validity.offset-2").get("valid"),
            field=f"{context}.validity.offset-2.valid",
        )
        cross_valid = _boolean(
            _mapping(validity.get("cross-item"), field=f"{context}.validity.cross-item").get(
                "valid"
            ),
            field=f"{context}.validity.cross-item.valid",
        )
        common_valid = _boolean(
            validity.get("common-valid"), field=f"{context}.validity.common-valid"
        )
        manifest_controls = _mapping(record.get("controls"), field="manifest controls")
        manifest_offset = _mapping(
            manifest_controls.get("offset_2"), field="manifest controls.offset_2"
        ).get("valid")
        manifest_cross = _mapping(
            manifest_controls.get("cross_item"), field="manifest controls.cross_item"
        ).get("valid")
        if (
            not correct_valid
            or offset_valid is not manifest_offset
            or cross_valid is not manifest_cross
            or common_valid is not manifest_controls.get("common_valid")
            or common_valid != (offset_valid and cross_valid)
        ):
            raise ValueError(f"six-setting controls validity differs from manifest: {context}")
        parsed_events: dict[str, bool | None] = {}
        for arm, valid in (
            ("correct", True),
            ("offset-2", offset_valid),
            ("cross-item", cross_valid),
        ):
            event = events.get(arm)
            if valid:
                parsed_events[arm] = _boolean(event, field=f"{context}.events.{arm}")
                expected_control_keys.add((str(pair_id), arm))
            elif event is not None:
                raise ValueError(f"invalid control arm has an event: {context}:{arm}")
            else:
                parsed_events[arm] = None
        fixed = _mapping(record.get("fixed_window"), field="manifest fixed_window")
        if parsed_events["correct"] is not fixed.get("event"):
            raise ValueError(f"correct event differs from manifest reference: {context}")
        outcomes[str(pair_id)] = ControlOutcome(
            correct_event=bool(parsed_events["correct"]),
            offset_valid=offset_valid,
            offset_event=parsed_events["offset-2"],
            cross_valid=cross_valid,
            cross_event=parsed_events["cross-item"],
            common_valid=common_valid,
        )
    if set(outcomes) != set(by_id):
        raise ValueError("six-setting controls statuses do not cover the restoration cohort")

    actual_control_keys: set[tuple[str, str]] = set()
    for line_number, _line, row in control_items:
        context = f"{paths['control_records.jsonl']}:{line_number}"
        pair_id = str(row.get("pair_id"))
        arm = row.get("control")
        record = by_id.get(pair_id)
        key = pair_id, str(arm)
        if (
            record is None
            or arm not in protocol.controls
            or key in actual_control_keys
            or row.get("schema_version") != "six-setting-patch-controls-record/v1"
            or row.get("paper_sha256") != PAPER_SHA256
            or row.get("valid") is not True
            or not _identity_matches(row, record)
            or type(row.get("event")) is not bool
            or row.get("event") is not outcomes[pair_id].event(str(arm))
        ):
            raise ValueError(f"six-setting control record differs: {context}")
        actual_control_keys.add(key)
    if actual_control_keys != expected_control_keys:
        raise ValueError("six-setting control records do not match planned-valid arms")

    counts = _mapping(run.get("counts"), field="controls counts")
    expected_counts = {
        "common_valid_pairs": sum(outcome.common_valid for outcome in outcomes.values()),
        "control_records": len(actual_control_keys),
        "pair_status_records": len(outcomes),
        "selected_pairs": len(outcomes),
        "settings": len(REBUTTAL_SETTINGS),
    }
    if dict(counts) != expected_counts:
        raise ValueError("six-setting controls run counts differ from validated records")
    return CompletedControlRun(
        outcomes=outcomes,
        run_sha256=sha256_file(run_path),
        control_records_sha256=sha256_file(paths["control_records.jsonl"]),
        status_records_sha256=sha256_file(paths["pair_status_records.jsonl"]),
    )


__all__ = ["CompletedControlRun", "ControlOutcome", "load_completed_control_run"]
