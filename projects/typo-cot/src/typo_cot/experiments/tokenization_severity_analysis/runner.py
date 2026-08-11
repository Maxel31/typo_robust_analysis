"""CPU-only tokenization-severity stratification over three patch arms."""

from __future__ import annotations

import csv
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

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
from typo_cot.experiments.tokenization_severity_analysis.protocol import (
    TokenizationSeverityProtocol,
    load_tokenization_severity_protocol,
)
from typo_cot.experiments.tokenization_severity_analysis.source import (
    CompletedControlRun,
    load_completed_control_run,
)

_RUN_SCHEMA = "tokenization-severity-analysis-run/v1"
_RECORD_SCHEMA = "tokenization-severity-analysis-record/v1"
_SUMMARY_SCHEMA = "tokenization-severity-analysis-summary/v1"
_PUBLIC_OUTPUTS = (
    "tokenization_severity_records.jsonl",
    "tokenization_severity_table.csv",
    "tokenization_severity_summary.json",
)


@dataclass(frozen=True, slots=True)
class TokenizationSeverityConfig:
    """Public command arguments; ``resume`` does not alter the analysis."""

    protocol_path: Path
    manifest_path: Path
    controls_run: Path
    output_dir: Path
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("protocol_path", "manifest_path", "controls_run", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        if type(self.resume) is not bool:
            raise TypeError("resume must be boolean")

    def public_arguments(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("protocol_path", "manifest_path", "controls_run", "output_dir"):
            payload[field] = str(Path(payload[field]).resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class TokenizationSeverityResult:
    records_path: Path
    table_path: Path
    summary_path: Path
    run_path: Path
    pairs: int
    record_rows: int
    table_rows: int
    empty_cells: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _token_count(value: object, *, field: str) -> int:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty token-index list")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
        raise ValueError(f"{field} contains an invalid token index")
    if any(left >= right for left, right in zip(value, value[1:], strict=False)):
        raise ValueError(f"{field} must be strictly increasing")
    return len(value)


def classify_tokenization_severity(
    edits: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Assign one prespecified bin in each dimension from aligned token indices."""

    try:
        normalized = tuple(edits)
    except TypeError as exc:
        raise ValueError("edits must be a sequence") from exc
    if not 1 <= len(normalized) <= 4 or any(not isinstance(edit, Mapping) for edit in normalized):
        raise ValueError("edits must contain one to four aligned edit objects")
    counts = tuple(
        (
            _token_count(edit.get("clean_token_indices"), field=f"edits[{index}].clean"),
            _token_count(edit.get("typo_token_indices"), field=f"edits[{index}].typo"),
        )
        for index, edit in enumerate(normalized)
    )
    edit_count = str(len(counts)) if len(counts) < 3 else "3-4"
    return {
        "subtoken-count-change": (
            "changed-any-edit"
            if any(clean != typo for clean, typo in counts)
            else "unchanged-all-edits"
        ),
        "typo-fragmentation": (
            "increased-any-edit" if any(typo > clean for clean, typo in counts) else "not-increased"
        ),
        "edit-count": edit_count,
        "clean-edited-word-tokenization": (
            "all-single-token" if all(clean == 1 for clean, _typo in counts) else "any-multi-token"
        ),
    }


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
        raise ValueError("tokenization severity table must not be empty")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("tokenization severity table fields differ")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _restoration_records(
    records: Sequence[Mapping[str, object]],
    *,
    protocol: TokenizationSeverityProtocol,
) -> tuple[Mapping[str, object], ...]:
    restoration = tuple(
        record
        for record in records
        if _mapping(record.get("cohorts"), field="record cohorts").get("restoration") is True
    )
    if len(restoration) != protocol.required_pairs:
        raise ValueError("tokenization severity restoration denominator differs")
    return restoration


def _record_rows(
    *,
    records: Sequence[Mapping[str, object]],
    source: CompletedControlRun,
    protocol: TokenizationSeverityProtocol,
    manifest_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        pair_id = str(record.get("pair_id"))
        outcome = source.outcomes.get(pair_id)
        edits = record.get("edits")
        if outcome is None or not isinstance(edits, list):
            raise ValueError("tokenization severity pair lacks source outcome or edits")
        if record.get("number_of_aligned_words") != len(edits):
            raise ValueError("aligned edit count differs from manifest edits")
        strata = classify_tokenization_severity(edits)
        if set(strata) != set(protocol.dimensions) or any(
            stratum not in protocol.dimensions[dimension] for dimension, stratum in strata.items()
        ):
            raise ValueError("tokenization severity classification escaped the protocol")
        rows.append(
            {
                "schema_version": _RECORD_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "protocol_sha256": protocol.config_sha256,
                "pair_manifest_sha256": manifest_sha256,
                "controls_run_sha256": source.run_sha256,
                "pair_id": pair_id,
                "sample_id": record.get("sample_id"),
                "model": record.get("model"),
                "task": record.get("task"),
                "target_rule": record.get("target_rule"),
                "strata": strata,
                "validity": {arm: outcome.valid(arm) for arm in protocol.controls}
                | {"common-valid": outcome.common_valid},
                "events": {arm: outcome.event(arm) for arm in protocol.controls},
            }
        )
    return rows


def _rate(successes: int, denominator: int) -> float | None:
    return successes / denominator if denominator else None


def _table_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    protocol: TokenizationSeverityProtocol,
) -> tuple[list[dict[str, object]], int]:
    scopes: list[tuple[str, str, str, Sequence[Mapping[str, object]]]] = [
        ("overall", "all", "all", rows)
    ]
    scopes.extend(
        (
            "setting",
            setting.model,
            setting.task,
            tuple(row for row in rows if (row.get("model"), row.get("task")) == setting.key),
        )
        for setting in REBUTTAL_SETTINGS
    )
    output: list[dict[str, object]] = []
    empty_cells = 0
    for scope, model, task, scope_rows in scopes:
        for dimension, bins in protocol.dimensions.items():
            for stratum in bins:
                cell = tuple(
                    row
                    for row in scope_rows
                    if _mapping(row.get("strata"), field="severity strata").get(dimension)
                    == stratum
                )
                empty_cells += int(not cell)
                common = tuple(
                    row
                    for row in cell
                    if _mapping(row.get("validity"), field="severity validity").get("common-valid")
                    is True
                )
                for arm in protocol.controls:
                    arm_valid = tuple(
                        row
                        for row in cell
                        if _mapping(row.get("validity"), field="severity validity").get(arm) is True
                    )
                    arm_successes = sum(
                        _mapping(row.get("events"), field="severity events").get(arm) is True
                        for row in arm_valid
                    )
                    common_successes = sum(
                        _mapping(row.get("events"), field="severity events").get(arm) is True
                        for row in common
                    )
                    output.append(
                        {
                            "scope": scope,
                            "model": model,
                            "task": task,
                            "dimension": dimension,
                            "stratum": stratum,
                            "arm": arm,
                            "n_pairs": len(cell),
                            "arm_valid_n": len(arm_valid),
                            "arm_valid_successes": arm_successes,
                            "arm_valid_rate": _rate(arm_successes, len(arm_valid)),
                            "common_valid_n": len(common),
                            "common_valid_successes": common_successes,
                            "common_valid_rate": _rate(common_successes, len(common)),
                        }
                    )
    return output, empty_cells


def _result_from_run(
    output_dir: Path,
    run: Mapping[str, object],
) -> TokenizationSeverityResult:
    outputs = _mapping(run.get("outputs"), field="completed severity outputs")
    if set(outputs) != set(_PUBLIC_OUTPUTS):
        raise ValueError("completed tokenization severity output inventory differs")
    for name in _PUBLIC_OUTPUTS:
        path = output_dir / name
        metadata = _mapping(outputs.get(name), field=f"completed severity output {name}")
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"completed tokenization severity output SHA-256 differs: {path}")
    records = tuple(
        row
        for _number, _line, row in iter_jsonl_objects(
            output_dir / "tokenization_severity_records.jsonl"
        )
    )
    if any(row.get("schema_version") != _RECORD_SCHEMA for row in records):
        raise ValueError("completed tokenization severity record schema differs")
    with (output_dir / "tokenization_severity_table.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        table = tuple(reader)
    if reader.fieldnames is None:
        raise ValueError("completed tokenization severity table has no header")
    summary = load_json_object(output_dir / "tokenization_severity_summary.json")
    counts = _mapping(run.get("counts"), field="completed severity counts")
    expected = {
        "pairs": len(records),
        "record_rows": len(records),
        "table_rows": len(table),
        "empty_cells": _mapping(summary.get("counts"), field="severity summary counts").get(
            "empty_cells"
        ),
    }
    if dict(counts) != expected:
        raise ValueError("completed tokenization severity counts differ from outputs")
    expected_output_counts = {
        "tokenization_severity_records.jsonl": len(records),
        "tokenization_severity_table.csv": len(table),
        "tokenization_severity_summary.json": 1,
    }
    for name, count in expected_output_counts.items():
        if (
            _mapping(outputs[name], field=f"completed severity output {name}").get("records")
            != count
        ):
            raise ValueError(f"completed tokenization severity output count differs: {name}")
    return TokenizationSeverityResult(
        records_path=output_dir / _PUBLIC_OUTPUTS[0],
        table_path=output_dir / _PUBLIC_OUTPUTS[1],
        summary_path=output_dir / _PUBLIC_OUTPUTS[2],
        run_path=output_dir / "run.json",
        pairs=int(counts["pairs"]),
        record_rows=int(counts["record_rows"]),
        table_rows=int(counts["table_rows"]),
        empty_cells=int(counts["empty_cells"]),
    )


def _completed_resume(
    config: TokenizationSeverityConfig,
    *,
    protocol: TokenizationSeverityProtocol,
    inputs: Mapping[str, object],
) -> TokenizationSeverityResult | None:
    run_path = config.output_dir / "run.json"
    if not config.resume or not run_path.is_file():
        return None
    run = load_json_object(run_path)
    if run.get("status") != "completed":
        return None
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "tokenization-severity-analysis"
        or run.get("arguments") != config.public_arguments()
        or run.get("protocol") != protocol.as_dict()
        or run.get("protocol_sha256") != protocol.config_sha256
        or run.get("inputs") != dict(inputs)
    ):
        raise ValueError("completed tokenization severity resume contract differs")
    return _result_from_run(config.output_dir, run)


def run_tokenization_severity_analysis(
    config: TokenizationSeverityConfig,
) -> TokenizationSeverityResult:
    """Validate upstream results and publish every prespecified severity cell."""

    if not isinstance(config, TokenizationSeverityConfig):
        raise TypeError("config must be TokenizationSeverityConfig")
    protocol = load_tokenization_severity_protocol(config.protocol_path)
    if not config.manifest_path.is_file():
        raise ValueError(f"rebuttal manifest is not a file: {config.manifest_path}")
    records = load_rebuttal_pair_manifest(config.manifest_path)
    restoration = _restoration_records(records, protocol=protocol)
    manifest_sha256 = sha256_file(config.manifest_path)
    source = load_completed_control_run(
        records=records,
        manifest_path=config.manifest_path,
        controls_run=config.controls_run,
        protocol=protocol,
    )
    inputs = {
        "pair_manifest_sha256": manifest_sha256,
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "controls_run_sha256": source.run_sha256,
        "control_records_sha256": source.control_records_sha256,
        "pair_status_records_sha256": source.status_records_sha256,
    }
    completed = _completed_resume(config, protocol=protocol, inputs=inputs)
    if completed is not None:
        return completed
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    record_rows = _record_rows(
        records=restoration,
        source=source,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
    )
    table_rows, empty_cells = _table_rows(record_rows, protocol=protocol)
    dimension_counts = {
        dimension: {
            stratum: sum(
                _mapping(row.get("strata"), field="severity strata").get(dimension) == stratum
                for row in record_rows
            )
            for stratum in bins
        }
        for dimension, bins in protocol.dimensions.items()
    }
    summary = {
        "schema_version": _SUMMARY_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "protocol_sha256": protocol.config_sha256,
        "counts": {
            "pairs": len(record_rows),
            "settings": len(REBUTTAL_SETTINGS),
            "dimensions": len(protocol.dimensions),
            "strata": sum(len(bins) for bins in protocol.dimensions.values()),
            "table_rows": len(table_rows),
            "empty_cells": empty_cells,
        },
        "dimension_counts": dimension_counts,
        "denominators": list(protocol.denominators),
        "empty_cell_policy": protocol.empty_cells,
        "additional_model_inference": False,
    }
    paths = {name: output_dir / name for name in _PUBLIC_OUTPUTS}
    _write_jsonl_atomic(paths[_PUBLIC_OUTPUTS[0]], record_rows)
    _write_csv_atomic(paths[_PUBLIC_OUTPUTS[1]], table_rows)
    _write_json_atomic(paths[_PUBLIC_OUTPUTS[2]], summary)
    counts = {
        "pairs": len(record_rows),
        "record_rows": len(record_rows),
        "table_rows": len(table_rows),
        "empty_cells": empty_cells,
    }
    output_counts = {
        _PUBLIC_OUTPUTS[0]: len(record_rows),
        _PUBLIC_OUTPUTS[1]: len(table_rows),
        _PUBLIC_OUTPUTS[2]: 1,
    }
    run = {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "tokenization-severity-analysis",
        "status": "completed",
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "inputs": inputs,
        "counts": counts,
        "outputs": {
            name: {"sha256": sha256_file(path), "records": output_counts[name]}
            for name, path in paths.items()
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu_required": False,
        },
        "completed_at": _now(),
    }
    _write_json_atomic(output_dir / "run.json", run)
    return _result_from_run(output_dir, run)


__all__ = [
    "TokenizationSeverityConfig",
    "TokenizationSeverityResult",
    "classify_tokenization_severity",
    "run_tokenization_severity_analysis",
]
