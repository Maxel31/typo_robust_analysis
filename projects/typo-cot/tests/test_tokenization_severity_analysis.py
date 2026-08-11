"""Contracts for the no-inference tokenization-severity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
)
from typo_cot.experiments.build_rebuttal_manifest.records import sha256_file
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.tokenization_severity_analysis.protocol import (
    load_tokenization_severity_protocol,
)
from typo_cot.experiments.tokenization_severity_analysis.runner import (
    TokenizationSeverityConfig,
    classify_tokenization_severity,
    run_tokenization_severity_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "tokenization-severity-analysis.yaml"
SIX_CONTROL_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "six-setting-patch-controls.yaml"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _edit(clean_tokens: list[int], typo_tokens: list[int]) -> dict[str, object]:
    return {
        "clean_token_indices": clean_tokens,
        "typo_token_indices": typo_tokens,
    }


def _record(*, model: str, task: str, index: int, pattern: int) -> dict[str, object]:
    if pattern == 0:
        edits = [_edit([5], [6])]
    elif pattern == 1:
        edits = [_edit([5], [6, 7]), _edit([9], [11])]
    elif pattern == 2:
        edits = [
            _edit([5, 6], [7, 8]),
            _edit([9], [11]),
            _edit([13], [15]),
            _edit([17], [19]),
        ]
    else:
        edits = [_edit([5, 6], [7]), _edit([9], [11]), _edit([13], [15])]
    pair_id = _digest(f"{model}\0{task}\0{index}")
    offset_valid = index % 4 != 0
    cross_valid = index % 5 != 0
    return {
        "pair_id": pair_id,
        "sample_id": f"sample-{index:04d}",
        "model": model,
        "task": task,
        "target_rule": "attribution-4",
        "number_of_aligned_words": len(edits),
        "edits": edits,
        "cohorts": {"restoration": True},
        "controls": {
            "correct": {"valid": True},
            "offset_2": {"valid": offset_valid},
            "cross_item": {"valid": cross_valid},
            "common_valid": offset_valid and cross_valid,
        },
        "fixed_window": {"event": index % 2 == 0},
    }


def _records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    primary = REBUTTAL_SETTINGS[0].key
    for setting in REBUTTAL_SETTINGS:
        for index in range(setting.paper_denominator):
            records.append(
                _record(
                    model=setting.model,
                    task=setting.task,
                    index=index,
                    pattern=0 if setting.key == primary else index % 4,
                )
            )
    return tuple(records)


def _event(record: dict[str, object], arm: str) -> bool:
    index = int(str(record["sample_id"]).rsplit("-", 1)[1])
    if arm == "correct":
        return bool(record["fixed_window"]["event"])  # type: ignore[index]
    if arm == "offset-2":
        return index % 3 == 0
    return index % 7 == 0


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_controls_run(
    directory: Path,
    *,
    manifest_path: Path,
    records: tuple[dict[str, object], ...],
    confirmatory: bool = True,
) -> None:
    directory.mkdir()
    control_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for record in records:
        controls = record["controls"]
        validity = {
            "correct": {"valid": True, "reason": None},
            "offset-2": {
                "valid": controls["offset_2"]["valid"],  # type: ignore[index]
                "reason": None,
            },
            "cross-item": {
                "valid": controls["cross_item"]["valid"],  # type: ignore[index]
                "reason": None,
            },
            "common-valid": controls["common_valid"],  # type: ignore[index]
        }
        events: dict[str, bool | None] = {}
        for arm in ("correct", "offset-2", "cross-item"):
            valid = validity[arm]["valid"]  # type: ignore[index]
            event = _event(record, arm) if valid else None
            events[arm] = event
            if valid:
                control_rows.append(
                    {
                        "schema_version": "six-setting-patch-controls-record/v1",
                        "paper_sha256": PAPER_SHA256,
                        "pair_id": record["pair_id"],
                        "sample_id": record["sample_id"],
                        "model": record["model"],
                        "task": record["task"],
                        "target_rule": record["target_rule"],
                        "control": arm,
                        "valid": True,
                        "event": event,
                    }
                )
        status_rows.append(
            {
                "schema_version": "six-setting-patch-controls-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "pair_id": record["pair_id"],
                "sample_id": record["sample_id"],
                "model": record["model"],
                "task": record["task"],
                "target_rule": record["target_rule"],
                "validity": validity,
                "events": events,
            }
        )

    paths = {
        "control_records.jsonl": directory / "control_records.jsonl",
        "pair_status_records.jsonl": directory / "pair_status_records.jsonl",
        "six_setting_control_table.csv": directory / "six_setting_control_table.csv",
        "common_denominator_flow.csv": directory / "common_denominator_flow.csv",
        "multiplicity_table.csv": directory / "multiplicity_table.csv",
        "macro_average.json": directory / "macro_average.json",
        "risk_difference_forest.svg": directory / "risk_difference_forest.svg",
    }
    _write_jsonl(paths["control_records.jsonl"], control_rows)
    _write_jsonl(paths["pair_status_records.jsonl"], status_rows)
    for name in (
        "six_setting_control_table.csv",
        "common_denominator_flow.csv",
        "multiplicity_table.csv",
    ):
        paths[name].write_text("value\n1\n", encoding="utf-8")
    paths["macro_average.json"].write_text("{}\n", encoding="utf-8")
    paths["risk_difference_forest.svg"].write_text("<svg></svg>\n", encoding="utf-8")
    row_counts = {
        "control_records.jsonl": len(control_rows),
        "pair_status_records.jsonl": len(status_rows),
        "six_setting_control_table.csv": 1,
        "common_denominator_flow.csv": 1,
        "multiplicity_table.csv": 1,
        "macro_average.json": 1,
        "risk_difference_forest.svg": 1,
    }
    run = {
        "schema_version": "six-setting-patch-controls-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "six-setting-patch-controls",
        "status": "completed",
        "confirmatory": confirmatory,
        "arguments": {
            "manifest_path": str(manifest_path.resolve()),
            "limit_per_setting": None if confirmatory else 1,
        },
        "protocol": json.loads(SIX_CONTROL_CONFIG.read_text(encoding="utf-8")),
        "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
        "pair_manifest_sha256": sha256_file(manifest_path),
        "failures": [],
        "counts": {
            "common_valid_pairs": sum(
                int(record["controls"]["common_valid"] is True)  # type: ignore[index]
                for record in records
            ),
            "control_records": len(control_rows),
            "pair_status_records": len(status_rows),
            "selected_pairs": len(status_rows),
            "settings": len(REBUTTAL_SETTINGS),
        },
        "outputs": {
            name: {"sha256": sha256_file(path), "records": row_counts[name]}
            for name, path in paths.items()
        },
    }
    (directory / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config(tmp_path: Path, *, resume: bool = False) -> TokenizationSeverityConfig:
    manifest = tmp_path / "manifest" / "pair_manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("fixture\n", encoding="utf-8")
    _write_controls_run(
        tmp_path / "controls",
        manifest_path=manifest,
        records=_records(),
    )
    return TokenizationSeverityConfig(
        protocol_path=DEFAULT_CONFIG,
        manifest_path=manifest,
        controls_run=tmp_path / "controls",
        output_dir=tmp_path / "output",
        resume=resume,
    )


def test_default_protocol_catalog_and_cli_match_the_frozen_readme() -> None:
    protocol = load_tokenization_severity_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "tokenization-severity-analysis-config/v1"
    assert protocol.controls == ("correct", "offset-2", "cross-item")
    assert protocol.denominators == ("arm-valid", "common-valid")
    assert protocol.scopes == ("overall", "setting")
    assert tuple(protocol.dimensions) == (
        "subtoken-count-change",
        "typo-fragmentation",
        "edit-count",
        "clean-edited-word-tokenization",
    )

    spec = get_experiment("tokenization-severity-analysis")
    assert spec.status == "implemented"
    assert spec.compute == "cpu"
    assert spec.required_arguments == (
        "--config",
        "--manifest",
        "--controls-run",
        "--output-dir",
    )
    assert spec.outputs == (
        "tokenization_severity_records.jsonl",
        "tokenization_severity_table.csv",
        "tokenization_severity_summary.json",
        "run.json",
    )

    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "tokenization-severity-analysis" in subparsers.choices


def test_protocol_rejects_dimension_or_empty_cell_policy_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["dimensions"][0]["bins"].reverse()
    changed = tmp_path / "changed.yaml"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dimension"):
        load_tokenization_severity_protocol(changed)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["empty_cells"] = "drop"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_tokenization_severity_protocol(changed)


@pytest.mark.parametrize(
    ("edits", "expected"),
    (
        (
            [_edit([1], [2])],
            {
                "subtoken-count-change": "unchanged-all-edits",
                "typo-fragmentation": "not-increased",
                "edit-count": "1",
                "clean-edited-word-tokenization": "all-single-token",
            },
        ),
        (
            [_edit([1], [2, 3]), _edit([4, 5], [6])],
            {
                "subtoken-count-change": "changed-any-edit",
                "typo-fragmentation": "increased-any-edit",
                "edit-count": "2",
                "clean-edited-word-tokenization": "any-multi-token",
            },
        ),
        (
            [_edit([1, 2], [3]), _edit([4], [5]), _edit([6], [7])],
            {
                "subtoken-count-change": "changed-any-edit",
                "typo-fragmentation": "not-increased",
                "edit-count": "3-4",
                "clean-edited-word-tokenization": "any-multi-token",
            },
        ),
    ),
)
def test_pair_classification_is_deterministic_and_word_aligned(
    edits: list[dict[str, object]],
    expected: dict[str, str],
) -> None:
    assert classify_tokenization_severity(edits) == expected


def test_analysis_emits_every_bin_and_both_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    from typo_cot.experiments.tokenization_severity_analysis import runner

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", lambda _path: records)
    config = _config(tmp_path)

    result = run_tokenization_severity_analysis(config)

    assert result.pairs == 1_241
    assert result.record_rows == 1_241
    assert result.table_rows == 189
    assert result.empty_cells > 0
    record_rows = [
        json.loads(line) for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(record_rows) == 1_241
    assert all(len(row["strata"]) == 4 for row in record_rows)

    with result.table_path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    assert len(rows) == 7 * 9 * 3
    empty = next(
        row
        for row in rows
        if row["scope"] == "setting"
        and row["model"] == REBUTTAL_SETTINGS[0].model
        and row["task"] == REBUTTAL_SETTINGS[0].task
        and row["dimension"] == "edit-count"
        and row["stratum"] == "2"
        and row["arm"] == "correct"
    )
    assert int(empty["n_pairs"]) == 0
    assert int(empty["arm_valid_n"]) == 0
    assert empty["arm_valid_rate"] == ""
    assert int(empty["common_valid_n"]) == 0
    assert empty["common_valid_rate"] == ""

    populated = next(
        row
        for row in rows
        if row["scope"] == "overall"
        and row["dimension"] == "subtoken-count-change"
        and row["stratum"] == "changed-any-edit"
        and row["arm"] == "offset-2"
    )
    assert int(populated["n_pairs"]) > 0
    assert int(populated["arm_valid_n"]) >= int(populated["common_valid_n"])
    assert 0.0 <= float(populated["arm_valid_rate"]) <= 1.0
    assert 0.0 <= float(populated["common_valid_rate"]) <= 1.0

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["pairs"] == 1_241
    assert summary["counts"]["empty_cells"] == result.empty_cells
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["inputs"]["controls_run_sha256"] == sha256_file(config.controls_run / "run.json")

    resumed = run_tokenization_severity_analysis(replace(config, resume=True))
    assert resumed == result


def test_analysis_rejects_nonconfirmatory_or_tampered_controls_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    from typo_cot.experiments.tokenization_severity_analysis import runner

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", lambda _path: records)
    config = _config(tmp_path)
    run_path = config.controls_run / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["confirmatory"] = False
    run["arguments"]["limit_per_setting"] = 1
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmatory"):
        run_tokenization_severity_analysis(config)

    other = tmp_path / "other"
    other.mkdir()
    manifest = other / "manifest.jsonl"
    manifest.write_text("fixture\n", encoding="utf-8")
    controls = other / "controls"
    _write_controls_run(controls, manifest_path=manifest, records=records)
    with (controls / "control_records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    tampered = TokenizationSeverityConfig(
        protocol_path=DEFAULT_CONFIG,
        manifest_path=manifest,
        controls_run=controls,
        output_dir=other / "output",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        run_tokenization_severity_analysis(tampered)


def test_completed_resume_rejects_tampered_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    from typo_cot.experiments.tokenization_severity_analysis import runner

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", lambda _path: records)
    config = _config(tmp_path)
    result = run_tokenization_severity_analysis(config)
    result.summary_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output SHA-256 differs"):
        run_tokenization_severity_analysis(replace(config, resume=True))
