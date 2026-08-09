"""Publication tests for the CPU one-token table builder."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from typo_cot.experiments.build_one_token_tables import (
    BuildOneTokenTablesConfig,
    InferenceProtocol,
    ValidatedOneTokenRun,
    run_build_one_token_tables,
)
from typo_cot.experiments.build_one_token_tables import runner as builder_runner
from typo_cot.experiments.catalog import PAPER_SHA256


OUTPUTS = {
    "one_token_tables.json",
    "table10_one_token.csv",
    "table11_position_controls.csv",
    "one_token_tables.md",
    "one_token_tables.tex",
    "figure5_validation.json",
    "run.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(index: int = 1) -> dict[str, object]:
    return {
        "sample_id": f"mmlu_synthetic_{index:03d}",
        "benchmark": "mmlu",
        "targeting": "attribution-4",
        "input_plan": {"clean_cot_ids": [1, 2, 3, 4, 5]},
        "profile": {
            "clean_to_edited_kl": [0.1, 2.0, 0.5, 0.2, 0.1],
            "clean_token_rank_under_clean": [1, 1, 1, 1, 1],
            "clean_token_rank_under_edited": [1, 2, 2, 2, 2],
        },
        "positions": {"selected": 1, "distant": 4, "adjacent": None},
        "arms": [],
        "events": {
            "table10": {
                "eligible": True,
                "selected_correct_to_incorrect": True,
                "control_correct_to_incorrect": False,
            },
            "distant_factorial": {
                "eligible": True,
                "selected_loss_events": 2,
                "control_loss_events": 0,
                "event_opportunities_per_position": 2,
                "selected_before_control": True,
            },
            "distant_factorial_submitted_producer": {
                "eligible": True,
                "selected_loss_events": 2,
                "control_loss_events": 0,
                "event_opportunities_per_position": 2,
                "selected_before_control": True,
                "exclusion_reasons": [],
            },
            "adjacent": None,
        },
    }


def _validated_run(tmp_path: Path) -> ValidatedOneTokenRun:
    run_dir = tmp_path / "producer" / "gemma1b_mmlu"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "run.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return ValidatedOneTokenRun(
        setting_id="gemma1b_mmlu",
        model="google/gemma-3-1b-it",
        benchmark="mmlu",
        cohort="extension",
        adjacent_requested=False,
        records=(_record(1), _record(2)),
        run_dir=run_dir.resolve(),
        position_controls=("distant",),
        manifest_sha256=_sha256(manifest),
        producer_code_identity={
            "algorithm": "fixture/v1",
            "python_file_count": 1,
            "sha256": "a" * 64,
        },
        input_files={
            "one_token_records.jsonl": {"sha256": "b" * 64, "bytes": 123},
        },
    )


def test_runner_publishes_exactly_the_documented_hash_bound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_run(tmp_path)
    monkeypatch.setattr(
        builder_runner,
        "discover_and_validate_runs",
        lambda root: (validated,),
    )
    config = BuildOneTokenTablesConfig(
        runs_root=tmp_path / "producer",
        output_dir=tmp_path / "tables",
    )

    result = run_build_one_token_tables(
        config,
        inference=InferenceProtocol(seed=42, bootstrap_draws=16, sign_flip_draws=16),
    )

    assert result.settings == 1
    assert result.output_dir == config.output_dir.resolve()
    assert {path.name for path in result.output_dir.iterdir()} == OUTPUTS
    analysis = json.loads(result.tables_path.read_text(encoding="utf-8"))
    assert analysis["schema_version"] == "one-token-tables/v1"
    assert analysis["coverage"]["available_settings"] == ["gemma1b_mmlu"]
    assert analysis["coverage"]["complete_14_extension_grid"] is False
    assert analysis["table10"]["extension_pool"] is None

    figure = json.loads(result.figure5_path.read_text(encoding="utf-8"))
    assert figure == analysis["figure5_validation"]
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "build-one-token-tables-run/v1"
    assert manifest["paper_sha256"] == PAPER_SHA256
    assert manifest["operation"] == "build-one-token-tables"
    assert manifest["status"] == "completed"
    assert manifest["inputs"][0]["setting_id"] == "gemma1b_mmlu"
    assert manifest["inputs"][0]["run_manifest_sha256"] == validated.manifest_sha256
    assert set(manifest["outputs"]) == OUTPUTS - {"run.json"}
    for name, metadata in manifest["outputs"].items():
        path = result.output_dir / name
        assert metadata == {"sha256": _sha256(path), "bytes": path.stat().st_size}

    with result.table10_csv_path.open(encoding="utf-8", newline="") as handle:
        table10 = list(csv.DictReader(handle))
    assert [row["setting_id"] for row in table10] == ["gemma1b_mmlu"]
    assert table10[0]["paired_eligible"] == "2"
    assert table10[0]["selected_numerator"] == "2"
    assert table10[0]["control_numerator"] == "0"
    markdown = result.markdown_path.read_text(encoding="utf-8")
    latex = result.latex_path.read_text(encoding="utf-8")
    assert "Table 10" in markdown
    assert "distant_pdf_literal" in markdown
    assert "Difference [95% CI] (pp)" in markdown
    assert "100.0 [100.0, 100.0]" in markdown
    assert "## Coverage" in markdown
    assert "## Historical-reference comparison" in markdown
    assert "Table 10 gemma1b_mmlu: differs" in markdown
    assert latex.count("\\begin{tabular}") == 2
    assert "distant\\_pdf\\_literal" in latex
    assert r"$\Delta$ [95\% CI] (pp)" in latex
    assert "100.0 [100.0, 100.0]" in latex


def test_runner_removes_private_staging_directory_after_render_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_run(tmp_path)
    monkeypatch.setattr(
        builder_runner,
        "discover_and_validate_runs",
        lambda root: (validated,),
    )

    def fail_after_partial_write(output_dir: Path, _analysis: object) -> None:
        (output_dir / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("injected render failure")

    monkeypatch.setattr(builder_runner, "render_artifacts", fail_after_partial_write)
    output = tmp_path / "tables"

    with pytest.raises(RuntimeError, match="injected render failure"):
        run_build_one_token_tables(
            BuildOneTokenTablesConfig(runs_root=tmp_path / "producer", output_dir=output),
            inference=InferenceProtocol(bootstrap_draws=4, sign_flip_draws=4),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".tables.*.tmp"))


def test_config_rejects_output_inside_the_input_tree(tmp_path: Path) -> None:
    runs_root = tmp_path / "producer"
    runs_root.mkdir()

    with pytest.raises(ValueError, match="separate"):
        BuildOneTokenTablesConfig(
            runs_root=runs_root,
            output_dir=runs_root / "tables",
        )
