"""Contract tests for the public, paper-aligned experiment catalog."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from typo_cot.cli import main
from typo_cot.experiments.catalog import (
    PAPER_EXPERIMENTS,
    PAPER_SHA256,
    ExperimentSpec,
    get_experiment,
)


EXPECTED_EXPERIMENTS = (
    "prepare-edited-pairs",
    "targeting-fidelity-audit",
    "layerwise-kl-patching",
    "layerwise-answer-patching",
    "fixed-window-answer-patching",
    "patch-coordinate-controls",
    "patch-position-controls",
    "patch-text-combination",
    "cot-swap",
    "answer-line-deletion",
    "clean-prefix-scan",
    "one-token-prefix-replacement",
    "edit-count-sensitivity",
    "model-scale-cot-swap",
    "typo-warning-prompt",
    "input-corrector-audit",
    "restoration-order-accuracy",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_catalog_identifies_the_user_supplied_final_pdf() -> None:
    assert PAPER_SHA256 == "2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0"


def test_catalog_covers_the_final_paper_experiments_in_reproduction_order() -> None:
    assert tuple(spec.slug for spec in PAPER_EXPERIMENTS) == EXPECTED_EXPERIMENTS


@pytest.mark.parametrize("spec", PAPER_EXPERIMENTS, ids=lambda spec: spec.slug)
def test_catalog_entries_are_public_operation_contracts(spec: ExperimentSpec) -> None:
    assert re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", spec.slug)
    assert not re.search(r"(?:^|-)rq[123](?:-|$)", spec.slug)
    assert spec.summary
    assert spec.paper_sections
    assert spec.required_arguments
    assert spec.outputs
    assert spec.compute in {"cpu", "gpu"}
    assert spec.status in {"catalogued", "implemented"}


def test_catalog_slugs_are_unique() -> None:
    slugs = [spec.slug for spec in PAPER_EXPERIMENTS]
    assert len(slugs) == len(set(slugs))


def test_get_experiment_rejects_unknown_operation() -> None:
    with pytest.raises(KeyError, match="unknown experiment"):
        get_experiment("rq1")


def test_cli_lists_machine_readable_experiment_contracts(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiments", "list", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [item["slug"] for item in payload] == list(EXPECTED_EXPERIMENTS)
    assert all(item["command"].startswith("uv run typo-cot ") for item in payload)


def test_cli_shows_experiment_specific_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiments", "show", "clean-prefix-scan", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["paper_question"] == "RQ3"
    assert payload["command"] == "uv run typo-cot clean-prefix-scan"
    assert "--target-set" in payload["required_arguments"]
    assert "--output-dir" in payload["required_arguments"]


def test_cli_returns_usage_error_for_unknown_experiment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["experiments", "show", "rq1"])

    assert exc_info.value.code == 2
    assert "unknown experiment" in capsys.readouterr().err


def test_public_experiment_guide_tracks_every_catalog_operation() -> None:
    guide = (PROJECT_ROOT / "docs" / "paper-experiments.md").read_text(encoding="utf-8")

    assert PAPER_SHA256 in guide
    for slug in EXPECTED_EXPERIMENTS:
        assert f"`{slug}`" in guide
