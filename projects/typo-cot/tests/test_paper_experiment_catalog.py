"""Contract tests for the public, paper-aligned experiment catalog."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import replace
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
import typo_cot.experiments as experiments_api
from typo_cot.cli import main
from typo_cot.experiments.catalog import (
    _PAPER_QUESTION_SLUG_PATTERN,
    _PUBLIC_SLUG_PATTERN,
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
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def _documented_target_commands(markdown: str) -> dict[str, set[str]]:
    """Parse continued ``uv run [options] typo-cot`` commands from bash fences."""
    commands: dict[str, set[str]] = {}
    for block in re.findall(r"```bash\n(.*?)\n```", markdown, flags=re.DOTALL):
        continued: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            starts_command = stripped.startswith("uv run ") and " typo-cot " in stripped
            if continued or starts_command:
                continued.append(stripped.removesuffix("\\").rstrip())
                if stripped.endswith("\\"):
                    continue

                tokens = shlex.split(" ".join(continued))
                continued = []
                if tokens[:2] != ["uv", "run"] or "typo-cot" not in tokens[2:]:
                    continue
                executable_index = tokens.index("typo-cot", 2)
                if executable_index + 1 >= len(tokens):
                    continue
                slug = tokens[executable_index + 1]
                assert slug not in commands, f"duplicate command example: {slug}"
                commands[slug] = set(tokens)

        assert not continued, "unterminated shell continuation in documentation"
    return commands


def test_catalog_identifies_the_user_supplied_final_pdf() -> None:
    assert PAPER_SHA256 == "2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0"


def test_experiments_package_exports_the_canonical_paper_fingerprint() -> None:
    assert experiments_api.PAPER_SHA256 == PAPER_SHA256


def test_catalog_covers_the_final_paper_experiments_in_reproduction_order() -> None:
    assert tuple(spec.slug for spec in PAPER_EXPERIMENTS) == EXPECTED_EXPERIMENTS


@pytest.mark.parametrize("spec", PAPER_EXPERIMENTS, ids=lambda spec: spec.slug)
def test_catalog_entries_are_public_operation_contracts(spec: ExperimentSpec) -> None:
    assert _PUBLIC_SLUG_PATTERN.fullmatch(spec.slug)
    assert not _PAPER_QUESTION_SLUG_PATTERN.search(spec.slug)
    assert spec.summary
    assert spec.paper_sections
    assert spec.required_arguments
    assert spec.outputs
    assert spec.compute in {"cpu", "gpu"}
    assert spec.status in {"catalogued", "implemented"}


def test_catalog_slugs_are_unique() -> None:
    slugs = [spec.slug for spec in PAPER_EXPERIMENTS]
    assert len(slugs) == len(set(slugs))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (("compute", "cup"), ("status", "in-progress")),
)
def test_experiment_spec_rejects_values_outside_the_public_contract(
    field: str, invalid_value: str
) -> None:
    with pytest.raises(ValueError, match=rf"{field}.*{re.escape(invalid_value)}"):
        replace(PAPER_EXPERIMENTS[0], **{field: invalid_value})


@pytest.mark.parametrize(
    "invalid_slug",
    ("Clean-Prefix-Scan", "clean_prefix_scan", "rq1-prefix-scan", "rq4-future-scan"),
)
def test_experiment_spec_rejects_non_public_operation_slugs(invalid_slug: str) -> None:
    with pytest.raises(ValueError, match=rf"slug.*{re.escape(invalid_slug)}"):
        replace(PAPER_EXPERIMENTS[0], slug=invalid_slug)


def test_get_experiment_rejects_unknown_operation() -> None:
    with pytest.raises(KeyError, match="unknown experiment"):
        get_experiment("rq1")


def test_cli_lists_machine_readable_experiment_contracts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["experiments", "list", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [item["slug"] for item in payload] == list(EXPECTED_EXPERIMENTS)
    assert all(item["target_command"].startswith("uv run typo-cot ") for item in payload)
    assert all("command" not in item for item in payload)


def test_cli_reports_the_canonical_paper_fingerprint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["experiments", "source", "--format", "json"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "sha256": PAPER_SHA256,
    }


def test_cli_text_list_handles_an_empty_catalog(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "PAPER_EXPERIMENTS", ())

    assert main(["experiments", "list"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_shows_experiment_specific_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiments", "show", "clean-prefix-scan", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["paper_question"] == "RQ3"
    assert payload["target_command"] == "uv run typo-cot clean-prefix-scan"
    assert "--target-set" in payload["required_arguments"]
    assert "--output-dir" in payload["required_arguments"]


def test_cli_text_show_includes_the_operation_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = get_experiment("clean-prefix-scan")

    assert main(["experiments", "show", spec.slug]) == 0

    assert spec.summary in capsys.readouterr().out


def test_cli_returns_usage_error_for_unknown_experiment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["experiments", "show", "rq1"])

    assert exc_info.value.code == 2
    assert "unknown experiment" in capsys.readouterr().err


def test_cli_text_list_aligns_catalogued_and_implemented_statuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    specs = (
        PAPER_EXPERIMENTS[0],
        replace(PAPER_EXPERIMENTS[1], status="implemented"),
    )
    monkeypatch.setattr(cli_module, "PAPER_EXPERIMENTS", specs)

    assert main(["experiments", "list"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert [line.index(spec.title) for line, spec in zip(lines, specs, strict=True)] == [
        len(lines[0]) - len(specs[0].title),
        len(lines[1]) - len(specs[1].title),
    ]
    assert lines[0].index(specs[0].title) == lines[1].index(specs[1].title)


def test_public_experiment_guide_tracks_every_catalog_operation() -> None:
    guide = (PROJECT_ROOT / "docs" / "paper-experiments.md").read_text(encoding="utf-8")
    commands = _documented_target_commands(guide)

    assert "typo-cot experiments source" in guide
    assert set(commands) == set(EXPECTED_EXPERIMENTS)
    for spec in PAPER_EXPERIMENTS:
        assert f"`{spec.slug}`" in guide
        for argument in spec.required_arguments:
            assert argument in commands[spec.slug], f"{spec.slug} example is missing {argument}"


def test_all_public_entry_points_read_the_pdf_fingerprint_from_the_catalog() -> None:
    public_documents = (
        REPOSITORY_ROOT / "README.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "paper-experiments.md",
    )

    for document in public_documents:
        contents = document.read_text(encoding="utf-8")
        assert PAPER_SHA256 not in contents, document
        assert "typo-cot experiments source" in contents, document


def test_relocated_legacy_readme_relative_links_resolve() -> None:
    readme = PROJECT_ROOT / "docs" / "legacy-development-readme.md"
    markdown = readme.read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown)

    for target in targets:
        if target.startswith(("#", "http://", "https://")):
            continue
        relative_path = target.partition("#")[0]
        assert (readme.parent / relative_path).exists(), target


def test_root_readme_accounts_for_retained_workspace_support() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "sha256sum" in readme
    for retained_path in (
        "datasets/",
        "utils/",
        "_sample_project/",
        "scripts/new_project.sh",
        ".github/workflows/claude-code-review.yml",
    ):
        assert retained_path in readme
