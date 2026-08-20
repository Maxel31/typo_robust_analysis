"""The public evaluation examples distinguish fresh runs from resume."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TYPO_COT_ROOT = PROJECT_ROOT.parent / "typo-cot"


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_initial_evaluation_command_does_not_request_resume(readme_name: str) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    command = readme.split("typo-cot evaluate-typo-robustness", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]

    assert "--resume" not in command
    assert "--evaluation-protocol" in command
    assert "--evaluation-data" in command
    assert "gemma4b-cycle3-64m" in command
    assert "--window-validation" in command
    assert "--resume" in readme.split(command, maxsplit=1)[1]


@pytest.mark.parametrize(
    ("project_root", "readme_name"),
    [
        (PROJECT_ROOT, "README.md"),
        (PROJECT_ROOT, "README.ja.md"),
        (TYPO_COT_ROOT, "README.md"),
        (TYPO_COT_ROOT, "README.ja.md"),
    ],
)
def test_documented_evaluation_command_matches_the_installed_plugin_cli(
    project_root: Path,
    readme_name: str,
) -> None:
    readme = (project_root / readme_name).read_text(encoding="utf-8")
    tail = readme.split("typo-cot evaluate-typo-robustness", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]
    tokens = shlex.split(("evaluate-typo-robustness" + tail).replace("\\\n", " "))

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)
    parsed = parser.parse_args(tokens)

    assert parsed.command == "evaluate-typo-robustness"
    assert parsed.evaluation_role == "tune"
    assert parsed.splits == ["same-task", "unseen-task", "unseen-content", "unseen-typo"]
    assert parsed.checkpoints
    assert "--data-manifest" not in tail
    assert "--base-model" not in tail
    assert "--checkpoints" not in tail
