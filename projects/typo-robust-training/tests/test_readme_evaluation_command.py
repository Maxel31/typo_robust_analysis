"""The public evaluation examples distinguish fresh runs from resume."""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_initial_evaluation_command_does_not_request_resume(readme_name: str) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    command = readme.split("typo-cot evaluate-typo-robustness", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]

    assert "--resume" not in command
    assert "--resume" in readme.split(command, maxsplit=1)[1]
