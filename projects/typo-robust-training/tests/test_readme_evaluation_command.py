"""The public evaluation examples distinguish fresh runs from resume."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TYPO_COT_ROOT = PROJECT_ROOT.parent / "typo-cot"


def _documented_plugin_invocations(readme: str) -> tuple[list[str], ...]:
    invocations: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)\n```", readme, flags=re.DOTALL):
        for statement in block.replace("\\\n", " ").splitlines():
            command = re.search(
                r"(?<![\w/-])typo-cot\s+([a-z0-9][a-z0-9-]*)",
                statement,
            )
            if command is not None:
                invocations.append(shlex.split(statement[command.start() :])[1:])
    return tuple(invocations)


def _plugin_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)
    return parser


def test_documented_invocation_extractor_ignores_project_path_prefix() -> None:
    readme = """```bash
uv run --project projects/typo-cot --locked \\
  typo-cot train-output-matching \\
  --config config.yaml --training-data data --seed 42 --gpu-id 0 \\
  --output-dir output
```"""

    (tokens,) = _documented_plugin_invocations(readme)

    assert tokens[0] == "train-output-matching"
    assert "projects/typo-cot" not in tokens
    with pytest.raises(SystemExit):
        _plugin_parser().parse_args(tokens)


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

    parsed = _plugin_parser().parse_args(tokens)

    assert parsed.command == "evaluate-typo-robustness"
    assert parsed.evaluation_role == "tune"
    assert parsed.splits == ["same-task", "unseen-task", "unseen-content", "unseen-typo"]
    assert parsed.checkpoints
    assert "--data-manifest" not in tail
    assert "--base-model" not in tail
    assert "--checkpoints" not in tail


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_evaluated_checkpoint_has_a_documented_training_producer(
    readme_name: str,
) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    invocations = _documented_plugin_invocations(readme)
    parser = _plugin_parser()

    evaluation_tokens = [
        tokens for tokens in invocations if tokens[0] == "evaluate-typo-robustness"
    ]
    assert len(evaluation_tokens) == 1
    evaluation = parser.parse_args(evaluation_tokens[0])
    assert len(evaluation.checkpoints) == 1
    (checkpoint,) = evaluation.checkpoints
    assert checkpoint.name == "adapter"

    producers = []
    for tokens in invocations:
        if tokens[0] != "train-localized-state-distillation":
            continue
        parsed = parser.parse_args(tokens)
        if parsed.output_dir == checkpoint.parent:
            producers.append(parsed)

    assert len(producers) == 1
    assert producers[0]._training_condition == "localized-state-distillation"
    assert producers[0].wandb_project == "${WANDB_PROJECT}"
    assert producers[0].window_validation is not None


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_cycle3_readme_uses_the_runtime_wandb_role_name(readme_name: str) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")

    assert "`Proposed method`" in readme
    assert "`Causal-window proposal`" not in readme


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_typo_cot_readme_training_catalog_matches_the_installed_plugin_cli(
    readme_name: str,
) -> None:
    readme = (TYPO_COT_ROOT / readme_name).read_text(encoding="utf-8")
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)
    plugin_commands = set(commands.choices)

    documented = [
        tokens
        for tokens in _documented_plugin_invocations(readme)
        if tokens and tokens[0] in plugin_commands
    ]

    assert documented
    for tokens in documented:
        parsed = parser.parse_args(tokens)
        assert parsed.command == tokens[0]
