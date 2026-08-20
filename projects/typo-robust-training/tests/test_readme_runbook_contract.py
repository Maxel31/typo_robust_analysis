"""Falsification tests for the executable README runbook contract."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_NAMES = ("README.md", "README.ja.md")


def _bash_blocks(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL))


def _documented_invocations(text: str) -> tuple[list[str], ...]:
    invocations: list[list[str]] = []
    for block in _bash_blocks(text):
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


@pytest.mark.parametrize("readme_name", README_NAMES)
def test_every_documented_command_parses_with_the_installed_cli(readme_name: str) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    parser = _plugin_parser()
    invocations = _documented_invocations(text)

    assert invocations
    for tokens in invocations:
        parsed = parser.parse_args(tokens)
        assert parsed.command == tokens[0]


@pytest.mark.parametrize("readme_name", README_NAMES)
def test_cycle3_producers_precede_consumers_and_all_comparison_arms(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    parser = _plugin_parser()
    commands = [parser.parse_args(tokens) for tokens in _documented_invocations(text)]

    cycle3_build = next(
        index
        for index, command in enumerate(commands)
        if command.command == "build-robustness-training-data"
        and command.output_dir == Path("${CYCLE3_DATA}")
    )
    evaluation_freeze = next(
        index
        for index, command in enumerate(commands)
        if command.command == "freeze-robustness-evaluation"
        and command.output_dir == Path("${EVALUATION_DATA}")
    )
    localization_freeze = next(
        index
        for index, command in enumerate(commands)
        if command.command == "freeze-generic-localization-pairs"
    )
    localization_select = next(
        index
        for index, command in enumerate(commands)
        if command.command == "select-generic-joint-patch-window"
    )
    localization_validate = next(
        index
        for index, command in enumerate(commands)
        if command.command == "validate-generic-joint-patch-window"
    )
    cycle3_training = [
        (index, command)
        for index, command in enumerate(commands)
        if command.command.startswith("train-") and "configs/cycle3/" in str(command.config)
    ]
    evaluation = next(
        index
        for index, command in enumerate(commands)
        if command.command == "evaluate-typo-robustness"
    )

    assert cycle3_training
    assert (
        cycle3_build
        < evaluation_freeze
        < localization_freeze
        < localization_select
        < localization_validate
        < min(index for index, _command in cycle3_training)
        < evaluation
    )

    frozen_evaluation = commands[evaluation_freeze]
    assert frozen_evaluation.exclude_data == Path("${CYCLE3_DATA}")
    generic_pairs = commands[localization_freeze]
    assert generic_pairs.exclude_data == [
        Path("${CYCLE3_DATA}"),
        Path("${EVALUATION_DATA}"),
    ]

    assert {command._training_condition for _index, command in cycle3_training} == {
        "output-matching",
        "localized-state-distillation",
        "random-window-state-distillation",
        "global-state-alignment",
    }
    for _index, command in cycle3_training:
        assert command.training_data == Path("${CYCLE3_DATA}")
        assert command.monitor_data == Path("${EVALUATION_DATA}")


@pytest.mark.parametrize("readme_name", README_NAMES)
def test_sae_runbook_has_no_ephemeral_paths_and_fails_closed_before_gpu(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    bash = "\n".join(_bash_blocks(text))
    sae = text.split("## 7.", maxsplit=1)[1]
    sae_blocks = _bash_blocks(sae)
    gpu_blocks = [block for block in sae_blocks if "CUDA_VISIBLE_DEVICES=" in block]
    build_block = next(block for block in sae_blocks if "build-sae-clean-corpus" in block)

    assert "/diskthalys/" not in bash
    assert "/tmp/typo-rebuttal-manifest" not in bash
    assert not re.search(r"(?m)^SAE_ROOT=", sae)
    assert not re.search(r"(?m)^ROOTED_REGISTRY=", sae)
    assert len(gpu_blocks) == 3
    for block in gpu_blocks:
        before_gpu = block[: block.index("CUDA_VISIBLE_DEVICES=")]
        for guard in (
            ': "${SAE_ROOT:?Set SAE_ROOT',
            ': "${ROOTED_REGISTRY:?Set ROOTED_REGISTRY',
            'case "${SAE_ROOT}" in /*)',
            'case "${ROOTED_REGISTRY}" in /*)',
            '[ -d "${SAE_ROOT}" ]',
            '[ -f "${ROOTED_REGISTRY}" ]',
            'for REQUIRED_INPUT in "${SAE_TRAINING_DATA}"',
            '[ -f "${REQUIRED_INPUT}" ]',
        ):
            assert guard in before_gpu, block
    for repository_artifact in (
        'SAE_TRAINING_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m/training_sources.jsonl"',
        'SAE_EVALUATION_DATA="${TRAIN_ROOT}/evaluation-data/robustness-v1"',
        'SAE_LOCALIZATION_DATA="${TRAIN_ROOT}/localization/generic-joint-window-v1/pairs"',
    ):
        assert repository_artifact in build_block
    assert "separately reviewed" in sae or "別途レビュー済み" in sae


@pytest.mark.parametrize("readme_name", README_NAMES)
def test_residual_window_artifact_reference_names_its_producer_section(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    sections = re.findall(
        r"(?ms)^## (\d+)\. (.*?)(?=^## \d+\.|\Z)",
        text,
    )
    producer_section = next(
        section_number
        for section_number, section_body in sections
        if "select-generic-joint-patch-window" in section_body
    )
    artifact_references = re.findall(r"Section (\d+)(?: artifacts|のartifact)", text)

    assert artifact_references == [producer_section]


@pytest.mark.parametrize("readme_name", README_NAMES)
def test_every_documented_bash_block_is_syntactically_valid(readme_name: str) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")

    for block in _bash_blocks(text):
        result = subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
