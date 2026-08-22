"""Falsification tests for the executable README runbook contract."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.data.config import load_training_data_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_NAMES = ("README.md", "README.ja.md")
_CYCLE3_RUNBOOK_CONFIG = Path("${TRAIN_PROJECT}/configs/cycle3/gemma4b-data-64m.yaml")
_TRAIN_PROJECT_PLACEHOLDER = Path("${TRAIN_PROJECT}")
SEED_INVENTORY_PROMISES = {
    "README.md": (
        "The later bounded residual-window training feature supplies proposed-method "
        "seeds 42, 43, and 44; its matched output-only baseline and scope controls "
        "remain frozen at seed 42."
    ),
    "README.ja.md": (
        "後続の有界な residual-window学習機能では提案手法をseed 42、43、44で実行し、"
        "matched output-only baselineと scope controlはseed 42に固定します。"
    ),
}


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


def _assert_cycle3_materialization_contract(
    commands: list[argparse.Namespace],
) -> int:
    producers = [
        (index, command)
        for index, command in enumerate(commands)
        if command.command == "build-robustness-training-data"
        and command.output_dir == Path("${CYCLE3_DATA}")
    ]
    assert len(producers) == 1, "runbook must document exactly one ${CYCLE3_DATA} producer"
    index, producer = producers[0]
    assert producer.config == _CYCLE3_RUNBOOK_CONFIG, (
        "${CYCLE3_DATA} producer must use the frozen Cycle 3 64M config"
    )
    resolved_config = PROJECT_ROOT / producer.config.relative_to(_TRAIN_PROJECT_PLACEHOLDER)
    protocol = load_training_data_config(resolved_config)
    assert protocol.training_token_budget == 64_000_000, (
        "documented Cycle 3 producer config must freeze a 64M-token budget"
    )
    return index


def _normalized_prose(text: str) -> str:
    return " ".join(text.split())


def _assert_seed_inventory_promise(readme_name: str, text: str) -> None:
    assert _normalized_prose(SEED_INVENTORY_PROMISES[readme_name]) in _normalized_prose(text)


def _cycle3_seed_inventory(text: str) -> dict[str, set[int]]:
    parser = _plugin_parser()
    inventory: dict[str, set[int]] = {}
    for tokens in _documented_invocations(text):
        command = parser.parse_args(tokens)
        if not command.command.startswith("train-") or "configs/cycle3/" not in str(command.config):
            continue
        inventory.setdefault(command._training_condition, set()).add(command.seed)
    return inventory


def _assert_sae_assignments_are_portable(text: str) -> None:
    sae_section = text.split("## 7.", maxsplit=1)[1]
    assignments = re.findall(r"(?m)^(SAE_[A-Z0-9_]+)=(.+)$", sae_section)
    assert assignments
    safe_literals = {
        "SAE_GPU_ID": '"0"',
        "SAE_WANDB_PROJECT": '"typo-robustness-sae"',
    }
    rooted_value = re.compile(r'^"\$\{(?:SAE_ROOT|TRAIN_ROOT|TRAIN_PROJECT)\}(?:/[^"\n]*)?"$')
    for name, value in assignments:
        if name in safe_literals:
            assert value == safe_literals[name]
            continue
        assert rooted_value.fullmatch(value), (
            f"{name} must be rooted in SAE_ROOT, TRAIN_ROOT, or TRAIN_PROJECT: {value}"
        )
        assert "/../" not in value and not value.endswith('/.."')


def _assert_bilingual_executable_blocks_match(english: str, japanese: str) -> None:
    assert _bash_blocks(english) == _bash_blocks(japanese)


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
def test_confirmatory_seed_inventory_promise_matches_the_runbook(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")

    _assert_seed_inventory_promise(readme_name, text)
    assert _cycle3_seed_inventory(text) == {
        "output-matching": {42},
        "localized-state-distillation": {42, 43, 44},
        "random-window-state-distillation": {42},
        "global-state-alignment": {42},
    }


@pytest.mark.parametrize("readme_name", README_NAMES)
def test_seed_inventory_contract_rejects_a_three_seed_baseline_control_promise(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    replacements = {
        "README.md": (
            "baseline and scope controls remain frozen at seed 42",
            "baseline and scope controls run at seeds 42, 43, and 44",
        ),
        "README.ja.md": (
            "baselineと\nscope controlはseed 42に固定します",
            "baselineと\nscope controlもseed 42、43、44で実行します",
        ),
    }
    correct, contradictory = replacements[readme_name]
    mutated = text.replace(correct, contradictory, 1)

    assert mutated != text
    assert contradictory in mutated
    with pytest.raises(AssertionError):
        _assert_seed_inventory_promise(readme_name, mutated)


def test_english_and_japanese_executable_blocks_are_identical() -> None:
    english = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (PROJECT_ROOT / "README.ja.md").read_text(encoding="utf-8")

    _assert_bilingual_executable_blocks_match(english, japanese)


def test_bilingual_contract_rejects_a_japanese_only_training_seed_change() -> None:
    english = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (PROJECT_ROOT / "README.ja.md").read_text(encoding="utf-8")
    all_layer_block = next(
        block
        for block in _bash_blocks(japanese)
        if "configs/cycle3/gemma4b-all-layer-state-10m.yaml" in block
    )
    mutated_block = all_layer_block.replace("--seed 42", "--seed 43", 1)
    mutated_japanese = japanese.replace(all_layer_block, mutated_block, 1)

    assert mutated_block != all_layer_block
    assert "--seed 43" in mutated_block
    assert "all-layer-state-distillation/seed-42" in mutated_block
    with pytest.raises(AssertionError):
        _assert_bilingual_executable_blocks_match(english, mutated_japanese)


@pytest.mark.parametrize("readme_name", README_NAMES)
def test_cycle3_producers_precede_consumers_and_all_comparison_arms(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    parser = _plugin_parser()
    commands = [parser.parse_args(tokens) for tokens in _documented_invocations(text)]

    cycle3_build = _assert_cycle3_materialization_contract(commands)
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
def test_cycle3_materialization_contract_rejects_the_sanity_config(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    cycle3_block = next(
        block
        for block in _bash_blocks(text)
        if "typo-cot build-robustness-training-data" in block
        and '--output-dir "${CYCLE3_DATA}"' in block
    )
    correct = '--config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-data-64m.yaml"'
    incorrect = '--config "${TRAIN_PROJECT}/configs/gemma4b-sanity.yaml"'
    mutated_block = cycle3_block.replace(correct, incorrect, 1)
    mutated = text.replace(cycle3_block, mutated_block, 1)

    assert mutated_block != cycle3_block
    parser = _plugin_parser()
    commands = [parser.parse_args(tokens) for tokens in _documented_invocations(mutated)]
    with pytest.raises(
        AssertionError,
        match="must use the frozen Cycle 3 64M config",
    ):
        _assert_cycle3_materialization_contract(commands)


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
    _assert_sae_assignments_are_portable(text)
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
def test_sae_portability_contract_rejects_an_ephemeral_absolute_assignment(
    readme_name: str,
) -> None:
    text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    calibration_block = next(
        block
        for block in _bash_blocks(text.split("## 7.", maxsplit=1)[1])
        if "calibrate-sparse-autoencoder-l1" in block
    )
    portable = 'SAE_TRAINING_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m/training_sources.jsonl"'
    ephemeral = (
        'SAE_TRAINING_DATA="/diskfoo/ssd14tc/operator/typo_artifacts/repo/'
        "projects/typo-robust-training/results/data/gemma4b-cycle3-64m/"
        'training_sources.jsonl"'
    )
    mutated_block = calibration_block.replace(portable, ephemeral, 1)
    mutated = text.replace(calibration_block, mutated_block, 1)

    assert mutated_block != calibration_block
    assert ephemeral in mutated
    with pytest.raises(AssertionError, match="must be rooted in"):
        _assert_sae_assignments_are_portable(mutated)


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
