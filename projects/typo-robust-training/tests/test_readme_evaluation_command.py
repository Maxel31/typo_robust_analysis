"""The public evaluation examples distinguish fresh runs from resume."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

import typo_cot.cli as typo_cot_cli
from typo_robust_training.cli import register_commands
from typo_robust_training.evaluation.checkpoints import AdapterDescriptor, PatchWindow
from typo_robust_training.evaluation.config import load_robustness_evaluation_config
from typo_robust_training.evaluation.runner import _validate_injected_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TYPO_COT_ROOT = PROJECT_ROOT.parent / "typo-cot"

CORE_COMMANDS = frozenset(
    {
        "answer-line-deletion",
        "build-input-corrector-summary",
        "build-one-token-tables",
        "build-rebuttal-manifest",
        "build-restoration-order-table",
        "build-typo-warning-summary",
        "clean-prefix-scan",
        "cot-swap",
        "edit-count-sensitivity",
        "experiments",
        "fixed-window-answer-patching",
        "held-out-window-evaluation",
        "input-corrector-audit",
        "layerwise-answer-patching",
        "layerwise-kl-patching",
        "model-scale-cot-swap",
        "multitoken-kl-readout",
        "one-token-prefix-replacement",
        "patch-coordinate-controls",
        "patch-harm-audit",
        "patch-position-controls",
        "patch-text-combination",
        "prepare-edited-pairs",
        "restoration-order-accuracy",
        "six-setting-patch-controls",
        "source-write-coordinate-grid",
        "subword-position-patching",
        "targeting-fidelity-audit",
        "tokenization-severity-analysis",
        "typo-warning-prompt",
    }
)

TRAINING_PLUGIN_COMMANDS = frozenset(
    {
        "prepare-kojima-faithful-data",
        "build-robustness-training-data",
        "build-probe-transition-data",
        "build-sae-clean-corpus",
        "calibrate-evaluation-v2-severity",
        "calibrate-sparse-autoencoder-l1",
        "evaluate-typo-robustness",
        "freeze-generic-localization-pairs",
        "freeze-robustness-evaluation",
        "freeze-tokenizer-attestation",
        "localize-robustness-components",
        "materialize-probe-output-factorial-configs",
        "materialize-probe-transition-state-training-config",
        "materialize-probe-semantic-training-config",
        "materialize-probe-transition-training-config",
        "run-probe-semantic-subspace-kill-test",
        "select-distillation-layers",
        "select-generic-joint-patch-window",
        "select-probe-transition",
        "train-global-state-alignment",
        "train-factorial-all-layers-all-tokens",
        "train-factorial-all-layers-downstream-horizon",
        "train-factorial-probe-suffix-all-tokens",
        "train-factorial-probe-suffix-downstream-horizon",
        "train-factorial-random-layers-downstream-horizon",
        "train-kojima-faithful-output-matching",
        "train-localized-state-distillation",
        "train-noisy-language-model",
        "train-output-matching",
        "train-probe-semantic-subspace-distillation",
        "train-probe-transition-output-matching",
        "train-probe-transition-single-layer-state-distillation",
        "train-random-window-state-distillation",
        "train-sparse-autoencoders",
        "validate-generic-joint-patch-window",
        "validate-probe-transition-single-layer-gate",
        "validate-sparse-autoencoders",
    }
)

DOCUMENTED_TRAINING_COMMANDS = frozenset(
    {
        "build-robustness-training-data",
        "evaluate-typo-robustness",
        "localize-robustness-components",
        "select-distillation-layers",
        "train-global-state-alignment",
        "train-localized-state-distillation",
        "train-noisy-language-model",
        "train-output-matching",
    }
)


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


def _evaluation_producers_by_role(readme: str) -> dict[str, dict[Path, argparse.Namespace]]:
    parser = _plugin_parser()
    commands = [parser.parse_args(tokens) for tokens in _documented_plugin_invocations(readme)]
    training_commands = [
        (index, command)
        for index, command in enumerate(commands)
        if command.command.startswith("train-")
    ]
    producers_by_role: dict[str, dict[Path, argparse.Namespace]] = {}

    for consumer_index, evaluation in enumerate(commands):
        if evaluation.command != "evaluate-typo-robustness":
            continue

        role = evaluation.evaluation_role
        assert role not in producers_by_role, f"duplicate evaluation role: {role}"
        producers: dict[Path, argparse.Namespace] = {}
        for checkpoint in evaluation.checkpoints:
            matching = [
                (producer_index, producer)
                for producer_index, producer in training_commands
                if producer.output_dir == checkpoint.parent
            ]
            assert matching, f"{role} checkpoint has no documented producer: {checkpoint}"
            assert len(matching) == 1, (
                f"{role} checkpoint has ambiguous documented producers: {checkpoint}"
            )
            producer_index, producer = matching[0]
            assert producer_index < consumer_index, (
                f"{role} checkpoint is consumed before its producer: {checkpoint}"
            )
            producers[checkpoint] = producer
        producers_by_role[role] = producers

    assert producers_by_role
    return producers_by_role


def _move_documented_command_after_block(
    readme: str,
    *,
    producer_marker: str,
    consumer_marker: str,
) -> str:
    producer_marker_index = readme.index(producer_marker)
    producer_start = readme.rfind("CUDA_VISIBLE_DEVICES=", 0, producer_marker_index)
    producer_output = readme.index("--output-dir", producer_marker_index)
    producer_end = readme.index("\n", producer_output)
    producer_command = readme[producer_start:producer_end]
    without_producer = readme[:producer_start] + readme[producer_end + 1 :]
    consumer = next(
        match
        for match in re.finditer(r"```bash\n.*?\n```", without_producer, flags=re.DOTALL)
        if consumer_marker in match.group()
    )
    return (
        without_producer[: consumer.end()]
        + "\n\n```bash\n"
        + producer_command
        + "\n```"
        + without_producer[consumer.end() :]
    )


def test_documented_invocation_extractor_ignores_project_path_prefix() -> None:
    readme = """```bash
uv run --project projects/typo-cot \\
  typo-cot train-output-matching \\
  --config config.yaml --training-data data --seed 42 --gpu-id 0 \\
  --output-dir output
```"""

    (tokens,) = _documented_plugin_invocations(readme)

    assert tokens[0] == "train-output-matching"
    assert "projects/typo-cot" not in tokens
    with pytest.raises(SystemExit):
        _plugin_parser().parse_args(tokens)


def test_real_cli_registers_the_exact_core_and_training_plugin_commands() -> None:
    plugin_parser = _plugin_parser()
    plugin_subparsers = next(
        action
        for action in plugin_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    plugin_commands = set(plugin_subparsers.choices)

    combined_parser = typo_cot_cli._parser()
    combined_subparsers = next(
        action
        for action in combined_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    combined_commands = set(combined_subparsers.choices)

    assert plugin_commands == TRAINING_PLUGIN_COMMANDS
    assert CORE_COMMANDS.isdisjoint(TRAINING_PLUGIN_COMMANDS)
    assert combined_commands - plugin_commands == CORE_COMMANDS
    assert combined_commands == CORE_COMMANDS | TRAINING_PLUGIN_COMMANDS
    assert len(combined_commands) == 66


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_initial_evaluation_command_does_not_request_resume(readme_name: str) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    command = readme.split("typo-cot evaluate-typo-robustness", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]

    assert "--resume" not in command
    assert "--evaluation-protocol" in command
    assert "--evaluation-data" in command
    assert '--training-data "${CYCLE3_DATA}"' in command
    assert 'CYCLE3_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m"' in readme
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
def test_every_evaluated_checkpoint_has_one_preceding_training_producer(
    readme_name: str,
) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    producers_by_role = _evaluation_producers_by_role(readme)

    assert set(producers_by_role) == {"tune", "pre-pr-gate"}
    assert len(producers_by_role["tune"]) == 4
    assert len(producers_by_role["pre-pr-gate"]) == 6
    assert all(
        checkpoint.name == "adapter"
        for producers in producers_by_role.values()
        for checkpoint in producers
    )
    assert {
        producer._training_condition
        for producers in producers_by_role.values()
        for producer in producers.values()
    } == {
        "output-matching",
        "localized-state-distillation",
        "random-window-state-distillation",
        "global-state-alignment",
    }
    assert all(
        producer.wandb_project == "${WANDB_PROJECT}"
        for producers in producers_by_role.values()
        for producer in producers.values()
    )
    localized = [
        producer
        for producer in producers_by_role["tune"].values()
        if producer._training_condition == "localized-state-distillation"
    ]
    assert len(localized) == 1
    assert localized[0].window_validation is not None


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_producer_order_contract_rejects_a_training_block_after_its_consumer(
    readme_name: str,
) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    mutated = _move_documented_command_after_block(
        readme,
        producer_marker="configs/cycle3/gemma4b-all-layer-state-10m.yaml",
        consumer_marker="--evaluation-role tune",
    )
    parsed = [
        _plugin_parser().parse_args(tokens) for tokens in _documented_plugin_invocations(mutated)
    ]
    producer_index = next(
        index
        for index, command in enumerate(parsed)
        if command.command == "train-global-state-alignment"
        and "configs/cycle3/gemma4b-all-layer-state-10m.yaml" in str(command.config)
    )
    consumer_index = next(
        index
        for index, command in enumerate(parsed)
        if command.command == "evaluate-typo-robustness" and command.evaluation_role == "tune"
    )
    assert producer_index > consumer_index

    with pytest.raises(AssertionError, match="consumed before its producer"):
        _evaluation_producers_by_role(mutated)


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_pre_pr_contract_rejects_a_checkpoint_without_a_training_producer(
    readme_name: str,
) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    pre_pr_block = next(
        block
        for block in re.findall(r"```bash\n(.*?)\n```", readme, flags=re.DOTALL)
        if "--evaluation-role pre-pr-gate" in block
    )
    mutated_block = pre_pr_block.replace(
        '  --checkpoint "${TRAIN_ROOT}/training/cycle3/output-matching/seed-42/adapter" \\\n',
        '  --checkpoint "${TRAIN_ROOT}/training/cycle3/output-matching/seed-42/adapter" \\\n'
        '  --checkpoint "${TRAIN_ROOT}/training/cycle3/output-matching/seed-43/adapter" \\\n',
        1,
    )
    assert mutated_block != pre_pr_block
    mutated = readme.replace(pre_pr_block, mutated_block, 1)
    bogus_checkpoint = Path("${TRAIN_ROOT}/training/cycle3/output-matching/seed-43/adapter")
    parsed = [
        _plugin_parser().parse_args(tokens) for tokens in _documented_plugin_invocations(mutated)
    ]
    pre_pr = next(
        command
        for command in parsed
        if command.command == "evaluate-typo-robustness"
        and command.evaluation_role == "pre-pr-gate"
    )
    assert bogus_checkpoint in pre_pr.checkpoints
    assert not any(
        command.command.startswith("train-") and command.output_dir == bogus_checkpoint.parent
        for command in parsed
    )

    with pytest.raises(AssertionError, match="has no documented producer"):
        _evaluation_producers_by_role(mutated)


def _descriptor_for_documented_checkpoint(path: Path) -> AdapterDescriptor:
    checkpoint = str(path)
    condition_by_fragment = {
        "/output-matching/": "output-matching",
        "/causal-window-state-distillation/": "localized-state-distillation",
        "/random-window-state-distillation/": "random-window-state-distillation",
        "/all-layer-state-distillation/": "global-state-alignment",
    }
    condition = next(
        value for fragment, value in condition_by_fragment.items() if fragment in checkpoint
    )
    seed_match = re.search(r"/seed-(42|43|44)/", checkpoint)
    assert seed_match is not None
    localization_sha256 = (
        "f" * 64
        if condition in {"localized-state-distillation", "random-window-state-distillation"}
        else None
    )
    config_sha256 = {
        "output-matching": "1" * 64,
        "localized-state-distillation": "2" * 64,
        "random-window-state-distillation": "3" * 64,
        "global-state-alignment": "4" * 64,
    }[condition]
    return AdapterDescriptor(
        path=path,
        condition=condition,
        seed=int(seed_match.group(1)),
        config_sha256=config_sha256,
        training_data_sha256="a" * 64,
        data_identity_sha256="b" * 64,
        localization_sha256=localization_sha256,
        adapter_sha256="c" * 64,
    )


@pytest.mark.parametrize("readme_name", ["README.md", "README.ja.md"])
def test_pre_pr_evaluation_is_documented_after_the_complete_proposal_seed_inventory(
    readme_name: str,
) -> None:
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    invocations = _documented_plugin_invocations(readme)
    parser = _plugin_parser()
    parsed = [parser.parse_args(tokens) for tokens in invocations]
    protocol = load_robustness_evaluation_config(PROJECT_ROOT / "configs/gemma4b-evaluation.yaml")

    tune_index = next(
        index
        for index, command in enumerate(parsed)
        if command.command == "evaluate-typo-robustness" and command.evaluation_role == "tune"
    )
    pre_pr_index = next(
        index
        for index, command in enumerate(parsed)
        if command.command == "evaluate-typo-robustness"
        and command.evaluation_role == "pre-pr-gate"
    )
    proposal_training = {
        command.seed: index
        for index, command in enumerate(parsed)
        if command.command == "train-localized-state-distillation"
        and "configs/cycle3/gemma4b-causal-window-10m.yaml" in str(command.config)
    }

    assert set(proposal_training) == set(protocol.seed_inventory)
    assert tune_index < proposal_training[43] < pre_pr_index
    assert tune_index < proposal_training[44] < pre_pr_index

    pre_pr = parsed[pre_pr_index]
    descriptors = tuple(_descriptor_for_documented_checkpoint(path) for path in pre_pr.checkpoints)
    localized_seeds = {
        descriptor.seed
        for descriptor in descriptors
        if descriptor.condition == "localized-state-distillation"
    }
    assert localized_seeds == set(protocol.seed_inventory)

    _validate_injected_inputs(
        SimpleNamespace(
            checkpoint_paths=pre_pr.checkpoints,
            evaluation_role=pre_pr.evaluation_role,
        ),
        protocol=protocol,
        data_bundle=None,
        corpus_bundle=None,
        descriptors=descriptors,
        patch_window=PatchWindow(
            start=0,
            stop=6,
            artifact_sha256="9" * 64,
            localization_sha256="f" * 64,
        ),
    )


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

    assert len(documented) == len(DOCUMENTED_TRAINING_COMMANDS)
    assert {tokens[0] for tokens in documented} == DOCUMENTED_TRAINING_COMMANDS
    for tokens in documented:
        parsed = parser.parse_args(tokens)
        assert parsed.command == tokens[0]
