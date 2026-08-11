"""Contracts for the pre-results rebuttal and robustness-training plans."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import typo_cot.cli as cli_module
from typo_cot.experiments.catalog import get_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]

REBUTTAL_COMMANDS = (
    "build-rebuttal-manifest",
    "six-setting-patch-controls",
    "source-write-coordinate-grid",
    "multitoken-kl-readout",
    "patch-harm-audit",
    "tokenization-severity-analysis",
    "subword-position-patching",
    "held-out-window-evaluation",
)

TRAINING_COMMANDS = (
    "build-robustness-training-data",
    "select-distillation-layers",
    "localize-robustness-components",
    "train-noisy-language-model",
    "train-output-matching",
    "train-global-state-alignment",
    "train-localized-state-distillation",
    "evaluate-typo-robustness",
)


def _bash_blocks(markdown: str) -> str:
    return "\n".join(re.findall(r"```bash\n(.*?)\n```", markdown, flags=re.DOTALL))


def test_readmes_freeze_one_descriptive_command_per_planned_operation() -> None:
    root_english = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    root_japanese = (REPOSITORY_ROOT / "README.ja.md").read_text(encoding="utf-8")
    project_english = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    project_japanese = (PROJECT_ROOT / "README.ja.md").read_text(encoding="utf-8")

    for contents in (root_english, root_japanese):
        assert "docs/rebuttal_analysis_plan_v1.md" in contents
        assert "docs/robustness_training_plan_v1.md" in contents

    for contents in (project_english, project_japanese):
        examples = _bash_blocks(contents)
        assert "interface-frozen" in contents
        for command in (*REBUTTAL_COMMANDS, *TRAINING_COMMANDS):
            assert f"typo-cot {command}" in examples

    assert "not yet runnable" in project_english
    assert "`build-rebuttal-manifest` is implemented and CPU-only" in project_english
    assert "`six-setting-patch-controls` is implemented and GPU-only" in project_english
    assert "prose-only pre-implementation label" in project_english
    assert "まだ実行できません" in project_japanese
    assert "`build-rebuttal-manifest` は実装済みのCPU専用コマンド" in project_japanese
    assert "`six-setting-patch-controls` は実装済みのGPU専用コマンド" in project_japanese
    assert "README上の実装前ラベル" in project_japanese

    english_examples = _bash_blocks(project_english)
    japanese_examples = _bash_blocks(project_japanese)
    assert japanese_examples == english_examples
    assert "--selection-run" not in english_examples
    assert "--max-per-setting" not in english_examples
    assert '--cohort-ids "${REBUTTAL_ROOT}/manifest/cohort_ids.json"' in english_examples

    documented_commands = set(
        re.findall(
            r"(?=\btypo-cot\s+([a-z0-9][a-z0-9-]*)\b)",
            english_examples,
        )
    )
    assert set((*REBUTTAL_COMMANDS, *TRAINING_COMMANDS)) <= documented_commands
    for command in documented_commands:
        assert re.search(r"(?:^|-)rq\d+(?:-|$)", command, flags=re.IGNORECASE) is None
        assert re.search(r"(?:^|-)p[01](?:-[a-d])?(?:-|$)", command, flags=re.IGNORECASE) is None

    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    registered = set(subparsers.choices)
    assert registered.intersection(REBUTTAL_COMMANDS) == {
        "build-rebuttal-manifest",
        "six-setting-patch-controls",
    }
    assert registered.isdisjoint(TRAINING_COMMANDS)
    assert get_experiment("build-rebuttal-manifest").status == "implemented"
    assert get_experiment("six-setting-patch-controls").status == "implemented"


def test_rebuttal_plan_freezes_cohorts_arms_statistics_and_claim_rules() -> None:
    path = PROJECT_ROOT / "docs" / "rebuttal_analysis_plan_v1.md"
    plan = path.read_text(encoding="utf-8")
    normalized = " ".join(plan.split())

    for fragment in (
        "800/1,241",
        "172",
        "232",
        "197",
        "209",
        "226",
        "205",
        "common-valid",
        "cyclic derangement",
        "Cochran's Q",
        "exact McNemar",
        "Holm",
        "nested bootstrap",
        "R_{2:16}",
        "right-to-wrong",
        "negative-result claim rule",
        "cohort_ids.json",
        "full clean-correct population",
        "10 percentage points",
        "Holm-adjusted",
    ):
        assert fragment in normalized

    for command in REBUTTAL_COMMANDS:
        assert f"`{command}`" in plan


def test_training_plan_freezes_data_separation_localization_and_pr_gate() -> None:
    path = PROJECT_ROOT / "docs" / "robustness_training_plan_v1.md"
    plan = path.read_text(encoding="utf-8")
    normalized = " ".join(plan.split())

    for fragment in (
        "FineWeb-Edu",
        "Dolma",
        "GitHub Typo Corpus",
        "CoT Collection",
        "unseen content",
        "unseen task",
        "unseen typo",
        "substitution",
        "deletion",
        "insertion",
        "duplication",
        "keyboard-neighbor",
        "layer -> component -> training",
        "MLP neuron",
        "attention head",
        "LoRA",
        "wrong-to-right",
        "right-to-wrong",
        "additional patch gain",
        "three seeds",
        "no training pull request",
        "Adjacent transpositions are excluded from every training and tuning source",
        "1e-6 nats",
        "intentionally stricter",
        "same evaluated checkpoint",
    ):
        assert fragment in normalized

    for command in TRAINING_COMMANDS:
        assert f"`{command}`" in plan
