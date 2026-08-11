"""Contracts for the pre-results rebuttal and robustness-training plans."""

from __future__ import annotations

import re
from pathlib import Path


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
        assert "not yet runnable" in project_english
        assert "まだ実行できません" in project_japanese
        for command in (*REBUTTAL_COMMANDS, *TRAINING_COMMANDS):
            assert f"typo-cot {command}" in examples

    for command in (*REBUTTAL_COMMANDS, *TRAINING_COMMANDS):
        assert re.search(r"(?:^|-)rq\d+(?:-|$)", command, flags=re.IGNORECASE) is None
        assert re.search(r"(?:^|-)p[01](?:-[a-d])?(?:-|$)", command, flags=re.IGNORECASE) is None


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
    ):
        assert fragment in normalized

    for command in TRAINING_COMMANDS:
        assert f"`{command}`" in plan
