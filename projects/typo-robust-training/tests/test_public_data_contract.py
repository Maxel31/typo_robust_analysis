"""Public command, configuration, and dataset-role contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from typo_robust_training.data.config import load_training_data_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-sanity.yaml"


def _bash_blocks(path: Path) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"```bash\n(.*?)\n```",
            path.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
    )


def test_readmes_freeze_matching_operation_commands_and_artifacts() -> None:
    english = PROJECT_ROOT / "README.md"
    japanese = PROJECT_ROOT / "README.ja.md"
    assert _bash_blocks(english) == _bash_blocks(japanese)

    contents = english.read_text(encoding="utf-8")
    for command in (
        "build-robustness-training-data",
        "select-distillation-layers",
        "localize-robustness-components",
        "train-noisy-language-model",
        "train-output-matching",
        "train-global-state-alignment",
        "train-localized-state-distillation",
        "evaluate-typo-robustness",
    ):
        assert f"typo-cot {command}" in "\n".join(_bash_blocks(english))
    for artifact in (
        "training_sources.jsonl",
        "diagnostic_manifest.jsonl",
        "tune_manifest.jsonl",
        "pre_pr_gate_manifest.jsonl",
        "final_test_manifest.jsonl",
        "evaluation_manifest.json",
        "decontamination_report.json",
        "run.json",
    ):
        assert artifact in contents
    assert "GPU_ID=3" in contents
    assert "no training pull request" in contents


def test_sanity_config_pins_source_revisions_roles_and_unseen_axes() -> None:
    config = load_training_data_config(DEFAULT_CONFIG)

    assert config.schema_version == "robustness-training-data-config/v1"
    assert config.stage == "sanity"
    assert config.seed == 42
    assert config.model == "google/gemma-3-4b-it"
    assert config.training_token_budget == 1_000_000
    assert config.max_sequence_length == 512
    assert config.training_mixture == {
        "fineweb_edu": 0.85,
        "reasoning": 0.10,
        "natural_typo": 0.05,
    }
    assert config.explicit_clean_pair_probability == 0.10
    assert config.edit_count_probabilities == {"1": 0.50, "2": 0.30, "3-4": 0.20}
    assert set(config.training_operations) == {
        "keyboard-neighbor-substitution",
        "deletion",
        "insertion",
        "duplication",
        "natural-statistics-substitution",
    }
    assert config.held_out_operations == ("adjacent-transposition",)
    assert config.natural_repository_split == {
        "train": 0.70,
        "tune": 0.10,
        "held_out": 0.20,
    }
    assert config.fineweb_content_split == {
        "train": 0.80,
        "tune": 0.10,
        "pre_pr_gate": 0.05,
        "final_test": 0.05,
    }
    assert config.diagnostic_tasks == ("gsm8k", "mmlu", "arc")
    assert config.unseen_tasks == ("mmlu_pro", "math_500", "commonsense_qa")
    assert config.unseen_domain == "dolma"

    sources = config.sources
    assert sources["fineweb_edu"].dataset == "HuggingFaceFW/fineweb-edu"
    assert sources["fineweb_edu"].subset == "sample-10BT"
    assert sources["fineweb_edu"].revision == "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5"
    assert sources["fineweb_edu"].license == "odc-by-1.0"
    assert sources["fineweb_edu"].streaming is True
    for source in sources.values():
        assert re.fullmatch(r"[0-9a-f]{40}", source.revision)
        assert source.license
        assert source.role


def test_config_rejects_duplicate_keys_nonstandard_numbers_and_moving_revisions(
    tmp_path: Path,
) -> None:
    source = DEFAULT_CONFIG.read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text('{"schema_version":"duplicate",' + source.lstrip()[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_training_data_config(duplicate)

    nonstandard = tmp_path / "nonstandard.yaml"
    nonstandard.write_text(source.replace("0.85", "NaN", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_training_data_config(nonstandard)

    payload = json.loads(source)
    payload["sources"]["fineweb_edu"]["revision"] = "main"
    moving = tmp_path / "moving.yaml"
    moving.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_training_data_config(moving)

