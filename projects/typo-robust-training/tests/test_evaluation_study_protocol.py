"""The model-independent evaluation study is frozen before cycle-2 training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typo_robust_training.evaluation.study import load_evaluation_study_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/robustness-evaluation-v1.yaml"


def test_default_study_freezes_tiers_population_typos_statistics_and_gates() -> None:
    protocol = load_evaluation_study_protocol(DEFAULT_PROTOCOL)

    assert protocol.schema_version == "robustness-evaluation-study/v1"
    assert protocol.protocol_id == "typo-robustness-evaluation-v1.2"
    assert protocol.seed == 42
    assert protocol.training_seeds == (42, 43, 44)
    assert protocol.monitor_task_accuracy_allowed is False
    assert protocol.tune_task_records_total == 500
    assert protocol.tune_natural_injection_records == 100
    assert protocol.role_tasks == {
        "pre_pr_gate": ("gsm8k", "mmlu", "arc", "mmlu_pro", "commonsense_qa"),
        "final_test": (
            "gsm8k",
            "mmlu",
            "arc",
            "mmlu_pro",
            "math_500",
            "commonsense_qa",
        ),
    }
    assert protocol.records_per_task == {
        "pre_pr_gate": {
            "gsm8k": 500,
            "mmlu": 500,
            "arc": 500,
            "mmlu_pro": 500,
            "commonsense_qa": 500,
        },
        "final_test": {
            "gsm8k": 500,
            "mmlu": 500,
            "arc": 500,
            "mmlu_pro": 500,
            "math_500": 466,
            "commonsense_qa": 500,
        },
    }
    assert protocol.maximum_openings == {"pre_pr_gate": 1, "final_test": 1}
    assert protocol.primary_typo_condition == "random-2"
    assert protocol.primary_edit_count == 2
    assert protocol.primary_operations == (
        "keyboard-neighbor-substitution",
        "deletion",
        "duplication",
    )
    assert protocol.minimum_word_letters == 3
    assert protocol.question_only is True
    assert protocol.corpus_counts == {
        "tune": {"fineweb_edu": 200, "dolma": 0, "natural_pairs": 100},
        "pre_pr_gate": {"fineweb_edu": 1000, "dolma": 500, "natural_pairs": 500},
        "final_test": {"fineweb_edu": 1000, "dolma": 1000, "natural_pairs": 1000},
    }
    assert protocol.corpus_max_tokens == 512
    assert protocol.corpus_ppl_sources == ("fineweb_edu", "dolma")
    assert protocol.audit_records == {"tune": 100, "pre_pr_gate": 500, "final_test": 500}
    assert protocol.shots == {
        "gsm8k": 8,
        "mmlu": 5,
        "arc": 5,
        "mmlu_pro": 5,
        "math_500": 4,
        "commonsense_qa": 5,
    }
    assert protocol.max_new_tokens == 512
    assert protocol.bootstrap_replicates == 10_000
    assert protocol.gates == {
        "clean_noninferiority_margin_points": 1.0,
        "minimum_task_clean_change_points": -3.0,
        "minimum_typo_gain_points": 2.0,
        "require_typo_ci_lower_above_zero": True,
        "maximum_clean_ppl_ratio": 1.02,
        "maximum_clean_kl_nats_per_token": 0.03,
        "minimum_directional_seeds": 2,
        "natural_minimum_point_change": -1.0,
        "natural_minimum_ci_lower": -2.0,
        "patch_audit_is_blocking": False,
    }


def test_study_rejects_scientific_drift_and_duplicate_json_keys(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    payload["typos"]["primary"]["edit_count"] = 1
    moved = tmp_path / "moved.yaml"
    moved.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="primary typo protocol differs"):
        load_evaluation_study_protocol(moved)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        DEFAULT_PROTOCOL.read_text(encoding="utf-8").replace(
            '"protocol_id": "typo-robustness-evaluation-v1.2",',
            '"protocol_id": "typo-robustness-evaluation-v1.2",\n'
            '  "protocol_id": "typo-robustness-evaluation-v1.2",',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_evaluation_study_protocol(duplicate)


def test_study_digest_changes_if_any_frozen_byte_changes(tmp_path: Path) -> None:
    protocol = load_evaluation_study_protocol(DEFAULT_PROTOCOL)
    copied = tmp_path / "copy.yaml"
    copied.write_bytes(DEFAULT_PROTOCOL.read_bytes() + b"\n")
    copied_protocol = load_evaluation_study_protocol(copied)

    assert copied_protocol.protocol_id == protocol.protocol_id
    assert copied_protocol.config_sha256 != protocol.config_sha256
