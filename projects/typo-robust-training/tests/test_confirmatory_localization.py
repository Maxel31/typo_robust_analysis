"""Confirmatory generic-text joint-window localization contract."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training import cli
from typo_robust_training.localization.confirmatory_config import (
    load_confirmatory_localization_config,
    window_width_for_decoder_layers,
)
from typo_robust_training.localization.confirmatory_records import JointWindowScan
from typo_robust_training.localization.confirmatory_scoring import (
    select_joint_patch_window,
    validate_joint_patch_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cycle3" / "gemma4b-generic-joint-window.yaml"


def test_confirmatory_protocol_is_generic_kl_only_and_fixed_before_behavior() -> None:
    protocol = load_confirmatory_localization_config(DEFAULT_CONFIG)

    assert protocol.schema_version == "robustness-confirmatory-localization-config/v1"
    assert protocol.model == "google/gemma-3-4b-it"
    assert protocol.model_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert protocol.decoder_layers == 34
    assert protocol.selection_records == 200
    assert protocol.validation_records == 200
    assert protocol.selection_source == "fineweb_edu"
    assert protocol.typo_operations == (
        "keyboard-neighbor-substitution",
        "deletion",
        "duplication",
    )
    assert protocol.teacher_forced_tokens == 16
    assert protocol.readout_token_range == (2, 16)
    assert protocol.denominator_min_exclusive == 1e-9
    assert protocol.window_width == 6
    assert protocol.window_selection_metric == "median-pairwise-kl-restoration/v1"
    assert protocol.tie_break == "smallest-start-layer/v1"
    assert protocol.validation_rule == "bootstrap-95ci-lower-strictly-positive/v1"
    assert not hasattr(protocol, "answer_weight")
    assert not hasattr(protocol, "clean_harm_weight")


@pytest.mark.parametrize(
    ("layers", "expected"),
    ((1, 1), (2, 1), (8, 1), (9, 2), (28, 5), (32, 5), (34, 6), (40, 7)),
)
def test_window_width_uses_one_predeclared_half_up_relative_rule(
    layers: int, expected: int
) -> None:
    assert window_width_for_decoder_layers(layers) == expected


def test_cli_separates_pair_freeze_selection_and_independent_validation() -> None:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command")
    register_commands(commands)

    freeze = root.parse_args(
        [
            "freeze-generic-localization-pairs",
            "--config",
            "confirmatory.json",
            "--exclude-data",
            "training",
            "--exclude-data",
            "evaluation",
            "--output-dir",
            "pairs",
        ]
    )
    assert freeze.exclude_data == [Path("training"), Path("evaluation")]

    selection = root.parse_args(
        [
            "select-generic-joint-patch-window",
            "--config",
            "confirmatory.json",
            "--selection-manifest",
            "selection.jsonl",
            "--gpu-id",
            "5",
            "--output-dir",
            "selection",
            "--resume",
        ]
    )
    assert selection.gpu_id == "5"
    assert selection.selection_manifest == Path("selection.jsonl")
    assert selection.resume is True

    validation = root.parse_args(
        [
            "validate-generic-joint-patch-window",
            "--config",
            "confirmatory.json",
            "--validation-manifest",
            "validation.jsonl",
            "--window-selection",
            "window.json",
            "--gpu-id",
            "6",
            "--output-dir",
            "validation",
        ]
    )
    assert validation.gpu_id == "6"
    assert validation.validation_manifest == Path("validation.jsonl")
    assert validation.window_selection == Path("window.json")


def _scan(
    record_id: str,
    *,
    untreated: float,
    patched: dict[int, float],
    decoder_layers: int = 3,
    window_width: int = 1,
    role: str = "selection",
) -> JointWindowScan:
    return JointWindowScan(
        record_id=record_id,
        role=role,
        decoder_layers=decoder_layers,
        window_width=window_width,
        target_token_ids=tuple(range(16)),
        untreated_kl_2_16=(untreated,) * 15,
        patched_kl_2_16_by_window={start: (value,) * 15 for start, value in patched.items()},
        invalid_reason=None,
        audit={"source": "fineweb_edu"},
    )


def test_selection_uses_median_of_pairwise_restoration_not_ratio_of_means() -> None:
    # Window 0 pairwise scores are [1, 0], whose median is 0.5.  A ratio of
    # aggregate means would incorrectly report 1 - 1/101 ~= 0.9901.
    scans = (
        _scan("large", untreated=100.0, patched={0: 0.0, 1: 50.0, 2: 50.0}),
        _scan("small", untreated=1.0, patched={0: 1.0, 1: 0.5, 2: 0.5}),
    )

    result = select_joint_patch_window(
        scans,
        denominator_min_exclusive=1e-9,
        minimum_eligible=2,
        minimum_eligible_fraction=1.0,
        bootstrap_replicates=20,
        bootstrap_seed=42,
        confidence_level=0.95,
        random_control_seed=42,
    )

    assert result.window_scores[0] == pytest.approx(0.5)
    assert result.selected_window == (0, 1)


def test_exact_window_score_tie_selects_shallower_start() -> None:
    scans = tuple(
        _scan(record_id, untreated=1.0, patched={0: 0.4, 1: 0.4, 2: 0.8})
        for record_id in ("a", "b", "c")
    )

    result = select_joint_patch_window(
        scans,
        denominator_min_exclusive=1e-9,
        minimum_eligible=3,
        minimum_eligible_fraction=1.0,
        bootstrap_replicates=20,
        bootstrap_seed=42,
        confidence_level=0.95,
        random_control_seed=42,
    )

    assert result.selected_window == (0, 1)
    assert result.random_control_window == (2, 3)


def test_independent_validation_uses_only_the_frozen_window_and_strict_positive_ci() -> None:
    passing = tuple(
        _scan(
            str(index),
            untreated=1.0,
            patched={1: 0.5},
            role="validation",
        )
        for index in range(10)
    )
    result = validate_joint_patch_window(
        passing,
        selected_window=(1, 2),
        denominator_min_exclusive=1e-9,
        minimum_eligible=10,
        minimum_eligible_fraction=1.0,
        bootstrap_replicates=100,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    assert result.median_pairwise_restoration == pytest.approx(0.5)
    assert result.confidence_interval == pytest.approx((0.5, 0.5))
    assert result.passed is True

    mixed = tuple(
        _scan(
            str(index),
            untreated=1.0,
            patched={1: 0.0 if index < 5 else 2.0},
            role="validation",
        )
        for index in range(10)
    )
    failed = validate_joint_patch_window(
        mixed,
        selected_window=(1, 2),
        denominator_min_exclusive=1e-9,
        minimum_eligible=10,
        minimum_eligible_fraction=1.0,
        bootstrap_replicates=100,
        bootstrap_seed=42,
        confidence_level=0.95,
    )
    assert failed.confidence_interval[0] <= 0.0
    assert failed.passed is False


def test_joint_window_scan_round_trips_and_rejects_missing_selection_windows() -> None:
    scan = _scan("a", untreated=1.0, patched={0: 0.2, 1: 0.4, 2: 0.8})
    assert JointWindowScan.from_dict(scan.as_dict()) == scan

    payload = scan.as_dict()
    payload["patched_kl_2_16_by_window"] = payload["patched_kl_2_16_by_window"][:-1]
    with pytest.raises(ValueError, match="every candidate window"):
        JointWindowScan.from_dict(payload)


def test_config_rejects_behavior_dependent_selection_fields(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["selection"]["answer_weight"] = 0.5
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selection fields differ"):
        load_confirmatory_localization_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("confidence_level", 0.90, "confidence level must equal 0.95"),
        ("minimum_following_tokens", 15, "minimum following tokens must equal teacher tokens"),
    ),
)
def test_config_enforces_frozen_validation_contract(
    tmp_path: Path, field: str, value: float | int, message: str
) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if field == "confidence_level":
        payload["statistics"][field] = value
    else:
        payload["data"][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_confirmatory_localization_config(path)


def test_validation_cli_returns_nonzero_when_frozen_window_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "typo_robust_training.localization.confirmatory_runner."
        "run_validate_generic_joint_patch_window",
        lambda _config: SimpleNamespace(
            passed=False,
            validation_path=tmp_path / "window_validation.json",
            scans_path=tmp_path / "scans.jsonl",
            run_path=tmp_path / "run.json",
        ),
    )
    args = SimpleNamespace(
        config=tmp_path / "config.json",
        validation_manifest=tmp_path / "validation.jsonl",
        window_selection=tmp_path / "selection.json",
        gpu_id="6",
        output_dir=tmp_path / "output",
        resume=False,
    )

    assert cli._run_validate_generic_joint_window(args) == 1
