from __future__ import annotations

from dataclasses import replace

import pytest

from typo_robust_training.probe.subspace_kill_config import SemanticSubspaceKillProtocol
from typo_robust_training.probe.subspace_kill_scoring import (
    PATCH_OPERATORS,
    SubspaceKillScoreRow,
    score_semantic_subspace_kill,
)


def _protocol(**changes) -> SemanticSubspaceKillProtocol:
    base = SemanticSubspaceKillProtocol(
        model="model",
        model_revision="a" * 40,
        code_revision="b" * 40,
        decoder_layers=34,
        hidden_size=32,
        parent_artifact_sha256="c" * 64,
        cohort_sha256="d" * 64,
        pca_activations_sha256="e" * 64,
        rank=16,
        primary_probe_seed=42,
        reproducibility_probe_seeds=(42, 43),
        random_basis_seed=101,
        complement_basis_seed=202,
        transition_layer_source="parent-probe-selected-transition/v1",
        hook_site="complete-decoder-block-residual-output",
        coordinate="edited-word-final-token/v1",
        patch_direction="clean-to-typo",
        operators=("untreated", *PATCH_OPERATORS),
        teacher_forced_tokens=16,
        readout_offsets=tuple(range(2, 17)),
        denominator_min_exclusive=1e-9,
        minimum_valid=160,
        minimum_valid_fraction=0.8,
        bootstrap_resamples=10_000,
        bootstrap_seed=1729,
        bootstrap_confidence=0.95,
        bootstrap_unit="source-group",
        semantic_full_ratio_lower=0.5,
        control_difference_lower=0.0,
        config_sha256="f" * 64,
    )
    return replace(base, **changes)


def _rows(
    *,
    count: int = 200,
    full: float = 0.8,
    semantic: float = 0.6,
    pca: float = 0.2,
    random: float = 0.1,
    complement: float = 0.15,
    layer: int = 7,
) -> tuple[SubspaceKillScoreRow, ...]:
    restoration = {
        "full-state": full,
        "semantic-rank16": semantic,
        "clean-fit-pca-rank16": pca,
        "deterministic-haar-random-rank16": random,
        "semantic-complement-rank16": complement,
    }
    return tuple(
        SubspaceKillScoreRow(
            pair_id=f"pair-{index}",
            source_group_sha256=f"{index + 1:064x}",
            transition_layer=layer,
            clean_word_final_token=4,
            typo_word_final_token=5,
            untreated_kl_2_16=(1.0,) * 15,
            patched_kl_2_16={
                operator: (1.0 - value,) * 15 for operator, value in restoration.items()
            },
        )
        for index in range(count)
    )


def test_both_sufficiency_and_control_gates_pass_from_raw_kl() -> None:
    summary = score_semantic_subspace_kill(_rows(), protocol=_protocol(), transition_layer=7)

    assert summary.passed is True
    assert summary.valid_records == 200
    assert summary.semantic_full_ratio_ci_lower == pytest.approx(0.75)
    assert all(value > 0 for value in summary.semantic_minus_control_ci_lower.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"full": -0.1},
        {"semantic": 0.39},
        {"pca": 0.61},
        {"random": 0.61},
        {"complement": 0.61},
    ],
)
def test_noncausal_full_or_insufficient_or_control_equivalent_semantic_fails(kwargs) -> None:
    summary = score_semantic_subspace_kill(
        _rows(**kwargs), protocol=_protocol(), transition_layer=7
    )

    assert summary.passed is False


def test_wrong_layer_and_too_few_valid_rows_fail_closed() -> None:
    with pytest.raises(ValueError, match="wrong patch layer"):
        score_semantic_subspace_kill(_rows(layer=6), protocol=_protocol(), transition_layer=7)
    with pytest.raises(ValueError, match="below"):
        score_semantic_subspace_kill(
            _rows(count=159), protocol=_protocol(), transition_layer=7
        )


def test_wrong_operator_inventory_or_invalid_coordinates_fail_closed() -> None:
    row = _rows(count=1)[0]
    with pytest.raises(ValueError, match="operator inventory"):
        SubspaceKillScoreRow(
            pair_id=row.pair_id,
            source_group_sha256=row.source_group_sha256,
            transition_layer=row.transition_layer,
            clean_word_final_token=row.clean_word_final_token,
            typo_word_final_token=row.typo_word_final_token,
            untreated_kl_2_16=row.untreated_kl_2_16,
            patched_kl_2_16={"full-state": (0.2,) * 15},
        )
    with pytest.raises(ValueError, match="coordinates"):
        SubspaceKillScoreRow(
            pair_id=row.pair_id,
            source_group_sha256=row.source_group_sha256,
            transition_layer=-1,
            clean_word_final_token=4,
            typo_word_final_token=5,
            untreated_kl_2_16=row.untreated_kl_2_16,
            patched_kl_2_16=row.patched_kl_2_16,
        )
