"""Adversarial tests for the one project-global WP-2 retrain claim."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from typo_robust_training.sae.retry import (
    Wp2RetryInputs,
    claim_wp2_retry_training,
    load_wp2_retry_lineage,
    record_wp2_retry_training_completion,
    require_claimed_wp2_retry_training,
)


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _failed_project(tmp_path: Path) -> tuple[Wp2RetryInputs, str, str]:
    project = tmp_path / "project"
    initial_config = "a" * 64
    initial_registry = "b" * 64
    retry_config = "c" * 64
    retry_registry = "d" * 64

    training_dir = project / "training"
    training_sha = _write(
        training_dir / "run.json",
        {
            "bindings": {
                "config_sha256": initial_config,
                "preregistration_sha256": initial_registry,
            }
        },
    )
    validation_dir = project / "validation"
    acceptance_sha = _write(
        validation_dir / "wp2_acceptance.json",
        {
            "passed": False,
            "config_sha256": initial_config,
            "preregistration_sha256": initial_registry,
        },
    )
    validation_sha = _write(
        validation_dir / "run.json",
        {
            "passed": False,
            "acceptance_sha256": acceptance_sha,
            "training_run_sha256": training_sha,
        },
    )
    ledger_path = project / "wp2_attempts.json"
    ledger_sha = _write(
        ledger_path,
        {
            "schema_version": "robustness-sae-wp2-attempt-ledger/v1",
            "config_sha256": initial_config,
            "preregistration_sha256": initial_registry,
            "maximum_retrains_after_failure": 1,
            "attempts": [
                {
                    "attempt": 1,
                    "checkpoint_run_sha256": training_sha,
                    "output_dir": str(validation_dir.resolve()),
                    "passed": False,
                    "acceptance_sha256": acceptance_sha,
                }
            ],
        },
    )
    authorization_path = tmp_path / "retry-authorization.json"
    _write(
        authorization_path,
        {
            "schema_version": "robustness-sae-wp2-retry-authorization/v1",
            "project_id": "test-project",
            "maximum_full_retrain_bundles": 1,
            "initial_failure": {
                "config_sha256": initial_config,
                "preregistration_sha256": initial_registry,
                "attempt_ledger_sha256": ledger_sha,
                "training_run_sha256": training_sha,
                "validation_run_sha256": validation_sha,
                "acceptance_sha256": acceptance_sha,
            },
            "retry": {
                "config_sha256": retry_config,
                "preregistration_sha256": retry_registry,
            },
        },
    )
    return (
        Wp2RetryInputs(
            authorization_path=authorization_path,
            initial_attempt_ledger_path=ledger_path,
            initial_training_dir=training_dir,
        ),
        retry_config,
        retry_registry,
    )


def _lineage(tmp_path: Path):
    inputs, config_sha, registry_sha = _failed_project(tmp_path)
    return load_wp2_retry_lineage(
        inputs,
        retry_config_sha256=config_sha,
        retry_preregistration_sha256=registry_sha,
    )


def _retry_bindings(lineage, **extra: object) -> dict[str, object]:
    return {
        "config_sha256": lineage.authorization.retry_config_sha256,
        "preregistration_sha256": lineage.authorization.retry_preregistration_sha256,
        **extra,
    }


def test_relocated_retry_cannot_mint_a_third_wp2_bundle(tmp_path: Path) -> None:
    """This is the falsification test: changing parent directories cannot reset the budget."""

    lineage = _lineage(tmp_path)
    claim_wp2_retry_training(
        lineage,
        output_dir=tmp_path / "attempt-2" / "training",
        training_bindings=_retry_bindings(lineage, source="first-retry"),
        resume=False,
    )

    with pytest.raises(ValueError, match="project-global retrain limit is exhausted"):
        claim_wp2_retry_training(
            lineage,
            output_dir=tmp_path / "unrelated-parent" / "attempt-3" / "training",
            training_bindings=_retry_bindings(lineage, source="third-bundle"),
            resume=False,
        )


def test_retry_resume_requires_the_same_preexisting_claim(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path)
    output = tmp_path / "retry" / "training"
    bindings = _retry_bindings(lineage, source_record_sha256="e" * 64)

    with pytest.raises(ValueError, match="requires the pre-training global claim"):
        claim_wp2_retry_training(
            lineage,
            output_dir=output,
            training_bindings=bindings,
            resume=True,
        )

    first = claim_wp2_retry_training(
        lineage,
        output_dir=output,
        training_bindings=bindings,
        resume=False,
    )
    resumed = claim_wp2_retry_training(
        lineage,
        output_dir=output,
        training_bindings=bindings,
        resume=True,
    )
    assert resumed.sha256 == first.sha256

    with pytest.raises(ValueError, match="resume differs"):
        claim_wp2_retry_training(
            lineage,
            output_dir=tmp_path / "other" / "training",
            training_bindings=bindings,
            resume=True,
        )


def test_concurrent_double_claim_allows_exactly_one_bundle(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path)

    def attempt(index: int) -> str:
        try:
            claim_wp2_retry_training(
                lineage,
                output_dir=tmp_path / f"retry-{index}" / "training",
                training_bindings=_retry_bindings(lineage, candidate=index),
                resume=False,
            )
        except ValueError:
            return "rejected"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (1, 2)))
    assert sorted(outcomes) == ["claimed", "rejected"]


def test_validation_accepts_only_the_claimed_training_run_sha(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path)
    checkpoint = tmp_path / "retry" / "training"
    base_bindings = _retry_bindings(lineage, source_record_sha256="e" * 64)
    claim = claim_wp2_retry_training(
        lineage,
        output_dir=checkpoint,
        training_bindings=base_bindings,
        resume=False,
    )
    _write(
        checkpoint / "run.json",
        {"bindings": {**base_bindings, "wp2_retry_claim_sha256": claim.sha256}},
    )
    record_wp2_retry_training_completion(
        lineage,
        claim=claim,
        training_run_path=checkpoint / "run.json",
    )
    assert require_claimed_wp2_retry_training(lineage, checkpoint_dir=checkpoint) == claim

    forged = tmp_path / "forged" / "training"
    _write(
        forged / "run.json",
        {"bindings": {**base_bindings, "wp2_retry_claim_sha256": claim.sha256}},
    )
    with pytest.raises(ValueError, match="not the claimed retry training run"):
        require_claimed_wp2_retry_training(lineage, checkpoint_dir=forged)


def test_retry_lineage_rejects_tampered_initial_failure(tmp_path: Path) -> None:
    inputs, config_sha, registry_sha = _failed_project(tmp_path)
    acceptance = inputs.initial_attempt_ledger_path.parent / "validation/wp2_acceptance.json"
    acceptance.write_text('{"passed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash chain differs"):
        load_wp2_retry_lineage(
            inputs,
            retry_config_sha256=config_sha,
            retry_preregistration_sha256=registry_sha,
        )
