"""Adversarial tests for the one project-global WP-2 retrain claim."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_robust_training.sae import retry as retry_module
from typo_robust_training.sae.retry import (
    Wp2RetryInputs,
    claim_wp2_retry_training,
    hold_wp2_retry_training_lease,
    load_wp2_retry_lineage,
    record_wp2_retry_training_completion,
    require_claimed_wp2_retry_training,
)


def _lease_worker(project_root, project_root_device, project_root_inode, ready, release, reached, outcomes) -> None:
    lineage = SimpleNamespace(
        project_root=Path(project_root),
        project_root_device=project_root_device,
        project_root_inode=project_root_inode,
    )
    try:
        with hold_wp2_retry_training_lease(lineage):
            reached.put(os.getpid())
            ready.set()
            release.wait(timeout=10)
            outcomes.put((os.getpid(), "completed"))
    except ValueError:
        outcomes.put((os.getpid(), "rejected"))


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _root_identity(path: Path) -> dict[str, int]:
    metadata = path.stat()
    return {
        "project_root_device": metadata.st_dev,
        "project_root_inode": metadata.st_ino,
    }


def _failed_project(
    tmp_path: Path,
    *,
    validation_dir: Path | None = None,
) -> tuple[Wp2RetryInputs, str, str]:
    project = tmp_path / "project"
    initial_config = "a" * 64
    initial_registry = "b" * 64
    retry_config = "c" * 64

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
    validation_dir = validation_dir or project / "validation"
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
    retry_preregistration_path = tmp_path / "retry-preregistration.json"
    retry_registry = _write(
        retry_preregistration_path,
        {
            "schema_version": "robustness-sae-preregistry/v2",
            "wp2_project_root": str(project),
            "wp2_project_root_identity": {
                "device": project.stat().st_dev,
                "inode": project.stat().st_ino,
            },
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
            project_root=project,
            project_root_device=project.stat().st_dev,
            project_root_inode=project.stat().st_ino,
            preregistration_path=retry_preregistration_path,
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


def test_retry_claim_never_follows_the_final_record_symlink(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path)
    external = tmp_path / "external-claim.json"
    lineage.claim_path.symlink_to(external)

    with pytest.raises(ValueError, match="project-global retrain limit is exhausted"):
        claim_wp2_retry_training(
            lineage,
            output_dir=tmp_path / "retry" / "training",
            training_bindings=_retry_bindings(lineage),
            resume=False,
        )

    assert lineage.claim_path.is_symlink()
    assert not external.exists()


def test_retry_claim_rejects_project_root_replaced_by_a_symlink(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path)
    original_root = tmp_path / "original-project"
    lineage.project_root.rename(original_root)
    substituted_root = tmp_path / "substituted-project"
    substituted_root.mkdir()
    lineage.project_root.symlink_to(substituted_root, target_is_directory=True)

    with pytest.raises(ValueError, match="pre-existing directory"):
        claim_wp2_retry_training(
            lineage,
            output_dir=tmp_path / "retry" / "training",
            training_bindings=_retry_bindings(lineage),
            resume=False,
        )

    assert not (substituted_root / "wp2_retry_claim.json").exists()
    assert not (original_root / "wp2_retry_claim.json").exists()


def test_loaded_lineage_does_not_follow_replaced_project_root_directory(
    tmp_path: Path,
) -> None:
    """A new regular directory at the reviewed pathname must not reset the budget."""

    lineage = _lineage(tmp_path)
    original_root = tmp_path / "original-project"
    lineage.project_root.rename(original_root)
    lineage.project_root.mkdir()

    with pytest.raises(ValueError, match="filesystem identity changed"):
        with hold_wp2_retry_training_lease(lineage):
            pytest.fail("a replacement inode must never acquire the retry lease")
    with pytest.raises(ValueError, match="pre-existing directory"):
        claim_wp2_retry_training(
            lineage,
            output_dir=tmp_path / "retry" / "training",
            training_bindings=_retry_bindings(lineage),
            resume=False,
        )

    assert not (lineage.project_root / "wp2_retry_training.lock").exists()
    assert not (lineage.project_root / "wp2_retry_claim.json").exists()
    assert not (original_root / "wp2_retry_training.lock").exists()
    assert not (original_root / "wp2_retry_claim.json").exists()


def test_new_invocation_cannot_recapture_replacement_root_identity(
    tmp_path: Path,
) -> None:
    """The reviewed preregistration, not each process, pins the root inode."""

    inputs, config_sha, registry_sha = _failed_project(tmp_path)
    lineage = load_wp2_retry_lineage(
        inputs,
        retry_config_sha256=config_sha,
        retry_preregistration_sha256=registry_sha,
    )
    claim_wp2_retry_training(
        lineage,
        output_dir=tmp_path / "retry-1" / "training",
        training_bindings=_retry_bindings(lineage),
        resume=False,
    )

    original_root = tmp_path / "original-project"
    inputs.project_root.rename(original_root)
    inputs.project_root.mkdir()
    shutil.copytree(original_root / "training", inputs.project_root / "training")
    shutil.copytree(original_root / "validation", inputs.project_root / "validation")
    shutil.copy2(
        original_root / "wp2_attempts.json",
        inputs.project_root / "wp2_attempts.json",
    )
    replacement_inputs = Wp2RetryInputs(
        project_root=inputs.project_root,
        project_root_device=inputs.project_root.stat().st_dev,
        project_root_inode=inputs.project_root.stat().st_ino,
        preregistration_path=inputs.preregistration_path,
        authorization_path=inputs.authorization_path,
        initial_attempt_ledger_path=inputs.project_root / "wp2_attempts.json",
        initial_training_dir=inputs.project_root / "training",
    )

    with pytest.raises(ValueError, match="preregistration project root identity differs"):
        load_wp2_retry_lineage(
            replacement_inputs,
            retry_config_sha256=config_sha,
            retry_preregistration_sha256=registry_sha,
        )

    assert not (inputs.project_root / "wp2_retry_claim.json").exists()
    assert (original_root / "wp2_retry_claim.json").is_file()


def test_initial_validation_reservation_rejects_project_root_symlink_substitution(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    project_root_device = project_root.stat().st_dev
    project_root_inode = project_root.stat().st_ino
    original_root = tmp_path / "original-project"
    project_root.rename(original_root)
    substituted_root = tmp_path / "substituted-project"
    substituted_root.mkdir()
    project_root.symlink_to(substituted_root, target_is_directory=True)

    with pytest.raises(ValueError, match="pre-existing directory"):
        retry_module.reserve_initial_wp2_validation(
            project_root=project_root,
            project_root_device=project_root_device,
            project_root_inode=project_root_inode,
            config_sha256="a" * 64,
            preregistration_sha256="b" * 64,
            checkpoint_dir=tmp_path / "training",
            checkpoint_run_sha256="c" * 64,
            output_dir=tmp_path / "validation",
        )

    assert not (substituted_root / "wp2_initial_validation_reservation.json").exists()
    assert not (original_root / "wp2_initial_validation_reservation.json").exists()


def test_initial_validation_reservation_rejects_project_root_inode_substitution(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    identity = _root_identity(project_root)
    original_root = tmp_path / "original-project"
    project_root.rename(original_root)
    project_root.mkdir()

    with pytest.raises(ValueError, match="pre-existing directory"):
        retry_module.reserve_initial_wp2_validation(
            project_root=project_root,
            **identity,
            config_sha256="a" * 64,
            preregistration_sha256="b" * 64,
            checkpoint_dir=tmp_path / "training",
            checkpoint_run_sha256="c" * 64,
            output_dir=tmp_path / "validation",
        )

    assert not (project_root / "wp2_initial_validation_reservation.json").exists()
    assert not (original_root / "wp2_initial_validation_reservation.json").exists()


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

    with pytest.raises(ValueError, match="resume differs"):
        claim_wp2_retry_training(
            lineage,
            output_dir=output,
            training_bindings={**bindings, "source_record_sha256": "f" * 64},
            resume=True,
        )


@pytest.mark.parametrize("artifact_kind", ("symlink", "hardlink", "fifo", "directory"))
def test_retry_resume_rejects_unsafe_existing_claim_authority_records(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    lineage = _lineage(tmp_path)
    output = tmp_path / "retry" / "training"
    bindings = _retry_bindings(lineage, source_record_sha256="e" * 64)
    claim_wp2_retry_training(
        lineage,
        output_dir=output,
        training_bindings=bindings,
        resume=False,
    )
    claim_bytes = lineage.claim_path.read_bytes()
    lineage.claim_path.unlink()
    external = tmp_path / "external-claim.json"
    external.write_bytes(claim_bytes)
    if artifact_kind == "symlink":
        lineage.claim_path.symlink_to(external)
    elif artifact_kind == "hardlink":
        os.link(external, lineage.claim_path)
    elif artifact_kind == "fifo":
        os.mkfifo(lineage.claim_path)
    else:
        lineage.claim_path.mkdir()

    with pytest.raises(ValueError, match="requires the pre-training global claim"):
        claim_wp2_retry_training(
            lineage,
            output_dir=output,
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


def test_retry_training_lease_allows_exactly_one_process_to_reach_runtime(
    tmp_path: Path,
) -> None:
    context = get_context("fork")
    ready = context.Event()
    release = context.Event()
    reached = context.Queue()
    outcomes = context.Queue()
    project_root = tmp_path / "project"
    project_root.mkdir()
    project_root_device = project_root.stat().st_dev
    project_root_inode = project_root.stat().st_ino

    first = context.Process(
        target=_lease_worker,
        args=(
            project_root,
            project_root_device,
            project_root_inode,
            ready,
            release,
            reached,
            outcomes,
        ),
    )
    first.start()
    assert ready.wait(timeout=10)
    second = context.Process(
        target=_lease_worker,
        args=(
            project_root,
            project_root_device,
            project_root_inode,
            ready,
            release,
            reached,
            outcomes,
        ),
    )
    second.start()
    second.join(timeout=10)
    assert not second.is_alive()
    rejected_pid, rejected_status = outcomes.get(timeout=2)
    assert (rejected_pid, rejected_status) == (second.pid, "rejected")

    assert reached.get(timeout=2) == first.pid
    release.set()
    first.join(timeout=10)
    assert not first.is_alive()
    completed_pid, completed_status = outcomes.get(timeout=2)
    assert (completed_pid, completed_status) == (first.pid, "completed")


def test_retry_training_lease_rejects_a_symlink(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    project_root_device = project_root.stat().st_dev
    project_root_inode = project_root.stat().st_ino
    target = tmp_path / "external-lock"
    target.write_text("not a lock\n", encoding="utf-8")
    (project_root / "wp2_retry_training.lock").symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        with hold_wp2_retry_training_lease(
            SimpleNamespace(
                project_root=project_root,
                project_root_device=project_root_device,
                project_root_inode=project_root_inode,
            )
        ):
            pytest.fail("a symlink must never acquire the retry lease")


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
    assert require_claimed_wp2_retry_training(lineage, checkpoint_dir=checkpoint).claim == claim

    forged = tmp_path / "forged" / "training"
    _write(
        forged / "run.json",
        {"bindings": {**base_bindings, "wp2_retry_claim_sha256": claim.sha256}},
    )
    with pytest.raises(ValueError, match="not the claimed retry training run"):
        require_claimed_wp2_retry_training(lineage, checkpoint_dir=forged)


def test_training_completion_idempotence_rejects_an_exact_payload_symlink(
    tmp_path: Path,
) -> None:
    lineage = _lineage(tmp_path)
    checkpoint = tmp_path / "retry" / "training"
    base_bindings = _retry_bindings(lineage, source_record_sha256="e" * 64)
    claim = claim_wp2_retry_training(
        lineage,
        output_dir=checkpoint,
        training_bindings=base_bindings,
        resume=False,
    )
    run_path = checkpoint / "run.json"
    run_sha256 = _write(
        run_path,
        {"bindings": {**base_bindings, "wp2_retry_claim_sha256": claim.sha256}},
    )
    external = tmp_path / "external-training-completion.json"
    _write(
        external,
        {
            "schema_version": "robustness-sae-wp2-retry-training-completion/v1",
            "project_id": lineage.authorization.project_id,
            "claim_sha256": claim.sha256,
            "training_run_path": str(run_path.resolve()),
            "training_run_sha256": run_sha256,
        },
    )
    lineage.training_completion_path.symlink_to(external)

    with pytest.raises(ValueError, match="missing or unsafe"):
        record_wp2_retry_training_completion(
            lineage,
            claim=claim,
            training_run_path=run_path,
        )


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


def test_retry_lineage_accepts_hash_bound_validation_output_outside_project_root(
    tmp_path: Path,
) -> None:
    """A legal absolute validation output must not make the authorized retry unusable."""

    external_validation = tmp_path / "shared-results" / "sae-validation"
    inputs, config_sha, registry_sha = _failed_project(
        tmp_path,
        validation_dir=external_validation,
    )

    lineage = load_wp2_retry_lineage(
        inputs,
        retry_config_sha256=config_sha,
        retry_preregistration_sha256=registry_sha,
    )
    assert lineage.project_root == inputs.initial_attempt_ledger_path.parent.resolve()


def test_copied_initial_bundle_and_new_authorization_cannot_mint_another_claim(
    tmp_path: Path,
) -> None:
    """The reviewed v2 project root, not a copied ledger, owns the retry claim."""

    inputs, config_sha, registry_sha = _failed_project(tmp_path)
    copied_project = tmp_path / "copied-project"
    shutil.copytree(inputs.project_root, copied_project)
    copied_authorization = tmp_path / "copied-authorization.json"
    authorization = json.loads(inputs.authorization_path.read_text(encoding="utf-8"))
    authorization["project_id"] = "copied-project"
    _write(copied_authorization, authorization)
    copied_inputs = Wp2RetryInputs(
        project_root=inputs.project_root,
        project_root_device=inputs.project_root_device,
        project_root_inode=inputs.project_root_inode,
        preregistration_path=inputs.preregistration_path,
        authorization_path=copied_authorization,
        initial_attempt_ledger_path=copied_project / "wp2_attempts.json",
        initial_training_dir=copied_project / "training",
    )

    with pytest.raises(ValueError, match="outside the preregistered project root"):
        load_wp2_retry_lineage(
            copied_inputs,
            retry_config_sha256=config_sha,
            retry_preregistration_sha256=registry_sha,
        )


def _completed_retry_training(tmp_path: Path):
    lineage = _lineage(tmp_path)
    checkpoint = tmp_path / "retry" / "training"
    bindings = _retry_bindings(lineage, source_record_sha256="e" * 64)
    claim = claim_wp2_retry_training(
        lineage,
        output_dir=checkpoint,
        training_bindings=bindings,
        resume=False,
    )
    run_sha = _write(
        checkpoint / "run.json",
        {"bindings": {**bindings, "wp2_retry_claim_sha256": claim.sha256}},
    )
    record_wp2_retry_training_completion(
        lineage,
        claim=claim,
        training_run_path=checkpoint / "run.json",
    )
    claimed = require_claimed_wp2_retry_training(lineage, checkpoint_dir=checkpoint)
    return lineage, checkpoint, claim, claimed, run_sha


def test_retry_validation_reservation_is_exclusive_before_any_output(tmp_path: Path) -> None:
    """Falsify a design that waits until validation completion to claim attempt 2."""

    lineage, checkpoint, _claim, claimed, _run_sha = _completed_retry_training(tmp_path)
    reserve = getattr(retry_module, "reserve_wp2_retry_validation")

    def attempt(index: int) -> str:
        try:
            reserve(
                lineage,
                claimed_training=claimed,
                checkpoint_dir=checkpoint,
                output_dir=tmp_path / f"validation-{index}",
            )
        except ValueError:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (1, 2)))
    assert sorted(outcomes) == ["rejected", "reserved"]
    assert not (tmp_path / "validation-1").exists()
    assert not (tmp_path / "validation-2").exists()


def test_initial_validation_reservation_is_exclusive_before_any_output(tmp_path: Path) -> None:
    """Falsify two concurrent initial validations that both reach artifact generation."""

    reserve = getattr(retry_module, "reserve_initial_wp2_validation")
    checkpoint = tmp_path / "training"
    checkpoint.mkdir()
    run_sha = _write(checkpoint / "run.json", {"bindings": {}})

    def attempt(index: int) -> str:
        try:
            reserve(
                project_root=tmp_path,
                **_root_identity(tmp_path),
                config_sha256="a" * 64,
                preregistration_sha256="b" * 64,
                checkpoint_dir=checkpoint,
                checkpoint_run_sha256=run_sha,
                output_dir=tmp_path / f"validation-{index}",
            )
        except ValueError:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (1, 2)))
    assert sorted(outcomes) == ["rejected", "reserved"]
    assert not (tmp_path / "validation-1").exists()
    assert not (tmp_path / "validation-2").exists()


def test_exclusive_reservation_fsyncs_the_project_root_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file-only fsync must not leave the consumed slot vulnerable to a crash."""

    checkpoint = tmp_path / "training"
    checkpoint.mkdir()
    run_sha = _write(checkpoint / "run.json", {"bindings": {}})
    observed_modes: list[int] = []
    real_fsync = retry_module.os.fsync

    def record_fsync(descriptor: int) -> None:
        observed_modes.append(retry_module.os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(retry_module.os, "fsync", record_fsync)
    retry_module.reserve_initial_wp2_validation(
        project_root=tmp_path,
        **_root_identity(tmp_path),
        config_sha256="a" * 64,
        preregistration_sha256="b" * 64,
        checkpoint_dir=checkpoint,
        checkpoint_run_sha256=run_sha,
        output_dir=tmp_path / "validation",
    )

    assert any(stat.S_ISDIR(mode) for mode in observed_modes)


def test_exclusive_reservation_refuses_to_create_a_missing_project_root(
    tmp_path: Path,
) -> None:
    """The reviewed project identity must pre-exist; a writer cannot mint it."""

    checkpoint = tmp_path / "training"
    checkpoint.mkdir()
    run_sha = _write(checkpoint / "run.json", {"bindings": {}})
    missing_root = tmp_path / "not-preregistered-on-disk"

    with pytest.raises(ValueError, match="pre-existing directory"):
        retry_module.reserve_initial_wp2_validation(
            project_root=missing_root,
            project_root_device=0,
            project_root_inode=1,
            config_sha256="a" * 64,
            preregistration_sha256="b" * 64,
            checkpoint_dir=checkpoint,
            checkpoint_run_sha256=run_sha,
            output_dir=tmp_path / "validation",
        )

    assert not missing_root.exists()


def test_directory_fsync_failure_leaves_the_validation_slot_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-write durability error must fail closed instead of deleting the claim."""

    checkpoint = tmp_path / "training"
    checkpoint.mkdir()
    run_sha = _write(checkpoint / "run.json", {"bindings": {}})
    real_fsync = retry_module.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(retry_module.os.fstat(descriptor).st_mode):
            raise OSError("simulated directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(retry_module.os, "fsync", fail_directory_fsync)
    kwargs = {
        "project_root": tmp_path,
        **_root_identity(tmp_path),
        "config_sha256": "a" * 64,
        "preregistration_sha256": "b" * 64,
        "checkpoint_dir": checkpoint,
        "checkpoint_run_sha256": run_sha,
        "output_dir": tmp_path / "validation",
    }
    with pytest.raises(OSError, match="simulated directory fsync failure"):
        retry_module.reserve_initial_wp2_validation(**kwargs)

    reservation_path = tmp_path / "wp2_initial_validation_reservation.json"
    assert reservation_path.is_file()
    with pytest.raises(ValueError, match="initial validation slot is already reserved"):
        retry_module.reserve_initial_wp2_validation(**kwargs)


def test_retry_training_run_is_revalidated_after_the_initial_claim_check(
    tmp_path: Path,
) -> None:
    """Falsify a validation path that trusts a mutable run.json after preflight."""

    lineage, checkpoint, _claim, claimed, _run_sha = _completed_retry_training(tmp_path)
    _write(checkpoint / "run.json", {"bindings": {"tampered": True}})

    revalidate = getattr(retry_module, "revalidate_claimed_wp2_retry_training")
    with pytest.raises(ValueError, match="changed after validation reservation"):
        revalidate(
            lineage,
            claimed_training=claimed,
            checkpoint_dir=checkpoint,
        )


def test_retry_validation_completion_reloads_claim_and_records_attempt_audit(
    tmp_path: Path,
) -> None:
    """Falsify completion code that trusts an in-memory claim and omits audit fields."""

    lineage, checkpoint, claim, claimed, run_sha = _completed_retry_training(tmp_path)
    reserve = getattr(retry_module, "reserve_wp2_retry_validation")
    reservation = reserve(
        lineage,
        claimed_training=claimed,
        checkpoint_dir=checkpoint,
        output_dir=tmp_path / "retry-validation",
    )
    output = tmp_path / "retry-validation"
    acceptance_sha = _write(
        output / "wp2_acceptance.json",
        {
            "schema_version": "robustness-sae-wp2-acceptance/v1",
            "passed": True,
            "config_sha256": lineage.authorization.retry_config_sha256,
            "preregistration_sha256": (
                lineage.authorization.retry_preregistration_sha256
            ),
        },
    )
    validation_run = output / "run.json"
    _write(
        validation_run,
        {
            "schema_version": "robustness-sae-validation-run/v1",
            "operation": "validate-sparse-autoencoders",
            "passed": True,
            "acceptance_sha256": acceptance_sha,
            "training_run_sha256": run_sha,
        },
    )

    # The disk claim is the authority. An in-memory claim must not let a changed
    # claim file produce a completion record.
    claim.path.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="global retry claim"):
        retry_module.record_wp2_retry_validation_completion(
            lineage,
            claimed_training=claimed,
            reservation=reservation,
            output_dir=output,
            validation_run_path=validation_run,
            acceptance_path=output / "wp2_acceptance.json",
        )


def test_retry_validation_completion_contains_passed_attempt_and_parent_audit(
    tmp_path: Path,
) -> None:
    lineage, checkpoint, _claim, claimed, run_sha = _completed_retry_training(tmp_path)
    reservation = getattr(retry_module, "reserve_wp2_retry_validation")(
        lineage,
        claimed_training=claimed,
        checkpoint_dir=checkpoint,
        output_dir=tmp_path / "retry-validation",
    )
    output = tmp_path / "retry-validation"
    acceptance_sha = _write(
        output / "wp2_acceptance.json",
        {
            "schema_version": "robustness-sae-wp2-acceptance/v1",
            "passed": False,
            "config_sha256": lineage.authorization.retry_config_sha256,
            "preregistration_sha256": (
                lineage.authorization.retry_preregistration_sha256
            ),
        },
    )
    validation_run = output / "run.json"
    _write(
        validation_run,
        {
            "schema_version": "robustness-sae-validation-run/v1",
            "operation": "validate-sparse-autoencoders",
            "passed": False,
            "acceptance_sha256": acceptance_sha,
            "training_run_sha256": run_sha,
        },
    )
    retry_module.record_wp2_retry_validation_completion(
        lineage,
        claimed_training=claimed,
        reservation=reservation,
        output_dir=output,
        validation_run_path=validation_run,
        acceptance_path=output / "wp2_acceptance.json",
    )

    completion = json.loads(lineage.validation_completion_path.read_text(encoding="utf-8"))
    assert completion["attempt"] == 2
    assert completion["passed"] is False
    assert completion["initial_attempt_ledger_sha256"] == (
        lineage.authorization.initial_attempt_ledger_sha256
    )
    assert completion["training_run_sha256"] == run_sha
    assert completion["validation_reservation_sha256"] == reservation.sha256
