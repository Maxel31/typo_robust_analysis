"""Project-global, fail-closed lineage for the single allowed WP-2 retrain."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from typo_robust_training.data.config import strict_loads
from typo_robust_training.sae.data import sha256_file


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLAIM_NAME = "wp2_retry_claim.json"
_TRAINING_COMPLETION_NAME = "wp2_retry_training_completion.json"
_INITIAL_VALIDATION_RESERVATION_NAME = "wp2_initial_validation_reservation.json"
_RETRY_VALIDATION_RESERVATION_NAME = "wp2_retry_validation_reservation.json"
_VALIDATION_COMPLETION_NAME = "wp2_retry_validation_completion.json"


@dataclass(frozen=True, slots=True)
class Wp2RetryInputs:
    authorization_path: Path
    initial_attempt_ledger_path: Path
    initial_training_dir: Path


@dataclass(frozen=True, slots=True)
class Wp2RetryAuthorization:
    path: Path
    sha256: str
    project_id: str
    initial_config_sha256: str
    initial_preregistration_sha256: str
    initial_attempt_ledger_sha256: str
    initial_training_run_sha256: str
    initial_validation_run_sha256: str
    initial_acceptance_sha256: str
    retry_config_sha256: str
    retry_preregistration_sha256: str


@dataclass(frozen=True, slots=True)
class Wp2RetryLineage:
    authorization: Wp2RetryAuthorization
    project_root: Path
    claim_path: Path
    training_completion_path: Path
    validation_completion_path: Path


@dataclass(frozen=True, slots=True)
class Wp2RetryClaim:
    path: Path
    sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Wp2ClaimedRetryTraining:
    claim: Wp2RetryClaim
    training_run_path: Path
    training_run_sha256: str


@dataclass(frozen=True, slots=True)
class Wp2ValidationReservation:
    path: Path
    sha256: str
    payload: Mapping[str, object]


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"WP-2 retry authorization {field} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, field: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"WP-2 retry authorization {field} fields differ")
    return value


def load_wp2_retry_authorization(path: Path) -> Wp2RetryAuthorization:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"WP-2 retry authorization is not a file: {resolved}")
    raw = resolved.read_bytes()
    payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    root = _mapping(
        payload,
        field="root",
        keys={
            "schema_version",
            "project_id",
            "maximum_full_retrain_bundles",
            "initial_failure",
            "retry",
        },
    )
    if (
        root.get("schema_version") != "robustness-sae-wp2-retry-authorization/v1"
        or root.get("maximum_full_retrain_bundles") != 1
        or not isinstance(root.get("project_id"), str)
        or not root.get("project_id")
    ):
        raise ValueError("WP-2 retry authorization root differs")
    initial = _mapping(
        root["initial_failure"],
        field="initial_failure",
        keys={
            "config_sha256",
            "preregistration_sha256",
            "attempt_ledger_sha256",
            "training_run_sha256",
            "validation_run_sha256",
            "acceptance_sha256",
        },
    )
    retry = _mapping(
        root["retry"],
        field="retry",
        keys={"config_sha256", "preregistration_sha256"},
    )
    return Wp2RetryAuthorization(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        project_id=str(root["project_id"]),
        initial_config_sha256=_sha256(initial["config_sha256"], field="initial config"),
        initial_preregistration_sha256=_sha256(
            initial["preregistration_sha256"], field="initial preregistration"
        ),
        initial_attempt_ledger_sha256=_sha256(
            initial["attempt_ledger_sha256"], field="initial attempt ledger"
        ),
        initial_training_run_sha256=_sha256(
            initial["training_run_sha256"], field="initial training run"
        ),
        initial_validation_run_sha256=_sha256(
            initial["validation_run_sha256"], field="initial validation run"
        ),
        initial_acceptance_sha256=_sha256(
            initial["acceptance_sha256"], field="initial acceptance"
        ),
        retry_config_sha256=_sha256(retry["config_sha256"], field="retry config"),
        retry_preregistration_sha256=_sha256(
            retry["preregistration_sha256"], field="retry preregistration"
        ),
    )


def _load_object(path: Path, *, description: str) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"WP-2 retry {description} is missing: {path}")
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"WP-2 retry {description} must be an object")
    return payload


def load_wp2_retry_lineage(
    inputs: Wp2RetryInputs,
    *,
    retry_config_sha256: str,
    retry_preregistration_sha256: str,
) -> Wp2RetryLineage:
    """Verify the complete failed-attempt chain and derive its one global root."""

    authorization = load_wp2_retry_authorization(inputs.authorization_path)
    if (
        retry_config_sha256 != authorization.retry_config_sha256
        or retry_preregistration_sha256 != authorization.retry_preregistration_sha256
    ):
        raise ValueError("WP-2 retry config/preregistration differs from authorization")

    ledger_path = Path(inputs.initial_attempt_ledger_path).resolve()
    if not ledger_path.is_file():
        raise ValueError(f"WP-2 retry initial attempt ledger is missing: {ledger_path}")
    if sha256_file(ledger_path) != authorization.initial_attempt_ledger_sha256:
        raise ValueError("WP-2 initial attempt ledger hash differs from authorization")
    ledger = _load_object(ledger_path, description="initial attempt ledger")
    if (
        ledger.get("schema_version") != "robustness-sae-wp2-attempt-ledger/v1"
        or ledger.get("config_sha256") != authorization.initial_config_sha256
        or ledger.get("preregistration_sha256")
        != authorization.initial_preregistration_sha256
        or ledger.get("maximum_retrains_after_failure") != 1
    ):
        raise ValueError("WP-2 initial attempt ledger lineage differs")
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError("WP-2 retry requires exactly one initial validation attempt")
    attempt = attempts[0]
    if (
        not isinstance(attempt, Mapping)
        or attempt.get("attempt") != 1
        or attempt.get("passed") is not False
        or attempt.get("checkpoint_run_sha256")
        != authorization.initial_training_run_sha256
        or attempt.get("acceptance_sha256") != authorization.initial_acceptance_sha256
        or not isinstance(attempt.get("output_dir"), str)
    ):
        raise ValueError("WP-2 retry parent attempt is not the authorized failed bundle")

    project_root = ledger_path.parent
    validation_dir = Path(str(attempt["output_dir"])).resolve()
    if validation_dir.parent != project_root:
        raise ValueError("WP-2 initial ledger is outside its recorded project root")
    acceptance_path = validation_dir / "wp2_acceptance.json"
    validation_run_path = validation_dir / "run.json"
    initial_training_run_path = Path(inputs.initial_training_dir).resolve() / "run.json"
    for path, description in (
        (acceptance_path, "initial acceptance"),
        (validation_run_path, "initial validation run"),
        (initial_training_run_path, "initial training run"),
    ):
        if not path.is_file():
            raise ValueError(f"WP-2 retry {description} is missing: {path}")
    if (
        sha256_file(acceptance_path) != authorization.initial_acceptance_sha256
        or sha256_file(validation_run_path) != authorization.initial_validation_run_sha256
        or sha256_file(initial_training_run_path)
        != authorization.initial_training_run_sha256
    ):
        raise ValueError("WP-2 initial failure artifact hash chain differs")
    acceptance = _load_object(acceptance_path, description="initial acceptance")
    validation_run = _load_object(validation_run_path, description="initial validation run")
    training_run = _load_object(initial_training_run_path, description="initial training run")
    training_bindings = training_run.get("bindings")
    if (
        acceptance.get("passed") is not False
        or acceptance.get("config_sha256") != authorization.initial_config_sha256
        or acceptance.get("preregistration_sha256")
        != authorization.initial_preregistration_sha256
        or validation_run.get("passed") is not False
        or validation_run.get("acceptance_sha256")
        != authorization.initial_acceptance_sha256
        or validation_run.get("training_run_sha256")
        != authorization.initial_training_run_sha256
        or not isinstance(training_bindings, Mapping)
        or training_bindings.get("config_sha256") != authorization.initial_config_sha256
        or training_bindings.get("preregistration_sha256")
        != authorization.initial_preregistration_sha256
    ):
        raise ValueError("WP-2 initial failure artifact lineage differs")
    return Wp2RetryLineage(
        authorization=authorization,
        project_root=project_root,
        claim_path=project_root / _CLAIM_NAME,
        training_completion_path=project_root / _TRAINING_COMPLETION_NAME,
        validation_completion_path=project_root / _VALIDATION_COMPLETION_NAME,
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: object) -> None:
    data = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def claim_wp2_retry_training(
    lineage: Wp2RetryLineage,
    *,
    output_dir: Path,
    training_bindings: Mapping[str, object],
    resume: bool,
) -> Wp2RetryClaim:
    """Consume the sole retry slot before model/runtime initialization."""

    if (
        training_bindings.get("config_sha256")
        != lineage.authorization.retry_config_sha256
        or training_bindings.get("preregistration_sha256")
        != lineage.authorization.retry_preregistration_sha256
    ):
        raise ValueError("SAE WP-2 retry training bindings differ from authorization")
    payload = {
        "schema_version": "robustness-sae-wp2-retry-claim/v1",
        "project_id": lineage.authorization.project_id,
        "attempt": 2,
        "authorization_sha256": lineage.authorization.sha256,
        "initial_attempt_ledger_sha256": (
            lineage.authorization.initial_attempt_ledger_sha256
        ),
        "training_output_dir": str(Path(output_dir).resolve()),
        "training_bindings": dict(training_bindings),
    }
    if resume:
        if not lineage.claim_path.is_file():
            raise ValueError("SAE WP-2 retry resume requires the pre-training global claim")
        observed = _load_object(lineage.claim_path, description="global retry claim")
        if observed != payload:
            raise ValueError("SAE WP-2 retry resume differs from the existing global claim")
        return Wp2RetryClaim(
            path=lineage.claim_path,
            sha256=sha256_file(lineage.claim_path),
            payload=payload,
        )
    try:
        _write_exclusive(lineage.claim_path, payload)
    except FileExistsError:
        raise ValueError("SAE WP-2 project-global retrain limit is exhausted") from None
    digest = sha256_file(lineage.claim_path)
    return Wp2RetryClaim(path=lineage.claim_path, sha256=digest, payload=payload)


def _verified_claim(lineage: Wp2RetryLineage) -> Wp2RetryClaim:
    payload = _load_object(lineage.claim_path, description="global retry claim")
    expected_keys = {
        "schema_version",
        "project_id",
        "attempt",
        "authorization_sha256",
        "initial_attempt_ledger_sha256",
        "training_output_dir",
        "training_bindings",
    }
    bindings = payload.get("training_bindings")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != "robustness-sae-wp2-retry-claim/v1"
        or payload.get("project_id") != lineage.authorization.project_id
        or payload.get("attempt") != 2
        or payload.get("authorization_sha256") != lineage.authorization.sha256
        or payload.get("initial_attempt_ledger_sha256")
        != lineage.authorization.initial_attempt_ledger_sha256
        or not isinstance(payload.get("training_output_dir"), str)
        or not isinstance(bindings, Mapping)
        or bindings.get("config_sha256") != lineage.authorization.retry_config_sha256
        or bindings.get("preregistration_sha256")
        != lineage.authorization.retry_preregistration_sha256
    ):
        raise ValueError("WP-2 global retry claim lineage differs")
    return Wp2RetryClaim(
        path=lineage.claim_path,
        sha256=sha256_file(lineage.claim_path),
        payload=payload,
    )


def record_wp2_retry_training_completion(
    lineage: Wp2RetryLineage,
    *,
    claim: Wp2RetryClaim,
    training_run_path: Path,
) -> None:
    observed_claim = _verified_claim(lineage)
    if claim.sha256 != observed_claim.sha256 or claim.payload != observed_claim.payload:
        raise ValueError("WP-2 retry training claim differs from the global claim")
    run_path = Path(training_run_path).resolve()
    run = _load_object(run_path, description="retry training run")
    bindings = run.get("bindings")
    claimed_bindings = claim.payload.get("training_bindings")
    expected_bindings = {
        **(dict(claimed_bindings) if isinstance(claimed_bindings, Mapping) else {}),
        "wp2_retry_claim_sha256": claim.sha256,
    }
    if (
        not isinstance(bindings, Mapping)
        or dict(bindings) != expected_bindings
        or run_path.parent != Path(str(claim.payload["training_output_dir"])).resolve()
    ):
        raise ValueError("WP-2 retry training run differs from the global claim")
    payload = {
        "schema_version": "robustness-sae-wp2-retry-training-completion/v1",
        "project_id": lineage.authorization.project_id,
        "claim_sha256": claim.sha256,
        "training_run_path": str(run_path),
        "training_run_sha256": sha256_file(run_path),
    }
    try:
        _write_exclusive(lineage.training_completion_path, payload)
    except FileExistsError:
        if _load_object(
            lineage.training_completion_path,
            description="retry training completion",
        ) != payload:
            raise ValueError("WP-2 retry training completion differs") from None


def require_claimed_wp2_retry_training(
    lineage: Wp2RetryLineage,
    *,
    checkpoint_dir: Path,
) -> Wp2ClaimedRetryTraining:
    claim = _verified_claim(lineage)
    claim_payload = claim.payload
    completion = _load_object(
        lineage.training_completion_path,
        description="retry training completion",
    )
    run_path = Path(checkpoint_dir).resolve() / "run.json"
    if not run_path.is_file():
        raise ValueError(f"WP-2 retry training run is missing: {run_path}")
    completion_keys = {
        "schema_version",
        "project_id",
        "claim_sha256",
        "training_run_path",
        "training_run_sha256",
    }
    if (
        set(completion) != completion_keys
        or completion.get("schema_version")
        != "robustness-sae-wp2-retry-training-completion/v1"
        or completion.get("project_id") != lineage.authorization.project_id
        or claim_payload.get("training_output_dir") != str(Path(checkpoint_dir).resolve())
        or completion.get("claim_sha256") != claim.sha256
        or completion.get("training_run_path") != str(run_path)
        or completion.get("training_run_sha256") != sha256_file(run_path)
    ):
        raise ValueError("WP-2 validation checkpoint is not the claimed retry training run")
    run = _load_object(run_path, description="retry training run")
    bindings = run.get("bindings")
    claimed_bindings = claim_payload.get("training_bindings")
    expected_bindings = {
        **(dict(claimed_bindings) if isinstance(claimed_bindings, Mapping) else {}),
        "wp2_retry_claim_sha256": claim.sha256,
    }
    if not isinstance(bindings, Mapping) or dict(bindings) != expected_bindings:
        raise ValueError("WP-2 validation checkpoint bindings differ from the global claim")
    return Wp2ClaimedRetryTraining(
        claim=claim,
        training_run_path=run_path,
        training_run_sha256=str(completion["training_run_sha256"]),
    )


def revalidate_claimed_wp2_retry_training(
    lineage: Wp2RetryLineage,
    *,
    claimed_training: Wp2ClaimedRetryTraining,
    checkpoint_dir: Path,
) -> None:
    """Fail closed when the claim or exact training run changed after preflight."""

    try:
        observed = require_claimed_wp2_retry_training(
            lineage,
            checkpoint_dir=checkpoint_dir,
        )
    except ValueError as exc:
        raise ValueError(
            "WP-2 retry training changed after validation reservation"
        ) from exc
    if observed != claimed_training:
        raise ValueError("WP-2 retry training changed after validation reservation")


def reserve_initial_wp2_validation(
    *,
    project_root: Path,
    config_sha256: str,
    preregistration_sha256: str,
    checkpoint_dir: Path,
    checkpoint_run_sha256: str,
    output_dir: Path,
) -> Wp2ValidationReservation:
    """Consume the initial validation slot before outputs, GPU, or runtime work."""

    path = Path(project_root).resolve() / _INITIAL_VALIDATION_RESERVATION_NAME
    payload = {
        "schema_version": "robustness-sae-wp2-validation-reservation/v1",
        "attempt": 1,
        "config_sha256": config_sha256,
        "preregistration_sha256": preregistration_sha256,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "checkpoint_run_sha256": checkpoint_run_sha256,
        "output_dir": str(Path(output_dir).resolve()),
    }
    try:
        _write_exclusive(path, payload)
    except FileExistsError:
        raise ValueError("SAE WP-2 initial validation slot is already reserved") from None
    return Wp2ValidationReservation(path=path, sha256=sha256_file(path), payload=payload)


def reserve_wp2_retry_validation(
    lineage: Wp2RetryLineage,
    *,
    claimed_training: Wp2ClaimedRetryTraining,
    checkpoint_dir: Path,
    output_dir: Path,
) -> Wp2ValidationReservation:
    """Consume the retry validation slot before outputs, GPU, or runtime work."""

    revalidate_claimed_wp2_retry_training(
        lineage,
        claimed_training=claimed_training,
        checkpoint_dir=checkpoint_dir,
    )
    path = lineage.project_root / _RETRY_VALIDATION_RESERVATION_NAME
    payload = {
        "schema_version": "robustness-sae-wp2-validation-reservation/v1",
        "project_id": lineage.authorization.project_id,
        "attempt": 2,
        "authorization_sha256": lineage.authorization.sha256,
        "claim_sha256": claimed_training.claim.sha256,
        "initial_attempt_ledger_sha256": (
            lineage.authorization.initial_attempt_ledger_sha256
        ),
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "training_run_sha256": claimed_training.training_run_sha256,
        "output_dir": str(Path(output_dir).resolve()),
    }
    try:
        _write_exclusive(path, payload)
    except FileExistsError:
        raise ValueError("SAE WP-2 retry validation slot is already reserved") from None
    return Wp2ValidationReservation(path=path, sha256=sha256_file(path), payload=payload)


def verify_initial_wp2_validation_reservation(
    reservation: Wp2ValidationReservation,
    *,
    checkpoint_dir: Path,
) -> None:
    observed = _load_object(reservation.path, description="initial validation reservation")
    if (
        observed != reservation.payload
        or sha256_file(reservation.path) != reservation.sha256
        or reservation.payload.get("attempt") != 1
        or reservation.payload.get("checkpoint_dir")
        != str(Path(checkpoint_dir).resolve())
        or reservation.payload.get("checkpoint_run_sha256")
        != sha256_file(Path(checkpoint_dir).resolve() / "run.json")
    ):
        raise ValueError("SAE WP-2 initial validation reservation changed during validation")


def _verify_retry_validation_reservation(
    lineage: Wp2RetryLineage,
    *,
    claimed_training: Wp2ClaimedRetryTraining,
    reservation: Wp2ValidationReservation,
    output_dir: Path,
) -> None:
    observed = _load_object(reservation.path, description="retry validation reservation")
    if (
        reservation.path != lineage.project_root / _RETRY_VALIDATION_RESERVATION_NAME
        or observed != reservation.payload
        or sha256_file(reservation.path) != reservation.sha256
        or reservation.payload.get("attempt") != 2
        or reservation.payload.get("claim_sha256") != claimed_training.claim.sha256
        or reservation.payload.get("training_run_sha256")
        != claimed_training.training_run_sha256
        or reservation.payload.get("output_dir") != str(Path(output_dir).resolve())
    ):
        raise ValueError("SAE WP-2 retry validation reservation changed during validation")


def record_wp2_retry_validation_completion(
    lineage: Wp2RetryLineage,
    *,
    claimed_training: Wp2ClaimedRetryTraining,
    reservation: Wp2ValidationReservation,
    output_dir: Path,
    validation_run_path: Path,
    acceptance_path: Path,
) -> None:
    observed_claim = _verified_claim(lineage)
    if observed_claim != claimed_training.claim:
        raise ValueError("WP-2 global retry claim changed before validation completion")
    revalidate_claimed_wp2_retry_training(
        lineage,
        claimed_training=claimed_training,
        checkpoint_dir=claimed_training.training_run_path.parent,
    )
    _verify_retry_validation_reservation(
        lineage,
        claimed_training=claimed_training,
        reservation=reservation,
        output_dir=output_dir,
    )
    resolved_output = Path(output_dir).resolve()
    run_path = Path(validation_run_path).resolve()
    resolved_acceptance = Path(acceptance_path).resolve()
    if run_path.parent != resolved_output or resolved_acceptance.parent != resolved_output:
        raise ValueError("WP-2 retry validation artifacts are outside the reserved output")
    validation_run = _load_object(run_path, description="retry validation run")
    acceptance = _load_object(resolved_acceptance, description="retry acceptance")
    passed = acceptance.get("passed")
    claimed_bindings = claimed_training.claim.payload.get("training_bindings")
    if (
        type(passed) is not bool
        or acceptance.get("schema_version") != "robustness-sae-wp2-acceptance/v1"
        or not isinstance(claimed_bindings, Mapping)
        or acceptance.get("config_sha256") != claimed_bindings.get("config_sha256")
        or acceptance.get("preregistration_sha256")
        != claimed_bindings.get("preregistration_sha256")
        or validation_run.get("schema_version")
        != "robustness-sae-validation-run/v1"
        or validation_run.get("operation") != "validate-sparse-autoencoders"
        or validation_run.get("passed") is not passed
        or validation_run.get("acceptance_sha256") != sha256_file(resolved_acceptance)
        or validation_run.get("training_run_sha256")
        != claimed_training.training_run_sha256
    ):
        raise ValueError("WP-2 retry validation result lineage differs")
    payload = {
        "schema_version": "robustness-sae-wp2-retry-validation-completion/v1",
        "project_id": lineage.authorization.project_id,
        "attempt": 2,
        "passed": passed,
        "authorization_sha256": lineage.authorization.sha256,
        "claim_sha256": claimed_training.claim.sha256,
        "initial_attempt_ledger_sha256": (
            lineage.authorization.initial_attempt_ledger_sha256
        ),
        "training_run_sha256": claimed_training.training_run_sha256,
        "validation_reservation_sha256": reservation.sha256,
        "output_dir": str(resolved_output),
        "validation_run_sha256": sha256_file(run_path),
        "acceptance_sha256": sha256_file(resolved_acceptance),
    }
    # Narrow the mutable-run TOCTOU window to the final exclusive record write.
    revalidate_claimed_wp2_retry_training(
        lineage,
        claimed_training=claimed_training,
        checkpoint_dir=claimed_training.training_run_path.parent,
    )
    if _verified_claim(lineage) != claimed_training.claim:
        raise ValueError("WP-2 global retry claim changed before validation completion")
    try:
        _write_exclusive(lineage.validation_completion_path, payload)
    except FileExistsError:
        raise ValueError("SAE WP-2 retry validation was already completed") from None


__all__ = [
    "Wp2ClaimedRetryTraining",
    "Wp2RetryAuthorization",
    "Wp2RetryClaim",
    "Wp2RetryInputs",
    "Wp2RetryLineage",
    "Wp2ValidationReservation",
    "claim_wp2_retry_training",
    "load_wp2_retry_authorization",
    "load_wp2_retry_lineage",
    "record_wp2_retry_training_completion",
    "record_wp2_retry_validation_completion",
    "reserve_initial_wp2_validation",
    "reserve_wp2_retry_validation",
    "revalidate_claimed_wp2_retry_training",
    "require_claimed_wp2_retry_training",
    "verify_initial_wp2_validation_reservation",
]
