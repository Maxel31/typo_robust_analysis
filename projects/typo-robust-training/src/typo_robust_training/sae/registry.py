"""Validation of the committed SAE preregistration artifact."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from typo_robust_training.data.config import strict_loads
from typo_robust_training.sae.config import SaeProtocol
from typo_robust_training.sae.retry import Wp2RetryInputs

if TYPE_CHECKING:
    from typo_robust_training.sae.data import PreparedSaeSources


_FORBIDDEN_DATA_ROLES = [
    "tune",
    "pre_pr_gate",
    "final_test",
    "localization-selection",
    "localization-validation",
]
_PROTECTED_RUNS = [
    "Output distribution matching (Kojima-style baseline), seed 42, 10M student tokens",
    "Causal-window localized state distillation, seed 42, 10M student tokens",
]
_PROHIBITED_CHANGES = [
    "training configuration",
    "rho or lambda_state",
    "data stream",
    "restart or termination",
    "task-accuracy evaluation before completion",
]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATA_CONTRACT_FIELDS = {
    "allowed",
    "forbidden_roles",
    "source_manifest_sha256",
    "current_run_exclusion_rule",
    "initial_eligible_records",
    "initial_eligible_source_tokens",
    "initial_eligible_record_ids_sha256",
    "deduplication_amendment",
    "identity_replay_amendment",
    "selected_record_id_sha256",
}
_DEDUPLICATION_AMENDMENT_FIELDS = {
    "amended_at",
    "trigger",
    "rule",
    "eligible_records_removed",
    "source_manifest_changed",
    "wp2_or_wp5_gate_changed",
}
_IDENTITY_REPLAY_AMENDMENT_FIELDS = {
    "amended_at",
    "trigger",
    "rule",
    "overlapping_record_ids_observed",
    "model_forward_started",
}
_NON_INTERFERENCE_FIELDS = {
    "protected_gpu_ids",
    "sae_gpu_id",
    "gpu_reassignment_amendment",
    "protected_runs",
    "prohibited_changes",
}
_GPU_REASSIGNMENT_AMENDMENT_FIELDS = {
    "amended_at",
    "trigger",
    "from_gpu_id",
    "to_gpu_id",
    "scientific_configuration_changed",
    "wp2_or_wp5_gate_changed",
}
_L1_CALIBRATION_AMENDMENT_FIELDS = {
    "amended_at",
    "trigger",
    "failed_wandb_run_id",
    "original_coefficients",
    "observed_median_l0_by_layer",
    "adjusted_coefficients",
    "original_calibration_tokens",
    "adjusted_calibration_tokens",
    "calibration_shuffle_buffer_activations",
    "calibration_activation_buffers",
    "calibration_activation_batch_size",
    "calibration_tokens_changed",
    "model_forward_completed",
    "wp2_or_wp5_gate_changed",
    "remaining_adjustments",
}
_ORIGINAL_L1_COEFFICIENTS = [0.0001, 0.0003, 0.001]
_ADJUSTED_L1_COEFFICIENTS = [0.01, 0.1, 1.0]
_ORIGINAL_CALIBRATION_TOKENS = 1_000_000
_ADJUSTED_CALIBRATION_TOKENS = 10_000_000
_CALIBRATION_SHUFFLE_BUFFER_ACTIVATIONS = 1_000_000
_CALIBRATION_ACTIVATION_BUFFERS = 10
_CALIBRATION_ACTIVATION_BATCH_SIZE = 2_048
_OBSERVED_MEDIAN_L0 = {
    "5": [5465.0, 5330.0, 4874.0],
    "20": [3890.0, 3881.0, 3856.0],
}


@dataclass(frozen=True, slots=True)
class SaePreregistration:
    path: Path
    sha256: str
    source_manifest_sha256: str
    sae_gpu_id: int
    initial_eligible_records: int
    initial_eligible_source_tokens: int
    initial_eligible_record_ids_sha256: str
    eligible_records_removed: int
    retry_inputs: Wp2RetryInputs | None


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"SAE preregistration {field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"SAE preregistration {field} must be a non-negative integer")
    return value


def validate_sae_prepared_sources(
    prepared: PreparedSaeSources,
    *,
    preregistration: SaePreregistration,
) -> None:
    """Bind derived protected-stream values to their committed preregistration."""

    observed = (
        prepared.protected_eligible_records,
        prepared.protected_eligible_source_tokens,
        prepared.protected_eligible_record_ids_sha256,
        prepared.protected_normalized_duplicates_removed,
    )
    expected = (
        preregistration.initial_eligible_records,
        preregistration.initial_eligible_source_tokens,
        preregistration.initial_eligible_record_ids_sha256,
        preregistration.eligible_records_removed,
    )
    if observed != expected:
        raise ValueError("SAE protected eligible stream differs from preregistration")


def load_sae_preregistration(path: Path, *, protocol: SaeProtocol) -> SaePreregistration:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"SAE preregistration is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("SAE preregistration is not UTF-8") from exc
    common_fields = {
        "schema_version",
        "registered_at",
        "track",
        "non_interference",
        "l1_calibration_amendment",
        "data_contract",
        "wp2_gates",
        "wp3_predictions",
        "wp4_predictions",
        "wp5_gates",
    }
    if not isinstance(payload, Mapping):
        raise ValueError("SAE preregistration fields differ")
    schema_version = payload.get("schema_version")
    expected_fields = (
        common_fields | {"wp2_retry"}
        if schema_version == "robustness-sae-preregistry/v2"
        else common_fields
    )
    if set(payload) != expected_fields:
        raise ValueError("SAE preregistration fields differ")
    if (
        schema_version
        not in {
            "robustness-sae-preregistry/v1",
            "robustness-sae-preregistry/v2",
        }
        or payload.get("track") != "diagnostic-and-future-study-only"
    ):
        raise ValueError("SAE preregistration schema or track differs")
    retry_inputs: Wp2RetryInputs | None = None
    if schema_version == "robustness-sae-preregistry/v2":
        retry = payload["wp2_retry"]
        retry_fields = {
            "authorization_path",
            "initial_attempt_ledger_path",
            "initial_training_dir",
        }
        if not isinstance(retry, Mapping) or set(retry) != retry_fields:
            raise ValueError("SAE WP-2 retry registration differs")

        def registered_path(field: str) -> Path:
            value = retry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SAE WP-2 retry {field} must be a non-empty path")
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = resolved.parent / candidate
            return candidate.resolve()

        retry_inputs = Wp2RetryInputs(
            authorization_path=registered_path("authorization_path"),
            initial_attempt_ledger_path=registered_path("initial_attempt_ledger_path"),
            initial_training_dir=registered_path("initial_training_dir"),
        )
    non_interference = payload["non_interference"]
    if (
        not isinstance(non_interference, Mapping)
        or set(non_interference) != _NON_INTERFERENCE_FIELDS
    ):
        raise ValueError("SAE non-interference registration differs")
    if non_interference.get("protected_gpu_ids") != [5, 6]:
        raise ValueError("SAE GPU non-interference registration differs")
    if (
        non_interference.get("protected_runs") != _PROTECTED_RUNS
        or non_interference.get("prohibited_changes") != _PROHIBITED_CHANGES
    ):
        raise ValueError("SAE non-interference protection lists differ")
    sae_gpu_id = _nonnegative_int(non_interference.get("sae_gpu_id"), field="sae_gpu_id")
    gpu_amendment = non_interference.get("gpu_reassignment_amendment")
    if (
        not isinstance(gpu_amendment, Mapping)
        or set(gpu_amendment) != _GPU_REASSIGNMENT_AMENDMENT_FIELDS
        or not isinstance(gpu_amendment.get("amended_at"), str)
        or not gpu_amendment.get("amended_at")
        or not isinstance(gpu_amendment.get("trigger"), str)
        or not gpu_amendment.get("trigger")
        or _nonnegative_int(
            gpu_amendment.get("from_gpu_id"), field="gpu_reassignment_amendment.from_gpu_id"
        )
        != 1
        or _nonnegative_int(
            gpu_amendment.get("to_gpu_id"), field="gpu_reassignment_amendment.to_gpu_id"
        )
        != sae_gpu_id
        or sae_gpu_id != 0
        or sae_gpu_id in non_interference["protected_gpu_ids"]
        or gpu_amendment.get("scientific_configuration_changed") is not False
        or gpu_amendment.get("wp2_or_wp5_gate_changed") is not False
    ):
        raise ValueError("SAE GPU reassignment amendment differs")
    calibration_amendment = payload["l1_calibration_amendment"]
    if (
        not isinstance(calibration_amendment, Mapping)
        or set(calibration_amendment) != _L1_CALIBRATION_AMENDMENT_FIELDS
        or not isinstance(calibration_amendment.get("amended_at"), str)
        or not calibration_amendment.get("amended_at")
        or not isinstance(calibration_amendment.get("trigger"), str)
        or not calibration_amendment.get("trigger")
        or calibration_amendment.get("failed_wandb_run_id") != "cc36b12ad5150642"
        or calibration_amendment.get("original_coefficients") != _ORIGINAL_L1_COEFFICIENTS
        or calibration_amendment.get("observed_median_l0_by_layer") != _OBSERVED_MEDIAN_L0
        or calibration_amendment.get("adjusted_coefficients") != _ADJUSTED_L1_COEFFICIENTS
        or list(protocol.l1_coefficients) != _ADJUSTED_L1_COEFFICIENTS
        or _positive_int(
            calibration_amendment.get("original_calibration_tokens"),
            field="l1_calibration_amendment.original_calibration_tokens",
        )
        != _ORIGINAL_CALIBRATION_TOKENS
        or _positive_int(
            calibration_amendment.get("adjusted_calibration_tokens"),
            field="l1_calibration_amendment.adjusted_calibration_tokens",
        )
        != _ADJUSTED_CALIBRATION_TOKENS
        or protocol.l1_calibration_tokens != _ADJUSTED_CALIBRATION_TOKENS
        or _positive_int(
            calibration_amendment.get("calibration_shuffle_buffer_activations"),
            field="l1_calibration_amendment.calibration_shuffle_buffer_activations",
        )
        != _CALIBRATION_SHUFFLE_BUFFER_ACTIVATIONS
        or _positive_int(
            calibration_amendment.get("calibration_activation_buffers"),
            field="l1_calibration_amendment.calibration_activation_buffers",
        )
        != _CALIBRATION_ACTIVATION_BUFFERS
        or protocol.shuffle_buffer_activations != _CALIBRATION_SHUFFLE_BUFFER_ACTIVATIONS
        or protocol.l1_calibration_tokens % protocol.shuffle_buffer_activations != 0
        or protocol.l1_calibration_tokens // protocol.shuffle_buffer_activations
        != _CALIBRATION_ACTIVATION_BUFFERS
        or _positive_int(
            calibration_amendment.get("calibration_activation_batch_size"),
            field="l1_calibration_amendment.calibration_activation_batch_size",
        )
        != _CALIBRATION_ACTIVATION_BATCH_SIZE
        or protocol.activation_batch_size != _CALIBRATION_ACTIVATION_BATCH_SIZE
        or calibration_amendment.get("calibration_tokens_changed") is not True
        or calibration_amendment.get("model_forward_completed") is not True
        or calibration_amendment.get("wp2_or_wp5_gate_changed") is not False
        or _nonnegative_int(
            calibration_amendment.get("remaining_adjustments"),
            field="l1_calibration_amendment.remaining_adjustments",
        )
        != 0
    ):
        raise ValueError("SAE L1 calibration amendment differs")
    data = payload["data_contract"]
    if (
        not isinstance(data, Mapping)
        or set(data) != _DATA_CONTRACT_FIELDS
        or data.get("allowed") != "clean FineWeb-Edu train split only"
        or data.get("forbidden_roles") != _FORBIDDEN_DATA_ROLES
    ):
        raise ValueError("SAE data preregistration differs")
    source_sha = data.get("source_manifest_sha256")
    if not isinstance(source_sha, str) or _SHA256.fullmatch(source_sha) is None:
        raise ValueError("SAE source-manifest preregistration differs")
    expected_exclusion = (
        "stable-training-order-prefix/v1:seed="
        f"{protocol.reserved_order_seed}:epoch={protocol.reserved_order_epoch}:"
        f"records={protocol.reserved_prefix_records}"
    )
    if data.get("current_run_exclusion_rule") != expected_exclusion:
        raise ValueError("SAE current-run exclusion registration differs")
    initial_records = _positive_int(
        data.get("initial_eligible_records"),
        field="initial_eligible_records",
    )
    initial_tokens = _positive_int(
        data.get("initial_eligible_source_tokens"),
        field="initial_eligible_source_tokens",
    )
    initial_digest = data.get("initial_eligible_record_ids_sha256")
    if not isinstance(initial_digest, str) or _SHA256.fullmatch(initial_digest) is None:
        raise ValueError("SAE initial eligible record-ID digest differs")
    deduplication = data.get("deduplication_amendment")
    if (
        not isinstance(deduplication, Mapping)
        or set(deduplication) != _DEDUPLICATION_AMENDMENT_FIELDS
        or not isinstance(deduplication.get("amended_at"), str)
        or not deduplication.get("amended_at")
        or not isinstance(deduplication.get("trigger"), str)
        or not deduplication.get("trigger")
        or not isinstance(deduplication.get("rule"), str)
        or not deduplication.get("rule")
        or deduplication.get("source_manifest_changed") is not False
        or deduplication.get("wp2_or_wp5_gate_changed") is not False
    ):
        raise ValueError("SAE deduplication amendment differs")
    removed = _positive_int(
        deduplication.get("eligible_records_removed"),
        field="eligible_records_removed",
    )
    identity_replay = data.get("identity_replay_amendment")
    if (
        not isinstance(identity_replay, Mapping)
        or set(identity_replay) != _IDENTITY_REPLAY_AMENDMENT_FIELDS
        or not isinstance(identity_replay.get("amended_at"), str)
        or not identity_replay.get("amended_at")
        or not isinstance(identity_replay.get("trigger"), str)
        or not identity_replay.get("trigger")
        or not isinstance(identity_replay.get("rule"), str)
        or not identity_replay.get("rule")
        or identity_replay.get("model_forward_started") is not False
    ):
        raise ValueError("SAE identity-replay amendment differs")
    _positive_int(
        identity_replay.get("overlapping_record_ids_observed"),
        field="overlapping_record_ids_observed",
    )
    selected_digest = data.get("selected_record_id_sha256")
    if selected_digest is not None and (
        not isinstance(selected_digest, str) or _SHA256.fullmatch(selected_digest) is None
    ):
        raise ValueError("SAE selected record-ID digest differs")
    wp2 = payload["wp2_gates"]
    if not isinstance(wp2, Mapping) or (
        wp2.get("fvu_max") != protocol.fvu_max
        or wp2.get("median_l0_inclusive") != list(protocol.median_l0_range)
        or wp2.get("dead_feature_probability_below") != protocol.dead_feature_probability_below
        or wp2.get("dead_feature_rate_max") != protocol.dead_feature_rate_max
        or wp2.get("splice_documents") != protocol.splice_documents
        or wp2.get("splice_kl_median_max_nats_per_token") != protocol.splice_kl_max
        or wp2.get("maximum_retrains_after_failure") != protocol.maximum_gate_retrains
    ):
        raise ValueError("SAE WP-2 preregistered gates differ from config")
    wp5 = payload["wp5_gates"]
    if not isinstance(wp5, Mapping) or (
        wp5.get("G1_feature_sufficiency")
        != f"median(R_z) >= {protocol.wp5_feature_sufficiency_ratio:.2f} * median(R_full)"
        or wp5.get("G2_spurious_suppression")
        != f"median(R_sup) >= {protocol.wp5_suppression_ratio:.2f} * median(R_full)"
    ):
        raise ValueError("SAE WP-5 preregistered gates differ from config")
    return SaePreregistration(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        source_manifest_sha256=source_sha,
        sae_gpu_id=sae_gpu_id,
        initial_eligible_records=initial_records,
        initial_eligible_source_tokens=initial_tokens,
        initial_eligible_record_ids_sha256=initial_digest,
        eligible_records_removed=removed,
        retry_inputs=retry_inputs,
    )


__all__ = [
    "SaePreregistration",
    "load_sae_preregistration",
    "validate_sae_prepared_sources",
]
