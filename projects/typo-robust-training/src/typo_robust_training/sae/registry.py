"""Validation of the committed SAE preregistration artifact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from typo_robust_training.data.config import strict_loads
from typo_robust_training.sae.config import SaeProtocol


@dataclass(frozen=True, slots=True)
class SaePreregistration:
    path: Path
    sha256: str
    source_manifest_sha256: str
    sae_gpu_id: int


def load_sae_preregistration(path: Path, *, protocol: SaeProtocol) -> SaePreregistration:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"SAE preregistration is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("SAE preregistration is not UTF-8") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "registered_at",
        "track",
        "non_interference",
        "data_contract",
        "wp2_gates",
        "wp3_predictions",
        "wp4_predictions",
        "wp5_gates",
    }:
        raise ValueError("SAE preregistration fields differ")
    if (
        payload.get("schema_version") != "robustness-sae-preregistry/v1"
        or payload.get("track") != "diagnostic-and-future-study-only"
    ):
        raise ValueError("SAE preregistration schema or track differs")
    non_interference = payload["non_interference"]
    if not isinstance(non_interference, Mapping):
        raise ValueError("SAE non-interference registration differs")
    if (
        non_interference.get("protected_gpu_ids") != [5, 6]
        or non_interference.get("sae_gpu_id") != 1
    ):
        raise ValueError("SAE GPU non-interference registration differs")
    data = payload["data_contract"]
    if not isinstance(data, Mapping) or data.get("allowed") != "clean FineWeb-Edu train split only":
        raise ValueError("SAE data preregistration differs")
    source_sha = data.get("source_manifest_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("SAE source-manifest preregistration differs")
    expected_exclusion = (
        "stable-training-order-prefix/v1:seed="
        f"{protocol.reserved_order_seed}:epoch={protocol.reserved_order_epoch}:"
        f"records={protocol.reserved_prefix_records}"
    )
    if data.get("current_run_exclusion_rule") != expected_exclusion:
        raise ValueError("SAE current-run exclusion registration differs")
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
        sae_gpu_id=1,
    )


__all__ = ["SaePreregistration", "load_sae_preregistration"]
