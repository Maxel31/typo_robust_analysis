"""Frozen preregistration for the semantic-subspace causal kill test."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.data.config import strict_loads


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_TOP = {
    "schema_version",
    "model",
    "inputs",
    "subspace",
    "intervention",
    "readout",
    "bootstrap",
    "gates",
}


def _object(value: object, *, field: str, keys: set[str]):
    from collections.abc import Mapping

    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} fields differ")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return float(value)


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class SemanticSubspaceKillProtocol:
    model: str
    model_revision: str
    parent_probe_code_revision: str
    kill_runtime_code_revision: str
    decoder_layers: int
    hidden_size: int
    parent_artifact_sha256: str
    cohort_sha256: str
    pca_manifest_sha256: str
    rank: int
    primary_probe_seed: int
    reproducibility_probe_seeds: tuple[int, int]
    random_basis_seed: int
    complement_basis_seed: int
    transition_layer_source: str
    hook_site: str
    coordinate: str
    patch_direction: str
    operators: tuple[str, ...]
    teacher_forced_tokens: int
    readout_offsets: tuple[int, ...]
    denominator_min_exclusive: float
    minimum_valid: int
    minimum_valid_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_confidence: float
    bootstrap_unit: str
    semantic_full_ratio_lower: float
    control_difference_lower: float
    config_sha256: str


def load_semantic_subspace_kill_config(path: Path) -> SemanticSubspaceKillProtocol:
    """Load the exact, result-independent kill-test contract."""

    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("semantic kill config must be one regular file")
    raw = supplied.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(supplied.resolve()))
    except UnicodeDecodeError as exc:
        raise ValueError("semantic kill config must be UTF-8") from exc
    root = _object(payload, field="semantic kill config", keys=_TOP)
    if root["schema_version"] != "probe-semantic-subspace-kill-config/v2":
        raise ValueError("semantic kill config schema differs")
    model = _object(
        root["model"],
        field="semantic kill model",
        keys={
            "id",
            "revision",
            "parent_probe_code_revision",
            "kill_runtime_code_revision",
            "decoder_layers",
            "hidden_size",
            "dtype",
        },
    )
    if not isinstance(model["id"], str) or not model["id"]:
        raise ValueError("semantic kill model id must be non-empty")
    if not isinstance(model["revision"], str) or _REVISION.fullmatch(model["revision"]) is None:
        raise ValueError("semantic kill model revision must be pinned")
    for field in ("parent_probe_code_revision", "kill_runtime_code_revision"):
        if not isinstance(model[field], str) or _REVISION.fullmatch(model[field]) is None:
            raise ValueError(f"semantic kill {field} must be pinned")
    if model["dtype"] != "bfloat16":
        raise ValueError("semantic kill model dtype differs")
    inputs = _object(
        root["inputs"],
        field="semantic kill inputs",
        keys={
            "parent_probe_artifact_sha256",
            "cohort_manifest_sha256",
            "pca_fit_manifest_sha256",
        },
    )
    subspace = _object(
        root["subspace"],
        field="semantic kill subspace",
        keys={
            "rank",
            "primary_probe_seed",
            "reproducibility_probe_seeds",
            "centering",
            "svd_dtype",
            "basis_sign_rule",
            "random_basis_seed",
            "complement_basis_seed",
        },
    )
    seeds_raw = subspace["reproducibility_probe_seeds"]
    if not isinstance(seeds_raw, list):
        raise ValueError("semantic kill probe seeds must be a list")
    seeds = tuple(_integer(seed, field="semantic kill probe seed") for seed in seeds_raw)
    if seeds != (42, 43):
        raise ValueError("semantic kill requires the frozen probe seeds 42 and 43")
    rank = _integer(subspace["rank"], field="semantic kill rank", minimum=1)
    primary = _integer(subspace["primary_probe_seed"], field="primary probe seed")
    if primary != 42:
        raise ValueError("semantic kill primary probe seed must be 42")
    if (
        rank != 16
        or subspace["centering"] != "subtract-class-mean-weight/v1"
        or subspace["svd_dtype"] != "float64"
        or subspace["basis_sign_rule"] != "largest-absolute-coordinate-positive/v1"
    ):
        raise ValueError("semantic kill subspace derivation differs")
    intervention = _object(
        root["intervention"],
        field="semantic kill intervention",
        keys={
            "transition_layer_source",
            "hook_site",
            "coordinate",
            "patch_direction",
            "operators",
        },
    )
    operators_raw = intervention["operators"]
    if not isinstance(operators_raw, list):
        raise ValueError("semantic kill operators must be a list")
    operators = tuple(operators_raw)
    expected_operators = (
        "untreated",
        "full-state",
        "semantic-rank16",
        "clean-fit-pca-rank16",
        "deterministic-haar-random-rank16",
        "semantic-complement-rank16",
    )
    if operators != expected_operators or any(not isinstance(value, str) for value in operators):
        raise ValueError("semantic kill operator inventory differs")
    identity = {
        "transition_layer_source": "parent-probe-selected-transition/v1",
        "hook_site": "complete-decoder-block-residual-output",
        "coordinate": "edited-word-final-token/v1",
        "patch_direction": "clean-to-typo",
    }
    if any(intervention[field] != expected for field, expected in identity.items()):
        raise ValueError("semantic kill intervention identity differs")
    readout = _object(
        root["readout"],
        field="semantic kill readout",
        keys={
            "teacher_forced_tokens",
            "offsets",
            "metric",
            "denominator_min_exclusive",
            "minimum_valid",
            "minimum_valid_fraction",
        },
    )
    offsets_raw = readout["offsets"]
    if not isinstance(offsets_raw, list):
        raise ValueError("semantic kill readout offsets must be a list")
    offsets = tuple(_integer(value, field="semantic kill offset", minimum=1) for value in offsets_raw)
    if (
        _integer(readout["teacher_forced_tokens"], field="teacher forced tokens") != 16
        or offsets != tuple(range(2, 17))
        or readout["metric"] != "forward-kl-restoration-r2-through-r16/v1"
    ):
        raise ValueError("semantic kill readout differs")
    bootstrap = _object(
        root["bootstrap"],
        field="semantic kill bootstrap",
        keys={"resamples", "seed", "confidence", "unit"},
    )
    gates = _object(
        root["gates"],
        field="semantic kill gates",
        keys={
            "full_ci_lower_strictly_positive",
            "semantic_ci_lower_strictly_positive",
            "semantic_full_ratio_ci_lower",
            "semantic_minus_each_control_ci_lower",
            "both_probe_seeds_must_pass",
        },
    )
    if (
        gates["full_ci_lower_strictly_positive"] is not True
        or gates["semantic_ci_lower_strictly_positive"] is not True
        or gates["both_probe_seeds_must_pass"] is not True
    ):
        raise ValueError("semantic kill boolean gates differ")
    protocol = SemanticSubspaceKillProtocol(
        model=model["id"],
        model_revision=model["revision"],
        parent_probe_code_revision=model["parent_probe_code_revision"],
        kill_runtime_code_revision=model["kill_runtime_code_revision"],
        decoder_layers=_integer(model["decoder_layers"], field="decoder layers", minimum=2),
        hidden_size=_integer(model["hidden_size"], field="hidden size", minimum=1),
        parent_artifact_sha256=_sha(
            inputs["parent_probe_artifact_sha256"], field="parent artifact hash"
        ),
        cohort_sha256=_sha(inputs["cohort_manifest_sha256"], field="cohort hash"),
        pca_manifest_sha256=_sha(
            inputs["pca_fit_manifest_sha256"], field="PCA manifest hash"
        ),
        rank=rank,
        primary_probe_seed=primary,
        reproducibility_probe_seeds=(seeds[0], seeds[1]),
        random_basis_seed=_integer(subspace["random_basis_seed"], field="random basis seed"),
        complement_basis_seed=_integer(
            subspace["complement_basis_seed"], field="complement basis seed"
        ),
        transition_layer_source=intervention["transition_layer_source"],
        hook_site=intervention["hook_site"],
        coordinate=intervention["coordinate"],
        patch_direction=intervention["patch_direction"],
        operators=operators,
        teacher_forced_tokens=16,
        readout_offsets=offsets,
        denominator_min_exclusive=_number(
            readout["denominator_min_exclusive"],
            field="semantic kill denominator",
            minimum=1e-12,
        ),
        minimum_valid=_integer(readout["minimum_valid"], field="minimum valid", minimum=1),
        minimum_valid_fraction=_number(
            readout["minimum_valid_fraction"], field="minimum valid fraction"
        ),
        bootstrap_resamples=_integer(
            bootstrap["resamples"], field="bootstrap resamples", minimum=1
        ),
        bootstrap_seed=_integer(bootstrap["seed"], field="bootstrap seed"),
        bootstrap_confidence=_number(bootstrap["confidence"], field="bootstrap confidence"),
        bootstrap_unit=bootstrap["unit"],
        semantic_full_ratio_lower=_number(
            gates["semantic_full_ratio_ci_lower"], field="semantic/full gate"
        ),
        control_difference_lower=_number(
            gates["semantic_minus_each_control_ci_lower"], field="control difference gate"
        ),
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    if (
        protocol.minimum_valid != 160
        or protocol.minimum_valid_fraction != 0.8
        or protocol.bootstrap_resamples != 10_000
        or protocol.bootstrap_confidence != 0.95
        or protocol.bootstrap_unit != "source-group"
        or protocol.semantic_full_ratio_lower != 0.5
        or protocol.control_difference_lower != 0.0
    ):
        raise ValueError("semantic kill frozen gate settings differ")
    return protocol


__all__ = ["SemanticSubspaceKillProtocol", "load_semantic_subspace_kill_config"]
