"""Strict, preregistered configuration for the SAE diagnostic track."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from typo_robust_training.data.config import strict_loads


_REVISION = re.compile(r"[0-9a-f]{40}")


def _object(value: object, *, field: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} fields differ")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be a finite positive number")
    return float(value)


def _probability(value: object, *, field: str) -> float:
    result = _positive_float(value, field=field)
    if result > 1.0:
        raise ValueError(f"{field} must be at most one")
    return result


def _integer_list(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(_nonnegative_int(item, field=field) for item in value)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{field} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class SaeProtocol:
    schema_version: str
    model: str
    model_revision: str
    model_dtype: str
    d_model: int
    decoder_layers: int
    probe_layers: tuple[int, ...]
    activation_subsample_layers: tuple[int, ...]
    source: str
    source_dataset: str
    source_subset: str
    source_revision: str
    source_split: str
    max_sequence_length: int
    document_character_limit: int
    near_duplicate_shingle_size: int
    near_duplicate_jaccard_threshold: float
    supplement_stream_order: str
    minimum_training_tokens: int
    preferred_training_tokens: int
    statistics_tokens: int
    activation_subsample_tokens: int
    shuffle_buffer_activations: int
    reserved_order_seed: int
    reserved_order_epoch: int
    reserved_prefix_records: int
    protected_student_token_budget: int
    expansion_factor: int
    activation: str
    regularizer: str
    l1_coefficients: tuple[float, ...]
    l1_calibration_tokens: int
    l1_selection_rule: str
    optimizer: str
    learning_rate: float
    adam_betas: tuple[float, float]
    adam_epsilon: float
    precision: str
    activation_batch_size: int
    decoder_normalization: str
    seeds_by_layer: Mapping[int, tuple[int, ...]]
    fvu_max: float
    median_l0_range: tuple[int, int]
    dead_feature_probability_below: float
    dead_feature_rate_max: float
    splice_documents: int
    splice_kl_max: float
    maximum_gate_retrains: int
    wp5_feature_sufficiency_ratio: float
    wp5_suppression_ratio: float
    wp5_requires_layer5_seed_replication: bool
    bootstrap_replicates: int
    bootstrap_seed: int
    config_path: Path
    config_sha256: str

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor


def load_sae_protocol(path: Path) -> SaeProtocol:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"SAE config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("SAE config is not UTF-8") from exc
    root = _object(
        payload,
        field="SAE config",
        keys={
            "schema_version",
            "model",
            "data",
            "sae",
            "acceptance",
            "feature_kill_test",
        },
    )
    if root["schema_version"] != "robustness-sae-training-config/v1":
        raise ValueError("SAE config schema differs")

    model = _object(
        root["model"],
        field="model",
        keys={
            "id",
            "revision",
            "dtype",
            "d_model",
            "decoder_layers",
            "probe_layers",
            "activation_subsample_layers",
        },
    )
    if not isinstance(model["id"], str) or not model["id"]:
        raise ValueError("model.id must be non-empty")
    if not isinstance(model["revision"], str) or _REVISION.fullmatch(model["revision"]) is None:
        raise ValueError("model.revision must be a pinned lowercase commit SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("model.dtype must be bfloat16")
    d_model = _positive_int(model["d_model"], field="model.d_model")
    decoder_layers = _positive_int(model["decoder_layers"], field="model.decoder_layers")
    probe_layers = _integer_list(model["probe_layers"], field="model.probe_layers")
    subsample_layers = _integer_list(
        model["activation_subsample_layers"], field="model.activation_subsample_layers"
    )
    if probe_layers != (5, 20) or subsample_layers != (5, 11, 20, 26):
        raise ValueError("SAE probe/subsample layers differ from preregistration")
    if any(layer >= decoder_layers for layer in (*probe_layers, *subsample_layers)):
        raise ValueError("SAE layer is outside the model")

    data = _object(
        root["data"],
        field="data",
        keys={
            "source",
            "dataset",
            "subset",
            "revision",
            "split",
            "clean_only",
            "max_sequence_length",
            "document_character_limit",
            "near_duplicate_shingle_size",
            "near_duplicate_jaccard_threshold",
            "supplement_stream_order",
            "minimum_training_tokens",
            "preferred_training_tokens",
            "statistics_tokens",
            "activation_subsample_tokens",
            "shuffle_buffer_activations",
            "current_run_exclusion",
        },
    )
    if (
        data["source"] != "fineweb_edu"
        or data["dataset"] != "HuggingFaceFW/fineweb-edu"
        or data["split"] != "train"
        or data["clean_only"] is not True
    ):
        raise ValueError("SAE data must be clean FineWeb-Edu train only")
    if not isinstance(data["subset"], str) or not data["subset"]:
        raise ValueError("data.subset must be non-empty")
    if not isinstance(data["revision"], str) or _REVISION.fullmatch(data["revision"]) is None:
        raise ValueError("data.revision must be a pinned lowercase commit SHA")
    if data["supplement_stream_order"] != "pinned-unshuffled-stream/v1":
        raise ValueError("SAE supplement stream order differs")
    minimum_tokens = _positive_int(
        data["minimum_training_tokens"], field="data.minimum_training_tokens"
    )
    preferred_tokens = _positive_int(
        data["preferred_training_tokens"], field="data.preferred_training_tokens"
    )
    if preferred_tokens < minimum_tokens:
        raise ValueError("preferred SAE token count cannot be below the minimum")
    exclusion = _object(
        data["current_run_exclusion"],
        field="data.current_run_exclusion",
        keys={"rule", "seed", "epoch", "reserved_records", "declared_student_token_budget"},
    )
    if exclusion["rule"] != "stable-training-order-prefix/v1":
        raise ValueError("SAE current-run exclusion rule differs")

    sae = _object(
        root["sae"],
        field="sae",
        keys={
            "expansion_factor",
            "activation",
            "regularizer",
            "l1_coefficients",
            "l1_calibration_tokens",
            "l1_selection_rule",
            "optimizer",
            "learning_rate",
            "adam_betas",
            "adam_epsilon",
            "precision",
            "activation_batch_size",
            "decoder_normalization",
            "seeds_by_layer",
        },
    )
    if sae["activation"] != "relu":
        raise ValueError("SAE activation must be relu")
    if sae["regularizer"] != "l1":
        raise ValueError("SAE regularizer must be l1")
    if sae["l1_selection_rule"] != "in-range-median-l0-then-lowest-fvu/v1":
        raise ValueError("SAE L1 selection rule differs")
    if sae["optimizer"] != "adam" or sae["precision"] != "float32":
        raise ValueError("SAE optimizer/precision differs")
    betas = sae["adam_betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) < 1.0
            for value in betas
        )
        or float(betas[0]) >= float(betas[1])
    ):
        raise ValueError("SAE Adam betas must be increasing values in [0, 1)")
    if sae["decoder_normalization"] != "unit-column-every-step/v1":
        raise ValueError("SAE decoder normalization differs")
    coefficients = sae["l1_coefficients"]
    if not isinstance(coefficients, list) or len(coefficients) != 3:
        raise ValueError("SAE L1 coefficient grid must contain exactly three values")
    l1_coefficients = tuple(
        _positive_float(value, field="sae.l1_coefficients") for value in coefficients
    )
    if tuple(sorted(set(l1_coefficients))) != l1_coefficients:
        raise ValueError("SAE L1 coefficients must be sorted and unique")
    l1_calibration_tokens = _positive_int(
        sae["l1_calibration_tokens"], field="sae.l1_calibration_tokens"
    )
    shuffle_buffer_activations = _positive_int(
        data["shuffle_buffer_activations"], field="data.shuffle_buffer_activations"
    )
    if l1_calibration_tokens > shuffle_buffer_activations:
        raise ValueError("SAE L1 calibration must fit in one frozen shuffle buffer")
    seeds_raw = sae["seeds_by_layer"]
    if not isinstance(seeds_raw, Mapping) or set(seeds_raw) != {"5", "20"}:
        raise ValueError("SAE layer seeds differ")
    seeds_by_layer: dict[int, tuple[int, ...]] = {}
    for layer_text, values in seeds_raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError("SAE seeds must be non-empty lists")
        seeds = tuple(_nonnegative_int(value, field="SAE seed") for value in values)
        if len(set(seeds)) != len(seeds):
            raise ValueError("SAE seeds must be unique")
        seeds_by_layer[int(layer_text)] = seeds
    if seeds_by_layer != {5: (42, 43), 20: (42,)}:
        raise ValueError("SAE seed replication differs from preregistration")

    acceptance = _object(
        root["acceptance"],
        field="acceptance",
        keys={
            "fvu_max",
            "median_l0_min",
            "median_l0_max",
            "dead_feature_probability_below",
            "dead_feature_rate_max",
            "splice_documents",
            "splice_kl_median_max",
            "maximum_gate_retrains",
        },
    )
    l0_min = _positive_int(acceptance["median_l0_min"], field="acceptance.median_l0_min")
    l0_max = _positive_int(acceptance["median_l0_max"], field="acceptance.median_l0_max")
    if l0_min > l0_max:
        raise ValueError("SAE L0 acceptance range is empty")

    kill = _object(
        root["feature_kill_test"],
        field="feature_kill_test",
        keys={
            "feature_sufficiency_ratio_min",
            "spurious_suppression_ratio_min",
            "requires_layer5_seed_replication",
            "bootstrap_replicates",
            "bootstrap_seed",
        },
    )
    if kill["requires_layer5_seed_replication"] is not True:
        raise ValueError("SAE feature kill test must require layer-5 seed replication")

    return SaeProtocol(
        schema_version=str(root["schema_version"]),
        model=str(model["id"]),
        model_revision=str(model["revision"]),
        model_dtype=str(model["dtype"]),
        d_model=d_model,
        decoder_layers=decoder_layers,
        probe_layers=probe_layers,
        activation_subsample_layers=subsample_layers,
        source=str(data["source"]),
        source_dataset=str(data["dataset"]),
        source_subset=str(data["subset"]),
        source_revision=str(data["revision"]),
        source_split=str(data["split"]),
        max_sequence_length=_positive_int(
            data["max_sequence_length"], field="data.max_sequence_length"
        ),
        document_character_limit=_positive_int(
            data["document_character_limit"], field="data.document_character_limit"
        ),
        near_duplicate_shingle_size=_positive_int(
            data["near_duplicate_shingle_size"],
            field="data.near_duplicate_shingle_size",
        ),
        near_duplicate_jaccard_threshold=_probability(
            data["near_duplicate_jaccard_threshold"],
            field="data.near_duplicate_jaccard_threshold",
        ),
        supplement_stream_order=str(data["supplement_stream_order"]),
        minimum_training_tokens=minimum_tokens,
        preferred_training_tokens=preferred_tokens,
        statistics_tokens=_positive_int(data["statistics_tokens"], field="data.statistics_tokens"),
        activation_subsample_tokens=_positive_int(
            data["activation_subsample_tokens"], field="data.activation_subsample_tokens"
        ),
        shuffle_buffer_activations=shuffle_buffer_activations,
        reserved_order_seed=_nonnegative_int(exclusion["seed"], field="exclusion.seed"),
        reserved_order_epoch=_nonnegative_int(exclusion["epoch"], field="exclusion.epoch"),
        reserved_prefix_records=_positive_int(
            exclusion["reserved_records"], field="exclusion.reserved_records"
        ),
        protected_student_token_budget=_positive_int(
            exclusion["declared_student_token_budget"],
            field="exclusion.declared_student_token_budget",
        ),
        expansion_factor=_positive_int(sae["expansion_factor"], field="sae.expansion_factor"),
        activation=str(sae["activation"]),
        regularizer=str(sae["regularizer"]),
        l1_coefficients=l1_coefficients,
        l1_calibration_tokens=l1_calibration_tokens,
        l1_selection_rule=str(sae["l1_selection_rule"]),
        optimizer=str(sae["optimizer"]),
        learning_rate=_positive_float(sae["learning_rate"], field="sae.learning_rate"),
        adam_betas=(float(betas[0]), float(betas[1])),
        adam_epsilon=_positive_float(sae["adam_epsilon"], field="sae.adam_epsilon"),
        precision=str(sae["precision"]),
        activation_batch_size=_positive_int(
            sae["activation_batch_size"], field="sae.activation_batch_size"
        ),
        decoder_normalization=str(sae["decoder_normalization"]),
        seeds_by_layer=MappingProxyType(seeds_by_layer),
        fvu_max=_probability(acceptance["fvu_max"], field="acceptance.fvu_max"),
        median_l0_range=(l0_min, l0_max),
        dead_feature_probability_below=_probability(
            acceptance["dead_feature_probability_below"],
            field="acceptance.dead_feature_probability_below",
        ),
        dead_feature_rate_max=_probability(
            acceptance["dead_feature_rate_max"], field="acceptance.dead_feature_rate_max"
        ),
        splice_documents=_positive_int(
            acceptance["splice_documents"], field="acceptance.splice_documents"
        ),
        splice_kl_max=_positive_float(
            acceptance["splice_kl_median_max"], field="acceptance.splice_kl_median_max"
        ),
        maximum_gate_retrains=_nonnegative_int(
            acceptance["maximum_gate_retrains"], field="acceptance.maximum_gate_retrains"
        ),
        wp5_feature_sufficiency_ratio=_probability(
            kill["feature_sufficiency_ratio_min"],
            field="feature_kill_test.feature_sufficiency_ratio_min",
        ),
        wp5_suppression_ratio=_probability(
            kill["spurious_suppression_ratio_min"],
            field="feature_kill_test.spurious_suppression_ratio_min",
        ),
        wp5_requires_layer5_seed_replication=True,
        bootstrap_replicates=_positive_int(
            kill["bootstrap_replicates"], field="feature_kill_test.bootstrap_replicates"
        ),
        bootstrap_seed=_nonnegative_int(
            kill["bootstrap_seed"], field="feature_kill_test.bootstrap_seed"
        ),
        config_path=resolved,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["SaeProtocol", "load_sae_protocol"]
