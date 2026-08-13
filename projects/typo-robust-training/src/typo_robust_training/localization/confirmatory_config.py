"""Strict config for behavior-independent confirmatory localization."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from typo_robust_training.data.config import strict_loads


_REVISION = re.compile(r"[0-9a-f]{40}")
_OPERATIONS = (
    "keyboard-neighbor-substitution",
    "deletion",
    "duplication",
)


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


def window_width_for_decoder_layers(decoder_layers: int) -> int:
    """Apply the frozen round-half-up `L / 6` rule without Python ties-to-even."""

    layers = _positive_int(decoder_layers, field="decoder layers")
    return max(1, math.floor(layers / 6.0 + 0.5))


@dataclass(frozen=True, slots=True)
class ConfirmatoryLocalizationProtocol:
    schema_version: str
    model: str
    model_revision: str
    dtype: str
    decoder_layers: int
    selection_source: str
    source_dataset: str
    source_subset: str
    source_revision: str
    source_split: str
    selection_records: int
    validation_records: int
    source_seed: int
    character_window: int
    max_sequence_length: int
    minimum_following_tokens: int
    typo_operations: tuple[str, ...]
    minimum_word_letters: int
    teacher_forced_tokens: int
    readout_token_range: tuple[int, int]
    denominator_min_exclusive: float
    minimum_eligible: int
    minimum_eligible_fraction: float
    window_width: int
    window_selection_metric: str
    tie_break: str
    validation_rule: str
    random_control_rule: str
    random_control_seed: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    config_path: Path
    config_sha256: str


def load_confirmatory_localization_config(path: Path) -> ConfirmatoryLocalizationProtocol:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"confirmatory localization config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("confirmatory localization config is not UTF-8") from exc
    root = _object(
        payload,
        field="confirmatory localization config",
        keys={
            "schema_version",
            "model",
            "data",
            "typos",
            "readout",
            "intervention",
            "selection",
            "statistics",
        },
    )
    if root["schema_version"] != "robustness-confirmatory-localization-config/v1":
        raise ValueError("confirmatory localization schema differs")

    model = _object(
        root["model"],
        field="model",
        keys={"id", "revision", "dtype", "decoder_layers"},
    )
    if not isinstance(model["id"], str) or not model["id"]:
        raise ValueError("model.id must be non-empty")
    if not isinstance(model["revision"], str) or _REVISION.fullmatch(model["revision"]) is None:
        raise ValueError("model.revision must be a pinned lowercase commit SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("confirmatory localization dtype must be bfloat16")
    decoder_layers = _positive_int(model["decoder_layers"], field="decoder layers")

    data = _object(
        root["data"],
        field="data",
        keys={
            "source",
            "dataset",
            "subset",
            "revision",
            "split",
            "selection_records",
            "validation_records",
            "seed",
            "character_window",
            "max_sequence_length",
            "minimum_following_tokens",
        },
    )
    if data["source"] != "fineweb_edu" or data["dataset"] != "HuggingFaceFW/fineweb-edu":
        raise ValueError("confirmatory localization source must be FineWeb-Edu")
    for field in ("subset", "split"):
        if not isinstance(data[field], str) or not data[field]:
            raise ValueError(f"data.{field} must be non-empty")
    if not isinstance(data["revision"], str) or _REVISION.fullmatch(data["revision"]) is None:
        raise ValueError("data.revision must be a pinned lowercase commit SHA")

    typos = _object(
        root["typos"],
        field="typos",
        keys={"edit_count", "minimum_word_letters", "operations", "assignment"},
    )
    if typos["edit_count"] != 1 or typos["assignment"] != "balanced-deterministic/v1":
        raise ValueError("confirmatory localization typo realization differs")
    if not isinstance(typos["operations"], list) or tuple(typos["operations"]) != _OPERATIONS:
        raise ValueError("confirmatory localization typo operations differ")

    readout = _object(
        root["readout"],
        field="readout",
        keys={
            "coordinate",
            "continuation",
            "teacher_forced_tokens",
            "token_start",
            "token_stop",
            "denominator_min_exclusive",
        },
    )
    if readout["coordinate"] != "edited-word-final-token/v1":
        raise ValueError("confirmatory localization coordinate differs")
    if readout["continuation"] != "first-16-clean-corpus-tokens-after-edit/v1":
        raise ValueError("confirmatory localization continuation differs")
    teacher_tokens = _positive_int(readout["teacher_forced_tokens"], field="teacher tokens")
    token_start = _positive_int(readout["token_start"], field="token start")
    token_stop = _positive_int(readout["token_stop"], field="token stop")
    if teacher_tokens != 16 or (token_start, token_stop) != (2, 16):
        raise ValueError("confirmatory localization readout must be tokens 2--16")

    intervention = _object(
        root["intervention"],
        field="intervention",
        keys={"direction", "patch_site", "scan_layers", "window_width_rule"},
    )
    expected_intervention = {
        "direction": "clean-to-typo",
        "patch_site": "complete-decoder-block-residual-output",
        "scan_layers": "all/v1",
        "window_width_rule": "round-half-up-decoder-layers-divided-by-six/v1",
    }
    if dict(intervention) != expected_intervention:
        raise ValueError("confirmatory localization intervention differs")

    selection = _object(
        root["selection"],
        field="selection",
        keys={
            "metric",
            "tie_break",
            "validation_rule",
            "minimum_eligible",
            "minimum_eligible_fraction",
            "random_control_rule",
            "random_control_seed",
        },
    )
    expected_selection = {
        "metric": "median-pairwise-kl-restoration/v1",
        "tie_break": "smallest-start-layer/v1",
        "validation_rule": "bootstrap-95ci-lower-strictly-positive/v1",
        "random_control_rule": "sha256-drawn-nonoverlapping-same-width/v1",
    }
    for field, expected in expected_selection.items():
        if selection[field] != expected:
            raise ValueError(f"selection.{field} differs")

    statistics = _object(
        root["statistics"],
        field="statistics",
        keys={"bootstrap_replicates", "bootstrap_seed", "confidence_level"},
    )
    return ConfirmatoryLocalizationProtocol(
        schema_version=str(root["schema_version"]),
        model=str(model["id"]),
        model_revision=str(model["revision"]),
        dtype=str(model["dtype"]),
        decoder_layers=decoder_layers,
        selection_source=str(data["source"]),
        source_dataset=str(data["dataset"]),
        source_subset=str(data["subset"]),
        source_revision=str(data["revision"]),
        source_split=str(data["split"]),
        selection_records=_positive_int(data["selection_records"], field="selection records"),
        validation_records=_positive_int(data["validation_records"], field="validation records"),
        source_seed=_nonnegative_int(data["seed"], field="data seed"),
        character_window=_positive_int(data["character_window"], field="character window"),
        max_sequence_length=_positive_int(
            data["max_sequence_length"], field="maximum sequence length"
        ),
        minimum_following_tokens=_positive_int(
            data["minimum_following_tokens"], field="minimum following tokens"
        ),
        typo_operations=_OPERATIONS,
        minimum_word_letters=_positive_int(
            typos["minimum_word_letters"], field="minimum word letters"
        ),
        teacher_forced_tokens=teacher_tokens,
        readout_token_range=(token_start, token_stop),
        denominator_min_exclusive=_positive_float(
            readout["denominator_min_exclusive"], field="denominator minimum"
        ),
        minimum_eligible=_positive_int(selection["minimum_eligible"], field="minimum eligible"),
        minimum_eligible_fraction=_probability(
            selection["minimum_eligible_fraction"], field="minimum eligible fraction"
        ),
        window_width=window_width_for_decoder_layers(decoder_layers),
        window_selection_metric=str(selection["metric"]),
        tie_break=str(selection["tie_break"]),
        validation_rule=str(selection["validation_rule"]),
        random_control_rule=str(selection["random_control_rule"]),
        random_control_seed=_nonnegative_int(
            selection["random_control_seed"], field="random control seed"
        ),
        bootstrap_replicates=_positive_int(
            statistics["bootstrap_replicates"], field="bootstrap replicates"
        ),
        bootstrap_seed=_nonnegative_int(statistics["bootstrap_seed"], field="bootstrap seed"),
        confidence_level=_probability(statistics["confidence_level"], field="confidence level"),
        config_path=resolved,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "ConfirmatoryLocalizationProtocol",
    "load_confirmatory_localization_config",
    "window_width_for_decoder_layers",
]
