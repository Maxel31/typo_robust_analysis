"""Strict, content-addressed evidence for linear-probe transition selection.

The selected layer is not accepted as an unattested scalar. Loading an
artifact resolves every referenced file inside the artifact bundle, checks
its digest, verifies source-level disjointness, and recomputes the selection
and independent validation from paired per-example probe losses.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.perturb import (
    classify_character_edit,
    is_keyboard_neighbor_substitution,
)
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.probe.config import (
    ProbeProducerProtocol,
    load_probe_producer_config,
)
from typo_robust_training.probe.scoring import ProbeSeedTrajectory, select_probe_transition


_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROLES = ("fit", "selection", "validation")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "operation",
    "model",
    "model_revision",
    "decoder_layers",
    "hook_site",
    "coordinate",
    "probe_seeds",
    "references",
    "selection_metric",
    "selection_rule",
    "tie_break",
    "stability_rule",
    "validation_rule",
    "bootstrap",
    "selected_transition_layer",
    "validation_passed",
}
_REFERENCE_FIELDS = {"relative_path", "sha256"}
_MANIFEST_FIELDS = {"schema_version", "role", "records"}
_COMMON_RECORD_FIELDS = {
    "record_id",
    "source_group_sha256",
    "parent_source_sha256",
    "normalized_clean_sha256",
    "clean_text",
    "clean_word_char_span",
    "class_id",
}
_PAIRED_RECORD_FIELDS = _COMMON_RECORD_FIELDS | {
    "pair_id",
    "normalized_noisy_sha256",
    "typo_text",
    "typo_word_char_span",
    "edit_type",
    "edit_count",
    "token_inflation_bucket",
}
_SCORE_FIELDS = {
    "schema_version",
    "role",
    "seed",
    "decoder_layers",
    "bindings",
    "records",
}
_SCORE_BINDING_FIELDS = {
    "model",
    "model_revision",
    "code_revision",
    "config_sha256",
    "class_inventory_sha256",
    "fit_manifest_sha256",
    "role_manifest_sha256",
    "probe_weights_sha256",
}
_WEIGHT_METADATA_FIELDS = {
    "schema_version",
    "seed",
    "config_sha256",
    "fit_manifest_sha256",
    "class_inventory_sha256",
    "model",
    "model_revision",
    "code_revision",
    "decoder_layers",
    "hidden_size",
    "class_count",
}
_SCORE_ROW_FIELDS = {
    "pair_id",
    "source_group_sha256",
    "class_id",
    "edit_type",
    "edit_count",
    "token_inflation_bucket",
    "clean_cross_entropy",
    "noisy_cross_entropy",
}


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _json_file(path: Path) -> Mapping[str, object]:
    try:
        value = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    except UnicodeDecodeError as exc:
        raise ValueError(f"artifact reference is not UTF-8: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact reference must contain a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _reference(value: object, *, root: Path, field: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != _REFERENCE_FIELDS:
        raise ValueError(f"{field} reference fields differ")
    relative_value = value["relative_path"]
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{field} relative path must be a non-empty string")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != relative_value:
        raise ValueError(f"{field} reference must be a canonical relative POSIX path")
    supplied = root / Path(*relative.parts)
    if supplied.is_symlink():
        raise ValueError(f"{field} reference must not be a symlink")
    path = supplied.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} reference escapes the artifact bundle") from exc
    expected = _sha(value["sha256"], field=f"{field} hash")
    if not path.is_file() or _digest(path) != expected:
        raise ValueError(f"{field} reference is missing or its hash differs")
    return path, expected


def _strict_number_list(value: object, *, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain one value per decoder layer")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field} must contain only JSON numbers")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError(f"{field} must contain finite non-negative values")
    return result


@dataclass(frozen=True, slots=True)
class _CohortRecord:
    record_id: str
    source_group_sha256: str
    parent_source_sha256: str
    normalized_clean_sha256: str
    clean_text: str
    clean_word_char_span: tuple[int, int]
    class_id: int
    pair_id: str | None = None
    normalized_noisy_sha256: str | None = None
    typo_text: str | None = None
    typo_word_char_span: tuple[int, int] | None = None
    edit_type: str | None = None
    edit_count: int | None = None
    token_inflation_bucket: str | None = None


def _load_class_inventory(path: Path) -> tuple[str, ...]:
    value = _json_file(path)
    if set(value) != {"schema_version", "classes"}:
        raise ValueError("probe class inventory fields differ")
    if value["schema_version"] != "typo-word-identity-classes/v1":
        raise ValueError("probe class inventory schema differs")
    rows = value["classes"]
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("probe class inventory must contain at least two classes")
    labels: list[str] = []
    for expected_id, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"class_id", "label"}:
            raise ValueError("probe class inventory row fields differ")
        if _integer(row["class_id"], field="probe class id") != expected_id:
            raise ValueError("probe class ids must be contiguous and ordered")
        labels.append(_string(row["label"], field="probe class label"))
    if len(set(labels)) != len(labels):
        raise ValueError("probe class labels must be unique")
    return tuple(labels)


def _char_span(value: object, *, text: str, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{field} must contain two integers")
    start, stop = value
    if (
        not 0 <= start < stop <= len(text)
        or any(character.isspace() for character in text[start:stop])
        or (start and text[start - 1].isalnum())
        or (stop < len(text) and text[stop].isalnum())
    ):
        raise ValueError(f"{field} is not exactly one word span")
    return start, stop


def _load_manifest(
    path: Path, *, expected_role: str, class_labels: Sequence[str]
) -> tuple[_CohortRecord, ...]:
    value = _json_file(path)
    if set(value) != _MANIFEST_FIELDS:
        raise ValueError(f"probe {expected_role} manifest fields differ")
    if value["schema_version"] != "typo-probe-cohort/v2" or value["role"] != expected_role:
        raise ValueError(f"probe {expected_role} manifest identity differs")
    rows = value["records"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"probe {expected_role} manifest must contain records")
    expected_fields = _COMMON_RECORD_FIELDS if expected_role == "fit" else _PAIRED_RECORD_FIELDS
    records: list[_CohortRecord] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError(f"probe {expected_role} record fields differ")
        class_id = _integer(row["class_id"], field="probe class id")
        if class_id >= len(class_labels):
            raise ValueError("probe record class id is outside the class inventory")
        clean_text = _string(row["clean_text"], field="probe clean text")
        clean_hash = _sha(row["normalized_clean_sha256"], field="normalized clean hash")
        if normalized_content_sha256(clean_text) != clean_hash:
            raise ValueError("probe normalized clean hash differs from the resolved text")
        clean_span = _char_span(
            row["clean_word_char_span"], text=clean_text, field="probe clean word span"
        )
        if clean_text[slice(*clean_span)] != class_labels[class_id]:
            raise ValueError("probe clean word does not match its class label")
        common = {
            "record_id": _string(row["record_id"], field="probe record id"),
            "source_group_sha256": _sha(row["source_group_sha256"], field="source group hash"),
            "parent_source_sha256": _sha(row["parent_source_sha256"], field="parent source hash"),
            "normalized_clean_sha256": clean_hash,
            "clean_text": clean_text,
            "clean_word_char_span": clean_span,
            "class_id": class_id,
        }
        if expected_role == "fit":
            records.append(_CohortRecord(**common))
        else:
            typo_text = _string(row["typo_text"], field="probe typo text")
            typo_hash = _sha(row["normalized_noisy_sha256"], field="normalized noisy hash")
            if normalized_content_sha256(typo_text) != typo_hash or typo_text == clean_text:
                raise ValueError("probe normalized typo hash differs or pair is a no-op")
            typo_span = _char_span(
                row["typo_word_char_span"], text=typo_text, field="probe typo word span"
            )
            edit_type = _string(row["edit_type"], field="probe edit type")
            if edit_type not in {
                "keyboard-neighbor-substitution",
                "deletion",
                "duplication",
            }:
                raise ValueError("probe edit type is outside the frozen operation inventory")
            observed_edit = classify_character_edit(
                clean=clean_text[slice(*clean_span)], typo=typo_text[slice(*typo_span)]
            )
            expected_observed = (
                "natural-statistics-substitution"
                if edit_type == "keyboard-neighbor-substitution"
                else edit_type
            )
            if observed_edit != expected_observed:
                raise ValueError("probe edit type differs from the resolved word pair")
            if edit_type == "keyboard-neighbor-substitution" and not (
                is_keyboard_neighbor_substitution(
                    clean=clean_text[slice(*clean_span)],
                    typo=typo_text[slice(*typo_span)],
                )
            ):
                raise ValueError(
                    "probe keyboard substitution is not a case-preserving neighbor"
                )
            if (
                clean_text[: clean_span[0]] != typo_text[: typo_span[0]]
                or clean_text[clean_span[1] :] != typo_text[typo_span[1] :]
            ):
                raise ValueError("probe pair does not isolate one aligned edited word")
            records.append(
                _CohortRecord(
                    **common,
                    pair_id=_string(row["pair_id"], field="probe pair id"),
                    normalized_noisy_sha256=typo_hash,
                    typo_text=typo_text,
                    typo_word_char_span=typo_span,
                    edit_type=edit_type,
                    edit_count=_integer(row["edit_count"], field="probe edit count", minimum=1),
                    token_inflation_bucket=_string(
                        row["token_inflation_bucket"], field="token inflation bucket"
                    ),
                )
            )
            if records[-1].edit_count != 1:
                raise ValueError("probe selection and validation require exactly one edit")
    record_ids = [row.record_id for row in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError(f"probe {expected_role} record ids must be unique")
    pair_ids = [row.pair_id for row in records if row.pair_id is not None]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"probe {expected_role} pair ids must be unique")
    _validate_within_role_identities(records, role=expected_role)
    counts = Counter(row.class_id for row in records)
    if set(counts) != set(range(len(class_labels))) or len(set(counts.values())) != 1:
        raise ValueError(f"probe {expected_role} cohort must be exactly class balanced")
    if min(counts.values()) < 2 or any(
        len({row.source_group_sha256 for row in records if row.class_id == class_id}) < 2
        for class_id in range(len(class_labels))
    ):
        raise ValueError(f"probe {expected_role} requires repeated independent sources per class")
    if expected_role != "fit":
        required_operations = {
            "keyboard-neighbor-substitution",
            "deletion",
            "duplication",
        }
        operation_counts = Counter(row.edit_type for row in records)
        if set(operation_counts) != required_operations or (
            max(operation_counts.values()) - min(operation_counts.values()) > 1
        ):
            raise ValueError(
                f"probe {expected_role} edit operation strata must include every frozen "
                "operation with counts differing by at most one"
            )
    return tuple(records)


def _validate_within_role_identities(
    records: Sequence[_CohortRecord],
    *,
    role: str,
) -> None:
    clean_hashes = [record.normalized_clean_sha256 for record in records]
    noisy_hashes = [
        record.normalized_noisy_sha256
        for record in records
        if record.normalized_noisy_sha256 is not None
    ]
    if len(set(clean_hashes)) != len(clean_hashes) or len(set(noisy_hashes)) != len(
        noisy_hashes
    ):
        raise ValueError(f"probe {role} normalized content must be unique within role")
    parent_to_group: dict[str, str] = {}
    for record in records:
        previous = parent_to_group.setdefault(
            record.parent_source_sha256,
            record.source_group_sha256,
        )
        if previous != record.source_group_sha256:
            raise ValueError(
                f"probe {role} parent source maps to multiple bootstrap groups"
            )


def _identity_sets(records: Sequence[_CohortRecord]) -> tuple[set[str], set[str], set[str]]:
    return (
        {row.source_group_sha256 for row in records},
        {row.parent_source_sha256 for row in records},
        {row.normalized_clean_sha256 for row in records}
        | {row.normalized_noisy_sha256 for row in records if row.normalized_noisy_sha256},
    )


def _all_identities(records: Sequence[_CohortRecord]) -> set[str]:
    return set().union(*_identity_sets(records))


def _stratum_key(record: _CohortRecord) -> str:
    if (
        record.edit_type is None
        or record.edit_count is None
        or record.token_inflation_bucket is None
    ):
        raise ValueError("probe paired record lacks one frozen stratum")
    return f"{record.edit_type}|{record.edit_count}|{record.token_inflation_bucket}"


def _validate_preregistered_manifest(
    records: Sequence[_CohortRecord],
    *,
    role: str,
    class_count: int,
    protocol: ProbeProducerProtocol,
) -> None:
    expected_per_class = protocol.records_per_class[role]
    observed_counts = Counter(record.class_id for record in records)
    if observed_counts != Counter(
        {class_id: expected_per_class for class_id in range(class_count)}
    ):
        raise ValueError(f"probe {role} class counts differ from the preregistration")
    minimum_groups = protocol.min_source_groups_per_class[role]
    for class_id in range(class_count):
        group_count = len(
            {
                record.source_group_sha256
                for record in records
                if record.class_id == class_id
            }
        )
        if group_count < minimum_groups:
            raise ValueError(
                f"probe {role} source-group count differs from the preregistration"
            )
    if role in {"selection", "validation"} and Counter(
        _stratum_key(record) for record in records
    ) != Counter(protocol.stratum_counts[role]):
        raise ValueError(f"probe {role} strata differ from the preregistration")


def _load_protected_registry(path: Path) -> tuple[set[str], set[str], set[str]]:
    value = _json_file(path)
    if set(value) != {"schema_version", "registries"}:
        raise ValueError("protected split registry fields differ")
    if value["schema_version"] != "typo-protected-split-registry/v1":
        raise ValueError("protected split registry schema differs")
    rows = value["registries"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("protected split registry must contain at least one tier")
    required_tiers = {"training", "localization", "tune", "pre-pr", "sealed"}
    identities_by_tier: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "tier",
            "source_group_sha256",
            "parent_source_sha256",
            "normalized_content_sha256",
        }:
            raise ValueError("protected split registry row fields differ")
        tier = _string(row["tier"], field="protected tier")
        if tier in identities_by_tier:
            raise ValueError("protected split tier names must be unique")
        tier_sets: list[set[str]] = []
        for field in (
            "source_group_sha256",
            "parent_source_sha256",
            "normalized_content_sha256",
        ):
            values = row[field]
            if not isinstance(values, list):
                raise ValueError(f"protected {field} must be a list")
            tier_sets.append({_sha(item, field=f"protected {field}") for item in values})
        if not any(tier_sets):
            raise ValueError("every protected split tier must contain an identity")
        identities_by_tier[tier] = (tier_sets[0], tier_sets[1], tier_sets[2])
    if set(identities_by_tier) != required_tiers:
        raise ValueError("protected split registry tier inventory differs")
    ordered_tiers = sorted(required_tiers)
    for left_index, left in enumerate(ordered_tiers):
        for right in ordered_tiers[left_index + 1 :]:
            if set().union(*identities_by_tier[left]) & set().union(*identities_by_tier[right]):
                raise ValueError("protected split tiers overlap transitively")
    return (
        set().union(*(identities_by_tier[tier][0] for tier in ordered_tiers)),
        set().union(*(identities_by_tier[tier][1] for tier in ordered_tiers)),
        set().union(*(identities_by_tier[tier][2] for tier in ordered_tiers)),
    )


@dataclass(frozen=True, slots=True)
class _PairedScore:
    pair_id: str
    source_group_sha256: str
    class_id: int
    edit_type: str
    edit_count: int
    token_inflation_bucket: str
    clean_cross_entropy: tuple[float, ...]
    noisy_cross_entropy: tuple[float, ...]

    @property
    def noise_penalty(self) -> tuple[float, ...]:
        return tuple(
            noisy - clean
            for clean, noisy in zip(self.clean_cross_entropy, self.noisy_cross_entropy, strict=True)
        )


def _load_scores(
    path: Path,
    *,
    role: str,
    seed: int,
    decoder_layers: int,
    manifest: Sequence[_CohortRecord],
    expected_bindings: Mapping[str, str],
) -> tuple[_PairedScore, ...]:
    value = _json_file(path)
    if set(value) != _SCORE_FIELDS:
        raise ValueError(f"probe {role} score fields differ")
    if (
        value["schema_version"] != "typo-paired-probe-scores/v1"
        or value["role"] != role
        or _integer(value["seed"], field="probe score seed") != seed
        or _integer(value["decoder_layers"], field="probe score decoder layers", minimum=2)
        != decoder_layers
    ):
        raise ValueError(f"probe {role} score identity differs")
    bindings = value["bindings"]
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != _SCORE_BINDING_FIELDS
        or dict(bindings) != dict(expected_bindings)
    ):
        raise ValueError(f"probe {role} score provenance bindings differ")
    raw_rows = value["records"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"probe {role} scores must contain records")
    expected = {
        row.pair_id: (
            row.source_group_sha256,
            row.class_id,
            row.edit_type,
            row.edit_count,
            row.token_inflation_bucket,
        )
        for row in manifest
    }
    rows: list[_PairedScore] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != _SCORE_ROW_FIELDS:
            raise ValueError(f"probe {role} score row fields differ")
        pair_id = _string(raw["pair_id"], field="probe score pair id")
        metadata = (
            _sha(raw["source_group_sha256"], field="probe score source group"),
            _integer(raw["class_id"], field="probe score class id"),
            _string(raw["edit_type"], field="probe score edit type"),
            _integer(raw["edit_count"], field="probe score edit count", minimum=1),
            _string(raw["token_inflation_bucket"], field="probe score token inflation bucket"),
        )
        if pair_id not in expected or metadata != expected[pair_id]:
            raise ValueError(f"probe {role} score does not match its paired cohort manifest")
        rows.append(
            _PairedScore(
                pair_id=pair_id,
                source_group_sha256=metadata[0],
                class_id=metadata[1],
                edit_type=metadata[2],
                edit_count=metadata[3],
                token_inflation_bucket=metadata[4],
                clean_cross_entropy=_strict_number_list(
                    raw["clean_cross_entropy"],
                    field="probe clean cross-entropy",
                    length=decoder_layers,
                ),
                noisy_cross_entropy=_strict_number_list(
                    raw["noisy_cross_entropy"],
                    field="probe noisy cross-entropy",
                    length=decoder_layers,
                ),
            )
        )
    if len(rows) != len(expected) or {row.pair_id for row in rows} != set(expected):
        raise ValueError(f"probe {role} scores must cover every paired manifest row exactly once")
    return tuple(rows)


def _group_mean_trajectory(rows: Sequence[_PairedScore], *, seed: int) -> ProbeSeedTrajectory:
    grouped: dict[str, list[_PairedScore]] = defaultdict(list)
    for row in rows:
        grouped[row.source_group_sha256].append(row)
    layer_count = len(rows[0].clean_cross_entropy)
    clean: list[float] = []
    noisy: list[float] = []
    for layer in range(layer_count):
        clean_by_group = [
            sum(row.clean_cross_entropy[layer] for row in group_rows) / len(group_rows)
            for group_rows in grouped.values()
        ]
        noisy_by_group = [
            sum(row.noisy_cross_entropy[layer] for row in group_rows) / len(group_rows)
            for group_rows in grouped.values()
        ]
        clean.append(sum(clean_by_group) / len(clean_by_group))
        noisy.append(sum(noisy_by_group) / len(noisy_by_group))
    return ProbeSeedTrajectory(seed, tuple(clean), tuple(noisy))


def _bootstrap_lower_bound(
    rows: Sequence[_PairedScore],
    *,
    selected_layer: int,
    resamples: int,
    seed: int,
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        penalty = row.noise_penalty
        grouped[row.source_group_sha256].append(
            penalty[selected_layer - 1] - penalty[selected_layer]
        )
    group_values = tuple(sum(values) / len(values) for _group, values in sorted(grouped.items()))
    if len(group_values) < 2:
        raise ValueError("probe validation requires at least two independent source groups")
    samples: list[float] = []
    for replicate in range(resamples):
        total = 0.0
        for draw in range(len(group_values)):
            material = f"probe-bootstrap/v1\0{seed}\0{replicate}\0{draw}".encode()
            index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(group_values)
            total += group_values[index]
        samples.append(total / len(group_values))
    samples.sort()
    return samples[max(0, math.ceil(0.025 * len(samples)) - 1)]


def _validate_probe_weights(
    path: Path,
    *,
    seed: int,
    protocol: ProbeProducerProtocol,
    class_count: int,
) -> str:
    from safetensors import SafetensorError, safe_open

    expected_metadata = {
        "schema_version": "typo-linear-probe-weights/v1",
        "seed": str(seed),
        "config_sha256": protocol.config_sha256,
        "fit_manifest_sha256": protocol.input_sha256["fit_manifest"],
        "class_inventory_sha256": protocol.input_sha256["class_inventory"],
        "model": protocol.model,
        "model_revision": protocol.model_revision,
        "code_revision": protocol.code_revision,
        "decoder_layers": str(protocol.decoder_layers),
        "hidden_size": str(protocol.hidden_size),
        "class_count": str(class_count),
    }
    expected_keys = {
        f"decoder_layer.{layer}.{kind}"
        for layer in range(protocol.decoder_layers)
        for kind in ("weight", "bias")
    }
    try:
        with safe_open(path, framework="np") as handle:
            metadata = handle.metadata()
            if (
                not isinstance(metadata, Mapping)
                or set(metadata) != _WEIGHT_METADATA_FIELDS
                or dict(metadata) != expected_metadata
            ):
                raise ValueError("probe weight provenance metadata differs")
            if set(handle.keys()) != expected_keys:
                raise ValueError("probe weight tensor inventory differs")
            tensor_digest = hashlib.sha256(b"typo-linear-probe-tensors/v1\0")
            for layer in range(protocol.decoder_layers):
                weight = handle.get_tensor(f"decoder_layer.{layer}.weight")
                bias = handle.get_tensor(f"decoder_layer.{layer}.bias")
                if (
                    weight.shape != (class_count, protocol.hidden_size)
                    or bias.shape != (class_count,)
                    or weight.dtype != np.float32
                    or bias.dtype != np.float32
                    or not np.isfinite(weight).all()
                    or not np.isfinite(bias).all()
                ):
                    raise ValueError("probe weight tensor shape, dtype, or values differ")
                for name, tensor in (("weight", weight), ("bias", bias)):
                    array = np.array(tensor, dtype=np.float32, order="C", copy=True)
                    # Canonicalize the two IEEE zero encodings before checking
                    # numerical seed independence.
                    array[array == 0.0] = 0.0
                    tensor_digest.update(name.encode())
                    tensor_digest.update(str(array.shape).encode())
                    tensor_digest.update(array.dtype.str.encode())
                    tensor_digest.update(array.tobytes(order="C"))
    except SafetensorError as exc:
        raise ValueError("probe weight file is not a valid safetensors artifact") from exc
    return tensor_digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProbeTransitionIdentityInventory:
    """Verified transitive identities reserved by the parent probe study.

    Derivative studies consume this immutable inventory instead of reparsing
    private producer files or trusting a caller-provided list.
    """

    fit: frozenset[str]
    selection: frozenset[str]
    validation: frozenset[str]
    protected: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        return frozenset().union(
            self.fit,
            self.selection,
            self.validation,
            self.protected,
        )


@dataclass(frozen=True, slots=True)
class ProbeFitRecord:
    """One validated clean row used to fit the parent probe.

    This deliberately exposes only the immutable information needed to
    reproduce downstream clean activations.  Consumers never need to trust a
    separately supplied activation matrix.
    """

    record_id: str
    source_group_sha256: str
    parent_source_sha256: str
    normalized_clean_sha256: str
    clean_text: str
    clean_word_char_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ProbeTransitionArtifact:
    """Validated evidence consumed by suffix-adapter training."""

    model: str
    model_revision: str
    code_revision: str
    decoder_layers: int
    hidden_size: int
    selected_transition_layer: int
    probe_seeds: tuple[int, int]
    class_count: int
    probe_weights_by_seed: Mapping[int, Path]
    artifact_sha256: str
    config_sha256: str
    selection_ci_lower_by_seed: Mapping[int, float]
    validation_ci_lower_by_seed: Mapping[int, float]
    identity_inventory: ProbeTransitionIdentityInventory
    protected_split_registry_sha256: str
    fit_records: tuple[ProbeFitRecord, ...]

    @property
    def suffix_layers(self) -> tuple[int, ...]:
        return tuple(range(self.selected_transition_layer, self.decoder_layers))

    @property
    def cohort_identities_by_role(self) -> Mapping[str, frozenset[str]]:
        """Compatibility view over the stronger parent identity inventory."""

        return MappingProxyType(
            {
                "fit": self.identity_inventory.fit,
                "selection": self.identity_inventory.selection,
                "validation": self.identity_inventory.validation,
            }
        )

    @property
    def protected_identities(self) -> frozenset[str]:
        """Protected identities bound to the attested split registry."""

        return self.identity_inventory.protected

    @property
    def all_reserved_identities(self) -> frozenset[str]:
        """Every transitive identity unavailable to a downstream diagnostic."""

        return self.identity_inventory.all


def _validation_peak(trajectory: ProbeSeedTrajectory) -> int:
    drops = trajectory.transition_drop
    maximum = max(drops)
    return next(index for index, value in enumerate(drops) if value == maximum) + 1


def load_probe_transition_artifact(path: Path) -> ProbeTransitionArtifact:
    """Resolve and independently verify a probe transition evidence bundle."""

    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        resolved = supplied.resolve()
        raise ValueError(f"probe transition artifact is not a file: {resolved}")
    resolved = supplied.resolve()
    raw = resolved.read_bytes()
    payload = _json_file(resolved)
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("probe transition artifact fields differ")
    identity = {
        "schema_version": "typo-denoising-probe-selection/v2",
        "operation": "select-linear-probe-denoising-transition",
        "hook_site": "complete-decoder-block-residual-output",
        "coordinate": "edited-word-final-token/v1",
        "selection_metric": "largest-group-mean-paired-noise-penalty-drop/v2",
        "selection_rule": "min-argmax-over-layers-one-through-last/v1",
        "tie_break": "smallest-layer/v1",
        "stability_rule": "selection-exact-and-validation-within-one-layer-for-both-seeds/v1",
        "validation_rule": "group-bootstrap-95pct-lower-positive-for-both-seeds/v1",
    }
    for field, expected in identity.items():
        if payload[field] != expected:
            raise ValueError(f"probe transition {field} differs")
    model = _string(payload["model"], field="probe transition model")
    revision = payload["model_revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("probe transition model revision must be a pinned lowercase commit SHA")
    decoder_layers = _integer(payload["decoder_layers"], field="decoder layers", minimum=2)
    seed_values = payload["probe_seeds"]
    if (
        not isinstance(seed_values, list)
        or len(seed_values) != 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_values
        )
        or seed_values != sorted(set(seed_values))
    ):
        raise ValueError("probe transition requires two sorted distinct non-negative seeds")
    seeds = (seed_values[0], seed_values[1])
    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
        "resamples",
        "seed",
        "confidence",
        "unit",
    }:
        raise ValueError("probe bootstrap fields differ")
    if (
        _integer(bootstrap["resamples"], field="probe bootstrap resamples", minimum=1) != 10_000
        or _integer(bootstrap["seed"], field="probe bootstrap seed") != 1729
        or bootstrap["confidence"] != 0.95
        or bootstrap["unit"] != "source-group"
    ):
        raise ValueError("probe bootstrap protocol differs")
    references = payload["references"]
    if not isinstance(references, Mapping) or set(references) != {
        "config",
        "class_inventory",
        "fit_manifest",
        "selection_manifest",
        "validation_manifest",
        "protected_split_registry",
        "probe_weights_by_seed",
        "selection_scores_by_seed",
        "validation_scores_by_seed",
    }:
        raise ValueError("probe transition reference fields differ")
    root = resolved.parent.resolve()
    config_path, config_sha256 = _reference(
        references["config"], root=root, field="probe config"
    )
    protocol = load_probe_producer_config(config_path)
    if config_sha256 != protocol.config_sha256:
        raise ValueError("probe config reference differs from its content hash")
    if (
        model != protocol.model
        or revision != protocol.model_revision
        or decoder_layers != protocol.decoder_layers
        or seeds != protocol.probe_seeds
        or payload["hook_site"] != protocol.hook_site
        or payload["coordinate"] != protocol.coordinate
        or payload["selection_metric"] != protocol.selection_metric
        or payload["selection_rule"] != protocol.selection_rule
        or payload["tie_break"] != protocol.tie_break
        or payload["stability_rule"] != protocol.stability_rule
        or payload["validation_rule"] != protocol.validation_rule
        or bootstrap
        != {
            "resamples": protocol.bootstrap_resamples,
            "seed": protocol.bootstrap_seed,
            "confidence": protocol.bootstrap_confidence,
            "unit": protocol.bootstrap_unit,
        }
    ):
        raise ValueError("probe artifact identity differs from its preregistration")
    classes_path, classes_sha256 = _reference(
        references["class_inventory"], root=root, field="probe class inventory"
    )
    labels = _load_class_inventory(classes_path)
    manifests: dict[str, tuple[_CohortRecord, ...]] = {}
    manifest_hashes: dict[str, str] = {}
    for role in _ROLES:
        manifest_path, manifest_hash = _reference(
            references[f"{role}_manifest"], root=root, field=f"probe {role} manifest"
        )
        manifest_hashes[role] = manifest_hash
        manifests[role] = _load_manifest(manifest_path, expected_role=role, class_labels=labels)
        _validate_preregistered_manifest(
            manifests[role],
            role=role,
            class_count=len(labels),
            protocol=protocol,
        )
    identity_unions = {role: _all_identities(manifests[role]) for role in _ROLES}
    for left_index, left in enumerate(_ROLES):
        for right in _ROLES[left_index + 1 :]:
            if identity_unions[left] & identity_unions[right]:
                raise ValueError("probe cohorts overlap transitively across roles")
    protected_path, protected_sha256 = _reference(
        references["protected_split_registry"], root=root, field="protected split registry"
    )
    protected = _load_protected_registry(protected_path)
    protected_union = set().union(*protected)
    if any(identity_unions[role] & protected_union for role in _ROLES):
        raise ValueError("probe cohort overlaps a protected training or evaluation split")
    resolved_input_hashes = {
        "class_inventory": classes_sha256,
        "fit_manifest": manifest_hashes["fit"],
        "selection_manifest": manifest_hashes["selection"],
        "validation_manifest": manifest_hashes["validation"],
        "protected_split_registry": protected_sha256,
    }
    if resolved_input_hashes != dict(protocol.input_sha256):
        raise ValueError("probe evidence inputs differ from their preregistration")
    weights_value = references["probe_weights_by_seed"]
    selection_refs = references["selection_scores_by_seed"]
    validation_refs = references["validation_scores_by_seed"]
    expected_seed_keys = {str(seed) for seed in seeds}
    if any(
        not isinstance(value, Mapping) or set(value) != expected_seed_keys
        for value in (weights_value, selection_refs, validation_refs)
    ):
        raise ValueError("probe seed references must exactly match the seed inventory")
    weights: dict[int, Path] = {}
    weight_hashes: dict[int, str] = {}
    selection_rows: dict[int, tuple[_PairedScore, ...]] = {}
    validation_rows: dict[int, tuple[_PairedScore, ...]] = {}
    for seed in seeds:
        weights[seed], weight_hashes[seed] = _reference(
            weights_value[str(seed)], root=root, field=f"probe seed {seed} weights"
        )
    if len(set(weights.values())) != 2 or len(set(weight_hashes.values())) != 2:
        raise ValueError("probe seeds must use distinct independently fitted weight artifacts")
    tensor_digests: dict[int, str] = {}
    for seed in seeds:
        tensor_digests[seed] = _validate_probe_weights(
            weights[seed],
            seed=seed,
            protocol=protocol,
            class_count=len(labels),
        )
    if len(set(tensor_digests.values())) != len(seeds):
        raise ValueError("independent probe seeds contain identical numerical tensors")
    for seed in seeds:
        selection_path, _ = _reference(
            selection_refs[str(seed)], root=root, field=f"probe seed {seed} selection scores"
        )
        validation_path, _ = _reference(
            validation_refs[str(seed)], root=root, field=f"probe seed {seed} validation scores"
        )
        selection_rows[seed] = _load_scores(
            selection_path,
            role="selection",
            seed=seed,
            decoder_layers=decoder_layers,
            manifest=manifests["selection"],
            expected_bindings={
                "model": protocol.model,
                "model_revision": protocol.model_revision,
                "code_revision": protocol.code_revision,
                "config_sha256": protocol.config_sha256,
                "class_inventory_sha256": classes_sha256,
                "fit_manifest_sha256": manifest_hashes["fit"],
                "role_manifest_sha256": manifest_hashes["selection"],
                "probe_weights_sha256": weight_hashes[seed],
            },
        )
        validation_rows[seed] = _load_scores(
            validation_path,
            role="validation",
            seed=seed,
            decoder_layers=decoder_layers,
            manifest=manifests["validation"],
            expected_bindings={
                "model": protocol.model,
                "model_revision": protocol.model_revision,
                "code_revision": protocol.code_revision,
                "config_sha256": protocol.config_sha256,
                "class_inventory_sha256": classes_sha256,
                "fit_manifest_sha256": manifest_hashes["fit"],
                "role_manifest_sha256": manifest_hashes["validation"],
                "probe_weights_sha256": weight_hashes[seed],
            },
        )
    selection_trajectories = tuple(
        _group_mean_trajectory(selection_rows[seed], seed=seed) for seed in seeds
    )
    recomputed = select_probe_transition(selection_trajectories)
    selected = _integer(
        payload["selected_transition_layer"], field="selected transition layer", minimum=1
    )
    if selected >= decoder_layers or selected != recomputed.selected_layer:
        raise ValueError("stored transition layer differs from recomputed selection")
    if any(layer != selected for _seed, layer in recomputed.seed_selected_layers):
        raise ValueError("probe transition layer is not stable across selection probe seeds")
    selection_ci_lower = {
        seed: _bootstrap_lower_bound(
            selection_rows[seed],
            selected_layer=selected,
            resamples=10_000,
            seed=1729,
        )
        for seed in seeds
    }
    if any(value <= 0.0 for value in selection_ci_lower.values()):
        raise ValueError("probe transition lacks a positive selection denoising drop")
    ci_lower: dict[int, float] = {}
    for seed in seeds:
        validation_trajectory = _group_mean_trajectory(validation_rows[seed], seed=seed)
        if abs(_validation_peak(validation_trajectory) - selected) > 1:
            raise ValueError("probe transition boundary is not stable on independent validation")
        ci_lower[seed] = _bootstrap_lower_bound(
            validation_rows[seed],
            selected_layer=selected,
            resamples=10_000,
            seed=1729,
        )
    if payload["validation_passed"] is not True or any(value <= 0.0 for value in ci_lower.values()):
        raise ValueError("probe transition did not pass group-bootstrap independent validation")
    return ProbeTransitionArtifact(
        model=model,
        model_revision=revision,
        code_revision=protocol.code_revision,
        decoder_layers=decoder_layers,
        hidden_size=protocol.hidden_size,
        selected_transition_layer=selected,
        probe_seeds=seeds,
        class_count=len(labels),
        probe_weights_by_seed=MappingProxyType(weights),
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        config_sha256=protocol.config_sha256,
        selection_ci_lower_by_seed=MappingProxyType(selection_ci_lower),
        validation_ci_lower_by_seed=MappingProxyType(ci_lower),
        identity_inventory=ProbeTransitionIdentityInventory(
            fit=frozenset(identity_unions["fit"]),
            selection=frozenset(identity_unions["selection"]),
            validation=frozenset(identity_unions["validation"]),
            protected=frozenset(protected_union),
        ),
        protected_split_registry_sha256=protected_sha256,
        fit_records=tuple(
            ProbeFitRecord(
                record_id=row.record_id,
                source_group_sha256=row.source_group_sha256,
                parent_source_sha256=row.parent_source_sha256,
                normalized_clean_sha256=row.normalized_clean_sha256,
                clean_text=row.clean_text,
                clean_word_char_span=row.clean_word_char_span,
            )
            for row in manifests["fit"]
        ),
    )


__all__ = [
    "ProbeFitRecord",
    "ProbeTransitionArtifact",
    "ProbeTransitionIdentityInventory",
    "load_probe_transition_artifact",
]
