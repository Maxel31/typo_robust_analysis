"""Fit and freeze content-addressed linear-probe transition evidence."""

from __future__ import annotations

import hashlib
import math
import platform
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.perturb import (
    classify_character_edit,
    is_keyboard_neighbor_substitution,
)
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.integrity import sha256_file
from typo_robust_training.probe.config import (
    ProbeProducerProtocol,
    load_probe_producer_config,
)
from typo_robust_training.probe.scoring import (
    ProbeSeedTrajectory,
    ProbeTransitionSelection,
    select_probe_transition,
)
from typo_robust_training.training.json_io import write_json_atomic


_SHA256_LENGTH = 64
_ROLES = ("fit", "selection", "validation")
_COMMON_RECORD_FIELDS = {
    "record_id",
    "source_group_sha256",
    "parent_source_sha256",
    "normalized_clean_sha256",
    "class_id",
    "clean_text",
    "clean_word_char_span",
}
_PAIRED_RECORD_FIELDS = _COMMON_RECORD_FIELDS | {
    "pair_id",
    "normalized_noisy_sha256",
    "edit_type",
    "edit_count",
    "token_inflation_bucket",
    "typo_text",
    "typo_word_char_span",
}


def _json_object(path: Path, *, label: str) -> Mapping[str, object]:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be one regular file")
    resolved = supplied.resolve()
    try:
        value = strict_loads(resolved.read_text(encoding="utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha(value: object, *, field: str) -> str:
    result = _string(value, field=field)
    if len(result) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _span(value: object, *, text: str, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or not 0 <= value[0] < value[1] <= len(text)
        or any(character.isspace() for character in text[value[0] : value[1]])
        or (value[0] and text[value[0] - 1].isalnum())
        or (value[1] < len(text) and text[value[1]].isalnum())
    ):
        raise ValueError(f"{field} must be one in-bounds character span")
    return value[0], value[1]


@dataclass(frozen=True, slots=True)
class ProbeCohortRecord:
    record_id: str
    source_group_sha256: str
    parent_source_sha256: str
    normalized_clean_sha256: str
    class_id: int
    clean_text: str
    clean_word_char_span: tuple[int, int]
    pair_id: str | None = None
    normalized_noisy_sha256: str | None = None
    edit_type: str | None = None
    edit_count: int | None = None
    token_inflation_bucket: str | None = None
    typo_text: str | None = None
    typo_word_char_span: tuple[int, int] | None = None


def _load_classes(path: Path) -> tuple[str, ...]:
    payload = _json_object(path, label="probe class inventory")
    if set(payload) != {"schema_version", "classes"}:
        raise ValueError("probe class inventory fields differ")
    if payload["schema_version"] != "typo-word-identity-classes/v1":
        raise ValueError("probe class inventory schema differs")
    rows = payload["classes"]
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


def _load_cohort(
    path: Path,
    *,
    role: str,
    labels: Sequence[str],
) -> tuple[ProbeCohortRecord, ...]:
    payload = _json_object(path, label=f"probe {role} manifest")
    if set(payload) != {"schema_version", "role", "records"}:
        raise ValueError(f"probe {role} manifest fields differ")
    if payload["schema_version"] != "typo-probe-cohort/v2" or payload["role"] != role:
        raise ValueError(f"probe {role} manifest identity differs")
    raw_rows = payload["records"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"probe {role} manifest must contain records")
    expected_fields = _COMMON_RECORD_FIELDS if role == "fit" else _PAIRED_RECORD_FIELDS
    records: list[ProbeCohortRecord] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError(f"probe {role} record fields differ")
        class_id = _integer(raw["class_id"], field="probe class id")
        if class_id >= len(labels):
            raise ValueError("probe class id is outside the class inventory")
        clean_text = _string(raw["clean_text"], field="probe clean text")
        clean_span = _span(
            raw["clean_word_char_span"], text=clean_text, field="clean word span"
        )
        if normalized_content_sha256(clean_text) != _sha(
            raw["normalized_clean_sha256"], field="normalized clean hash"
        ):
            raise ValueError("probe clean text hash differs")
        if clean_text[slice(*clean_span)] != labels[class_id]:
            raise ValueError("probe clean word span differs from its class label")
        common = {
            "record_id": _string(raw["record_id"], field="probe record id"),
            "source_group_sha256": _sha(
                raw["source_group_sha256"], field="source group hash"
            ),
            "parent_source_sha256": _sha(
                raw["parent_source_sha256"], field="parent source hash"
            ),
            "normalized_clean_sha256": raw["normalized_clean_sha256"],
            "class_id": class_id,
            "clean_text": clean_text,
            "clean_word_char_span": clean_span,
        }
        if role == "fit":
            records.append(ProbeCohortRecord(**common))
            continue
        typo_text = _string(raw["typo_text"], field="probe typo text")
        typo_span = _span(raw["typo_word_char_span"], text=typo_text, field="typo word span")
        if normalized_content_sha256(typo_text) != _sha(
            raw["normalized_noisy_sha256"], field="normalized noisy hash"
        ):
            raise ValueError("probe typo text hash differs")
        edit_type = _string(raw["edit_type"], field="probe edit type")
        if edit_type not in {
            "keyboard-neighbor-substitution",
            "deletion",
            "duplication",
        }:
            raise ValueError("probe edit type is outside the frozen operation inventory")
        observed_edit = classify_character_edit(
            clean=clean_text[slice(*clean_span)],
            typo=typo_text[slice(*typo_span)],
        )
        expected_edit = (
            "natural-statistics-substitution"
            if edit_type == "keyboard-neighbor-substitution"
            else edit_type
        )
        if observed_edit != expected_edit:
            raise ValueError("probe edit type differs from the resolved word pair")
        if edit_type == "keyboard-neighbor-substitution" and not (
            is_keyboard_neighbor_substitution(
                clean=clean_text[slice(*clean_span)],
                typo=typo_text[slice(*typo_span)],
            )
        ):
            raise ValueError("probe keyboard substitution is not a case-preserving neighbor")
        edit_count = _integer(raw["edit_count"], field="probe edit count", minimum=1)
        if edit_count != 1:
            raise ValueError("probe selection and validation require exactly one edit")
        if (
            clean_text[: clean_span[0]] != typo_text[: typo_span[0]]
            or clean_text[clean_span[1] :] != typo_text[typo_span[1] :]
            or clean_text[slice(*clean_span)] == typo_text[slice(*typo_span)]
        ):
            raise ValueError("probe pair must differ only inside the registered word span")
        records.append(
            ProbeCohortRecord(
                **common,
                pair_id=_string(raw["pair_id"], field="probe pair id"),
                normalized_noisy_sha256=raw["normalized_noisy_sha256"],
                edit_type=edit_type,
                edit_count=edit_count,
                token_inflation_bucket=_string(
                    raw["token_inflation_bucket"], field="token inflation bucket"
                ),
                typo_text=typo_text,
                typo_word_char_span=typo_span,
            )
        )
    record_ids = [record.record_id for record in records]
    pair_ids = [record.pair_id for record in records if record.pair_id is not None]
    if len(set(record_ids)) != len(record_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"probe {role} record and pair ids must be unique")
    _validate_within_role_identities(records, role=role)
    counts = Counter(record.class_id for record in records)
    if set(counts) != set(range(len(labels))) or len(set(counts.values())) != 1:
        raise ValueError(f"probe {role} cohort must be exactly class balanced")
    return tuple(records)


def _validate_within_role_identities(
    records: Sequence[ProbeCohortRecord],
    *,
    role: str,
) -> None:
    """Reject pseudo-replication disguised with fresh bootstrap group ids."""

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


def _stratum_key(record: ProbeCohortRecord) -> str:
    if (
        record.edit_type is None
        or record.edit_count is None
        or record.token_inflation_bucket is None
    ):
        raise ValueError("probe paired record lacks one frozen stratum")
    return f"{record.edit_type}|{record.edit_count}|{record.token_inflation_bucket}"


def _validate_preregistered_cohort(
    records: Sequence[ProbeCohortRecord],
    *,
    role: str,
    class_count: int,
    protocol: ProbeProducerProtocol,
) -> None:
    counts = Counter(record.class_id for record in records)
    expected_per_class = protocol.records_per_class[role]
    if counts != Counter({class_id: expected_per_class for class_id in range(class_count)}):
        raise ValueError(f"probe {role} class counts differ from the preregistration")
    minimum_groups = protocol.min_source_groups_per_class[role]
    for class_id in range(class_count):
        groups = {
            record.source_group_sha256
            for record in records
            if record.class_id == class_id
        }
        if len(groups) < minimum_groups:
            raise ValueError(
                f"probe {role} source-group count differs from the preregistration"
            )
    if role in {"selection", "validation"}:
        observed = Counter(_stratum_key(record) for record in records)
        expected = Counter(protocol.stratum_counts[role])
        if observed != expected:
            raise ValueError(f"probe {role} strata differ from the preregistration")


def _identity_sets(
    records: Sequence[ProbeCohortRecord],
) -> tuple[set[str], set[str], set[str]]:
    return (
        {record.source_group_sha256 for record in records},
        {record.parent_source_sha256 for record in records},
        {record.normalized_clean_sha256 for record in records}
        | {
            record.normalized_noisy_sha256
            for record in records
            if record.normalized_noisy_sha256 is not None
        },
    )


def _load_protected_registry(path: Path) -> tuple[set[str], set[str], set[str]]:
    payload = _json_object(path, label="protected split registry")
    if set(payload) != {"schema_version", "registries"}:
        raise ValueError("protected split registry fields differ")
    if payload["schema_version"] != "typo-protected-split-registry/v1":
        raise ValueError("protected split registry schema differs")
    rows = payload["registries"]
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
        for values, field in zip(
            (
                row["source_group_sha256"],
                row["parent_source_sha256"],
                row["normalized_content_sha256"],
            ),
            ("source group", "parent source", "normalized content"),
            strict=True,
        ):
            if not isinstance(values, list):
                raise ValueError(f"protected {field} hashes must be a list")
            tier_sets.append(
                {_sha(value, field=f"protected {field} hash") for value in values}
            )
        if not any(tier_sets):
            raise ValueError("every protected split tier must contain an identity")
        identities_by_tier[tier] = (tier_sets[0], tier_sets[1], tier_sets[2])
    if set(identities_by_tier) != required_tiers:
        raise ValueError("protected split registry tier inventory differs")
    ordered = sorted(required_tiers)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if set().union(*identities_by_tier[left]) & set().union(
                *identities_by_tier[right]
            ):
                raise ValueError("protected split tiers overlap transitively")
    return (
        set().union(*(identities_by_tier[tier][0] for tier in ordered)),
        set().union(*(identities_by_tier[tier][1] for tier in ordered)),
        set().union(*(identities_by_tier[tier][2] for tier in ordered)),
    )


def _validate_role_isolation(
    cohorts: Mapping[str, Sequence[ProbeCohortRecord]],
    protected: tuple[set[str], set[str], set[str]],
) -> None:
    identities = {
        role: set().union(*_identity_sets(cohorts[role])) for role in _ROLES
    }
    for left_index, left in enumerate(_ROLES):
        for right in _ROLES[left_index + 1 :]:
            if identities[left] & identities[right]:
                raise ValueError("probe cohorts overlap transitively across roles")
    protected_union = set().union(*protected)
    if any(identities[role] & protected_union for role in _ROLES):
        raise ValueError("probe cohort overlaps a protected training or evaluation split")


class ProbeActivationProvider(Protocol):
    """Model-dependent, read-only source of edited-word-final residuals."""

    model: str
    model_revision: str
    code_revision: str
    decoder_layers: int
    hidden_size: int
    base_model_frozen: bool

    def activations(
        self,
        records: Sequence[ProbeCohortRecord],
        *,
        side: str,
    ) -> np.ndarray: ...

    def provenance(self) -> Mapping[str, object]: ...

    def token_inflation_bucket(self, record: ProbeCohortRecord) -> str: ...


@dataclass(frozen=True, slots=True)
class _ProbeWeights:
    weight: np.ndarray
    bias: np.ndarray


def _probe_tensor_digest(weights: _ProbeWeights) -> str:
    """Hash numerical probe tensors only, deliberately excluding seed metadata."""

    digest = hashlib.sha256(b"typo-linear-probe-tensors/v1\0")
    for name, value in (("weight", weights.weight), ("bias", weights.bias)):
        array = np.array(value, dtype=np.float32, order="C", copy=True)
        # IEEE-754 has two zero encodings.  Probe independence is numerical,
        # so metadata or the sign bit of an exact zero must not make two fits
        # appear distinct.
        array[array == 0.0] = 0.0
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _activation_matrix(
    provider: ProbeActivationProvider,
    records: Sequence[ProbeCohortRecord],
    *,
    side: str,
    decoder_layers: int,
    hidden_size: int | None,
) -> tuple[np.ndarray, int]:
    raw = provider.activations(records, side=side)
    matrix = np.asarray(raw, dtype=np.float32)
    if (
        matrix.ndim != 3
        or matrix.shape[0] != len(records)
        or matrix.shape[1] != decoder_layers
        or matrix.shape[2] <= 0
        or (hidden_size is not None and matrix.shape[2] != hidden_size)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("probe activation provider returned an invalid activation matrix")
    return np.ascontiguousarray(matrix), int(matrix.shape[2])


def _fit_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    seed: int,
    protocol: ProbeProducerProtocol,
) -> _ProbeWeights:
    import torch

    if labels.shape != (activations.shape[0],):
        raise ValueError("probe fit labels differ from the activation inventory")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.from_numpy(activations)
    targets = torch.from_numpy(labels.astype(np.int64, copy=False))
    scale = 1.0 / math.sqrt(activations.shape[2])
    weight = torch.nn.Parameter(
        torch.randn(
            activations.shape[1],
            activations.shape[2],
            class_count,
            generator=generator,
            dtype=torch.float32,
        )
        * scale
    )
    bias = torch.nn.Parameter(torch.zeros(activations.shape[1], class_count))
    optimizer = torch.optim.AdamW(
        (weight, bias),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
        betas=(protocol.beta1, protocol.beta2),
        eps=protocol.epsilon,
    )
    for _epoch in range(protocol.epochs):
        order = torch.randperm(len(targets), generator=generator)
        for start in range(0, len(targets), protocol.batch_size):
            indices = order[start : start + protocol.batch_size]
            batch = values[indices]
            batch_targets = targets[indices]
            logits = torch.einsum("bld,ldc->blc", batch, weight) + bias
            expanded = batch_targets[:, None].expand(-1, activations.shape[1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, class_count), expanded.reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return _ProbeWeights(
        weight=weight.detach().cpu().numpy(),
        bias=bias.detach().cpu().numpy(),
    )


def _cross_entropy_by_layer(
    activations: np.ndarray,
    labels: np.ndarray,
    weights: _ProbeWeights,
) -> np.ndarray:
    logits = np.einsum("nld,ldc->nlc", activations, weights.weight) + weights.bias[None]
    maximum = logits.max(axis=2, keepdims=True)
    log_partition = maximum[..., 0] + np.log(np.exp(logits - maximum).sum(axis=2))
    correct = np.take_along_axis(logits, labels[:, None, None], axis=2)[..., 0]
    losses = log_partition - correct
    if not np.isfinite(losses).all() or (losses < -1e-6).any():
        raise FloatingPointError("linear probe produced invalid cross-entropy")
    return np.maximum(losses, 0.0)


def _score_payload(
    records: Sequence[ProbeCohortRecord],
    *,
    role: str,
    seed: int,
    decoder_layers: int,
    clean_loss: np.ndarray,
    noisy_loss: np.ndarray,
    bindings: Mapping[str, str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if (
            record.pair_id is None
            or record.edit_type is None
            or record.edit_count is None
            or record.token_inflation_bucket is None
        ):
            raise ValueError("probe score cohort is not paired")
        rows.append(
            {
                "pair_id": record.pair_id,
                "source_group_sha256": record.source_group_sha256,
                "class_id": record.class_id,
                "edit_type": record.edit_type,
                "edit_count": record.edit_count,
                "token_inflation_bucket": record.token_inflation_bucket,
                "clean_cross_entropy": [float(value) for value in clean_loss[index]],
                "noisy_cross_entropy": [float(value) for value in noisy_loss[index]],
            }
        )
    return {
        "schema_version": "typo-paired-probe-scores/v1",
        "role": role,
        "seed": seed,
        "decoder_layers": decoder_layers,
        "bindings": dict(bindings),
        "records": rows,
    }


def _group_mean_trajectory(payload: Mapping[str, object]) -> ProbeSeedTrajectory:
    raw_rows = payload["records"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("probe score payload has no records")
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("probe score row must be an object")
        groups[str(raw["source_group_sha256"])].append(raw)
    decoder_layers = int(payload["decoder_layers"])
    clean: list[float] = []
    noisy: list[float] = []
    for layer in range(decoder_layers):
        clean_groups = [
            sum(float(row["clean_cross_entropy"][layer]) for row in rows) / len(rows)  # type: ignore[index]
            for rows in groups.values()
        ]
        noisy_groups = [
            sum(float(row["noisy_cross_entropy"][layer]) for row in rows) / len(rows)  # type: ignore[index]
            for rows in groups.values()
        ]
        clean.append(sum(clean_groups) / len(clean_groups))
        noisy.append(sum(noisy_groups) / len(noisy_groups))
    return ProbeSeedTrajectory(int(payload["seed"]), tuple(clean), tuple(noisy))


def _bootstrap_lower_bound(
    payload: Mapping[str, object],
    *,
    selected_layer: int,
    seed: int,
    resamples: int,
    confidence: float,
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    raw_rows = payload["records"]
    assert isinstance(raw_rows, list)
    for raw in raw_rows:
        assert isinstance(raw, Mapping)
        clean = raw["clean_cross_entropy"]
        noisy = raw["noisy_cross_entropy"]
        assert isinstance(clean, list) and isinstance(noisy, list)
        before = float(noisy[selected_layer - 1]) - float(clean[selected_layer - 1])
        after = float(noisy[selected_layer]) - float(clean[selected_layer])
        grouped[str(raw["source_group_sha256"])].append(before - after)
    group_values = tuple(
        sum(values) / len(values) for _group, values in sorted(grouped.items())
    )
    if len(group_values) < 2:
        raise ValueError("probe validation requires at least two source groups")
    samples: list[float] = []
    for replicate in range(resamples):
        total = 0.0
        for draw in range(len(group_values)):
            material = f"probe-bootstrap/v1\0{seed}\0{replicate}\0{draw}".encode()
            index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(
                group_values
            )
            total += group_values[index]
        samples.append(total / len(group_values))
    samples.sort()
    lower_tail = (1.0 - confidence) / 2.0
    return samples[max(0, math.ceil(lower_tail * len(samples)) - 1)]


def _validation_peak(trajectory: ProbeSeedTrajectory) -> int:
    maximum = max(trajectory.transition_drop)
    return next(
        index for index, value in enumerate(trajectory.transition_drop) if value == maximum
    ) + 1


def _addressed_copy(source: Path, output_dir: Path, *, label: str) -> Path:
    digest = sha256_file(source)
    suffix = source.suffix or ".bin"
    target = output_dir / f"{label}-{digest}{suffix}"
    target.write_bytes(source.read_bytes())
    return target


def _addressed_json(output_dir: Path, *, label: str, payload: object) -> Path:
    temporary = output_dir / f".{label}.json"
    write_json_atomic(temporary, payload)
    digest = sha256_file(temporary)
    target = output_dir / f"{label}-{digest}.json"
    temporary.replace(target)
    return target


def _addressed_weights(
    output_dir: Path,
    *,
    seed: int,
    weights: _ProbeWeights,
    protocol: ProbeProducerProtocol,
    class_count: int,
) -> Path:
    from safetensors.numpy import save_file

    if weights.weight.shape != (
        protocol.decoder_layers,
        protocol.hidden_size,
        class_count,
    ) or weights.bias.shape != (protocol.decoder_layers, class_count):
        raise ValueError("probe weight tensors differ from the preregistered inventory")
    tensors: dict[str, np.ndarray] = {}
    for layer in range(protocol.decoder_layers):
        tensors[f"decoder_layer.{layer}.weight"] = np.ascontiguousarray(
            weights.weight[layer].T,
            dtype=np.float32,
        )
        tensors[f"decoder_layer.{layer}.bias"] = np.ascontiguousarray(
            weights.bias[layer],
            dtype=np.float32,
        )
    metadata = {
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
    temporary = output_dir / f".probe-weights-seed-{seed}.safetensors"
    save_file(tensors, temporary, metadata=metadata)
    digest = sha256_file(temporary)
    target = output_dir / f"probe-weights-seed-{seed}-{digest}.safetensors"
    temporary.replace(target)
    return target


def _reference(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True, slots=True)
class ProbeTransitionProducerRunConfig:
    config_path: Path
    class_inventory_path: Path
    fit_manifest_path: Path
    selection_manifest_path: Path
    validation_manifest_path: Path
    protected_registry_path: Path
    gpu_id: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ProbeTransitionProducerResult:
    selected_transition_layer: int
    validation_passed: bool
    artifact_path: Path
    run_path: Path
    weights_by_seed: Mapping[int, Path]
    selection_scores_by_seed: Mapping[int, Path]
    validation_scores_by_seed: Mapping[int, Path]


ProviderFactory = Callable[[ProbeProducerProtocol, str], ProbeActivationProvider]


def _default_provider_factory(
    protocol: ProbeProducerProtocol,
    gpu_id: str,
) -> ProbeActivationProvider:
    from typo_robust_training.probe.runtime import HuggingFaceProbeActivationProvider

    return HuggingFaceProbeActivationProvider(protocol=protocol, gpu_id=gpu_id)


def run_select_probe_transition(
    config: ProbeTransitionProducerRunConfig,
    *,
    activation_provider: ProbeActivationProvider | None = None,
    provider_factory: ProviderFactory = _default_provider_factory,
) -> ProbeTransitionProducerResult:
    """Fit two probes, score paired cohorts, and derive one transition layer."""

    if not isinstance(config, ProbeTransitionProducerRunConfig):
        raise TypeError("probe producer run config has the wrong type")
    protocol = load_probe_producer_config(config.config_path)
    input_paths = {
        "class_inventory": config.class_inventory_path,
        "fit_manifest": config.fit_manifest_path,
        "selection_manifest": config.selection_manifest_path,
        "validation_manifest": config.validation_manifest_path,
        "protected_split_registry": config.protected_registry_path,
    }
    actual_input_hashes = {
        name: sha256_file(Path(path).resolve()) for name, path in input_paths.items()
    }
    if actual_input_hashes != dict(protocol.input_sha256):
        raise ValueError("probe producer input hashes differ from the preregistration")
    labels = _load_classes(config.class_inventory_path)
    cohorts = {
        "fit": _load_cohort(config.fit_manifest_path, role="fit", labels=labels),
        "selection": _load_cohort(
            config.selection_manifest_path, role="selection", labels=labels
        ),
        "validation": _load_cohort(
            config.validation_manifest_path, role="validation", labels=labels
        ),
    }
    for role in _ROLES:
        _validate_preregistered_cohort(
            cohorts[role],
            role=role,
            class_count=len(labels),
            protocol=protocol,
        )
    protected = _load_protected_registry(config.protected_registry_path)
    _validate_role_isolation(cohorts, protected)
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"probe producer output directory is not empty: {output_dir}")

    provider = activation_provider or provider_factory(protocol, config.gpu_id)
    if (
        provider.model != protocol.model
        or provider.model_revision != protocol.model_revision
        or provider.code_revision != protocol.code_revision
        or provider.decoder_layers != protocol.decoder_layers
        or provider.hidden_size != protocol.hidden_size
        or provider.base_model_frozen is not True
    ):
        raise ValueError("probe activation provider identity or freeze contract differs")
    provider_provenance = dict(provider.provenance())
    expected_provider_identity = {
        "model": protocol.model,
        "model_revision": protocol.model_revision,
        "code_revision": protocol.code_revision,
        "base_model_frozen": True,
    }
    if any(
        provider_provenance.get(field) != value
        for field, value in expected_provider_identity.items()
    ):
        raise ValueError("probe activation provider provenance identity differs")
    for role in ("selection", "validation"):
        for record in cohorts[role]:
            observed_bucket = provider.token_inflation_bucket(record)
            if observed_bucket != record.token_inflation_bucket:
                raise ValueError(
                    "probe token inflation bucket differs from the runtime tokenizer"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_activations, hidden_size = _activation_matrix(
        provider,
        cohorts["fit"],
        side="clean",
        decoder_layers=protocol.decoder_layers,
        hidden_size=protocol.hidden_size,
    )
    fit_labels = np.asarray([record.class_id for record in cohorts["fit"]], dtype=np.int64)
    paired_activations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for role in ("selection", "validation"):
        clean, clean_hidden = _activation_matrix(
            provider,
            cohorts[role],
            side="clean",
            decoder_layers=protocol.decoder_layers,
            hidden_size=hidden_size,
        )
        noisy, noisy_hidden = _activation_matrix(
            provider,
            cohorts[role],
            side="typo",
            decoder_layers=protocol.decoder_layers,
            hidden_size=hidden_size,
        )
        if clean_hidden != noisy_hidden:
            raise ValueError("probe clean/noisy hidden dimensions differ")
        paired_activations[role] = clean, noisy

    copied = {
        "config": _addressed_copy(config.config_path, output_dir, label="probe-config"),
        "class_inventory": _addressed_copy(
            config.class_inventory_path, output_dir, label="class-inventory"
        ),
        "fit_manifest": _addressed_copy(
            config.fit_manifest_path, output_dir, label="fit-manifest"
        ),
        "selection_manifest": _addressed_copy(
            config.selection_manifest_path, output_dir, label="selection-manifest"
        ),
        "validation_manifest": _addressed_copy(
            config.validation_manifest_path, output_dir, label="validation-manifest"
        ),
        "protected_split_registry": _addressed_copy(
            config.protected_registry_path, output_dir, label="protected-registry"
        ),
    }
    weights_by_seed: dict[int, Path] = {}
    selection_paths: dict[int, Path] = {}
    validation_paths: dict[int, Path] = {}
    selection_payloads: dict[int, dict[str, object]] = {}
    validation_payloads: dict[int, dict[str, object]] = {}
    tensor_digests: set[str] = set()
    for seed in protocol.probe_seeds:
        weights = _fit_probe(
            fit_activations,
            fit_labels,
            class_count=len(labels),
            seed=seed,
            protocol=protocol,
        )
        tensor_digest = _probe_tensor_digest(weights)
        if tensor_digest in tensor_digests:
            raise ValueError(
                "independent probe seeds produced identical numerical tensors"
            )
        tensor_digests.add(tensor_digest)
        weights_by_seed[seed] = _addressed_weights(
            output_dir,
            seed=seed,
            weights=weights,
            protocol=protocol,
            class_count=len(labels),
        )
        weight_sha256 = sha256_file(weights_by_seed[seed])
        for role in ("selection", "validation"):
            records = cohorts[role]
            role_labels = np.asarray([record.class_id for record in records], dtype=np.int64)
            clean_loss = _cross_entropy_by_layer(
                paired_activations[role][0], role_labels, weights
            )
            noisy_loss = _cross_entropy_by_layer(
                paired_activations[role][1], role_labels, weights
            )
            payload = _score_payload(
                records,
                role=role,
                seed=seed,
                decoder_layers=protocol.decoder_layers,
                clean_loss=clean_loss,
                noisy_loss=noisy_loss,
                bindings={
                    "model": protocol.model,
                    "model_revision": protocol.model_revision,
                    "code_revision": protocol.code_revision,
                    "config_sha256": protocol.config_sha256,
                    "class_inventory_sha256": protocol.input_sha256[
                        "class_inventory"
                    ],
                    "fit_manifest_sha256": protocol.input_sha256["fit_manifest"],
                    "role_manifest_sha256": protocol.input_sha256[f"{role}_manifest"],
                    "probe_weights_sha256": weight_sha256,
                },
            )
            path = _addressed_json(
                output_dir, label=f"{role}-scores-seed-{seed}", payload=payload
            )
            if role == "selection":
                selection_payloads[seed] = payload
                selection_paths[seed] = path
            else:
                validation_payloads[seed] = payload
                validation_paths[seed] = path

    if len({sha256_file(path) for path in weights_by_seed.values()}) != len(
        protocol.probe_seeds
    ):
        raise ValueError("independent probe seeds produced identical weight artifacts")

    selection: ProbeTransitionSelection = select_probe_transition(
        tuple(
            _group_mean_trajectory(selection_payloads[seed])
            for seed in protocol.probe_seeds
        )
    )
    validation_trajectories = {
        seed: _group_mean_trajectory(validation_payloads[seed])
        for seed in protocol.probe_seeds
    }
    selection_lower = {
        seed: _bootstrap_lower_bound(
            selection_payloads[seed],
            selected_layer=selection.selected_layer,
            seed=protocol.bootstrap_seed,
            resamples=protocol.bootstrap_resamples,
            confidence=protocol.bootstrap_confidence,
        )
        for seed in protocol.probe_seeds
    }
    validation_lower = {
        seed: _bootstrap_lower_bound(
            validation_payloads[seed],
            selected_layer=selection.selected_layer,
            seed=protocol.bootstrap_seed,
            resamples=protocol.bootstrap_resamples,
            confidence=protocol.bootstrap_confidence,
        )
        for seed in protocol.probe_seeds
    }
    passed = (
        all(layer == selection.selected_layer for _seed, layer in selection.seed_selected_layers)
        and all(value > 0.0 for value in selection_lower.values())
        and all(
            abs(_validation_peak(validation_trajectories[seed]) - selection.selected_layer) <= 1
            for seed in protocol.probe_seeds
        )
        and all(value > 0.0 for value in validation_lower.values())
    )
    references = {
        **{name: _reference(path, root=output_dir) for name, path in copied.items()},
        "probe_weights_by_seed": {
            str(seed): _reference(weights_by_seed[seed], root=output_dir)
            for seed in protocol.probe_seeds
        },
        "selection_scores_by_seed": {
            str(seed): _reference(selection_paths[seed], root=output_dir)
            for seed in protocol.probe_seeds
        },
        "validation_scores_by_seed": {
            str(seed): _reference(validation_paths[seed], root=output_dir)
            for seed in protocol.probe_seeds
        },
    }
    artifact_payload = {
        "schema_version": "typo-denoising-probe-selection/v2",
        "operation": "select-linear-probe-denoising-transition",
        "model": protocol.model,
        "model_revision": protocol.model_revision,
        "decoder_layers": protocol.decoder_layers,
        "hook_site": protocol.hook_site,
        "coordinate": protocol.coordinate,
        "probe_seeds": list(protocol.probe_seeds),
        "references": references,
        "selection_metric": protocol.selection_metric,
        "selection_rule": protocol.selection_rule,
        "tie_break": protocol.tie_break,
        "stability_rule": protocol.stability_rule,
        "validation_rule": protocol.validation_rule,
        "bootstrap": {
            "resamples": protocol.bootstrap_resamples,
            "seed": protocol.bootstrap_seed,
            "confidence": protocol.bootstrap_confidence,
            "unit": protocol.bootstrap_unit,
        },
        "selected_transition_layer": selection.selected_layer,
        "validation_passed": passed,
    }
    artifact_path = _addressed_json(
        output_dir, label="probe-transition", payload=artifact_payload
    )
    run_path = output_dir / "run.json"
    write_json_atomic(
        run_path,
        {
            "schema_version": "typo-linear-probe-producer-run/v1",
            "operation": "select-linear-probe-denoising-transition",
            "status": "completed" if passed else "validation-failed",
            "config_sha256": protocol.config_sha256,
            "input_sha256": actual_input_hashes,
            "model": protocol.model,
            "model_revision": protocol.model_revision,
            "code_revision": protocol.code_revision,
            "decoder_layers": protocol.decoder_layers,
            "hidden_size": protocol.hidden_size,
            "probe_seeds": list(protocol.probe_seeds),
            "base_model_frozen": True,
            "selected_transition_layer": selection.selected_layer,
            "selection_ci_lower_by_seed": {
                str(seed): selection_lower[seed] for seed in protocol.probe_seeds
            },
            "validation_ci_lower_by_seed": {
                str(seed): validation_lower[seed] for seed in protocol.probe_seeds
            },
            "validation_passed": passed,
            "artifact": _reference(artifact_path, root=output_dir),
            "runtime": provider_provenance,
            "python": platform.python_version(),
        },
    )
    return ProbeTransitionProducerResult(
        selected_transition_layer=selection.selected_layer,
        validation_passed=passed,
        artifact_path=artifact_path,
        run_path=run_path,
        weights_by_seed=weights_by_seed,
        selection_scores_by_seed=selection_paths,
        validation_scores_by_seed=validation_paths,
    )


__all__ = [
    "ProbeActivationProvider",
    "ProbeCohortRecord",
    "ProbeTransitionProducerResult",
    "ProbeTransitionProducerRunConfig",
    "run_select_probe_transition",
]
