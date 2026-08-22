"""Content-addressed causal-gate evidence for single-layer state KD."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.perturb import (
    classify_character_edit,
    is_keyboard_neighbor_substitution,
)
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.probe import load_probe_transition_artifact
from typo_robust_training.state_gate.config import (
    SingleLayerGateProtocol,
    load_single_layer_gate_config,
)
from typo_robust_training.state_gate.scoring import (
    GateObservation,
    GateScore,
    score_single_layer_gate,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REFERENCE_FIELDS = {"relative_path", "sha256"}
_TOP = {
    "schema_version",
    "operation",
    "model",
    "model_revision",
    "code_revision",
    "decoder_layers",
    "selected_transition_layer",
    "hook_site",
    "coordinate",
    "readout",
    "controls",
    "references",
    "passed",
}
_REFERENCES = {
    "config",
    "parent_probe_artifact",
    "cohort_manifest",
    "protected_split_registry",
    "donor_plan",
    "runtime_manifest",
    "raw_kl",
}
_MANIFEST_FIELDS = {"schema_version", "role", "records"}
_RECORD_FIELDS = {
    "record_id",
    "pair_id",
    "source_group_sha256",
    "parent_source_sha256",
    "normalized_clean_sha256",
    "normalized_noisy_sha256",
    "clean_text",
    "typo_text",
    "clean_word_char_span",
    "typo_word_char_span",
    "edit_type",
    "edit_count",
    "token_inflation_bucket",
}
_RAW_FIELDS = {"schema_version", "bindings", "records"}
_RAW_BINDINGS = {
    "config_sha256",
    "parent_probe_artifact_sha256",
    "cohort_manifest_sha256",
    "protected_registry_sha256",
    "donor_plan_sha256",
    "runtime_manifest_sha256",
}
_RAW_RECORD_FIELDS = {
    "pair_id",
    "source_group_sha256",
    "stratum",
    "transition_layer",
    "clean_word_final_token",
    "typo_word_final_token",
    "offset_donor_clean_token",
    "offset_patch_token",
    "cross_donor_pair_id",
    "cross_donor_clean_word_final_token",
    "cross_donor_clean_prompt_offsets",
    "clean_prompt_offsets",
    "typo_prompt_offsets",
    "target_token_ids",
    "untreated_kl_2_16",
    "correct_kl_2_16",
    "offset_kl_2_16",
    "cross_kl_2_16",
    "self_copy_kl_2_16",
    "invalid_reason",
}


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _json(path: Path) -> Mapping[str, object]:
    try:
        value = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    except UnicodeDecodeError as exc:
        raise ValueError(f"gate artifact reference is not UTF-8: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"gate artifact reference must be an object: {path}")
    return value


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _reference(value: object, *, root: Path, field: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != _REFERENCE_FIELDS:
        raise ValueError(f"{field} reference fields differ")
    relative_raw = value["relative_path"]
    if not isinstance(relative_raw, str) or not relative_raw:
        raise ValueError(f"{field} relative path must be non-empty")
    relative = PurePosixPath(relative_raw)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != relative_raw:
        raise ValueError(f"{field} reference must be a canonical relative POSIX path")
    candidate = root / Path(*relative.parts)
    if candidate.is_symlink():
        raise ValueError(f"{field} reference is a symlink")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} reference escapes the gate bundle") from exc
    expected = _sha(value["sha256"], field=f"{field} hash")
    if not path.is_file() or path.is_symlink() or _digest(path) != expected:
        raise ValueError(f"{field} reference is missing, a symlink, or hash-mismatched")
    return path, expected


def _span(value: object, *, text: str, field: str) -> tuple[int, int]:
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


@dataclass(frozen=True, slots=True)
class SingleLayerGateRecord:
    record_id: str
    pair_id: str
    source_group_sha256: str
    parent_source_sha256: str
    normalized_clean_sha256: str
    normalized_noisy_sha256: str
    clean_text: str
    typo_text: str
    clean_word_char_span: tuple[int, int]
    typo_word_char_span: tuple[int, int]
    edit_type: str
    edit_count: int
    token_inflation_bucket: str

    @property
    def stratum(self) -> str:
        return f"{self.edit_type}|{self.edit_count}|{self.token_inflation_bucket}"

    @property
    def identities(self) -> frozenset[str]:
        return frozenset(
            {
                self.source_group_sha256,
                self.parent_source_sha256,
                self.normalized_clean_sha256,
                self.normalized_noisy_sha256,
            }
        )


def load_gate_cohort_manifest(
    path: Path,
    *,
    protocol: SingleLayerGateProtocol,
) -> tuple[SingleLayerGateRecord, ...]:
    value = _json(path)
    if set(value) != _MANIFEST_FIELDS or value["schema_version"] != "typo-single-layer-gate-cohort/v1" or value["role"] != "independent-generic-fineweb":
        raise ValueError("single-layer gate cohort manifest identity differs")
    rows = value["records"]
    if not isinstance(rows, list) or len(rows) != protocol.records:
        raise ValueError("single-layer gate cohort record count differs")
    parsed: list[SingleLayerGateRecord] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _RECORD_FIELDS:
            raise ValueError("single-layer gate cohort row fields differ")
        clean = _string(row["clean_text"], field="gate clean text")
        typo = _string(row["typo_text"], field="gate typo text")
        clean_hash = _sha(row["normalized_clean_sha256"], field="gate clean hash")
        typo_hash = _sha(row["normalized_noisy_sha256"], field="gate typo hash")
        if normalized_content_sha256(clean) != clean_hash or normalized_content_sha256(typo) != typo_hash or clean == typo:
            raise ValueError("single-layer gate resolved text hash differs or pair is a no-op")
        clean_span = _span(row["clean_word_char_span"], text=clean, field="clean span")
        typo_span = _span(row["typo_word_char_span"], text=typo, field="typo span")
        if clean[: clean_span[0]] != typo[: typo_span[0]] or clean[clean_span[1] :] != typo[typo_span[1] :]:
            raise ValueError("single-layer gate pair must isolate one edited word")
        edit_type = _string(row["edit_type"], field="gate edit type")
        observed = classify_character_edit(
            clean=clean[slice(*clean_span)], typo=typo[slice(*typo_span)]
        )
        expected = "natural-statistics-substitution" if edit_type == "keyboard-neighbor-substitution" else edit_type
        if edit_type not in {"keyboard-neighbor-substitution", "deletion", "duplication"} or observed != expected:
            raise ValueError("single-layer gate edit operation differs")
        if edit_type == "keyboard-neighbor-substitution" and not is_keyboard_neighbor_substitution(
            clean=clean[slice(*clean_span)], typo=typo[slice(*typo_span)]
        ):
            raise ValueError("single-layer gate substitution is not a QWERTY neighbor")
        parsed.append(
            SingleLayerGateRecord(
                record_id=_string(row["record_id"], field="gate record id"),
                pair_id=_string(row["pair_id"], field="gate pair id"),
                source_group_sha256=_sha(row["source_group_sha256"], field="gate source group"),
                parent_source_sha256=_sha(row["parent_source_sha256"], field="gate parent source"),
                normalized_clean_sha256=clean_hash,
                normalized_noisy_sha256=typo_hash,
                clean_text=clean,
                typo_text=typo,
                clean_word_char_span=clean_span,
                typo_word_char_span=typo_span,
                edit_type=edit_type,
                edit_count=_integer(row["edit_count"], field="gate edit count", minimum=1),
                token_inflation_bucket=_string(row["token_inflation_bucket"], field="gate token inflation"),
            )
        )
        if parsed[-1].edit_count != 1:
            raise ValueError("single-layer gate requires exactly one typo")
    if len({row.record_id for row in parsed}) != len(parsed) or len({row.pair_id for row in parsed}) != len(parsed):
        raise ValueError("single-layer gate record and pair ids must be unique")
    normalized_hashes = [
        digest
        for row in parsed
        for digest in (row.normalized_clean_sha256, row.normalized_noisy_sha256)
    ]
    if len(set(normalized_hashes)) != len(normalized_hashes):
        raise ValueError(
            "single-layer gate normalized clean/noisy content must be unique"
        )
    parent_to_group: dict[str, str] = {}
    for row in parsed:
        previous = parent_to_group.setdefault(
            row.parent_source_sha256,
            row.source_group_sha256,
        )
        if previous != row.source_group_sha256:
            raise ValueError(
                "single-layer gate parent source maps to multiple bootstrap groups"
            )
    if Counter(row.stratum for row in parsed) != Counter(protocol.stratum_counts):
        raise ValueError("single-layer gate manifest strata differ from preregistration")
    return tuple(parsed)


def _load_protected_registry(path: Path) -> frozenset[str]:
    value = _json(path)
    if set(value) != {"schema_version", "registries"} or value["schema_version"] != "typo-protected-split-registry/v1":
        raise ValueError("gate protected split registry identity differs")
    rows = value["registries"]
    required = {"training", "localization", "tune", "pre-pr", "sealed"}
    if (
        not isinstance(rows, list)
        or len(rows) != len(required)
        or {row.get("tier") for row in rows if isinstance(row, Mapping)} != required
    ):
        raise ValueError("gate protected split tier inventory differs")
    by_tier: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"tier", "source_group_sha256", "parent_source_sha256", "normalized_content_sha256"}:
            raise ValueError("gate protected split row fields differ")
        tier = _string(row["tier"], field="gate protected tier")
        identities: set[str] = set()
        for field in ("source_group_sha256", "parent_source_sha256", "normalized_content_sha256"):
            values = row[field]
            if not isinstance(values, list):
                raise ValueError("gate protected identity inventory must be lists")
            identities.update(_sha(item, field=f"gate protected {field}") for item in values)
        if not identities:
            raise ValueError("gate protected tier must not be empty")
        if tier in by_tier:
            raise ValueError("gate protected split tier names must be unique")
        by_tier[tier] = identities
    for left_index, left in enumerate(sorted(required)):
        for right in sorted(required)[left_index + 1 :]:
            if by_tier[left] & by_tier[right]:
                raise ValueError("gate protected tiers overlap transitively")
    return frozenset().union(*(frozenset(values) for values in by_tier.values()))


def deterministic_cross_item_donor_plan(
    records: Sequence[SingleLayerGateRecord],
) -> Mapping[str, str]:
    """Return the first cyclic permutation with no self/source-group donor."""

    rows = tuple(sorted(records, key=lambda row: row.pair_id))
    if len(rows) < 2:
        raise ValueError("cross-item donor plan requires at least two records")
    for shift in range(1, len(rows)):
        donors = rows[shift:] + rows[:shift]
        if all(row.source_group_sha256 != donor.source_group_sha256 for row, donor in zip(rows, donors, strict=True)):
            return MappingProxyType(
                {row.pair_id: donor.pair_id for row, donor in zip(rows, donors, strict=True)}
            )
    raise ValueError("gate cohort has no source-group-disjoint cyclic derangement")


def _load_donor_plan(
    path: Path, *, records: Sequence[SingleLayerGateRecord]
) -> Mapping[str, str]:
    value = _json(path)
    if set(value) != {"schema_version", "rule", "records"} or value["schema_version"] != "typo-cross-item-donor-plan/v1" or value["rule"] != "first-valid-cyclic-source-group-derangement/v1":
        raise ValueError("single-layer gate donor plan identity differs")
    rows = value["records"]
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) or set(row) != {"pair_id", "donor_pair_id"} for row in rows):
        raise ValueError("single-layer gate donor plan rows differ")
    observed = {
        _string(row["pair_id"], field="gate donor pair id"): _string(
            row["donor_pair_id"], field="gate donor target pair id"
        )
        for row in rows
    }
    if len(observed) != len(rows) or observed != dict(deterministic_cross_item_donor_plan(records)):
        raise ValueError("single-layer gate donor plan differs from deterministic derivation")
    return MappingProxyType(observed)


def _load_runtime_manifest(
    path: Path, *, protocol: SingleLayerGateProtocol
) -> Mapping[str, object]:
    value = _json(path)
    expected = {
        "schema_version", "provider", "model", "model_revision", "teacher_revision",
        "student_revision", "tokenizer_revision", "code_revision", "source_tree_sha256",
        "decoder_layers", "dtype", "hook_site", "coordinate", "readout",
        "base_model_frozen", "packages", "hardware",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("single-layer gate runtime manifest fields differ")
    if (
        value["schema_version"] != "single-layer-gate-runtime/v2"
        or value["provider"] != "hugging-face-single-layer-gate/v1"
        or value["model"] != protocol.model
        or value["model_revision"] != protocol.model_revision
        or value["teacher_revision"] != protocol.model_revision
        or value["student_revision"] != protocol.model_revision
        or value["tokenizer_revision"] != protocol.model_revision
        or value["code_revision"] != protocol.code_revision
        or value["decoder_layers"] != protocol.decoder_layers
        or value["dtype"] != "bfloat16"
        or value["hook_site"] != "complete-decoder-block-residual-output"
        or value["coordinate"] != "edited-word-final-token/v1"
        or value["readout"] != "teacher-forced-tokens-2-through-16-inclusive/v1"
        or value["base_model_frozen"] is not True
        or not isinstance(value["packages"], Mapping)
        or not isinstance(value["hardware"], Mapping)
    ):
        raise ValueError("single-layer gate runtime manifest identity differs")
    _sha(value["source_tree_sha256"], field="single-layer gate runtime source tree")
    json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def _offsets(value: object, *, field: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty offset list")
    result: list[tuple[int, int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in row):
            raise ValueError(f"{field} contains an invalid token offset")
        result.append((row[0], row[1]))
    return tuple(result)


def _word_final(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> int:
    positions = [index for index, (start, stop) in enumerate(offsets) if stop > start and start < span[1] and stop > span[0]]
    if not positions or max(offsets[index][1] for index in positions) < span[1]:
        raise ValueError("gate raw offsets do not cover the edited word")
    return positions[-1]


def _trajectory(value: object, *, field: str, valid: bool) -> tuple[float, ...]:
    expected = 15 if valid else 0
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{field} trajectory length differs")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError(f"{field} trajectory values differ")
    return result


def _load_raw_observations(
    path: Path,
    *,
    protocol: SingleLayerGateProtocol,
    records: Sequence[SingleLayerGateRecord],
    donor_plan: Mapping[str, str],
    transition_layer: int,
    expected_bindings: Mapping[str, str],
) -> tuple[GateObservation, ...]:
    value = _json(path)
    if set(value) != _RAW_FIELDS or value["schema_version"] != "single-layer-gate-raw-kl/v1":
        raise ValueError("single-layer gate raw KL identity differs")
    if not isinstance(value["bindings"], Mapping) or set(value["bindings"]) != _RAW_BINDINGS or dict(value["bindings"]) != dict(expected_bindings):
        raise ValueError("single-layer gate raw KL provenance differs")
    rows = value["records"]
    if not isinstance(rows, list) or len(rows) != len(records):
        raise ValueError("single-layer gate raw KL inventory differs")
    manifests = {row.pair_id: row for row in records}
    result: list[GateObservation] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != _RAW_RECORD_FIELDS:
            raise ValueError("single-layer gate raw KL row fields differ")
        pair_id = _string(raw["pair_id"], field="gate raw pair id")
        if pair_id not in manifests:
            raise ValueError("single-layer gate raw row is outside the cohort")
        record = manifests[pair_id]
        donor = manifests[donor_plan[pair_id]]
        if raw["source_group_sha256"] != record.source_group_sha256 or raw["stratum"] != record.stratum or raw["cross_donor_pair_id"] != donor.pair_id:
            raise ValueError("single-layer gate raw row metadata differs")
        layer = _integer(raw["transition_layer"], field="gate raw transition layer", minimum=1)
        if layer != transition_layer:
            raise ValueError("single-layer gate raw row patched the wrong layer")
        clean_offsets = _offsets(raw["clean_prompt_offsets"], field="gate clean prompt offsets")
        typo_offsets = _offsets(raw["typo_prompt_offsets"], field="gate typo prompt offsets")
        clean_position = _integer(raw["clean_word_final_token"], field="gate clean word-final")
        typo_position = _integer(raw["typo_word_final_token"], field="gate typo word-final")
        if clean_position != _word_final(clean_offsets, record.clean_word_char_span) or typo_position != _word_final(typo_offsets, record.typo_word_char_span):
            raise ValueError("single-layer gate raw row patched a non-word-final token")
        offset_donor = _integer(
            raw["offset_donor_clean_token"],
            field="gate raw offset donor clean token",
        )
        if offset_donor != clean_position + protocol.offset_control_tokens:
            raise ValueError("single-layer gate raw offset donor differs from clean +2")
        if raw["offset_patch_token"] != typo_position + protocol.offset_control_tokens:
            raise ValueError("single-layer gate raw offset control differs from +2")
        cross_position = _integer(
            raw["cross_donor_clean_word_final_token"],
            field="gate cross donor word-final",
        )
        cross_offsets = _offsets(
            raw["cross_donor_clean_prompt_offsets"],
            field="gate cross donor clean prompt offsets",
        )
        if cross_position != _word_final(cross_offsets, donor.clean_word_char_span):
            raise ValueError("single-layer gate cross donor patched a non-word-final token")
        invalid_reason = raw["invalid_reason"]
        valid = invalid_reason is None
        if not valid and (not isinstance(invalid_reason, str) or not invalid_reason):
            raise ValueError("single-layer gate invalid row requires a reason")
        target_ids = raw["target_token_ids"]
        if not isinstance(target_ids, list) or len(target_ids) != (16 if valid else 0) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in target_ids):
            raise ValueError("single-layer gate target token inventory differs")
        untreated = _trajectory(raw["untreated_kl_2_16"], field="untreated", valid=valid)
        patched = {
            condition: _trajectory(raw[f"{field}_kl_2_16"], field=condition, valid=valid)
            for condition, field in (
                ("correct", "correct"), ("offset", "offset"), ("cross", "cross"), ("self_copy", "self_copy")
            )
        }
        result.append(
            GateObservation(
                pair_id=pair_id,
                source_group_sha256=record.source_group_sha256,
                stratum=record.stratum,
                transition_layer=layer,
                untreated_kl_2_16=untreated,
                patched_kl_2_16=patched if valid else {},
                invalid_reason=invalid_reason,
            )
        )
    if {row.pair_id for row in result} != set(manifests):
        raise ValueError("single-layer gate raw KL must cover the cohort exactly once")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SingleLayerGateArtifact:
    model: str
    model_revision: str
    decoder_layers: int
    selected_transition_layer: int
    parent_probe_artifact_sha256: str
    artifact_sha256: str
    config_sha256: str
    valid_records: int
    scores: Mapping[str, GateScore]


def load_single_layer_gate_artifact(path: Path) -> SingleLayerGateArtifact:
    """Load all bound inputs and recompute every causal gate decision."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("single-layer gate artifact must not be a symlink")
    resolved = supplied.resolve()
    if not resolved.is_file():
        raise ValueError(f"single-layer gate artifact is not a file: {resolved}")
    raw_artifact = resolved.read_bytes()
    payload = _json(resolved)
    if set(payload) != _TOP:
        raise ValueError("single-layer gate artifact fields differ")
    expected_identity = {
        "schema_version": "probe-transition-single-layer-gate/v1",
        "operation": "validate-probe-transition-single-layer-causal-gate",
        "hook_site": "complete-decoder-block-residual-output",
        "coordinate": "edited-word-final-token/v1",
        "readout": "teacher-forced-tokens-2-through-16-inclusive/v1",
        "controls": ["offset-plus-two", "cross-item-derangement", "self-copy-identity"],
    }
    for field, expected in expected_identity.items():
        if payload[field] != expected:
            raise ValueError(f"single-layer gate {field} differs")
    references = payload["references"]
    if not isinstance(references, Mapping) or set(references) != _REFERENCES:
        raise ValueError("single-layer gate reference fields differ")
    root = resolved.parent.resolve()
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for field in _REFERENCES:
        paths[field], hashes[field] = _reference(references[field], root=root, field=field)
    protocol = load_single_layer_gate_config(paths["config"])
    if protocol.config_sha256 != hashes["config"]:
        raise ValueError("single-layer gate config content hash differs")
    parent = load_probe_transition_artifact(paths["parent_probe_artifact"])
    model = _string(payload["model"], field="gate model")
    revision = _string(payload["model_revision"], field="gate model revision")
    code_revision = _string(payload["code_revision"], field="gate code revision")
    layers = _integer(payload["decoder_layers"], field="gate decoder layers", minimum=2)
    transition = _integer(payload["selected_transition_layer"], field="gate transition", minimum=1)
    if (
        model != protocol.model
        or revision != protocol.model_revision
        or code_revision != protocol.code_revision
        or layers != protocol.decoder_layers
        or (model, revision, layers) != (parent.model, parent.model_revision, parent.decoder_layers)
        or transition != parent.selected_transition_layer
        or hashes["parent_probe_artifact"] != parent.artifact_sha256
        or protocol.input_sha256["parent_probe_artifact"] != parent.artifact_sha256
    ):
        raise ValueError("single-layer gate identity differs from config or parent probe")
    expected_inputs = {
        "cohort_manifest": hashes["cohort_manifest"],
        "protected_registry": hashes["protected_split_registry"],
        "donor_plan": hashes["donor_plan"],
        "runtime_manifest": hashes["runtime_manifest"],
    }
    if any(protocol.input_sha256[key] != digest for key, digest in expected_inputs.items()):
        raise ValueError("single-layer gate input differs from preregistration")
    records = load_gate_cohort_manifest(paths["cohort_manifest"], protocol=protocol)
    protected = _load_protected_registry(paths["protected_split_registry"])
    if hashes["protected_split_registry"] != parent.protected_split_registry_sha256 or protected != parent.identity_inventory.protected:
        raise ValueError("single-layer gate protected registry differs from the parent probe")
    cohort_identities = frozenset().union(*(record.identities for record in records))
    if cohort_identities & parent.identity_inventory.all or cohort_identities & protected:
        raise ValueError("single-layer gate cohort overlaps parent or protected data transitively")
    donor_plan = _load_donor_plan(paths["donor_plan"], records=records)
    _load_runtime_manifest(paths["runtime_manifest"], protocol=protocol)
    observations = _load_raw_observations(
        paths["raw_kl"],
        protocol=protocol,
        records=records,
        donor_plan=donor_plan,
        transition_layer=transition,
        expected_bindings={
            "config_sha256": protocol.config_sha256,
            "parent_probe_artifact_sha256": parent.artifact_sha256,
            "cohort_manifest_sha256": hashes["cohort_manifest"],
            "protected_registry_sha256": hashes["protected_split_registry"],
            "donor_plan_sha256": hashes["donor_plan"],
            "runtime_manifest_sha256": hashes["runtime_manifest"],
        },
    )
    result = score_single_layer_gate(
        observations, protocol=protocol, transition_layer=transition
    )
    if payload["passed"] is not True or not result.passed:
        raise ValueError("single-layer causal gate did not pass recomputation")
    return SingleLayerGateArtifact(
        model=model,
        model_revision=revision,
        decoder_layers=layers,
        selected_transition_layer=transition,
        parent_probe_artifact_sha256=parent.artifact_sha256,
        artifact_sha256=hashlib.sha256(raw_artifact).hexdigest(),
        config_sha256=protocol.config_sha256,
        valid_records=result.valid_records,
        scores=result.scores,
    )


__all__ = [
    "SingleLayerGateArtifact",
    "SingleLayerGateRecord",
    "deterministic_cross_item_donor_plan",
    "load_gate_cohort_manifest",
    "load_single_layer_gate_artifact",
]
