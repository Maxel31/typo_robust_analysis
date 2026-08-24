"""Freeze externally pinned JSONL manifests into the probe split registry.

The consumer-facing ``registry.json`` intentionally retains the compact
``typo-protected-split-registry/v1`` schema.  The adjacent producer record is
the trust boundary: callers must pin its self-hash externally, and the loader
replays every input validation before returning the registry.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.records import record_id_for
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.integrity import sha256_file
from typo_robust_training.probe.cohort_builder import (
    probe_parent_source_sha256,
    probe_source_group_sha256,
)
from typo_robust_training.training.json_io import write_json_atomic
from typo_robust_training.training.pairs import TrainingSource


INVENTORY_SCHEMA = "typo-protected-split-inventory/v1"
REGISTRY_SCHEMA = "typo-protected-split-registry/v1"
PRODUCER_SCHEMA = "freeze-protected-split-registry-run/v1"
OVERLAP_AUDIT_SCHEMA = "typo-protected-split-overlap-audit/v1"
TIERS = ("training", "localization", "tune", "pre-pr", "sealed")
_RECORD_SCHEMA_ORDER = (
    "robustness-clean-record/v1",
    "robustness-evaluation-corpus-record/v1",
    "robustness-fixed-typo-pair/v1",
    "robustness-natural-pair/v1",
)
ALLOWED_RECORD_SCHEMAS = frozenset(_RECORD_SCHEMA_ORDER)

_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INVENTORY_TOP = {"schema_version", "tiers"}
_TIER_FIELDS = {"tier", "inputs"}
_INPUT_FIELDS = {"relative_path", "sha256", "accepted_schemas", "role"}
_RUN_FILENAME = "freeze_protected_split_registry_run.json"
_REGISTRY_FILENAME = "registry.json"
_INVENTORY_FILENAME = "inventory.json"
_IDENTITY_RULES = {
    "source_group": "typo-probe-source-group/v1",
    "parent_source": "typo-probe-parent-source/v1",
    "normalized_content": "casefold-collapse-whitespace-sha256/v1",
    "transitive_overlap": "record-component-union-find/v1",
}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _lexical_absolute(path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject symlinks before any caller resolves the supplied path."""

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {current}")


def _regular_file(path: Path, *, label: str) -> Path:
    absolute = _lexical_absolute(path)
    _reject_symlink_components(absolute, label=label)
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {absolute}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be one regular file: {absolute}")
    if metadata.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked: {absolute}")
    return absolute


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    regular = _regular_file(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(regular, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError(f"{label} changed away from one unlinked regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        current = regular.lstat()
        if (
            after.st_nlink != 1
            or current.st_nlink != 1
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ValueError(f"{label} changed while it was being read")
        return raw
    finally:
        os.close(descriptor)


def _assert_tree_without_symlinks(root: Path, *, label: str) -> None:
    absolute = _lexical_absolute(root)
    _reject_symlink_components(absolute, label=label)
    if not absolute.is_dir():
        raise ValueError(f"{label} must be a directory")

    def fail_on_walk_error(error: OSError) -> None:
        raise ValueError(f"{label} cannot be traversed completely") from error

    for current, directories, filenames in os.walk(
        absolute,
        followlinks=False,
        onerror=fail_on_walk_error,
    ):
        for name in (*directories, *filenames):
            candidate = Path(current) / name
            if candidate.is_symlink():
                raise ValueError(f"{label} contains a symlink: {candidate}")


def _relative_path(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if "\\" in text:
        raise ValueError(f"{field} must use canonical POSIX separators")
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or text != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{field} must be one canonical relative path without traversal")
    return text


def _strict_json(raw: bytes, *, context: str) -> Mapping[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context} must be UTF-8") from exc
    value = strict_loads(text, context=context)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must contain one JSON object")
    return value


@dataclass(frozen=True, slots=True)
class ProtectedInputSpec:
    tier: str
    relative_path: str
    sha256: str
    accepted_schemas: tuple[str, ...]
    role: str


@dataclass(frozen=True, slots=True)
class ProtectedSplitInventory:
    root: Path
    path: Path
    external_sha256: str
    inputs: tuple[ProtectedInputSpec, ...]
    raw: bytes


def load_protected_split_inventory(
    path: Path,
    *,
    expected_sha256: str,
) -> ProtectedSplitInventory:
    expected = _sha(expected_sha256, field="external inventory SHA-256")
    inventory_path = _regular_file(path, label="protected split inventory")
    raw = _read_regular_bytes(inventory_path, label="protected split inventory")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("protected split inventory differs from its external SHA-256")
    inputs = _decode_inventory(raw, context=str(inventory_path))
    root = inventory_path.parent
    _assert_tree_without_symlinks(root, label="protected split inventory tree")
    for item in inputs:
        source = _regular_file(root / item.relative_path, label="protected JSONL input")
        if source.parent != (root / PurePosixPath(item.relative_path)).parent:
            raise ValueError("protected input path substitution detected")
        if hashlib.sha256(
            _read_regular_bytes(source, label="protected JSONL input")
        ).hexdigest() != (item.sha256):
            raise ValueError("protected JSONL input differs from its expected SHA-256")
    return ProtectedSplitInventory(
        root=root,
        path=inventory_path,
        external_sha256=expected,
        inputs=inputs,
        raw=raw,
    )


def _decode_inventory(raw: bytes, *, context: str) -> tuple[ProtectedInputSpec, ...]:
    payload = _strict_json(raw, context=context)
    if raw != _canonical_bytes(payload):
        raise ValueError("protected split inventory must use canonical JSON serialization")
    if set(payload) != _INVENTORY_TOP or payload.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("protected split inventory schema or fields differ")
    rows = payload.get("tiers")
    if not isinstance(rows, list) or len(rows) != len(TIERS):
        raise ValueError("protected split inventory must enumerate exactly five tiers")
    inputs: list[ProtectedInputSpec] = []
    observed_tiers: list[str] = []
    observed_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _TIER_FIELDS:
            raise ValueError("protected split inventory tier fields differ")
        tier = _text(row.get("tier"), field="protected inventory tier")
        if tier not in TIERS or tier in observed_tiers:
            raise ValueError("protected split inventory tier identity differs")
        observed_tiers.append(tier)
        tier_inputs = row.get("inputs")
        if not isinstance(tier_inputs, list) or not tier_inputs:
            raise ValueError("every protected split tier must enumerate at least one input")
        for item in tier_inputs:
            if not isinstance(item, Mapping) or set(item) != _INPUT_FIELDS:
                raise ValueError("protected split inventory input fields differ")
            relative = _relative_path(
                item.get("relative_path"),
                field="protected input relative_path",
            )
            if relative in observed_paths:
                raise ValueError("protected split inventory input paths must be unique")
            observed_paths.add(relative)
            raw_schemas = item.get("accepted_schemas")
            if not isinstance(raw_schemas, list) or not raw_schemas:
                raise ValueError("protected input accepted_schemas must be a non-empty list")
            schemas = tuple(_text(schema, field="accepted schema") for schema in raw_schemas)
            if (
                len(set(schemas)) != len(schemas)
                or schemas != tuple(sorted(schemas))
                or any(schema not in ALLOWED_RECORD_SCHEMAS for schema in schemas)
            ):
                raise ValueError(
                    "protected input accepted_schemas must be unique, supported, and canonical"
                )
            inputs.append(
                ProtectedInputSpec(
                    tier=tier,
                    relative_path=relative,
                    sha256=_sha(item.get("sha256"), field="protected input SHA-256"),
                    accepted_schemas=schemas,
                    role=_text(item.get("role"), field="protected input role"),
                )
            )
    if tuple(observed_tiers) != TIERS:
        raise ValueError("protected split inventory tiers must use the canonical order")
    return tuple(inputs)


@dataclass(frozen=True, slots=True)
class _RecordIdentity:
    canonical_sha256: str
    address: str
    line_number: int
    source_group_sha256: str
    parent_source_sha256: str
    normalized_content_sha256: tuple[str, ...]


def _raw_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_optional_hash(
    row: Mapping[str, object],
    field: str,
    expected: str,
) -> None:
    if field in row and row[field] != expected:
        raise ValueError(f"protected input {field} differs from its full text")


def _evaluation_pair_record_id(
    value: Mapping[str, object],
    *,
    parent_record_id: str,
    role: str,
    context: str,
) -> str:
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context}.metadata must be an object")
    condition = _text(
        metadata.get("evaluation_condition"),
        field=f"{context}.metadata.evaluation_condition",
    )
    seed = _nonnegative_integer(value.get("generator_seed"), field=f"{context}.generator_seed")
    variant = _nonnegative_integer(
        value.get("generator_variant"),
        field=f"{context}.generator_variant",
    )
    edit_count = _positive_integer(value.get("edit_count"), field=f"{context}.edit_count")
    return hashlib.sha256(
        (
            "frozen-evaluation-pair/v4\0"
            f"{role}\0{condition}\0{seed}\0{variant}\0{edit_count}\0{parent_record_id}"
        ).encode("utf-8")
    ).hexdigest()


def _record_identity(
    value: object,
    *,
    spec: ProtectedInputSpec,
    context: str,
    line_number: int,
) -> _RecordIdentity:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must contain one JSON object")
    schema = value.get("schema_version")
    if schema not in spec.accepted_schemas:
        raise ValueError(f"{context} schema differs from the inventory")
    if value.get("split") != spec.role:
        raise ValueError(f"{context} role differs from the inventory")
    source = _text(value.get("source"), field=f"{context}.source")
    revision = _text(value.get("source_revision"), field=f"{context}.source_revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError(f"{context}.source_revision must be pinned")
    source_id = _text(value.get("source_id"), field=f"{context}.source_id")
    group_id = _text(value.get("group_id"), field=f"{context}.group_id")
    record_id = _sha(value.get("record_id"), field=f"{context}.record_id")
    parent_record_id = record_id_for(
        source=source,
        source_revision=revision,
        source_id=source_id,
    )
    if schema == "robustness-fixed-typo-pair/v1":
        valid_record_ids = {parent_record_id}
        if record_id != parent_record_id:
            valid_record_ids.add(
                _evaluation_pair_record_id(
                    value,
                    parent_record_id=parent_record_id,
                    role=spec.role,
                    context=context,
                )
            )
        if record_id not in valid_record_ids:
            raise ValueError(f"{context}.record_id differs from its fixed-pair identity")
    elif record_id != parent_record_id:
        raise ValueError(f"{context}.record_id differs from the source identity")

    if schema in {
        "robustness-clean-record/v1",
        "robustness-evaluation-corpus-record/v1",
    }:
        clean = _text(value.get("text"), field=f"{context}.text")
        typo: str | None = None
        if value.get("content_sha256") != _raw_sha256(clean):
            raise ValueError(f"{context}.content_sha256 differs from the full text")
        if schema == "robustness-clean-record/v1" and value.get(
            "normalized_content_sha256"
        ) != normalized_content_sha256(clean):
            raise ValueError(f"{context}.normalized_content_sha256 differs")
    else:
        clean = _text(value.get("clean_text"), field=f"{context}.clean_text")
        typo = _text(value.get("typo_text"), field=f"{context}.typo_text")
        if clean == typo:
            raise ValueError(f"{context} clean and typo texts must differ")
        if schema == "robustness-natural-pair/v1" and (
            value.get("clean_sha256") != _raw_sha256(clean)
            or value.get("typo_sha256") != _raw_sha256(typo)
        ):
            raise ValueError(f"{context} natural-pair text hash differs")

    clean_normalized = normalized_content_sha256(clean)
    _validate_optional_hash(value, "clean_sha256", _raw_sha256(clean))
    _validate_optional_hash(value, "content_sha256", _raw_sha256(clean))
    _validate_optional_hash(value, "normalized_clean_sha256", clean_normalized)
    _validate_optional_hash(value, "normalized_content_sha256", clean_normalized)
    normalized = [clean_normalized]
    if typo is not None:
        typo_normalized = normalized_content_sha256(typo)
        _validate_optional_hash(value, "typo_sha256", _raw_sha256(typo))
        _validate_optional_hash(value, "normalized_typo_sha256", typo_normalized)
        _validate_optional_hash(value, "normalized_noisy_sha256", typo_normalized)
        normalized.append(typo_normalized)

    source_view = TrainingSource(
        kind="clean" if typo is None else "natural",
        record_id=record_id,
        source=source,
        source_revision=revision,
        source_split=_text(value.get("source_split"), field=f"{context}.source_split"),
        source_id=source_id,
        group_id=group_id,
        clean_text=clean,
        typo_text=typo,
        task=None,
        answer=None,
        operation=None,
        metadata={},
        token_count=1,
    )
    return _RecordIdentity(
        canonical_sha256=_canonical_sha256(dict(value)),
        address=record_id,
        line_number=line_number,
        source_group_sha256=probe_source_group_sha256(source_view),
        parent_source_sha256=probe_parent_source_sha256(source_view),
        normalized_content_sha256=tuple(normalized),
    )


def _jsonl_records_from_raw(
    raw: bytes,
    *,
    path: Path,
    spec: ProtectedInputSpec,
) -> tuple[_RecordIdentity, ...]:
    if b"\r" in raw:
        raise ValueError("protected JSONL input must use LF, never CR or CRLF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("protected JSONL input must be UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError("protected JSONL input must end with one final LF")
    lines = text[:-1].split("\n")
    if not lines or any(not line for line in lines):
        raise ValueError("protected JSONL input must be non-empty and contain no blank lines")
    records: list[_RecordIdentity] = []
    for number, line in enumerate(lines, 1):
        context = f"{path}:{number}"
        value = strict_loads(line, context=context)
        records.append(
            _record_identity(
                value,
                spec=spec,
                context=context,
                line_number=number,
            )
        )
    return tuple(records)


def _jsonl_records(path: Path, *, spec: ProtectedInputSpec) -> tuple[_RecordIdentity, ...]:
    return _jsonl_records_from_raw(
        _read_regular_bytes(path, label="protected JSONL input"),
        path=path,
        spec=spec,
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


@dataclass(frozen=True, slots=True)
class _InputAudit:
    spec: ProtectedInputSpec
    source_path: Path
    copied_relative_path: str
    records: int
    unique_records: int


@dataclass(frozen=True, slots=True)
class ProtectedSplitIdentitySets:
    """Immutable union of every identity in the five verified tiers."""

    source_group_sha256: frozenset[str]
    parent_source_sha256: frozenset[str]
    normalized_content_sha256: frozenset[str]


@dataclass(frozen=True, slots=True)
class ProtectedOverlapIdentity:
    """One hash-only identity participating in a cross-tier collision."""

    kind: str
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ProtectedOverlapOccurrence:
    """Safe source location for one colliding record; never includes record text."""

    tier: str
    source_relative_path: str
    line_number: int
    record_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "source_relative_path": self.source_relative_path,
            "line_number": self.line_number,
            "record_id": self.record_id,
        }


@dataclass(frozen=True, slots=True)
class ProtectedOverlapComponent:
    """Deterministic hash-only audit of one transitive collision component."""

    tiers: tuple[str, ...]
    identities: tuple[ProtectedOverlapIdentity, ...]
    occurrences: tuple[ProtectedOverlapOccurrence, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "tiers": list(self.tiers),
            "identities": [identity.as_dict() for identity in self.identities],
            "occurrences": [occurrence.as_dict() for occurrence in self.occurrences],
        }


class ProtectedSplitOverlapError(ValueError):
    """Strict split-certification failure with a body-free collision audit."""

    def __init__(self, components: tuple[ProtectedOverlapComponent, ...]) -> None:
        if not components:
            raise ValueError("protected overlap error requires at least one component")
        self.components = components
        super().__init__(
            f"protected split tiers overlap transitively; collision_components={len(components)}"
        )

    @property
    def audit_report(self) -> dict[str, object]:
        """Return a deterministic report containing hashes and source locations only."""

        return {
            "schema_version": OVERLAP_AUDIT_SCHEMA,
            "collision_components": [component.as_dict() for component in self.components],
        }

    @property
    def audit_json(self) -> str:
        """Return the canonical single-line representation used by the CLI."""

        return json.dumps(
            self.audit_report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class _RegistryBuild:
    payload: Mapping[str, object]
    audits: tuple[_InputAudit, ...]
    tier_record_counts: Mapping[str, int]
    tier_unique_record_counts: Mapping[str, int]
    identity_sets: ProtectedSplitIdentitySets


def _build_registry(
    specs_and_paths: Sequence[tuple[ProtectedInputSpec, Path, str]],
    *,
    captured_inputs: Sequence[bytes] | None = None,
) -> _RegistryBuild:
    if captured_inputs is not None and len(captured_inputs) != len(specs_and_paths):
        raise ValueError("captured protected input inventory differs")
    identities: dict[str, dict[str, set[str]]] = {
        tier: {
            "source_group_sha256": set(),
            "parent_source_sha256": set(),
            "normalized_content_sha256": set(),
        }
        for tier in TIERS
    }
    union = _UnionFind()
    node_tiers: dict[str, set[str]] = {}
    node_occurrences: dict[str, set[ProtectedOverlapOccurrence]] = {}
    addresses: dict[tuple[str, str], str] = {}
    exact_by_tier: dict[str, set[str]] = {tier: set() for tier in TIERS}
    tier_records = {tier: 0 for tier in TIERS}
    audits: list[_InputAudit] = []
    for index, (spec, path, copied_relative) in enumerate(specs_and_paths):
        records = (
            _jsonl_records(path, spec=spec)
            if captured_inputs is None
            else _jsonl_records_from_raw(
                captured_inputs[index],
                path=path,
                spec=spec,
            )
        )
        unique_in_file: set[str] = set()
        for record in records:
            tier_records[spec.tier] += 1
            address_key = (spec.tier, record.address)
            previous = addresses.get(address_key)
            if previous is not None and previous != record.canonical_sha256:
                raise ValueError("same-tier protected record identity has conflicting records")
            addresses[address_key] = record.canonical_sha256
            nodes = (
                f"source_group_sha256:{record.source_group_sha256}",
                f"parent_source_sha256:{record.parent_source_sha256}",
                *(
                    f"normalized_content_sha256:{digest}"
                    for digest in record.normalized_content_sha256
                ),
            )
            occurrence = ProtectedOverlapOccurrence(
                tier=spec.tier,
                source_relative_path=spec.relative_path,
                line_number=record.line_number,
                record_id=record.address,
            )
            for node in nodes:
                node_occurrences.setdefault(node, set()).add(occurrence)
            if record.canonical_sha256 in exact_by_tier[spec.tier]:
                unique_in_file.add(record.canonical_sha256)
                continue
            exact_by_tier[spec.tier].add(record.canonical_sha256)
            unique_in_file.add(record.canonical_sha256)
            for node in nodes:
                union.add(node)
                node_tiers.setdefault(node, set()).add(spec.tier)
            for node in nodes[1:]:
                union.union(nodes[0], node)
            identities[spec.tier]["source_group_sha256"].add(record.source_group_sha256)
            identities[spec.tier]["parent_source_sha256"].add(record.parent_source_sha256)
            identities[spec.tier]["normalized_content_sha256"].update(
                record.normalized_content_sha256
            )
        audits.append(
            _InputAudit(
                spec=spec,
                source_path=path,
                copied_relative_path=copied_relative,
                records=len(records),
                unique_records=len(unique_in_file),
            )
        )
    component_tiers: dict[str, set[str]] = {}
    component_nodes: dict[str, set[str]] = {}
    for node, tiers in node_tiers.items():
        root = union.find(node)
        component_tiers.setdefault(root, set()).update(tiers)
        component_nodes.setdefault(root, set()).add(node)
    overlap_components: list[ProtectedOverlapComponent] = []
    tier_order = {tier: index for index, tier in enumerate(TIERS)}
    for root, tiers in component_tiers.items():
        if len(tiers) <= 1:
            continue
        nodes = component_nodes[root]
        identities = tuple(
            ProtectedOverlapIdentity(kind=kind, sha256=digest)
            for kind, digest in sorted(node.split(":", 1) for node in nodes)
        )
        occurrences = tuple(
            sorted(
                {occurrence for node in nodes for occurrence in node_occurrences[node]},
                key=lambda occurrence: (
                    tier_order[occurrence.tier],
                    occurrence.source_relative_path,
                    occurrence.line_number,
                    occurrence.record_id,
                ),
            )
        )
        overlap_components.append(
            ProtectedOverlapComponent(
                tiers=tuple(sorted(tiers, key=tier_order.__getitem__)),
                identities=identities,
                occurrences=occurrences,
            )
        )
    if overlap_components:
        overlap_components.sort(
            key=lambda component: (
                component.tiers,
                tuple((identity.kind, identity.sha256) for identity in component.identities),
                tuple(
                    (
                        occurrence.tier,
                        occurrence.source_relative_path,
                        occurrence.line_number,
                        occurrence.record_id,
                    )
                    for occurrence in component.occurrences
                ),
            )
        )
        raise ProtectedSplitOverlapError(tuple(overlap_components))
    registries = [
        {
            "tier": tier,
            "source_group_sha256": sorted(identities[tier]["source_group_sha256"]),
            "parent_source_sha256": sorted(identities[tier]["parent_source_sha256"]),
            "normalized_content_sha256": sorted(identities[tier]["normalized_content_sha256"]),
        }
        for tier in TIERS
    ]
    if any(not any(row[field] for field in row if field != "tier") for row in registries):
        raise ValueError("every protected split tier must contain an identity")
    return _RegistryBuild(
        payload={"schema_version": REGISTRY_SCHEMA, "registries": registries},
        audits=tuple(audits),
        tier_record_counts=tier_records,
        tier_unique_record_counts={tier: len(exact_by_tier[tier]) for tier in TIERS},
        identity_sets=ProtectedSplitIdentitySets(
            source_group_sha256=frozenset().union(
                *(identities[tier]["source_group_sha256"] for tier in TIERS)
            ),
            parent_source_sha256=frozenset().union(
                *(identities[tier]["parent_source_sha256"] for tier in TIERS)
            ),
            normalized_content_sha256=frozenset().union(
                *(identities[tier]["normalized_content_sha256"] for tier in TIERS)
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _CheckoutAttestation:
    revision: str
    project_tree: str

    def as_dict(self) -> dict[str, str]:
        return {"revision": self.revision, "project_tree": self.project_tree}


def _git(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("protected registry producer checkout cannot be attested")
    return result.stdout.strip()


def _attest_checkout() -> _CheckoutAttestation:
    module = Path(__file__)
    root = Path(_git("rev-parse", "--show-toplevel", cwd=module.parent))
    relative_project = Path("projects/typo-robust-training")
    try:
        module.absolute().relative_to(root.absolute() / relative_project)
    except ValueError as exc:
        raise ValueError("protected registry producer is outside the expected checkout") from exc
    if _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative_project.as_posix(),
        cwd=root,
    ):
        raise ValueError("protected registry producer project tree is dirty")
    revision = _git("rev-parse", "HEAD", cwd=root)
    project_tree = _git("rev-parse", f"HEAD:{relative_project.as_posix()}", cwd=root)
    if _GIT_OBJECT.fullmatch(revision) is None or _GIT_OBJECT.fullmatch(project_tree) is None:
        raise ValueError("protected registry producer checkout identity is unavailable")
    return _CheckoutAttestation(revision=revision, project_tree=project_tree)


def _new_output_target(path: Path) -> Path:
    target = _lexical_absolute(path)
    _reject_symlink_components(target, label="protected registry output")
    if os.path.lexists(target):
        raise FileExistsError(f"protected registry output already exists: {target}")
    parent = target.parent
    _reject_symlink_components(parent, label="protected registry output parent")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(parent, label="protected registry output parent")
    if not parent.is_dir():
        raise ValueError("protected registry output parent must be a directory")
    return target


def _publish_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish one sibling directory without replacing a race winner."""

    if source.parent != target.parent:
        raise ValueError("protected registry staging directory must be a target sibling")
    parent = target.parent
    _reject_symlink_components(parent, label="protected registry output parent")
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, parent_flags)
    try:
        opened_parent = os.fstat(parent_fd)
        current_parent = parent.lstat()
        if not stat.S_ISDIR(opened_parent.st_mode) or (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ) != (current_parent.st_dev, current_parent.st_ino):
            raise ValueError("protected registry output parent changed before publication")
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace directory publication is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        rename_noreplace = 1
        if (
            renameat2(
                parent_fd,
                os.fsencode(source.name),
                parent_fd,
                os.fsencode(target.name),
                rename_noreplace,
            )
            != 0
        ):
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(
                    f"protected registry output appeared before publish: {target}"
                )
            raise OSError(error, os.strerror(error), target)
        current_parent = parent.lstat()
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            current_parent.st_dev,
            current_parent.st_ino,
        ):
            raise ValueError("protected registry output parent changed during publication")
    finally:
        os.close(parent_fd)


def _before_final_rehash() -> None:
    """Private test seam for simulating a source mutation before publication."""


@dataclass(frozen=True, slots=True)
class ProtectedSplitRegistryFreezeResult:
    root: Path
    registry_path: Path
    inventory_path: Path
    run_path: Path
    producer_record_sha256: str
    input_records: int


@dataclass(frozen=True, slots=True)
class ProtectedSplitRegistryBundle:
    root: Path
    registry_path: Path
    inventory_path: Path
    input_paths: tuple[Path, ...]
    run_path: Path
    producer_record_sha256: str
    input_records: int
    identity_sets: ProtectedSplitIdentitySets


def _input_copy_name(index: int, spec: ProtectedInputSpec) -> str:
    suffix = PurePosixPath(spec.relative_path).suffix
    return f"inputs/{index:03d}-{spec.tier}{suffix or '.jsonl'}"


def _run_payload(
    *,
    checkout: _CheckoutAttestation,
    inventory: ProtectedSplitInventory,
    inventory_copy: Path,
    registry_path: Path,
    build: _RegistryBuild,
) -> dict[str, object]:
    inputs = [
        {
            "tier": audit.spec.tier,
            "role": audit.spec.role,
            "accepted_schemas": list(audit.spec.accepted_schemas),
            "source_relative_path": audit.spec.relative_path,
            "copied_relative_path": audit.copied_relative_path,
            "expected_sha256": audit.spec.sha256,
            "sha256": sha256_file(audit.source_path),
            "bytes": audit.source_path.stat().st_size,
            "records": audit.records,
            "unique_records": audit.unique_records,
        }
        for audit in build.audits
    ]
    return {
        "schema_version": PRODUCER_SCHEMA,
        "status": "completed",
        "checkout_attestation": checkout.as_dict(),
        "inventory": {
            "relative_path": _INVENTORY_FILENAME,
            "external_sha256": inventory.external_sha256,
            "sha256": sha256_file(inventory_copy),
            "bytes": inventory_copy.stat().st_size,
        },
        "inputs": inputs,
        "identity_rules": dict(_IDENTITY_RULES),
        "output": {
            "relative_path": _REGISTRY_FILENAME,
            "schema_version": REGISTRY_SCHEMA,
            "sha256": sha256_file(registry_path),
            "bytes": registry_path.stat().st_size,
            "input_records": sum(build.tier_record_counts.values()),
            "tier_record_counts": dict(build.tier_record_counts),
            "tier_unique_record_counts": dict(build.tier_unique_record_counts),
            "registry_identity_counts": {
                row["tier"]: {
                    field: len(row[field])
                    for field in (
                        "source_group_sha256",
                        "parent_source_sha256",
                        "normalized_content_sha256",
                    )
                }
                for row in build.payload["registries"]  # type: ignore[index]
            },
        },
    }


def freeze_protected_split_registry(
    *,
    inventory_path: Path,
    inventory_sha256: str,
    output_dir: Path,
) -> ProtectedSplitRegistryFreezeResult:
    """Freeze one closed, externally attestable protected-split bundle."""

    target = _new_output_target(output_dir)
    inventory = load_protected_split_inventory(
        inventory_path,
        expected_sha256=inventory_sha256,
    )
    checkout = _attest_checkout()
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        inputs_dir = temporary / "inputs"
        inputs_dir.mkdir()
        inventory_copy = temporary / _INVENTORY_FILENAME
        inventory_copy.write_bytes(inventory.raw)
        specs_and_paths: list[tuple[ProtectedInputSpec, Path, str]] = []
        for index, spec in enumerate(inventory.inputs):
            source = inventory.root / spec.relative_path
            copied_relative = _input_copy_name(index, spec)
            copied = temporary / copied_relative
            copied.write_bytes(_read_regular_bytes(source, label="protected JSONL input"))
            specs_and_paths.append((spec, copied, copied_relative))
        build = _build_registry(specs_and_paths)
        registry_path = temporary / _REGISTRY_FILENAME
        write_json_atomic(registry_path, build.payload)
        unsigned_run = _run_payload(
            checkout=checkout,
            inventory=inventory,
            inventory_copy=inventory_copy,
            registry_path=registry_path,
            build=build,
        )
        producer_record_sha256 = _canonical_sha256(unsigned_run)
        run_payload = {**unsigned_run, "record_sha256": producer_record_sha256}
        run_path = temporary / _RUN_FILENAME
        write_json_atomic(run_path, run_payload)
        load_protected_split_registry_bundle(
            run_path,
            expected_producer_record_sha256=producer_record_sha256,
        )

        _before_final_rehash()
        if _attest_checkout() != checkout:
            raise ValueError("protected registry producer checkout changed before publication")
        if (
            hashlib.sha256(
                _read_regular_bytes(inventory.path, label="protected split inventory")
            ).hexdigest()
            != inventory.external_sha256
        ):
            raise ValueError("protected split inventory changed before publication")
        _assert_tree_without_symlinks(
            inventory.root,
            label="protected split inventory tree",
        )
        for audit in build.audits:
            source = inventory.root / audit.spec.relative_path
            if (
                hashlib.sha256(
                    _read_regular_bytes(source, label="protected JSONL input")
                ).hexdigest()
                != audit.spec.sha256
            ):
                raise ValueError("protected JSONL input changed before publication")
        _publish_directory_noreplace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ProtectedSplitRegistryFreezeResult(
        root=target,
        registry_path=target / _REGISTRY_FILENAME,
        inventory_path=target / _INVENTORY_FILENAME,
        run_path=target / _RUN_FILENAME,
        producer_record_sha256=producer_record_sha256,
        input_records=sum(build.tier_record_counts.values()),
    )


def _closed_bundle_files(root: Path, expected: set[str]) -> None:
    _assert_tree_without_symlinks(root, label="protected registry bundle")
    observed: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        relative_current = Path(current).relative_to(root)
        for directory in directories:
            relative = (relative_current / directory).as_posix()
            if relative != "inputs":
                raise ValueError("protected registry bundle contains an unexpected directory")
        for filename in filenames:
            path = Path(current) / filename
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(
                    "protected registry bundle contains a non-regular or hard-linked artifact"
                )
            observed.add((relative_current / filename).as_posix())
    if observed != expected:
        raise ValueError("protected registry bundle file inventory differs")


def _load_canonical_run(
    path: Path,
    *,
    expected_producer_record_sha256: str,
) -> tuple[Path, Mapping[str, object]]:
    expected = _sha(
        expected_producer_record_sha256,
        field="expected protected registry producer-record SHA-256",
    )
    run_path = _regular_file(path, label="protected registry producer record")
    raw = _read_regular_bytes(run_path, label="protected registry producer record")
    payload = _strict_json(raw, context=str(run_path))
    if raw != _canonical_bytes(payload):
        raise ValueError("protected registry producer record must use canonical JSON")
    if payload.get("record_sha256") != expected:
        raise ValueError("protected registry producer record differs from its external SHA-256")
    unsigned = dict(payload)
    del unsigned["record_sha256"]
    if _canonical_sha256(unsigned) != expected:
        raise ValueError("protected registry producer record self-hash differs")
    return run_path, payload


def load_protected_split_registry_bundle(
    producer_run_path: Path,
    *,
    expected_producer_record_sha256: str,
) -> ProtectedSplitRegistryBundle:
    """Verify a frozen registry and replay every copied JSONL input."""

    run_path, run = _load_canonical_run(
        producer_run_path,
        expected_producer_record_sha256=expected_producer_record_sha256,
    )
    expected_top = {
        "schema_version",
        "status",
        "checkout_attestation",
        "inventory",
        "inputs",
        "identity_rules",
        "output",
        "record_sha256",
    }
    if (
        set(run) != expected_top
        or run.get("schema_version") != PRODUCER_SCHEMA
        or run.get("status") != "completed"
    ):
        raise ValueError("protected registry producer record fields or status differ")
    if run.get("identity_rules") != _IDENTITY_RULES:
        raise ValueError("protected registry identity rules differ")
    checkout = run.get("checkout_attestation")
    if not isinstance(checkout, Mapping) or set(checkout) != {"revision", "project_tree"}:
        raise ValueError("protected registry checkout attestation fields differ")
    if (
        not isinstance(checkout.get("revision"), str)
        or _GIT_OBJECT.fullmatch(checkout["revision"]) is None
        or not isinstance(checkout.get("project_tree"), str)
        or _GIT_OBJECT.fullmatch(checkout["project_tree"]) is None
    ):
        raise ValueError("protected registry checkout attestation differs")
    inventory_record = run.get("inventory")
    if not isinstance(inventory_record, Mapping) or set(inventory_record) != {
        "relative_path",
        "external_sha256",
        "sha256",
        "bytes",
    }:
        raise ValueError("protected registry producer inventory record differs")
    if inventory_record.get("relative_path") != _INVENTORY_FILENAME:
        raise ValueError("protected registry producer inventory path differs")
    inventory_sha = _sha(
        inventory_record.get("external_sha256"),
        field="producer inventory external SHA-256",
    )
    if inventory_record.get("sha256") != inventory_sha:
        raise ValueError("producer inventory copy differs from its external SHA-256")
    inventory_bytes = _positive_integer(
        inventory_record.get("bytes"),
        field="producer inventory bytes",
    )
    inputs = run.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("protected registry producer input inventory differs")
    input_paths: list[Path] = []
    captured_inputs: list[bytes] = []
    expected_files = {_RUN_FILENAME, _REGISTRY_FILENAME, _INVENTORY_FILENAME}
    run_specs: list[ProtectedInputSpec] = []
    run_records: list[Mapping[str, object]] = []
    for index, record in enumerate(inputs):
        if not isinstance(record, Mapping) or set(record) != {
            "tier",
            "role",
            "accepted_schemas",
            "source_relative_path",
            "copied_relative_path",
            "expected_sha256",
            "sha256",
            "bytes",
            "records",
            "unique_records",
        }:
            raise ValueError("protected registry producer input record fields differ")
        raw_schemas = record.get("accepted_schemas")
        if not isinstance(raw_schemas, list) or not raw_schemas:
            raise ValueError("protected registry producer input schemas differ")
        schemas = tuple(_text(schema, field="producer input schema") for schema in raw_schemas)
        if (
            len(set(schemas)) != len(schemas)
            or schemas != tuple(sorted(schemas))
            or any(schema not in ALLOWED_RECORD_SCHEMAS for schema in schemas)
        ):
            raise ValueError("protected registry producer input schemas differ")
        spec = ProtectedInputSpec(
            tier=_text(record.get("tier"), field="producer input tier"),
            relative_path=_relative_path(
                record.get("source_relative_path"),
                field="producer input source path",
            ),
            sha256=_sha(record.get("expected_sha256"), field="producer input expected hash"),
            accepted_schemas=schemas,
            role=_text(record.get("role"), field="producer input role"),
        )
        if spec.tier not in TIERS:
            raise ValueError("protected registry producer input tier differs")
        copied_relative = _relative_path(
            record.get("copied_relative_path"),
            field="producer input copied path",
        )
        if copied_relative != _input_copy_name(index, spec):
            raise ValueError("protected registry producer input copy path differs")
        if record.get("sha256") != spec.sha256:
            raise ValueError("protected registry producer input hash differs")
        _positive_integer(record.get("bytes"), field="producer input bytes")
        _positive_integer(record.get("records"), field="producer input records")
        _positive_integer(record.get("unique_records"), field="producer unique records")
        path = _regular_file(run_path.parent / copied_relative, label="protected input copy")
        raw_input = _read_regular_bytes(path, label="protected input copy")
        if (
            len(raw_input) != record["bytes"]
            or hashlib.sha256(raw_input).hexdigest() != spec.sha256
        ):
            raise ValueError("protected registry copied input bytes differ")
        input_paths.append(path)
        captured_inputs.append(raw_input)
        expected_files.add(copied_relative)
        run_specs.append(spec)
        run_records.append(record)
    _closed_bundle_files(run_path.parent, expected_files)
    inventory_path = _regular_file(
        run_path.parent / _INVENTORY_FILENAME,
        label="protected registry inventory copy",
    )
    inventory_raw = _read_regular_bytes(
        inventory_path,
        label="protected registry inventory copy",
    )
    if (
        len(inventory_raw) != inventory_bytes
        or hashlib.sha256(inventory_raw).hexdigest() != inventory_sha
    ):
        raise ValueError("protected registry inventory copy bytes differ")
    inventory = ProtectedSplitInventory(
        root=run_path.parent,
        path=inventory_path,
        external_sha256=inventory_sha,
        inputs=_decode_inventory(inventory_raw, context=str(inventory_path)),
        raw=inventory_raw,
    )
    if inventory.inputs != tuple(run_specs):
        raise ValueError("protected registry producer inputs differ from its inventory copy")

    build = _build_registry(
        tuple(
            (spec, path, str(record["copied_relative_path"]))
            for spec, path, record in zip(run_specs, input_paths, run_records, strict=True)
        ),
        captured_inputs=tuple(captured_inputs),
    )
    for audit, record in zip(build.audits, run_records, strict=True):
        if audit.records != record["records"] or audit.unique_records != record["unique_records"]:
            raise ValueError("protected registry producer input counts differ")
    registry_path = _regular_file(
        run_path.parent / _REGISTRY_FILENAME,
        label="protected split registry",
    )
    registry_raw = _read_regular_bytes(registry_path, label="protected split registry")
    registry = _strict_json(registry_raw, context=str(registry_path))
    if registry_raw != _canonical_bytes(registry) or dict(registry) != dict(build.payload):
        raise ValueError("protected split registry differs from replayed inputs")
    output = run.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "relative_path",
        "schema_version",
        "sha256",
        "bytes",
        "input_records",
        "tier_record_counts",
        "tier_unique_record_counts",
        "registry_identity_counts",
    }:
        raise ValueError("protected registry producer output record differs")
    expected_output = _run_payload(
        checkout=_CheckoutAttestation(
            revision=str(checkout["revision"]),
            project_tree=str(checkout["project_tree"]),
        ),
        inventory=inventory,
        inventory_copy=inventory_path,
        registry_path=registry_path,
        build=build,
    )["output"]
    if dict(output) != expected_output:
        raise ValueError("protected registry producer output accounting differs")
    for path, captured in zip(input_paths, captured_inputs, strict=True):
        if _read_regular_bytes(path, label="protected input copy") != captured:
            raise ValueError("protected registry copied input changed during verification")
    if (
        _read_regular_bytes(inventory_path, label="protected registry inventory copy")
        != inventory_raw
        or _read_regular_bytes(registry_path, label="protected split registry") != registry_raw
    ):
        raise ValueError("protected registry bundle changed during verification")
    _, final_run = _load_canonical_run(
        run_path,
        expected_producer_record_sha256=expected_producer_record_sha256,
    )
    if dict(final_run) != dict(run):
        raise ValueError("protected registry producer record changed during verification")
    _closed_bundle_files(run_path.parent, expected_files)
    return ProtectedSplitRegistryBundle(
        root=run_path.parent,
        registry_path=registry_path,
        inventory_path=inventory_path,
        input_paths=tuple(input_paths),
        run_path=run_path,
        producer_record_sha256=expected_producer_record_sha256,
        input_records=sum(build.tier_record_counts.values()),
        identity_sets=build.identity_sets,
    )


__all__ = [
    "ALLOWED_RECORD_SCHEMAS",
    "INVENTORY_SCHEMA",
    "OVERLAP_AUDIT_SCHEMA",
    "PRODUCER_SCHEMA",
    "ProtectedOverlapComponent",
    "ProtectedOverlapIdentity",
    "ProtectedOverlapOccurrence",
    "ProtectedSplitRegistryBundle",
    "ProtectedSplitRegistryFreezeResult",
    "ProtectedSplitIdentitySets",
    "ProtectedSplitOverlapError",
    "REGISTRY_SCHEMA",
    "TIERS",
    "freeze_protected_split_registry",
    "load_protected_split_inventory",
    "load_protected_split_registry_bundle",
]
