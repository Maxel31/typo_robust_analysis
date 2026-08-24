"""Freeze the unused FineWeb-Edu shard used to construct word-probe cohorts.

The source pool is deliberately model-output free.  It preserves the original
FineWeb document identity so the historical protected-identity denylist can
exclude a document even when another pipeline observed it through a different
text window.  The consumer-facing JSONL uses the existing training-source schema;
the upstream ``token_count`` is retained only as source metadata and is not
used to select the probe boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.protected_denylist import (
    DENYLIST_PURPOSE,
    ProtectedExclusionDenylistBundle,
    load_protected_exclusion_denylist_bundle,
)
from typo_robust_training.data.protected_registry import ProtectedSplitIdentitySets
from typo_robust_training.data.records import CleanRecord, record_id_for
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.integrity import sha256_file, sha256_tree
from typo_robust_training.probe.attestation import attest_runtime_checkout
from typo_robust_training.probe.cohort_builder import (
    probe_parent_source_sha256,
    probe_source_group_sha256,
)
from typo_robust_training.training.filesystem import (
    publish_directory_noreplace,
    reject_path_symlink_components,
)
from typo_robust_training.training.pairs import TrainingSource


_DATASET = "HuggingFaceFW/fineweb-edu"
_SOURCE = "fineweb_edu"
_REVISION = "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5"
_SUBSET = "sample-10BT"
_SPLIT = "train"
_SHARD = "sample/10BT/013_00000.parquet"
_SHARD_SHA256 = "b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de"
_SHARD_BYTES = 540_632_672
_EXPECTED_PARQUET_COLUMNS = (
    "text",
    "id",
    "dump",
    "url",
    "file_path",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
)
_EXPECTED_PARQUET_TYPES = (
    "string",
    "string",
    "string",
    "string",
    "string",
    "string",
    "double",
    "int64",
    "double",
    "int64",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_FILENAME = "probe_source_pool.jsonl"
_DECISIONS_FILENAME = "probe_source_pool_decisions.jsonl"
_PROTECTED_EXCLUSION_DIR = "protected_exclusion"
_RUN_FILENAME = "freeze_probe_source_pool_run.json"
_RUN_SCHEMA = "freeze-probe-source-pool-run/v2"
_DECISION_SCHEMA = "freeze-probe-source-pool-decision/v1"
_METADATA_FIELDS = {
    "dataset",
    "dataset_revision",
    "dataset_subset",
    "dataset_shard",
    "shard_row_index",
    "upstream_token_count",
    "upstream_token_count_semantics",
    "dump",
    "url",
    "file_path",
    "language",
    "language_score",
    "score",
    "int_score",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def _record_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _descriptor_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_regular_descriptor(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, int, os.stat_result]:
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise ValueError(f"{label} expected SHA-256 must be one lowercase digest")
    supplied = Path(path)
    reject_path_symlink_components(supplied, artifact=label)
    try:
        resolved = supplied.resolve(strict=True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"{label} must be one regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        visible = resolved.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (visible.st_dev, visible.st_ino)
        ):
            raise ValueError(f"{label} must be one unlinked regular file")
        if expected_sha256 is not None and _descriptor_sha256(descriptor) != expected_sha256:
            raise ValueError(f"{label} differs from its externally pinned SHA-256")
        return resolved, descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def _assert_descriptor_unchanged(
    path: Path,
    descriptor: int,
    *,
    expected_sha256: str,
    expected_identity: os.stat_result,
    label: str,
) -> None:
    try:
        final = os.fstat(descriptor)
        visible = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} changed while the source pool was being frozen") from exc
    if (
        _identity(final) != _identity(expected_identity)
        or (final.st_dev, final.st_ino) != (visible.st_dev, visible.st_ino)
        or not stat.S_ISREG(visible.st_mode)
        or _descriptor_sha256(descriptor) != expected_sha256
    ):
        raise ValueError(f"{label} changed while the source pool was being frozen")


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    resolved, descriptor, metadata = _open_regular_descriptor(path, label=label)
    try:
        raw = _descriptor_bytes(descriptor)
        digest = hashlib.sha256(raw).hexdigest()
        _assert_descriptor_unchanged(
            resolved,
            descriptor,
            expected_sha256=digest,
            expected_identity=metadata,
            label=label,
        )
        return raw
    finally:
        os.close(descriptor)


def _output_path(path: Path) -> tuple[Path, int, tuple[int, int]]:
    supplied = Path(path)
    reject_path_symlink_components(supplied, artifact="probe source pool output")
    target = supplied.absolute()
    if os.path.lexists(target):
        raise FileExistsError(f"probe source pool output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    reject_path_symlink_components(supplied, artifact="probe source pool output")
    resolved = supplied.resolve()
    if os.path.lexists(resolved):
        raise FileExistsError(f"probe source pool output already exists: {resolved}")
    parent_metadata = resolved.parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("probe source pool output parent must be one directory")
    descriptor = os.open(
        resolved.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (
        parent_metadata.st_dev,
        parent_metadata.st_ino,
    ):
        os.close(descriptor)
        raise ValueError("probe source pool output parent changed")
    return resolved, descriptor, (opened.st_dev, opened.st_ino)


def _assert_output_parent(
    target: Path,
    expected_identity: tuple[int, int],
) -> None:
    try:
        current = target.parent.lstat()
    except OSError as exc:
        raise ValueError("probe source pool output parent changed") from exc
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != expected_identity:
        raise ValueError("probe source pool output parent changed")


def _create_staging_directory(
    target: Path,
    *,
    parent_descriptor: int,
    expected_parent_identity: tuple[int, int],
) -> Path:
    for _ in range(100):
        name = f".{target.name}.{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        temporary = target.parent / name
        try:
            _assert_output_parent(target, expected_parent_identity)
            pinned = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            visible = temporary.lstat()
            if not stat.S_ISDIR(pinned.st_mode) or (pinned.st_dev, pinned.st_ino) != (
                visible.st_dev,
                visible.st_ino,
            ):
                raise ValueError("probe source pool staging path was substituted")
            return temporary
        except Exception:
            shutil.rmtree(name, dir_fd=parent_descriptor, ignore_errors=True)
            raise
    raise FileExistsError("could not allocate a unique probe source pool staging directory")


def _jsonl_row(value: Mapping[str, object]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _optional_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("FineWeb-Edu metadata contains a non-finite number")
        return value
    raise ValueError("FineWeb-Edu metadata contains a non-scalar value")


def _iter_parquet_rows(handle: BinaryIO) -> Iterator[Mapping[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency via datasets
        raise RuntimeError("pyarrow is required to read the pinned FineWeb-Edu shard") from exc
    parquet = pq.ParquetFile(handle)
    schema = parquet.schema_arrow
    if (
        tuple(schema.names) != _EXPECTED_PARQUET_COLUMNS
        or tuple(str(field.type) for field in schema) != _EXPECTED_PARQUET_TYPES
        or any(field.nullable is not True for field in schema)
    ):
        raise ValueError("FineWeb-Edu parquet schema differs")
    for batch in parquet.iter_batches(batch_size=1024, columns=list(_EXPECTED_PARQUET_COLUMNS)):
        for row in batch.to_pylist():
            if not isinstance(row, Mapping):
                raise ValueError("FineWeb-Edu parquet emitted a non-object row")
            yield row


def _clean_payload(row: Mapping[str, object], *, row_index: int) -> dict[str, object]:
    if set(row) != set(_EXPECTED_PARQUET_COLUMNS):
        raise ValueError("FineWeb-Edu row column inventory differs")
    text = row.get("text")
    source_key = row.get("id")
    upstream_tokens = row.get("token_count")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("FineWeb-Edu row text must be non-empty")
    if not isinstance(source_key, str) or not source_key:
        raise ValueError("FineWeb-Edu row id must be non-empty")
    if (
        isinstance(upstream_tokens, bool)
        or not isinstance(upstream_tokens, int)
        or upstream_tokens <= 0
    ):
        raise ValueError("FineWeb-Edu upstream token_count must be positive")
    for field in ("dump", "url", "file_path", "language"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"FineWeb-Edu row {field} must be non-empty")
    for field in ("language_score", "score"):
        value = row.get(field)
        if not isinstance(value, float) or not float("-inf") < value < float("inf"):
            raise ValueError(f"FineWeb-Edu row {field} must be one finite float64")
    int_score = row.get("int_score")
    if isinstance(int_score, bool) or not isinstance(int_score, int):
        raise ValueError("FineWeb-Edu row int_score must be one int64")
    source_id = f"{_SOURCE}:{source_key}"
    record = CleanRecord(
        source=_SOURCE,
        source_revision=_REVISION,
        source_split=_SPLIT,
        source_id=source_id,
        group_id=source_id,
        text=text,
        task=None,
        answer=None,
        metadata={},
    )
    metadata = {
        "dataset": _DATASET,
        "dataset_revision": _REVISION,
        "dataset_subset": _SUBSET,
        "dataset_shard": _SHARD,
        "shard_row_index": row_index,
        "upstream_token_count": upstream_tokens,
        "upstream_token_count_semantics": "fineweb-edu-published-metadata-only/v1",
    }
    for key in (
        "dump",
        "url",
        "file_path",
        "language",
        "language_score",
        "score",
        "int_score",
    ):
        metadata[key] = _optional_scalar(row.get(key))
    return {
        "schema_version": "robustness-clean-record/v1",
        "kind": "clean",
        "record_id": record.record_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_split": record.source_split,
        "source_id": record.source_id,
        "group_id": record.group_id,
        "split": "train",
        "text": text,
        "task": None,
        "answer": None,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_content_sha256": normalized_content_sha256(text),
        "metadata": metadata,
        # Required by the shared schema, but never used by the probe producer.
        "token_count": upstream_tokens,
    }


@dataclass(frozen=True, slots=True)
class _SourceIdentities:
    source_group_sha256: str
    parent_source_sha256: str
    normalized_content_sha256: str


def _source_identities(payload: Mapping[str, object]) -> _SourceIdentities:
    source = TrainingSource.from_dict(payload)
    return _SourceIdentities(
        source_group_sha256=probe_source_group_sha256(source),
        parent_source_sha256=probe_parent_source_sha256(source),
        normalized_content_sha256=str(payload["normalized_content_sha256"]),
    )


def _is_protected(
    identities: _SourceIdentities,
    protected: ProtectedSplitIdentitySets,
) -> bool:
    return (
        identities.source_group_sha256 in protected.source_group_sha256
        or identities.parent_source_sha256 in protected.parent_source_sha256
        or identities.normalized_content_sha256 in protected.normalized_content_sha256
    )


def _validate_source_payload(
    value: object,
    *,
    expected_row_index: int | None = None,
) -> tuple[TrainingSource, _SourceIdentities, int]:
    source = TrainingSource.from_dict(value)
    if not isinstance(value, Mapping):  # narrowed by TrainingSource.from_dict
        raise ValueError("probe source pool row must be one object")
    metadata = source.metadata
    row_index = metadata.get("shard_row_index")
    upstream_tokens = metadata.get("upstream_token_count")
    if (
        source.kind != "clean"
        or source.source != _SOURCE
        or source.source_revision != _REVISION
        or source.source_split != _SPLIT
        or source.task is not None
        or source.answer is not None
        or not source.source_id.startswith(f"{_SOURCE}:")
        or source.source_id == f"{_SOURCE}:"
        or source.group_id != source.source_id
        or source.record_id
        != record_id_for(
            source=source.source,
            source_revision=source.source_revision,
            source_id=source.source_id,
        )
        or set(metadata) != _METADATA_FIELDS
        or metadata.get("dataset") != _DATASET
        or metadata.get("dataset_revision") != _REVISION
        or metadata.get("dataset_subset") != _SUBSET
        or metadata.get("dataset_shard") != _SHARD
        or isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or row_index < 0
        or (expected_row_index is not None and row_index != expected_row_index)
        or isinstance(upstream_tokens, bool)
        or not isinstance(upstream_tokens, int)
        or upstream_tokens <= 0
        or upstream_tokens != source.token_count
        or metadata.get("upstream_token_count_semantics")
        != "fineweb-edu-published-metadata-only/v1"
    ):
        raise ValueError("probe source pool row identity or metadata differs")
    for field in ("dump", "url", "file_path", "language"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError("probe source pool row string metadata differs")
    for field in ("language_score", "score"):
        metadata_value = metadata[field]
        if not isinstance(metadata_value, float) or not float("-inf") < metadata_value < float(
            "inf"
        ):
            raise ValueError("probe source pool row float metadata differs")
    int_score = metadata["int_score"]
    if isinstance(int_score, bool) or not isinstance(int_score, int):
        raise ValueError("probe source pool row integer metadata differs")
    identities = _source_identities(value)
    return source, identities, row_index


class _IdentityLedger:
    """Disk-backed uniqueness ledger so shard size does not determine RAM use."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE source_identity (record_id TEXT PRIMARY KEY, source_id TEXT UNIQUE)"
        )
        self.connection.execute(
            "CREATE TABLE retained_content (normalized_content_sha256 TEXT PRIMARY KEY)"
        )

    def __enter__(self) -> _IdentityLedger:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.connection.close()

    def observe_source(self, *, record_id: str, source_id: str) -> None:
        try:
            self.connection.execute(
                "INSERT INTO source_identity(record_id, source_id) VALUES (?, ?)",
                (record_id, source_id),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("FineWeb-Edu shard contains a duplicate source identity") from exc

    def retain_normalized(self, normalized: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO retained_content(normalized_content_sha256) VALUES (?)",
            (normalized,),
        )
        return cursor.rowcount == 1


class _CanonicalJsonlReader:
    """Stream canonical JSONL while hashing the exact descriptor bytes."""

    def __init__(self, handle: BinaryIO, *, context: str) -> None:
        self.handle = handle
        self.context = context
        self.digest = hashlib.sha256()
        self.bytes_read = 0
        self.rows = 0
        self._started = False

    def __iter__(self) -> Iterator[tuple[int, Mapping[str, object]]]:
        if self._started:
            raise RuntimeError("canonical JSONL reader cannot be replayed")
        self._started = True
        for line_number, raw in enumerate(self.handle, 1):
            self.digest.update(raw)
            self.bytes_read += len(raw)
            if not raw.endswith(b"\n") or raw == b"\n":
                raise ValueError(f"{self.context} has a partial or blank row at {line_number}")
            try:
                text = raw[:-1].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{self.context} is not UTF-8 at {line_number}") from exc
            value = strict_loads(text, context=f"{self.context}:{line_number}")
            if not isinstance(value, Mapping) or raw != _jsonl_row(value):
                raise ValueError(f"{self.context} row is not canonical at {line_number}")
            self.rows += 1
            yield line_number, value

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProbeSourcePoolFreezeConfig:
    parquet_path: Path
    parquet_sha256: str
    protected_exclusion_run_path: Path
    protected_exclusion_producer_sha256: str
    code_revision: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ProbeSourcePoolFreezeResult:
    source_manifest_path: Path
    decision_ledger_path: Path
    protected_exclusion_path: Path
    protected_exclusion_run_path: Path
    run_path: Path
    run_sha256: str
    records: int
    protected_records_removed: int
    duplicate_records_removed: int


def _publish_bundle(
    temporary: Path,
    target: Path,
    *,
    expected_parent_identity: tuple[int, int],
) -> None:
    """Publish the already validated closed-world directory in one operation."""

    publish_directory_noreplace(
        temporary,
        target,
        expected_parent_identity=expected_parent_identity,
    )


def _load_protected_exclusion_bundle(
    run_path: Path,
    *,
    expected_producer_record_sha256: str,
) -> ProtectedExclusionDenylistBundle:
    """Load only the typed exclusion-only artifact; no raw/strict fallback."""

    bundle = load_protected_exclusion_denylist_bundle(
        run_path,
        expected_producer_record_sha256=expected_producer_record_sha256,
    )
    if bundle.purpose != DENYLIST_PURPOSE or bundle.split_certified is not False:
        raise ValueError("protected exclusion denylist discriminator differs")
    return bundle


def _copy_protected_bundle(
    bundle: ProtectedExclusionDenylistBundle,
    *,
    destination: Path,
) -> ProtectedExclusionDenylistBundle:
    shutil.copytree(bundle.root, destination)
    copied_run = destination / bundle.run_path.relative_to(bundle.root)
    copied = _load_protected_exclusion_bundle(
        copied_run,
        expected_producer_record_sha256=bundle.producer_record_sha256,
    )
    if (
        copied.identity_sets != bundle.identity_sets
        or copied.purpose != DENYLIST_PURPOSE
        or copied.split_certified is not False
    ):
        raise ValueError("copied protected exclusion identities differ")
    return copied


def _decision_payload(
    payload: Mapping[str, object],
    identities: _SourceIdentities,
    *,
    decision: str,
) -> dict[str, object]:
    return {
        "schema_version": _DECISION_SCHEMA,
        "shard_row_index": payload["metadata"]["shard_row_index"],  # type: ignore[index]
        "record_id": payload["record_id"],
        "source_id": payload["source_id"],
        "group_id": payload["group_id"],
        "source_group_sha256": identities.source_group_sha256,
        "parent_source_sha256": identities.parent_source_sha256,
        "normalized_content_sha256": identities.normalized_content_sha256,
        "decision": decision,
        "source_record": dict(payload),
    }


def freeze_probe_source_pool(
    config: ProbeSourcePoolFreezeConfig,
) -> ProbeSourcePoolFreezeResult:
    """Freeze one exact unused shard after historical protected-identity exclusion."""

    if not isinstance(config, ProbeSourcePoolFreezeConfig):
        raise TypeError("probe source pool config has the wrong type")
    if config.parquet_sha256 != _SHARD_SHA256:
        raise ValueError("probe source pool must use the preregistered final FineWeb-Edu shard")
    if (
        not isinstance(config.code_revision, str)
        or _REVISION_RE.fullmatch(config.code_revision) is None
    ):
        raise ValueError("probe source pool code revision must be one exact commit")
    parquet, parquet_descriptor, parquet_identity = _open_regular_descriptor(
        config.parquet_path,
        expected_sha256=config.parquet_sha256,
        label="FineWeb-Edu parquet shard",
    )
    if parquet_identity.st_size != _SHARD_BYTES:
        os.close(parquet_descriptor)
        raise ValueError("FineWeb-Edu parquet shard byte count differs")
    output_parent_descriptor: int | None = None
    try:
        checkout = attest_runtime_checkout(config.code_revision)
        protected_source = _load_protected_exclusion_bundle(
            config.protected_exclusion_run_path,
            expected_producer_record_sha256=config.protected_exclusion_producer_sha256,
        )
        target, output_parent_descriptor, output_parent_identity = _output_path(config.output_dir)
        if (
            target == protected_source.root
            or target.is_relative_to(protected_source.root)
            or protected_source.root.is_relative_to(target)
        ):
            raise ValueError("probe source pool output overlaps the protected exclusion bundle")
        temporary = _create_staging_directory(
            target,
            parent_descriptor=output_parent_descriptor,
            expected_parent_identity=output_parent_identity,
        )
    except Exception:
        if output_parent_descriptor is not None:
            os.close(output_parent_descriptor)
        os.close(parquet_descriptor)
        raise
    records = 0
    protected_removed = 0
    duplicate_removed = 0
    observed_rows = 0
    published = False
    try:
        protected = _copy_protected_bundle(
            protected_source,
            destination=temporary / _PROTECTED_EXCLUSION_DIR,
        )
        source_path = temporary / _SOURCE_FILENAME
        decisions_path = temporary / _DECISIONS_FILENAME
        ledger_path = temporary / ".source-pool-identities.sqlite3"
        with (
            source_path.open("xb") as source_handle,
            decisions_path.open("xb") as decision_handle,
            _IdentityLedger(ledger_path) as ledger,
            os.fdopen(os.dup(parquet_descriptor), "rb") as parquet_handle,
        ):
            for row_index, row in enumerate(_iter_parquet_rows(parquet_handle)):
                observed_rows = row_index + 1
                payload = _clean_payload(row, row_index=row_index)
                source, identities, _ = _validate_source_payload(
                    payload,
                    expected_row_index=row_index,
                )
                ledger.observe_source(
                    record_id=source.record_id,
                    source_id=source.source_id,
                )
                if _is_protected(identities, protected.identity_sets):
                    decision = "protected"
                    protected_removed += 1
                elif not ledger.retain_normalized(identities.normalized_content_sha256):
                    decision = "duplicate"
                    duplicate_removed += 1
                else:
                    decision = "retained"
                    source_handle.write(_jsonl_row(payload))
                    records += 1
                decision_handle.write(
                    _jsonl_row(_decision_payload(payload, identities, decision=decision))
                )
        ledger_path.unlink()
        if records == 0:
            raise ValueError("probe source pool contains no eligible records")
        _assert_descriptor_unchanged(
            parquet,
            parquet_descriptor,
            expected_sha256=config.parquet_sha256,
            expected_identity=parquet_identity,
            label="FineWeb-Edu parquet shard",
        )
        artifacts = {
            "source_manifest": {
                "relative_path": _SOURCE_FILENAME,
                "sha256": sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            },
            "decision_ledger": {
                "relative_path": _DECISIONS_FILENAME,
                "sha256": sha256_file(decisions_path),
                "bytes": decisions_path.stat().st_size,
            },
            "protected_exclusion_bundle": {
                "relative_run_path": (protected.run_path.relative_to(temporary).as_posix()),
                "producer_record_sha256": protected.producer_record_sha256,
                "tree_sha256": sha256_tree(protected.root),
            },
        }
        payload: dict[str, object] = {
            "schema_version": _RUN_SCHEMA,
            "status": "completed",
            "model_outputs_observed": False,
            "code": checkout.as_dict(),
            "source": {
                "dataset": _DATASET,
                "revision": _REVISION,
                "subset": _SUBSET,
                "split": _SPLIT,
                "shard": _SHARD,
                "sha256": config.parquet_sha256,
                "bytes": parquet_identity.st_size,
                "columns": list(_EXPECTED_PARQUET_COLUMNS),
                "column_types": list(_EXPECTED_PARQUET_TYPES),
                "columns_nullable": True,
            },
            "identity_policy": {
                "protected_exclusion_producer_sha256": protected.producer_record_sha256,
                "protected_exclusion_purpose": protected.purpose,
                "protected_exclusion_split_certified": protected.split_certified,
                "protected_dimensions": [
                    "source_group_sha256",
                    "parent_source_sha256",
                    "normalized_content_sha256",
                ],
                "within_pool_duplicates": "first-parquet-row-by-normalized-content/v1",
                "decision_priority": ["protected", "duplicate", "retained"],
            },
            "token_count": {
                "provider": "fineweb-edu-published-metadata-only/v1",
                "used_for_probe_selection": False,
            },
            "counts": {
                "records": records,
                "protected_records_removed": protected_removed,
                "duplicate_records_removed": duplicate_removed,
                "observed_parquet_rows": observed_rows,
            },
            "artifacts": artifacts,
        }
        run_sha = _record_sha256(payload)
        payload["record_sha256"] = run_sha
        (temporary / _RUN_FILENAME).write_bytes(_pretty_bytes(payload))
        load_probe_source_pool_bundle(
            temporary / _RUN_FILENAME,
            expected_run_sha256=run_sha,
            expected_code_revision=config.code_revision,
        )
        _publish_bundle(
            temporary,
            target,
            expected_parent_identity=output_parent_identity,
        )
        published = True
        return ProbeSourcePoolFreezeResult(
            source_manifest_path=target / _SOURCE_FILENAME,
            decision_ledger_path=target / _DECISIONS_FILENAME,
            protected_exclusion_path=(
                target
                / _PROTECTED_EXCLUSION_DIR
                / protected.denylist_path.relative_to(protected.root)
            ),
            protected_exclusion_run_path=(
                target / _PROTECTED_EXCLUSION_DIR / protected.run_path.relative_to(protected.root)
            ),
            run_path=target / _RUN_FILENAME,
            run_sha256=run_sha,
            records=records,
            protected_records_removed=protected_removed,
            duplicate_records_removed=duplicate_removed,
        )
    finally:
        os.close(parquet_descriptor)
        if not published:
            shutil.rmtree(
                temporary.name,
                dir_fd=output_parent_descriptor,
                ignore_errors=True,
            )
        os.close(output_parent_descriptor)


def load_probe_source_pool_bundle(
    run_path: Path,
    *,
    expected_run_sha256: str,
    expected_code_revision: str,
) -> ProbeSourcePoolFreezeResult:
    """Revalidate and replay a frozen pool from an external producer digest."""

    if _SHA256.fullmatch(expected_run_sha256 or "") is None:
        raise ValueError("expected probe source pool run SHA-256 is invalid")
    if _REVISION_RE.fullmatch(expected_code_revision or "") is None:
        raise ValueError("expected probe source pool code revision is invalid")
    supplied = Path(run_path)
    if supplied.name != _RUN_FILENAME:
        raise ValueError("probe source pool run filename differs")
    resolved_run, run_descriptor, run_identity = _open_regular_descriptor(
        supplied,
        label="probe source pool producer record",
    )
    source_descriptor: int | None = None
    decision_descriptor: int | None = None
    try:
        raw = _descriptor_bytes(run_descriptor)
        try:
            payload = strict_loads(raw.decode("utf-8"), context=str(resolved_run))
        except UnicodeDecodeError as exc:
            raise ValueError("probe source pool run must be UTF-8") from exc
        if not isinstance(payload, dict) or raw != _pretty_bytes(payload):
            raise ValueError("probe source pool run must be canonical JSON")
        record_sha = payload.get("record_sha256")
        unsigned = dict(payload)
        unsigned.pop("record_sha256", None)
        if record_sha != expected_run_sha256 or _record_sha256(unsigned) != expected_run_sha256:
            raise ValueError("probe source pool run differs from its externally pinned digest")
        expected_fields = {
            "schema_version",
            "status",
            "model_outputs_observed",
            "code",
            "source",
            "identity_policy",
            "token_count",
            "counts",
            "artifacts",
            "record_sha256",
        }
        if (
            set(payload) != expected_fields
            or payload.get("schema_version") != _RUN_SCHEMA
            or payload.get("status") != "completed"
            or payload.get("model_outputs_observed") is not False
        ):
            raise ValueError("probe source pool run fields or status differ")
        code = payload.get("code")
        runtime_sources = (
            code.get("typo_cot_runtime_sources") if isinstance(code, Mapping) else None
        )
        if (
            not isinstance(code, Mapping)
            or set(code)
            != {
                "revision",
                "typo_robust_training_tree",
                "typo_cot_tree",
                "typo_cot_runtime_sources",
            }
            or code.get("revision") != expected_code_revision
            or _REVISION_RE.fullmatch(str(code.get("typo_robust_training_tree"))) is None
            or _REVISION_RE.fullmatch(str(code.get("typo_cot_tree"))) is None
            or not isinstance(runtime_sources, list)
            or not runtime_sources
            or any(
                not isinstance(item, str)
                or not item.startswith("projects/typo-cot/src/typo_cot/")
                or ".." in PurePosixPath(item).parts
                for item in runtime_sources
            )
            or runtime_sources != sorted(set(runtime_sources))
        ):
            raise ValueError("probe source pool code attestation differs")
        source = payload.get("source")
        expected_source_fields = {
            "dataset",
            "revision",
            "subset",
            "split",
            "shard",
            "sha256",
            "bytes",
            "columns",
            "column_types",
            "columns_nullable",
        }
        if (
            not isinstance(source, Mapping)
            or set(source) != expected_source_fields
            or source.get("dataset") != _DATASET
            or source.get("revision") != _REVISION
            or source.get("subset") != _SUBSET
            or source.get("split") != _SPLIT
            or source.get("shard") != _SHARD
            or source.get("sha256") != _SHARD_SHA256
            or source.get("bytes") != _SHARD_BYTES
            or source.get("columns") != list(_EXPECTED_PARQUET_COLUMNS)
            or source.get("column_types") != list(_EXPECTED_PARQUET_TYPES)
            or source.get("columns_nullable") is not True
        ):
            raise ValueError("probe source pool dataset identity differs")
        identity_policy = payload.get("identity_policy")
        token_count = payload.get("token_count")
        if (
            not isinstance(identity_policy, Mapping)
            or set(identity_policy)
            != {
                "protected_exclusion_producer_sha256",
                "protected_exclusion_purpose",
                "protected_exclusion_split_certified",
                "protected_dimensions",
                "within_pool_duplicates",
                "decision_priority",
            }
            or _SHA256.fullmatch(str(identity_policy.get("protected_exclusion_producer_sha256")))
            is None
            or identity_policy.get("protected_exclusion_purpose") != DENYLIST_PURPOSE
            or identity_policy.get("protected_exclusion_split_certified") is not False
            or identity_policy.get("protected_dimensions")
            != [
                "source_group_sha256",
                "parent_source_sha256",
                "normalized_content_sha256",
            ]
            or identity_policy.get("within_pool_duplicates")
            != "first-parquet-row-by-normalized-content/v1"
            or identity_policy.get("decision_priority") != ["protected", "duplicate", "retained"]
            or not isinstance(token_count, Mapping)
            or dict(token_count)
            != {
                "provider": "fineweb-edu-published-metadata-only/v1",
                "used_for_probe_selection": False,
            }
        ):
            raise ValueError("probe source pool identity or token-count policy differs")
        counts = payload.get("counts")
        count_fields = {
            "records",
            "protected_records_removed",
            "duplicate_records_removed",
            "observed_parquet_rows",
        }
        if (
            not isinstance(counts, Mapping)
            or set(counts) != count_fields
            or any(
                isinstance(counts.get(field), bool)
                or not isinstance(counts.get(field), int)
                or counts[field] < 0
                for field in count_fields
            )
            or counts["records"] <= 0
            or counts["observed_parquet_rows"]
            != counts["records"]
            + counts["protected_records_removed"]
            + counts["duplicate_records_removed"]
        ):
            raise ValueError("probe source pool counts differ")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "source_manifest",
            "decision_ledger",
            "protected_exclusion_bundle",
        }:
            raise ValueError("probe source pool artifact inventory differs")
        root = resolved_run.parent
        expected_root_names = {
            _RUN_FILENAME,
            _SOURCE_FILENAME,
            _DECISIONS_FILENAME,
            _PROTECTED_EXCLUSION_DIR,
        }
        observed_root_names = {path.name for path in root.iterdir()}
        if observed_root_names != expected_root_names:
            raise ValueError("probe source pool closed-world inventory differs")
        protected_root = root / _PROTECTED_EXCLUSION_DIR
        protected_metadata = protected_root.lstat()
        if not stat.S_ISDIR(protected_metadata.st_mode):
            raise ValueError("protected exclusion bundle must be one directory")

        artifact_paths: dict[str, Path] = {}
        artifact_descriptors: dict[str, tuple[int, os.stat_result, str]] = {}
        for artifact_name, filename in (
            ("source_manifest", _SOURCE_FILENAME),
            ("decision_ledger", _DECISIONS_FILENAME),
        ):
            record = artifacts[artifact_name]
            if (
                not isinstance(record, Mapping)
                or set(record) != {"relative_path", "sha256", "bytes"}
                or record.get("relative_path") != filename
                or _SHA256.fullmatch(str(record.get("sha256"))) is None
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
            ):
                raise ValueError("probe source pool artifact record differs")
            artifact_path, descriptor, descriptor_identity = _open_regular_descriptor(
                root / filename,
                label=f"probe source pool {artifact_name}",
                expected_sha256=str(record["sha256"]),
            )
            if descriptor_identity.st_size != record["bytes"]:
                os.close(descriptor)
                raise ValueError("probe source pool artifact byte count differs")
            artifact_paths[artifact_name] = artifact_path
            artifact_descriptors[artifact_name] = (
                descriptor,
                descriptor_identity,
                str(record["sha256"]),
            )
        source_descriptor = artifact_descriptors["source_manifest"][0]
        decision_descriptor = artifact_descriptors["decision_ledger"][0]

        protected_record = artifacts["protected_exclusion_bundle"]
        if not isinstance(protected_record, Mapping) or set(protected_record) != {
            "relative_run_path",
            "producer_record_sha256",
            "tree_sha256",
        }:
            raise ValueError("probe source pool protected bundle record differs")
        relative_run = protected_record.get("relative_run_path")
        if not isinstance(relative_run, str) or "\\" in relative_run:
            raise ValueError("probe source pool protected run path differs")
        relative_candidate = PurePosixPath(relative_run)
        if (
            relative_run != relative_candidate.as_posix()
            or relative_candidate.is_absolute()
            or not relative_candidate.parts
            or relative_candidate.parts[0] != _PROTECTED_EXCLUSION_DIR
            or any(part in {"", ".", ".."} for part in relative_candidate.parts)
            or _SHA256.fullmatch(str(protected_record.get("producer_record_sha256"))) is None
            or _SHA256.fullmatch(str(protected_record.get("tree_sha256"))) is None
        ):
            raise ValueError("probe source pool protected bundle binding differs")
        producer_sha = str(protected_record["producer_record_sha256"])
        if producer_sha != identity_policy["protected_exclusion_producer_sha256"]:
            raise ValueError("probe source pool protected producer binding differs")
        protected = _load_protected_exclusion_bundle(
            root / relative_candidate,
            expected_producer_record_sha256=producer_sha,
        )
        if (
            protected.root != protected_root
            or protected.producer_record_sha256 != producer_sha
            or sha256_tree(protected.root) != protected_record["tree_sha256"]
        ):
            raise ValueError("probe source pool protected exclusion bundle differs")

        observed = 0
        observed_protected = 0
        observed_duplicates = 0
        source_handle = os.fdopen(os.dup(source_descriptor), "rb")
        decision_handle = os.fdopen(os.dup(decision_descriptor), "rb")
        try:
            source_reader = _CanonicalJsonlReader(
                source_handle,
                context="probe source pool manifest",
            )
            decision_reader = _CanonicalJsonlReader(
                decision_handle,
                context="probe source pool decision ledger",
            )
            source_iterator = iter(source_reader)
            decision_fields = {
                "schema_version",
                "shard_row_index",
                "record_id",
                "source_id",
                "group_id",
                "source_group_sha256",
                "parent_source_sha256",
                "normalized_content_sha256",
                "decision",
                "source_record",
            }
            with tempfile.TemporaryDirectory(prefix="probe-source-pool-replay-") as ledger_dir:
                with _IdentityLedger(Path(ledger_dir) / "identities.sqlite3") as ledger:
                    for expected_index, (_, decision) in enumerate(decision_reader):
                        if (
                            set(decision) != decision_fields
                            or decision.get("schema_version") != _DECISION_SCHEMA
                            or decision.get("shard_row_index") != expected_index
                            or decision.get("decision")
                            not in {"protected", "duplicate", "retained"}
                        ):
                            raise ValueError("probe source pool decision row differs")
                        source_row, identities, row_index = _validate_source_payload(
                            decision.get("source_record"),
                            expected_row_index=expected_index,
                        )
                        if (
                            row_index != expected_index
                            or decision.get("record_id") != source_row.record_id
                            or decision.get("source_id") != source_row.source_id
                            or decision.get("group_id") != source_row.group_id
                            or decision.get("source_group_sha256") != identities.source_group_sha256
                            or decision.get("parent_source_sha256")
                            != identities.parent_source_sha256
                            or decision.get("normalized_content_sha256")
                            != identities.normalized_content_sha256
                        ):
                            raise ValueError("probe source pool decision identity differs")
                        ledger.observe_source(
                            record_id=source_row.record_id,
                            source_id=source_row.source_id,
                        )
                        if _is_protected(identities, protected.identity_sets):
                            expected_decision = "protected"
                            observed_protected += 1
                        elif ledger.retain_normalized(identities.normalized_content_sha256):
                            expected_decision = "retained"
                            observed += 1
                        else:
                            expected_decision = "duplicate"
                            observed_duplicates += 1
                        if decision["decision"] != expected_decision:
                            raise ValueError("probe source pool decision replay differs")
                        if expected_decision == "retained":
                            try:
                                _, retained = next(source_iterator)
                            except StopIteration as exc:
                                raise ValueError(
                                    "probe source pool manifest ended before replay"
                                ) from exc
                            if retained != decision["source_record"]:
                                raise ValueError("probe source pool retained manifest row differs")
                    try:
                        next(source_iterator)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError("probe source pool manifest has extra rows")
        finally:
            source_handle.close()
            decision_handle.close()
        source_record = artifacts["source_manifest"]
        decision_record = artifacts["decision_ledger"]
        if (
            source_reader.rows != observed
            or source_reader.bytes_read != source_record["bytes"]
            or source_reader.sha256 != source_record["sha256"]
            or decision_reader.rows != observed + observed_protected + observed_duplicates
            or decision_reader.bytes_read != decision_record["bytes"]
            or decision_reader.sha256 != decision_record["sha256"]
            or counts["records"] != observed
            or counts["protected_records_removed"] != observed_protected
            or counts["duplicate_records_removed"] != observed_duplicates
            or counts["observed_parquet_rows"] != decision_reader.rows
        ):
            raise ValueError("probe source pool replay accounting differs")
        for artifact_name, path in artifact_paths.items():
            descriptor, descriptor_identity, digest = artifact_descriptors[artifact_name]
            _assert_descriptor_unchanged(
                path,
                descriptor,
                expected_sha256=digest,
                expected_identity=descriptor_identity,
                label=f"probe source pool {artifact_name}",
            )
        final_protected = _load_protected_exclusion_bundle(
            protected.run_path,
            expected_producer_record_sha256=producer_sha,
        )
        if (
            final_protected.identity_sets != protected.identity_sets
            or sha256_tree(final_protected.root) != protected_record["tree_sha256"]
            or _read_regular_bytes(
                resolved_run,
                label="probe source pool producer record",
            )
            != raw
            or {path.name for path in root.iterdir()} != expected_root_names
        ):
            raise ValueError("probe source pool bundle changed during replay")
        _assert_descriptor_unchanged(
            resolved_run,
            run_descriptor,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_identity=run_identity,
            label="probe source pool producer record",
        )
        return ProbeSourcePoolFreezeResult(
            source_manifest_path=artifact_paths["source_manifest"],
            decision_ledger_path=artifact_paths["decision_ledger"],
            protected_exclusion_path=protected.denylist_path,
            protected_exclusion_run_path=protected.run_path,
            run_path=resolved_run,
            run_sha256=expected_run_sha256,
            records=observed,
            protected_records_removed=observed_protected,
            duplicate_records_removed=observed_duplicates,
        )
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if decision_descriptor is not None:
            os.close(decision_descriptor)
        os.close(run_descriptor)


__all__ = [
    "ProbeSourcePoolFreezeConfig",
    "ProbeSourcePoolFreezeResult",
    "freeze_probe_source_pool",
    "load_probe_source_pool_bundle",
]
