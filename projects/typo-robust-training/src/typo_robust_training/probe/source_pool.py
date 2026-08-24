"""Freeze the unused FineWeb-Edu shard used to construct word-probe cohorts.

The source pool is deliberately model-output free.  It preserves the original
FineWeb document identity so the five-tier protected registry can exclude a
document even when another pipeline observed it through a different text
window.  The consumer-facing JSONL uses the existing training-source schema;
the upstream ``token_count`` is retained only as source metadata and is not
used to select the probe boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.records import CleanRecord
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.integrity import sha256_file
from typo_robust_training.probe.attestation import attest_runtime_checkout
from typo_robust_training.probe.cohort_builder import (
    probe_parent_source_sha256,
    probe_source_group_sha256,
)
from typo_robust_training.probe.producer import _load_protected_registry
from typo_robust_training.training.pairs import TrainingSource


_DATASET = "HuggingFaceFW/fineweb-edu"
_SOURCE = "fineweb_edu"
_REVISION = "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5"
_SUBSET = "sample-10BT"
_SPLIT = "train"
_SHARD = "sample/10BT/013_00000.parquet"
_SHARD_SHA256 = "b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de"
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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_FILENAME = "probe_source_pool.jsonl"
_REGISTRY_FILENAME = "protected_split_registry.json"
_RUN_FILENAME = "freeze_probe_source_pool_run.json"


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


def _regular_file(path: Path, *, expected_sha256: str, label: str) -> Path:
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} expected SHA-256 must be one lowercase digest")
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be one regular file")
    resolved = supplied.resolve()
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} differs from its externally pinned SHA-256")
    return resolved


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat(follow_symlinks=False)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _assert_input_unchanged(
    path: Path,
    *,
    expected_sha256: str,
    expected_identity: tuple[int, int, int, int, int],
    label: str,
) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or _file_identity(path) != expected_identity
        or sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"{label} changed while the source pool was being frozen")


def _output_path(path: Path) -> Path:
    target = Path(path).absolute()
    if os.path.lexists(target):
        raise FileExistsError(f"probe source pool output already exists: {target}")
    ancestor = target.parent
    while not os.path.lexists(ancestor):
        ancestor = ancestor.parent
    if ancestor.is_symlink():
        raise ValueError("probe source pool output ancestor must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or target.parent.resolve() != target.parent:
        raise ValueError("probe source pool output parent must not contain symlinks")
    return target


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


def _iter_parquet_rows(path: Path) -> Iterator[Mapping[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency via datasets
        raise RuntimeError("pyarrow is required to read the pinned FineWeb-Edu shard") from exc
    parquet = pq.ParquetFile(path)
    if tuple(parquet.schema_arrow.names) != _EXPECTED_PARQUET_COLUMNS:
        raise ValueError("FineWeb-Edu parquet column inventory differs")
    for batch in parquet.iter_batches(batch_size=1024, columns=list(_EXPECTED_PARQUET_COLUMNS)):
        for row in batch.to_pylist():
            if not isinstance(row, Mapping):
                raise ValueError("FineWeb-Edu parquet emitted a non-object row")
            yield row


RowProvider = Callable[[Path], Iterator[Mapping[str, object]]]


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


def _protected_union(path: Path) -> set[str]:
    return set().union(*_load_protected_registry(path))


def _source_identities(payload: Mapping[str, object]) -> frozenset[str]:
    source = TrainingSource.from_dict(payload)
    return frozenset(
        {
            probe_source_group_sha256(source),
            probe_parent_source_sha256(source),
            str(payload["normalized_content_sha256"]),
        }
    )


@dataclass(frozen=True, slots=True)
class ProbeSourcePoolFreezeConfig:
    parquet_path: Path
    parquet_sha256: str
    protected_registry_path: Path
    protected_registry_sha256: str
    code_revision: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ProbeSourcePoolFreezeResult:
    source_manifest_path: Path
    protected_registry_path: Path
    run_path: Path
    run_sha256: str
    records: int
    protected_records_removed: int
    duplicate_records_removed: int


def _publish_bundle(temporary: Path, target: Path) -> None:
    """Publish with a no-clobber directory and make the run file the commit point."""

    target.mkdir(parents=False, exist_ok=False)
    ordered = (_SOURCE_FILENAME, _REGISTRY_FILENAME, _RUN_FILENAME)
    try:
        for name in ordered:
            os.link(temporary / name, target / name, follow_symlinks=False)
    except Exception:
        # Do not erase a path another process may have added.  An incomplete
        # directory has no trusted run manifest and therefore fails closed.
        raise


def freeze_probe_source_pool(
    config: ProbeSourcePoolFreezeConfig,
    *,
    row_provider: RowProvider = _iter_parquet_rows,
) -> ProbeSourcePoolFreezeResult:
    """Freeze one exact unused shard after five-tier identity exclusion."""

    if not isinstance(config, ProbeSourcePoolFreezeConfig):
        raise TypeError("probe source pool config has the wrong type")
    if config.parquet_sha256 != _SHARD_SHA256:
        raise ValueError("probe source pool must use the preregistered final FineWeb-Edu shard")
    if (
        not isinstance(config.code_revision, str)
        or _REVISION_RE.fullmatch(config.code_revision) is None
    ):
        raise ValueError("probe source pool code revision must be one exact commit")
    parquet = _regular_file(
        config.parquet_path,
        expected_sha256=config.parquet_sha256,
        label="FineWeb-Edu parquet shard",
    )
    protected_path = _regular_file(
        config.protected_registry_path,
        expected_sha256=config.protected_registry_sha256,
        label="protected split registry",
    )
    parquet_identity = _file_identity(parquet)
    protected_identity = _file_identity(protected_path)
    checkout = attest_runtime_checkout(config.code_revision)
    protected = _protected_union(protected_path)
    target = _output_path(config.output_dir)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    records = 0
    protected_removed = 0
    duplicate_removed = 0
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    seen_normalized: set[str] = set()
    try:
        source_path = temporary / _SOURCE_FILENAME
        with source_path.open("xb") as handle:
            for row_index, row in enumerate(row_provider(parquet)):
                payload = _clean_payload(row, row_index=row_index)
                record_id = str(payload["record_id"])
                source_id = str(payload["source_id"])
                normalized = str(payload["normalized_content_sha256"])
                if record_id in seen_record_ids or source_id in seen_source_ids:
                    raise ValueError("FineWeb-Edu shard contains a duplicate source identity")
                seen_record_ids.add(record_id)
                seen_source_ids.add(source_id)
                if _source_identities(payload) & protected:
                    protected_removed += 1
                    continue
                if normalized in seen_normalized:
                    duplicate_removed += 1
                    continue
                seen_normalized.add(normalized)
                handle.write(_jsonl_row(payload))
                records += 1
        if records == 0:
            raise ValueError("probe source pool contains no eligible records")
        _assert_input_unchanged(
            parquet,
            expected_sha256=config.parquet_sha256,
            expected_identity=parquet_identity,
            label="FineWeb-Edu parquet shard",
        )
        _assert_input_unchanged(
            protected_path,
            expected_sha256=config.protected_registry_sha256,
            expected_identity=protected_identity,
            label="protected split registry",
        )
        (temporary / _REGISTRY_FILENAME).write_bytes(protected_path.read_bytes())
        _assert_input_unchanged(
            protected_path,
            expected_sha256=config.protected_registry_sha256,
            expected_identity=protected_identity,
            label="protected split registry",
        )
        artifacts = {
            "source_manifest": {
                "relative_path": _SOURCE_FILENAME,
                "sha256": sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            },
            "protected_registry": {
                "relative_path": _REGISTRY_FILENAME,
                "sha256": sha256_file(temporary / _REGISTRY_FILENAME),
                "bytes": (temporary / _REGISTRY_FILENAME).stat().st_size,
            },
        }
        payload: dict[str, object] = {
            "schema_version": "freeze-probe-source-pool-run/v1",
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
                "bytes": parquet.stat().st_size,
                "columns": list(_EXPECTED_PARQUET_COLUMNS),
            },
            "identity_policy": {
                "protected_registry_sha256": config.protected_registry_sha256,
                "protected_dimensions": [
                    "source_group_sha256",
                    "parent_source_sha256",
                    "normalized_content_sha256",
                ],
                "within_pool_duplicates": "first-parquet-row-by-normalized-content/v1",
            },
            "token_count": {
                "provider": "fineweb-edu-published-metadata-only/v1",
                "used_for_probe_selection": False,
            },
            "counts": {
                "records": records,
                "protected_records_removed": protected_removed,
                "duplicate_records_removed": duplicate_removed,
                "observed_parquet_rows": len(seen_record_ids),
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
        _publish_bundle(temporary, target)
        return ProbeSourcePoolFreezeResult(
            source_manifest_path=target / _SOURCE_FILENAME,
            protected_registry_path=target / _REGISTRY_FILENAME,
            run_path=target / _RUN_FILENAME,
            run_sha256=run_sha,
            records=records,
            protected_records_removed=protected_removed,
            duplicate_records_removed=duplicate_removed,
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def load_probe_source_pool_bundle(
    run_path: Path,
    *,
    expected_run_sha256: str,
    expected_code_revision: str,
) -> ProbeSourcePoolFreezeResult:
    """Revalidate a frozen pool from an externally stored producer digest."""

    if _SHA256.fullmatch(expected_run_sha256 or "") is None:
        raise ValueError("expected probe source pool run SHA-256 is invalid")
    if _REVISION_RE.fullmatch(expected_code_revision or "") is None:
        raise ValueError("expected probe source pool code revision is invalid")
    supplied = Path(run_path)
    if supplied.name != _RUN_FILENAME or supplied.is_symlink() or not supplied.is_file():
        raise ValueError("probe source pool run must be the regular canonical run file")
    raw = supplied.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(supplied))
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
        or payload.get("schema_version") != "freeze-probe-source-pool-run/v1"
        or payload.get("status") != "completed"
        or payload.get("model_outputs_observed") is not False
    ):
        raise ValueError("probe source pool run fields or status differ")
    code = payload.get("code")
    if not isinstance(code, Mapping) or code.get("revision") != expected_code_revision:
        raise ValueError("probe source pool code revision differs")
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
        or isinstance(source.get("bytes"), bool)
        or not isinstance(source.get("bytes"), int)
        or source["bytes"] <= 0
        or source.get("columns") != list(_EXPECTED_PARQUET_COLUMNS)
    ):
        raise ValueError("probe source pool dataset identity differs")
    identity_policy = payload.get("identity_policy")
    token_count = payload.get("token_count")
    if (
        not isinstance(identity_policy, Mapping)
        or set(identity_policy)
        != {
            "protected_registry_sha256",
            "protected_dimensions",
            "within_pool_duplicates",
        }
        or _SHA256.fullmatch(str(identity_policy.get("protected_registry_sha256"))) is None
        or identity_policy.get("protected_dimensions")
        != [
            "source_group_sha256",
            "parent_source_sha256",
            "normalized_content_sha256",
        ]
        or identity_policy.get("within_pool_duplicates")
        != "first-parquet-row-by-normalized-content/v1"
        or not isinstance(token_count, Mapping)
        or dict(token_count)
        != {
            "provider": "fineweb-edu-published-metadata-only/v1",
            "used_for_probe_selection": False,
        }
    ):
        raise ValueError("probe source pool identity or token-count policy differs")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "source_manifest",
        "protected_registry",
    }:
        raise ValueError("probe source pool artifact inventory differs")
    resolved: dict[str, Path] = {}
    for name, filename in (
        ("source_manifest", _SOURCE_FILENAME),
        ("protected_registry", _REGISTRY_FILENAME),
    ):
        record = artifacts[name]
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "sha256",
            "bytes",
        }:
            raise ValueError("probe source pool artifact record differs")
        path = supplied.parent / filename
        if (
            record.get("relative_path") != filename
            or path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != supplied.parent.resolve()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError("probe source pool artifact differs")
        resolved[name] = path
    protected = _protected_union(resolved["protected_registry"])
    if identity_policy.get("protected_registry_sha256") != artifacts["protected_registry"].get(
        "sha256"
    ):
        raise ValueError("probe source pool protected registry binding differs")
    counts = payload.get("counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts)
        != {
            "records",
            "protected_records_removed",
            "duplicate_records_removed",
            "observed_parquet_rows",
        }
        or any(
            isinstance(counts.get(field), bool)
            or not isinstance(counts.get(field), int)
            or counts[field] < 0
            for field in counts
        )
        or counts["records"] <= 0
        or counts["observed_parquet_rows"]
        != counts["records"]
        + counts["protected_records_removed"]
        + counts["duplicate_records_removed"]
    ):
        raise ValueError("probe source pool counts differ")
    observed = 0
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    seen_normalized: set[str] = set()
    for line_number, line in read_lf_jsonl_lines(
        resolved["source_manifest"], context="probe source pool manifest"
    ):
        value = strict_loads(line, context=f"{resolved['source_manifest']}:{line_number}")
        source_row = TrainingSource.from_dict(value)
        if (
            source_row.kind != "clean"
            or source_row.source != _SOURCE
            or source_row.source_revision != _REVISION
            or source_row.source_split != _SPLIT
            or source_row.task is not None
        ):
            raise ValueError("probe source pool row identity differs")
        identities = _source_identities(value)
        normalized = str(value["normalized_content_sha256"])
        if identities & protected:
            raise ValueError("probe source pool overlaps a protected tier")
        if (
            source_row.record_id in seen_record_ids
            or source_row.source_id in seen_source_ids
            or normalized in seen_normalized
        ):
            raise ValueError("probe source pool contains a duplicate identity")
        seen_record_ids.add(source_row.record_id)
        seen_source_ids.add(source_row.source_id)
        seen_normalized.add(normalized)
        observed += 1
    if counts.get("records") != observed or observed == 0:
        raise ValueError("probe source pool record count differs")
    return ProbeSourcePoolFreezeResult(
        source_manifest_path=resolved["source_manifest"],
        protected_registry_path=resolved["protected_registry"],
        run_path=supplied.resolve(),
        run_sha256=expected_run_sha256,
        records=observed,
        protected_records_removed=counts["protected_records_removed"],
        duplicate_records_removed=counts["duplicate_records_removed"],
    )


__all__ = [
    "ProbeSourcePoolFreezeConfig",
    "ProbeSourcePoolFreezeResult",
    "freeze_probe_source_pool",
    "load_probe_source_pool_bundle",
]
