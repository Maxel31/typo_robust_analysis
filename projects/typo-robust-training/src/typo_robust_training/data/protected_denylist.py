"""Freeze overlapping historical splits into an exclusion-only identity denylist.

This artifact is intentionally incapable of certifying split disjointness.  It
exists only to prevent historical protected identities from entering a new
source pool.  Its loader replays the same parsing and identity computation as
the strict protected-split registry before returning immutable identity sets.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from typo_robust_training.data import protected_registry as registry
from typo_robust_training.integrity import sha256_file
from typo_robust_training.training.json_io import write_json_atomic


DENYLIST_SCHEMA = "typo-protected-exclusion-denylist/v1"
DENYLIST_PRODUCER_SCHEMA = "freeze-protected-exclusion-denylist-run/v1"
DENYLIST_PURPOSE = "source-pool-exclusion-only"

_DENYLIST_FILENAME = "denylist.json"
_OVERLAP_AUDIT_FILENAME = "overlap_audit.json"
_RUN_FILENAME = "freeze_protected_exclusion_denylist_run.json"


@dataclass(frozen=True, slots=True)
class ProtectedExclusionDenylistFreezeResult:
    root: Path
    denylist_path: Path
    overlap_audit_path: Path
    inventory_path: Path
    run_path: Path
    producer_record_sha256: str
    input_records: int
    overlap_audit_sha256: str


@dataclass(frozen=True, slots=True)
class ProtectedExclusionDenylistBundle:
    """Verified historical identities that may only be used for exclusion."""

    root: Path
    denylist_path: Path
    overlap_audit_path: Path
    inventory_path: Path
    input_paths: tuple[Path, ...]
    run_path: Path
    producer_record_sha256: str
    input_records: int
    overlap_audit_sha256: str
    identity_sets: registry.ProtectedSplitIdentitySets
    purpose: Literal["source-pool-exclusion-only"]
    split_certified: Literal[False]


def _identity_sets_payload(
    identity_sets: registry.ProtectedSplitIdentitySets,
) -> dict[str, list[str]]:
    return {
        "source_group_sha256": sorted(identity_sets.source_group_sha256),
        "parent_source_sha256": sorted(identity_sets.parent_source_sha256),
        "normalized_content_sha256": sorted(identity_sets.normalized_content_sha256),
    }


def _denylist_payload(build: registry._RegistryBuild) -> dict[str, object]:  # noqa: SLF001
    return {
        "schema_version": DENYLIST_SCHEMA,
        "purpose": DENYLIST_PURPOSE,
        "split_certified": False,
        "identity_rules": dict(registry._IDENTITY_RULES),  # noqa: SLF001
        "identity_sets": _identity_sets_payload(build.identity_sets),
    }


def _input_records(build: registry._RegistryBuild) -> list[dict[str, object]]:  # noqa: SLF001
    return [
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


def _run_payload(
    *,
    checkout: registry._CheckoutAttestation,  # noqa: SLF001
    inventory: registry.ProtectedSplitInventory,
    inventory_copy: Path,
    denylist_path: Path,
    overlap_audit_path: Path,
    build: registry._RegistryBuild,  # noqa: SLF001
) -> dict[str, object]:
    identity_counts = {
        name: len(values) for name, values in _identity_sets_payload(build.identity_sets).items()
    }
    return {
        "schema_version": DENYLIST_PRODUCER_SCHEMA,
        "status": "completed",
        "purpose": DENYLIST_PURPOSE,
        "split_certified": False,
        "checkout_attestation": checkout.as_dict(),
        "inventory": {
            "relative_path": registry._INVENTORY_FILENAME,  # noqa: SLF001
            "external_sha256": inventory.external_sha256,
            "sha256": sha256_file(inventory_copy),
            "bytes": inventory_copy.stat().st_size,
        },
        "inputs": _input_records(build),
        "identity_rules": dict(registry._IDENTITY_RULES),  # noqa: SLF001
        "outputs": {
            "denylist": {
                "relative_path": _DENYLIST_FILENAME,
                "schema_version": DENYLIST_SCHEMA,
                "purpose": DENYLIST_PURPOSE,
                "split_certified": False,
                "sha256": sha256_file(denylist_path),
                "bytes": denylist_path.stat().st_size,
                "input_records": sum(build.tier_record_counts.values()),
                "tier_record_counts": dict(build.tier_record_counts),
                "tier_unique_record_counts": dict(build.tier_unique_record_counts),
                "identity_counts": identity_counts,
            },
            "overlap_audit": {
                "relative_path": _OVERLAP_AUDIT_FILENAME,
                "schema_version": registry.OVERLAP_AUDIT_SCHEMA,
                "sha256": sha256_file(overlap_audit_path),
                "bytes": overlap_audit_path.stat().st_size,
                "collision_components": len(build.overlap_components),
            },
        },
    }


def _before_denylist_final_rehash() -> None:
    """Private test seam for simulating mutation before publication."""


def freeze_protected_exclusion_denylist(
    *,
    inventory_path: Path,
    inventory_sha256: str,
    output_dir: Path,
) -> ProtectedExclusionDenylistFreezeResult:
    """Freeze a closed exclusion-only bundle from an externally pinned inventory."""

    target = registry._new_output_target(output_dir)  # noqa: SLF001
    inventory = registry.load_protected_split_inventory(
        inventory_path,
        expected_sha256=inventory_sha256,
    )
    checkout = registry._attest_checkout()  # noqa: SLF001
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (temporary / "inputs").mkdir()
        inventory_copy = temporary / registry._INVENTORY_FILENAME  # noqa: SLF001
        inventory_copy.write_bytes(inventory.raw)
        specs_and_paths: list[tuple[registry.ProtectedInputSpec, Path, str]] = []
        for index, spec in enumerate(inventory.inputs):
            source = inventory.root / spec.relative_path
            copied_relative = registry._input_copy_name(index, spec)  # noqa: SLF001
            copied = temporary / copied_relative
            copied.write_bytes(
                registry._read_regular_bytes(source, label="protected JSONL input")  # noqa: SLF001
            )
            specs_and_paths.append((spec, copied, copied_relative))
        build = registry._audit_protected_inputs(specs_and_paths)  # noqa: SLF001
        denylist_path = temporary / _DENYLIST_FILENAME
        write_json_atomic(denylist_path, _denylist_payload(build))
        overlap_audit_path = temporary / _OVERLAP_AUDIT_FILENAME
        write_json_atomic(
            overlap_audit_path,
            registry._overlap_audit_payload(build.overlap_components),  # noqa: SLF001
        )
        unsigned_run = _run_payload(
            checkout=checkout,
            inventory=inventory,
            inventory_copy=inventory_copy,
            denylist_path=denylist_path,
            overlap_audit_path=overlap_audit_path,
            build=build,
        )
        producer_record_sha256 = registry._canonical_sha256(unsigned_run)  # noqa: SLF001
        run_path = temporary / _RUN_FILENAME
        write_json_atomic(
            run_path,
            {**unsigned_run, "record_sha256": producer_record_sha256},
        )
        load_protected_exclusion_denylist_bundle(
            run_path,
            expected_producer_record_sha256=producer_record_sha256,
        )

        _before_denylist_final_rehash()
        if registry._attest_checkout() != checkout:  # noqa: SLF001
            raise ValueError("protected denylist producer checkout changed before publication")
        if (
            hashlib.sha256(
                registry._read_regular_bytes(  # noqa: SLF001
                    inventory.path,
                    label="protected split inventory",
                )
            ).hexdigest()
            != inventory.external_sha256
        ):
            raise ValueError("protected split inventory changed before publication")
        registry._assert_tree_without_symlinks(  # noqa: SLF001
            inventory.root,
            label="protected split inventory tree",
        )
        for audit in build.audits:
            source = inventory.root / audit.spec.relative_path
            if (
                hashlib.sha256(
                    registry._read_regular_bytes(  # noqa: SLF001
                        source,
                        label="protected JSONL input",
                    )
                ).hexdigest()
                != audit.spec.sha256
            ):
                raise ValueError("protected JSONL input changed before publication")
        registry._publish_directory_noreplace(temporary, target)  # noqa: SLF001
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ProtectedExclusionDenylistFreezeResult(
        root=target,
        denylist_path=target / _DENYLIST_FILENAME,
        overlap_audit_path=target / _OVERLAP_AUDIT_FILENAME,
        inventory_path=target / registry._INVENTORY_FILENAME,  # noqa: SLF001
        run_path=target / _RUN_FILENAME,
        producer_record_sha256=producer_record_sha256,
        input_records=sum(build.tier_record_counts.values()),
        overlap_audit_sha256=sha256_file(target / _OVERLAP_AUDIT_FILENAME),
    )


def _producer_input_spec(
    record: Mapping[str, object],
    *,
    index: int,
) -> tuple[registry.ProtectedInputSpec, str]:
    if set(record) != {
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
        raise ValueError("protected denylist producer input record fields differ")
    raw_schemas = record.get("accepted_schemas")
    if not isinstance(raw_schemas, list) or not raw_schemas:
        raise ValueError("protected denylist producer input schemas differ")
    schemas = tuple(registry._text(value, field="producer input schema") for value in raw_schemas)  # noqa: SLF001
    if (
        len(set(schemas)) != len(schemas)
        or schemas != tuple(sorted(schemas))
        or any(schema not in registry.ALLOWED_RECORD_SCHEMAS for schema in schemas)
    ):
        raise ValueError("protected denylist producer input schemas differ")
    spec = registry.ProtectedInputSpec(
        tier=registry._text(record.get("tier"), field="producer input tier"),  # noqa: SLF001
        relative_path=registry._relative_path(  # noqa: SLF001
            record.get("source_relative_path"),
            field="producer input source path",
        ),
        sha256=registry._sha(  # noqa: SLF001
            record.get("expected_sha256"),
            field="producer input expected hash",
        ),
        accepted_schemas=schemas,
        role=registry._text(record.get("role"), field="producer input role"),  # noqa: SLF001
    )
    if spec.tier not in registry.TIERS:
        raise ValueError("protected denylist producer input tier differs")
    copied_relative = registry._relative_path(  # noqa: SLF001
        record.get("copied_relative_path"),
        field="producer input copied path",
    )
    if copied_relative != registry._input_copy_name(index, spec):  # noqa: SLF001
        raise ValueError("protected denylist producer input copy path differs")
    if record.get("sha256") != spec.sha256:
        raise ValueError("protected denylist producer input hash differs")
    registry._positive_integer(record.get("bytes"), field="producer input bytes")  # noqa: SLF001
    registry._positive_integer(record.get("records"), field="producer input records")  # noqa: SLF001
    registry._positive_integer(  # noqa: SLF001
        record.get("unique_records"),
        field="producer unique records",
    )
    return spec, copied_relative


def load_protected_exclusion_denylist_bundle(
    producer_run_path: Path,
    *,
    expected_producer_record_sha256: str,
) -> ProtectedExclusionDenylistBundle:
    """Verify and replay an exclusion-only bundle before exposing its identities."""

    run_path, run = registry._load_canonical_run(  # noqa: SLF001
        producer_run_path,
        expected_producer_record_sha256=expected_producer_record_sha256,
    )
    if set(run) != {
        "schema_version",
        "status",
        "purpose",
        "split_certified",
        "checkout_attestation",
        "inventory",
        "inputs",
        "identity_rules",
        "outputs",
        "record_sha256",
    }:
        raise ValueError("protected denylist producer record fields differ")
    if (
        run.get("schema_version") != DENYLIST_PRODUCER_SCHEMA
        or run.get("status") != "completed"
        or run.get("purpose") != DENYLIST_PURPOSE
        or run.get("split_certified") is not False
    ):
        raise ValueError("protected denylist producer contract differs")
    if run.get("identity_rules") != registry._IDENTITY_RULES:  # noqa: SLF001
        raise ValueError("protected denylist identity rules differ")
    checkout = run.get("checkout_attestation")
    if not isinstance(checkout, Mapping) or set(checkout) != {"revision", "project_tree"}:
        raise ValueError("protected denylist checkout attestation fields differ")
    revision = checkout.get("revision")
    project_tree = checkout.get("project_tree")
    if (
        not isinstance(revision, str)
        or registry._GIT_OBJECT.fullmatch(revision) is None  # noqa: SLF001
        or not isinstance(project_tree, str)
        or registry._GIT_OBJECT.fullmatch(project_tree) is None  # noqa: SLF001
    ):
        raise ValueError("protected denylist checkout attestation differs")

    inventory_record = run.get("inventory")
    if not isinstance(inventory_record, Mapping) or set(inventory_record) != {
        "relative_path",
        "external_sha256",
        "sha256",
        "bytes",
    }:
        raise ValueError("protected denylist producer inventory record differs")
    if inventory_record.get("relative_path") != registry._INVENTORY_FILENAME:  # noqa: SLF001
        raise ValueError("protected denylist producer inventory path differs")
    inventory_sha = registry._sha(  # noqa: SLF001
        inventory_record.get("external_sha256"),
        field="producer inventory external SHA-256",
    )
    if inventory_record.get("sha256") != inventory_sha:
        raise ValueError("protected denylist inventory copy differs from its external SHA-256")
    inventory_bytes = registry._positive_integer(  # noqa: SLF001
        inventory_record.get("bytes"),
        field="producer inventory bytes",
    )

    inputs = run.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("protected denylist producer input inventory differs")
    input_paths: list[Path] = []
    captured_inputs: list[bytes] = []
    run_specs: list[registry.ProtectedInputSpec] = []
    run_records: list[Mapping[str, object]] = []
    expected_files = {
        _RUN_FILENAME,
        _DENYLIST_FILENAME,
        _OVERLAP_AUDIT_FILENAME,
        registry._INVENTORY_FILENAME,  # noqa: SLF001
    }
    for index, record in enumerate(inputs):
        if not isinstance(record, Mapping):
            raise ValueError("protected denylist producer input record differs")
        spec, copied_relative = _producer_input_spec(record, index=index)
        path = registry._regular_file(  # noqa: SLF001
            run_path.parent / copied_relative,
            label="protected denylist input copy",
        )
        raw = registry._read_regular_bytes(  # noqa: SLF001
            path,
            label="protected denylist input copy",
        )
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != spec.sha256:
            raise ValueError("protected denylist copied input bytes differ")
        input_paths.append(path)
        captured_inputs.append(raw)
        run_specs.append(spec)
        run_records.append(record)
        expected_files.add(copied_relative)
    registry._closed_bundle_files(run_path.parent, expected_files)  # noqa: SLF001

    inventory_path = registry._regular_file(  # noqa: SLF001
        run_path.parent / registry._INVENTORY_FILENAME,  # noqa: SLF001
        label="protected denylist inventory copy",
    )
    inventory_raw = registry._read_regular_bytes(  # noqa: SLF001
        inventory_path,
        label="protected denylist inventory copy",
    )
    if (
        len(inventory_raw) != inventory_bytes
        or hashlib.sha256(inventory_raw).hexdigest() != inventory_sha
    ):
        raise ValueError("protected denylist inventory copy bytes differ")
    inventory = registry.ProtectedSplitInventory(
        root=run_path.parent,
        path=inventory_path,
        external_sha256=inventory_sha,
        inputs=registry._decode_inventory(  # noqa: SLF001
            inventory_raw,
            context=str(inventory_path),
        ),
        raw=inventory_raw,
    )
    if inventory.inputs != tuple(run_specs):
        raise ValueError("protected denylist producer inputs differ from its inventory copy")

    build = registry._audit_protected_inputs(  # noqa: SLF001
        tuple(
            (spec, path, str(record["copied_relative_path"]))
            for spec, path, record in zip(run_specs, input_paths, run_records, strict=True)
        ),
        captured_inputs=tuple(captured_inputs),
    )
    for audit, record in zip(build.audits, run_records, strict=True):
        if audit.records != record["records"] or audit.unique_records != record["unique_records"]:
            raise ValueError("protected denylist producer input counts differ")

    denylist_path = registry._regular_file(  # noqa: SLF001
        run_path.parent / _DENYLIST_FILENAME,
        label="protected exclusion denylist",
    )
    denylist_raw = registry._read_regular_bytes(  # noqa: SLF001
        denylist_path,
        label="protected exclusion denylist",
    )
    denylist = registry._strict_json(denylist_raw, context=str(denylist_path))  # noqa: SLF001
    expected_denylist = _denylist_payload(build)
    if denylist_raw != registry._canonical_bytes(denylist) or dict(denylist) != expected_denylist:  # noqa: SLF001
        raise ValueError("protected exclusion denylist differs from replayed inputs")

    overlap_audit_path = registry._regular_file(  # noqa: SLF001
        run_path.parent / _OVERLAP_AUDIT_FILENAME,
        label="protected overlap audit",
    )
    overlap_audit_raw = registry._read_regular_bytes(  # noqa: SLF001
        overlap_audit_path,
        label="protected overlap audit",
    )
    overlap_audit = registry._strict_json(  # noqa: SLF001
        overlap_audit_raw,
        context=str(overlap_audit_path),
    )
    expected_audit = registry._overlap_audit_payload(build.overlap_components)  # noqa: SLF001
    if (
        overlap_audit_raw != registry._canonical_bytes(overlap_audit)  # noqa: SLF001
        or dict(overlap_audit) != expected_audit
    ):
        raise ValueError("protected overlap audit differs from replayed inputs")

    outputs = run.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"denylist", "overlap_audit"}:
        raise ValueError("protected denylist output inventory differs")
    expected_outputs = _run_payload(
        checkout=registry._CheckoutAttestation(  # noqa: SLF001
            revision=revision,
            project_tree=project_tree,
        ),
        inventory=inventory,
        inventory_copy=inventory_path,
        denylist_path=denylist_path,
        overlap_audit_path=overlap_audit_path,
        build=build,
    )["outputs"]
    if dict(outputs) != expected_outputs:
        raise ValueError("protected denylist output accounting differs")

    for path, captured in zip(input_paths, captured_inputs, strict=True):
        if registry._read_regular_bytes(path, label="protected denylist input copy") != captured:  # noqa: SLF001
            raise ValueError("protected denylist copied input changed during verification")
    if (
        registry._read_regular_bytes(  # noqa: SLF001
            inventory_path,
            label="protected denylist inventory copy",
        )
        != inventory_raw
        or registry._read_regular_bytes(  # noqa: SLF001
            denylist_path,
            label="protected exclusion denylist",
        )
        != denylist_raw
        or registry._read_regular_bytes(  # noqa: SLF001
            overlap_audit_path,
            label="protected overlap audit",
        )
        != overlap_audit_raw
    ):
        raise ValueError("protected denylist bundle changed during verification")
    _, final_run = registry._load_canonical_run(  # noqa: SLF001
        run_path,
        expected_producer_record_sha256=expected_producer_record_sha256,
    )
    if dict(final_run) != dict(run):
        raise ValueError("protected denylist producer record changed during verification")
    registry._closed_bundle_files(run_path.parent, expected_files)  # noqa: SLF001
    return ProtectedExclusionDenylistBundle(
        root=run_path.parent,
        denylist_path=denylist_path,
        overlap_audit_path=overlap_audit_path,
        inventory_path=inventory_path,
        input_paths=tuple(input_paths),
        run_path=run_path,
        producer_record_sha256=expected_producer_record_sha256,
        input_records=sum(build.tier_record_counts.values()),
        overlap_audit_sha256=hashlib.sha256(overlap_audit_raw).hexdigest(),
        identity_sets=build.identity_sets,
        purpose=DENYLIST_PURPOSE,
        split_certified=False,
    )


__all__ = [
    "DENYLIST_PRODUCER_SCHEMA",
    "DENYLIST_PURPOSE",
    "DENYLIST_SCHEMA",
    "ProtectedExclusionDenylistBundle",
    "ProtectedExclusionDenylistFreezeResult",
    "freeze_protected_exclusion_denylist",
    "load_protected_exclusion_denylist_bundle",
]
