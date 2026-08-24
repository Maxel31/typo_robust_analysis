from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.data import protected_denylist as denylist_module
from typo_robust_training.data import protected_registry as registry_module
from typo_robust_training.data.config import DatasetSource
from typo_robust_training.data.protected_denylist import freeze_protected_exclusion_denylist
from typo_robust_training.data.protected_registry import (
    INVENTORY_SCHEMA,
    TIERS,
)
from typo_robust_training.data.records import record_id_for
from typo_robust_training.data.sources import _format_huggingface_record
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.integrity import sha256_file
from typo_robust_training.probe import source_pool
from typo_robust_training.probe.source_pool import (
    ProbeSourcePoolFreezeConfig,
    ProbeSourcePoolFreezeResult,
    freeze_probe_source_pool,
    load_probe_source_pool_bundle,
)


REVISION = source_pool._REVISION
CODE_REVISION = "a" * 40


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _canonical_jsonl(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


@dataclass(frozen=True)
class _Checkout:
    revision: str

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "typo_robust_training_tree": "1" * 40,
            "typo_cot_tree": "2" * 40,
            "typo_cot_runtime_sources": ["projects/typo-cot/src/typo_cot/__init__.py"],
        }


def _row(identifier: str, text: str, *, tokens: int = 12) -> dict[str, object]:
    return {
        "text": text,
        "id": identifier,
        "dump": "CC-MAIN-2026-01",
        "url": f"https://example.test/{identifier}",
        "file_path": f"crawl/{identifier}.warc.gz",
        "language": "en",
        "language_score": 0.99,
        "token_count": tokens,
        "score": 4.2,
        "int_score": 4,
    }


def _protected_row(
    *,
    role: str,
    index: int,
    text: str,
    source: str = "protected-fixture",
    source_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, object]:
    identifier = source_id or f"protected-{index}"
    return {
        "schema_version": "robustness-clean-record/v1",
        "kind": "clean",
        "record_id": record_id_for(
            source=source,
            source_revision=REVISION,
            source_id=identifier,
        ),
        "source": source,
        "source_revision": REVISION,
        "source_split": "train",
        "source_id": identifier,
        "group_id": group_id or f"protected-group-{index}",
        "split": role,
        "text": text,
        "content_sha256": _sha(text),
        "normalized_content_sha256": normalized_content_sha256(text),
        "metadata": {"fixture": index},
    }


def _protected_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    training_row: dict[str, object] | None = None,
) -> denylist_module.ProtectedExclusionDenylistFreezeResult:
    checkout = registry_module._CheckoutAttestation(  # noqa: SLF001
        revision="b" * 40,
        project_tree="c" * 40,
    )
    monkeypatch.setattr(registry_module, "_attest_checkout", lambda: checkout)
    roles = {
        "training": "train",
        "localization": "diagnostic",
        "tune": "tune",
        "pre-pr": "pre_pr_gate",
        "sealed": "final_test",
    }
    snapshot = tmp_path / "protected-snapshot"
    manifests = snapshot / "manifests"
    manifests.mkdir(parents=True)
    tiers: list[dict[str, object]] = []
    for index, tier in enumerate(TIERS):
        if tier == "training" and training_row is not None:
            row = training_row
        elif tier in {"localization", "tune"}:
            row = _protected_row(
                role=roles[tier],
                index=index,
                text="Known historical cross-tier clean-text overlap.",
                source="protected-overlap",
                source_id="protected-overlap:shared-parent",
                group_id="protected-overlap:shared-group",
            )
        else:
            row = _protected_row(
                role=roles[tier],
                index=index,
                text=f"Distinct protected fixture text for {tier} number {index}.",
                source=f"protected-fixture-{tier}",
            )
        path = manifests / f"{tier}.jsonl"
        path.write_bytes(_canonical_jsonl(row))
        tiers.append(
            {
                "tier": tier,
                "inputs": [
                    {
                        "relative_path": f"manifests/{tier}.jsonl",
                        "sha256": sha256_file(path),
                        "accepted_schemas": ["robustness-clean-record/v1"],
                        "role": roles[tier],
                    }
                ],
            }
        )
    inventory = snapshot / "inventory.json"
    inventory.write_bytes(_canonical_json({"schema_version": INVENTORY_SCHEMA, "tiers": tiers}))
    return freeze_protected_exclusion_denylist(
        inventory_path=inventory,
        inventory_sha256=sha256_file(inventory),
        output_dir=tmp_path / "protected-published",
    )


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: tuple[dict[str, object], ...],
    training_row: dict[str, object] | None = None,
    output_name: str = "pool",
) -> ProbeSourcePoolFreezeConfig:
    protected = _protected_bundle(tmp_path, monkeypatch, training_row=training_row)
    parquet = tmp_path / "013_00000.parquet"
    parquet.write_bytes(b"pinned-test-parquet")
    parquet_sha = sha256_file(parquet)
    monkeypatch.setattr(source_pool, "_SHARD_SHA256", parquet_sha)
    monkeypatch.setattr(source_pool, "_SHARD_BYTES", parquet.stat().st_size)
    monkeypatch.setattr(source_pool, "_iter_parquet_rows", lambda _handle: iter(rows))
    monkeypatch.setattr(
        source_pool,
        "attest_runtime_checkout",
        lambda revision: _Checkout(revision),
    )
    return ProbeSourcePoolFreezeConfig(
        parquet_path=parquet,
        parquet_sha256=parquet_sha,
        protected_exclusion_run_path=protected.run_path,
        protected_exclusion_producer_sha256=protected.producer_record_sha256,
        code_revision=CODE_REVISION,
        output_dir=tmp_path / output_name,
    )


def _freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: tuple[dict[str, object], ...] | None = None,
    training_row: dict[str, object] | None = None,
    output_name: str = "pool",
) -> tuple[ProbeSourcePoolFreezeConfig, ProbeSourcePoolFreezeResult]:
    source_rows = rows or (_row("eligible", "One eligible source document."),)
    config = _config(
        tmp_path,
        monkeypatch,
        rows=source_rows,
        training_row=training_row,
        output_name=output_name,
    )
    return config, freeze_probe_source_pool(config)


def _update_artifact(run: dict[str, Any], name: str, path: Path) -> None:
    run["artifacts"][name]["sha256"] = sha256_file(path)
    run["artifacts"][name]["bytes"] = path.stat().st_size


def _rewrite_run(result: ProbeSourcePoolFreezeResult, run: dict[str, Any]) -> str:
    unsigned = dict(run)
    unsigned.pop("record_sha256", None)
    digest = source_pool._record_sha256(unsigned)
    run["record_sha256"] = digest
    result.run_path.write_bytes(source_pool._pretty_bytes(run))
    return digest


def test_freezer_replays_protected_and_normalized_duplicate_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected_text = "This protected document must never be reused."
    protected = _protected_row(
        role="train",
        index=50,
        text=protected_text,
        source="fineweb_edu",
        source_id="fineweb_edu:protected",
        group_id="fineweb_edu:protected",
    )
    rows = (
        _row("eligible", "A clean eligible document with a stable identity."),
        _row("duplicate", "  a CLEAN eligible document with a stable identity.  "),
        _row("protected", protected_text),
    )
    _, result = _freeze(
        tmp_path,
        monkeypatch,
        rows=rows,
        training_row=protected,
    )

    assert (result.records, result.protected_records_removed) == (1, 1)
    assert result.duplicate_records_removed == 1
    frozen = load_probe_source_pool_bundle(
        result.run_path,
        expected_run_sha256=result.run_sha256,
        expected_code_revision=CODE_REVISION,
    )
    assert frozen.records == 1
    decisions = [json.loads(line) for line in result.decision_ledger_path.read_text().splitlines()]
    assert [row["decision"] for row in decisions] == [
        "retained",
        "duplicate",
        "protected",
    ]
    assert all("source_record" in row for row in decisions)
    run = json.loads(result.run_path.read_text())
    assert run["model_outputs_observed"] is False
    assert run["token_count"]["used_for_probe_selection"] is False
    assert run["identity_policy"]["protected_exclusion_purpose"] == ("source-pool-exclusion-only")
    assert run["identity_policy"]["protected_exclusion_split_certified"] is False
    assert result.protected_exclusion_path.name == "denylist.json"
    assert (result.protected_exclusion_path.parent / "overlap_audit.json").is_file()


@pytest.mark.parametrize("dimension", ["source_group", "parent_source", "full_text"])
def test_every_protected_identity_dimension_excludes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimension: str,
) -> None:
    candidate_id = "fineweb_edu:candidate"
    candidate_text = "Candidate text whose exclusion dimension is under test."
    source_id = candidate_id if dimension == "parent_source" else "fineweb_edu:other-parent"
    group_id = candidate_id if dimension == "source_group" else "fineweb_edu:other-group"
    protected_text = candidate_text if dimension == "full_text" else f"{dimension} protected text"
    protected = _protected_row(
        role="train",
        index=60,
        text=protected_text,
        source="fineweb_edu",
        source_id=source_id,
        group_id=group_id,
    )
    _, result = _freeze(
        tmp_path,
        monkeypatch,
        rows=(
            _row("candidate", candidate_text),
            _row("control", "An unrelated eligible control document."),
        ),
        training_row=protected,
    )
    assert result.records == 1
    assert result.protected_records_removed == 1


def test_source_identity_matches_the_existing_fineweb_builder() -> None:
    row = _row("stable-id", "Stable FineWeb identity text.")
    dataset = DatasetSource(
        name="fineweb_edu",
        dataset=source_pool._DATASET,
        revision=REVISION,
        subset=source_pool._SUBSET,
        splits=("train",),
        role="pretraining",
        license="fixture",
        streaming=True,
        task=None,
    )
    existing = _format_huggingface_record("fineweb_edu", dataset, "train", 17, row)
    frozen = source_pool._clean_payload(row, row_index=17)
    assert frozen["source_id"] == existing.source_id == "fineweb_edu:stable-id"
    assert frozen["group_id"] == existing.group_id == "fineweb_edu:stable-id"
    assert frozen["record_id"] == existing.record_id


def test_regular_byte_reader_closes_its_descriptor_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"fixture":true}\n')
    closed_descriptors: list[int] = []
    original_close = source_pool.os.close

    def observe_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(source_pool.os, "close", observe_close)
    assert source_pool._read_regular_bytes(artifact, label="fixture") == artifact.read_bytes()
    assert len(closed_descriptors) == 1


def test_model_output_freeze_has_no_injectable_row_provider() -> None:
    assert tuple(inspect.signature(freeze_probe_source_pool).parameters) == ("config",)
    source_text = inspect.getsource(source_pool)
    assert "_load_protected_registry" not in source_text
    assert "load_protected_split_registry_bundle" not in source_text


def test_freezer_uses_the_public_protected_bundle_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = source_pool.load_protected_exclusion_denylist_bundle

    def observe(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(source_pool, "load_protected_exclusion_denylist_bundle", observe)
    _freeze(tmp_path, monkeypatch)
    assert calls >= 3


def test_freezer_rejects_wrong_fixed_shard_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, rows=())
    with pytest.raises(ValueError, match="preregistered final"):
        freeze_probe_source_pool(replace(config, parquet_sha256="f" * 64))


def test_freezer_rejects_wrong_fixed_shard_byte_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, rows=())
    monkeypatch.setattr(source_pool, "_SHARD_BYTES", config.parquet_path.stat().st_size + 1)
    with pytest.raises(ValueError, match="byte count"):
        freeze_probe_source_pool(config)


def test_atomic_publish_does_not_clobber_a_racing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool._publish_bundle

    def race(
        temporary: Path,
        target: Path,
        *,
        expected_parent_identity: tuple[int, int],
        bindings: source_pool._PublicationBindings,
    ) -> None:
        target.mkdir()
        (target / "owned").write_text("keep")
        original(
            temporary,
            target,
            expected_parent_identity=expected_parent_identity,
            bindings=bindings,
        )

    monkeypatch.setattr(source_pool, "_publish_bundle", race)
    with pytest.raises(FileExistsError, match="appeared"):
        freeze_probe_source_pool(config)
    assert (config.output_dir / "owned").read_text() == "keep"
    assert not (config.output_dir / source_pool._SOURCE_FILENAME).exists()
    assert not list(tmp_path.glob(f".{config.output_dir.name}.*"))


def test_publish_failure_removes_the_complete_staging_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )

    def fail(
        _temporary: Path,
        _target: Path,
        *,
        expected_parent_identity: tuple[int, int],
        bindings: source_pool._PublicationBindings,
    ) -> None:
        assert expected_parent_identity
        assert bindings.staging_run_sha256
        raise RuntimeError("simulated atomic publication failure")

    monkeypatch.setattr(source_pool, "_publish_bundle", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(f".{config.output_dir.name}.*"))


@pytest.mark.parametrize("attack", ["source", "protected", "parquet"])
def test_freezer_never_publishes_staging_or_input_mutated_before_final_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    """Prove final validation is adjacent to atomic publication."""

    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool._publish_bundle

    def mutate_then_publish(
        temporary: Path,
        target: Path,
        *,
        expected_parent_identity: tuple[int, int],
        bindings: source_pool._PublicationBindings,
    ) -> None:
        if attack == "source":
            source = temporary / source_pool._SOURCE_FILENAME
            source.write_bytes(source.read_bytes().replace(b"Enough", b"BROKEN"))
        elif attack == "protected":
            denylist = bindings.protected_root / "denylist.json"
            denylist.write_bytes(
                denylist.read_bytes().replace(b'"schema_version"', b'"schema_Xersion"', 1)
            )
        else:
            config.parquet_path.write_bytes(b"changed-before-final-attestation")
        original(
            temporary,
            target,
            expected_parent_identity=expected_parent_identity,
            bindings=bindings,
        )

    monkeypatch.setattr(source_pool, "_publish_bundle", mutate_then_publish)
    with pytest.raises(ValueError):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(f".{config.output_dir.name}.*"))


def test_freezer_rechecks_checkout_immediately_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    calls = 0

    def attest(revision: str) -> _Checkout:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated dirty checkout before publication")
        return _Checkout(revision)

    monkeypatch.setattr(source_pool, "attest_runtime_checkout", attest)
    with pytest.raises(ValueError, match="dirty checkout"):
        freeze_probe_source_pool(config)
    assert calls == 2
    assert not config.output_dir.exists()


def test_final_tree_binding_rejects_mutation_after_offline_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool.load_probe_source_pool_bundle

    def load_then_mutate(*args: object, **kwargs: object) -> ProbeSourcePoolFreezeResult:
        result = original(*args, **kwargs)
        result.source_manifest_path.write_bytes(
            result.source_manifest_path.read_bytes().replace(b"Enough", b"BROKEN")
        )
        return result

    monkeypatch.setattr(source_pool, "load_probe_source_pool_bundle", load_then_mutate)
    with pytest.raises(ValueError, match="tree changed"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()


def test_atomic_rename_rejects_staging_root_inode_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool.publish_directory_noreplace

    def substitute(
        staged: Path,
        output: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_staged_identity: tuple[int, int] | None = None,
        expected_tree_attestation: source_pool.DirectoryTreeAttestation | None = None,
    ) -> None:
        moved = staged.with_name(f"{staged.name}.moved")
        staged.rename(moved)
        staged.mkdir()
        try:
            original(
                staged,
                output,
                expected_parent_identity=expected_parent_identity,
                expected_staged_identity=expected_staged_identity,
                expected_tree_attestation=expected_tree_attestation,
            )
        finally:
            if staged.exists():
                staged.rmdir()
            if moved.exists():
                moved.rename(staged)

    monkeypatch.setattr(source_pool, "publish_directory_noreplace", substitute)
    with pytest.raises(ValueError, match="source changed"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()


def test_atomic_rename_rejects_file_mutation_inside_publication_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file write must not slip between the outer final hash and rename."""

    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool.publish_directory_noreplace

    def mutate_file_then_rename(
        staged: Path,
        output: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_staged_identity: tuple[int, int] | None = None,
        expected_tree_attestation: source_pool.DirectoryTreeAttestation | None = None,
    ) -> None:
        source = staged / source_pool._SOURCE_FILENAME
        source.write_bytes(source.read_bytes().replace(b"Enough", b"BROKEN"))
        original(
            staged,
            output,
            expected_parent_identity=expected_parent_identity,
            expected_staged_identity=expected_staged_identity,
            expected_tree_attestation=expected_tree_attestation,
        )

    monkeypatch.setattr(source_pool, "publish_directory_noreplace", mutate_file_then_rename)
    with pytest.raises(ValueError, match="tree changed"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()


def test_atomic_rename_rejects_unattested_nested_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed-world inventory includes empty directories, unlike a file-only digest."""

    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool.load_probe_source_pool_bundle

    def load_then_add_directory(*args: object, **kwargs: object) -> ProbeSourcePoolFreezeResult:
        result = original(*args, **kwargs)
        nested = result.protected_exclusion_path.parent / "inputs" / "unattested-empty"
        nested.mkdir()
        return result

    monkeypatch.setattr(source_pool, "load_probe_source_pool_bundle", load_then_add_directory)
    with pytest.raises(ValueError, match="tree changed"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("attack", ["symlink", "fifo", "hardlink"])
def test_atomic_rename_rejects_unattested_nonregular_or_hardlinked_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool.publish_directory_noreplace

    def inject_then_rename(
        staged: Path,
        output: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_staged_identity: tuple[int, int] | None = None,
        expected_tree_attestation: source_pool.DirectoryTreeAttestation | None = None,
    ) -> None:
        injected = staged / "unattested-node"
        if attack == "symlink":
            injected.symlink_to(staged / source_pool._SOURCE_FILENAME)
        elif attack == "fifo":
            os.mkfifo(injected)
        else:
            os.link(staged / source_pool._SOURCE_FILENAME, injected)
        original(
            staged,
            output,
            expected_parent_identity=expected_parent_identity,
            expected_staged_identity=expected_staged_identity,
            expected_tree_attestation=expected_tree_attestation,
        )

    monkeypatch.setattr(source_pool, "publish_directory_noreplace", inject_then_rename)
    with pytest.raises(ValueError, match="hard-linked|symlink|special"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()


def test_post_rename_loader_quarantines_mutation_before_success_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )
    original = source_pool.publish_directory_noreplace

    def publish_then_mutate(
        staged: Path,
        output: Path,
        *,
        expected_parent_identity: tuple[int, int] | None = None,
        expected_staged_identity: tuple[int, int] | None = None,
        expected_tree_attestation: source_pool.DirectoryTreeAttestation | None = None,
    ) -> None:
        original(
            staged,
            output,
            expected_parent_identity=expected_parent_identity,
            expected_staged_identity=expected_staged_identity,
            expected_tree_attestation=expected_tree_attestation,
        )
        source = output / source_pool._SOURCE_FILENAME
        source.write_bytes(source.read_bytes().replace(b"Enough", b"BROKEN"))

    monkeypatch.setattr(source_pool, "publish_directory_noreplace", publish_then_mutate)
    with pytest.raises(ValueError, match="was removed"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(f".{config.output_dir.name}.invalid-*"))


def test_publish_rejects_output_parent_inode_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
        output_name="publication/pool",
    )
    original = source_pool._publish_bundle

    def substitute(
        temporary: Path,
        target: Path,
        *,
        expected_parent_identity: tuple[int, int],
        bindings: source_pool._PublicationBindings,
    ) -> None:
        moved = target.parent.with_name("publication-moved")
        target.parent.rename(moved)
        target.parent.mkdir()
        original(
            temporary,
            target,
            expected_parent_identity=expected_parent_identity,
            bindings=bindings,
        )

    monkeypatch.setattr(source_pool, "_publish_bundle", substitute)
    with pytest.raises(ValueError, match="parent changed"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()
    assert not list((tmp_path / "publication-moved").glob(".pool.*"))


@pytest.mark.parametrize("kind", ["symlink", "ancestor", "hardlink"])
def test_freezer_rejects_parquet_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    config = _config(tmp_path, monkeypatch, rows=())
    if kind == "symlink":
        supplied = tmp_path / "shard-link.parquet"
        supplied.symlink_to(config.parquet_path)
    elif kind == "ancestor":
        link = tmp_path / "parquet-parent-link"
        link.symlink_to(tmp_path, target_is_directory=True)
        supplied = link / config.parquet_path.name
    else:
        supplied = config.parquet_path
        os.link(config.parquet_path, tmp_path / "parquet-hardlink")
    with pytest.raises(ValueError, match="symlink|unlinked"):
        freeze_probe_source_pool(replace(config, parquet_path=supplied))


def test_freezer_rejects_parquet_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        rows=(_row("a", "Enough source text for one row."),),
    )

    def mutate(_handle: object):
        yield _row("a", "Enough source text for one row.")
        config.parquet_path.write_bytes(b"changed-after-pinned-read")

    monkeypatch.setattr(source_pool, "_iter_parquet_rows", mutate)
    with pytest.raises(ValueError, match="changed while"):
        freeze_probe_source_pool(config)
    assert not config.output_dir.exists()


def test_freezer_rejects_hardlinked_protected_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, rows=())
    os.link(config.protected_exclusion_run_path, tmp_path / "producer-hardlink.json")
    with pytest.raises(ValueError, match="hard-linked"):
        freeze_probe_source_pool(config)


def test_loader_requires_external_run_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _freeze(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="externally pinned"):
        load_probe_source_pool_bundle(
            result.run_path,
            expected_run_sha256="0" * 64,
            expected_code_revision=CODE_REVISION,
        )


def test_loader_reads_the_run_once_per_pinned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _freeze(tmp_path, monkeypatch)
    observed_descriptors: list[int] = []
    original = source_pool._descriptor_bytes

    def observe(descriptor: int) -> bytes:
        observed_descriptors.append(descriptor)
        return original(descriptor)

    monkeypatch.setattr(source_pool, "_descriptor_bytes", observe)
    load_probe_source_pool_bundle(
        result.run_path,
        expected_run_sha256=result.run_sha256,
        expected_code_revision=CODE_REVISION,
    )
    assert len(observed_descriptors) == 2
    assert observed_descriptors[0] != observed_descriptors[1]


def test_loader_rejects_rehashed_adjacent_artifact_without_run_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _freeze(tmp_path, monkeypatch)
    result.source_manifest_path.write_bytes(
        result.source_manifest_path.read_bytes().replace(b"eligible", b"alteredx")
    )
    with pytest.raises(ValueError, match="externally pinned SHA-256"):
        load_probe_source_pool_bundle(
            result.run_path,
            expected_run_sha256=result.run_sha256,
            expected_code_revision=CODE_REVISION,
        )


def test_loader_rejects_fully_self_rehashed_record_id_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _freeze(tmp_path, monkeypatch)
    source_row = json.loads(result.source_manifest_path.read_text())
    decision = json.loads(result.decision_ledger_path.read_text())
    source_row["record_id"] = "f" * 64
    decision["record_id"] = "f" * 64
    decision["source_record"]["record_id"] = "f" * 64
    result.source_manifest_path.write_bytes(_canonical_jsonl(source_row))
    result.decision_ledger_path.write_bytes(_canonical_jsonl(decision))
    run = json.loads(result.run_path.read_text())
    _update_artifact(run, "source_manifest", result.source_manifest_path)
    _update_artifact(run, "decision_ledger", result.decision_ledger_path)
    forged_run_sha = _rewrite_run(result, run)
    with pytest.raises(ValueError, match="identity or metadata"):
        load_probe_source_pool_bundle(
            result.run_path,
            expected_run_sha256=forged_run_sha,
            expected_code_revision=CODE_REVISION,
        )


def test_loader_rejects_fully_rehashed_decision_count_spoof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _freeze(
        tmp_path,
        monkeypatch,
        rows=(
            _row("first", "First distinct eligible document."),
            _row("second", "Second distinct eligible document."),
        ),
    )
    source_rows = result.source_manifest_path.read_text().splitlines()
    decisions = [json.loads(line) for line in result.decision_ledger_path.read_text().splitlines()]
    decisions[0]["decision"] = "protected"
    result.source_manifest_path.write_bytes((source_rows[1] + "\n").encode())
    result.decision_ledger_path.write_bytes(b"".join(_canonical_jsonl(row) for row in decisions))
    run = json.loads(result.run_path.read_text())
    run["counts"].update(
        {
            "records": 1,
            "protected_records_removed": 1,
            "duplicate_records_removed": 0,
            "observed_parquet_rows": 2,
        }
    )
    _update_artifact(run, "source_manifest", result.source_manifest_path)
    _update_artifact(run, "decision_ledger", result.decision_ledger_path)
    forged_run_sha = _rewrite_run(result, run)
    with pytest.raises(ValueError, match="decision replay"):
        load_probe_source_pool_bundle(
            result.run_path,
            expected_run_sha256=forged_run_sha,
            expected_code_revision=CODE_REVISION,
        )


@pytest.mark.parametrize("attack", ["extra", "hardlink", "ancestor-symlink"])
def test_loader_rejects_closed_world_and_link_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    _, result = _freeze(tmp_path, monkeypatch)
    run_path = result.run_path
    if attack == "extra":
        (run_path.parent / "unattested").write_text("extra")
    elif attack == "hardlink":
        os.link(result.source_manifest_path, tmp_path / "external-hardlink")
    else:
        link = tmp_path / "pool-link"
        link.symlink_to(run_path.parent, target_is_directory=True)
        run_path = link / run_path.name
    with pytest.raises(ValueError, match="inventory|unlinked|symlink"):
        load_probe_source_pool_bundle(
            run_path,
            expected_run_sha256=result.run_sha256,
            expected_code_revision=CODE_REVISION,
        )


def test_loader_detects_manifest_inode_swap_during_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _freeze(tmp_path, monkeypatch)
    original = source_pool._validate_source_payload
    swapped = False

    def swap(value: object, *, expected_row_index: int | None = None):
        nonlocal swapped
        validated = original(value, expected_row_index=expected_row_index)
        if not swapped:
            swapped = True
            replacement = result.source_manifest_path.with_name(".replacement")
            replacement.write_bytes(result.source_manifest_path.read_bytes())
            os.replace(replacement, result.source_manifest_path)
        return validated

    monkeypatch.setattr(source_pool, "_validate_source_payload", swap)
    with pytest.raises(ValueError, match="changed while"):
        load_probe_source_pool_bundle(
            result.run_path,
            expected_run_sha256=result.run_sha256,
            expected_code_revision=CODE_REVISION,
        )


def test_upstream_token_counts_do_not_change_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = (
        _row("first", "A token-independent retained document.", tokens=1),
        _row("duplicate", " a TOKEN-independent retained document. ", tokens=999_999),
        _row("second", "Another token-independent retained document.", tokens=2),
    )
    _, first = _freeze(tmp_path / "first", monkeypatch, rows=rows)
    changed = tuple({**row, "token_count": 10_000_000 - index} for index, row in enumerate(rows))
    _, second = _freeze(tmp_path / "second", monkeypatch, rows=changed)
    first_decisions = [
        (row["source_id"], row["decision"])
        for row in map(json.loads, first.decision_ledger_path.read_text().splitlines())
    ]
    second_decisions = [
        (row["source_id"], row["decision"])
        for row in map(json.loads, second.decision_ledger_path.read_text().splitlines())
    ]
    assert first_decisions == second_decisions
    assert first.records == second.records == 2


def test_identity_ledgers_are_disk_backed_for_streaming_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_databases: list[object] = []
    original = source_pool.sqlite3.connect

    def observe(database: object, *args: object, **kwargs: object):
        observed_databases.append(database)
        return original(database, *args, **kwargs)

    monkeypatch.setattr(source_pool.sqlite3, "connect", observe)
    rows = tuple(
        _row(f"row-{index}", f"Unique streaming source document number {index}.")
        for index in range(500)
    )
    _, result = _freeze(tmp_path, monkeypatch, rows=rows)
    assert result.records == 500
    assert observed_databases
    assert all(database != ":memory:" for database in observed_databases)


def test_parquet_schema_types_are_exact(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    row = _row("schema", "Exact schema fixture text.")
    columns = {
        key: pa.array(
            [value],
            type=pa.int32() if key == "token_count" else None,
        )
        for key, value in row.items()
    }
    path = tmp_path / "wrong-schema.parquet"
    pq.write_table(pa.table(columns), path)
    with path.open("rb") as handle, pytest.raises(ValueError, match="schema differs"):
        tuple(source_pool._iter_parquet_rows(handle))


def test_cli_requires_external_protected_producer_identity() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)
    args = parser.parse_args(
        [
            "freeze-probe-source-pool",
            "--source-parquet",
            "shard.parquet",
            "--source-parquet-sha256",
            "a" * 64,
            "--protected-exclusion-run",
            "protected/freeze_protected_exclusion_denylist_run.json",
            "--protected-exclusion-producer-sha256",
            "b" * 64,
            "--code-revision",
            "c" * 40,
            "--output-dir",
            "output",
        ]
    )
    assert args.source_parquet == Path("shard.parquet")
    assert args.protected_exclusion_run.name == "freeze_protected_exclusion_denylist_run.json"
    assert args.protected_exclusion_producer_sha256 == "b" * 64
    assert args.code_revision == "c" * 40
