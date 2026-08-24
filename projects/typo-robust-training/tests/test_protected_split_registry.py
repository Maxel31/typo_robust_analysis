from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.data import protected_registry as registry_module
from typo_robust_training.data.protected_registry import (
    INVENTORY_SCHEMA,
    PRODUCER_SCHEMA,
    REGISTRY_SCHEMA,
    TIERS,
    freeze_protected_split_registry,
    load_protected_split_registry_bundle,
)
from typo_robust_training.data.records import record_id_for
from typo_robust_training.data.splits import normalized_content_sha256


REVISION = "a" * 40
CHECKOUT = registry_module._CheckoutAttestation(  # noqa: SLF001
    revision="b" * 40,
    project_tree="c" * 40,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(source: str, source_id: str) -> str:
    return record_id_for(source=source, source_revision=REVISION, source_id=source_id)


def _common(
    *,
    schema: str,
    role: str,
    index: int,
    text: str,
    source: str = "fixture-source",
    source_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    identifier = source_id or f"record-{index}"
    row: dict[str, Any] = {
        "schema_version": schema,
        "record_id": _identity(source, identifier),
        "source": source,
        "source_revision": REVISION,
        "source_split": "fixture",
        "source_id": identifier,
        "group_id": group_id or f"group-{index}",
        "split": role,
        "metadata": {"fixture": index},
    }
    if schema == "robustness-clean-record/v1":
        row.update(
            {
                "kind": "clean",
                "text": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "normalized_content_sha256": normalized_content_sha256(text),
            }
        )
    elif schema == "robustness-evaluation-corpus-record/v1":
        row.update(
            {
                "kind": "clean-corpus",
                "text": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    else:
        typo = f"{text} typoo"
        row.update({"clean_text": text, "typo_text": typo})
        if schema == "robustness-fixed-typo-pair/v1":
            row.update({"kind": "synthetic", "operation": "deletion"})
        elif schema == "robustness-natural-pair/v1":
            row.update(
                {
                    "kind": "natural",
                    "operation": "natural",
                    "clean_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "typo_sha256": hashlib.sha256(typo.encode()).hexdigest(),
                }
            )
        else:  # pragma: no cover - fixture invariant
            raise AssertionError(schema)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_inventory(
    tmp_path: Path,
    *,
    rows_by_tier: dict[str, list[dict[str, Any]]] | None = None,
    schemas: dict[str, str] | None = None,
) -> tuple[Path, str, dict[str, Path], dict[str, list[dict[str, Any]]]]:
    root = tmp_path / "snapshot-2026-08-22"
    root.mkdir(parents=True)
    default_schemas = {
        "training": "robustness-clean-record/v1",
        "localization": "robustness-fixed-typo-pair/v1",
        "tune": "robustness-natural-pair/v1",
        "pre-pr": "robustness-evaluation-corpus-record/v1",
        "sealed": "robustness-fixed-typo-pair/v1",
    }
    if schemas is not None:
        default_schemas.update(schemas)
    roles = {
        "training": "train",
        "localization": "diagnostic",
        "tune": "tune",
        "pre-pr": "pre_pr_gate",
        "sealed": "final_test",
    }
    if rows_by_tier is None:
        rows_by_tier = {
            tier: [
                _common(
                    schema=default_schemas[tier],
                    role=roles[tier],
                    index=index,
                    text=f"Complete protected text for {tier} number {index}.",
                )
            ]
            for index, tier in enumerate(TIERS)
        }
    paths: dict[str, Path] = {}
    tiers: list[dict[str, object]] = []
    for tier in TIERS:
        path = root / "manifests" / f"{tier}.jsonl"
        _write_jsonl(path, rows_by_tier[tier])
        paths[tier] = path
        tiers.append(
            {
                "tier": tier,
                "inputs": [
                    {
                        "relative_path": f"manifests/{tier}.jsonl",
                        "sha256": _digest(path),
                        "accepted_schema": default_schemas[tier],
                        "role": roles[tier],
                    }
                ],
            }
        )
    inventory = root / "inventory.json"
    inventory.write_bytes(_canonical({"schema_version": INVENTORY_SCHEMA, "tiers": tiers}))
    return inventory, _digest(inventory), paths, rows_by_tier


@pytest.fixture(autouse=True)
def _fixed_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, "_attest_checkout", lambda: CHECKOUT)


def _freeze(tmp_path: Path) -> tuple[registry_module.ProtectedSplitRegistryFreezeResult, str]:
    inventory, inventory_sha, _, _ = _fixture_inventory(tmp_path)
    result = freeze_protected_split_registry(
        inventory_path=inventory,
        inventory_sha256=inventory_sha,
        output_dir=tmp_path / "published",
    )
    return result, inventory_sha


def test_freeze_and_external_hash_verification_round_trip(tmp_path: Path) -> None:
    result, inventory_sha = _freeze(tmp_path)
    assert result.input_records == 5
    assert (
        result.inventory_path.read_bytes()
        == (tmp_path / "snapshot-2026-08-22" / "inventory.json").read_bytes()
    )
    registry = json.loads(result.registry_path.read_text())
    assert registry["schema_version"] == REGISTRY_SCHEMA
    assert [row["tier"] for row in registry["registries"]] == list(TIERS)
    assert all(row["source_group_sha256"] for row in registry["registries"])
    assert all(row["parent_source_sha256"] for row in registry["registries"])
    assert all(row["normalized_content_sha256"] for row in registry["registries"])

    run = json.loads(result.run_path.read_text())
    assert run["schema_version"] == PRODUCER_SCHEMA
    assert run["inventory"]["external_sha256"] == inventory_sha
    assert run["checkout_attestation"] == CHECKOUT.as_dict()
    assert run["output"]["input_records"] == 5
    bundle = load_protected_split_registry_bundle(
        result.run_path,
        expected_producer_record_sha256=result.producer_record_sha256,
    )
    assert bundle.registry_path == result.registry_path
    assert len(bundle.input_paths) == 5


def test_pair_registers_both_clean_and_typo_full_texts(tmp_path: Path) -> None:
    result, _ = _freeze(tmp_path)
    registry = json.loads(result.registry_path.read_text())
    localization = next(row for row in registry["registries"] if row["tier"] == "localization")
    original = json.loads(
        (tmp_path / "snapshot-2026-08-22/manifests/localization.jsonl").read_text().splitlines()[0]
    )
    assert sorted(localization["normalized_content_sha256"]) == sorted(
        (
            normalized_content_sha256(original["clean_text"]),
            normalized_content_sha256(original["typo_text"]),
        )
    )


def test_fixed_evaluation_pair_derived_record_id_is_supported(tmp_path: Path) -> None:
    inventory, _, paths, rows = _fixture_inventory(tmp_path)
    row = rows["sealed"][0]
    parent_record_id = row["record_id"]
    row["metadata"]["evaluation_condition"] = "random-2"
    row.update({"generator_seed": 42, "generator_variant": 7, "edit_count": 2})
    row["record_id"] = hashlib.sha256(
        (
            f"frozen-evaluation-pair/v4\0final_test\0random-2\0{42}\0{7}\0{2}\0{parent_record_id}"
        ).encode()
    ).hexdigest()
    _write_jsonl(paths["sealed"], rows["sealed"])
    payload = json.loads(inventory.read_text())
    payload["tiers"][4]["inputs"][0]["sha256"] = _digest(paths["sealed"])
    inventory.write_bytes(_canonical(payload))
    result = freeze_protected_split_registry(
        inventory_path=inventory,
        inventory_sha256=_digest(inventory),
        output_dir=tmp_path / "output",
    )
    assert result.input_records == 5


def test_same_tier_fixed_variants_may_share_one_parent_source(tmp_path: Path) -> None:
    inventory, _, paths, rows = _fixture_inventory(tmp_path)
    parent_record_id = rows["sealed"][0]["record_id"]
    variants: list[dict[str, Any]] = []
    for variant in (7, 8):
        row = json.loads(json.dumps(rows["sealed"][0]))
        row["metadata"]["evaluation_condition"] = "random-2"
        row["typo_text"] = f"{row['clean_text']} variant-{variant}"
        row.update({"generator_seed": 42, "generator_variant": variant, "edit_count": 2})
        row["record_id"] = hashlib.sha256(
            (
                "frozen-evaluation-pair/v4\0"
                f"final_test\0random-2\0{42}\0{variant}\0{2}\0{parent_record_id}"
            ).encode()
        ).hexdigest()
        variants.append(row)
    rows["sealed"] = variants
    _write_jsonl(paths["sealed"], variants)
    payload = json.loads(inventory.read_text())
    payload["tiers"][4]["inputs"][0]["sha256"] = _digest(paths["sealed"])
    inventory.write_bytes(_canonical(payload))
    result = freeze_protected_split_registry(
        inventory_path=inventory,
        inventory_sha256=_digest(inventory),
        output_dir=tmp_path / "output",
    )
    run = json.loads(result.run_path.read_text())
    assert run["output"]["tier_record_counts"]["sealed"] == 2
    assert run["output"]["tier_unique_record_counts"]["sealed"] == 2


def test_same_tier_exact_duplicates_dedupe_but_conflicts_fail(tmp_path: Path) -> None:
    inventory, inventory_sha, paths, rows = _fixture_inventory(tmp_path)
    duplicated = rows["training"] * 2
    _write_jsonl(paths["training"], duplicated)
    payload = json.loads(inventory.read_text())
    payload["tiers"][0]["inputs"][0]["sha256"] = _digest(paths["training"])
    inventory.write_bytes(_canonical(payload))
    result = freeze_protected_split_registry(
        inventory_path=inventory,
        inventory_sha256=_digest(inventory),
        output_dir=tmp_path / "deduped",
    )
    run = json.loads(result.run_path.read_text())
    assert run["output"]["tier_record_counts"]["training"] == 2
    assert run["output"]["tier_unique_record_counts"]["training"] == 1

    conflict = dict(rows["training"][0])
    conflict["text"] = "A conflicting full text."
    conflict["content_sha256"] = hashlib.sha256(conflict["text"].encode()).hexdigest()
    conflict["normalized_content_sha256"] = normalized_content_sha256(conflict["text"])
    _write_jsonl(paths["training"], [rows["training"][0], conflict])
    payload["tiers"][0]["inputs"][0]["sha256"] = _digest(paths["training"])
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="conflicting"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "conflict",
        )


def test_three_tier_transitive_identity_bridge_is_rejected(tmp_path: Path) -> None:
    schemas = {tier: "robustness-clean-record/v1" for tier in TIERS}
    roles = {
        "training": "train",
        "localization": "diagnostic",
        "tune": "tune",
        "pre-pr": "pre_pr_gate",
        "sealed": "final_test",
    }
    rows = {
        tier: [
            _common(
                schema=schemas[tier],
                role=roles[tier],
                index=index,
                text=f"Disjoint full text {tier}",
            )
        ]
        for index, tier in enumerate(TIERS)
    }
    rows["training"][0]["group_id"] = "bridge-group"
    rows["localization"][0]["group_id"] = "bridge-group"
    rows["localization"][0]["source_id"] = "bridge-parent"
    rows["localization"][0]["record_id"] = _identity("fixture-source", "bridge-parent")
    rows["tune"][0]["source_id"] = "bridge-parent"
    rows["tune"][0]["record_id"] = _identity("fixture-source", "bridge-parent")
    _inventory, inventory_sha, _, _ = _fixture_inventory(
        tmp_path,
        rows_by_tier=rows,
        schemas=schemas,
    )
    with pytest.raises(ValueError, match="overlap transitively"):
        freeze_protected_split_registry(
            inventory_path=_inventory,
            inventory_sha256=inventory_sha,
            output_dir=tmp_path / "output",
        )


def test_clean_typo_cross_tier_overlap_is_rejected(tmp_path: Path) -> None:
    inventory, inventory_sha, paths, rows = _fixture_inventory(tmp_path)
    shared = rows["training"][0]["text"]
    rows["tune"][0]["typo_text"] = shared
    rows["tune"][0]["typo_sha256"] = hashlib.sha256(shared.encode()).hexdigest()
    _write_jsonl(paths["tune"], rows["tune"])
    payload = json.loads(inventory.read_text())
    payload["tiers"][2]["inputs"][0]["sha256"] = _digest(paths["tune"])
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="overlap transitively"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "output",
        )


def test_content_identity_uses_full_text_not_prefix(tmp_path: Path) -> None:
    inventory, inventory_sha, paths, rows = _fixture_inventory(tmp_path)
    prefix = "Identical first 200 characters " + "x" * 200
    for tier, ending in (("training", " alpha"), ("pre-pr", " beta")):
        field = "text"
        rows[tier][0][field] = prefix + ending
        rows[tier][0]["content_sha256"] = hashlib.sha256(rows[tier][0][field].encode()).hexdigest()
        if tier == "training":
            rows[tier][0]["normalized_content_sha256"] = normalized_content_sha256(
                rows[tier][0][field]
            )
        _write_jsonl(paths[tier], rows[tier])
    payload = json.loads(inventory.read_text())
    for tier_index, tier in ((0, "training"), (3, "pre-pr")):
        payload["tiers"][tier_index]["inputs"][0]["sha256"] = _digest(paths[tier])
    inventory.write_bytes(_canonical(payload))
    result = freeze_protected_split_registry(
        inventory_path=inventory,
        inventory_sha256=_digest(inventory),
        output_dir=tmp_path / "output",
    )
    registry = json.loads(result.registry_path.read_text())
    hashes = {row["tier"]: set(row["normalized_content_sha256"]) for row in registry["registries"]}
    assert hashes["training"].isdisjoint(hashes["pre-pr"])


@pytest.mark.parametrize("attack", ["inventory", "input", "tree"])
def test_source_tree_symlinks_are_rejected(tmp_path: Path, attack: str) -> None:
    inventory, inventory_sha, paths, _ = _fixture_inventory(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("outside\n")
    if attack == "inventory":
        link = tmp_path / "inventory-link.json"
        link.symlink_to(inventory)
        inventory = link
    elif attack == "input":
        original = paths["training"]
        copy = tmp_path / "training-copy.jsonl"
        copy.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(copy)
    else:
        (inventory.parent / "unrelated-link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=tmp_path / "output",
        )


def test_inventory_rejects_traversal_unknown_schema_and_wrong_external_hash(tmp_path: Path) -> None:
    inventory, inventory_sha, _, _ = _fixture_inventory(tmp_path)
    with pytest.raises(ValueError, match="external SHA-256"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256="f" * 64,
            output_dir=tmp_path / "wrong-hash",
        )
    payload = json.loads(inventory.read_text())
    payload["tiers"][0]["inputs"][0]["relative_path"] = "../escape.jsonl"
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="traversal"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "traversal",
        )
    payload["tiers"][0]["inputs"][0]["relative_path"] = "manifests/training.jsonl"
    payload["tiers"][0]["inputs"][0]["accepted_schema"] = "unknown/v1"
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="unsupported"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "unknown",
        )


def test_inventory_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    inventory, _, _, _ = _fixture_inventory(tmp_path)
    original = inventory.read_text(encoding="utf-8")
    inventory.write_text(
        original.replace(
            '  "schema_version":',
            '  "schema_version":"duplicate",\n  "schema_version":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "output",
        )


def test_duplicate_json_key_missing_field_utf8_and_newline_are_rejected(tmp_path: Path) -> None:
    inventory, _, paths, _ = _fixture_inventory(tmp_path)
    original = paths["training"].read_text().strip()
    paths["training"].write_text(original[:-1] + ',"source":"duplicate"}\n')
    payload = json.loads(inventory.read_text())
    payload["tiers"][0]["inputs"][0]["sha256"] = _digest(paths["training"])
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "duplicate",
        )

    for raw, message, name in (
        (
            b'{"schema_version":"robustness-clean-record/v1","split":"train"}\n',
            "source",
            "missing",
        ),
        (b"\xff\n", "UTF-8", "utf8"),
        (original.encode(), "final LF", "newline"),
        (original.encode() + b"\r\n", "LF, never CR", "crlf"),
    ):
        paths["training"].write_bytes(raw)
        payload["tiers"][0]["inputs"][0]["sha256"] = _digest(paths["training"])
        inventory.write_bytes(_canonical(payload))
        with pytest.raises(ValueError, match=message):
            freeze_protected_split_registry(
                inventory_path=inventory,
                inventory_sha256=_digest(inventory),
                output_dir=tmp_path / name,
            )


def test_existing_relevant_hashes_are_recomputed(tmp_path: Path) -> None:
    inventory, _, paths, rows = _fixture_inventory(tmp_path)
    rows["sealed"][0]["normalized_noisy_sha256"] = "0" * 64
    _write_jsonl(paths["sealed"], rows["sealed"])
    payload = json.loads(inventory.read_text())
    payload["tiers"][4]["inputs"][0]["sha256"] = _digest(paths["sealed"])
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="normalized_noisy_sha256"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "output",
        )


def test_final_rehash_detects_toctou_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, inventory_sha, paths, _ = _fixture_inventory(tmp_path)

    def mutate() -> None:
        paths["training"].write_bytes(paths["training"].read_bytes() + b" \n")

    monkeypatch.setattr(registry_module, "_before_final_rehash", mutate)
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="changed before publication"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".published.*"))


def test_final_rehash_rejects_checkout_attestation_change_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, inventory_sha, _, _ = _fixture_inventory(tmp_path)
    changed = registry_module._CheckoutAttestation(  # noqa: SLF001
        revision="d" * 40,
        project_tree="e" * 40,
    )
    observed = iter((CHECKOUT, changed))
    monkeypatch.setattr(registry_module, "_attest_checkout", lambda: next(observed))
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="checkout changed before publication"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".published.*"))


def test_preexisting_output_is_preserved(tmp_path: Path) -> None:
    inventory, inventory_sha, _, _ = _fixture_inventory(tmp_path)
    output = tmp_path / "published"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("do not replace")
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=output,
        )
    assert sentinel.read_text() == "do not replace"


def test_loader_rejects_input_substitution_and_bundle_symlink(tmp_path: Path) -> None:
    result, _ = _freeze(tmp_path)
    run = json.loads(result.run_path.read_text())
    copied = result.root / run["inputs"][0]["copied_relative_path"]
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="copied input"):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=result.producer_record_sha256,
        )

    result2, _ = _freeze(tmp_path / "second")
    run2 = json.loads(result2.run_path.read_text())
    copied2 = result2.root / run2["inputs"][0]["copied_relative_path"]
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(copied2.read_bytes())
    copied2.unlink()
    copied2.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        load_protected_split_registry_bundle(
            result2.run_path,
            expected_producer_record_sha256=result2.producer_record_sha256,
        )


def test_external_producer_hash_defeats_adjacent_rehash_forgery(tmp_path: Path) -> None:
    result, _ = _freeze(tmp_path)
    run = json.loads(result.run_path.read_text())
    run["output"]["input_records"] += 1
    unsigned = dict(run)
    del unsigned["record_sha256"]
    forged = registry_module._canonical_sha256(unsigned)  # noqa: SLF001
    run["record_sha256"] = forged
    result.run_path.write_bytes(_canonical(run))
    with pytest.raises(ValueError, match="external SHA-256"):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=result.producer_record_sha256,
        )


def test_external_producer_hash_defeats_source_copy_and_adjacent_hash_forgery(
    tmp_path: Path,
) -> None:
    result, _ = _freeze(tmp_path)
    externally_pinned = result.producer_record_sha256
    run = json.loads(result.run_path.read_text())
    inventory = json.loads(result.inventory_path.read_text())
    copied = result.root / run["inputs"][0]["copied_relative_path"]
    row = json.loads(copied.read_text().strip())
    row["text"] = "A substituted but internally rehashed training record."
    row["content_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    row["normalized_content_sha256"] = normalized_content_sha256(row["text"])
    _write_jsonl(copied, [row])
    substituted_sha = _digest(copied)

    inventory["tiers"][0]["inputs"][0]["sha256"] = substituted_sha
    result.inventory_path.write_bytes(_canonical(inventory))
    inventory_sha = _digest(result.inventory_path)
    run["inventory"].update(
        {
            "external_sha256": inventory_sha,
            "sha256": inventory_sha,
            "bytes": result.inventory_path.stat().st_size,
        }
    )
    run["inputs"][0].update(
        {
            "expected_sha256": substituted_sha,
            "sha256": substituted_sha,
            "bytes": copied.stat().st_size,
        }
    )
    unsigned = dict(run)
    del unsigned["record_sha256"]
    run["record_sha256"] = registry_module._canonical_sha256(unsigned)  # noqa: SLF001
    result.run_path.write_bytes(_canonical(run))

    with pytest.raises(ValueError, match="external SHA-256"):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=externally_pinned,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tier", "invented", "input tier differs"),
        ("accepted_schema", "invented/v1", "input schema differs"),
    ),
)
def test_loader_rejects_resigned_unknown_input_contract(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    result, _ = _freeze(tmp_path)
    run = json.loads(result.run_path.read_text())
    run["inputs"][0][field] = value
    unsigned = dict(run)
    del unsigned["record_sha256"]
    resigned = registry_module._canonical_sha256(unsigned)  # noqa: SLF001
    run["record_sha256"] = resigned
    result.run_path.write_bytes(_canonical(run))
    with pytest.raises(ValueError, match=message):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=resigned,
        )


def test_loader_rejects_resigned_identity_rule_substitution(tmp_path: Path) -> None:
    result, _ = _freeze(tmp_path)
    run = json.loads(result.run_path.read_text())
    run["identity_rules"]["normalized_content"] = "prefix-only-hash/v0"
    unsigned = dict(run)
    del unsigned["record_sha256"]
    resigned = registry_module._canonical_sha256(unsigned)  # noqa: SLF001
    run["record_sha256"] = resigned
    result.run_path.write_bytes(_canonical(run))
    with pytest.raises(ValueError, match="identity rules differ"):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=resigned,
        )


def test_inventory_substitution_after_freeze_is_rejected(tmp_path: Path) -> None:
    result, _ = _freeze(tmp_path)
    inventory = json.loads(result.inventory_path.read_text())
    inventory["tiers"][0]["inputs"][0]["role"] = "substituted"
    result.inventory_path.write_bytes(_canonical(inventory))
    with pytest.raises(ValueError, match="inventory copy bytes"):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=result.producer_record_sha256,
        )


def test_cli_exposes_freeze_and_verification_only_commands(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)
    freeze = parser.parse_args(
        [
            "freeze-protected-split-registry",
            "--inventory",
            str(tmp_path / "inventory.json"),
            "--inventory-sha256",
            "a" * 64,
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert freeze.command == "freeze-protected-split-registry"
    verify = parser.parse_args(
        [
            "verify-protected-split-registry",
            "--producer-run",
            str(tmp_path / "run.json"),
            "--producer-record-sha256",
            "b" * 64,
        ]
    )
    assert verify.command == "verify-protected-split-registry"


def test_output_root_and_ancestor_symlinks_are_rejected(tmp_path: Path) -> None:
    inventory, inventory_sha, _, _ = _fixture_inventory(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "alias-parent"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=alias / "output",
        )


def test_loader_rejects_unexpected_closed_world_file(tmp_path: Path) -> None:
    result, _ = _freeze(tmp_path)
    (result.root / "unexpected.txt").write_text("not attested")
    with pytest.raises(ValueError, match="file inventory"):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=result.producer_record_sha256,
        )
