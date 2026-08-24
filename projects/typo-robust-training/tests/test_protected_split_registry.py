from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.data import protected_denylist as denylist_module
from typo_robust_training.data import protected_registry as registry_module
from typo_robust_training.data.protected_denylist import (
    DENYLIST_PRODUCER_SCHEMA,
    DENYLIST_PURPOSE,
    DENYLIST_SCHEMA,
    ProtectedExclusionDenylistBundle,
    freeze_protected_exclusion_denylist,
    load_protected_exclusion_denylist_bundle,
)
from typo_robust_training.data.protected_registry import (
    INVENTORY_SCHEMA,
    OVERLAP_AUDIT_SCHEMA,
    PRODUCER_SCHEMA,
    REGISTRY_SCHEMA,
    TIERS,
    ProtectedSplitOverlapError,
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
                        "accepted_schemas": [default_schemas[tier]],
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


def _overlapping_historical_inventory(
    tmp_path: Path,
) -> tuple[Path, str, dict[str, Path], dict[str, list[dict[str, Any]]]]:
    roles = {
        "training": "train",
        "localization": "diagnostic",
        "tune": "tune",
        "pre-pr": "pre_pr_gate",
        "sealed": "final_test",
    }
    shared_natural = "A historical natural correction with exact content."
    shared_benchmark = "Which duplicated benchmark question has the correct answer?"
    rows = {
        "training": [
            _common(
                schema="robustness-clean-record/v1",
                role=roles["training"],
                index=0,
                text="Unique historical training text.",
            )
        ],
        "localization": [
            _common(
                schema="robustness-fixed-typo-pair/v1",
                role=roles["localization"],
                index=1,
                text="Unique historical localization text.",
            )
        ],
        "tune": [
            _common(
                schema="robustness-natural-pair/v1",
                role=roles["tune"],
                index=2,
                source="gtc-tune",
                source_id="natural-tune",
                group_id="natural-tune-group",
                text=shared_natural,
            )
        ],
        "pre-pr": [
            _common(
                schema="robustness-natural-pair/v1",
                role=roles["pre-pr"],
                index=3,
                source="gtc-pre-pr",
                source_id="natural-pre-pr",
                group_id="natural-pre-pr-group",
                text=shared_natural,
            ),
            _common(
                schema="robustness-evaluation-corpus-record/v1",
                role=roles["pre-pr"],
                index=4,
                source="mmlu-pre-pr",
                source_id="question-pre-pr",
                group_id="question-pre-pr-group",
                text=shared_benchmark,
            ),
        ],
        "sealed": [
            _common(
                schema="robustness-evaluation-corpus-record/v1",
                role=roles["sealed"],
                index=5,
                source="mmlu-pro-sealed",
                source_id="question-sealed",
                group_id="question-sealed-group",
                text=shared_benchmark,
            )
        ],
    }
    inventory, _, paths, rows = _fixture_inventory(
        tmp_path,
        rows_by_tier=rows,
        schemas={
            "pre-pr": "robustness-natural-pair/v1",
            "sealed": "robustness-evaluation-corpus-record/v1",
        },
    )
    payload = json.loads(inventory.read_text())
    payload["tiers"][3]["inputs"][0]["accepted_schemas"] = sorted(
        [
            "robustness-natural-pair/v1",
            "robustness-evaluation-corpus-record/v1",
        ]
    )
    inventory.write_bytes(_canonical(payload))
    return inventory, _digest(inventory), paths, rows


def _freeze_denylist(
    tmp_path: Path,
) -> tuple[denylist_module.ProtectedExclusionDenylistFreezeResult, Path, str]:
    inventory, inventory_sha, _, _ = _overlapping_historical_inventory(tmp_path)
    result = freeze_protected_exclusion_denylist(
        inventory_path=inventory,
        inventory_sha256=inventory_sha,
        output_dir=tmp_path / "historical-denylist",
    )
    return result, inventory, inventory_sha


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
    assert len(bundle.identity_sets.source_group_sha256) == 5
    assert len(bundle.identity_sets.parent_source_sha256) == 5
    assert len(bundle.identity_sets.normalized_content_sha256) == 8
    with pytest.raises(AttributeError):
        bundle.identity_sets.source_group_sha256.add("0" * 64)  # type: ignore[attr-defined]


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


def test_one_input_accepts_declared_mixed_real_schemas_and_rejects_an_undeclared_row(
    tmp_path: Path,
) -> None:
    inventory, _, paths, rows = _fixture_inventory(tmp_path)
    corpus = rows["pre-pr"][0]
    natural = _common(
        schema="robustness-natural-pair/v1",
        role="pre_pr_gate",
        index=91,
        text="A real natural typo pair in the mixed evaluation corpus.",
    )
    _write_jsonl(paths["pre-pr"], [corpus, natural])
    payload = json.loads(inventory.read_text())
    input_record = payload["tiers"][3]["inputs"][0]
    input_record["accepted_schemas"] = [
        "robustness-evaluation-corpus-record/v1",
        "robustness-natural-pair/v1",
    ]
    input_record["sha256"] = _digest(paths["pre-pr"])
    inventory.write_bytes(_canonical(payload))
    result = freeze_protected_split_registry(
        inventory_path=inventory,
        inventory_sha256=_digest(inventory),
        output_dir=tmp_path / "accepted",
    )
    run = json.loads(result.run_path.read_text())
    assert run["output"]["tier_record_counts"]["pre-pr"] == 2

    undeclared = _common(
        schema="robustness-fixed-typo-pair/v1",
        role="pre_pr_gate",
        index=92,
        text="An undeclared third schema must fail closed.",
    )
    _write_jsonl(paths["pre-pr"], [corpus, natural, undeclared])
    input_record["sha256"] = _digest(paths["pre-pr"])
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="schema differs from the inventory"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "undeclared",
        )


@pytest.mark.parametrize(
    "schemas",
    (
        [],
        ["robustness-natural-pair/v1", "robustness-natural-pair/v1"],
        ["robustness-natural-pair/v1", "robustness-clean-record/v1"],
    ),
)
def test_inventory_schema_allowlist_must_be_nonempty_unique_and_canonical(
    tmp_path: Path,
    schemas: list[str],
) -> None:
    inventory, _, _, _ = _fixture_inventory(tmp_path)
    payload = json.loads(inventory.read_text())
    payload["tiers"][0]["inputs"][0]["accepted_schemas"] = schemas
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="accepted_schemas"):
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=tmp_path / "output",
        )


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
    output = tmp_path / "output"
    with pytest.raises(ProtectedSplitOverlapError, match="overlap transitively") as exc_info:
        freeze_protected_split_registry(
            inventory_path=_inventory,
            inventory_sha256=inventory_sha,
            output_dir=output,
        )
    error = exc_info.value
    assert len(error.components) == 1
    component = error.components[0]
    assert component.tiers == ("training", "localization", "tune")
    assert {identity.kind for identity in component.identities} == {
        "source_group_sha256",
        "parent_source_sha256",
        "normalized_content_sha256",
    }
    assert [(row.tier, row.line_number) for row in component.occurrences] == [
        ("training", 1),
        ("localization", 1),
        ("tune", 1),
    ]
    assert all(row.source_relative_path.endswith(".jsonl") for row in component.occurrences)
    report = error.audit_report
    assert report["schema_version"] == OVERLAP_AUDIT_SCHEMA
    assert json.loads(error.audit_json) == report
    for row in rows.values():
        for record in row:
            for forbidden_field in ("text", "clean_text", "typo_text"):
                forbidden = record.get(forbidden_field)
                if isinstance(forbidden, str):
                    assert forbidden not in error.audit_json
    assert not output.exists()

    with pytest.raises(ProtectedSplitOverlapError) as repeated:
        freeze_protected_split_registry(
            inventory_path=_inventory,
            inventory_sha256=inventory_sha,
            output_dir=tmp_path / "repeated-output",
        )
    assert repeated.value.audit_json == error.audit_json


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


@pytest.mark.parametrize("attack", ["inventory", "input", "bundle"])
def test_hardlink_substitution_is_rejected(tmp_path: Path, attack: str) -> None:
    inventory, inventory_sha, paths, _ = _fixture_inventory(tmp_path)
    if attack == "inventory":
        alias = tmp_path / "inventory-hardlink.json"
        alias.hardlink_to(inventory)
        with pytest.raises(ValueError, match="hard-linked"):
            freeze_protected_split_registry(
                inventory_path=inventory,
                inventory_sha256=inventory_sha,
                output_dir=tmp_path / "output",
            )
    elif attack == "input":
        alias = tmp_path / "training-hardlink.jsonl"
        alias.hardlink_to(paths["training"])
        with pytest.raises(ValueError, match="hard-linked"):
            freeze_protected_split_registry(
                inventory_path=inventory,
                inventory_sha256=inventory_sha,
                output_dir=tmp_path / "output",
            )
    else:
        result = freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=tmp_path / "output",
        )
        run = json.loads(result.run_path.read_text())
        copied = result.root / run["inputs"][0]["copied_relative_path"]
        alias = tmp_path / "bundle-hardlink.jsonl"
        alias.hardlink_to(copied)
        with pytest.raises(ValueError, match="hard-linked"):
            load_protected_split_registry_bundle(
                result.run_path,
                expected_producer_record_sha256=result.producer_record_sha256,
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
    payload["tiers"][0]["inputs"][0]["accepted_schemas"] = ["unknown/v1"]
    inventory.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="supported"):
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
        (original.encode() + b"\n\n", "blank lines", "blank-line"),
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


def test_atomic_noreplace_publish_preserves_a_race_winner(tmp_path: Path) -> None:
    staging = tmp_path / ".published.staging"
    staging.mkdir()
    (staging / "new").write_text("new")
    target = tmp_path / "published"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("race winner")
    with pytest.raises(FileExistsError, match="appeared before publish"):
        registry_module._publish_directory_noreplace(staging, target)  # noqa: SLF001
    assert sentinel.read_text() == "race winner"
    assert (staging / "new").read_text() == "new"


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


def test_loader_detects_input_toctou_after_capturing_pinned_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _freeze(tmp_path)
    run = json.loads(result.run_path.read_text())
    copied = result.root / run["inputs"][0]["copied_relative_path"]
    original_read = registry_module._read_regular_bytes  # noqa: SLF001
    attacked = False

    def mutate_after_read(path: Path, *, label: str) -> bytes:
        nonlocal attacked
        raw = original_read(path, label=label)
        if path == copied and label == "protected input copy" and not attacked:
            attacked = True
            row = json.loads(raw.decode().strip())
            row["metadata"]["attacker"] = True
            _write_jsonl(copied, [row])
        return raw

    monkeypatch.setattr(registry_module, "_read_regular_bytes", mutate_after_read)
    with pytest.raises(ValueError, match="changed during verification"):
        load_protected_split_registry_bundle(
            result.run_path,
            expected_producer_record_sha256=result.producer_record_sha256,
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
        ("accepted_schemas", ["invented/v1"], "input schemas differ"),
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


def test_exclusion_denylist_accepts_realistic_historical_collisions_only_for_exclusion(
    tmp_path: Path,
) -> None:
    inventory, inventory_sha, _, rows = _overlapping_historical_inventory(tmp_path)
    strict_output = tmp_path / "strict-output"
    with pytest.raises(ProtectedSplitOverlapError) as strict_error:
        freeze_protected_split_registry(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=strict_output,
        )
    assert [component.tiers for component in strict_error.value.components] == [
        ("pre-pr", "sealed"),
        ("tune", "pre-pr"),
    ]
    assert not strict_output.exists()

    result = freeze_protected_exclusion_denylist(
        inventory_path=inventory,
        inventory_sha256=inventory_sha,
        output_dir=tmp_path / "historical-denylist",
    )
    bundle = load_protected_exclusion_denylist_bundle(
        result.run_path,
        expected_producer_record_sha256=result.producer_record_sha256,
    )
    assert isinstance(bundle, ProtectedExclusionDenylistBundle)
    assert not isinstance(bundle, registry_module.ProtectedSplitRegistryBundle)
    assert bundle.purpose == DENYLIST_PURPOSE
    assert bundle.split_certified is False
    assert bundle.input_records == 6
    assert isinstance(bundle.identity_sets.source_group_sha256, frozenset)
    assert isinstance(bundle.identity_sets.parent_source_sha256, frozenset)
    assert isinstance(bundle.identity_sets.normalized_content_sha256, frozenset)
    assert bundle.overlap_audit_sha256 == _digest(result.overlap_audit_path)

    run = json.loads(result.run_path.read_text())
    assert run["schema_version"] == DENYLIST_PRODUCER_SCHEMA
    assert run["purpose"] == DENYLIST_PURPOSE
    assert run["split_certified"] is False
    assert run["outputs"]["overlap_audit"]["sha256"] == bundle.overlap_audit_sha256
    denylist = json.loads(result.denylist_path.read_text())
    assert denylist["schema_version"] == DENYLIST_SCHEMA
    assert denylist["purpose"] == DENYLIST_PURPOSE
    assert denylist["split_certified"] is False
    assert denylist["identity_sets"] == {
        "source_group_sha256": sorted(bundle.identity_sets.source_group_sha256),
        "parent_source_sha256": sorted(bundle.identity_sets.parent_source_sha256),
        "normalized_content_sha256": sorted(bundle.identity_sets.normalized_content_sha256),
    }
    audit = json.loads(result.overlap_audit_path.read_text())
    assert audit["schema_version"] == OVERLAP_AUDIT_SCHEMA
    assert len(audit["collision_components"]) == 2
    assert [component["tiers"] for component in audit["collision_components"]] == [
        ["pre-pr", "sealed"],
        ["tune", "pre-pr"],
    ]
    audit_text = result.overlap_audit_path.read_text()
    for tier_rows in rows.values():
        for row in tier_rows:
            for field in ("text", "clean_text", "typo_text"):
                value = row.get(field)
                if isinstance(value, str):
                    assert value not in audit_text


def test_exclusion_denylist_cannot_masquerade_as_strict_certification(tmp_path: Path) -> None:
    denylist_result, _, _ = _freeze_denylist(tmp_path)
    with pytest.raises(ValueError, match="producer record fields or status differ"):
        load_protected_split_registry_bundle(
            denylist_result.run_path,
            expected_producer_record_sha256=denylist_result.producer_record_sha256,
        )

    strict_result, _ = _freeze(tmp_path / "strict")
    with pytest.raises(ValueError, match="denylist producer record fields differ"):
        load_protected_exclusion_denylist_bundle(
            strict_result.run_path,
            expected_producer_record_sha256=strict_result.producer_record_sha256,
        )


def test_exclusion_denylist_rejects_self_rehash_forgery_and_replays_inputs(
    tmp_path: Path,
) -> None:
    result, _, _ = _freeze_denylist(tmp_path)
    externally_pinned = result.producer_record_sha256
    denylist = json.loads(result.denylist_path.read_text())
    denylist["identity_sets"]["normalized_content_sha256"].append("f" * 64)
    denylist["identity_sets"]["normalized_content_sha256"].sort()
    result.denylist_path.write_bytes(_canonical(denylist))
    run = json.loads(result.run_path.read_text())
    run["outputs"]["denylist"]["sha256"] = _digest(result.denylist_path)
    run["outputs"]["denylist"]["bytes"] = result.denylist_path.stat().st_size
    run["outputs"]["denylist"]["identity_counts"]["normalized_content_sha256"] += 1
    unsigned = dict(run)
    del unsigned["record_sha256"]
    forged_sha = registry_module._canonical_sha256(unsigned)  # noqa: SLF001
    run["record_sha256"] = forged_sha
    result.run_path.write_bytes(_canonical(run))

    with pytest.raises(ValueError, match="external SHA-256"):
        load_protected_exclusion_denylist_bundle(
            result.run_path,
            expected_producer_record_sha256=externally_pinned,
        )
    with pytest.raises(ValueError, match="differs from replayed inputs"):
        load_protected_exclusion_denylist_bundle(
            result.run_path,
            expected_producer_record_sha256=forged_sha,
        )


@pytest.mark.parametrize("substitution", ["symlink", "hardlink"])
def test_exclusion_denylist_rejects_link_substitution(
    tmp_path: Path,
    substitution: str,
) -> None:
    result, _, _ = _freeze_denylist(tmp_path)
    target = result.overlap_audit_path
    outside = tmp_path / "outside-audit.json"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    if substitution == "symlink":
        target.symlink_to(outside)
        message = "symlink"
    else:
        target.hardlink_to(outside)
        message = "hard-linked"
    with pytest.raises(ValueError, match=message):
        load_protected_exclusion_denylist_bundle(
            result.run_path,
            expected_producer_record_sha256=result.producer_record_sha256,
        )


def test_exclusion_denylist_loader_detects_toctou_and_unexpected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = _freeze_denylist(tmp_path)
    original_read = registry_module._read_regular_bytes  # noqa: SLF001
    attacked = False

    def mutate_after_read(path: Path, *, label: str) -> bytes:
        nonlocal attacked
        raw = original_read(path, label=label)
        if (
            path == result.denylist_path
            and label == "protected exclusion denylist"
            and not attacked
        ):
            attacked = True
            path.write_bytes(raw + b" ")
        return raw

    monkeypatch.setattr(registry_module, "_read_regular_bytes", mutate_after_read)
    with pytest.raises(
        ValueError,
        match="output accounting differs|changed during verification",
    ):
        load_protected_exclusion_denylist_bundle(
            result.run_path,
            expected_producer_record_sha256=result.producer_record_sha256,
        )
    assert attacked

    monkeypatch.setattr(registry_module, "_read_regular_bytes", original_read)
    second, _, _ = _freeze_denylist(tmp_path / "unexpected")
    (second.root / "unattested.txt").write_text("not in the closed bundle")
    with pytest.raises(ValueError, match="file inventory differs"):
        load_protected_exclusion_denylist_bundle(
            second.run_path,
            expected_producer_record_sha256=second.producer_record_sha256,
        )


def test_exclusion_denylist_freeze_rehashes_sources_and_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, inventory_sha, paths, _ = _overlapping_historical_inventory(tmp_path)

    def mutate_source() -> None:
        paths["training"].write_bytes(paths["training"].read_bytes() + b" \n")

    monkeypatch.setattr(denylist_module, "_before_denylist_final_rehash", mutate_source)
    output = tmp_path / "mutated-output"
    with pytest.raises(ValueError, match="changed before publication"):
        freeze_protected_exclusion_denylist(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".mutated-output.*"))

    monkeypatch.setattr(denylist_module, "_before_denylist_final_rehash", lambda: None)
    existing = tmp_path / "existing-output"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("do not replace")
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_protected_exclusion_denylist(
            inventory_path=inventory,
            inventory_sha256=_digest(inventory),
            output_dir=existing,
        )
    assert sentinel.read_text() == "do not replace"


def test_exclusion_denylist_freeze_reverifies_staging_after_final_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, inventory_sha, _, _ = _overlapping_historical_inventory(tmp_path)
    output = tmp_path / "reverified-output"

    def mutate_staging() -> None:
        denylist = next(tmp_path.glob(".reverified-output.*")) / "denylist.json"
        denylist.write_bytes(denylist.read_bytes() + b" ")

    monkeypatch.setattr(denylist_module, "_before_denylist_final_rehash", mutate_staging)
    with pytest.raises(ValueError, match="differs from replayed inputs"):
        freeze_protected_exclusion_denylist(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".reverified-output.*"))


def test_exclusion_denylist_atomic_publish_preserves_race_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, inventory_sha, _, _ = _overlapping_historical_inventory(tmp_path)
    output = tmp_path / "race-output"
    original_publish = registry_module._publish_directory_noreplace  # noqa: SLF001

    def install_race_winner(source: Path, target: Path) -> None:
        target.mkdir()
        (target / "sentinel").write_text("race winner")
        original_publish(source, target)

    monkeypatch.setattr(
        registry_module,
        "_publish_directory_noreplace",
        install_race_winner,
    )
    with pytest.raises(FileExistsError, match="appeared before publish"):
        freeze_protected_exclusion_denylist(
            inventory_path=inventory,
            inventory_sha256=inventory_sha,
            output_dir=output,
        )
    assert (output / "sentinel").read_text() == "race winner"
    assert not list(tmp_path.glob(".race-output.*"))


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
    exclusion = parser.parse_args(
        [
            "freeze-protected-exclusion-denylist",
            "--inventory",
            str(tmp_path / "historical-inventory.json"),
            "--inventory-sha256",
            "c" * 64,
            "--output-dir",
            str(tmp_path / "denylist-output"),
        ]
    )
    assert exclusion.command == "freeze-protected-exclusion-denylist"
    exclusion_verify = parser.parse_args(
        [
            "verify-protected-exclusion-denylist",
            "--producer-run",
            str(tmp_path / "denylist-run.json"),
            "--producer-record-sha256",
            "d" * 64,
        ]
    )
    assert exclusion_verify.command == "verify-protected-exclusion-denylist"


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
