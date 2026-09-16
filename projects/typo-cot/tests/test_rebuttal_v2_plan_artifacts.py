"""Artifact-boundary tests for frozen rebuttal-v2 global plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typo_cot.experiments.rebuttal_v2 import plan_artifacts
from typo_cot.experiments.rebuttal_v2.identity import identity_hash
from typo_cot.experiments.rebuttal_v2.planning import GENERATION_ID_DOMAIN, GlobalPlan
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "configs/rebuttal_v2/protocol.json"


def _write(path: Path, raw: bytes = b"fixture\n") -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path.resolve(), hashlib.sha256(raw).hexdigest()


def _ref(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def _bundles(tmp_path: Path):
    paths = {}
    for name in (
        "pair_manifest.jsonl",
        "pair_manifest.meta.json",
        "input_audit_records.jsonl",
        "input_audit_records.meta.json",
        "semantic_labels.jsonl",
        "labels-run.json",
        "runtime_lock.json",
    ):
        path, digest = _write(tmp_path / "upstream" / name)
        paths[name] = (path, digest)
    manifest = SimpleNamespace(
        path=paths["pair_manifest.jsonl"][0],
        metadata_path=paths["pair_manifest.meta.json"][0],
        manifest_ref=_ref(*paths["pair_manifest.jsonl"]),
        metadata_ref=_ref(*paths["pair_manifest.meta.json"]),
        snapshots={
            paths["pair_manifest.jsonl"][0]: paths["pair_manifest.jsonl"][1],
            paths["pair_manifest.meta.json"][0]: paths["pair_manifest.meta.json"][1],
        },
    )
    audit = SimpleNamespace(
        path=paths["input_audit_records.jsonl"][0],
        metadata_path=paths["input_audit_records.meta.json"][0],
        audit_ref=_ref(*paths["input_audit_records.jsonl"]),
        metadata_ref=_ref(*paths["input_audit_records.meta.json"]),
        manifest=manifest,
        snapshots={
            paths["input_audit_records.jsonl"][0]: paths["input_audit_records.jsonl"][1],
            paths["input_audit_records.meta.json"][0]: paths["input_audit_records.meta.json"][1],
        },
    )
    labels = SimpleNamespace(
        path=paths["semantic_labels.jsonl"][0],
        run_path=paths["labels-run.json"][0],
        reference=_ref(*paths["semantic_labels.jsonl"]),
        run_ref=_ref(*paths["labels-run.json"]),
        snapshots={
            paths["semantic_labels.jsonl"][0]: paths["semantic_labels.jsonl"][1],
            paths["labels-run.json"][0]: paths["labels-run.json"][1],
        },
    )
    runtime = SimpleNamespace(
        path=paths["runtime_lock.json"][0],
        reference=_ref(*paths["runtime_lock.json"]),
        snapshots={paths["runtime_lock.json"][0]: paths["runtime_lock.json"][1]},
    )
    snapshots = {}
    for bundle in (manifest, audit, labels, runtime):
        snapshots.update(bundle.snapshots)
    return manifest, audit, labels, runtime, snapshots


def _row(plan_id: str, *, valid: bool, arm: str) -> dict:
    return {
        "schema_version": "rebuttal-plan/v2",
        "plan_id": plan_id,
        "pair_id": "a" * 64,
        "original_problem_group_id": "group-a",
        "setting": "R3",
        "window": None if arm in {"clean", "typo"} else [0, 6],
        "arm": arm,
        "eligibility": "valid" if valid else "not_available",
        "reason_codes": [] if valid else ["donor_bank_not_available"],
    }


def _global_plan(*, blocked: bool = False, zero: bool = False) -> GlobalPlan:
    if blocked:
        return GlobalPlan(
            rows=[],
            preflight={
                "schema_version": "rebuttal-plan-preflight/v2",
                "status": "blocked",
                "blockers": [
                    {
                        "pair_id": "a" * 64,
                        "setting": "R3",
                        "reason_codes": ["original_problem_group_id_unknown"],
                    },
                    {
                        "pair_id": "b" * 64,
                        "setting": "R3",
                        "reason_codes": ["original_problem_group_id_unknown"],
                    },
                ],
                "blocking_pair_ids": ["a" * 64, "b" * 64],
                "experiments": ["R3"],
            },
            coverage={
                "schema_version": "rebuttal-plan-coverage/v2",
                "counts": {"intended_plan_row_count": 12},
            },
            donor_assignment=None,
            scientific_config_sha256=None,
            expected_generation_ids=[],
        )
    scientific = "9" * 64
    rows = (
        []
        if zero
        else [_row("1" * 64, valid=True, arm="clean"), _row("2" * 64, valid=False, arm="cross")]
    )
    expected = sorted(
        identity_hash(
            GENERATION_ID_DOMAIN,
            {"plan_id": row["plan_id"], "scientific_config_sha256": scientific},
        )
        for row in rows
        if row["eligibility"] == "valid"
    )
    return GlobalPlan(
        rows=rows,
        preflight={
            "schema_version": "rebuttal-plan-preflight/v2",
            "status": "passed",
            "blockers": [],
            "blocking_pair_ids": [],
            "experiments": ["R3"],
        },
        coverage={
            "schema_version": "rebuttal-plan-coverage/v2",
            "counts": {"intended_plan_row_count": len(rows)},
        },
        donor_assignment={
            "schema_version": "rebuttal-donor-assignment/v2",
            "assignments": [],
            "unused_donor_pair_ids": [],
            "donor_bank_available": False,
        },
        scientific_config_sha256=scientific,
        expected_generation_ids=expected,
    )


def _patch_run(monkeypatch, tmp_path: Path, plan: GlobalPlan):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    manifest, audit, labels, runtime, snapshots = _bundles(tmp_path)
    monkeypatch.setattr(
        plan_artifacts,
        "_load_inputs",
        lambda *args, **kwargs: (
            protocol,
            protocol_bytes,
            manifest,
            audit,
            labels,
            runtime,
            None,
            dict(snapshots),
        ),
    )
    monkeypatch.setattr(plan_artifacts, "build_run_plan", lambda *args, **kwargs: plan)
    return protocol, manifest, audit, labels, runtime


def _run(tmp_path: Path, monkeypatch, plan: GlobalPlan, name: str, shards: int = 1):
    _patch_run(monkeypatch, tmp_path / name, plan)
    return plan_artifacts.run_plan(
        "manifest",
        "audit",
        "labels",
        "runtime",
        PROTOCOL_PATH,
        tmp_path / name / "out",
        experiments=["R3"],
        num_shards=shards,
    )


def test_complete_plan_keeps_ineligible_rows_and_null_bank(tmp_path: Path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, _global_plan(), "complete")
    output = Path(result["output_dir"])

    assert result["status"] == "complete"
    assert result["plan_count"] == 2
    assert result["expected_generation_count"] == 1
    assert {path.name for path in output.iterdir()} == {
        "plan.jsonl",
        "plan.meta.json",
        "preflight.json",
        "planning_coverage.json",
        "donor_assignment.json",
        "expected_generations.jsonl",
        "shard_assignment.json",
        "protocol.json",
        "run.json",
    }
    metadata = json.loads((output / "plan.meta.json").read_text(encoding="utf-8"))
    assert metadata["donor_bank_ref"] is None
    assert metadata["donor_bank_sha256"] is None
    assert metadata["ineligible_plan_rows"] == [
        {
            "plan_id": "2" * 64,
            "eligibility": "not_available",
            "reason_codes": ["donor_bank_not_available"],
        }
    ]


def test_shard_count_changes_only_separate_assignment(tmp_path: Path, monkeypatch) -> None:
    first = _run(tmp_path, monkeypatch, _global_plan(), "one", shards=1)
    second = _run(tmp_path, monkeypatch, _global_plan(), "six", shards=6)
    one, six = Path(first["output_dir"]), Path(second["output_dir"])

    for name in ("plan.jsonl", "donor_assignment.json", "expected_generations.jsonl"):
        assert (one / name).read_bytes() == (six / name).read_bytes()
    assert (one / "shard_assignment.json").read_bytes() != (
        six / "shard_assignment.json"
    ).read_bytes()


def test_blocked_plan_publishes_only_diagnostics_and_all_pair_ids(
    tmp_path: Path, monkeypatch
) -> None:
    result = _run(tmp_path, monkeypatch, _global_plan(blocked=True), "blocked")
    output = Path(result["output_dir"])

    assert result["status"] == "blocked"
    assert result["blocking_pair_ids"] == ["a" * 64, "b" * 64]
    assert {path.name for path in output.iterdir()} == {
        "preflight.json",
        "planning_coverage.json",
        "protocol.json",
        "run.json",
    }
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["planning_status"] == "blocked"
    assert run["run_status"] == "failed"
    assert run["expected_ids"] == run["completed_ids"] == []


def test_zero_row_plan_still_binds_every_scientific_input(tmp_path: Path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, _global_plan(zero=True), "zero")
    metadata = json.loads(
        (Path(result["output_dir"]) / "plan.meta.json").read_text(encoding="utf-8")
    )
    assert result["plan_count"] == 0
    assert metadata["manifest_ref"]["path"].endswith("pair_manifest.jsonl")
    assert metadata["input_audit_ref"]["path"].endswith("input_audit_records.jsonl")
    assert metadata["runtime_lock_ref"]["path"].endswith("runtime_lock.json")


def test_no_overwrite_and_structural_failure_publish_nothing(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep").write_text("owned", encoding="utf-8")
    with pytest.raises(IntakeError, match="absent or empty"):
        plan_artifacts.run_plan(
            "missing",
            "missing",
            "missing",
            "missing",
            PROTOCOL_PATH,
            output,
            experiments=["R3"],
        )
    assert (output / "keep").read_text(encoding="utf-8") == "owned"

    fresh = tmp_path / "fresh"
    with pytest.raises(IntakeError):
        plan_artifacts.run_plan(
            "missing",
            "missing",
            "missing",
            "missing",
            tmp_path / "missing-protocol.json",
            fresh,
            experiments=["R3"],
        )
    assert not fresh.exists()


def test_protocol_is_parsed_from_the_same_captured_bytes(tmp_path: Path, monkeypatch) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(PROTOCOL_PATH.read_bytes())
    original_capture = plan_artifacts.artifact_io.capture_file
    changed = False

    def change_before_capture(path, snapshots, expected=None):
        nonlocal changed
        if Path(path).resolve() == protocol.resolve() and not changed:
            changed = True
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["protocol_revision"] = "changed-after-first-read"
            protocol.write_text(json.dumps(payload), encoding="utf-8")
        return original_capture(path, snapshots, expected)

    monkeypatch.setattr(plan_artifacts.artifact_io, "capture_file", change_before_capture)
    output = tmp_path / "output"
    with pytest.raises(IntakeError, match="protocol_revision"):
        plan_artifacts.run_plan(
            "unused",
            "unused",
            "unused",
            "unused",
            protocol,
            output,
            experiments=["R3"],
        )
    assert not output.exists()


def _patch_load(monkeypatch, result: dict, fixtures) -> None:
    _, manifest, audit, labels, runtime = fixtures
    monkeypatch.setattr(plan_artifacts, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(plan_artifacts, "load_input_audit", lambda path: audit)
    monkeypatch.setattr(plan_artifacts, "load_semantic_labels", lambda path, bundle: labels)
    monkeypatch.setattr(plan_artifacts, "load_runtime_lock", lambda *args: runtime)
    monkeypatch.setattr(plan_artifacts, "build_run_plan", lambda *args, **kwargs: result)


def _rehash(metadata_path: Path, field: str, target: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_load_plan_reconstructs_global_inventory_and_rejects_rehashed_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    core = _global_plan()
    fixtures = _patch_run(monkeypatch, tmp_path / "load", core)
    result = plan_artifacts.run_plan(
        "manifest",
        "audit",
        "labels",
        "runtime",
        PROTOCOL_PATH,
        tmp_path / "load/out",
        experiments=["R3"],
    )
    _patch_load(monkeypatch, core, fixtures)
    path = Path(result["plan_path"])
    bundle = plan_artifacts.load_plan(path)
    assert bundle.rows == core.rows
    assert bundle.generation_rows[0]["generation_id"] == core.expected_generation_ids[0]

    path.write_text(json.dumps({**core.rows[0], "arm": "typo"}) + "\n", encoding="utf-8")
    _rehash(path.with_name("plan.meta.json"), "plan_ref", path)
    with pytest.raises(IntakeError, match="plan rows differs"):
        plan_artifacts.load_plan(path)


@pytest.mark.parametrize("target", ["expected", "metadata"])
def test_load_plan_rejects_rehashed_expected_grid_and_metadata_inventory(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    core = _global_plan()
    fixtures = _patch_run(monkeypatch, tmp_path / target, core)
    result = plan_artifacts.run_plan(
        "manifest",
        "audit",
        "labels",
        "runtime",
        PROTOCOL_PATH,
        tmp_path / target / "out",
        experiments=["R3"],
    )
    _patch_load(monkeypatch, core, fixtures)
    plan_path = Path(result["plan_path"])
    metadata_path = plan_path.with_name("plan.meta.json")
    if target == "expected":
        expected_path = plan_path.with_name("expected_generations.jsonl")
        row = json.loads(expected_path.read_text(encoding="utf-8"))
        row["arm"] = "typo"
        expected_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        _rehash(metadata_path, "expected_generations_ref", expected_path)
        message = "expected generation inventory differs"
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["row_counts"][0]["count"] += 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        message = "row counts differ"
    with pytest.raises(IntakeError, match=message):
        plan_artifacts.load_plan(plan_path)


def test_cli_help_is_cpu_only_and_r5_is_rejected() -> None:
    import subprocess
    import sys

    help_result = subprocess.run(
        [sys.executable, "-m", "typo_cot.cli", "rebuttal-v2", "plan", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--runtime-lock" in help_result.stdout
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "typo_cot.cli",
            "rebuttal-v2",
            "plan",
            "--manifest",
            "m",
            "--input-audit",
            "a",
            "--labels",
            "l",
            "--runtime-lock",
            "r",
            "--protocol",
            "p",
            "--experiments",
            "R5",
            "--output-dir",
            "o",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
