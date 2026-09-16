"""Public intake-to-global-plan acceptance chain for rebuttal v2."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typo_cot.experiments.rebuttal_v2.plan_artifacts import load_plan, run_plan
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "configs/rebuttal_v2/protocol.json"


def _test_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEMANTIC = _test_module("test_rebuttal_v2_semantic_labels.py", "_plan_semantic_pipeline")
_PLANNING = _test_module("test_rebuttal_v2_planning.py", "_plan_runtime_fixture")


def _runtime_lock(root: Path, audit) -> Path:
    runtime = _PLANNING._runtime(root, audit, ["R3"])
    settings = runtime.settings
    for entry in settings.values():
        for ref in entry["tokenizer_policy"]["tokenizer_files"].values():
            matches = [path for path, digest in audit.snapshots.items() if digest == ref["sha256"]]
            assert len(matches) == 1
            ref["path"] = str(matches[0])
    runtime.path.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-runtime-lock/v2",
                "settings": settings,
                "provenance": {"fixture": "public-pipeline"},
            }
        ),
        encoding="utf-8",
    )
    return runtime.path


def _case(root: Path):
    semantic = _SEMANTIC._pipeline(root / "pipeline")
    runtime = _runtime_lock(root / "runtime", semantic.audit)
    return semantic, runtime


def test_real_public_pipeline_freezes_and_reloads_missing_bank_plan(tmp_path: Path) -> None:
    case, runtime = _case(tmp_path)
    output = tmp_path / "plan"

    result = run_plan(
        case.audit.manifest.path,
        case.audit.path,
        case.semantic,
        runtime,
        PROTOCOL,
        output,
        experiments=["R3"],
        donor_bank_path=None,
        num_shards=6,
    )
    bundle = load_plan(output / "plan.jsonl")

    assert result["status"] == "complete"
    assert len(bundle.rows) == 6
    assert len(bundle.expected_generations) == sum(
        row["eligibility"] == "valid" for row in bundle.rows
    )
    assert bundle.metadata["donor_bank_ref"] is None
    assert bundle.metadata["donor_bank_sha256"] is None
    cross = next(row for row in bundle.rows if row["arm"] == "cross")
    assert cross["eligibility"] == "not_available"
    assert cross["donor_bank_sha256"] is None
    assert bundle.shard_assignment["num_shards"] == 6


def test_real_plan_rejects_changed_upstream_and_structural_runtime_without_output(
    tmp_path: Path,
) -> None:
    case, runtime = _case(tmp_path)
    output = tmp_path / "plan"
    run_plan(
        case.audit.manifest.path,
        case.audit.path,
        case.semantic,
        runtime,
        PROTOCOL,
        output,
        experiments=["R3"],
    )
    manifest = case.audit.manifest.path
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n")
    with pytest.raises(IntakeError, match="SHA-256 mismatch|changed"):
        load_plan(output / "plan.jsonl")
    manifest.write_bytes(original)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    payload["settings"]["R3"]["cache_behavior"]["use_cache"] = 1
    runtime.write_text(json.dumps(payload), encoding="utf-8")
    failed = tmp_path / "structural-failure"
    with pytest.raises(IntakeError):
        run_plan(
            case.audit.manifest.path,
            case.audit.path,
            case.semantic,
            runtime,
            PROTOCOL,
            failed,
            experiments=["R3"],
        )
    assert not failed.exists()
