"""Independent public-adapter -> intake -> R1 CLI acceptance fixtures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from typo_cot.cli import main
from typo_cot.experiments.rebuttal_v2.source import run_intake


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "configs/rebuttal_v2/protocol.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixture(root: Path, *, labels_known: bool = True, gold_known: bool = True) -> Path:
    slots = [{"arm": arm, "window": None} for arm in ("clean", "typo")]
    slots += [
        {"arm": arm, "window": window}
        for arm in ("self", "correct", "offset", "cross")
        for window in ([0, 6], [6, 12])
    ]
    pair = {
        "schema_version": "rebuttal-source-pair/v2",
        "source_record_id": "pair-record",
        "source_pair_key": "original-1",
        "task": "mmlu-pro",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "dataset_id": "mmlu-pro",
        "split": "test",
        "original_problem_id": "question-1",
        "cohort_ids": ["R4"],
        "clean_text": "Choose the correct option.",
        "typo_text": "Choose the corrcet option.",
        "valid_labels": list("ABCDEFGHIJ") if labels_known else None,
        "gold": "I" if gold_known else None,
        "canonical_gold": "I" if gold_known else None,
        "gold_canonicalization_version": "fixture-label/v1" if gold_known else None,
        "expected_generations": slots,
    }
    generation_rows = []
    for index, slot in enumerate(slots):
        # One expected generation is deliberately absent, never replaced.
        if slot == {"arm": "cross", "window": [6, 12]}:
            continue
        generation_rows.append(
            {
                "schema_version": "rebuttal-source-generation/v2",
                "source_record_id": f"generation-{index}",
                "source_generation_key": f"generation-{index}",
                "source_pair_key": "original-1",
                **slot,
                "raw_text": (
                    "The answer is incorrect" if slot["arm"] == "typo" else "The answer is I."
                ),
                "stopping_metadata": {"eos_observed": True, "length_cap_reached": True},
            }
        )
    sources = []
    for role, records, adapter in (
        ("pairs", [pair], "rebuttal-source-pair/v2"),
        ("generations", generation_rows, "rebuttal-source-generation/v2"),
    ):
        path = root / f"{role}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        sources.append(
            {
                "source_id": role,
                "source_kind": "submitted-archive",
                "role": role,
                "path": path.name,
                "expected_sha256": _sha(path),
                "format": "jsonl",
                "adapter_version": adapter,
                "source_commit": None,
                "model_revision": None,
                "tokenizer_revision": None,
                "prompt_template_revision": None,
                "generation_config": None,
                "availability": "available",
                "reason_codes": [],
            }
        )
    ids = root / "ids.json"
    ids.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-cohort-ids/v2",
                "cohort_id": "R4",
                "source_namespace": "r1-independent-fixture",
                "source_pair_keys": ["original-1", "missing-original"],
            }
        ),
        encoding="utf-8",
    )
    index = root / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-archive-index/v2",
                "created_at": "2026-09-16T00:00:00Z",
                "source_namespace": "r1-independent-fixture",
                "sources": sources,
                "cohorts": [
                    {
                        "cohort_id": "R4",
                        "setting": "R4",
                        "historical_n_reference": 97,
                        "expected_ids_ref": {"path": ids.name, "sha256": _sha(ids)},
                        "id_namespace": "r1-independent-fixture",
                        "membership_provenance_ref": None,
                    }
                ],
                "identity_aliases_ref": None,
                "notes": "Synthetic acceptance fixture, not paper results.",
            }
        ),
        encoding="utf-8",
    )
    run_intake(index, PROTOCOL, root / "intake")
    return root / "intake/pair_manifest.jsonl"


def _run(manifest: Path, output: Path) -> int:
    return main(
        ["rebuttal-v2", "answer-audit", "--manifest", str(manifest), "--output-dir", str(output)]
    )


def test_r4_cli_preserves_shared_baselines_and_missing_generations(tmp_path, capsys):
    manifest = _fixture(tmp_path)
    before = {path: _sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "audit"
    assert _run(manifest, output) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["run_status"] == "complete"
    assert run["model_execution_performed"] is False
    assert all(_sha(path) == digest for path, digest in before.items())
    audits = _rows(output / "answer_audit_records.jsonl")
    coverage = _rows(output / "scoring_coverage.jsonl")
    assert len(audits) == 18
    assert len(coverage) == 20
    assert sum(row["scoring_status"] == "output_missing" for row in coverage) == 2
    assert {row["termination"] for row in audits} == {"eos"}
    assert len([row for row in audits if row["arm"] == "clean"]) == 2
    typo = {row["parser_id"]: row for row in audits if row["arm"] == "typo"}
    assert typo["explicit_answer_v2"]["extraction_status"] == "unextractable"
    assert typo["explicit_answer_v2"]["gold_match"] is False
    assert typo["public_v1_proxy"]["raw_value"] == "I"
    assert typo["public_v1_proxy"]["gold_match"] is True
    baseline = _json(output / "baseline_transition_table.json")
    assert len(baseline) == 1
    assert baseline[0]["primary_state"] == "eligible"
    assert baseline[0]["proxy_state"] == "ineligible"
    assert baseline[0]["comparison_kind"] == "archived-baselines-not-fresh"
    summary = _json(output / "parser_audit_summary.json")
    assert "missing-original" in json.dumps(summary["source_cohort_coverage"])
    assert summary["parser_independent_semantic_claim"] == "inconclusive"
    blind = _rows(output / "parser_disagreement_blind.jsonl")
    assert blind
    assert all(set(row) == {"schema_version", "blind_id", "raw_text"} for row in blind)
    for name, ref in run["output_refs"].items():
        assert ref["path"] == name
        assert _sha(output / name) == ref["sha256"]
    assert _run(manifest, output) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("labels_known,gold_known", [(False, True), (True, False)])
def test_availability_never_invents_a_model_failure(tmp_path, capsys, labels_known, gold_known):
    manifest = _fixture(tmp_path, labels_known=labels_known, gold_known=gold_known)
    output = tmp_path / "audit"
    assert _run(manifest, output) == 0
    capsys.readouterr()
    audits = _rows(output / "answer_audit_records.jsonl")
    primary = [row for row in audits if row["parser_id"] == "explicit_answer_v2"]
    if not labels_known:
        assert primary == []
        coverage = _rows(output / "scoring_coverage.jsonl")
        assert sum(row["scoring_status"] == "parser_unavailable" for row in coverage) == 9
        assert all(row["score_status"] == "scored" for row in audits)
    else:
        assert len(primary) == 9
        assert all(row["score_status"] == "not_scored" for row in audits)
        assert all(row["gold_match"] is None for row in audits)
        assert any(row["extraction_status"] == "unextractable" for row in primary)


def test_answer_audit_help_and_execution_are_cpu_only(tmp_path):
    manifest = _fixture(tmp_path)
    script = (
        "import sys; from typo_cot.cli import main; "
        f"result=main(['rebuttal-v2','answer-audit','--manifest',{str(manifest)!r},"
        f"'--output-dir',{str(tmp_path / 'audit')!r}]); "
        "assert result == 0; assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from typo_cot.cli import main; main(['rebuttal-v2','answer-audit','--help'])",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
