"""Evaluation data stay hash-bound, role-sealed, and explicitly stratified."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import typo_robust_training.evaluation.data as evaluation_data_module
from typo_robust_training.evaluation.data import (
    complete_evaluation_role,
    load_evaluation_bundle,
)


MODEL = "google/gemma-3-4b-it"
REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"


def _synthetic(
    index: int,
    *,
    source: str,
    task: str | None,
    answer: str | None,
    operation: str,
    split: str,
) -> dict[str, object]:
    clean = "The airport is open."
    typo = (
        "The arport is open." if operation != "adjacent-transposition" else "The ai rport is open."
    )
    if operation == "adjacent-transposition":
        typo = "The ariport is open."
        typo_word = "ariport"
    else:
        typo_word = "arport"
    return {
        "schema_version": "robustness-fixed-typo-pair/v1",
        "kind": "synthetic",
        "record_id": f"{index:064x}",
        "source": source,
        "source_revision": "a" * 40,
        "source_split": "test",
        "source_id": f"{source}-{index}",
        "group_id": f"{source}-{index}",
        "split": split,
        "clean_text": clean,
        "typo_text": typo,
        "task": task,
        "answer": answer,
        "metadata": {},
        "operation": operation,
        "operations": [operation],
        "edit_count": 1,
        "generator_seed": 42,
        "generator_variant": index,
        "edits": [
            {
                "operation": operation,
                "clean_word": "airport",
                "typo_word": typo_word,
                "clean_char_span": [4, 11],
                "typo_char_span": [4, 4 + len(typo_word)],
            }
        ],
    }


def _natural(index: int, *, split: str) -> dict[str, object]:
    clean = "The airport is open."
    typo = "The arport is open."
    return {
        "schema_version": "robustness-natural-pair/v1",
        "kind": "natural",
        "record_id": f"{index:064x}",
        "source": "github_typo_corpus",
        "source_revision": "b" * 64,
        "source_split": "corpus",
        "source_id": f"commit-{index}",
        "group_id": "https://example.test/held-out.git",
        "split": split,
        "clean_text": clean,
        "typo_text": typo,
        "task": None,
        "answer": None,
        "operation": "deletion",
        "training_eligible": True,
        "repository": "https://example.test/held-out.git",
        "repository_license": "mit",
        "clean_sha256": hashlib.sha256(clean.encode()).hexdigest(),
        "typo_sha256": hashlib.sha256(typo.encode()).hexdigest(),
        "metadata": {},
    }


def _write_data(root: Path) -> None:
    root.mkdir()
    role_rows = {
        "tune_manifest.jsonl": [
            _synthetic(
                1,
                source="gsm8k",
                task="gsm8k",
                answer="12",
                operation="deletion",
                split="tune",
            )
        ],
        "pre_pr_gate_manifest.jsonl": [
            _synthetic(
                2,
                source="gsm8k",
                task="gsm8k",
                answer="12",
                operation="deletion",
                split="pre_pr_gate",
            ),
            _synthetic(
                3,
                source="mmlu_pro",
                task="mmlu_pro",
                answer="B",
                operation="adjacent-transposition",
                split="pre_pr_gate",
            ),
            _natural(4, split="pre_pr_gate"),
            _synthetic(
                6,
                source="fineweb_edu",
                task=None,
                answer=None,
                operation="deletion",
                split="pre_pr_gate",
            ),
        ],
        "final_test_manifest.jsonl": [
            _synthetic(
                5,
                source="commonsense_qa",
                task="commonsense_qa",
                answer="C",
                operation="adjacent-transposition",
                split="final_test",
            ),
            _synthetic(
                7,
                source="gsm8k",
                task="gsm8k",
                answer="12",
                operation="deletion",
                split="final_test",
            ),
            _synthetic(
                8,
                source="fineweb_edu",
                task=None,
                answer=None,
                operation="deletion",
                split="final_test",
            ),
            _natural(9, split="final_test"),
        ],
    }
    hashes: dict[str, str] = {}
    for name, rows in role_rows.items():
        path = root / name
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evaluation = {
        "schema_version": "robustness-evaluation-manifest/v1",
        "protocol_sha256": "c" * 64,
        "source_revisions": {},
        "split_roles": {
            "tune": "iteration-only",
            "pre_pr_gate": "one-use-before-training-pr",
            "final_test": "sealed-until-method-freeze",
        },
        "training_operations": ["deletion"],
        "operation_probabilities": {"deletion": 1.0},
        "held_out_operations": ["adjacent-transposition"],
        "pre_pr_gate_consumed": False,
        "final_test_opened": False,
        "artifact_sha256": hashes,
        "data_identity_sha256": "d" * 64,
    }
    evaluation_path = root / "evaluation_manifest.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    hashes["evaluation_manifest.json"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    run = {
        "schema_version": "build-robustness-training-data-run/v1",
        "status": "completed",
        "protocol": {
            "model": MODEL,
            "model_revision": REVISION,
            "typos": {"held_out_operations": ["adjacent-transposition"]},
        },
        "outputs": {name: {"sha256": digest, "records": 1} for name, digest in hashes.items()},
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")


def test_loader_filters_union_of_requested_strata_and_infers_natural_edit(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_data(data)
    output = tmp_path / "evaluation"
    bundle = load_evaluation_bundle(
        data,
        evaluation_role="pre-pr-gate",
        splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
        model=MODEL,
        model_revision=REVISION,
        access_binding_sha256="e" * 64,
        experiment_binding_sha256="9" * 64,
        output_dir=output,
        confirm_sealed_role=True,
        resume=False,
    )

    assert [record.record_id for record in bundle.records] == [
        f"{index:064x}" for index in (2, 3, 4, 6)
    ]
    assert bundle.records[0].strata == ("same-task",)
    assert bundle.records[1].strata == ("unseen-task", "unseen-typo")
    assert bundle.records[2].strata == ("unseen-typo",)
    assert bundle.records[2].edits[0].clean_word == "airport"
    assert bundle.records[2].edits[0].typo_word == "arport"
    assert bundle.records[3].strata == ("unseen-content",)
    assert (
        bundle.manifest_sha256
        == hashlib.sha256((data / "pre_pr_gate_manifest.jsonl").read_bytes()).hexdigest()
    )
    access = json.loads((data / "evaluation_access.json").read_text(encoding="utf-8"))
    assert access["roles"]["pre_pr_gate"]["status"] == "opened"
    assert access["roles"]["pre_pr_gate"]["output_dir"] == str(output.resolve())
    assert access["roles"]["pre_pr_gate"]["experiment_binding_sha256"] == "9" * 64


def test_sealed_roles_require_confirmation_exact_resume_and_passing_pre_pr_gate(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_data(data)
    with pytest.raises(ValueError, match="complete frozen split inventory"):
        load_evaluation_bundle(
            data,
            evaluation_role="pre-pr-gate",
            splits=("same-task",),
            model=MODEL,
            model_revision=REVISION,
            access_binding_sha256="e" * 64,
            experiment_binding_sha256="9" * 64,
            output_dir=tmp_path / "partial-gate",
            confirm_sealed_role=True,
            resume=False,
        )
    arguments = {
        "evaluation_role": "pre-pr-gate",
        "splits": ("same-task", "unseen-task", "unseen-content", "unseen-typo"),
        "model": MODEL,
        "model_revision": REVISION,
        "access_binding_sha256": "e" * 64,
        "experiment_binding_sha256": "9" * 64,
        "output_dir": tmp_path / "gate",
    }
    with pytest.raises(ValueError, match="confirmation"):
        load_evaluation_bundle(
            data,
            **arguments,
            confirm_sealed_role=False,
            resume=False,
        )
    bundle = load_evaluation_bundle(
        data,
        **arguments,
        confirm_sealed_role=True,
        resume=False,
    )
    with pytest.raises(ValueError, match="already opened"):
        load_evaluation_bundle(
            data,
            **arguments,
            confirm_sealed_role=True,
            resume=False,
        )
    resumed = load_evaluation_bundle(
        data,
        **arguments,
        confirm_sealed_role=True,
        resume=True,
    )
    assert resumed.records == bundle.records

    complete_evaluation_role(
        data,
        evaluation_role="pre-pr-gate",
        access_binding_sha256="e" * 64,
        report_sha256="f" * 64,
        gate_passed=False,
    )
    with pytest.raises(ValueError, match="already completed"):
        load_evaluation_bundle(
            data,
            **arguments,
            confirm_sealed_role=True,
            resume=True,
        )
    with pytest.raises(ValueError, match="opened role"):
        complete_evaluation_role(
            data,
            evaluation_role="pre-pr-gate",
            access_binding_sha256="e" * 64,
            report_sha256="1" * 64,
            gate_passed=True,
        )
    with pytest.raises(ValueError, match="passing pre-PR"):
        load_evaluation_bundle(
            data,
            evaluation_role="final-test",
            splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
            model=MODEL,
            model_revision=REVISION,
            access_binding_sha256="1" * 64,
            experiment_binding_sha256="9" * 64,
            output_dir=tmp_path / "final",
            confirm_sealed_role=True,
            resume=False,
        )

    passing_data = tmp_path / "passing-data"
    _write_data(passing_data)
    passing_arguments = {
        **arguments,
        "output_dir": tmp_path / "passing-gate",
    }
    load_evaluation_bundle(
        passing_data,
        **passing_arguments,
        confirm_sealed_role=True,
        resume=False,
    )
    complete_evaluation_role(
        passing_data,
        evaluation_role="pre-pr-gate",
        access_binding_sha256="e" * 64,
        report_sha256="f" * 64,
        gate_passed=True,
    )
    with pytest.raises(ValueError, match="candidate differs"):
        load_evaluation_bundle(
            passing_data,
            evaluation_role="final-test",
            splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
            model=MODEL,
            model_revision=REVISION,
            access_binding_sha256="1" * 64,
            experiment_binding_sha256="8" * 64,
            output_dir=tmp_path / "wrong-final",
            confirm_sealed_role=True,
            resume=False,
        )
    final = load_evaluation_bundle(
        passing_data,
        evaluation_role="final-test",
        splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
        model=MODEL,
        model_revision=REVISION,
        access_binding_sha256="1" * 64,
        experiment_binding_sha256="9" * 64,
        output_dir=tmp_path / "final",
        confirm_sealed_role=True,
        resume=False,
    )
    assert len(final.records) == 4


def test_loader_rejects_role_hash_tampering(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_data(data)
    with (data / "tune_manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="hash differs"):
        load_evaluation_bundle(
            data,
            evaluation_role="tune",
            splits=("same-task",),
            model=MODEL,
            model_revision=REVISION,
            access_binding_sha256="e" * 64,
            experiment_binding_sha256="9" * 64,
            output_dir=tmp_path / "tune",
            confirm_sealed_role=False,
            resume=False,
        )


def test_loader_fails_closed_when_artifact_and_declared_hash_are_missing(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_data(data)
    (data / "tune_manifest.jsonl").unlink()
    run_path = data / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["outputs"]["tune_manifest.jsonl"]["sha256"] = None
    run_path.write_text(json.dumps(run), encoding="utf-8")
    evaluation_path = data / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["artifact_sha256"]["tune_manifest.jsonl"] = None
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(ValueError, match="hash input must be one regular file"):
        load_evaluation_bundle(
            data,
            evaluation_role="tune",
            splits=("same-task",),
            model=MODEL,
            model_revision=REVISION,
            access_binding_sha256="e" * 64,
            experiment_binding_sha256="9" * 64,
            output_dir=tmp_path / "tune",
            confirm_sealed_role=False,
            resume=False,
        )


def test_sealed_role_claim_is_serialized_across_concurrent_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    _write_data(data)
    barrier = threading.Barrier(2)
    original_registry = evaluation_data_module._access_registry

    def delayed_registry(path: Path) -> dict[str, object]:
        time.sleep(0.05)
        return original_registry(path)

    monkeypatch.setattr(evaluation_data_module, "_access_registry", delayed_registry)

    def claim(index: int) -> str:
        barrier.wait()
        try:
            evaluation_data_module._claim_evaluation_role(
                data,
                role="pre-pr-gate",
                binding=f"{index + 1:064x}",
                experiment_binding="9" * 64,
                output_dir=tmp_path / f"output-{index}",
                confirm=True,
                resume=False,
            )
        except ValueError as exc:
            return str(exc)
        return "opened"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(claim, range(2)))

    assert outcomes.count("opened") == 1
    assert sum("already opened" in outcome for outcome in outcomes) == 1
