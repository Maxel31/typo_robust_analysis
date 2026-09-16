"""Public-input integration fixtures for independent coordinates and R2 CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers

from typo_cot.cli import main
from typo_cot.experiments.rebuttal_v2.input_audit import load_input_audit, run_input_audit
from typo_cot.experiments.rebuttal_v2.source import run_intake


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "configs/rebuttal_v2/protocol.json"


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, *, missing_lock=False, pair_changes=None, reference_text=False):
    root.mkdir(parents=True, exist_ok=True)
    clean, typo = "red cats eat 12 fish.", "red cats eat 13 fish."
    prefix = "Example: red cats eat 12 fish.\nQuestion: "
    pair = {
        "schema_version": "rebuttal-source-pair/v2",
        "source_record_id": "p1",
        "source_pair_key": "original-1",
        "cohort_ids": ["R3"],
        "task": "gsm8k",
        "model_id": "google/gemma-3-4b-it",
        "dataset_id": "gsm8k",
        "split": "test",
        "original_problem_id": "question-1",
        "clean_text": clean,
        "typo_text": typo,
        "clean_prompt": prefix + clean,
        "typo_prompt": prefix + typo,
        "clean_query_span": [len(prefix), len(prefix + clean)],
        "typo_query_span": [len(prefix), len(prefix + typo)],
        "tokenizer_revision": "a" * 40,
        "gold": "12",
        "canonical_gold": {"numerator": "12", "denominator": "1"},
        "gold_canonicalization_version": "fixture/v1",
        "expected_generations": [],
        "intended_edits": [
            {
                "audit_adapter": "rebuttal-intended-token/v2",
                "clean_token_index": 2,
                "actual_clean_span": [14, 15],
                "actual_typo_span": [14, 15],
            }
        ],
        "archived_token_positions": [
            {"audit_adapter": "rebuttal-word-endpoints/v2", "clean_endpoint": 4, "typo_endpoint": 4}
        ],
        "prompt_regions": {
            side: [{"kind": "stem", "start": len(prefix), "end": len(prefix + clean)}]
            for side in ("clean", "typo")
        },
        **(pair_changes or {}),
    }
    if reference_text:
        for field in ("clean_text", "typo_text", "clean_prompt", "typo_prompt"):
            text_path = root / f"{field}.txt"
            text_path.write_text(pair[field], encoding="utf-8")
            pair[field] = None
            pair[f"{field}_ref"] = {"path": text_path.name, "sha256": _sha(text_path)}
    pair_path = _write(root / "pairs.jsonl", pair)
    ids = _write(
        root / "ids.json",
        {
            "schema_version": "rebuttal-cohort-ids/v2",
            "cohort_id": "R3",
            "source_namespace": "r2-fixture",
            "source_pair_keys": ["original-1", "missing-original"],
        },
    )
    index = _write(
        root / "index.json",
        {
            "schema_version": "rebuttal-archive-index/v2",
            "created_at": "2026-09-16T00:00:00Z",
            "source_namespace": "r2-fixture",
            "sources": [
                {
                    "source_id": "pairs",
                    "source_kind": "submitted-archive",
                    "role": "pairs",
                    "path": pair_path.name,
                    "expected_sha256": _sha(pair_path),
                    "format": "jsonl",
                    "adapter_version": "rebuttal-source-pair/v2",
                    "source_commit": None,
                    "model_revision": None,
                    "tokenizer_revision": None,
                    "prompt_template_revision": None,
                    "generation_config": None,
                    "availability": "available",
                    "reason_codes": [],
                }
            ],
            "cohorts": [
                {
                    "cohort_id": "R3",
                    "setting": "R3",
                    "historical_n_reference": 172,
                    "expected_ids_ref": {"path": ids.name, "sha256": _sha(ids)},
                    "id_namespace": "r2-fixture",
                    "membership_provenance_ref": None,
                }
            ],
            "identity_aliases_ref": None,
            "notes": "Synthetic R2 fixture, not paper data.",
        },
    )
    run_intake(index, PROTOCOL, root / "intake")
    backend = Tokenizer(
        models.WordLevel(
            {
                word: index
                for index, word in enumerate(
                    ["[UNK]", "Example:", "red", "cats", "eat", "12", "fish.", "Question:", "13"]
                )
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer_path = root / "tokenizer.json"
    tokenizer_path.write_text(backend.to_str(), encoding="utf-8")
    entry = {
        "tokenizer_id": "google/gemma-3-4b-it",
        "revision": "a" * 40,
        "tokenizer_files": {
            "tokenizer.json": {"path": tokenizer_path.name, "sha256": _sha(tokenizer_path)}
        },
        "library": {"name": "tokenizers", "version": tokenizers.__version__},
        "special_token_ids": {},
        "chat_template": None,
        "add_special_tokens": False,
        "normalization": None,
        "padding_side": "left",
        "offset_convention": "unicode-codepoint-half-open",
    }
    lock = _write(
        root / "tokenizer_lock.json",
        {
            "schema_version": "rebuttal-tokenizer-lock/v2",
            "models": {} if missing_lock else {pair["model_id"]: entry},
        },
    )
    return root / "intake/pair_manifest.jsonl", lock


def test_exact_second_query_endpoint_is_independent_of_target_and_archive(tmp_path):
    manifest, lock = _fixture(tmp_path)
    before = {path: _sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "audit"
    run = run_input_audit(manifest, lock, output)
    assert run["run_status"] == "complete"
    assert run["experiment_status"] == "partial"
    row = _rows(output / "input_audit_records.jsonl")[0]
    assert row["span_audit"]["edit_spans"] == [
        {"clean_start": 14, "clean_end": 15, "typo_start": 14, "typo_end": 15}
    ]
    assert row["alignment"]["eligibility"] == "valid"
    assert row["alignment"]["aligned_edits"][0]["clean_endpoint"] == 10
    assert row["alignment"]["aligned_edits"][0]["typo_endpoint"] == 10
    assert row["alignment"]["clean"]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6]
    assert row["intended_token_hit"]["status"] == "miss"
    assert row["archived_coordinate_agreement"]["status"] == "differs"
    assert row["risk_flags"]["flags"]["number"] is True
    assert row["risk_flags"]["flags"]["stem"] is True
    assert row["alignment"]["clean"]["archived_revision_match"] is True
    assert row["alignment"]["clean"]["matches_archived_tokenizer"] is None
    assert all(_sha(path) == digest for path, digest in before.items())
    summary = _read(output / "input_audit_summary.json")
    assert "missing-original" in json.dumps(summary["source_cohort_coverage"])
    assert summary["semantic_status"] == "not_run"
    bundle = load_input_audit(output / "input_audit_records.jsonl")
    assert bundle.rows == [row]
    for name, ref in run["output_refs"].items():
        assert _sha(output / name) == ref["sha256"]
    with pytest.raises(ValueError, match="absent or empty"):
        run_input_audit(manifest, lock, output)


def test_unknown_tokenizer_keeps_independent_text_audit(tmp_path):
    manifest, lock = _fixture(tmp_path, missing_lock=True)
    run_input_audit(manifest, lock, tmp_path / "audit")
    row = _rows(tmp_path / "audit/input_audit_records.jsonl")[0]
    assert row["span_audit"]["diff_status"] == "changed"
    assert row["alignment"]["eligibility"] == "unknown"
    assert row["alignment"]["clean"]["prompt_token_ids"] is None
    assert row["risk_flags"]["flags"]["number"] is True
    assert row["intended_token_hit"]["status"] == "unknown"


def test_reference_only_text_is_resolved_without_mutating_manifest(tmp_path):
    manifest, lock = _fixture(tmp_path, reference_text=True)
    run_input_audit(manifest, lock, tmp_path / "audit")
    bundle = load_input_audit(tmp_path / "audit/input_audit_records.jsonl")
    assert bundle.pairs[0]["typo_text"] == "red cats eat 13 fish."
    assert bundle.manifest.pairs[0]["typo_text"] is None
    assert bundle.rows[0]["alignment"]["eligibility"] == "valid"


@pytest.mark.parametrize("tamper", ["endpoint", "bool-offset", "missing-row"])
def test_rebound_audit_cannot_forge_independent_results(tmp_path, tamper):
    manifest, lock = _fixture(tmp_path)
    run_input_audit(manifest, lock, tmp_path / "audit")
    path = tmp_path / "audit/input_audit_records.jsonl"
    row = _rows(path)[0]
    if tamper == "endpoint":
        row["alignment"]["aligned_edits"][0]["clean_endpoint"] = 4
    elif tamper == "bool-offset":
        row["alignment"]["clean"]["offset_mapping"][0][0] = False
    path.write_text("" if tamper == "missing-row" else json.dumps(row) + "\n", encoding="utf-8")
    meta_path = path.with_name("input_audit_records.meta.json")
    meta = _read(meta_path)
    meta["audit_ref"]["sha256"] = _sha(path)
    _write(meta_path, meta)
    with pytest.raises(ValueError, match="verified independent audit"):
        load_input_audit(path)


def test_input_audit_cli_is_cpu_only(tmp_path):
    manifest, lock = _fixture(tmp_path)
    command = [
        "rebuttal-v2",
        "input-audit",
        "--manifest",
        str(manifest),
        "--tokenizer-lock",
        str(lock),
        "--output-dir",
        str(tmp_path / "audit"),
    ]
    script = f"import sys; from typo_cot.cli import main; assert main({command!r}) == 0; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["command"] == "input-audit"


def test_stage_b_cli_requires_explicit_completed_a_batch(tmp_path, capsys):
    assert (
        main(
            [
                "rebuttal-v2",
                "annotation-export",
                "--audit",
                str(tmp_path / "missing.jsonl"),
                "--stage",
                "b",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    assert "requires --stage-a-labels and --annotation-batch" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "missing", ["clean_prompt", "typo_prompt", "both_prompts", "clean_text", "blank_clean_text"]
)
def test_public_missing_material_keeps_annotation_coverage_and_explicit_judgments(
    tmp_path, missing
):
    from typo_cot.experiments.rebuttal_v2 import annotations as annotation

    changes = (
        {"clean_prompt": None, "typo_prompt": None}
        if missing == "both_prompts"
        else {"clean_text": " \t", "clean_query_span": None}
        if missing == "blank_clean_text"
        else {missing: None}
    )
    manifest, lock = _fixture(tmp_path, pair_changes=changes)
    run_input_audit(manifest, lock, tmp_path / "audit")
    audit = tmp_path / "audit/input_audit_records.jsonl"
    stage_a, stage_b = tmp_path / "stage-a", tmp_path / "stage-b"
    annotation.export_stage_a(audit, stage_a)
    form = _rows(stage_a / "annotation_stage_a.jsonl")[0]
    assert form["typo_query"] == "red cats eat 13 fish."
    assert "clean_query" not in form
    batch = _read(stage_a / "annotation_batch.json")
    coverage = _read(stage_a / "annotation_coverage.json")
    assert coverage["pair_coverage"][0]["annotation_status"] == "selected"
    reasons = coverage["pair_coverage"][0]["reason_codes"]
    for side in ("clean", "typo"):
        if changes.get(f"{side}_prompt", "present") is None:
            assert f"{side}_prompt_unavailable_for_declared_regions" in reasons
    missing_clean = missing in {"clean_text", "blank_clean_text"}
    if missing_clean:
        assert "clean_query_unavailable" in reasons
    returned_a = tmp_path / "returned-a.jsonl"
    returned_a.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": annotation.A_SCHEMA,
                    "blind_id": form["blind_id"],
                    "rater_id": rater,
                    "timestamp": "2026-09-16T12:00:00Z",
                    "interpretations": [],
                    "ambiguity": "unassessable",
                    "rationale": "Synthetic workflow fixture only.",
                }
            )
            + "\n"
            for rater in batch["required_raters"]
        ),
        encoding="utf-8",
    )
    annotation.export_stage_b(audit, returned_a, stage_a / "annotation_batch.json", stage_b)
    forms_b = _rows(stage_b / "annotation_stage_b.jsonl")
    assert len(forms_b) == 2
    if missing_clean:
        assert all(row["allowed_labels"] == ["unassessable"] for row in forms_b)
    else:
        assert all(set(row["allowed_labels"]) == annotation.LABELS for row in forms_b)
    returned_b = tmp_path / "returned-b.jsonl"
    returned_b.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": annotation.B_SCHEMA,
                    "blind_id": row["blind_id"],
                    "rater_id": row["rater_id"],
                    "timestamp": "2026-09-16T12:01:00Z",
                    "stage_a_lock_sha256": row["stage_a_lock_sha256"],
                    "label": "unassessable",
                    "rationale": "Synthetic explicit rating, never auto-generated by import.",
                }
            )
            + "\n"
            for row in forms_b
        ),
        encoding="utf-8",
    )
    imported = tmp_path / "semantic"
    annotation.import_adjudicated_labels(
        audit, returned_b, stage_b / "annotation_batch.json", imported
    )
    semantic = _rows(imported / "semantic_labels.jsonl")
    assert len(semantic) == 1
    assert semantic[0]["label"] == "unassessable"
    assert len(semantic[0]["stage_b_judgments"]) == 2
    assert "missing-original" in json.dumps(_read(imported / "annotation_coverage.json"))


def test_public_cli_two_stage_human_forms_preserve_original_membership(tmp_path, capsys):
    from typo_cot.experiments.rebuttal_v2.annotations import A_SCHEMA, B_SCHEMA, ADJ_SCHEMA

    manifest, lock = _fixture(tmp_path, reference_text=True)
    run_input_audit(manifest, lock, tmp_path / "audit")
    audit = tmp_path / "audit/input_audit_records.jsonl"
    stage_a, stage_b, imported = (tmp_path / name for name in ("stage-a", "stage-b", "semantic"))
    assert (
        main(
            [
                "rebuttal-v2",
                "annotation-export",
                "--audit",
                str(audit),
                "--stage",
                "a",
                "--raters",
                "reader-one",
                "reader-two",
                "--output-dir",
                str(stage_a),
            ]
        )
        == 0
    )
    capsys.readouterr()
    reviewer = _rows(stage_a / "annotation_stage_a.jsonl")
    assert len(reviewer) == 1
    assert set(reviewer[0]) == {
        "schema_version",
        "blind_id",
        "stage",
        "typo_query",
        "typo_context",
        "task_instructions",
        "presented_options",
    }
    assert reviewer[0]["typo_query"] == "red cats eat 13 fish."
    assert reviewer[0]["typo_context"] == []
    batch_a = _read(stage_a / "annotation_batch.json")
    blind = reviewer[0]["blind_id"]
    ratings_a = [
        {
            "schema_version": A_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": "2026-09-16T12:00:00Z",
            "interpretations": ["Synthetic reading, not actual human data."],
            "ambiguity": "unique",
            "rationale": f"Private A text from {rater}",
        }
        for rater in batch_a["required_raters"]
    ]
    returned_a = tmp_path / "returned-a.jsonl"
    returned_a.write_text("".join(json.dumps(row) + "\n" for row in ratings_a), encoding="utf-8")
    assert (
        main(
            [
                "rebuttal-v2",
                "annotation-export",
                "--audit",
                str(audit),
                "--stage",
                "b",
                "--stage-a-labels",
                str(returned_a),
                "--annotation-batch",
                str(stage_a / "annotation_batch.json"),
                "--output-dir",
                str(stage_b),
            ]
        )
        == 0
    )
    capsys.readouterr()
    forms_b = _rows(stage_b / "annotation_stage_b.jsonl")
    assert len(forms_b) == 2
    assert all(row["clean_query"] == "red cats eat 12 fish." for row in forms_b)
    assert "Private A text" not in (stage_b / "annotation_stage_b.jsonl").read_text()
    batch_b = _read(stage_b / "annotation_batch.json")
    lock_hash = batch_b["stage_a_lock_ref"]["sha256"]
    ratings_b = [
        {
            "schema_version": B_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": "2026-09-16T12:01:00Z",
            "stage_a_lock_sha256": lock_hash,
            "label": label,
            "rationale": "Synthetic fixture judgment.",
        }
        for rater, label in zip(
            batch_b["required_raters"], ("unique_preserved", "task_changed"), strict=True
        )
    ]
    ratings_b.append(
        {
            "schema_version": ADJ_SCHEMA,
            "blind_id": blind,
            "adjudicator_id": "third-reader",
            "timestamp": "2026-09-16T12:02:00Z",
            "stage_a_lock_sha256": lock_hash,
            "final_label": "task_changed",
            "disagreement_reason": "The quantity differs in this synthetic test.",
        }
    )
    returned_b = tmp_path / "returned-b.jsonl"
    returned_b.write_text("".join(json.dumps(row) + "\n" for row in ratings_b), encoding="utf-8")
    assert (
        main(
            [
                "rebuttal-v2",
                "annotation-import",
                "--audit",
                str(audit),
                "--labels",
                str(returned_b),
                "--annotation-batch",
                str(stage_b / "annotation_batch.json"),
                "--output-dir",
                str(imported),
            ]
        )
        == 0
    )
    run = json.loads(capsys.readouterr().out)
    assert run["experiment_status"] == "partial"
    assert run["annotation_stage_status"] == "complete"
    semantic = _rows(imported / "semantic_labels.jsonl")
    assert len(semantic) == 1
    assert semantic[0]["label"] == "task_changed"
    assert semantic[0]["adjudication_status"] == "adjudicated"
    assert len(semantic[0]["stage_a_judgments"]) == len(semantic[0]["stage_b_judgments"]) == 2
    agreement = _read(imported / "annotation_agreement.json")
    assert agreement["rater_count"] == 2
    assert agreement["pre_adjudication_agreement_rate"] == 0.0
    assert agreement["adjudicated_count"] == 1
    assert "missing-original" in json.dumps(_read(imported / "annotation_coverage.json"))
    assert len(_rows(manifest)) == 1
