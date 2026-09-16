"""End-to-end CPU acceptance of multi-setting rebuttal-v2 plan artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers

from typo_cot.cli import main
from typo_cot.experiments.rebuttal_v2 import annotations
from typo_cot.experiments.rebuttal_v2.donor_artifacts import load_donor_bank
from typo_cot.experiments.rebuttal_v2.input_audit import load_input_audit, run_input_audit
from typo_cot.experiments.rebuttal_v2.plan_artifacts import load_plan, run_plan
from typo_cot.experiments.rebuttal_v2.schemas import (
    IntakeError,
    canonical_json,
    canonical_sha256,
    make_ref,
    sha256_text,
)
from typo_cot.experiments.rebuttal_v2.source import run_intake


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "configs/rebuttal_v2/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
TIME = "2026-09-16T12:00:00Z"
R3_MODEL = "google/gemma-3-4b-it"
R4_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _source_row(
    source_key: str,
    *,
    setting: str,
    original_problem_id: str | None,
    cohort: bool,
    variant: str,
) -> dict:
    model = R3_MODEL if setting == "R3" else R4_MODEL
    task = "gsm8k" if setting == "R3" else "mmlu-pro"
    animal = "cats" if setting == "R3" else "dogs"
    clean = f"{variant} {animal} eat 12 fish."
    typo = f"{variant} {animal} eat 13 fish."
    prefix = f"Example: {animal} eat 12 fish.\nQuestion: "
    return {
        "schema_version": "rebuttal-source-pair/v2",
        "source_record_id": f"record-{source_key}",
        "source_pair_key": source_key,
        "cohort_ids": [setting] if cohort else [],
        "task": task,
        "model_id": model,
        "dataset_id": task,
        "split": "test",
        "original_problem_id": original_problem_id,
        "target_rule": "rule-a",
        "perturbation_id": source_key,
        "clean_text": clean,
        "typo_text": typo,
        "clean_prompt": prefix + clean,
        "typo_prompt": prefix + typo,
        "clean_query_span": [len(prefix), len(prefix + clean)],
        "typo_query_span": [len(prefix), len(prefix + typo)],
        "tokenizer_revision": "a" * 40,
        "gold": "12" if setting == "R3" else "A",
        "canonical_gold": {"numerator": "12", "denominator": "1"} if setting == "R3" else "A",
        "gold_canonicalization_version": "fixture/v1",
        "expected_generations": [],
        "intended_edits": [],
        "archived_token_positions": [],
        "prompt_regions": {
            side: [{"kind": "stem", "start": len(prefix), "end": len(prefix + text)}]
            for side, text in (("clean", clean), ("typo", typo))
        },
    }


def _tokenizer(root: Path) -> tuple[Path, Path]:
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                word: index
                for index, word in enumerate(
                    [
                        "[UNK]",
                        "Example:",
                        "Question:",
                        "alpha",
                        "beta",
                        "gamma",
                        "donor3",
                        "donor4",
                        "cats",
                        "dogs",
                        "eat",
                        "12",
                        "13",
                        "fish.",
                    ]
                )
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer_path = root / "tokenizer.json"
    tokenizer_path.write_text(tokenizer.to_str(), encoding="utf-8")

    def descriptor(model_id: str) -> dict:
        return {
            "tokenizer_id": model_id,
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

    lock = _json(
        root / "tokenizer_lock.json",
        {
            "schema_version": "rebuttal-tokenizer-lock/v2",
            "models": {R3_MODEL: descriptor(R3_MODEL), R4_MODEL: descriptor(R4_MODEL)},
        },
    )
    return tokenizer_path, lock


def _index(
    root: Path,
    rows: list[dict],
    *,
    cohorts: dict[str, list[str]] | None,
    namespace: str,
) -> tuple[Path, Path]:
    source = _jsonl(root / "pairs.jsonl", rows)
    cohort_rows = []
    for setting, ids in (cohorts or {}).items():
        frozen = _json(
            root / f"{setting.lower()}_ids.json",
            {
                "schema_version": "rebuttal-cohort-ids/v2",
                "cohort_id": setting,
                "source_namespace": namespace,
                "source_pair_keys": ids,
            },
        )
        cohort_rows.append(
            {
                "cohort_id": setting,
                "setting": setting,
                "historical_n_reference": 172 if setting == "R3" else 97,
                "expected_ids_ref": {"path": frozen.name, "sha256": _sha(frozen)},
                "id_namespace": namespace,
                "membership_provenance_ref": None,
            }
        )
    index = _json(
        root / "index.json",
        {
            "schema_version": "rebuttal-archive-index/v2",
            "created_at": "2026-09-16T00:00:00Z",
            "source_namespace": namespace,
            "sources": [
                {
                    "source_id": "pairs",
                    "source_kind": "submitted-archive",
                    "role": "pairs",
                    "path": source.name,
                    "expected_sha256": _sha(source),
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
            "cohorts": cohort_rows,
            "identity_aliases_ref": None,
            "notes": "Synthetic CPU-only multi-setting acceptance fixture.",
        },
    )
    return index, source


def _annotation_pipeline(root: Path, audit_path: Path, manifest) -> Path:
    stage_a = root / "stage-a"
    annotations.export_stage_a(audit_path, stage_a)
    batch_a = _read(stage_a / "annotation_batch.json")
    ratings_a = [
        {
            "schema_version": annotations.A_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": TIME,
            "interpretations": ["Synthetic interpretation."],
            "ambiguity": "unique",
            "rationale": "Synthetic Stage A rationale.",
        }
        for blind in batch_a["selected_blind_ids"]
        for rater in batch_a["required_raters"]
    ]
    labels_a = _jsonl(root / "returned-a.jsonl", ratings_a)
    stage_b = root / "stage-b"
    annotations.export_stage_b(audit_path, labels_a, stage_a / "annotation_batch.json", stage_b)
    batch_b = _read(stage_b / "annotation_batch.json")
    mapping = {
        row["blind_id"]: row["pair_id"] for row in _rows(stage_b / "annotation_mapping.jsonl")
    }
    source_key = {row["pair_id"]: row["source_pair_key"] for row in manifest.pairs}
    chosen = {"r3-a": "unique_preserved", "r3-b": "task_changed", "r4-a": "ambiguous"}
    ratings_b = [
        {
            "schema_version": annotations.B_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": TIME,
            "stage_a_lock_sha256": batch_b["stage_a_lock_ref"]["sha256"],
            "label": chosen[source_key[mapping[blind]]],
            "rationale": "Synthetic Stage B rationale.",
        }
        for blind in batch_b["selected_blind_ids"]
        for rater in batch_b["required_raters"]
    ]
    labels_b = _jsonl(root / "returned-b.jsonl", ratings_b)
    imported = root / "labels"
    annotations.import_adjudicated_labels(
        audit_path, labels_b, stage_b / "annotation_batch.json", imported
    )
    return imported / "semantic_labels.jsonl"


def _donor_bank(root: Path, audit, source_path: Path, lock_path: Path) -> Path:
    audit_by_pair = {row["pair_id"]: row for row in audit.rows}
    bank_path = root / "donor_bank.jsonl"
    rows = []
    for index, pair in enumerate(audit.pairs):
        audit_row = audit_by_pair[pair["pair_id"]]
        prompt = root / f"donor_prompt_{index}.txt"
        prompt.write_text(pair["clean_prompt"], encoding="utf-8")
        endpoints = [edit["clean_endpoint"] for edit in audit_row["alignment"]["aligned_edits"]]
        rows.append(
            {
                "schema_version": "rebuttal-donor-bank/v2",
                "pair_id": pair["pair_id"],
                "original_problem_group_id": pair["original_problem_group_id"],
                "task": pair["task"],
                "model_id": pair["model_id"],
                "target_rule": pair["target_rule"],
                "clean_prompt_ref": make_ref(prompt, bank_path),
                "clean_prompt_sha256": _sha(prompt),
                "aligned_word_count": len(endpoints),
                "clean_endpoints": endpoints,
                "tokenizer_lock_ref": make_ref(lock_path, bank_path),
                "input_audit_ref": {
                    "artifact": make_ref(audit.path, bank_path),
                    "record_id": audit_row["input_audit_id"],
                },
                "provenance": {
                    "manifest_ref": {
                        "artifact": make_ref(audit.manifest.path, bank_path),
                        "record_id": pair["pair_id"],
                    },
                    "source_ref": {
                        "artifact": make_ref(source_path, bank_path),
                        "record_id": pair["source_record_id"],
                    },
                },
            }
        )
    return _jsonl(bank_path, rows)


def _runtime_lock(root: Path, audit, donor_bank) -> Path:
    donor_rows = {row["pair_id"]: row for row in donor_bank.rows}
    donor_audits = donor_bank.audit_rows_by_pair_id
    audit_rows = {row["pair_id"]: row for row in audit.rows}
    protocol_settings = {row["id"]: row for row in PROTOCOL["settings"]}

    def entry(setting: str) -> dict:
        model = protocol_settings[setting]["model"]
        descriptor = copy.deepcopy(audit.tokenizer_lock["models"][model])
        tokenizer_id = descriptor.pop("tokenizer_id")
        revision = descriptor.pop("revision")
        recipients = [pair for pair in audit.pairs if pair["model_id"] == model]
        donors = [pair for pair in donor_bank.pairs if pair["model_id"] == model]
        return {
            "model_id": model,
            "model_revision": "1" * 40,
            "model_files": {"model.safetensors": "2" * 64},
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": revision,
            "tokenizer_policy": descriptor,
            "generation": copy.deepcopy(PROTOCOL["generation"]),
            "attention_backend": "sdpa",
            "cache_behavior": {"use_cache": True, "implementation": "dynamic"},
            "effective_eos_ids": [1, 2],
            "library_versions": {
                "tokenizers": descriptor["library"]["version"],
                "transformers": "4.57.6",
                "torch": "2.10.0",
            },
            "code_commit": "3" * 40,
            "code_tree_sha256": "4" * 64,
            "device_identity": {
                "type": "cuda",
                "model": "Fixture GPU",
                "compute_capability": "8.0",
            },
            "compute_settings": {
                "cuda_version": "13.0",
                "driver_version": "999.0",
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "matmul_precision": "highest",
            },
            "prompt_sha256": {
                pair["pair_id"]: {
                    side: sha256_text(pair[f"{side}_prompt"]) for side in ("clean", "typo")
                }
                for pair in recipients
            },
            "prompt_token_ids_sha256": {
                pair["pair_id"]: {
                    side: canonical_sha256(
                        audit_rows[pair["pair_id"]]["alignment"][side]["prompt_token_ids"]
                    )
                    for side in ("clean", "typo")
                }
                for pair in recipients
            },
            "donor_prompt_sha256": {
                pair["pair_id"]: sha256_text(pair["clean_prompt"]) for pair in donors
            },
            "donor_prompt_token_ids_sha256": {
                pair["pair_id"]: canonical_sha256(
                    donor_audits[pair["pair_id"]]["alignment"]["clean"]["prompt_token_ids"]
                )
                for pair in donors
                if pair["pair_id"] in donor_rows
            },
        }

    return _json(
        root / "runtime_lock.json",
        {
            "schema_version": "rebuttal-runtime-lock/v2",
            "settings": {setting: entry(setting) for setting in ("R3", "R4")},
        },
    )


def _pipeline(root: Path, *, null_r3_groups=False, r3_missing=False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    _, tokenizer_lock = _tokenizer(root)
    recipients = [
        _source_row(
            "r3-a",
            setting="R3",
            original_problem_id=None if null_r3_groups else "shared-r3",
            cohort=True,
            variant="alpha",
        ),
        _source_row(
            "r3-b",
            setting="R3",
            original_problem_id=None if null_r3_groups else "shared-r3",
            cohort=True,
            variant="beta",
        ),
        _source_row(
            "r4-a",
            setting="R4",
            original_problem_id="r4-question",
            cohort=True,
            variant="gamma",
        ),
    ]
    r3_ids = ["r3-a", "r3-b"] + (["r3-missing"] if r3_missing else [])
    index, _ = _index(
        root / "recipients",
        recipients,
        cohorts={"R3": r3_ids, "R4": ["r4-a"]},
        namespace="acceptance-recipients",
    )
    intake = root / "recipient-intake"
    run_intake(index, PROTOCOL_PATH, intake)
    audit_dir = root / "recipient-audit"
    run_input_audit(intake / "pair_manifest.jsonl", tokenizer_lock, audit_dir)
    audit = load_input_audit(audit_dir / "input_audit_records.jsonl")
    labels = _annotation_pipeline(root / "annotations", audit.path, audit.manifest)

    donors = [
        _source_row(
            "donor-r3",
            setting="R3",
            original_problem_id="donor-r3-question",
            cohort=False,
            variant="donor3",
        ),
        _source_row(
            "donor-r4",
            setting="R4",
            original_problem_id="donor-r4-question",
            cohort=False,
            variant="donor4",
        ),
    ]
    donor_index, donor_source = _index(
        root / "donors",
        donors,
        cohorts=None,
        namespace="acceptance-donors",
    )
    donor_intake = root / "donor-intake"
    run_intake(donor_index, PROTOCOL_PATH, donor_intake)
    donor_audit_dir = root / "donor-audit"
    run_input_audit(donor_intake / "pair_manifest.jsonl", tokenizer_lock, donor_audit_dir)
    donor_audit = load_input_audit(donor_audit_dir / "input_audit_records.jsonl")
    bank_path = _donor_bank(root, donor_audit, donor_source, tokenizer_lock)
    bank = load_donor_bank(bank_path, PROTOCOL)
    runtime = _runtime_lock(root, audit, bank)
    return {
        "root": root,
        "manifest": audit.manifest.path,
        "audit": audit.path,
        "labels": labels,
        "runtime": runtime,
        "bank": bank_path,
        "audit_bundle": audit,
        "bank_bundle": bank,
    }


@pytest.fixture(scope="module")
def accepted(tmp_path_factory):
    case = _pipeline(tmp_path_factory.mktemp("plan-acceptance"))
    one = case["root"] / "plan-one"
    six = case["root"] / "plan-six"
    run_plan(
        case["manifest"],
        case["audit"],
        case["labels"],
        case["runtime"],
        PROTOCOL_PATH,
        one,
        experiments=["R4", "R3"],
        donor_bank_path=case["bank"],
        num_shards=1,
    )
    run_plan(
        case["manifest"],
        case["audit"],
        case["labels"],
        case["runtime"],
        PROTOCOL_PATH,
        six,
        experiments=["R3", "R4"],
        donor_bank_path=case["bank"],
        num_shards=6,
    )
    return {**case, "one": one, "six": six}


def test_multi_item_multi_setting_pipeline_freezes_exact_intended_grid(accepted):
    one = load_plan(accepted["one"] / "plan.jsonl")
    six = load_plan(accepted["six"] / "plan.jsonl")
    assert len(one.rows) == 22 == one.metadata["intended_row_count"]
    pair_by_key = {pair["source_pair_key"]: pair for pair in one.manifest.pairs}
    r4_id = pair_by_key["r4-a"]["pair_id"]
    r4_rows = [row for row in one.rows if row["pair_id"] == r4_id]
    assert len(r4_rows) == 10
    assert len([row for row in r4_rows if row["window"] is None]) == 2
    assert {row["semantic_label"] for row in one.rows if row["pair_id"] == r4_id} == {"ambiguous"}
    assert len([row for row in one.rows if row["arm"] in {"clean", "typo"}]) == 6
    assert one.path.read_bytes() == six.path.read_bytes()
    assert (accepted["one"] / "donor_assignment.json").read_bytes() == (
        accepted["six"] / "donor_assignment.json"
    ).read_bytes()
    assert (accepted["one"] / "expected_generations.jsonl").read_bytes() == (
        accepted["six"] / "expected_generations.jsonl"
    ).read_bytes()


def test_group_sharding_semantics_and_cross_coordinates_are_frozen(accepted):
    bundle = load_plan(accepted["six"] / "plan.jsonl")
    pairs = {pair["source_pair_key"]: pair for pair in bundle.manifest.pairs}
    r3_ids = {pairs[key]["pair_id"] for key in ("r3-a", "r3-b")}
    r3_groups = {
        pair["original_problem_group_id"]
        for pair in bundle.manifest.pairs
        if pair["pair_id"] in r3_ids
    }
    assert len(r3_groups) == 1
    shard_group = next(
        row
        for row in bundle.shard_assignment["groups"]
        if row["original_problem_group_id"] in r3_groups
    )
    assert r3_ids <= set(shard_group["pair_ids"])
    donor_groups = {pair["original_problem_group_id"] for pair in bundle.donor_bank.pairs}
    assert donor_groups.isdisjoint(r3_groups)

    assignments = {
        row["pair_id"]: row
        for row in _read(accepted["six"] / "donor_assignment.json")["assignments"]
    }
    valid_recipient = next(pid for pid, row in assignments.items() if row["donor_pair_id"])
    donor_id = assignments[valid_recipient]["donor_pair_id"]
    cross = next(
        row for row in bundle.rows if row["pair_id"] == valid_recipient and row["arm"] == "cross"
    )
    donor_row = next(row for row in bundle.donor_bank.rows if row["pair_id"] == donor_id)
    donor_audit = bundle.donor_bank.audit_rows_by_pair_id[donor_id]
    recipient_audit = next(
        row for row in bundle.input_audit.rows if row["pair_id"] == valid_recipient
    )
    assert cross["source_positions"] == donor_row["clean_endpoints"]
    assert cross["write_positions"] == [
        edit["typo_endpoint"] for edit in recipient_audit["alignment"]["aligned_edits"]
    ]
    assert cross["source_prompt_ref"] == {
        **donor_audit["manifest_ref"],
        "field": "clean_prompt",
    }
    r3_labels = {
        row["pair_id"]: row["semantic_label"]
        for row in bundle.rows
        if row["pair_id"] in r3_ids and row["arm"] == "clean"
    }
    assert set(r3_labels.values()) == {"unique_preserved", "task_changed"}
    assert len(r3_labels) == 2


def test_declared_partial_r3_preserves_missing_id_and_plans_recovered_subset(tmp_path):
    case = _pipeline(tmp_path, r3_missing=True)
    output = tmp_path / "plan"
    result = run_plan(
        case["manifest"],
        case["audit"],
        case["labels"],
        case["runtime"],
        PROTOCOL_PATH,
        output,
        experiments=["R3", "R4"],
        donor_bank_path=case["bank"],
    )
    assert result["status"] == "complete"
    coverage = _read(output / "planning_coverage.json")
    r3 = next(row for row in coverage["source_cohort_coverage"] if row["setting"] == "R3")
    assert r3["coverage_status"] == "partial"
    assert r3["missing_source_pair_keys"] == ["r3-missing"]
    assert coverage["counts"]["selected_pair_count_by_setting"] == {"R3": 2, "R4": 1}
    assert len(_rows(output / "plan.jsonl")) == 22


def test_two_raw_null_groups_block_cli_and_publish_only_diagnostics(tmp_path, capsys):
    case = _pipeline(tmp_path, null_r3_groups=True)
    output = tmp_path / "blocked-plan"
    code = main(
        [
            "rebuttal-v2",
            "plan",
            "--manifest",
            str(case["manifest"]),
            "--input-audit",
            str(case["audit"]),
            "--labels",
            str(case["labels"]),
            "--runtime-lock",
            str(case["runtime"]),
            "--protocol",
            str(PROTOCOL_PATH),
            "--donor-bank",
            str(case["bank"]),
            "--experiments",
            "R4",
            "R3",
            "--num-shards",
            "6",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert {path.name for path in output.iterdir()} == {
        "preflight.json",
        "planning_coverage.json",
        "protocol.json",
        "run.json",
    }
    preflight = _read(output / "preflight.json")
    pairs = load_input_audit(case["audit"]).pairs
    expected = sorted(pair["pair_id"] for pair in pairs if pair["source_pair_key"].startswith("r3"))
    assert preflight["blocking_pair_ids"] == expected
    assert all(
        "original_problem_group_id_unknown" in row["reason_codes"]
        for row in preflight["blockers"]
        if row["pair_id"] in expected
    )


@pytest.mark.parametrize(
    "name,mutation,match",
    [
        (
            "plan.jsonl",
            lambda value: value[0].update(semantic_label="unassessable"),
            "plan rows differs",
        ),
        (
            "expected_generations.jsonl",
            lambda value: value[0].update(generation_id="0" * 64),
            "expected generation inventory differs",
        ),
        (
            "donor_assignment.json",
            lambda value: value["unused_donor_pair_ids"].append("forged-donor"),
            "donor assignment differs",
        ),
    ],
)
def test_rehashed_derived_plan_artifacts_are_reconstructed_and_rejected(
    accepted, tmp_path, name, mutation, match
):
    copied = tmp_path / "copied"
    shutil.copytree(accepted["one"], copied)
    path = copied / name
    value = _rows(path) if name.endswith(".jsonl") else _read(path)
    mutation(value)
    if name.endswith(".jsonl"):
        _jsonl(path, value)
    else:
        _json(path, value)
    metadata_path = copied / "plan.meta.json"
    metadata = _read(metadata_path)
    field = {
        "plan.jsonl": "plan_ref",
        "expected_generations.jsonl": "expected_generations_ref",
        "donor_assignment.json": "donor_assignment_ref",
    }[name]
    metadata[field]["sha256"] = _sha(path)
    _json(metadata_path, metadata)
    with pytest.raises(IntakeError, match=match):
        load_plan(copied / "plan.jsonl")
