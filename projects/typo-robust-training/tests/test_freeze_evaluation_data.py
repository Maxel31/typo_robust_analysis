"""Evaluation text is frozen independently from every training-cycle config."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from typo_robust_training.data.config import DatasetSource, load_training_data_config
from typo_robust_training.data.records import CleanRecord, NaturalTypoRecord
from typo_robust_training.data.splits import NearDuplicateTextIndex
from typo_robust_training.evaluation.freeze import (
    FreezeEvaluationRunConfig,
    _Exclusions,
    _exclusions,
    _could_retain_smallest,
    _natural_edit,
    _retain_smallest,
    _select_corpus,
    _select_natural,
    _supports_frozen_typo_grid,
    _supports_transposition,
    run_freeze_robustness_evaluation,
)
from typo_robust_training.evaluation.data import (
    load_evaluation_bundle,
    load_evaluation_corpus_bundle,
)
from typo_robust_training.evaluation.study import load_evaluation_study_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDY = PROJECT_ROOT / "configs/robustness-evaluation-v1.yaml"
SOURCES = PROJECT_ROOT / "configs" / "cycle3" / "gemma4b-data-64m.yaml"


def test_frozen_typo_grid_counts_distinct_words_not_occurrences() -> None:
    protocol = load_evaluation_study_protocol(STUDY)
    record = CleanRecord(
        source="gsm8k",
        source_revision="a" * 40,
        source_split="test",
        source_id="gsm8k:repeated-word-grid",
        group_id="gsm8k:repeated-word-grid",
        text="Add value value value value now.",
        task="gsm8k",
        answer="42",
        metadata={},
    )

    assert not _supports_frozen_typo_grid(record, protocol=protocol)


def test_transposition_eligibility_does_not_shrink_primary_population() -> None:
    protocol = load_evaluation_study_protocol(STUDY)
    record = CleanRecord(
        source="math_500",
        source_revision="a" * 40,
        source_split="test",
        source_id="math_500:non-transposable-primary",
        group_id="math_500:non-transposable-primary",
        text="Aaa bbb ccc ddd.",
        task="math_500",
        answer="42",
        metadata={},
    )

    assert _supports_frozen_typo_grid(record, protocol=protocol)
    assert not _supports_transposition(record, protocol=protocol)


def _letters(index: int) -> str:
    value = index
    chars: list[str] = []
    while True:
        chars.append(chr(ord("a") + value % 26))
        value //= 26
        if value == 0:
            return "".join(reversed(chars))


def _task_record(source: str, revision: str, split: str, index: int) -> CleanRecord:
    if source in {"mmlu", "arc", "mmlu_pro", "commonsense_qa"}:
        text = (
            "Which reliable airport serves the northern research district carefully?\n"
            "A. Alpha terminal\nB. Northern airport\nC. Third terminal"
        )
        answer = "B"
    elif source == "math_500":
        text = "Determine the perimeter of a rectangular garden after careful measurement."
        answer = "42"
    else:
        text = (
            "Reliable gardeners carefully measure the airport garden before calculating "
            "the combined perimeter for tomorrow morning."
        )
        answer = "42"
    metadata: dict[str, object] = {"dataset_row_index": index}
    if source == "mmlu":
        metadata["subject"] = ("history", "science", "law")[index % 3]
    elif source == "mmlu_pro":
        metadata["category"] = ("biology", "business", "physics")[index % 3]
    elif source == "math_500":
        metadata.update({"subject": ("algebra", "geometry")[index % 2], "level": index % 5})
    source_id = f"{source}:{split}-{index}"
    return CleanRecord(
        source=source,
        source_revision=revision,
        source_split=split,
        source_id=source_id,
        group_id=source_id,
        text=text,
        task=source,
        answer=answer,
        metadata=metadata,
    )


class _Provider:
    def __init__(self, sources: Mapping[str, DatasetSource]) -> None:
        self.sources = sources

    def iter_records(
        self,
        source_name: str,
        source: DatasetSource,
    ) -> Iterable[CleanRecord | NaturalTypoRecord]:
        if source_name == "github_typo_corpus":
            natural_words = (
                "reliable",
                "airport",
                "northern",
                "research",
                "district",
                "carefully",
                "terminal",
                "gardeners",
                "measure",
                "garden",
                "calculating",
                "combined",
                "perimeter",
                "tomorrow",
                "morning",
                "determine",
                "rectangular",
                "after",
                "measurement",
                "serves",
            )
            for index in range(3_400):
                if index < 200:
                    repository = f"https://example.test/train-{index // 10}.git"
                elif index < 300:
                    repository = f"https://example.test/tune-{index // 10}.git"
                else:
                    repository = f"https://example.test/eval-{index // 5}.git"
                clean_word = natural_words[index % len(natural_words)]
                typo_word = clean_word[:2] + clean_word[3:]
                suffix = _letters(index)
                clean = f"The {clean_word} record remains readable near the district {suffix}."
                typo = f"The {typo_word} record remains readable near the district {suffix}."
                yield NaturalTypoRecord(
                    source=source_name,
                    source_revision=source.revision,
                    source_split="corpus",
                    source_id=f"natural-{index}",
                    group_id=repository,
                    clean_text=clean,
                    typo_text=typo,
                    repository=repository,
                    repository_license="MIT",
                    operation="deletion",
                    training_eligible=True,
                    metadata={},
                )
            return
        if source_name == "fineweb_edu":
            for index in range(2_500):
                source_id = f"fineweb_edu:train-{index}"
                yield CleanRecord(
                    source=source_name,
                    source_revision=source.revision,
                    source_split="train",
                    source_id=source_id,
                    group_id=source_id,
                    text=f"Educational document {_letters(index)} explains a reliable concept.",
                    task=None,
                    answer=None,
                    metadata={},
                )
            return
        if source_name == "dolma":
            for index in range(1_600):
                source_id = f"dolma:train-{index}"
                yield CleanRecord(
                    source=source_name,
                    source_revision=source.revision,
                    source_split="train",
                    source_id=source_id,
                    group_id=source_id,
                    text=f"Unseen domain document {_letters(index)} remains readable.",
                    task=None,
                    answer=None,
                    metadata={},
                )
            return
        split_counts = {
            "gsm8k": (("train", 700), ("test", 1_100)),
            "mmlu": (("auxiliary_train", 700), ("dev", 100), ("test", 1_100)),
            "arc": (("train", 700), ("test", 1_100)),
            "mmlu_pro": (("test", 1_100),),
            "math_500": (("test", 500),),
            "commonsense_qa": (("validation", 1_100),),
        }[source_name]
        for split, count in split_counts:
            for index in range(count):
                yield _task_record(source_name, source.revision, split, index)

    def provenance(self) -> Mapping[str, object]:
        return {"provider": "evaluation-fixture/v1"}


def _write_exclusions(root: Path) -> None:
    root.mkdir()
    training = [
        {
            "record_id": f"{index:064x}",
            "source": "github_typo_corpus",
            "source_id": f"natural-{index}",
            "group_id": f"https://example.test/train-{index // 10}.git",
            "kind": "natural",
            "repository": f"https://example.test/train-{index // 10}.git",
            "clean_text": f"The reliable airport example {index} remains readable.",
        }
        for index in range(200)
    ]
    diagnostic = [
        {
            "record_id": f"{10_000 + index:064x}",
            "source": "gsm8k",
            "source_id": f"gsm8k:train-{index}",
            "group_id": f"gsm8k:train-{index}",
            "clean_text": f"The reliable GSM8K diagnostic example {index} remains readable.",
        }
        for index in range(10)
    ]
    tune = [
        *[
            {
                "record_id": f"{20_000 + index:064x}",
                "source": "gsm8k",
                "source_id": f"gsm8k:train-{100 + index}",
                "group_id": f"gsm8k:train-{100 + index}",
                "clean_text": f"The reliable GSM8K tune example {index} remains readable.",
            }
            for index in range(100)
        ],
        *[
            {
                "record_id": f"{30_000 + index:064x}",
                "source": "fineweb_edu",
                "source_id": f"fineweb_edu:train-{index}",
                "group_id": f"fineweb_edu:train-{index}",
                "clean_text": f"Educational document {_letters(index)} explains a reliable concept.",
            }
            for index in range(200)
        ],
        *[
            {
                "record_id": f"{40_000 + index:064x}",
                "source": "github_typo_corpus",
                "source_id": f"natural-{200 + index}",
                "group_id": f"https://example.test/tune-{(200 + index) // 10}.git",
                "kind": "natural",
                "repository": f"https://example.test/tune-{(200 + index) // 10}.git",
                "clean_text": f"The reliable airport tune example {index} remains readable.",
            }
            for index in range(100)
        ],
    ]
    for name, rows in (
        ("training_sources.jsonl", training),
        ("diagnostic_manifest.jsonl", diagnostic),
        ("tune_manifest.jsonl", tune),
    ):
        (root / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "build-robustness-training-data-run/v1",
                "status": "completed",
                "protocol_sha256": hashlib.sha256(SOURCES.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_natural_dictionary_excludes_case_only_corrections() -> None:
    record = NaturalTypoRecord(
        source="github_typo_corpus",
        source_revision="a" * 40,
        source_split="corpus",
        source_id="case-only",
        group_id="https://example.test/repository.git",
        clean_text="The Internet remains available.",
        typo_text="The internet remains available.",
        repository="https://example.test/repository.git",
        repository_license="MIT",
        operation="natural-statistics-substitution",
        training_eligible=True,
        metadata={},
    )

    assert _natural_edit(record, minimum_word_letters=3) is None


def test_exclusions_index_clean_text_and_reject_rows_without_text(tmp_path: Path) -> None:
    exclusion_root = tmp_path / "exclusions"
    _write_exclusions(exclusion_root)
    exclusions = _exclusions(exclusion_root)
    tune_rows = [
        json.loads(line)
        for line in (exclusion_root / "tune_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]

    assert exclusions.prior_tune_near_duplicates.contains_near_duplicate(
        str(tune_rows[0]["clean_text"])
    )

    broken = dict(tune_rows[0])
    broken.pop("clean_text")
    tune_rows[0] = broken
    (exclusion_root / "tune_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tune_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row has no clean text"):
        _exclusions(exclusion_root)


def test_bottom_k_prefilter_skips_noncompetitive_near_duplicate_queries() -> None:
    heap: list[tuple[int, str, CleanRecord]] = []
    records = tuple(_task_record("gsm8k", "a" * 40, "test", index) for index in range(3))
    _retain_smallest(heap, records[0], key=10, limit=2)
    _retain_smallest(heap, records[1], key=20, limit=2)

    assert not _could_retain_smallest(
        heap,
        record_id=records[2].record_id,
        key=30,
        limit=2,
    )
    assert _could_retain_smallest(
        heap,
        record_id=records[2].record_id,
        key=5,
        limit=2,
    )
    assert not _could_retain_smallest([], record_id="unused", key=0, limit=0)


def _rows(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def _natural_record(*, source_id: str, repository: str, word: str) -> NaturalTypoRecord:
    typo_word = word[:2] + word[3:]
    return NaturalTypoRecord(
        source="github_typo_corpus",
        source_revision="a" * 64,
        source_split="corpus",
        source_id=source_id,
        group_id=repository,
        clean_text=f"The {word} context remains readable.",
        typo_text=f"The {typo_word} context remains readable.",
        repository=repository,
        repository_license="MIT",
        operation="deletion",
        training_eligible=True,
    )


def test_natural_lm_pairs_are_repository_held_out_while_injection_words_are_disjoint() -> None:
    protocol = replace(
        load_evaluation_study_protocol(STUDY),
        corpus_counts=MappingProxyType(
            {
                "tune": {"fineweb_edu": 0, "dolma": 0, "natural_pairs": 2},
                "pre_pr_gate": {"fineweb_edu": 0, "dolma": 0, "natural_pairs": 2},
                "final_test": {"fineweb_edu": 0, "dolma": 0, "natural_pairs": 3},
            }
        ),
    )
    train = _natural_record(source_id="train", repository="train/repo", word="shared")
    tune = tuple(
        _natural_record(source_id=f"tune-{index}", repository="tune/repo", word="shared")
        for index in range(2)
    )
    sealed = tuple(
        _natural_record(
            source_id=f"sealed-{index}",
            repository=f"sealed/repo-{index}",
            word=f"word{_letters(index)}",
        )
        for index in range(12)
    )
    exclusions = _Exclusions(
        hard_source_ids=frozenset(),
        hard_groups=frozenset(),
        prior_tune_source_ids=frozenset(record.source_id for record in tune),
        prior_tune_groups=frozenset((record.source, record.group_id) for record in tune),
        hard_near_duplicates=NearDuplicateTextIndex(()),
        prior_tune_near_duplicates=NearDuplicateTextIndex(()),
        training_repositories=frozenset({train.repository}),
        tune_repositories=frozenset({tune[0].repository}),
        artifact_sha256={},
    )

    selected, dictionaries = _select_natural(
        (train, *tune, *sealed), protocol=protocol, exclusions=exclusions
    )

    assert len(selected["tune"]) == 2
    assert all("shared" in record.clean_text for record in selected["tune"])
    assert "shared" not in dictionaries["tune"]
    assert len(selected["pre_pr_gate"]) == 2
    assert len(selected["final_test"]) == 3


def test_sealed_corpus_excludes_near_duplicates_of_prior_tune_text() -> None:
    protocol = replace(
        load_evaluation_study_protocol(STUDY),
        corpus_counts=MappingProxyType(
            {
                "tune": {"fineweb_edu": 0, "dolma": 0, "natural_pairs": 0},
                "pre_pr_gate": {"fineweb_edu": 1, "dolma": 1, "natural_pairs": 0},
                "final_test": {"fineweb_edu": 1, "dolma": 1, "natural_pairs": 0},
            }
        ),
    )

    def corpus_record(source: str, index: int, text: str) -> CleanRecord:
        return CleanRecord(
            source=source,
            source_revision="a" * 40,
            source_split="train",
            source_id=f"{source}-{index}",
            group_id=f"{source}-group-{index}",
            text=text,
            task=None,
            answer=None,
            metadata={"fixture": True},
        )

    prior_text = " ".join(f"token{index:04d}" for index in range(1000))
    near_prior = f"{prior_text[:-1]}8"
    collected = {
        source: (
            corpus_record(source, 0, near_prior),
            corpus_record(source, 1, f"Unique {source} document alpha with sufficient prose."),
            corpus_record(source, 2, f"Unique {source} document beta with different prose."),
        )
        for source in ("fineweb_edu", "dolma")
    }
    exclusions = _Exclusions(
        hard_source_ids=frozenset(),
        hard_groups=frozenset(),
        prior_tune_source_ids=frozenset(),
        prior_tune_groups=frozenset(),
        hard_near_duplicates=NearDuplicateTextIndex(()),
        prior_tune_near_duplicates=NearDuplicateTextIndex(
            (prior_text,), shingle_size=5, threshold=0.99
        ),
        training_repositories=frozenset(),
        tune_repositories=frozenset(),
        artifact_sha256={},
    )
    selected = _select_corpus(collected, protocol=protocol, exclusions=exclusions)

    assert all(
        record.text != near_prior
        for role in ("pre_pr_gate", "final_test")
        for source in ("fineweb_edu", "dolma")
        for record in selected[role][source]
    )


def test_freeze_writes_fixed_disjoint_primary_secondary_and_corpus_artifacts(
    tmp_path: Path,
) -> None:
    exclude = tmp_path / "exclude"
    _write_exclusions(exclude)
    source_protocol = load_training_data_config(SOURCES)
    output = tmp_path / "frozen"
    result = run_freeze_robustness_evaluation(
        FreezeEvaluationRunConfig(
            protocol_path=STUDY,
            source_config_path=SOURCES,
            exclude_data_dir=exclude,
            output_dir=output,
        ),
        source_provider=_Provider(source_protocol.sources),
    )

    assert result.registry_path == output / "registry.json"
    assert result.run_path == output / "run.json"
    assert {path.name for path in output.iterdir()} == {
        "tune_manifest.jsonl",
        "pre_pr_gate_manifest.jsonl",
        "final_test_manifest.jsonl",
        "tune_corpus_manifest.jsonl",
        "pre_pr_gate_corpus_manifest.jsonl",
        "final_test_corpus_manifest.jsonl",
        "registry.json",
        "run.json",
    }

    pre_pr = _rows(output / "pre_pr_gate_manifest.jsonl")
    final = _rows(output / "final_test_manifest.jsonl")
    primary_pre = [row for row in pre_pr if row["metadata"]["evaluation_condition"] == "random-2"]
    primary_final = [row for row in final if row["metadata"]["evaluation_condition"] == "random-2"]
    assert len(primary_pre) == 2_500
    assert len(primary_final) == 2_940
    assert {row["task"] for row in primary_pre} == {
        "gsm8k",
        "mmlu",
        "arc",
        "mmlu_pro",
        "commonsense_qa",
    }
    assert {row["task"] for row in primary_final} == {
        "gsm8k",
        "mmlu",
        "arc",
        "mmlu_pro",
        "math_500",
        "commonsense_qa",
    }
    assert all(row["edit_count"] == 2 for row in (*primary_pre, *primary_final))
    assert sum(row["metadata"]["mechanistic_audit"] for row in primary_pre) == 500
    assert sum(row["metadata"]["mechanistic_audit"] for row in primary_final) == 500
    assert all(
        row["typo_text"][row["typo_text"].find("\n") :]
        == row["clean_text"][row["clean_text"].find("\n") :]
        for row in primary_final
        if row["task"] in {"mmlu", "arc", "mmlu_pro", "commonsense_qa"}
    )
    assert sum(row["metadata"]["evaluation_condition"] == "random-1" for row in final) == 1_200
    assert sum(row["metadata"]["evaluation_condition"] == "random-4" for row in final) == 1_200
    assert sum(row["metadata"]["evaluation_condition"] == "transposition-2" for row in final) == 500
    assert (
        sum(row["metadata"]["evaluation_condition"] == "natural-injection" for row in final) == 500
    )

    pre_groups = {(row["source"], row["group_id"]) for row in primary_pre}
    final_groups = {(row["source"], row["group_id"]) for row in primary_final}
    assert pre_groups.isdisjoint(final_groups)
    registry = json.loads(result.registry_path.read_text(encoding="utf-8"))
    assert registry["schema_version"] == "robustness-evaluation-registry/v1"
    assert registry["protocol_sha256"] == hashlib.sha256(STUDY.read_bytes()).hexdigest()
    assert registry["roles"]["pre_pr_gate"]["maximum_openings"] == 1
    assert registry["roles"]["final_test"]["maximum_openings"] == 1
    assert registry["roles"]["final_test"]["task_primary_records"] == 2_940
    assert registry["roles"]["final_test"]["corpus_records"] == 3_000
    assert registry["opening_order"] == ["pre_pr_gate", "final_test"]
    assert registry["generator"] == "frozen-evaluation-typo/v3"
    assert registry["task_capacity_census"]["sealed"]["math_500"] == {
        "source_split_records": 500,
        "after_exclusions": 500,
        "typo_grid_eligible": 500,
        "transposition_eligible": 500,
        "required": 440,
    }
    assert registry["exclusion_data_protocol_sha256"] == registry["source_config_sha256"]
    assert registry["natural_evaluation_axes"] == {
        "language_model_pairs": "repository-disjoint/v1",
        "task_injection": "corrected-word-disjoint/v1",
        "corrected_word_split": {
            "train": 0.60,
            "tune": 0.10,
            "pre_pr_gate": 0.10,
            "final_test": 0.20,
        },
    }

    bundle = load_evaluation_bundle(
        output,
        evaluation_role="tune",
        splits=("same-task",),
        model="google/gemma-3-4b-it",
        model_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        access_binding_sha256="f" * 64,
        experiment_binding_sha256="e" * 64,
        output_dir=tmp_path / "evaluation-output",
        confirm_sealed_role=False,
        resume=False,
        study_protocol_sha256=hashlib.sha256(STUDY.read_bytes()).hexdigest(),
    )
    assert len(bundle.records) == 600
    assert (
        sum(record.metadata["evaluation_condition"] == "random-2" for record in bundle.records)
        == 500
    )
    assert (
        sum(
            record.metadata["evaluation_condition"] == "natural-injection"
            for record in bundle.records
        )
        == 100
    )
    corpus = load_evaluation_corpus_bundle(
        output,
        evaluation_role="tune",
        study_protocol_sha256=hashlib.sha256(STUDY.read_bytes()).hexdigest(),
        access_binding_sha256="f" * 64,
        experiment_binding_sha256="e" * 64,
        output_dir=tmp_path / "evaluation-output",
        confirm_sealed_role=False,
        resume=False,
    )
    assert len(corpus.records) == 300
    assert sum(record.kind == "clean-corpus" for record in corpus.records) == 200
    assert sum(record.kind == "natural" for record in corpus.records) == 100

    sealed_arguments = {
        "evaluation_role": "pre-pr-gate",
        "study_protocol_sha256": hashlib.sha256(STUDY.read_bytes()).hexdigest(),
        "access_binding_sha256": "a" * 64,
        "experiment_binding_sha256": "b" * 64,
        "output_dir": tmp_path / "sealed-evaluation-output",
        "confirm_sealed_role": True,
        "resume": False,
    }
    sealed_tasks = load_evaluation_bundle(
        output,
        splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
        model="google/gemma-3-4b-it",
        model_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        **sealed_arguments,
    )
    sealed_corpus = load_evaluation_corpus_bundle(output, **sealed_arguments)
    assert sealed_tasks.records
    assert sealed_corpus.records

    with pytest.raises(ValueError, match="completed passing pre-PR gate"):
        load_evaluation_corpus_bundle(
            output,
            evaluation_role="final-test",
            study_protocol_sha256=hashlib.sha256(STUDY.read_bytes()).hexdigest(),
            access_binding_sha256="f" * 64,
            experiment_binding_sha256="e" * 64,
            output_dir=tmp_path / "evaluation-output",
            confirm_sealed_role=True,
            resume=False,
        )


def test_freeze_replays_identical_bytes_and_rejects_nonempty_output(tmp_path: Path) -> None:
    exclude = tmp_path / "exclude"
    _write_exclusions(exclude)
    protocol = load_training_data_config(SOURCES)
    digests: list[dict[str, str]] = []
    for name in ("one", "two"):
        output = tmp_path / name
        result = run_freeze_robustness_evaluation(
            FreezeEvaluationRunConfig(
                protocol_path=STUDY,
                source_config_path=SOURCES,
                exclude_data_dir=exclude,
                output_dir=output,
            ),
            source_provider=_Provider(protocol.sources),
        )
        registry = json.loads(result.registry_path.read_text(encoding="utf-8"))
        digests.append(registry["artifact_sha256"])
    assert digests[0] == digests[1]


def test_freeze_rejects_source_config_not_bound_to_exclusion_data(tmp_path: Path) -> None:
    exclude = tmp_path / "exclude"
    _write_exclusions(exclude)
    payload = json.loads(SOURCES.read_text(encoding="utf-8"))
    payload["training"]["token_budget"] += 1
    changed = tmp_path / "changed-source-config.json"
    changed.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    protocol = load_training_data_config(changed)

    with pytest.raises(
        ValueError,
        match="source config differs from the exclusion data build",
    ):
        run_freeze_robustness_evaluation(
            FreezeEvaluationRunConfig(
                protocol_path=STUDY,
                source_config_path=changed,
                exclude_data_dir=exclude,
                output_dir=tmp_path / "frozen",
            ),
            source_provider=_Provider(protocol.sources),
        )
