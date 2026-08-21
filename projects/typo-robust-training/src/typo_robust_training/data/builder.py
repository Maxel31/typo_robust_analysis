"""Build hash-bound train, diagnostic, tune, and sealed evaluation manifests."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typo_robust_training.data.config import (
    DatasetSource,
    TrainingDataProtocol,
    load_training_data_config,
)
from typo_robust_training.data.natural_words import (
    natural_corrected_word,
    natural_dictionary_role_for_word,
)
from typo_robust_training.data.perturb import (
    TRAINING_OPERATIONS,
    TypoGenerator,
    eligible_word_spans,
)
from typo_robust_training.data.records import (
    CleanRecord,
    NaturalTypoRecord,
    TypoEdit,
    TypoPair,
)
from typo_robust_training.data.splits import (
    assign_balanced_group_roles,
    assign_content_splits,
    assign_repository_split,
    cluster_near_duplicates,
    normalized_content_sha256,
    stable_weighted_split,
    validate_group_disjointness,
)
from typo_robust_training.data.task_splits import (
    REASONING_DIAGNOSTIC_SPLITS,
    REASONING_TRAINING_SPLITS,
    TRAINING_DATA_EVALUATION_SPLITS,
)
from typo_robust_training.data.typo_stats import (
    derive_natural_typo_statistics,
    substitutions_from_statistics,
)


_RUN_SCHEMA = "build-robustness-training-data-run/v1"
_PUBLIC_OUTPUTS = (
    "training_sources.jsonl",
    "typo_statistics.json",
    "diagnostic_manifest.jsonl",
    "tune_manifest.jsonl",
    "pre_pr_gate_manifest.jsonl",
    "final_test_manifest.jsonl",
    "evaluation_manifest.json",
    "decontamination_report.json",
)
_REASONING_SOURCES = ("gsm8k", "mmlu", "arc")
_UNSEEN_SOURCES = ("dolma", "mmlu_pro", "math_500", "commonsense_qa")
_REASONING_V1_SPLITS = {
    "gsm8k": frozenset({"train"}),
    "mmlu": frozenset({"dev"}),
    "arc": frozenset({"train"}),
}


class DataSourceProvider(Protocol):
    """Injectable source boundary; unit tests never require network access."""

    def iter_records(
        self,
        source_name: str,
        source: DatasetSource,
    ) -> Iterable[CleanRecord | NaturalTypoRecord]: ...

    def provenance(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class BuildTrainingDataConfig:
    config_path: Path
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def public_arguments(self) -> dict[str, str]:
        return {
            "config_path": str(self.config_path.resolve()),
            "output_dir": str(self.output_dir.resolve()),
        }


@dataclass(frozen=True, slots=True)
class BuildTrainingDataResult:
    training_sources_path: Path
    typo_statistics_path: Path
    diagnostic_manifest_path: Path
    tune_manifest_path: Path
    pre_pr_gate_manifest_path: Path
    final_test_manifest_path: Path
    evaluation_manifest_path: Path
    decontamination_report_path: Path
    run_path: Path
    training_records: int
    training_tokens: int
    diagnostic_records: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_order_key(value: str, *, seed: int, namespace: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{namespace}\0{seed}\0{value}".encode("utf-8")).hexdigest()
    return digest, value


def source_collection_limit(source_name: str, protocol: TrainingDataProtocol) -> int:
    fixed = protocol.fixed_pairs_per_source_split
    if source_name == "fineweb_edu":
        target = protocol.training_token_budget * protocol.training_mixture["fineweb_edu"]
        expected_before_split = target / protocol.fineweb_content_split["train"]
        return max(5_000, int(expected_before_split // 64) + 1, fixed * 25)
    if source_name == "github_typo_corpus":
        return max(5_000, fixed * 100)
    if source_name in _REASONING_SOURCES:
        return max(5_000, protocol.diagnostic_per_task * 10 + fixed * 20)
    return max(1_000, fixed * 10)


def collect_sources(
    protocol: TrainingDataProtocol,
    provider: DataSourceProvider,
    *,
    record_preparer: Callable[[CleanRecord], CleanRecord] | None = None,
    source_names: Sequence[str] | None = None,
) -> dict[str, tuple[CleanRecord | NaturalTypoRecord, ...]]:
    collected: dict[str, tuple[CleanRecord | NaturalTypoRecord, ...]] = {}
    global_ids: set[str] = set()
    selected_names = tuple(protocol.sources) if source_names is None else tuple(source_names)
    if len(set(selected_names)) != len(selected_names) or any(
        name not in protocol.sources for name in selected_names
    ):
        raise ValueError("source collection names must be unique configured sources")
    for name in selected_names:
        source = protocol.sources[name]
        rows: list[CleanRecord | NaturalTypoRecord] = []
        split_counts: dict[str, int] = defaultdict(int)
        per_split_limit = source_collection_limit(name, protocol)
        iterator = iter(provider.iter_records(name, source))
        try:
            for record in iterator:
                if not isinstance(record, (CleanRecord, NaturalTypoRecord)):
                    raise TypeError(f"source provider emitted an invalid record for {name}")
                if isinstance(record, CleanRecord) and record_preparer is not None:
                    record = record_preparer(record)
                    if not isinstance(record, CleanRecord):
                        raise TypeError("record preparer must return a CleanRecord")
                if record.source != name or record.source_revision != source.revision:
                    raise ValueError(f"source provider provenance differs for {name}")
                if name != "github_typo_corpus" and record.source_split not in source.splits:
                    raise ValueError(f"source provider emitted an undeclared split for {name}")
                if record.record_id in global_ids:
                    raise ValueError(
                        f"source provider duplicated record identity: {record.record_id}"
                    )
                if split_counts[record.source_split] >= per_split_limit:
                    continue
                global_ids.add(record.record_id)
                rows.append(record)
                split_counts[record.source_split] += 1
                if all(split_counts[split] >= per_split_limit for split in source.splits):
                    break
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if not rows:
            raise ValueError(f"source provider returned no records for {name}")
        collected[name] = tuple(rows)
    return collected


def _clean_records(
    records: Sequence[CleanRecord | NaturalTypoRecord],
) -> tuple[CleanRecord, ...]:
    return tuple(record for record in records if isinstance(record, CleanRecord))


def _diagnostic_selection(
    records: Sequence[CleanRecord],
    *,
    count: int,
    seed: int,
    clusters: Mapping[str, str],
    excluded_clusters: set[str],
) -> tuple[CleanRecord, ...]:
    eligible = [
        record
        for record in records
        if record.source_split not in {"test", "validation-test"}
        and clusters[record.source_id] not in excluded_clusters
    ]
    ordered = sorted(
        eligible,
        key=lambda record: _stable_order_key(
            record.record_id,
            seed=seed,
            namespace="diagnostic-selection/v1",
        ),
    )
    selected: list[CleanRecord] = []
    seen_clusters: set[str] = set()
    for record in ordered:
        cluster = clusters[record.source_id]
        if cluster in seen_clusters:
            continue
        selected.append(record)
        seen_clusters.add(cluster)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"diagnostic source {records[0].source if records else 'unknown'} has fewer than "
            f"{count} disjoint records"
        )
    return tuple(selected)


def _clean_payload(
    record: CleanRecord, *, split: str, token_count: int | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "robustness-clean-record/v1",
        "kind": "clean",
        "record_id": record.record_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_split": record.source_split,
        "source_id": record.source_id,
        "group_id": record.group_id,
        "split": split,
        "text": record.text,
        "task": record.task,
        "answer": record.answer,
        "content_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
        "normalized_content_sha256": normalized_content_sha256(record.text),
        "metadata": dict(record.metadata),
    }
    if token_count is not None:
        payload["token_count"] = token_count
    return payload


def _natural_payload(
    record: NaturalTypoRecord, *, split: str, token_count: int | None = None
) -> dict[str, object]:
    metadata = dict(record.metadata)
    existing_condition = metadata.get("evaluation_condition")
    if existing_condition not in {None, "natural-lm-pair"}:
        raise ValueError("natural record evaluation condition conflicts with its payload")
    metadata["evaluation_condition"] = "natural-lm-pair"
    payload: dict[str, object] = {
        "schema_version": "robustness-natural-pair/v1",
        "kind": "natural",
        "record_id": record.record_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_split": record.source_split,
        "source_id": record.source_id,
        "group_id": record.group_id,
        "split": split,
        "clean_text": record.clean_text,
        "typo_text": record.typo_text,
        "task": None,
        "answer": None,
        "operation": record.operation,
        "training_eligible": record.training_eligible,
        "repository": record.repository,
        "repository_license": record.repository_license,
        "clean_sha256": hashlib.sha256(record.clean_text.encode("utf-8")).hexdigest(),
        "typo_sha256": hashlib.sha256(record.typo_text.encode("utf-8")).hexdigest(),
        "metadata": metadata,
    }
    if token_count is not None:
        payload["token_count"] = token_count
    return payload


def _select_to_token_budget(
    records: Sequence[CleanRecord | NaturalTypoRecord],
    *,
    target_tokens: int,
    seed: int,
    namespace: str,
    token_counter: Callable[[str], int],
    max_sequence_length: int,
) -> tuple[list[dict[str, object]], int]:
    ordered = sorted(
        records,
        key=lambda record: _stable_order_key(record.record_id, seed=seed, namespace=namespace),
    )
    rows: list[dict[str, object]] = []
    tokens = 0
    for record in ordered:
        text = record.typo_text if isinstance(record, NaturalTypoRecord) else record.text
        count = token_counter(text)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"token counter returned an invalid count for {record.record_id}")
        count = min(count, max_sequence_length)
        row = (
            _natural_payload(record, split="train", token_count=count)
            if isinstance(record, NaturalTypoRecord)
            else _clean_payload(record, split="train", token_count=count)
        )
        rows.append(row)
        tokens += count
        if tokens >= target_tokens:
            return rows, tokens
    raise ValueError(f"{namespace} has insufficient data for {target_tokens} tokens")


def _transpose_word(word: str) -> str | None:
    for index in range(len(word) - 1):
        if word[index] != word[index + 1]:
            return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    return None


def _heldout_transpositions(
    record: CleanRecord,
    *,
    edit_count: int,
) -> tuple[str, tuple[TypoEdit, ...]]:
    selected: list[tuple[int, int, str, str]] = []
    selected_words: set[str] = set()
    for start, stop in eligible_word_spans(record.text):
        clean_word = record.text[start:stop]
        normalized_word = clean_word.casefold()
        if normalized_word in selected_words:
            continue
        typo_word = _transpose_word(clean_word)
        if typo_word is None:
            continue
        selected.append((start, stop, clean_word, typo_word))
        selected_words.add(normalized_word)
        if len(selected) == edit_count:
            break
    if len(selected) != edit_count:
        raise ValueError(
            f"record has fewer than {edit_count} distinct adjacent-transposition targets: "
            f"{record.source_id}"
        )

    pieces: list[str] = []
    edits: list[TypoEdit] = []
    cursor = 0
    typo_length = 0
    for start, stop, clean_word, typo_word in selected:
        unchanged = record.text[cursor:start]
        pieces.extend((unchanged, typo_word))
        typo_length += len(unchanged)
        typo_start = typo_length
        typo_length += len(typo_word)
        edits.append(
            TypoEdit(
                operation="adjacent-transposition",
                clean_word=clean_word,
                typo_word=typo_word,
                clean_char_span=(start, stop),
                typo_char_span=(typo_start, typo_length),
            )
        )
        cursor = stop
    pieces.append(record.text[cursor:])
    return "".join(pieces), tuple(edits)


def _synthetic_pair_payload(
    record: CleanRecord,
    *,
    split: str,
    operation: str,
    protocol: TrainingDataProtocol,
    natural_substitutions: Mapping[str, Mapping[str, int]],
    variant: int,
) -> dict[str, object]:
    if operation == "adjacent-transposition":
        typo_text, edits = _heldout_transpositions(record, edit_count=2)
    else:
        pair = TypoGenerator(
            seed=protocol.seed,
            operation_weights=protocol.operation_probabilities,
            natural_substitutions=natural_substitutions,
            minimum_word_letters=protocol.minimum_word_letters,
        ).generate(
            record,
            epoch=0,
            variant=variant,
            force_operations=(operation,),
            force_edit_count=1,
        )
        typo_text, edits = pair.typo_text, pair.edits
    return _pair_payload(
        record,
        split=split,
        typo_text=typo_text,
        edits=edits,
        protocol=protocol,
        variant=variant,
    )


def _pair_payload(
    record: CleanRecord,
    *,
    split: str,
    typo_text: str,
    edits: Sequence[TypoEdit],
    protocol: TrainingDataProtocol,
    variant: int,
) -> dict[str, object]:
    edit_count = len(edits)
    metadata = dict(record.metadata)
    if split != "diagnostic":
        if edit_count not in {1, 2, 4}:
            raise ValueError("fixed synthetic evaluation pairs require 1, 2, or 4 edits")
        evaluation_condition = (
            "transposition-2"
            if edits and all(edit.operation == "adjacent-transposition" for edit in edits)
            else f"random-{edit_count}"
        )
        existing_condition = metadata.get("evaluation_condition")
        if existing_condition not in {None, evaluation_condition}:
            raise ValueError("clean record evaluation condition conflicts with its typo payload")
        metadata["evaluation_condition"] = evaluation_condition
    return {
        "schema_version": "robustness-fixed-typo-pair/v1",
        "kind": "synthetic",
        "record_id": record.record_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_split": record.source_split,
        "source_id": record.source_id,
        "group_id": record.group_id,
        "split": split,
        "clean_text": record.text,
        "typo_text": typo_text,
        "task": record.task,
        "answer": record.answer,
        "metadata": metadata,
        "operation": edits[0].operation if edit_count == 1 else "multiple",
        "operations": [edit.operation for edit in edits],
        "edit_count": edit_count,
        "generator_seed": protocol.seed,
        "generator_variant": variant,
        "edits": [
            {
                "operation": edit.operation,
                "clean_word": edit.clean_word,
                "typo_word": edit.typo_word,
                "clean_char_span": list(edit.clean_char_span),
                "typo_char_span": list(edit.typo_char_span),
            }
            for edit in edits
        ],
    }


def _diagnostic_rows(
    records: Sequence[CleanRecord],
    *,
    protocol: TrainingDataProtocol,
    natural_substitutions: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    generator = TypoGenerator(
        seed=protocol.seed,
        operation_weights=protocol.operation_probabilities,
        edit_count_weights=protocol.edit_count_probabilities,
        natural_substitutions=natural_substitutions,
        minimum_word_letters=protocol.minimum_word_letters,
    )
    rows: list[dict[str, object]] = []
    for variant, record in enumerate(sorted(records, key=lambda candidate: candidate.record_id)):
        pair: TypoPair = generator.generate(record, epoch=0, variant=variant)
        rows.append(
            _pair_payload(
                record,
                split="diagnostic",
                typo_text=pair.typo_text,
                edits=pair.edits,
                protocol=protocol,
                variant=variant,
            )
        )
    return rows


def _fixed_rows(
    clean_by_split: Mapping[str, Sequence[CleanRecord]],
    natural_by_split: Mapping[str, Sequence[NaturalTypoRecord]],
    *,
    protocol: TrainingDataProtocol,
    natural_substitutions: Mapping[str, Mapping[str, int]],
) -> dict[str, list[dict[str, object]]]:
    outputs = {"tune": [], "pre_pr_gate": [], "final_test": []}
    training_operations = tuple(sorted(TRAINING_OPERATIONS))
    for split in outputs:
        by_source: dict[str, list[CleanRecord]] = defaultdict(list)
        for record in clean_by_split.get(split, ()):
            by_source[record.source].append(record)
        for source, source_records in sorted(by_source.items()):
            ordered = sorted(
                source_records,
                key=lambda record: _stable_order_key(
                    record.record_id,
                    seed=protocol.seed,
                    namespace=f"fixed-{split}-{source}/v1",
                ),
            )[: protocol.fixed_pairs_per_source_split]
            for index, record in enumerate(ordered):
                operation = (
                    "adjacent-transposition"
                    if split != "tune" and index % 2 == 0
                    else training_operations[index % len(training_operations)]
                )
                outputs[split].append(
                    _synthetic_pair_payload(
                        record,
                        split=split,
                        operation=operation,
                        protocol=protocol,
                        natural_substitutions=natural_substitutions,
                        variant=index,
                    )
                )
        natural_by_source: dict[str, list[NaturalTypoRecord]] = defaultdict(list)
        for record in natural_by_split.get(split, ()):
            natural_by_source[record.source].append(record)
        for source, source_records in sorted(natural_by_source.items()):
            ordered_natural = sorted(
                source_records,
                key=lambda record: _stable_order_key(
                    record.record_id,
                    seed=protocol.seed,
                    namespace=f"fixed-natural-{split}-{source}/v1",
                ),
            )[: protocol.fixed_pairs_per_source_split]
            outputs[split].extend(
                _natural_payload(record, split=split) for record in ordered_natural
            )
        outputs[split].sort(key=lambda row: str(row["record_id"]))
    return outputs


def _partition_clean_records(
    collected: Mapping[str, Sequence[CleanRecord | NaturalTypoRecord]],
    *,
    protocol: TrainingDataProtocol,
) -> tuple[
    dict[str, list[CleanRecord]],
    tuple[CleanRecord, ...],
    dict[str, object],
]:
    all_clean = tuple(
        record
        for source_records in collected.values()
        for record in source_records
        if isinstance(record, CleanRecord)
    )
    clusters = cluster_near_duplicates(all_clean, shingle_size=5, threshold=0.99)
    legacy_v1 = protocol.schema_version == "robustness-training-data-config/v1"
    training_splits = _REASONING_V1_SPLITS if legacy_v1 else REASONING_TRAINING_SPLITS
    diagnostic_splits = _REASONING_V1_SPLITS if legacy_v1 else REASONING_DIAGNOSTIC_SPLITS
    by_split: dict[str, list[CleanRecord]] = defaultdict(list)
    diagnostic: list[CleanRecord] = []
    unseen_records = tuple(
        record
        for source_name in _UNSEEN_SOURCES
        for record in _clean_records(collected[source_name])
    )
    same_task_evaluation_records = tuple(
        record
        for source_name in _REASONING_SOURCES
        for record in _clean_records(collected[source_name])
        if (
            record.source_split not in training_splits[source_name]
            if legacy_v1
            else record.source_split in TRAINING_DATA_EVALUATION_SPLITS[source_name]
        )
    )
    evaluation_records = (*same_task_evaluation_records, *unseen_records)
    evaluation_clusters = {clusters[record.source_id] for record in evaluation_records}
    diagnostic_clusters: set[str] = set()

    for source_name in _REASONING_SOURCES:
        records = tuple(
            record
            for record in _clean_records(collected[source_name])
            if record.source_split in diagnostic_splits[source_name]
        )
        selected = _diagnostic_selection(
            records,
            count=protocol.diagnostic_per_task,
            seed=protocol.seed,
            clusters=clusters,
            excluded_clusters=evaluation_clusters | diagnostic_clusters,
        )
        diagnostic.extend(selected)
        diagnostic_clusters.update(clusters[record.source_id] for record in selected)

    general_sources = ("fineweb_edu", *_REASONING_SOURCES)
    general_records = tuple(
        record
        for source_name in general_sources
        for record in _clean_records(collected[source_name])
        if (source_name == "fineweb_edu" or record.source_split in training_splits[source_name])
        if clusters[record.source_id] not in diagnostic_clusters
        and clusters[record.source_id] not in evaluation_clusters
    )
    general_clusters = {record.source_id: clusters[record.source_id] for record in general_records}
    assignments = assign_content_splits(
        general_records,
        clusters=general_clusters,
        seed=protocol.seed,
        weights=protocol.fineweb_content_split,
    )
    validate_group_disjointness(
        general_records,
        assignments,
        clusters=general_clusters,
    )
    for record in general_records:
        by_split[assignments[record.source_id]].append(record)

    for record in evaluation_records:
        split = stable_weighted_split(
            clusters[record.source_id],
            seed=protocol.seed,
            namespace="evaluation-content-split/v1",
            weights={"pre_pr_gate": 0.5, "final_test": 0.5},
        )
        by_split[split].append(record)

    excluded_general_records = sum(
        1
        for source_name in general_sources
        for record in _clean_records(collected[source_name])
        if (source_name == "fineweb_edu" or record.source_split in training_splits[source_name])
        and clusters[record.source_id] in evaluation_clusters
    )

    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in by_split.items():
        for record in records:
            cluster_splits[clusters[record.source_id]].add(split)
    for record in diagnostic:
        cluster_splits[clusters[record.source_id]].add("diagnostic")
    leaked_clusters = {
        cluster: sorted(splits) for cluster, splits in cluster_splits.items() if len(splits) > 1
    }
    if leaked_clusters:
        raise ValueError("near-duplicate clusters cross scientific data roles")
    report = {
        "schema_version": "robustness-decontamination-report/v1",
        "normalization": "unicode-casefold-and-whitespace-collapse/v1",
        "near_duplicate_method": "character-5gram-minhash32-lsh8x4-exact-jaccard-0.99/v1",
        "input_clean_records": len(all_clean),
        "near_duplicate_clusters": len(set(clusters.values())),
        "diagnostic_clusters": len(diagnostic_clusters),
        "same_task_evaluation_records": len(same_task_evaluation_records),
        "unseen_evaluation_records": len(unseen_records),
        "evaluation_clusters": len(evaluation_clusters),
        "training_records_excluded_for_evaluation_overlap": excluded_general_records,
        "cross_role_cluster_violations": 0,
        "task_denylist": ["mmlu_pro", "math_500", "commonsense_qa"],
        "training_excludes_unseen_sources": list(_UNSEEN_SOURCES),
    }
    return by_split, tuple(sorted(diagnostic, key=lambda record: record.record_id)), report


def _partition_natural_records(
    records: Sequence[CleanRecord | NaturalTypoRecord],
    *,
    protocol: TrainingDataProtocol,
) -> dict[str, list[NaturalTypoRecord]]:
    natural = tuple(record for record in records if isinstance(record, NaturalTypoRecord))
    records_by_repository: dict[str, list[NaturalTypoRecord]] = defaultdict(list)
    for record in natural:
        records_by_repository[record.repository].append(record)
    if protocol.schema_version == "robustness-training-data-config/v1":
        repository_roles = {
            repository: assign_repository_split(
                repository,
                seed=protocol.seed,
                weights=protocol.natural_repository_split,
            )
            for repository in records_by_repository
        }
        held_out_roles = {
            repository: stable_weighted_split(
                repository,
                seed=protocol.seed,
                namespace="held-out-natural-evaluation/v1",
                weights=protocol.held_out_repository_evaluation_split,
            )
            for repository, role in repository_roles.items()
            if role == "held_out"
        }
    else:
        repository_roles = assign_balanced_group_roles(
            {
                repository: len(repository_records)
                for repository, repository_records in records_by_repository.items()
            },
            seed=protocol.seed,
            namespace="github-typo-repository-split/v2",
            weights=protocol.natural_repository_split,
        )
        held_out_roles = assign_balanced_group_roles(
            {
                repository: len(records_by_repository[repository])
                for repository, role in repository_roles.items()
                if role == "held_out"
            },
            seed=protocol.seed,
            namespace="held-out-natural-evaluation/v2",
            weights=protocol.held_out_repository_evaluation_split,
        )
    output: dict[str, list[NaturalTypoRecord]] = defaultdict(list)
    for record in natural:
        role = repository_roles[record.repository]
        if role == "train":
            corrected_word = (
                natural_corrected_word(record)
                if protocol.natural_dictionary_word_split is not None
                else None
            )
            dictionary_role = (
                natural_dictionary_role_for_word(
                    corrected_word,
                    seed=protocol.seed,
                    weights=protocol.natural_dictionary_word_split,
                )
                if corrected_word is not None and protocol.natural_dictionary_word_split is not None
                else None
            )
            if (
                record.training_eligible
                and record.operation in protocol.training_operations
                and record.operation not in protocol.held_out_operations
                and (protocol.natural_dictionary_word_split is None or dictionary_role == "train")
            ):
                output["train"].append(record)
        elif role == "tune":
            if record.operation not in protocol.held_out_operations:
                output["tune"].append(record)
        else:
            output[held_out_roles[record.repository]].append(record)
    for split in output:
        output[split].sort(key=lambda record: record.record_id)
    repositories_by_split: dict[str, set[str]] = defaultdict(set)
    for split, split_records in output.items():
        repositories_by_split[split].update(record.repository for record in split_records)
    splits = tuple(repositories_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if repositories_by_split[left] & repositories_by_split[right]:
                raise ValueError("natural typo repository crosses scientific splits")
    return output


def _build_training_rows(
    clean_by_split: Mapping[str, Sequence[CleanRecord]],
    natural_by_split: Mapping[str, Sequence[NaturalTypoRecord]],
    *,
    protocol: TrainingDataProtocol,
    token_counter: Callable[[str], int],
) -> tuple[list[dict[str, object]], int, dict[str, int], dict[str, dict[str, int]]]:
    pools: dict[str, Sequence[CleanRecord | NaturalTypoRecord]] = {
        "fineweb_edu": tuple(
            record for record in clean_by_split["train"] if record.source == "fineweb_edu"
        ),
        "reasoning": tuple(
            record for record in clean_by_split["train"] if record.source in _REASONING_SOURCES
        ),
        "natural_typo": tuple(natural_by_split.get("train", ())),
    }
    rows: list[dict[str, object]] = []
    token_counts: dict[str, int] = {}
    source_token_counts: dict[str, dict[str, int]] = {}
    for category, fraction in protocol.training_mixture.items():
        if fraction == 0.0:
            continue
        target = round(protocol.training_token_budget * fraction)
        if target <= 0:
            raise ValueError(f"positive training mixture {category} rounded to zero tokens")
        if (
            category == "reasoning"
            and protocol.schema_version != "robustness-training-data-config/v1"
        ):
            selected_by_task: list[dict[str, object]] = []
            task_tokens: dict[str, int] = {}
            for task, task_fraction in protocol.reasoning_task_mixture.items():
                if task_fraction == 0.0:
                    continue
                task_target = round(target * task_fraction)
                if task_target <= 0:
                    raise ValueError(f"positive reasoning mixture {task} rounded to zero tokens")
                selected, tokens = _select_to_token_budget(
                    tuple(record for record in pools[category] if record.source == task),
                    target_tokens=task_target,
                    seed=protocol.seed,
                    namespace=f"training-mixture-{category}-{task}/v1",
                    token_counter=token_counter,
                    max_sequence_length=protocol.max_sequence_length,
                )
                selected_by_task.extend(selected)
                task_tokens[task] = tokens
            rows.extend(selected_by_task)
            token_counts[category] = sum(task_tokens.values())
            source_token_counts[category] = task_tokens
            continue
        selected, tokens = _select_to_token_budget(
            pools[category],
            target_tokens=target,
            seed=protocol.seed,
            namespace=f"training-mixture-{category}/v1",
            token_counter=token_counter,
            max_sequence_length=protocol.max_sequence_length,
        )
        rows.extend(selected)
        token_counts[category] = tokens
        per_source: dict[str, int] = defaultdict(int)
        for row in selected:
            per_source[str(row["source"])] += int(row["token_count"])
        if sum(per_source.values()) != tokens:
            raise RuntimeError("selected source-token accounting differs from the token budget")
        source_token_counts[category] = dict(sorted(per_source.items()))
    rows.sort(key=lambda row: str(row["record_id"]))
    return rows, sum(token_counts.values()), token_counts, source_token_counts


def _assert_output_group_disjointness(
    groups_by_artifact: Mapping[str, set[tuple[str, str]]],
) -> None:
    names = tuple(groups_by_artifact)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups_by_artifact[left] & groups_by_artifact[right]
            if overlap:
                raise ValueError(f"data artifacts {left} and {right} share source groups")


def _result(output_dir: Path, counts: Mapping[str, int]) -> BuildTrainingDataResult:
    return BuildTrainingDataResult(
        training_sources_path=output_dir / "training_sources.jsonl",
        typo_statistics_path=output_dir / "typo_statistics.json",
        diagnostic_manifest_path=output_dir / "diagnostic_manifest.jsonl",
        tune_manifest_path=output_dir / "tune_manifest.jsonl",
        pre_pr_gate_manifest_path=output_dir / "pre_pr_gate_manifest.jsonl",
        final_test_manifest_path=output_dir / "final_test_manifest.jsonl",
        evaluation_manifest_path=output_dir / "evaluation_manifest.json",
        decontamination_report_path=output_dir / "decontamination_report.json",
        run_path=output_dir / "run.json",
        training_records=counts["training_records"],
        training_tokens=counts["training_tokens"],
        diagnostic_records=counts["diagnostic_records"],
    )


def run_build_training_data(
    config: BuildTrainingDataConfig,
    *,
    source_provider: DataSourceProvider | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> BuildTrainingDataResult:
    """Build all data roles, failing closed on leakage or provenance drift."""

    if not isinstance(config, BuildTrainingDataConfig):
        raise TypeError("config must be BuildTrainingDataConfig")
    protocol = load_training_data_config(config.config_path)
    output_dir = config.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_provider is None:
        from typo_robust_training.data.sources import HuggingFaceDataSourceProvider

        source_provider = HuggingFaceDataSourceProvider(protocol=protocol)
    if token_counter is None:
        from typo_robust_training.data.sources import HuggingFaceTokenCounter

        token_counter = HuggingFaceTokenCounter(
            model=protocol.model,
            revision=protocol.model_revision,
            max_sequence_length=protocol.max_sequence_length,
        )
    provider_provenance = dict(source_provider.provenance())
    token_counter_provenance = getattr(token_counter, "provenance", None)
    token_counter_record = (
        dict(token_counter_provenance())
        if callable(token_counter_provenance)
        else {"provider": "injected-token-counter/v1"}
    )
    started_at = _now()
    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "operation": "build-robustness-training-data",
        "status": "running",
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "source_provider": provider_provenance,
        "token_counter": token_counter_record,
        "started_at": started_at,
        "updated_at": started_at,
        "failures": [],
    }
    _write_json_atomic(output_dir / "run.json", base_run)
    try:
        prepare_record = getattr(token_counter, "prepare_record", None)
        collected = collect_sources(
            protocol,
            source_provider,
            record_preparer=prepare_record if callable(prepare_record) else None,
        )
        clean_by_split, diagnostic, decontamination = _partition_clean_records(
            collected,
            protocol=protocol,
        )
        natural_by_split = _partition_natural_records(
            collected["github_typo_corpus"],
            protocol=protocol,
        )
        natural_dictionary_word_counts: dict[str, int] | None = None
        natural_dictionary_record_counts: dict[str, int] | None = None
        if protocol.natural_dictionary_word_split is not None:
            words_by_role: dict[str, set[str]] = defaultdict(set)
            records_by_role: dict[str, int] = defaultdict(int)
            for record in collected["github_typo_corpus"]:
                if not isinstance(record, NaturalTypoRecord):
                    continue
                word = natural_corrected_word(record)
                if word is None:
                    continue
                role = natural_dictionary_role_for_word(
                    word,
                    seed=protocol.seed,
                    weights=protocol.natural_dictionary_word_split,
                )
                words_by_role[role].add(word)
                records_by_role[role] += 1
            natural_dictionary_word_counts = {
                role: len(words_by_role[role]) for role in protocol.natural_dictionary_word_split
            }
            natural_dictionary_record_counts = {
                role: records_by_role[role] for role in protocol.natural_dictionary_word_split
            }
            train_words = {
                word
                for record in natural_by_split.get("train", ())
                if (word := natural_corrected_word(record)) is not None
            }
            held_out_words = set().union(
                *(words_by_role[role] for role in words_by_role if role != "train")
            )
            if train_words & held_out_words:
                raise ValueError("natural corrected words cross training/evaluation roles")
        typo_statistics = derive_natural_typo_statistics(tuple(natural_by_split.get("train", ())))
        natural_substitutions = substitutions_from_statistics(typo_statistics)
        (
            training_rows,
            training_tokens,
            mixture_tokens,
            mixture_source_tokens,
        ) = _build_training_rows(
            clean_by_split,
            natural_by_split,
            protocol=protocol,
            token_counter=token_counter,
        )
        diagnostic_rows = _diagnostic_rows(
            diagnostic,
            protocol=protocol,
            natural_substitutions=natural_substitutions,
        )
        fixed = _fixed_rows(
            clean_by_split,
            natural_by_split,
            protocol=protocol,
            natural_substitutions=natural_substitutions,
        )
        artifact_rows = {
            "training_sources.jsonl": training_rows,
            "diagnostic_manifest.jsonl": diagnostic_rows,
            "tune_manifest.jsonl": fixed["tune"],
            "pre_pr_gate_manifest.jsonl": fixed["pre_pr_gate"],
            "final_test_manifest.jsonl": fixed["final_test"],
        }
        if any(not rows for rows in artifact_rows.values()):
            empty = sorted(name for name, rows in artifact_rows.items() if not rows)
            raise ValueError(f"data builder produced empty required artifacts: {empty}")
        _assert_output_group_disjointness(
            {
                name: {(str(row["source"]), str(row["group_id"])) for row in rows}
                for name, rows in artifact_rows.items()
            }
        )
        decontamination.update(
            {
                "natural_repository_split": dict(protocol.natural_repository_split),
                "natural_dictionary_word_split": (
                    dict(protocol.natural_dictionary_word_split)
                    if protocol.natural_dictionary_word_split is not None
                    else None
                ),
                "natural_dictionary_unique_word_counts": natural_dictionary_word_counts,
                "natural_dictionary_record_counts": natural_dictionary_record_counts,
                "natural_dictionary_training_evaluation_disjoint": (
                    protocol.natural_dictionary_word_split is not None
                ),
                "natural_repository_grouping": (
                    "exact-repository-url/v1"
                    if protocol.schema_version == "robustness-training-data-config/v1"
                    else "exact-repository-url-balanced-by-record-count/v2"
                ),
                "natural_repository_counts": {
                    split: len({record.repository for record in records})
                    for split, records in sorted(natural_by_split.items())
                },
                "natural_record_counts": {
                    split: len(records) for split, records in sorted(natural_by_split.items())
                },
                "artifact_group_disjoint": True,
            }
        )
        for name, rows in artifact_rows.items():
            _write_jsonl_atomic(output_dir / name, rows)
        _write_json_atomic(output_dir / "typo_statistics.json", typo_statistics)
        _write_json_atomic(output_dir / "decontamination_report.json", decontamination)

        row_hashes = {
            name: _sha256_file(output_dir / name)
            for name in (
                *artifact_rows,
                "typo_statistics.json",
                "decontamination_report.json",
            )
        }
        evaluation_manifest = {
            "schema_version": "robustness-evaluation-manifest/v1",
            "protocol_sha256": protocol.config_sha256,
            "source_revisions": {
                name: source.revision for name, source in sorted(protocol.sources.items())
            },
            "split_roles": {
                "tune": "iteration-only",
                "pre_pr_gate": "one-use-before-training-pr",
                "final_test": "sealed-until-method-freeze",
            },
            "training_operations": list(protocol.training_operations),
            "operation_probabilities": dict(protocol.operation_probabilities),
            "held_out_operations": list(protocol.held_out_operations),
            "pre_pr_gate_consumed": False,
            "final_test_opened": False,
            "artifact_sha256": row_hashes,
            "data_identity_sha256": _canonical_sha256(
                {
                    name: [str(row["record_id"]) for row in rows]
                    for name, rows in artifact_rows.items()
                }
            ),
        }
        _write_json_atomic(output_dir / "evaluation_manifest.json", evaluation_manifest)
        counts = {
            "training_records": len(training_rows),
            "training_tokens": training_tokens,
            "diagnostic_records": len(diagnostic_rows),
            "tune_records": len(fixed["tune"]),
            "pre_pr_gate_records": len(fixed["pre_pr_gate"]),
            "final_test_records": len(fixed["final_test"]),
        }
        outputs = {
            name: {
                "sha256": _sha256_file(output_dir / name),
                "records": (len(artifact_rows[name]) if name in artifact_rows else 1),
            }
            for name in _PUBLIC_OUTPUTS
        }
        completed = {
            **base_run,
            "status": "completed",
            "counts": counts,
            "mixture_tokens": mixture_tokens,
            "mixture_source_tokens": mixture_source_tokens,
            "outputs": outputs,
            "updated_at": _now(),
        }
        _write_json_atomic(output_dir / "run.json", completed)
        return _result(output_dir, counts)
    except Exception as exc:
        for name in _PUBLIC_OUTPUTS:
            (output_dir / name).unlink(missing_ok=True)
        failed = {
            **base_run,
            "status": "failed",
            "failures": [{"type": type(exc).__name__, "message": str(exc)}],
            "updated_at": _now(),
        }
        _write_json_atomic(output_dir / "run.json", failed)
        raise


__all__ = [
    "BuildTrainingDataConfig",
    "BuildTrainingDataResult",
    "DataSourceProvider",
    "collect_sources",
    "run_build_training_data",
    "source_collection_limit",
]
