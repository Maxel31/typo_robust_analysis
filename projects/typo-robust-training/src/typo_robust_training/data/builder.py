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
    assign_content_splits,
    assign_repository_split,
    cluster_near_duplicates,
    normalized_content_sha256,
    stable_weighted_split,
    validate_group_disjointness,
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
_REASONING_TRAINING_SPLITS = {
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


def _collect_limit(source_name: str, protocol: TrainingDataProtocol) -> int:
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


def _collect_sources(
    protocol: TrainingDataProtocol,
    provider: DataSourceProvider,
    *,
    record_preparer: Callable[[CleanRecord], CleanRecord] | None = None,
) -> dict[str, tuple[CleanRecord | NaturalTypoRecord, ...]]:
    collected: dict[str, tuple[CleanRecord | NaturalTypoRecord, ...]] = {}
    global_ids: set[str] = set()
    for name, source in protocol.sources.items():
        rows: list[CleanRecord | NaturalTypoRecord] = []
        split_counts: dict[str, int] = defaultdict(int)
        per_split_limit = _collect_limit(name, protocol)
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
        "metadata": dict(record.metadata),
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


def _heldout_transposition(record: CleanRecord) -> tuple[str, TypoEdit]:
    for start, stop in eligible_word_spans(record.text):
        clean_word = record.text[start:stop]
        typo_word = _transpose_word(clean_word)
        if typo_word is None:
            continue
        typo_text = record.text[:start] + typo_word + record.text[stop:]
        return typo_text, TypoEdit(
            operation="adjacent-transposition",
            clean_word=clean_word,
            typo_word=typo_word,
            clean_char_span=(start, stop),
            typo_char_span=(start, start + len(typo_word)),
        )
    raise ValueError(f"record has no adjacent-transposition target: {record.source_id}")


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
        typo_text, edit = _heldout_transposition(record)
        edits = (edit,)
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
        "metadata": dict(record.metadata),
        "operation": edits[0].operation if len(edits) == 1 else "multiple",
        "operations": [edit.operation for edit in edits],
        "edit_count": len(edits),
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
        if record.source_split not in _REASONING_TRAINING_SPLITS[source_name]
    )
    evaluation_records = (*same_task_evaluation_records, *unseen_records)
    evaluation_clusters = {clusters[record.source_id] for record in evaluation_records}
    diagnostic_clusters: set[str] = set()

    for source_name in _REASONING_SOURCES:
        records = tuple(
            record
            for record in _clean_records(collected[source_name])
            if record.source_split in _REASONING_TRAINING_SPLITS[source_name]
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
        if (
            source_name == "fineweb_edu"
            or record.source_split in _REASONING_TRAINING_SPLITS[source_name]
        )
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
        if (
            source_name == "fineweb_edu"
            or record.source_split in _REASONING_TRAINING_SPLITS[source_name]
        )
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
    repository_roles: dict[str, str] = {}
    output: dict[str, list[NaturalTypoRecord]] = defaultdict(list)
    for record in natural:
        role = repository_roles.setdefault(
            record.repository,
            assign_repository_split(
                record.repository,
                seed=protocol.seed,
                weights=protocol.natural_repository_split,
            ),
        )
        if role == "train":
            if record.training_eligible and record.operation not in protocol.held_out_operations:
                output["train"].append(record)
        elif role == "tune":
            if record.operation not in protocol.held_out_operations:
                output["tune"].append(record)
        else:
            evaluation_split = stable_weighted_split(
                record.repository,
                seed=protocol.seed,
                namespace="held-out-natural-evaluation/v1",
                weights=protocol.held_out_repository_evaluation_split,
            )
            output[evaluation_split].append(record)
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
) -> tuple[list[dict[str, object]], int, dict[str, int]]:
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
    for category, fraction in protocol.training_mixture.items():
        target = max(1, round(protocol.training_token_budget * fraction))
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
    rows.sort(key=lambda row: str(row["record_id"]))
    return rows, sum(token_counts.values()), token_counts


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
    started_at = _now()
    base_run: dict[str, object] = {
        "schema_version": _RUN_SCHEMA,
        "operation": "build-robustness-training-data",
        "status": "running",
        "arguments": config.public_arguments(),
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.config_sha256,
        "source_provider": provider_provenance,
        "started_at": started_at,
        "updated_at": started_at,
        "failures": [],
    }
    _write_json_atomic(output_dir / "run.json", base_run)
    try:
        prepare_record = getattr(token_counter, "prepare_record", None)
        collected = _collect_sources(
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
        typo_statistics = derive_natural_typo_statistics(tuple(natural_by_split.get("train", ())))
        natural_substitutions = substitutions_from_statistics(typo_statistics)
        training_rows, training_tokens, mixture_tokens = _build_training_rows(
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
                "natural_repository_grouping": "exact-repository-url/v1",
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
    "run_build_training_data",
]
