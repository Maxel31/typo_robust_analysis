"""Freeze disjoint generic-text pairs before confirmatory localization."""

from __future__ import annotations

import gzip
import hashlib
import platform
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.records import CleanRecord
from typo_robust_training.data.sources import segment_document
from typo_robust_training.localization.artifacts import (
    FIXED_TYPO_PAIR_FIELDS as _PAIR_FIELDS,
    sha256_file as _sha256_file,
    utc_now as _now,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from typo_robust_training.localization.confirmatory_config import (
    ConfirmatoryLocalizationProtocol,
    load_confirmatory_localization_config,
)
from typo_robust_training.localization.corpus_targets import clean_corpus_targets
from typo_robust_training.localization.prompting import word_final_token_positions


def _sha256_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenericLocalizationPairFreezeConfig:
    config_path: Path
    exclude_data_paths: tuple[Path, ...]
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(
            self,
            "exclude_data_paths",
            tuple(Path(path) for path in self.exclude_data_paths),
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.exclude_data_paths:
            raise ValueError("at least one --exclude-data path is required")


@dataclass(frozen=True, slots=True)
class GenericLocalizationPairFreezeResult:
    selection_manifest_path: Path
    validation_manifest_path: Path
    registry_path: Path
    run_path: Path


class GenericLocalizationPairProvider(Protocol):
    def iter_records(self, protocol: ConfirmatoryLocalizationProtocol) -> Iterable[CleanRecord]: ...

    def realize_pair(
        self,
        record: CleanRecord,
        *,
        operation: str,
        variant: int,
        protocol: ConfirmatoryLocalizationProtocol,
    ) -> dict[str, object] | None: ...

    def provenance(self) -> Mapping[str, object]: ...


def _flat_ids(value: object, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"tokenizer {field} must be one non-empty integer sequence")
    return tuple(value)


class HuggingFaceGenericLocalizationPairProvider:
    """Stream the pinned FineWeb-Edu revision and realize one eligible typo."""

    def __init__(self, *, protocol: ConfirmatoryLocalizationProtocol) -> None:
        from transformers import AutoTokenizer

        self.protocol = protocol
        self.tokenizer = AutoTokenizer.from_pretrained(
            protocol.model,
            revision=protocol.model_revision,
            use_fast=True,
        )
        self.generator = TypoGenerator(
            seed=protocol.source_seed,
            operation_weights={
                operation: 1.0 / len(protocol.typo_operations)
                for operation in protocol.typo_operations
            },
            edit_count_weights={"1": 1.0, "2": 0.0, "3-4": 0.0},
            explicit_clean_pair_probability=0.0,
            minimum_word_letters=protocol.minimum_word_letters,
        )
        self._datasets_version: str | None = None

    def iter_records(self, protocol: ConfirmatoryLocalizationProtocol) -> Iterable[CleanRecord]:
        if protocol != self.protocol:
            raise ValueError("pair provider protocol differs")
        import datasets
        from datasets import load_dataset

        self._datasets_version = datasets.__version__
        dataset = load_dataset(
            protocol.source_dataset,
            name=protocol.source_subset,
            split=protocol.source_split,
            revision=protocol.source_revision,
            streaming=True,
        )
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=protocol.source_seed, buffer_size=10_000)
        for index, row in enumerate(dataset):
            if not isinstance(row, Mapping):
                raise ValueError("FineWeb-Edu emitted a non-object row")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            upstream_id = str(row.get("id") or f"{protocol.source_split}-{index}")
            source_id = f"fineweb_edu:{upstream_id}"
            metadata = {
                "dataset_row_index": index,
                **{
                    key: row[key]
                    for key in ("url", "dump", "date", "language", "score", "token_count")
                    if key in row and isinstance(row[key], (str, int, float, bool, type(None)))
                },
            }
            record = CleanRecord(
                source="fineweb_edu",
                source_revision=protocol.source_revision,
                source_split=protocol.source_split,
                source_id=source_id,
                group_id=f"fineweb_edu:{row.get('id') or row.get('url') or upstream_id}",
                text=text.strip(),
                task=None,
                answer=None,
                metadata=metadata,
            )
            yield segment_document(record, character_limit=protocol.character_window)

    def _token_ids(self, text: str) -> tuple[int, ...]:
        encoded = self.tokenizer(text, add_special_tokens=True)
        if not isinstance(encoded, Mapping):
            raise ValueError("tokenizer output must be an object")
        return _flat_ids(encoded.get("input_ids"), field="input_ids")

    def _coordinate_is_usable(self, *, clean_text: str, typo_text: str, edit: Any) -> bool:
        clean_end, typo_end = edit.clean_char_span[1], edit.typo_char_span[1]
        clean_prefix, typo_prefix = clean_text[:clean_end], typo_text[:typo_end]
        clean_span = (edit.clean_char_span,)
        typo_span = (edit.typo_char_span,)
        try:
            clean_position = word_final_token_positions(
                self.tokenizer, text=clean_prefix, spans=clean_span
            )[-1]
            typo_position = word_final_token_positions(
                self.tokenizer, text=typo_prefix, spans=typo_span
            )[-1]
            clean_prompt_ids = self._token_ids(clean_prefix)
            typo_prompt_ids = self._token_ids(typo_prefix)
            full_clean_ids = self._token_ids(clean_text)
            targets, reason = clean_corpus_targets(
                full_clean_token_ids=full_clean_ids,
                clean_prompt_token_ids=clean_prompt_ids,
                final_edited_token=clean_position,
                count=self.protocol.minimum_following_tokens,
            )
        except ValueError:
            return False
        return (
            reason is None
            and len(targets) == self.protocol.minimum_following_tokens
            and typo_position == len(typo_prompt_ids) - 1
            and len(clean_prompt_ids) + len(targets) - 1 <= self.protocol.max_sequence_length
            and len(typo_prompt_ids) + len(targets) - 1 <= self.protocol.max_sequence_length
        )

    def realize_pair(
        self,
        record: CleanRecord,
        *,
        operation: str,
        variant: int,
        protocol: ConfirmatoryLocalizationProtocol,
    ) -> dict[str, object] | None:
        if protocol != self.protocol or operation not in protocol.typo_operations:
            raise ValueError("pair realization request differs from protocol")
        for attempt in range(128):
            realized_variant = variant * 128 + attempt
            try:
                pair = self.generator.generate(
                    record,
                    epoch=0,
                    variant=realized_variant,
                    force_operations=(operation,),
                    force_edit_count=1,
                )
            except ValueError:
                return None
            edit = pair.edits[0]
            if not self._coordinate_is_usable(
                clean_text=record.text,
                typo_text=pair.typo_text,
                edit=edit,
            ):
                continue
            return {
                "schema_version": "robustness-fixed-typo-pair/v1",
                "kind": "synthetic",
                "record_id": record.record_id,
                "source": record.source,
                "source_revision": record.source_revision,
                "source_split": record.source_split,
                "source_id": record.source_id,
                "group_id": record.group_id,
                "split": "candidate",
                "clean_text": record.text,
                "typo_text": pair.typo_text,
                "task": None,
                "answer": None,
                "metadata": {
                    **dict(record.metadata),
                    "localization_candidate_attempt": attempt,
                },
                "operation": operation,
                "operations": [operation],
                "edit_count": 1,
                "generator_seed": protocol.source_seed,
                "generator_variant": realized_variant,
                "edits": [
                    {
                        "operation": edit.operation,
                        "clean_word": edit.clean_word,
                        "typo_word": edit.typo_word,
                        "clean_char_span": list(edit.clean_char_span),
                        "typo_char_span": list(edit.typo_char_span),
                    }
                ],
            }
        return None

    def provenance(self) -> Mapping[str, object]:
        try:
            import transformers

            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = "not-installed"
        return {
            "provider": "pinned-fineweb-edu-confirmatory/v1",
            "dataset": self.protocol.source_dataset,
            "subset": self.protocol.source_subset,
            "split": self.protocol.source_split,
            "source_revision": self.protocol.source_revision,
            "model": self.protocol.model,
            "model_revision": self.protocol.model_revision,
            "datasets_version": self._datasets_version,
            "transformers_version": transformers_version,
            "streaming_shuffle_seed": self.protocol.source_seed,
            "streaming_shuffle_buffer": 10_000,
        }


def _exclusion_files(path: Path) -> tuple[Path, ...]:
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"exclusion path does not exist: {resolved}")
    if resolved.is_file():
        return (resolved,)
    files = tuple(
        sorted(
            candidate
            for candidate in resolved.rglob("*")
            if candidate.is_file()
            and (candidate.suffix == ".jsonl" or candidate.name.endswith(".jsonl.gz"))
        )
    )
    return files


def _iter_jsonl_objects(path: Path) -> Iterable[Mapping[str, object]]:
    if path.stat().st_size == 0:
        return
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = strict_loads(line, context=f"{path}:{line_number}")
                if not isinstance(payload, Mapping):
                    raise ValueError(f"exclusion JSONL row must be an object: {path}:{line_number}")
                yield payload
        return
    for line_number, line in read_lf_jsonl_lines(path, context="localization exclusion"):
        payload = strict_loads(line, context=f"{path}:{line_number}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"exclusion JSONL row must be an object: {path}:{line_number}")
        yield payload


def _load_exclusions(
    paths: tuple[Path, ...],
) -> tuple[set[str], set[str], set[str], list[dict[str, object]]]:
    record_ids: set[str] = set()
    group_ids: set[str] = set()
    source_ids: set[str] = set()
    files: list[dict[str, object]] = []
    seen_files: set[Path] = set()
    for path in paths:
        for file_path in _exclusion_files(path):
            if file_path in seen_files:
                continue
            seen_files.add(file_path)
            files.append(
                {
                    "path": str(file_path),
                    "sha256": _sha256_file(file_path),
                    "bytes": file_path.stat().st_size,
                }
            )
            for row in _iter_jsonl_objects(file_path):
                for field, destination in (
                    ("record_id", record_ids),
                    ("group_id", group_ids),
                    ("source_id", source_ids),
                ):
                    value = row.get(field)
                    if isinstance(value, str) and value:
                        destination.add(value)
    return record_ids, group_ids, source_ids, files


def _validate_pair(
    value: object,
    *,
    record: CleanRecord,
    operation: str,
    protocol: ConfirmatoryLocalizationProtocol,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PAIR_FIELDS:
        raise ValueError("localization pair fields differ")
    pair = dict(value)
    if (
        pair["schema_version"] != "robustness-fixed-typo-pair/v1"
        or pair["kind"] != "synthetic"
        or pair["record_id"] != record.record_id
        or pair["source"] != protocol.selection_source
        or pair["source_revision"] != protocol.source_revision
        or pair["source_id"] != record.source_id
        or pair["group_id"] != record.group_id
        or pair["split"] != "candidate"
        or pair["operation"] != operation
        or pair["operations"] != [operation]
        or pair["edit_count"] != 1
        or not isinstance(pair["edits"], list)
        or len(pair["edits"]) != 1
        or pair["clean_text"] == pair["typo_text"]
    ):
        raise ValueError("localization pair protocol binding differs")
    return pair


def run_freeze_generic_localization_pairs(
    config: GenericLocalizationPairFreezeConfig,
    *,
    provider: GenericLocalizationPairProvider | None = None,
) -> GenericLocalizationPairFreezeResult:
    if not isinstance(config, GenericLocalizationPairFreezeConfig):
        raise TypeError("config must be GenericLocalizationPairFreezeConfig")
    protocol = load_confirmatory_localization_config(config.config_path)
    output_dir = config.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"generic localization pair output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    record_ids, group_ids, source_ids, exclusion_files = _load_exclusions(config.exclude_data_paths)
    if provider is None:
        provider = HuggingFaceGenericLocalizationPairProvider(protocol=protocol)
    run_path = output_dir / "run.json"
    run_base: dict[str, object] = {
        "schema_version": "freeze-generic-localization-pairs-run/v1",
        "operation": "freeze-generic-localization-pairs",
        "config_path": str(protocol.config_path),
        "config_sha256": protocol.config_sha256,
        "exclude_data": [str(path.resolve()) for path in config.exclude_data_paths],
        "exclusion_files": exclusion_files,
        "excluded_record_ids": len(record_ids),
        "excluded_group_ids": len(group_ids),
        "excluded_source_ids": len(source_ids),
        "python": platform.python_version(),
        "started_at": _now(),
    }
    _write_json(run_path, {**run_base, "status": "running"})
    required = protocol.selection_records + protocol.validation_records
    accepted: list[dict[str, object]] = []
    seen_records: set[str] = set()
    seen_groups: set[str] = set()
    attempted = 0
    rejected_excluded = 0
    rejected_duplicate = 0
    rejected_unusable = 0
    try:
        for record in provider.iter_records(protocol):
            if not isinstance(record, CleanRecord):
                raise TypeError("localization pair provider must yield CleanRecord values")
            attempted += 1
            if (
                record.record_id in record_ids
                or record.group_id in group_ids
                or record.source_id in source_ids
            ):
                rejected_excluded += 1
                continue
            if record.record_id in seen_records or record.group_id in seen_groups:
                rejected_duplicate += 1
                continue
            operation = protocol.typo_operations[len(accepted) % len(protocol.typo_operations)]
            pair = provider.realize_pair(
                record,
                operation=operation,
                variant=len(accepted),
                protocol=protocol,
            )
            if pair is None:
                rejected_unusable += 1
                continue
            accepted.append(
                _validate_pair(pair, record=record, operation=operation, protocol=protocol)
            )
            seen_records.add(record.record_id)
            seen_groups.add(record.group_id)
            if len(accepted) == required:
                break
        if len(accepted) != required:
            raise ValueError(
                "insufficient eligible FineWeb-Edu localization records: "
                f"required={required}, accepted={len(accepted)}, attempted={attempted}"
            )
        selection = [
            {**row, "split": "localization-selection"}
            for row in accepted[: protocol.selection_records]
        ]
        validation = [
            {**row, "split": "localization-validation"}
            for row in accepted[protocol.selection_records :]
        ]
        selection_path = output_dir / "selection_manifest.jsonl"
        validation_path = output_dir / "validation_manifest.jsonl"
        _write_jsonl(selection_path, selection)
        _write_jsonl(validation_path, validation)
        provider_provenance = dict(provider.provenance())
        registry = {
            "schema_version": "robustness-generic-localization-pair-registry/v1",
            "config_sha256": protocol.config_sha256,
            "source": protocol.selection_source,
            "source_revision": protocol.source_revision,
            "roles_disjoint": True,
            "training_and_evaluation_exclusions_applied": True,
            "selection_manifest": {
                "path": selection_path.name,
                "records": len(selection),
                "sha256": _sha256_file(selection_path),
            },
            "validation_manifest": {
                "path": validation_path.name,
                "records": len(validation),
                "sha256": _sha256_file(validation_path),
            },
            "selection_record_ids_sha256": _sha256_values(
                str(row["record_id"]) for row in selection
            ),
            "validation_record_ids_sha256": _sha256_values(
                str(row["record_id"]) for row in validation
            ),
            "operations": list(protocol.typo_operations),
            "provider": provider_provenance,
        }
        registry_path = output_dir / "registry.json"
        _write_json(registry_path, registry)
        _write_json(
            run_path,
            {
                **run_base,
                "status": "completed",
                "completed_at": _now(),
                "attempted_records": attempted,
                "accepted_records": len(accepted),
                "rejected_excluded": rejected_excluded,
                "rejected_duplicate": rejected_duplicate,
                "rejected_unusable": rejected_unusable,
                "provider": provider_provenance,
                "outputs": {
                    path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
                    for path in (selection_path, validation_path, registry_path)
                },
            },
        )
    except BaseException as exc:
        _write_json(
            run_path,
            {
                **run_base,
                "status": "failed",
                "failed_at": _now(),
                "attempted_records": attempted,
                "accepted_records": len(accepted),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    return GenericLocalizationPairFreezeResult(
        selection_manifest_path=selection_path,
        validation_manifest_path=validation_path,
        registry_path=registry_path,
        run_path=run_path,
    )


__all__ = [
    "GenericLocalizationPairFreezeConfig",
    "GenericLocalizationPairFreezeResult",
    "GenericLocalizationPairProvider",
    "HuggingFaceGenericLocalizationPairProvider",
    "run_freeze_generic_localization_pairs",
]
