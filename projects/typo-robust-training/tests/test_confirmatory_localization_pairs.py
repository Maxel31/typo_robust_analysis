"""Frozen, disjoint generic-text localization pair construction."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from typo_robust_training.data.perturb import apply_typo_operation_to_word
from typo_robust_training.data.records import CleanRecord
from typo_robust_training.localization.confirmatory_config import (
    ConfirmatoryLocalizationProtocol,
    load_confirmatory_localization_config,
)
from typo_robust_training.localization.confirmatory_pairs import (
    GenericLocalizationPairFreezeConfig,
    run_freeze_generic_localization_pairs,
)
from typo_robust_training.localization.corpus_targets import clean_corpus_targets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cycle3" / "gemma4b-generic-joint-window.yaml"


def _record(index: int) -> CleanRecord:
    return CleanRecord(
        source="fineweb_edu",
        source_revision="fc9850dff5e2d0f8f776efe41b24a1c49556cfc5",
        source_split="train",
        source_id=f"fineweb_edu:doc-{index}",
        group_id=f"fineweb_edu:group-{index}",
        text="Airport systems provide reliable transport for communities and schools.",
        task=None,
        answer=None,
        metadata={"dataset_row_index": index},
    )


class _Provider:
    def __init__(self, records: tuple[CleanRecord, ...]) -> None:
        self.records = records
        self.realized: list[tuple[str, str, int]] = []

    def iter_records(self, protocol: ConfirmatoryLocalizationProtocol) -> tuple[CleanRecord, ...]:
        assert protocol.selection_source == "fineweb_edu"
        return self.records

    def realize_pair(
        self,
        record: CleanRecord,
        *,
        operation: str,
        variant: int,
        protocol: ConfirmatoryLocalizationProtocol,
    ) -> dict[str, object] | None:
        del protocol
        start, stop = 0, len("Airport")
        typo_word = apply_typo_operation_to_word(
            record.text[start:stop], operation, random.Random(variant + 99)
        )
        typo_text = typo_word + record.text[stop:]
        self.realized.append((record.record_id, operation, variant))
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
            "typo_text": typo_text,
            "task": None,
            "answer": None,
            "metadata": dict(record.metadata),
            "operation": operation,
            "operations": [operation],
            "edit_count": 1,
            "generator_seed": 42,
            "generator_variant": variant,
            "edits": [
                {
                    "operation": operation,
                    "clean_word": "Airport",
                    "typo_word": typo_word,
                    "clean_char_span": [start, stop],
                    "typo_char_span": [start, len(typo_word)],
                }
            ],
        }

    def provenance(self) -> dict[str, object]:
        return {"provider": "offline-confirmatory-fixture/v1"}


def _small_config(tmp_path: Path) -> Path:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["data"]["selection_records"] = 2
    payload["data"]["validation_records"] = 2
    payload["selection"]["minimum_eligible"] = 1
    payload["statistics"]["bootstrap_replicates"] = 20
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pair_freeze_excludes_prior_ids_balances_operations_and_hash_binds_outputs(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index) for index in range(6))
    excluded = tmp_path / "excluded"
    excluded.mkdir()
    (excluded / "training_sources.jsonl").write_text(
        json.dumps(
            {
                "record_id": records[0].record_id,
                "group_id": records[0].group_id,
                "source_id": records[0].source_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = _Provider(records)
    output = tmp_path / "pairs"

    result = run_freeze_generic_localization_pairs(
        GenericLocalizationPairFreezeConfig(
            config_path=_small_config(tmp_path),
            exclude_data_paths=(excluded,),
            output_dir=output,
        ),
        provider=provider,
    )

    selection = _rows(result.selection_manifest_path)
    validation = _rows(result.validation_manifest_path)
    assert len(selection) == len(validation) == 2
    assert {row["split"] for row in selection} == {"localization-selection"}
    assert {row["split"] for row in validation} == {"localization-validation"}
    assert not ({row["record_id"] for row in selection} & {row["record_id"] for row in validation})
    assert records[0].record_id not in {row["record_id"] for row in (*selection, *validation)}
    operations = [str(row["operation"]) for row in (*selection, *validation)]
    counts = {operation: operations.count(operation) for operation in set(operations)}
    assert max(counts.values()) - min(counts.values()) <= 1

    registry = json.loads(result.registry_path.read_text(encoding="utf-8"))
    assert registry["selection_manifest"]["records"] == 2
    assert registry["validation_manifest"]["records"] == 2
    assert len(registry["selection_manifest"]["sha256"]) == 64
    assert registry["selection_record_ids_sha256"] != registry["validation_record_ids_sha256"]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["excluded_record_ids"] >= 1
    assert run["provider"]["provider"] == "offline-confirmatory-fixture/v1"

    with pytest.raises(FileExistsError, match="not empty"):
        run_freeze_generic_localization_pairs(
            GenericLocalizationPairFreezeConfig(
                config_path=_small_config(tmp_path),
                exclude_data_paths=(excluded,),
                output_dir=output,
            ),
            provider=_Provider(records),
        )


def test_pair_freeze_fails_closed_when_unique_capacity_is_insufficient(tmp_path: Path) -> None:
    records = tuple(_record(index) for index in range(3))
    excluded = tmp_path / "excluded"
    excluded.mkdir()
    (excluded / "empty.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="insufficient eligible FineWeb-Edu"):
        run_freeze_generic_localization_pairs(
            GenericLocalizationPairFreezeConfig(
                config_path=_small_config(tmp_path),
                exclude_data_paths=(excluded,),
                output_dir=tmp_path / "pairs",
            ),
            provider=_Provider(records),
        )


def test_pair_freeze_requires_every_declared_exclusion_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exclusion path does not exist"):
        run_freeze_generic_localization_pairs(
            GenericLocalizationPairFreezeConfig(
                config_path=_small_config(tmp_path),
                exclude_data_paths=(tmp_path / "missing",),
                output_dir=tmp_path / "pairs",
            ),
            provider=_Provider(tuple(_record(index) for index in range(6))),
        )


def test_confirmatory_config_fixes_model_sequence_limit() -> None:
    assert load_confirmatory_localization_config(DEFAULT_CONFIG).max_sequence_length == 512


def test_corpus_targets_begin_after_a_stable_edited_word_prefix() -> None:
    targets, reason = clean_corpus_targets(
        full_clean_token_ids=(10, 20, 30, 40, 50),
        clean_prompt_token_ids=(10, 20),
        final_edited_token=1,
        count=3,
    )

    assert targets == (30, 40, 50)
    assert reason is None


def test_corpus_targets_reject_unstable_or_short_continuations() -> None:
    with pytest.raises(ValueError, match="stable prefix"):
        clean_corpus_targets(
            full_clean_token_ids=(10, 99, 30),
            clean_prompt_token_ids=(10, 20),
            final_edited_token=1,
            count=1,
        )

    targets, reason = clean_corpus_targets(
        full_clean_token_ids=(10, 20, 30),
        clean_prompt_token_ids=(10, 20),
        final_edited_token=1,
        count=2,
    )
    assert targets == ()
    assert reason == "fewer-than-16-clean-corpus-tokens-after-final-edit"
