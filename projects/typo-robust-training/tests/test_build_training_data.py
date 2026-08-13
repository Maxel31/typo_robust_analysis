"""End-to-end fixture tests for the public data-building operation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_robust_training.data.builder import (
    BuildTrainingDataConfig,
    DataSourceProvider,
    _diagnostic_rows,
    _pair_payload,
    run_build_training_data,
)
from typo_robust_training.data.config import DatasetSource, load_training_data_config
from typo_robust_training.data.records import CleanRecord, NaturalTypoRecord, TypoEdit
from typo_robust_training.evaluation.data import EvaluationPair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-sanity.yaml"
CYCLE3_CONFIG = PROJECT_ROOT / "configs" / "cycle3" / "gemma4b-data-64m.yaml"


def _clean(
    source: str,
    revision: str,
    index: int,
    *,
    task: str | None,
    source_split: str,
) -> CleanRecord:
    task_label = task or "general educational prose"
    return CleanRecord(
        source=source,
        source_revision=revision,
        source_split=source_split,
        source_id=f"{source}-{index:04d}",
        group_id=f"{source}-group-{index:04d}",
        text=(
            f"This {source} {task_label} example number {index} explains a reliable concept "
            "with enough alphabetic words for deterministic perturbations."
        ),
        task=task,
        answer="A" if task is not None else None,
        metadata={"fixture": True},
    )


class _Provider(DataSourceProvider):
    def __init__(self, sources: Mapping[str, DatasetSource]) -> None:
        self._sources = sources
        self.calls: list[str] = []

    def iter_records(
        self,
        source_name: str,
        source: DatasetSource,
    ) -> Iterable[CleanRecord | NaturalTypoRecord]:
        self.calls.append(source_name)
        if source_name == "github_typo_corpus":
            for index in range(120):
                operation = (
                    "adjacent-transposition"
                    if index % 7 == 0
                    else "natural-statistics-substitution"
                    if index % 3 == 0
                    else "deletion"
                )
                yield NaturalTypoRecord(
                    source="github_typo_corpus",
                    source_revision=source.revision,
                    source_split="v1.0.0",
                    source_id=f"natural-{index:04d}",
                    group_id=f"https://github.com/fixture/repository-{index // 3}",
                    clean_text=f"The reliable airport example {index} remains readable.",
                    typo_text=(
                        f"The reliable ariport example {index} remains readable."
                        if operation == "adjacent-transposition"
                        else f"The reliable airpirt example {index} remains readable."
                        if operation == "natural-statistics-substitution"
                        else f"The reliable arport example {index} remains readable."
                    ),
                    repository=f"https://github.com/fixture/repository-{index // 3}",
                    repository_license="MIT",
                    operation=operation,
                    training_eligible=operation != "adjacent-transposition",
                    metadata={"fixture": True},
                )
            return
        task = source.task
        for index in range(120):
            yield _clean(
                source_name,
                source.revision,
                index,
                task=task,
                source_split=source.splits[index % len(source.splits)],
            )

    def provenance(self) -> Mapping[str, object]:
        return {"provider": "offline-fixture/v1"}


def _small_config(tmp_path: Path) -> Path:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["training"]["token_budget"] = 240
    payload["sampling"]["diagnostic_per_task"] = 2
    payload["sampling"]["fixed_pairs_per_source_split"] = 2
    payload["streaming"]["shuffle_buffer_size"] = 32
    path = tmp_path / "fixture-config.yaml"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _small_cycle3_config(tmp_path: Path) -> Path:
    payload = json.loads(CYCLE3_CONFIG.read_text(encoding="utf-8"))
    payload["training"]["token_budget"] = 240
    payload["sampling"]["diagnostic_per_task"] = 2
    payload["sampling"]["fixed_pairs_per_source_split"] = 2
    payload["streaming"]["shuffle_buffer_size"] = 32
    payload["splits"]["natural_dictionary_word"] = {
        "train": 1.0,
        "tune": 0.0,
        "pre_pr_gate": 0.0,
        "final_test": 0.0,
    }
    path = tmp_path / "cycle3-fixture-config.yaml"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _rows(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def test_builder_pair_payload_satisfies_evaluation_parser_condition_contract() -> None:
    record = CleanRecord(
        source="gsm8k",
        source_revision="b" * 40,
        source_split="test",
        source_id="gsm8k:test:1",
        group_id="gsm8k:test:1",
        text="The quick brown fox jumps over the lazy dog",
        task="gsm8k",
        answer="42",
    )
    edit = TypoEdit(
        operation="deletion",
        clean_word="quick",
        typo_word="quik",
        clean_char_span=(4, 9),
        typo_char_span=(4, 8),
    )
    row = _pair_payload(
        record,
        split="pre_pr_gate",
        typo_text="The quik brown fox jumps over the lazy dog",
        edits=(edit,),
        protocol=SimpleNamespace(seed=42),
        variant=0,
    )

    pair = EvaluationPair.from_dict(
        row,
        expected_role="pre_pr_gate",
        held_out_operations=frozenset({"adjacent-transposition"}),
    )

    assert pair.record_id == record.record_id
    assert pair.metadata["evaluation_condition"] == "random-1"


def test_builder_writes_hash_bound_disjoint_replayable_artifacts(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)
    protocol = load_training_data_config(config_path)
    provider = _Provider(protocol.sources)
    output = tmp_path / "run-a"
    result = run_build_training_data(
        BuildTrainingDataConfig(config_path=config_path, output_dir=output),
        source_provider=provider,
        token_counter=lambda text: len(text.split()),
    )

    assert result.training_tokens >= 240
    assert result.diagnostic_records == 6
    assert result.training_records > 0
    assert set(provider.calls) == set(protocol.sources)
    expected = {
        "training_sources.jsonl",
        "typo_statistics.json",
        "diagnostic_manifest.jsonl",
        "tune_manifest.jsonl",
        "pre_pr_gate_manifest.jsonl",
        "final_test_manifest.jsonl",
        "evaluation_manifest.json",
        "decontamination_report.json",
        "run.json",
    }
    assert {path.name for path in output.iterdir()} == expected

    training = _rows(result.training_sources_path)
    diagnostic = _rows(result.diagnostic_manifest_path)
    tune = _rows(result.tune_manifest_path)
    gate = _rows(result.pre_pr_gate_manifest_path)
    final = _rows(result.final_test_manifest_path)
    assert training and diagnostic and tune and gate and final
    assert all(row["split"] == "train" for row in training)
    assert {row["task"] for row in diagnostic} == {"gsm8k", "mmlu", "arc"}
    assert all(row["kind"] == "synthetic" for row in diagnostic)
    assert all(row["clean_text"] != row["typo_text"] for row in diagnostic)
    assert all(1 <= len(row["edits"]) <= 4 for row in diagnostic)
    assert all(isinstance(row["metadata"], dict) for row in diagnostic)
    assert all(
        row["metadata"]["evaluation_condition"]
        in {"random-1", "random-2", "random-4", "transposition-2", "natural-lm-pair"}
        for row in (*tune, *gate, *final)
        if row["kind"] != "clean"
    )
    allowed_training_splits = {"gsm8k": "train", "mmlu": "dev", "arc": "train"}
    for row in (*training, *diagnostic):
        if row["source"] in allowed_training_splits:
            assert row["source_split"] == allowed_training_splits[row["source"]]
    assert any(
        row["source"] in allowed_training_splits
        and row["source_split"] != allowed_training_splits[row["source"]]
        for row in (*gate, *final)
    )
    assert all(
        row["operation"] != "adjacent-transposition" for row in training if row["kind"] == "natural"
    )
    assert any(row["operation"] == "adjacent-transposition" for row in (*gate, *final))

    group_sets = [
        {(row["source"], row["group_id"]) for row in rows}
        for rows in (training, diagnostic, tune, gate, final)
    ]
    for index, left in enumerate(group_sets):
        for right in group_sets[index + 1 :]:
            assert left.isdisjoint(right)

    evaluation = json.loads(result.evaluation_manifest_path.read_text(encoding="utf-8"))
    assert evaluation["held_out_operations"] == ["adjacent-transposition"]
    assert evaluation["final_test_opened"] is False
    assert evaluation["pre_pr_gate_consumed"] is False
    assert evaluation["source_revisions"] == {
        name: source.revision for name, source in sorted(protocol.sources.items())
    }
    statistics = json.loads((output / "typo_statistics.json").read_text(encoding="utf-8"))
    assert statistics["source_role"] == "natural-training-repositories-only"
    assert statistics["input_records"] > 0
    assert statistics["operation_counts"]["adjacent-transposition"] == 0
    assert statistics["substitutions"]
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["source_provider"] == {"provider": "offline-fixture/v1"}
    category_sources = {
        "fineweb_edu": {"fineweb_edu"},
        "reasoning": {"gsm8k", "mmlu", "arc"},
        "natural_typo": {"github_typo_corpus"},
    }
    for category, sources in category_sources.items():
        expected: dict[str, int] = {}
        for row in training:
            source = str(row["source"])
            if source in sources:
                expected[source] = expected.get(source, 0) + int(row["token_count"])
        assert run["mixture_source_tokens"][category] == expected
    for filename, metadata in run["outputs"].items():
        path = output / filename
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("config_path", (DEFAULT_CONFIG, CYCLE3_CONFIG))
def test_diagnostic_manifest_accepts_three_edit_bucket(config_path: Path) -> None:
    protocol = load_training_data_config(config_path)
    records = tuple(
        _clean(
            "gsm8k",
            protocol.sources["gsm8k"].revision,
            index,
            task="gsm8k",
            source_split="train",
        )
        for index in range(protocol.diagnostic_per_task)
    )
    substitutions = {letter: {"x": 1} for letter in "abcdefghijklmnopqrstuvwxyz" if letter != "x"}

    rows = _diagnostic_rows(
        records,
        protocol=protocol,
        natural_substitutions=substitutions,
    )

    assert len(rows) == protocol.diagnostic_per_task
    assert any(row["edit_count"] == 3 for row in rows)


def test_builder_is_byte_deterministic_except_for_run_provenance(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)
    protocol = load_training_data_config(config_path)
    outputs: list[Path] = []
    for label in ("a", "b"):
        output = tmp_path / label
        run_build_training_data(
            BuildTrainingDataConfig(config_path=config_path, output_dir=output),
            source_provider=_Provider(protocol.sources),
            token_counter=lambda text: len(text.split()),
        )
        outputs.append(output)
    for filename in (
        "training_sources.jsonl",
        "typo_statistics.json",
        "diagnostic_manifest.jsonl",
        "tune_manifest.jsonl",
        "pre_pr_gate_manifest.jsonl",
        "final_test_manifest.jsonl",
        "evaluation_manifest.json",
        "decontamination_report.json",
    ):
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()


def test_cycle3_builder_uses_only_positive_mixture_sources_and_records_word_disjointness(
    tmp_path: Path,
) -> None:
    config_path = _small_cycle3_config(tmp_path)
    protocol = load_training_data_config(config_path)
    output = tmp_path / "cycle3"

    result = run_build_training_data(
        BuildTrainingDataConfig(config_path=config_path, output_dir=output),
        source_provider=_Provider(protocol.sources),
        token_counter=lambda text: len(text.split()),
    )

    assert {row["source"] for row in _rows(result.training_sources_path)} == {"fineweb_edu"}
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["mixture_tokens"] == {"fineweb_edu": result.training_tokens}
    assert run["mixture_source_tokens"] == {"fineweb_edu": {"fineweb_edu": result.training_tokens}}
    report = json.loads(result.decontamination_report_path.read_text(encoding="utf-8"))
    assert report["natural_dictionary_training_evaluation_disjoint"] is True
    assert report["natural_repository_grouping"].endswith("balanced-by-record-count/v2")


def test_builder_failure_is_visible_and_does_not_publish_final_artifacts(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)
    protocol = load_training_data_config(config_path)

    class _FailingProvider(_Provider):
        def iter_records(
            self,
            source_name: str,
            source: DatasetSource,
        ) -> Iterable[CleanRecord | NaturalTypoRecord]:
            if source_name == "mmlu":
                raise RuntimeError("injected source failure")
            yield from super().iter_records(source_name, source)

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected source failure"):
        run_build_training_data(
            BuildTrainingDataConfig(config_path=config_path, output_dir=output),
            source_provider=_FailingProvider(protocol.sources),
            token_counter=lambda text: len(text.split()),
        )
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["failures"][0]["type"] == "RuntimeError"
    assert not (output / "evaluation_manifest.json").exists()
