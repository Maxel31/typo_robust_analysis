"""End-to-end fixture tests for the public data-building operation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from typo_robust_training.data.builder import (
    BuildTrainingDataConfig,
    DataSourceProvider,
    run_build_training_data,
)
from typo_robust_training.data.config import DatasetSource, load_training_data_config
from typo_robust_training.data.records import CleanRecord, NaturalTypoRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-sanity.yaml"


def _clean(source: str, revision: str, index: int, *, task: str | None) -> CleanRecord:
    task_label = task or "general educational prose"
    return CleanRecord(
        source=source,
        source_revision=revision,
        source_split="train",
        source_id=f"{source}-{index:04d}",
        group_id=f"{source}-group-{index:04d}",
        text=(
            f"This {task_label} example number {index} explains a reliable concept "
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
                operation = "adjacent-transposition" if index % 7 == 0 else "deletion"
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
            yield _clean(source_name, source.revision, index, task=task)

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


def _rows(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


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
    assert all(row["operation"] != "adjacent-transposition" for row in training if row["kind"] == "natural")
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
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["source_provider"] == {"provider": "offline-fixture/v1"}
    for filename, metadata in run["outputs"].items():
        path = output / filename
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


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
        "diagnostic_manifest.jsonl",
        "tune_manifest.jsonl",
        "pre_pr_gate_manifest.jsonl",
        "final_test_manifest.jsonl",
        "evaluation_manifest.json",
        "decontamination_report.json",
    ):
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()


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

