"""Complete-grid, micro-pooling, and paired-test contracts for Table 13."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_cot.experiments.restoration_order_accuracy import aggregation
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.restoration_order_accuracy.aggregation import (
    BuildRestorationOrderTableConfig,
    RestorationOrderTableInputError,
    build_restoration_order_table,
)
from typo_cot.experiments.restoration_order_accuracy.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.restoration_order_accuracy.planning import (
    build_restoration_plan,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    ALL_CONDITION_IDS,
    EditGroup,
    GENERATION,
    PAPER_BENCHMARKS,
    PAPER_BUDGETS,
    PAPER_MODELS,
    PAPER_ORDERS,
    PAPER_SOURCE_RECORD_COUNTS,
    PROTOCOL_SHA256,
    build_edit_groups,
    canonical_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _source_record_from_public_row(row: dict[str, object]) -> dict[str, object]:
    arms = row["arms"]
    context = row["prompt_context"]
    prefix_length = context["prefix_length"]
    suffix_length = context["suffix_length"]

    def endpoint(condition_id: str, *, correct: bool) -> dict[str, object]:
        arm = arms[condition_id]
        prompt = arm["prompt"]
        return {
            "editable_text": arm["editable_text"],
            "prompt": prompt,
            "editable_prompt_span": {
                "start": prefix_length,
                "end": len(prompt) - suffix_length,
            },
            "answer": {"is_correct": correct},
        }

    return {
        "sample_id": row["sample_id"],
        "gold_answer": row["gold_answer"],
        "target_attempts": [
            {
                "target_token_index": group["target_token_index"],
                "selection_rank": group["selection_rank"],
                "relevance": group["relevance"],
            }
            for group in row["edit_groups"]
        ],
        "clean": endpoint("clean:k4", correct=True),
        "edited": endpoint("edited:k0", correct=False),
    }


class _FixtureValidatedSource:
    def __init__(
        self,
        *,
        records: tuple[dict[str, object], ...],
        plans: tuple[object, ...],
        payload: dict[str, object],
    ) -> None:
        self.records = records
        self.plans = plans
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)

    def assert_unchanged(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _install_validated_source_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(
        pairs_path: Path,
        *,
        model: str,
        benchmark: str,
        limit: int | None = None,
    ) -> _FixtureValidatedSource:
        del model, benchmark
        assert limit is None
        setting = Path(pairs_path).parent.parent
        run = _read_json(setting / "run.json")
        public_rows = [
            json.loads(line)
            for line in (setting / "restoration_order_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if json.loads(line).get("schema_version")
            == "restoration-order-record/v1"
        ]
        records = tuple(_source_record_from_public_row(row) for row in public_rows)
        plans = []
        for record in records:
            groups = build_edit_groups(
                record["clean"]["editable_text"],
                record["edited"]["editable_text"],
                record["target_attempts"],
            )
            plans.append(
                build_restoration_plan(
                    sample_id=record["sample_id"],
                    clean_text=record["clean"]["editable_text"],
                    edited_text=record["edited"]["editable_text"],
                    groups=groups,
                    seed=42,
                )
            )
        return _FixtureValidatedSource(
            records=records,
            plans=tuple(plans),
            payload=dict(run["source"]),
        )

    monkeypatch.setattr(aggregation, "load_restoration_order_source", load, raising=False)


def _write_setting(root: Path, model: str, benchmark: str) -> Path:
    directory = root / model.rsplit("/", 1)[-1].lower() / benchmark
    directory.mkdir(parents=True)
    records_path = directory / "restoration_order_records.jsonl"
    rows = []
    for sample_index in range(2):
        clean_text = "a b c d"
        edited_text = "A B C D"
        groups = build_edit_groups(
            clean_text,
            edited_text,
            [
                {
                    "target_token_index": index,
                    "selection_rank": index + 1,
                    "relevance": float(4 - index),
                }
                for index in range(4)
            ],
        )
        sample_id = f"sample-{sample_index}"
        plan = build_restoration_plan(
            sample_id=sample_id,
            clean_text=clean_text,
            edited_text=edited_text,
            groups=groups,
            seed=42,
        )
        conditions = {
            condition.condition_id: condition for condition in plan.conditions
        }
        prefix = "Question: "
        suffix = "\nAnswer:"
        arms: dict[str, object] = {}
        gold_answer = "4" if benchmark == "gsm8k" else "A"
        wrong_answer = "9" if benchmark == "gsm8k" else "B"
        for condition_id in ALL_CONDITION_IDS:
            condition = conditions[condition_id]
            if condition_id == "clean:k4":
                correct = True
            elif condition_id.startswith("high-relevance-first:"):
                correct = sample_index == 0
            else:
                correct = False
            prompt = prefix + condition.text + suffix
            extracted_answer = gold_answer if correct else wrong_answer
            arms[condition_id] = {
                "order": condition.order,
                "budget": condition.budget,
                "restored_group_indices": list(condition.restored_group_indices),
                "editable_text": condition.text,
                "editable_text_sha256": hashlib.sha256(
                    condition.text.encode()
                ).hexdigest(),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "sample_id": sample_id,
                "is_extracted": True,
                "is_correct": correct,
                "extracted_answer": extracted_answer,
                "method": "primary:pattern_1",
                "primary_method": "pattern_1",
                "token_ids": [41, 99],
                "text": f"The answer is {extracted_answer}",
                "termination": "eos",
                "individual_retry": False,
            }
        row = {
                "schema_version": "restoration-order-record/v1",
                "sample_id": sample_id,
                "model": model,
                "benchmark": benchmark,
                "gold_answer": gold_answer,
                "realized_edit_groups": 4,
                "source_selection": {
                    "clean_correct": True,
                    "edited_wrong": True,
                    "separable": True,
                },
                "source_record_sha256": "pending",
                "prompt_context": {
                    "prefix_length": len(prefix),
                    "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                    "suffix_length": len(suffix),
                    "suffix_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
                },
                "order_group_indices": {
                    name: list(indices) for name, indices in plan.order_indices
                },
                "edit_groups": [group.to_dict() for group in plan.groups],
                "plan_sha256": plan.to_dict()["sha256"],
                "arms": arms,
            }
        row["source_record_sha256"] = canonical_sha256(
            _source_record_from_public_row(row)
        )
        rows.append(row)
    records_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary_path = directory / "restoration_order_summary.json"
    conditions = {
        condition: {
            "correct": sum(int(row["arms"][condition]["is_correct"]) for row in rows),
            "extracted": sum(
                int(row["arms"][condition]["is_extracted"]) for row in rows
            ),
            "total": len(rows),
            "accuracy": sum(
                int(row["arms"][condition]["is_correct"]) for row in rows
            )
            / len(rows),
        }
        for condition in ALL_CONDITION_IDS
    }
    _write_json(
        summary_path,
        {
            "schema_version": "restoration-order-summary/v1",
            "model": model,
            "benchmark": benchmark,
            "cohort": {
                "source_records": PAPER_SOURCE_RECORD_COUNTS[benchmark],
                "source_flip": 2,
                "separable": 2,
                "selected": 2,
                "limited": False,
            },
            "conditions": conditions,
        },
    )
    run_path = directory / "run.json"
    _write_json(
        run_path,
        {
            "schema_version": "restoration-order-run/v1",
            "paper_sha256": PAPER_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "operation": "restoration-order-accuracy",
            "run_id": "c" * 32,
            "status": "completed",
            "started_at": "2026-08-09T00:00:00+00:00",
            "updated_at": "2026-08-09T00:01:00+00:00",
            "completed_at": "2026-08-09T00:01:00+00:00",
            "arguments": {
                "model": model,
                "benchmark": benchmark,
                "pairs": str((directory / "source" / "pairs.jsonl").resolve()),
                "orders": list(PAPER_ORDERS),
                "budgets": list(PAPER_BUDGETS),
                "seed": 42,
                "batch_size": 8,
                "gpu_id": "1",
                "limit": None,
                "output_dir": str(directory.resolve()),
            },
            "source": {
                "input_kind": "completed-prepare-edited-pairs/v1",
                "model": model,
                "benchmark": benchmark,
                "model_revision": "1" * 40,
                "records": PAPER_SOURCE_RECORD_COUNTS[benchmark],
                "source_records": PAPER_SOURCE_RECORD_COUNTS[benchmark],
                "source_selected": 2,
                "separable": 2,
                "selected": 2,
                "limit": None,
                "pairs_sha256": "2" * 64,
                "run_sha256": "3" * 64,
                "ordered_sample_ids_sha256": "4" * 64,
                "dataset_records_sha256": "5" * 64,
                "selected_sample_ids_sha256": canonical_sha256(
                    [row["sample_id"] for row in rows]
                ),
                "cohort_sha256": canonical_sha256(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "sample_ids": [row["sample_id"] for row in rows],
                        "plans": [row["plan_sha256"] for row in rows],
                    }
                ),
                "exclusions": [],
            },
            "runtime": {
                "operation": "restoration-order-accuracy",
                "runtime": "HuggingFaceRestorationRuntime",
                "python": "3.12.7",
                "torch": "2.10.0",
                "transformers": "4.57.6",
                "accelerate": "1.12.0",
                "device": "cuda:0",
                "cuda": "fixture-cuda",
                "cuda_visible_devices": "1",
                "gpu_name": "fixture-gpu",
                "gpu_total_memory_bytes": 123456,
                "model": model,
                "requested_revision": "1" * 40,
                "model_revision": "1" * 40,
                "model_revision_source": "explicit-load-revision",
                "tokenizer_revision": "1" * 40,
                "tokenizer_revision_source": "explicit-load-revision",
                "dtype": "bfloat16",
                "protocol_sha256": PROTOCOL_SHA256,
                "answer_extraction": (
                    "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1"
                ),
                "generation": GENERATION,
                "base_generation_config_sha256": "a" * 64,
                "effective_generation_config_sha256": "b" * 64,
                "effective_eos_token_ids": [99],
                "effective_eos_token_ids_source": "model-generation-config",
                "benchmark_extractor": benchmark,
                "historical_extraction_difference": "submitted-table13-primary-only",
                "implementation_code": implementation_code_identity(),
            },
            "implementation_code": implementation_code_identity(),
            "progress": {"completed_batches": 11, "total_batches": 11},
            "failure": None,
            "counts": {"selected": 2, "records": 2},
            "generation_retries": {
                "individual_retry_batches": 0,
                "individual_retry_items": 0,
                "batches": [],
            },
            "outputs": {
                "records": {
                    "path": records_path.name,
                    "sha256": _sha256(records_path),
                    "records": 2,
                },
                "summary": {
                    "path": summary_path.name,
                    "sha256": _sha256(summary_path),
                },
            },
        },
    )
    return directory


def _complete_grid(root: Path) -> list[Path]:
    return [
        _write_setting(root, model, benchmark)
        for model in PAPER_MODELS
        for benchmark in PAPER_BENCHMARKS
    ]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("run_id", (None, "c" * 31, "g" * 32))
def test_builder_rejects_a_missing_or_invalid_producer_run_id(
    tmp_path: Path,
    run_id: str | None,
) -> None:
    runs_root = tmp_path / "runs"
    setting = _complete_grid(runs_root)[0]
    run_path = setting / "run.json"
    run = _read_json(run_path)
    if run_id is None:
        run.pop("run_id")
    else:
        run["run_id"] = run_id
    _write_json(run_path, run)

    with pytest.raises(RestorationOrderTableInputError, match="run ID"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(
                runs_root=runs_root,
                output_dir=tmp_path / "table",
            )
        )


def test_builder_micro_pools_six_settings_and_uses_exact_paired_binomial(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _complete_grid(runs_root)

    result = build_restoration_order_table(
        BuildRestorationOrderTableConfig(
            runs_root=runs_root,
            output_dir=tmp_path / "table",
        )
    )

    assert result.settings == 6
    table = _read_json(result.table_path)
    assert table["pooling"] == "micro-by-model-task-sample-identity"
    assert table["cohort"]["items"] == 12
    assert table["cohort"]["kind"] == "fresh-final-pdf-protocol-replication"
    reference = table["paper_published_reference"]
    assert reference["cohort_items"] == 1582
    assert "conditions" not in reference
    assert "setting_retained_items" not in reference
    high_k1 = table["conditions"]["high-relevance-first:k1"]
    assert high_k1 == {"accuracy": 0.5, "correct": 6, "total": 12}
    random_k1 = table["conditions"]["seeded-random:k1"]
    assert random_k1 == {"accuracy": 0.0, "correct": 0, "total": 12}
    paired = table["paired_tests"]["k1"]
    assert paired["high_only"] == 6
    assert paired["random_only"] == 0
    assert paired["p_value"] == pytest.approx(0.03125)
    assert paired["method"] == "two-sided-exact-mcnemar-binomial"
    assert result.csv_path.is_file()
    assert result.csv_path.read_text(encoding="utf-8").splitlines()[0] == (
        "Result kind,Cohort n,Restoration order,Zero restored,One,Two,Three,All restored"
    )
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "No edits" not in markdown
    assert "Fresh final-PDF protocol replication; n=12" in markdown
    assert "historical final-PDF reference; n=1,582" in markdown
    assert result.markdown_path.is_file()
    assert result.latex_path.is_file()
    assert _read_json(result.run_path)["status"] == "completed"


def test_builder_rejects_missing_setting_and_never_publishes_partial_tables(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    settings = _complete_grid(runs_root)
    (settings[-1] / "run.json").unlink()
    output = tmp_path / "table"

    with pytest.raises(RestorationOrderTableInputError, match="six|missing|grid"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(runs_root=runs_root, output_dir=output)
        )

    assert not output.exists() or not any(output.iterdir())


def test_builder_ignores_owned_hidden_work_and_publish_manifests(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _complete_grid(runs_root)
    for suffix in ("restoration-order-work", "restoration-order-publish"):
        hidden = runs_root / f".stale.{suffix}"
        hidden.mkdir()
        _write_json(
            hidden / "run.json",
            {
                "operation": "restoration-order-accuracy",
                "status": "running",
            },
        )

    result = build_restoration_order_table(
        BuildRestorationOrderTableConfig(
            runs_root=runs_root,
            output_dir=tmp_path / "table",
        )
    )

    assert result.settings == 6


def test_builder_serializes_publishers_for_the_same_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    _complete_grid(runs_root)
    operations: list[int] = []
    real_flock = aggregation.fcntl.flock

    def record_flock(descriptor: int, operation: int) -> None:
        operations.append(operation)
        real_flock(descriptor, operation)

    monkeypatch.setattr(aggregation.fcntl, "flock", record_flock)

    build_restoration_order_table(
        BuildRestorationOrderTableConfig(
            runs_root=runs_root,
            output_dir=tmp_path / "table",
        )
    )

    assert operations == [aggregation.fcntl.LOCK_EX, aggregation.fcntl.LOCK_UN]


def test_builder_never_replaces_a_destination_created_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    _complete_grid(runs_root)
    output = tmp_path / "table"
    real_commit = aggregation._commit_table_directory

    def create_destination_then_commit(stage: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_text("other publisher\n", encoding="utf-8")
        real_commit(stage, destination)

    monkeypatch.setattr(
        aggregation,
        "_commit_table_directory",
        create_destination_then_commit,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(runs_root=runs_root, output_dir=output)
        )

    assert (output / "owner.txt").read_text(encoding="utf-8") == "other publisher\n"
    assert not list(tmp_path.glob(".table.*.staging"))


def test_linux_atomic_commit_does_not_replace_an_existing_empty_directory(
    tmp_path: Path,
) -> None:
    from typo_cot.experiments.restoration_order_accuracy.publication import (
        rename_directory_no_replace,
    )

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "artifact.txt").write_text("complete\n", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        rename_directory_no_replace(stage, destination)

    assert destination.is_dir() and not any(destination.iterdir())
    assert (stage / "artifact.txt").is_file()


def test_builder_commit_uses_the_atomic_no_replace_primitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    destination = tmp_path / "destination"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        aggregation,
        "rename_directory_no_replace",
        lambda source, target: calls.append((source, target)),
        raising=False,
    )

    aggregation._commit_table_directory(stage, destination)

    assert calls == [(stage, destination)]


def test_builder_commit_failure_leaves_no_public_or_staged_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    _complete_grid(runs_root)
    output = tmp_path / "table"

    def fail_commit(stage: Path, destination: Path) -> None:
        del stage, destination
        raise OSError("injected commit failure")

    monkeypatch.setattr(aggregation, "_commit_table_directory", fail_commit)

    with pytest.raises(OSError, match="injected commit failure"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(runs_root=runs_root, output_dir=output)
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".table.*.staging"))


def test_builder_post_commit_sync_failure_does_not_mask_published_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    _complete_grid(runs_root)
    output = tmp_path / "table"

    def fail_parent_sync(parent: Path) -> None:
        del parent
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(aggregation, "_fsync_published_parent", fail_parent_sync)

    result = build_restoration_order_table(
        BuildRestorationOrderTableConfig(runs_root=runs_root, output_dir=output)
    )

    assert result.table_path.is_file()
    assert _read_json(result.run_path)["status"] == "completed"


def test_builder_rechecks_analysis_code_identity_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    _complete_grid(runs_root)
    calls = 0

    def changing_identity() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"algorithm": "fixture", "sha256": f"{calls:064x}"}

    monkeypatch.setattr(aggregation, "analysis_code_identity", changing_identity)
    output = tmp_path / "table"

    with pytest.raises(RestorationOrderTableInputError, match="analysis.*changed|identity"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(runs_root=runs_root, output_dir=output)
        )

    assert calls >= 2
    assert not output.exists()


@pytest.mark.parametrize("mutation", ("limit", "arms", "hash"))
def test_builder_fails_closed_on_smoke_runs_arm_drift_or_hash_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    runs_root = tmp_path / "runs"
    setting = _complete_grid(runs_root)[0]
    run_path = setting / "run.json"
    run = _read_json(run_path)
    if mutation == "limit":
        run["arguments"]["limit"] = 1
        _write_json(run_path, run)
    elif mutation == "arms":
        records_path = setting / "restoration_order_records.jsonl"
        rows = [json.loads(line) for line in records_path.read_text().splitlines()]
        rows[0]["arms"].pop("seeded-random:k2")
        records_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        run["outputs"]["records"]["sha256"] = _sha256(records_path)
        _write_json(run_path, run)
    else:
        with (setting / "restoration_order_records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")

    with pytest.raises(RestorationOrderTableInputError, match="limit|arm|hash|integrity"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(
                runs_root=runs_root,
                output_dir=tmp_path / "table",
            )
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "plan",
        "prompt",
        "correctness",
        "selected_ids",
        "summary_extracted",
        "runtime_model",
        "retry_ledger",
    ),
)
def test_builder_rejects_internally_inconsistent_hashed_artifacts(
    tmp_path: Path,
    mutation: str,
) -> None:
    runs_root = tmp_path / "runs"
    setting = _complete_grid(runs_root)[0]
    run_path = setting / "run.json"
    run = _read_json(run_path)

    if mutation in {"plan", "prompt", "correctness"}:
        records_path = setting / "restoration_order_records.jsonl"
        rows = [json.loads(line) for line in records_path.read_text().splitlines()]
        arm = rows[0]["arms"]["high-relevance-first:k1"]
        if mutation == "correctness":
            arm["is_correct"] = False
        else:
            field = "editable_text" if mutation == "plan" else "prompt"
            arm[field] = f"tampered-{field}"
            arm[f"{field}_sha256"] = hashlib.sha256(arm[field].encode()).hexdigest()
        records_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        run["outputs"]["records"]["sha256"] = _sha256(records_path)
        if mutation == "correctness":
            summary_path = setting / "restoration_order_summary.json"
            summary = _read_json(summary_path)
            stored = summary["conditions"]["high-relevance-first:k1"]
            stored["correct"] -= 1
            stored["accuracy"] = stored["correct"] / stored["total"]
            _write_json(summary_path, summary)
            run["outputs"]["summary"]["sha256"] = _sha256(summary_path)
    elif mutation == "selected_ids":
        run["source"]["selected_sample_ids_sha256"] = "f" * 64
    elif mutation == "summary_extracted":
        summary_path = setting / "restoration_order_summary.json"
        summary = _read_json(summary_path)
        summary["conditions"]["edited:k0"]["extracted"] = 0
        _write_json(summary_path, summary)
        run["outputs"]["summary"]["sha256"] = _sha256(summary_path)
    elif mutation == "runtime_model":
        run["runtime"]["model"] = "wrong/model"
    else:
        run["generation_retries"]["individual_retry_batches"] = 1
    _write_json(run_path, run)

    with pytest.raises(
        RestorationOrderTableInputError,
        match=(
            "plan|prompt|sample|summary|runtime|identity|integrity|"
            "extracted|correct|retry"
        ),
    ):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(
                runs_root=runs_root,
                output_dir=tmp_path / "table",
            )
        )


def test_builder_rejects_an_unlimited_run_that_omits_separable_items(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    setting = _complete_grid(runs_root)[0]
    run_path = setting / "run.json"
    run = _read_json(run_path)
    run["source"]["source_selected"] = 3
    run["source"]["separable"] = 3
    summary_path = setting / "restoration_order_summary.json"
    summary = _read_json(summary_path)
    summary["cohort"]["source_flip"] = 3
    summary["cohort"]["separable"] = 3
    _write_json(summary_path, summary)
    run["outputs"]["summary"]["sha256"] = _sha256(summary_path)
    _write_json(run_path, run)

    with pytest.raises(RestorationOrderTableInputError, match="complete|separable|selected"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(
                runs_root=runs_root,
                output_dir=tmp_path / "table",
            )
        )


def test_builder_rejects_a_self_consistent_noncanonical_edit_group_partition(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    setting = _complete_grid(runs_root)[0]
    records_path = setting / "restoration_order_records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    row = rows[0]
    clean_text = "abcd"
    edited_text = "ABCD"
    noncanonical_groups = tuple(
        EditGroup(
            index=index,
            clean_start=index,
            clean_end=index + 1,
            edited_start=index,
            edited_end=index + 1,
            clean_text=clean_text[index],
            edited_text=edited_text[index],
            selection_rank=index + 1,
            target_token_index=index,
            relevance=float(4 - index),
        )
        for index in range(4)
    )
    plan = build_restoration_plan(
        sample_id=row["sample_id"],
        clean_text=clean_text,
        edited_text=edited_text,
        groups=noncanonical_groups,
        seed=42,
    )
    conditions = {condition.condition_id: condition for condition in plan.conditions}
    prefix = "Question: "
    suffix = "\nAnswer:"
    for condition_id in ALL_CONDITION_IDS:
        condition = conditions[condition_id]
        arm = row["arms"][condition_id]
        prompt = prefix + condition.text + suffix
        arm.update(
            {
                "order": condition.order,
                "budget": condition.budget,
                "restored_group_indices": list(condition.restored_group_indices),
                "editable_text": condition.text,
                "editable_text_sha256": hashlib.sha256(
                    condition.text.encode()
                ).hexdigest(),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    row["edit_groups"] = [group.to_dict() for group in plan.groups]
    row["order_group_indices"] = {
        name: list(indices) for name, indices in plan.order_indices
    }
    row["plan_sha256"] = plan.to_dict()["sha256"]
    records_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
        encoding="utf-8",
    )
    run_path = setting / "run.json"
    run = _read_json(run_path)
    run["outputs"]["records"]["sha256"] = _sha256(records_path)
    run["source"]["cohort_sha256"] = canonical_sha256(
        {
            "model": run["arguments"]["model"],
            "benchmark": run["arguments"]["benchmark"],
            "sample_ids": [value["sample_id"] for value in rows],
            "plans": [value["plan_sha256"] for value in rows],
        }
    )
    _write_json(run_path, run)

    with pytest.raises(RestorationOrderTableInputError, match="canonical|group"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(
                runs_root=runs_root,
                output_dir=tmp_path / "table",
            )
        )


def test_builder_reloads_source_and_rejects_self_consistent_fictional_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    setting = _complete_grid(runs_root)[0]
    run_path = setting / "run.json"
    original_run = _read_json(run_path)
    pairs_path = Path(original_run["arguments"]["pairs"])
    fixture_loader = aggregation.load_restoration_order_source
    expected_source = fixture_loader(
        pairs_path,
        model=original_run["arguments"]["model"],
        benchmark=original_run["arguments"]["benchmark"],
        limit=None,
    )

    records_path = setting / "restoration_order_records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    row = rows[0]
    fictional_prefix = "Fictional question: "
    suffix_length = row["prompt_context"]["suffix_length"]
    for arm in row["arms"].values():
        old_prompt = arm["prompt"]
        suffix = old_prompt[len(old_prompt) - suffix_length :]
        prompt = fictional_prefix + arm["editable_text"] + suffix
        arm["prompt"] = prompt
        arm["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    row["prompt_context"]["prefix_length"] = len(fictional_prefix)
    row["prompt_context"]["prefix_sha256"] = hashlib.sha256(
        fictional_prefix.encode()
    ).hexdigest()
    records_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
        encoding="utf-8",
    )
    changed_run = _read_json(run_path)
    changed_run["outputs"]["records"]["sha256"] = _sha256(records_path)
    _write_json(run_path, changed_run)

    def reload_original_source(
        path: Path,
        *,
        model: str,
        benchmark: str,
        limit: int | None = None,
    ) -> object:
        if Path(path) == pairs_path:
            return expected_source
        return fixture_loader(
            path,
            model=model,
            benchmark=benchmark,
            limit=limit,
        )

    monkeypatch.setattr(
        aggregation,
        "load_restoration_order_source",
        reload_original_source,
    )

    with pytest.raises(RestorationOrderTableInputError, match="source|prompt"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(
                runs_root=runs_root,
                output_dir=tmp_path / "table",
            )
        )


def test_builder_rejects_eos_termination_beyond_the_generation_cap(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    setting = _complete_grid(runs_root)[0]
    records_path = setting / "restoration_order_records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0]["arms"]["high-relevance-first:k1"]["token_ids"] = [41] * 512 + [99]
    records_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
        encoding="utf-8",
    )
    run_path = setting / "run.json"
    run = _read_json(run_path)
    run["outputs"]["records"]["sha256"] = _sha256(records_path)
    _write_json(run_path, run)

    with pytest.raises(RestorationOrderTableInputError, match="cap|token|512"):
        build_restoration_order_table(
            BuildRestorationOrderTableConfig(
                runs_root=runs_root,
                output_dir=tmp_path / "table",
            )
        )


def test_exact_test_is_symmetric_and_stable_for_the_full_public_scale() -> None:
    from typo_cot.experiments.restoration_order_accuracy.statistics import (
        exact_mcnemar_p_value,
    )

    assert exact_mcnemar_p_value(6, 0) == pytest.approx(0.03125)
    assert exact_mcnemar_p_value(3, 7) == exact_mcnemar_p_value(7, 3)
    assert exact_mcnemar_p_value(1582, 0) == 0.0
    assert exact_mcnemar_p_value(791, 791) == 1.0
