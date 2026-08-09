"""Restart and publication contracts for the Table 13 producer."""

from __future__ import annotations

import hashlib
import gc
import json
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

from typo_cot.experiments.restoration_order_accuracy.protocol import (
    PAPER_BUDGETS,
    PAPER_ORDERS,
)
from typo_cot.experiments.restoration_order_accuracy.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.restoration_order_accuracy.runner import (
    RestorationGeneration,
    RestorationOrderConfig,
    run_restoration_order_accuracy,
)


MODEL = "google/gemma-3-4b-it"


def _pair_record(sample_id: str = "sample-a") -> dict[str, object]:
    clean_text = "alpha beta gamma delta"
    edited_text = "alpga beeta gamna delxa"
    prefix = f"few-shot::{sample_id}\nQuestion: "
    suffix = "\nAnswer:"
    attempts = []
    cursor = 0
    for rank, (clean_word, relevance) in enumerate(
        zip(clean_text.split(), (4.0, -3.0, 2.0, -1.0), strict=True),
        start=1,
    ):
        start = clean_text.index(clean_word, cursor)
        cursor = start + len(clean_word)
        attempts.append(
            {
                "selection_rank": rank,
                "target_token_index": rank * 10,
                "relevance": relevance,
            }
        )
    return {
        "sample_id": sample_id,
        "model": MODEL,
        "benchmark": "gsm8k",
        "gold_answer": "4",
        "num_target_attempts": 4,
        "target_attempts": attempts,
        "clean": {
            "editable_text": clean_text,
            "editable_prompt_span": {
                "start": len(prefix),
                "end": len(prefix) + len(clean_text),
            },
            "prompt": prefix + clean_text + suffix,
            "answer": {"is_correct": True},
        },
        "edited": {
            "editable_text": edited_text,
            "editable_prompt_span": {
                "start": len(prefix),
                "end": len(prefix) + len(edited_text),
            },
            "prompt": prefix + edited_text + suffix,
            "answer": {"is_correct": False},
        },
    }


class _Source:
    def __init__(
        self,
        records: tuple[dict[str, object], ...],
        *,
        fail_on_unchanged_call: int | None = None,
    ) -> None:
        self.model = MODEL
        self.benchmark = "gsm8k"
        self.model_revision = "1" * 40
        self.records = records
        self.pairs_sha256 = "2" * 64
        self.run_sha256 = "3" * 64
        self.ordered_sample_ids_sha256 = "4" * 64
        self.dataset_records_sha256 = "5" * 64
        self.unchanged_checks = 0
        self.fail_on_unchanged_call = fail_on_unchanged_call

    def assert_unchanged(self) -> None:
        self.unchanged_checks += 1
        if self.unchanged_checks == self.fail_on_unchanged_call:
            raise RuntimeError("fixture source changed")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_kind": "completed-prepare-edited-pairs/v1",
            "model": self.model,
            "benchmark": self.benchmark,
            "model_revision": self.model_revision,
            "records": len(self.records),
            "pairs_sha256": self.pairs_sha256,
            "run_sha256": self.run_sha256,
            "ordered_sample_ids_sha256": self.ordered_sample_ids_sha256,
            "dataset_records_sha256": self.dataset_records_sha256,
        }


class _Runtime:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls: list[tuple[str, ...]] = []
        self.closed = False

    def generate_batch(
        self,
        prompts: tuple[str, ...] | list[str],
        *,
        sample_ids: tuple[str, ...] | list[str],
        gold_answers: tuple[str, ...] | list[str],
    ) -> tuple[RestorationGeneration, ...]:
        assert len(prompts) == len(sample_ids) == len(gold_answers)
        self.calls.append(tuple(prompts))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("fixture interruption")
        rows = []
        for prompt, sample_id, gold in zip(
            prompts, sample_ids, gold_answers, strict=True
        ):
            is_correct = "alpga beeta gamna delxa" not in prompt
            rows.append(
                RestorationGeneration(
                    sample_id=sample_id,
                    token_ids=(41, 99),
                    text=f"reasoning **{gold}**",
                    termination="eos",
                    extracted_answer=gold,
                    is_extracted=True,
                    is_correct=is_correct,
                    method="fallback:N3_bold",
                    primary_method="no_match",
                )
            )
        return tuple(rows)

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "fixture",
            "model_revision": "1" * 40,
            "tokenizer_revision": "1" * 40,
            "implementation_code": implementation_code_identity(),
        }

    def close(self) -> None:
        self.closed = True


class _SingletonFallbackRuntime(_Runtime):
    def generate_batch(
        self,
        prompts: tuple[str, ...] | list[str],
        *,
        sample_ids: tuple[str, ...] | list[str],
        gold_answers: tuple[str, ...] | list[str],
    ) -> tuple[RestorationGeneration, ...]:
        if len(prompts) > 1:
            self.calls.append(tuple(prompts))
            raise RuntimeError("fixture batched generation failure")
        return super().generate_batch(
            prompts,
            sample_ids=sample_ids,
            gold_answers=gold_answers,
        )


class _CloseFailRuntime(_Runtime):
    def close(self) -> None:
        raise RuntimeError("fixture close failure")


class _Allocation:
    pass


class _TracebackReleasingRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.failed_allocation: weakref.ReferenceType[_Allocation] | None = None
        self.alive_during_recovery: list[bool] = []

    def generate_batch(
        self,
        prompts: tuple[str, ...] | list[str],
        *,
        sample_ids: tuple[str, ...] | list[str],
        gold_answers: tuple[str, ...] | list[str],
    ) -> tuple[RestorationGeneration, ...]:
        if len(prompts) > 1:
            allocation = _Allocation()
            self.failed_allocation = weakref.ref(allocation)
            raise RuntimeError("fixture OOM with traceback-held allocation")
        return super().generate_batch(
            prompts,
            sample_ids=sample_ids,
            gold_answers=gold_answers,
        )

    def recover_after_batch_failure(self) -> None:
        gc.collect()
        assert self.failed_allocation is not None
        self.alive_during_recovery.append(self.failed_allocation() is not None)


class _MismatchedCodeRuntime(_Runtime):
    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload["implementation_code"] = {
            **implementation_code_identity(),
            "sha256": "f" * 64,
        }
        return payload


def _config(tmp_path: Path, *, resume: bool = False) -> RestorationOrderConfig:
    return RestorationOrderConfig(
        model=MODEL,
        benchmark="gsm8k",
        pairs=tmp_path / "source" / "pairs.jsonl",
        orders=PAPER_ORDERS,
        budgets=PAPER_BUDGETS,
        seed=42,
        batch_size=8,
        gpu_id="1",
        output_dir=tmp_path / "run",
        resume=resume,
    )


def _work_dir(config: RestorationOrderConfig) -> Path:
    output = config.output_dir.resolve()
    return output.parent / f".{output.name}.restoration-order-work"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_runner_generates_shared_endpoints_and_nine_order_budget_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    runtime = _Runtime()
    monkeypatch.setattr(runner_module, "load_restoration_order_source", lambda *a, **k: source)

    result = run_restoration_order_accuracy(
        _config(tmp_path),
        runtime_factory=lambda *a, **k: runtime,
    )

    assert result.records == 1
    assert len(runtime.calls) == 11
    assert runtime.closed is True
    rows = [
        json.loads(line)
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["sample-a"]
    arms = rows[0]["arms"]
    assert list(arms) == [
        "edited:k0",
        "high-relevance-first:k1",
        "high-relevance-first:k2",
        "high-relevance-first:k3",
        "seeded-random:k1",
        "seeded-random:k2",
        "seeded-random:k3",
        "low-relevance-first:k1",
        "low-relevance-first:k2",
        "low-relevance-first:k3",
        "clean:k4",
    ]
    assert arms["edited:k0"]["is_correct"] is False
    assert arms["clean:k4"]["is_correct"] is True
    assert rows[0]["realized_edit_groups"] == 4
    prefix = "few-shot::sample-a\nQuestion: "
    suffix = "\nAnswer:"
    assert rows[0]["prompt_context"] == {
        "prefix_length": len(prefix),
        "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "suffix_length": len(suffix),
        "suffix_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
    }
    assert source.unchanged_checks >= 1

    summary = _read_json(result.summary_path)
    assert summary["cohort"]["selected"] == 1
    assert summary["conditions"]["edited:k0"]["correct"] == 0
    assert summary["conditions"]["clean:k4"]["correct"] == 1
    manifest = _read_json(result.run_path)
    assert manifest["status"] == "completed"
    assert manifest["arguments"]["gpu_id"] == "1"
    assert manifest["outputs"]["records"]["sha256"]


def test_completed_resume_validates_outputs_without_loading_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(runner_module, "load_restoration_order_source", lambda *a, **k: source)
    first = run_restoration_order_accuracy(
        _config(tmp_path), runtime_factory=lambda *a, **k: _Runtime()
    )

    resumed = run_restoration_order_accuracy(
        replace(_config(tmp_path), resume=True),
        runtime_factory=lambda *a, **k: pytest.fail("completed resume loaded the model"),
    )

    assert resumed == first


def test_resume_after_power_loss_immediately_after_first_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(
        runner_module, "load_restoration_order_source", lambda *a, **k: source
    )
    real_write = runner_module._write_json_atomic
    interrupted = False

    def interrupt_after_checkpoint(path: Path, payload: object) -> None:
        nonlocal interrupted
        if (
            not interrupted
            and path.name == "run.json"
            and any(tmp_path.rglob("checkpoints/**/*.json"))
        ):
            interrupted = True
            raise KeyboardInterrupt("simulated power loss")
        real_write(path, payload)

    monkeypatch.setattr(runner_module, "_write_json_atomic", interrupt_after_checkpoint)
    with pytest.raises(KeyboardInterrupt, match="power loss"):
        run_restoration_order_accuracy(
            _config(tmp_path), runtime_factory=lambda *a, **k: _Runtime()
        )
    monkeypatch.setattr(runner_module, "_write_json_atomic", real_write)

    result = run_restoration_order_accuracy(
        replace(_config(tmp_path), resume=True),
        runtime_factory=lambda *a, **k: _Runtime(),
    )
    assert result.records == 1


def test_failed_run_keeps_complete_batch_checkpoints_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(runner_module, "load_restoration_order_source", lambda *a, **k: source)
    interrupted = _Runtime(fail_after=3)
    with pytest.raises(RuntimeError, match="fixture interruption|failed"):
        run_restoration_order_accuracy(
            _config(tmp_path), runtime_factory=lambda *a, **k: interrupted
        )

    manifest = _read_json(_work_dir(_config(tmp_path)) / "run.json")
    assert manifest["status"] == "failed"
    assert len(list(_work_dir(_config(tmp_path)).rglob("*.json"))) >= 3

    resumed_runtime = _Runtime()
    result = run_restoration_order_accuracy(
        replace(_config(tmp_path), resume=True),
        runtime_factory=lambda *a, **k: resumed_runtime,
    )
    assert result.records == 1
    assert len(resumed_runtime.calls) == 8
    assert not _work_dir(_config(tmp_path)).exists()


def test_config_rejects_nonpaper_grid_or_generation_settings(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="paper.*orders|orders"):
        replace(config, orders=("high-relevance-first",))
    with pytest.raises(ValueError, match="budgets"):
        replace(config, budgets=(1, 2, 3))
    with pytest.raises(ValueError, match="seed.*42|seed"):
        replace(config, seed=7)
    with pytest.raises(ValueError, match="batch.*8|batch"):
        replace(config, batch_size=4)
    with pytest.raises(ValueError, match="paper grid|model"):
        replace(config, model="example/other")


def test_output_identity_lock_rejects_a_concurrent_invocation(tmp_path: Path) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    output = _config(tmp_path).output_dir.resolve()
    with runner_module._exclusive_output_lock(output):
        with pytest.raises(RuntimeError, match="another.*owns|invocation"):
            with runner_module._exclusive_output_lock(output):
                pytest.fail("second invocation acquired the same output lock")


def test_completed_outputs_retain_singleton_fallback_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record("sample-a"), _pair_record("sample-b")))
    runtime = _SingletonFallbackRuntime()
    monkeypatch.setattr(
        runner_module, "load_restoration_order_source", lambda *a, **k: source
    )

    result = run_restoration_order_accuracy(
        _config(tmp_path), runtime_factory=lambda *a, **k: runtime
    )

    rows = [json.loads(line) for line in result.records_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(
        arm["individual_retry"] is True
        for row in rows
        for arm in row["arms"].values()
    )
    manifest = _read_json(result.run_path)
    retries = manifest["generation_retries"]
    assert retries["individual_retry_batches"] == 11
    assert retries["individual_retry_items"] == 22


def test_batch_failure_releases_traceback_before_cuda_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record("sample-a"), _pair_record("sample-b")))
    runtime = _TracebackReleasingRuntime()
    monkeypatch.setattr(
        runner_module, "load_restoration_order_source", lambda *a, **k: source
    )

    result = run_restoration_order_accuracy(
        _config(tmp_path), runtime_factory=lambda *a, **k: runtime
    )

    assert result.records == 2
    assert runtime.alive_during_recovery == [False] * 11


def test_runtime_provenance_must_match_the_runner_code_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(
        runner_module, "load_restoration_order_source", lambda *a, **k: source
    )

    with pytest.raises(RuntimeError, match="implementation|runtime.*code"):
        run_restoration_order_accuracy(
            _config(tmp_path),
            runtime_factory=lambda *a, **k: _MismatchedCodeRuntime(),
        )

    assert not _config(tmp_path).output_dir.exists()


def test_completed_resume_never_deletes_an_unowned_private_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(
        runner_module, "load_restoration_order_source", lambda *a, **k: source
    )
    run_restoration_order_accuracy(
        _config(tmp_path), runtime_factory=lambda *a, **k: _Runtime()
    )
    unowned = _work_dir(_config(tmp_path))
    unowned.mkdir()
    marker = unowned / "unrelated-owner.txt"
    marker.write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="owner|owned|private"):
        run_restoration_order_accuracy(
            replace(_config(tmp_path), resume=True),
            runtime_factory=lambda *a, **k: pytest.fail("resume loaded the model"),
        )

    assert marker.read_text(encoding="utf-8") == "do not delete\n"


def test_publication_commit_failure_leaves_no_public_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(
        runner_module, "load_restoration_order_source", lambda *a, **k: source
    )
    monkeypatch.setattr(
        runner_module,
        "_commit_publish_directory",
        lambda *a, **k: (_ for _ in ()).throw(OSError("fixture commit failure")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="commit failure|publication"):
        run_restoration_order_accuracy(
            _config(tmp_path), runtime_factory=lambda *a, **k: _Runtime()
        )

    assert not _config(tmp_path).output_dir.exists()
    manifests = list(tmp_path.rglob("run.json"))
    assert len(manifests) == 1
    assert _read_json(manifests[0])["status"] == "failed"


def test_producer_commit_uses_the_atomic_no_replace_primitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    stage = tmp_path / "stage"
    stage.mkdir()
    destination = tmp_path / "destination"
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        runner_module,
        "rename_directory_no_replace",
        lambda source, target: calls.append((source, target)),
        raising=False,
    )

    runner_module._commit_publish_directory(stage, destination)

    assert calls == [(stage, destination)]


def test_runtime_close_failure_marks_private_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(
        runner_module, "load_restoration_order_source", lambda *a, **k: source
    )

    with pytest.raises(RuntimeError, match="close failure"):
        run_restoration_order_accuracy(
            _config(tmp_path), runtime_factory=lambda *a, **k: _CloseFailRuntime()
        )

    assert not _config(tmp_path).output_dir.exists()
    manifests = list(tmp_path.rglob("run.json"))
    assert len(manifests) == 1
    manifest = _read_json(manifests[0])
    assert manifest["status"] == "failed"
    assert manifest["failure"]["message"] == "fixture close failure"


def test_source_change_before_publication_marks_the_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),), fail_on_unchanged_call=1)
    monkeypatch.setattr(runner_module, "load_restoration_order_source", lambda *a, **k: source)

    with pytest.raises(RuntimeError, match="source changed|publication"):
        run_restoration_order_accuracy(
            _config(tmp_path), runtime_factory=lambda *a, **k: _Runtime()
        )

    manifest = _read_json(_work_dir(_config(tmp_path)) / "run.json")
    assert manifest["status"] == "failed"
    assert manifest["failure"]["message"] == "fixture source changed"
    assert not (_config(tmp_path).output_dir / "restoration_order_records.jsonl").exists()
    assert _work_dir(_config(tmp_path)).is_dir()


def test_resume_rejects_a_tampered_complete_batch_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typo_cot.experiments.restoration_order_accuracy.runner as runner_module

    source = _Source((_pair_record(),))
    monkeypatch.setattr(runner_module, "load_restoration_order_source", lambda *a, **k: source)
    with pytest.raises(RuntimeError):
        run_restoration_order_accuracy(
            _config(tmp_path),
            runtime_factory=lambda *a, **k: _Runtime(fail_after=1),
        )
    checkpoints = sorted(
        (_work_dir(_config(tmp_path)) / "checkpoints").rglob("*.json")
    )
    assert len(checkpoints) == 1
    payload = _read_json(checkpoints[0])
    payload["generations"][0]["is_correct"] = not payload["generations"][0]["is_correct"]
    checkpoints[0].write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="integrity|checkpoint"):
        run_restoration_order_accuracy(
            replace(_config(tmp_path), resume=True),
            runtime_factory=lambda *a, **k: _Runtime(),
        )
