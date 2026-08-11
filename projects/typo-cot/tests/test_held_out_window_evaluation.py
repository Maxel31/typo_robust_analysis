"""Contracts for diagnostic selection and disjoint held-out window evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
)
from typo_cot.experiments.build_rebuttal_manifest.records import REBUTTAL_COHORT_SCHEMA
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.held_out_window_evaluation.protocol import (
    load_held_out_window_protocol,
)
from typo_cot.experiments.held_out_window_evaluation.runner import (
    HeldOutGeneration,
    HeldOutWindowConfig,
    HeldOutWindowRunError,
    WindowEvaluationScan,
    WindowSelectionScan,
    run_held_out_window_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "held-out-window-evaluation.yaml"
CANDIDATES = ((0, 6), (6, 12), (12, 18), (18, 24), (22, 28))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _record(
    setting_index: int,
    item_index: int,
    *,
    selection_count: int,
) -> dict[str, object]:
    setting = REBUTTAL_SETTINGS[setting_index]
    pair_id = _digest(f"held-out\0{setting.model}\0{setting.task}\0{item_index}")
    selection = item_index < selection_count
    return {
        "pair_id": pair_id,
        "sample_id": f"{setting.task}-{item_index:04d}",
        "model": setting.model,
        "task": setting.task,
        "target_rule": "attribution-4",
        "gold_answer": "42",
        "clean_answer": "42",
        "typo_answer": "99",
        "clean_correct": True,
        "typo_correct": False,
        "clean_text": "clean prompt",
        "typo_text": "typo prompt",
        "clean_continuation": " solution",
        "clean_prompt_token_count": 20,
        "typo_prompt_token_count": 21,
        "number_of_aligned_words": 1,
        "edits": [
            {
                "clean_char_span": [0, 1],
                "typo_char_span": [0, 1],
                "clean_token_indices": [5],
                "typo_token_indices": [6],
                "clean_word_final_token": 5,
                "typo_word_final_token": 6,
            }
        ],
        "cohorts": {
            "restoration": True,
            "window_selection": selection,
            "window_evaluation": not selection,
        },
        "source": {
            "source_record_sha256": _digest(pair_id),
            "model_revision": _digest(setting.model),
        },
    }


def _records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for setting_index, setting in enumerate(REBUTTAL_SETTINGS):
        records.extend(
            _record(setting_index, index, selection_count=setting.paper_denominator)
            for index in range(setting.paper_denominator)
        )
    for record in records:
        selection = int(str(record["sample_id"]).rsplit("-", 1)[1]) % 2 == 0
        record["cohorts"] = {
            "restoration": True,
            "window_selection": selection,
            "window_evaluation": not selection,
        }
    return tuple(records)


def _write_inputs(tmp_path: Path, records: tuple[dict[str, object], ...]) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest" / "pair_manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("fixture\n", encoding="utf-8")
    restoration = sorted(str(record["pair_id"]) for record in records)
    selection = sorted(
        str(record["pair_id"]) for record in records if record["cohorts"]["window_selection"]
    )
    evaluation = sorted(
        str(record["pair_id"]) for record in records if record["cohorts"]["window_evaluation"]
    )
    cohort_ids = manifest.with_name("cohort_ids.json")
    cohort_ids.write_text(
        json.dumps(
            {
                "schema_version": REBUTTAL_COHORT_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "manifest_protocol_sha256": REBUTTAL_MANIFEST_PROTOCOL.sha256(),
                "pair_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "identity": "sha256-canonical-model-task-target-rule-sample-id/v1",
                "window_split": {
                    "algorithm": "sha256-order-sample-group-half-per-task/v2",
                    "seed": 42,
                    "outcome_independent": True,
                },
                "cohorts": {
                    "restoration": restoration,
                    "window_selection": selection,
                    "window_evaluation": evaluation,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, cohort_ids


def _config(
    tmp_path: Path,
    records: tuple[dict[str, object], ...],
    *,
    limit_per_setting: int | None = None,
    resume: bool = False,
) -> HeldOutWindowConfig:
    manifest, cohort_ids = _write_inputs(tmp_path, records)
    return HeldOutWindowConfig(
        protocol_path=DEFAULT_CONFIG,
        manifest_path=manifest,
        cohort_ids_path=cohort_ids,
        gpu_id="3",
        output_dir=tmp_path / "output",
        limit_per_setting=limit_per_setting,
        resume=resume,
    )


def _generation(*, success: bool, label: str) -> HeldOutGeneration:
    return HeldOutGeneration(
        token_ids=(10, 1),
        text=label,
        termination="eos",
        value="42" if success else "99",
        is_extracted=True,
        is_correct=success,
        method="fixture",
        primary_method="fixture-primary",
    )


class _Runtime:
    selection_calls = 0
    evaluation_calls = 0
    expected_selection_calls = 0
    selection_artifact: Path | None = None

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        self.model = model
        self.task = task
        self.revision = revision
        self.gpu_id = gpu_id
        self.num_layers = 34 if "gemma" in model else 28 if "Llama" in model else 32

    def provenance(self) -> Mapping[str, object]:
        return {
            "operation": "held-out-window-evaluation",
            "runtime": "held-out-window-fixture",
            "model": self.model,
            "task": self.task,
            "requested_revision": self.revision,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "cuda_visible_devices": self.gpu_id,
            "direction": "clean-to-typo",
            "coordinate_source": "manifest-edited-word-final-token/v1",
            "diagnostic_readout": "first-clean-continuation-token-distribution/v1",
            "candidate_windows": [list(window) for window in CANDIDATES],
            "effective_eos_token_ids": [1, 107],
            "effective_eos_token_ids_source": "fixture",
            "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
            "answer_extraction": "primary-then-empty-only-positional/v1",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_new_tokens": 512,
                "use_cache": True,
                "return_dict_in_generate": False,
                "output_scores": False,
                "padding_side": "left",
                "patch_application": "selected-window-on-typo-prefill-exactly-once/v1",
            },
        }

    def scan_selection(
        self,
        pair: Mapping[str, object],
        *,
        candidate_windows: tuple[tuple[int, int], ...],
    ) -> WindowSelectionScan:
        _Runtime.selection_calls += 1
        assert candidate_windows == CANDIDATES
        return WindowSelectionScan(
            available=True,
            invalid_reason=None,
            target_token_id=123,
            target_token_text="solution",
            untreated_kl=1.0,
            patched_kl=(0.2, 0.4, 0.6, 0.8, 0.9),
        )

    def scan_evaluation(
        self,
        pair: Mapping[str, object],
        *,
        windows: Mapping[str, tuple[int, int]],
    ) -> WindowEvaluationScan:
        assert _Runtime.selection_calls == _Runtime.expected_selection_calls
        assert _Runtime.selection_artifact is not None
        assert _Runtime.selection_artifact.is_file()
        _Runtime.evaluation_calls += 1
        index = int(str(pair["sample_id"]).rsplit("-", 1)[1])
        assert windows == {"selected": (0, 6), "runner-up": (6, 12)}
        return WindowEvaluationScan(
            generations={
                "selected": _generation(success=index % 3 != 0, label="selected"),
                "runner-up": _generation(success=index % 5 == 0, label="runner-up"),
            }
        )


def test_protocol_catalog_cli_and_readme_freeze_the_public_contract() -> None:
    protocol = load_held_out_window_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "held-out-window-evaluation-config/v1"
    assert protocol.candidate_windows == CANDIDATES
    assert protocol.window_width == 6
    assert protocol.selection_metric == "median-normalized-first-token-kl-restoration/v1"
    assert protocol.cross_setting_score == "equal-setting-macro-mean/v1"
    assert protocol.ranking == "score-descending-then-start-ascending/v1"
    assert protocol.untreated_kl_min_exclusive == 1e-9
    assert protocol.pair_bootstrap_replicates == 10_000
    assert protocol.nested_bootstrap_replicates == 10_000
    assert protocol.bootstrap_seed == 42
    assert protocol.multiplicity == "holm-6-held-out-setting-contrasts/v1"

    spec = get_experiment("held-out-window-evaluation")
    assert spec.status == "implemented"
    assert spec.compute == "gpu"
    assert spec.required_arguments == (
        "--config",
        "--manifest",
        "--cohort-ids",
        "--gpu-id",
        "--output-dir",
    )
    assert spec.outputs == (
        "window_selection_records.jsonl",
        "window_selection.json",
        "held_out_window_records.jsonl",
        "held_out_window_table.csv",
        "held_out_window_contrasts.csv",
        "held_out_window_summary.json",
        "pair_status_records.jsonl",
        "run.json",
    )
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    help_text = subparsers.choices["held-out-window-evaluation"].format_help()
    assert "--cohort-ids" in help_text
    assert "--limit-per-setting" in help_text
    for readme in (PROJECT_ROOT / "README.md", PROJECT_ROOT / "README.ja.md"):
        contents = readme.read_text(encoding="utf-8")
        assert "typo-cot held-out-window-evaluation" in contents
        assert "window_selection.json" in contents
        assert "[22,28)" in contents


def test_protocol_rejects_candidate_metric_and_tie_break_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed = tmp_path / "changed.yaml"
    payload["diagnostic"]["candidate_windows"].reverse()
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic"):
        load_held_out_window_protocol(changed)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["diagnostic"]["ranking"] = "score-only"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic"):
        load_held_out_window_protocol(changed)


def test_scan_payloads_reject_partial_or_inconsistent_results() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        WindowSelectionScan(
            available=False,
            invalid_reason=None,
            target_token_id=None,
            target_token_text=None,
            untreated_kl=None,
            patched_kl=(),
        )
    with pytest.raises(ValueError, match="candidate"):
        WindowSelectionScan(
            available=True,
            invalid_reason=None,
            target_token_id=1,
            target_token_text="x",
            untreated_kl=1.0,
            patched_kl=(0.1,),
        )
    with pytest.raises(ValueError, match="arms"):
        WindowEvaluationScan(generations={"selected": _generation(success=True, label="x")})


def test_runner_commits_selection_before_disjoint_evaluation_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typo_cot.experiments.held_out_window_evaluation import runner

    records = _records()
    load_calls: list[Path] = []

    def load_manifest(path: Path) -> tuple[dict[str, object], ...]:
        load_calls.append(path)
        return records

    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", load_manifest)
    config = _config(tmp_path, records)
    expected_selection = sum((setting.paper_denominator + 1) // 2 for setting in REBUTTAL_SETTINGS)
    expected_evaluation = sum(setting.paper_denominator // 2 for setting in REBUTTAL_SETTINGS)
    _Runtime.selection_calls = 0
    _Runtime.evaluation_calls = 0
    _Runtime.expected_selection_calls = expected_selection
    _Runtime.selection_artifact = config.output_dir / "window_selection.json"

    result = run_held_out_window_evaluation(config, runtime_factory=_Runtime)

    assert result.selection_pairs == expected_selection == 622
    assert result.evaluation_pairs == expected_evaluation == 619
    assert result.evaluation_records == expected_evaluation * 2
    assert result.table_rows == 6
    assert result.contrast_rows == 6
    assert _Runtime.selection_calls == expected_selection
    assert _Runtime.evaluation_calls == expected_evaluation
    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    assert selection["selected_window"] == {"start": 0, "stop": 6}
    assert selection["runner_up_window"] == {"start": 6, "stop": 12}
    assert selection["selection_pair_ids_sha256"] != selection["evaluation_pair_ids_sha256"]
    assert (
        selection["selection_records_sha256"]
        == hashlib.sha256(result.selection_records_path.read_bytes()).hexdigest()
    )
    with result.contrasts_path.open(encoding="utf-8", newline="") as handle:
        contrasts = tuple(csv.DictReader(handle))
    assert len(contrasts) == 6
    assert all(row["mcnemar_p_holm_defined"] == "True" for row in contrasts)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["confirmatory"] is True
    assert summary["initial_six_advantage_reproduced"] is True
    assert summary["macro_difference"]["settings"] == 6
    statuses = tuple(
        json.loads(line)
        for line in result.status_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    assert len(statuses) == 1_241
    assert {row["phase"] for row in statuses} == {"selection", "evaluation"}
    assert all(row["status"] == "completed" for row in statuses)
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert (
        run["selection_commit"]["sha256"]
        == hashlib.sha256(result.selection_path.read_bytes()).hexdigest()
    )
    assert load_calls == [config.manifest_path]

    resumed = run_held_out_window_evaluation(
        replace(config, resume=True),
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume must not load a runtime")
        ),
    )
    assert resumed == result
    assert load_calls == [config.manifest_path, config.manifest_path]

    result.summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output SHA-256 differs"):
        run_held_out_window_evaluation(replace(config, resume=True), runtime_factory=_Runtime)


def test_eval_failure_keeps_frozen_selection_and_active_pair_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typo_cot.experiments.held_out_window_evaluation import runner

    records = _records()
    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", lambda _path: records)
    config = _config(tmp_path, records, limit_per_setting=2)
    _Runtime.selection_calls = 0
    _Runtime.evaluation_calls = 0
    _Runtime.expected_selection_calls = 12
    _Runtime.selection_artifact = config.output_dir / "window_selection.json"

    class FailingRuntime(_Runtime):
        def scan_evaluation(self, *args: object, **kwargs: object) -> WindowEvaluationScan:
            if _Runtime.evaluation_calls == 1:
                raise RuntimeError("injected held-out interruption")
            return super().scan_evaluation(*args, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(HeldOutWindowRunError):
        run_held_out_window_evaluation(config, runtime_factory=FailingRuntime)
    selection_bytes = (config.output_dir / "window_selection.json").read_bytes()
    failed = json.loads((config.output_dir / "run.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["phase"] == "evaluation"
    assert failed["failures"][0]["pair_id"]
    assert failed["failures"][0]["sample_id"]
    statuses = tuple(
        json.loads(line)
        for line in (config.output_dir / "pair_status_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    assert any(row["status"] == "failed" and row["phase"] == "evaluation" for row in statuses)

    selection_calls = _Runtime.selection_calls
    resumed = run_held_out_window_evaluation(replace(config, resume=True), runtime_factory=_Runtime)
    assert _Runtime.selection_calls == selection_calls
    assert resumed.evaluation_pairs == 12
    assert resumed.selection_path.read_bytes() == selection_bytes


def test_split_overlap_and_cohort_sidecar_mismatch_fail_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typo_cot.experiments.held_out_window_evaluation import runner

    records = _records()
    monkeypatch.setattr(runner, "load_rebuttal_pair_manifest", lambda _path: records)
    config = _config(tmp_path, records, limit_per_setting=1)
    payload = json.loads(config.cohort_ids_path.read_text(encoding="utf-8"))
    payload["cohorts"]["window_evaluation"][0] = payload["cohorts"]["window_selection"][0]
    config.cohort_ids_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disjoint"):
        run_held_out_window_evaluation(
            config,
            runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("invalid split must fail before runtime loading")
            ),
        )
