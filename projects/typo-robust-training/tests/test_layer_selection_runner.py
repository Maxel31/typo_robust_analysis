"""Hash-bound, resumable orchestration for the expensive all-layer scan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.localization.records import LayerScan
from typo_robust_training.localization.runner import (
    LayerSelectionRunConfig,
    run_select_distillation_layers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-layer-selection.yaml"


def _config(tmp_path: Path) -> Path:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["diagnostic"]["minimum_kl_eligible_per_task"] = 1
    payload["diagnostic"]["minimum_kl_eligible_fraction_per_task"] = 0.0
    payload["diagnostic"]["minimum_answer_cohort_per_task"] = 1
    payload["selection"]["window_width"] = 2
    payload["selection"]["bootstrap_replicates"] = 20
    path = tmp_path / "selection.yaml"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "diagnostic.jsonl"
    rows: list[dict[str, object]] = []
    for task in ("gsm8k", "mmlu", "arc"):
        for cohort in ("repair", "harm"):
            record_id = f"{task}-{cohort}"
            rows.append(
                {
                    "schema_version": "robustness-fixed-typo-pair/v1",
                    "kind": "synthetic",
                    "record_id": record_id,
                    "source": task,
                    "source_revision": "a" * 40,
                    "source_split": "train" if task != "mmlu" else "dev",
                    "source_id": record_id,
                    "group_id": record_id,
                    "split": "diagnostic",
                    "clean_text": "The airport answer is two.",
                    "typo_text": "The arport answer is two.",
                    "task": task,
                    "answer": "2" if task == "gsm8k" else "A",
                    "metadata": {"cohort": cohort},
                    "operation": "deletion",
                    "operations": ["deletion"],
                    "edit_count": 1,
                    "generator_seed": 42,
                    "generator_variant": 0,
                    "edits": [
                        {
                            "operation": "deletion",
                            "clean_word": "airport",
                            "typo_word": "arport",
                            "clean_char_span": [4, 11],
                            "typo_char_span": [4, 10],
                        }
                    ],
                }
            )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


class _Runtime:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls: list[str] = []
        self.fail_after = fail_after

    def scan_pair(self, record: dict[str, object]) -> LayerScan:
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("injected scan failure")
        record_id = str(record["record_id"])
        self.calls.append(record_id)
        cohort = record["metadata"]["cohort"]
        if cohort == "repair":
            clean_correct, typo_correct = True, False
            patched = (False, True, True, False)
        else:
            clean_correct, typo_correct = True, True
            patched = (True, True, False, True)
        return LayerScan(
            record_id=record_id,
            task=str(record["task"]),
            target_token_ids=tuple(range(16)),
            untreated_kl_2_16=(1.0,) * 15,
            patched_kl_2_16_by_layer=(
                (0.8,) * 15,
                (0.4,) * 15,
                (0.5,) * 15,
                (0.9,) * 15,
            ),
            clean_correct=clean_correct,
            typo_correct=typo_correct,
            patched_correct_by_layer=patched,
            kl_invalid_reason=None,
            audit={"fixture": True},
        )

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "offline-fixture/v1",
            "model_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
            "num_decoder_layers": 4,
        }


def _run_config(tmp_path: Path, *, resume: bool) -> LayerSelectionRunConfig:
    return LayerSelectionRunConfig(
        config_path=_config(tmp_path),
        diagnostic_manifest_path=_manifest(tmp_path),
        tasks=("gsm8k", "mmlu", "arc"),
        gpu_id="3",
        output_dir=tmp_path / "output",
        resume=resume,
    )


def test_runner_writes_per_record_selection_and_hash_bound_run(tmp_path: Path) -> None:
    runtime = _Runtime()
    result = run_select_distillation_layers(_run_config(tmp_path, resume=False), runtime=runtime)

    assert len(runtime.calls) == 6
    assert result.records == 6
    assert result.selected_window == (0, 2)
    assert result.scans_path.name == "layer_scans.jsonl"
    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    assert selection["selected_window"] == {"start": 0, "stop": 2}
    assert (
        selection["diagnostic_manifest_sha256"]
        == hashlib.sha256(_manifest(tmp_path).read_bytes()).hexdigest()
    )
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["gpu_id"] == "3"
    assert (
        run["outputs"]["layer_selection.json"]["sha256"]
        == hashlib.sha256(result.selection_path.read_bytes()).hexdigest()
    )


def test_resume_reuses_only_hash_matching_checkpoints(tmp_path: Path) -> None:
    config = _run_config(tmp_path, resume=True)
    failing = _Runtime(fail_after=3)
    with pytest.raises(RuntimeError, match="injected scan failure"):
        run_select_distillation_layers(config, runtime=failing)
    failed_run = json.loads((config.output_dir / "run.json").read_text(encoding="utf-8"))
    assert failed_run["status"] == "failed"
    assert len(failing.calls) == 3

    resumed = _Runtime()
    result = run_select_distillation_layers(config, runtime=resumed)
    assert len(resumed.calls) == 3
    assert result.records == 6


def test_runner_rejects_non_diagnostic_or_duplicate_manifest_rows(tmp_path: Path) -> None:
    config = _run_config(tmp_path, resume=False)
    lines = config.diagnostic_manifest_path.read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[0])
    bad["split"] = "pre_pr_gate"
    config.diagnostic_manifest_path.write_text(
        json.dumps(bad) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="diagnostic split"):
        run_select_distillation_layers(config, runtime=_Runtime())


def test_runner_treats_only_lf_as_jsonl_record_separator(tmp_path: Path) -> None:
    config = _run_config(tmp_path, resume=False)
    rows = [
        json.loads(line) for line in config.diagnostic_manifest_path.read_text().split("\n") if line
    ]
    rows[0]["clean_text"] = "Line one\u2028line two"
    config.diagnostic_manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = run_select_distillation_layers(config, runtime=_Runtime())

    assert result.records == 6
