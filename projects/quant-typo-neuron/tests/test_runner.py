"""実験ランナーのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from quant_typo_neuron.benchmarks.base import BenchmarkExample, BenchmarkResult
from quant_typo_neuron.output import make_run_id, save_metrics, save_predictions


def test_make_run_id():
    rid = make_run_id(
        model="llama-3.2-1b",
        quant="gptq_w4",
        benchmark="arc_easy",
        typo_type="clean",
        num_typos=0,
    )
    assert "llama" in rid
    assert "gptq" in rid
    assert "arc" in rid


def test_make_run_id_typo():
    rid = make_run_id(
        model="gemma-2-2b",
        quant="none",
        benchmark="mmlu",
        typo_type="swap",
        num_typos=4,
    )
    assert "swap" in rid
    assert "4" in rid


def test_save_predictions_creates_jsonl(tmp_path):
    results = [
        BenchmarkResult(
            example_id="q1",
            question_text="Q?",
            typo_annotations=[],
            model_output="A",
            predicted=0,
            correct=True,
        ),
        BenchmarkResult(
            example_id="q2",
            question_text="Q2?",
            typo_annotations=[],
            model_output="B",
            predicted=1,
            correct=False,
        ),
    ]
    path = tmp_path / "predictions.jsonl"
    save_predictions(results, path)
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["example_id"] == "q1"
    assert row["correct"] is True


def test_save_metrics_creates_json(tmp_path):
    metrics = {"accuracy": 0.85, "robustness_gap": 0.05}
    path = tmp_path / "metrics.json"
    save_metrics(metrics, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["accuracy"] == 0.85
    assert data["robustness_gap"] == 0.05


def test_save_metrics_overwrites(tmp_path):
    path = tmp_path / "metrics.json"
    save_metrics({"a": 1}, path)
    save_metrics({"b": 2}, path)
    data = json.loads(path.read_text())
    assert "b" in data
    assert "a" not in data
