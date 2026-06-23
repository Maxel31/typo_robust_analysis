"""分析スクリプトのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from quant_typo_neuron.analysis.aggregate import aggregate_results


def _create_mock_results(root: Path, n=3):
    for i in range(n):
        run_dir = root / f"exp_{i}" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model": f"model_{i}",
                    "quantization_method": "gptq",
                    "bits": 4,
                    "benchmark": "arc_easy",
                    "typo_type": "clean",
                    "num_typos": 0,
                    "accuracy": 0.5 + i * 0.1,
                    "num_examples": 100,
                }
            )
        )


def test_aggregate_results_returns_dataframe(tmp_path):
    _create_mock_results(tmp_path)
    df = aggregate_results(tmp_path)
    assert len(df) == 3
    assert "accuracy" in df.columns
    assert "model" in df.columns


def test_aggregate_results_correct_values(tmp_path):
    _create_mock_results(tmp_path)
    df = aggregate_results(tmp_path)
    accuracies = sorted(df["accuracy"].tolist())
    assert accuracies == [0.5, 0.6, 0.7]


def test_aggregate_results_empty_dir(tmp_path):
    df = aggregate_results(tmp_path)
    assert len(df) == 0


def test_aggregate_to_csv(tmp_path):
    _create_mock_results(tmp_path)
    df = aggregate_results(tmp_path)
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)
    assert csv_path.exists()
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 rows
