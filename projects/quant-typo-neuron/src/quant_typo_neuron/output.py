"""実験結果の保存ユーティリティ。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quant_typo_neuron.benchmarks.base import BenchmarkResult


def make_run_id(
    model: str,
    quantization: str,
    benchmark: str,
    typo_type: str,
    num_typos: int,
    calibration: str = "none",
) -> str:
    model_short = model.split("/")[-1]
    if calibration != "none":
        quantization_dir = f"{quantization}_{calibration}"
    else:
        quantization_dir = quantization
    if typo_type == "clean":
        typo_dir = "clean"
    else:
        typo_dir = f"{typo_type}_n{num_typos}"
    return f"{model_short}/{quantization_dir}/{benchmark}/{typo_dir}"


def save_predictions(results: list[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
