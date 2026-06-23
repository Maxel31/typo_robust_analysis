"""実験結果の保存ユーティリティ。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quant_typo_neuron.benchmarks.base import BenchmarkResult


def make_run_id(
    model: str,
    quant: str,
    benchmark: str,
    typo_type: str,
    num_typos: int,
) -> str:
    model_short = model.split("/")[-1].lower()
    parts = [model_short, quant, benchmark, typo_type]
    if typo_type != "clean":
        parts.append(f"n{num_typos}")
    return "__".join(parts)


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
