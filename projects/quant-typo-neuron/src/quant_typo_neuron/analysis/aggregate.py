"""実験結果の集約。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def aggregate_results(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    records: list[dict] = []

    for metrics_file in results_root.rglob("metrics.json"):
        try:
            data = json.loads(metrics_file.read_text())
            data["_source"] = str(metrics_file)
            records.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
