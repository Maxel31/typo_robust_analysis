"""ItemResult JSONL I/O and M3 long-form conversion (README §4.3).

Public API
----------
write_items(path, items) -> None
    Serialise a list of ItemResult to a JSONL file.  Parent directories are
    created automatically.

read_items(path) -> list[ItemResult]
    Deserialise a JSONL file produced by write_items.

to_long_records(items) -> list[dict]
    Expand each ItemResult into TWO rows suitable for the M3 GLMM:
      row 0  typo_present=0, correct=correct_clean
      row 1  typo_present=1, correct=correct_typo
    Each row also carries: quantization, model, method, bit, dataset,
    typo_type, eps, seed, item_id.

to_long_df(items) -> pandas.DataFrame
    Thin wrapper around to_long_records that returns a DataFrame.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from quant_typo_neuron.contracts import ItemResult

__all__ = ["write_items", "read_items", "to_long_records", "to_long_df"]


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_items(path: str | Path, items: list[ItemResult]) -> None:
    """Write *items* to a JSONL file at *path*, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def read_items(path: str | Path) -> list[ItemResult]:
    """Read a JSONL file and return a list of ItemResult objects."""
    path = Path(path)
    results: list[ItemResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(ItemResult.from_dict(json.loads(line)))
    return results


# ---------------------------------------------------------------------------
# Long-form conversion for GLMM (M3)
# ---------------------------------------------------------------------------

def _item_to_rows(item: ItemResult) -> list[dict]:
    """Expand one ItemResult into two long-form rows."""
    quantization = 0 if item.method == "fp16" else 1
    shared = {
        "quantization": quantization,
        "model": item.model,
        "method": item.method,
        "bit": item.bit,
        "dataset": item.dataset,
        "typo_type": item.typo_type,
        "eps": item.eps,
        "seed": item.seed,
        "item_id": item.item_id,
    }
    return [
        {"typo_present": 0, "correct": item.correct_clean, **shared},
        {"typo_present": 1, "correct": item.correct_typo, **shared},
    ]


def to_long_records(items: list[ItemResult]) -> list[dict]:
    """Return a flat list of dicts, two rows per ItemResult, for the M3 GLMM."""
    rows: list[dict] = []
    for item in items:
        rows.extend(_item_to_rows(item))
    return rows


def to_long_df(items: list[ItemResult]) -> "pd.DataFrame":
    """Return a pandas DataFrame with the long-form records (two rows per item)."""
    import pandas as pd  # lazy import -- pandas is a CPU dependency

    records = to_long_records(items)
    columns = [
        "typo_present",
        "correct",
        "quantization",
        "model",
        "method",
        "bit",
        "dataset",
        "typo_type",
        "eps",
        "seed",
        "item_id",
    ]
    if records:
        return pd.DataFrame(records, columns=columns)
    # Return an empty DataFrame with the correct columns when items is empty.
    return pd.DataFrame(columns=columns)
