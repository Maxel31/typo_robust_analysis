"""JSAI2026 アーカイブ (読み取り専用) への薄いアクセス層.

configs/paths.yaml が指すアーカイブのディレクトリ規約:
- baseline:  {outputs}/baseline/{model}_{benchmark}/results.json
- perturbed: {outputs}/perturbed/{model}_{benchmark}_{suffix}/results.json
  (suffix は master_table.CONDITION_TO_ARCHIVE_SUFFIX)
- analysis:  {outputs}/analysis/{benchmark}/{model}/{suffix}/full_results.json

本モジュールは読み取りとパス解決だけを行う。アーカイブへの書き込みは行わない。
master table 完成後は、このモジュール経由の直接読みを parquet 読みに
一行で差し替えられるよう、データアクセスをここに隔離する。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from typo_cot.data.master_table import CONDITION_TO_ARCHIVE_SUFFIX


def load_paths_config(path: Path | str) -> dict[str, Any]:
    """configs/paths.yaml を読み込む."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path | str) -> Any:
    """JSON ファイルを読み込む."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """ファイルの sha256 hex digest を返す (移行同一性検証用)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def baseline_dir(outputs_root: Path | str, model: str, benchmark: str) -> Path:
    """baseline (clean) の結果ディレクトリ."""
    return Path(outputs_root) / "baseline" / f"{model}_{benchmark}"


def perturbed_dir(
    outputs_root: Path | str, model: str, benchmark: str, condition: str
) -> Path:
    """摂動条件の結果ディレクトリ. condition は master table の条件名."""
    suffix = CONDITION_TO_ARCHIVE_SUFFIX[condition]
    return Path(outputs_root) / "perturbed" / f"{model}_{benchmark}_{suffix}"


def analysis_condition_dir(
    analysis_root: Path | str, model: str, benchmark: str, condition: str
) -> Path:
    """analysis の (benchmark, model, condition) ディレクトリ."""
    suffix = CONDITION_TO_ARCHIVE_SUFFIX[condition]
    return Path(analysis_root) / benchmark / model / suffix


def load_analysis_sample_results(
    analysis_root: Path | str, model: str, benchmark: str, condition: str
) -> list[dict] | None:
    """full_results.json の sample_results を返す (無ければ None)."""
    path = analysis_condition_dir(analysis_root, model, benchmark, condition) / "full_results.json"
    if not path.exists():
        return None
    data = load_json(path)
    return data.get("sample_results", [])


def load_analysis_partial_correlations(
    analysis_root: Path | str, model: str, benchmark: str, condition: str
) -> list[dict] | None:
    """full_results.json の partial_correlations を返す (無ければ None)."""
    path = analysis_condition_dir(analysis_root, model, benchmark, condition) / "full_results.json"
    if not path.exists():
        return None
    data = load_json(path)
    return data.get("partial_correlations", [])


def load_summary_accuracy(result_dir: Path | str) -> float | None:
    """summary.json の overall accuracy を返す (無ければ None)."""
    path = Path(result_dir) / "summary.json"
    if not path.exists():
        return None
    data = load_json(path)
    return (data.get("overall_metrics") or {}).get("accuracy")
