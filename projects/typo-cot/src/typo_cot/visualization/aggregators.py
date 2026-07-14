"""分析結果JSONを横断的に集約するユーティリティ.

`outputs/analysis/<dataset>/<model>/k<k>_<type>/...json` および
`outputs/baseline/`, `outputs/perturbed/` 配下の summary を読み込み、
論文の図表生成に必要な形に整形する。

ファイル単位の読み込みだけを行い、描画・整形ロジックは含めない（responsibility分離）。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ANALYSIS_DIR_PATTERN = re.compile(r"^k(\d+)_(importance|random|bottom_k)$")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_analysis_dirs(analysis_root: Path) -> Iterable[tuple[str, str, int, str, Path]]:
    """`outputs/analysis/<dataset>/<model>/k<k>_<type>/` を列挙する.

    Yields:
        (dataset, model, k, perturbation_type, dir_path)
    """
    if not analysis_root.is_dir():
        return
    for ds_dir in sorted(analysis_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for model_dir in sorted(ds_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for k_dir in sorted(model_dir.iterdir()):
                if not k_dir.is_dir():
                    continue
                m = ANALYSIS_DIR_PATTERN.match(k_dir.name)
                if not m:
                    continue
                yield ds_dir.name, model_dir.name, int(m.group(1)), m.group(2), k_dir


def collect_overall_metrics(
    analysis_root: Path,
    pert_type: str = "importance",
) -> pd.DataFrame:
    """各 (dataset, model, k) の `exp2-a.json` / `exp2-b.json` から平均値を集約.

    Returns:
        long-form DataFrame:
            columns = [dataset, model, k, perturbation_type, metric, mean, std, n]
    """
    rows: list[dict] = []
    for ds, model, k, ptype, dir_path in iter_analysis_dirs(analysis_root):
        if ptype != pert_type:
            continue
        for json_name in ("exp2-a.json", "exp2-b.json"):
            data = _load_json(dir_path / json_name)
            if data is None:
                continue
            metrics = data.get("overall_metrics", {})
            for metric_name, stats in metrics.items():
                if not isinstance(stats, dict) or "mean" not in stats:
                    continue
                rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "k": k,
                        "perturbation_type": ptype,
                        "metric": metric_name,
                        "mean": stats["mean"],
                        "std": stats.get("std", float("nan")),
                        "n": stats.get("n", 0),
                    }
                )
    return pd.DataFrame(rows)


def collect_q_cot_correlations(
    analysis_root: Path,
    k: int = 4,
    pert_type: str = "importance",
) -> pd.DataFrame:
    """各 (dataset, model) の `exp3.json` から Q↔CoT Spearman ρ を集約.

    Figure 3 用。group_name == "all" のみを集める。
    """
    rows: list[dict] = []
    for ds, model, k_val, ptype, dir_path in iter_analysis_dirs(analysis_root):
        if ptype != pert_type or k_val != k:
            continue
        data = _load_json(dir_path / "exp3.json")
        if data is None:
            continue
        for c in data.get("correlations", []):
            if c.get("group_name", "all") != "all":
                continue
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "k": k_val,
                    "var_q": c["variable1"],
                    "var_cot": c["variable2"],
                    "spearman_rho": c["spearman_rho"],
                    "spearman_p": c["spearman_p"],
                    "n": c.get("n", 0),
                }
            )
    return pd.DataFrame(rows)


def collect_exclusion_stats(
    analysis_root: Path,
) -> pd.DataFrame:
    """各 (dataset, model, k, perturbation_type) の回答スパン未検出による除外統計を集約.

    `full_results.json` の `metadata.total_samples` および
    `metadata.excluded_no_answer_count` から、除外件数と割合を計算する。

    Returns:
        DataFrame: columns = [
            dataset, model, k, perturbation_type,
            total_samples, excluded_count, total_with_excluded, excluded_pct
        ]
        total_with_excluded = total_samples + excluded_count（除外前のサンプル数）
        excluded_pct = excluded_count / total_with_excluded * 100
    """
    rows: list[dict] = []
    for ds, model, k_val, ptype, dir_path in iter_analysis_dirs(analysis_root):
        data = _load_json(dir_path / "full_results.json")
        if data is None:
            continue
        md = data.get("metadata", {})
        total = md.get("total_samples")
        excluded = md.get("excluded_no_answer_count", 0)
        if total is None:
            continue
        total_with_excluded = total + excluded
        excluded_pct = (excluded / total_with_excluded * 100) if total_with_excluded > 0 else 0.0
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "k": k_val,
                "perturbation_type": ptype,
                "total_samples": total,
                "excluded_count": excluded,
                "total_with_excluded": total_with_excluded,
                "excluded_pct": excluded_pct,
            }
        )
    return pd.DataFrame(rows)


def collect_partial_correlations(
    analysis_root: Path,
    k: int = 4,
    pert_type: str = "importance",
) -> pd.DataFrame:
    """各 (dataset, model) の `full_results.json` から偏相関を集約.

    Table 3 / Table 6 用。`partial_correlations` 配列をそのまま long-form に展開。
    """
    rows: list[dict] = []
    for ds, model, k_val, ptype, dir_path in iter_analysis_dirs(analysis_root):
        if ptype != pert_type or k_val != k:
            continue
        data = _load_json(dir_path / "full_results.json")
        if data is None:
            continue
        for pc in data.get("partial_correlations", []):
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "k": k_val,
                    "variable": pc["variable"],
                    "control_variable": pc["control_variable"],
                    "target_variable": pc["target_variable"],
                    "group": pc.get("group", "all"),
                    "n": pc["n"],
                    "partial_r": pc["partial_r"],
                    "partial_p": pc["partial_p"],
                }
            )
    return pd.DataFrame(rows)


def collect_accuracy_summary(
    baseline_root: Path,
    perturbed_root: Path,
    models: Iterable[str],
    benchmarks: Iterable[str],
    ks: Iterable[int] = (1, 2, 4, 8),
) -> pd.DataFrame:
    """baseline / perturbed の `summary.json` から accuracy を集約.

    Table 5 用。Ori. = baseline accuracy、LXT-k = perturbed (importance) accuracy、
    Rnd-k = perturbed (random) accuracy。
    """
    rows: list[dict] = []
    for model in models:
        for bench in benchmarks:
            base = _load_json(baseline_root / f"{model}_{bench}" / "summary.json")
            base_acc = (
                base.get("overall_metrics", {}).get("accuracy") if base else None
            )
            row: dict = {
                "model": model,
                "benchmark": bench,
                "original": base_acc,
            }
            for k in ks:
                for pert_type, label in (("importance", "LXT"), ("random", "Rnd")):
                    pert = _load_json(
                        perturbed_root
                        / f"{model}_{bench}_k{k}_{pert_type}"
                        / "summary.json"
                    )
                    acc = (
                        pert.get("overall_metrics", {}).get("accuracy") if pert else None
                    )
                    row[f"{label}_{k}"] = acc
            rows.append(row)
    return pd.DataFrame(rows)


def load_sample_results(analysis_dir: Path) -> pd.DataFrame:
    """単一 `<analysis_dir>/full_results.json` の `sample_results` を DataFrame 化.

    Table 6（C→I 偏相関）の再計算用。各サンプルの metrics をフラット化する。
    """
    data = _load_json(analysis_dir / "full_results.json")
    if data is None:
        return pd.DataFrame()
    rows: list[dict] = []
    for sr in data.get("sample_results", []):
        row: dict = {
            "sample_id": sr["sample_id"],
            "pattern": sr["pattern"],
            "answer_changed": sr["answer_changed"],
            "before_correct": sr["before_correct"],
            "after_correct": sr["after_correct"],
        }
        # question_metrics の構造例: {"spearman_r": float, "jaccard": {"top3":..., "top10":...}}
        qm = sr.get("question_metrics") or {}
        for key, value in qm.items():
            if isinstance(value, dict):
                for sub_k, sub_v in value.items():
                    row[f"q_{key}_{sub_k}"] = sub_v
            else:
                row[f"q_{key}"] = value
        # cot_metrics の構造例:
        #   {"rouge_l": {"precision":..., "recall":..., "f1":...},
        #    "jaccard": {"top3":..., "top10":...}}
        cm = sr.get("cot_metrics") or {}
        for key, value in cm.items():
            if isinstance(value, dict):
                for sub_k, sub_v in value.items():
                    row[f"cot_{key}_{sub_k}"] = sub_v
            else:
                row[f"cot_{key}"] = value
        # Table 6 互換のエイリアス: cot_jaccard_top{k} / cot_rouge_l_f1
        for k in (3, 5, 10, 15, 20):
            v = (cm.get("jaccard") or {}).get(f"top{k}")
            if v is not None:
                row[f"cot_jaccard_top{k}"] = v
        rouge = cm.get("rouge_l")
        if isinstance(rouge, dict) and "f1" in rouge:
            row["cot_rouge_l_f1"] = rouge["f1"]
        rows.append(row)
    return pd.DataFrame(rows)
