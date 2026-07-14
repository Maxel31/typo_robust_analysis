#!/usr/bin/env python3
"""Weights & Biases への実験結果アップロードスクリプト.

outputs/analysis 以下のJSONファイルを読み取り、
実験シートに対応する統合テーブルとグラフをW&B上に自動生成する。

実験構成:
- 実験1: lxp-perturbation後の推論性能
- 実験2-a: 質問文への影響指標
- 実験2-b: CoT推論過程への影響指標
- 実験3: Q指標とCoT指標の相関
- 実験4-a: CoT指標と回答変化の偏相関
- 実験4-b: CoT指標と不正解転落の偏相関

使用方法:
    # 一括アップロード
    uv run python scripts/upload_to_wandb.py --analysis_dir outputs/analysis

    # 監視モード（バックグラウンド実行）
    nohup uv run python scripts/upload_to_wandb.py --analysis_dir outputs/analysis --watch &
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy import stats

# .envファイルからAPIキーを読み込み
load_dotenv()

from watchdog.events import FileSystemEventHandler  # noqa: E402
from watchdog.observers import Observer  # noqa: E402

import wandb  # noqa: E402

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Run ID保存ファイル
RUN_ID_FILE = ".wandb_run_id"

# グローバルカラーパレット
MODEL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def format_pvalue(p: float | None) -> str:
    """p-valueをフォーマットする.

    極小値の場合は科学的表記を使用する。

    Args:
        p: p-value（Noneの場合は空文字を返す）

    Returns:
        フォーマットされた文字列
    """
    if p is None:
        return ""
    if p < 1e-100:
        return "<1e-100"
    if p < 0.0001:
        # 科学的表記: 1.23e-05 のように表示
        return f"{p:.2e}"
    return f"{p:.4f}"


class ExperimentDataCollector:
    """実験データを収集・整理するクラス."""

    def __init__(self):
        """初期化."""
        self.exp1_data: list[dict] = []  # exp1: Accuracy変化
        self.full_results_data: list[dict] = []  # サンプルレベルデータ
        self.loaded_files: set[str] = set()

    def clear(self) -> None:
        """データをクリア."""
        self.exp1_data.clear()
        self.full_results_data.clear()
        self.loaded_files.clear()

    def load_file(self, json_path: Path) -> bool:
        """JSONファイルを読み込む."""
        file_key = str(json_path)
        if file_key in self.loaded_files:
            return False

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            filename = json_path.name
            if filename == "full_results.json":
                self.full_results_data.append(data)
            else:
                return False

            self.loaded_files.add(file_key)
            logger.info(f"読み込み完了: {json_path.name}")
            return True

        except Exception as e:
            logger.error(f"読み込みエラー: {json_path}: {e}")
            return False

    def load_directory(self, analysis_dir: Path) -> int:
        """ディレクトリ内のJSONファイルを読み込む."""
        count = 0
        for json_file in sorted(analysis_dir.rglob("full_results.json")):
            if self.load_file(json_file):
                count += 1
        return count

    def load_exp1_data(self, outputs_dir: Path) -> int:
        """exp1用データを baseline/perturbed ディレクトリから読み込む."""
        baseline_dir = outputs_dir / "baseline"
        perturbed_dir = outputs_dir / "perturbed"
        count = 0

        # Baseline データの読み込み
        if baseline_dir.exists():
            for summary_file in baseline_dir.glob("*/summary.json"):
                file_key = str(summary_file)
                if file_key in self.loaded_files:
                    continue

                try:
                    with open(summary_file, encoding="utf-8") as f:
                        data = json.load(f)

                    exp_info = data.get("experiment_info", {})
                    model_full = exp_info.get("model", "")
                    model = model_full.split("/")[-1] if "/" in model_full else model_full
                    benchmark = exp_info.get("benchmark", "")

                    if not model or not benchmark:
                        continue

                    self.exp1_data.append(
                        {
                            "type": "baseline",
                            "model": model,
                            "benchmark": benchmark,
                            "k": 0,
                            "perturbation_type": None,
                            "accuracy": data.get("overall_metrics", {}).get("accuracy", 0),
                            "em_score": data.get("overall_metrics", {}).get("em_score", 0),
                            "total_samples": data.get("overall_metrics", {}).get(
                                "total_samples", 0
                            ),
                        }
                    )
                    self.loaded_files.add(file_key)
                    count += 1
                except Exception as e:
                    logger.error(f"exp1 baseline読み込みエラー: {summary_file}: {e}")

        # Perturbed データの読み込み
        if perturbed_dir.exists():
            pattern = re.compile(r"_k(\d+)_(importance|random|bottom_k)$")

            for summary_file in perturbed_dir.glob("*/summary.json"):
                file_key = str(summary_file)
                if file_key in self.loaded_files:
                    continue

                try:
                    dir_name = summary_file.parent.name
                    match = pattern.search(dir_name)
                    if not match:
                        continue

                    k = int(match.group(1))
                    ptype = match.group(2)

                    with open(summary_file, encoding="utf-8") as f:
                        data = json.load(f)

                    exp_info = data.get("experiment_info", {})
                    model_full = exp_info.get("model", "")
                    model = model_full.split("/")[-1] if "/" in model_full else model_full
                    benchmark = exp_info.get("benchmark", "")

                    if not model or not benchmark:
                        continue

                    self.exp1_data.append(
                        {
                            "type": "perturbed",
                            "model": model,
                            "benchmark": benchmark,
                            "k": k,
                            "perturbation_type": ptype,
                            "accuracy": data.get("overall_metrics", {}).get("accuracy", 0),
                            "em_score": data.get("overall_metrics", {}).get("em_score", 0),
                            "total_samples": data.get("overall_metrics", {}).get(
                                "total_samples", 0
                            ),
                        }
                    )
                    self.loaded_files.add(file_key)
                    count += 1
                except Exception as e:
                    logger.error(f"exp1 perturbed読み込みエラー: {summary_file}: {e}")

        return count


# ========================================
# 実験1: 推論性能テーブル・グラフ
# ========================================


def create_exp1_table(data_list: list[dict]) -> wandb.Table:
    """実験1: 推論性能テーブル.

    カラム: ベンチマーク, モデル, baseline, random-4, lxp-1, lxp-2, lxp-4, lxp-8, anti-lxp-1, anti-lxp-2, anti-lxp-4, anti-lxp-8
    """
    columns = [
        "Benchmark",
        "Model",
        "Baseline",
        "Random-4",
        "LXP-1",
        "LXP-2",
        "LXP-4",
        "LXP-8",
        "Anti-LXP-1",
        "Anti-LXP-2",
        "Anti-LXP-4",
        "Anti-LXP-8",
    ]
    table = wandb.Table(columns=columns)

    # Dataset x Model でグループ化
    grouped: dict[tuple[str, str], dict] = {}
    for data in data_list:
        key = (data["benchmark"], data["model"])
        if key not in grouped:
            grouped[key] = {"baseline": None, "importance": {}, "random": {}, "bottom_k": {}}

        benchmark = data["benchmark"]
        # SQuAD-v2はEMスコア、それ以外はAccuracy
        score_key = "em_score" if "squad" in benchmark.lower() else "accuracy"
        score = data.get(score_key, data.get("accuracy", 0))

        if data["type"] == "baseline":
            grouped[key]["baseline"] = score
        elif data["type"] == "perturbed":
            ptype = data["perturbation_type"]
            k = data["k"]
            grouped[key][ptype][k] = score

    # 行を追加
    for (benchmark, model), data in sorted(grouped.items()):
        baseline = data["baseline"]
        imp = data["importance"]
        rand = data["random"]
        bottom = data["bottom_k"]

        table.add_data(
            benchmark,
            model,
            f"{baseline:.2%}" if baseline is not None else "",
            f"{rand.get(4):.2%}" if rand.get(4) is not None else "",
            f"{imp.get(1):.2%}" if imp.get(1) is not None else "",
            f"{imp.get(2):.2%}" if imp.get(2) is not None else "",
            f"{imp.get(4):.2%}" if imp.get(4) is not None else "",
            f"{imp.get(8):.2%}" if imp.get(8) is not None else "",
            f"{bottom.get(1):.2%}" if bottom.get(1) is not None else "",
            f"{bottom.get(2):.2%}" if bottom.get(2) is not None else "",
            f"{bottom.get(4):.2%}" if bottom.get(4) is not None else "",
            f"{bottom.get(8):.2%}" if bottom.get(8) is not None else "",
        )

    return table


def create_exp1_charts(run: wandb.sdk.wandb_run.Run, data_list: list[dict]) -> None:
    """実験1: ベンチマーク別の推論性能折れ線グラフ."""
    # Dataset別にグループ化
    by_dataset: dict[str, list[dict]] = {}
    for data in data_list:
        dataset = data["benchmark"]
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(data)

    for dataset, dataset_data in sorted(by_dataset.items()):
        # SQuAD-v2はEMスコア、それ以外はAccuracy
        score_key = "em_score" if "squad" in dataset.lower() else "accuracy"
        y_label = "EM Score" if "squad" in dataset.lower() else "Accuracy"

        # Model別のデータ
        models = sorted({d["model"] for d in dataset_data})
        model_color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

        fig = go.Figure()

        for model in models:
            color = model_color_map[model]
            model_data = [d for d in dataset_data if d["model"] == model]

            # Baseline
            baseline_score = None
            for d in model_data:
                if d["type"] == "baseline":
                    baseline_score = d.get(score_key, d.get("accuracy", 0))
                    break

            # lxp-perturbation (importance)
            imp_points = []
            for d in model_data:
                if d["type"] == "perturbed" and d["perturbation_type"] == "importance":
                    score = d.get(score_key, d.get("accuracy", 0))
                    imp_points.append((d["k"], score))
            imp_points.sort(key=lambda x: x[0])

            # anti-lxp-perturbation (bottom_k)
            bottom_points = []
            for d in model_data:
                if d["type"] == "perturbed" and d["perturbation_type"] == "bottom_k":
                    score = d.get(score_key, d.get("accuracy", 0))
                    bottom_points.append((d["k"], score))
            bottom_points.sort(key=lambda x: x[0])

            # Random k=4
            random_k4_score = None
            for d in model_data:
                if d["type"] == "perturbed" and d["perturbation_type"] == "random" and d["k"] == 4:
                    random_k4_score = d.get(score_key, d.get("accuracy", 0))
                    break

            # lxp-perturbation 折れ線
            if imp_points:
                fig.add_trace(
                    go.Scatter(
                        x=[p[0] for p in imp_points],
                        y=[p[1] for p in imp_points],
                        mode="lines+markers",
                        name=f"{model} (LXP)",
                        line={"width": 3, "color": color},
                        marker={"size": 10, "symbol": "circle"},
                    )
                )

            # anti-lxp-perturbation 折れ線
            if bottom_points:
                fig.add_trace(
                    go.Scatter(
                        x=[p[0] for p in bottom_points],
                        y=[p[1] for p in bottom_points],
                        mode="lines+markers",
                        name=f"{model} (Anti-LXP)",
                        line={"width": 3, "color": color, "dash": "dot"},
                        marker={"size": 10, "symbol": "diamond"},
                    )
                )

            # Baseline 水平線
            all_points = imp_points + bottom_points
            if baseline_score is not None and all_points:
                k_range = [min(p[0] for p in all_points), max(p[0] for p in all_points)]
                fig.add_trace(
                    go.Scatter(
                        x=k_range,
                        y=[baseline_score, baseline_score],
                        mode="lines",
                        name=f"{model} (Baseline)",
                        line={"width": 2, "color": color, "dash": "dash"},
                    )
                )

            # Random k=4 点
            if random_k4_score is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[4],
                        y=[random_k4_score],
                        mode="markers",
                        name=f"{model} (Random-4)",
                        marker={"size": 14, "symbol": "x", "color": color, "line": {"width": 2}},
                    )
                )

        fig.update_layout(
            title=f"[{dataset}] 実験1: 推論性能 by k",
            xaxis_title="k (摂動回数)",
            yaxis_title=y_label,
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            template="plotly_white",
        )
        run.log({f"exp1/{dataset}/performance_by_k": wandb.Plotly(fig)})


# ========================================
# 実験2-a: 質問文への影響指標
# ========================================


def create_exp2a_table(
    aggregated_data: dict[str, dict[str, dict[int, dict]]],
) -> wandb.Table:
    """実験2-a: 質問文への影響指標テーブル.

    カラム: Benchmark, Model, Metric, k=1, k=2, k=4, k=8
    """
    columns = ["Benchmark", "Model", "Metric", "k=1", "k=2", "k=4", "k=8"]
    table = wandb.Table(columns=columns)

    metrics = [
        ("token_diff", "ΔToken Num"),
        ("q_jaccard_3", "Q:Jaccard@3"),
        ("q_jaccard_5", "Q:Jaccard@5"),
        ("q_jaccard_10", "Q:Jaccard@10"),
        ("q_spearman_r", "Q:Spearman-ρ"),
    ]

    for benchmark, model_data in sorted(aggregated_data.items()):
        for model, k_data in sorted(model_data.items()):
            for metric_key, metric_name in metrics:
                row = [benchmark, model, metric_name]
                for k in [1, 2, 4, 8]:
                    if k in k_data and metric_key in k_data[k]:
                        vals = k_data[k][metric_key]
                        if vals:
                            mean_val = np.mean(vals)
                            std_val = np.std(vals)
                            row.append(f"{mean_val:.3f}±{std_val:.3f}")
                        else:
                            row.append("")
                    else:
                        row.append("")
                table.add_data(*row)

    return table


def create_exp2a_charts(
    run: wandb.sdk.wandb_run.Run,
    aggregated_data: dict[str, dict[str, dict[int, dict]]],
) -> None:
    """実験2-a: 質問文への影響指標折れ線グラフ."""
    metrics = [
        ("token_diff", "ΔToken Num"),
        ("q_jaccard_3", "Q:Jaccard@3"),
        ("q_jaccard_5", "Q:Jaccard@5"),
        ("q_jaccard_10", "Q:Jaccard@10"),
        ("q_spearman_r", "Q:Spearman-ρ"),
    ]

    for benchmark, model_data in sorted(aggregated_data.items()):
        models = sorted(model_data.keys())
        model_color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

        for metric_key, metric_name in metrics:
            fig = go.Figure()

            for model in models:
                color = model_color_map[model]
                k_data = model_data[model]

                points = []
                for k in sorted(k_data):
                    if metric_key in k_data[k] and k_data[k][metric_key]:
                        mean_val = np.mean(k_data[k][metric_key])
                        points.append((k, mean_val))

                if points:
                    fig.add_trace(
                        go.Scatter(
                            x=[p[0] for p in points],
                            y=[p[1] for p in points],
                            mode="lines+markers",
                            name=model,
                            line={"width": 2, "color": color},
                            marker={"size": 8},
                        )
                    )

            fig.update_layout(
                title=f"[{benchmark}] 実験2-a: {metric_name} by k",
                xaxis_title="k (摂動回数)",
                yaxis_title=metric_name,
                xaxis={"tickmode": "linear", "dtick": 1},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                template="plotly_white",
            )
            safe_key = metric_key.replace("@", "_at_")
            run.log({f"exp2a/{benchmark}/{safe_key}": wandb.Plotly(fig)})


# ========================================
# 実験2-b: CoT推論過程への影響指標
# ========================================


def create_exp2b_table(
    aggregated_data: dict[str, dict[str, dict[int, dict]]],
) -> wandb.Table:
    """実験2-b: CoT推論過程への影響指標テーブル.

    カラム: Benchmark, Model, Metric, k=1, k=2, k=4, k=8
    """
    columns = ["Benchmark", "Model", "Metric", "k=1", "k=2", "k=4", "k=8"]
    table = wandb.Table(columns=columns)

    metrics = [
        ("cot_rouge_l", "CoT:ROUGE-L"),
        ("cot_jaccard_3", "CoT:Jaccard@3"),
        ("cot_jaccard_5", "CoT:Jaccard@5"),
        ("cot_jaccard_10", "CoT:Jaccard@10"),
        ("cot_jaccard_15", "CoT:Jaccard@15"),
        ("cot_jaccard_20", "CoT:Jaccard@20"),
    ]

    for benchmark, model_data in sorted(aggregated_data.items()):
        for model, k_data in sorted(model_data.items()):
            for metric_key, metric_name in metrics:
                row = [benchmark, model, metric_name]
                for k in [1, 2, 4, 8]:
                    if k in k_data and metric_key in k_data[k]:
                        vals = k_data[k][metric_key]
                        if vals:
                            mean_val = np.mean(vals)
                            std_val = np.std(vals)
                            row.append(f"{mean_val:.3f}±{std_val:.3f}")
                        else:
                            row.append("")
                    else:
                        row.append("")
                table.add_data(*row)

    return table


def create_exp2b_charts(
    run: wandb.sdk.wandb_run.Run,
    aggregated_data: dict[str, dict[str, dict[int, dict]]],
) -> None:
    """実験2-b: CoT推論過程への影響指標折れ線グラフ."""
    metrics = [
        ("cot_rouge_l", "CoT:ROUGE-L"),
        ("cot_jaccard_3", "CoT:Jaccard@3"),
        ("cot_jaccard_5", "CoT:Jaccard@5"),
        ("cot_jaccard_10", "CoT:Jaccard@10"),
        ("cot_jaccard_15", "CoT:Jaccard@15"),
        ("cot_jaccard_20", "CoT:Jaccard@20"),
    ]

    for benchmark, model_data in sorted(aggregated_data.items()):
        models = sorted(model_data.keys())
        model_color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

        for metric_key, metric_name in metrics:
            fig = go.Figure()

            for model in models:
                color = model_color_map[model]
                k_data = model_data[model]

                points = []
                for k in sorted(k_data):
                    if metric_key in k_data[k] and k_data[k][metric_key]:
                        mean_val = np.mean(k_data[k][metric_key])
                        points.append((k, mean_val))

                if points:
                    fig.add_trace(
                        go.Scatter(
                            x=[p[0] for p in points],
                            y=[p[1] for p in points],
                            mode="lines+markers",
                            name=model,
                            line={"width": 2, "color": color},
                            marker={"size": 8},
                        )
                    )

            fig.update_layout(
                title=f"[{benchmark}] 実験2-b: {metric_name} by k",
                xaxis_title="k (摂動回数)",
                yaxis_title=metric_name,
                xaxis={"tickmode": "linear", "dtick": 1},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                template="plotly_white",
            )
            safe_key = metric_key.replace("@", "_at_")
            run.log({f"exp2b/{benchmark}/{safe_key}": wandb.Plotly(fig)})


# ========================================
# 実験3: Q指標とCoT指標の相関
# ========================================


def create_exp3_table(
    correlation_data: dict[str, dict[str, dict[int, dict[str, tuple[float, float]]]]],
) -> wandb.Table:
    """実験3: 相関テーブル（モデル別）.

    カラム: Benchmark, Model, k, Q-Metric, CoT-Metric, ρ, p-value
    """
    columns = ["Benchmark", "Model", "k", "Q-Metric", "CoT-Metric", "ρ", "p-value"]
    table = wandb.Table(columns=columns)

    for benchmark, model_data in sorted(correlation_data.items()):
        for model, k_data in sorted(model_data.items()):
            for k, corr_data in sorted(k_data.items()):
                for corr_key, (rho, p) in corr_data.items():
                    # corr_key format: "q_metric|cot_metric"
                    parts = corr_key.split("|")
                    if len(parts) == 2:
                        q_metric, cot_metric = parts
                        sig = (
                            "***"
                            if p < 0.001
                            else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                        )
                        table.add_data(
                            benchmark,
                            model,
                            k,
                            q_metric,
                            cot_metric,
                            f"{rho:.3f}{sig}",
                            format_pvalue(p),
                        )

    return table


def create_exp3_heatmaps(
    run: wandb.sdk.wandb_run.Run,
    correlation_data: dict[str, dict[str, dict[int, dict[str, tuple[float, float]]]]],
) -> None:
    """実験3: 相関ヒートマップ（モデル別・k別サブプロット）."""
    q_metrics = ["ΔToken Num", "Q:Jaccard@3", "Q:Jaccard@5", "Q:Jaccard@10", "Q:Spearman-ρ"]
    cot_metrics = [
        "CoT:ROUGE-L",
        "CoT:Jaccard@3",
        "CoT:Jaccard@5",
        "CoT:Jaccard@10",
        "CoT:Jaccard@15",
        "CoT:Jaccard@20",
    ]

    for benchmark, model_data in sorted(correlation_data.items()):
        for model, k_data in sorted(model_data.items()):
            k_values = sorted(k_data)
            if not k_values:
                continue

            fig = make_subplots(
                rows=1,
                cols=len(k_values),
                subplot_titles=[f"k={k}" for k in k_values],
                horizontal_spacing=0.05,
            )

            for col_idx, k in enumerate(k_values, 1):
                corr_matrix = np.zeros((len(q_metrics), len(cot_metrics)))

                for i, q_m in enumerate(q_metrics):
                    for j, cot_m in enumerate(cot_metrics):
                        key = f"{q_m}|{cot_m}"
                        if key in k_data[k]:
                            rho, _p = k_data[k][key]
                            corr_matrix[i, j] = rho

                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix,
                        x=cot_metrics,
                        y=q_metrics,
                        colorscale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        showscale=(col_idx == len(k_values)),
                        text=[
                            [f"{corr_matrix[i, j]:.2f}" for j in range(len(cot_metrics))]
                            for i in range(len(q_metrics))
                        ],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                    ),
                    row=1,
                    col=col_idx,
                )

            fig.update_layout(
                title=f"[{benchmark}][{model}] 実験3: Q指標 vs CoT指標 相関",
                height=400,
                template="plotly_white",
            )
            run.log({f"exp3/{benchmark}/{model}/correlation_heatmap": wandb.Plotly(fig)})


# ========================================
# 実験4-a: CoT指標と回答変化の偏相関
# ========================================


def create_exp4a_table(
    partial_corr_data: dict[str, dict[str, dict[int, dict]]],
) -> wandb.Table:
    """実験4-a: 偏相関テーブル（モデル別）.

    各Jaccard@m (m=3,5,10) について、ROUGE-L|Jaccard@m と Jaccard@m|ROUGE-L の偏相関を表示.
    """
    columns = [
        "Benchmark",
        "Model",
        "k",
        "N(Ans_unch)",
        "N(Ans_ch)",
        "Jaccard@m",
        "ROUGE-L|Jaccard ρ",
        "ROUGE-L|Jaccard p",
        "Jaccard|ROUGE-L ρ",
        "Jaccard|ROUGE-L p",
    ]
    table = wandb.Table(columns=columns)

    jaccard_keys = [3, 5, 10, 15, 20]

    for benchmark, model_data in sorted(partial_corr_data.items()):
        for model, k_data in sorted(model_data.items()):
            # k値をソート（整数を先に、"average"を最後に）
            int_keys = sorted([k for k in k_data if isinstance(k, int)])
            str_keys = sorted([k for k in k_data if isinstance(k, str)])
            sorted_keys = int_keys + str_keys

            for k in sorted_keys:
                data = k_data[k]
                for m in jaccard_keys:
                    rouge_rho = data.get(f"rouge_l_partial_{m}", (None, None))[0]
                    rouge_p = data.get(f"rouge_l_partial_{m}", (None, None))[1]
                    jaccard_rho = data.get(f"jaccard_partial_{m}", (None, None))[0]
                    jaccard_p = data.get(f"jaccard_partial_{m}", (None, None))[1]

                    table.add_data(
                        benchmark,
                        model,
                        str(k),  # kを文字列に変換（整数と"average"の混在対応）
                        data.get("n_unchanged", 0),
                        data.get("n_changed", 0),
                        f"@{m}",
                        f"{rouge_rho:.3f}" if rouge_rho is not None else "",
                        format_pvalue(rouge_p),
                        f"{jaccard_rho:.3f}" if jaccard_rho is not None else "",
                        format_pvalue(jaccard_p),
                    )

    return table


def create_exp4a_q_table(
    partial_corr_data: dict[str, dict[str, dict[int, dict]]],
) -> wandb.Table:
    """実験4-a: Q指標と回答変化の相関テーブル（モデル別）.

    Q:ΔToken-Num, Q:Jaccard@Top-k, Q:Spearman-r と回答変化の相関を表示.
    """
    columns = [
        "Benchmark",
        "Model",
        "k",
        "N(Ans_unch)",
        "N(Ans_ch)",
        "Q:ΔToken ρ",
        "Q:ΔToken p",
        "Q:Jaccard@3 ρ",
        "Q:Jaccard@3 p",
        "Q:Jaccard@5 ρ",
        "Q:Jaccard@5 p",
        "Q:Jaccard@10 ρ",
        "Q:Jaccard@10 p",
        "Q:Spearman ρ",
        "Q:Spearman p",
    ]
    table = wandb.Table(columns=columns)

    for benchmark, model_data in sorted(partial_corr_data.items()):
        for model, k_data in sorted(model_data.items()):
            # k値をソート（整数を先に、"average"を最後に）
            int_keys = sorted([k for k in k_data if isinstance(k, int)])
            str_keys = sorted([k for k in k_data if isinstance(k, str)])
            sorted_keys = int_keys + str_keys

            for k in sorted_keys:
                data = k_data[k]
                token_rho, token_p = data.get("q_token_diff", (None, None))
                j3_rho, j3_p = data.get("q_jaccard_3", (None, None))
                j5_rho, j5_p = data.get("q_jaccard_5", (None, None))
                j10_rho, j10_p = data.get("q_jaccard_10", (None, None))
                sp_rho, sp_p = data.get("q_spearman_r", (None, None))

                table.add_data(
                    benchmark,
                    model,
                    str(k),  # kを文字列に変換（整数と"average"の混在対応）
                    data.get("n_unchanged", 0),
                    data.get("n_changed", 0),
                    f"{token_rho:.3f}" if token_rho is not None else "",
                    format_pvalue(token_p),
                    f"{j3_rho:.3f}" if j3_rho is not None else "",
                    format_pvalue(j3_p),
                    f"{j5_rho:.3f}" if j5_rho is not None else "",
                    format_pvalue(j5_p),
                    f"{j10_rho:.3f}" if j10_rho is not None else "",
                    format_pvalue(j10_p),
                    f"{sp_rho:.3f}" if sp_rho is not None else "",
                    format_pvalue(sp_p),
                )

    return table


# ========================================
# 実験4-b: CoT指標と不正解転落の偏相関
# ========================================


def create_exp4b_table(
    partial_corr_data: dict[str, dict[str, dict[int, dict]]],
) -> wandb.Table:
    """実験4-b: 偏相関テーブル（正解→正解 vs 正解→不正解、モデル別）.

    各Jaccard@m (m=3,5,10) について、ROUGE-L|Jaccard@m と Jaccard@m|ROUGE-L の偏相関を表示.
    """
    columns = [
        "Benchmark",
        "Model",
        "k",
        "N(C→C)",
        "N(C→I)",
        "Jaccard@m",
        "ROUGE-L|Jaccard ρ",
        "ROUGE-L|Jaccard p",
        "Jaccard|ROUGE-L ρ",
        "Jaccard|ROUGE-L p",
    ]
    table = wandb.Table(columns=columns)

    jaccard_keys = [3, 5, 10, 15, 20]

    for benchmark, model_data in sorted(partial_corr_data.items()):
        for model, k_data in sorted(model_data.items()):
            # k値をソート（整数を先に、"average"を最後に）
            int_keys = sorted([k for k in k_data if isinstance(k, int)])
            str_keys = sorted([k for k in k_data if isinstance(k, str)])
            sorted_keys = int_keys + str_keys

            for k in sorted_keys:
                data = k_data[k]
                for m in jaccard_keys:
                    rouge_rho = data.get(f"rouge_l_partial_{m}", (None, None))[0]
                    rouge_p = data.get(f"rouge_l_partial_{m}", (None, None))[1]
                    jaccard_rho = data.get(f"jaccard_partial_{m}", (None, None))[0]
                    jaccard_p = data.get(f"jaccard_partial_{m}", (None, None))[1]

                    table.add_data(
                        benchmark,
                        model,
                        str(k),  # kを文字列に変換（整数と"average"の混在対応）
                        data.get("n_cc", 0),
                        data.get("n_ci", 0),
                        f"@{m}",
                        f"{rouge_rho:.3f}" if rouge_rho is not None else "",
                        format_pvalue(rouge_p),
                        f"{jaccard_rho:.3f}" if jaccard_rho is not None else "",
                        format_pvalue(jaccard_p),
                    )

    return table


def create_exp4b_q_table(
    partial_corr_data: dict[str, dict[str, dict[int, dict]]],
) -> wandb.Table:
    """実験4-b: Q指標と不正解転落の相関テーブル（モデル別）.

    Q:ΔToken-Num, Q:Jaccard@Top-k, Q:Spearman-r と不正解転落の相関を表示.
    """
    columns = [
        "Benchmark",
        "Model",
        "k",
        "N(C→C)",
        "N(C→I)",
        "Q:ΔToken ρ",
        "Q:ΔToken p",
        "Q:Jaccard@3 ρ",
        "Q:Jaccard@3 p",
        "Q:Jaccard@5 ρ",
        "Q:Jaccard@5 p",
        "Q:Jaccard@10 ρ",
        "Q:Jaccard@10 p",
        "Q:Spearman ρ",
        "Q:Spearman p",
    ]
    table = wandb.Table(columns=columns)

    for benchmark, model_data in sorted(partial_corr_data.items()):
        for model, k_data in sorted(model_data.items()):
            # k値をソート（整数を先に、"average"を最後に）
            int_keys = sorted([k for k in k_data if isinstance(k, int)])
            str_keys = sorted([k for k in k_data if isinstance(k, str)])
            sorted_keys = int_keys + str_keys

            for k in sorted_keys:
                data = k_data[k]
                token_rho, token_p = data.get("q_token_diff", (None, None))
                j3_rho, j3_p = data.get("q_jaccard_3", (None, None))
                j5_rho, j5_p = data.get("q_jaccard_5", (None, None))
                j10_rho, j10_p = data.get("q_jaccard_10", (None, None))
                sp_rho, sp_p = data.get("q_spearman_r", (None, None))

                table.add_data(
                    benchmark,
                    model,
                    str(k),  # kを文字列に変換（整数と"average"の混在対応）
                    data.get("n_cc", 0),
                    data.get("n_ci", 0),
                    f"{token_rho:.3f}" if token_rho is not None else "",
                    format_pvalue(token_p),
                    f"{j3_rho:.3f}" if j3_rho is not None else "",
                    format_pvalue(j3_p),
                    f"{j5_rho:.3f}" if j5_rho is not None else "",
                    format_pvalue(j5_p),
                    f"{j10_rho:.3f}" if j10_rho is not None else "",
                    format_pvalue(j10_p),
                    f"{sp_rho:.3f}" if sp_rho is not None else "",
                    format_pvalue(sp_p),
                )

    return table


# ========================================
# メインの集計・アップロード処理
# ========================================


def aggregate_sample_data(
    full_results_data: list[dict],
) -> dict[str, dict[str, dict[int, dict]]]:
    """サンプルデータを集約.

    Returns:
        {benchmark: {model: {k: {metric: [values], samples: [...]}}}}
    """
    aggregated: dict[str, dict[str, dict[int, dict]]] = {}

    for data in full_results_data:
        meta = data.get("metadata", {})
        benchmark = meta.get("dataset", "unknown")
        model = meta.get("model", "unknown")
        k = meta.get("num_perturbations", 0)
        ptype = meta.get("perturbation_type", "unknown")

        # lxp-perturbation (importance) のみを対象
        # Note: bottom_k (Anti-LXP) はExp1のみで使用し、Exp2-4の分析には含めない
        if ptype != "importance":
            continue

        if benchmark not in aggregated:
            aggregated[benchmark] = {}
        if model not in aggregated[benchmark]:
            aggregated[benchmark][model] = {}
        if k not in aggregated[benchmark][model]:
            aggregated[benchmark][model][k] = {
                "token_diff": [],
                "q_jaccard_3": [],
                "q_jaccard_5": [],
                "q_jaccard_10": [],
                "q_spearman_r": [],
                "cot_rouge_l": [],
                "cot_jaccard_3": [],
                "cot_jaccard_5": [],
                "cot_jaccard_10": [],
                "cot_jaccard_15": [],
                "cot_jaccard_20": [],
                "samples": [],
            }

        for sr in data.get("sample_results", []):
            token_count = sr.get("token_count", {})
            q_metrics = sr.get("question_metrics", {})
            cot_metrics = sr.get("cot_metrics", {})

            token_diff = token_count.get("diff", 0)
            q_jacc = q_metrics.get("jaccard", {})
            q_spearman = q_metrics.get("spearman_r", 0.0)
            rouge = cot_metrics.get("rouge_l", {}).get("f1", 0.0)
            cot_jacc = cot_metrics.get("jaccard", {})

            agg = aggregated[benchmark][model][k]
            agg["token_diff"].append(token_diff)
            agg["q_jaccard_3"].append(q_jacc.get("top3", 0.0))
            agg["q_jaccard_5"].append(q_jacc.get("top5", 0.0))
            agg["q_jaccard_10"].append(q_jacc.get("top10", 0.0))
            agg["q_spearman_r"].append(q_spearman)
            agg["cot_rouge_l"].append(rouge)
            agg["cot_jaccard_3"].append(cot_jacc.get("top3", 0.0))
            agg["cot_jaccard_5"].append(cot_jacc.get("top5", 0.0))
            agg["cot_jaccard_10"].append(cot_jacc.get("top10", 0.0))
            agg["cot_jaccard_15"].append(cot_jacc.get("top15", 0.0))
            agg["cot_jaccard_20"].append(cot_jacc.get("top20", 0.0))

            agg["samples"].append(
                {
                    "pattern": sr.get("pattern", ""),
                    "answer_changed": sr.get("answer_changed", False),
                    "token_diff": token_diff,
                    "q_jaccard_3": q_jacc.get("top3", 0.0),
                    "q_jaccard_5": q_jacc.get("top5", 0.0),
                    "q_jaccard_10": q_jacc.get("top10", 0.0),
                    "q_spearman_r": q_spearman,
                    "cot_rouge_l": rouge,
                    "cot_jaccard_3": cot_jacc.get("top3", 0.0),
                    "cot_jaccard_5": cot_jacc.get("top5", 0.0),
                    "cot_jaccard_10": cot_jacc.get("top10", 0.0),
                    "cot_jaccard_15": cot_jacc.get("top15", 0.0),
                    "cot_jaccard_20": cot_jacc.get("top20", 0.0),
                }
            )

    return aggregated


def compute_correlations(
    aggregated_data: dict[str, dict[str, dict[int, dict]]],
) -> dict[str, dict[str, dict[int, dict[str, tuple[float, float]]]]]:
    """実験3: Q指標とCoT指標の相関を計算（モデル別）.

    Returns:
        {benchmark: {model: {k: {"q_metric|cot_metric": (rho, p)}}}}
    """
    result: dict[str, dict[str, dict[int, dict[str, tuple[float, float]]]]] = {}

    q_metrics = [
        ("token_diff", "ΔToken Num"),
        ("q_jaccard_3", "Q:Jaccard@3"),
        ("q_jaccard_5", "Q:Jaccard@5"),
        ("q_jaccard_10", "Q:Jaccard@10"),
        ("q_spearman_r", "Q:Spearman-ρ"),
    ]
    cot_metrics = [
        ("cot_rouge_l", "CoT:ROUGE-L"),
        ("cot_jaccard_3", "CoT:Jaccard@3"),
        ("cot_jaccard_5", "CoT:Jaccard@5"),
        ("cot_jaccard_10", "CoT:Jaccard@10"),
        ("cot_jaccard_15", "CoT:Jaccard@15"),
        ("cot_jaccard_20", "CoT:Jaccard@20"),
    ]

    for benchmark, model_data in aggregated_data.items():
        result[benchmark] = {}

        # モデル別に相関を計算
        for model, k_data in model_data.items():
            result[benchmark][model] = {}

            for k, metrics in k_data.items():
                result[benchmark][model][k] = {}

                for q_key, q_name in q_metrics:
                    for cot_key, cot_name in cot_metrics:
                        q_vals = np.array(metrics.get(q_key, []))
                        cot_vals = np.array(metrics.get(cot_key, []))

                        if len(q_vals) >= 10 and np.std(q_vals) > 1e-9 and np.std(cot_vals) > 1e-9:
                            rho, p = stats.spearmanr(q_vals, cot_vals)
                            result[benchmark][model][k][f"{q_name}|{cot_name}"] = (rho, p)

    return result


def compute_partial_correlations_4a(
    aggregated_data: dict[str, dict[str, dict[int, dict]]],
) -> dict[str, dict[str, dict[int, dict]]]:
    """実験4-a: CoT指標・Q指標と回答変化の偏相関を計算（モデル別）.

    Returns:
        {benchmark: {model: {k: {n_unchanged, n_changed,
                        rouge_l_partial_3, rouge_l_partial_5, rouge_l_partial_10,
                        jaccard_partial_3, jaccard_partial_5, jaccard_partial_10,
                        q_token_diff, q_jaccard_3, q_jaccard_5, q_jaccard_10, q_spearman_r}}}}
    """
    try:
        import pingouin as pg
    except ImportError:
        logger.warning("pingouinがインストールされていないため、偏相関をスキップ")
        return {}

    result: dict[str, dict[str, dict[int, dict]]] = {}
    jaccard_keys = [3, 5, 10, 15, 20]

    for benchmark, model_data in aggregated_data.items():
        result[benchmark] = {}

        # モデル別に計算
        for model, k_data in model_data.items():
            result[benchmark][model] = {}

            for k, metrics in k_data.items():
                samples = metrics.get("samples", [])
                if len(samples) < 20:
                    continue

                n_unchanged = sum(1 for s in samples if not s["answer_changed"])
                n_changed = sum(1 for s in samples if s["answer_changed"])

                result[benchmark][model][k] = {
                    "n_unchanged": n_unchanged,
                    "n_changed": n_changed,
                }

                # 各Jaccard@m (m=3,5,10) について偏相関を計算
                for m in jaccard_keys:
                    jaccard_col = f"cot_jaccard_{m}"

                    df = pd.DataFrame(
                        {
                            "rouge_l": [s["cot_rouge_l"] for s in samples],
                            "jaccard": [s[jaccard_col] for s in samples],
                            "answer_changed": [1 if s["answer_changed"] else 0 for s in samples],
                        }
                    )

                    # ROUGE-L vs answer_changed (Jaccard@m統制)
                    try:
                        res = pg.partial_corr(
                            data=df,
                            x="rouge_l",
                            y="answer_changed",
                            covar="jaccard",
                            method="spearman",
                        )
                        result[benchmark][model][k][f"rouge_l_partial_{m}"] = (
                            float(res["r"].values[0]),
                            float(res["p-val"].values[0]),
                        )
                    except Exception:
                        result[benchmark][model][k][f"rouge_l_partial_{m}"] = (None, None)

                    # Jaccard@m vs answer_changed (ROUGE-L統制)
                    try:
                        res = pg.partial_corr(
                            data=df,
                            x="jaccard",
                            y="answer_changed",
                            covar="rouge_l",
                            method="spearman",
                        )
                        result[benchmark][model][k][f"jaccard_partial_{m}"] = (
                            float(res["r"].values[0]),
                            float(res["p-val"].values[0]),
                        )
                    except Exception:
                        result[benchmark][model][k][f"jaccard_partial_{m}"] = (None, None)

                # Q指標と回答変化の偏相関（他のQ指標を統制）
                df_q = pd.DataFrame(
                    {
                        "token_diff": [s["token_diff"] for s in samples],
                        "q_jaccard_3": [s["q_jaccard_3"] for s in samples],
                        "q_jaccard_5": [s["q_jaccard_5"] for s in samples],
                        "q_jaccard_10": [s["q_jaccard_10"] for s in samples],
                        "q_spearman_r": [s["q_spearman_r"] for s in samples],
                        "answer_changed": [1 if s["answer_changed"] else 0 for s in samples],
                    }
                )

                # Q:ΔToken-Num vs answer_changed (Q:Jaccard@10, Q:Spearman-rを統制)
                try:
                    res = pg.partial_corr(
                        data=df_q,
                        x="token_diff",
                        y="answer_changed",
                        covar=["q_jaccard_10", "q_spearman_r"],
                        method="spearman",
                    )
                    result[benchmark][model][k]["q_token_diff"] = (
                        float(res["r"].values[0]),
                        float(res["p-val"].values[0]),
                    )
                except Exception:
                    result[benchmark][model][k]["q_token_diff"] = (None, None)

                # Q:Jaccard@Top-k vs answer_changed (Q:ΔToken-Num, Q:Spearman-rを統制)
                for m in jaccard_keys:
                    try:
                        res = pg.partial_corr(
                            data=df_q,
                            x=f"q_jaccard_{m}",
                            y="answer_changed",
                            covar=["token_diff", "q_spearman_r"],
                            method="spearman",
                        )
                        result[benchmark][model][k][f"q_jaccard_{m}"] = (
                            float(res["r"].values[0]),
                            float(res["p-val"].values[0]),
                        )
                    except Exception:
                        result[benchmark][model][k][f"q_jaccard_{m}"] = (None, None)

                # Q:Spearman-r vs answer_changed (Q:ΔToken-Num, Q:Jaccard@10を統制)
                try:
                    res = pg.partial_corr(
                        data=df_q,
                        x="q_spearman_r",
                        y="answer_changed",
                        covar=["token_diff", "q_jaccard_10"],
                        method="spearman",
                    )
                    result[benchmark][model][k]["q_spearman_r"] = (
                        float(res["r"].values[0]),
                        float(res["p-val"].values[0]),
                    )
                except Exception:
                    result[benchmark][model][k]["q_spearman_r"] = (None, None)

            # k=average: 全kの平均相関を計算
            k_values = [kv for kv in result[benchmark][model] if isinstance(kv, int)]
            if k_values:
                avg_data: dict = {
                    "n_unchanged": sum(
                        result[benchmark][model][kv].get("n_unchanged", 0) for kv in k_values
                    ),
                    "n_changed": sum(
                        result[benchmark][model][kv].get("n_changed", 0) for kv in k_values
                    ),
                }

                # 各指標について平均を計算
                metric_keys = []
                for m in jaccard_keys:
                    metric_keys.extend([f"rouge_l_partial_{m}", f"jaccard_partial_{m}"])
                metric_keys.extend(
                    ["q_token_diff", "q_spearman_r"] + [f"q_jaccard_{m}" for m in [3, 5, 10]]
                )

                for metric_key in metric_keys:
                    rho_values = []
                    for kv in k_values:
                        val = result[benchmark][model][kv].get(metric_key, (None, None))
                        if val[0] is not None:
                            rho_values.append(val[0])
                    if rho_values:
                        avg_data[metric_key] = (float(np.mean(rho_values)), None)
                    else:
                        avg_data[metric_key] = (None, None)

                result[benchmark][model]["average"] = avg_data

    return result


def compute_partial_correlations_4b(
    aggregated_data: dict[str, dict[str, dict[int, dict]]],
) -> dict[str, dict[str, dict[int, dict]]]:
    """実験4-b: CoT指標・Q指標と不正解転落の偏相関を計算（モデル別）.

    Returns:
        {benchmark: {model: {k: {n_cc, n_ci,
                        rouge_l_partial_3, rouge_l_partial_5, rouge_l_partial_10,
                        jaccard_partial_3, jaccard_partial_5, jaccard_partial_10,
                        q_token_diff, q_jaccard_3, q_jaccard_5, q_jaccard_10, q_spearman_r}}}}
    """
    try:
        import pingouin as pg
    except ImportError:
        logger.warning("pingouinがインストールされていないため、偏相関をスキップ")
        return {}

    result: dict[str, dict[str, dict[int, dict]]] = {}
    jaccard_keys = [3, 5, 10, 15, 20]

    for benchmark, model_data in aggregated_data.items():
        result[benchmark] = {}

        # モデル別に計算
        for model, k_data in model_data.items():
            result[benchmark][model] = {}

            for k, metrics in k_data.items():
                samples = metrics.get("samples", [])

                # 正解→正解 or 正解→不正解 のみを抽出
                filtered = [
                    s for s in samples if s["pattern"] in ["correct→correct", "correct→incorrect"]
                ]

                if len(filtered) < 20:
                    continue

                n_cc = sum(1 for s in filtered if s["pattern"] == "correct→correct")
                n_ci = sum(1 for s in filtered if s["pattern"] == "correct→incorrect")

                result[benchmark][model][k] = {
                    "n_cc": n_cc,
                    "n_ci": n_ci,
                }

                # 各Jaccard@m (m=3,5,10) について偏相関を計算
                for m in jaccard_keys:
                    jaccard_col = f"cot_jaccard_{m}"

                    # 正解→不正解を1、正解→正解を0とする
                    df = pd.DataFrame(
                        {
                            "rouge_l": [s["cot_rouge_l"] for s in filtered],
                            "jaccard": [s[jaccard_col] for s in filtered],
                            "incorrect_fall": [
                                1 if s["pattern"] == "correct→incorrect" else 0 for s in filtered
                            ],
                        }
                    )

                    # ROUGE-L vs incorrect_fall (Jaccard@m統制)
                    try:
                        res = pg.partial_corr(
                            data=df,
                            x="rouge_l",
                            y="incorrect_fall",
                            covar="jaccard",
                            method="spearman",
                        )
                        result[benchmark][model][k][f"rouge_l_partial_{m}"] = (
                            float(res["r"].values[0]),
                            float(res["p-val"].values[0]),
                        )
                    except Exception:
                        result[benchmark][model][k][f"rouge_l_partial_{m}"] = (None, None)

                    # Jaccard@m vs incorrect_fall (ROUGE-L統制)
                    try:
                        res = pg.partial_corr(
                            data=df,
                            x="jaccard",
                            y="incorrect_fall",
                            covar="rouge_l",
                            method="spearman",
                        )
                        result[benchmark][model][k][f"jaccard_partial_{m}"] = (
                            float(res["r"].values[0]),
                            float(res["p-val"].values[0]),
                        )
                    except Exception:
                        result[benchmark][model][k][f"jaccard_partial_{m}"] = (None, None)

                # Q指標と不正解転落の偏相関（他のQ指標を統制）
                df_q = pd.DataFrame(
                    {
                        "token_diff": [s["token_diff"] for s in filtered],
                        "q_jaccard_3": [s["q_jaccard_3"] for s in filtered],
                        "q_jaccard_5": [s["q_jaccard_5"] for s in filtered],
                        "q_jaccard_10": [s["q_jaccard_10"] for s in filtered],
                        "q_spearman_r": [s["q_spearman_r"] for s in filtered],
                        "incorrect_fall": [
                            1 if s["pattern"] == "correct→incorrect" else 0 for s in filtered
                        ],
                    }
                )

                # Q:ΔToken-Num vs incorrect_fall (Q:Jaccard@10, Q:Spearman-rを統制)
                try:
                    res = pg.partial_corr(
                        data=df_q,
                        x="token_diff",
                        y="incorrect_fall",
                        covar=["q_jaccard_10", "q_spearman_r"],
                        method="spearman",
                    )
                    result[benchmark][model][k]["q_token_diff"] = (
                        float(res["r"].values[0]),
                        float(res["p-val"].values[0]),
                    )
                except Exception:
                    result[benchmark][model][k]["q_token_diff"] = (None, None)

                # Q:Jaccard@Top-k vs incorrect_fall (Q:ΔToken-Num, Q:Spearman-rを統制)
                for m in jaccard_keys:
                    try:
                        res = pg.partial_corr(
                            data=df_q,
                            x=f"q_jaccard_{m}",
                            y="incorrect_fall",
                            covar=["token_diff", "q_spearman_r"],
                            method="spearman",
                        )
                        result[benchmark][model][k][f"q_jaccard_{m}"] = (
                            float(res["r"].values[0]),
                            float(res["p-val"].values[0]),
                        )
                    except Exception:
                        result[benchmark][model][k][f"q_jaccard_{m}"] = (None, None)

                # Q:Spearman-r vs incorrect_fall (Q:ΔToken-Num, Q:Jaccard@10を統制)
                try:
                    res = pg.partial_corr(
                        data=df_q,
                        x="q_spearman_r",
                        y="incorrect_fall",
                        covar=["token_diff", "q_jaccard_10"],
                        method="spearman",
                    )
                    result[benchmark][model][k]["q_spearman_r"] = (
                        float(res["r"].values[0]),
                        float(res["p-val"].values[0]),
                    )
                except Exception:
                    result[benchmark][model][k]["q_spearman_r"] = (None, None)

            # k=average: 全kの平均相関を計算
            k_values = [kv for kv in result[benchmark][model] if isinstance(kv, int)]
            if k_values:
                avg_data: dict = {
                    "n_cc": sum(result[benchmark][model][kv].get("n_cc", 0) for kv in k_values),
                    "n_ci": sum(result[benchmark][model][kv].get("n_ci", 0) for kv in k_values),
                }

                # 各指標について平均を計算
                metric_keys = []
                for m in jaccard_keys:
                    metric_keys.extend([f"rouge_l_partial_{m}", f"jaccard_partial_{m}"])
                metric_keys.extend(
                    ["q_token_diff", "q_spearman_r"] + [f"q_jaccard_{m}" for m in [3, 5, 10]]
                )

                for metric_key in metric_keys:
                    rho_values = []
                    for kv in k_values:
                        val = result[benchmark][model][kv].get(metric_key, (None, None))
                        if val[0] is not None:
                            rho_values.append(val[0])
                    if rho_values:
                        avg_data[metric_key] = (float(np.mean(rho_values)), None)
                    else:
                        avg_data[metric_key] = (None, None)

                result[benchmark][model]["average"] = avg_data

    return result


class WandbUploader:
    """W&Bへのアップロードを管理するクラス."""

    def __init__(self, project: str = "lxp-perturbation-analysis"):
        """初期化."""
        self.project = project
        self.run: wandb.sdk.wandb_run.Run | None = None
        self.run_id: str | None = None

    def _load_run_id(self) -> str | None:
        """保存されたRun IDを読み込む."""
        if os.path.exists(RUN_ID_FILE):
            with open(RUN_ID_FILE, encoding="utf-8") as f:
                return f.read().strip()
        return None

    def _save_run_id(self, run_id: str) -> None:
        """Run IDを保存."""
        with open(RUN_ID_FILE, "w", encoding="utf-8") as f:
            f.write(run_id)

    def start_run(self, resume: bool = True) -> None:
        """W&B Runを開始."""
        if resume:
            self.run_id = self._load_run_id()

        if self.run_id:
            self.run = wandb.init(
                project=self.project,
                id=self.run_id,
                resume="allow",
            )
        else:
            self.run = wandb.init(
                project=self.project,
                name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            self.run_id = self.run.id
            self._save_run_id(self.run_id)

        logger.info(f"W&B Run開始: {self.run.url}")

    def finish_run(self) -> None:
        """W&B Runを終了."""
        if self.run:
            self.run.finish()
            self.run = None

    def upload_all(self, collector: ExperimentDataCollector) -> None:
        """すべてのデータをアップロード."""
        if not self.run:
            logger.error("Runが開始されていません")
            return

        # ========================================
        # 実験1: 推論性能
        # ========================================
        if collector.exp1_data:
            logger.info("実験1: 推論性能テーブル・グラフを作成中...")
            table = create_exp1_table(collector.exp1_data)
            self.run.log({"exp1/performance_table": table})
            create_exp1_charts(self.run, collector.exp1_data)
            logger.info("実験1: 完了")

        # ========================================
        # 実験2-4: サンプルレベル分析
        # ========================================
        if collector.full_results_data:
            logger.info("サンプルデータを集約中...")
            aggregated_data = aggregate_sample_data(collector.full_results_data)

            if aggregated_data:
                # 実験2-a: 質問文への影響
                logger.info("実験2-a: 質問文への影響テーブル・グラフを作成中...")
                table_2a = create_exp2a_table(aggregated_data)
                self.run.log({"exp2a/question_impact_table": table_2a})
                create_exp2a_charts(self.run, aggregated_data)
                logger.info("実験2-a: 完了")

                # 実験2-b: CoT推論過程への影響
                logger.info("実験2-b: CoT推論過程への影響テーブル・グラフを作成中...")
                table_2b = create_exp2b_table(aggregated_data)
                self.run.log({"exp2b/cot_impact_table": table_2b})
                create_exp2b_charts(self.run, aggregated_data)
                logger.info("実験2-b: 完了")

                # 実験3: Q指標とCoT指標の相関
                logger.info("実験3: 相関分析中...")
                correlation_data = compute_correlations(aggregated_data)
                if correlation_data:
                    table_3 = create_exp3_table(correlation_data)
                    self.run.log({"exp3/correlation_table": table_3})
                    create_exp3_heatmaps(self.run, correlation_data)
                logger.info("実験3: 完了")

                # 実験4-a: 回答変化との偏相関
                logger.info("実験4-a: 回答変化との偏相関を計算中...")
                partial_4a = compute_partial_correlations_4a(aggregated_data)
                if partial_4a:
                    table_4a = create_exp4a_table(partial_4a)
                    self.run.log({"exp4a/answer_change_partial_corr_table": table_4a})
                    # Q指標と回答変化の相関テーブル
                    table_4a_q = create_exp4a_q_table(partial_4a)
                    self.run.log({"exp4a/q_metrics_answer_change_table": table_4a_q})
                logger.info("実験4-a: 完了")

                # 実験4-b: 不正解転落との偏相関
                logger.info("実験4-b: 不正解転落との偏相関を計算中...")
                partial_4b = compute_partial_correlations_4b(aggregated_data)
                if partial_4b:
                    table_4b = create_exp4b_table(partial_4b)
                    self.run.log({"exp4b/incorrect_fall_partial_corr_table": table_4b})
                    # Q指標と不正解転落の相関テーブル
                    table_4b_q = create_exp4b_q_table(partial_4b)
                    self.run.log({"exp4b/q_metrics_incorrect_fall_table": table_4b_q})
                logger.info("実験4-b: 完了")

        logger.info("すべてのアップロードが完了しました")


class JsonFileHandler(FileSystemEventHandler):
    """JSONファイルの変更を監視するハンドラ."""

    def __init__(
        self,
        analysis_dir: Path,
        outputs_dir: Path,
        uploader: WandbUploader,
    ):
        """初期化."""
        self.analysis_dir = analysis_dir
        self.outputs_dir = outputs_dir
        self.uploader = uploader
        self.last_reload = 0.0
        self.reload_interval = 5.0  # 5秒間隔

    def on_created(self, event):
        """ファイル作成時."""
        if event.is_directory:
            return
        if event.src_path.endswith(".json"):
            self._reload_and_upload()

    def on_modified(self, event):
        """ファイル変更時."""
        if event.is_directory:
            return
        if event.src_path.endswith(".json"):
            self._reload_and_upload()

    def _reload_and_upload(self):
        """データを再読み込みしてアップロード."""
        now = time.time()
        if now - self.last_reload < self.reload_interval:
            return
        self.last_reload = now

        logger.info("ファイル変更を検出。再読み込み中...")

        collector = ExperimentDataCollector()
        collector.load_exp1_data(self.outputs_dir)
        collector.load_directory(self.analysis_dir)

        if collector.exp1_data or collector.full_results_data:
            self.uploader.upload_all(collector)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース."""
    parser = argparse.ArgumentParser(
        description="W&Bへの実験結果アップロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="./outputs/analysis",
        help="分析結果ディレクトリ",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="./outputs",
        help="出力ルートディレクトリ（baseline/perturbed含む）",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="lxp-perturbation-analysis",
        help="W&Bプロジェクト名",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="監視モード（ファイル追加時に自動更新）",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="新規Runを作成（既存Runを再利用しない）",
    )

    return parser.parse_args()


def run_watch_mode(
    analysis_dir: Path,
    outputs_dir: Path,
    uploader: WandbUploader,
) -> None:
    """監視モードで実行."""
    handler = JsonFileHandler(analysis_dir, outputs_dir, uploader)
    observer = Observer()
    observer.schedule(handler, str(analysis_dir), recursive=True)
    observer.start()

    logger.info(f"監視モード開始: {analysis_dir}")
    logger.info("Ctrl+Cで終了")

    def signal_handler(sig, frame):
        logger.info("終了シグナルを受信...")
        observer.stop()
        uploader.finish_run()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    analysis_dir = Path(args.analysis_dir)
    outputs_dir = Path(args.outputs_dir)

    if not analysis_dir.exists():
        logger.warning(f"分析ディレクトリが存在しません: {analysis_dir}")

    # データ収集
    collector = ExperimentDataCollector()

    # exp1データ読み込み
    if outputs_dir.exists():
        exp1_count = collector.load_exp1_data(outputs_dir)
        logger.info(f"exp1データ: {exp1_count}件読み込み")

    # 分析結果読み込み
    if analysis_dir.exists():
        analysis_count = collector.load_directory(analysis_dir)
        logger.info(f"分析結果: {analysis_count}件読み込み")

    if not collector.exp1_data and not collector.full_results_data:
        logger.error("アップロードするデータがありません")
        sys.exit(1)

    # W&Bアップロード
    uploader = WandbUploader(project=args.project)
    uploader.start_run(resume=not args.new_run)

    try:
        uploader.upload_all(collector)

        if args.watch:
            run_watch_mode(analysis_dir, outputs_dir, uploader)
        else:
            uploader.finish_run()

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        uploader.finish_run()
        raise


if __name__ == "__main__":
    main()
