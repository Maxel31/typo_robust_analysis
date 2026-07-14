"""論文Figure 2 / Figure 3 を再生成する描画モジュール.

- Figure 2a: Question側メトリクス vs k（Q:ΔToken / Q:Jaccard@10 / Q:Spearman-ρ）
- Figure 2b: CoT側メトリクス vs k（CoT:ROUGE-L / CoT:Jaccard@10）
- Figure 3: Q↔CoT Spearman ρ ヒートマップ（k=4）

集約済みDataFrame（`aggregators.collect_*`）を受け取り、matplotlibで描画する。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ORDER = [
    # 論文準拠 (v1)
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "Mistral-7B-Instruct-v0.3",
    # v2 拡張
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct",
]
DEFAULT_DATASET_ORDER = [
    "gsm8k",
    "mmlu",
    "mmlu_pro",
    "arc",
    "commonsense_qa",
    "squad_v2",
    "bbh",
    "math",
    "strategy_qa",
]

MODEL_DISPLAY = {
    "Llama-3.2-1B-Instruct": "Llama-3.2 (1B)",
    "Llama-3.2-3B-Instruct": "Llama-3.2 (3B)",
    "gemma-3-1b-it": "Gemma-3 (1B)",
    "gemma-3-4b-it": "Gemma-3 (4B)",
    "gemma-3-12b-it": "Gemma-3 (12B)",
    "gemma-3-27b-it": "Gemma-3 (27B)",
    "Mistral-7B-Instruct-v0.3": "Mistral (7B)",
    "Qwen2.5-0.5B-Instruct": "Qwen-2.5 (0.5B)",
    "Qwen2.5-1.5B-Instruct": "Qwen-2.5 (1.5B)",
    "Qwen2.5-3B-Instruct": "Qwen-2.5 (3B)",
    "Qwen2.5-7B-Instruct": "Qwen-2.5 (7B)",
    "Qwen2.5-32B-Instruct": "Qwen-2.5 (32B)",
}
DATASET_DISPLAY = {
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
    "mmlu_pro": "MMLU-Pro",
    "arc": "ARC",
    "commonsense_qa": "CommonsenseQA",
    "squad_v2": "SQuAD v2",
    "bbh": "BBH",
    "math": "MATH",
    "strategy_qa": "StrategyQA",
}


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _aggregate_over_datasets(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """指定 metric について model × k のサンプル数加重平均を返す."""
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        return sub
    sub["weighted"] = sub["mean"] * sub["n"]
    agg = (
        sub.groupby(["model", "k"], as_index=False)
        .agg(weighted_sum=("weighted", "sum"), n_total=("n", "sum"))
    )
    agg["value"] = agg["weighted_sum"] / agg["n_total"]
    return agg[["model", "k", "value"]]


def _plot_metric_vs_k(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    title: str,
    ylabel: str,
    model_order: list[str],
    ylim: tuple[float, float] | None = None,
) -> None:
    for model in model_order:
        sub = df_agg[df_agg["model"] == model].sort_values("k")
        if sub.empty:
            continue
        ax.plot(
            sub["k"], sub["value"], marker="o", label=MODEL_DISPLAY.get(model, model)
        )
    ax.set_xlabel("k (number of perturbed tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_figure2a(
    overall_df: pd.DataFrame,
    out_path: Path,
    model_order: list[str] | None = None,
) -> None:
    """Figure 2a: Question-side metrics (Q:ΔToken / Q:Jaccard@10 / Q:Spearman-ρ) vs k."""
    model_order = model_order or DEFAULT_MODEL_ORDER
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    spec = [
        ("token_count_diff", "Q:ΔToken", None),
        ("q_jaccard_top10", "Q:Jaccard@10", (0.0, 1.0)),
        ("q_spearman_r", "Q:Spearman-ρ", (0.0, 1.0)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, spec, strict=True):
        agg = _aggregate_over_datasets(overall_df, metric)
        _plot_metric_vs_k(ax, agg, title=ylabel, ylabel=ylabel, model_order=model_order, ylim=ylim)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Figure 2a: Effect of perturbations on questions")
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 2a → {out_path}")


def plot_figure2b(
    overall_df: pd.DataFrame,
    out_path: Path,
    model_order: list[str] | None = None,
) -> None:
    """Figure 2b: CoT-side metrics (CoT:ROUGE-L / CoT:Jaccard@10) vs k."""
    model_order = model_order or DEFAULT_MODEL_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    spec = [
        ("cot_rouge_l_f1", "CoT:ROUGE-L", (0.0, 1.0)),
        ("cot_jaccard_top10", "CoT:Jaccard@10", (0.0, 1.0)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, spec, strict=True):
        agg = _aggregate_over_datasets(overall_df, metric)
        _plot_metric_vs_k(ax, agg, title=ylabel, ylabel=ylabel, model_order=model_order, ylim=ylim)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Figure 2b: Effect of perturbations on CoT reasoning")
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 2b → {out_path}")


def _pivot_q_cot_heatmap(corr_df: pd.DataFrame) -> pd.DataFrame:
    """`var_q` × `var_cot` のピボットを返す（model平均）."""
    if corr_df.empty:
        return corr_df
    pv = (
        corr_df.groupby(["var_q", "var_cot"])["spearman_rho"]
        .mean()
        .unstack("var_cot")
    )
    return pv


def plot_figure3(
    corr_df: pd.DataFrame,
    out_path: Path,
    model_order: list[str] | None = None,
) -> None:
    """Figure 3: Q↔CoT Spearman ρ ヒートマップ（モデル別パネル）."""
    model_order = model_order or DEFAULT_MODEL_ORDER
    present_models = [m for m in model_order if m in set(corr_df["model"])]
    if not present_models:
        logger.warning("Figure 3: no model data available")
        return
    n = len(present_models)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0), squeeze=False)
    axes = axes[0]
    vmin, vmax = -0.5, 0.5
    im = None
    for ax, model in zip(axes, present_models, strict=False):
        sub = corr_df[corr_df["model"] == model]
        pv = _pivot_q_cot_heatmap(sub)
        if pv.empty:
            ax.set_visible(False)
            continue
        im = ax.imshow(pv.values, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(pv.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(pv.index, fontsize=7)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=9)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(v) < 0.3 else "white")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Spearman ρ")
    fig.suptitle("Figure 3: Q ↔ CoT metric correlation (k=4)")
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 3 → {out_path}")
