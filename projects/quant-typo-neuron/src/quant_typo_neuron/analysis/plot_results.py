"""実験結果の可視化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typo_utils.viz.theme import set_theme


def plot_accuracy_heatmap(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    set_theme()
    pivot = df.pivot_table(
        index="model",
        columns=["quantization_method", "bits"],
        values="accuracy",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax)
    ax.set_title("Accuracy by Model and Quantization")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_robustness_comparison(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    set_theme()
    clean = df[df["typo_type"] == "clean"].set_index(["model", "quantization_method", "bits", "benchmark"])
    typo = df[df["typo_type"] != "clean"]

    if "accuracy" in clean.columns and "accuracy" in typo.columns:
        merged = typo.merge(
            clean[["accuracy"]],
            left_on=["model", "quantization_method", "bits", "benchmark"],
            right_index=True,
            suffixes=("_typo", "_clean"),
        )
        merged["robustness_gap"] = merged["accuracy_clean"] - merged["accuracy_typo"]
    else:
        merged = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(12, 6))
    if not merged.empty:
        sns.boxplot(data=merged, x="quantization_method", y="robustness_gap", hue="typo_type", ax=ax)
        ax.set_title("Robustness Gap by Quantization Method")
        ax.set_ylabel("Accuracy Drop (clean - typo)")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_perplexity_bars(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    set_theme()
    ppl_df = df[df["mean_perplexity"].notna()] if "mean_perplexity" in df.columns else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(12, 6))
    if not ppl_df.empty:
        sns.barplot(
            data=ppl_df,
            x="model",
            y="mean_perplexity",
            hue="quantization_method",
            ax=ax,
        )
        ax.set_title("Perplexity by Model and Quantization")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
