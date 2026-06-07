"""定番プロット（typo 率 vs 精度 など）。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def plot_robustness_curve(
    typo_rates: Sequence[float],
    accuracies: Sequence[float],
    label: str | None = None,
    save_path: str | Path | None = None,
):
    """typo 率に対する精度の頑健性カーブを描く。"""
    import matplotlib.pyplot as plt

    from typo_utils.viz.theme import set_theme

    set_theme()
    fig, ax = plt.subplots()
    ax.plot(typo_rates, accuracies, marker="o", label=label)
    ax.set_xlabel("typo rate")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    if label:
        ax.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig, ax
