"""matplotlib / seaborn の共通テーマ。図の見た目をプロジェクト間で統一する。"""

from __future__ import annotations


def set_theme(context: str = "talk", style: str = "whitegrid") -> None:
    """seaborn テーマを適用する。"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context=context, style=style)
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
