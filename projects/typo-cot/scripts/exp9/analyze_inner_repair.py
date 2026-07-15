"""実験9: 修復スコアの集計・回帰・図表出力.

run_inner_repair.py が出力した word_rows_*.jsonl を読み、
- flip ~ 修復スコア + 分割増分 + Zipf頻度 + R_Q のロジスティック回帰
  (クラスタロバスト SE; item=sample_id) の係数表 CSV
- flip / non-flip 群の平均層別 cos カーブ図 (PNG)
- 集計サマリ JSON
を出力する。GPU 不要。

使用例:
    uv run python scripts/exp9/analyze_inner_repair.py \
        --input-dir results/smoke/exp9 --output-dir results/smoke/exp9/analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from typo_cot.repair.regression import filter_clean_correct, fit_flip_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp9.analyze")

FEATURES = ["repair_score", "split_increment", "zipf_freq", "r_q"]


def load_rows(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("word_rows_*.jsonl")):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"word_rows_*.jsonl が見つかりません: {input_dir}")
    return pd.DataFrame(rows)


def plot_cos_curves(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for flip, color, label in [(False, "tab:blue", "no flip"), (True, "tab:red", "flip")]:
        sub = df[df["flip"] == flip]
        if len(sub) == 0:
            continue
        curves = np.array([c for c in sub["cos_curve"]])
        mean = curves.mean(axis=0)
        se = curves.std(axis=0) / np.sqrt(len(curves))
        layers = np.arange(len(mean))
        ax.plot(layers, mean, color=color, label=f"{label} (n={len(sub)})")
        ax.fill_between(layers, mean - se, mean + se, color=color, alpha=0.2)
    ax.set_xlabel("layer")
    ax.set_ylabel("cos(clean, typo) at span-end token")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="実験9: 集計・回帰・図表")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clean-correct-only", action="store_true",
                   help="clean 正解サンプルの語行に限定 (主推定量の規約; 分析側で条件付け)")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(input_dir)
    if args.clean_correct_only:
        n_before = len(df)
        df = filter_clean_correct(df)
        logger.info("clean 正解条件付け: %d -> %d 行", n_before, len(df))
    logger.info("語レベル行: %d (モデル=%s)", len(df), sorted(df["model"].unique()))

    summary: dict = {"n_rows": int(len(df)), "groups": {}}
    for (model, benchmark, condition), sub in df.groupby(["model", "benchmark", "condition"]):
        tag = f"{model}_{benchmark}_{condition}"
        group_info: dict = {
            "n_rows": int(len(sub)),
            "n_flip": int(sub["flip"].sum()),
            "mean_repair_flip": float(sub.loc[sub["flip"], "repair_score"].mean())
            if sub["flip"].any()
            else None,
            "mean_repair_noflip": float(sub.loc[~sub["flip"], "repair_score"].mean())
            if (~sub["flip"]).any()
            else None,
        }
        # 回帰 (flip の両クラスが十分あるときのみ)
        try:
            result = fit_flip_regression(sub, feature_cols=FEATURES, cluster_col="sample_id")
            result.coefs.to_csv(out_dir / f"regression_{tag}.csv")
            group_info["regression"] = {
                "n_obs": result.n_obs,
                "n_clusters": result.n_clusters,
                "repair_coef": float(result.coefs.loc["repair_score", "coef"]),
                "repair_p": float(result.coefs.loc["repair_score", "p"]),
            }
        except (ValueError, KeyError) as e:
            group_info["regression"] = {"error": str(e)}
        # 図
        plot_cos_curves(sub, out_dir / f"cos_curves_{tag}.png", tag)
        summary["groups"][tag] = group_info
        logger.info("%s: %s", tag, group_info)

    # 条件プール版 (ベンチマーク・条件横断、モデル別)
    for model, sub in df.groupby("model"):
        try:
            result = fit_flip_regression(sub, feature_cols=FEATURES, cluster_col="sample_id")
            result.coefs.to_csv(out_dir / f"regression_pooled_{model}.csv")
            summary["groups"][f"pooled_{model}"] = {
                "n_obs": result.n_obs,
                "repair_coef": float(result.coefs.loc["repair_score", "coef"]),
                "repair_p": float(result.coefs.loc["repair_score", "p"]),
            }
        except (ValueError, KeyError) as e:
            summary["groups"][f"pooled_{model}"] = {"error": str(e)}

    # 全設定プール版 (モデル横断; 参考値。主報告はモデル別 pooled)
    try:
        result = fit_flip_regression(df, feature_cols=FEATURES, cluster_col="sample_id")
        result.coefs.to_csv(out_dir / "regression_pooled_all.csv")
        summary["groups"]["pooled_all"] = {
            "n_obs": result.n_obs,
            "repair_coef": float(result.coefs.loc["repair_score", "coef"]),
            "repair_p": float(result.coefs.loc["repair_score", "p"]),
        }
    except (ValueError, KeyError) as e:
        summary["groups"]["pooled_all"] = {"error": str(e)}
    summary["clean_correct_only"] = bool(args.clean_correct_only)

    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("保存: %s", out_dir / "analysis_summary.json")


if __name__ == "__main__":
    main()
