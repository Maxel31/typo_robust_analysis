"""実験16 骨格: 統一 GLMM 用のサンプル×特徴 中間データフレーム (partial).

後続の統一 GLMM:
  flip ~ repair_min + KL_sum + fixed_Jaccard + ROUGE + 設定レベルモデレーター
         + (1|item) + (1|setting)

本スクリプトは実験11・12 の出力と Step0 マスタから、現時点で利用可能な特徴を
サンプル単位で結合する。Gini(実験13)・noCoT_flip(実験14)は未完のため列を持たず、
README に追記する旨を明記する。

結合元:
  exp11_sample_table.parquet  … repair_min/mean, KL_sum, flip, 統制 (S1/G→S2 連鎖)
  Step0 <model>/<bench>/{lxt4,random4}.parquet … ROUGE-L(cot_rouge_l_f1),
                                    Jaccard(cot_jaccard_top10), per-sample r_c/r_q
  exp12 rc_composition_by_setting.csv … 設定レベルモデレーター
                                    (share_*, delta_rho_top10, deletion_rd_k4, family, task_type)

出力: features_partial.parquet  (+ README.md は手書き)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP11 = HERE.parent / "exp11_chain_mediation" / "exp11_sample_table.parquet"
EXP12 = HERE.parent / "exp12_rc_composition" / "rc_composition_by_setting.csv"
STEP0 = Path("/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-step0/projects/typo-cot/data")

COND_TO_STEP0 = {"importance": "lxt4", "random": "random4"}


def load_step0() -> pd.DataFrame:
    frames = []
    for cond, s0cond in COND_TO_STEP0.items():
        for f in STEP0.glob(f"*/*/{s0cond}.parquet"):
            model = f.parts[-3]
            bench = f.parts[-2]
            # r_c/r_q は Step0 では JSON 直列化ランキング(~3KB 文字列)なので取り込まない
            # (統一 GLMM はスカラ特徴のみ; per-sample R_Q は exp11 rq_mean が担う)
            d = pd.read_parquet(f, columns=["sample_id", "cot_rouge_l_f1",
                                            "cot_jaccard_top10"])
            d["model"] = model
            d["benchmark"] = bench
            d["condition"] = cond
            frames.append(d)
    s0 = pd.concat(frames, ignore_index=True)
    return s0.rename(columns={"cot_rouge_l_f1": "rouge_l_f1"})


def main():
    df = pd.read_parquet(EXP11)
    s0 = load_step0()
    df = df.merge(s0, on=["model", "benchmark", "condition", "sample_id"], how="left")

    rc = pd.read_csv(EXP12)
    mod = rc[["model", "benchmark", "family", "task_type", "share_conclusion",
              "share_numeric", "share_content", "share_function",
              "share_content_plus_numeric", "delta_rho_top10", "deletion_rd_k4"]]
    df = df.merge(mod, on=["model", "benchmark"], how="left")

    order = [
        "setting", "model", "benchmark", "condition", "family", "task_type",
        "is_core", "sample_id", "included", "flip", "cot_changed",
        "repair_min", "repair_mean", "kl_sum",
        "zipf_mean", "split_mean", "rq_mean", "n_words",
        "rouge_l_f1", "cot_jaccard_top10",
        "share_conclusion", "share_numeric", "share_content", "share_function",
        "share_content_plus_numeric", "delta_rho_top10", "deletion_rd_k4",
    ]
    order = [c for c in order if c in df.columns]
    df = df[order]
    df.to_parquet(HERE / "features_partial.parquet", index=False)

    print(f"features_partial: {len(df)} rows, {df['setting'].nunique()} settings")
    print("columns:", list(df.columns))
    cov = {c: float(df[c].notna().mean()) for c in
           ["repair_min", "kl_sum", "rouge_l_f1", "cot_jaccard_top10",
            "share_conclusion", "delta_rho_top10", "deletion_rd_k4"]}
    print("non-null coverage:", {k: round(v, 3) for k, v in cov.items()})
    print("included w/ repair+kl+rouge:",
          int((df["included"] & df["repair_min"].notna() & df["kl_sum"].notna()
               & df["rouge_l_f1"].notna()).sum()))


if __name__ == "__main__":
    main()
