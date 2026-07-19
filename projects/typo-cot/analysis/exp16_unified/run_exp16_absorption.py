"""実験16 (P6): 家族効果の吸収テスト.

仮説 P6: モデル家族 (Gemma/Llama/Mistral/Qwen) の flip 主効果は、ERDC 連鎖
パラメータ (M1符号化 ~ split/rq, M2内部修復 ~ repair_min, S2分岐 ~ kl_sum,
CoT搬送 ~ rouge/jaccard, M3読み出し集中 ~ gini_rc/delta_rho) を共変量に入れると
縮小・消失するか。段階的にネストした固定効果ロジット (cluster-robust SE, setting
クラスタ) + BBGLM(1|setting) 確認で検定。

出力:
  features_full.parquet     (features_partial + family(model派生) + gini_rc)
  exp16_absorption.json      M0-M3 の family 係数/SE/p, 縮小率, N, fitter
  exp16_summary.md           P6 判定 (別途 human 記述)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sstats

HERE = Path(__file__).resolve().parent
EXP13 = Path("/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/"
             ".claude/worktrees/exp-13-readout/projects/typo-cot/analysis/"
             "exp13_readout_concentration/exp13_summary.json")

FAM_MAP = {
    "gemma-3-1b-it": "Gemma", "gemma-3-4b-it": "Gemma",
    "Llama-3.2-1B-Instruct": "Llama", "Llama-3.2-3B-Instruct": "Llama",
    "Mistral-7B-Instruct-v0.3": "Mistral", "Qwen2.5-7B-Instruct": "Qwen",
}
REF_FAMILY = "Gemma"  # treatment-coding reference

# ERDC 共変量 (段階投入)
CARRY = ["rouge_l_f1", "cot_jaccard_top10"]                 # M1: CoT 搬送
REPAIR_DIVERT = ["repair_min", "kl_sum"]                    # M2: 内部修復 / S2 分岐
READOUT_ENC = ["delta_rho_top10", "rq_mean", "split_mean"]  # M3: 読み出し集中/符号化
ALL_COVS = CARRY + REPAIR_DIVERT + READOUT_ENC


def zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        v = out[c].astype(float)
        out[c + "_z"] = (v - v.mean()) / v.std(ddof=0)
    return out


def build_full():
    df = pd.read_parquet(HERE / "features_partial.parquet")
    df["family"] = df["model"].map(FAM_MAP)  # 完全 (Qwen 含む) 家族を model から派生
    g = json.load(open(EXP13))
    grows = (pd.DataFrame(g["rows"])[["model", "benchmark", "loo_gini"]]
             .rename(columns={"loo_gini": "gini_rc"}))
    df = df.merge(grows, on=["model", "benchmark"], how="left")
    df.to_parquet(HERE / "features_full.parquet", index=False)
    return df


def fit_logit_cluster(data: pd.DataFrame, cov_z: list[str]):
    terms = [f"C(family, Treatment('{REF_FAMILY}'))"] + [c + "_z" for c in cov_z]
    formula = "flip ~ " + " + ".join(terms)
    res = smf.logit(formula, data=data).fit(
        disp=0, maxiter=200,
        cov_type="cluster", cov_kwds={"groups": data["setting"].values})
    return res, formula


def family_terms(res):
    return [p for p in res.params.index if p.startswith("C(family")]


def extract_family(res):
    out = {}
    for p in family_terms(res):
        lvl = p.split("[T.")[-1].rstrip("]")
        out[lvl] = {"coef": float(res.params[p]), "se": float(res.bse[p]),
                    "z": float(res.tvalues[p]), "p": float(res.pvalues[p])}
    return out


def family_block_wald(res):
    fam = family_terms(res)
    try:
        w = res.wald_test(fam, scalar=False)
        return {"chi2": float(np.ravel(w.statistic)[0]), "df": len(fam),
                "p": float(w.pvalue)}
    except Exception as e:
        return {"error": str(e)}


def l2(d):
    return float(np.sqrt(sum(v["coef"] ** 2 for v in d.values())))


def fam_range(d):
    vals = [0.0] + [v["coef"] for v in d.values()]
    return float(max(vals) - min(vals))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals)); m = len(pvals); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def try_bbglm(data: pd.DataFrame, cov_z: list[str], group="setting"):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    terms = [f"C(family, Treatment('{REF_FAMILY}'))"] + [c + "_z" for c in cov_z]
    formula = "flip ~ " + " + ".join(terms)
    vc = {group: f"0 + C({group})"}
    m = BinomialBayesMixedGLM.from_formula(formula, vc, data)
    r = m.fit_vb(verbose=False)
    names = list(r.model.exog_names)
    out = {}
    for i, nm in enumerate(names):
        if nm.startswith("C(family"):
            lvl = nm.split("[T.")[-1].rstrip("]")
            mean = float(r.fe_mean[i]); sd = float(r.fe_sd[i])
            z = mean / sd if sd > 0 else np.nan
            p = float(2 * (1 - sstats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
            out[lvl] = {"coef": mean, "se": sd, "z": float(z), "p": p}
    return out


def main():
    df = build_full()
    inc = df[df["included"]].copy()
    n_included = len(inc)
    cc = inc.dropna(subset=ALL_COVS).copy()
    n_cc = len(cc)
    n_excluded = n_included - n_cc

    cc["family"] = pd.Categorical(cc["family"])
    cc = zscore(cc, ALL_COVS)

    stages = {
        "M0_base": [],
        "M1_carry": CARRY,
        "M2_repair_divert": CARRY + REPAIR_DIVERT,
        "M3_readout_enc": CARRY + REPAIR_DIVERT + READOUT_ENC,
    }

    results = {}
    fam_at = {}
    block_ps = []
    for name, covs in stages.items():
        res, formula = fit_logit_cluster(cc, covs)
        fam = extract_family(res)
        blk = family_block_wald(res)
        fam_at[name] = fam
        block_ps.append(blk.get("p", np.nan))
        results[name] = {
            "formula": formula, "n": int(res.nobs),
            "n_clusters_setting": int(cc["setting"].nunique()),
            "converged": bool(res.mle_retvals.get("converged", True)),
            "family_coef": fam,
            "family_block_wald": blk,
            "family_L2_logodds": l2(fam),
            "family_range_logodds": fam_range(fam),
            "included_covariates": covs,
        }

    base = fam_at["M0_base"]
    for name in stages:
        red = {}
        for lvl, b0 in base.items():
            bk = fam_at[name].get(lvl, {}).get("coef", np.nan)
            red[lvl] = (100.0 * (abs(b0["coef"]) - abs(bk)) / abs(b0["coef"])
                        if b0["coef"] != 0 else np.nan)
        l2_base = results["M0_base"]["family_L2_logodds"]
        l2_k = results[name]["family_L2_logodds"]
        results[name]["family_logodds_reduction_pct_vs_M0"] = red
        results[name]["family_L2_reduction_pct_vs_M0"] = (
            100.0 * (l2_base - l2_k) / l2_base if l2_base else np.nan)

    holm_adj = holm(block_ps) if all(np.isfinite(block_ps)) else None
    for i, name in enumerate(stages):
        results[name]["family_block_p_holm"] = (holm_adj[i] if holm_adj else None)

    # Robustness A: BBGLM(1|setting) confirmation on same CC
    bbglm = {}
    for name, covs in stages.items():
        try:
            bbglm[name] = {"family_coef": try_bbglm(cc, covs), "status": "ok"}
        except Exception as e:
            bbglm[name] = {"status": f"failed: {type(e).__name__}: {e}"}

    # Robustness B: M3 + gini_rc on gsm8k/mmlu subset
    gini_block = {}
    try:
        ccg = cc.dropna(subset=["gini_rc"]).copy()
        ccg = zscore(ccg, ["gini_rc"])
        res0g, _ = fit_logit_cluster(ccg, [])
        terms = ([f"C(family, Treatment('{REF_FAMILY}'))"]
                 + [c + "_z" for c in ALL_COVS] + ["gini_rc_z"])
        resg = smf.logit("flip ~ " + " + ".join(terms), data=ccg).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": ccg["setting"].values})
        fam0 = extract_family(res0g)
        famg = extract_family(resg)
        red = {lvl: (100.0 * (abs(fam0[lvl]["coef"]) - abs(famg[lvl]["coef"]))
                     / abs(fam0[lvl]["coef"])) for lvl in fam0}
        gini_block = {
            "n": int(resg.nobs), "benchmarks": sorted(ccg["benchmark"].unique()),
            "n_clusters_setting": int(ccg["setting"].nunique()),
            "M0subset_family_coef": fam0,
            "M3plusGini_family_coef": famg,
            "M3plusGini_family_reduction_pct_vs_M0subset": red,
            "gini_rc_z_coef": float(resg.params["gini_rc_z"]),
            "gini_rc_z_p": float(resg.pvalues["gini_rc_z"]),
            "note": "gini_rc = exp13 loo_gini (model x benchmark, gsm8k/mmlu のみ, "
                    "Qwen 無し)。gsm8k+mmlu subset に限定した頑健性チェック。",
        }
    except Exception as e:
        gini_block = {"status": f"failed: {type(e).__name__}: {e}"}

    # Descriptive: full-sample M0 with all 4 families (incl Qwen)
    try:
        full4 = inc.copy()
        full4["family"] = pd.Categorical(full4["family"])
        res4, _ = fit_logit_cluster(full4, [])
        m0_full = {"n": int(res4.nobs),
                   "n_clusters_setting": int(full4["setting"].nunique()),
                   "families": sorted(full4["family"].unique().tolist()),
                   "family_coef": extract_family(res4),
                   "note": "全included (4家族,全benchmark)。記述用 M0。ネスト吸収テストとは "
                           "N/家族集合が異なる (Qwen は carry特徴全欠損で M1+ 不可)。"}
    except Exception as e:
        m0_full = {"status": f"failed: {type(e).__name__}: {e}"}

    out = {
        "hypothesis": "P6 family absorption by ERDC covariates",
        "primary_fitter": "fixed-effects logit + cluster-robust SE (cluster=setting)",
        "fitter_note": ("BBGLM(VB) を交差ランダム効果 (item ~17k levels) で回すのは非現実的"
                        "なため、主解析は task 指定 fallback の fixed-effects logit + "
                        "cluster-robust(setting) を採用。BBGLM(1|setting) は robustness で併記。"),
        "family_reference_level": REF_FAMILY,
        "family_coding": "treatment (dummy vs Gemma)",
        "continuous_covariates": "z-standardized on the complete-case sample",
        "n_included": int(n_included),
        "n_complete_case": int(n_cc),
        "n_excluded_from_included": int(n_excluded),
        "complete_case_families": sorted(cc["family"].unique().tolist()),
        "complete_case_benchmarks": sorted(cc["benchmark"].unique().tolist()),
        "complete_case_settings": int(cc["setting"].nunique()),
        "qwen_status": ("Qwen は rouge_l_f1/cot_jaccard_top10/kl_sum/delta_rho が全欠損 "
                        "(Step0 CoT-ROUGE 未計算, exp12 moderator 欠損) のため M1+ に投入不可。"
                        "ネスト吸収テストは Gemma/Llama/Mistral の3家族に限定。"),
        "math_status": "math benchmark は全モデルで rouge 欠損のため complete-case から脱落。",
        "nan_family_handling": ("features_partial の family 列は Qwen 設定で 18701 NaN "
                                "(exp12 moderator 欠損)。model からの決定論的写像で family を"
                                "再構成し完全化 (Qwen 含む)。"),
        "multiple_comparison": "family-block Wald p を Holm 補正 (4段階)。効果サイズ(縮小率)中心。",
        "models": results,
        "robustness_bbglm_setting_RE": bbglm,
        "robustness_M3_plus_gini_gsm8k_mmlu": gini_block,
        "descriptive_M0_full_4family": m0_full,
    }

    (HERE / "exp16_absorption.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("WROTE", HERE / "exp16_absorption.json")
    for name in stages:
        r = results[name]
        print(name, "N", r["n"], "familyL2", round(r["family_L2_logodds"], 4),
              "L2red%", round(r.get("family_L2_reduction_pct_vs_M0", float('nan')), 1),
              "blockp", r["family_block_wald"].get("p"))
    return out


if __name__ == "__main__":
    main()
