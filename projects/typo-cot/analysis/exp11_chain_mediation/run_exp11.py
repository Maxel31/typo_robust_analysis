"""実験11: 連鎖媒介分析 (H11)  修復 → 分岐(KL_sum) → flip.

サンプル i ごとに:
  repair_min_i = 4 摂動語の最小修復スコア (弱リンク仮説; mean 版は感度分析)
  KL_sum_i     = exp01_03 divergence/<sid>.json の kl_sum (clean vs typo 分岐)
  flip_i       = exp01_03 outcomes.json の TE flip (answers[B] != answers[A]),
                 included = not exclude and a_correct (Qwen は dedup-on exclude 上書き)

設定内 2 段回帰 (予測子は設定内 z 化):
  第1段 (OLS):  KL_sum ~ repair_min + Zipf + split_increment + R_Q
  第2段a(logit): flip ~ repair_min + 統制
  第2段b(logit): flip ~ repair_min + KL_sum + 統制
  媒介率 = (a - a') / a   (repair 係数の KL_sum 投入による減衰率)

横断: GLMM flip ~ repair_min (+ KL_sum) + (1|setting)  で pooled 媒介率。

判定(事前登録):
  第1段が負に有意が過半 & repair 直接効果が KL_sum 統制で 50% 以上減衰
    → S1/G→S2 接続を支持。減衰せず → 修復は分岐を介さず読み出し段に直接効く。

出力:
  exp11_sample_table.parquet   サンプル×特徴
  mediation_by_setting.csv     設定別 2 段係数・媒介率
  mediation_pooled.json        pooled/GLMM + 判定
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

WR_DIR = Path("/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-09-inner-repair/projects/typo-cot/results/exp9")
EX01 = Path("/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-01-03-transplant/projects/typo-cot/results/exp01_03")
QWEN_DEDUP = HERE / "qwen_dedup_exclude.json"

CORE_MODELS = {"gemma-3-1b-it", "gemma-3-4b-it", "Llama-3.2-1B-Instruct",
               "Llama-3.2-3B-Instruct", "Mistral-7B-Instruct-v0.3"}
MODELS = ["gemma-3-1b-it", "gemma-3-4b-it", "Llama-3.2-1B-Instruct",
          "Llama-3.2-3B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-7B-Instruct"]
BENCHES = ["arc", "commonsense_qa", "gsm8k", "math", "mmlu", "mmlu_pro"]


def parse_wr_file(fname: str) -> tuple[str, str, str] | None:
    stem = fname[len("word_rows_"):-len(".jsonl")]
    for m in MODELS:
        if not stem.startswith(m + "_"):
            continue
        rest = stem[len(m) + 1:]
        for b in BENCHES:
            if rest == b or rest.startswith(b + "_"):
                tail = rest[len(b):]
                if "lxt4" in tail:
                    return m, b, "importance"
                if "random4" in tail:
                    return m, b, "random"
        return None
    return None


def load_word_features() -> dict[tuple, dict[str, dict]]:
    by_setting = defaultdict(lambda: defaultdict(list))
    for f in sorted(os.listdir(WR_DIR)):
        if not f.startswith("word_rows_") or not f.endswith(".jsonl"):
            continue
        key = parse_wr_file(f)
        if key is None:
            continue
        for line in open(WR_DIR / f):
            r = json.loads(line)
            if r.get("repair_score") is None:
                continue
            by_setting[key][r["sample_id"]].append(r)
    out = {}
    for key, samples in by_setting.items():
        feat = {}
        for sid, rows in samples.items():
            rs = np.array([r["repair_score"] for r in rows], float)
            zi = np.array([r.get("zipf_freq") if r.get("zipf_freq") is not None else np.nan for r in rows], float)
            sp = np.array([r.get("split_increment", 0) or 0 for r in rows], float)
            rq = np.array([r.get("r_q") if r.get("r_q") is not None else np.nan for r in rows], float)
            wl = int(np.argmin(rs))
            feat[sid] = {
                "repair_min": float(rs.min()), "repair_mean": float(rs.mean()),
                "zipf_mean": float(np.nanmean(zi)) if np.isfinite(zi).any() else np.nan,
                "zipf_wl": float(zi[wl]) if np.isfinite(zi[wl]) else np.nan,
                "split_mean": float(sp.mean()), "split_wl": float(sp[wl]),
                "rq_mean": float(np.nanmean(rq)) if np.isfinite(rq).any() else np.nan,
                "rq_wl": float(rq[wl]) if np.isfinite(rq[wl]) else np.nan,
                "n_words": len(rows),
            }
        out[key] = feat
    return out


def exp01_dirs(model: str, bench: str, cond: str) -> list[Path]:
    pat = re.compile(rf"^{re.escape(model)}_{re.escape(bench)}_k4_{cond}(__p\d+)?$")
    return [EX01 / d for d in os.listdir(EX01) if pat.match(d)]


def load_outcomes_kl(model, bench, cond, qwen_override) -> dict[str, dict]:
    out = {}
    base = f"{model}_{bench}_k4_{cond}"
    override = qwen_override.get(base) if model.startswith("Qwen") else None
    for d in exp01_dirs(model, bench, cond):
        kl = {}
        divdir = d / "divergence"
        if divdir.is_dir():
            for jf in os.listdir(divdir):
                if jf.endswith(".json"):
                    dj = json.load(open(divdir / jf))
                    if dj.get("ok") and dj.get("kl_sum") is not None:
                        kl[dj["sample_id"]] = float(dj["kl_sum"])
        for o in json.load(open(d / "outcomes.json")):
            sid = o["sample_id"]
            ans = o.get("answers", {})
            if "A" not in ans or "B" not in ans:
                continue
            flip = int(str(ans["B"]).strip() != str(ans["A"]).strip())
            exclude = bool(o["exclude"])
            if override is not None and sid in override:
                exclude = bool(override[sid])
            out[sid] = {"flip": flip, "a_correct": bool(o["a_correct"]),
                        "exclude": exclude, "cot_changed": bool(o.get("cot_changed", False)),
                        "kl_sum": kl.get(sid, np.nan)}
    return out


def build_table() -> pd.DataFrame:
    qwen_override = json.load(open(QWEN_DEDUP)) if QWEN_DEDUP.exists() else {}
    wfeat = load_word_features()
    rows = []
    for (model, bench, cond), feat in sorted(wfeat.items()):
        oc = load_outcomes_kl(model, bench, cond, qwen_override)
        for sid, f in feat.items():
            o = oc.get(sid)
            if o is None:
                continue
            included = (not o["exclude"]) and o["a_correct"]
            rows.append({
                "setting": f"{model}_{bench}_{cond}", "model": model,
                "benchmark": bench, "condition": cond,
                "is_core": model in CORE_MODELS, "sample_id": sid,
                **f, "kl_sum": o["kl_sum"], "flip": o["flip"],
                "a_correct": o["a_correct"], "exclude": o["exclude"],
                "cot_changed": o["cot_changed"], "included": included,
            })
    return pd.DataFrame(rows)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 1e-9 else s * 0.0


def mediation_one(df: pd.DataFrame, repair_col="repair_min") -> dict:
    import statsmodels.api as sm
    sub = df[df["included"]].dropna(subset=[repair_col, "kl_sum", "flip",
                                            "zipf_mean", "split_mean", "rq_mean"]).copy()
    n = len(sub)
    n_flip = int(sub["flip"].sum())
    if n < 30 or n_flip < 5 or sub["flip"].nunique() < 2:
        return {"n": n, "n_flip": n_flip, "ok": False, "reason": "insufficient"}
    for c in [repair_col, "kl_sum", "zipf_mean", "split_mean", "rq_mean"]:
        sub[c + "_z"] = zscore(sub[c])
    ctrl = ["zipf_mean_z", "split_mean_z", "rq_mean_z"]
    rz = repair_col + "_z"
    res = {"n": n, "n_flip": n_flip, "ok": True}
    X1 = sm.add_constant(sub[[rz] + ctrl])
    m1 = sm.OLS(sub["kl_sum_z"], X1).fit()
    res["stage1_repair_coef"] = float(m1.params[rz])
    res["stage1_repair_p"] = float(m1.pvalues[rz])
    res["stage1_repair_neg_sig"] = bool(m1.params[rz] < 0 and m1.pvalues[rz] < 0.05)
    try:
        X2a = sm.add_constant(sub[[rz] + ctrl])
        m2a = sm.Logit(sub["flip"], X2a).fit(disp=0, method="bfgs", maxiter=200)
        a = float(m2a.params[rz])
        res["stage2a_repair_coef"] = a
        res["stage2a_repair_p"] = float(m2a.pvalues[rz])
        X2b = sm.add_constant(sub[[rz, "kl_sum_z"] + ctrl])
        m2b = sm.Logit(sub["flip"], X2b).fit(disp=0, method="bfgs", maxiter=200)
        aprime = float(m2b.params[rz])
        res["stage2b_repair_coef"] = aprime
        res["stage2b_repair_p"] = float(m2b.pvalues[rz])
        res["stage2b_kl_coef"] = float(m2b.params["kl_sum_z"])
        res["stage2b_kl_p"] = float(m2b.pvalues["kl_sum_z"])
        res["prop_mediated"] = float((a - aprime) / a) if abs(a) > 1e-6 else np.nan
    except Exception as e:
        res["stage2_error"] = str(e)[:120]
    return res


def glmm_pooled(df: pd.DataFrame, repair_col="repair_min", tag="") -> dict:
    import statsmodels.api as sm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    sub = df[df["included"]].dropna(subset=[repair_col, "kl_sum", "flip"]).copy()
    for c in [repair_col, "kl_sum"]:
        sub[c + "_z"] = sub.groupby("setting")[c].transform(zscore)
    sub = sub.rename(columns={repair_col + "_z": "repair_z"})
    out = {"tag": tag, "n": int(len(sub)), "n_settings": int(sub["setting"].nunique())}
    try:
        m_tot = BinomialBayesMixedGLM.from_formula(
            "flip ~ repair_z", {"setting": "0 + C(setting)"}, sub).fit_vb()
        names = list(m_tot.model.exog_names)
        a = float(m_tot.fe_mean[names.index("repair_z")])
        m_dir = BinomialBayesMixedGLM.from_formula(
            "flip ~ repair_z + kl_sum_z", {"setting": "0 + C(setting)"}, sub).fit_vb()
        names = list(m_dir.model.exog_names)
        aprime = float(m_dir.fe_mean[names.index("repair_z")])
        akl = float(m_dir.fe_mean[names.index("kl_sum_z")])
        out.update({"glmm_repair_total": a, "glmm_repair_direct": aprime, "glmm_kl_coef": akl,
                    "glmm_prop_mediated": float((a - aprime) / a) if abs(a) > 1e-6 else None})
    except Exception as e:
        out["glmm_error"] = str(e)[:200]
    try:
        Xt = sm.add_constant(pd.concat([sub[["repair_z"]],
                             pd.get_dummies(sub["setting"], drop_first=True, dtype=float)], axis=1))
        mt = sm.Logit(sub["flip"], Xt).fit(disp=0, method="bfgs", maxiter=300)
        Xd = sm.add_constant(pd.concat([sub[["repair_z", "kl_sum_z"]],
                             pd.get_dummies(sub["setting"], drop_first=True, dtype=float)], axis=1))
        md = sm.Logit(sub["flip"], Xd).fit(disp=0, method="bfgs", maxiter=300)
        a = float(mt.params["repair_z"]); ap = float(md.params["repair_z"])
        out.update({"fe_repair_total": a, "fe_repair_direct": ap,
                    "fe_kl_coef": float(md.params["kl_sum_z"]),
                    "fe_prop_mediated": float((a - ap) / a) if abs(a) > 1e-6 else None})
    except Exception as e:
        out["fe_error"] = str(e)[:200]
    return out


def main():
    df = build_table()
    df.to_parquet(HERE / "exp11_sample_table.parquet", index=False)
    print(f"sample table: {len(df)} rows, {df['setting'].nunique()} settings")
    print("included w/ kl:", int((df["included"] & df["kl_sum"].notna()).sum()))

    recs = []
    for setting, g in df.groupby("setting"):
        model = g["model"].iloc[0]
        rec = {"setting": setting, "model": model, "benchmark": g["benchmark"].iloc[0],
               "condition": g["condition"].iloc[0], "is_core": bool(g["is_core"].iloc[0]),
               "qwen_dedup_on": model.startswith("Qwen")}
        r_min = mediation_one(g, "repair_min")
        r_mean = mediation_one(g, "repair_mean")
        for k, v in r_min.items():
            rec["min__" + k] = v
        for k in ("prop_mediated", "stage1_repair_coef", "stage1_repair_neg_sig"):
            if k in r_mean:
                rec["mean__" + k] = r_mean[k]
        recs.append(rec)
    mdf = pd.DataFrame(recs)
    mdf.to_csv(HERE / "mediation_by_setting.csv", index=False)

    def judge(frame, label):
        ok = frame[frame.get("min__ok") == True]  # noqa: E712
        n_ok = len(ok)
        s1 = int(ok["min__stage1_repair_neg_sig"].sum()) if n_ok else 0
        pm = ok["min__prop_mediated"].dropna() if n_ok else pd.Series(dtype=float)
        pm_valid = pm[(pm > -2) & (pm < 3)]
        return {"label": label, "n_settings_ok": int(n_ok),
                "stage1_neg_sig_count": s1,
                "stage1_neg_sig_frac": float(s1 / n_ok) if n_ok else None,
                "prop_mediated_median": float(pm_valid.median()) if len(pm_valid) else None,
                "prop_mediated_mean": float(pm_valid.mean()) if len(pm_valid) else None,
                "n_settings_mediated_ge50pct": int((pm_valid >= 0.5).sum())}

    judg = {
        "all": judge(mdf, "all_settings"),
        "core5": judge(mdf[mdf["is_core"]], "core5_models"),
        "core_MC": judge(mdf[mdf["is_core"] & mdf["benchmark"].isin(
            ["arc", "commonsense_qa", "mmlu", "mmlu_pro"])], "core5_MC"),
        "validation_qwen": judge(mdf[mdf["model"].str.startswith("Qwen")], "qwen_dedup_on"),
        "validation_math": judge(mdf[mdf["benchmark"] == "math"], "math_shards"),
    }
    pooled = {
        "glmm_core5_min": glmm_pooled(df[df["is_core"]], "repair_min", "core5"),
        "glmm_all_min": glmm_pooled(df, "repair_min", "all"),
        "glmm_core5_mean": glmm_pooled(df[df["is_core"]], "repair_mean", "core5_sensitivity"),
    }
    c = judg["core5"]
    verdict = {
        "criterion_stage1_neg_sig_majority": {
            "frac": c["stage1_neg_sig_frac"], "pass": (c["stage1_neg_sig_frac"] or 0) > 0.5},
        "criterion_mediation_ge50pct": {
            "pooled_glmm_prop_mediated": pooled["glmm_core5_min"].get("glmm_prop_mediated"),
            "pooled_fe_prop_mediated": pooled["glmm_core5_min"].get("fe_prop_mediated"),
            "setting_median_prop_mediated": c["prop_mediated_median"],
            "pass_pooled": (pooled["glmm_core5_min"].get("glmm_prop_mediated") or 0) >= 0.5,
            "pass_median": (c["prop_mediated_median"] or 0) >= 0.5},
    }
    verdict["H11_supported"] = bool(
        verdict["criterion_stage1_neg_sig_majority"]["pass"]
        and (verdict["criterion_mediation_ge50pct"]["pass_pooled"]
             or verdict["criterion_mediation_ge50pct"]["pass_median"]))

    out = {"judgment_by_group": judg, "pooled": pooled, "verdict": verdict,
           "meta": {"n_settings": int(mdf["setting"].nunique()), "qwen_dedup_on": True,
                    "primary_predictor": "repair_min (weak-link)",
                    "flip": "TE flip (answers[B]!=answers[A]), included=not exclude & a_correct"}}
    json.dump(out, open(HERE / "mediation_pooled.json", "w"), indent=2, ensure_ascii=False,
              default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)
    print("\n=== JUDGMENT (H11) ===")
    print(json.dumps(judg, indent=1, default=str))
    print("\n=== POOLED ===")
    print(json.dumps(pooled, indent=1, default=str))
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=1, default=str))


if __name__ == "__main__":
    main()
