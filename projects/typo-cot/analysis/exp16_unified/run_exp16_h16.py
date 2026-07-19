"""実験16 事前登録 H16: 設定分散 吸収テスト (正式 estimand).

registry lines 712-736 の判定基準:
  base GLMM: flip ~ 1 + (1|setting)               -> sigma2_setting(base)
  mod  GLMM: flip ~ M1 + M2 + M3a + M3b + (1|setting) -> sigma2_setting(mod)
  吸収率 A = 1 - sigma2_setting(mod)/sigma2_setting(base)
  判定: A >= 0.5 で Supported。A < 0.5 なら falsification branch =
        二層レジーム (free-form: carry/IE 優位 vs MC/selection: DE/shortcut 優位)。

設定レベルモデレーター:
  M1  = repair_min の設定平均 (内部修復)
  M2  = share_conclusion (exp12, R_C 組成; 既に設定レベル)
  M3a = gini_rc (exp13 loo_gini; 設定レベル; gsm8k/mmlu のみ)
  M3b = nocot_flip_rate (exp14 settings.csv; no-CoT shortcut)
全て z 標準化。fitter は BinomialBayesMixedGLM(VB), (1|setting)。
sigma2 は VB 事後 (log-SD パラメータ) から exp(2*vcp_mean)。

出力: exp16_h16_absorption.json (既存 md へ節追記は別途)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

HERE = Path(__file__).resolve().parent
SETTINGS_CSV = Path("/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/"
                    ".claude/worktrees/exp-14-nocot/projects/typo-cot/results/"
                    "exp14_nocot/analysis/settings.csv")

FREEFORM_BENCH = {"gsm8k", "math"}  # 自由生成 (残りは MC/selection)


def load_data():
    ff = pd.read_parquet(HERE / "features_full.parquet")
    s = pd.read_csv(SETTINGS_CSV)
    s2 = (s.rename(columns={"perturbation": "condition"})
          [["model", "benchmark", "condition", "nocot_flip_rate"]])
    m = ff.merge(s2, on=["model", "benchmark", "condition"], how="left")
    inc = m[m["included"]].copy()
    inc["repair_setting_mean"] = inc.groupby("setting")["repair_min"].transform("mean")
    inc["regime"] = np.where(inc["benchmark"].isin(FREEFORM_BENCH), "free_form", "mc")
    return inc


def zc(df, cols):
    out = df.copy()
    for c in cols:
        v = out[c].astype(float)
        out[c + "_z"] = (v - v.mean()) / v.std(ddof=0)
    return out


def fit_sigma2(data, rhs_terms):
    """BBGLM(VB) with (1|setting)。返り値: dict(sigma2, sd, vcp_mean, vcp_sd, CI, fe)."""
    formula = "flip ~ " + (" + ".join(rhs_terms) if rhs_terms else "1")
    vc = {"setting": "0 + C(setting)"}
    r = BinomialBayesMixedGLM.from_formula(formula, vc, data).fit_vb(verbose=False)
    vcp = float(np.asarray(r.vcp_mean)[0])
    vcp_sd = float(np.asarray(r.vcp_sd)[0])
    sd = np.exp(vcp); sigma2 = np.exp(2 * vcp)
    sd_lb, sd_ub = np.exp(vcp - 1.96 * vcp_sd), np.exp(vcp + 1.96 * vcp_sd)
    fe = {}
    names = list(r.model.exog_names)
    for i, nm in enumerate(names):
        mean = float(r.fe_mean[i]); s = float(r.fe_sd[i])
        z = mean / s if s > 0 else np.nan
        p = float(2 * (1 - sstats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        fe[nm] = {"coef": mean, "sd": s, "z": float(z), "p": p}
    return {
        "formula": formula, "n": int(r.model.exog.shape[0]),
        "vcp_mean_logSD": vcp, "vcp_sd": vcp_sd,
        "sigma_setting_SD": float(sd), "sigma2_setting": float(sigma2),
        "sigma2_CI95": [float(sd_lb ** 2), float(sd_ub ** 2)],
        "converged": bool(getattr(r, "converged", True)),
        "fixed_effects": fe,
    }


def absorption(base, mod):
    return 1.0 - mod["sigma2_setting"] / base["sigma2_setting"]


def run_pair(data, mods, label):
    d = zc(data, mods)
    rhs = [c + "_z" for c in mods]
    base = fit_sigma2(d, [])
    mod = fit_sigma2(d, rhs)
    A = absorption(base, mod)
    return {
        "label": label, "moderators": mods,
        "n_settings": int(d["setting"].nunique()), "n_samples": int(len(d)),
        "sigma2_setting_base": base["sigma2_setting"],
        "sigma2_setting_mod": mod["sigma2_setting"],
        "A_absorption": float(A),
        "base": base, "mod": mod,
    }


def main():
    inc = load_data()

    # ---- Primary: 4 moderator (M1,M2,M3a=gini,M3b=nocot) ----
    cc = inc.dropna(subset=["repair_setting_mean", "share_conclusion",
                            "gini_rc", "nocot_flip_rate"]).copy()
    mods = ["repair_setting_mean", "share_conclusion", "gini_rc", "nocot_flip_rate"]
    primary = run_pair(cc, mods, "primary_4mod_gini_gsm8k_mmlu")
    A = primary["A_absorption"]

    # ---- Robustness: + family covariate (同一 CC) ----
    ccf = zc(cc, mods)
    ccf["family"] = pd.Categorical(ccf["family"])
    base_f = fit_sigma2(ccf, [])
    mod_f = fit_sigma2(ccf, [c + "_z" for c in mods] + ["C(family)"])
    fam_ver = {
        "label": "primary_4mod_plus_family",
        "n_settings": int(ccf["setting"].nunique()), "n_samples": int(len(ccf)),
        "sigma2_setting_base": base_f["sigma2_setting"],
        "sigma2_setting_mod": mod_f["sigma2_setting"],
        "A_absorption": float(1.0 - mod_f["sigma2_setting"] / base_f["sigma2_setting"]),
        "mod_fixed_effects": mod_f["fixed_effects"],
    }

    # ---- Robustness: no-gini (3 moderator), 広い設定被覆 ----
    ccn = inc.dropna(subset=["repair_setting_mean", "share_conclusion",
                             "nocot_flip_rate"]).copy()
    nog = run_pair(ccn, ["repair_setting_mean", "share_conclusion",
                         "nocot_flip_rate"], "robust_3mod_nogini")

    out = {
        "hypothesis": "H16 preregistered setting-variance absorption (registry 712-736)",
        "estimand": "A = 1 - sigma2_setting(mod)/sigma2_setting(base)",
        "decision_rule": "A >= 0.5 -> Supported; A < 0.5 -> two-regime falsification branch",
        "fitter": "BinomialBayesMixedGLM (VB), random intercept (1|setting)",
        "fitter_note": ("item RE (~17k 交差水準) は非現実的なため (1|setting) のみ。"
                        "sigma2 は VB 事後の log-SD パラメータより exp(2*vcp_mean)。"),
        "moderators": {
            "M1_repair": "repair_min の設定平均",
            "M2_composition": "share_conclusion (exp12, 設定レベル R_C 組成)",
            "M3a_readout_gini": "gini_rc = exp13 loo_gini (gsm8k/mmlu のみ, Qwen 無し)",
            "M3b_nocot_shortcut": "nocot_flip_rate (exp14 settings.csv, 設定レベル, 全設定被覆)",
        },
        "coverage_note": ("gini_rc が gsm8k/mmlu×5非Qwenモデルのみのため、4モデレーター "
                          "complete-case は 20 設定 (gsm8k+mmlu, 3家族, 両条件, N=20,500) に限定。"
                          "Qwen と arc/csqa/math 設定は gini 欠損で脱落。"),
        "primary": primary,
        "robustness_plus_family": fam_ver,
        "robustness_no_gini_wider": nog,
        "primary_A": A,
        "judgment": ("Supported (A>=0.5)" if A >= 0.5
                     else "Refuted -> two-regime falsification branch (A<0.5)"),
    }

    # ---- Falsification branch: regime-split A (exploratory) ----
    if A < 0.5:
        regime_A = {}
        for reg in ["free_form", "mc"]:
            sub = cc[cc["regime"] == reg]
            if sub["setting"].nunique() >= 4:
                try:
                    regime_A[reg] = run_pair(sub, mods, f"regime_{reg}")
                except Exception as e:
                    regime_A[reg] = {"error": f"{type(e).__name__}: {e}"}
            else:
                regime_A[reg] = {"note": f"only {sub['setting'].nunique()} settings"}
        out["falsification_two_regime"] = {
            "trigger": "A < 0.5",
            "regime_definition": "free_form={gsm8k,math} vs mc={arc,csqa,mmlu,mmlu_pro}",
            "regime_split_A": {k: (v.get("A_absorption") if isinstance(v, dict) else None)
                               for k, v in regime_A.items()},
            "regime_detail": regime_A,
            "consistency_prior_H": {
                "H12": "MC-only R_C purity r=+0.705 (free-form では崩れる)",
                "H13": "Mistral double dissociation (集中は Llama並でも RD_content 桁違い低)",
                "H14": "MC 層別 nocot-vs-DE rho=+0.726 vs 全体 Simpson 反転",
            },
        }

    (HERE / "exp16_h16_absorption.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("WROTE", HERE / "exp16_h16_absorption.json")
    print(f"PRIMARY: n_settings={primary['n_settings']} N={primary['n_samples']}")
    print(f"  sigma2_base={primary['sigma2_setting_base']:.4f} "
          f"sigma2_mod={primary['sigma2_setting_mod']:.4f} A={A:.3f} -> {out['judgment']}")
    print(f"  +family: A={fam_ver['A_absorption']:.3f}")
    print(f"  no-gini(3mod, {nog['n_settings']} settings): "
          f"sigma2_base={nog['sigma2_setting_base']:.4f} "
          f"sigma2_mod={nog['sigma2_setting_mod']:.4f} A={nog['A_absorption']:.3f}")
    if A < 0.5:
        print("  regime-split A:", out["falsification_two_regime"]["regime_split_A"])
    return out


if __name__ == "__main__":
    main()
