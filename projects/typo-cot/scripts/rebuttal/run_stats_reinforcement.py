#!/usr/bin/env python3
"""Rebuttal 実験④: 統計補強 (AxQH #4).

既存分析ログ (outputs/analysis/{bench}/{model}/k4_importance/full_results.json) から:
(a) 主要偏相関 ρ(J|R), ρ(R|J) のブートストラップ 95%CI (percentile, B=2000)
(b) Holm 補正 p 値 (25 設定 × 各ファミリー内で補正)
(c) mixed-effects logistic regression:
    flip ~ z(CoT:ROUGE-L) + z(CoT:Jaccard@10) + (1|model) + (1|benchmark)
    (statsmodels BinomialBayesMixedGLM, 変分ベイズ)

偏相関の点推定は analyzer.py:2263-2330 と同じ定義 (単一統制変数の場合、
OLS 残差同士の Pearson = 一次偏相関公式) を用いる。p 値は two-sided。

使用例:
  uv run --no-sync python scripts/rebuttal/run_stats_reinforcement.py \
    --analysis_dir outputs/analysis --output_dir outputs/rebuttal
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

MODELS = [
    "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Mistral-7B-Instruct-v0.3",
    "gemma-3-1b-it", "gemma-3-4b-it",
]
BENCHES = ["gsm8k", "mmlu", "mmlu_pro", "commonsense_qa", "arc"]
B_BOOT = 2000
SEED = 42


def partial_corr(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """一次偏相関 r_xy.z (analyzer の OLS 残差 Pearson と数学的に同値)."""
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom < 1e-12:
        return np.nan
    return (rxy - rxz * ryz) / denom


def partial_corr_p(r: float, n: int) -> float:
    """偏相関の two-sided p 値 (t 分布, 自由度 n-3)."""
    if np.isnan(r) or n <= 3 or abs(r) >= 1:
        return np.nan
    t = r * np.sqrt((n - 3) / (1 - r**2))
    return float(2 * stats.t.sf(abs(t), df=n - 3))


def bootstrap_ci(y, x, z, rng, b=B_BOOT):
    n = len(y)
    vals = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        vals[i] = partial_corr(y[idx], x[idx], z[idx])
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni 補正 (NaN は補正対象外)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adjusted = [np.nan] * len(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        adj = max(adj, prev)  # 単調性
        adjusted[i] = adj
        prev = adj
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="統計補強 (bootstrap CI / Holm / GLMM)")
    parser.add_argument("--analysis_dir", type=str, default="outputs/analysis")
    parser.add_argument("--output_dir", type=str, default="outputs/rebuttal")
    parser.add_argument("--skip_glmm", action="store_true")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    settings = []
    pooled_rows = []

    for bench in BENCHES:
        for model in MODELS:
            fr_path = analysis_dir / bench / model / "k4_importance" / "full_results.json"
            if not fr_path.exists():
                print(f"skip (missing): {fr_path}")
                continue
            with open(fr_path, encoding="utf-8") as f:
                sr = json.load(f)["sample_results"]

            rows = [
                {
                    "flip": int(bool(s["answer_changed"])),
                    "before_correct": bool(s["before_correct"]),
                    "ci": int(s["pattern"] == "correct→incorrect"),
                    "jac10": float(s["cot_metrics"]["jaccard"]["top10"]),
                    "rouge": float(s["cot_metrics"]["rouge_l"]["f1"]),
                    "model": model,
                    "bench": bench,
                }
                for s in sr
            ]
            pooled_rows.extend(rows)

            y = np.array([r["flip"] for r in rows], dtype=float)
            xj = np.array([r["jac10"] for r in rows])
            xr = np.array([r["rouge"] for r in rows])
            n = len(rows)

            rec = {"model": model, "bench": bench, "n_all": n}

            # target: answer_changed (all samples)
            r_j = partial_corr(y, xj, xr)
            r_r = partial_corr(y, xr, xj)
            rec["flip_rho_J_given_R"] = float(r_j)
            rec["flip_rho_J_given_R_p"] = partial_corr_p(r_j, n)
            rec["flip_rho_J_given_R_ci95"] = bootstrap_ci(y, xj, xr, rng)
            rec["flip_rho_R_given_J"] = float(r_r)
            rec["flip_rho_R_given_J_p"] = partial_corr_p(r_r, n)
            rec["flip_rho_R_given_J_ci95"] = bootstrap_ci(y, xr, xj, rng)

            # target: correct→incorrect (correct-before subset)
            sub = [r for r in rows if r["before_correct"]]
            ys = np.array([r["ci"] for r in sub], dtype=float)
            xjs = np.array([r["jac10"] for r in sub])
            xrs = np.array([r["rouge"] for r in sub])
            ns = len(sub)
            rec["n_correct_before"] = ns
            if ns >= 20 and ys.std() > 0:
                r_j2 = partial_corr(ys, xjs, xrs)
                r_r2 = partial_corr(ys, xrs, xjs)
                rec["ci_rho_J_given_R"] = float(r_j2)
                rec["ci_rho_J_given_R_p"] = partial_corr_p(r_j2, ns)
                rec["ci_rho_J_given_R_ci95"] = bootstrap_ci(ys, xjs, xrs, rng)
                rec["ci_rho_R_given_J"] = float(r_r2)
                rec["ci_rho_R_given_J_p"] = partial_corr_p(r_r2, ns)
                rec["ci_rho_R_given_J_ci95"] = bootstrap_ci(ys, xrs, xjs, rng)

            settings.append(rec)
            print(
                f"{model} {bench}: n={n} ρ(J|R)={r_j:.3f} "
                f"CI={rec['flip_rho_J_given_R_ci95']} ρ(R|J)={r_r:.3f}"
            )

    # Holm 補正 (ファミリー = 各指標×ターゲット, 25 設定間)
    for key in [
        "flip_rho_J_given_R_p", "flip_rho_R_given_J_p",
        "ci_rho_J_given_R_p", "ci_rho_R_given_J_p",
    ]:
        pvals = [s.get(key, np.nan) for s in settings]
        adj = holm(pvals)
        for s, a in zip(settings, adj, strict=True):
            if key in s:
                s[key.replace("_p", "_p_holm")] = None if np.isnan(a) else float(a)

    result = {"settings": settings, "bootstrap_B": B_BOOT, "seed": SEED}

    # (c) mixed-effects logistic regression
    if not args.skip_glmm:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        df = pd.DataFrame(pooled_rows)
        df["z_jac10"] = (df["jac10"] - df["jac10"].mean()) / df["jac10"].std()
        df["z_rouge"] = (df["rouge"] - df["rouge"].mean()) / df["rouge"].std()

        glmm = BinomialBayesMixedGLM.from_formula(
            "flip ~ z_rouge + z_jac10",
            vc_formulas={"model": "0 + C(model)", "bench": "0 + C(bench)"},
            data=df,
        )
        fit = glmm.fit_vb()
        fe_names = list(fit.model.exog_names)
        fe_mean = fit.fe_mean.tolist()
        fe_sd = fit.fe_sd.tolist()
        glmm_out = {
            "formula": "flip ~ z_rouge + z_jac10 + (1|model) + (1|benchmark)",
            "method": "BinomialBayesMixedGLM (variational Bayes)",
            "n": int(len(df)),
            "fixed_effects": [
                {
                    "name": nm,
                    "posterior_mean": m,
                    "posterior_sd": s,
                    "approx_z": m / s if s > 0 else None,
                }
                for nm, m, s in zip(fe_names, fe_mean, fe_sd, strict=True)
            ],
        }
        result["glmm_flip"] = glmm_out
        print("\nGLMM (flip):")
        for fe in glmm_out["fixed_effects"]:
            print(f"  {fe['name']}: {fe['posterior_mean']:.4f} ± {fe['posterior_sd']:.4f} (z≈{fe['approx_z']:.1f})")

    with open(output_dir / "stats_reinforcement.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Markdown サマリ
    lines = [
        "# Statistical reinforcement (k=4 importance, Jaccard@10)",
        "",
        "## Partial correlations vs answer flip (all samples), bootstrap 95% CI, Holm-adjusted p",
        "",
        "| Model | Bench | n | ρ(J\\|R) | 95% CI | p (Holm) | ρ(R\\|J) | 95% CI | p (Holm) |",
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for s in settings:
        ci_j = s["flip_rho_J_given_R_ci95"]
        ci_r = s["flip_rho_R_given_J_ci95"]
        lines.append(
            f"| {s['model']} | {s['bench']} | {s['n_all']} "
            f"| {s['flip_rho_J_given_R']:.3f} | [{ci_j[0]:.3f}, {ci_j[1]:.3f}] "
            f"| {s.get('flip_rho_J_given_R_p_holm', float('nan')):.2e} "
            f"| {s['flip_rho_R_given_J']:.3f} | [{ci_r[0]:.3f}, {ci_r[1]:.3f}] "
            f"| {s.get('flip_rho_R_given_J_p_holm', float('nan')):.2e} |"
        )
    if "glmm_flip" in result:
        lines += ["", "## Mixed-effects logistic regression (pooled)", "",
                  f"`{result['glmm_flip']['formula']}` (n={result['glmm_flip']['n']})", ""]
        for fe in result["glmm_flip"]["fixed_effects"]:
            lines.append(
                f"- {fe['name']}: {fe['posterior_mean']:.4f} ± {fe['posterior_sd']:.4f} (z≈{fe['approx_z']:.1f})"
            )
    with open(output_dir / "stats_reinforcement.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n出力: {output_dir}/stats_reinforcement.json / .md")


if __name__ == "__main__":
    main()
