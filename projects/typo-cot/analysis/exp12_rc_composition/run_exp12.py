"""実験12: R_C 組成分析 (H12, M2 測定).

各設定 (31 = 既存25 + MATH6) の clean 側 R_C top-10 を 4 カテゴリに分類し、
設定レベルのシェアを算出。Δρ(top10)・削除RD と回帰/相関する。

- R_C ランキング: exp/02 (commit fef3958) の再構築ローダー
  `rc_word_ranking_from_cot_pt`。Mistral は word_scores 結合不良のため
  token_scores 貪欲整列で再構築 (loader が degenerate 検出で自動切替)。
- Δρ(top10): exp/04 prod + prod_math の delta_rho_table.json (ソース直接)。
- 削除RD: exp/02 prod/exp2 の <setting>_core/summary.json,
  flip_rate(top_rc_unrestricted_delete_k4) - flip_rate(stratum_matched_random_delete_k4)。

出力:
  rc_composition_by_setting.csv   31設定 × 4カテゴリシェア + Δρ + 削除RD
  correlations.json               設定レベル回帰・相関 + 事前登録判定
  rc_top10_examples.json          各設定の代表 top-10 (監査用)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP02 = "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-02-target-deletion/projects/typo-cot"
sys.path.insert(0, EXP02 + "/src")
import torch  # noqa: E402
from typo_cot.intervention.loo_scorer import (  # noqa: E402
    rc_word_ranking_from_cot_pt,
    word_scores_degenerate,
)
from typo_cot.models.prompts import create_prompt_template  # noqa: E402

sys.path.insert(0, str(HERE))
from rc_classifier import CATEGORIES, classify_entry, compose_top10  # noqa: E402

ARCHIVE = "/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline"
DELTA_PROD = "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-04-fixed-target/projects/typo-cot/results/prod/delta_rho/delta_rho_table.json"
DELTA_MATH = "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-04-fixed-target/projects/typo-cot/results/prod_math/delta_rho/delta_rho_table.json"
EXP2_DIR = "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-02-target-deletion/projects/typo-cot/results/prod/exp2"

MAX_SAMPLES = 500          # 設定あたり最大サンプル数 (シェア推定は安定; 決定論的先頭N)
MC_BENCH = {"arc", "commonsense_qa", "mmlu", "mmlu_pro"}
MODEL_FAMILY = {
    "gemma-3-1b-it": "Gemma", "gemma-3-4b-it": "Gemma",
    "Llama-3.2-1B-Instruct": "Llama", "Llama-3.2-3B-Instruct": "Llama",
    "Mistral-7B-Instruct-v0.3": "Mistral", "Qwen2.5-7B-Instruct": "Qwen",
}


def parse_setting(setting: str) -> tuple[str, str]:
    for m in MODEL_FAMILY:
        if setting.startswith(m + "_"):
            return m, setting[len(m) + 1:]
    raise ValueError(setting)


def build_prompt(template, benchmark, entry):
    if benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
        pr = template.generate(question=entry["question"], choices=entry.get("choices"),
                               subject=entry.get("subset"))
    elif benchmark == "gsm8k":
        pr = template.generate(question=entry["question"])
    elif benchmark in ["bbh", "math", "strategy_qa"]:
        pr = template.generate(question=entry["question"], subject=entry.get("subset"))
    else:
        pr = template.generate(question=entry["question"])
    return pr.get_full_prompt()


def load_delta_rho_top10() -> dict[str, float]:
    out = {}
    for f in (DELTA_PROD, DELTA_MATH):
        d = json.load(open(f))
        for r in d["rows"]:
            if r["k"] == "top10":
                out[r["setting"]] = r["delta_rho"]
    return out


def load_deletion_rd() -> dict[str, float]:
    out = {}
    for name in os.listdir(EXP2_DIR):
        if not name.endswith("_core") or ".broken" in name:
            continue
        setting = name[:-5]
        p = Path(EXP2_DIR) / name / "summary.json"
        if not p.exists():
            continue
        arms = json.load(open(p))["strata"]["all"]["arms"]
        top = arms.get("top_rc_unrestricted_delete_k4")
        rnd = arms.get("stratum_matched_random_delete_k4")
        if top and rnd and top.get("flip_rate") is not None and rnd.get("flip_rate") is not None:
            out[setting] = top["flip_rate"] - rnd["flip_rate"]
    return out


def compose_setting(setting: str) -> dict:
    model, benchmark = parse_setting(setting)
    run_dir = Path(ARCHIVE) / setting
    results = json.load(open(run_dir / "results.json"))
    template = create_prompt_template(benchmark)
    counts = {c: 0 for c in CATEGORIES}
    n_used = n_degenerate = 0
    examples = []
    for e in results[:MAX_SAMPLES]:
        sid = e["sample_id"]
        pt = run_dir / "importance_scores" / f"{sid}_cot.pt"
        if not pt.exists():
            continue
        data = torch.load(pt, map_location="cpu", weights_only=False)
        deg = word_scores_degenerate(data)
        full = build_prompt(template, benchmark, e) + e["generated_text"]
        ranking = rc_word_ranking_from_cot_pt(data, full_text=full)
        words = [str(d["word"]) for d in ranking]
        if not words:
            continue
        c = compose_top10(words)
        for k in CATEGORIES:
            counts[k] += c[k]
        n_used += 1
        n_degenerate += int(deg)
        if len(examples) < 5:
            examples.append({"sample_id": sid, "degenerate": deg,
                             "top10": [(w, classify_entry(w)) for w in words[:10]]})
    total = sum(counts.values()) or 1
    shares = {f"share_{k}": counts[k] / total for k in CATEGORIES}
    return {
        "setting": setting, "model": model, "benchmark": benchmark,
        "family": MODEL_FAMILY[model],
        "task_type": "MC" if benchmark in MC_BENCH else "OpenNum",
        "n_samples": n_used, "n_degenerate": n_degenerate,
        "reconstruct_loader": n_degenerate > 0,
        **shares,
        "share_content_plus_numeric": shares["share_content"] + shares["share_numeric"],
        "_examples": examples,
    }


def main():
    delta = load_delta_rho_top10()
    rd = load_deletion_rd()
    settings = sorted(delta.keys())
    print(f"settings (delta_rho top10): {len(settings)}")

    rows, examples = [], {}
    for s in settings:
        if not (Path(ARCHIVE) / s / "results.json").exists():
            print(f"  SKIP {s}: no archive results.json")
            continue
        r = compose_setting(s)
        examples[s] = r.pop("_examples")
        r["delta_rho_top10"] = delta.get(s)
        r["deletion_rd_k4"] = rd.get(s)
        rows.append(r)
        rd_s = "  n/a" if r["deletion_rd_k4"] is None else f"{r['deletion_rd_k4']:+.3f}"
        print(f"  {s:42s} n={r['n_samples']:4d} deg={r['reconstruct_loader']!s:5s} "
              f"concl={r['share_conclusion']:.3f} num={r['share_numeric']:.3f} "
              f"cont={r['share_content']:.3f} func={r['share_function']:.3f} "
              f"dRho={r['delta_rho_top10']:+.3f} RD={rd_s}")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "rc_composition_by_setting.csv", index=False)
    json.dump(examples, open(HERE / "rc_top10_examples.json", "w"), indent=1, ensure_ascii=False)

    from scipy import stats
    import statsmodels.api as sm

    def reg(y, x, d):
        sub = d.dropna(subset=[y, x])
        if len(sub) < 4:
            return None
        X = sm.add_constant(sub[x])
        m = sm.OLS(sub[y], X).fit()
        pear = stats.pearsonr(sub[x], sub[y])
        spear = stats.spearmanr(sub[x], sub[y])
        return {"y": y, "x": x, "n": int(len(sub)),
                "slope": float(m.params[x]), "intercept": float(m.params["const"]),
                "r2": float(m.rsquared), "p_slope": float(m.pvalues[x]),
                "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
                "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue)}

    corr = {}
    corr["dRho_vs_conclusion_all31"] = reg("delta_rho_top10", "share_conclusion", df)
    d2 = df.copy()
    d2["abs_delta_rho_top10"] = d2["delta_rho_top10"].abs()
    corr["absDRho_vs_conclusion_all31"] = reg("abs_delta_rho_top10", "share_conclusion", d2)
    # MC 設定のみ (結論句定型が意味を持つのは MC/答え定型設定)
    corr["dRho_vs_conclusion_MC"] = reg("delta_rho_top10", "share_conclusion",
                                        df[df["task_type"] == "MC"])
    corr["deletionRD_vs_content_plus_numeric"] = reg(
        "deletion_rd_k4", "share_content_plus_numeric", df)
    corr["deletionRD_vs_conclusion"] = reg("deletion_rd_k4", "share_conclusion", df)

    # 家系レベル対比 (conclusion share と Δρ 符号)
    fam_stats = {}
    for fam in ["Gemma", "Llama", "Mistral", "Qwen"]:
        sub = df[df["family"] == fam]
        if len(sub):
            fam_stats[fam] = {
                "n": int(len(sub)),
                "mean_conclusion": float(sub["share_conclusion"].mean()),
                "mean_delta_rho": float(sub["delta_rho_top10"].mean()),
                "frac_delta_rho_positive": float((sub["delta_rho_top10"] > 0).mean()),
            }
    corr["family_contrast"] = fam_stats

    mc = df[df["task_type"] == "MC"]
    gl_mc = mc[mc["family"].isin(["Gemma", "Llama"])]
    mistral = df[df["family"] == "Mistral"]
    numeric_tasks = df[df["benchmark"].isin(["gsm8k", "math"])]
    r_dRho = corr["dRho_vs_conclusion_all31"]["pearson_r"] if corr["dRho_vs_conclusion_all31"] else None

    verdict = {
        "gemma_llama_MC_conclusion_gt_0.5": {
            "shares": gl_mc.set_index("setting")["share_conclusion"].round(3).to_dict(),
            "n_pass": int((gl_mc["share_conclusion"] > 0.5).sum()), "n_total": int(len(gl_mc)),
            "mean": float(gl_mc["share_conclusion"].mean())},
        "mistral_conclusion_lt_0.3": {
            "shares": mistral.set_index("setting")["share_conclusion"].round(3).to_dict(),
            "n_pass": int((mistral["share_conclusion"] < 0.3).sum()), "n_total": int(len(mistral)),
            "mean": float(mistral["share_conclusion"].mean())},
        "gsm8k_math_numeric_plus_content_gt_0.7": {
            "shares": numeric_tasks.set_index("setting")["share_content_plus_numeric"].round(3).to_dict(),
            "n_pass": int((numeric_tasks["share_content_plus_numeric"] > 0.7).sum()),
            "n_total": int(len(numeric_tasks)),
            "mean": float(numeric_tasks["share_content_plus_numeric"].mean())},
        "abs_r_conclusion_dRho_ge_0.7": {
            "pearson_r": r_dRho,
            "abs_pearson_r": abs(r_dRho) if r_dRho is not None else None,
            "pass": (abs(r_dRho) >= 0.7) if r_dRho is not None else None},
    }
    out = {"correlations": corr, "verdict": verdict,
           "meta": {"n_settings": int(len(df)), "max_samples_per_setting": MAX_SAMPLES,
                    "classifier": "rc_classifier.py (POS-proxy stoplist + template match)"}}
    json.dump(out, open(HERE / "correlations.json", "w"), indent=2, ensure_ascii=False)
    print("\n=== VERDICT (H12) ===")
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    print("\n=== CORRELATIONS ===")
    print(json.dumps(corr, indent=1))


if __name__ == "__main__":
    main()
