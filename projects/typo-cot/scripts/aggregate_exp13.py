#!/usr/bin/env python3
"""実験13: 読み出し集中度(M3) の設定レベル突合と H13 判定.

以下を突合して analysis/exp13_readout_concentration/ に出力する:
  - LOO 集中度 (loo_concentration.json; compute_loo_concentration.py の出力)
  - attention 集中度代理 (results/attention/*/summary.json)
  - 削除RD (exp2 の {model}_{benchmark}_core/summary.json の k=4 risk_difference)

判定 (事前登録 H13):
  (1) Gini モデルファミリー順位 Llama > Gemma > Mistral
  (2) rank-corr(setting Gini, 削除RD) >= 0.7

出力: setting_table.csv, exp13_summary.json (rank-corr・family順位・H13判定・
      LOO Gini と attention Gini の代理妥当性)。日付フィールドなし。
"""

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import spearmanr

MODELS = [
    "gemma-3-1b-it", "gemma-3-4b-it",
    "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]


def family_of(model: str) -> str:
    m = model.lower()
    if "gemma" in m:
        return "gemma"
    if "llama" in m:
        return "llama"
    if "mistral" in m:
        return "mistral"
    return "other"


def load_deletion_rd(exp2_dir: Path, model: str, benchmark: str) -> dict:
    """exp2 の core summary から k=4 削除RD (all / content) を抽出する."""
    p = exp2_dir / f"{model}_{benchmark}_core" / "summary.json"
    if not p.exists():
        return {"rd_all_k4": None, "rd_content_k4": None, "exp2_summary": None}
    s = json.load(open(p, encoding="utf-8"))
    rd_all = rd_content = None
    for c in s.get("contrasts", []):
        if c.get("k") == 4 and c.get("op") == "delete":
            if c.get("stratum") == "all" and "unrestricted" in c.get("arm_a", ""):
                rd_all = c.get("risk_difference")
            elif c.get("stratum") == "content":
                rd_content = c.get("risk_difference")
    return {"rd_all_k4": rd_all, "rd_content_k4": rd_content, "exp2_summary": str(p)}


def load_attention(attn_dir: Path, model: str, benchmark: str) -> dict:
    """attention 集中度代理の設定サマリを読む (無ければ None)."""
    for label in ("clean_attn",):
        p = attn_dir / f"{model}_{benchmark}_{label}" / "summary.json"
        if p.exists():
            s = json.load(open(p, encoding="utf-8"))
            m = s.get("metrics", {})
            return {
                "attn_gini_mean": m.get("mean_attn_gini_mean"),
                "attn_gini_agg": m.get("mean_attn_gini_agg"),
                "attn_top1_agg": m.get("mean_attn_top1_agg"),
                "attn_n": s.get("stats", {}).get("scored"),
            }
    return {"attn_gini_mean": None, "attn_gini_agg": None,
            "attn_top1_agg": None, "attn_n": None}


def _spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"rho": None, "p": None, "n": len(pairs)}
    xr, yr = zip(*pairs)
    rho, p = spearmanr(xr, yr)
    return {"rho": float(rho), "p": float(p), "n": len(pairs)}


def main() -> None:
    ap = argparse.ArgumentParser(description="exp13 aggregation + H13 judgment")
    ap.add_argument("--out_dir", type=str,
                    default="analysis/exp13_readout_concentration")
    ap.add_argument("--exp2_dir", type=str, default=(
        "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/"
        "worktrees/exp-02-target-deletion/projects/typo-cot/results/prod/exp2"))
    ap.add_argument("--attn_dir", type=str, default="results/attention")
    ap.add_argument("--benchmarks", type=str, default="gsm8k,mmlu",
                    help="LOO/主判定に使うベンチ (カンマ区切り)")
    ap.add_argument("--rd_primary", type=str, default="rd_content_k4",
                    choices=["rd_content_k4", "rd_all_k4"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    exp2_dir = Path(args.exp2_dir)
    attn_dir = Path(args.attn_dir)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    loo_path = out_dir / "loo_concentration.json"
    loo_settings = {}
    if loo_path.exists():
        loo_settings = json.load(open(loo_path, encoding="utf-8")).get("settings", {})

    def loo_for(model, benchmark):
        key = f"{model}_{benchmark}_clean_occ"
        return loo_settings.get(key)

    rows = []
    for model in MODELS:
        for benchmark in benchmarks:
            loo = loo_for(model, benchmark)
            rd = load_deletion_rd(exp2_dir, model, benchmark)
            attn = load_attention(attn_dir, model, benchmark)
            rows.append({
                "model": model,
                "family": family_of(model),
                "benchmark": benchmark,
                "loo_n": loo.get("n_samples") if loo else None,
                "loo_gini": loo.get("mean_gini") if loo else None,
                "loo_top1": loo.get("mean_top1_share") if loo else None,
                "loo_top4": loo.get("mean_top4_share") if loo else None,
                "loo_eff_count": loo.get("mean_effective_count") if loo else None,
                "loo_n_words": loo.get("mean_n_words") if loo else None,
                # 内容語限定 (削除RD の content 層と同区分)
                "content_gini": loo.get("mean_content_gini") if loo else None,
                "content_top1": loo.get("mean_content_top1_share") if loo else None,
                "content_mass_share": (
                    loo.get("mean_content_mass_share") if loo else None),
                "frac_top1_is_content": (
                    loo.get("frac_top1_is_content") if loo else None),
                **attn,
                **rd,
            })

    # ---- 設定レベル rank-corr ----
    rd_key = args.rd_primary
    loo_gini = [r["loo_gini"] for r in rows]
    attn_gini = [r["attn_gini_mean"] for r in rows]
    rd_vals = [r[rd_key] for r in rows]

    corr = {
        "loo_gini_vs_deletion_rd": _spearman(loo_gini, rd_vals),
        "loo_top1_vs_deletion_rd": _spearman([r["loo_top1"] for r in rows], rd_vals),
        "attn_gini_vs_deletion_rd": _spearman(attn_gini, rd_vals),
        "loo_gini_vs_attn_gini": _spearman(loo_gini, attn_gini),
        # 内容語限定の集中度・内容語追跡 vs 削除RD (content 層)
        "content_gini_vs_deletion_rd": _spearman(
            [r["content_gini"] for r in rows], rd_vals),
        "content_top1_vs_deletion_rd": _spearman(
            [r["content_top1"] for r in rows], rd_vals),
        "content_mass_share_vs_deletion_rd": _spearman(
            [r["content_mass_share"] for r in rows], rd_vals),
        "frac_top1_content_vs_deletion_rd": _spearman(
            [r["frac_top1_is_content"] for r in rows], rd_vals),
        "rd_key": rd_key,
    }
    # sensitivity: RD_all
    corr["loo_gini_vs_rd_all"] = _spearman(loo_gini, [r["rd_all_k4"] for r in rows])

    # ---- family Gini 順位 ----
    fam_gini = {}
    for fam in ("llama", "gemma", "mistral"):
        gs = [r["loo_gini"] for r in rows
              if r["family"] == fam and r["loo_gini"] is not None]
        fam_gini[fam] = (sum(gs) / len(gs)) if gs else None
    fam_rank_ok = (
        fam_gini["llama"] is not None and fam_gini["gemma"] is not None
        and fam_gini["mistral"] is not None
        and fam_gini["llama"] > fam_gini["gemma"] > fam_gini["mistral"]
    )
    fam_order = sorted(
        (f for f in fam_gini if fam_gini[f] is not None),
        key=lambda f: fam_gini[f], reverse=True,
    )

    # ---- H13 判定 ----
    rho = corr["loo_gini_vs_deletion_rd"]["rho"]
    corr_ok = rho is not None and rho >= 0.7
    n_loo_settings = sum(1 for r in rows if r["loo_gini"] is not None)
    judgment = {
        "family_rank_llama_gt_gemma_gt_mistral": fam_rank_ok,
        "family_gini_order": fam_order,
        "family_gini": fam_gini,
        "rankcorr_gini_vs_deletion_rd": rho,
        "rankcorr_threshold_met (>=0.7)": corr_ok,
        "H13_supported": bool(fam_rank_ok and corr_ok),
        "n_loo_settings_available": n_loo_settings,
        "note": ("H13 = M3(読み出し集中度)が削除介入効果量を説明。"
                 "2条件 (family順位 & rank-corr>=0.7) 両立で支持。"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "family", "benchmark", "loo_n", "loo_gini", "loo_top1",
              "loo_top4", "loo_eff_count", "loo_n_words",
              "content_gini", "content_top1", "content_mass_share",
              "frac_top1_is_content", "attn_gini_mean",
              "attn_gini_agg", "attn_top1_agg", "attn_n",
              "rd_content_k4", "rd_all_k4"]
    with open(out_dir / "setting_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "benchmarks": benchmarks,
        "rd_primary": rd_key,
        "rows": rows,
        "correlations": corr,
        "H13_judgment": judgment,
    }
    with open(out_dir / "exp13_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- コンソール出力 ----
    print(f"\n=== 設定別表 (RD primary={rd_key}) ===")
    print(f"{'model':26s} {'bm':6s} {'loo_gini':>8s} {'loo_top1':>8s} "
          f"{'attn_gini':>9s} {'rd_cont':>8s} {'rd_all':>8s}")
    for r in rows:
        def fmt(x, w=8, p=4):
            return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}")
        print(f"{r['model']:26s} {r['benchmark']:6s} {fmt(r['loo_gini'])} "
              f"{fmt(r['loo_top1'])} {fmt(r['attn_gini_mean'], 9)} "
              f"{fmt(r['rd_content_k4'])} {fmt(r['rd_all_k4'])}")
    print(f"\nfamily Gini: {fam_gini}  order={fam_order}")
    print(f"rank-corr(LOO Gini(全語), 削除RD[{rd_key}]) = "
          f"{corr['loo_gini_vs_deletion_rd']}")
    print(f"rank-corr(内容語Gini, 削除RD) = {corr['content_gini_vs_deletion_rd']}")
    print(f"rank-corr(内容語質量シェア, 削除RD) = "
          f"{corr['content_mass_share_vs_deletion_rd']}")
    print(f"rank-corr(top1が内容語の割合, 削除RD) = "
          f"{corr['frac_top1_content_vs_deletion_rd']}")
    print(f"rank-corr(attn Gini, 削除RD) = {corr['attn_gini_vs_deletion_rd']}")
    print(f"proxy: rank-corr(LOO Gini, attn Gini) = {corr['loo_gini_vs_attn_gini']}")
    print(f"\n=== H13 判定 ===\n{json.dumps(judgment, ensure_ascii=False, indent=2)}")
    print(f"\n書き出し: {out_dir}/setting_table.csv, exp13_summary.json")


if __name__ == "__main__":
    main()
