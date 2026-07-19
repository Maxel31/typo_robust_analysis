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

    # ---- family 平均 (Gini と 削除RD) ----
    def _fam_mean(key):
        out = {}
        for fam in ("llama", "gemma", "mistral"):
            vs = [r[key] for r in rows if r["family"] == fam and r[key] is not None]
            out[fam] = (sum(vs) / len(vs)) if vs else None
        return out

    def _order(d):
        return sorted((f for f in d if d[f] is not None),
                      key=lambda f: d[f], reverse=True)

    def _is_llama_gemma_mistral(d):
        return (d["llama"] is not None and d["gemma"] is not None
                and d["mistral"] is not None
                and d["llama"] > d["gemma"] > d["mistral"])

    fam_gini = _fam_mean("loo_gini")
    fam_rd_content = _fam_mean("rd_content_k4")
    fam_rd_all = _fam_mean("rd_all_k4")
    fam_rank_ok = _is_llama_gemma_mistral(fam_gini)

    # ---- H13 判定 (scope-aware) ----
    rho_content = corr["loo_gini_vs_deletion_rd"]["rho"]  # 全語Gini vs RD_content
    rho_all = corr["loo_gini_vs_rd_all"]["rho"]           # 全語Gini vs RD_all (scope一致)
    n_loo_settings = sum(1 for r in rows if r["loo_gini"] is not None)
    judgment = {
        # 事前登録の文字通りの判定 (削除RD = content 層, 二重乖離の介入側順位)
        "preregistered_literal": {
            "rd_scope": "content",
            "family_gini_order": _order(fam_gini),
            "family_gini": fam_gini,
            "family_rank_llama_gt_gemma_gt_mistral": fam_rank_ok,
            "rankcorr_gini_vs_rd_content": rho_content,
            "rankcorr_threshold_met (>=0.7)": bool(
                rho_content is not None and rho_content >= 0.7),
            "H13_supported": bool(
                fam_rank_ok and rho_content is not None and rho_content >= 0.7),
        },
        # scope 一致の探索的判定 (全語Gini ↔ 全語削除RD)
        "scope_matched_exploratory": {
            "rd_scope": "all (unrestricted)",
            "family_rd_all_order": _order(fam_rd_all),
            "family_rd_all": fam_rd_all,
            "family_rd_all_llama_gt_gemma_gt_mistral": _is_llama_gemma_mistral(fam_rd_all),
            "rankcorr_gini_vs_rd_all": rho_all,
            "rankcorr_threshold_met (>=0.7)": bool(rho_all is not None and rho_all >= 0.7),
        },
        "family_rd_content": fam_rd_content,
        "family_rd_content_order": _order(fam_rd_content),
        "n_loo_settings_available": n_loo_settings,
        "interpretation": (
            "全語 LOO Gini は scope 一致の RD_all を強く予測 (rho={:.2f}) するが、"
            "content 層 RD_content とは逆符号 (rho={:.2f})。→ 集中度が numeric/機能語に"
            "由来する場合 (Gemma gsm8k 等) は content 削除に効かない。Gini family 順位は"
            "Gemma≈Llama>Mistral で、事前登録の Llama>Gemma>Mistral とは Gemma/Llama が"
            "近接・逆転。Mistral は観察的集中 (LOO/内容語質量) は Llama と同程度でも "
            "RD_content が桁違いに低い = 因果読み出しが冗長/分散 (削除に強い)。"
        ).format(
            rho_all if rho_all is not None else float("nan"),
            rho_content if rho_content is not None else float("nan"),
        ),
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
    print(f"\nfamily Gini: { {k: round(v,3) if v else v for k,v in fam_gini.items()} }"
          f"  order={_order(fam_gini)}")
    print(f"family RD_content order={_order(fam_rd_content)}; "
          f"RD_all order={_order(fam_rd_all)}")
    print(f"rank-corr(LOO Gini(全語), RD_content) = "
          f"{corr['loo_gini_vs_deletion_rd']}")
    print(f"rank-corr(LOO Gini(全語), RD_all[scope一致]) = "
          f"{corr['loo_gini_vs_rd_all']}")
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
