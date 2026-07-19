#!/usr/bin/env python3
"""実験7 最終集計: 校正器3段 x 25設定の精度回復表と byte-identical サマリ.

入力:
  (a) アーカイブ baseline: {archive}/outputs/baseline/{model}_{bench}/summary.json
  (b) アーカイブ摂動:      {archive}/outputs/perturbed/{model}_{bench}_k4_importance/summary.json
  (c) 校正後評価生成:      results/prod/exp7/generation/{model}_{bench}_k4_{mode}/summary.json
  (d) 復元統計:            data/exp7/corrected/{model}_{bench}_k4_{mode}/restoration_stats.json

出力: --output に JSON、--output_md に Markdown 表 (dev notes 貼り付け用)。

使用例:
  uv run python scripts/exp7/aggregate_final_results.py \
    --output results/prod/exp7/analysis/final_summary.json \
    --output_md results/prod/exp7/analysis/final_summary.md
"""

import argparse
import json
from pathlib import Path

ARCHIVE = Path("/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs")
MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "Mistral-7B-Instruct-v0.3",
]
BENCHES = ["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa"]
MODES = ["spellfix", "neuralfix", "llmfix"]
CORRECTOR_LABEL = {
    "spellfix": "pyspell",
    "neuralfix": "T5-large-spell",
    "llmfix": "Qwen2.5-7B-Instruct",
}


def read_acc(summary_path: Path):
    if not summary_path.is_file():
        return None
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)
    m = s["overall_metrics"]
    return {"accuracy": m["accuracy"], "correct": m["total_correct"], "n": m["total_samples"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="実験7 最終集計")
    parser.add_argument("--generation_dir", default="results/prod/exp7/generation")
    parser.add_argument("--corrected_dir", default="data/exp7/corrected")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    gen = Path(args.generation_dir)
    cor = Path(args.corrected_dir)

    # --- (1) 精度回復表: 25 設定 x {clean, perturbed, 3 校正器} ---
    rows = []
    missing = []
    for m in MODELS:
        for b in BENCHES:
            row = {"model": m, "benchmark": b}
            row["clean"] = read_acc(ARCHIVE / "baseline" / f"{m}_{b}" / "summary.json")
            row["perturbed"] = read_acc(
                ARCHIVE / "perturbed" / f"{m}_{b}_k4_importance" / "summary.json"
            )
            for mode in MODES:
                row[mode] = read_acc(gen / f"{m}_{b}_k4_{mode}" / "summary.json")
                if row[mode] is None:
                    missing.append(f"{m}_{b}_k4_{mode}")
            rows.append(row)

    # --- (2) byte-identical 復元率 (fully_restored) の校正器別サマリ ---
    resto = {mode: [] for mode in MODES}
    for m in MODELS:
        for b in BENCHES:
            for mode in MODES:
                p = cor / f"{m}_{b}_k4_{mode}" / "restoration_stats.json"
                if not p.is_file():
                    continue
                with open(p, encoding="utf-8") as f:
                    st = json.load(f)
                resto[mode].append(
                    {
                        "model": m,
                        "benchmark": b,
                        "n_samples": st["aggregate"]["n_samples"],
                        "fully_restored": st["aggregate"]["fully_restored"],
                        "full_restoration_rate": st["rates"]["full_restoration_rate"],
                        "word_restoration_rate": st["rates"]["word_restoration_rate"],
                        "llm_parse_failures": st["aggregate"].get("llm_parse_failures", 0),
                    }
                )

    resto_summary = {}
    for mode in MODES:
        rs = resto[mode]
        n_total = sum(r["n_samples"] for r in rs)
        fr_total = sum(r["fully_restored"] for r in rs)
        resto_summary[mode] = {
            "corrector": CORRECTOR_LABEL[mode],
            "n_configs": len(rs),
            "pooled": {
                "n_samples": n_total,
                "fully_restored": fr_total,
                "byte_identical_rate": fr_total / n_total if n_total else None,
            },
            "macro_mean_byte_identical_rate": (
                sum(r["full_restoration_rate"] for r in rs) / len(rs) if rs else None
            ),
            "macro_mean_word_restoration_rate": (
                sum(r["word_restoration_rate"] for r in rs) / len(rs) if rs else None
            ),
            "min_byte_identical": (
                min(rs, key=lambda r: r["full_restoration_rate"]) if rs else None
            ),
            "max_byte_identical": (
                max(rs, key=lambda r: r["full_restoration_rate"]) if rs else None
            ),
            "total_llm_parse_failures": sum(r["llm_parse_failures"] for r in rs),
            "per_config": rs,
        }

    # --- (3) 校正器別の精度回復サマリ (マクロ平均) ---
    def macro(key):
        vals = [r[key]["accuracy"] for r in rows if r.get(key)]
        return sum(vals) / len(vals) if vals else None

    recovery_summary = {}
    for mode in MODES:
        pairs = [
            (r["clean"]["accuracy"], r["perturbed"]["accuracy"], r[mode]["accuracy"])
            for r in rows
            if r["clean"] and r["perturbed"] and r.get(mode)
        ]
        if pairs:
            rec = [
                (c_acc - p_acc, x_acc - p_acc, (x_acc - p_acc) / (c_acc - p_acc) if c_acc != p_acc else None)
                for c_acc, p_acc, x_acc in pairs
            ]
            ratios = [r[2] for r in rec if r[2] is not None]
            recovery_summary[mode] = {
                "corrector": CORRECTOR_LABEL[mode],
                "n_configs": len(pairs),
                "macro_mean_accuracy": sum(p[2] for p in pairs) / len(pairs),
                "macro_mean_recovered_points": sum(r[1] for r in rec) / len(rec),
                "macro_mean_recovery_ratio": sum(ratios) / len(ratios) if ratios else None,
            }

    out = {
        "n_configs": len(rows),
        "n_missing_corrected_runs": len(missing),
        "missing_corrected_runs": missing,
        "macro_mean_accuracy": {
            "clean": macro("clean"),
            "perturbed": macro("perturbed"),
            **{mode: macro(mode) for mode in MODES},
        },
        "recovery_summary": recovery_summary,
        "restoration_summary": resto_summary,
        "table": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # --- Markdown ---
    def fmt(cell):
        return f"{cell['accuracy']:.4f}" if cell else "—"

    lines = [
        "### 校正器3段 x 25設定 精度回復表 (accuracy)",
        "",
        "| モデル | ベンチ | clean | 摂動(k4) | spellfix | neuralfix | llmfix |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['benchmark']} | {fmt(r['clean'])} | {fmt(r['perturbed'])} | "
            f"{fmt(r.get('spellfix'))} | {fmt(r.get('neuralfix'))} | {fmt(r.get('llmfix'))} |"
        )
    mm = out["macro_mean_accuracy"]
    if all(mm.get(k) is not None for k in ["clean", "perturbed", *MODES]):
        lines.append(
            f"| **マクロ平均** | 25設定 | {mm['clean']:.4f} | {mm['perturbed']:.4f} | "
            f"{mm['spellfix']:.4f} | {mm['neuralfix']:.4f} | {mm['llmfix']:.4f} |"
        )
    lines += [
        "",
        "### 校正器別 回復サマリ",
        "",
        "| 校正器 | マクロ平均 acc | 回復幅 (pt, 対摂動) | 回復率 (対 clean-摂動 gap) |",
        "|---|---|---|---|",
    ]
    for mode in MODES:
        rs = recovery_summary.get(mode)
        if rs:
            lines.append(
                f"| {mode} ({rs['corrector']}) | {rs['macro_mean_accuracy']:.4f} | "
                f"+{rs['macro_mean_recovered_points'] * 100:.2f} | "
                f"{rs['macro_mean_recovery_ratio'] * 100:.1f}% |"
            )
    lines += [
        "",
        "### byte-identical 復元率 (fully_restored) 校正器別サマリ",
        "",
        "| 校正器 | pooled byte-identical | マクロ平均 | 語復元率 (マクロ平均) | 最小設定 | 最大設定 |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        s = resto_summary[mode]
        p = s["pooled"]
        mn, mx = s["min_byte_identical"], s["max_byte_identical"]
        if p["n_samples"]:
            lines.append(
                f"| {mode} ({s['corrector']}) | {p['fully_restored']}/{p['n_samples']} "
                f"({p['byte_identical_rate'] * 100:.1f}%) | {s['macro_mean_byte_identical_rate'] * 100:.1f}% | "
                f"{s['macro_mean_word_restoration_rate'] * 100:.1f}% | "
                f"{mn['model']}x{mn['benchmark']} {mn['full_restoration_rate'] * 100:.1f}% | "
                f"{mx['model']}x{mx['benchmark']} {mx['full_restoration_rate'] * 100:.1f}% |"
            )
    if missing:
        lines += ["", f"未完了ラン ({len(missing)}): " + ", ".join(missing)]

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
