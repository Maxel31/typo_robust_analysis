#!/usr/bin/env python3
"""実験6-(i)〜(iii) 集計: 手法別 J_method@10 と ρ(J_method@10|R).

run_attribution_family.py が出力した clean / LXT-4 摂動の手法ランキングを
sample_id で対応付けて内的軸 J_method@10 (clean vs perturbed の CoT 語
ランキング Jaccard) を再計算し、アーカイブの analysis full_results.json の
サンプル別指標 (R = cot_metrics.rouge_l.f1, 参照 J_RC = cot_metrics.jaccard.top10)
と結合して Spearman ρ(J_method@10 | R) を求める。

「帰属手法を替えても内的軸の相関構造が保持されるか」(実験6 の主要検定) の
判定に使う。規約は scripts/analyze_loo_rankings.py (exp6-iv) と同一。

使用例 (CPU のみ):
  uv run python scripts/analyze_attribution_family.py \\
    --results_root results/attribution_family \\
    --archive_analysis_root /home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/analysis
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

DEFAULT_SETTINGS = [
    ("gemma-3-4b-it", "gsm8k"),
    ("gemma-3-4b-it", "mmlu"),
    ("Llama-3.2-3B-Instruct", "gsm8k"),
    ("Llama-3.2-3B-Instruct", "mmlu"),
    ("Mistral-7B-Instruct-v0.3", "gsm8k"),
    ("Mistral-7B-Instruct-v0.3", "mmlu"),
]
DEFAULT_METHODS = ["gxi", "ig", "rollout"]


def compute_method_jaccard_pairs(
    clean_entries: list[dict],
    perturbed_entries: list[dict],
    k: int = 10,
) -> list[dict]:
    """clean / perturbed の手法ランキングを sample_id で対応付けて Jaccard@k を計算する.

    run_attribution_family.py の results.json エントリ
    (sample_id / method_word_scores) をそのまま受け取れる。

    Returns:
        [{"sample_id": str, "method_jaccard": float}, ...] (clean 側の順序、
        片側にしか無いサンプルはスキップ)
    """
    from typo_cot.intervention.loo_scorer import loo_jaccard_topk

    perturbed_by_id = {e["sample_id"]: e for e in perturbed_entries}
    pairs: list[dict] = []
    for clean in clean_entries:
        pert = perturbed_by_id.get(clean["sample_id"])
        if pert is None:
            continue
        pairs.append(
            {
                "sample_id": clean["sample_id"],
                "method_jaccard": loo_jaccard_topk(
                    clean.get("method_word_scores", []),
                    pert.get("method_word_scores", []),
                    k=k,
                ),
            }
        )
    return pairs


def join_pairs_with_archive(
    pairs: list[dict], sample_results: list[dict]
) -> list[dict]:
    """手法 Jaccard ペアをアーカイブのサンプル別指標と sample_id で結合する.

    Returns:
        [{"sample_id", "j_method", "rouge_f1", "j_rc"}] — 両側に存在し
        rouge_f1 を持つサンプルのみ (pairs の順序を保持)。
    """
    by_id: dict[str, tuple[float | None, float | None]] = {}
    for sr in sample_results:
        cm = sr.get("cot_metrics") or {}
        rouge = (cm.get("rouge_l") or {}).get("f1")
        j_rc = (cm.get("jaccard") or {}).get("top10")
        by_id[sr["sample_id"]] = (rouge, j_rc)

    rows: list[dict] = []
    for p in pairs:
        hit = by_id.get(p["sample_id"])
        if hit is None or hit[0] is None:
            continue
        rouge, j_rc = hit
        rows.append(
            {
                "sample_id": p["sample_id"],
                "j_method": float(p["method_jaccard"]),
                "rouge_f1": float(rouge),
                "j_rc": float(j_rc) if j_rc is not None else None,
            }
        )
    return rows


def _spearman(x: list[float], y: list[float]) -> dict:
    """Spearman ρ (n<3 では計算しない)."""
    if len(x) < 3:
        return {"rho": None, "p": None, "n": len(x)}
    from typo_cot.analysis.metrics import spearman_correlation

    res = spearman_correlation(x, y)
    return {"rho": res["correlation"], "p": res["p_value"], "n": len(x)}


def _mean_vs_rc(entries: list[dict], k: int) -> float | None:
    vals = [
        e.get(f"vs_rc_jaccard_top{k}")
        for e in entries
        if e.get(f"vs_rc_jaccard_top{k}") is not None
    ]
    return statistics.mean(vals) if vals else None


def compute_setting_summary(
    clean_entries: list[dict],
    perturbed_entries: list[dict],
    sample_results: list[dict],
    k: int = 10,
) -> dict:
    """1設定 (model x benchmark x method) の J_method@k と ρ(J_method@k|R) を集計する."""
    pairs = compute_method_jaccard_pairs(clean_entries, perturbed_entries, k=k)
    rows = join_pairs_with_archive(pairs, sample_results)

    j_all = [p["method_jaccard"] for p in pairs]
    rc_rows = [r for r in rows if r["j_rc"] is not None]
    return {
        "n_pairs": len(pairs),
        "n_joined": len(rows),
        "mean_j_method": statistics.mean(j_all) if j_all else None,
        "median_j_method": statistics.median(j_all) if j_all else None,
        "mean_j_method_joined": (
            statistics.mean(r["j_method"] for r in rows) if rows else None
        ),
        "rho_j_method_vs_rouge": _spearman(
            [r["j_method"] for r in rows], [r["rouge_f1"] for r in rows]
        ),
        "rho_j_rc_vs_rouge": _spearman(
            [r["j_rc"] for r in rc_rows], [r["rouge_f1"] for r in rc_rows]
        ),
        "mean_j_rc_joined": (
            statistics.mean(r["j_rc"] for r in rc_rows) if rc_rows else None
        ),
        "mean_vs_rc_jaccard_clean": _mean_vs_rc(clean_entries, k),
        "mean_vs_rc_jaccard_lxt4": _mean_vs_rc(perturbed_entries, k),
    }


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate attribution-family rankings (exp6 i-iii)"
    )
    parser.add_argument("--results_root", type=str,
                        default="results/attribution_family")
    parser.add_argument("--archive_analysis_root", type=str, required=True,
                        help="アーカイブ outputs/analysis のルート (読み取り専用)")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--perturbation", type=str, default="k4_importance",
                        help="アーカイブ analysis の摂動条件ディレクトリ名")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.results_root)
    arch_root = Path(args.archive_analysis_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_path = (
        Path(args.output)
        if args.output
        else root / "aggregate_attribution_family.json"
    )

    settings_out: dict[str, dict] = {}
    for model_short, benchmark in DEFAULT_SETTINGS:
        full_results = None
        arch_path = (
            arch_root / benchmark / model_short / args.perturbation
            / "full_results.json"
        )
        if arch_path.exists():
            full_results = _read_json(arch_path)
        for method in methods:
            name = f"{model_short}_{benchmark}_{method}"
            clean_dir = root / f"{model_short}_{benchmark}_{method}_clean"
            pert_dir = root / f"{model_short}_{benchmark}_{method}_lxt4"
            if not (clean_dir / "results.json").exists() or not (
                pert_dir / "results.json"
            ).exists():
                settings_out[name] = {"status": "missing_runs"}
                continue
            if full_results is None:
                settings_out[name] = {"status": "missing_archive"}
                continue

            summary = compute_setting_summary(
                _read_json(clean_dir / "results.json"),
                _read_json(pert_dir / "results.json"),
                full_results.get("sample_results", []),
                k=args.k,
            )
            summary["status"] = "ok"
            settings_out[name] = summary

    out = {
        "experiment_info": {
            "analysis": "exp6 i-iii attribution-family aggregation",
            "methods": methods,
            "k": args.k,
            "perturbation": args.perturbation,
            "results_root": str(root),
            "archive_analysis_root": str(arch_root),
            "timestamp": datetime.now().isoformat(),
        },
        "settings": settings_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # コンソール表
    print(f"\n=== exp6 i-iii aggregation (k={args.k}) ===")
    header = (
        f"{'setting':<44} {'n':>4} {'vsRC(cl)':>8} {'J_mthd':>7} "
        f"{'rho(J_m|R)':>11} {'rho(J_RC|R)':>12}"
    )
    print(header)
    for name, s in settings_out.items():
        if s.get("status") != "ok":
            print(f"{name:<44} {s.get('status')}")
            continue
        vs_rc = s["mean_vs_rc_jaccard_clean"]
        rho_m = s["rho_j_method_vs_rouge"]["rho"]
        rho_rc = s["rho_j_rc_vs_rouge"]["rho"]
        print(
            f"{name:<44} {s['n_pairs']:>4} "
            f"{(f'{vs_rc:.4f}' if vs_rc is not None else 'n/a'):>8} "
            f"{s['mean_j_method']:>7.4f} "
            f"{(f'{rho_m:.4f}' if rho_m is not None else 'n/a'):>11} "
            f"{(f'{rho_rc:.4f}' if rho_rc is not None else 'n/a'):>12}"
        )
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
