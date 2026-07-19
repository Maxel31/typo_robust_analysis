#!/usr/bin/env python3
"""実験6-(iv) 集計: LOO 版 CoT:Jaccard@10 と ρ(J_LOO@10|R).

run_loo_scoring.py が出力した clean / LXT-4 摂動の LOO ランキングを
sample_id で対応付けて LOO 版 CoT:Jaccard@10 を計算し、アーカイブの
analysis full_results.json のサンプル別指標
(R = cot_metrics.rouge_l.f1, 参照 J_RC = cot_metrics.jaccard.top10)
と結合して Spearman ρ(J_LOO@10 | R) を求める。

内的軸 (CoT 重要語ランキングの摂動安定性) が帰属フリーの LOO でも
再構成できるか — R3 の leave-one-out 要求への最終回答 — の判定に使う。

使用例 (CPU のみ):
  uv run python scripts/analyze_loo_rankings.py \\
    --loo_root results/loo \\
    --archive_analysis_root /home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/analysis \\
    --mode occ
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


def join_pairs_with_archive(
    pairs: list[dict], sample_results: list[dict]
) -> list[dict]:
    """LOO Jaccard ペアをアーカイブのサンプル別指標と sample_id で結合する.

    Args:
        pairs: compute_loo_jaccard_pairs の出力 [{sample_id, loo_jaccard}]
        sample_results: アーカイブ full_results.json の sample_results
            (cot_metrics.rouge_l.f1 = R, cot_metrics.jaccard.top10 = J_RC)

    Returns:
        [{"sample_id", "j_loo", "rouge_f1", "j_rc"}] — 両側に存在し
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
                "j_loo": float(p["loo_jaccard"]),
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


def compute_setting_summary(
    clean_entries: list[dict],
    perturbed_entries: list[dict],
    sample_results: list[dict],
    k: int = 10,
) -> dict:
    """1設定 (model x benchmark) の LOO 版 Jaccard@k と ρ(J_LOO@k|R) を集計する."""
    from typo_cot.intervention.loo_scorer import compute_loo_jaccard_pairs

    pairs = compute_loo_jaccard_pairs(clean_entries, perturbed_entries, k=k)
    rows = join_pairs_with_archive(pairs, sample_results)

    j_loo_all = [p["loo_jaccard"] for p in pairs]
    rc_rows = [r for r in rows if r["j_rc"] is not None]
    return {
        "n_loo_pairs": len(pairs),
        "n_joined": len(rows),
        "mean_j_loo": statistics.mean(j_loo_all) if j_loo_all else None,
        "median_j_loo": statistics.median(j_loo_all) if j_loo_all else None,
        "mean_j_loo_joined": (
            statistics.mean(r["j_loo"] for r in rows) if rows else None
        ),
        "rho_j_loo_vs_rouge": _spearman(
            [r["j_loo"] for r in rows], [r["rouge_f1"] for r in rows]
        ),
        "rho_j_rc_vs_rouge": _spearman(
            [r["j_rc"] for r in rc_rows], [r["rouge_f1"] for r in rc_rows]
        ),
        "mean_j_rc_joined": (
            statistics.mean(r["j_rc"] for r in rc_rows) if rc_rows else None
        ),
    }


def archive_reference_rho(full_results: dict) -> dict | None:
    """アーカイブ全数での ρ(J_RC@10|R) 参照値 (correlations, group=all) を引く."""
    for c in full_results.get("correlations", []):
        if (
            c.get("variable1") == "cot_jaccard_top10"
            and c.get("variable2") == "cot_rouge_l_f1"
            and c.get("group_name") == "all"
        ):
            return {"rho": c.get("spearman_rho"), "p": c.get("spearman_p"),
                    "n": c.get("n")}
    return None


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate LOO rankings: LOO-Jaccard@k and rho(J_LOO|R) (exp6-iv)"
    )
    parser.add_argument("--loo_root", type=str, default="results/loo")
    parser.add_argument("--archive_analysis_root", type=str, required=True,
                        help="アーカイブ outputs/analysis のルート (読み取り専用)")
    parser.add_argument("--mode", type=str, default="occ", choices=["occ", "type"],
                        help="run_label の接尾辞 (clean_{mode} / lxt4_{mode})")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--perturbation", type=str, default="k4_importance",
                        help="アーカイブ analysis の摂動条件ディレクトリ名")
    parser.add_argument("--output", type=str, default=None,
                        help="集計 JSON の出力先 (省略時 {loo_root}/aggregate_loo_{mode}.json)")
    args = parser.parse_args()

    loo_root = Path(args.loo_root)
    arch_root = Path(args.archive_analysis_root)
    out_path = (
        Path(args.output)
        if args.output
        else loo_root / f"aggregate_loo_{args.mode}.json"
    )

    settings_out: dict[str, dict] = {}
    for model_short, benchmark in DEFAULT_SETTINGS:
        name = f"{model_short}_{benchmark}"
        clean_dir = loo_root / f"{name}_clean_{args.mode}"
        pert_dir = loo_root / f"{name}_lxt4_{args.mode}"
        if not (clean_dir / "results.json").exists() or not (
            pert_dir / "results.json"
        ).exists():
            settings_out[name] = {"status": "missing_runs"}
            continue

        clean_entries = _read_json(clean_dir / "results.json")
        pert_entries = _read_json(pert_dir / "results.json")
        full_results = _read_json(
            arch_root / benchmark / model_short / args.perturbation
            / "full_results.json"
        )

        summary = compute_setting_summary(
            clean_entries, pert_entries,
            full_results.get("sample_results", []), k=args.k,
        )
        summary["status"] = "ok"
        summary["archive_reference_rho_full_n"] = archive_reference_rho(full_results)
        # 各 run の LOO vs R_C Jaccard@10 (summary.json から転記)
        for side, d in [("clean", clean_dir), ("lxt4", pert_dir)]:
            try:
                s = _read_json(d / "summary.json")
                summary[f"{side}_mean_loo_vs_rc_jaccard"] = s["metrics"].get(
                    f"mean_loo_vs_rc_jaccard_top{args.k}"
                )
                summary[f"{side}_stats"] = s.get("stats")
            except FileNotFoundError:
                summary[f"{side}_mean_loo_vs_rc_jaccard"] = None
        settings_out[name] = summary

    out = {
        "experiment_info": {
            "analysis": "exp6-iv LOO ranking aggregation",
            "mode": args.mode,
            "k": args.k,
            "perturbation": args.perturbation,
            "loo_root": str(loo_root),
            "archive_analysis_root": str(arch_root),
            "timestamp": datetime.now().isoformat(),
        },
        "settings": settings_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # コンソール表
    print(f"\n=== exp6-iv LOO aggregation (mode={args.mode}, k={args.k}) ===")
    header = (
        f"{'setting':<38} {'n_pair':>6} {'J_LOO':>7} {'rho(J_LOO|R)':>13} "
        f"{'rho(J_RC|R)':>12} {'ref_rho(full)':>13}"
    )
    print(header)
    for name, s in settings_out.items():
        if s.get("status") != "ok":
            print(f"{name:<38} {s.get('status')}")
            continue
        rho_loo = s["rho_j_loo_vs_rouge"]["rho"]
        rho_rc = s["rho_j_rc_vs_rouge"]["rho"]
        ref = (s.get("archive_reference_rho_full_n") or {}).get("rho")
        print(
            f"{name:<38} {s['n_loo_pairs']:>6} "
            f"{s['mean_j_loo']:>7.4f} "
            f"{(f'{rho_loo:.4f}' if rho_loo is not None else 'n/a'):>13} "
            f"{(f'{rho_rc:.4f}' if rho_rc is not None else 'n/a'):>12} "
            f"{(f'{ref:.4f}' if ref is not None else 'n/a'):>13}"
        )
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
