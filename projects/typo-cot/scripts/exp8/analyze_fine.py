#!/usr/bin/env python3
"""実験8-fine の集計・可視化・判定スクリプト.

run_patching_fine.py が各設定 (<model>_<benchmark>) 下の lxt4/ rnd4/ に出力した
per-pair 結果 JSON を読み、設定別に

    - 単層 denoising / 累積 / noising / sham の層プロファイル CSV
    - 相対深さ×回復率の重ね描き PNG (Fig.5 差替候補; 単層 + 累積)
    - H8f-1〜5 の事前登録判定 JSON (設定別 + モデル別プール + 総括)

を生成する。GPU 不要。集計は fine_analysis の純関数を使用。

例:
    uv run --package typo-cot python scripts/exp8/analyze_fine.py \
        --results-dir results/prod/exp8_fine \
        --out-dir analysis/exp8_fine
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from typo_cot.intervention.fine_analysis import (
    argmax_layer,
    collect_by_layer,
    judge_h8f1_peak_depth,
    judge_h8f2_plateau_vs_spike,
    judge_h8f3_cumulative_saturation,
    judge_h8f4_late_null,
    judge_h8f5_noising_sufficiency,
    summarize_by_layer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analyze_fine")

VAL_LAYERS = (14, 20, 26)
EARLY = list(range(12))
# モデル短名 → 表示色 (相対深さ重ね描き用)
MODEL_COLORS = {
    "gemma-3-4b-it": "#1b9e77",
    "Llama-3.2-3B-Instruct": "#d95f02",
    "Mistral-7B-Instruct-v0.3": "#7570b3",
}


def load_setting_cells(setting_dir: Path) -> tuple[list[dict], int | None, dict]:
    """設定ディレクトリ配下の per-pair JSON から全 cell を平坦化して返す.

    Returns:
        (cells, n_layers, stats) — cells は全ペアの cell を連結。
        stats に n_pairs / n_excluded / n_error を格納。
    """
    cells: list[dict] = []
    n_layers: int | None = None
    n_pairs = n_excluded = n_error = 0
    for cond in ("lxt4", "rnd4"):
        cond_dir = setting_dir / cond
        if not cond_dir.is_dir():
            continue
        for jf in sorted(cond_dir.glob("*.json")):
            try:
                with open(jf, encoding="utf-8") as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if "excluded" in payload:
                n_excluded += 1
                continue
            if "error" in payload or "cells" not in payload:
                n_error += 1
                continue
            n_pairs += 1
            if n_layers is None:
                n_layers = payload.get("n_layers")
            cells.extend(payload["cells"])
    return cells, n_layers, {"n_pairs": n_pairs, "n_excluded": n_excluded, "n_error": n_error}


def write_profile_csv(path: Path, summary: dict, n_layers: int) -> None:
    """{層 → {n,mean,ci_lo,ci_hi}} を層昇順 CSV に書く (相対深さ列付き)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["layer", "rel_depth", "n", "median", "median_lo", "median_hi", "mean", "ci_lo", "ci_hi"]
        )
        for layer in sorted(summary):
            s = summary[layer]
            rel = layer / n_layers if n_layers else ""
            w.writerow([
                layer, rel, s["n"],
                s.get("median"), s.get("median_lo"), s.get("median_hi"),
                s.get("mean"), s.get("ci_lo"), s.get("ci_hi"),
            ])


def write_semantic_csv(out_dir: Path, sem_single: dict[str, dict]) -> None:
    """設定別 semantic 単層プロファイルを CSV 出力 (typo 比較用)."""
    for name, summary in sem_single.items():
        # 総層数は不明でも rel_depth は近似で層/最大層。ここでは layer 昇順のみ出力。
        path = out_dir / name / "a3c_semantic_single_profile.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["layer", "n", "mean", "ci_lo", "ci_hi"])
            for layer in sorted(summary):
                s = summary[layer]
                w.writerow([layer, s["n"], s["mean"], s.get("ci_lo"), s.get("ci_hi")])


def _abs_max_mean(summary: dict, layers, stat: str = "median") -> float | None:
    """指定層のうち |median| の最大 (sham / other_span が ~0 かの確認用)."""
    vals = [
        abs(summary[li][stat])
        for li in layers
        if li in summary and summary[li].get(stat) is not None
    ]
    return max(vals) if vals else None


def summarize_setting(cells: list[dict], n_layers: int) -> dict:
    """1 設定の single/cumulative/noising/sham 要約 + flip 逆転 + 判定を作る."""
    single = summarize_by_layer(
        collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
    )
    cumulative = summarize_by_layer(
        collect_by_layer(cells, "cumulative", "clean_to_pert", "s2_kl_recovery")
    )
    noising = summarize_by_layer(
        collect_by_layer(cells, "noising", "pert_to_clean", "s2_kl_recovery")
    )
    sham = summarize_by_layer(
        collect_by_layer(cells, "sham_single", "clean_to_pert", "s2_kl_recovery")
    )
    flip_rev = summarize_by_layer(
        collect_by_layer(cells, "single", "clean_to_pert", "answer_matches_donor")
    )
    # A3 統制
    other_span = summarize_by_layer(
        collect_by_layer(cells, "other_span", "clean_to_pert", "s2_kl_recovery")
    )
    all_positions = summarize_by_layer(
        collect_by_layer(cells, "all_positions", "clean_to_pert", "s2_kl_recovery")
    )

    best = argmax_layer(single, restrict=EARLY)
    a3 = {
        "a_other_span_early_abs_max": _abs_max_mean(other_span, EARLY),
        "a_other_span_supported": (
            (_abs_max_mean(other_span, EARLY) is not None)
            and (_abs_max_mean(other_span, EARLY) < 0.2)
        ),
        "b_all_positions_min": _min_mean(all_positions),
        "b_all_positions_supported": (
            (_min_mean(all_positions) is not None) and (_min_mean(all_positions) > 0.8)
        ),
    }
    judgments = {
        "H8f-1": judge_h8f1_peak_depth(single, n_layers, restrict_early=EARLY),
        "H8f-2": (
            judge_h8f2_plateau_vs_spike(single, best) if best is not None else {"supported": None}
        ),
        "H8f-3": judge_h8f3_cumulative_saturation(single, cumulative, n_layers),
        "H8f-4": judge_h8f4_late_null(single, VAL_LAYERS),
        "H8f-5": (
            judge_h8f5_noising_sufficiency(noising, best) if best is not None else {"supported": None}
        ),
        "A3": a3,
        "best_early_layer": best,
        "sham_early_abs_max": _abs_max_mean(sham, EARLY),
    }
    return {
        "single": single,
        "cumulative": cumulative,
        "noising": noising,
        "sham": sham,
        "other_span": other_span,
        "all_positions": all_positions,
        "flip_reversal": flip_rev,
        "judgments": judgments,
    }


def _min_mean(summary: dict, stat: str = "median") -> float | None:
    """全層のうち median の最小 (all_positions が ~1 かの確認用)."""
    vals = [s[stat] for s in summary.values() if s.get(stat) is not None]
    return min(vals) if vals else None


def profile_isomorphism(typo_single: dict, sem_single: dict, layers=EARLY, stat="median") -> dict:
    """typo と semantic の単層プロファイルの同形性 (Pearson 相関 + ピーク一致; median)."""
    import math

    xs, ys = [], []
    for li in layers:
        if li in typo_single and li in sem_single:
            a = typo_single[li].get(stat)
            b = sem_single[li].get(stat)
            if a is not None and b is not None:
                xs.append(a)
                ys.append(b)
    if len(xs) < 3:
        return {"pearson": None, "peak_match": None, "isomorphic": None, "n": len(xs)}
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    pearson = cov / (vx * vy) if vx > 0 and vy > 0 else None
    peak_t = argmax_layer(typo_single, restrict=layers)
    peak_s = argmax_layer(sem_single, restrict=layers)
    peak_match = (peak_t is not None and peak_s is not None and abs(peak_t - peak_s) <= 1)
    iso = bool(pearson is not None and pearson > 0.8 and peak_match)
    return {
        "pearson": pearson,
        "peak_typo": peak_t,
        "peak_semantic": peak_s,
        "peak_match": peak_match,
        "isomorphic": iso,
        "n": len(xs),
        "interpretation": (
            "read-out generic; typo-specific effect in LXT/Random magnitude"
            if iso
            else "typo-specific depth localization"
        ),
    }


def make_overlay_png(
    per_model_single: dict[str, dict],
    per_model_cumulative: dict[str, dict],
    per_model_nlayers: dict[str, int],
    out_path: Path,
) -> bool:
    """相対深さ×回復率の重ね描き (単層 + 累積) PNG を生成する."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        logger.warning("matplotlib 不可のため PNG をスキップ: %s", e)
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        ("Single-layer denoising", per_model_single),
        ("Cumulative (0..l)", per_model_cumulative),
    ]
    for ax, (title, per_model) in zip(axes, panels):
        for model, summary in per_model.items():
            L = per_model_nlayers.get(model)
            if not L:
                continue
            pts = sorted(
                (li / L, summary[li]["median"])
                for li in summary
                if summary[li].get("median") is not None
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs, ys, marker="o", ms=3, lw=1.5,
                color=MODEL_COLORS.get(model, None), label=model,
            )
        ax.axvspan(0, 0.2, color="grey", alpha=0.10, label="li/L < 0.2")
        ax.axhline(0.0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("relative depth  li / L")
        ax.set_ylabel("S2 KL recovery (median)")
        ax.set_title(title)
        ax.set_xlim(0, 1)
    # 重複ラベル除去
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict = {}
    for h, la in zip(handles, labels):
        seen.setdefault(la, h)
    axes[0].legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")
    fig.suptitle("Exp 8-fine: injection-layer profile (S2 KL recovery vs relative depth)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def infer_model(setting_name: str) -> str:
    """設定名 (<model>_<benchmark>) からモデル短名を推定."""
    for m in MODEL_COLORS:
        if setting_name.startswith(m):
            return m
    return setting_name.rsplit("_", 1)[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, help="設定ディレクトリを含む結果ルート")
    p.add_argument("--out-dir", required=True, help="解析出力先")
    p.add_argument(
        "--semantic-results-dir", default=None,
        help="A3(c) 意味置換 run の結果ルート (typo との深さプロファイル比較に使用)",
    )
    return p.parse_args()


def load_semantic_single(results_root: Path) -> dict[str, dict]:
    """意味置換 run の設定別 single (kind='semantic') プロファイル要約を返す."""
    out: dict[str, dict] = {}
    if not results_root or not results_root.is_dir():
        return out
    for sd in sorted(results_root.iterdir()):
        if not sd.is_dir() or not ((sd / "lxt4").is_dir() or (sd / "rnd4").is_dir()):
            continue
        cells, _, _ = load_setting_cells(sd)
        out[sd.name] = summarize_by_layer(
            collect_by_layer(cells, "semantic", "clean_to_pert", "s2_kl_recovery")
        )
    return out


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setting_dirs = sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and ((d / "lxt4").is_dir() or (d / "rnd4").is_dir())
    )
    logger.info("設定 %d 件を検出: %s", len(setting_dirs), [d.name for d in setting_dirs])

    all_judgments: dict[str, dict] = {}
    per_setting_single: dict[str, dict] = {}
    per_model_single: dict[str, dict] = {}
    per_model_cumulative: dict[str, dict] = {}
    per_model_nlayers: dict[str, int] = {}
    model_cells: dict[str, list[dict]] = defaultdict(list)
    model_nlayers: dict[str, int] = {}

    for sd in setting_dirs:
        cells, n_layers, stats = load_setting_cells(sd)
        if not cells or not n_layers:
            logger.warning("%s: 有効な cell なし (%s)", sd.name, stats)
            continue
        res = summarize_setting(cells, n_layers)
        setting_out = out_dir / sd.name
        write_profile_csv(setting_out / "single_profile.csv", res["single"], n_layers)
        write_profile_csv(setting_out / "cumulative_profile.csv", res["cumulative"], n_layers)
        write_profile_csv(setting_out / "noising_profile.csv", res["noising"], n_layers)
        write_profile_csv(setting_out / "sham_profile.csv", res["sham"], n_layers)
        write_profile_csv(setting_out / "flip_reversal_profile.csv", res["flip_reversal"], n_layers)
        write_profile_csv(setting_out / "a3_other_span_profile.csv", res["other_span"], n_layers)
        write_profile_csv(setting_out / "a3_all_positions_profile.csv", res["all_positions"], n_layers)
        all_judgments[sd.name] = {"n_layers": n_layers, "stats": stats, **res["judgments"]}
        per_setting_single[sd.name] = res["single"]
        logger.info(
            "%s: n_pairs=%d best=%s H8f-1=%s H8f-3=%s",
            sd.name, stats["n_pairs"], res["judgments"]["best_early_layer"],
            res["judgments"]["H8f-1"].get("supported"),
            res["judgments"]["H8f-3"].get("supported"),
        )
        model = infer_model(sd.name)
        model_cells[model].extend(cells)
        model_nlayers[model] = n_layers

    model_judgments: dict[str, dict] = {}
    for model, cells in model_cells.items():
        n_layers = model_nlayers[model]
        res = summarize_setting(cells, n_layers)
        write_profile_csv(out_dir / f"pooled_{model}" / "single_profile.csv", res["single"], n_layers)
        write_profile_csv(
            out_dir / f"pooled_{model}" / "cumulative_profile.csv", res["cumulative"], n_layers
        )
        write_profile_csv(out_dir / f"pooled_{model}" / "noising_profile.csv", res["noising"], n_layers)
        write_profile_csv(out_dir / f"pooled_{model}" / "sham_profile.csv", res["sham"], n_layers)
        model_judgments[model] = {"n_layers": n_layers, **res["judgments"]}
        per_model_single[model] = res["single"]
        per_model_cumulative[model] = res["cumulative"]
        per_model_nlayers[model] = n_layers

    png_ok = make_overlay_png(
        per_model_single, per_model_cumulative, per_model_nlayers,
        out_dir / "fig5_relative_depth_overlay.png",
    )

    # A3(c) 意味置換プロファイルとの同形性比較 (設定別)
    semantic_compare: dict[str, dict] = {}
    if args.semantic_results_dir:
        sem_single = load_semantic_single(Path(args.semantic_results_dir))
        for name, j in all_judgments.items():
            if name in sem_single and name in all_judgments:
                typo_single = per_setting_single.get(name)
                if typo_single is not None:
                    cmp = profile_isomorphism(typo_single, sem_single[name])
                    semantic_compare[name] = cmp
                    j["A3"]["c_semantic"] = cmp
        write_semantic_csv(out_dir, sem_single)

    overall = {}
    for h in ("H8f-1", "H8f-2", "H8f-3", "H8f-4", "H8f-5"):
        n_sup = sum(1 for j in all_judgments.values() if j.get(h, {}).get("supported") is True)
        n_tot = sum(1 for j in all_judgments.values() if j.get(h, {}).get("supported") is not None)
        overall[h] = {"supported": n_sup, "evaluated": n_tot}
    overall["A3"] = {
        "a_other_span_supported": sum(
            1 for j in all_judgments.values() if j.get("A3", {}).get("a_other_span_supported") is True
        ),
        "b_all_positions_supported": sum(
            1 for j in all_judgments.values() if j.get("A3", {}).get("b_all_positions_supported") is True
        ),
        "c_semantic_isomorphic": sum(
            1 for c in semantic_compare.values() if c.get("isomorphic") is True
        ),
        "c_semantic_evaluated": len(semantic_compare),
        "evaluated": len(all_judgments),
    }

    judgment = {
        "experiment": "exp8_fine",
        "settings": all_judgments,
        "pooled_by_model": model_judgments,
        "overall_supported_over_settings": overall,
        "semantic_compare": semantic_compare,
        "fig5_png": str(out_dir / "fig5_relative_depth_overlay.png") if png_ok else None,
    }
    with open(out_dir / "judgment.json", "w", encoding="utf-8") as f:
        json.dump(judgment, f, ensure_ascii=False, indent=2, default=str)
    logger.info("総括 (設定横断 supported): %s", json.dumps(overall, ensure_ascii=False))
    logger.info("出力: %s", out_dir / "judgment.json")


if __name__ == "__main__":
    main()
