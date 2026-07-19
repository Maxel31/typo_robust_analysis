#!/usr/bin/env python3
"""実験13: LOO 重要度分布の集中度 (M3) を LOO results.json から算出する.

run_loo_scoring.py が出力した LOO result ディレクトリ (results.json に
loo_word_scores を含む) を読み、各サンプルの Gini係数 / top-1シェア /
top-4シェア / 有効語数を計算し、設定レベルに集計する。GPU 不要。

入力 results.json スキーマ (使用フィールドのみ):
    [{"sample_id": str, "loo_word_scores": [{"word": str, "score": float}, ...]}, ...]
出力: {out_dir}/loo/{setting}.json (設定ごと・per_sample 付き) と
      {out_dir}/loo_concentration.json (全設定集計)。日付フィールドなし。
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from typo_cot.analysis.concentration import loo_sample_concentration


def _clean(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _mean(xs):
    xs = _clean(xs)
    return statistics.mean(xs) if xs else None


def _median(xs):
    xs = _clean(xs)
    return statistics.median(xs) if xs else None


def compute_setting(results_path: Path, clip_negative: bool = True) -> dict:
    """1設定の LOO results.json → 設定レベル集中度集計 + 各サンプル値."""
    with open(results_path, encoding="utf-8") as f:
        entries = json.load(f)
    per_sample = []
    for e in entries:
        ws = e.get("loo_word_scores") or []
        conc = loo_sample_concentration(ws, clip_negative=clip_negative)
        per_sample.append(
            {
                "sample_id": e.get("sample_id"),
                "n_words": conc["n_words"],
                "gini": conc["gini"],
                "top1_share": conc["top1_share"],
                "top4_share": conc["top4_share"],
                "effective_count": conc["effective_count"],
            }
        )
    ginis = [s["gini"] for s in per_sample]
    return {
        "n_samples": len(per_sample),
        "clip_negative": clip_negative,
        "aggregates": {
            "mean_gini": _mean(ginis),
            "median_gini": _median(ginis),
            "mean_top1_share": _mean([s["top1_share"] for s in per_sample]),
            "median_top1_share": _median([s["top1_share"] for s in per_sample]),
            "mean_top4_share": _mean([s["top4_share"] for s in per_sample]),
            "mean_effective_count": _mean([s["effective_count"] for s in per_sample]),
            "mean_n_words": _mean([s["n_words"] for s in per_sample]),
        },
        "per_sample": per_sample,
    }


def parse_setting_name(dirname: str) -> tuple[str, str]:
    """'{model}_{benchmark}_clean_occ' → (model, benchmark)."""
    stem = dirname
    for suffix in ("_clean_occ", "_clean_type", "_lxt4_occ", "_lxt4_type"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    model, _, benchmark = stem.rpartition("_")
    return model, benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="LOO concentration (exp13 M3)")
    ap.add_argument("--loo_dir", type=str, default="results/loo",
                    help="LOO result ディレクトリ群の親")
    ap.add_argument("--pattern", type=str, default="*_clean_occ",
                    help="対象サブディレクトリの glob")
    ap.add_argument("--out_dir", type=str,
                    default="analysis/exp13_readout_concentration")
    ap.add_argument("--no_clip_negative", action="store_true",
                    help="負のLOOスコアをクリップしない (感度分析)")
    args = ap.parse_args()

    loo_dir = Path(args.loo_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "loo").mkdir(parents=True, exist_ok=True)
    clip = not args.no_clip_negative

    settings = sorted(
        d for d in loo_dir.glob(args.pattern)
        if d.is_dir() and (d / "results.json").exists()
    )
    overview = {}
    for d in settings:
        model, benchmark = parse_setting_name(d.name)
        res = compute_setting(d / "results.json", clip_negative=clip)
        res["model"] = model
        res["benchmark"] = benchmark
        res["setting"] = d.name
        with open(out_dir / "loo" / f"{d.name}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        overview[d.name] = {
            "model": model,
            "benchmark": benchmark,
            "n_samples": res["n_samples"],
            **res["aggregates"],
        }
        agg = res["aggregates"]
        print(f"{d.name:48s} n={res['n_samples']:4d} "
              f"gini={agg['mean_gini']:.4f} top1={agg['mean_top1_share']:.4f} "
              f"eff={agg['mean_effective_count']:.2f} nw={agg['mean_n_words']:.1f}")

    with open(out_dir / "loo_concentration.json", "w", encoding="utf-8") as f:
        json.dump({"clip_negative": clip, "settings": overview}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n書き出し: {out_dir/'loo_concentration.json'} ({len(overview)} settings)")


if __name__ == "__main__":
    main()
