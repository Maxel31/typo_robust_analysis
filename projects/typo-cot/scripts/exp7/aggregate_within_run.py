#!/usr/bin/env python3
"""実験7: within-run byte-identical flip 検証の集計.

within_run_flip.py の結果 (校正器×設定別) を表にまとめ、同じ byte-identical
サンプル集合でのクロスラン flip 率 (本番評価生成 vs アーカイブ baseline;
再現性ノイズフロアの参考値) と対比する。

使用例:
  uv run python scripts/exp7/aggregate_within_run.py \
    --output results/prod/exp7/within_run/within_run_summary.json \
    --output_md results/prod/exp7/within_run/within_run_summary.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

ARCHIVE_BASELINE = Path(
    "/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline"
)
CORRECTOR_LABEL = {
    "spellfix": "pyspell",
    "neuralfix": "T5-large-spell",
    "llmfix": "Qwen2.5-7B-Instruct",
}


def load_answers(results_path: Path) -> dict[str, str | None]:
    with open(results_path, encoding="utf-8") as f:
        return {r["sample_id"]: r["extracted_answer"] for r in json.load(f)}


def cross_run_flip(
    sample_ids: list[str],
    generation_path: Path,
    baseline_path: Path,
) -> dict | None:
    """同じ byte-identical 集合でのクロスラン flip (参考ノイズフロア)."""
    if not generation_path.is_file() or not baseline_path.is_file():
        return None
    gen = load_answers(generation_path)
    base = load_answers(baseline_path)
    common = [s for s in sample_ids if s in gen and s in base]
    if not common:
        return None
    n_flip = sum(gen[s] != base[s] for s in common)
    return {"n": len(common), "n_flip": n_flip, "flip_rate": n_flip / len(common)}


def main() -> None:
    parser = argparse.ArgumentParser(description="within-run flip 検証の集計")
    parser.add_argument("--within_run_dir", default="results/prod/exp7/within_run")
    parser.add_argument("--generation_dir", default="results/prod/exp7/generation")
    parser.add_argument("--archive_baseline", default=str(ARCHIVE_BASELINE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    wr_dir = Path(args.within_run_dir)
    gen_dir = Path(args.generation_dir)
    arch = Path(args.archive_baseline)

    rows = []
    pooled_n = pooled_flip = 0
    pooled_x_n = pooled_x_flip = 0
    for res_path in sorted(wr_dir.glob("*/within_run_results.json")):
        with open(res_path, encoding="utf-8") as f:
            d = json.load(f)
        cfg = d["config"]
        name = cfg["config_name"]
        model_short = cfg["model"].split("/")[-1]
        bench = cfg["benchmark"]
        agg = d["aggregate"]
        sel = d["selection_stats"]
        ids = [r["sample_id"] for r in d["records"]]
        xr = cross_run_flip(
            ids,
            gen_dir / name / "results.json",
            arch / f"{model_short}_{bench}" / "results.json",
        )
        rows.append(
            {
                "config": name,
                "model": model_short,
                "benchmark": bench,
                "corrector": CORRECTOR_LABEL.get(
                    cfg["corrector_mode"], cfg["corrector_mode"]
                ),
                "n_samples": sel["n_samples"],
                "n_byte_identical": sel["n_byte_identical"],
                "n_measured": agg["n"],
                "within_run_n_flip": agg["n_flip"],
                "within_run_flip_rate": agg["flip_rate"],
                "n_gen_identical": agg.get("n_gen_identical"),
                "flip_ids": agg["flip_ids"],
                "n_failed": len(d.get("failed_sample_ids", [])),
                "cross_run": xr,
            }
        )
        pooled_n += agg["n"]
        pooled_flip += agg["n_flip"]
        if xr:
            pooled_x_n += xr["n"]
            pooled_x_flip += xr["n_flip"]

    pooled = {
        "within_run": {
            "n": pooled_n,
            "n_flip": pooled_flip,
            "flip_rate": pooled_flip / pooled_n if pooled_n else None,
        },
        "cross_run_reference": {
            "n": pooled_x_n,
            "n_flip": pooled_x_flip,
            "flip_rate": pooled_x_flip / pooled_x_n if pooled_x_n else None,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "n_configs": len(rows),
                "pooled": pooled,
                "rows": rows,
            },
            f, ensure_ascii=False, indent=1,
        )

    md = [
        "| モデル | ベンチ | 校正器 | byte-identical n/N | within-run flip | "
        "クロスラン flip (参考) |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["model"], r["benchmark"], r["corrector"])):
        xr = r["cross_run"]
        xr_s = f"{xr['n_flip']}/{xr['n']} ({xr['flip_rate']:.1%})" if xr else "—"
        md.append(
            f"| {r['model']} | {r['benchmark']} | {r['corrector']} | "
            f"{r['n_byte_identical']}/{r['n_samples']} | "
            f"{r['within_run_n_flip']}/{r['n_measured']} "
            f"({r['within_run_flip_rate']:.1%}) | {xr_s} |"
        )
    md.append("")
    wr = pooled["within_run"]
    xr = pooled["cross_run_reference"]
    md.append(
        f"合計: within-run flip {wr['n_flip']}/{wr['n']} "
        f"({(wr['flip_rate'] or 0):.2%}) vs クロスラン参考 "
        f"{xr['n_flip']}/{xr['n']} ({(xr['flip_rate'] or 0):.2%})"
    )
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
