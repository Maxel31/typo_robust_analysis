#!/usr/bin/env python3
"""実験10④: A/B 比較表の作成 (A=LXT-4合成typo vs B=自然typo, 標的語固定).

入力:
- baseline: アーカイブ outputs/baseline/gemma-3-4b-it_{bench}/results.json
- A側:     アーカイブ outputs/perturbed/gemma-3-4b-it_{bench}_k4_importance/results.json
- B側:     outputs/perturbed/gemma-3-4b-it_{bench}_k4_natural/results.json (本実験)

指標 (baseline/A/B の共通 sample_id 上で計算):
- 精度 (baseline, A, B) と Δacc
- flip率 = P(摂動後不正解 | baseline正解)  ※主指標
- 回復率 = P(摂動後正解 | baseline不正解)
- 回答変化率 = extracted_answer が baseline から変わった割合
- A vs B の flip の対応比較 (McNemar 検定, 連続性補正付き)
- flip 集合の Jaccard (A/B が同じサンプルを flip させるか)
- 編集操作の実現分布 (A: 合成3種 / B: 自然4種)

注: 内的軸相関 (LRP重要度の Jaccard) は B 側で AttnLRP を省略したため対象外。

実行 (CPU):
  uv run --no-sync python scripts/exp10_natural_typo/compare_ab.py
"""

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARKS = ["gsm8k", "mmlu"]
MODEL_SHORT = "gemma-3-4b-it"
ARCHIVE = "/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs"


def load_rows(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["sample_id"]: r for r in json.load(f)}


def mcnemar_p(n10: int, n01: int) -> float:
    """McNemar 検定 (連続性補正付き χ², df=1) の p 値."""
    n = n10 + n01
    if n == 0:
        return 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2))


def analyze_benchmark(bench: str, b_results_dir: Path) -> dict | None:
    base_path = Path(ARCHIVE) / "baseline" / f"{MODEL_SHORT}_{bench}" / "results.json"
    a_path = (
        Path(ARCHIVE) / "perturbed" / f"{MODEL_SHORT}_{bench}_k4_importance" / "results.json"
    )
    b_path = b_results_dir / f"{MODEL_SHORT}_{bench}_k4_natural" / "results.json"
    if not b_path.exists():
        logger.warning(f"[{bench}] B側の結果がまだありません: {b_path}")
        return None

    base = load_rows(base_path)
    side_a = load_rows(a_path)
    side_b = load_rows(b_path)
    common = sorted(set(base) & set(side_a) & set(side_b))
    logger.info(
        f"[{bench}] 共通サンプル {len(common)} 件 "
        f"(baseline {len(base)} / A {len(side_a)} / B {len(side_b)})"
    )

    def acc(rows: dict[str, dict]) -> float:
        return sum(1 for s in common if rows[s].get("is_correct")) / len(common)

    base_correct = [s for s in common if base[s].get("is_correct")]
    base_incorrect = [s for s in common if not base[s].get("is_correct")]

    def flips(rows: dict[str, dict]) -> set[str]:
        """baseline正解 → 摂動後不正解のサンプル集合."""
        return {s for s in base_correct if not rows[s].get("is_correct")}

    def recoveries(rows: dict[str, dict]) -> set[str]:
        return {s for s in base_incorrect if rows[s].get("is_correct")}

    def answer_changed(rows: dict[str, dict]) -> int:
        n = 0
        for s in common:
            if str(rows[s].get("extracted_answer")) != str(base[s].get("extracted_answer")):
                n += 1
        return n

    flips_a, flips_b = flips(side_a), flips(side_b)
    n10 = len(flips_a - flips_b)  # A のみ flip
    n01 = len(flips_b - flips_a)  # B のみ flip
    n11 = len(flips_a & flips_b)
    union = len(flips_a | flips_b)

    def op_dist(rows: dict[str, dict]) -> dict[str, float]:
        counter = Counter(
            pt["perturbation_type"]
            for s in common
            for pt in rows[s].get("perturbed_tokens", [])
        )
        total = sum(counter.values())
        return {k: round(v / total, 4) for k, v in sorted(counter.items())} if total else {}

    return {
        "benchmark": bench,
        "n_common": len(common),
        "n_base_correct": len(base_correct),
        "accuracy": {
            "baseline": round(acc(base), 4),
            "A_synthetic": round(acc(side_a), 4),
            "B_natural": round(acc(side_b), 4),
        },
        "delta_accuracy": {
            "A_synthetic": round(acc(side_a) - acc(base), 4),
            "B_natural": round(acc(side_b) - acc(base), 4),
        },
        "flip_rate_correct_to_incorrect": {
            "A_synthetic": round(len(flips_a) / len(base_correct), 4),
            "B_natural": round(len(flips_b) / len(base_correct), 4),
        },
        "recovery_rate_incorrect_to_correct": {
            "A_synthetic": round(len(recoveries(side_a)) / max(len(base_incorrect), 1), 4),
            "B_natural": round(len(recoveries(side_b)) / max(len(base_incorrect), 1), 4),
        },
        "answer_change_rate": {
            "A_synthetic": round(answer_changed(side_a) / len(common), 4),
            "B_natural": round(answer_changed(side_b) / len(common), 4),
        },
        "flip_agreement": {
            "n_flip_A_only": n10,
            "n_flip_B_only": n01,
            "n_flip_both": n11,
            "jaccard_flip_sets": round(n11 / union, 4) if union else None,
            "mcnemar_p": round(mcnemar_p(n10, n01), 6),
        },
        "op_distribution": {
            "A_synthetic": op_dist(side_a),
            "B_natural": op_dist(side_b),
        },
    }


def to_markdown(results: list[dict]) -> str:
    lines = [
        "# 実験10④ A/B 比較: 合成typo (LXT-4) vs 自然typo (GitHub Typo Corpus 分布)",
        "",
        "モデル: gemma-3-4b-it / 標的語は A/B で同一 (LXT-4 の標的トークンを固定) / k=4",
        "",
        "| 指標 | "
        + " | ".join(f"{r['benchmark']} A(合成) | {r['benchmark']} B(自然)" for r in results)
        + " |",
        "|---|" + "---|" * (2 * len(results)),
    ]

    def row(label: str, getter) -> str:
        cells = []
        for r in results:
            a, b = getter(r)
            cells.append(str(a))
            cells.append(str(b))
        return f"| {label} | " + " | ".join(cells) + " |"

    lines.append(
        row("精度 (baseline)", lambda r: (r["accuracy"]["baseline"], r["accuracy"]["baseline"]))
    )
    lines.append(
        row("精度 (摂動後)", lambda r: (r["accuracy"]["A_synthetic"], r["accuracy"]["B_natural"]))
    )
    lines.append(
        row("Δ精度", lambda r: (r["delta_accuracy"]["A_synthetic"], r["delta_accuracy"]["B_natural"]))
    )
    lines.append(
        row(
            "flip率 (正→誤)",
            lambda r: (
                r["flip_rate_correct_to_incorrect"]["A_synthetic"],
                r["flip_rate_correct_to_incorrect"]["B_natural"],
            ),
        )
    )
    lines.append(
        row(
            "回答変化率",
            lambda r: (r["answer_change_rate"]["A_synthetic"], r["answer_change_rate"]["B_natural"]),
        )
    )
    lines.append("")
    for r in results:
        fa = r["flip_agreement"]
        lines.append(
            f"- **{r['benchmark']}**: flip一致 Jaccard={fa['jaccard_flip_sets']}, "
            f"Aのみflip={fa['n_flip_A_only']}, Bのみflip={fa['n_flip_B_only']}, "
            f"両方flip={fa['n_flip_both']}, McNemar p={fa['mcnemar_p']} "
            f"(n_correct={r['n_base_correct']})"
        )
        lines.append(
            f"  - 操作分布 A: {r['op_distribution']['A_synthetic']} / "
            f"B: {r['op_distribution']['B_natural']}"
        )
    lines.append("")
    lines.append("注: 内的軸相関 (LRP重要度 Jaccard) は B側で AttnLRP を省略したため対象外。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b_results_dir", default="./outputs/perturbed")
    parser.add_argument("--output_dir", default="./analysis/exp10_natural_typo")
    args = parser.parse_args()

    results = []
    for bench in BENCHMARKS:
        r = analyze_benchmark(bench, Path(args.b_results_dir))
        if r is not None:
            results.append(r)
    if not results:
        logger.error("比較可能なベンチマークがありません")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ab_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    md = to_markdown(results)
    with open(out_dir / "ab_comparison.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    logger.info(f"保存: {out_dir}/ab_comparison.json, ab_comparison.md")
    print(md)


if __name__ == "__main__":
    main()
