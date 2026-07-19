#!/usr/bin/env python3
"""実験1+3 本番キューのシャード一覧 (TSV) を生成する.

1シャード = 1設定 (モデル×ベンチ) × 1摂動条件。大設定 (結合ペア > SPLIT_THRESHOLD)
は --start/--n でサンプル分割する。出力 TSV 列:
    name  model  benchmark  baseline_dir  perturbed_dir  start  n
start/n が "-" のときは全量。

拡張 (Qwen2.5-7B / R1蒸留 / MATH-500) は MODELS / BENCHMARKS に追記して再生成するか、
生成済み TSV に行を追記するだけでキューが拾う (queue_worker.sh はループ毎に再読込)。

使い方:
    python scripts/exp01_03/make_shards.py > scripts/exp01_03/shards_all.tsv
"""

import json
import sys
from pathlib import Path

ARCHIVE = Path("/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs")

# (アーカイブディレクトリ名の接頭辞, HuggingFace モデル名) — 実行順 = この順
MODELS = [
    ("gemma-3-4b-it", "google/gemma-3-4b-it"),  # 検証シャード (スモーク同一設定) を先頭に
    ("gemma-3-1b-it", "google/gemma-3-1b-it"),
    ("Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
]
BENCHMARKS = ["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa"]
PERTURBATIONS = ["k4_importance", "k4_random"]

# 1シャードの目安 20〜40 分 (GPU ロック保持のフェアネス)。
# gemma-3-4b × gsm8k スモーク実測 ≈0.6 s/sample → 2000 件 ≈ 20〜40 分圏内。
SPLIT_THRESHOLD = 2000


def joined_count(baseline_dir: Path, perturbed_dir: Path) -> int:
    def ids(p: Path) -> set:
        with open(p / "results.json") as f:
            d = json.load(f)
        recs = d["results"] if isinstance(d, dict) and "results" in d else d
        return {r["sample_id"] for r in recs}

    return len(ids(baseline_dir) & ids(perturbed_dir))


def main() -> None:
    rows = []
    for arc_model, hf_model in MODELS:
        for bench in BENCHMARKS:
            baseline = ARCHIVE / "baseline" / f"{arc_model}_{bench}"
            for pert in PERTURBATIONS:
                perturbed = ARCHIVE / "perturbed" / f"{arc_model}_{bench}_{pert}"
                if not baseline.is_dir() or not perturbed.is_dir():
                    print(f"# SKIP (archive missing): {arc_model}_{bench}_{pert}", file=sys.stderr)
                    continue
                n_pairs = joined_count(baseline, perturbed)
                base_name = f"{arc_model}_{bench}_{pert}"
                if n_pairs > SPLIT_THRESHOLD:
                    half = (n_pairs + 1) // 2
                    parts = [(0, half), (half, "-")]
                    for i, (start, n) in enumerate(parts):
                        rows.append(
                            (f"{base_name}__p{i}", hf_model, bench, baseline, perturbed, start, n)
                        )
                else:
                    rows.append((base_name, hf_model, bench, baseline, perturbed, "-", "-"))

    print("# name\tmodel\tbenchmark\tbaseline_dir\tperturbed_dir\tstart\tn")
    for r in rows:
        print("\t".join(str(x) for x in r))
    print(f"# total shards: {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
