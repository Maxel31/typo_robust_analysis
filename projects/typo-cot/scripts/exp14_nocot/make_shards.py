#!/usr/bin/env python3
"""実験14 no-CoT 本番キューのシャード一覧 (TSV) を生成する.

1 シャード = model × benchmark × condition。condition は
  clean      … baseline results.json の質問 (摂動なし)
  importance … LXT-4 (k4_importance) 摂動質問
  random     … Random-4 (k4_random) 摂動質問
大設定 (サンプル数 > SPLIT_THRESHOLD) は --start/--n で分割する。

データソース:
  clean: アーカイブ baseline/{short}_{bench}
  importance/random (既存5モデル): アーカイブ perturbed/{short}_{bench}_k4_{mode}
  importance/random (Qwen2.5-7B): exp-10-scope worktree perturbed/... (id はアーカイブ
    baseline と完全一致することを確認済み)
アーカイブ / exp-10-scope は読み取り専用。

出力 TSV 列: name  model  benchmark  condition  source_dir  start  n
start/n が "-" のときは全量。

使い方:
    python scripts/exp14_nocot/make_shards.py > scripts/exp14_nocot/shards_all.tsv
"""

import json
import sys
from pathlib import Path

ARCHIVE = Path("/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs")
EXP10 = Path(
    "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis"
    "/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs"
)

# (アーカイブ接頭辞, HuggingFace モデル名)。gemma-3-4b × gsm8k をスモークと同一設定
# として先頭に置く。
CORE_MODELS = [
    ("gemma-3-4b-it", "google/gemma-3-4b-it"),
    ("gemma-3-1b-it", "google/gemma-3-1b-it"),
    ("Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
]
QWEN = ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
BENCHMARKS = ["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa", "math"]

SPLIT_THRESHOLD = 1600


def count(source: Path) -> int | None:
    rj = source / "results.json"
    if not rj.exists():
        return None
    try:
        with open(rj) as f:
            return len(json.load(f))
    except Exception:
        return None


def clean_source(short: str, bench: str) -> Path:
    return ARCHIVE / "baseline" / f"{short}_{bench}"


def pert_source(short: str, bench: str, mode: str, is_qwen: bool) -> Path:
    name = f"{short}_{bench}_k4_{mode}"
    # Qwen は exp-10-scope が正典。既存5モデルはアーカイブ優先だが、一部
    # (Mistral math random 等) はアーカイブに無く exp-10-scope にのみ存在するため
    # アーカイブに results.json が無ければ exp-10-scope へフォールバックする。
    order = [EXP10, ARCHIVE] if is_qwen else [ARCHIVE, EXP10]
    for root in order:
        cand = root / "perturbed" / name
        if (cand / "results.json").exists():
            return cand
    return order[0] / "perturbed" / name  # 存在しない → emit で SKIP される


def emit(rows: list, short: str, hf: str, bench: str, condition: str, source: Path) -> None:
    n = count(source)
    if n is None:
        print(f"# SKIP (no results.json): {source}", file=sys.stderr)
        return
    name_base = f"{short}_{bench}_{condition}"
    if n > SPLIT_THRESHOLD:
        start = 0
        idx = 0
        while start < n:
            take = min(SPLIT_THRESHOLD, n - start)
            rows.append(
                (f"{name_base}__p{idx}", hf, bench, condition, str(source), str(start), str(take))
            )
            start += take
            idx += 1
    else:
        rows.append((name_base, hf, bench, condition, str(source), "-", "-"))


def main() -> None:
    rows: list = []
    models = CORE_MODELS + [QWEN]
    for short, hf in models:
        is_qwen = short == QWEN[0]
        for bench in BENCHMARKS:
            emit(rows, short, hf, bench, "clean", clean_source(short, bench))
            emit(rows, short, hf, bench, "importance", pert_source(short, bench, "importance", is_qwen))
            emit(rows, short, hf, bench, "random", pert_source(short, bench, "random", is_qwen))

    print("\t".join(["name", "model", "benchmark", "condition", "source_dir", "start", "n"]))
    for r in rows:
        print("\t".join(r))
    print(f"# total shards: {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
