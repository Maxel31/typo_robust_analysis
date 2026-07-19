#!/usr/bin/env python3
"""実験2 本番キューのシャード一覧 (TSV) を生成する.

P1〜P4 (dev notes §6 + 2026-07-15 両建て更新) の全設定を
results/prod/exp2/queue/shards_active.tsv に書き出す。行は追記で拡張可能
(queue_worker.sh はループ毎に再読込する)。

TSV 列: name <TAB> script <TAB> done_dir <TAB> args
  - name: シャード名 (claim / failed / ログのキー)
  - script: 実行する CLI (project 相対)
  - done_dir: summary.json が生成される出力ディレクトリ (project 相対、冪等スキップ判定)
  - args: CLI 引数 (スペース区切り。値にスペースを含めないこと)

R_C ソース: アーカイブ既定 R_C (`_cot.pt`)。実験4の fixed-target 出力は
摂動側 CoT のランキングであり clean CoT の編集には使えない (dev notes §6.1)。
clean 正解サンプル (主分析母集団) では既定 R_C ≡ fixed-target R_C (標的=自答=正答)。
"""

import argparse
from pathlib import Path

ARCHIVE = "/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs"

M5 = [
    ("google/gemma-3-1b-it", "gemma-3-1b-it"),
    ("google/gemma-3-4b-it", "gemma-3-4b-it"),
    ("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
]
M3 = [m for m in M5 if m[1] in
      ("gemma-3-4b-it", "Llama-3.2-3B-Instruct", "Mistral-7B-Instruct-v0.3")]
B5 = ["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa"]
B2 = ["gsm8k", "mmlu"]

DELETION = "scripts/exp2/run_target_deletion.py"
RECOVERY = "scripts/exp2/run_recovery_curve.py"
OUT = "results/prod/exp2"


def deletion_row(hf: str, ms: str, bench: str, preset: str, label: str,
                 n: int, extra: str = "") -> tuple[str, str, str, str]:
    name = f"{preset}_{ms}_{bench}"
    args = (
        f"--baseline_dir {ARCHIVE}/baseline/{ms}_{bench} "
        f"--model {hf} --benchmark {bench} --arms {preset} "
        f"--clean_correct_only --n {n} --rc_source cot_pt --batch_size 8 "
        f"--output_dir {OUT} --run_label {label}"
    )
    if extra:
        args += f" {extra}"
    return (name, DELETION, f"{OUT}/{ms}_{bench}_{label}", args.strip())


def recovery_row(hf: str, ms: str, bench: str, n: int) -> tuple[str, str, str, str]:
    name = f"recovery_{ms}_{bench}"
    args = (
        f"--baseline_dir {ARCHIVE}/baseline/{ms}_{bench} "
        f"--perturbed_dir {ARCHIVE}/perturbed/{ms}_{bench}_k4_importance "
        f"--model {hf} --benchmark {bench} --n {n} --rc_source cot_pt "
        f"--batch_size 8 --output_dir {OUT} --run_label recovery"
    )
    return (name, RECOVERY, f"{OUT}/{ms}_{bench}_recovery", args)


def build_rows(n_core: int, n_deep: int, n_recovery: int) -> list[tuple]:
    rows: list[tuple] = []
    # P1: コア対比 (両建て) M5×B5。検証用に gemma-3-4b × B2 を先頭に置く
    p1_order = [(hf, ms, b) for b in B5 for (hf, ms) in M5]
    p1_order.sort(key=lambda t: (t[1] != "gemma-3-4b-it", t[2] not in B2))
    for hf, ms, b in p1_order:
        rows.append(deletion_row(hf, ms, b, "core", "core", n_core))
    # P2: 完全グリッド (39腕) Gemma-3-4B×B2、LOO はインライン (exp6 本番未完)
    for b in B2:
        rows.append(deletion_row(
            "google/gemma-3-4b-it", "gemma-3-4b-it", b, "full", "full", n_deep,
            extra="--loo_inline --loo_deletion_mode occurrence"))
    # P3: LOO 腕 M3×B2 の残り (gemma-3-4b は P2 が含む)
    for hf, ms in M3:
        if ms == "gemma-3-4b-it":
            continue
        for b in B2:
            rows.append(deletion_row(
                hf, ms, b, "loo", "loo", n_deep,
                extra="--loo_inline --loo_deletion_mode occurrence"))
    # P4: 回復曲線 M3×B2
    for hf, ms in M3:
        for b in B2:
            rows.append(recovery_row(hf, ms, b, n_recovery))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_core", type=int, default=500,
                    help="P1 の clean 正解サンプル数/設定")
    ap.add_argument("--n_deep", type=int, default=500,
                    help="P2/P3 の clean 正解サンプル数/設定")
    ap.add_argument("--n_recovery", type=int, default=300,
                    help="P4 の flip 事例数/設定")
    ap.add_argument("--out", type=str,
                    default="results/prod/exp2/queue/shards_active.tsv")
    args = ap.parse_args()

    rows = build_rows(args.n_core, args.n_deep, args.n_recovery)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# exp2 production queue (P1 core 25 / P2 full 2 / P3 loo 4 / P4 recovery 6)\n")
        f.write("# columns: name\tscript\tdone_dir\targs\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(rows)} shards -> {out}")


if __name__ == "__main__":
    main()
