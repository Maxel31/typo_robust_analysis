#!/usr/bin/env bash
# 実験5: M5 x B5 = 25 設定の Matched-Rnd-4 データセットを一括作成 (CPU のみ)。
# SMD 表は全 25 設定で必要なため全設定でデータセットを作る。
# 生成 (GPU) は 23 設定のみ (gemma-3-4b-it x gsm8k/mmlu は rebuttal ログ流用) —
# 生成コマンドは docs/dev_notes_05_matched_control.md を参照。
#
# 使用例:
#   bash scripts/exp5/run_all_matched_datasets.sh [output_dir]
set -euo pipefail

ARCHIVE=${ARCHIVE:-/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline}
OUTPUT_DIR=${1:-data/exp5/matched_rnd}
LOG_DIR=logs/exp5
mkdir -p "$LOG_DIR"

MODELS=(
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  "gemma-3-1b-it"
  "gemma-3-4b-it"
  "Mistral-7B-Instruct-v0.3"
)
BENCHMARKS=(gsm8k mmlu mmlu_pro arc commonsense_qa)

for model in "${MODELS[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    baseline="$ARCHIVE/${model}_${bench}"
    if [ ! -d "$baseline" ]; then
      echo "[skip] baseline なし: $baseline" >&2
      continue
    fi
    out_name="${model}_${bench}_k4_matched_rnd"
    if [ -f "$OUTPUT_DIR/$out_name/perturbed_dataset.json" ]; then
      echo "[skip] 既存: $out_name" >&2
      continue
    fi
    echo "[run] $out_name"
    PYTHONHASHSEED=42 uv run --no-sync python scripts/exp5/make_matched_twin_dataset.py \
      --baseline_dir "$baseline" -k 4 --output_dir "$OUTPUT_DIR" \
      > "$LOG_DIR/make_${model}_${bench}.log" 2>&1
  done
done

echo "完了。SMD 表: $OUTPUT_DIR/*/matched_stats.json"
