#!/usr/bin/env bash
# 実験9 スモーク: Gemma-3-4B-it x GSM8K/MMLU, LXT-4 n=32 + Random-4 n=16
# GPU ヘルパー経由で呼ぶ:
#   bash /diskthalys/.../tmp/gpu-locks/run_with_gpu.sh bash scripts/exp9/smoke.sh
# forward のみ (生成なし)。ロック保持時間 << 30 分。
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=results/smoke/exp9

uv run python scripts/exp9/run_inner_repair.py \
    --model gemma-3-4b-it --benchmarks gsm8k mmlu \
    --conditions lxt4 --n 32 --output-dir "$OUT"

uv run python scripts/exp9/run_inner_repair.py \
    --model gemma-3-4b-it --benchmarks gsm8k mmlu \
    --conditions random4 --n 16 --output-dir "$OUT"

echo "[smoke] done: $OUT"
