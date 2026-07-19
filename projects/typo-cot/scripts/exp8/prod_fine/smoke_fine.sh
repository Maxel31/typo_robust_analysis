#!/usr/bin/env bash
# 実験8-fine スモーク: Gemma-3-4B × GSM8K, flip ペア n=16.
# 単層 0-11 + 検証点 14/20/26 + 累積 0-11 + noising 0-7 + sham 0-11 を実行し、
# 早期層でピーク / 検証点≈0 / sham≈0 を確認する。
# GPU は必ずロックヘルパー経由 (allowed 0,3,4,5,6)。
# Usage: setsid nohup bash scripts/exp8/prod_fine/smoke_fine.sh \
#        > logs/exp8_fine/smoke.log 2>&1 < /dev/null &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"          # projects/typo-cot
cd "$ROOT"
GPU_HELPER=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
ARCH=/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026
OUTDIR="$ROOT/results/exp8_fine/smoke_gemma-3-4b-it_gsm8k"
LOGDIR="$ROOT/logs/exp8_fine"
mkdir -p "$OUTDIR" "$LOGDIR"

m=gemma-3-4b-it
b=gsm8k
echo "[$(date '+%F %T')] SMOKE START $m $b (n=16)"
bash "$GPU_HELPER" uv run --package typo-cot python scripts/exp8/run_patching_fine.py \
    --model google/${m} \
    --benchmark "$b" \
    --baseline-dir "$ARCH/outputs/baseline/${m}_${b}" \
    --perturbed-dir-lxt "$ARCH/outputs/perturbed/${m}_${b}_k4_importance" \
    --perturbed-dir-rnd "$ARCH/outputs/perturbed/${m}_${b}_k4_random" \
    --output-dir "$OUTDIR" \
    --n-pairs 16
rc=$?
echo "[$(date '+%F %T')] SMOKE DONE rc=$rc"
exit $rc
