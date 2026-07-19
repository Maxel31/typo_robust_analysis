#!/usr/bin/env bash
# 実験8-fine 本番キュー: 3モデル × 2ベンチ = 6 設定, flip ペア n=150.
# 各設定は単層 0-11 + 検証 14/20/26 + 累積 0-11 + noising 0-7 + sham 0-11。
# GPU は必ずロックヘルパー経由 (allowed 0,3,4,5,6; 並行系統とロック共有)。
# 冪等: run_summary_fine.json があればスキップ。
# Usage: setsid nohup bash scripts/exp8/prod_fine/run_fine_queue.sh \
#        > logs/exp8_fine/queue.log 2>&1 < /dev/null &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"          # projects/typo-cot
cd "$ROOT"
GPU_HELPER=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
ARCH=/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026
OUTROOT="$ROOT/results/prod/exp8_fine"
LOGDIR="$ROOT/logs/exp8_fine"
mkdir -p "$OUTROOT" "$LOGDIR"
N_PAIRS="${N_PAIRS:-150}"

log() { echo "[$(date '+%F %T')] $*"; }

declare -A HF_NAME=(
    [gemma-3-4b-it]="google/gemma-3-4b-it"
    [Llama-3.2-3B-Instruct]="meta-llama/Llama-3.2-3B-Instruct"
    [Mistral-7B-Instruct-v0.3]="mistralai/Mistral-7B-Instruct-v0.3"
)
MODELS=(gemma-3-4b-it Llama-3.2-3B-Instruct Mistral-7B-Instruct-v0.3)
BENCHMARKS=(gsm8k mmlu)

for m in "${MODELS[@]}"; do
    for b in "${BENCHMARKS[@]}"; do
        setting="${m}_${b}"
        SETTING_OUTDIR="$OUTROOT/${setting}"
        if [ -f "$SETTING_OUTDIR/run_summary_fine.json" ]; then
            log "SKIP $setting (already done)"
            continue
        fi
        mkdir -p "$SETTING_OUTDIR"
        log "START $setting (n=$N_PAIRS)"
        bash "$GPU_HELPER" uv run --package typo-cot python scripts/exp8/run_patching_fine.py \
            --model "${HF_NAME[$m]}" \
            --benchmark "$b" \
            --baseline-dir "$ARCH/outputs/baseline/${m}_${b}" \
            --perturbed-dir-lxt "$ARCH/outputs/perturbed/${m}_${b}_k4_importance" \
            --perturbed-dir-rnd "$ARCH/outputs/perturbed/${m}_${b}_k4_random" \
            --output-dir "$SETTING_OUTDIR" \
            --n-pairs "$N_PAIRS" \
            > "$LOGDIR/${setting}.log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            log "DONE $setting"
        elif [ "$rc" -eq 86 ]; then
            log "PAUSED $setting (SMOKE_PAUSED); leaving pending and continuing"
        else
            log "FAIL $setting (rc=$rc)"
        fi
    done
done
log "QUEUE FINISHED"
