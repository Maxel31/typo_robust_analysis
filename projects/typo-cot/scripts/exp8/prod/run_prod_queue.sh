#!/usr/bin/env bash
# 実験8 本番: M3×B2 全 flip ペア, GPU 0 専用
# Usage: setsid nohup bash scripts/exp8/prod/run_prod_queue.sh > logs/exp8/prod/queue.log 2>&1 < /dev/null &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
ARCH=/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026
LOGDIR="$ROOT/logs/exp8/prod"
OUTDIR="$ROOT/results/prod/exp8"
mkdir -p "$LOGDIR" "$OUTDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# GPU 0 専用: CUDA_VISIBLE_DEVICES を直接設定 (ユーザー許可済み 2026-07-16)
export CUDA_VISIBLE_DEVICES=0

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
        SETTING_OUTDIR="$OUTDIR/${setting}"
        done_file="$SETTING_OUTDIR/run_summary.json"
        if [ -f "$done_file" ]; then
            log "SKIP $setting (already done)"
            continue
        fi
        mkdir -p "$SETTING_OUTDIR"
        log "START $setting (all flip pairs, 3 sites)"
        uv run python scripts/exp8/run_patching.py \
            --model "${HF_NAME[$m]}" \
            --benchmark "$b" \
            --baseline-dir "$ARCH/outputs/baseline/${m}_${b}" \
            --perturbed-dir-lxt "$ARCH/outputs/perturbed/${m}_${b}_k4_importance" \
            --perturbed-dir-rnd "$ARCH/outputs/perturbed/${m}_${b}_k4_random" \
            --output-dir "$SETTING_OUTDIR" \
            --n-pairs 0 \
            --sites residual attn mlp \
            --max-new-tokens 16 \
            > "$LOGDIR/${setting}.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            log "DONE $setting"
        else
            log "FAIL $setting (rc=$rc)"
        fi
    done
done
log "QUEUE FINISHED"
