#!/usr/bin/env bash
# 実験15 本番キュー: M3 × B2、全 flip ペア (n-pairs 上限 300/設定)、
# 早期/中期/後期 × denoise/noise、sham(no-op)検証つき、CoT 自由生成。
#
# GPU は必ず共有ヘルパー run_with_gpu.sh 経由 (GPU 3,4,5,6 のロックプールを
# 並行系統と共有)。1 設定 = 1 invocation。ペア×条件ごとに JSON を書く冪等実行
# なので、途中中断しても同じコマンドで再開する (done ペアはスキップ)。
#
# 起動:
#   setsid nohup bash scripts/exp15/prod/run_queue.sh \
#     > logs/exp15/prod/queue.log 2>&1 < /dev/null &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # <worktree>/projects/typo-cot
cd "$ROOT"
# GPU ヘルパーと torch 入り .venv はワークツリーではなくメインリポジトリ側にある。
# (worktree は .../typo_robust_analysis/.claude/worktrees/<name>/projects/typo-cot)
MAIN_REPO="${MAIN_REPO:-/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis}"
ARCH=/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026
HELPER="$MAIN_REPO/tmp/gpu-locks/run_with_gpu.sh"
VENV="$MAIN_REPO/.venv/bin/python"
LOGDIR="$ROOT/logs/exp15/prod"
OUTDIR="$ROOT/results/exp15"
mkdir -p "$LOGDIR" "$OUTDIR"

export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

declare -A HF_NAME=(
    [gemma-3-4b-it]="google/gemma-3-4b-it"
    [Llama-3.2-3B-Instruct]="meta-llama/Llama-3.2-3B-Instruct"
    [Mistral-7B-Instruct-v0.3]="mistralai/Mistral-7B-Instruct-v0.3"
)
MODELS=(gemma-3-4b-it Llama-3.2-3B-Instruct Mistral-7B-Instruct-v0.3)
BENCHMARKS=(gsm8k mmlu)
N_PAIRS=300   # LXT-4/Random-4 半々 (=150+150)。flip がそれ未満の設定は全数。

for m in "${MODELS[@]}"; do
    for b in "${BENCHMARKS[@]}"; do
        setting="${m}_${b}"
        SETTING_OUTDIR="$OUTDIR/${setting}"
        done_file="$SETTING_OUTDIR/run_summary.json"
        if [ -f "$done_file" ]; then
            log "SKIP $setting (run_summary.json exists)"
            continue
        fi
        mkdir -p "$SETTING_OUTDIR"
        log "START $setting (n_pairs<=$N_PAIRS, early/mid/late x denoise/noise, sham)"
        bash "$HELPER" env \
            PYTHONPATH="$ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
            "$VENV" scripts/exp15/run_free_generation.py \
            --model "${HF_NAME[$m]}" \
            --benchmark "$b" \
            --baseline-dir "$ARCH/outputs/baseline/${m}_${b}" \
            --perturbed-dir-lxt "$ARCH/outputs/perturbed/${m}_${b}_k4_importance" \
            --perturbed-dir-rnd "$ARCH/outputs/perturbed/${m}_${b}_k4_random" \
            --output-dir "$SETTING_OUTDIR" \
            --n-pairs "$N_PAIRS" \
            --levels early mid late \
            --directions denoise noise \
            --noop-check \
            --max-new-tokens 512 \
            > "$LOGDIR/${setting}.log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            log "DONE $setting"
        elif [ "$rc" -eq 86 ]; then
            log "PAUSED $setting (SMOKE_PAUSED directive) — stopping queue"
            exit 86
        else
            log "FAIL $setting (rc=$rc) — continuing to next setting"
        fi
    done
done
log "QUEUE FINISHED"
