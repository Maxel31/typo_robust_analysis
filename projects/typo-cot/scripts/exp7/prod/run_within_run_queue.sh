#!/usr/bin/env bash
# 実験7: byte-identical 復元サンプルの within-run flip 検証キュー.
#
# 1 シャード = 1 設定 × 1 校正器 (byte-identical 集合のみを生成するため短時間)。
# 冪等: results/prod/exp7/within_run/{config}/within_run_results.json が
# あればスキップ。GPU はヘルパー run_with_gpu.sh (GPU 3/4/5/6, flock 排他) 経由。
#
# usage: run_within_run_queue.sh [phase1|rest|all] [stride] [offset]
#   phase1 = M3×B2 (gemma-3-4b-it / Llama-3.2-3B / Mistral-7B × gsm8k / mmlu)
#            × 3 校正器 = 18 設定 (実行順は集合の小さい spellfix から)
#   rest   = 残り 57 設定
#   stride/offset = 並列ワーカー用の間引き (例: 2 0 と 2 1 で2ワーカー)
#
# 起動例 (2 ワーカー, setsid 切り離し):
#   setsid nohup bash scripts/exp7/prod/run_within_run_queue.sh phase1 2 0 \
#     > logs/exp7/within_run_queue_w0.log 2>&1 &
#   setsid nohup bash scripts/exp7/prod/run_within_run_queue.sh phase1 2 1 \
#     > logs/exp7/within_run_queue_w1.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.."   # projects/typo-cot

PHASE=${1:-phase1}
STRIDE=${2:-1}
OFFSET=${3:-0}

RUN_WITH_GPU=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
CORRECTED=data/exp7/corrected
OUT=results/prod/exp7/within_run
mkdir -p logs/exp7 "$OUT"
export GPU_LOCK_TIMEOUT=86400

M3=(gemma-3-4b-it Llama-3.2-3B-Instruct Mistral-7B-Instruct-v0.3)
ALL_MODELS=(Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct gemma-3-1b-it gemma-3-4b-it Mistral-7B-Instruct-v0.3)
B2=(gsm8k mmlu)
ALL_BENCHES=(gsm8k mmlu mmlu_pro arc commonsense_qa)
# 集合の小さい順に流す (spellfix 10-30% < llmfix 28-63% < neuralfix 37-84%)
MODES=(spellfix llmfix neuralfix)

in_phase1() {
  local cfg=$1 m b md
  for m in "${M3[@]}"; do for b in "${B2[@]}"; do for md in "${MODES[@]}"; do
    [ "$cfg" = "${m}_${b}_k4_${md}" ] && return 0
  done; done; done
  return 1
}

configs=()
if [ "$PHASE" = "phase1" ]; then
  for md in "${MODES[@]}"; do for b in "${B2[@]}"; do for m in "${M3[@]}"; do
    configs+=("${m}_${b}_k4_${md}")
  done; done; done
else
  for md in "${MODES[@]}"; do for b in "${ALL_BENCHES[@]}"; do for m in "${ALL_MODELS[@]}"; do
    cfg="${m}_${b}_k4_${md}"
    if [ "$PHASE" = "rest" ] && in_phase1 "$cfg"; then continue; fi
    configs+=("$cfg")
  done; done; done
fi

echo "[queue] phase=$PHASE stride=$STRIDE offset=$OFFSET configs=${#configs[@]} $(date '+%F %T')"
n_done=0; n_skip=0; n_fail=0
i=0
for cfg in "${configs[@]}"; do
  idx=$i; i=$((i + 1))
  [ $((idx % STRIDE)) -ne "$OFFSET" ] && continue
  if [ ! -f "$CORRECTED/$cfg/perturbed_dataset.json" ]; then
    echo "[miss] $cfg (校正済みデータなし)"
    n_fail=$((n_fail + 1))
    continue
  fi
  if [ -f "$OUT/$cfg/within_run_results.json" ]; then
    echo "[skip] $cfg (既済)"
    n_skip=$((n_skip + 1))
    continue
  fi
  echo "[run ] $cfg $(date '+%F %T')"
  if bash "$RUN_WITH_GPU" uv run --no-sync python scripts/exp7/within_run_flip.py \
      --config "$cfg" --gpu_id 0 --output_dir "$OUT"; then
    echo "[done] $cfg $(date '+%F %T')"
    n_done=$((n_done + 1))
  else
    rc=$?
    echo "[fail] $cfg rc=$rc $(date '+%F %T')"
    n_fail=$((n_fail + 1))
    if [ "$rc" -eq 86 ]; then
      echo "[queue] GPU 一時停止指示 (SMOKE_PAUSED) のためキューを終了します"
      exit 86
    fi
  fi
done
echo "[queue] 完了: done=$n_done skip=$n_skip fail=$n_fail $(date '+%F %T')"
