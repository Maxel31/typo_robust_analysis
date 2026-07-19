#!/usr/bin/env bash
# 実験7 本番: 校正後テキストの評価生成キュー (75 ラン = 25 設定 × 3 校正器)。
#
# 各設定は「摂動元モデル = 評価モデル」(LXT-4 摂動が当該モデルの帰属に基づく
# ため、rebuttal と同じく同一モデルで評価する)。rebuttal 済みの
# pyspell × gemma-3-4b-it × {gsm8k, mmlu} も手続き統一のため再実行に含める
# (ユーザー決定 2026-07-14: 統一再生成、revision notes に明記)。
#
# - 1 シャード = 1 回の run_with_gpu.sh 呼び出し (目安 20〜40 分)。
# - 冪等: シャード JSON があればスキップ。全シャード完了で merge。
# - 前提: 校正済みデータセット (run_pyspell_grid.sh / run_correction_queue.sh)
#   が完了していること。未完成の設定はスキップして後で再実行すればよい。
#
# 起動 (exp4/exp5 のキューが掃けた後):
#   nohup bash scripts/exp7/prod/run_eval_generation_queue.sh \
#     > logs/exp7/eval_queue.log 2>&1 &
# 特定校正器のみ: run_eval_generation_queue.sh spellfix
set -u
cd "$(dirname "$0")/../../.."   # projects/typo-cot

ONLY_MODE=${1:-}
SHARD_SIZE=${2:-800}

RUN_WITH_GPU=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
CORRECTED=data/exp7/corrected
OUT=results/prod/exp7/generation
PROGRESS=logs/exp7/eval_queue_progress.json
mkdir -p logs/exp7 "$OUT"
export GPU_LOCK_TIMEOUT=86400

MODELS=(Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct gemma-3-1b-it gemma-3-4b-it Mistral-7B-Instruct-v0.3)
declare -A HF_NAME=(
  [Llama-3.2-1B-Instruct]=meta-llama/Llama-3.2-1B-Instruct
  [Llama-3.2-3B-Instruct]=meta-llama/Llama-3.2-3B-Instruct
  [gemma-3-1b-it]=google/gemma-3-1b-it
  [gemma-3-4b-it]=google/gemma-3-4b-it
  [Mistral-7B-Instruct-v0.3]=mistralai/Mistral-7B-Instruct-v0.3
)
BENCHES=(gsm8k mmlu mmlu_pro arc commonsense_qa)
declare -A N_TOTAL=([gsm8k]=1319 [mmlu]=2850 [mmlu_pro]=1400 [arc]=1172 [commonsense_qa]=1221)
MODES=(spellfix neuralfix llmfix)

n_done=0; n_skip=0; n_fail=0; n_pending=0
progress() {
  printf '{"done_shards": %d, "skipped": %d, "failed": %d, "pending_datasets": %d, "updated": "%s"}\n' \
    "$n_done" "$n_skip" "$n_fail" "$n_pending" "$(date -Is)" > "$PROGRESS"
}

for mode in "${MODES[@]}"; do
  [ -n "$ONLY_MODE" ] && [ "$mode" != "$ONLY_MODE" ] && continue
  for m in "${MODELS[@]}"; do
    for b in "${BENCHES[@]}"; do
      data="$CORRECTED/${m}_${b}_k4_${mode}/perturbed_dataset.json"
      exp_dir="$OUT/${m}_${b}_k4_${mode}"
      if [ ! -f "$data" ]; then
        echo "[wait] $m x $b x $mode (校正済みデータ未完成)"
        n_pending=$((n_pending + 1))
        continue
      fi
      if [ -f "$exp_dir/summary.json" ]; then
        echo "[skip] $m x $b x $mode (結合済み)"
        continue
      fi
      n=${N_TOTAL[$b]}
      all_shards_ok=1
      start=0
      while [ "$start" -lt "$n" ]; do
        end=$((start + SHARD_SIZE)); [ "$end" -gt "$n" ] && end=$n
        shard_file=$(printf '%s/shards/%05d_%05d.json' "$exp_dir" "$start" "$end")
        if [ -f "$shard_file" ]; then
          n_skip=$((n_skip + 1))
        else
          echo "[run ] $m x $b x $mode shard $start-$end $(date '+%F %T')"
          if bash "$RUN_WITH_GPU" uv run python scripts/rebuttal/run_generation_only.py \
              --model "${HF_NAME[$m]}" --benchmark "$b" \
              --perturbed_data "$data" \
              --batch_size 4 --gpu_id 0 \
              --start "$start" --limit $((end - start)) --shard \
              --output_dir "$OUT"; then
            n_done=$((n_done + 1))
          else
            rc=$?
            echo "[FAIL] $m x $b x $mode shard $start-$end (rc=$rc)" >&2
            n_fail=$((n_fail + 1)); all_shards_ok=0
            [ "$rc" = 86 ] && { echo "GPU 一時停止指示を検出、キュー終了"; progress; exit 86; }
          fi
          progress
        fi
        start=$end
      done
      if [ "$all_shards_ok" = 1 ]; then
        echo "[merge] $m x $b x $mode"
        uv run python scripts/exp7/merge_generation_shards.py \
          --experiment_dir "$exp_dir" --expected_total "$n" \
          || { echo "[FAIL] merge $m x $b x $mode" >&2; n_fail=$((n_fail + 1)); }
        progress
      fi
    done
  done
done
progress
echo "=== 評価生成キュー完了: done=$n_done skip=$n_skip fail=$n_fail pending=$n_pending ==="
