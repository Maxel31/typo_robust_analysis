#!/usr/bin/env bash
# 実験7 本番: GPU 校正段 (neural=T5-large-spell / llm=Qwen2.5-7B-Instruct) の
# シャードキュー。25 データセット (5 摂動元モデル × 5 ベンチ) を処理する。
#
# - 1 シャード = 1 回の run_with_gpu.sh 呼び出し (目安 20〜40 分)。
#   シャード間でロックを解放し、実験10 の生成ジョブ等と交互に進む。
# - 冪等: 完了済みシャード (shards/*.json が存在) はスキップ。中断後の再実行で
#   続きから進む。全シャード完了時に merge_corrected_shards.py で結合する。
# - ロック待ちは正常な状態。GPU_LOCK_TIMEOUT=86400 (24h)。
#
# 起動 (exp4/exp5 のキューが掃けた後):
#   nohup bash scripts/exp7/prod/run_correction_queue.sh neural \
#     > logs/exp7/correction_queue_neural.log 2>&1 &
#   nohup bash scripts/exp7/prod/run_correction_queue.sh llm \
#     > logs/exp7/correction_queue_llm.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.."   # projects/typo-cot

CORRECTOR=${1:?"usage: run_correction_queue.sh <neural|llm> [shard_size]"}
case "$CORRECTOR" in
  neural) MODE=neuralfix; SHARD_SIZE=${2:-1500} ;;
  llm)    MODE=llmfix;    SHARD_SIZE=${2:-500} ;;
  *) echo "corrector は neural か llm" >&2; exit 2 ;;
esac

RUN_WITH_GPU=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
ARCHIVE=/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/datasets/perturbed
OUT=data/exp7/corrected
PROGRESS=logs/exp7/correction_queue_${CORRECTOR}_progress.json
mkdir -p logs/exp7 "$OUT"
export GPU_LOCK_TIMEOUT=86400

MODELS=(Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct gemma-3-1b-it gemma-3-4b-it Mistral-7B-Instruct-v0.3)
BENCHES=(gsm8k mmlu mmlu_pro arc commonsense_qa)
# ベンチごとのサンプル総数 (アーカイブ LXT-4 データセットの実数)
declare -A N_TOTAL=([gsm8k]=1319 [mmlu]=2850 [mmlu_pro]=1400 [arc]=1172 [commonsense_qa]=1221)

n_done=0; n_skip=0; n_fail=0
progress() {
  printf '{"corrector": "%s", "done_shards": %d, "skipped": %d, "failed": %d, "updated": "%s"}\n' \
    "$CORRECTOR" "$n_done" "$n_skip" "$n_fail" "$(date -Is)" > "$PROGRESS"
}

for m in "${MODELS[@]}"; do
  for b in "${BENCHES[@]}"; do
    src="$ARCHIVE/${m}_${b}_k4_with_choices/perturbed_dataset.json"
    dst="$OUT/${m}_${b}_k4_${MODE}"
    if [ -f "$dst/restoration_stats.json" ]; then
      echo "[skip] $m x $b (結合済み)"
      continue
    fi
    n=${N_TOTAL[$b]}
    all_shards_ok=1
    start=0
    while [ "$start" -lt "$n" ]; do
      end=$((start + SHARD_SIZE)); [ "$end" -gt "$n" ] && end=$n
      shard_file=$(printf '%s/shards/%05d_%05d.json' "$dst" "$start" "$end")
      if [ -f "$shard_file" ]; then
        echo "[skip] $m x $b shard $start-$end"
        n_skip=$((n_skip + 1))
      else
        echo "[run ] $m x $b shard $start-$end $(date '+%F %T')"
        if bash "$RUN_WITH_GPU" uv run python scripts/exp7/make_corrected_dataset.py \
            --input "$src" --corrector "$CORRECTOR" --device cuda \
            --start "$start" --limit $((end - start)) --shard \
            --output_dir "$OUT"; then
          n_done=$((n_done + 1))
        else
          rc=$?
          echo "[FAIL] $m x $b shard $start-$end (rc=$rc)" >&2
          n_fail=$((n_fail + 1)); all_shards_ok=0
          [ "$rc" = 86 ] && { echo "GPU 一時停止指示を検出、キュー終了"; progress; exit 86; }
        fi
        progress
      fi
      start=$end
    done
    if [ "$all_shards_ok" = 1 ]; then
      echo "[merge] $m x $b"
      uv run python scripts/exp7/merge_corrected_shards.py \
        --dataset_dir "$dst" --source "$src" \
        || { echo "[FAIL] merge $m x $b" >&2; n_fail=$((n_fail + 1)); }
      progress
    fi
  done
done
progress
echo "=== ${CORRECTOR} 校正キュー完了: done=$n_done skip=$n_skip fail=$n_fail ==="
