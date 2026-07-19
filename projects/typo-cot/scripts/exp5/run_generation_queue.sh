#!/usr/bin/env bash
# 実験5: Matched-Rnd-4 の本番生成キュー (25 設定、1 設定 = 1 シャード)。
#
# - 各シャードは run_with_gpu.sh 経由で GPU 3/4 のロックを取得し、シャード間で解放する。
# - 完了スキップ: results/exp5/perturbed/<name>/summary.json があれば飛ばす。
# - データセット未構築 (run_all_matched_datasets.sh が並行実行中) の設定は後回しにし、
#   全設定が done/failed になるまでパスを繰り返す。
# - 進捗 JSON: logs/exp5/queue_progress.json (queue_progress.py --print で監視)。
# - run_with_gpu.sh が 86 (SMOKE_PAUSED) を返したらキュー全体を停止する。
#
# 起動例:
#   nohup bash scripts/exp5/run_generation_queue.sh > logs/exp5/queue.log 2>&1 &
set -u

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
SETTINGS_FILE="$PROJ/scripts/exp5/settings_25.txt"
DATA="$PROJ/data/exp5/matched_rnd"
OUT="$PROJ/results/exp5/perturbed"
LOGDIR="$PROJ/logs/exp5/gen"
STATE="$PROJ/logs/exp5/queue_state"
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
mkdir -p "$LOGDIR" "$STATE" "$OUT"

update_progress() {
  (cd "$PROJ" && uv run --no-sync python scripts/exp5/queue_progress.py \
     --max_attempts "$MAX_ATTEMPTS") || true
}

# settings_25.txt を読み込む (コメント・空行を除く)
mapfile -t SETTINGS < <(grep -vE '^\s*(#|$)' "$SETTINGS_FILE")
echo "[queue] $(date -Is) 開始: ${#SETTINGS[@]} 設定, batch_size=$BATCH_SIZE"
update_progress

while :; do
  remaining=0
  ran_shard=0
  for line in "${SETTINGS[@]}"; do
    read -r model bench <<<"$line"
    name="${model}_${bench}_k4_matched_rnd"

    # 完了スキップ
    [ -f "$OUT/$name/summary.json" ] && continue

    att=$(cat "$STATE/$name.attempts" 2>/dev/null || echo 0)
    if [ "$att" -ge "$MAX_ATTEMPTS" ]; then
      continue  # 永続失敗 (progress JSON では failed)
    fi
    remaining=1

    # データセット未構築なら後回し (matched_stats.json が構築完了マーカー)
    if [ ! -f "$DATA/$name/matched_stats.json" ]; then
      continue
    fi

    hf=$(python3 -c "import json;print(json.load(open('$DATA/$name/config.json'))['source_model'])") || {
      echo "[queue] $(date -Is) $name: config.json 読取失敗" >&2
      echo $((att + 1)) > "$STATE/$name.attempts"
      continue
    }

    echo $((att + 1)) > "$STATE/$name.attempts"
    touch "$STATE/$name.running"
    update_progress
    echo "[queue] $(date -Is) シャード開始: $name (model=$hf, attempt $((att + 1)))"
    GPU_LOCK_TIMEOUT=86400 bash "$WRAPPER" bash -c "cd '$PROJ' && PYTHONHASHSEED=42 uv run --no-sync python scripts/rebuttal/run_generation_only.py --model '$hf' --benchmark '$bench' --perturbed_data 'data/exp5/matched_rnd/$name/perturbed_dataset.json' --gpu_id \"\$CUDA_VISIBLE_DEVICES\" --batch_size $BATCH_SIZE --output_dir 'results/exp5/perturbed'" >> "$LOGDIR/$name.log" 2>&1
    rc=$?
    rm -f "$STATE/$name.running"
    ran_shard=1
    if [ "$rc" -eq 86 ]; then
      update_progress
      echo "[queue] $(date -Is) SMOKE_PAUSED (rc=86) を検出。キューを停止します。"
      exit 86
    fi
    if [ "$rc" -eq 0 ] && [ -f "$OUT/$name/summary.json" ]; then
      echo "[queue] $(date -Is) シャード完了: $name"
    else
      echo "[queue] $(date -Is) シャード失敗: $name (rc=$rc, attempt $((att + 1))/$MAX_ATTEMPTS)" >&2
    fi
    update_progress
  done

  if [ "$remaining" -eq 0 ]; then
    break
  fi
  # 残りがあるのにこのパスで 1 シャードも走らなかった場合はデータセット構築待ち
  if [ "$ran_shard" -eq 0 ]; then
    sleep 60
  fi
done

update_progress
echo "[queue] $(date -Is) キュー終了 (全設定 done または failed)"
