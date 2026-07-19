#!/usr/bin/env bash
# 実験13: LOO 集中度拡張キュー — 1B モデル (gemma-3-1b, Llama-3.2-1B) x {gsm8k, mmlu}。
#
# 既存の exp6 LOO (3B/4B/7B x gsm8k/mmlu, clean, occurrence, n=300, seed=42) を
# 1B 2モデルへ拡張し、5モデル x 2ベンチ = 10 設定の LOO 集中度 (M3) を揃える。
#
# GPU は必ず tmp/gpu-locks/run_with_gpu.sh 経由 (GPU 3,4,5,6 を他系統とロック共有)。
# 冪等: summary.json があるシャードはスキップ。多重起動安全: mkdir で claim。
# 進捗: results/loo/queue/progress/{shard}.json (updated は ISO-8601)。
#
# 使い方 (worktree の projects/typo-cot から):
#   setsid nohup bash scripts/run_exp13_loo_queue.sh \
#     > results/loo/queue/logs/exp13_worker_$$.out 2>&1 &
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJ/../../../../.." && pwd)"
LOCKS="$REPO_ROOT/tmp/gpu-locks"
ARCHIVE="/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs"
QUEUE="$PROJ/results/loo/queue"
N=300
SEED=42
BATCH=8

mkdir -p "$QUEUE/claims" "$QUEUE/progress" "$QUEUE/logs"

# name|model|benchmark|run_dir|clean_run_dir|deletion_mode|run_label
SHARDS=(
  "gemma1b_gsm8k_clean_occ|google/gemma-3-1b-it|gsm8k|$ARCHIVE/baseline/gemma-3-1b-it_gsm8k|$ARCHIVE/baseline/gemma-3-1b-it_gsm8k|occurrence|clean_occ"
  "gemma1b_mmlu_clean_occ|google/gemma-3-1b-it|mmlu|$ARCHIVE/baseline/gemma-3-1b-it_mmlu|$ARCHIVE/baseline/gemma-3-1b-it_mmlu|occurrence|clean_occ"
  "llama1b_gsm8k_clean_occ|meta-llama/Llama-3.2-1B-Instruct|gsm8k|$ARCHIVE/baseline/Llama-3.2-1B-Instruct_gsm8k|$ARCHIVE/baseline/Llama-3.2-1B-Instruct_gsm8k|occurrence|clean_occ"
  "llama1b_mmlu_clean_occ|meta-llama/Llama-3.2-1B-Instruct|mmlu|$ARCHIVE/baseline/Llama-3.2-1B-Instruct_mmlu|$ARCHIVE/baseline/Llama-3.2-1B-Instruct_mmlu|occurrence|clean_occ"
)

progress() {  # name status rc
  local name="$1" status="$2" rc="${3:-null}"
  cat > "$QUEUE/progress/$name.json" <<EOF
{"shard": "$name", "status": "$status", "rc": $rc,
 "worker_pid": $$, "updated": "$(date -Iseconds)"}
EOF
}

out_dir_for() {  # model benchmark label
  local model_short="${1##*/}"
  echo "$PROJ/results/loo/${model_short}_${2}_${3}"
}

cd "$PROJ"
ran_any=1
while [ "$ran_any" = 1 ]; do
  ran_any=0
  for spec in "${SHARDS[@]}"; do
    IFS='|' read -r name model bench run_dir clean_dir mode label <<< "$spec"
    out_dir="$(out_dir_for "$model" "$bench" "$label")"
    if [ -f "$out_dir/summary.json" ]; then
      progress "$name" done 0
      continue
    fi
    if ! mkdir "$QUEUE/claims/$name" 2>/dev/null; then continue; fi
    ran_any=1
    progress "$name" running
    echo "[exp13-loo $$] start $name -> $out_dir" >&2
    bash "$LOCKS/run_with_gpu.sh" uv run python scripts/run_loo_scoring.py \
      --run_dir "$run_dir" \
      --clean_run_dir "$clean_dir" \
      --model "$model" --benchmark "$bench" \
      --n "$N" --seed "$SEED" --batch_size "$BATCH" \
      --deletion-mode "$mode" \
      --output_dir results/loo --run_label "$label" \
      > "$QUEUE/logs/$name.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$out_dir/summary.json" ]; then
      progress "$name" done 0
      echo "[exp13-loo $$] done $name" >&2
    else
      progress "$name" failed "$rc"
      echo "[exp13-loo $$] FAILED $name (rc=$rc), see $QUEUE/logs/$name.log" >&2
      if [ "$rc" -eq 86 ] || [ "$rc" -eq 124 ]; then exit "$rc"; fi
    fi
  done
done
echo "[exp13-loo $$] no more claimable shards" >&2
