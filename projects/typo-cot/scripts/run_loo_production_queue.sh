#!/usr/bin/env bash
# 実験6-(iv) LOO ランキング本番キュー (M3 x B2 x {clean, LXT-4}).
#
# 1シャード = 1設定 x 1条件 = run_loo_scoring.py 1回 (n=300, seed=42)。
# GPU は必ず tmp/gpu-locks/run_with_gpu.sh 経由 (GPU 3,4,5,6 を他系統とロック共有)。
#
# 冪等性: 出力ディレクトリの summary.json があるシャードはスキップ。
# 多重起動安全: mkdir によるアトミックなシャード claim。
# 進捗: results/loo/queue/progress/{shard}.json (status/started/finished/rc)。
#
# 使い方 (worktree の projects/typo-cot から):
#   setsid nohup bash scripts/run_loo_production_queue.sh [--only SHARD_REGEX] \
#     > results/loo/queue/logs/worker_$$.out 2>&1 &
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJ/../../../../.." && pwd)"  # typo_robust_analysis 本体
LOCKS="$REPO_ROOT/tmp/gpu-locks"
ARCHIVE="/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs"
QUEUE="$PROJ/results/loo/queue"
N=300
SEED=42
BATCH=8

ONLY_RE=""
if [ "${1:-}" = "--only" ]; then ONLY_RE="${2:?--only requires a regex}"; fi

mkdir -p "$QUEUE/claims" "$QUEUE/progress" "$QUEUE/logs"

# shard 定義: name|model|benchmark|run_dir|clean_run_dir|deletion_mode|run_label
SHARDS=(
  # 主定義 (occurrence, 案B) — M3 x B2 x {clean, LXT-4}
  "gemma_gsm8k_clean_occ|google/gemma-3-4b-it|gsm8k|$ARCHIVE/baseline/gemma-3-4b-it_gsm8k|$ARCHIVE/baseline/gemma-3-4b-it_gsm8k|occurrence|clean_occ"
  "gemma_gsm8k_lxt4_occ|google/gemma-3-4b-it|gsm8k|$ARCHIVE/perturbed/gemma-3-4b-it_gsm8k_k4_importance|$ARCHIVE/baseline/gemma-3-4b-it_gsm8k|occurrence|lxt4_occ"
  "gemma_mmlu_clean_occ|google/gemma-3-4b-it|mmlu|$ARCHIVE/baseline/gemma-3-4b-it_mmlu|$ARCHIVE/baseline/gemma-3-4b-it_mmlu|occurrence|clean_occ"
  "gemma_mmlu_lxt4_occ|google/gemma-3-4b-it|mmlu|$ARCHIVE/perturbed/gemma-3-4b-it_mmlu_k4_importance|$ARCHIVE/baseline/gemma-3-4b-it_mmlu|occurrence|lxt4_occ"
  "llama_gsm8k_clean_occ|meta-llama/Llama-3.2-3B-Instruct|gsm8k|$ARCHIVE/baseline/Llama-3.2-3B-Instruct_gsm8k|$ARCHIVE/baseline/Llama-3.2-3B-Instruct_gsm8k|occurrence|clean_occ"
  "llama_gsm8k_lxt4_occ|meta-llama/Llama-3.2-3B-Instruct|gsm8k|$ARCHIVE/perturbed/Llama-3.2-3B-Instruct_gsm8k_k4_importance|$ARCHIVE/baseline/Llama-3.2-3B-Instruct_gsm8k|occurrence|lxt4_occ"
  "llama_mmlu_clean_occ|meta-llama/Llama-3.2-3B-Instruct|mmlu|$ARCHIVE/baseline/Llama-3.2-3B-Instruct_mmlu|$ARCHIVE/baseline/Llama-3.2-3B-Instruct_mmlu|occurrence|clean_occ"
  "llama_mmlu_lxt4_occ|meta-llama/Llama-3.2-3B-Instruct|mmlu|$ARCHIVE/perturbed/Llama-3.2-3B-Instruct_mmlu_k4_importance|$ARCHIVE/baseline/Llama-3.2-3B-Instruct_mmlu|occurrence|lxt4_occ"
  "mistral_gsm8k_clean_occ|mistralai/Mistral-7B-Instruct-v0.3|gsm8k|$ARCHIVE/baseline/Mistral-7B-Instruct-v0.3_gsm8k|$ARCHIVE/baseline/Mistral-7B-Instruct-v0.3_gsm8k|occurrence|clean_occ"
  "mistral_gsm8k_lxt4_occ|mistralai/Mistral-7B-Instruct-v0.3|gsm8k|$ARCHIVE/perturbed/Mistral-7B-Instruct-v0.3_gsm8k_k4_importance|$ARCHIVE/baseline/Mistral-7B-Instruct-v0.3_gsm8k|occurrence|lxt4_occ"
  "mistral_mmlu_clean_occ|mistralai/Mistral-7B-Instruct-v0.3|mmlu|$ARCHIVE/baseline/Mistral-7B-Instruct-v0.3_mmlu|$ARCHIVE/baseline/Mistral-7B-Instruct-v0.3_mmlu|occurrence|clean_occ"
  "mistral_mmlu_lxt4_occ|mistralai/Mistral-7B-Instruct-v0.3|mmlu|$ARCHIVE/perturbed/Mistral-7B-Instruct-v0.3_mmlu_k4_importance|$ARCHIVE/baseline/Mistral-7B-Instruct-v0.3_mmlu|occurrence|lxt4_occ"
  # 感度分析 (type, 案A) — Gemma-3-4B x B2 のみ
  "gemma_gsm8k_clean_type|google/gemma-3-4b-it|gsm8k|$ARCHIVE/baseline/gemma-3-4b-it_gsm8k|$ARCHIVE/baseline/gemma-3-4b-it_gsm8k|type|clean_type"
  "gemma_gsm8k_lxt4_type|google/gemma-3-4b-it|gsm8k|$ARCHIVE/perturbed/gemma-3-4b-it_gsm8k_k4_importance|$ARCHIVE/baseline/gemma-3-4b-it_gsm8k|type|lxt4_type"
  "gemma_mmlu_clean_type|google/gemma-3-4b-it|mmlu|$ARCHIVE/baseline/gemma-3-4b-it_mmlu|$ARCHIVE/baseline/gemma-3-4b-it_mmlu|type|clean_type"
  "gemma_mmlu_lxt4_type|google/gemma-3-4b-it|mmlu|$ARCHIVE/perturbed/gemma-3-4b-it_mmlu_k4_importance|$ARCHIVE/baseline/gemma-3-4b-it_mmlu|type|lxt4_type"
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
    if [ -n "$ONLY_RE" ] && ! [[ "$name" =~ $ONLY_RE ]]; then continue; fi
    out_dir="$(out_dir_for "$model" "$bench" "$label")"
    # 冪等: 完了済み (summary.json あり) はスキップ
    if [ -f "$out_dir/summary.json" ]; then
      progress "$name" done 0
      continue
    fi
    # アトミック claim (他 worker と競合しない)
    if ! mkdir "$QUEUE/claims/$name" 2>/dev/null; then continue; fi
    ran_any=1
    progress "$name" running
    echo "[queue $$] start $name -> $out_dir" >&2
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
      echo "[queue $$] done $name" >&2
    else
      progress "$name" failed "$rc"
      echo "[queue $$] FAILED $name (rc=$rc), see $QUEUE/logs/$name.log" >&2
      # claim は保持する (無限リトライ防止)。原因調査後に
      # claims/{name} を手で消してキューを再起動するとリトライされる。
      # run_with_gpu.sh の PAUSE (86) / timeout (124) なら worker を止める
      if [ "$rc" -eq 86 ] || [ "$rc" -eq 124 ]; then exit "$rc"; fi
    fi
  done
done
echo "[queue $$] no more claimable shards" >&2
