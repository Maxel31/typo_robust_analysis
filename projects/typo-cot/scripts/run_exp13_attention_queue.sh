#!/usr/bin/env bash
# 実験13: 答え->CoT attention 集中度代理キュー.
#
# 既定は 5モデル x {gsm8k, mmlu} = 10 設定 (LOO と同一 seed=42/n=300 で
# サンプルを揃え、LOO Gini との代理妥当性を per-sample/設定レベルで検証)。
# --all で arc/commonsense_qa/mmlu_pro を加え 25 設定へ拡張 (削除RD との
# 広い突合用、GPU 軽負荷: forward 1回/サンプル)。
#
# GPU は必ず tmp/gpu-locks/run_with_gpu.sh 経由。冪等: summary.json でスキップ。
# 多重起動安全: mkdir claim。progress の updated は ISO-8601。
# 使い方 (projects/typo-cot から):
#   setsid nohup bash scripts/run_exp13_attention_queue.sh [--all] \
#     > results/attention/queue/logs/worker_$$.out 2>&1 &
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJ/../../../../.." && pwd)"
LOCKS="$REPO_ROOT/tmp/gpu-locks"
ARCHIVE="/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs"
QUEUE="$PROJ/results/attention/queue"
N=300
SEED=42

BENCHES=(gsm8k mmlu)
if [ "${1:-}" = "--all" ]; then BENCHES=(gsm8k mmlu arc commonsense_qa mmlu_pro); fi

declare -A HF=(
  ["gemma-3-1b-it"]="google/gemma-3-1b-it"
  ["gemma-3-4b-it"]="google/gemma-3-4b-it"
  ["Llama-3.2-1B-Instruct"]="meta-llama/Llama-3.2-1B-Instruct"
  ["Llama-3.2-3B-Instruct"]="meta-llama/Llama-3.2-3B-Instruct"
  ["Mistral-7B-Instruct-v0.3"]="mistralai/Mistral-7B-Instruct-v0.3"
)
MODELS=(gemma-3-1b-it gemma-3-4b-it Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Mistral-7B-Instruct-v0.3)

mkdir -p "$QUEUE/claims" "$QUEUE/progress" "$QUEUE/logs"

progress() {  # name status rc
  local name="$1" status="$2" rc="${3:-null}"
  cat > "$QUEUE/progress/$name.json" <<EOF
{"shard": "$name", "status": "$status", "rc": $rc,
 "worker_pid": $$, "updated": "$(date -Iseconds)"}
EOF
}

cd "$PROJ"
ran_any=1
while [ "$ran_any" = 1 ]; do
  ran_any=0
  for model_short in "${MODELS[@]}"; do
    model="${HF[$model_short]}"
    for bench in "${BENCHES[@]}"; do
      name="${model_short}_${bench}"
      run_dir="$ARCHIVE/baseline/${model_short}_${bench}"
      out_dir="$PROJ/results/attention/${model_short}_${bench}_clean_attn"
      [ -d "$run_dir" ] || { echo "[attn $$] skip missing archive $run_dir" >&2; continue; }
      if [ -f "$out_dir/summary.json" ]; then progress "$name" done 0; continue; fi
      if ! mkdir "$QUEUE/claims/$name" 2>/dev/null; then continue; fi
      ran_any=1
      progress "$name" running
      echo "[attn $$] start $name -> $out_dir" >&2
      bash "$LOCKS/run_with_gpu.sh" uv run python scripts/run_attention_concentration.py \
        --run_dir "$run_dir" --clean_run_dir "$run_dir" \
        --model "$model" --benchmark "$bench" \
        --n "$N" --seed "$SEED" \
        --output_dir results/attention --run_label clean_attn \
        > "$QUEUE/logs/$name.log" 2>&1
      rc=$?
      if [ "$rc" -eq 0 ] && [ -f "$out_dir/summary.json" ]; then
        progress "$name" done 0
        echo "[attn $$] done $name" >&2
      else
        progress "$name" failed "$rc"
        echo "[attn $$] FAILED $name (rc=$rc), see $QUEUE/logs/$name.log" >&2
        if [ "$rc" -eq 86 ] || [ "$rc" -eq 124 ]; then exit "$rc"; fi
      fi
    done
  done
done
echo "[attn $$] no more claimable shards" >&2
