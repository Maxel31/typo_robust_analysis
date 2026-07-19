#!/usr/bin/env bash
# 実験6-(i)〜(iii) 帰属ファミリー本番キュー (M3 x B2 x {clean, LXT-4} x {gxi, rollout, ig}).
#
# 1シャード = 1設定 x 1手法 x 1条件 = run_attribution_family.py 1回 (n=300, seed=42、
# exp6-iv LOO 本番と同一サンプル)。軽い手法 (gxi, rollout) を先に、重い IG を後に並べる。
# GPU は必ず tmp/gpu-locks/run_with_gpu.sh 経由 (GPU 3,4,5,6 を他系統とロック共有)。
#
# 冪等性: 出力ディレクトリの summary.json があるシャードはスキップ。
# 多重起動安全: mkdir によるアトミックなシャード claim。
# 進捗: results/attribution_family/queue/progress/{shard}.json。
#
# 使い方 (worktree の projects/typo-cot から):
#   setsid nohup bash scripts/run_attribution_family_queue.sh [--only SHARD_REGEX] \
#     > results/attribution_family/queue/logs/worker_$$.out 2>&1 &
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJ/../../../../.." && pwd)"  # typo_robust_analysis 本体
LOCKS="$REPO_ROOT/tmp/gpu-locks"
ARCHIVE="/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs"
QUEUE="$PROJ/results/attribution_family/queue"
N=300
SEED=42
IG_STEPS=16
IG_STEP_BATCH=4

ONLY_RE=""
if [ "${1:-}" = "--only" ]; then ONLY_RE="${2:?--only requires a regex}"; fi

mkdir -p "$QUEUE/claims" "$QUEUE/progress" "$QUEUE/logs"

# 設定定義: model|benchmark|model_short
CONFIGS=(
  "google/gemma-3-4b-it|gsm8k|gemma-3-4b-it"
  "google/gemma-3-4b-it|mmlu|gemma-3-4b-it"
  "meta-llama/Llama-3.2-3B-Instruct|gsm8k|Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct|mmlu|Llama-3.2-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3|gsm8k|Mistral-7B-Instruct-v0.3"
  "mistralai/Mistral-7B-Instruct-v0.3|mmlu|Mistral-7B-Instruct-v0.3"
)

# shard 定義: name|model|benchmark|method|run_dir|clean_run_dir|label
SHARDS=()
for method in gxi rollout ig; do
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r model bench short <<< "$cfg"
    mkey="$(echo "$short" | tr '[:upper:]' '[:lower:]' | cut -d- -f1)"
    baseline="$ARCHIVE/baseline/${short}_${bench}"
    perturbed="$ARCHIVE/perturbed/${short}_${bench}_k4_importance"
    SHARDS+=("${mkey}_${bench}_${method}_clean|$model|$bench|$method|$baseline|$baseline|clean")
    SHARDS+=("${mkey}_${bench}_${method}_lxt4|$model|$bench|$method|$perturbed|$baseline|lxt4")
  done
done

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
  for spec in "${SHARDS[@]}"; do
    IFS='|' read -r name model bench method run_dir clean_dir label <<< "$spec"
    if [ -n "$ONLY_RE" ] && ! [[ "$name" =~ $ONLY_RE ]]; then continue; fi
    model_short="${model##*/}"
    out_dir="$PROJ/results/attribution_family/${model_short}_${bench}_${method}_${label}"
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
    bash "$LOCKS/run_with_gpu.sh" uv run --no-sync python scripts/run_attribution_family.py \
      --run_dir "$run_dir" \
      --clean_run_dir "$clean_dir" \
      --model "$model" --benchmark "$bench" --method "$method" \
      --n "$N" --seed "$SEED" \
      --ig_steps "$IG_STEPS" --ig_step_batch "$IG_STEP_BATCH" \
      --output_dir results/attribution_family --run_label "$label" \
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
