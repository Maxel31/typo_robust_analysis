#!/usr/bin/env bash
# 実験10②: MATH-500 全面新規再生成キュー (M5+Qwen の6モデル、GPU 3/4)
#
# パイプライン (モデルごとに直列):
#   clean 生成シャード → merge → [検証ゲート: 最初のモデルのみ]
#   → LXT-4 / Random-4 摂動データセット作成 (CPU)
#   → 摂動側生成シャード ×2条件 → merge
#
# 使い方 (プロジェクトルートから):
#   nohup bash scripts/exp10_math500/run_queue.sh A > logs/exp10_math500/driverA.log 2>&1 &
#   nohup bash scripts/exp10_math500/run_queue.sh B > logs/exp10_math500/driverB.log 2>&1 &
#
# - 1シャード = 1 GPUヘルパー呼び出し (run_with_gpu.sh が GPU 3/4 を flock 排他)
# - 完了済みシャードはスキップ (再実行安全 = 中断後は同コマンドの再実行で再開)
# - 進捗: logs/exp10_math500/progress_<driver>.json
# - 検証ゲート: 各ドライバの最初のモデルの clean merge 後、
#   scripts/exp10_math500/VERIFY_OK が作成されるまで待機
#   (verify_vs_archive.py でアーカイブと照合してから touch する)
set -u

DRIVER="${1:?usage: run_queue.sh <A|B>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HELPER=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
export GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-86400}"
# 摂動データセット作成の乱択が hash() を使うため再現性のため固定
export PYTHONHASHSEED=42
GATE_FILE="$ROOT/scripts/exp10_math500/VERIFY_OK"
LOGDIR="$ROOT/logs/exp10_math500"
PROGRESS="$LOGDIR/progress_${DRIVER}.json"
mkdir -p "$LOGDIR"

# モデル定義: "HF名|短縮名|シャード境界(カンマ区切り)"
# シャード境界は 1シャード 20〜40分目安 (batch_size=1, AttnLRP込み) で設定
if [ "$DRIVER" = "A" ]; then
  MODELS=(
    "google/gemma-3-1b-it|gemma-3-1b-it|0,250,500"
    "google/gemma-3-4b-it|gemma-3-4b-it|0,125,250,375,500"
    "mistralai/Mistral-7B-Instruct-v0.3|Mistral-7B-Instruct-v0.3|0,84,168,252,336,420,500"
  )
else
  MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct|0,250,500"
    "meta-llama/Llama-3.2-3B-Instruct|Llama-3.2-3B-Instruct|0,125,250,375,500"
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|0,84,168,252,336,420,500"
  )
fi

log() { echo "[$(date '+%F %T')] [driver$DRIVER] $*"; }

progress() { # progress <task_id> <status>
  python3 - "$PROGRESS" "$1" "$2" <<'PY'
import datetime
import json
import pathlib
import sys

path, task, status = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
data = json.loads(p.read_text()) if p.exists() else {}
data[task] = {"status": status, "at": datetime.datetime.now().isoformat()}
p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
PY
}

# shards/results_{start:05d}_*.json が自身の範囲ぶんの行数を持てば完了
# (サンプル欠損で end が切り詰められたシャードも自己申告の範囲で判定)
shard_complete() { # shard_complete <outdir> <start> <end>
  python3 - "$1" "$2" "$3" <<'PY'
import json
import pathlib
import re
import sys

outdir, start = pathlib.Path(sys.argv[1]), int(sys.argv[2])
pattern = re.compile(rf"results_{start:05d}_(\d{{5}})\.json$")
for f in sorted((outdir / "shards").glob(f"results_{start:05d}_*.json")):
    m = pattern.search(f.name)
    if not m:
        continue
    file_end = int(m.group(1))
    try:
        rows = json.load(open(f))
    except Exception:
        continue
    if isinstance(rows, list) and len(rows) >= file_end - start:
        sys.exit(0)
sys.exit(1)
PY
}

run_shard() { # run_shard <task_id> <outdir> <start> <end> <cmd...>
  local task="$1" outdir="$2" start="$3" end="$4" attempt rc
  shift 4
  if shard_complete "$outdir" "$start" "$end"; then
    log "SKIP (完了済み): $task"
    progress "$task" done
    return 0
  fi
  for attempt in 1 2 3; do
    progress "$task" "running(attempt$attempt)"
    log "RUN [$attempt/3]: $task"
    bash "$HELPER" "$@"
    rc=$?
    if [ "$rc" -eq 0 ] && shard_complete "$outdir" "$start" "$end"; then
      progress "$task" done
      log "DONE: $task"
      return 0
    fi
    if [ "$rc" -eq 86 ]; then
      progress "$task" paused
      log "PAUSED: GPU作業がユーザー指示で一時停止中 (SMOKE_PAUSED)。キューを終了します。"
      log "再開方法: nohup bash scripts/exp10_math500/run_queue.sh $DRIVER >> logs/exp10_math500/driver${DRIVER}.log 2>&1 &"
      exit 86
    fi
    log "FAIL rc=$rc: $task (60s後にリトライ)"
    sleep 60
  done
  # 3回失敗: シャードファイルに部分結果があれば partial として続行 (欠損は検証/分析で顕在化する)
  if python3 - "$outdir" "$start" <<'PY'
import json
import pathlib
import sys

outdir, start = pathlib.Path(sys.argv[1]), int(sys.argv[2])
files = sorted((outdir / "shards").glob(f"results_{start:05d}_*.json"))
for f in files:
    try:
        if len(json.load(open(f))) > 0:
            sys.exit(0)
    except Exception:
        continue
sys.exit(1)
PY
  then
    progress "$task" partial
    log "WARN: $task は3回失敗、部分結果ありのため partial として続行"
    return 0
  fi
  progress "$task" failed
  log "ABORT: $task が3回失敗 (結果ゼロ)。依存タスク保護のためキュー停止"
  exit 1
}

run_cpu() { # run_cpu <task_id> <skip_if_exists(空=チェックなし)> <cmd...>
  local task="$1" check="$2"
  shift 2
  if [ -n "$check" ] && [ -f "$check" ]; then
    log "SKIP (存在): $task"
    progress "$task" done
    return 0
  fi
  progress "$task" running
  log "RUN: $task"
  if "$@"; then
    progress "$task" done
    log "DONE: $task"
  else
    progress "$task" failed
    log "ABORT: $task 失敗。キュー停止"
    exit 1
  fi
}

FIRST_MODEL_GATED=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r HF SHORT BOUNDS <<<"$entry"
  IFS=',' read -ra B <<<"$BOUNDS"
  CLEAN_DIR="outputs/baseline/${SHORT}_math"

  # 1) clean 生成シャード
  for ((i = 0; i < ${#B[@]} - 1; i++)); do
    s=${B[i]}
    e=${B[i + 1]}
    run_shard "clean/${SHORT}/${s}-${e}" "$CLEAN_DIR" "$s" "$e" \
      uv run --no-sync python scripts/run_inference.py \
      --model "$HF" --benchmark math --batch_size 1 --seed 42 \
      --max_new_tokens 512 --no_heatmaps --output_dir outputs/baseline \
      --start "$s" --end "$e"
  done

  # 2) clean merge (CPU)
  run_cpu "clean_merge/${SHORT}" "$CLEAN_DIR/summary.json" \
    uv run --no-sync python scripts/run_inference.py \
    --model "$HF" --benchmark math --batch_size 1 --output_dir outputs/baseline --merge

  # 3) 検証ゲート (各ドライバの最初のモデルの clean 完了後)
  if [ "$FIRST_MODEL_GATED" -eq 0 ]; then
    FIRST_MODEL_GATED=1
    if [ ! -f "$GATE_FILE" ]; then
      log "検証ゲート待機: verify_vs_archive.py --model_short $SHORT を実行し、"
      log "問題なければ touch $GATE_FILE で続行"
      progress "gate/${SHORT}" waiting
      while [ ! -f "$GATE_FILE" ]; do sleep 60; done
    fi
    progress "gate/${SHORT}" passed
    log "検証ゲート通過"
  fi

  # 4) 摂動データセット作成 (CPU; LXT-4 と Random-4)
  LXT_DS="datasets/perturbed/${SHORT}_math_k4_with_choices/perturbed_dataset.json"
  RND_DS="datasets/perturbed/${SHORT}_math_k4_random_with_choices/perturbed_dataset.json"
  run_cpu "perturb_ds/${SHORT}/lxt4" "$LXT_DS" \
    uv run --no-sync python scripts/run_perturbation.py \
    --baseline_dir "$CLEAN_DIR" -k 4 --seed 42
  run_cpu "perturb_ds/${SHORT}/random4" "$RND_DS" \
    uv run --no-sync python scripts/run_perturbation.py \
    --baseline_dir "$CLEAN_DIR" -k 4 --seed 42 --random_perturbation

  # 5) 摂動側生成シャード ×2条件 → merge
  for cond in importance random; do
    if [ "$cond" = importance ]; then DS="$LXT_DS"; else DS="$RND_DS"; fi
    PDIR="outputs/perturbed/${SHORT}_math_k4_${cond}"
    for ((i = 0; i < ${#B[@]} - 1; i++)); do
      s=${B[i]}
      e=${B[i + 1]}
      run_shard "pert_${cond}/${SHORT}/${s}-${e}" "$PDIR" "$s" "$e" \
        uv run --no-sync python scripts/run_inference.py \
        --model "$HF" --benchmark math --perturbed_data "$DS" \
        --batch_size 1 --seed 42 --max_new_tokens 512 --no_heatmaps \
        --start "$s" --end "$e"
    done
    run_cpu "pert_${cond}_merge/${SHORT}" "$PDIR/summary.json" \
      uv run --no-sync python scripts/run_inference.py \
      --model "$HF" --benchmark math --perturbed_data "$DS" --batch_size 1 --merge
  done

  log "モデル完了: $SHORT"
done

log "driver$DRIVER 全タスク完了"
progress "driver_${DRIVER}" all_done
