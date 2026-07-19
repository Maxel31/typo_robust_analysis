#!/usr/bin/env bash
# 実験10③: DeepSeek-R1-Distill-Qwen-7B 摂動側生成キュー (GPU 3/4/5/6, ヘルパー排他)
#
# 6条件 = {gsm8k, math, mmlu} × {LXT-4(importance), Random-4(random)}。
# シャード境界は clean 生成 (dev_notes_exp10_scope.md) と同一:
#   gsm8k: 220刻み / math: 250刻み / mmlu: 300刻み
#
# driver A = importance (LXT-4) 系統 / driver B = random (Random-4) 系統。
#
# 使い方 (プロジェクトルートから。再開も同じコマンド=完了済みシャードは自動スキップ):
#   setsid nohup bash scripts/exp10_r1_perturbed/run_queue.sh A \
#     >> logs/exp10_r1_perturbed/driverA.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/exp10_r1_perturbed/run_queue.sh B \
#     >> logs/exp10_r1_perturbed/driverB.log 2>&1 < /dev/null &
#
# - 1シャード = 1 GPUヘルパー呼び出し (run_with_gpu.sh が flock 排他)
# - 検証ゲート: 各ドライバの最初のシャード (gsm8k [0,220)) 完了後、
#   scripts/exp10_r1_perturbed/VERIFY_OK が作成されるまで待機。
#   verify_first_shard.py で スキーマ / 摂動適用 / 精度方向 を確認して touch する。
# - 進捗: logs/exp10_r1_perturbed/progress_<driver>.json
set -u

DRIVER="${1:?usage: run_queue.sh <A|B>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HELPER=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
export GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-86400}"
export PYTHONHASHSEED=42
GATE_FILE="$ROOT/scripts/exp10_r1_perturbed/VERIFY_OK"
LOGDIR="$ROOT/logs/exp10_r1_perturbed"
PROGRESS="$LOGDIR/progress_${DRIVER}.json"
mkdir -p "$LOGDIR"

MODEL_SHORT="DeepSeek-R1-Distill-Qwen-7B"

if [ "$DRIVER" = "A" ]; then
  COND=importance
  DS_SUFFIX="k4_with_choices"
else
  COND=random
  DS_SUFFIX="k4_random_with_choices"
fi

# "ベンチ|バッチサイズ|シャード境界(カンマ区切り)" — clean と同一分割
BENCHES=(
  "gsm8k|16|0,220,440,660,880,1100,1319"
  "math|12|0,250,500"
  "mmlu|16|0,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,3900,4200,4500,4800,5100,5400,5700"
)

log() { echo "[$(date '+%F %T')] [driver$DRIVER/$COND] $*"; }

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
      log "再開方法: setsid nohup bash scripts/exp10_r1_perturbed/run_queue.sh $DRIVER >> logs/exp10_r1_perturbed/driver${DRIVER}.log 2>&1 < /dev/null &"
      exit 86
    fi
    log "FAIL rc=$rc: $task (60s後にリトライ)"
    sleep 60
  done
  # 3回失敗: 部分結果があれば partial として続行 (欠損は merge/検証で顕在化)
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

run_cpu() { # run_cpu <task_id> <skip_if_exists> <cmd...>
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

GATED=0
for entry in "${BENCHES[@]}"; do
  IFS='|' read -r BENCH BATCH BOUNDS <<<"$entry"
  IFS=',' read -ra B <<<"$BOUNDS"
  DS="datasets/perturbed/${MODEL_SHORT}_${BENCH}_${DS_SUFFIX}/perturbed_dataset.json"
  PDIR="outputs/perturbed/${MODEL_SHORT}_${BENCH}_k4_${COND}"

  if [ ! -f "$DS" ]; then
    log "ABORT: 摂動データセットがありません: $DS"
    exit 1
  fi

  for ((i = 0; i < ${#B[@]} - 1; i++)); do
    s=${B[i]}
    e=${B[i + 1]}
    run_shard "pert_${COND}/${BENCH}/${s}-${e}" "$PDIR" "$s" "$e" \
      uv run --no-sync python scripts/run_inference_reasoning.py \
      --benchmark "$BENCH" --perturbed_data "$DS" \
      --start "$s" --end "$e" --batch_size "$BATCH" --seed 42

    # 検証ゲート: 各ドライバの最初のシャード完了後
    if [ "$GATED" -eq 0 ]; then
      GATED=1
      if [ ! -f "$GATE_FILE" ]; then
        log "検証ゲート待機: uv run --no-sync python scripts/exp10_r1_perturbed/verify_first_shard.py --condition $COND を実行し、"
        log "問題なければ touch $GATE_FILE で続行"
        progress "gate/${COND}" waiting
        while [ ! -f "$GATE_FILE" ]; do sleep 60; done
      fi
      progress "gate/${COND}" passed
      log "検証ゲート通過"
    fi
  done

  run_cpu "pert_${COND}_merge/${BENCH}" "$PDIR/summary.json" \
    uv run --no-sync python scripts/run_inference_reasoning.py \
    --benchmark "$BENCH" --perturbed_data "$DS" --merge

  log "ベンチ完了: $BENCH ($COND)"
done

log "driver$DRIVER ($COND) 全タスク完了"
progress "driver_${DRIVER}" all_done
