#!/usr/bin/env bash
# 実験10④: Qwen2.5-7B-Instruct 摂動側生成キュー (GPU 3/4/5/6, ヘルパー排他)
#
# 10条件 = {gsm8k, mmlu, mmlu_pro, arc, commonsense_qa} × {LXT-4, Random-4}
# R_Q はアーカイブ baseline に完備 → 摂動データセットは CPU で事前作成済み前提。
#
# パイプライン (driver ごとに直列):
#   摂動データセットが無い場合は create_datasets.py で作成 (CPU)
#   → 摂動側生成シャード × 各ベンチ → merge
#
# driver A = importance (LXT-4) 系統 / driver B = random (Random-4) 系統。
#
# 使い方 (プロジェクトルートから。再開も同じコマンド=完了済みシャードは自動スキップ):
#   setsid nohup bash scripts/exp10_qwen_perturbed/run_queue.sh A \
#     >> logs/exp10_qwen_perturbed/driverA.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/exp10_qwen_perturbed/run_queue.sh B \
#     >> logs/exp10_qwen_perturbed/driverB.log 2>&1 < /dev/null &
#
# シャード境界 (batch_size=1, AttnLRP, 1シャード ≈ 30-90min 目安):
#   gsm8k (1319):  100刻み + 末尾 → 14シャード
#   mmlu (5700):   200刻み → 29シャード
#   mmlu_pro (1400): 100刻み → 14シャード
#   arc (1172):    100刻み + 末尾 → 12シャード
#   commonsense_qa (1221): 100刻み + 末尾 → 13シャード
set -u

DRIVER="${1:?usage: run_queue.sh <A|B>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HELPER=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
export GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-86400}"
export PYTHONHASHSEED=42
LOGDIR="$ROOT/logs/exp10_qwen_perturbed"
PROGRESS="$LOGDIR/progress_${DRIVER}.json"
mkdir -p "$LOGDIR"

MODEL_HF="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT="Qwen2.5-7B-Instruct"

if [ "$DRIVER" = "A" ]; then
  COND=importance
  DS_SUFFIX="k4_with_choices"
else
  COND=random
  DS_SUFFIX="k4_random_with_choices"
fi

# "ベンチ|シャード境界(カンマ区切り)"
BENCHES=(
  "gsm8k|0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1319"
  "mmlu|0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5700"
  "mmlu_pro|0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400"
  "arc|0,100,200,300,400,500,600,700,800,900,1000,1100,1172"
  "commonsense_qa|0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1221"
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
      log "再開方法: setsid nohup bash scripts/exp10_qwen_perturbed/run_queue.sh $DRIVER >> logs/exp10_qwen_perturbed/driver${DRIVER}.log 2>&1 < /dev/null &"
      exit 86
    fi
    log "FAIL rc=$rc: $task (60s後にリトライ)"
    sleep 60
  done
  # 3回失敗: 部分結果があれば partial として続行
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

# === 摂動データセット作成 (CPU、最初に全ベンチ分) ===
# driver A のみで作成 (driver B は完了待ち)
if [ "$DRIVER" = "A" ]; then
  for entry in "${BENCHES[@]}"; do
    IFS='|' read -r BENCH _ <<<"$entry"
    DS_LXT="datasets/perturbed/${MODEL_SHORT}_${BENCH}_k4_with_choices/perturbed_dataset.json"
    DS_RND="datasets/perturbed/${MODEL_SHORT}_${BENCH}_k4_random_with_choices/perturbed_dataset.json"
    if [ ! -f "$DS_LXT" ] || [ ! -f "$DS_RND" ]; then
      run_cpu "create_ds/${BENCH}" "" \
        uv run --no-sync python scripts/exp10_qwen_perturbed/create_datasets.py \
        --benchmarks "$BENCH"
    else
      log "SKIP (存在): create_ds/${BENCH}"
      progress "create_ds/${BENCH}" done
    fi
  done
fi

# driver B: 摂動データセットが全ベンチ揃うまで待機
if [ "$DRIVER" = "B" ]; then
  for entry in "${BENCHES[@]}"; do
    IFS='|' read -r BENCH _ <<<"$entry"
    DS="datasets/perturbed/${MODEL_SHORT}_${BENCH}_${DS_SUFFIX}/perturbed_dataset.json"
    while [ ! -f "$DS" ]; do
      log "待機中: $DS (driver A がデータセット作成中)"
      sleep 30
    done
  done
  log "全摂動データセット確認完了、生成キュー開始"
fi

# === 摂動側生成 ===
for entry in "${BENCHES[@]}"; do
  IFS='|' read -r BENCH BOUNDS <<<"$entry"
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
      uv run --no-sync python scripts/run_inference.py \
      --model "$MODEL_HF" --benchmark "$BENCH" --perturbed_data "$DS" \
      --batch_size 1 --seed 42 --max_new_tokens 512 --no_heatmaps \
      --start "$s" --end "$e"
  done

  run_cpu "pert_${COND}_merge/${BENCH}" "$PDIR/summary.json" \
    uv run --no-sync python scripts/run_inference.py \
    --model "$MODEL_HF" --benchmark "$BENCH" --perturbed_data "$DS" \
    --batch_size 1 --merge

  log "ベンチ完了: $BENCH ($COND)"
done

log "driver$DRIVER ($COND) 全タスク完了"
progress "driver_${DRIVER}" all_done
