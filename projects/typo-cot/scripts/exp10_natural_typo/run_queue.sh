#!/usr/bin/env bash
# 実験10④: 自然typo (B側) 摂動側生成キュー (GPU 3/4/5/6, ヘルパー排他)
#
# 2条件 = {gsm8k, mmlu} × natural (A側 = アーカイブ LXT-4 生成ログを流用、再生成不要)
# 摂動データセットは CPU で事前作成済み前提 (create_datasets.py)。
#
# driver G = gsm8k / driver M = mmlu (並列実行可、ヘルパーが別GPUを確保)。
#
# 使い方 (プロジェクトルートから。再開も同じコマンド=完了済みシャードは自動スキップ):
#   setsid nohup bash scripts/exp10_natural_typo/run_queue.sh G \
#     >> logs/exp10_natural_typo/driverG.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/exp10_natural_typo/run_queue.sh M \
#     >> logs/exp10_natural_typo/driverM.log 2>&1 < /dev/null &
set -u

DRIVER="${1:?usage: run_queue.sh <G|M>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HELPER=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
export GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-86400}"
export PYTHONHASHSEED=42
LOGDIR="$ROOT/logs/exp10_natural_typo"
PROGRESS="$LOGDIR/progress_${DRIVER}.json"
mkdir -p "$LOGDIR"

MODEL_HF="google/gemma-3-4b-it"
MODEL_SHORT="gemma-3-4b-it"
BATCH_SIZE=8

if [ "$DRIVER" = "G" ]; then
  BENCH=gsm8k
  BOUNDS="0,440,880,1319"
else
  BENCH=mmlu
  BOUNDS="0,475,950,1425,1900,2375,2850"
fi

DS="datasets/perturbed/${MODEL_SHORT}_${BENCH}_k4_natural_with_choices/perturbed_dataset.json"
PDIR="outputs/perturbed/${MODEL_SHORT}_${BENCH}_k4_natural"

log() { echo "[$(date '+%F %T')] [driver$DRIVER/$BENCH] $*"; }

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

run_shard() { # run_shard <task_id> <start> <end>
  local task="$1" start="$2" end="$3" attempt rc
  if shard_complete "$PDIR" "$start" "$end"; then
    log "SKIP (完了済み): $task"
    progress "$task" done
    return 0
  fi
  for attempt in 1 2 3; do
    progress "$task" "running(attempt$attempt)"
    log "RUN [$attempt/3]: $task"
    bash "$HELPER" uv run --no-sync python scripts/exp10_natural_typo/run_generation.py \
      --model "$MODEL_HF" --benchmark "$BENCH" --perturbed_data "$DS" \
      --batch_size "$BATCH_SIZE" --seed 42 --max_new_tokens 512 \
      --start "$start" --end "$end"
    rc=$?
    if [ "$rc" -eq 0 ] && shard_complete "$PDIR" "$start" "$end"; then
      progress "$task" done
      log "DONE: $task"
      return 0
    fi
    if [ "$rc" -eq 86 ]; then
      progress "$task" paused
      log "PAUSED: GPU作業がユーザー指示で一時停止中 (SMOKE_PAUSED)。キューを終了します。"
      exit 86
    fi
    log "FAIL rc=$rc: $task (60s後にリトライ)"
    sleep 60
  done
  progress "$task" failed
  log "ABORT: $task が3回失敗。キュー停止"
  exit 1
}

if [ ! -f "$DS" ]; then
  log "ABORT: 摂動データセットがありません: $DS (create_datasets.py を先に実行)"
  exit 1
fi

IFS=',' read -ra B <<<"$BOUNDS"
for ((i = 0; i < ${#B[@]} - 1; i++)); do
  run_shard "natural/${BENCH}/${B[i]}-${B[i + 1]}" "${B[i]}" "${B[i + 1]}"
done

progress "natural_merge/${BENCH}" running
if uv run --no-sync python scripts/exp10_natural_typo/run_generation.py \
  --model "$MODEL_HF" --benchmark "$BENCH" --perturbed_data "$DS" --merge; then
  progress "natural_merge/${BENCH}" done
else
  progress "natural_merge/${BENCH}" failed
  log "ABORT: merge 失敗"
  exit 1
fi

log "driver$DRIVER ($BENCH) 全タスク完了"
progress "driver_${DRIVER}" all_done
