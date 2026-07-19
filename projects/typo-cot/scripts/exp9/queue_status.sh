#!/usr/bin/env bash
# 実験9 キューの進捗を一覧表示する (読み取り専用)。
# 使い方: bash scripts/exp9/queue_status.sh
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUEUE="$PROJ/results/exp9/queue"
LIST="${SHARD_LIST:-$QUEUE/shards_active.tsv}"

[ -f "$LIST" ] || { echo "shard list not found: $LIST"; exit 1; }

done_n=0; failed_n=0; running_n=0; pending_n=0
running_names=(); failed_names=(); pending_names=()
while IFS=$'\t' read -r name _rest; do
    [ -z "${name:-}" ] && continue
    case "$name" in \#*) continue ;; esac
    if [ -f "$PROJ/results/exp9/summary_${name}.json" ]; then
        done_n=$((done_n + 1))
    elif [ -f "$QUEUE/failed/$name" ]; then
        failed_n=$((failed_n + 1)); failed_names+=("$name")
    elif [ -d "$QUEUE/claims/$name" ]; then
        running_n=$((running_n + 1)); running_names+=("$name")
    else
        pending_n=$((pending_n + 1)); pending_names+=("$name")
    fi
done < "$LIST"

echo "=== exp9 queue status ($(date -Is)) ==="
echo "list: $LIST"
echo "done=$done_n running=$running_n pending=$pending_n failed=$failed_n"
[ "$running_n" -gt 0 ] && printf 'running: %s\n' "${running_names[@]}"
[ "$failed_n" -gt 0 ] && printf 'FAILED : %s\n' "${failed_names[@]}"
[ "$pending_n" -gt 0 ] && [ "$pending_n" -le 10 ] && printf 'pending: %s\n' "${pending_names[@]}"

echo "--- workers ---"
for p in "$QUEUE"/progress_*.json; do
    [ -f "$p" ] || continue
    python3 - "$p" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
alive = os.path.exists(f"/proc/{d['pid']}")
print(f"{d['worker_id']}: pid={d['pid']} sid={d.get('sid', '?')} alive={alive} "
      f"status={d['status']} shard={d['shard'] or '-'} at {d['time']}")
PY
done
