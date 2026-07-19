#!/usr/bin/env bash
# 実験2 本番キューの状態表示 (読み取りのみ)。
# 使用: bash scripts/exp2/queue_status.sh
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUEUE="$PROJ/results/prod/exp2/queue"
LIST="${SHARD_LIST:-$QUEUE/shards_active.tsv}"

[ -f "$LIST" ] || { echo "no shard list: $LIST"; exit 1; }

done_n=0; run_n=0; fail_n=0; pend_n=0
while IFS=$'\t' read -r name script done_dir args; do
    [ -z "${name:-}" ] && continue
    case "$name" in \#*) continue ;; esac
    if [ -f "$PROJ/$done_dir/summary.json" ]; then
        st="DONE"; done_n=$((done_n+1))
    elif [ -f "$QUEUE/failed/$name" ]; then
        st="FAILED($(cat "$QUEUE/failed/$name"))"; fail_n=$((fail_n+1))
    elif [ -d "$QUEUE/claims/$name" ]; then
        cpid="$(cat "$QUEUE/claims/$name/pid" 2>/dev/null || echo '?')"
        if [ "$cpid" != "?" ] && kill -0 "$cpid" 2>/dev/null; then
            n_rec="?"
            if [ -f "$PROJ/$done_dir/results.json" ]; then
                n_rec=$(python3 -c "import json;print(len(json.load(open('$PROJ/$done_dir/results.json'))))" 2>/dev/null || echo "?")
            fi
            st="RUNNING(pid=$cpid records=$n_rec)"; run_n=$((run_n+1))
        else
            st="STALE_CLAIM(pid=$cpid)"; pend_n=$((pend_n+1))
        fi
    else
        st="pending"; pend_n=$((pend_n+1))
    fi
    printf "%-45s %s\n" "$name" "$st"
done < "$LIST"

echo "----"
echo "done=$done_n running=$run_n failed=$fail_n pending=$pend_n"
[ -f "$QUEUE/STOP" ] && echo "STOP file present!"
for p in "$QUEUE"/progress_*.json; do
    [ -f "$p" ] || continue
    python3 -c "import json;d=json.load(open('$p'));print(f\"worker {d['worker_id']}: {d['status']} {d.get('shard','')} (pid={d['pid']} sid={d['sid']} at {d['time']})\")" 2>/dev/null
done
