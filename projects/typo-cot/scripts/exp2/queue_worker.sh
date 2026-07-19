#!/usr/bin/env bash
# 実験2 本番キュー (P1〜P4) のワーカー。exp1+3 の queue_worker.sh 方式を踏襲。
#
# - シャード一覧 (TSV: name / script / done_dir / args) をループ毎に再読込
#   → 行追記でキュー拡張可能 (生成: scripts/exp2/make_queue.py)
# - 冪等スキップ: $PROJ/<done_dir>/summary.json が存在すれば skip
#   (CLI 自体も sample_id ベース resume 対応なので途中死は再実行で継続)
# - 排他: results/prod/exp2/queue/claims/<name> を mkdir で原子的に確保
#   (死んだワーカーの stale claim は pid 生存確認で自動回収)
# - 失敗マーカー: results/prod/exp2/queue/failed/<name> (再試行はこのファイルを削除)
# - GPU は必ず run_with_gpu.sh 経由 (flock 排他)。rc=86 (PAUSED) はワーカー終了、
#   rc=124 (ロック待ちタイムアウト) は failed にせず後で再試行
# - 進捗: results/prod/exp2/queue/progress_<WORKER_ID>.json
#
# 起動 (必ず setsid で切り離す):
#   cd <project> && setsid nohup bash scripts/exp2/queue_worker.sh \
#       < /dev/null >> logs/exp2_queue/worker_w1.log 2>&1 &
# 環境変数:
#   WORKER_ID   ワーカー名 (既定: w<pid>)
#   SHARD_LIST  シャード一覧 TSV (既定: results/prod/exp2/queue/shards_active.tsv)
# 停止: touch results/prod/exp2/queue/STOP (実行中シャードは完走してから終了)
set -u

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh"
QUEUE="$PROJ/results/prod/exp2/queue"
LIST="${SHARD_LIST:-$QUEUE/shards_active.tsv}"
WORKER_ID="${WORKER_ID:-w$$}"
PROG="$QUEUE/progress_${WORKER_ID}.json"

mkdir -p "$QUEUE/claims" "$QUEUE/failed" "$PROJ/logs/exp2_queue"

log() { echo "[$(date -Is)] [$WORKER_ID] $*"; }

progress() { # progress <status> <shard>
    python3 - "$PROG" "$WORKER_ID" "$$" "$1" "$2" <<'PY'
import datetime, json, os, sys
path, wid, pid, status, shard = sys.argv[1:6]
d = {}
if os.path.exists(path):
    try:
        d = json.load(open(path))
    except Exception:
        d = {}
now = datetime.datetime.now().isoformat(timespec="seconds")
d.update({"worker_id": wid, "pid": int(pid), "sid": os.getsid(int(pid)),
          "status": status, "shard": shard, "time": now})
d.setdefault("history", []).append({"t": now, "status": status, "shard": shard})
json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
PY
}

log "worker start (pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ') list=$LIST)"
progress start ""

while :; do
    if [ -f "$QUEUE/STOP" ]; then
        log "STOP file found; exiting"
        progress stopped ""
        exit 0
    fi

    ran=0
    pending=0
    while IFS=$'\t' read -r name script done_dir args; do
        [ -z "${name:-}" ] && continue
        case "$name" in \#*) continue ;; esac
        [ -f "$PROJ/$done_dir/summary.json" ] && continue   # 冪等スキップ
        [ -f "$QUEUE/failed/$name" ] && continue            # 失敗は手動再試行
        pending=1

        claim="$QUEUE/claims/$name"
        if ! mkdir "$claim" 2>/dev/null; then
            cpid="$(cat "$claim/pid" 2>/dev/null || true)"
            if [ -n "$cpid" ] && kill -0 "$cpid" 2>/dev/null; then
                continue                                    # 他ワーカーが実行中
            fi
            log "stale claim on $name (pid=${cpid:-?}); reclaiming"
            rm -rf "$claim"
            mkdir "$claim" 2>/dev/null || continue
        fi
        echo "$$" > "$claim/pid"
        echo "$WORKER_ID" > "$claim/worker"
        date -Is > "$claim/started"

        shard_log="$PROJ/logs/exp2_queue/$name.log"
        log "running $name (log: $shard_log)"
        progress running "$name"
        # shellcheck disable=SC2086 -- args は自前生成 TSV の制御された引数列
        (cd "$PROJ" && bash "$HELPER" uv run python "$script" $args) \
            >> "$shard_log" 2>&1
        rc=$?

        if [ "$rc" -eq 0 ] && [ -f "$PROJ/$done_dir/summary.json" ]; then
            log "done $name"
            progress done "$name"
        elif [ "$rc" -eq 86 ]; then
            log "GPU PAUSED (rc=86); exiting worker (shard $name left pending)"
            progress paused "$name"
            rm -rf "$claim"
            exit 86
        elif [ "$rc" -eq 124 ]; then
            log "GPU lock timeout on $name (rc=124); will retry later"
            progress lock_timeout "$name"
            rm -rf "$claim"
            sleep 120
        else
            log "FAILED $name (rc=$rc); marker: $QUEUE/failed/$name"
            echo "rc=$rc $(date -Is)" > "$QUEUE/failed/$name"
            progress failed "$name"
        fi
        rm -rf "$claim"
        ran=1
        break   # 一覧を先頭から再読込 (追記・完了を反映)
    done < "$LIST"

    if [ "$ran" -eq 0 ]; then
        if [ "$pending" -eq 0 ]; then
            log "no pending shards; exiting"
            progress all_done ""
            exit 0
        fi
        sleep 60   # 全 pending が他ワーカーの claim 下 → 待って再確認
    fi
done
