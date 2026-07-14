#!/usr/bin/env bash
# tmux 内で実行する一括実行ラッパ. GPU 0 を使い、論文準拠スコープで3パス回す.
# Pass 1: importance × k∈{1,2,4,8}  (paper main analysis / Table 5 / Figure 2)
# Pass 2: random/bottom_k × k=4     (paper Section 3.2 + Appendix F)
# Pass 3: Phase 4-5 のみ            (analysis + figures, 除外フィルタ反映)
#
# ログは /tmp/jsai_pipeline_<timestamp>.log に tee. すべて非ストップ.

set -uo pipefail   # 失敗時に途中で止めず errexit は外す。各 phase は独立で扱う。

cd "$(dirname "$0")/.."

LOG_DIR="${LOG_DIR:-/tmp}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/jsai_pipeline_${TS}.log"

run_pass() {
    local pass_name="$1"; shift
    {
        echo ""
        echo "============================================================"
        echo "[$(date)] ${pass_name} 開始"
        echo "  args: $*"
        echo "============================================================"
    } | tee -a "${LOG}"
    "$@" 2>&1 | tee -a "${LOG}"
    local rc=${PIPESTATUS[0]}
    {
        echo "[$(date)] ${pass_name} 終了 (rc=${rc})"
    } | tee -a "${LOG}"
    return "${rc}"
}

{
    echo "============================================================"
    echo "JSAI2026 一括実行パイプライン"
    echo "開始: $(date)"
    echo "ログ: ${LOG}"
    echo "GPU : 0"
    echo "============================================================"
} | tee "${LOG}"

# Pass 1: paper main (LXT at all k)
GPU_ID=0 PERTURBATION_MODES="importance" K_LIST="1 2 4 8" SKIP_PHASES="4,5" \
    run_pass "Pass 1 (importance @ k=1,2,4,8)" \
    bash scripts/run_full_pipeline.sh
rc1=$?

# Pass 2: paper auxiliary (Random + Anti-LXT at k=4)
GPU_ID=0 PERTURBATION_MODES="random bottom_k" K_LIST="4" SKIP_PHASES="4,5" \
    run_pass "Pass 2 (random/bottom_k @ k=4)" \
    bash scripts/run_full_pipeline.sh
rc2=$?

# Pass 3: Phase 4-5 (analysis + figures) — exclude_no_answer_span フィルタ反映
GPU_ID=0 SKIP_PHASES="1,2,3" \
    run_pass "Pass 3 (Phase 4-5 only)" \
    bash scripts/run_full_pipeline.sh
rc3=$?

{
    echo "============================================================"
    echo "完了: $(date)"
    echo "  Pass 1 (importance):    rc=${rc1}"
    echo "  Pass 2 (random/bottom): rc=${rc2}"
    echo "  Pass 3 (Phase 4-5):     rc=${rc3}"
    echo "  Figure/Table 出力: outputs/figures/"
    echo "  実行ログ: ${LOG}"
    echo "============================================================"
} | tee -a "${LOG}"
