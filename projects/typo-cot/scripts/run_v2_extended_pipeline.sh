#!/usr/bin/env bash
# v2 実行スクリプト: 論文版 5 モデル + 拡張モデル（Gemma 12B/27B + Qwen 2.5 全サイズ）
#
# このファイルは git 追跡される実験記録ファイル。MODELS と BENCHMARKS は
# 「どの組合せで一括実行を行ったか」が後から明確に分かるよう、ハードコード
# している（環境変数による override 可）。
#
# 実行内容:
#   Pass 1: importance × k∈{1,2,4,8}
#   Pass 2: random/bottom_k × k=4
#   Pass 3: Phase 4-5（分析+図表生成。除外フィルタ反映）
#
# GPU 構成:
#   既定 GPU_ID="0,1" → 2 GPU で device_map="auto" 分散読込み.
#   AttnLRP の gradient 計算分込みで Gemma-3 27B / Qwen-2.5 32B も
#   2 GPU (合計 ~190GB) なら載る想定. 単体 GPU(95GB) では 27B/32B は
#   AttnLRP backward pass で OOM するため要 2 GPU.
#
# 使用例:
#   bash scripts/run_v2_extended_pipeline.sh
#   GPU_ID="0" bash scripts/run_v2_extended_pipeline.sh    # 小型モデルのみ
#   SKIP_PHASES="1,2,3" bash scripts/run_v2_extended_pipeline.sh    # Phase 4-5 のみ

set -uo pipefail

cd "$(dirname "$0")/.."

# ───── 実験対象（v2） ─────────────────────────────────────────────────
# 論文準拠の 5 モデル（v1 と同じ）
MODELS_V1=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
# v2 で追加するモデル（27B/32B は実用時間制約により除外）
MODELS_V2_ADD=(
    # Gemma-3 同ファミリーの大型版（27B 除外: 単 run に 19h+ かかるため）
    "google/gemma-3-12b-it"
    # Qwen-2.5 Instruct 系列（32B 除外: 同様に時間過大）
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)
ALL_MODELS=("${MODELS_V1[@]}" "${MODELS_V2_ADD[@]}")

# NLP 網羅性確保のため以下を追加:
# - BBH (BIG-Bench Hard): CoT 評価の標準. 23 サブタスクの推論問題.
# - MATH (Hendrycks): 高校数学コンテスト. GSM8K より難しい数学推論.
#
# 廃止: SQuAD v2 (1サンプル 20-30秒で実用時間制約に合わず),
#       StrategyQA (1300+サンプル × 全モデルで時間過大)
BENCHMARKS_LIST=(gsm8k mmlu mmlu_pro arc commonsense_qa bbh math)

# 環境変数で override 可
MODELS="${MODELS:-${ALL_MODELS[*]}}"
BENCHMARKS="${BENCHMARKS:-${BENCHMARKS_LIST[*]}}"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
FIGURES_DIR="${FIGURES_DIR:-${OUTPUTS_ROOT}/figures}"

LOG_DIR="${LOG_DIR:-/tmp}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/jsai_pipeline_v2_${TS}.log"

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
    echo "JSAI2026 拡張モデル一括実行パイプライン (v2)"
    echo "開始: $(date)"
    echo "ログ: ${LOG}"
    echo "GPU : ${GPU_ID}"
    echo "------ MODELS (${#ALL_MODELS[@]}) ------"
    for m in $MODELS; do echo "  - $m"; done
    echo "------ BENCHMARKS ------"
    for b in $BENCHMARKS; do echo "  - $b"; done
    echo "============================================================"
} | tee "${LOG}"

# Pass 1: importance @ k=1,2,4,8
MODELS="${MODELS}" BENCHMARKS="${BENCHMARKS}" \
GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" \
OUTPUTS_ROOT="${OUTPUTS_ROOT}" FIGURES_DIR="${FIGURES_DIR}" \
PERTURBATION_MODES="importance" K_LIST="1 2 4 8" SKIP_PHASES="4,5" \
    run_pass "Pass 1 (importance @ k=1,2,4,8)" \
    bash scripts/run_full_pipeline.sh
rc1=$?

# Pass 2: random/bottom_k @ k=4
MODELS="${MODELS}" BENCHMARKS="${BENCHMARKS}" \
GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" \
OUTPUTS_ROOT="${OUTPUTS_ROOT}" FIGURES_DIR="${FIGURES_DIR}" \
PERTURBATION_MODES="random bottom_k" K_LIST="4" SKIP_PHASES="4,5" \
    run_pass "Pass 2 (random/bottom_k @ k=4)" \
    bash scripts/run_full_pipeline.sh
rc2=$?

# Pass 3: Phase 4-5 のみ
MODELS="${MODELS}" BENCHMARKS="${BENCHMARKS}" \
GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" \
OUTPUTS_ROOT="${OUTPUTS_ROOT}" FIGURES_DIR="${FIGURES_DIR}" \
SKIP_PHASES="1,2,3" \
    run_pass "Pass 3 (Phase 4-5 only)" \
    bash scripts/run_full_pipeline.sh
rc3=$?

{
    echo "============================================================"
    echo "完了: $(date)"
    echo "  Pass 1 (importance):    rc=${rc1}"
    echo "  Pass 2 (random/bottom): rc=${rc2}"
    echo "  Pass 3 (Phase 4-5):     rc=${rc3}"
    echo "  Figure/Table 出力: ${FIGURES_DIR}"
    echo "  実行ログ: ${LOG}"
    echo "============================================================"
} | tee -a "${LOG}"
