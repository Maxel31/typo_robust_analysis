#!/usr/bin/env bash
# v2 並列ランチャー: GPU 0 / GPU 1 にモデルを分割して並行実行.
#
# 分割:
#   GPU 0: Llama + Gemma 系 (5 モデル)
#   GPU 1: Mistral + Qwen 系 (5 モデル)
#
# 構成:
#   1. GPU 0/1 で Phase 1-3（推論）を並列に実行（各々 SKIP_PHASES="4,5"）
#   2. 両プロセス完了後、シリアルに Phase 4-5（分析+図表）を 1 回だけ実行
#
# 使用例:
#   bash scripts/run_v2_parallel.sh

set -uo pipefail

cd "$(dirname "$0")/.."

# ───── モデル分割（GPU ごとに均衡） ───────────────────────────────────
MODELS_GPU0=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
)
MODELS_GPU1=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

BENCHMARKS_LIST=(gsm8k mmlu mmlu_pro arc commonsense_qa bbh math)

LOG_DIR="${LOG_DIR:-/tmp}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_GPU0="${LOG_DIR}/jsai_v2_par_gpu0_${TS}.log"
LOG_GPU1="${LOG_DIR}/jsai_v2_par_gpu1_${TS}.log"
LOG_FINAL="${LOG_DIR}/jsai_v2_par_final_${TS}.log"

run_one_gpu() {
    # 1 GPU 内で Pass 1 (importance) → Pass 2 (random/bottom_k) を順次実行
    # Phase 4-5 はスキップ（後で一括）
    local gpu="$1"
    local models="$2"
    local benchmarks="$3"

    # Pass 1: importance × k=1,2,4,8
    MODELS="${models}" BENCHMARKS="${benchmarks}" \
        GPU_ID="${gpu}" BATCH_SIZE="1" \
        PERTURBATION_MODES="importance" K_LIST="1 2 4 8" SKIP_PHASES="4,5" \
        bash scripts/run_full_pipeline.sh
    echo "[GPU ${gpu}] Pass 1 終了 (rc=$?)"

    # Pass 2: random/bottom_k × k=4
    MODELS="${models}" BENCHMARKS="${benchmarks}" \
        GPU_ID="${gpu}" BATCH_SIZE="1" \
        PERTURBATION_MODES="random bottom_k" K_LIST="4" SKIP_PHASES="4,5" \
        bash scripts/run_full_pipeline.sh
    echo "[GPU ${gpu}] Pass 2 終了 (rc=$?)"
}

MODELS_GPU0_STR="${MODELS_GPU0[*]}"
MODELS_GPU1_STR="${MODELS_GPU1[*]}"
BENCHMARKS_STR="${BENCHMARKS_LIST[*]}"

echo "============================================================"
echo "v2 並列パイプライン起動"
echo "開始: $(date)"
echo "------ GPU 0 (Llama + Gemma) ------"
for m in "${MODELS_GPU0[@]}"; do echo "  - $m"; done
echo "  log: ${LOG_GPU0}"
echo "------ GPU 1 (Mistral + Qwen) ------"
for m in "${MODELS_GPU1[@]}"; do echo "  - $m"; done
echo "  log: ${LOG_GPU1}"
echo "------ BENCHMARKS (7) ------"
for b in "${BENCHMARKS_LIST[@]}"; do echo "  - $b"; done
echo "============================================================"

# GPU 0 を background で起動
run_one_gpu "0" "${MODELS_GPU0_STR}" "${BENCHMARKS_STR}" > "${LOG_GPU0}" 2>&1 &
PID_GPU0=$!

# GPU 1 を background で起動
run_one_gpu "1" "${MODELS_GPU1_STR}" "${BENCHMARKS_STR}" > "${LOG_GPU1}" 2>&1 &
PID_GPU1=$!

echo "[PID] GPU 0: ${PID_GPU0}, GPU 1: ${PID_GPU1}"
echo "両プロセスの完了を待機中..."

wait "${PID_GPU0}"
rc0=$?
echo "[$(date)] GPU 0 プロセス完了 (rc=${rc0})"

wait "${PID_GPU1}"
rc1=$?
echo "[$(date)] GPU 1 プロセス完了 (rc=${rc1})"

# Phase 4-5: 分析 + Figure/Table 生成（並列の両結果を統合）
echo "============================================================"
echo "Phase 4-5: 分析 + Figure/Table 生成"
echo "============================================================"
GPU_ID="0" BATCH_SIZE="1" SKIP_PHASES="1,2,3" \
    bash scripts/run_full_pipeline.sh 2>&1 | tee "${LOG_FINAL}"
rc_final=$?

echo "============================================================"
echo "並列パイプライン完了: $(date)"
echo "  GPU 0 推論プロセス: rc=${rc0}, log=${LOG_GPU0}"
echo "  GPU 1 推論プロセス: rc=${rc1}, log=${LOG_GPU1}"
echo "  Phase 4-5 統合: rc=${rc_final}, log=${LOG_FINAL}"
echo "  Figure/Table 出力: outputs/figures/"
echo "============================================================"
