#!/usr/bin/env bash
# 論文 (ARR2026 May WIP) と同じ実験フローを一括実行するスクリプト.
#
# 実行内容:
#   Phase 1: ベースライン推論 + AttnLRP 重要度分析
#   Phase 2: 摂動データセット生成（importance, random, bottom_k × k ∈ {1,2,4,8}）
#   Phase 3: 摂動後推論
#   Phase 4: 摂動前後の分析（exp*.json / full_results.json 出力）
#   Phase 5: 論文 Figure 2/3 / Table 3/5/6 を一括生成
#
# 使用例:
#   bash scripts/run_full_pipeline.sh                   # 既定モデル・ベンチマーク
#   MODELS="google/gemma-3-4b-it" BENCHMARKS="mmlu" bash scripts/run_full_pipeline.sh
#   K_LIST="4" PERTURBATION_MODES="importance" bash scripts/run_full_pipeline.sh
#   SKIP_PHASES="1,2,3" bash scripts/run_full_pipeline.sh    # 既存出力から Phase4-5 のみ
#
# 環境変数:
#   MODELS              スペース区切りモデル一覧（HF ID）
#   BENCHMARKS          スペース区切りベンチマーク一覧
#   K_LIST              摂動トークン数 k のスペース区切り（例: "1 2 4 8"）
#   PERTURBATION_MODES  スペース区切り摂動モード（importance / random / bottom_k）
#   GPU_ID              GPU 指定（例: "0" / "0,1"）。指定しなければ "0".
#   BATCH_SIZE          推論バッチサイズ（既定 1）
#   OUTPUTS_ROOT        出力ルート（既定 outputs）
#   FIGURES_DIR         Figure/Table 出力先（既定 ${OUTPUTS_ROOT}/figures）
#   SKIP_PHASES         実行しないフェーズ番号のカンマ区切り（例: "1,2"）

set -euo pipefail

MODELS=${MODELS:-"meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct google/gemma-3-1b-it google/gemma-3-4b-it mistralai/Mistral-7B-Instruct-v0.3"}
BENCHMARKS=${BENCHMARKS:-"gsm8k mmlu mmlu_pro arc commonsense_qa"}
K_LIST=${K_LIST:-"1 2 4 8"}
PERTURBATION_MODES=${PERTURBATION_MODES:-"importance random bottom_k"}
GPU_ID=${GPU_ID:-"0"}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUTS_ROOT=${OUTPUTS_ROOT:-outputs}
FIGURES_DIR=${FIGURES_DIR:-${OUTPUTS_ROOT}/figures}
SKIP_PHASES=${SKIP_PHASES:-""}

BASELINE_DIR=${OUTPUTS_ROOT}/baseline
PERTURBED_DATA_DIR=${PERTURBED_DATA_DIR:-datasets/perturbed}
PERTURBED_DIR=${OUTPUTS_ROOT}/perturbed
ANALYSIS_DIR=${OUTPUTS_ROOT}/analysis

mkdir -p "${BASELINE_DIR}" "${PERTURBED_DATA_DIR}" "${PERTURBED_DIR}" "${ANALYSIS_DIR}" "${FIGURES_DIR}"

skip_phase() {
    [[ ",${SKIP_PHASES}," == *",$1,"* ]]
}

model_basename() {
    local id="$1"
    echo "${id##*/}"
}

perturbation_flag() {
    case "$1" in
        importance) echo "" ;;
        random)     echo "--random_perturbation" ;;
        bottom_k)   echo "--bottom_k" ;;
        *) echo "[ERROR] unknown perturbation mode: $1" >&2; exit 1 ;;
    esac
}

# run_perturbation.py が生成するディレクトリ名のサフィックス。
# importance: なし / random: "_random" / bottom_k: "_bottom_k"
# その後ろに常に "_with_choices" が付く（include_choices のデフォルトが True のため）
perturbation_dir_name() {
    local model_bench="$1" k="$2" mode="$3"
    local mode_suffix=""
    case "${mode}" in
        importance) mode_suffix="" ;;
        random)     mode_suffix="_random" ;;
        bottom_k)   mode_suffix="_bottom_k" ;;
    esac
    echo "${model_bench}_k${k}${mode_suffix}_with_choices"
}

#───────────────────────────── Phase 1: ベースライン推論 ─────────────────────────────
if ! skip_phase 1; then
    echo "===== Phase 1: ベースライン推論 ====="
    for model in ${MODELS}; do
        for bench in ${BENCHMARKS}; do
            mb=$(model_basename "${model}")
            out_dir="${BASELINE_DIR}/${mb}_${bench}"
            if [[ -f "${out_dir}/summary.json" ]]; then
                echo "[skip] ${mb}_${bench} は既に存在: ${out_dir}"
                continue
            fi
            echo "[run]  ${mb}_${bench}"
            uv run python scripts/run_inference.py \
                --model "${model}" \
                --benchmark "${bench}" \
                --batch_size "${BATCH_SIZE}" \
                --gpu_id "${GPU_ID}" \
                --output_dir "${BASELINE_DIR}"
        done
    done
fi

#───────────────────────────── Phase 2: 摂動データセット生成 ─────────────────────────────
if ! skip_phase 2; then
    echo "===== Phase 2: 摂動データセット生成 ====="
    for model in ${MODELS}; do
        for bench in ${BENCHMARKS}; do
            mb=$(model_basename "${model}")
            baseline_subdir="${BASELINE_DIR}/${mb}_${bench}"
            if [[ ! -d "${baseline_subdir}" ]]; then
                echo "[warn] baseline 未生成: ${baseline_subdir}（Phase 1 を先に実行）"
                continue
            fi
            for k in ${K_LIST}; do
                for mode in ${PERTURBATION_MODES}; do
                    dataset_subdir=$(perturbation_dir_name "${mb}_${bench}" "${k}" "${mode}")
                    dataset_path="${PERTURBED_DATA_DIR}/${dataset_subdir}/perturbed_dataset.json"
                    if [[ -f "${dataset_path}" ]]; then
                        echo "[skip] 既存 perturbed dataset: ${dataset_path}"
                        continue
                    fi
                    flag=$(perturbation_flag "${mode}")
                    echo "[run]  ${mb}_${bench} k=${k} ${mode}"
                    uv run python scripts/run_perturbation.py \
                        --baseline_dir "${baseline_subdir}" \
                        --num_perturbations "${k}" \
                        --output_dir "${PERTURBED_DATA_DIR}" \
                        ${flag}
                done
            done
        done
    done
fi

#───────────────────────────── Phase 3: 摂動後推論 ─────────────────────────────
if ! skip_phase 3; then
    echo "===== Phase 3: 摂動後推論 ====="
    for model in ${MODELS}; do
        for bench in ${BENCHMARKS}; do
            mb=$(model_basename "${model}")
            for k in ${K_LIST}; do
                for mode in ${PERTURBATION_MODES}; do
                    dataset_subdir=$(perturbation_dir_name "${mb}_${bench}" "${k}" "${mode}")
                    pert_data="${PERTURBED_DATA_DIR}/${dataset_subdir}/perturbed_dataset.json"
                    if [[ ! -f "${pert_data}" ]]; then
                        echo "[warn] 摂動データ未生成: ${pert_data}"
                        continue
                    fi
                    # 既に Phase 3 出力が存在する場合はスキップ
                    perturbed_out="${PERTURBED_DIR}/${mb}_${bench}_k${k}_${mode}"
                    if [[ -f "${perturbed_out}/summary.json" ]]; then
                        echo "[skip] 既存 perturbed inference: ${perturbed_out}"
                        continue
                    fi
                    echo "[run]  perturbed ${mb}_${bench} k=${k} ${mode}"
                    uv run python scripts/run_inference.py \
                        --model "${model}" \
                        --benchmark "${bench}" \
                        --perturbed_data "${pert_data}" \
                        --batch_size "${BATCH_SIZE}" \
                        --gpu_id "${GPU_ID}" \
                        --output_dir "${PERTURBED_DIR}"
                done
            done
        done
    done
fi

#───────────────────────────── Phase 4: 摂動前後の分析 ─────────────────────────────
if ! skip_phase 4; then
    echo "===== Phase 4: 摂動前後の分析 ====="
    uv run python scripts/run_analysis.py --outputs_dir "${OUTPUTS_ROOT}" --output_dir "${ANALYSIS_DIR}"
fi

#───────────────────────────── Phase 5: 論文 Figure / Table 生成 ─────────────────────────────
if ! skip_phase 5; then
    echo "===== Phase 5: Figure / Table 生成 ====="
    uv run python scripts/build_figures_tables.py \
        --analysis_dir "${ANALYSIS_DIR}" \
        --baseline_dir "${BASELINE_DIR}" \
        --perturbed_dir "${PERTURBED_DIR}" \
        --output_dir "${FIGURES_DIR}"
fi

echo "===== 完了 ====="
echo "Figure/Table 出力: ${FIGURES_DIR}"
