#!/usr/bin/env bash
# 全パターンE2Eテスト: Llama-3.2-3B × (none + 8 quant methods) × 8 benchmarks × 6 typo conditions
# Phase 1: quant環境で量子化 → Phase 2: vllm環境で推論
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

GPU_IDS="0,1,4,5"
MODEL="meta-llama/Llama-3.2-3B"
MODEL_SHORT="Llama-3.2-3B"
QUANTIZED_ROOT="./data/quantized"
CALIBRATION_CLEAN="./data/calibration/fineweb_clean.jsonl"
CALIBRATION_NOISY="./data/calibration/fineweb_noisy.jsonl"
CALIBRATION_TYPES=("clean" "noisy")

BENCHMARKS=(
    "arc_easy"
    "arc_challenge"
    "hellaswag"
    "mmlu"
    "piqa"
    "gsm8k"
    "wikitext2"
    "c4"
)

TYPO_CONDITIONS=(
    "clean:0"
    "swap:1"
    "swap:2"
    "swap:4"
    "random:4"
    "replace:4"
)

QUANTIZATION_METHODS=(
    "none:16"
    "gptq:4"
    "gptq:8"
    "awq:4"
    "awq:8"
    # smoothquant:8 は SM120 (Blackwell) で Int8 非対応のため除外
    "smoothquant:4"
    "qep:4"
    "qep:8"
)

# ============================================================
# Phase 1: Quantization (quant environment)
# ============================================================
echo "=============================================="
echo "Phase 1: Switching to quant environment"
echo "=============================================="
uv sync --package quant-typo-neuron --extra quant

if [ ! -f "$CALIBRATION_CLEAN" ]; then
    echo "Preparing calibration data from FineWeb sample-10BT (clean + noisy)..."
    uv run python scripts/prepare_calibration_data.py \
        --dataset fineweb \
        --num-samples 512 \
        --output "$CALIBRATION_CLEAN" \
        --output-noisy "$CALIBRATION_NOISY" \
        --typo-type random \
        --num-typos 4
fi

for quantization_spec in "${QUANTIZATION_METHODS[@]}"; do
    method="${quantization_spec%%:*}"
    bits="${quantization_spec##*:}"

    if [ "$method" = "none" ]; then
        continue
    fi

    for calibration_type in "${CALIBRATION_TYPES[@]}"; do
        if [ "$calibration_type" = "clean" ]; then
            calibration_data_path="$CALIBRATION_CLEAN"
        else
            calibration_data_path="$CALIBRATION_NOISY"
        fi

        output_dir="${QUANTIZED_ROOT}/${MODEL_SHORT}/${method}_w${bits}_${calibration_type}"
        if [ -d "$output_dir" ]; then
            echo "SKIP quantization: $output_dir already exists"
            continue
        fi

        echo "=== Quantizing: ${MODEL} | method=${method} bits=${bits} calibration=${calibration_type} ==="
        CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run python scripts/quantize_model.py \
            --model "$MODEL" \
            --method "$method" \
            --bits "$bits" \
            --output-dir "$output_dir" \
            --calibration-data "$calibration_data_path" \
            --num-calibration-samples 128 \
            --gpu-ids "$GPU_IDS" \
        || { echo "FAILED quantization: ${method} w${bits} calibration=${calibration_type}"; continue; }
    done
done

# ============================================================
# Phase 2: Evaluation (vllm environment, 4GPU parallel)
# ============================================================
echo ""
echo "=============================================="
echo "Phase 2: Switching to vllm environment"
echo "=============================================="
uv sync --package quant-typo-neuron --extra vllm

IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"
NUM_GPUS=${#GPU_LIST[@]}

JOB_FILE=$(mktemp)
STATUS_DIR=$(mktemp -d)

for quantization_spec in "${QUANTIZATION_METHODS[@]}"; do
    method="${quantization_spec%%:*}"
    bits="${quantization_spec##*:}"

    if [ "$method" = "none" ]; then
        calibration_types_eval=("none")
    else
        calibration_types_eval=("${CALIBRATION_TYPES[@]}")
    fi

    for calibration_type in "${calibration_types_eval[@]}"; do
        if [ "$method" = "none" ]; then
            model_path="$MODEL"
        else
            model_path="${QUANTIZED_ROOT}/${MODEL_SHORT}/${method}_w${bits}_${calibration_type}"
            if [ ! -d "$model_path" ]; then
                echo "SKIP eval: quantized model not found at $model_path"
                continue
            fi
        fi

        for benchmark_name in "${BENCHMARKS[@]}"; do
            for typo_spec in "${TYPO_CONDITIONS[@]}"; do
                typo_type="${typo_spec%%:*}"
                num_typos="${typo_spec##*:}"
                echo "${model_path}|${method}|${bits}|${calibration_type}|${benchmark_name}|${typo_type}|${num_typos}" >> "$JOB_FILE"
            done
        done
    done
done

TOTAL=$(wc -l < "$JOB_FILE")
echo "Total evaluation jobs: ${TOTAL} (running ${NUM_GPUS} GPUs in parallel)"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    : > "${JOB_FILE}.gpu${i}"
done
job_index=0
while IFS= read -r job_line || [ -n "$job_line" ]; do
    gpu_slot=$((job_index % NUM_GPUS))
    echo "$job_line" >> "${JOB_FILE}.gpu${gpu_slot}"
    job_index=$((job_index + 1))
done < "$JOB_FILE"

run_gpu_queue() {
    local gpu_id="$1"
    local queue_file="$2"
    local status_dir="$3"
    local master_port=$((29500 + gpu_id))

    while IFS= read -r job_line || [ -n "$job_line" ]; do
        IFS='|' read -r model_path method bits calibration_type benchmark_name typo_type num_typos <<< "$job_line"
        local job_id="${method}_w${bits}_${calibration_type}__${benchmark_name}__${typo_type}_n${num_typos}"

        echo "[GPU ${gpu_id}] START: ${job_id}"
        if CUDA_VISIBLE_DEVICES="$gpu_id" MASTER_PORT="$master_port" uv run python experiments/run_eval.py \
            --config configs/base_eval.yaml \
            --gpu-ids "$gpu_id" \
            "model.name=$model_path" \
            "model.quantization_method=$method" \
            "model.bits=$bits" \
            "model.calibration=${calibration_type}" \
            "model.tensor_parallel_size=1" \
            "model.gpu_memory_utilization=0.9" \
            "benchmark.name=$benchmark_name" \
            "typo.type=$typo_type" \
            "typo.num_typos=$num_typos" 2>&1; then
            echo "SUCCESS" > "${status_dir}/${job_id}.status"
            echo "[GPU ${gpu_id}] SUCCESS: ${job_id}"
        else
            echo "FAILED" > "${status_dir}/${job_id}.status"
            echo "[GPU ${gpu_id}] FAILED: ${job_id}"
        fi
    done < "$queue_file"
}

for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id="${GPU_LIST[$i]}"
    run_gpu_queue "$gpu_id" "${JOB_FILE}.gpu${i}" "$STATUS_DIR" &
done

wait

SUCCESS=$(grep -rl "SUCCESS" "$STATUS_DIR" 2>/dev/null | wc -l)
FAIL=$(grep -rl "FAILED" "$STATUS_DIR" 2>/dev/null | wc -l)

echo "=============================================="
echo "All E2E tests complete."
echo "Total: ${TOTAL} | Success: ${SUCCESS} | Failed: ${FAIL}"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed jobs:"
    grep -rl "FAILED" "$STATUS_DIR" | while read -r f; do
        echo "  - $(basename "${f%.status}")"
    done
fi
echo "=============================================="

rm -f "$JOB_FILE" "${JOB_FILE}".gpu*
rm -rf "$STATUS_DIR"
