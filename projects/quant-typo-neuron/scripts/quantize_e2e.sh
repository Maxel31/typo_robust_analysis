#!/usr/bin/env bash
# Llama-3.2-3B E2E量子化: 全8手法
set -euo pipefail

GPU_IDS="0,1,2,3"
MODEL="meta-llama/Llama-3.2-3B"
OUTPUT_ROOT="./data/quantized"
CALIBRATION_DATA="./data/calibration/fineweb.jsonl"
CALIBRATION_SAMPLES=512

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

if [ ! -f "$CALIBRATION_DATA" ]; then
    echo "Preparing calibration data from FineWeb sample-10BT..."
    uv run python scripts/prepare_calibration_data.py \
        --dataset fineweb \
        --num-samples "$CALIBRATION_SAMPLES" \
        --output "$CALIBRATION_DATA"
fi

METHODS=("gptq:4" "gptq:8" "awq:4" "awq:8" "smoothquant:4" "smoothquant:8" "qep:4" "qep:8")

PASSED=0
FAILED_LIST=()

for spec in "${METHODS[@]}"; do
    method="${spec%%:*}"
    bits="${spec##*:}"
    output_dir="${OUTPUT_ROOT}/Llama-3.2-3B/${method}_w${bits}"

    if [ -d "$output_dir" ]; then
        echo "SKIP: $output_dir already exists"
        PASSED=$((PASSED + 1))
        continue
    fi

    echo "=============================================="
    echo "=== Quantizing: $MODEL | $method w$bits ==="
    echo "=============================================="

    if uv run python scripts/quantize_model.py \
        --model "$MODEL" \
        --method "$method" \
        --bits "$bits" \
        --output-dir "$output_dir" \
        --calibration-data "$CALIBRATION_DATA" \
        --num-calibration-samples "$CALIBRATION_SAMPLES" \
        --gpu-ids "$GPU_IDS"; then
        PASSED=$((PASSED + 1))
        echo "SUCCESS: $method w$bits"
    else
        FAILED_LIST+=("${method}_w${bits}")
        echo "FAILED: $method w$bits"
    fi
    echo ""
done

echo "=============================================="
echo "Quantization Summary"
echo "=============================================="
echo "Total: ${#METHODS[@]} | Passed: $PASSED | Failed: ${#FAILED_LIST[@]}"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "Failed methods:"
    for f in "${FAILED_LIST[@]}"; do
        echo "  - $f"
    done
fi
echo "=============================================="
