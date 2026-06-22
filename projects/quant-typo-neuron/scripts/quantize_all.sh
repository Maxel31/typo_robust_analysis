#!/usr/bin/env bash
# 全モデル x 全量子化手法の一括量子化スクリプト
#
# Usage: bash scripts/quantize_all.sh --gpu-ids 2,3
set -euo pipefail

GPU_IDS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$GPU_IDS" ]; then
    echo "Error: --gpu-ids is required (e.g., --gpu-ids 2,3)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data/quantized}"
CALIBRATION_DATA="${CALIBRATION_DATA:-./data/calibration/wikitext.jsonl}"

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "google/gemma-2-2b"
    "google/gemma-2-9b"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
    "mistralai/Mistral-7B-v0.3"
)

METHODS_AND_BITS=(
    "gptq:4"
    "gptq:8"
    "awq:4"
    "awq:8"
    "smoothquant:8"
    "smoothquant:4"
    "qep:4"
    "qep:8"
)

# Prepare calibration data if not exists
if [ ! -f "$CALIBRATION_DATA" ]; then
    echo "Preparing calibration data..."
    uv run python scripts/prepare_calibration_data.py \
        --dataset wikitext \
        --num-samples "$CALIBRATION_SAMPLES" \
        --output "$CALIBRATION_DATA"
fi

for model in "${MODELS[@]}"; do
    model_short=$(basename "$model")
    for method_bits in "${METHODS_AND_BITS[@]}"; do
        method="${method_bits%%:*}"
        bits="${method_bits##*:}"
        output_dir="${OUTPUT_ROOT}/${model_short}/${method}_w${bits}"

        if [ -d "$output_dir" ]; then
            echo "SKIP: $output_dir already exists"
            continue
        fi

        echo "=== Quantizing: $model | method=$method bits=$bits ==="
        uv run python scripts/quantize_model.py \
            --model "$model" \
            --method "$method" \
            --bits "$bits" \
            --output-dir "$output_dir" \
            --calibration-data "$CALIBRATION_DATA" \
            --num-calibration-samples "$CALIBRATION_SAMPLES" \
            --gpu-ids "$GPU_IDS" \
        || echo "FAILED: $model $method w$bits"
    done
done

echo "All quantization jobs complete."
