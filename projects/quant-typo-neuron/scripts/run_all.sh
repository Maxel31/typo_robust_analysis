#!/usr/bin/env bash
# 全実験の一括実行スクリプト
#
# Usage: bash scripts/run_all.sh --gpu-ids 2,3
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

QUANT_METHODS=(
    "none:16"
    "gptq:4"
    "gptq:8"
    "awq:4"
    "awq:8"
    "smoothquant:8"
    "smoothquant:4"
    "qep:4"
    "qep:8"
)

QUANTIZED_ROOT="${QUANTIZED_ROOT:-./data/quantized}"

for model in "${MODELS[@]}"; do
    model_short=$(basename "$model")
    for quant_spec in "${QUANT_METHODS[@]}"; do
        method="${quant_spec%%:*}"
        bits="${quant_spec##*:}"

        if [ "$method" = "none" ]; then
            model_path="$model"
        else
            model_path="${QUANTIZED_ROOT}/${model_short}/${method}_w${bits}"
            if [ ! -d "$model_path" ]; then
                echo "SKIP: quantized model not found at $model_path"
                continue
            fi
        fi

        for benchmark in "${BENCHMARKS[@]}"; do
            for typo_spec in "${TYPO_CONDITIONS[@]}"; do
                typo_type="${typo_spec%%:*}"
                num_typos="${typo_spec##*:}"

                echo "=== $model_short | $method w$bits | $benchmark | $typo_type n$num_typos ==="
                uv run python experiments/run_eval.py \
                    --config configs/base_eval.yaml \
                    --gpu-ids "$GPU_IDS" \
                    "model.name=$model_path" \
                    "model.quant_method=$method" \
                    "model.bits=$bits" \
                    "benchmark.name=$benchmark" \
                    "typo.type=$typo_type" \
                    "typo.num_typos=$num_typos" \
                || echo "FAILED: $model_short $method w$bits $benchmark $typo_type n$num_typos"
            done
        done
    done
done

echo "All experiments complete."
