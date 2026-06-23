#!/usr/bin/env bash
# E2Eテスト（量子化パイプライン検証）
# Llama-3.2-3B × 全量子化手法 × 限定ベンチマーク(arc_easy, hellaswag, wikitext2) × 限定typo(clean, random:4)
# Phase 1: quant環境で量子化 → Phase 2: vllm環境で推論・評価
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

GPU_IDS="0,1,2,3"
MODEL="meta-llama/Llama-3.2-3B"
MODEL_SHORT="Llama-3.2-3B"
QUANTIZED_ROOT="./data/quantized"
CALIBRATION_CLEAN="./data/calibration/fineweb_clean.jsonl"
CALIBRATION_NOISY="./data/calibration/fineweb_noisy.jsonl"
CALIBRATION_TYPES=("clean" "noisy")

BENCHMARKS=("arc_easy" "hellaswag" "wikitext2")
TYPO_CONDITIONS=("clean:0" "random:4")
MAX_SAMPLES=1000

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
echo "Phase 1: Quantization"
echo "=============================================="
uv sync --package quant-typo-neuron --extra quant 2>&1 | tail -3

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

QUANTIZATION_PASSED=0
QUANTIZATION_FAILED=()

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
            QUANTIZATION_PASSED=$((QUANTIZATION_PASSED + 1))
            continue
        fi

        echo "=============================================="
        echo "=== Quantizing: ${MODEL} | ${method} w${bits} | calibration=${calibration_type} ==="
        echo "=============================================="
        if CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run python scripts/quantize_model.py \
            --model "$MODEL" \
            --method "$method" \
            --bits "$bits" \
            --output-dir "$output_dir" \
            --calibration-data "$calibration_data_path" \
            --num-calibration-samples 512 \
            --gpu-ids "$GPU_IDS"; then
            QUANTIZATION_PASSED=$((QUANTIZATION_PASSED + 1))
            echo "SUCCESS: ${method} w${bits} calibration=${calibration_type}"
        else
            QUANTIZATION_FAILED+=("${method}_w${bits}_${calibration_type}")
            echo "FAILED quantization: ${method} w${bits} calibration=${calibration_type}"
        fi
    done
done

echo ""
echo "=============================================="
echo "Quantization Summary: Passed=${QUANTIZATION_PASSED} Failed=${#QUANTIZATION_FAILED[@]}"
if [ ${#QUANTIZATION_FAILED[@]} -gt 0 ]; then
    for f in "${QUANTIZATION_FAILED[@]}"; do echo "  FAILED: $f"; done
fi
echo "=============================================="

# ============================================================
# Phase 2: Evaluation (vllm environment)
# ============================================================
echo ""
echo "=============================================="
echo "Phase 2: Evaluation (switching to vllm environment)"
echo "=============================================="
uv sync --package quant-typo-neuron --extra vllm 2>&1 | tail -3

TOTAL=0
SUCCESS=0
FAIL=0
FAIL_LIST=()

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
                TOTAL=$((TOTAL + 1))

                echo "=============================================="
                echo "=== ${method} w${bits} calibration=${calibration_type} | ${benchmark_name} | ${typo_type} n${num_typos} ==="
                echo "=============================================="
                if uv run python experiments/run_eval.py \
                    --config configs/base_eval.yaml \
                    --gpu-ids "$GPU_IDS" \
                    "model.name=$model_path" \
                    "model.quantization_method=$method" \
                    "model.bits=$bits" \
                    "model.calibration=${calibration_type}" \
                    "benchmark.name=$benchmark_name" \
                    "benchmark.max_samples=$MAX_SAMPLES" \
                    "typo.type=$typo_type" \
                    "typo.num_typos=$num_typos"; then
                    SUCCESS=$((SUCCESS + 1))
                else
                    FAIL=$((FAIL + 1))
                    FAIL_LIST+=("${method}_w${bits}_${calibration_type}/${benchmark_name}/${typo_type}_n${num_typos}")
                    echo "FAILED: ${method} w${bits} calibration=${calibration_type} ${benchmark_name} ${typo_type} n${num_typos}"
                fi
                echo ""
            done
        done
    done
done

echo "=============================================="
echo "E2E Test Summary"
echo "=============================================="
echo "Quantization: Passed=${QUANTIZATION_PASSED} Failed=${#QUANTIZATION_FAILED[@]}"
echo "Evaluation:   Total=${TOTAL} Success=${SUCCESS} Failed=${FAIL}"
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
    echo "Failed evaluations:"
    for f in "${FAIL_LIST[@]}"; do echo "  - $f"; done
fi
echo "=============================================="
