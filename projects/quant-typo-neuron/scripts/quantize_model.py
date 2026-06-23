#!/usr/bin/env python3
"""単一モデルの量子化スクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize a single model")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--method", required=True, choices=["gptq", "awq", "smoothquant", "qep"],
                        help="Quantization method")
    parser.add_argument("--bits", type=int, required=True, help="Bit width (4 or 8)")
    parser.add_argument("--output-dir", required=True, help="Output directory for quantized model")
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument("--num-calibration-samples", type=int, default=512,
                        help="Number of calibration samples")
    parser.add_argument("--calibration-data", type=str, default=None,
                        help="Path to calibration JSONL file")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '2,3')")
    args = parser.parse_args()

    from typo_utils.quant.base import QuantConfig
    from typo_utils.quant.llm_compressor import create_quantizer

    config = QuantConfig(
        method=args.method,
        bits=args.bits,
        group_size=args.group_size,
        num_calibration_samples=args.num_calibration_samples,
    )

    calibration_data = None
    if args.calibration_data:
        calibration_data = []
        with open(args.calibration_data) as f:
            for line in f:
                calibration_data.append(json.loads(line)["text"])

    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    quantizer = create_quantizer(args.method)
    output_path = quantizer.quantize(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        config=config,
        calibration_data=calibration_data,
        gpu_ids=gpu_ids,
    )
    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
