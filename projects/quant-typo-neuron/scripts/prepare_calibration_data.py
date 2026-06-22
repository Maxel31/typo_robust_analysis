#!/usr/bin/env python3
"""較正データの準備スクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare calibration data for quantization")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name (default: wikitext)")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of calibration samples")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    args = parser.parse_args()

    from typo_utils.quant.calibration import prepare_calibration_data

    texts = prepare_calibration_data(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Saved {len(texts)} calibration samples to {output_path}")


if __name__ == "__main__":
    main()
