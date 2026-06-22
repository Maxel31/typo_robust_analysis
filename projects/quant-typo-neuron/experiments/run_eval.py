#!/usr/bin/env python3
"""評価実験のエントリーポイント。

Usage:
    uv run python experiments/run_eval.py --config configs/base_eval.yaml --gpu-ids 2,3
    uv run python experiments/run_eval.py --config configs/base_eval.yaml --gpu-ids 2,3 model.name=meta-llama/Llama-3.2-3B
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation experiment with config + overrides")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU IDs (e.g., '2,3')")
    parser.add_argument("overrides", nargs="*", help="OmegaConf-style overrides (key=value)")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    from typo_utils.config import load_config

    cfg = load_config(args.config, overrides=args.overrides or None)

    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(cfg, {"gpu_ids": gpu_ids})

    from quant_typo_neuron.runner import run_experiment

    metrics = run_experiment(cfg)
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
