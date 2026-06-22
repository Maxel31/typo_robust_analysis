#!/usr/bin/env python3
"""評価実験のエントリーポイント。

Usage:
    python experiments/run_eval.py --config configs/base_eval.yaml
    python experiments/run_eval.py --config configs/base_eval.yaml model.name=meta-llama/Llama-3.2-3B
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation experiment with config + overrides")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="OmegaConf-style overrides (key=value)")
    args = parser.parse_args()

    from typo_utils.config import load_config

    cfg = load_config(args.config, overrides=args.overrides or None)

    from quant_typo_neuron.runner import run_experiment

    metrics = run_experiment(cfg)
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
