#!/usr/bin/env python3
"""実験10③: R1蒸留の摂動データセット6件を一括作成する CLI.

R1PerturbedDatasetCreator (Qwen2 トークナイザの選択肢マーカー断片 '(A' を
摂動対象から除外する追記型サブクラス) を使う。共有 run_perturbation.py は
並行キューが使用中のため変更しない。

再現手順 (PYTHONHASHSEED=42 必須: 乱択とtypo実現値が hash() でシードされる):
  PYTHONHASHSEED=42 uv run --no-sync python \
    scripts/exp10_r1_perturbed/create_datasets.py
"""

import argparse
import logging
import os
import sys

from typo_cot.perturbation.r1_dataset import create_r1_perturbed_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARKS = ["gsm8k", "math", "mmlu"]
MODEL_SHORT = "DeepSeek-R1-Distill-Qwen-7B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    parser.add_argument("-k", "--num_perturbations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_root", default="outputs/baseline")
    parser.add_argument("--output_dir", default="./datasets/perturbed")
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "42":
        logger.error("PYTHONHASHSEED=42 を設定してください (再現性のため必須)")
        sys.exit(1)

    for bench in args.benchmarks:
        baseline_dir = f"{args.baseline_root}/{MODEL_SHORT}_{bench}"
        for random_perturbation in (False, True):
            mode = "Random-4" if random_perturbation else "LXT-4"
            logger.info(f"=== {bench} {mode} ===")
            path = create_r1_perturbed_dataset(
                baseline_dir=baseline_dir,
                num_perturbations=args.num_perturbations,
                output_dir=args.output_dir,
                seed=args.seed,
                random_perturbation=random_perturbation,
            )
            logger.info(f"保存: {path}")


if __name__ == "__main__":
    main()
