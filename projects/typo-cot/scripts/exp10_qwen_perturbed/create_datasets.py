#!/usr/bin/env python3
"""実験10④: Qwen2.5-7B-Instruct の摂動データセット10件を一括作成する CLI.

アーカイブの baseline (重要度スコア付き) を参照して、5ベンチ × 2条件(LXT-4, Random-4)
の摂動データセットを作成する。

再現手順 (PYTHONHASHSEED=42 必須: 乱択とtypo実現値が hash() でシードされる):
  cd <PROJ_ROOT>
  PYTHONHASHSEED=42 uv run --no-sync python \
    scripts/exp10_qwen_perturbed/create_datasets.py
"""

import argparse
import logging
import os
import sys

from typo_cot.perturbation.dataset import create_perturbed_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARKS = ["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa"]
MODEL_SHORT = "Qwen2.5-7B-Instruct"
ARCHIVE_BASELINE = "/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS
    )
    parser.add_argument("-k", "--num_perturbations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./datasets/perturbed")
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "42":
        logger.error("PYTHONHASHSEED=42 を設定してください (再現性のため必須)")
        sys.exit(1)

    for bench in args.benchmarks:
        baseline_dir = f"{ARCHIVE_BASELINE}/{MODEL_SHORT}_{bench}"
        for random_perturbation in (False, True):
            mode = "Random-4" if random_perturbation else "LXT-4"
            logger.info(f"=== {bench} {mode} ===")
            path = create_perturbed_dataset(
                baseline_dir=baseline_dir,
                num_perturbations=args.num_perturbations,
                output_dir=args.output_dir,
                seed=args.seed,
                random_perturbation=random_perturbation,
                include_choices=True,
            )
            logger.info(f"保存: {path}")


if __name__ == "__main__":
    main()
