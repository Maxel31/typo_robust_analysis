#!/usr/bin/env python3
"""few-shot 例のプリキャッシュスクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache few-shot examples for all benchmarks")
    parser.add_argument("--output-dir", default="data/few_shot", help="Cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-pool", type=int, default=500, help="Max pool size for sampling")
    args = parser.parse_args()

    from quant_typo_neuron.benchmarks import BENCHMARKS
    from quant_typo_neuron.few_shot import FewShotCache

    cache = FewShotCache(cache_dir=Path(args.output_dir))

    for name, bench_cls in BENCHMARKS.items():
        bench = bench_cls()
        if bench.num_few_shot == 0:
            print(f"SKIP {name}: num_few_shot=0")
            continue
        pool = bench.load(max_samples=args.max_pool)
        shots = cache.get_or_create(
            benchmark=name,
            num_shots=bench.num_few_shot,
            seed=args.seed,
            pool=pool,
        )
        print(f"Cached {len(shots)} few-shot examples for {name}")

    print("Done.")


if __name__ == "__main__":
    main()
