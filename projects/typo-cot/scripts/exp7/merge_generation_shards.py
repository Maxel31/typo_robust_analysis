#!/usr/bin/env python3
"""実験7: run_generation_only.py --shard の出力を結合して results/summary を作る.

非シャード実行と同じスキーマの results.json / summary.json を出力する。
シャードが [0, N) を隙間なく被覆していない場合はエラーで停止する。

使用例:
  uv run python scripts/exp7/merge_generation_shards.py \
    --experiment_dir results/prod/exp7/generation/gemma-3-4b-it_mmlu_k4_neuralfix \
    --expected_total 2850
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from typo_cot.defense.sharding import merge_generation_results


def main() -> None:
    parser = argparse.ArgumentParser(description="評価生成シャードの結合")
    parser.add_argument("--experiment_dir", required=True,
                        help="shards/ サブディレクトリを含む実験出力ディレクトリ")
    parser.add_argument("--expected_total", type=int, default=None,
                        help="期待サンプル総数 (不一致ならエラー)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    shard_paths = sorted((exp_dir / "shards").glob("*.json"))
    if not shard_paths:
        raise SystemExit(f"シャードが見つかりません: {exp_dir}/shards")
    shards = []
    for p in shard_paths:
        with open(p, encoding="utf-8") as f:
            shards.append(json.load(f))

    results, summary_metrics = merge_generation_results(shards)
    if args.expected_total is not None and len(results) != args.expected_total:
        raise SystemExit(
            f"結合結果 {len(results)} 件が期待値 {args.expected_total} と"
            f"一致しません (未完了シャードあり?)"
        )

    config = {}
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = {
        "experiment_info": {
            "model": config.get("model"),
            "benchmark": config.get("benchmark"),
            "num_samples_per_subset": None,
            "batch_size": config.get("batch_size"),
            "total_samples": len(results),
            "n_shards": len(shards),
            "timestamp": datetime.now().isoformat(),
        },
        **summary_metrics,
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"結合完了: {exp_dir} ({len(shards)} シャード, {len(results)} サンプル, "
          f"accuracy={summary_metrics['overall_metrics']['accuracy']:.4f})")


if __name__ == "__main__":
    main()
