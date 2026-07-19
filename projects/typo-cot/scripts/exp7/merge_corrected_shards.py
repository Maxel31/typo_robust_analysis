#!/usr/bin/env python3
"""実験7: make_corrected_dataset.py --shard の出力を結合して最終形にする.

非シャード実行と同じスキーマの
perturbed_dataset.json / config.json / restoration_stats.json を出力する。
シャードが [0, N) を隙間なく被覆していない場合はエラーで停止する。

使用例:
  uv run python scripts/exp7/merge_corrected_shards.py \
    --dataset_dir data/exp7/corrected/gemma-3-4b-it_mmlu_k4_neuralfix \
    --source <LXT-4 perturbed_dataset.json>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from typo_cot.defense.sharding import merge_correction_shards

MODE_NAMES = {"pyspell": "spellfix", "neural": "neuralfix", "llm": "llmfix"}


def main() -> None:
    parser = argparse.ArgumentParser(description="校正シャードの結合")
    parser.add_argument("--dataset_dir", required=True,
                        help="shards/ サブディレクトリを含む出力ディレクトリ")
    parser.add_argument("--source", required=True,
                        help="元の LXT-4 perturbed_dataset.json (metadata 取得用)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    shard_paths = sorted((dataset_dir / "shards").glob("*.json"))
    if not shard_paths:
        raise SystemExit(f"シャードが見つかりません: {dataset_dir}/shards")
    shards = []
    for p in shard_paths:
        with open(p, encoding="utf-8") as f:
            shards.append(json.load(f))

    with open(args.source, encoding="utf-8") as f:
        source = json.load(f)
    n_source = len(source["samples"])

    merged = merge_correction_shards(shards)
    if merged["aggregate"]["n_samples"] != n_source:
        raise SystemExit(
            f"シャード合計 {merged['aggregate']['n_samples']} が元データセット "
            f"{n_source} と一致しません (未完了シャードあり?)"
        )

    corrector = shards[0]["corrector"]
    new_metadata = dict(source["metadata"])
    new_metadata["perturbation_mode"] = MODE_NAMES[corrector]
    new_metadata["correction_source"] = args.source
    new_metadata["corrector"] = corrector
    new_metadata["corrector_model"] = shards[0]["corrector_model"]
    new_metadata["created_at"] = datetime.now().isoformat()
    new_metadata["total_samples"] = len(merged["samples"])
    new_metadata["n_shards"] = len(shards)

    with open(dataset_dir / "perturbed_dataset.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": new_metadata, "samples": merged["samples"]}, f,
                  ensure_ascii=False, indent=2)
    with open(dataset_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)
    with open(dataset_dir / "restoration_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {"aggregate": merged["aggregate"], "rates": merged["rates"],
             "per_sample": merged["per_sample"]},
            f, ensure_ascii=False, indent=2,
        )
    print(f"結合完了: {dataset_dir} ({len(shards)} シャード, "
          f"{len(merged['samples'])} サンプル)")
    print(json.dumps(merged["rates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
