"""シャード分割/統合ヘルパー (実験10②: MATH-500 全面新規再生成).

run_inference.py の --start/--end/--merge で使う共通ロジック。
命名規約・統合規則は scripts/run_inference_reasoning.py と同一:

- シャードファイル: <output_dir>/shards/results_{start:05d}_{end:05d}.json
- 統合: ファイル名昇順に読み、同一 sample_id は後勝ち(再実行シャード優先)
- summary スキーマ: アーカイブ outputs/baseline/*_math/summary.json 互換
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SHARD_NAME_RE = re.compile(r"results_(\d{5})_(\d{5})\.json$")


def shard_results_path(output_dir: Path, start: int, end: int) -> Path:
    """シャード結果ファイルのパスを返す(命名規約の一元管理)."""
    return Path(output_dir) / "shards" / f"results_{start:05d}_{end:05d}.json"


def load_shard_rows(path: Path) -> list[dict]:
    """シャードファイルを読み込む。欠損・破損時は空リスト(再実行で上書きされる)."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"シャード読み込み失敗(空扱い): {path}: {e}")
        return []
    if not isinstance(rows, list):
        logger.warning(f"シャード形式不正(空扱い): {path}")
        return []
    return rows


def merge_shard_results(output_dir: Path) -> tuple[list[dict], list[tuple[int, int]]]:
    """shards/results_*.json を統合する.

    Returns:
        (sample_id 昇順の結果リスト, カバーした [start, end) 範囲のリスト)
    """
    shard_dir = Path(output_dir) / "shards"
    merged: dict[str, dict] = {}
    covered: list[tuple[int, int]] = []
    for path in sorted(shard_dir.glob("results_*.json")):
        m = _SHARD_NAME_RE.search(path.name)
        if m:
            covered.append((int(m.group(1)), int(m.group(2))))
        for row in load_shard_rows(path):
            sample_id = row.get("sample_id")
            if sample_id is None:
                continue
            merged[sample_id] = row  # 後勝ち(再実行シャード優先)
    results = [merged[k] for k in sorted(merged)]
    return results, sorted(covered)


def build_summary_from_results(
    model: str,
    benchmark: str,
    results: list[dict],
    num_samples_per_subset: int | None = None,
    batch_size: int = 1,
    merged_shards: list[tuple[int, int]] | None = None,
) -> dict:
    """アーカイブ互換の summary.json 用 dict を構築する.

    スキーマ: experiment_info / overall_metrics / per_subset_metrics
    (アーカイブ outputs/baseline/<model>_<bench>/summary.json と同一。
    merged_shards がある場合のみ experiment_info に追記される)。
    """
    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)

    per_subset_counts: dict[str, dict[str, int]] = {}
    for r in results:
        subset = r.get("subset") or "default"
        stats = per_subset_counts.setdefault(subset, {"correct": 0, "total": 0})
        stats["total"] += 1
        if r.get("is_correct"):
            stats["correct"] += 1

    per_subset_metrics = {
        subset: {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "correct": stats["correct"],
            "total": stats["total"],
        }
        for subset, stats in per_subset_counts.items()
    }

    experiment_info: dict = {
        "model": model,
        "benchmark": benchmark,
        "num_samples_per_subset": num_samples_per_subset,
        "batch_size": batch_size,
        "total_samples": total,
        "timestamp": datetime.now().isoformat(),
    }
    if merged_shards is not None:
        experiment_info["merged_shards"] = [list(c) for c in merged_shards]

    return {
        "experiment_info": experiment_info,
        "overall_metrics": {
            "accuracy": correct / total if total > 0 else 0,
            "total_correct": correct,
            "total_samples": total,
        },
        "per_subset_metrics": per_subset_metrics,
    }
