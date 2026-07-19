"""実験7 本番実行のシャード分割・マージ (GPU ロックを細かく解放するため).

本番は 1 シャード = 1 GPU ヘルパー呼び出し (目安 20〜40 分) に分割し、
シャード間でロックを解放して他実験と交互に進む。本モジュールは
シャード範囲の計算と、シャード出力の決定的なマージを提供する。
マージは開始位置順に整列し、範囲が [0, N) を隙間なく被覆することを検証する。
"""


def shard_ranges(n_total: int, shard_size: int) -> list[tuple[int, int]]:
    """[0, n_total) を shard_size ごとの半開区間に分割する."""
    if shard_size <= 0:
        raise ValueError("shard_size は正の整数が必要です")
    return [
        (start, min(start + shard_size, n_total))
        for start in range(0, n_total, shard_size)
    ]


def _validate_contiguous(ranges: list[tuple[int, int]]) -> None:
    if not ranges:
        raise ValueError("シャードがありません")
    if ranges[0][0] != 0:
        raise ValueError(f"先頭シャードが 0 から始まっていません: start={ranges[0][0]}")
    for (_, prev_end), (start, _) in zip(ranges, ranges[1:]):
        if start != prev_end:
            raise ValueError(
                f"シャード範囲が連続していません: {prev_end} の次が {start}"
            )


def merge_correction_shards(shards: list[dict]) -> dict:
    """make_corrected_dataset.py のシャード出力をマージする.

    Args:
        shards: {"start", "end", "samples", "per_sample", "aggregate"} の列。
                順不同でよい (start で整列する)。

    Returns:
        {"samples", "per_sample", "aggregate", "rates"}
    """
    ordered = sorted(shards, key=lambda s: s["start"])
    _validate_contiguous([(s["start"], s["end"]) for s in ordered])

    samples: list[dict] = []
    per_sample: list[dict] = []
    agg: dict[str, int] = {}
    for sh in ordered:
        samples.extend(sh["samples"])
        per_sample.extend(sh["per_sample"])
        for k, v in sh["aggregate"].items():
            agg[k] = agg.get(k, 0) + v

    rates = {
        "word_restoration_rate": agg["word_restored"] / agg["word_total"]
        if agg.get("word_total") else 0.0,
        "full_restoration_rate": agg["fully_restored"] / agg["n_samples"]
        if agg.get("n_samples") else 0.0,
        "all_perturbed_restored_rate": (
            agg.get("perturbed_words_all_restored", 0) / agg["n_samples"]
            if agg.get("n_samples") else 0.0
        ),
    }
    return {"samples": samples, "per_sample": per_sample,
            "aggregate": agg, "rates": rates}


def merge_generation_results(shards: list[dict]) -> tuple[list[dict], dict]:
    """run_generation_only.py のシャード出力をマージする.

    Args:
        shards: {"start", "end", "results"} の列 (順不同)。

    Returns:
        (results 連結リスト, summary.json 互換の集計
         {"overall_metrics", "per_subset_metrics"})
    """
    ordered = sorted(shards, key=lambda s: s["start"])
    _validate_contiguous([(s["start"], s["end"]) for s in ordered])

    results: list[dict] = []
    for sh in ordered:
        results.extend(sh["results"])

    correct = sum(1 for r in results if r.get("is_correct"))
    subset_stats: dict[str, dict[str, int]] = {}
    for r in results:
        st = subset_stats.setdefault(
            r.get("subset") or "default", {"correct": 0, "total": 0}
        )
        st["total"] += 1
        if r.get("is_correct"):
            st["correct"] += 1

    summary = {
        "overall_metrics": {
            "accuracy": correct / len(results) if results else 0.0,
            "total_correct": correct,
            "total_samples": len(results),
        },
        "per_subset_metrics": {
            k: {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in subset_stats.items()
        },
    }
    return results, summary
