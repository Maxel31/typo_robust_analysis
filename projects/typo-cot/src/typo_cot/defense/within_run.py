"""実験7: byte-identical 復元サンプルの within-run flip 検証ロジック.

校正済みデータセット (data/exp7/corrected/*) から「校正後プロンプトが clean
プロンプトとバイト同一」のサンプルを抽出し、clean/校正後を同一バッチで生成
して flip を測るための純粋ロジック (プロンプト構築・選別・バッチ分割・集計)。

本番評価生成 (run_generation_only.py) はアーカイブ baseline とのクロスラン
比較になり再現性ノイズが乗るため、byte-identical → flip 0% の主張は
本モジュールによる within-run 測定で正式検証する。

プロンプト構築規約は run_generation_only.py / rerun_within_run_reference.py と
同一: MC ベンチの clean は choices 引数から、校正後は choices 埋め込み済み
テキスト + choices=None で構築する。
"""

# choices を持つ多肢選択ベンチ (run_generation_only.py:139 と同一)
MC_BENCHMARKS = frozenset({"mmlu", "mmlu_pro", "arc", "commonsense_qa"})


def build_prompt_pair(template, benchmark: str, sample: dict) -> tuple[str, str]:
    """clean / 校正後のプロンプト対を Phase 3 と同一規約で構築する.

    Args:
        template: create_prompt_template(benchmark) の戻り値
        benchmark: ベンチマーク名
        sample: 校正済み perturbed_dataset.json の 1 サンプル
                (original_question=clean, perturbed_question=校正後)

    Returns:
        (clean_prompt, corrected_prompt)
    """
    subset = sample.get("subset")
    if benchmark in MC_BENCHMARKS:
        clean = template.generate(
            question=sample["original_question"],
            choices=sample["choices"],
            subject=subset,
        ).get_full_prompt()
        corr = template.generate(
            question=sample["perturbed_question"],
            choices=sample.get("perturbed_choices"),
            subject=subset,
        ).get_full_prompt()
    else:
        clean = template.generate(
            question=sample["original_question"]
        ).get_full_prompt()
        corr = template.generate(
            question=sample["perturbed_question"]
        ).get_full_prompt()
    return clean, corr


def select_byte_identical(
    samples: list[dict],
    fully_restored_flags: dict[str, bool],
    template,
    benchmark: str,
) -> tuple[list[dict], dict]:
    """校正後プロンプトが clean とバイト同一のサンプルを選別する.

    選別の正はプロンプト厳密一致 (byte-identical)。restoration_stats.json の
    fully_restored フラグ (空白正規化の全文一致) は照合のみに使い、
    不一致サンプルを stats["flag_mismatch_ids"] に記録する。

    Returns:
        (pairs, stats)
        pairs: [{"index", "sample_id", "prompt", "correct_answer", "subset"}]
               (clean == 校正後 なのでプロンプトは 1 本)
        stats: {"n_samples", "n_byte_identical", "n_fully_restored_flag",
                "flag_mismatch_ids"}
    """
    pairs: list[dict] = []
    mismatch: list[str] = []
    n_flag = 0
    for idx, s in enumerate(samples):
        sid = s["sample_id"]
        clean, corr = build_prompt_pair(template, benchmark, s)
        identical = clean == corr
        flag = bool(fully_restored_flags.get(sid, False))
        n_flag += flag
        if identical != flag:
            mismatch.append(sid)
        if identical:
            pairs.append(
                {
                    "index": idx,
                    "sample_id": sid,
                    "prompt": clean,
                    "correct_answer": s["correct_answer"],
                    "subset": s.get("subset"),
                }
            )
    stats = {
        "n_samples": len(samples),
        "n_byte_identical": len(pairs),
        "n_fully_restored_flag": n_flag,
        "flag_mismatch_ids": mismatch,
    }
    return pairs, stats


def iter_pair_batches(pairs: list[dict], pairs_per_batch: int):
    """pairs を pairs_per_batch ずつのバッチに分割する (最終バッチは端数可).

    clean/校正後の 2 行が必ず同一バッチに入るよう、分割はペア単位で行う。
    """
    if pairs_per_batch <= 0:
        raise ValueError("pairs_per_batch は正の整数が必要です")
    for start in range(0, len(pairs), pairs_per_batch):
        yield pairs[start : start + pairs_per_batch]


def batch_rows(batch_pairs: list[dict]) -> list[str]:
    """1 バッチの生成行を構築する: [clean_0, corr_0, clean_1, corr_1, ...].

    行 2i が clean、行 2i+1 が校正後。byte-identical 集合では両行は同一文字列
    だが、within-run 測定の定義どおり両方を独立の行として生成する。
    """
    rows: list[str] = []
    for p in batch_pairs:
        rows.append(p["prompt"])
        rows.append(p["prompt"])
    return rows


def aggregate_within_run(records: list[dict]) -> dict:
    """within-run 生成結果の flip / accuracy を集計する.

    Args:
        records: [{"sample_id", "ans_clean", "ans_corr",
                   "correct_clean", "correct_corr"}]
    """
    n = len(records)
    flips = [r for r in records if r["ans_corr"] != r["ans_clean"]]
    return {
        "n": n,
        "n_flip": len(flips),
        "flip_rate": (len(flips) / n) if n else None,
        "flip_ids": [r["sample_id"] for r in flips],
        "accuracy_clean": (sum(r["correct_clean"] for r in records) / n) if n else None,
        "accuracy_corrected": (
            sum(r["correct_corr"] for r in records) / n
        ) if n else None,
    }
