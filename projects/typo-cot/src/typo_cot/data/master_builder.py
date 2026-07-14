"""アーカイブ済み生成/解析ログから Step 0 統合テーブルの行を構築する純粋ロジック.

ファイル io は行わない (io は archive_reader / master_table に隔離)。
入力はアーカイブの results.json / full_results.json(sample_results) と
同じ構造の list[dict] を想定する。
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from typo_cot.data.master_table import (
    CONDITIONS,
    MASTER_COLUMNS,
    coerce_master_dtypes,
)
from typo_cot.evaluation.extractor import create_extractor


def _to_json_or_none(value: Any) -> str | None:
    """list/dict を JSON 文字列に変換 (None はそのまま)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def sample_metrics_from_analysis(
    sample_results: list[dict],
) -> dict[str, dict[str, Any]]:
    """analysis full_results.json の `sample_results` を sample_id -> 指標 dict に変換.

    抽出する指標:
        flip (answer_changed), pattern, cot_rouge_l_f1, cot_jaccard_top{3,5,10,15,20}
    """
    metrics: dict[str, dict[str, Any]] = {}
    for sr in sample_results:
        cm = sr.get("cot_metrics") or {}
        rouge = cm.get("rouge_l") or {}
        jaccard = cm.get("jaccard") or {}
        entry: dict[str, Any] = {
            "flip": bool(sr["answer_changed"]),
            "pattern": sr.get("pattern"),
            "cot_rouge_l_f1": rouge.get("f1"),
        }
        for k in (3, 5, 10, 15, 20):
            entry[f"cot_jaccard_top{k}"] = jaccard.get(f"top{k}")
        metrics[sr["sample_id"]] = entry
    return metrics


def build_condition_df(
    baseline_results: list[dict],
    perturbed_results: list[dict] | None,
    sample_metrics: dict[str, dict[str, Any]] | None,
    *,
    model: str,
    benchmark: str,
    condition: str,
    seed: int,
    prompt_id: str,
    source_path: str,
) -> pd.DataFrame:
    """1 つの (model, benchmark, condition) の統合テーブル行を構築する.

    Args:
        baseline_results: baseline results.json の list[dict]
        perturbed_results: 摂動条件の results.json (clean のときは None)
        sample_metrics: `sample_metrics_from_analysis` の出力
            (clean のときは None。分析から除外されたサンプルは含まれず、
            該当行の flip / CoT 指標は NA になる)
        model / benchmark / condition: キー
        seed: レジストリに凍結された seed
        prompt_id: レジストリに凍結されたプロンプト ID
        source_path: 由来の results.json パス (プロベナンス)

    Returns:
        MASTER_COLUMNS スキーマの DataFrame
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    is_clean = condition == "clean"
    if is_clean:
        results = baseline_results
    else:
        if perturbed_results is None:
            raise ValueError(f"perturbed_results is required for condition={condition}")
        results = perturbed_results

    extractor = create_extractor(benchmark)
    metrics = sample_metrics or {}

    rows: list[dict[str, Any]] = []
    for r in results:
        sample_id = r["sample_id"]
        generated_text = r.get("generated_text", "") or ""
        span = extractor.extract_strict(generated_text).strip()
        m = metrics.get(sample_id, {}) if not is_clean else {}
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "model": model,
            "benchmark": benchmark,
            "condition": condition,
            "question_text": r.get("question"),
            "cot_text": generated_text,
            "answer_span": span if span else None,
            "answer_pred": r.get("extracted_answer"),
            "answer_gold": r.get("correct_answer"),
            "is_correct": r.get("is_correct"),
            "flip": m.get("flip"),
            "pattern": m.get("pattern"),
            "cot_rouge_l_f1": m.get("cot_rouge_l_f1"),
            "cot_jaccard_top3": m.get("cot_jaccard_top3"),
            "cot_jaccard_top5": m.get("cot_jaccard_top5"),
            "cot_jaccard_top10": m.get("cot_jaccard_top10"),
            "cot_jaccard_top15": m.get("cot_jaccard_top15"),
            "cot_jaccard_top20": m.get("cot_jaccard_top20"),
            "r_q": _to_json_or_none(r.get("question_top_k_words")),
            "r_c": _to_json_or_none(r.get("cot_top_k_words")),
            "span_extract_ok": bool(span),
            "seed": seed,
            "prompt_id": prompt_id,
            "subset": r.get("subset"),
            "original_question": r.get("original_question"),
            "perturbed_tokens": _to_json_or_none(r.get("perturbed_tokens")),
            "source_path": source_path,
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=list(MASTER_COLUMNS))
    return coerce_master_dtypes(df)


def derive_union_exclusion(
    clean_span_ok: dict[str, bool],
    condition_span_ok: dict[str, dict[str, bool]],
) -> set[str]:
    """span_extract_ok から union 除外集合を導出する.

    `typo_cot.analysis.analyzer.compute_unified_exclusion` と同じ意味論:
    clean で strict 未検出、またはいずれかの摂動条件で strict 未検出の
    sample_id の和集合を返す。

    Args:
        clean_span_ok: sample_id -> clean 条件の span_extract_ok
        condition_span_ok: condition -> (sample_id -> span_extract_ok)

    Returns:
        除外すべき sample_id の集合
    """
    excluded: set[str] = set()
    for cond_ok in condition_span_ok.values():
        for sid, ok in cond_ok.items():
            if not ok or not clean_span_ok.get(sid, False):
                excluded.add(sid)
    return excluded
