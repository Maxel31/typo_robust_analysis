"""実験4: fixed-target 帰属 (R_C^fixed) のコアロジック.

rebuttal 期の scripts/rebuttal/run_fixed_target_attribution.py (4設定版) を
全設定に一般化するためのモジュール化。回答パターン検出・splice 計画・統計カウントは
rebuttal 実装と同一規約 (テストで ANSWER_PATTERNS の同一性を検証している)。

方式 (force-decode splice):
- 摂動後 generated_text 中の最終回答スパン ("The answer is (X)" 等) の選択肢/数値部分を、
  baseline (clean) 生成の回答文字列に置換した spliced_text を構築する。
- 既存パイプライン analyze_combined と同一手順・同一の compute_relevance 規約
  (回答トークン位置の logit を backprop; 自由記述は各位置の平均) で
  CoT→Answer relevance を再計算する。
- 回答が flip していないサンプルでは spliced_text == 元テキストとなり、
  R_C^fixed は default 版 R_C と定義上一致する。

GPU 依存部 (analyze_cot_fixed / build_prompt) は遅延 import で分離してあり、
本モジュールの計画・統計ロジックは GPU なしでテスト可能。
"""

import re
from dataclasses import dataclass
from typing import Any

# lrp/analyzer.py の回答パターン (テキストマッチ部分のみ使用)。
# analyzer._find_answer_pattern と同一の正規表現・同一の「最後のマッチ採用」規約。
# rebuttal スクリプトの ANSWER_PATTERNS と同一であること (テストで検証)。
ANSWER_PATTERNS = [
    (r"[Tt]he\s+answer\s+is[:\s]*\(([A-Ja-j])\)", "choice"),
    (r"[Tt]he\s+answer\s+is[:\s]*([A-Ja-j])(?:\.|,|\s|$)", "choice"),
    (r"[Aa]nswer[:\s]+\(([A-Ja-j])\)", "choice"),
    (r"[Aa]nswer[:\s]+([A-Ja-j])(?:\.|,|\s|$)", "choice"),
    (r"\*\*\(([A-Ja-j])\)\*\*", "choice"),
    (r"\*\*([A-Ja-j])\*\*", "choice"),
    (r"(?:correct|right)\s+(?:answer|option)\s+is[:\s]*\(?([A-Ja-j])\)?", "choice"),
    (r"[Tt]he\s+answer\s+is[:\s]*(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"[Tt]he\s+answer\s+is[:\s]*\$?(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"####\s*(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"[Aa]nswer[:\s]+(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"(?:^|\n)\s*\(?([A-Ja-j])\)?\s*\.?\s*$", "choice"),
]


def find_answer_match(text: str) -> tuple[re.Match | None, str | None]:
    """lrp.analyzer._find_answer_pattern と同じ規約で最終回答のマッチを返す.

    パターンは定義順に試し、最初にヒットしたパターンの「最後のマッチ」を採用する。

    Returns:
        (match, pattern_type) または (None, None)
    """
    for pattern, ptype in ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1], ptype
    return None, None


@dataclass
class SplicePlan:
    """1サンプルの fixed-target 再計算計画.

    Attributes:
        sample_id: サンプル ID
        skip_reason: スキップ理由 (None なら処理対象)
        spliced: 回答文字列を置換したか (flip 事例なら True)
        spliced_text: AttnLRP に流す生成テキスト (splice 済み)
        baseline_answer: baseline (clean) 生成の回答文字列 (=固定ターゲット)
        perturbed_answer: 摂動後生成の回答文字列 (生成テキスト中の生スパン)
        baseline_pattern_type: baseline 側でマッチしたパターン種別 (choice/number)
        perturbed_pattern_type: 摂動側でマッチしたパターン種別
    """

    sample_id: str
    skip_reason: str | None = None
    spliced: bool = False
    spliced_text: str | None = None
    baseline_answer: str | None = None
    perturbed_answer: str | None = None
    baseline_pattern_type: str | None = None
    perturbed_pattern_type: str | None = None


def plan_splice(
    baseline_text: str, perturbed_text: str, sample_id: str
) -> SplicePlan:
    """baseline / 摂動後の生成テキストから splice 計画を立てる.

    flip 判定は生成テキスト中の生スパン文字列の比較 (rebuttal 規約: 大文字化しない)。
    """
    base_match, base_type = find_answer_match(baseline_text)
    if base_match is None:
        return SplicePlan(
            sample_id=sample_id, skip_reason="no_baseline_answer_pattern"
        )
    base_answer_str = base_match.group(1)

    pert_match, pert_type = find_answer_match(perturbed_text)
    if pert_match is None:
        return SplicePlan(
            sample_id=sample_id,
            skip_reason="no_perturbed_answer_pattern",
            baseline_answer=base_answer_str,
            baseline_pattern_type=base_type,
        )

    s, e = pert_match.span(1)
    pert_answer_str = perturbed_text[s:e]
    if pert_answer_str == base_answer_str:
        spliced_text = perturbed_text
        spliced = False
    else:
        spliced_text = perturbed_text[:s] + base_answer_str + perturbed_text[e:]
        spliced = True

    return SplicePlan(
        sample_id=sample_id,
        spliced=spliced,
        spliced_text=spliced_text,
        baseline_answer=base_answer_str,
        perturbed_answer=pert_answer_str,
        baseline_pattern_type=base_type,
        perturbed_pattern_type=pert_type,
    )


def plan_run(
    baseline_by_id: dict[str, dict[str, Any]],
    perturbed_results: list[dict[str, Any]],
    limit: int | None = None,
    sample_ids: set[str] | None = None,
) -> tuple[list[SplicePlan], dict[str, Any]]:
    """run 全体の splice 計画と統計を作る (rebuttal fixed_target_stats.json と同キー).

    Args:
        baseline_by_id: baseline results.json の sample_id -> entry
        perturbed_results: 摂動後 results.json のエントリリスト (順序保存)
        limit: 先頭 N 件に制限 (パイロット用; rebuttal --limit と同じ先頭スライス)
        sample_ids: 指定時はこの ID 集合のみ対象 (limit より先に適用)

    Returns:
        (処理対象の SplicePlan リスト, 統計辞書)
    """
    samples = perturbed_results
    if sample_ids is not None:
        samples = [r for r in samples if r["sample_id"] in sample_ids]
    if limit is not None:
        samples = samples[:limit]

    stats: dict[str, Any] = {
        "total": 0,
        "processed": 0,
        "spliced": 0,
        "identical": 0,
        "skip_no_baseline": 0,
        "skip_no_base_answer": 0,
        "skip_no_pert_answer": 0,
        "errors": 0,
    }
    skipped_ids: dict[str, str] = {}
    plans: list[SplicePlan] = []

    for entry in samples:
        sid = entry["sample_id"]
        stats["total"] += 1

        base = baseline_by_id.get(sid)
        if base is None:
            stats["skip_no_baseline"] += 1
            skipped_ids[sid] = "no_baseline"
            continue

        plan = plan_splice(base["generated_text"], entry["generated_text"], sid)
        if plan.skip_reason == "no_baseline_answer_pattern":
            stats["skip_no_base_answer"] += 1
            skipped_ids[sid] = plan.skip_reason
            continue
        if plan.skip_reason == "no_perturbed_answer_pattern":
            stats["skip_no_pert_answer"] += 1
            skipped_ids[sid] = plan.skip_reason
            continue

        if plan.spliced:
            stats["spliced"] += 1
        else:
            stats["identical"] += 1
        stats["processed"] += 1
        plans.append(plan)

    stats["skipped_ids"] = skipped_ids
    return plans, stats


def top_k_token_set(token_scores: list, k: int = 10) -> set:
    """analysis/metrics.py top_k_jaccard_by_token と同じ規約 (トークン文字列 dedup, max)."""
    best: dict[str, float] = {}
    for tok, score in token_scores:
        if tok not in best or score > best[tok]:
            best[tok] = score
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return {t for t, _ in ranked[: min(k, len(ranked))]}


def compare_cot_payloads(
    new: dict[str, Any], ref: dict[str, Any], ks: tuple[int, ...] = (5, 10, 20)
) -> dict[str, Any]:
    """_cot.pt ペイロード同士の一致度レポートを作る.

    スモーク検証で (a) rebuttal 参照との一致、(b) 非flip事例の default 版との
    一致を数値化する。

    Returns:
        {"top{k}_jaccard": float, "cot_range_match": bool,
         "n_tokens_match": bool, "max_abs_score_diff": float | None}
    """
    new_scores = list(new.get("token_scores", []))
    ref_scores = list(ref.get("token_scores", []))

    report: dict[str, Any] = {}
    for k in ks:
        s1 = top_k_token_set(new_scores, k)
        s2 = top_k_token_set(ref_scores, k)
        union = s1 | s2
        report[f"top{k}_jaccard"] = (
            len(s1 & s2) / len(union) if union else 1.0
        )

    report["cot_range_match"] = bool(
        new.get("cot_token_start") == ref.get("cot_token_start")
        and new.get("cot_token_end") == ref.get("cot_token_end")
    )
    report["n_tokens_match"] = len(new_scores) == len(ref_scores)
    if report["n_tokens_match"]:
        report["max_abs_score_diff"] = max(
            (abs(float(a[1]) - float(b[1])) for a, b in zip(new_scores, ref_scores, strict=True)),
            default=0.0,
        )
    else:
        report["max_abs_score_diff"] = None
    return report


# ---------------------------------------------------------------------------
# GPU 依存部 (AttnLRP backward)。ユニットテスト対象外・スクリプトから使用。
# ---------------------------------------------------------------------------


def build_prompt(template, benchmark: str, entry: dict) -> str:
    """run_inference.generate_prompt_for_sample と同一のプロンプト再構築.

    rebuttal 実装 (run_fixed_target_attribution.build_prompt) と同一。
    """
    if benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
        prompt_result = template.generate(
            question=entry["question"],
            choices=entry.get("choices"),  # Phase3 では None (選択肢は question 埋め込み済)
            subject=entry.get("subset"),
        )
    elif benchmark == "gsm8k":
        prompt_result = template.generate(question=entry["question"])
    elif benchmark in ["bbh", "math", "strategy_qa"]:
        prompt_result = template.generate(
            question=entry["question"], subject=entry.get("subset")
        )
    else:
        prompt_result = template.generate(question=entry["question"])
    return prompt_result.get_full_prompt()


def analyze_cot_fixed(analyzer, prompt: str, generated_text: str) -> dict:
    """analyze_combined の CoT→Answer パスのみを同一手順で実行 (rebuttal 実装と同一).

    generated_text には plan_splice 済みの spliced_text を渡す。
    質問側パス (Question→CoT) は固定ターゲット化の影響を受けないため省略する。
    """
    import torch

    full_text = prompt + generated_text
    prompt_length = len(prompt)

    prompt_inputs = analyzer.tokenizer(prompt, return_tensors="pt")
    prompt_token_count = prompt_inputs["input_ids"].shape[1]

    full_inputs = analyzer.tokenizer(full_text, return_tensors="pt")
    full_input_ids = full_inputs["input_ids"].to(analyzer.device)
    total_token_count = full_input_ids.shape[1]

    offset_list = analyzer._compute_offset_mapping_fallback(
        full_text, full_input_ids[0].tolist()
    )
    tokens = [analyzer.tokenizer.decode([tid]) for tid in full_input_ids[0].tolist()]

    answer_token_start, answer_token_end, answer_choice_position = (
        analyzer._find_answer_pattern(
            generated_text=generated_text,
            tokens=tokens,
            prompt_token_count=prompt_token_count,
            offset_list=offset_list,
            prompt_length=prompt_length,
        )
    )

    cot_token_start = prompt_token_count
    if answer_token_start is not None:
        cot_token_end = answer_token_start - 1
    else:
        cot_token_end = total_token_count - 2

    if answer_choice_position is not None:
        target_position = answer_choice_position
    else:
        target_position = -1

    cot_relevance = analyzer.compute_relevance(
        full_input_ids, target_position=target_position
    )

    # analyze_combined と同一のフィルタリング (プロンプト部と回答部を0に)
    cot_filtered_relevance = cot_relevance.clone()
    for i in range(prompt_token_count):
        cot_filtered_relevance[i] = 0.0
    if answer_token_start is not None:
        for i in range(answer_token_start, total_token_count):
            cot_filtered_relevance[i] = 0.0

    cot_word_scores = analyzer.tokens_to_words(tokens, cot_filtered_relevance)
    cot_token_scores = [
        (tokens[i], cot_relevance[i].item()) for i in range(len(tokens))
    ]

    result = {
        "type": "cot",
        "token_scores": cot_token_scores,
        "word_scores": [
            {"word": ws.word, "score": ws.score, "token_indices": ws.token_indices}
            for ws in cot_word_scores
        ],
        "raw_relevance": cot_filtered_relevance.cpu(),
        "cot_token_start": cot_token_start,
        "cot_token_end": cot_token_end,
        # 追加メタ情報 (既存スキーマには無いが analysis 側は未使用キーを無視する)
        "fixed_target_position": target_position,
        "fixed_answer_token_start": answer_token_start,
    }
    del cot_relevance, cot_filtered_relevance, full_input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
