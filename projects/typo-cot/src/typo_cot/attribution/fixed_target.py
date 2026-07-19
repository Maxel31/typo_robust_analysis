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


# MATH-500 (boxed 形式) 対応。evaluation.extractor.MATHAnswerExtractor と同じ
# 「最後の閉じた \boxed{...} を採用・未閉じは無視」規約で中身スパンを特定する。
BOXED_RE = re.compile(r"\\boxed\s*\{")
# boxed 直前の回答トリガー句 ("The answer is" / "The final answer is: $" 等)。
# 回答スパン開始 (= CoT 終端の決定) にのみ使用する。
BOXED_TRIGGER_RE = re.compile(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?\s*$")


@dataclass
class BoxedAnswer:
    """最後の閉じた \\boxed{...} の位置情報 (文字オフセット).

    Attributes:
        pattern_start: 回答パターン開始 (トリガー句があればその先頭、なければ \\boxed)
        content_start: 中身の開始 (\\boxed{ の直後)
        content_end: 中身の終了 (閉じ括弧の位置; exclusive)
        content: 中身の生文字列 (strip しない — splice スパンと同一)
    """

    pattern_start: int
    content_start: int
    content_end: int
    content: str


def find_boxed_answer(text: str) -> BoxedAnswer | None:
    """最後の「閉じた」\\boxed{...} の中身スパンを括弧追跡で返す.

    extractor の MATHAnswerExtractor.extract と同じ採用規約:
    閉じた boxed のうち最後のものを採用し、末尾の未閉じ boxed は無視する。
    """
    last: tuple[int, int, int] | None = None
    for m in BOXED_RE.finditer(text):
        content_start = m.end()
        depth = 1
        i = content_start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:  # 未閉じ (途中打ち切り) は無視
            continue
        last = (m.start(), content_start, i)
    if last is None:
        return None
    boxed_start, content_start, content_end = last
    trigger = BOXED_TRIGGER_RE.search(text, 0, boxed_start)
    pattern_start = trigger.start() if trigger else boxed_start
    return BoxedAnswer(
        pattern_start=pattern_start,
        content_start=content_start,
        content_end=content_end,
        content=text[content_start:content_end],
    )


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

    MATH (boxed 形式): どちらかのテキストに \\boxed があれば boxed 経路を優先する。
    片側にしか閉じた boxed が無いサンプルはスキップ (union 除外の strict=boxed
    規約と同じ母集団になる)。両側に \\boxed が無ければ従来経路 (回帰なし)。
    """
    base_boxed = find_boxed_answer(baseline_text)
    pert_boxed = find_boxed_answer(perturbed_text)
    if base_boxed is not None or pert_boxed is not None:
        if base_boxed is None:
            return SplicePlan(
                sample_id=sample_id, skip_reason="no_baseline_answer_pattern"
            )
        if pert_boxed is None:
            return SplicePlan(
                sample_id=sample_id,
                skip_reason="no_perturbed_answer_pattern",
                baseline_answer=base_boxed.content,
                baseline_pattern_type="boxed",
            )
        s, e = pert_boxed.content_start, pert_boxed.content_end
        if pert_boxed.content == base_boxed.content:
            spliced_text = perturbed_text
            spliced = False
        else:
            spliced_text = (
                perturbed_text[:s] + base_boxed.content + perturbed_text[e:]
            )
            spliced = True
        return SplicePlan(
            sample_id=sample_id,
            spliced=spliced,
            spliced_text=spliced_text,
            baseline_answer=base_boxed.content,
            perturbed_answer=pert_boxed.content,
            baseline_pattern_type="boxed",
            perturbed_pattern_type="boxed",
        )

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


def fixed_target_entry(entry: dict[str, Any], plan: SplicePlan) -> dict[str, Any]:
    """results.json 用エントリを作る (摂動 entry のコピー + fixed_target メタデータ).

    flip (GPU 再計算) と非flip (default 再利用) の両方で同一のスキーマを使う。
    元 entry は変更しない。
    """
    out = dict(entry)
    out["fixed_target"] = {
        "baseline_answer": plan.baseline_answer,
        "perturbed_answer": plan.perturbed_answer,
        "spliced": plan.spliced,
        "baseline_pattern_type": plan.baseline_pattern_type,
        "perturbed_pattern_type": plan.perturbed_pattern_type,
    }
    return out


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


def map_answer_char_spans_to_tokens(
    offset_list: list[tuple[int, int]],
    prompt_token_count: int,
    answer_char_start: int,
    answer_char_end: int,
    choice_char_start: int,
    choice_char_end: int,
) -> tuple[int | None, int | None, int | None]:
    """文字スパンをトークン位置へ写像する (lrp.analyzer._find_answer_pattern と同一規約).

    - answer_token_start: プロンプト以降で最初の「end > answer_char_start」トークン
    - answer_choice_position: 最初の「start < choice_char_end かつ end > choice_char_start」
    - answer_token_end: 最後の「start < answer_char_end」トークン
    プロンプト範囲 (i < prompt_token_count) のトークンはスパンに重なっても不採用。
    """
    answer_token_start: int | None = None
    answer_token_end: int | None = None
    answer_choice_position: int | None = None
    for i, (start, end) in enumerate(offset_list):
        if i < prompt_token_count:
            continue
        if answer_token_start is None and end > answer_char_start:
            answer_token_start = i
        if (
            answer_choice_position is None
            and start < choice_char_end
            and end > choice_char_start
        ):
            answer_choice_position = i
        if start < answer_char_end:
            answer_token_end = i
    return answer_token_start, answer_token_end, answer_choice_position


def find_answer_token_positions(
    generated_text: str,
    tokens: list[str],
    prompt_token_count: int,
    offset_list: list[tuple[int, int]],
    prompt_length: int,
    analyzer: Any = None,
) -> tuple[int | None, int | None, int | None]:
    """回答スパンのトークン位置を boxed 優先で決定する.

    - 生成テキストに閉じた \\boxed があれば boxed 経路:
      ターゲット = boxed 中身の先頭トークン、回答スパン = トリガー句先頭〜閉じ括弧。
    - なければ analyzer._find_answer_pattern (choice/number の従来規約) へ委譲。
      analyzer が None の場合は (None, None, None)。
    """
    boxed = find_boxed_answer(generated_text)
    if boxed is not None:
        return map_answer_char_spans_to_tokens(
            offset_list=offset_list,
            prompt_token_count=prompt_token_count,
            answer_char_start=prompt_length + boxed.pattern_start,
            answer_char_end=prompt_length + boxed.content_end + 1,  # 閉じ括弧含む
            choice_char_start=prompt_length + boxed.content_start,
            choice_char_end=prompt_length + boxed.content_end,
        )
    if analyzer is None:
        return None, None, None
    return analyzer._find_answer_pattern(
        generated_text=generated_text,
        tokens=tokens,
        prompt_token_count=prompt_token_count,
        offset_list=offset_list,
        prompt_length=prompt_length,
    )


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
        find_answer_token_positions(
            generated_text=generated_text,
            tokens=tokens,
            prompt_token_count=prompt_token_count,
            offset_list=offset_list,
            prompt_length=prompt_length,
            analyzer=analyzer,
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
