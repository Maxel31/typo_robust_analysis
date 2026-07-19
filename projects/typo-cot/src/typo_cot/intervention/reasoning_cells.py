"""実験1+3: reasoning モデル (DeepSeek-R1蒸留系) のセル入力サポート.

基底5モデルは生プロンプト (few-shot completion) + "The answer is" トリガー前提。
R1蒸留系は差分を持つ:
- (a) チャットテンプレート形式のゼロショット (DeepSeek 公式推奨)
- (b) CoT が <think>...</think> タグに包まれ、最終回答は </think> 後の本文
- (c) 答えトリガーの位置・形式が異なる (Answer:, ANSWER:, \\boxed{} 等も使う)

本モジュールは cell_builder / runner に **注入する** 差分部品のみを提供し、
4セル (A/B/C/D) の分解ロジックや除外・dedup の枠組みは既存実装を再利用する。

## CoT 移植点 (切断点) の定義

「<think> 開始 〜 </think> 直後の答え文の答え宣言直前」。すなわち
forced CoT = <think>本文 + </think> + (答え文中の最初の答え宣言の直前まで)。
モデルは宣言 + 答えスパンだけを teacher-forcing で再生成する (基底モデルと
同じ短スパン再生成。答えスパンのみなので max_new_tokens は短くて可)。

## 凍結除外規則の <think> 構造への自然拡張

- no_trigger: (a) </think> が生成されず CoT 途中切断、または (b) </think> 後の
  答え文に答え宣言が見つからない場合。基底の no_trigger と同一の意味 (答えに
  到達しない生成) なので同じフラグに集約する。
- multi_trigger: R1 は同一答えを反復宣言する習性が強い (Qwen の先例と同様)。
  最初の宣言直前で切断すれば再生成は最初の宣言を復元するので、反復は曖昧では
  ない。cell_builder の dedup_same_answer_triggers=True と組み合わせ、全宣言が
  同一答えを指す場合は multi_trigger 除外を課さない (答えが途中で変わる真の
  曖昧さのみ除外)。除外込みの値は感度分析として併記する。
- early_trigger: <think> 本文は常に丸ごと移植されるため「答えが序盤に出た」
  概念が当てはまらない。常に False。
- residual_fragment: 答え文の宣言直前プロセにのみ適用 (<think> 本文には
  適用しない — 推論句として "the answer is" 風の表現が常に含まれ得るため)。

decoding は既存パイプラインと同じ greedy (do_sample=False)。
"""

import re

from typo_cot.evaluation.extractor import BaseAnswerExtractor, create_extractor
from typo_cot.intervention.cell_builder import RESIDUAL_PATTERNS, TruncationResult
from typo_cot.models.reasoning import (
    ReasoningSplit,
    build_full_prompt,
    extract_reasoning_answer,
    split_reasoning_output,
)

_THINK_CLOSE = "</think>"

# </think> 後の答え文で使われる答え宣言トリガー。基底の "The answer is" に加え
# R1 が実際に使う宣言形を含める (実測: gsm8k/mmlu/math の baseline+摂動で
# 反復宣言 dedup 込み包含率 gsm8k≈87-92% / mmlu≈73-75% / math≈66-71%)。
REASONING_ANSWER_TRIGGER = (
    r"[Tt]he (?:final |correct )?answer is"
    r"|[Tt]he correct (?:option|choice|answer) is"
    r"|(?:^|\n)\s*(?:\*\*)?[Aa]nswer\s*(?:choice)?\s*[:=]"
    r"|(?:^|\n)\s*(?:\*\*)?ANSWER\s*[:=]"
    r"|\\boxed\{"
)


def truncate_reasoning_cot(
    cot: str,
    benchmark: str,
    trigger_pattern: str | None = None,
) -> TruncationResult:
    """R1 生成テキストを「</think> 後の答え宣言直前」で切断する.

    cell_builder.build_cell_inputs に ``truncator=`` として注入する。返り値は
    既存 TruncationResult なので除外・dedup ロジックはそのまま流用される。

    Args:
        cot: R1 の生成テキスト (先頭 <think> タグはプロンプト側にあるため通常
            含まないが、含む場合も </think> の絶対位置で正しく切断する)
        benchmark: ベンチマーク名 (dedup 用の答え抽出器選択に使用)
        trigger_pattern: 答え宣言の正規表現 (None なら REASONING_ANSWER_TRIGGER)

    Returns:
        TruncationResult。切断点 (prefix) は <think>本文 + </think> +
        答え文の宣言直前まで。trigger_count/trigger_answers は **答え文内** の
        宣言に対して数える (<think> 本文内の推論句は数えない)。
    """
    pattern = trigger_pattern if trigger_pattern is not None else REASONING_ANSWER_TRIGGER
    split = split_reasoning_output(cot)

    not_found = TruncationResult(
        prefix=cot,
        trigger_found=False,
        trigger_count=0,
        early_trigger=False,
        residual_fragment=False,
        trigger_char_start=None,
        trigger_answers=[],
        trigger_answers_identical=False,
    )
    if not split.has_think_close:
        return not_found

    answer = split.answer_text
    matches = list(re.finditer(pattern, answer))
    if not matches:
        return not_found

    # </think> の絶対文字位置 (先頭 <think> の有無に依らず答え文の開始を確定)
    close_idx = cot.find(_THINK_CLOSE)
    answer_start = close_idx + len(_THINK_CLOSE)
    first = matches[0]
    cut = answer_start + first.start()
    prefix = cot[:cut]

    # residual は答え文の宣言直前プロセにのみ適用 (<think> 本文は対象外)
    pre_decl = answer[: first.start()]
    residual = any(re.search(p, pre_decl) for p in RESIDUAL_PATTERNS)

    # 各宣言区間 [decl_i, decl_{i+1}) の答えを抽出 (同一答えの反復 vs 真の曖昧さ)
    trigger_answers: list[str] = []
    if len(matches) > 1:
        extractor = create_extractor(benchmark)
        for idx, m in enumerate(matches):
            seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(answer)
            trigger_answers.append(
                extractor.extract(answer[m.start(): seg_end]).extracted_answer.strip()
            )
    nonempty = [a for a in trigger_answers if a]
    identical = (
        len(trigger_answers) > 1
        and len(nonempty) == len(trigger_answers)
        and len(set(nonempty)) == 1
    )

    return TruncationResult(
        prefix=prefix,
        trigger_found=True,
        trigger_count=len(matches),
        early_trigger=False,
        residual_fragment=residual,
        trigger_char_start=cut,
        trigger_answers=trigger_answers,
        trigger_answers_identical=identical,
    )


def make_reasoning_prompt_builder(tokenizer):
    """チャットテンプレートベースの prompt_builder を返す.

    cell_builder.build_cell_inputs に ``prompt_builder=`` として注入する。
    シグネチャは既定の _build_prompt と同じ (benchmark, question, choices, subset)。
    subset は R1 のゼロショット指示文では未使用 (基底の subject few-shot と異なる)。
    """

    def _builder(
        benchmark: str,
        question: str,
        choices: list[str] | None,
        subset: str | None,
    ) -> str:
        return build_full_prompt(tokenizer, benchmark, question, choices).full_prompt

    return _builder


def make_reasoning_extract_fn(extractor: BaseAnswerExtractor):
    """再生成した答えスパンから R1 抽出チェーンで答えを取り出す関数を返す.

    runner.run_cells に ``extract_fn=`` として注入する。基底の
    extractor.extract よりロバスト ($ 記号・boxed フォールバックを含む
    reasoning.extract_reasoning_answer チェーンを利用)。teacher-forcing で
    宣言直前から再生成されたスパンを answer_text とみなして抽出する。
    """

    def _extract(span: str) -> str:
        split = ReasoningSplit(cot_text="", answer_text=span, has_think_close=True)
        return extract_reasoning_answer(extractor, split).extracted_answer

    return _extract
