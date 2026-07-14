"""reasoning モデル(DeepSeek-R1蒸留系)サポートモジュール.

実験10③: R1蒸留系(reasoning特化 7〜8B級)は実験1・3のみ参加する。
既存の few-shot 生プロンプト規約と異なり、
- DeepSeek 公式推奨に従いゼロショット(few-shot は性能劣化)+チャットテンプレート
- CoT は <think>...</think> タグに包まれ、最終回答はタグ閉鎖後のセクション
- 長大 CoT のため max_new_tokens はベンチマーク別に拡大
という差分を持つ。本モジュールはプロンプト構築(質問スパン追跡つき)、
<think> 分離、回答抽出チェーンを提供する。

decoding は Step 0 凍結レジストリ(configs/registry.yaml)と同じ greedy/seed=42。
"""

import re
from dataclasses import dataclass

from typo_cot.evaluation.extractor import (
    BaseAnswerExtractor,
    ExtractionResult,
    MATHAnswerExtractor,
)

R1_DISTILL_QWEN_7B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ベンチマーク別の zero-shot 指示文。答えトリガーは既存 extractor の
# canonical パターン("The answer is ...")に合わせる。MATH のみ boxed。
_INSTRUCTIONS: dict[str, str] = {
    "gsm8k": (
        "Solve the following math problem step by step. "
        'At the end, give your final answer in the format "The answer is [number]."'
    ),
    "mmlu": (
        "The following is a multiple-choice question. Think through the problem "
        'step by step, then give your final answer in the format "The answer is (X)" '
        "where X is the correct letter choice."
    ),
    "math": (
        "Solve the following math problem step by step. "
        "Put your final answer within \\boxed{}."
    ),
}

# ベンチマーク別のデフォルト max_new_tokens(長い CoT を考慮)
REASONING_MAX_NEW_TOKENS: dict[str, int] = {
    "gsm8k": 4096,
    "mmlu": 4096,
    "math": 8192,
}

_CHOICE_LETTERS = "ABCDEFGHIJ"

_THINK_OPEN_RE = re.compile(r"^\s*<think>\s*")
_THINK_CLOSE = "</think>"


@dataclass
class ReasoningSplit:
    """<think> タグによる CoT / 回答セクションの分離結果.

    Attributes:
        cot_text: CoT 本文(<think> タグの中身、タグ自体は含まない)
        answer_text: </think> 以降の最終回答セクション(閉じタグが無ければ空)
        has_think_close: </think> が生成されたか(False は CoT 途中切断)
    """

    cot_text: str
    answer_text: str
    has_think_close: bool


@dataclass
class ReasoningPrompt:
    """チャットテンプレート適用後のプロンプトと質問スパン.

    オフセットはすべて full_prompt 内の文字位置(既存 PromptResult と同じ規約)。
    """

    full_prompt: str
    user_message: str
    question_text: str
    question_start: int
    question_end: int
    question_with_choices_end: int


def build_user_message(
    benchmark: str,
    question: str,
    choices: list[str] | None = None,
) -> tuple[str, int, int, int]:
    """ゼロショットのユーザーメッセージを構築する.

    Args:
        benchmark: ベンチマーク名 (gsm8k / mmlu / math)
        question: 質問文(摂動データの場合は選択肢が埋め込まれていることがある)
        choices: 選択肢リスト(None の場合は question に埋め込み済みとみなす)

    Returns:
        (メッセージ, 質問開始, 質問終了, 質問+選択肢終了) のタプル。
        オフセットはメッセージ内の文字位置。

    Raises:
        ValueError: 未対応のベンチマーク名の場合
    """
    if benchmark not in _INSTRUCTIONS:
        raise ValueError(
            f"reasoning モデル未対応のベンチマーク: {benchmark}. "
            f"利用可能: {sorted(_INSTRUCTIONS)}"
        )

    instruction = _INSTRUCTIONS[benchmark]

    if benchmark == "mmlu":
        if choices:
            options_str = " ".join(
                f"({_CHOICE_LETTERS[i]}) {c}" for i, c in enumerate(choices)
            )
            body = f"{question}\n{options_str}"
            q_len = len(question)
        else:
            # 摂動データセット: question に選択肢が埋め込まれている。
            # 既存 MMLUPromptTemplate と同じく最初の改行までを質問文とみなす。
            body = question
            first_newline = question.find("\n")
            q_len = first_newline if first_newline > 0 else len(question)
        prefix = f"{instruction}\n\nQuestion: "
        msg = f"{prefix}{body}"
        q_start = len(prefix)
        return msg, q_start, q_start + q_len, q_start + len(body)

    # 自由記述系 (gsm8k / math): 選択肢なし
    prefix = f"{instruction}\n\nProblem: "
    msg = f"{prefix}{question}"
    q_start = len(prefix)
    q_end = q_start + len(question)
    return msg, q_start, q_end, q_end


def build_full_prompt(
    tokenizer,
    benchmark: str,
    question: str,
    choices: list[str] | None = None,
) -> ReasoningPrompt:
    """チャットテンプレートを適用した完全プロンプトを構築する.

    質問スパンのオフセットは、テンプレート適用後のテキスト内で
    ユーザーメッセージを逐語検索して補正する。

    Args:
        tokenizer: apply_chat_template を持つトークナイザー
        benchmark: ベンチマーク名
        question: 質問文
        choices: 選択肢リスト

    Returns:
        ReasoningPrompt
    """
    msg, q_start, q_end, q_choices_end = build_user_message(benchmark, question, choices)

    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=True,
    )

    offset = full_prompt.find(msg)
    if offset < 0:
        raise ValueError(
            "チャットテンプレート適用後のプロンプトにユーザーメッセージが "
            "逐語で見つかりません(テンプレートが本文を改変しています)"
        )

    return ReasoningPrompt(
        full_prompt=full_prompt,
        user_message=msg,
        question_text=question,
        question_start=offset + q_start,
        question_end=offset + q_end,
        question_with_choices_end=offset + q_choices_end,
    )


def split_reasoning_output(generated_text: str) -> ReasoningSplit:
    """生成テキストを CoT(<think>内)と最終回答セクションに分離する.

    テンプレートが <think>\\n を事前付与する版(生成が本文から始まる)と、
    モデルが自分で <think> を書く版の両方を処理する。

    Args:
        generated_text: モデルの生成テキスト(プロンプト部を除く)

    Returns:
        ReasoningSplit
    """
    text = generated_text
    m = _THINK_OPEN_RE.match(text)
    body = text[m.end() :] if m else text

    close_idx = body.find(_THINK_CLOSE)
    if close_idx >= 0:
        cot = body[:close_idx].strip()
        answer = body[close_idx + len(_THINK_CLOSE) :]
        return ReasoningSplit(cot_text=cot, answer_text=answer, has_think_close=True)

    return ReasoningSplit(cot_text=body.strip(), answer_text="", has_think_close=False)


def think_prefix_end(generated_text: str) -> int:
    """生成テキスト内で CoT 本文が始まる文字位置を返す.

    R_Q(質問トークンの CoT 開始への寄与)計算で、backward の対象を
    <think> タグ直後の最初の実トークンに合わせるために使う。

    Args:
        generated_text: モデルの生成テキスト

    Returns:
        <think> タグ+直後の空白の終了位置(タグが無ければ 0)
    """
    m = _THINK_OPEN_RE.match(generated_text)
    return m.end() if m else 0


_BOXED_NUMBER_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")


def extract_reasoning_answer(
    extractor: BaseAnswerExtractor,
    split: ReasoningSplit,
) -> ExtractionResult:
    """回答セクションからベンチマーク抽出器で答えを抽出する.

    チェーン:
    1. answer_text(</think> 以降)にベンチマーク抽出器を適用
    2. 失敗したら answer_text の \\boxed{...} を試す(R1系の既定出力形式)
    3. answer_text が空(CoT 途中切断)なら cot_text 全文に抽出器を適用

    Args:
        extractor: ベンチマーク別の回答抽出器
        split: split_reasoning_output の結果

    Returns:
        ExtractionResult(extraction_method に使用経路を記録)
    """
    if split.answer_text.strip():
        result = extractor.extract(split.answer_text)
        if result.extracted_answer:
            return result

        # boxed フォールバック(MATH 抽出器は既に boxed を試している)
        if not isinstance(extractor, MATHAnswerExtractor):
            boxed = MATHAnswerExtractor._extract_boxed_content(split.answer_text)
            if boxed is not None:
                normalized = boxed.replace(",", "") if _BOXED_NUMBER_RE.match(boxed) else boxed
                return ExtractionResult(
                    extracted_answer=normalized,
                    raw_text=split.answer_text,
                    confidence=0.8,
                    extraction_method="boxed_fallback",
                )
        return result

    # 回答セクションが無い(切断)場合: CoT 全文から救済抽出
    result = extractor.extract(split.cot_text)
    if result.extracted_answer:
        return ExtractionResult(
            extracted_answer=result.extracted_answer,
            raw_text=split.cot_text,
            confidence=min(0.3, result.confidence),
            extraction_method=f"cot_{result.extraction_method}",
        )
    return result
