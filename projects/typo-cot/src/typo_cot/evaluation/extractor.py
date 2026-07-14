"""回答抽出モジュール.

各ベンチマークの回答形式に対応した回答抽出機能を提供する。
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """回答抽出結果.

    Attributes:
        extracted_answer: 抽出された回答
        raw_text: 元のテキスト
        confidence: 抽出の信頼度（0.0-1.0）
        extraction_method: 使用した抽出方法
    """

    extracted_answer: str
    raw_text: str
    confidence: float
    extraction_method: str


class BaseAnswerExtractor(ABC):
    """回答抽出器の基底クラス."""

    # 厳密モードで受け入れる extraction_method 名. サブクラスでオーバライド可.
    # ベンチマーク標準の canonical な回答フォーマット (例: "The answer is X")
    # のみを strict とし、fallback (last_letter / first_sentence / 行末数値 等)
    # は除外対象とする.
    STRICT_METHODS: tuple[str, ...] = ("pattern_1", "pattern_2")

    @abstractmethod
    def extract(self, generated_text: str) -> ExtractionResult:
        """生成テキストから回答を抽出.

        Args:
            generated_text: モデルの生成テキスト

        Returns:
            抽出結果
        """
        pass

    @abstractmethod
    def is_correct(self, extracted: str, correct_answer: str) -> bool:
        """回答が正解かどうかを判定.

        Args:
            extracted: 抽出された回答
            correct_answer: 正解

        Returns:
            正解の場合True
        """
        pass

    def extract_strict(self, generated_text: str) -> str:
        """厳密モードでの抽出.

        `STRICT_METHODS` に含まれる extraction_method でマッチした場合のみ
        回答文字列を返す. fallback パターンや no_match の場合は空文字を返す.

        Args:
            generated_text: モデルの生成テキスト

        Returns:
            厳密マッチ時の抽出文字列、それ以外は空文字
        """
        if not generated_text:
            return ""
        result = self.extract(generated_text)
        if result.extraction_method in self.STRICT_METHODS and result.extracted_answer:
            return result.extracted_answer
        return ""


class GSM8KAnswerExtractor(BaseAnswerExtractor):
    """GSM8K用回答抽出器.

    "The answer is [数値]." 形式から数値を抽出する。
    """

    # 回答パターン（優先度順）
    PATTERNS = [
        r"[Tt]he answer is[:\s]*(-?[\d,]+(?:\.\d+)?)",  # The answer is 123
        r"[Aa]nswer[:\s]*(-?[\d,]+(?:\.\d+)?)",  # Answer: 123
        r"=\s*(-?[\d,]+(?:\.\d+)?)\s*$",  # = 123（行末）
        r"(-?[\d,]+(?:\.\d+)?)\s*$",  # 123（行末の数値）
    ]
    # canonical な "The answer is N" / "Answer: N" のみ strict.
    # 行末 "= N" / "N" 単独は緩い fallback として除外対象.
    STRICT_METHODS = ("pattern_1", "pattern_2")

    def extract(self, generated_text: str) -> ExtractionResult:
        """数値回答を抽出."""
        # 各パターンを試行
        for i, pattern in enumerate(self.PATTERNS):
            match = re.search(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
            if match:
                # カンマを除去して数値を正規化
                answer = match.group(1).replace(",", "")
                confidence = 1.0 - (i * 0.2)  # パターンの優先度に応じた信頼度

                return ExtractionResult(
                    extracted_answer=answer,
                    raw_text=generated_text,
                    confidence=max(0.2, confidence),
                    extraction_method=f"pattern_{i + 1}",
                )

        # パターンにマッチしない場合
        return ExtractionResult(
            extracted_answer="",
            raw_text=generated_text,
            confidence=0.0,
            extraction_method="no_match",
        )

    def is_correct(self, extracted: str, correct_answer: str) -> bool:
        """数値の一致を判定."""
        if not extracted:
            return False

        try:
            # 小数点以下の精度を考慮した比較
            extracted_num = float(extracted)
            correct_num = float(correct_answer.replace(",", ""))
            return abs(extracted_num - correct_num) < 1e-6
        except ValueError:
            return extracted.strip() == correct_answer.strip()


class MMLUAnswerExtractor(BaseAnswerExtractor):
    """MMLU用回答抽出器.

    "The answer is (X)" 形式から選択肢（A-D）を抽出する。
    """

    # 回答パターン（優先度順）
    PATTERNS = [
        r"[Tt]he answer is\s*\(?([A-Da-d])\)?",  # The answer is (A) or The answer is A
        r"[Aa]nswer[:\s]*\(?([A-Da-d])\)?",  # Answer: A or Answer: (A)
        r"\b([A-Da-d])\s*[.:]?\s*$",  # A. (行末)
        r"^\s*\(?([A-Da-d])\)?[.:]?\s*$",  # (A) (行頭)
    ]
    # canonical な "The answer is X" / "Answer: X" のみ strict.
    # pattern_3/4 (行末/行頭の A-D 単独) と last_letter は緩い fallback.
    STRICT_METHODS = ("pattern_1", "pattern_2")

    def extract(self, generated_text: str) -> ExtractionResult:
        """選択肢を抽出."""
        # 各パターンを試行
        for i, pattern in enumerate(self.PATTERNS):
            match = re.search(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper()
                confidence = 1.0 - (i * 0.15)

                return ExtractionResult(
                    extracted_answer=answer,
                    raw_text=generated_text,
                    confidence=max(0.4, confidence),
                    extraction_method=f"pattern_{i + 1}",
                )

        # パターンにマッチしない場合、テキスト内のA-Dを探す
        letters_found = re.findall(r"\b([A-Da-d])\b", generated_text)
        if letters_found:
            # 最後に出現した選択肢を採用
            return ExtractionResult(
                extracted_answer=letters_found[-1].upper(),
                raw_text=generated_text,
                confidence=0.3,
                extraction_method="last_letter",
            )

        return ExtractionResult(
            extracted_answer="",
            raw_text=generated_text,
            confidence=0.0,
            extraction_method="no_match",
        )

    def is_correct(self, extracted: str, correct_answer: str) -> bool:
        """選択肢の一致を判定."""
        return extracted.upper() == correct_answer.upper()


class MMLUProAnswerExtractor(MMLUAnswerExtractor):
    """MMLU-Pro用回答抽出器.

    MMLUと同じ形式だが、選択肢がA-Jまで拡張されている。
    """

    # 回答パターン（A-Jに対応）
    PATTERNS = [
        r"[Tt]he answer is\s*\(?([A-Ja-j])\)?",
        r"[Aa]nswer[:\s]*\(?([A-Ja-j])\)?",
        r"\b([A-Ja-j])\s*[.:]?\s*$",
        r"^\s*\(?([A-Ja-j])\)?[.:]?\s*$",
    ]
    STRICT_METHODS = ("pattern_1", "pattern_2")

    def extract(self, generated_text: str) -> ExtractionResult:
        """選択肢を抽出（A-J対応）."""
        for i, pattern in enumerate(self.PATTERNS):
            match = re.search(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper()
                confidence = 1.0 - (i * 0.15)

                return ExtractionResult(
                    extracted_answer=answer,
                    raw_text=generated_text,
                    confidence=max(0.4, confidence),
                    extraction_method=f"pattern_{i + 1}",
                )

        # パターンにマッチしない場合
        letters_found = re.findall(r"\b([A-Ja-j])\b", generated_text)
        if letters_found:
            return ExtractionResult(
                extracted_answer=letters_found[-1].upper(),
                raw_text=generated_text,
                confidence=0.3,
                extraction_method="last_letter",
            )

        return ExtractionResult(
            extracted_answer="",
            raw_text=generated_text,
            confidence=0.0,
            extraction_method="no_match",
        )


class SQuADv2AnswerExtractor(BaseAnswerExtractor):
    """SQuAD v2用回答抽出器.

    読解QAの回答テキストを抽出する。
    """

    # SQuAD v2 は抽出型 QA で「canonical な単一パターン」が存在しないため、
    # 唯一明確な signal である "unanswerable_detected" のみ strict 扱い.
    # first_sentence / full_text は緩い fallback として除外.
    STRICT_METHODS = ("unanswerable_detected",)

    def extract(self, generated_text: str) -> ExtractionResult:
        """回答テキストを抽出."""
        # テキストをクリーンアップ
        answer = generated_text.strip()

        # 回答不可能パターンの検出
        unanswerable_patterns = [
            r"unanswerable",
            r"cannot be answered",
            r"no answer",
            r"not mentioned",
            r"not stated",
        ]

        for pattern in unanswerable_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return ExtractionResult(
                    extracted_answer="",
                    raw_text=generated_text,
                    confidence=0.8,
                    extraction_method="unanswerable_detected",
                )

        # 最初の文または句を抽出（簡易的な回答抽出）
        # ピリオド、改行、または特定のパターンで区切る
        first_sentence_match = re.match(r"^([^.!?\n]+)", answer)
        if first_sentence_match:
            extracted = first_sentence_match.group(1).strip()
            return ExtractionResult(
                extracted_answer=extracted,
                raw_text=generated_text,
                confidence=0.7,
                extraction_method="first_sentence",
            )

        return ExtractionResult(
            extracted_answer=answer,
            raw_text=generated_text,
            confidence=0.5,
            extraction_method="full_text",
        )

    def is_correct(self, extracted: str, correct_answer: str) -> bool:
        """回答の一致を判定.

        SQuAD v2では完全一致ではなく、正規化後の一致を判定する。
        """
        if not correct_answer:
            # 回答不可能な質問の場合
            return extracted == ""

        # 正規化
        def normalize(text: str) -> str:
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)  # 句読点除去
            text = " ".join(text.split())  # 空白正規化
            return text

        return normalize(extracted) == normalize(correct_answer)

    def compute_f1(self, extracted: str, correct_answer: str) -> float:
        """F1スコアを計算.

        Args:
            extracted: 抽出された回答
            correct_answer: 正解

        Returns:
            F1スコア（0.0-1.0）
        """
        if not correct_answer:
            return 1.0 if not extracted else 0.0

        def get_tokens(text: str) -> set[str]:
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            return set(text.split())

        extracted_tokens = get_tokens(extracted)
        correct_tokens = get_tokens(correct_answer)

        if not extracted_tokens or not correct_tokens:
            return 0.0

        common = extracted_tokens & correct_tokens
        if not common:
            return 0.0

        precision = len(common) / len(extracted_tokens)
        recall = len(common) / len(correct_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def compute_em(self, extracted: str, correct_answer: str) -> float:
        """Exact Match (EM) スコアを計算.

        正規化後の文字列が完全一致する場合は1.0、それ以外は0.0を返す。

        Args:
            extracted: 抽出された回答
            correct_answer: 正解

        Returns:
            EMスコア（0.0または1.0）
        """
        if not correct_answer:
            # 回答不可能な質問の場合
            return 1.0 if not extracted else 0.0

        def normalize(text: str) -> str:
            """テキストを正規化."""
            text = text.lower()
            # 冠詞を除去
            text = re.sub(r"\b(a|an|the)\b", " ", text)
            # 句読点を除去
            text = re.sub(r"[^\w\s]", "", text)
            # 空白を正規化
            text = " ".join(text.split())
            return text

        return 1.0 if normalize(extracted) == normalize(correct_answer) else 0.0

    def compute_scores(self, extracted: str, correct_answer: str) -> dict[str, float]:
        """EMスコアとF1スコアの両方を計算.

        Args:
            extracted: 抽出された回答
            correct_answer: 正解

        Returns:
            {"em": EMスコア, "f1": F1スコア} の辞書
        """
        return {
            "em": self.compute_em(extracted, correct_answer),
            "f1": self.compute_f1(extracted, correct_answer),
        }


class CommonsenseQAAnswerExtractor(MMLUAnswerExtractor):
    """CommonsenseQA用回答抽出器.

    MMLUと同じ形式だが、選択肢がA-Eの5択に対応。
    """

    # 回答パターン（A-Eに対応）
    PATTERNS = [
        r"[Tt]he answer is\s*\(?([A-Ea-e])\)?",
        r"[Aa]nswer[:\s]*\(?([A-Ea-e])\)?",
        r"\b([A-Ea-e])\s*[.:]?\s*$",
        r"^\s*\(?([A-Ea-e])\)?[.:]?\s*$",
    ]
    STRICT_METHODS = ("pattern_1", "pattern_2")

    def extract(self, generated_text: str) -> ExtractionResult:
        """選択肢を抽出（A-E対応）."""
        for i, pattern in enumerate(self.PATTERNS):
            match = re.search(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper()
                confidence = 1.0 - (i * 0.15)

                return ExtractionResult(
                    extracted_answer=answer,
                    raw_text=generated_text,
                    confidence=max(0.4, confidence),
                    extraction_method=f"pattern_{i + 1}",
                )

        # パターンにマッチしない場合
        letters_found = re.findall(r"\b([A-Ea-e])\b", generated_text)
        if letters_found:
            return ExtractionResult(
                extracted_answer=letters_found[-1].upper(),
                raw_text=generated_text,
                confidence=0.3,
                extraction_method="last_letter",
            )

        return ExtractionResult(
            extracted_answer="",
            raw_text=generated_text,
            confidence=0.0,
            extraction_method="no_match",
        )


class BBHAnswerExtractor(BaseAnswerExtractor):
    """BBH 用回答抽出器.

    23 サブタスクの解答形式が多様（多肢選択、Yes/No、数値、リスト等）の
    ため、"The answer is X" / "Answer: X" の形式で抽出し、案件ごとに
    正規化したうえで完全一致判定する.
    """

    PATTERNS = [
        r"[Tt]he answer is[:\s]*\(?([A-Za-z0-9.,/\-+_ ]+?)\)?\s*\.?\s*$",
        r"[Tt]he answer is[:\s]*\(?([^\n\.]+?)\)?\s*\.",
        r"[Aa]nswer[:\s]*\(?([^\n\.]+?)\)?\s*\.?\s*$",
    ]
    # BBH のパターンは全て canonical "The answer is X" / "Answer: X" 系で
    # fallback を持たないため、全 3 パターン strict.
    STRICT_METHODS = ("pattern_1", "pattern_2", "pattern_3")

    def extract(self, generated_text: str) -> ExtractionResult:
        for i, pat in enumerate(self.PATTERNS):
            m = re.search(pat, generated_text, re.MULTILINE)
            if m:
                ans = m.group(1).strip().rstrip(".").strip()
                return ExtractionResult(
                    extracted_answer=ans,
                    raw_text=generated_text,
                    confidence=max(0.3, 1.0 - i * 0.2),
                    extraction_method=f"pattern_{i + 1}",
                )
        return ExtractionResult(
            extracted_answer="",
            raw_text=generated_text,
            confidence=0.0,
            extraction_method="no_match",
        )

    def is_correct(self, extracted: str, correct_answer: str) -> bool:
        if not extracted:
            return False
        e = extracted.strip().lower().strip("().,")
        c = correct_answer.strip().lower().strip("().,")
        return e == c


class MATHAnswerExtractor(BaseAnswerExtractor):
    """MATH 用回答抽出器.

    \\boxed{...} 内の式を回答として抽出する. ネスト中括弧に対応するため
    \\boxed{} の対応括弧を文字単位で追跡する.
    """

    BOXED_RE = re.compile(r"\\boxed\s*\{")
    # MATH の canonical 出力は \\boxed{...} のみ strict.
    # "The answer is X" 形式の answer_is は曖昧でフォールバック扱い.
    STRICT_METHODS = ("boxed",)

    @staticmethod
    def _extract_boxed_content(text: str) -> str | None:
        """\\boxed{...} 内の中身（ネスト括弧対応）を返す。"""
        m = MATHAnswerExtractor.BOXED_RE.search(text)
        if not m:
            return None
        i = m.end()
        depth = 1
        buf: list[str] = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                buf.append(ch)
            else:
                buf.append(ch)
            i += 1
        if depth != 0:
            return None
        return "".join(buf).strip()

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.strip()
        s = s.rstrip(". \t\n")
        s = re.sub(r"\s+", "", s)
        s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        return s

    def extract(self, generated_text: str) -> ExtractionResult:
        # 最後の \\boxed{...} を採用（モデルが複数書く場合に最終答えを優先）
        last_match_text = None
        text = generated_text
        while True:
            content = self._extract_boxed_content(text)
            if content is None:
                break
            last_match_text = content
            next_pos = self.BOXED_RE.search(text)
            if next_pos is None:
                break
            text = text[next_pos.end():]
        if last_match_text is not None:
            return ExtractionResult(
                extracted_answer=last_match_text,
                raw_text=generated_text,
                confidence=0.9,
                extraction_method="boxed",
            )
        # フォールバック: "The answer is X"
        m = re.search(r"[Tt]he answer is[:\s]*([^\n\.]+)", generated_text)
        if m:
            return ExtractionResult(
                extracted_answer=m.group(1).strip(),
                raw_text=generated_text,
                confidence=0.5,
                extraction_method="answer_is",
            )
        return ExtractionResult(
            extracted_answer="",
            raw_text=generated_text,
            confidence=0.0,
            extraction_method="no_match",
        )

    def is_correct(self, extracted: str, correct_answer: str) -> bool:
        if not extracted:
            return False
        return self._normalize(extracted) == self._normalize(correct_answer)


class StrategyQAAnswerExtractor(BaseAnswerExtractor):
    """StrategyQA 用回答抽出器.

    yes/no の二択. "The answer is Yes/No" / "Yes" / "No" を許容する.
    """

    PATTERNS = [
        r"[Tt]he answer is[:\s]*(yes|no)\b",
        r"[Aa]nswer[:\s]*(yes|no)\b",
        r"\b(yes|no)\s*\.?\s*$",
    ]
    # canonical "The answer is Yes/No" / "Answer: Yes/No" のみ strict.
    # pattern_3 (Yes/No 単独で行末) は緩いので除外.
    STRICT_METHODS = ("pattern_1", "pattern_2")

    def extract(self, generated_text: str) -> ExtractionResult:
        for i, pat in enumerate(self.PATTERNS):
            m = re.search(pat, generated_text, re.IGNORECASE | re.MULTILINE)
            if m:
                ans = m.group(1).strip().lower()
                return ExtractionResult(
                    extracted_answer=ans,
                    raw_text=generated_text,
                    confidence=max(0.3, 1.0 - i * 0.2),
                    extraction_method=f"pattern_{i + 1}",
                )
        return ExtractionResult(
            extracted_answer="",
            raw_text=generated_text,
            confidence=0.0,
            extraction_method="no_match",
        )

    def is_correct(self, extracted: str, correct_answer: str) -> bool:
        if not extracted:
            return False
        return extracted.strip().lower() == correct_answer.strip().lower()


def create_extractor(benchmark: str) -> BaseAnswerExtractor:
    """ベンチマーク名から回答抽出器を作成するファクトリ関数.

    Args:
        benchmark: ベンチマーク名

    Returns:
        対応する回答抽出器

    Raises:
        ValueError: 不明なベンチマーク名の場合
    """
    extractors = {
        "mmlu": MMLUAnswerExtractor,
        "mmlu_pro": MMLUProAnswerExtractor,
        "gsm8k": GSM8KAnswerExtractor,
        "squad_v2": SQuADv2AnswerExtractor,
        "arc": MMLUAnswerExtractor,  # 4択 (A-D) なのでMMLUと同じ
        "commonsense_qa": CommonsenseQAAnswerExtractor,  # 5択 (A-E)
        "bbh": BBHAnswerExtractor,
        "math": MATHAnswerExtractor,
        "strategy_qa": StrategyQAAnswerExtractor,
    }

    if benchmark not in extractors:
        raise ValueError(f"不明なベンチマーク: {benchmark}. 利用可能: {list(extractors.keys())}")

    return extractors[benchmark]()
