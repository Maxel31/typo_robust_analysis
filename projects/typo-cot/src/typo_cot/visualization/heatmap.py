"""ヒートマップ可視化モジュール.

lxtライブラリを使用して、重要度スコアのヒートマップをPDF形式で生成する。
"""

import logging
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _clean_tokens_for_latex(tokens: list[str]) -> list[str]:
    """トークンをLaTeX互換形式にクリーンアップ.

    lxt.utils.clean_tokensの代替実装。
    一部のトークナイザー（Gemma等）でclean_tokensがエラーを起こすため、
    独自にクリーニングを行う。

    pdflatexはUnicode文字をサポートしていないため、
    非ASCII文字は適切に変換または除去する。

    Args:
        tokens: トークン文字列のリスト

    Returns:
        クリーンアップされたトークンのリスト
    """
    # よく使われるUnicode文字のASCII代替マッピング
    unicode_replacements = {
        "α": "a",
        "β": "b",
        "γ": "g",
        "ɣ": "g",  # U+0263 (問題の文字)
        "δ": "d",
        "ε": "e",
        "ζ": "z",
        "η": "n",
        "θ": "th",
        "λ": "l",
        "μ": "u",
        "π": "pi",
        "σ": "s",
        "τ": "t",
        "φ": "ph",
        "ω": "w",
        "Δ": "D",
        "Σ": "S",
        "Ω": "O",
        "∞": "inf",
        "≈": "~",
        "≠": "!=",
        "≤": "<=",
        "≥": ">=",
        "×": "x",
        "÷": "/",
        "±": "+/-",
        "°": "deg",
        "′": "'",
        "″": '"',
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "•": "*",
        "…": "...",
        "—": "--",
        "–": "-",
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
    }

    cleaned = []
    for token in tokens:
        # サブワードプレフィックスを除去（▁, Ġ など）
        t = token.replace("▁", " ").replace("Ġ", " ")

        # Unicode文字を置換
        for unicode_char, replacement in unicode_replacements.items():
            t = t.replace(unicode_char, replacement)

        # LaTeX特殊文字をエスケープ
        t = t.replace("\\", "\\textbackslash{}")
        t = t.replace("{", "\\{")
        t = t.replace("}", "\\}")
        t = t.replace("$", "\\$")
        t = t.replace("%", "\\%")
        t = t.replace("&", "\\&")
        t = t.replace("#", "\\#")
        t = t.replace("_", "\\_")
        t = t.replace("^", "\\textasciicircum{}")
        t = t.replace("~", "\\textasciitilde{}")

        # 制御文字を除去
        t = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", t)

        # 残りの非ASCII文字を除去（pdflatex互換性のため）
        t = t.encode("ascii", errors="ignore").decode("ascii")

        cleaned.append(t)
    return cleaned


def generate_pdf_heatmap(
    tokens: list[str],
    relevance: torch.Tensor,
    output_path: str | Path,
    backend: str = "pdflatex",
) -> bool:
    """重要度スコアのPDFヒートマップを生成.

    lxtのpdf_heatmap関数を使用して、トークンごとの重要度を可視化する。
    https://lxt.readthedocs.io/en/latest/quickstart.html に準拠。

    Args:
        tokens: トークン文字列のリスト
        relevance: 各トークンの重要度スコア (seq_len,)
        output_path: 出力PDFファイルのパス
        backend: LaTeXバックエンド（"pdflatex" または "xelatex"）

    Returns:
        生成に成功した場合はTrue、失敗した場合はFalse
    """
    try:
        from lxt.utils import pdf_heatmap
    except ImportError as e:
        logger.warning(f"lxt.utilsのインポートに失敗しました: {e}")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # relevanceを正規化（-1から1の範囲に）
    if relevance.abs().max() > 0:
        normalized_relevance = relevance / relevance.abs().max()
    else:
        normalized_relevance = relevance

    # トークンをクリーンアップ（LaTeX互換性のない文字を除去）
    # lxt.utils.clean_tokensは一部のトークナイザーでエラーを起こすため、独自実装を使用
    cleaned_tokens = _clean_tokens_for_latex(tokens)

    try:
        pdf_heatmap(
            cleaned_tokens,
            normalized_relevance.cpu() if normalized_relevance.is_cuda else normalized_relevance,
            path=str(output_path),
            backend=backend,
        )
        logger.info(f"ヒートマップを保存: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"ヒートマップ生成に失敗: {e}")
        return False


def generate_question_heatmap(
    tokens: list[str],
    relevance: torch.Tensor,
    offset_mapping: list[tuple[int, int]],
    question_char_start: int,
    question_char_end: int,
    output_path: str | Path,
    backend: str = "pdflatex",
) -> bool:
    """質問文（+選択肢）部分のPDFヒートマップを生成.

    プロンプト全体から指定された文字範囲に対応するトークンを抽出して
    ヒートマップを生成する。選択肢を含める場合は、question_char_endに
    question_with_choices_endの値を渡す。

    Args:
        tokens: プロンプト全体のトークン文字列リスト
        relevance: プロンプト全体の重要度スコア (seq_len,)
        offset_mapping: 各トークンの文字位置 [(start, end), ...]
        question_char_start: 質問文の開始文字位置
        question_char_end: 質問文の終了文字位置（選択肢を含める場合はその終了位置）
        output_path: 出力PDFファイルのパス
        backend: LaTeXバックエンド

    Returns:
        生成に成功した場合はTrue、失敗した場合はFalse
    """
    # 指定範囲内のトークンのみを抽出
    question_tokens: list[str] = []
    question_relevance: list[float] = []

    for i, (start, end) in enumerate(offset_mapping):
        # トークンが指定範囲内にある場合
        if not (end <= question_char_start or start >= question_char_end):
            question_tokens.append(tokens[i])
            question_relevance.append(relevance[i].item())

    if not question_tokens:
        logger.warning("指定範囲内のトークンが見つかりません")
        return False

    # テンソルに変換
    question_relevance_tensor = torch.tensor(question_relevance)

    logger.info(f"質問文+選択肢ヒートマップ: {len(question_tokens)}トークン")

    return generate_pdf_heatmap(
        tokens=question_tokens,
        relevance=question_relevance_tensor,
        output_path=output_path,
        backend=backend,
    )


def generate_cot_heatmap(
    tokens: list[str],
    relevance: torch.Tensor,
    cot_token_start: int,
    cot_token_end: int,
    output_path: str | Path,
    backend: str = "pdflatex",
) -> bool:
    """CoT推論過程部分のみのPDFヒートマップを生成.

    生成テキスト全体からCoT推論過程に対応するトークンのみを抽出して
    ヒートマップを生成する。

    Args:
        tokens: 全体のトークン文字列リスト（プロンプト + 生成テキスト）
        relevance: 全体の重要度スコア (seq_len,)
        cot_token_start: CoT開始トークン位置
        cot_token_end: CoT終了トークン位置
        output_path: 出力PDFファイルのパス
        backend: LaTeXバックエンド

    Returns:
        生成に成功した場合はTrue、失敗した場合はFalse
    """
    # CoT範囲内のトークンのみを抽出
    cot_tokens: list[str] = []
    cot_relevance: list[float] = []

    for i in range(cot_token_start, min(cot_token_end + 1, len(tokens))):
        cot_tokens.append(tokens[i])
        cot_relevance.append(relevance[i].item())

    if not cot_tokens:
        logger.warning("CoT範囲内のトークンが見つかりません")
        return False

    # テンソルに変換
    cot_relevance_tensor = torch.tensor(cot_relevance)

    logger.info(f"CoTヒートマップ: {len(cot_tokens)}トークン")

    return generate_pdf_heatmap(
        tokens=cot_tokens,
        relevance=cot_relevance_tensor,
        output_path=output_path,
        backend=backend,
    )
