"""文字レベル摂動生成モジュール."""

import random
from dataclasses import dataclass
from enum import Enum


class PerturbationType(Enum):
    """摂動の種類."""

    PROXIMITY = "proximity"  # キーボード上の隣接キーに置換
    DOUBLE_TYPING = "double_typing"  # 同じ文字を重複挿入
    OMISSION = "omission"  # 1文字削除


@dataclass
class PerturbationResult:
    """摂動結果のデータクラス.

    Attributes:
        original: 元のテキスト
        perturbed: 摂動後のテキスト
        perturbation_type: 適用した摂動の種類
        position: 摂動を適用した位置
        original_char: 元の文字（削除・置換の場合）
        new_char: 新しい文字（挿入・置換の場合）
    """

    original: str
    perturbed: str
    perturbation_type: PerturbationType
    position: int
    original_char: str | None = None
    new_char: str | None = None


class CharacterPerturbationGenerator:
    """文字レベル摂動生成器.

    3種類の摂動を適用:
    - Proximity: キーボード上で隣接するキーに置換
    - Double typing: 同じ文字を重複挿入
    - Omission: 1文字削除
    """

    # QWERTYキーボードレイアウトの隣接キー定義（小文字）
    KEYBOARD_NEIGHBORS: dict[str, list[str]] = {
        "q": ["w", "a"],
        "w": ["q", "e", "a", "s"],
        "e": ["w", "r", "s", "d"],
        "r": ["e", "t", "d", "f"],
        "t": ["r", "y", "f", "g"],
        "y": ["t", "u", "g", "h"],
        "u": ["y", "i", "h", "j"],
        "i": ["u", "o", "j", "k"],
        "o": ["i", "p", "k", "l"],
        "p": ["o", "l"],
        "a": ["q", "w", "s", "z"],
        "s": ["a", "w", "e", "d", "z", "x"],
        "d": ["s", "e", "r", "f", "x", "c"],
        "f": ["d", "r", "t", "g", "c", "v"],
        "g": ["f", "t", "y", "h", "v", "b"],
        "h": ["g", "y", "u", "j", "b", "n"],
        "j": ["h", "u", "i", "k", "n", "m"],
        "k": ["j", "i", "o", "l", "m"],
        "l": ["k", "o", "p"],
        "z": ["a", "s", "x"],
        "x": ["z", "s", "d", "c"],
        "c": ["x", "d", "f", "v"],
        "v": ["c", "f", "g", "b"],
        "b": ["v", "g", "h", "n"],
        "n": ["b", "h", "j", "m"],
        "m": ["n", "j", "k"],
    }

    def __init__(self, seed: int | None = None) -> None:
        """初期化.

        Args:
            seed: ランダムシード（再現性のため）
        """
        self.rng = random.Random(seed)

    def _is_alphabet(self, char: str) -> bool:
        """アルファベットかどうかを判定.

        Args:
            char: 判定する文字

        Returns:
            アルファベットの場合True
        """
        return char.isalpha() and char.isascii()

    def _get_proximity_char(self, char: str) -> str | None:
        """キーボード上で隣接する文字を取得.

        Args:
            char: 元の文字

        Returns:
            隣接する文字、または隣接キーがない場合はNone
        """
        lower_char = char.lower()

        if lower_char not in self.KEYBOARD_NEIGHBORS:
            return None

        neighbors = self.KEYBOARD_NEIGHBORS[lower_char]
        if not neighbors:
            return None

        new_char = self.rng.choice(neighbors)

        # 元の文字が大文字の場合は大文字で返す
        if char.isupper():
            return new_char.upper()
        return new_char

    def proximity_replace(self, text: str) -> PerturbationResult | None:
        """ランダムな位置の文字をキーボード上の隣接キーに置換.

        Args:
            text: 入力テキスト

        Returns:
            摂動結果、または置換できない場合はNone
        """
        if len(text) == 0:
            return None

        # 置換可能な位置（アルファベットの位置）を特定
        replaceable_positions = [
            i for i, c in enumerate(text) if self._is_alphabet(c)
        ]

        if not replaceable_positions:
            return None

        position = self.rng.choice(replaceable_positions)
        original_char = text[position]
        new_char = self._get_proximity_char(original_char)

        if new_char is None:
            return None

        perturbed = text[:position] + new_char + text[position + 1 :]

        return PerturbationResult(
            original=text,
            perturbed=perturbed,
            perturbation_type=PerturbationType.PROXIMITY,
            position=position,
            original_char=original_char,
            new_char=new_char,
        )

    def double_typing(self, text: str) -> PerturbationResult | None:
        """ランダムな位置のアルファベットの後に同じ文字を挿入（タイプミス風）.

        例: apple -> applle

        Args:
            text: 入力テキスト

        Returns:
            摂動結果、または挿入できない場合はNone
        """
        if len(text) == 0:
            return None

        # 挿入可能な位置（アルファベットの位置）を特定
        insertable_positions = [
            i for i, c in enumerate(text) if self._is_alphabet(c)
        ]

        if not insertable_positions:
            return None

        position = self.rng.choice(insertable_positions)
        char_to_double = text[position]

        # 対象文字の後に同じ文字を挿入
        perturbed = text[: position + 1] + char_to_double + text[position + 1 :]

        return PerturbationResult(
            original=text,
            perturbed=perturbed,
            perturbation_type=PerturbationType.DOUBLE_TYPING,
            position=position,
            original_char=None,
            new_char=char_to_double,
        )

    def omission(self, text: str) -> PerturbationResult | None:
        """ランダムな位置の1文字を削除.

        Args:
            text: 入力テキスト

        Returns:
            摂動結果、または削除できない場合はNone
        """
        if len(text) <= 1:
            # 1文字以下の場合は削除しない
            return None

        # 削除可能な位置（アルファベットの位置）を特定
        deletable_positions = [
            i for i, c in enumerate(text) if self._is_alphabet(c)
        ]

        if not deletable_positions:
            return None

        position = self.rng.choice(deletable_positions)
        original_char = text[position]
        perturbed = text[:position] + text[position + 1 :]

        return PerturbationResult(
            original=text,
            perturbed=perturbed,
            perturbation_type=PerturbationType.OMISSION,
            position=position,
            original_char=original_char,
            new_char=None,
        )

    def perturb(self, text: str) -> PerturbationResult | None:
        """ランダムに1種類の摂動を適用.

        Args:
            text: 入力テキスト

        Returns:
            摂動結果、または摂動できない場合はNone
        """
        if len(text) == 0:
            return None

        # 1文字の場合はOmissionを除外
        if len(text) == 1:
            perturbation_types = [PerturbationType.PROXIMITY, PerturbationType.DOUBLE_TYPING]
        else:
            perturbation_types = list(PerturbationType)

        # ランダムに摂動タイプを選択して試行
        self.rng.shuffle(perturbation_types)

        for ptype in perturbation_types:
            if ptype == PerturbationType.OMISSION:
                result = self.omission(text)
            elif ptype == PerturbationType.PROXIMITY:
                result = self.proximity_replace(text)
            else:  # DOUBLE_TYPING
                result = self.double_typing(text)

            if result is not None:
                return result

        return None

    def perturb_token_in_text(
        self, text: str, token: str, token_start: int
    ) -> tuple[str, PerturbationResult | None]:
        """テキスト内の特定トークンに摂動を適用.

        Args:
            text: 全体のテキスト
            token: 摂動対象のトークン
            token_start: トークンの開始位置（文字単位）

        Returns:
            (摂動後のテキスト, 摂動結果) のタプル
        """
        # トークンに摂動を適用
        result = self.perturb(token)

        if result is None:
            return text, None

        # テキスト内のトークンを置換
        token_end = token_start + len(token)
        perturbed_text = text[:token_start] + result.perturbed + text[token_end:]

        return perturbed_text, result
