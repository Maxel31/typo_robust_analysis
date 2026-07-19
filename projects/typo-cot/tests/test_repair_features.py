"""repair.features のテスト (実験9: 表層統制特徴量).

分割数増分・Zipf 頻度・先頭トークン ID の各特徴量。
GPU 不要。トークナイザはモックで代替する。
"""

from typo_cot.repair.features import first_token_id, split_increment, zipf_frequency


class FakeTokenizer:
    """encode() だけを持つ簡易トークナイザ.

    語彙: 空白始まりの既知語は 1 トークン、未知語は文字数ぶんに分割される、
    という挙動を辞書で模す。
    """

    def __init__(self, vocab: dict[str, list[int]]) -> None:
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False, "特徴量計算では special tokens を付けない"
        if text in self.vocab:
            return self.vocab[text]
        # 未知語: 1文字=1トークン扱い
        return [ord(c) for c in text]


class TestSplitIncrement:
    def test_typo_splits_into_more_pieces(self) -> None:
        tok = FakeTokenizer({" ducks": [101], " dicks": [102, 103]})
        assert split_increment(tok, "ducks", "dicks") == 1

    def test_no_change(self) -> None:
        tok = FakeTokenizer({" lay": [7], " ly": [8]})
        assert split_increment(tok, "lay", "ly") == 0

    def test_leading_space_is_prepended(self) -> None:
        # 単語は文中出現を模して先頭スペース付きでトークナイズする
        tok = FakeTokenizer({" Janet": [1], " Janeet": [2, 3, 4]})
        assert split_increment(tok, "Janet", "Janeet") == 2


class TestZipfFrequency:
    def test_common_word_is_more_frequent(self) -> None:
        assert zipf_frequency("the") > zipf_frequency("xylophone")

    def test_nonword_is_zero(self) -> None:
        assert zipf_frequency("qzqzqzqz") == 0.0

    def test_case_insensitive(self) -> None:
        assert zipf_frequency("The") == zipf_frequency("the")


class TestFirstTokenId:
    def test_returns_first_piece(self) -> None:
        tok = FakeTokenizer({" Janet": [11, 22]})
        assert first_token_id(tok, "Janet") == 11

    def test_without_leading_space(self) -> None:
        tok = FakeTokenizer({"Janet": [33]})
        assert first_token_id(tok, "Janet", with_leading_space=False) == 33
