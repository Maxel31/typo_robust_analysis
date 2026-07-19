"""実験2: 同品詞・同頻度帯の置換語サンプラ (操作(c) replace 用).

計画 §4 実験2-2 操作(c)「同品詞・同頻度帯の別語への置換」。
- 語彙は wordfreq の上位語 (既定 top_n=20000) から構築
- POS タガーは注入可能。既定は軽量ヒューリスティック (接尾辞規則)。
  spaCy (en_core_web_sm, appendix extra) を使う場合はタガーを差し替える —
  本番採用タガーは open question としてユーザー判断待ち
- 頻度帯は Zipf 半幅 <0.5 → <1.0 → <2.0 → ∞ と緩和
"""

from collections.abc import Callable
from dataclasses import dataclass

from wordfreq import top_n_list, zipf_frequency

ZIPF_SCHEDULE: list[float | None] = [0.5, 1.0, 2.0, None]


@dataclass(frozen=True)
class VocabEntry:
    """置換語彙のエントリ."""

    word: str
    pos: str
    zipf: float


def heuristic_pos(word: str) -> str:
    """接尾辞ベースの軽量 POS 推定 (NOUN / VERB / ADJ / ADV).

    ユニットテスト・スモーク用の既定タガー。本番の採用タガー (spaCy 等) は
    open question。曖昧な場合は NOUN に落とす。
    """
    w = word.lower()
    if w.endswith("ly"):
        return "ADV"
    if w.endswith(("ing", "ed", "ize", "ise", "ify")):
        return "VERB"
    if w.endswith(("ous", "ful", "ive", "able", "ible", "ish", "al", "ic")):
        return "ADJ"
    return "NOUN"


class ReplacementSampler:
    """同品詞・同頻度帯の別語を抽選するサンプラ.

    Args:
        vocab: 置換語彙 (None なら wordfreq top_n から構築)
        tagger: 語 → POS の callable (None ならヒューリスティック)
        top_n: 既定語彙の規模
    """

    def __init__(
        self,
        vocab: list[VocabEntry] | None = None,
        tagger: Callable[[str], str] | None = None,
        top_n: int = 20000,
    ) -> None:
        self.tagger = tagger or heuristic_pos
        self.vocab = vocab if vocab is not None else self._build_default_vocab(top_n)

    def _build_default_vocab(self, top_n: int) -> list[VocabEntry]:
        out: list[VocabEntry] = []
        for w in top_n_list("en", top_n):
            if len(w) < 2 or not w.isalpha():
                continue
            out.append(VocabEntry(word=w, pos=self.tagger(w), zipf=zipf_frequency(w, "en")))
        return out

    def sample(self, word: str, rng) -> str | None:
        """word と同品詞・同頻度帯の別語を返す (候補なしなら None).

        大小文字は元語のパターンに合わせる (先頭大文字 / 全大文字)。
        """
        pos = self.tagger(word)
        wz = zipf_frequency(word.lower(), "en")
        lower = word.lower()
        for zipf_tol in ZIPF_SCHEDULE:
            band = [
                v
                for v in self.vocab
                if v.pos == pos
                and v.word.lower() != lower
                and (zipf_tol is None or abs(v.zipf - wz) < zipf_tol)
            ]
            if band:
                chosen = rng.choice(sorted(band, key=lambda v: v.word))
                return self._match_case(word, chosen.word)
        return None

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        if len(original) > 1 and original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement
