"""実験5(双子語統制): 層化マッチドサンプラ.

LXT-4 の各標的語に対し、同一質問内から次の 5 特徴をマッチさせた「双子語」を選ぶ:

1. 内容語/機能語の別 (classify_fn; spaCy POS または機能語リスト)
2. 文字長 ±1
3. Zipf 頻度ビン (0.5 刻み; zipf_fn = wordfreq.zipf_frequency)
4. 同一摂動タイプを仮適用したときのサブワード分割数の増分
5. (第2優先) 質問文埋め込みとの cos 類似ビン (embed_fn = sentence-transformers)

マッチングは完全一致優先 → 失敗時に caliper 緩和 (頻度 ±0.5 / 長さ ±1 拡張) し、
緩和レベルを MatchRecord に記録する。マッチバランスは compute_smd_table() で
標準化平均差 (SMD) 表として自動生成する。

重い依存 (wordfreq / spaCy / sentence-transformers / HF tokenizer) はすべて
注入可能にしてあり、ユニットテストは GPU・外部モデル無しで完結する。
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from typo_cot.perturbation.generator import (
    CharacterPerturbationGenerator,
    PerturbationResult,
)

logger = logging.getLogger(__name__)

ZIPF_BIN_WIDTH = 0.5
CENTRALITY_BIN_WIDTH = 0.1

# 緩和ラダー (level = リスト内の位置)
RELAXATION_LABELS = (
    "exact",  # 5特徴すべて一致 (文字長は ±1)
    "no_centrality",  # 中心性ビン (第2優先) を落とす
    "caliper",  # 頻度 ±0.5 (隣接ビン) / 長さ ±2 / 分割増分 ±1
    "class_len",  # クラス一致 + 文字長差最小 (rebuttal 相当)
    "any",  # 文字長差最小のみ
    "unmatched",  # プール枯渇
)

SUBWORD_MARKERS = ("▁", "Ġ")

# 英語機能語リスト (scripts/rebuttal/make_matched_random_dataset.py と同一)。
# spaCy 非依存のフォールバック分類器として使う。
FUNCTION_WORDS = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "each", "every", "either",
    "neither", "some", "any", "no", "all", "both", "few", "many", "much", "more",
    "most", "other", "another", "such", "what", "which", "whose",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves",
    "themselves", "who", "whom", "someone", "anyone", "everyone", "nothing",
    "something", "anything", "everything", "one", "none",
    # prepositions
    "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "of", "off", "over", "under", "again", "further", "than", "as", "per",
    "via", "within", "without", "upon", "among", "across", "behind", "beyond",
    "near", "onto", "toward", "towards",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "if", "because", "although", "though",
    "while", "when", "whenever", "where", "wherever", "since", "unless", "until",
    "whether", "once",
    # auxiliaries / copula
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "ought", "need", "dare",
    # negation / common adverbs of degree
    "not", "n't", "only", "just", "also", "too", "very", "then", "there", "here",
    "how", "why", "out",
}

# spaCy POS で機能語とみなす品詞 (Universal POS)
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ", "PUNCT", "SYM", "NUM"}


def strip_token(token: str) -> str:
    """サブワードマーカー・空白を除去した表層形を返す."""
    text = token.strip()
    for marker in SUBWORD_MARKERS:
        text = text.replace(marker, "")
    return text


def normalize_surface(token: str) -> str:
    """分類・頻度参照用に小文字化し前後の記号を落とす."""
    return strip_token(token).lower().strip("'\"()[].,:;!?")


def function_word_class(token: str) -> str:
    """機能語リストによる内容語/機能語の2値分類 (spaCy 非依存フォールバック)."""
    return "function" if normalize_surface(token) in FUNCTION_WORDS else "content"


def make_spacy_classifier(nlp) -> Callable[[str], str]:
    """spaCy (en_core_web_sm) による内容語/機能語分類器を返す.

    単語単体を POS タグ付けし、FUNCTION_POS なら "function"。
    解析不能・空文字は機能語リストにフォールバックする。
    """
    cache: dict[str, str] = {}

    def classify(token: str) -> str:
        surface = normalize_surface(token)
        if not surface:
            return "function"
        if surface in cache:
            return cache[surface]
        doc = nlp(surface)
        if len(doc) == 0:
            result = function_word_class(token)
        else:
            result = "function" if doc[0].pos_ in FUNCTION_POS else "content"
        cache[surface] = result
        return result

    return classify


def zipf_bin(zipf: float) -> float:
    """Zipf 頻度を 0.5 刻みのビン下端に丸める."""
    return math.floor(zipf / ZIPF_BIN_WIDTH) * ZIPF_BIN_WIDTH


def centrality_bin(cos_sim: float) -> int:
    """cos 類似度を 0.1 幅のビン番号に変換する."""
    return math.floor(cos_sim / CENTRALITY_BIN_WIDTH)


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


@dataclass
class TokenFeatures:
    """マッチング用のトークン特徴量."""

    token_index: int
    token: str
    surface: str
    word_class: str  # "content" | "function"
    char_len: int
    zipf: float
    zipf_bin: float
    split_increment: int
    perturbation_type: str | None
    centrality: float | None
    centrality_bin: int | None

    def to_dict(self) -> dict:
        return {
            "token_index": self.token_index,
            "token": self.token,
            "surface": self.surface,
            "word_class": self.word_class,
            "char_len": self.char_len,
            "zipf": self.zipf,
            "zipf_bin": self.zipf_bin,
            "split_increment": self.split_increment,
            "perturbation_type": self.perturbation_type,
            "centrality": self.centrality,
            "centrality_bin": self.centrality_bin,
        }


@dataclass
class MatchRecord:
    """1標的語のマッチ結果 (緩和レベル込み)."""

    sample_id: str
    target: TokenFeatures
    matched: TokenFeatures | None
    relaxation_level: int
    relaxation_label: str

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "target": self.target.to_dict(),
            "matched": self.matched.to_dict() if self.matched else None,
            "relaxation_level": self.relaxation_level,
            "relaxation_label": self.relaxation_label,
        }


class FeatureExtractor:
    """マッチング特徴量の計算器 (依存はすべて注入可能).

    Args:
        tokenizer: `tokenize(text) -> list[str]` を持つトークナイザ
            (HF AutoTokenizer 互換)。None なら分割増分は常に 0。
        zipf_fn: 表層形 (小文字) -> Zipf 頻度。None なら常に 0.0。
        classify_fn: トークン -> "content"/"function"。None なら機能語リスト。
        embed_fn: テキスト -> ベクトル。None なら中心性特徴を無効化。
        seed: dataset.py と同一の token_seed 式 hash((seed, sample_id, token))
            に使うシード。
    """

    def __init__(
        self,
        tokenizer=None,
        zipf_fn: Callable[[str], float] | None = None,
        classify_fn: Callable[[str], str] | None = None,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.zipf_fn = zipf_fn
        self.classify_fn = classify_fn or function_word_class
        self.embed_fn = embed_fn
        self.seed = seed

    # --- 摂動の仮適用 -------------------------------------------------

    def perturbation_for(self, sample_id: str, token: str) -> PerturbationResult | None:
        """dataset.py の適用ループと同一の token_seed 式で摂動を仮適用する."""
        token_seed = hash((self.seed, sample_id, token))
        return CharacterPerturbationGenerator(seed=token_seed).perturb(token)

    def _count_pieces(self, surface: str) -> int:
        if self.tokenizer is None or not surface:
            return 1
        # 文中の出現を模すため先頭に空白を付けてトークナイズする
        return max(1, len(self.tokenizer.tokenize(" " + surface)))

    def split_increment(self, sample_id: str, token: str, ptype: str | None) -> int:
        """指定タイプの摂動を仮適用したときのサブワード分割数の増分.

        指定タイプが適用不能な場合はタイプ自由の仮適用にフォールバックし、
        それも不能なら 0 を返す。
        """
        if self.tokenizer is None:
            return 0
        surface = strip_token(token)
        if not surface:
            return 0

        token_seed = hash((self.seed, sample_id, token))
        generator = CharacterPerturbationGenerator(seed=token_seed)
        result: PerturbationResult | None = None
        if ptype == "proximity":
            result = generator.proximity_replace(surface)
        elif ptype == "double_typing":
            result = generator.double_typing(surface)
        elif ptype == "omission":
            result = generator.omission(surface)
        if result is None:
            result = generator.perturb(surface)
        if result is None:
            return 0

        return self._count_pieces(result.perturbed) - self._count_pieces(surface)

    # --- 埋め込み中心性 -----------------------------------------------

    def question_embedding(self, question_text: str) -> Sequence[float] | None:
        """質問文の埋め込み (embed_fn 無効時は None)."""
        if self.embed_fn is None or not question_text:
            return None
        return self.embed_fn(question_text)

    # --- 特徴量 ---------------------------------------------------------

    def features(
        self,
        sample_id: str,
        token_tuple: tuple[int, str, float],
        question_vec: Sequence[float] | None = None,
        ptype: str | None = None,
    ) -> TokenFeatures:
        """1トークンの特徴量を計算する.

        Args:
            sample_id: サンプルID (token_seed の一部)
            token_tuple: (token_index, token, importance_score)
            question_vec: 質問文埋め込み (centrality 計算用; None で無効)
            ptype: 分割増分の仮適用に使う摂動タイプ。None ならこのトークン
                自身の仮適用タイプ (dataset.py と同一の抽選) を使う。
        """
        token_index, token, _score = token_tuple
        surface = strip_token(token)

        if ptype is None:
            tentative = self.perturbation_for(sample_id, token)
            ptype_used = tentative.perturbation_type.value if tentative else None
        else:
            ptype_used = ptype

        zipf = float(self.zipf_fn(normalize_surface(token))) if self.zipf_fn else 0.0

        centrality: float | None = None
        c_bin: int | None = None
        if question_vec is not None and self.embed_fn is not None and surface:
            token_vec = self.embed_fn(surface)
            centrality = _cosine(token_vec, question_vec)
            c_bin = centrality_bin(centrality)

        return TokenFeatures(
            token_index=token_index,
            token=token,
            surface=surface,
            word_class=self.classify_fn(token),
            char_len=len(surface),
            zipf=zipf,
            zipf_bin=zipf_bin(zipf),
            split_increment=self.split_increment(sample_id, token, ptype_used),
            perturbation_type=ptype_used,
            centrality=centrality,
            centrality_bin=c_bin,
        )


def _level_predicate(level: int, target: TokenFeatures, cand: TokenFeatures) -> bool:
    """緩和レベル level で候補 cand が標的 target にマッチするか."""
    d_len = abs(cand.char_len - target.char_len)
    d_zipf_bin = abs(cand.zipf_bin - target.zipf_bin)
    d_inc = abs(cand.split_increment - target.split_increment)
    same_class = cand.word_class == target.word_class
    centrality_ok = (
        target.centrality_bin is None
        or cand.centrality_bin is None
        or cand.centrality_bin == target.centrality_bin
    )

    if level == 0:  # exact
        return same_class and d_len <= 1 and d_zipf_bin == 0.0 and d_inc == 0 and centrality_ok
    if level == 1:  # no_centrality
        return same_class and d_len <= 1 and d_zipf_bin == 0.0 and d_inc == 0
    if level == 2:  # caliper: 頻度 ±0.5 / 長さ ±2 / 増分 ±1
        return same_class and d_len <= 2 and d_zipf_bin <= ZIPF_BIN_WIDTH and d_inc <= 1
    if level == 3:  # class_len (文字長差最小は _distance で選ぶ)
        return same_class
    # level 4: any
    return True


def _distance(target: TokenFeatures, cand: TokenFeatures) -> float:
    """レベル内での選好距離 (小さいほど良い)."""
    d = abs(cand.char_len - target.char_len)
    d += abs(cand.zipf_bin - target.zipf_bin)
    d += abs(cand.split_increment - target.split_increment)
    if target.centrality_bin is not None and cand.centrality_bin is not None:
        d += abs(cand.centrality_bin - target.centrality_bin) * CENTRALITY_BIN_WIDTH
    return d


class MatchedTwinSampler:
    """LXT-4 標的語に対する双子語の層化マッチング.

    選択部は rebuttal 版 (scripts/rebuttal/make_matched_random_dataset.py) の
    random モードミラーと同一の骨格: 重要度降順の top-k を標的とし、それ以外を
    プールとして非復元でマッチする。乱数は hash((seed, sample_id,
    "matched_selection")) で初期化し、タイブレークにのみ使う。
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        num_perturbations: int = 4,
        seed: int = 42,
    ) -> None:
        self.extractor = extractor
        self.num_perturbations = num_perturbations
        self.seed = seed

    def select(
        self,
        sample_id: str,
        question_tokens: list[tuple[int, str, float]],
        question_text: str | None = None,
    ) -> tuple[list[tuple[int, str, float]], list[MatchRecord]]:
        """双子語を選び、摂動候補リストとマッチ記録を返す.

        Returns:
            (候補トークンリスト, マッチ記録): 候補リストは
            [マッチした双子語 (標的順), シャッフルしたバックアップ] の順。
            バックアップは perturb() 失敗時の充填用で、top-k 標的は含まない。
        """
        if not question_tokens:
            return [], []

        sorted_by_importance = sorted(question_tokens, key=lambda x: x[2], reverse=True)

        if len(sorted_by_importance) > self.num_perturbations:
            top_k = sorted_by_importance[: self.num_perturbations]
            pool = list(sorted_by_importance[self.num_perturbations :])
        else:
            # rebuttal 版と同じフォールバック: 全トークンをプールにする
            top_k = sorted_by_importance
            pool = list(sorted_by_importance)

        rng = random.Random(hash((self.seed, sample_id, "matched_selection")))

        question_vec = (
            self.extractor.question_embedding(question_text) if question_text else None
        )

        pool_features = {
            t[0]: self.extractor.features(sample_id, t, question_vec=question_vec)
            for t in pool
        }
        pool_tuples = {t[0]: t for t in pool}

        matched_tuples: list[tuple[int, str, float]] = []
        records: list[MatchRecord] = []

        for target_tuple in top_k:
            target_feat = self.extractor.features(
                sample_id, target_tuple, question_vec=question_vec
            )

            if not pool_features:
                records.append(
                    MatchRecord(
                        sample_id=sample_id,
                        target=target_feat,
                        matched=None,
                        relaxation_level=len(RELAXATION_LABELS) - 1,
                        relaxation_label="unmatched",
                    )
                )
                continue

            # 候補側の分割増分は「標的と同じ摂動タイプの仮適用」で再計算する
            cand_features = {
                idx: self._with_target_type(sample_id, feat, target_feat.perturbation_type)
                for idx, feat in pool_features.items()
            }

            chosen_idx: int | None = None
            level_used = len(RELAXATION_LABELS) - 1
            for level in range(len(RELAXATION_LABELS) - 1):
                eligible = [
                    idx
                    for idx, cand in cand_features.items()
                    if _level_predicate(level, target_feat, cand)
                ]
                if eligible:
                    min_d = min(_distance(target_feat, cand_features[i]) for i in eligible)
                    best = [
                        i
                        for i in eligible
                        if _distance(target_feat, cand_features[i]) == min_d
                    ]
                    chosen_idx = rng.choice(best)
                    level_used = level
                    break

            if chosen_idx is None:
                records.append(
                    MatchRecord(
                        sample_id=sample_id,
                        target=target_feat,
                        matched=None,
                        relaxation_level=len(RELAXATION_LABELS) - 1,
                        relaxation_label="unmatched",
                    )
                )
                continue

            records.append(
                MatchRecord(
                    sample_id=sample_id,
                    target=target_feat,
                    matched=cand_features[chosen_idx],
                    relaxation_level=level_used,
                    relaxation_label=RELAXATION_LABELS[level_used],
                )
            )
            matched_tuples.append(pool_tuples[chosen_idx])
            # 非復元抽出
            pool_features.pop(chosen_idx)
            pool_tuples.pop(chosen_idx, None)

        # perturb() 失敗に備えたバックアップ (残プールをシャッフルして後置)
        backup = list(pool_tuples.values())
        rng.shuffle(backup)

        return matched_tuples + backup, records

    def _with_target_type(
        self, sample_id: str, feat: TokenFeatures, target_ptype: str | None
    ) -> TokenFeatures:
        """候補特徴量の分割増分を標的の摂動タイプで再計算したコピーを返す."""
        if target_ptype is None or feat.perturbation_type == target_ptype:
            return feat
        increment = self.extractor.split_increment(sample_id, feat.token, target_ptype)
        return TokenFeatures(
            token_index=feat.token_index,
            token=feat.token,
            surface=feat.surface,
            word_class=feat.word_class,
            char_len=feat.char_len,
            zipf=feat.zipf,
            zipf_bin=feat.zipf_bin,
            split_increment=increment,
            perturbation_type=target_ptype,
            centrality=feat.centrality,
            centrality_bin=feat.centrality_bin,
        )


def _smd(target_values: list[float], matched_values: list[float]) -> float:
    """標準化平均差 (SMD) = (mean_t - mean_m) / pooled_sd."""
    n_t, n_m = len(target_values), len(matched_values)
    if n_t == 0 or n_m == 0:
        return float("nan")
    mean_t = sum(target_values) / n_t
    mean_m = sum(matched_values) / n_m
    var_t = (
        sum((v - mean_t) ** 2 for v in target_values) / (n_t - 1) if n_t > 1 else 0.0
    )
    var_m = (
        sum((v - mean_m) ** 2 for v in matched_values) / (n_m - 1) if n_m > 1 else 0.0
    )
    pooled_sd = math.sqrt((var_t + var_m) / 2.0)
    if pooled_sd == 0.0:
        return 0.0 if mean_t == mean_m else math.inf
    return (mean_t - mean_m) / pooled_sd


def compute_smd_table(records: list[MatchRecord]) -> dict:
    """マッチバランス表 (SMD・クラス一致率・緩和率) を計算する.

    unmatched レコードは SMD の対象外 (n_targets には数える)。
    """
    paired = [r for r in records if r.matched is not None]

    smd: dict[str, float] = {}
    for name in ("char_len", "zipf", "split_increment"):
        smd[name] = _smd(
            [float(getattr(r.target, name)) for r in paired],
            [float(getattr(r.matched, name)) for r in paired],
        )
    centrality_pairs = [
        (r.target.centrality, r.matched.centrality)
        for r in paired
        if r.target.centrality is not None and r.matched.centrality is not None
    ]
    if centrality_pairs:
        smd["centrality"] = _smd(
            [p[0] for p in centrality_pairs], [p[1] for p in centrality_pairs]
        )

    n = len(paired)
    relaxation_counts: dict[str, int] = {label: 0 for label in RELAXATION_LABELS}
    for r in records:
        relaxation_counts[r.relaxation_label] += 1

    return {
        "n_targets": len(records),
        "n_matched": n,
        "smd": smd,
        "class_match_rate": (
            sum(r.matched.word_class == r.target.word_class for r in paired) / n
            if n
            else 0.0
        ),
        "exact_len_match_rate": (
            sum(r.matched.char_len == r.target.char_len for r in paired) / n if n else 0.0
        ),
        "mean_len_diff": (
            sum(abs(r.matched.char_len - r.target.char_len) for r in paired) / n
            if n
            else 0.0
        ),
        "relaxation_rates": {
            label: (count / len(records) if records else 0.0)
            for label, count in relaxation_counts.items()
        },
    }
