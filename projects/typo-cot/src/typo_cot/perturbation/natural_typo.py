"""実験10④: GitHub Typo Corpus 由来の自然typo分布サンプラー.

「LXT-4 の合成 typo(ランダム編集)は人工的」という批判への予防実験用モジュール。

- `extract_single_edit`: 修正前後の文ペアから単一文字編集操作を抽出
  (コーパス→経験分布の推定に使用。intended=修正後, typo=修正前)
- `NaturalTypoDistribution`: 編集操作の経験分布
  (操作比率・語内位置傾向・置換/挿入の文字条件付き分布) の入れ物 + JSON入出力
- `NaturalTypoGenerator`: 経験分布に従って 1 語に 1 typo を適用する生成器
  (既存 `generator.CharacterPerturbationGenerator` と同じ「1語1編集」の粒度)
- `apply_natural_typos_to_targets`: 標的語(token_index)を固定したまま
  自然分布 typo を適用する A/B 設計の B 側適用関数

再現性: 既存パイプラインと同じく `hash((seed, sample_id, token_str))` を
トークン単位シードに使うため、実行時に PYTHONHASHSEED=42 が必要。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from typo_cot.perturbation.dataset import PerturbedToken

OPERATIONS = ("substitution", "insertion", "deletion", "transposition")
BUCKETS = ("first", "internal", "last")


def _is_ascii_alpha(char: str) -> bool:
    """ASCII アルファベットかどうか (既存 generator と同じ判定)."""
    return len(char) == 1 and char.isalpha() and char.isascii()


def position_bucket(word: str, index: int) -> str:
    """語内位置バケット (first / internal / last) を返す.

    Args:
        word: 対象の語
        index: 語内の文字位置 (0-origin)

    Returns:
        "first" (先頭) / "last" (末尾) / "internal" (それ以外)。
        1文字語は "first"。
    """
    if index <= 0:
        return "first"
    if index >= len(word) - 1:
        return "last"
    return "internal"


@dataclass
class SingleEdit:
    """コーパスから抽出した単一文字編集操作.

    Attributes:
        operation: 操作種別 (substitution / insertion / deletion / transposition)
        word: 編集を受けた語 (intended テキスト側のアルファベット連続列)
        bucket: 語内位置バケット
        intended_char: 意図した文字 (置換・削除)
        typed_char: 実際に打鍵された文字 (置換・挿入)
        prev_char: 挿入位置の直前の文字 (挿入のみ)
    """

    operation: str
    word: str
    bucket: str
    intended_char: str | None = None
    typed_char: str | None = None
    prev_char: str | None = None


def _alpha_run(text: str, index: int) -> tuple[str, int]:
    """text[index] を含むアルファベット連続列 (語) と語内位置を返す.

    index の位置がアルファベットでない場合は直前の文字を基準にする。
    """
    if index >= len(text):
        index = len(text) - 1
    if index < 0:
        return "", 0
    if not _is_ascii_alpha(text[index]) and index > 0:
        index -= 1
    if not _is_ascii_alpha(text[index]):
        return "", 0
    start = index
    while start > 0 and _is_ascii_alpha(text[start - 1]):
        start -= 1
    end = index
    while end < len(text) - 1 and _is_ascii_alpha(text[end + 1]):
        end += 1
    return text[start : end + 1], index - start


def extract_single_edit(intended: str, typo: str) -> SingleEdit | None:
    """修正後(intended)→修正前(typo)の差分から単一文字編集操作を抽出.

    以下のいずれか 1 操作のみで説明できるペアだけを受理する
    (編集箇所が複数、または非アルファベット文字の編集は None):

    - substitution: 1文字置換 (両文字ともアルファベット)
    - insertion: 1文字挿入 (挿入文字がアルファベット)
    - deletion: 1文字削除 (削除文字がアルファベット)
    - transposition: 隣接2文字の入れ替え

    Args:
        intended: 意図したテキスト (コーパスの tgt = 修正後)
        typo: 実際のテキスト (コーパスの src = 修正前)

    Returns:
        SingleEdit または None
    """
    if intended == typo:
        return None

    matcher = SequenceMatcher(a=intended, b=typo, autojunk=False)
    diffs = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if len(diffs) == 1:
        tag, i1, i2, j1, j2 = diffs[0]
        if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
            a_char, b_char = intended[i1], typo[j1]
            if _is_ascii_alpha(a_char) and _is_ascii_alpha(b_char):
                word, idx = _alpha_run(intended, i1)
                return SingleEdit(
                    operation="substitution",
                    word=word,
                    bucket=position_bucket(word, idx),
                    intended_char=a_char.lower(),
                    typed_char=b_char.lower(),
                )
        elif tag == "replace" and i2 - i1 == 2 and j2 - j1 == 2:
            a_pair, b_pair = intended[i1:i2], typo[j1:j2]
            if (
                a_pair == b_pair[::-1]
                and a_pair[0] != a_pair[1]
                and all(_is_ascii_alpha(c) for c in a_pair)
            ):
                word, idx = _alpha_run(intended, i1)
                return SingleEdit(
                    operation="transposition",
                    word=word,
                    bucket=position_bucket(word, idx),
                    intended_char=a_pair[0].lower(),
                    typed_char=b_pair[0].lower(),
                )
        elif tag == "insert" and j2 - j1 == 1 and i1 == i2:
            ins = typo[j1]
            if _is_ascii_alpha(ins):
                # 同一文字の連続への挿入は位置が曖昧なので右側に正規化する
                # (例: "apple"→"applle" は「l の後に l を挿入」= 重複打鍵と数える)
                while i1 < len(intended) and intended[i1] == ins:
                    i1 += 1
                prev = intended[i1 - 1] if i1 > 0 else None
                word, idx = _alpha_run(intended, max(i1 - 1, 0))
                return SingleEdit(
                    operation="insertion",
                    word=word,
                    bucket=position_bucket(word, idx),
                    typed_char=ins.lower(),
                    prev_char=prev.lower() if prev and _is_ascii_alpha(prev) else None,
                )
        elif tag == "delete" and i2 - i1 == 1 and j1 == j2:
            deleted = intended[i1]
            if _is_ascii_alpha(deleted):
                word, idx = _alpha_run(intended, i1)
                return SingleEdit(
                    operation="deletion",
                    word=word,
                    bucket=position_bucket(word, idx),
                    intended_char=deleted.lower(),
                )
        return None

    # SequenceMatcher が転置を 2 個の差分に割ることがあるため、
    # 同長・2文字違い・隣接スワップのパターンを追加判定する
    if len(intended) == len(typo) and len(diffs) == 2:
        positions = [k for k in range(len(intended)) if intended[k] != typo[k]]
        if (
            len(positions) == 2
            and positions[1] == positions[0] + 1
            and intended[positions[0]] == typo[positions[1]]
            and intended[positions[1]] == typo[positions[0]]
            and all(_is_ascii_alpha(intended[k]) for k in positions)
        ):
            word, idx = _alpha_run(intended, positions[0])
            return SingleEdit(
                operation="transposition",
                word=word,
                bucket=position_bucket(word, idx),
                intended_char=intended[positions[0]].lower(),
                typed_char=typo[positions[0]].lower(),
            )
    return None


@dataclass
class NaturalTypoDistribution:
    """自然 typo の経験分布.

    Attributes:
        op_probs: 編集操作の比率 {substitution, insertion, deletion, transposition}
        position_probs: 語内位置傾向 {first, internal, last}
        substitution_given_intended: P(打鍵文字 | 意図文字) 小文字
        substitution_marginal: P(打鍵文字) 小文字 (条件付きが無い文字のフォールバック)
        insertion_given_prev: P(挿入文字 | 直前文字) 小文字
        insertion_marginal: P(挿入文字) 小文字 (フォールバック)
        metadata: 推定条件などの記録
    """

    op_probs: dict[str, float]
    position_probs: dict[str, float]
    substitution_given_intended: dict[str, dict[str, float]]
    substitution_marginal: dict[str, float]
    insertion_given_prev: dict[str, dict[str, float]]
    insertion_marginal: dict[str, float]
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """分布を JSON に保存."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": self.metadata,
            "op_probs": self.op_probs,
            "position_probs": self.position_probs,
            "substitution_given_intended": self.substitution_given_intended,
            "substitution_marginal": self.substitution_marginal,
            "insertion_given_prev": self.insertion_given_prev,
            "insertion_marginal": self.insertion_marginal,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NaturalTypoDistribution":
        """JSON から分布を読み込み."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            op_probs=data["op_probs"],
            position_probs=data["position_probs"],
            substitution_given_intended=data["substitution_given_intended"],
            substitution_marginal=data["substitution_marginal"],
            insertion_given_prev=data["insertion_given_prev"],
            insertion_marginal=data["insertion_marginal"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class NaturalTypoResult:
    """自然 typo の適用結果.

    Attributes:
        original: 元のテキスト
        perturbed: 摂動後のテキスト
        operation: 適用した編集操作
        position: 編集位置 (transposition は左側の文字位置)
        original_char: 元の文字 (置換・削除)
        new_char: 新しい文字 (置換・挿入)
    """

    original: str
    perturbed: str
    operation: str
    position: int
    original_char: str | None = None
    new_char: str | None = None


class NaturalTypoGenerator:
    """経験分布に従って 1 語に 1 typo を適用する生成器.

    既存 `CharacterPerturbationGenerator.perturb` と同じ粒度
    (1 トークン 1 編集、アルファベット位置のみ対象)。
    """

    _LOWER = "abcdefghijklmnopqrstuvwxyz"

    def __init__(
        self, distribution: NaturalTypoDistribution, seed: int | None = None
    ) -> None:
        """初期化.

        Args:
            distribution: 自然 typo の経験分布
            seed: ランダムシード (再現性のため)
        """
        self.dist = distribution
        self.rng = random.Random(seed)

    # -- 内部ヘルパー -------------------------------------------------------

    def _weighted_choice(self, probs: dict[str, float]) -> str | None:
        """重み付きサンプリング (キーの辞書順で決定的に走査)."""
        items = [(k, w) for k, w in sorted(probs.items()) if w > 0]
        total = sum(w for _, w in items)
        if total <= 0:
            return None
        r = self.rng.random() * total
        cum = 0.0
        for key, w in items:
            cum += w
            if r <= cum:
                return key
        return items[-1][0]

    def _feasible_ops(self, text: str) -> list[str]:
        ops = ["substitution", "insertion"]
        if len(text) > 1:
            ops.append("deletion")
        if self._transposable_positions(text):
            ops.append("transposition")
        return ops

    @staticmethod
    def _transposable_positions(text: str) -> list[int]:
        """隣接2文字が共にアルファベットかつ異なる位置 (左側) のリスト."""
        return [
            i
            for i in range(len(text) - 1)
            if _is_ascii_alpha(text[i])
            and _is_ascii_alpha(text[i + 1])
            and text[i].lower() != text[i + 1].lower()
        ]

    def _sample_position(self, text: str, candidates: list[int]) -> int:
        """位置傾向バケットに従って編集位置を選ぶ.

        バケットを (実現可能なものの中から) 重みでサンプルし、
        バケット内は一様に選ぶ。
        """
        by_bucket: dict[str, list[int]] = {}
        for pos in candidates:
            bucket = self._bucket_of(text, pos)
            by_bucket.setdefault(bucket, []).append(pos)
        probs = {b: self.dist.position_probs.get(b, 0.0) for b in by_bucket}
        if all(w <= 0 for w in probs.values()):
            probs = {b: 1.0 for b in by_bucket}
        bucket = self._weighted_choice(probs)
        if bucket is None:
            bucket = sorted(by_bucket)[0]
        return self.rng.choice(by_bucket[bucket])

    @staticmethod
    def _bucket_of(text: str, index: int) -> str:
        """トークン内のアルファベット連続列 (語) を基準に位置バケットを返す."""
        word, idx = _alpha_run(text, index)
        if not word:
            return "internal"
        return position_bucket(word, idx)

    @staticmethod
    def _match_case(char: str, reference: str) -> str:
        return char.upper() if reference.isupper() else char

    def _sample_substitution_char(self, original: str) -> str | None:
        lower = original.lower()
        cond = dict(self.dist.substitution_given_intended.get(lower, {}))
        cond.pop(lower, None)
        choice = self._weighted_choice(cond) if cond else None
        if choice is None:
            marginal = dict(self.dist.substitution_marginal)
            marginal.pop(lower, None)
            choice = self._weighted_choice(marginal)
        if choice is None:
            others = [c for c in self._LOWER if c != lower]
            choice = self.rng.choice(others)
        return choice

    def _sample_insertion_char(self, prev: str) -> str | None:
        lower = prev.lower()
        cond = self.dist.insertion_given_prev.get(lower, {})
        choice = self._weighted_choice(cond) if cond else None
        if choice is None:
            choice = self._weighted_choice(self.dist.insertion_marginal)
        if choice is None:
            choice = self.rng.choice(self._LOWER)
        return choice

    # -- 公開 API -----------------------------------------------------------

    def perturb(self, text: str) -> NaturalTypoResult | None:
        """経験分布に従って 1 編集を適用.

        Args:
            text: 入力トークン

        Returns:
            NaturalTypoResult、または適用不可 (アルファベットなし等) の場合 None
        """
        if not text:
            return None
        alpha_positions = [i for i, c in enumerate(text) if _is_ascii_alpha(c)]
        if not alpha_positions:
            return None

        feasible = self._feasible_ops(text)
        op_probs = {op: self.dist.op_probs.get(op, 0.0) for op in feasible}
        if all(w <= 0 for w in op_probs.values()):
            op_probs = {op: 1.0 for op in feasible}
        operation = self._weighted_choice(op_probs)
        if operation is None:
            return None

        if operation == "transposition":
            candidates = self._transposable_positions(text)
            pos = self._sample_position(text, candidates)
            perturbed = text[:pos] + text[pos + 1] + text[pos] + text[pos + 2 :]
            return NaturalTypoResult(
                original=text,
                perturbed=perturbed,
                operation=operation,
                position=pos,
                original_char=text[pos],
                new_char=text[pos + 1],
            )

        pos = self._sample_position(text, alpha_positions)
        original_char = text[pos]

        if operation == "substitution":
            new_lower = self._sample_substitution_char(original_char)
            if new_lower is None:
                return None
            new_char = self._match_case(new_lower, original_char)
            perturbed = text[:pos] + new_char + text[pos + 1 :]
            return NaturalTypoResult(
                original=text,
                perturbed=perturbed,
                operation=operation,
                position=pos,
                original_char=original_char,
                new_char=new_char,
            )

        if operation == "insertion":
            new_lower = self._sample_insertion_char(original_char)
            if new_lower is None:
                return None
            # 直前文字と同じ文字 (重複打鍵) の場合はケースを揃える
            if new_lower == original_char.lower():
                new_char = self._match_case(new_lower, original_char)
            else:
                new_char = new_lower
            perturbed = text[: pos + 1] + new_char + text[pos + 1 :]
            return NaturalTypoResult(
                original=text,
                perturbed=perturbed,
                operation=operation,
                position=pos,
                original_char=None,
                new_char=new_char,
            )

        # deletion
        perturbed = text[:pos] + text[pos + 1 :]
        return NaturalTypoResult(
            original=text,
            perturbed=perturbed,
            operation=operation,
            position=pos,
            original_char=original_char,
            new_char=None,
        )


def apply_natural_typos_to_targets(
    text: str,
    targets: list[dict],
    offset_mapping: dict[int, tuple[int, int]],
    text_char_start: int,
    distribution: NaturalTypoDistribution,
    seed: int,
    sample_id: str,
) -> tuple[str, list[PerturbedToken], list[str]]:
    """標的語を固定したまま自然分布 typo を適用する (A/B 設計の B 側).

    A 側 (LXT-4 合成分布) と同一の標的トークン (token_index) に対し、
    編集操作の分布だけを自然分布に差し替えて適用する。
    文字位置のずれを避けるため右 (後方) の標的から順に適用する。

    Args:
        text: 摂動対象テキスト (質問文、または質問文+選択肢)
        targets: 標的のリスト。各要素は
            {"token_index": int, "original_token": str, "importance_score": float}
        offset_mapping: token_index -> (char_start, char_end) プロンプト全体基準
        text_char_start: text のプロンプト全体上での開始位置
        distribution: 自然 typo の経験分布
        seed: 実験シード (トークン単位シードは hash((seed, sample_id, token))
              で導出 = 既存パイプラインと同じ規約。PYTHONHASHSEED=42 必須)
        sample_id: サンプル ID (再現性のため)

    Returns:
        (摂動後テキスト, PerturbedToken リスト (token_index 昇順), 警告リスト)
    """
    warnings: list[str] = []
    resolved = []
    for target in targets:
        token_index = target["token_index"]
        span = offset_mapping.get(token_index)
        if span is None:
            warnings.append(f"{sample_id}: token_index={token_index} の offset がありません")
            continue
        rel_start = span[0] - text_char_start
        rel_end = span[1] - text_char_start
        if rel_start < 0 or rel_end > len(text) or rel_start >= rel_end:
            warnings.append(
                f"{sample_id}: token_index={token_index} の範囲が不正 "
                f"({rel_start}, {rel_end})"
            )
            continue
        span_text = text[rel_start:rel_end]
        expected = str(target.get("original_token", "")).strip()
        if expected and span_text != expected:
            warnings.append(
                f"{sample_id}: token_index={token_index} の文字列不一致 "
                f"(span={span_text!r}, expected={expected!r})"
            )
        resolved.append((rel_start, rel_end, target, span_text))

    perturbed_text = text
    entries: list[PerturbedToken] = []
    # 右から適用すれば左側の位置はずれない
    for rel_start, rel_end, target, span_text in sorted(
        resolved, key=lambda item: item[0], reverse=True
    ):
        token_str = target.get("original_token", span_text)
        token_seed = hash((seed, sample_id, token_str))
        generator = NaturalTypoGenerator(distribution, seed=token_seed)
        result = generator.perturb(span_text)
        if result is None:
            warnings.append(
                f"{sample_id}: token_index={target['token_index']} に摂動を適用できません "
                f"(span={span_text!r})"
            )
            continue
        perturbed_text = (
            perturbed_text[:rel_start] + result.perturbed + perturbed_text[rel_end:]
        )
        entries.append(
            PerturbedToken(
                token_index=target["token_index"],
                original_token=token_str,
                perturbed_token=result.perturbed,
                importance_score=float(target.get("importance_score", 0.0)),
                perturbation_type=result.operation,
                char_position=result.position,
            )
        )

    entries.sort(key=lambda e: e.token_index)
    return perturbed_text, entries, warnings
