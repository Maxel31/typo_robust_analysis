"""実験1 (CoT移植 2×2) のセル入力構築.

各サンプルについて 4 条件を構成する:

    A = (clean 質問, clean CoT)   … 基準
    B = (typo 質問,  typo CoT)    … 総効果 TE (既存ログの再現でもある)
    C = (typo 質問,  clean CoT)   … 直接効果 DE
    D = (clean 質問, typo CoT)    … 間接効果 IE

CoT は生成済みテキストを答え句テンプレート ("The answer is" 等) の
最初の出現位置の直前で切断し、teacher-forcing で与える。
few-shot 文脈・プロンプト骨格は既存の generation パイプライン
(scripts/run_inference.py → models/prompts.py) と完全同一で、
プロンプトはプレーンテキスト連結 (chat template 不使用) である。
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.intervention.records import PairRecord
from typo_cot.models.prompts import create_prompt_template

# 答え句テンプレート (最初の出現位置で切断)。既存パイプラインの全ベンチマークは
# "The answer is ..." 形式で統一されている。DeepSeek-R1-Distill 系など別の
# 答え句を使うモデルは trigger_pattern で差し替える。
DEFAULT_TRIGGER_PATTERN = r"[Tt]he answer is"

# 切断後の prefix に残っていたら「答え句断片の残留」とみなすパターン
# (残留事例は主分析から除外し、含めた版を感度分析とする)
RESIDUAL_PATTERNS = [
    r"[Tt]he answer is",
    r"[Aa]nswer\s*[:=]",
    r"[Ff]inal [Aa]nswer",
]

# 「序盤に答え句が出現」とみなす相対位置のしきい値 (先頭 25% 以内)
EARLY_TRIGGER_RATIO = 0.25


@dataclass
class TruncationResult:
    """CoT 切断の結果.

    Attributes:
        prefix: 答え句直前までの CoT (トリガー未検出時は全文)
        trigger_found: 答え句テンプレートが見つかったか
        trigger_count: 答え句テンプレートの出現回数
        early_trigger: 最初の出現が CoT 先頭 25% 以内か (フラグ→除外)
        residual_fragment: 切断後 prefix に答え句の断片 (変種) が残るか
        trigger_char_start: 最初の出現の文字位置 (未検出時は None)
        trigger_answers: 各トリガー区間で抽出された答え (トリガー順)。
            Qwen 等の「同一答えを複数回述べる」癖の検出に使う。
        trigger_answers_identical: 全トリガーが同一の非空答えを指すか
            (dedup 判定用。全区間が非空かつ一意で True)
    """

    prefix: str
    trigger_found: bool
    trigger_count: int
    early_trigger: bool
    residual_fragment: bool
    trigger_char_start: int | None = None
    trigger_answers: list[str] = field(default_factory=list)
    trigger_answers_identical: bool = False


def truncate_before_answer(
    cot: str,
    benchmark: str,
    trigger_pattern: str | None = None,
) -> TruncationResult:
    """CoT を答え句テンプレートの最初の出現位置の直前で切断する.

    Args:
        cot: 生成済み CoT テキスト (答え句込み)
        benchmark: ベンチマーク名 (現状は全ベンチ共通トリガー。将来の分岐用)
        trigger_pattern: 答え句の正規表現 (None なら既定の "The answer is")

    Returns:
        TruncationResult
    """
    pattern = trigger_pattern if trigger_pattern is not None else DEFAULT_TRIGGER_PATTERN
    matches = list(re.finditer(pattern, cot))

    if not matches:
        return TruncationResult(
            prefix=cot,
            trigger_found=False,
            trigger_count=0,
            early_trigger=False,
            residual_fragment=False,
            trigger_char_start=None,
        )

    first = matches[0]
    prefix = cot[: first.start()]
    early = len(cot) > 0 and (first.start() / len(cot)) < EARLY_TRIGGER_RATIO

    residual = any(re.search(p, prefix) for p in RESIDUAL_PATTERNS)

    # 各トリガー区間 [trigger_i, trigger_{i+1}) の答えを抽出する。区間で切ることで
    # 「同一答えを繰り返すだけ」(良性の重複) と「途中で答えが変わる」(真の曖昧さ)
    # を区別できる。切断点 (prefix) は従来どおり最初のトリガー直前で不変。
    trigger_answers: list[str] = []
    if len(matches) > 1:
        extractor = create_extractor(benchmark)
        for idx, m in enumerate(matches):
            seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cot)
            seg = cot[m.start(): seg_end]
            trigger_answers.append(extractor.extract(seg).extracted_answer.strip())
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
        early_trigger=early,
        residual_fragment=residual,
        trigger_char_start=first.start(),
        trigger_answers=trigger_answers,
        trigger_answers_identical=identical,
    )


@dataclass
class CellInputs:
    """1サンプル分の 4 セル teacher-forcing 入力.

    Attributes:
        sample_id: サンプル ID
        prompts: セル名 → few-shot プロンプト (質問まで。CoT は含まない)
        forced_cots: セル名 → teacher-forcing する切断済み CoT prefix
        truncation: "clean"/"typo" → TruncationResult
        exclude: 主分析から除外すべきか
        exclude_reasons: 除外理由のリスト
    """

    sample_id: str
    prompts: dict[str, str]
    forced_cots: dict[str, str]
    truncation: dict[str, TruncationResult]
    exclude: bool
    exclude_reasons: list[str] = field(default_factory=list)

    def full_input(self, cell: str) -> str:
        """セルの完全な teacher-forcing 入力 (プロンプト + 切断済み CoT)."""
        return self.prompts[cell] + self.forced_cots[cell]


# セル名 → (質問側, CoT側) の対応。"clean"/"typo"
CELL_DEFINITIONS: dict[str, tuple[str, str]] = {
    "A": ("clean", "clean"),
    "B": ("typo", "typo"),
    "C": ("typo", "clean"),
    "D": ("clean", "typo"),
}


def _build_prompt(
    benchmark: str,
    question: str,
    choices: list[str] | None,
    subset: str | None,
) -> str:
    """既存パイプラインと同一のプロンプト骨格を構築する.

    scripts/run_inference.py の generate_prompt_for_sample と同じ分岐。
    """
    template = create_prompt_template(benchmark)
    if benchmark in ("mmlu", "mmlu_pro", "arc", "commonsense_qa"):
        prompt_result = template.generate(question=question, choices=choices, subject=subset)
    elif benchmark in ("bbh", "math", "strategy_qa"):
        prompt_result = template.generate(question=question, subject=subset)
    else:
        prompt_result = template.generate(question=question)
    return prompt_result.get_full_prompt()


def build_cell_inputs(
    pair: PairRecord,
    trigger_pattern: str | None = None,
    dedup_same_answer_triggers: bool = False,
    prompt_builder: Callable[[str, str, list[str] | None, str | None], str] | None = None,
    truncator: Callable[[str, str, str | None], TruncationResult] | None = None,
    strip_conclusion_mode: str | None = None,
) -> CellInputs:
    """4 セルの teacher-forcing 入力を構築する.

    Args:
        pair: clean × typo のサンプル対
        trigger_pattern: 答え句の正規表現 (モデル別に差し替え可)
        dedup_same_answer_triggers: True のとき、複数トリガーでも全区間が同一の
            答えを指す (良性の重複、例: Qwen の "The answer is X. The answer is X.")
            場合は multi_trigger 除外を課さない。既定 False は従来挙動を完全維持
            (既存5モデルの再解析で数値不変)。切断点は常に最初のトリガー直前で不変
            のため、生成の再利用 (再解析のみ) が可能。
        prompt_builder: プロンプト骨格構築関数 (benchmark, question, choices,
            subset) → プロンプト文字列。None なら基底の _build_prompt (生プロンプト
            few-shot)。R1蒸留系はチャットテンプレート builder を注入する
            (reasoning_cells.make_reasoning_prompt_builder)。
        truncator: CoT 切断関数 (cot, benchmark, trigger_pattern) → TruncationResult。
            None なら基底の truncate_before_answer。R1蒸留系は <think> 構造対応の
            reasoning_cells.truncate_reasoning_cot を注入する。
        strip_conclusion_mode: A2 (ii) 結論剥ぎ。None 以外 ("last_line"/"last_sentence")
            を指定すると **C セル (typo質問 + clean CoT) の強制 CoT の末尾** を
            leak_audit.strip_conclusion で除去する。GSM8K で末尾の読み上げ計算行に
            金答え数値が載る事例の「丸写し」経路を潰すための介入。A/B/D は不変。

    Returns:
        CellInputs
    """
    _truncate = truncator if truncator is not None else truncate_before_answer
    _prompt = prompt_builder if prompt_builder is not None else _build_prompt

    trunc = {
        "clean": _truncate(pair.cot_clean, pair.benchmark, trigger_pattern),
        "typo": _truncate(pair.cot_typo, pair.benchmark, trigger_pattern),
    }

    prompt_by_q = {
        "clean": _prompt(pair.benchmark, pair.question_clean, pair.choices_clean, pair.subset),
        "typo": _prompt(pair.benchmark, pair.question_typo, pair.choices_typo, pair.subset),
    }

    prompts: dict[str, str] = {}
    forced_cots: dict[str, str] = {}
    for cell, (q_side, cot_side) in CELL_DEFINITIONS.items():
        prompts[cell] = prompt_by_q[q_side]
        forced_cots[cell] = trunc[cot_side].prefix

    if strip_conclusion_mode:
        # C セル (DE 条件) の clean CoT の末尾のみ剥ぐ (A2 ii)。他セルは不変。
        from typo_cot.intervention.leak_audit import strip_conclusion

        forced_cots["C"] = strip_conclusion(forced_cots["C"], mode=strip_conclusion_mode)

    reasons: list[str] = []
    for side in ("clean", "typo"):
        t = trunc[side]
        if not t.trigger_found:
            reasons.append(f"no_trigger_{side}")
        else:
            benign_dup = dedup_same_answer_triggers and t.trigger_answers_identical
            if t.trigger_count > 1 and not benign_dup:
                reasons.append(f"multi_trigger_{side}")
            if t.early_trigger:
                reasons.append(f"early_trigger_{side}")
            if t.residual_fragment:
                reasons.append(f"residual_fragment_{side}")

    return CellInputs(
        sample_id=pair.sample_id,
        prompts=prompts,
        forced_cots=forced_cots,
        truncation=trunc,
        exclude=len(reasons) > 0,
        exclude_reasons=reasons,
    )
