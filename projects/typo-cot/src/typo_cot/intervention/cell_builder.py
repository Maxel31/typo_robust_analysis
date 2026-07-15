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
from dataclasses import dataclass, field

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
    """

    prefix: str
    trigger_found: bool
    trigger_count: int
    early_trigger: bool
    residual_fragment: bool
    trigger_char_start: int | None = None


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

    return TruncationResult(
        prefix=prefix,
        trigger_found=True,
        trigger_count=len(matches),
        early_trigger=early,
        residual_fragment=residual,
        trigger_char_start=first.start(),
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
) -> CellInputs:
    """4 セルの teacher-forcing 入力を構築する.

    Args:
        pair: clean × typo のサンプル対
        trigger_pattern: 答え句の正規表現 (モデル別に差し替え可)

    Returns:
        CellInputs
    """
    trunc = {
        "clean": truncate_before_answer(pair.cot_clean, pair.benchmark, trigger_pattern),
        "typo": truncate_before_answer(pair.cot_typo, pair.benchmark, trigger_pattern),
    }

    prompt_by_q = {
        "clean": _build_prompt(
            pair.benchmark, pair.question_clean, pair.choices_clean, pair.subset
        ),
        "typo": _build_prompt(pair.benchmark, pair.question_typo, pair.choices_typo, pair.subset),
    }

    prompts: dict[str, str] = {}
    forced_cots: dict[str, str] = {}
    for cell, (q_side, cot_side) in CELL_DEFINITIONS.items():
        prompts[cell] = prompt_by_q[q_side]
        forced_cots[cell] = trunc[cot_side].prefix

    reasons: list[str] = []
    for side in ("clean", "typo"):
        t = trunc[side]
        if not t.trigger_found:
            reasons.append(f"no_trigger_{side}")
        else:
            if t.trigger_count > 1:
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
