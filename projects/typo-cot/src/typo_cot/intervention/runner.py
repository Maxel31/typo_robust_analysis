"""実験1: 4 セル teacher-forcing 生成の実行コア.

モデル依存部分は generate_fn (プロンプトリスト → 継続テキストリスト) として
注入する。ユニットテストではモック、GPU 実行時は ModelWrapper.generate_batch
を包んだクロージャを渡す。答え抽出は 4 セルで完全同一
(evaluation.extractor の該当ベンチマーク抽出器)。
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.intervention.cell_builder import CELL_DEFINITIONS, build_cell_inputs
from typo_cot.intervention.records import PairRecord

logger = logging.getLogger(__name__)

GenerateFn = Callable[[list[str]], list[str]]

CELLS = tuple(CELL_DEFINITIONS.keys())  # ("A", "B", "C", "D")


@dataclass
class CellOutcome:
    """1サンプルの 4 セル生成・抽出結果.

    Attributes:
        sample_id: サンプル ID
        answers: セル名 → 抽出された答え
        generated: セル名 → 生成された継続テキスト (答えスパン)
        correct_answer: 正解
        exclude: 切断フラグにより主分析から除外するか
        exclude_reasons: 除外理由
        cot_changed: typo CoT (切断後 prefix) が clean と実際に異なるか
        a_correct: 基準セル A の答えが正解か (主推定量の条件付け)
        te_match: 再生成した B セルの答えがアーカイブの answer_typo と一致するか
    """

    sample_id: str
    answers: dict[str, str]
    generated: dict[str, str]
    correct_answer: str
    exclude: bool
    exclude_reasons: list[str] = field(default_factory=list)
    cot_changed: bool = False
    a_correct: bool = False
    te_match: bool | None = None


def _normalize_ws(text: str) -> str:
    """空白の揺れを潰して CoT 変化判定に使う."""
    return re.sub(r"\s+", " ", text).strip()


def run_cells(
    pairs: list[PairRecord],
    generate_fn: GenerateFn,
    batch_size: int = 8,
    trigger_pattern: str | None = None,
    dedup_same_answer_triggers: bool = False,
    prompt_builder=None,
    truncator=None,
    extract_fn: Callable[[str], str] | None = None,
) -> list[CellOutcome]:
    """全ペアの 4 セルを teacher-forcing で流し、答えを抽出する.

    Args:
        pairs: PairRecord のリスト (同一 benchmark を想定)
        generate_fn: プロンプトリスト → 継続テキストリスト (greedy を想定)
        batch_size: generate_fn 1 回に渡す最大プロンプト数
        trigger_pattern: 答え句の正規表現 (モデル別差し替え)
        dedup_same_answer_triggers: 同一答えの複数トリガーを除外しない
            (cell_builder.build_cell_inputs に委譲。既定 False で従来挙動)
        prompt_builder: プロンプト骨格構築関数 (R1: チャットテンプレート)。
            build_cell_inputs へ委譲 (None なら基底の生プロンプト)。
        truncator: CoT 切断関数 (R1: <think> 構造対応)。build_cell_inputs へ委譲。
        extract_fn: 答えスパン → 答え文字列。None なら基底ベンチ抽出器の extract。
            R1蒸留系は reasoning 抽出チェーンを注入する。

    Returns:
        pairs と同順の CellOutcome リスト
    """
    if not pairs:
        return []

    extractor = create_extractor(pairs[0].benchmark)
    _extract = (
        extract_fn if extract_fn is not None else (lambda span: extractor.extract(span).extracted_answer)
    )

    # 4 セル × 全サンプルのタスクを平坦化してバッチ処理
    cell_inputs = [
        build_cell_inputs(
            p,
            trigger_pattern=trigger_pattern,
            dedup_same_answer_triggers=dedup_same_answer_triggers,
            prompt_builder=prompt_builder,
            truncator=truncator,
        )
        for p in pairs
    ]
    tasks: list[tuple[int, str, str]] = []  # (pair_idx, cell, full_input)
    for i, ci in enumerate(cell_inputs):
        for cell in CELLS:
            tasks.append((i, cell, ci.full_input(cell)))

    generated: dict[tuple[int, str], str] = {}
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        outputs = generate_fn([t[2] for t in chunk])
        if len(outputs) != len(chunk):
            raise ValueError(
                f"generate_fn が {len(chunk)} 件に対し {len(outputs)} 件を返しました"
            )
        for (pair_idx, cell, _), out in zip(chunk, outputs, strict=True):
            generated[(pair_idx, cell)] = out

    outcomes: list[CellOutcome] = []
    for i, (pair, ci) in enumerate(zip(pairs, cell_inputs, strict=True)):
        gen = {cell: generated[(i, cell)] for cell in CELLS}
        answers = {cell: _extract(gen[cell]) for cell in CELLS}

        ans_b = answers["B"].strip()
        archive_typo = (pair.answer_typo or "").strip()
        te_match = ans_b == archive_typo
        if not te_match and ans_b and archive_typo:
            # 抽出器の正規化差 (カンマ・括弧等) を許容
            te_match = extractor.is_correct(ans_b, archive_typo)

        cot_changed = _normalize_ws(ci.forced_cots["A"]) != _normalize_ws(ci.forced_cots["B"])

        outcomes.append(
            CellOutcome(
                sample_id=pair.sample_id,
                answers=answers,
                generated=gen,
                correct_answer=pair.correct_answer,
                exclude=ci.exclude,
                exclude_reasons=list(ci.exclude_reasons),
                cot_changed=cot_changed,
                a_correct=extractor.is_correct(answers["A"], pair.correct_answer),
                te_match=te_match,
            )
        )

    return outcomes
