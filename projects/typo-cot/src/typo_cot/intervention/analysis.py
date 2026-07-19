"""実験1: flip 表・効果分解 (TE/DE/IE)・bootstrap CI・GLMM.

セル定義 (cell_builder.CELL_DEFINITIONS):
    A = (clean, clean) 基準 / B = (typo, typo) TE /
    C = (typo, clean) DE / D = (clean, typo) IE

flip は「基準セル A の答えと異なる」こと。主分析は除外フラグなし &
A 正解のサンプルに限定する。DE と IE の和は TE に一致しなくてよい
(GLMM で交互作用込みに同時推定する)。
"""

import logging
import random as _random

from typo_cot.intervention.runner import CellOutcome

logger = logging.getLogger(__name__)

# 効果名 → 対応セル
EFFECT_CELLS = {"TE": "B", "DE": "C", "IE": "D"}


def _flipped(outcome: CellOutcome, cell: str) -> bool:
    return outcome.answers[cell].strip() != outcome.answers["A"].strip()


def _rate(num: int, den: int) -> float | None:
    return num / den if den > 0 else None


def flip_table(outcomes: list[CellOutcome]) -> dict:
    """4 条件の flip 表と主要派生指標を集計する.

    Returns:
        dict:
            n_total / n_excluded / n_a_incorrect / n_included
            flip_rate: {"TE","DE","IE"} → 率 (included 上)
            flip_count: 同カウント
            headline_restore_rate: TE で flip した事例のうち
                clean CoT (C セル) で元の答えに戻った割合
            ie_flip_rate_given_cot_changed: CoT が実際に変化した事例に
                条件付けた IE flip 率
            te_match_rate: 再生成 B セルとアーカイブ answer_typo の一致率
            flip_rate_sensitivity: 除外を含めた版 (感度分析)
    """
    n_total = len(outcomes)
    n_excluded = sum(1 for o in outcomes if o.exclude)
    included = [o for o in outcomes if not o.exclude and o.a_correct]
    n_included = len(included)
    n_a_incorrect = sum(1 for o in outcomes if not o.exclude and not o.a_correct)

    flip_count = {
        eff: sum(1 for o in included if _flipped(o, cell)) for eff, cell in EFFECT_CELLS.items()
    }
    flip_rate = {eff: _rate(c, n_included) for eff, c in flip_count.items()}

    # 感度分析: 除外事例も含めた版 (A 正解の条件は維持)
    sens = [o for o in outcomes if o.a_correct]
    flip_rate_sensitivity = {
        eff: _rate(sum(1 for o in sens if _flipped(o, cell)), len(sens))
        for eff, cell in EFFECT_CELLS.items()
    }

    te_flipped = [o for o in included if _flipped(o, "B")]
    headline_restore_rate = _rate(
        sum(1 for o in te_flipped if not _flipped(o, "C")), len(te_flipped)
    )

    cot_changed = [o for o in included if o.cot_changed]
    ie_given_changed = _rate(sum(1 for o in cot_changed if _flipped(o, "D")), len(cot_changed))

    te_known = [o for o in outcomes if o.te_match is not None]
    te_match_rate = _rate(sum(1 for o in te_known if o.te_match), len(te_known))

    return {
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_a_incorrect": n_a_incorrect,
        "n_included": n_included,
        "flip_count": flip_count,
        "flip_rate": flip_rate,
        "flip_rate_sensitivity": flip_rate_sensitivity,
        "headline_restore_rate": headline_restore_rate,
        "n_te_flipped": len(te_flipped),
        "ie_flip_rate_given_cot_changed": ie_given_changed,
        "n_cot_changed": len(cot_changed),
        "te_match_rate": te_match_rate,
    }


def bootstrap_ci(
    values: list[int] | list[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """二値列の平均のパーセンタイル bootstrap CI.

    Args:
        values: 0/1 の列 (flip 指示子など)
        n_boot: リサンプリング回数
        seed: 乱数シード
        alpha: 両側有意水準 (0.05 → 95% CI)

    Returns:
        (下限, 上限)。空列なら (None, None)
    """
    n = len(values)
    if n == 0:
        return (None, None)
    rng = _random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (means[lo_idx], means[hi_idx])


def bootstrap_flip_cis(
    outcomes: list[CellOutcome],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float | None, float | None]]:
    """included サンプル上の TE/DE/IE flip 率の bootstrap CI."""
    included = [o for o in outcomes if not o.exclude and o.a_correct]
    return {
        eff: bootstrap_ci([int(_flipped(o, cell)) for o in included], n_boot=n_boot, seed=seed)
        for eff, cell in EFFECT_CELLS.items()
    }


def glmm_decomposition(outcomes: list[CellOutcome]) -> dict | None:
    """GLMM: flip ~ q_typo * cot_typo + (1|item) を変分ベイズで推定する.

    各サンプルを 4 セルに展開する (A セルは構造的に flip=0 で切片を係留。
    切片の縮退はベイズ事前分布で正則化される)。statsmodels の
    BinomialBayesMixedGLM を使用。収束失敗時は None。

    Returns:
        {"Intercept": {"coef","sd"}, "q_typo": {...}, "cot_typo": {...},
         "q_typo:cot_typo": {...}} または None
    """
    included = [o for o in outcomes if not o.exclude and o.a_correct]
    if len(included) < 5:
        logger.warning("GLMM: included サンプルが少なすぎます (n=%d)", len(included))
        return None

    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as e:  # pragma: no cover
        logger.warning("GLMM: statsmodels/pandas が利用できません: %s", e)
        return None

    from typo_cot.intervention.cell_builder import CELL_DEFINITIONS

    rows = []
    for o in included:
        for cell, (q_side, cot_side) in CELL_DEFINITIONS.items():
            rows.append(
                {
                    "item": o.sample_id,
                    "q_typo": int(q_side == "typo"),
                    "cot_typo": int(cot_side == "typo"),
                    "flip": int(_flipped(o, cell)),
                }
            )
    df = pd.DataFrame(rows)

    try:
        model = BinomialBayesMixedGLM.from_formula(
            "flip ~ q_typo * cot_typo", {"item": "0 + C(item)"}, df
        )
        result = model.fit_vb()
    except Exception as e:
        logger.warning("GLMM の推定に失敗: %s", e)
        return None

    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(result.model.exog_names):
        out[name] = {
            "coef": float(result.params[i]),
            "sd": float(result.fe_sd[i]) if hasattr(result, "fe_sd") else float("nan"),
        }
    return out
