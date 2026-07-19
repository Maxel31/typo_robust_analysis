"""intervention.analysis のテスト (実験1: flip 表・効果分解・bootstrap・GLMM).

GPU 不要。合成 CellOutcome のみで検証する。
"""

import pytest

from typo_cot.intervention.analysis import (
    bootstrap_ci,
    flip_table,
    glmm_decomposition,
)
from typo_cot.intervention.runner import CellOutcome


def make_outcome(
    sample_id: str,
    a: str = "18",
    b: str = "17",
    c: str = "18",
    d: str = "17",
    exclude: bool = False,
    cot_changed: bool = True,
    a_correct: bool = True,
    te_match: bool = True,
) -> CellOutcome:
    return CellOutcome(
        sample_id=sample_id,
        answers={"A": a, "B": b, "C": c, "D": d},
        generated={k: f"The answer is {v}." for k, v in {"A": a, "B": b, "C": c, "D": d}.items()},
        correct_answer="18",
        exclude=exclude,
        exclude_reasons=["residual_fragment_typo"] if exclude else [],
        cot_changed=cot_changed,
        a_correct=a_correct,
        te_match=te_match,
    )


class TestFlipTable:
    def test_cot_mediated_pattern(self):
        # 10 サンプル: 8 件は CoT 媒介 flip (B,D flip / C 復帰)、2 件は flip なし
        outcomes = [make_outcome(f"s{i}") for i in range(8)]
        outcomes += [
            make_outcome(f"s{i}", b="18", c="18", d="18", cot_changed=False) for i in (8, 9)
        ]
        table = flip_table(outcomes)

        assert table["n_total"] == 10
        assert table["n_included"] == 10
        assert table["flip_rate"]["TE"] == pytest.approx(0.8)
        assert table["flip_rate"]["DE"] == pytest.approx(0.0)
        assert table["flip_rate"]["IE"] == pytest.approx(0.8)
        # 見出し: TE で flip した 8 件全てで clean CoT (Cセル) が元の答えに復帰
        assert table["headline_restore_rate"] == pytest.approx(1.0)
        # IE を CoT 変化事例に条件付け: 変化 8 件全てが flip
        assert table["ie_flip_rate_given_cot_changed"] == pytest.approx(1.0)
        assert table["te_match_rate"] == pytest.approx(1.0)

    def test_excluded_and_incorrect_filtered(self):
        outcomes = [
            make_outcome("s0"),
            make_outcome("s1", exclude=True),
            make_outcome("s2", a_correct=False),
        ]
        table = flip_table(outcomes)
        assert table["n_total"] == 3
        assert table["n_excluded"] == 1
        # 主分析は除外なし & A 正解のみ
        assert table["n_included"] == 1

    def test_direct_pathway_pattern(self):
        # DE 優位パターン: C セル (clean CoT 強制) でも flip する
        outcomes = [make_outcome(f"s{i}", c="17") for i in range(4)]
        table = flip_table(outcomes)
        assert table["flip_rate"]["DE"] == pytest.approx(1.0)
        assert table["headline_restore_rate"] == pytest.approx(0.0)

    def test_empty(self):
        table = flip_table([])
        assert table["n_total"] == 0
        assert table["flip_rate"]["TE"] is None


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        values = [1] * 70 + [0] * 30
        lo, hi = bootstrap_ci(values, n_boot=500, seed=0)
        assert lo <= 0.7 <= hi
        assert 0.5 < lo < 0.7
        assert 0.7 < hi < 0.9

    def test_degenerate(self):
        lo, hi = bootstrap_ci([1, 1, 1], n_boot=100, seed=0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_empty(self):
        assert bootstrap_ci([], n_boot=10, seed=0) == (None, None)


class TestGLMM:
    def test_cot_effect_dominant(self):
        # CoT 媒介: cot_typo で flip、q_typo 単独では flip しない
        outcomes = [make_outcome(f"s{i}") for i in range(60)]
        outcomes += [
            make_outcome(f"t{i}", b="18", c="18", d="18", cot_changed=False) for i in range(20)
        ]
        res = glmm_decomposition(outcomes)
        assert res is not None
        assert "cot_typo" in res and "q_typo" in res and "q_typo:cot_typo" in res
        # CoT 主効果が質問主効果より大きい
        assert res["cot_typo"]["coef"] > res["q_typo"]["coef"]
        for v in res.values():
            assert v["coef"] == v["coef"]  # not NaN
