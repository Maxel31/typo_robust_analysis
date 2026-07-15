"""intervention.runner のテスト (GPU 不要、generate_fn をモック).

run_cells は 4 セルの teacher-forcing 入力を構築し、注入された
generate_fn (プロンプトリスト → 継続テキストリスト) で答えスパンを
生成させ、抽出・照合して CellOutcome を返す。
"""

import pytest

from typo_cot.intervention.records import PairRecord
from typo_cot.intervention.runner import CellOutcome, run_cells


def make_pair(sample_id: str = "gsm8k_00000", flip: bool = True) -> PairRecord:
    """clean=18, typo=17 (flip=True) / typo=18 (flip=False) の合成ペア."""
    answer_typo = "17" if flip else "18"
    return PairRecord(
        sample_id=sample_id,
        model="google/gemma-3-4b-it",
        benchmark="gsm8k",
        question_clean="Janet has 16 eggs. How many dollars?",
        question_typo="Janeet has 16 egs. How many dollars?",
        choices_clean=None,
        choices_typo=None,
        subset="default",
        correct_answer="18",
        cot_clean="\nShe computes 9 * 2 = 18.\nThe answer is 18.\n",
        cot_typo=f" She computes 9 * 2 = {answer_typo}.\nThe answer is {answer_typo}.\n",
        answer_clean="18",
        answer_typo=answer_typo,
        is_correct_clean=True,
    )


def fake_generate(prompts: list[str]) -> list[str]:
    """CoT prefix の最後の計算結果をそのまま答えとして返す決定的モック.

    「CoT が無事なら答え段は復帰する」挙動 (IE 優位) を模す:
    forced CoT に "= 17" があれば 17、なければ 18 を返す。
    """
    out = []
    for p in prompts:
        if "= 17" in p:
            out.append("The answer is 17.\n")
        else:
            out.append("The answer is 18.\n")
    return out


class TestRunCells:
    def test_outcomes_structure(self):
        pairs = [make_pair()]
        outcomes = run_cells(pairs, fake_generate)
        assert len(outcomes) == 1
        o = outcomes[0]
        assert isinstance(o, CellOutcome)
        assert set(o.answers.keys()) == {"A", "B", "C", "D"}
        assert set(o.generated.keys()) == {"A", "B", "C", "D"}

    def test_answers_follow_cot(self):
        # CoT 媒介のモック: 答えは CoT 側に追従する
        outcomes = run_cells([make_pair()], fake_generate)
        o = outcomes[0]
        assert o.answers["A"] == "18"  # clean q + clean cot
        assert o.answers["B"] == "17"  # typo q + typo cot
        assert o.answers["C"] == "18"  # typo q + clean cot (DE: 復帰)
        assert o.answers["D"] == "17"  # clean q + typo cot (IE: flip)

    def test_te_match_against_archive(self):
        o = run_cells([make_pair()], fake_generate)[0]
        # 再生成した B セルの答えがアーカイブの answer_typo と一致
        assert o.te_match is True

    def test_a_correct_and_cot_changed(self):
        o = run_cells([make_pair(flip=True)], fake_generate)[0]
        assert o.a_correct is True
        assert o.cot_changed is True

        o2 = run_cells([make_pair(flip=False)], fake_generate)[0]
        # cot_typo は "18" 版で文言が異なるので変化あり
        assert o2.answers["B"] == "18"
        assert o2.te_match is True

    def test_exclusion_propagated(self):
        pair = make_pair()
        pair.cot_typo = "never concludes"
        o = run_cells([pair], fake_generate)[0]
        assert o.exclude is True
        assert "no_trigger_typo" in o.exclude_reasons

    def test_batching_multiple_pairs(self):
        calls: list[int] = []

        def counting_generate(prompts: list[str]) -> list[str]:
            calls.append(len(prompts))
            return fake_generate(prompts)

        pairs = [make_pair(f"gsm8k_{i:05d}") for i in range(3)]
        outcomes = run_cells(pairs, counting_generate, batch_size=2)
        assert len(outcomes) == 3
        # バッチサイズ 2 を超える呼び出しはない
        assert all(c <= 2 for c in calls)
        # 4 セル × 3 サンプル = 12 プロンプトが流れる
        assert sum(calls) == 12
