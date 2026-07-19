"""intervention.cell_builder のテスト (実験1: CoT移植 2×2).

GPU 不要。切断関数と4セルプロンプト構築のみを検証する。
"""

import pytest

from typo_cot.intervention.cell_builder import (
    CellInputs,
    TruncationResult,
    build_cell_inputs,
    truncate_before_answer,
)
from typo_cot.intervention.records import PairRecord


class TestTruncateBeforeAnswer:
    """答え句テンプレート直前での CoT 切断."""

    def test_basic_truncation(self):
        cot = "\nShe has 16 - 3 - 4 = 9 eggs left. 9 * 2 = 18 dollars.\nThe answer is 18.\n"
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert isinstance(res, TruncationResult)
        assert res.trigger_found is True
        assert res.prefix == "\nShe has 16 - 3 - 4 = 9 eggs left. 9 * 2 = 18 dollars.\n"
        assert res.trigger_count == 1
        assert res.early_trigger is False

    def test_lowercase_trigger(self):
        cot = "Reasoning here. the answer is (B)."
        res = truncate_before_answer(cot, benchmark="mmlu")
        assert res.trigger_found is True
        assert res.prefix == "Reasoning here. "

    def test_no_trigger(self):
        cot = "Some reasoning that never concludes."
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert res.trigger_found is False
        assert res.prefix == cot

    def test_multiple_triggers_flagged(self):
        cot = "The answer is maybe 3. Wait. More work. The answer is 5."
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert res.trigger_count == 2
        # 最初の出現位置で切断
        assert res.prefix == ""

    def test_early_trigger_flagged(self):
        # 序盤 (先頭25%以内) に出現 → early_trigger
        cot = "The answer is 5. Because of a long explanation that follows " + "x" * 200
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert res.early_trigger is True

    def test_late_trigger_not_early(self):
        cot = "y" * 200 + " The answer is 5."
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert res.early_trigger is False

    def test_residual_fragment_flagged(self):
        # トリガーが複数あっても最初の出現で切断される
        cot = "I think the answer is likely large. Compute 2+3=5.\nThe answer is 5."
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert res.prefix == "I think "
        assert res.trigger_count == 2

    def test_answer_fragment_in_prefix(self):
        # トリガーの変種 ("Answer:" 等) が prefix に残ると residual_fragment
        cot = "Answer: 4 seems wrong. Recompute. The answer is 5."
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert res.prefix == "Answer: 4 seems wrong. Recompute. "
        assert res.residual_fragment is True

    def test_clean_prefix_no_residual(self):
        cot = "Compute 2+3=5.\nThe answer is 5."
        res = truncate_before_answer(cot, benchmark="gsm8k")
        assert res.residual_fragment is False

    def test_custom_trigger_pattern(self):
        # DeepSeek-R1-Distill 系などモデル別トリガーを差し替え可能
        cot = "<think>reasoning</think>\n**Final Answer**: 5"
        res = truncate_before_answer(
            cot, benchmark="gsm8k", trigger_pattern=r"\*\*Final Answer\*\*"
        )
        assert res.trigger_found is True
        assert res.prefix == "<think>reasoning</think>\n"


class TestBuildCellInputs:
    """4セル (A/B/C/D) の teacher-forcing 入力構築."""

    @pytest.fixture
    def pair(self) -> PairRecord:
        return PairRecord(
            sample_id="gsm8k_00000",
            model="google/gemma-3-4b-it",
            benchmark="gsm8k",
            question_clean="Janet has 16 eggs. How many dollars?",
            question_typo="Janeet has 16 egs. How many dollars?",
            choices_clean=None,
            choices_typo=None,
            subset="default",
            correct_answer="18",
            cot_clean="\nShe computes 9 * 2 = 18.\nThe answer is 18.\n",
            cot_typo=" She computes 9 * 2 = 17.\nThe answer is 17.\n",
            answer_clean="18",
            answer_typo="17",
            is_correct_clean=True,
        )

    def test_four_cells_built(self, pair):
        cells = build_cell_inputs(pair)
        assert isinstance(cells, CellInputs)
        assert set(cells.prompts.keys()) == {"A", "B", "C", "D"}

    def test_cell_question_cot_combination(self, pair):
        cells = build_cell_inputs(pair)
        # A = (Q_c, C_c), B = (Q_p, C_p), C = (Q_p, C_c), D = (Q_c, C_p)
        assert pair.question_clean in cells.prompts["A"]
        assert pair.question_clean in cells.prompts["D"]
        assert pair.question_typo in cells.prompts["B"]
        assert pair.question_typo in cells.prompts["C"]
        clean_prefix = "\nShe computes 9 * 2 = 18.\n"
        typo_prefix = " She computes 9 * 2 = 17.\n"
        assert cells.forced_cots["A"] == clean_prefix
        assert cells.forced_cots["C"] == clean_prefix
        assert cells.forced_cots["B"] == typo_prefix
        assert cells.forced_cots["D"] == typo_prefix

    def test_prompt_skeleton_identical_across_cells(self, pair):
        # few-shot 文脈は4セルで完全同一 (質問部分のみ異なる)
        cells = build_cell_inputs(pair)
        pa = cells.prompts["A"]
        pc = cells.prompts["C"]
        # 質問を除いた骨格が同じ: clean 質問を typo 質問に置換すると一致する
        assert pa.replace(pair.question_clean, pair.question_typo) == pc

    def test_answer_phrase_not_in_forced_cot(self, pair):
        cells = build_cell_inputs(pair)
        for cot in cells.forced_cots.values():
            assert "The answer is" not in cot

    def test_exclusion_flags_propagated(self, pair):
        cells = build_cell_inputs(pair)
        assert cells.truncation["clean"].trigger_found is True
        assert cells.truncation["typo"].trigger_found is True
        assert cells.exclude is False

    def test_exclude_when_no_trigger(self, pair):
        pair.cot_typo = "never concludes anything"
        cells = build_cell_inputs(pair)
        assert cells.exclude is True
        assert "no_trigger_typo" in cells.exclude_reasons

    def test_dedup_off_by_default_excludes_same_answer_multitrigger(self, pair):
        # Qwen 癖: 同一答えを2回述べる ("The answer is 18. --- The answer is 18.")
        # 既定 (dedup 無効) では従来どおり multi_trigger で除外される (後方互換)
        pair.cot_clean = (
            "\nShe computes 9 * 2 = 18.\nThe answer is 18.\n--- The answer is 18.\n"
        )
        cells = build_cell_inputs(pair)
        assert cells.truncation["clean"].trigger_count == 2
        assert "multi_trigger_clean" in cells.exclude_reasons
        assert cells.exclude is True

    def test_truncation_records_trigger_answers(self, pair):
        pair.cot_clean = (
            "\nShe computes 9 * 2 = 18.\nThe answer is 18.\n--- The answer is 18.\n"
        )
        cells = build_cell_inputs(pair)
        t = cells.truncation["clean"]
        assert t.trigger_answers == ["18", "18"]
        assert t.trigger_answers_identical is True

    def test_truncation_different_answers_not_identical(self, pair):
        pair.cot_clean = "Reason.\nThe answer is 3.\nWait, recompute.\nThe answer is 5.\n"
        cells = build_cell_inputs(pair)
        t = cells.truncation["clean"]
        assert t.trigger_answers == ["3", "5"]
        assert t.trigger_answers_identical is False

    def test_dedup_on_keeps_same_answer_multitrigger(self, pair):
        # dedup 有効時: 同一答えの重複は良性とみなし除外しない (n_incl 回復)
        pair.cot_clean = (
            "\nShe computes 9 * 2 = 18.\nThe answer is 18.\n--- The answer is 18.\n"
        )
        pair.cot_typo = " She computes 9 * 2 = 17.\nThe answer is 17.\n--- The answer is 17.\n"
        cells = build_cell_inputs(pair, dedup_same_answer_triggers=True)
        assert "multi_trigger_clean" not in cells.exclude_reasons
        assert "multi_trigger_typo" not in cells.exclude_reasons
        assert cells.exclude is False
        # 切断点は最初のトリガー直前のまま (再生成不要)
        assert cells.forced_cots["A"] == "\nShe computes 9 * 2 = 18.\n"
        assert "The answer is" not in cells.forced_cots["A"]

    def test_dedup_on_still_excludes_different_answers(self, pair):
        # 異なる答えのトリガーは真の曖昧さ → dedup 有効でも除外を維持
        pair.cot_clean = "Reason.\nThe answer is 3.\nWait, recompute.\nThe answer is 5.\n"
        cells = build_cell_inputs(pair, dedup_same_answer_triggers=True)
        assert "multi_trigger_clean" in cells.exclude_reasons
        assert cells.exclude is True

    def test_dedup_on_preserves_early_trigger_exclusion(self, pair):
        # 同一答えでも序盤 (early_trigger) は別基準として除外を維持
        pair.cot_clean = "The answer is 18. " + "x" * 300 + " The answer is 18."
        cells = build_cell_inputs(pair, dedup_same_answer_triggers=True)
        assert cells.truncation["clean"].trigger_answers_identical is True
        assert "multi_trigger_clean" not in cells.exclude_reasons
        assert "early_trigger_clean" in cells.exclude_reasons
        assert cells.exclude is True

    def test_strip_conclusion_removes_last_line_of_cell_c_only(self):
        # A2 (ii): セルC (typo質問+clean CoT) の強制CoT末尾行を除去するオプション。
        # 末尾の読み上げ行に金答え数値が載る GSM8K で、これを消しても restore が
        # 保たれれば「丸写しでなく再導出」を支持する。
        pair = PairRecord(
            sample_id="gsm8k_00001",
            model="google/gemma-3-4b-it",
            benchmark="gsm8k",
            question_clean="Q clean?",
            question_typo="Q tpyo?",
            choices_clean=None,
            choices_typo=None,
            subset="default",
            correct_answer="18",
            cot_clean="\nFirst 16 - 3 - 4 = 9 eggs.\nSo she makes 9 * 2 = 18 dollars.\nThe answer is 18.\n",
            cot_typo="\nFirst 16 - 3 - 4 = 9 eggs.\nShe makes 9 * 2 = 17.\nThe answer is 17.\n",
            answer_clean="18",
            answer_typo="17",
            is_correct_clean=True,
        )
        base = build_cell_inputs(pair)
        stripped = build_cell_inputs(pair, strip_conclusion_mode="last_line")
        # C セルは末尾行 (= 18 dollars) が消え、金答え数値がリークしなくなる
        assert "18 dollars" in base.forced_cots["C"]
        assert "18 dollars" not in stripped.forced_cots["C"]
        assert "9 eggs" in stripped.forced_cots["C"]  # 前段の推論は保持
        # A/B/D セルは不変 (C のみ剥ぐ)
        assert stripped.forced_cots["A"] == base.forced_cots["A"]
        assert stripped.forced_cots["B"] == base.forced_cots["B"]
        assert stripped.forced_cots["D"] == base.forced_cots["D"]

    def test_strip_conclusion_none_is_noop(self):
        pair = PairRecord(
            sample_id="gsm8k_00002",
            model="google/gemma-3-4b-it",
            benchmark="gsm8k",
            question_clean="Q?",
            question_typo="Q typo?",
            choices_clean=None,
            choices_typo=None,
            subset="default",
            correct_answer="18",
            cot_clean="\nStep.\nSo 9 * 2 = 18.\nThe answer is 18.\n",
            cot_typo="\nStep.\nSo 9 * 2 = 17.\nThe answer is 17.\n",
            answer_clean="18",
            answer_typo="17",
            is_correct_clean=True,
        )
        a = build_cell_inputs(pair)
        b = build_cell_inputs(pair, strip_conclusion_mode=None)
        assert a.forced_cots == b.forced_cots

    def test_mmlu_prompt_uses_inline_choices_question(self):
        # MMLU 摂動データは選択肢込み質問 (choices=None) — そのまま骨格に入る
        pair = PairRecord(
            sample_id="mmlu_x_0001",
            model="google/gemma-3-4b-it",
            benchmark="mmlu",
            question_clean="Q stmt?",
            question_typo="Q stmtt?\n(A) x (B) y (C) z (D) w",
            choices_clean=["x", "y", "z", "w"],
            choices_typo=None,
            subset="abstract_algebra",
            correct_answer="A",
            cot_clean="Reason.\nThe answer is (A).",
            cot_typo="Reason2.\nThe answer is (B).",
            answer_clean="A",
            answer_typo="B",
            is_correct_clean=True,
        )
        cells = build_cell_inputs(pair)
        assert "(A) x (B) y (C) z (D) w" in cells.prompts["A"]  # choices リストから整形
        assert "Q stmtt?" in cells.prompts["C"]
