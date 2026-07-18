"""実験7: byte-identical 復元サンプルの within-run flip 検証ロジックのテスト.

対象: typo_cot.defense.within_run
- プロンプト対の構築が run_generation_only.py の規約と一致すること
- byte-identical 選別がプロンプト厳密一致で行われ、restoration_stats の
  fully_restored フラグとの不一致を検出できること
- ペア単位のバッチ分割 (clean/校正後が必ず同一バッチに入る) と flip 集計
"""

import pytest

from typo_cot.defense.restoration import build_reference
from typo_cot.defense.within_run import (
    aggregate_within_run,
    batch_rows,
    build_prompt_pair,
    iter_pair_batches,
    select_byte_identical,
)
from typo_cot.models.prompts import create_prompt_template


def make_sample(sample_id, original, corrected, choices=None, subset=None):
    return {
        "sample_id": sample_id,
        "original_question": original,
        "perturbed_question": corrected,
        "choices": choices,
        "perturbed_choices": None,
        "correct_answer": "A",
        "subset": subset,
    }


class TestBuildPromptPair:
    def test_gsm8k_identical_question_gives_identical_prompts(self):
        template = create_prompt_template("gsm8k")
        s = make_sample("s1", "What is 2+2?", "What is 2+2?")
        clean, corr = build_prompt_pair(template, "gsm8k", s)
        assert clean == corr
        assert "What is 2+2?" in clean

    def test_gsm8k_differing_question_gives_differing_prompts(self):
        template = create_prompt_template("gsm8k")
        s = make_sample("s1", "What is 2+2?", "Wha is 2+2?")
        clean, corr = build_prompt_pair(template, "gsm8k", s)
        assert clean != corr

    def test_mc_choices_embedding_matches_build_reference(self):
        """MC ベンチでは clean は choices 引数から、校正後は埋め込み済み
        テキストから構築する。完全復元 (= build_reference と同一) なら
        プロンプトはバイト同一になる (rebuttal within-run 実装と同じ性質)。"""
        template = create_prompt_template("arc")
        choices = ["red", "blue", "green", "yellow"]
        original = "What color is the sky?"
        corrected = build_reference(original, choices)
        s = make_sample("s1", original, corrected, choices=choices,
                        subset="ARC-Challenge")
        clean, corr = build_prompt_pair(template, "arc", s)
        assert clean == corr


class TestSelectByteIdentical:
    def test_selects_only_prompt_identical_samples(self):
        template = create_prompt_template("gsm8k")
        samples = [
            make_sample("a", "Q one?", "Q one?"),
            make_sample("b", "Q two?", "Q twoo?"),
            make_sample("c", "Q three?", "Q three?"),
        ]
        flags = {"a": True, "b": False, "c": True}
        pairs, stats = select_byte_identical(samples, flags, template, "gsm8k")
        assert [p["sample_id"] for p in pairs] == ["a", "c"]
        assert stats["n_samples"] == 3
        assert stats["n_byte_identical"] == 2
        assert stats["n_fully_restored_flag"] == 2
        assert stats["flag_mismatch_ids"] == []

    def test_pairs_carry_prompt_and_answer(self):
        template = create_prompt_template("gsm8k")
        samples = [make_sample("a", "Q one?", "Q one?")]
        pairs, _ = select_byte_identical(samples, {"a": True}, template, "gsm8k")
        assert pairs[0]["correct_answer"] == "A"
        assert "Q one?" in pairs[0]["prompt"]

    def test_flag_mismatch_is_reported_both_directions(self):
        template = create_prompt_template("gsm8k")
        samples = [
            make_sample("a", "Q one?", "Q one?"),    # identical, flag False
            make_sample("b", "Q two?", "Q twoo?"),   # not identical, flag True
        ]
        flags = {"a": False, "b": True}
        pairs, stats = select_byte_identical(samples, flags, template, "gsm8k")
        # 選別はプロンプト厳密一致が正: a は選ばれ b は選ばれない
        assert [p["sample_id"] for p in pairs] == ["a"]
        assert sorted(stats["flag_mismatch_ids"]) == ["a", "b"]


class TestPairBatching:
    def test_batch_rows_interleaves_clean_and_corrected(self):
        pairs = [{"prompt": "p1"}, {"prompt": "p2"}]
        rows = batch_rows(pairs)
        assert rows == ["p1", "p1", "p2", "p2"]

    def test_iter_pair_batches_keeps_pairs_together(self):
        pairs = [{"prompt": f"p{i}"} for i in range(5)]
        batches = list(iter_pair_batches(pairs, pairs_per_batch=2))
        assert [len(b) for b in batches] == [2, 2, 1]
        assert batches[2][0]["prompt"] == "p4"

    def test_iter_pair_batches_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            list(iter_pair_batches([], pairs_per_batch=0))


class TestAggregate:
    def test_aggregate_counts_flips_and_accuracy(self):
        records = [
            {"sample_id": "a", "ans_clean": "4", "ans_corr": "4",
             "correct_clean": True, "correct_corr": True},
            {"sample_id": "b", "ans_clean": "1", "ans_corr": "2",
             "correct_clean": True, "correct_corr": False},
            {"sample_id": "c", "ans_clean": None, "ans_corr": None,
             "correct_clean": False, "correct_corr": False},
        ]
        agg = aggregate_within_run(records)
        assert agg["n"] == 3
        assert agg["n_flip"] == 1
        assert agg["flip_rate"] == pytest.approx(1 / 3)
        assert agg["flip_ids"] == ["b"]
        assert agg["accuracy_clean"] == pytest.approx(2 / 3)
        assert agg["accuracy_corrected"] == pytest.approx(1 / 3)

    def test_aggregate_empty(self):
        agg = aggregate_within_run([])
        assert agg["n"] == 0
        assert agg["flip_rate"] is None
