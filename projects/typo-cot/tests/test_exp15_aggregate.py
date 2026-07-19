"""scripts/exp15/aggregate.py の純関数テスト (実験15 集計・H15 判定).

GPU 不要。合成 payload (run_free_generation の出力スキーマ) で summarize /
is_fresh_flip / h15_verdict を検証する。aggregate.py は scripts/ 配下のため
importlib で読み込む。
"""

import importlib.util
from pathlib import Path

import pytest

_AGG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "exp15" / "aggregate.py"
_spec = importlib.util.spec_from_file_location("exp15_aggregate", _AGG_PATH)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def _pair(
    sample_id,
    clean_correct,
    typo_correct,
    early_denoise_correct,
    early_rouge_gain,
    early_onset,
    late_denoise_correct=False,
    late_rouge_gain=0.02,
    noise_correct=False,
    noise_onset=3,
    sham_identical=True,
    baseline_onset=4,
):
    return {
        "sample_id": sample_id,
        "baseline": {
            "clean": {"is_correct": clean_correct, "answer": "18"},
            "typo": {"is_correct": typo_correct, "answer": "12"},
            "rouge_l_typo_vs_clean": 0.40,
            "onset_typo_vs_clean": baseline_onset,
        },
        "sham": {"generation_identical_to_typo": sham_identical, "answer_unchanged": True},
        "cells": [
            {
                "level": "early", "direction": "denoise", "is_correct": early_denoise_correct,
                "rouge_l_vs_clean": 0.40 + early_rouge_gain, "rouge_l_vs_typo": 0.5,
                "rouge_gain_vs_typo": early_rouge_gain, "onset_vs_clean": early_onset,
            },
            {
                "level": "late", "direction": "denoise", "is_correct": late_denoise_correct,
                "rouge_l_vs_clean": 0.40 + late_rouge_gain, "rouge_l_vs_typo": 0.9,
                "rouge_gain_vs_typo": late_rouge_gain, "onset_vs_clean": 2,
            },
            {
                "level": "early", "direction": "noise", "is_correct": noise_correct,
                "rouge_l_vs_clean": 0.5, "rouge_l_vs_typo": 0.7, "onset_vs_clean": noise_onset,
            },
        ],
    }


class TestIsFreshFlip:
    def test_true_when_clean_correct_typo_wrong(self):
        assert agg.is_fresh_flip(_pair("a", True, False, True, 0.4, None)) is True

    def test_false_when_typo_also_correct(self):
        assert agg.is_fresh_flip(_pair("a", True, True, True, 0.4, None)) is False

    def test_false_when_clean_wrong(self):
        assert agg.is_fresh_flip(_pair("a", False, False, True, 0.4, None)) is False

    def test_false_when_excluded(self):
        assert agg.is_fresh_flip({"excluded": "span_not_found"}) is False


class TestSummarize:
    def test_counts_and_sham(self):
        payloads = [
            _pair("s1", True, False, True, 0.45, None),   # fresh flip, early restores
            _pair("s2", True, False, False, 0.05, 1),      # fresh flip, early fails
            _pair("s3", True, True, True, 0.4, None),       # not a flip (typo correct)
            {"sample_id": "s4", "excluded": "span_not_found"},
        ]
        s = agg.summarize(payloads)
        assert s["n_done"] == 3
        assert s["n_excluded"] == 1
        assert s["n_fresh_flip"] == 2
        assert s["n_clean_correct"] == 3
        assert s["sham_identical_rate"] == pytest.approx(1.0)

    def test_early_denoise_metrics_on_fresh_only(self):
        payloads = [
            _pair("s1", True, False, True, 0.45, None),
            _pair("s2", True, False, False, 0.05, 1),
            _pair("s3", True, True, True, 0.4, None),  # excluded from fresh
        ]
        s = agg.summarize(payloads)
        early = s["levels"]["early"]["denoise"]
        assert early["n"] == 2  # only fresh flips
        assert early["restoration_rate"] == pytest.approx(0.5)  # 1 of 2 restored
        assert early["mean_rouge_gain"] == pytest.approx((0.45 + 0.05) / 2)
        # onset disappeared: s1 None -> True, s2 onset 1 < baseline 4 -> False
        assert early["onset_disappear_rate"] == pytest.approx(0.5)

    def test_noise_uses_clean_correct_subset(self):
        payloads = [
            _pair("s1", True, False, True, 0.45, None, noise_correct=False),
            _pair("s3", True, True, True, 0.4, None, noise_correct=True),
        ]
        s = agg.summarize(payloads)
        noise = s["levels"]["early"]["noise"]
        assert noise["n"] == 2  # both clean-correct
        assert noise["induced_flip_rate"] == pytest.approx(0.5)  # s1 wrong, s3 correct


class TestH15Verdict:
    def test_all_pass(self):
        # early strong restore + rouge gain, late null, noise induces flip
        payloads = [
            _pair(f"s{i}", True, False, True, 0.45, None, late_rouge_gain=0.01, noise_correct=False)
            for i in range(8)
        ]
        s = agg.summarize(payloads)
        v = agg.h15_verdict(s)
        assert v["rouge_gain_ge_0.15"] is True
        assert v["flip_halved"] is True
        assert v["onset_majority_disappeared"] is True
        assert v["late_near_null"] is True
        assert v["noise_induces_flip"] is True
        assert v["overall_backbone_closed"] is True

    def test_fails_when_early_weak(self):
        payloads = [
            _pair(f"s{i}", True, False, False, 0.05, 1, noise_correct=True) for i in range(8)
        ]
        s = agg.summarize(payloads)
        v = agg.h15_verdict(s)
        assert v["rouge_gain_ge_0.15"] is False
        assert v["flip_halved"] is False
        assert v["noise_induces_flip"] is False
        assert v["overall_backbone_closed"] is False
