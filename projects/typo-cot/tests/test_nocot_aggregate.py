"""実験14: aggregate.py のヘルパー (設定名パース / Mantel-Haenszel OR) のテスト.

aggregate.py は scripts 配下なので importlib でファイルから読み込む。
"""

import importlib.util
from pathlib import Path

import pytest

_AGG = (
    Path(__file__).resolve().parents[1] / "scripts" / "exp14_nocot" / "aggregate.py"
)
_spec = importlib.util.spec_from_file_location("exp14_aggregate", _AGG)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


class TestParseName:
    @pytest.mark.parametrize(
        "name,tails,expected",
        [
            ("gemma-3-1b-it_commonsense_qa_clean", agg.CONDITIONS,
             ("gemma-3-1b-it", "commonsense_qa", "clean")),
            ("Llama-3.2-1B-Instruct_mmlu_importance__p1", agg.CONDITIONS,
             ("Llama-3.2-1B-Instruct", "mmlu", "importance")),
            ("Mistral-7B-Instruct-v0.3_mmlu_pro_random", agg.CONDITIONS,
             ("Mistral-7B-Instruct-v0.3", "mmlu_pro", "random")),
            ("gemma-3-1b-it_commonsense_qa_k4_importance", ["k4_importance", "k4_random"],
             ("gemma-3-1b-it", "commonsense_qa", "k4_importance")),
            ("Qwen2.5-7B-Instruct_math_k4_random", ["k4_importance", "k4_random"],
             ("Qwen2.5-7B-Instruct", "math", "k4_random")),
        ],
    )
    def test_parse(self, name, tails, expected) -> None:
        assert agg.parse_name(name, tails) == expected

    def test_unmatched_returns_none(self) -> None:
        assert agg.parse_name("garbage_name_foo", agg.CONDITIONS) is None


class TestMantelHaenszel:
    def test_single_stratum_matches_or(self) -> None:
        # 1 層なら MH は素の OR (Haldane 不要ケース)
        res = agg.mantel_haenszel_or([(20, 5, 5, 20)], haldane=False)
        # MH = (a*d/n) / (b*c/n) = ad/bc
        assert abs(res["mh_or"] - (20 * 20) / (5 * 5)) < 1e-9

    def test_all_strata_bc_zero_defined_with_haldane(self) -> None:
        # 全層で b*c=0 でも Haldane で定義される (crude None 回避)
        res = agg.mantel_haenszel_or([(38, 0, 2, 60), (2, 0, 1, 97)], haldane=True)
        assert res["mh_or"] is not None
        assert res["mh_or"] > 1

    def test_all_strata_bc_zero_undefined_without_haldane(self) -> None:
        res = agg.mantel_haenszel_or([(38, 0, 2, 60)], haldane=False)
        assert res["mh_or"] is None

    def test_positive_association_pools_gt_one(self) -> None:
        res = agg.mantel_haenszel_or([(30, 3, 3, 30), (25, 2, 4, 40)], haldane=False)
        assert res["mh_or"] > 3
