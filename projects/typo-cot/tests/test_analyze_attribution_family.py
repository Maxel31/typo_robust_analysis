"""実験6-(i)〜(iii) 集計 CLI (scripts/analyze_attribution_family.py) のテスト.

各手法の clean / perturbed CoT 語ランキングから内的軸 J_method@10 を再計算し、
アーカイブのサンプル別指標 (cot_rouge_l.f1 = R, cot_jaccard.top10 = J_RC 参照値)
と sample_id で結合して ρ(J_method@10 | R) を Spearman で計算する経路を検証する。
(tests/test_analyze_loo_rankings.py と同一の規約・合成データ形式)
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load("analyze_attribution_family")


def _sample_results():
    """アーカイブ full_results.json の sample_results 形式 (合成値)."""
    return [
        {
            "sample_id": f"s{i}",
            "cot_metrics": {
                "rouge_l": {"f1": 0.5 + 0.1 * i},
                "jaccard": {"top10": 0.2 + 0.1 * i},
            },
        }
        for i in range(5)
    ]


def _method_entries(rankings: dict[str, list[str]], jacc_vs_rc: float | None = 0.5):
    """run_attribution_family.py の results.json 形式 (合成値)."""
    return [
        {
            "sample_id": sid,
            "method_word_scores": [
                {"word": w, "score": float(len(words) - i)}
                for i, w in enumerate(words)
            ],
            "vs_rc_jaccard_top10": jacc_vs_rc,
        }
        for sid, words in rankings.items()
    ]


class TestMethodJaccardPairs:
    def test_pairs_by_sample_id(self, cli):
        clean = _method_entries({"s0": ["a", "b"], "s1": ["c", "d"]})
        pert = _method_entries({"s1": ["c", "d"], "s2": ["e", "f"]})
        pairs = cli.compute_method_jaccard_pairs(clean, pert, k=10)
        assert [p["sample_id"] for p in pairs] == ["s1"]
        assert pairs[0]["method_jaccard"] == pytest.approx(1.0)


class TestSettingSummary:
    def test_summary_metrics(self, cli):
        # s0: 完全一致 (J=1), s1: 半分一致, s2: 不一致
        clean = _method_entries(
            {"s0": ["a", "b"], "s1": ["c", "d"], "s2": ["e", "f"]}
        )
        pert = _method_entries(
            {"s0": ["a", "b"], "s1": ["c", "x"], "s2": ["y", "z"]}
        )
        summary = cli.compute_setting_summary(
            clean, pert, _sample_results(), k=10
        )
        assert summary["n_pairs"] == 3
        assert summary["n_joined"] == 3
        assert summary["mean_j_method"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)
        # J_method は s0 > s1 > s2、R (rouge_f1) は s0 < s1 < s2 → rho = -1
        assert summary["rho_j_method_vs_rouge"]["rho"] == pytest.approx(-1.0)
        # J_RC 参照値 (アーカイブ AttnLRP) も同じ結合行から計算される
        assert summary["rho_j_rc_vs_rouge"]["rho"] == pytest.approx(1.0)
        # clean 側の手法 vs R_C Jaccard@10 (results.json の付帯値) の平均
        assert summary["mean_vs_rc_jaccard_clean"] == pytest.approx(0.5)

    def test_unpaired_samples_are_skipped(self, cli):
        clean = _method_entries({"s0": ["a"], "s1": ["b"]})
        pert = _method_entries({"s0": ["a"]})
        summary = cli.compute_setting_summary(
            clean, pert, _sample_results(), k=10
        )
        assert summary["n_pairs"] == 1
        assert summary["n_joined"] == 1
        assert summary["rho_j_method_vs_rouge"]["rho"] is None  # n<3 では計算しない
