"""実験6-(iv) 集計 CLI (scripts/analyze_loo_rankings.py) のテスト.

LOO 版 CoT:Jaccard@10 (clean-LOO vs perturbed-LOO) をアーカイブの
サンプル別指標 (cot_rouge_l.f1 = R, cot_jaccard.top10 = J_RC 参照値) と
sample_id で結合し、ρ(J_LOO@10 | R) を Spearman で計算する経路を検証する。
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
    return _load("analyze_loo_rankings")


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


class TestJoinPairsWithArchive:
    def test_joins_on_sample_id(self, cli):
        pairs = [
            {"sample_id": "s1", "loo_jaccard": 0.4},
            {"sample_id": "s3", "loo_jaccard": 0.8},
        ]
        rows = cli.join_pairs_with_archive(pairs, _sample_results())
        assert [r["sample_id"] for r in rows] == ["s1", "s3"]
        assert rows[0]["j_loo"] == pytest.approx(0.4)
        assert rows[0]["rouge_f1"] == pytest.approx(0.6)
        assert rows[0]["j_rc"] == pytest.approx(0.3)

    def test_missing_archive_sample_is_dropped(self, cli):
        pairs = [
            {"sample_id": "s1", "loo_jaccard": 0.4},
            {"sample_id": "zzz", "loo_jaccard": 0.9},
        ]
        rows = cli.join_pairs_with_archive(pairs, _sample_results())
        assert [r["sample_id"] for r in rows] == ["s1"]

    def test_empty_pairs(self, cli):
        assert cli.join_pairs_with_archive([], _sample_results()) == []


class TestSettingSummary:
    def _loo_entries(self, rankings: dict[str, list[str]]):
        """run_loo_scoring.py の results.json 形式 (word のみ意味を持つ合成値)."""
        return [
            {
                "sample_id": sid,
                "loo_word_scores": [
                    {"word": w, "score": float(len(words) - i)}
                    for i, w in enumerate(words)
                ],
            }
            for sid, words in rankings.items()
        ]

    def test_summary_metrics(self, cli):
        # s0: 完全一致 (J=1), s1: 半分一致, s2: 不一致
        clean = self._loo_entries(
            {"s0": ["a", "b"], "s1": ["c", "d"], "s2": ["e", "f"]}
        )
        pert = self._loo_entries(
            {"s0": ["a", "b"], "s1": ["c", "x"], "s2": ["y", "z"]}
        )
        summary = cli.compute_setting_summary(
            clean, pert, _sample_results(), k=10
        )
        assert summary["n_loo_pairs"] == 3
        assert summary["n_joined"] == 3
        assert summary["mean_j_loo"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)
        # J_LOO は s0 > s1 > s2、R (rouge_f1) は s0 < s1 < s2 → rho = -1
        assert summary["rho_j_loo_vs_rouge"]["rho"] == pytest.approx(-1.0)
        # J_RC 参照値も同じ結合行から計算される
        assert summary["rho_j_rc_vs_rouge"]["rho"] == pytest.approx(1.0)
        assert summary["mean_j_rc_joined"] == pytest.approx(0.3)

    def test_unpaired_samples_are_skipped(self, cli):
        clean = self._loo_entries({"s0": ["a"], "s1": ["b"]})
        pert = self._loo_entries({"s0": ["a"]})
        summary = cli.compute_setting_summary(
            clean, pert, _sample_results(), k=10
        )
        assert summary["n_loo_pairs"] == 1
        assert summary["n_joined"] == 1
        assert summary["rho_j_loo_vs_rouge"]["rho"] is None  # n<3 では計算しない
