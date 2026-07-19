"""実験4: fixed-target 統計 (偏相関・bootstrap・Holm・Δρ) のテスト.

typo_cot.analysis.fixed_stats を検証する。GPU 不要・合成データのみ。
偏相関の規約は analysis/analyzer.py の _compute_partial_correlation
(z を線形回帰で除去した残差同士の Pearson 相関 = 一次偏相関) と同一。
"""

import numpy as np
import pytest

from typo_cot.analysis.fixed_stats import (
    bootstrap_partial_corr_ci,
    cot_jaccard_from_scores,
    format_meta_comparison,
    holm_adjust,
    join_fixed_default_records,
    paired_bootstrap_delta_rho,
    partial_corr_flip,
)


def _synthetic(n=300, seed=0, strength=-3.0):
    """flip が jaccard に依存する合成データ (rouge は独立ノイズ)."""
    rng = np.random.default_rng(seed)
    jaccard = rng.uniform(0, 1, n)
    rouge = rng.uniform(0, 1, n)
    logits = strength * (jaccard - 0.5)
    flip = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logits))).astype(float)
    return jaccard, flip, rouge


class TestPartialCorrFlip:
    def test_matches_pingouin(self):
        """残差 Pearson 方式は pingouin の一次偏相関と一致する."""
        import pandas as pd
        import pingouin as pg

        j, flip, rouge = _synthetic()
        r, p, n = partial_corr_flip(j, flip, rouge)
        df = pd.DataFrame({"x": j, "y": flip, "z": rouge})
        ref = pg.partial_corr(data=df, x="x", y="y", covar="z")
        assert n == len(j)
        assert r == pytest.approx(float(ref["r"].iloc[0]), abs=1e-10)
        p_col = "p_val" if "p_val" in ref.columns else "p-val"
        assert p == pytest.approx(float(ref[p_col].iloc[0]), rel=1e-6)

    def test_negative_correlation_detected(self):
        j, flip, rouge = _synthetic(strength=-3.0)
        r, p, _ = partial_corr_flip(j, flip, rouge)
        assert r < -0.2
        assert p < 0.001

    def test_nan_rows_dropped(self):
        j, flip, rouge = _synthetic(n=50)
        j[0] = np.nan
        rouge[1] = np.inf
        r, p, n = partial_corr_flip(j, flip, rouge)
        assert n == 48
        assert np.isfinite(r)


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        j, flip, rouge = _synthetic()
        r, _, _ = partial_corr_flip(j, flip, rouge)
        lo, hi = bootstrap_partial_corr_ci(j, flip, rouge, n_boot=200, seed=42)
        assert lo < r < hi
        assert -1 <= lo < hi <= 1

    def test_deterministic_given_seed(self):
        j, flip, rouge = _synthetic()
        ci1 = bootstrap_partial_corr_ci(j, flip, rouge, n_boot=100, seed=7)
        ci2 = bootstrap_partial_corr_ci(j, flip, rouge, n_boot=100, seed=7)
        assert ci1 == ci2


class TestHolm:
    def test_known_example(self):
        # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone [0.03, 0.06, 0.06]
        adj = holm_adjust([0.01, 0.04, 0.03])
        assert adj[0] == pytest.approx(0.03)
        assert adj[1] == pytest.approx(0.06)
        assert adj[2] == pytest.approx(0.06)

    def test_known_example_four(self):
        pvals = [0.2, 0.005, 0.05, 0.0001]
        adj = holm_adjust(pvals)
        # sorted: 0.0001*4=0.0004, 0.005*3=0.015, 0.05*2=0.1, 0.2*1=0.2
        assert adj == pytest.approx([0.2, 0.015, 0.1, 0.0004])

    def test_capped_at_one(self):
        adj = holm_adjust([0.9, 0.8])
        assert all(a <= 1.0 for a in adj)


class TestPairedBootstrapDeltaRho:
    def test_identical_metrics_give_zero_delta(self):
        j, flip, rouge = _synthetic()
        res = paired_bootstrap_delta_rho(
            j, j, flip, rouge, n_boot=100, seed=3
        )
        assert res["delta_rho"] == pytest.approx(0.0)
        assert res["p_value"] == pytest.approx(1.0)

    def test_attenuation_detected(self):
        """default が強相関・fixed が無相関なら Δρ>0 が有意に出る."""
        rng = np.random.default_rng(1)
        j_default, flip, rouge = _synthetic(n=500, strength=-4.0)
        j_fixed = rng.uniform(0, 1, 500)  # 無相関
        res = paired_bootstrap_delta_rho(
            j_default, j_fixed, flip, rouge, n_boot=300, seed=5
        )
        assert res["rho_default"] < -0.3
        assert abs(res["rho_fixed"]) < 0.15
        assert res["delta_rho"] > 0.2
        assert res["p_value"] < 0.05
        lo, hi = res["ci95"]
        assert lo < res["delta_rho"] < hi

    def test_deterministic_given_seed(self):
        j, flip, rouge = _synthetic(n=100)
        j2 = np.clip(j + 0.1, 0, 1)
        r1 = paired_bootstrap_delta_rho(j, j2, flip, rouge, n_boot=50, seed=9)
        r2 = paired_bootstrap_delta_rho(j, j2, flip, rouge, n_boot=50, seed=9)
        assert r1 == r2


class TestCotJaccardFromScores:
    def test_matches_metrics_module(self):
        """token_scores ペアからの Jaccard@k は top_k_jaccard_by_token と一致する."""
        from typo_cot.analysis.metrics import top_k_jaccard_by_token

        clean = [("a", 1.0), ("b", 0.9), ("c", 0.5), ("d", 0.1)]
        other = [("a", 0.8), ("x", 0.7), ("c", 0.6), ("y", 0.2)]
        out = cot_jaccard_from_scores(clean, other, ks=(2, 3))
        for k in (2, 3):
            t1, s1 = zip(*clean, strict=True)
            t2, s2 = zip(*other, strict=True)
            assert out[f"top{k}"] == pytest.approx(
                top_k_jaccard_by_token(t1, s1, t2, s2, k=k)
            )

    def test_empty_returns_zero(self):
        out = cot_jaccard_from_scores([], [("a", 1.0)], ks=(5,))
        assert out["top5"] == 0.0


class TestJoinFixedDefaultRecords:
    """full_results.json の sample_results (default/fixed) を結合して配列化する."""

    def _rec(self, sid, flip, j10, rouge):
        return {
            "sample_id": sid,
            "answer_changed": flip,
            "cot_metrics": {
                "jaccard": {"top5": j10, "top10": j10, "top20": j10},
                "rouge_l": {"f1": rouge},
            },
        }

    def test_join_aligned(self):
        default = [self._rec("a", True, 0.2, 0.5), self._rec("b", False, 0.9, 0.8)]
        fixed = [self._rec("b", False, 0.95, 0.81), self._rec("a", True, 0.6, 0.5)]
        out = join_fixed_default_records(default, fixed, k="top10")
        assert out["sample_ids"] == ["a", "b"]
        assert list(out["j_default"]) == [0.2, 0.9]
        assert list(out["j_fixed"]) == [0.6, 0.95]
        assert list(out["flip"]) == [1.0, 0.0]
        assert list(out["rouge_default"]) == [0.5, 0.8]
        assert list(out["rouge_fixed"]) == [0.5, 0.81]

    def test_missing_ids_dropped(self):
        default = [self._rec("a", True, 0.2, 0.5), self._rec("c", False, 0.7, 0.6)]
        fixed = [self._rec("a", True, 0.6, 0.5)]
        out = join_fixed_default_records(default, fixed, k="top10")
        assert out["sample_ids"] == ["a"]
        assert out["n"] == 1


class TestDeltaRhoOwnRouge:
    def test_rouge_fixed_used_for_fixed_side(self):
        """rouge_fixed を渡すと fixed 側の偏相関はそれで統制される."""
        j, flip, rouge = _synthetic(n=200, seed=2)
        rng = np.random.default_rng(4)
        rouge_fixed = rng.uniform(0, 1, 200)
        res = paired_bootstrap_delta_rho(
            j, j, flip, rouge, rouge_fixed=rouge_fixed, n_boot=50, seed=1
        )
        r_def, _, _ = partial_corr_flip(j, flip, rouge)
        r_fix, _, _ = partial_corr_flip(j, flip, rouge_fixed)
        assert res["rho_default"] == pytest.approx(r_def, abs=1e-12)
        assert res["rho_fixed"] == pytest.approx(r_fix, abs=1e-12)


class TestFormatMetaComparison:
    """自由記述 vs 多肢選択の Δρ メタ比較 (設定レベル permutation 検定)."""

    def _rows(self, free_vals, mc_vals, k="top10"):
        rows = []
        for i, v in enumerate(free_vals):
            rows.append({"setting": f"free{i}", "k": k, "delta_rho": v})
        for i, v in enumerate(mc_vals):
            rows.append({"setting": f"mc{i}", "k": k, "delta_rho": v})
        fmt = {f"free{i}": "free" for i in range(len(free_vals))}
        fmt |= {f"mc{i}": "mc" for i in range(len(mc_vals))}
        return rows, fmt

    def test_group_means_and_diff(self):
        rows, fmt = self._rows([0.0, 0.02], [0.3, 0.5], k="top10")
        res = format_meta_comparison(rows, fmt, k="top10", n_perm=200, seed=0)
        assert res["mean_delta_free"] == pytest.approx(0.01)
        assert res["mean_delta_mc"] == pytest.approx(0.4)
        assert res["diff_mc_minus_free"] == pytest.approx(0.39)
        assert res["n_free"] == 2
        assert res["n_mc"] == 2

    def test_separated_groups_low_p(self):
        rows, fmt = self._rows(
            [0.0, 0.01, -0.01, 0.02, 0.0, 0.01],
            [0.3, 0.4, 0.35, 0.5, 0.45, 0.38],
        )
        res = format_meta_comparison(rows, fmt, k="top10", n_perm=2000, seed=1)
        assert res["p_value"] < 0.05

    def test_deterministic(self):
        rows, fmt = self._rows([0.0, 0.1], [0.2, 0.3])
        r1 = format_meta_comparison(rows, fmt, k="top10", n_perm=100, seed=5)
        r2 = format_meta_comparison(rows, fmt, k="top10", n_perm=100, seed=5)
        assert r1 == r2

    def test_unmapped_settings_ignored(self):
        rows, fmt = self._rows([0.0], [0.2])
        rows.append({"setting": "unknown", "k": "top10", "delta_rho": 9.9})
        res = format_meta_comparison(rows, fmt, k="top10", n_perm=50, seed=2)
        assert res["n_free"] == 1 and res["n_mc"] == 1
