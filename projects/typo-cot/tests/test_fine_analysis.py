"""実験8-fine の集計・判定 (H8f-1〜5) の純関数テスト.

GPU 不要。合成した cell レコード列に対して層プロファイル集計・最良層・
プラトー判定・累積飽和・検証点ヌル・noising 十分性の各判定を検証する。
"""

import pytest

from typo_cot.intervention.fine_analysis import (
    argmax_layer,
    collect_by_layer,
    judge_h8f1_peak_depth,
    judge_h8f2_plateau_vs_spike,
    judge_h8f3_cumulative_saturation,
    judge_h8f4_late_null,
    judge_h8f5_noising_sufficiency,
    mean_ci,
    plateau_layers,
    saturation_layer,
    summarize_by_layer,
)


def _cell(kind, layer, direction, **fields):
    d = {"kind": kind, "layer": layer, "direction": direction}
    d.update(fields)
    return d


def _plateau_single_cells():
    """早期プラトー型 (層1-4 が高原、以降低下) の合成 single cells (2 ペア分)."""
    means = {0: 0.30, 1: 0.85, 2: 0.88, 3: 0.86, 4: 0.84, 5: 0.55,
             6: 0.30, 7: 0.15, 8: 0.10, 9: 0.05, 10: 0.02, 11: 0.01,
             14: 0.0, 20: -0.01, 26: 0.0}
    cells = []
    for rep in (0.0, 0.02):  # 2 サンプル (わずかに違う値)
        for li, m in means.items():
            cells.append(_cell("single", li, "clean_to_pert",
                               s2_kl_recovery=m + rep, answer_matches_donor=(m > 0.5)))
    return cells


class TestCollectByLayer:
    def test_filters_kind_direction_field(self):
        cells = [
            _cell("single", 0, "clean_to_pert", s2_kl_recovery=0.5),
            _cell("single", 0, "clean_to_pert", s2_kl_recovery=0.7),
            _cell("cumulative", 0, "clean_to_pert", s2_kl_recovery=0.9),  # 別 kind
            _cell("single", 0, "pert_to_clean", s2_kl_recovery=0.1),      # 別 direction
            _cell("single", 1, "clean_to_pert", s2_kl_recovery=None),     # None は除外
        ]
        by = collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
        assert by[0] == [0.5, 0.7]
        assert 1 not in by  # None のみ → 空 → キーなし

    def test_missing_field_skipped(self):
        cells = [_cell("single", 3, "clean_to_pert")]  # フィールド無し
        assert collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery") == {}


class TestMeanCi:
    def test_mean_and_interval_contains_mean(self):
        vals = [0.8, 0.82, 0.79, 0.81, 0.83]
        mean, lo, hi = mean_ci(vals, n_boot=500, seed=1)
        assert mean == pytest.approx(sum(vals) / len(vals))
        assert lo <= mean <= hi

    def test_empty_returns_nones(self):
        assert mean_ci([]) == (None, None, None)


class TestSummarizeAndArgmax:
    def test_argmax_picks_peak_layer(self):
        cells = _plateau_single_cells()
        by = collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
        summ = summarize_by_layer(by)
        # 最良層は 2 (mean 0.88 近傍)
        assert argmax_layer(summ) == 2

    def test_argmax_restrict_to_layer_subset(self):
        cells = _plateau_single_cells()
        summ = summarize_by_layer(
            collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
        )
        # 早期帯 0-7 に限定しても 2
        assert argmax_layer(summ, restrict=list(range(8))) == 2


class TestPlateauLayers:
    def test_detects_contiguous_plateau(self):
        cells = _plateau_single_cells()
        summ = summarize_by_layer(
            collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
        )
        best = argmax_layer(summ)
        plat = plateau_layers(summ, best, rel=0.9)
        # 層 1-4 が max の 90% 以上 → プラトー幅 >= 2
        assert set(plat) >= {1, 2, 3, 4}
        assert len(plat) >= 2

    def test_spike_has_isolated_peak(self):
        # 単層スパイク: 層3 だけ高い
        cells = []
        vals = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.9, 4: 0.05, 5: 0.05}
        for li, m in vals.items():
            cells.append(_cell("single", li, "clean_to_pert", s2_kl_recovery=m))
        summ = summarize_by_layer(
            collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
        )
        best = argmax_layer(summ)
        assert best == 3
        assert plateau_layers(summ, best, rel=0.9) == [3]  # 孤立 → スパイク


class TestSaturationLayer:
    def test_first_layer_reaching_fraction_of_max(self):
        # 累積: 早期で急伸、li>=2 で飽和
        cum = {0: 0.4, 1: 0.7, 2: 0.90, 3: 0.95, 4: 0.97, 5: 0.98}
        summ = {li: {"mean": m, "median": m, "n": 5} for li, m in cum.items()}
        # max=0.98, 90% = 0.882 → 最初に到達するのは層2 (0.90 >= 0.882)
        assert saturation_layer(summ, frac=0.9) == 2


class TestJudgments:
    def test_h8f1_peak_depth_below_0p2(self):
        cells = _plateau_single_cells()
        summ = summarize_by_layer(
            collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
        )
        v = judge_h8f1_peak_depth(summ, n_layers=34, restrict_early=list(range(12)))
        assert v["best_layer"] == 2
        assert v["rel_depth"] == pytest.approx(2 / 34)
        assert v["supported"] is True  # 2/34 < 0.2

    def test_h8f2_plateau(self):
        cells = _plateau_single_cells()
        summ = summarize_by_layer(
            collect_by_layer(cells, "single", "clean_to_pert", "s2_kl_recovery")
        )
        best = argmax_layer(summ)
        v = judge_h8f2_plateau_vs_spike(summ, best)
        assert v["shape"] == "plateau"
        assert v["plateau_width"] >= 2

    def test_h8f3_cumulative_saturation(self):
        single = {li: {"mean": m, "median": m, "n": 4} for li, m in
                  {0: 0.3, 1: 0.5, 2: 0.55, 3: 0.4, 4: 0.2}.items()}
        cumulative = {li: {"mean": m, "median": m, "n": 4} for li, m in
                      {0: 0.3, 1: 0.6, 2: 0.75, 3: 0.78, 4: 0.79}.items()}
        v = judge_h8f3_cumulative_saturation(single, cumulative, n_layers=34, frac=0.9)
        assert v["single_max"] == pytest.approx(0.55)
        assert v["cumulative_max"] == pytest.approx(0.79)
        assert v["ratio_cum_over_single"] == pytest.approx(0.79 / 0.55)
        assert v["ratio_cum_over_single"] >= 1.2
        # 飽和層の相対深さ
        assert v["saturation_layer"] == 2
        assert v["supported"] is True

    def test_h8f4_late_null(self):
        # 検証点 14/20/26 が ~0
        summ = {14: {"mean": 0.01, "median": 0.01, "ci_lo": -0.02, "ci_hi": 0.04, "n": 10},
                20: {"mean": -0.01, "median": -0.01, "ci_lo": -0.03, "ci_hi": 0.02, "n": 10},
                26: {"mean": 0.0, "median": 0.0, "ci_lo": -0.02, "ci_hi": 0.02, "n": 10},
                2: {"mean": 0.88, "median": 0.88, "ci_lo": 0.8, "ci_hi": 0.95, "n": 10}}
        v = judge_h8f4_late_null(summ, val_layers=[14, 20, 26], thresh=0.1)
        assert v["supported"] is True
        assert all(abs(x) < 0.1 for x in v["val_means"].values())

    def test_h8f5_noising_sufficiency(self):
        # noising s2_kl_recovery が best±1 で >= 0.5
        summ = {1: {"mean": 0.55, "median": 0.55, "n": 10}, 2: {"mean": 0.62, "median": 0.62, "n": 10},
                3: {"mean": 0.58, "median": 0.58, "n": 10}, 5: {"mean": 0.2, "median": 0.2, "n": 10}}
        v = judge_h8f5_noising_sufficiency(summ, best_layer=2, thresh=0.5)
        assert set(v["layers"]) == {1, 2, 3}
        assert v["supported"] is True
        assert v["mean_at_best"] == pytest.approx(0.62)
