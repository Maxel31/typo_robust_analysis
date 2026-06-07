"""TDD tests for M0 再現ゲート②: seed/definition stability (stability.py).

All tests run on CPU with synthetic numpy data -- no model loading required.

Covers:
- jaccard: hand values (identical=1.0, disjoint=0.0, partial known)
- layer_distribution: count of selected neurons per layer
- spearman_rank_correlation: vs known values incl. ties (hand-computed)
- stability_report: on 3 synthetic NeuronMasks
- stability_gate_decision: pass/fail by thresholds
"""
from __future__ import annotations

import pytest
import numpy as np

from quant_typo_neuron.neuron_identification.stability import (
    jaccard,
    layer_distribution,
    spearman_rank_correlation,
    stability_report,
    stability_gate_decision,
)
from typo_utils.neurons import NeuronMask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask(data: dict[int, list[int]]) -> NeuronMask:
    """Cast to NeuronMask (dict[int, list[int]])."""
    return {int(k): list(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_masks_return_one(self):
        """Identical masks -> Jaccard = 1.0."""
        mask = _mask({0: [1, 2, 3], 1: [0, 4]})
        assert jaccard(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks_return_zero(self):
        """Completely disjoint masks -> Jaccard = 0.0."""
        mask_a = _mask({0: [0, 1], 1: [2]})
        mask_b = _mask({0: [3, 4], 1: [5]})
        assert jaccard(mask_a, mask_b) == pytest.approx(0.0)

    def test_partial_overlap_known_value(self):
        """Hand-computed: |intersection|=2, |union|=4 -> Jaccard=0.5."""
        # mask_a neurons: (0,0),(0,1),(1,2)  -- 3 pairs
        # mask_b neurons: (0,1),(1,2),(1,3)  -- 3 pairs
        # intersection: (0,1),(1,2) -> 2
        # union: (0,0),(0,1),(1,2),(1,3) -> 4
        # jaccard = 2/4 = 0.5
        mask_a = _mask({0: [0, 1], 1: [2]})
        mask_b = _mask({0: [1], 1: [2, 3]})
        assert jaccard(mask_a, mask_b) == pytest.approx(0.5)

    def test_one_empty_mask_returns_zero(self):
        """If one mask is empty (no neurons), Jaccard = 0.0."""
        mask_a = _mask({0: [1, 2]})
        mask_b = _mask({0: [], 1: []})
        assert jaccard(mask_a, mask_b) == pytest.approx(0.0)

    def test_both_empty_masks_return_one(self):
        """Two empty masks are identical -> Jaccard = 1.0 (0/0 -> 1 by convention)."""
        mask_a: NeuronMask = {}
        mask_b: NeuronMask = {}
        assert jaccard(mask_a, mask_b) == pytest.approx(1.0)

    def test_symmetric(self):
        """Jaccard is symmetric."""
        mask_a = _mask({0: [0, 2, 4], 1: [1]})
        mask_b = _mask({0: [0, 3], 1: [1, 5]})
        assert jaccard(mask_a, mask_b) == pytest.approx(jaccard(mask_b, mask_a))

    def test_partial_overlap_different_layers(self):
        """Neurons in different layers don't intersect even if dim is same."""
        # mask_a: (0,1),(1,2)   mask_b: (0,2),(1,1)
        # intersection: {} -> 0 , union -> 4, jaccard = 0.0
        mask_a = _mask({0: [1], 1: [2]})
        mask_b = _mask({0: [2], 1: [1]})
        assert jaccard(mask_a, mask_b) == pytest.approx(0.0)

    def test_known_three_quarter_overlap(self):
        """Hand-computed: |intersection|=3, |union|=4 -> Jaccard=0.75."""
        # mask_a: (0,0),(0,1),(0,2)  3 neurons
        # mask_b: (0,0),(0,1),(0,2),(0,3)  4 neurons
        # intersection: 3, union: 4, jaccard = 0.75
        mask_a = _mask({0: [0, 1, 2]})
        mask_b = _mask({0: [0, 1, 2, 3]})
        assert jaccard(mask_a, mask_b) == pytest.approx(0.75)

    def test_return_type_is_float(self):
        mask = _mask({0: [0, 1]})
        result = jaccard(mask, mask)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# layer_distribution
# ---------------------------------------------------------------------------


class TestLayerDistribution:
    def test_basic_counts(self):
        """Count neurons per layer, in order."""
        mask = _mask({0: [1, 3, 5], 1: [0], 2: [4, 7]})
        dist = layer_distribution(mask, num_layers=3)
        assert dist == [3, 1, 2]

    def test_empty_layer_counts_zero(self):
        """Layers with no neurons -> count 0."""
        mask = _mask({0: [0, 1], 2: [0]})
        dist = layer_distribution(mask, num_layers=4)
        assert dist == [2, 0, 1, 0]

    def test_all_empty(self):
        """No neurons in any layer -> all zeros."""
        mask: NeuronMask = {}
        dist = layer_distribution(mask, num_layers=3)
        assert dist == [0, 0, 0]

    def test_length_equals_num_layers(self):
        """Output length always equals num_layers."""
        mask = _mask({0: [0, 1]})
        for n in [1, 2, 5, 10]:
            dist = layer_distribution(mask, num_layers=n)
            assert len(dist) == n

    def test_all_layers_present(self):
        """All layers with neurons accounted for."""
        mask = _mask({i: list(range(i + 1)) for i in range(4)})
        dist = layer_distribution(mask, num_layers=4)
        assert dist == [1, 2, 3, 4]

    def test_returns_list_of_int(self):
        mask = _mask({0: [0, 1, 2]})
        dist = layer_distribution(mask, num_layers=2)
        assert isinstance(dist, list)
        assert all(isinstance(v, int) for v in dist)


# ---------------------------------------------------------------------------
# spearman_rank_correlation
# ---------------------------------------------------------------------------


class TestSpearmanRankCorrelation:
    def test_perfect_positive_correlation(self):
        """Identical sequences -> rho = 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert spearman_rank_correlation(x, x) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation(self):
        """Reversed sequences -> rho = -1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert spearman_rank_correlation(x, y) == pytest.approx(-1.0, abs=1e-9)

    def test_no_correlation_known_value(self):
        """Hand-computed: x=[1,2,3,4,5] y=[3,4,5,1,2].

        ranks_x = [1,2,3,4,5]
        ranks_y = [3,4,5,1,2]
        d = [-2,-2,-2,3,3]
        d^2 = [4,4,4,9,9], sum_d2 = 30
        rho = 1 - 6*30/(5*(25-1)) = 1 - 180/120 = -0.5
        """
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 4.0, 5.0, 1.0, 2.0]
        assert spearman_rank_correlation(x, y) == pytest.approx(-0.5, abs=1e-9)

    def test_ties_average_ranks(self):
        """Ties handled by average ranks (Pearson correlation on ranked data).

        x = [1, 2, 2, 4]
        ranks_x: 1->1.0, 2,2->(2+3)/2=2.5, 4->4.0  => [1.0,2.5,2.5,4.0]
        y = [3, 1, 2, 4]
        ranks_y: 1->1.0, 2->2.0, 3->3.0, 4->4.0  => [3.0,1.0,2.0,4.0]
        mean_rx=2.5, mean_ry=2.5
        dx=[-1.5,0,0,1.5], dy=[0.5,-1.5,-0.5,1.5]
        sum(dx*dy)=1.5, sum(dx^2)=4.5, sum(dy^2)=5.0
        rho = 1.5/sqrt(4.5*5.0) = 1.5/sqrt(22.5) = sqrt(0.1) ~= 0.31623
        NOTE: simplified Spearman formula 1-6*sum_d2/(n*(n^2-1)) gives 0.35
              but is only exact without ties; Pearson on ranks is the correct form.
        """
        x = [1.0, 2.0, 2.0, 4.0]
        y = [3.0, 1.0, 2.0, 4.0]
        import math
        expected = 1.5 / math.sqrt(22.5)  # = sqrt(0.1) ~= 0.31623
        assert spearman_rank_correlation(x, y) == pytest.approx(expected, abs=1e-9)

    def test_all_ties_in_x(self):
        """All equal in x -> rho = 0.0 (no variation, so no correlation)."""
        x = [5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0]
        result = spearman_rank_correlation(x, y)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_two_elements_positive(self):
        """n=2, positive order -> rho = 1.0."""
        assert spearman_rank_correlation([1.0, 2.0], [3.0, 9.0]) == pytest.approx(1.0)

    def test_two_elements_negative(self):
        """n=2, negative order -> rho = -1.0."""
        assert spearman_rank_correlation([2.0, 1.0], [3.0, 9.0]) == pytest.approx(-1.0)

    def test_accepts_numpy_arrays(self):
        """Accepts numpy arrays as input."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0])
        assert spearman_rank_correlation(x, y) == pytest.approx(-1.0)

    def test_known_positive_partial_correlation(self):
        """x=[1,2,3,4,5] y=[2,1,4,3,5].

        ranks_x = [1,2,3,4,5]
        ranks_y: [2,1,4,3,5]
        d = [-1,1,-1,1,0], d^2 = [1,1,1,1,0], sum_d2 = 4
        rho = 1 - 6*4/(5*24) = 1 - 24/120 = 0.8
        """
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 1.0, 4.0, 3.0, 5.0]
        assert spearman_rank_correlation(x, y) == pytest.approx(0.8, abs=1e-9)

    def test_return_type_is_float(self):
        result = spearman_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# stability_report
# ---------------------------------------------------------------------------


class TestStabilityReport:
    @pytest.fixture()
    def three_masks(self) -> list[NeuronMask]:
        """3 synthetic NeuronMasks: 2 layers, 10 dims each."""
        mask_0 = _mask({0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})
        mask_1 = _mask({0: [0, 1, 2, 3], 1: [5, 6, 7, 8, 9]})
        mask_2 = _mask({0: [2, 3, 4, 5], 1: [6, 7, 8, 9]})
        return [mask_0, mask_1, mask_2]

    def test_report_has_required_keys(self, three_masks):
        """stability_report must return dict with mean_jaccard and mean_spearman keys."""
        report = stability_report(three_masks)
        assert "mean_jaccard" in report
        assert "mean_spearman" in report

    def test_mean_jaccard_in_range(self, three_masks):
        """mean pairwise Jaccard must be in [0, 1]."""
        report = stability_report(three_masks)
        assert 0.0 <= report["mean_jaccard"] <= 1.0

    def test_mean_spearman_in_range(self, three_masks):
        """mean pairwise Spearman rho must be in [-1, 1]."""
        report = stability_report(three_masks)
        assert -1.0 <= report["mean_spearman"] <= 1.0

    def test_identical_masks_report_perfect_scores(self):
        """3 identical masks -> mean_jaccard=1.0, mean_spearman=1.0."""
        mask = _mask({0: [0, 1, 2], 1: [3, 4]})
        masks = [mask, mask, mask]
        report = stability_report(masks)
        assert report["mean_jaccard"] == pytest.approx(1.0)
        assert report["mean_spearman"] == pytest.approx(1.0)

    def test_pairwise_count_with_three_masks(self, three_masks):
        """With 3 masks there are C(3,2)=3 pairs; report should aggregate them."""
        report = stability_report(three_masks)
        masks = three_masks
        # Determine num_layers from the masks
        num_layers = max(max(m.keys()) for m in masks if m) + 1
        pairs = [(0, 1), (0, 2), (1, 2)]
        jaccards = [jaccard(masks[i], masks[j]) for i, j in pairs]
        spearmans = [
            spearman_rank_correlation(
                layer_distribution(masks[i], num_layers),
                layer_distribution(masks[j], num_layers),
            )
            for i, j in pairs
        ]
        expected_j = float(np.mean(jaccards))
        expected_s = float(np.mean(spearmans))
        assert report["mean_jaccard"] == pytest.approx(expected_j, abs=1e-9)
        assert report["mean_spearman"] == pytest.approx(expected_s, abs=1e-9)

    def test_two_masks_uses_single_pair(self):
        """With 2 masks -> exactly 1 pair, mean = the single pair value."""
        mask_a = _mask({0: [0, 1], 1: [2, 3]})
        mask_b = _mask({0: [0, 2], 1: [2, 4]})
        report = stability_report([mask_a, mask_b])
        expected_j = jaccard(mask_a, mask_b)
        assert report["mean_jaccard"] == pytest.approx(expected_j, abs=1e-9)

    def test_single_mask_raises_value_error(self):
        """With fewer than 2 masks, raise ValueError (no pairs to compute)."""
        with pytest.raises(ValueError):
            stability_report([_mask({0: [1, 2]})])

    def test_report_values_are_floats(self, three_masks):
        report = stability_report(three_masks)
        assert isinstance(report["mean_jaccard"], float)
        assert isinstance(report["mean_spearman"], float)

    def test_returns_dict(self, three_masks):
        report = stability_report(three_masks)
        assert isinstance(report, dict)


# ---------------------------------------------------------------------------
# stability_gate_decision
# ---------------------------------------------------------------------------


class TestStabilityGateDecision:
    @pytest.fixture()
    def passing_report(self) -> dict:
        """Report that exceeds both thresholds."""
        return {"mean_jaccard": 0.75, "mean_spearman": 0.85}

    @pytest.fixture()
    def failing_report_jaccard(self) -> dict:
        """Report that fails only on Jaccard."""
        return {"mean_jaccard": 0.40, "mean_spearman": 0.85}

    @pytest.fixture()
    def failing_report_spearman(self) -> dict:
        """Report that fails only on Spearman."""
        return {"mean_jaccard": 0.75, "mean_spearman": 0.30}

    @pytest.fixture()
    def failing_report_both(self) -> dict:
        """Report that fails both thresholds."""
        return {"mean_jaccard": 0.30, "mean_spearman": 0.20}

    def test_passes_when_both_above_threshold(self, passing_report):
        decision = stability_gate_decision(
            passing_report, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert decision["passed"] is True

    def test_fails_when_jaccard_below_threshold(self, failing_report_jaccard):
        decision = stability_gate_decision(
            failing_report_jaccard, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert decision["passed"] is False

    def test_fails_when_spearman_below_threshold(self, failing_report_spearman):
        decision = stability_gate_decision(
            failing_report_spearman, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert decision["passed"] is False

    def test_fails_when_both_below_threshold(self, failing_report_both):
        decision = stability_gate_decision(
            failing_report_both, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert decision["passed"] is False

    def test_passed_key_is_bool(self, passing_report):
        decision = stability_gate_decision(
            passing_report, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert isinstance(decision["passed"], bool)

    def test_decision_contains_passed_key(self, passing_report):
        decision = stability_gate_decision(
            passing_report, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert "passed" in decision

    def test_decision_includes_observed_values(self, passing_report):
        """Decision dict should carry observed values to diagnose failures."""
        decision = stability_gate_decision(
            passing_report, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert "mean_jaccard" in decision
        assert "mean_spearman" in decision

    def test_exact_boundary_jaccard_passes(self):
        """Exactly at threshold -> passed (>=)."""
        report = {"mean_jaccard": 0.5, "mean_spearman": 0.8}
        decision = stability_gate_decision(report, min_jaccard=0.5, min_rank_corr=0.7)
        assert decision["passed"] is True

    def test_exact_boundary_spearman_passes(self):
        """Exactly at threshold -> passed (>=)."""
        report = {"mean_jaccard": 0.6, "mean_spearman": 0.7}
        decision = stability_gate_decision(report, min_jaccard=0.5, min_rank_corr=0.7)
        assert decision["passed"] is True

    def test_returns_dict(self, passing_report):
        decision = stability_gate_decision(
            passing_report, min_jaccard=0.5, min_rank_corr=0.7
        )
        assert isinstance(decision, dict)
