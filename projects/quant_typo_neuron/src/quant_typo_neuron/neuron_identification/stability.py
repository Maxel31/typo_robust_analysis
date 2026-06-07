"""M0 再現ゲート②: seed/definition stability metrics.

Contract (README §5, M0 stability gate):
- jaccard(mask_a, mask_b) -> float
    Jaccard similarity over the set of (layer, dim) NeuronIndex pairs.
- layer_distribution(mask, num_layers) -> list[int]
    Count of selected neurons per layer (length = num_layers).
- spearman_rank_correlation(x, y) -> float
    Spearman rho implemented from scratch with numpy; handles ties via average ranks.
- stability_report(masks) -> dict
    Given a list of NeuronMasks, compute mean pairwise Jaccard and mean pairwise
    layer-distribution Spearman rho.
- stability_gate_decision(report, *, min_jaccard, min_rank_corr) -> dict
    {passed: bool, mean_jaccard: float, mean_spearman: float, ...}
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from typo_utils.neurons import NeuronMask


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


def jaccard(mask_a: NeuronMask, mask_b: NeuronMask) -> float:
    """Jaccard similarity over the set of (layer, dim) NeuronIndex pairs.

    Args:
        mask_a: NeuronMask = dict[int, list[int]] (layer -> selected dims).
        mask_b: NeuronMask = dict[int, list[int]].

    Returns:
        Jaccard similarity in [0, 1]. Returns 1.0 if both masks are empty
        (convention: two empty sets are identical). Returns 0.0 if one is
        empty and the other is not.
    """
    # Build sets of (layer, dim) tuples
    set_a: set[tuple[int, int]] = {
        (layer, dim)
        for layer, dims in mask_a.items()
        for dim in dims
    }
    set_b: set[tuple[int, int]] = {
        (layer, dim)
        for layer, dims in mask_b.items()
        for dim in dims
    }

    # Both empty -> identical by convention
    if not set_a and not set_b:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 1.0  # defensive branch (already covered above)

    return float(intersection / union)


# ---------------------------------------------------------------------------
# layer_distribution
# ---------------------------------------------------------------------------


def layer_distribution(mask: NeuronMask, num_layers: int) -> list[int]:
    """Count of selected neurons per layer.

    Args:
        mask: NeuronMask = dict[int, list[int]].
        num_layers: total number of layers in the model.

    Returns:
        list[int] of length num_layers; entry i = number of selected neurons
        in layer i. Layers absent from the mask contribute 0.
    """
    counts = [0] * num_layers
    for layer, dims in mask.items():
        if 0 <= layer < num_layers:
            counts[layer] = int(len(dims))
    return counts


# ---------------------------------------------------------------------------
# spearman_rank_correlation
# ---------------------------------------------------------------------------


def _rank_with_average_ties(arr: np.ndarray) -> np.ndarray:
    """Compute ranks with ties broken by the average rank (1-based).

    Args:
        arr: 1-D numpy array.

    Returns:
        Array of ranks (floats) of same length, where tied values receive
        the average of the ranks they would have occupied.
    """
    n = len(arr)
    # argsort gives indices that would sort arr ascending
    sorter = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[sorter] = np.arange(1, n + 1, dtype=float)

    # Identify ties: where consecutive sorted values are equal
    sorted_arr = arr[sorter]
    i = 0
    while i < n:
        j = i
        # Find the end of the run of equal values
        while j < n - 1 and sorted_arr[j] == sorted_arr[j + 1]:
            j += 1
        if j > i:
            # Ties: assign average rank to all tied positions
            avg_rank = float(i + 1 + j + 1) / 2.0  # average of 1-based ranks i+1..j+1
            for k in range(i, j + 1):
                ranks[sorter[k]] = avg_rank
        i = j + 1

    return ranks


def spearman_rank_correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """Spearman rank correlation coefficient (rho) implemented from scratch.

    Handles ties via average ranks (same as scipy.stats.spearmanr).
    Returns 0.0 when either sequence has zero variance (all ties).

    Args:
        x: First sequence of values (list or numpy array).
        y: Second sequence of values, same length as x.

    Returns:
        Spearman rho in [-1, 1] as a float.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length.")

    n = len(x_arr)
    if n < 2:
        raise ValueError("Need at least 2 elements to compute Spearman rho.")

    rx = _rank_with_average_ties(x_arr)
    ry = _rank_with_average_ties(y_arr)

    # Pearson correlation on the ranks
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    dx = rx - rx_mean
    dy = ry - ry_mean

    denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
    if denom == 0.0:
        # Zero variance in at least one variable -> no correlation
        return 0.0

    return float(np.sum(dx * dy) / denom)


# ---------------------------------------------------------------------------
# stability_report
# ---------------------------------------------------------------------------


def stability_report(masks: list[NeuronMask]) -> dict:
    """Compute mean pairwise Jaccard and mean pairwise layer-distribution Spearman.

    Args:
        masks: list of NeuronMasks (e.g., one per seed or per responsibility
               definition). Must have at least 2 masks.

    Returns:
        dict with keys:
            "mean_jaccard"  (float): mean of all C(n,2) pairwise Jaccard values.
            "mean_spearman" (float): mean of all C(n,2) pairwise Spearman rho values
                computed over layer_distribution vectors.
            "num_masks"     (int): number of masks supplied.
            "num_pairs"     (int): number of pairs evaluated (= C(n,2)).

    Raises:
        ValueError: if fewer than 2 masks are provided.
    """
    if len(masks) < 2:
        raise ValueError(
            "stability_report requires at least 2 masks; "
            f"got {len(masks)}."
        )

    # Determine the number of layers from the maximum layer key across all masks
    all_layer_keys = [k for m in masks for k in m.keys()]
    num_layers = (max(all_layer_keys) + 1) if all_layer_keys else 1

    pair_indices = list(combinations(range(len(masks)), 2))

    jaccards: list[float] = []
    spearmans: list[float] = []

    for i, j in pair_indices:
        jaccards.append(jaccard(masks[i], masks[j]))
        dist_i = layer_distribution(masks[i], num_layers)
        dist_j = layer_distribution(masks[j], num_layers)
        spearmans.append(spearman_rank_correlation(dist_i, dist_j))

    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "mean_spearman": float(np.mean(spearmans)),
        "num_masks": len(masks),
        "num_pairs": len(pair_indices),
    }


# ---------------------------------------------------------------------------
# stability_gate_decision
# ---------------------------------------------------------------------------


def stability_gate_decision(
    report: dict,
    *,
    min_jaccard: float,
    min_rank_corr: float,
) -> dict:
    """Determine whether the stability gate is passed.

    Args:
        report: dict as returned by stability_report (must contain
                "mean_jaccard" and "mean_spearman").
        min_jaccard: minimum required mean pairwise Jaccard (inclusive).
        min_rank_corr: minimum required mean pairwise Spearman rho (inclusive).

    Returns:
        dict with keys:
            "passed"        (bool): True iff both criteria are met.
            "mean_jaccard"  (float): observed mean Jaccard (from report).
            "mean_spearman" (float): observed mean Spearman rho (from report).
            "min_jaccard"   (float): threshold used.
            "min_rank_corr" (float): threshold used.
    """
    observed_j = float(report["mean_jaccard"])
    observed_s = float(report["mean_spearman"])

    passed = (observed_j >= min_jaccard) and (observed_s >= min_rank_corr)

    return {
        "passed": bool(passed),
        "mean_jaccard": observed_j,
        "mean_spearman": observed_s,
        "min_jaccard": float(min_jaccard),
        "min_rank_corr": float(min_rank_corr),
    }


__all__ = [
    "jaccard",
    "layer_distribution",
    "spearman_rank_correlation",
    "stability_report",
    "stability_gate_decision",
]
