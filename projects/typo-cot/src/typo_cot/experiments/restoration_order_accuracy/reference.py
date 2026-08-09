"""Rounded values printed in the final PDF for Table 13."""

from __future__ import annotations

PAPER_PUBLISHED_REFERENCE = {
    "role": "final-pdf-rounded-reference-not-fresh-run-acceptance-target",
    "cohort_items": 1582,
    "accuracy_percent": {
        "edited:k0": 12.0,
        "high-relevance-first:k1": 42.3,
        "high-relevance-first:k2": 56.1,
        "high-relevance-first:k3": 68.0,
        "seeded-random:k1": 37.9,
        "seeded-random:k2": 52.0,
        "seeded-random:k3": 64.8,
        "low-relevance-first:k1": 37.0,
        "low-relevance-first:k2": 48.9,
        "low-relevance-first:k3": 59.9,
        "clean:k4": 88.9,
    },
    "paired_high_vs_random_p_value": {
        "k1": 0.0018,
        "k2": 0.0053,
        "k3": 0.019,
    },
}

__all__ = ["PAPER_PUBLISHED_REFERENCE"]
