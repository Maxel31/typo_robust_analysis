"""Shared strict source adapters for the one-token diagnostic.

The final PDF states that clean-prefix and one-token diagnostics reuse the
same source cohorts.  This module intentionally delegates to the reviewed
clean-prefix source adapter instead of creating a second parser or accepting
opaque legacy target manifests.
"""

from typo_cot.experiments.clean_prefix_scan.source import (
    SourceBundle,
    SourceCase,
    SourceCohort,
    SourceFileSnapshot,
    load_source_bundle,
    validate_source_snapshot,
)

__all__ = [
    "SourceBundle",
    "SourceCase",
    "SourceCohort",
    "SourceFileSnapshot",
    "load_source_bundle",
    "validate_source_snapshot",
]
