"""Versioned numeric and categorical contract for the rebuttal manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RebuttalSetting:
    """One submitted Table 6 model/task setting and its exact outcome counts."""

    model: str
    task: str
    paper_denominator: int
    paper_successes: int

    @property
    def key(self) -> tuple[str, str]:
        return self.model, self.task

    @property
    def slug(self) -> str:
        return f"{self.model.rsplit('/', 1)[-1]}/{self.task}"


@dataclass(frozen=True, slots=True)
class RebuttalManifestProtocol:
    """Immutable pre-results contract consumed by the builder and its tests."""

    schema_version: str
    target_rules: tuple[str, ...]
    source_seed: int
    requested_edits: int
    source_max_new_tokens: int
    fixed_direction: str
    fixed_window: str
    offset_tokens: int
    window_split_seed: int
    settings: tuple[RebuttalSetting, ...]

    @property
    def restoration_pairs(self) -> int:
        return sum(setting.paper_denominator for setting in self.settings)

    @property
    def fixed_window_successes(self) -> int:
        return sum(setting.paper_successes for setting in self.settings)

    def as_dict(self) -> dict[str, object]:
        """Return the complete JSON-serializable contract for run provenance."""

        return {
            "schema_version": self.schema_version,
            "target_rules": list(self.target_rules),
            "prepared_source": {
                "seed": self.source_seed,
                "requested_edits": self.requested_edits,
                "max_new_tokens": self.source_max_new_tokens,
                "generation_protocol": "explicit-greedy-generation/v1",
                "termination_protocol": "effective-eos-vs-length-cap/v1",
            },
            "fixed_reference": {
                "direction": self.fixed_direction,
                "window": self.fixed_window,
                "restoration_pairs": self.restoration_pairs,
                "successes": self.fixed_window_successes,
            },
            "planning": {
                "offset_tokens": self.offset_tokens,
                "offset_validity": ("all-prompt-interior-non-edited-coordinates/v1"),
                "cross_item_donor": (
                    "task-model-target-rule-aligned-word-count-cyclic-derangement/v1"
                ),
                "window_split_seed": self.window_split_seed,
                "window_split_algorithm": (
                    "sha256-order-first-floor-half-per-model-task-target-rule/v1"
                ),
                "harm_selection": ("all-prepared-clean-correct-typo-correct-aligned-uncapped/v1"),
                "clean_correct_coverage": ("full-plus-patch-eligible-plus-alignment-ineligible/v1"),
                "restoration_selection": "fixed-regenerated-paper-denominator/v1",
                "net_balance_scope": ("repair-harm-conditional-composite-not-full-population/v1"),
                "uncovered_prepared_wrong": "retained-and-reported/v1",
            },
            "settings": [
                {
                    "model": setting.model,
                    "task": setting.task,
                    "paper_denominator": setting.paper_denominator,
                    "paper_successes": setting.paper_successes,
                }
                for setting in self.settings
            ],
        }

    def sha256(self) -> str:
        """Fingerprint the canonical protocol payload."""

        encoded = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


REBUTTAL_MANIFEST_PROTOCOL = RebuttalManifestProtocol(
    schema_version="rebuttal-manifest-protocol/v1",
    target_rules=("attribution-4", "random-4"),
    source_seed=42,
    requested_edits=4,
    source_max_new_tokens=512,
    fixed_direction="clean-to-edited",
    fixed_window="0:6",
    offset_tokens=2,
    window_split_seed=42,
    settings=(
        RebuttalSetting("google/gemma-3-4b-it", "gsm8k", 172, 129),
        RebuttalSetting("meta-llama/Llama-3.2-3B-Instruct", "gsm8k", 232, 161),
        RebuttalSetting("mistralai/Mistral-7B-Instruct-v0.3", "gsm8k", 197, 93),
        RebuttalSetting("google/gemma-3-4b-it", "mmlu", 209, 135),
        RebuttalSetting("meta-llama/Llama-3.2-3B-Instruct", "mmlu", 226, 142),
        RebuttalSetting("mistralai/Mistral-7B-Instruct-v0.3", "mmlu", 205, 140),
    ),
)
REBUTTAL_SETTINGS = REBUTTAL_MANIFEST_PROTOCOL.settings


__all__ = [
    "REBUTTAL_MANIFEST_PROTOCOL",
    "REBUTTAL_SETTINGS",
    "RebuttalManifestProtocol",
    "RebuttalSetting",
]
