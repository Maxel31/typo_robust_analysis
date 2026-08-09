"""Pure token planning and deterministic sampling for clean-prefix scans."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


RELATIVE_BUDGETS = (
    0.0,
    0.02,
    0.05,
    0.08,
    0.12,
    0.16,
    0.2,
    0.25,
    0.325,
    0.4,
    0.5,
    0.65,
    0.8,
    1.0,
)
ABSOLUTE_BUDGETS = (1, 2, 4, 8, 16, 32, 64)

_MIN_CLEAN_COT_TOKENS = 8
_MAX_CLEAN_COT_TOKENS = 512
_DEFAULT_TARGETING_CAP = 400


def _token_ids(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    try:
        token_ids = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of token IDs") from exc
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError(f"{field} must contain only non-negative integer token IDs")
    return token_ids


def _positive_int(value: int, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class PrefixBudget:
    """One distinct integer prefix budget and every grid point that produced it."""

    k: int
    cot_token_count: int
    relative_sources: tuple[float, ...] = ()
    absolute_sources: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        token_count = _positive_int(self.cot_token_count, field="cot_token_count")
        if (
            not isinstance(self.k, int)
            or isinstance(self.k, bool)
            or not 0 <= self.k <= token_count
        ):
            raise ValueError("prefix budget k must be between zero and cot_token_count")
        if not self.relative_sources and not self.absolute_sources:
            raise ValueError("prefix budget must retain at least one grid origin")

    @property
    def is_no_prefix(self) -> bool:
        return self.k == 0

    @property
    def is_complete_prefix(self) -> bool:
        return self.k == self.cot_token_count

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "cot_token_count": self.cot_token_count,
            "relative_sources": list(self.relative_sources),
            "absolute_sources": list(self.absolute_sources),
            "is_no_prefix": self.is_no_prefix,
            "is_complete_prefix": self.is_complete_prefix,
        }


@dataclass(frozen=True, slots=True)
class CleanCotAlignment:
    """Clean-CoT suffix accepted by the legacy selection-stage comparison."""

    clean_cot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clean_cot_ids",
            _token_ids(self.clean_cot_ids, field="clean_cot_ids"),
        )

    @property
    def cot_token_count(self) -> int:
        return len(self.clean_cot_ids)

    @property
    def eligible_length(self) -> bool:
        return _MIN_CLEAN_COT_TOKENS <= self.cot_token_count <= _MAX_CLEAN_COT_TOKENS

    def to_dict(self) -> dict[str, object]:
        return {
            "clean_cot_ids": list(self.clean_cot_ids),
            "cot_token_count": self.cot_token_count,
            "eligible_length": self.eligible_length,
            "eligible_length_min_inclusive": _MIN_CLEAN_COT_TOKENS,
            "eligible_length_max_inclusive": _MAX_CLEAN_COT_TOKENS,
        }


@dataclass(frozen=True, slots=True)
class PrefixInputPlan:
    """Token-exact edited prompt plus clean-CoT suffix used at every budget."""

    edited_prompt_ids: tuple[int, ...]
    clean_cot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        prompt_ids = _token_ids(self.edited_prompt_ids, field="edited_prompt_ids")
        cot_ids = _token_ids(self.clean_cot_ids, field="clean_cot_ids")
        if not prompt_ids:
            raise ValueError("edited_prompt_ids must not be empty")
        object.__setattr__(self, "edited_prompt_ids", prompt_ids)
        object.__setattr__(self, "clean_cot_ids", cot_ids)

    @property
    def cot_token_count(self) -> int:
        return len(self.clean_cot_ids)

    def input_ids(self, budget: int) -> tuple[int, ...]:
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or not 0 <= budget <= self.cot_token_count
        ):
            raise ValueError("prefix budget must be between zero and cot_token_count")
        return self.edited_prompt_ids + self.clean_cot_ids[:budget]

    def to_dict(self) -> dict[str, object]:
        return {
            "edited_prompt_ids": list(self.edited_prompt_ids),
            "clean_cot_ids": list(self.clean_cot_ids),
            "edited_prompt_token_count": len(self.edited_prompt_ids),
            "cot_token_count": self.cot_token_count,
        }


def align_clean_cot_suffixes(
    *,
    clean_prompt_ids: Sequence[int],
    clean_full_ids: Sequence[int],
    edited_prompt_ids: Sequence[int],
    edited_full_ids: Sequence[int],
) -> CleanCotAlignment:
    """Apply the submitted selection-stage suffix comparison.

    Selection first recovered the suffix after each independently tokenized
    prompt and compared those suffixes.  It did not yet require the edited
    prompt to be a token prefix of the edited full input; that second audit is
    deliberately retained in :func:`build_prefix_input_plan`.
    """

    clean_prompt = _token_ids(clean_prompt_ids, field="clean_prompt_ids")
    clean_full = _token_ids(clean_full_ids, field="clean_full_ids")
    edited_prompt = _token_ids(edited_prompt_ids, field="edited_prompt_ids")
    edited_full = _token_ids(edited_full_ids, field="edited_full_ids")
    if not clean_prompt:
        raise ValueError("clean_prompt_ids must not be empty")
    if not edited_prompt:
        raise ValueError("edited_prompt_ids must not be empty")
    # The submitted selection helper used prompt *lengths* as offsets and
    # compared only the resulting suffixes.  It did not audit either prompt
    # prefix at this stage.
    clean_suffix = clean_full[len(clean_prompt) :]
    # This slicing intentionally does not assert the edited prefix.  Keeping
    # the submitted two-stage audit reproduces the six later exclusions.
    edited_suffix = edited_full[len(edited_prompt) :]
    if clean_suffix != edited_suffix:
        raise ValueError("clean and edited full inputs do not have aligned CoT suffixes")
    return CleanCotAlignment(clean_suffix)


def build_prefix_input_plan(
    *,
    edited_prompt_ids: Sequence[int],
    edited_full_ids: Sequence[int],
    alignment: CleanCotAlignment,
) -> PrefixInputPlan:
    """Perform the scan-stage boundary audit and retain token IDs verbatim."""

    if not isinstance(alignment, CleanCotAlignment):
        raise TypeError("alignment must be a CleanCotAlignment")
    edited_prompt = _token_ids(edited_prompt_ids, field="edited_prompt_ids")
    edited_full = _token_ids(edited_full_ids, field="edited_full_ids")
    if not edited_prompt:
        raise ValueError("edited_prompt_ids must not be empty")
    if edited_full[: len(edited_prompt)] != edited_prompt:
        raise ValueError("edited full input does not preserve the edited prompt prefix")
    if edited_full[len(edited_prompt) :] != alignment.clean_cot_ids:
        raise ValueError("edited full input CoT suffix changed after selection alignment")
    return PrefixInputPlan(
        edited_prompt_ids=edited_prompt,
        clean_cot_ids=alignment.clean_cot_ids,
    )


def build_budget_grid(
    cot_token_count: int,
    *,
    relative_budgets: Sequence[float] = RELATIVE_BUDGETS,
    absolute_budgets: Sequence[int] = ABSOLUTE_BUDGETS,
) -> tuple[PrefixBudget, ...]:
    """Build the deduplicated absolute/relative union grid.

    Python's built-in :func:`round` is intentional: the submitted producer
    used its ties-to-even behavior when mapping relative budgets to tokens.
    """

    token_count = _positive_int(cot_token_count, field="cot_token_count")
    try:
        relative_values = tuple(relative_budgets)
    except TypeError as exc:
        raise ValueError("relative budgets must be a sequence") from exc
    if not relative_values:
        raise ValueError("relative budgets must not be empty")

    normalized_relative: list[float] = []
    for value in relative_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("relative budgets must be finite numbers from 0 through 1")
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError("relative budgets must be finite numbers from 0 through 1")
        normalized_relative.append(normalized)
    if len(set(normalized_relative)) != len(normalized_relative):
        raise ValueError("relative budgets must not contain duplicates")
    if 0.0 not in normalized_relative or 1.0 not in normalized_relative:
        raise ValueError("relative budgets must include both 0 and 1 endpoints")

    try:
        absolute_values = tuple(absolute_budgets)
    except TypeError as exc:
        raise ValueError("absolute budgets must be a sequence") from exc
    normalized_absolute: list[int] = []
    for value in absolute_values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("absolute budgets must be positive integers")
        normalized_absolute.append(value)
    if len(set(normalized_absolute)) != len(normalized_absolute):
        raise ValueError("absolute budgets must not contain duplicates")

    relative_by_k: dict[int, set[float]] = {}
    for relative in normalized_relative:
        k = round(relative * token_count)
        relative_by_k.setdefault(k, set()).add(relative)
    absolute_by_k: dict[int, set[int]] = {}
    for absolute in normalized_absolute:
        if absolute <= token_count:
            absolute_by_k.setdefault(absolute, set()).add(absolute)

    distinct_k = sorted(set(relative_by_k) | set(absolute_by_k))
    return tuple(
        PrefixBudget(
            k=k,
            cot_token_count=token_count,
            relative_sources=tuple(sorted(relative_by_k.get(k, ()))),
            absolute_sources=tuple(sorted(absolute_by_k.get(k, ()))),
        )
        for k in distinct_k
    )


def _sample_ids(values: Sequence[str], *, arm: str) -> tuple[str, ...]:
    try:
        sample_ids = tuple(values)
    except TypeError as exc:
        raise ValueError(f"sample IDs for arm {arm!r} must be a sequence") from exc
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError(f"sample IDs for arm {arm!r} must be non-empty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"sample IDs for arm {arm!r} must be unique")
    return tuple(sorted(sample_ids))


def _systematic_take(values: tuple[str, ...], count: int) -> tuple[str, ...]:
    if count >= len(values):
        return values
    if count <= 0:
        return ()
    return tuple(values[(index * len(values)) // count] for index in range(count))


def select_extension_sample_ids(
    sample_ids_by_targeting: Mapping[str, Sequence[str]],
    *,
    max_pairs: int,
    cap_per_targeting: int = _DEFAULT_TARGETING_CAP,
) -> tuple[tuple[str, str], ...]:
    """Apply arm cap, proportional quota, and systematic floor sampling."""

    target_count = _positive_int(max_pairs, field="max_pairs")
    arm_cap = _positive_int(cap_per_targeting, field="cap_per_targeting")
    if not isinstance(sample_ids_by_targeting, Mapping) or not sample_ids_by_targeting:
        raise ValueError("sample_ids_by_targeting must be a non-empty mapping")

    capped: dict[str, tuple[str, ...]] = {}
    for arm, values in sample_ids_by_targeting.items():
        if not isinstance(arm, str) or not arm:
            raise ValueError("targeting arm names must be non-empty strings")
        capped[arm] = _sample_ids(values, arm=arm)[:arm_cap]

    arms = tuple(sorted(capped))
    available = sum(len(capped[arm]) for arm in arms)
    if available == 0:
        return ()
    take_total = min(target_count, available)
    if take_total == available:
        quota = {arm: len(capped[arm]) for arm in arms}
    else:
        raw = {arm: take_total * len(capped[arm]) / available for arm in arms}
        quota = {arm: min(len(capped[arm]), round(raw[arm])) for arm in arms}
        # Match the submitted deterministic remainder adjustment.  Ties in
        # arm size resolve by the already sorted arm name.
        while sum(quota.values()) != take_total:
            step = 1 if sum(quota.values()) < take_total else -1
            moved = False
            for arm in sorted(arms, key=lambda name: -len(capped[name])):
                candidate = quota[arm] + step
                if 0 <= candidate <= len(capped[arm]):
                    quota[arm] = candidate
                    moved = True
                    break
            if not moved:  # Defensive: valid bounded quotas should always move.
                raise RuntimeError("could not reconcile proportional extension quotas")

    return tuple(
        (arm, sample_id) for arm in arms for sample_id in _systematic_take(capped[arm], quota[arm])
    )


__all__ = [
    "ABSOLUTE_BUDGETS",
    "RELATIVE_BUDGETS",
    "CleanCotAlignment",
    "PrefixBudget",
    "PrefixInputPlan",
    "align_clean_cot_suffixes",
    "build_budget_grid",
    "build_prefix_input_plan",
    "select_extension_sample_ids",
]
