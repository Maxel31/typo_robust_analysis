"""Pure token and position planning for the one-token diagnostic."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Literal


def _token_tuple(value: object, *, field: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be a sequence of token IDs")
    result = tuple(value)
    if not result and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in result):
        raise ValueError(f"{field} must contain non-negative integer token IDs")
    return result


@dataclass(frozen=True, slots=True)
class OneTokenInputPlan:
    """Exact full inputs for profiling and prompt-prefix inputs for generation."""

    clean_prompt_ids: tuple[int, ...]
    edited_prompt_ids: tuple[int, ...]
    clean_full_ids: tuple[int, ...]
    edited_full_ids: tuple[int, ...]
    clean_cot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for field in (
            "clean_prompt_ids",
            "edited_prompt_ids",
            "clean_full_ids",
            "edited_full_ids",
            "clean_cot_ids",
        ):
            object.__setattr__(self, field, _token_tuple(getattr(self, field), field=field))
        clean_suffix = self.clean_full_ids[len(self.clean_prompt_ids) :]
        edited_suffix = self.edited_full_ids[len(self.edited_prompt_ids) :]
        if self.clean_full_ids[: len(self.clean_prompt_ids)] != self.clean_prompt_ids:
            raise ValueError("clean full input must preserve the exact clean prompt boundary")
        if self.edited_full_ids[: len(self.edited_prompt_ids)] != self.edited_prompt_ids:
            raise ValueError("edited full input must preserve the exact edited prompt boundary")
        if clean_suffix != self.clean_cot_ids or edited_suffix != self.clean_cot_ids:
            raise ValueError("full clean and edited inputs must share the exact clean-CoT suffix")

    @property
    def profile_clean_input_ids(self) -> tuple[int, ...]:
        return self.clean_full_ids

    @property
    def profile_edited_input_ids(self) -> tuple[int, ...]:
        return self.edited_full_ids

    @property
    def cot_token_count(self) -> int:
        return len(self.clean_cot_ids)

    def generation_input_ids(self, position: int, forced_token_id: int) -> tuple[int, ...]:
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError("position must be an integer")
        if not 0 <= position < self.cot_token_count:
            raise ValueError("position must index the clean CoT")
        if (
            not isinstance(forced_token_id, int)
            or isinstance(forced_token_id, bool)
            or forced_token_id < 0
        ):
            raise ValueError("forced_token_id must be a non-negative integer")
        return (*self.clean_prompt_ids, *self.clean_cot_ids[:position], forced_token_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "clean_prompt_ids": list(self.clean_prompt_ids),
            "edited_prompt_ids": list(self.edited_prompt_ids),
            "clean_full_ids": list(self.clean_full_ids),
            "edited_full_ids": list(self.edited_full_ids),
            "clean_cot_ids": list(self.clean_cot_ids),
        }


@dataclass(frozen=True, slots=True)
class OneTokenProfile:
    """Positionwise clean-to-edited next-token profile over the clean CoT."""

    clean_to_edited_kl: tuple[float, ...]
    clean_token_rank_under_clean: tuple[int, ...]
    clean_token_rank_under_edited: tuple[int, ...]
    edited_top1_ids: tuple[int, ...]
    edited_top1_is_admissible: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "clean_to_edited_kl", tuple(self.clean_to_edited_kl))
        for field in (
            "clean_token_rank_under_clean",
            "clean_token_rank_under_edited",
            "edited_top1_ids",
            "edited_top1_is_admissible",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        length = len(self.clean_to_edited_kl)
        if length == 0 or any(
            len(getattr(self, field)) != length
            for field in (
                "clean_token_rank_under_clean",
                "clean_token_rank_under_edited",
                "edited_top1_ids",
                "edited_top1_is_admissible",
            )
        ):
            raise ValueError("all profile fields must have one common non-zero length")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < -1e-5
            for value in self.clean_to_edited_kl
        ):
            raise ValueError("KL values must be finite and non-negative")
        for field in ("clean_token_rank_under_clean", "clean_token_rank_under_edited"):
            if any(
                not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0
                for rank in getattr(self, field)
            ):
                raise ValueError(f"{field} must contain positive integer ranks")
        _token_tuple(self.edited_top1_ids, field="edited_top1_ids")
        if any(type(value) is not bool for value in self.edited_top1_is_admissible):
            raise ValueError("edited_top1_is_admissible must contain booleans")

    def to_dict(self) -> dict[str, object]:
        return {
            "clean_to_edited_kl": [float(value) for value in self.clean_to_edited_kl],
            "clean_token_rank_under_clean": list(self.clean_token_rank_under_clean),
            "clean_token_rank_under_edited": list(self.clean_token_rank_under_edited),
            "edited_top1_ids": list(self.edited_top1_ids),
            "edited_top1_is_admissible": list(self.edited_top1_is_admissible),
        }


@dataclass(frozen=True, slots=True)
class DistantPositionSelection:
    selected_position: int
    distant_position: int | None
    selected_edited_top1_id: int
    distant_edited_top1_id: int | None
    candidate_count: int
    distant_candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_position": self.selected_position,
            "distant_position": self.distant_position,
            "selected_edited_top1_id": self.selected_edited_top1_id,
            "distant_edited_top1_id": self.distant_edited_top1_id,
            "candidate_count": self.candidate_count,
            "distant_candidate_count": self.distant_candidate_count,
        }


def choose_distant_positions(
    profile: OneTokenProfile,
    *,
    min_distance: int = 3,
) -> DistantPositionSelection:
    """Select final-PDF P and its lower-median distant control."""

    if not isinstance(profile, OneTokenProfile):
        raise TypeError("profile must be a OneTokenProfile")
    if not isinstance(min_distance, int) or isinstance(min_distance, bool) or min_distance <= 0:
        raise ValueError("min_distance must be a positive integer")
    candidates = [
        index for index, rank in enumerate(profile.clean_token_rank_under_edited) if rank > 1
    ]
    if not candidates:
        raise ValueError("no-position-with-clean-token-below-edited-top1")
    selected = max(candidates, key=lambda index: profile.clean_to_edited_kl[index])
    distant = [index for index in candidates if abs(index - selected) >= min_distance]
    control: int | None = None
    if distant:
        median = statistics.median_low([profile.clean_to_edited_kl[index] for index in distant])
        control = min(index for index in distant if profile.clean_to_edited_kl[index] == median)
    return DistantPositionSelection(
        selected_position=selected,
        distant_position=control,
        selected_edited_top1_id=profile.edited_top1_ids[selected],
        distant_edited_top1_id=(None if control is None else profile.edited_top1_ids[control]),
        candidate_count=len(candidates),
        distant_candidate_count=len(distant),
    )


def choose_adjacent_position(
    clean_to_edited_kl: tuple[float, ...] | list[float],
    *,
    selected_position: int,
    tie_key: str,
) -> int | None:
    """Choose the producer-backed nearest strictly-lower-KL control."""

    values = tuple(float(value) for value in clean_to_edited_kl)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("adjacent KL profile must be non-empty and finite")
    if not 0 <= selected_position < len(values):
        raise ValueError("selected_position is outside the KL profile")
    if not isinstance(tie_key, str) or not tie_key:
        raise ValueError("tie_key must be a non-empty string")
    lower = [
        index
        for index, value in enumerate(values)
        if index != selected_position and value < values[selected_position]
    ]
    if not lower:
        return None
    distance = min(abs(index - selected_position) for index in lower)
    tied = [index for index in lower if abs(index - selected_position) == distance]
    prefer_right = int(hashlib.sha256(tie_key.encode("utf-8")).hexdigest(), 16) % 2
    tied.sort(
        key=lambda index: (
            0 if ((index > selected_position) == bool(prefer_right)) else 1,
            index,
        )
    )
    return tied[0]


ArmPosition = Literal["selected", "distant", "adjacent"]


@dataclass(frozen=True, slots=True)
class OneTokenArmSpec:
    name: str
    position_label: ArmPosition
    position: int
    forced_token_id: int
    clean_token_id: int
    token_source: Literal["clean", "selected-edited-top1", "distant-edited-top1"]
    token_is_admissible: bool | None

    @property
    def is_noop(self) -> bool:
        return self.forced_token_id == self.clean_token_id

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "position_label": self.position_label,
            "position": self.position,
            "forced_token_id": self.forced_token_id,
            "clean_token_id": self.clean_token_id,
            "token_source": self.token_source,
            "token_is_admissible": self.token_is_admissible,
            "is_noop": self.is_noop,
        }


def build_arm_specs(
    plan: OneTokenInputPlan,
    profile: OneTokenProfile,
    positions: DistantPositionSelection,
    *,
    adjacent_position: int | None,
) -> tuple[OneTokenArmSpec, ...]:
    """Freeze six distant arms plus two optional adjacent arms."""

    selected = positions.selected_position
    distant = positions.distant_position
    distant_token = positions.distant_edited_top1_id
    if distant is None or distant_token is None:
        raise ValueError("distant position is unavailable")
    selected_token = positions.selected_edited_top1_id

    def arm(
        name: str,
        label: ArmPosition,
        position: int,
        token: int,
        source: Literal["clean", "selected-edited-top1", "distant-edited-top1"],
        admissible: bool | None,
    ) -> OneTokenArmSpec:
        return OneTokenArmSpec(
            name=name,
            position_label=label,
            position=position,
            forced_token_id=token,
            clean_token_id=plan.clean_cot_ids[position],
            token_source=source,
            token_is_admissible=admissible,
        )

    specs = [
        arm("selected_keep", "selected", selected, plan.clean_cot_ids[selected], "clean", None),
        arm(
            "selected_from_selected",
            "selected",
            selected,
            selected_token,
            "selected-edited-top1",
            profile.edited_top1_is_admissible[selected],
        ),
        arm(
            "selected_from_distant",
            "selected",
            selected,
            distant_token,
            "distant-edited-top1",
            profile.edited_top1_is_admissible[distant],
        ),
        arm("distant_keep", "distant", distant, plan.clean_cot_ids[distant], "clean", None),
        arm(
            "distant_from_selected",
            "distant",
            distant,
            selected_token,
            "selected-edited-top1",
            profile.edited_top1_is_admissible[selected],
        ),
        arm(
            "distant_from_distant",
            "distant",
            distant,
            distant_token,
            "distant-edited-top1",
            profile.edited_top1_is_admissible[distant],
        ),
    ]
    if adjacent_position is not None:
        specs.extend(
            (
                arm(
                    "adjacent_keep",
                    "adjacent",
                    adjacent_position,
                    plan.clean_cot_ids[adjacent_position],
                    "clean",
                    None,
                ),
                arm(
                    "adjacent_from_selected",
                    "adjacent",
                    adjacent_position,
                    selected_token,
                    "selected-edited-top1",
                    profile.edited_top1_is_admissible[selected],
                ),
            )
        )
    return tuple(specs)


__all__ = [
    "DistantPositionSelection",
    "OneTokenArmSpec",
    "OneTokenInputPlan",
    "OneTokenProfile",
    "build_arm_specs",
    "choose_adjacent_position",
    "choose_distant_positions",
]
