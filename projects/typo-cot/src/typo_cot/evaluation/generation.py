"""Shared fail-closed generation termination helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

GenerationTermination = Literal["eos", "length-cap"]


def _normalized_eos_token_ids(value: object) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return tuple(
        sorted(
            {
                token
                for token in values
                if isinstance(token, int) and not isinstance(token, bool) and token >= 0
            }
        )
    )


def resolve_effective_eos_token_ids(
    *,
    generation_config: object,
    tokenizer: object,
    operation: str,
) -> tuple[tuple[int, ...], str]:
    """Resolve every effective EOS ID while rejecting other stop mechanisms."""
    if not isinstance(operation, str) or not operation:
        raise ValueError("generation operation must be a non-empty string")
    if getattr(generation_config, "stop_strings", None):
        raise ValueError(f"{operation} does not support stop_strings")
    if getattr(generation_config, "forced_eos_token_id", None) is not None:
        raise ValueError(f"{operation} does not support forced_eos_token_id")
    values = _normalized_eos_token_ids(getattr(generation_config, "eos_token_id", None))
    if values:
        return values, "model-generation-config"
    values = _normalized_eos_token_ids(getattr(tokenizer, "eos_token_id", None))
    if values:
        return values, "tokenizer-fallback"
    raise ValueError(f"{operation} exposes no EOS token ID")


def classify_generated_token_ids(
    token_ids: Sequence[int],
    *,
    effective_eos_token_ids: Sequence[int],
    max_new_tokens: int,
    field: str,
) -> tuple[tuple[int, ...], GenerationTermination]:
    """Return the first-EOS continuation and its explicit stopping reason."""
    if not isinstance(field, str) or not field:
        raise ValueError("generation field must be a non-empty string")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    raw = tuple(token_ids)
    if not raw:
        raise ValueError(f"{field} generated no continuation token")
    if len(raw) > max_new_tokens:
        raise ValueError(f"{field} exceeded the generation token cap")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in raw):
        raise ValueError(f"{field} generated invalid token IDs")
    eos_ids = frozenset(_normalized_eos_token_ids(effective_eos_token_ids))
    if not eos_ids:
        raise ValueError("effective_eos_token_ids must contain at least one token ID")
    eos_index = next(
        (index for index, token in enumerate(raw) if token in eos_ids),
        None,
    )
    if eos_index is not None:
        return raw[: eos_index + 1], "eos"
    if len(raw) != max_new_tokens:
        raise ValueError(f"{field} stopped without EOS before the token cap")
    return raw, "length-cap"


__all__ = [
    "GenerationTermination",
    "classify_generated_token_ids",
    "resolve_effective_eos_token_ids",
]
