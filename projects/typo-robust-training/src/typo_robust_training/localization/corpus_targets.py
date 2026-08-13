"""Teacher-forced continuation coordinates for generic-text localization."""

from __future__ import annotations

from collections.abc import Sequence


def clean_corpus_targets(
    *,
    full_clean_token_ids: Sequence[int],
    clean_prompt_token_ids: Sequence[int],
    final_edited_token: int,
    count: int,
) -> tuple[tuple[int, ...], str | None]:
    """Take corpus targets strictly after a stable edited-word-final prefix."""

    full = tuple(full_clean_token_ids)
    prompt = tuple(clean_prompt_token_ids)
    if (
        isinstance(final_edited_token, bool)
        or not isinstance(final_edited_token, int)
        or final_edited_token < 0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ValueError("generic corpus target coordinates are invalid")
    if final_edited_token != len(prompt) - 1:
        raise ValueError("generic clean prompt must end at the final edited-word token")
    if tuple(full[: len(prompt)]) != prompt:
        raise ValueError("generic clean prompt tokenization is not a stable prefix")
    targets = full[len(prompt) : len(prompt) + count]
    if len(targets) != count:
        return (), "fewer-than-16-clean-corpus-tokens-after-final-edit"
    return targets, None


__all__ = ["clean_corpus_targets"]
