"""Training pairs replay exactly and align only unchanged token targets."""

from __future__ import annotations

import hashlib

import pytest

from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.training.pairs import (
    TrainingSource,
    align_unchanged_token_positions,
    edited_word_final_token_positions,
    materialize_training_pair,
    stable_epoch_sources,
)


NATURAL_SUBSTITUTIONS = {
    character: {"z" if character != "z" else "x": 1.0} for character in "abcdefghijklmnopqrstuvwxyz"
}


def _clean_source(record_id: str = "a" * 64) -> TrainingSource:
    text = "The airport supports reliable international travel."
    return TrainingSource.from_dict(
        {
            "schema_version": "robustness-clean-record/v1",
            "kind": "clean",
            "record_id": record_id,
            "source": "fineweb_edu",
            "source_revision": "b" * 40,
            "source_split": "train",
            "source_id": f"source-{record_id[:8]}",
            "group_id": f"group-{record_id[:8]}",
            "split": "train",
            "text": text,
            "task": None,
            "answer": None,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "normalized_content_sha256": normalized_content_sha256(text),
            "metadata": {},
            "token_count": 8,
        }
    )


def test_epoch_order_and_on_the_fly_pair_are_counter_based_and_resume_safe() -> None:
    sources = tuple(_clean_source(f"{index:064x}") for index in range(12))
    first_order = stable_epoch_sources(sources, seed=42, epoch=3)
    assert first_order == stable_epoch_sources(tuple(reversed(sources)), seed=42, epoch=3)
    assert first_order != stable_epoch_sources(sources, seed=42, epoch=4)

    generator = TypoGenerator(seed=42, natural_substitutions=NATURAL_SUBSTITUTIONS)
    first = materialize_training_pair(first_order[7], generator=generator, epoch=3)
    resumed = materialize_training_pair(first_order[7], generator=generator, epoch=3)
    assert first == resumed
    assert first.typo_text != first.clean_text
    assert first.edits


def test_natural_pair_orientation_infers_the_edited_word_without_synthetic_generation() -> None:
    clean = "The airport is busy."
    typo = "The arport is busy."
    source = TrainingSource.from_dict(
        {
            "schema_version": "robustness-natural-pair/v1",
            "kind": "natural",
            "record_id": "e" * 64,
            "source": "github_typo_corpus",
            "source_revision": "f" * 40,
            "source_split": "v1.0.0",
            "source_id": "commit:0",
            "group_id": "https://github.com/example/repo",
            "split": "train",
            "clean_text": clean,
            "typo_text": typo,
            "task": None,
            "answer": None,
            "operation": "deletion",
            "training_eligible": True,
            "repository": "https://github.com/example/repo",
            "repository_license": "MIT",
            "clean_sha256": hashlib.sha256(clean.encode()).hexdigest(),
            "typo_sha256": hashlib.sha256(typo.encode()).hexdigest(),
            "metadata": {},
            "token_count": 6,
        }
    )
    pair = materialize_training_pair(
        source,
        generator=TypoGenerator(seed=42, natural_substitutions=NATURAL_SUBSTITUTIONS),
        epoch=99,
    )
    assert pair.clean_text == "The airport is busy."
    assert pair.typo_text == "The arport is busy."
    assert len(pair.edits) == 1
    assert pair.edits[0].clean_char_span == (4, 11)
    assert pair.edits[0].typo_char_span == (4, 10)


def test_alignment_excludes_changed_subwords_but_keeps_shifted_text_and_punctuation() -> None:
    clean = "The airport works."
    typo = "The airrport works."
    clean_offsets = ((0, 0), (0, 3), (4, 11), (12, 17), (17, 18))
    typo_offsets = ((0, 0), (0, 3), (4, 7), (7, 12), (13, 18), (18, 19))
    aligned = align_unchanged_token_positions(
        clean_text=clean,
        typo_text=typo,
        clean_edit_spans=((4, 11),),
        typo_edit_spans=((4, 12),),
        clean_offsets=clean_offsets,
        typo_offsets=typo_offsets,
    )
    assert aligned == ((1, 1), (3, 4), (4, 5))
    # Causal logits immediately before these targets are therefore also aligned.
    assert tuple((clean_index - 1, typo_index - 1) for clean_index, typo_index in aligned) == (
        (0, 0),
        (2, 3),
        (3, 4),
    )


def test_alignment_handles_token_count_decrease_and_left_padding() -> None:
    clean = "An airport works"
    typo = "An arport works"
    clean_offsets = ((0, 0), (0, 0), (0, 2), (3, 6), (6, 10), (11, 16))
    typo_offsets = ((0, 0), (0, 2), (3, 9), (10, 15))
    aligned = align_unchanged_token_positions(
        clean_text=clean,
        typo_text=typo,
        clean_edit_spans=((3, 10),),
        typo_edit_spans=((3, 9),),
        clean_offsets=clean_offsets,
        typo_offsets=typo_offsets,
    )
    assert aligned == ((2, 1), (5, 3))


def test_word_final_positions_choose_the_last_subword() -> None:
    offsets = ((0, 0), (0, 3), (4, 7), (7, 11), (11, 12))
    assert edited_word_final_token_positions(offsets, ((4, 11),)) == (3,)


def test_word_final_positions_allow_punctuation_in_the_same_token() -> None:
    text = "See </div> now"
    offsets = ((0, 0), (0, 3), (4, 10), (11, 14))

    assert edited_word_final_token_positions(offsets, ((6, 9),), text=text) == (2,)


def test_word_final_positions_reject_ambiguous_partial_word_boundaries() -> None:
    offsets = ((0, 0), (0, 3), (4, 10), (11, 14))
    with pytest.raises(ValueError, match="not exactly covered"):
        edited_word_final_token_positions(offsets, ((6, 9),))

    with pytest.raises(ValueError, match="inside an alphanumeric word"):
        edited_word_final_token_positions(offsets, ((5, 9),), text="See alphabet now")
