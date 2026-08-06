"""Paper-contract tests for clean/edited pair preparation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.prepare_edited_pairs.protocol import (
    CandidateToken,
    CharacterEdit,
    apply_paper_edits,
    build_aligned_words,
    eligible_candidates,
    order_candidates,
)
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PrepareEditedPairsConfig,
    run_prepare_edited_pairs,
)


def test_catalog_marks_pair_preparation_as_implemented() -> None:
    assert get_experiment("prepare-edited-pairs").status == "implemented"


def test_cli_dispatches_experiment_specific_pair_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[PrepareEditedPairsConfig] = []

    def fake_run(config: PrepareEditedPairsConfig) -> None:
        captured.append(config)

    monkeypatch.setattr(cli_module, "run_prepare_edited_pairs", fake_run)

    assert (
        main(
            [
                "prepare-edited-pairs",
                "--model",
                "google/gemma-3-4b-it",
                "--benchmark",
                "gsm8k",
                "--targeting",
                "attribution-4",
                "--num-edits",
                "4",
                "--output-dir",
                "results/pairs",
                "--gpu-id",
                "3",
                "--limit",
                "1",
            ]
        )
        == 0
    )

    assert captured == [
        PrepareEditedPairsConfig(
            model="google/gemma-3-4b-it",
            benchmark="gsm8k",
            targeting="attribution-4",
            num_edits=4,
            output_dir=Path("results/pairs"),
            gpu_id="3",
            limit=1,
        )
    ]


@pytest.mark.parametrize("value", ("rq1", "importance", "population-random"))
def test_cli_rejects_non_paper_targeting_names(value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "prepare-edited-pairs",
                "--model",
                "google/gemma-3-4b-it",
                "--benchmark",
                "gsm8k",
                "--targeting",
                value,
                "--num-edits",
                "4",
                "--output-dir",
                "results/pairs",
            ]
        )

    assert exc_info.value.code == 2


def test_eligible_candidates_follow_the_paper_span_rules() -> None:
    candidates = eligible_candidates(
        token_texts=["<bos>", " Alpha", " 42", " (A)", " beta", "?"],
        relevances=[0.0, -0.8, 9.0, 5.0, 0.3, 2.0],
        offsets=[(0, 0), (10, 16), (16, 19), (19, 23), (23, 28), (28, 29)],
        editable_prompt_start=10,
        editable_prompt_end=29,
    )

    assert [(item.token_index, item.text, item.relevance) for item in candidates] == [
        (1, " Alpha", -0.8),
        (4, " beta", 0.3),
    ]


def test_attribution_order_uses_largest_absolute_relevance() -> None:
    candidates = [
        CandidateToken(2, " two", -0.9, 4, 8),
        CandidateToken(1, " one", 0.4, 0, 4),
        CandidateToken(3, " three", 0.7, 8, 14),
    ]

    ordered = order_candidates(
        candidates,
        targeting="attribution-4",
        num_edits=2,
        seed=42,
        sample_id="sample-1",
    )

    assert [item.token_index for item in ordered.candidates] == [2, 3, 1]
    assert [item.token_index for item in ordered.excluded_top] == []


def test_random_control_excludes_attribution_top_k_and_is_deterministic() -> None:
    candidates = [
        CandidateToken(i, f" word{i}", float(10 - i), i * 6, (i + 1) * 6) for i in range(8)
    ]

    first = order_candidates(
        candidates,
        targeting="random-4",
        num_edits=4,
        seed=42,
        sample_id="sample-1",
    )
    second = order_candidates(
        list(reversed(candidates)),
        targeting="random-4",
        num_edits=4,
        seed=42,
        sample_id="sample-1",
    )

    assert [item.token_index for item in first.excluded_top] == [0, 1, 2, 3]
    assert first == second
    assert {item.token_index for item in first.candidates}.isdisjoint({0, 1, 2, 3})


def test_edit_attempts_preserve_the_reported_cumulative_shift_behavior() -> None:
    # Rank 1 duplicates a character in a later token. The historical paper
    # protocol applies its +1 cumulative shift to rank 2 even though rank 2 is
    # earlier in the text; rank 2 therefore lands outside its intended token.
    candidates = [
        CandidateToken(9, " beta", 1.0, 16, 21, attribution_rank=1),
        CandidateToken(3, "Alpha", 0.8, 10, 15, attribution_rank=2),
    ]

    def controlled_edit(text: str, _: int) -> CharacterEdit | None:
        if text.strip() == "beta":
            return CharacterEdit(
                original=text,
                edited=text + text[-1],
                operation="duplication",
                character_index=len(text) - 1,
                original_character=text[-1],
                new_character=text[-1],
            )
        if text:
            return CharacterEdit(
                original=text,
                edited="X" + text[1:],
                operation="substitution",
                character_index=0,
                original_character=text[0],
                new_character="X",
            )
        return None

    result = apply_paper_edits(
        editable_text="Alpha beta",
        editable_prompt_start=10,
        candidate_order=candidates,
        num_edits=2,
        seed=42,
        sample_id="sample-1",
        edit_token=controlled_edit,
    )

    assert len(result.attempts) == 2
    assert result.attempts[0].landed_on_intended_token is True
    assert result.attempts[1].landed_on_intended_token is False
    assert result.attempts[0].selection_rank == 1
    assert result.attempts[1].selection_rank == 2


class _OffsetTokenizer:
    """Whitespace tokenizer with an intentional edited-word split."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if "beeta" in text:
            start = text.index("beeta")
            return [(0, 0), (0, 5), (6, start), (start, start + 2), (start + 2, start + 5)]
        start = text.index("beta")
        return [(0, 0), (0, 5), (6, start), (start, start + 4)]


def test_alignment_records_actual_distinct_words_and_word_final_tokens() -> None:
    clean_prompt = "Alpha beta"
    edited_prompt = "Alpha beeta"
    tokenizer = _OffsetTokenizer()

    application = apply_paper_edits(
        editable_text=clean_prompt,
        editable_prompt_start=0,
        candidate_order=[CandidateToken(3, " beta", 1.0, 6, 10, attribution_rank=1)],
        num_edits=1,
        seed=42,
        sample_id="sample-1",
        edit_token=lambda text, _seed: CharacterEdit(
            original=text,
            edited=" beeta",
            operation="duplication",
            character_index=2,
            original_character="e",
            new_character="e",
        ),
    )

    aligned = build_aligned_words(
        clean_editable=clean_prompt,
        edited_editable=edited_prompt,
        editable_prompt_start=0,
        attempts=application.attempts,
        clean_token_offsets=tokenizer.offsets(clean_prompt),
        edited_token_offsets=tokenizer.offsets(edited_prompt),
    )

    assert len(aligned) == 1
    word = asdict(aligned[0])
    assert word["clean_text"] == "beta"
    assert word["edited_text"] == "beeta"
    assert word["target_ranks"] == (1,)
    assert word["clean_final_token"] == 3
    assert word["edited_final_token"] == 4


class _FakeRuntime:
    def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
        return [{"sample_id": "b"}, {"sample_id": "a"}]

    def prepare_pair(
        self, sample: dict[str, str], config: PrepareEditedPairsConfig
    ) -> dict[str, object]:
        return {
            "schema_version": "prepare-edited-pairs/v1",
            "sample_id": sample["sample_id"],
            "model": config.model,
            "benchmark": config.benchmark,
            "targeting": config.targeting,
            "target_attempts": [],
            "aligned_words": [],
        }

    def provenance(self) -> dict[str, str]:
        return {"runtime": "fake"}


def test_runner_writes_versioned_pairs_and_completed_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "pairs"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )

    result = run_prepare_edited_pairs(config, runtime=_FakeRuntime())

    rows = [json.loads(line) for line in (output_dir / "pairs.jsonl").read_text().splitlines()]
    manifest = json.loads((output_dir / "run.json").read_text())
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert all(row["schema_version"] == "prepare-edited-pairs/v1" for row in rows)
    assert manifest["schema_version"] == "prepare-edited-pairs-run/v1"
    assert manifest["status"] == "completed"
    assert manifest["counts"] == {"discovered": 2, "written": 2, "failed": 0}
    assert result.pairs_path == output_dir / "pairs.jsonl"


def test_readme_starts_from_the_complete_public_command() -> None:
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    for fragment in (
        "typo-cot prepare-edited-pairs",
        "--model google/gemma-3-4b-it",
        "--benchmark gsm8k",
        "--targeting attribution-4",
        "--num-edits 4",
        "--output-dir results/prepare-edited-pairs",
        "pairs.jsonl",
        "run.json",
    ):
        assert fragment in readme
