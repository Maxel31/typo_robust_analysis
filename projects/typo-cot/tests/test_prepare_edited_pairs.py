"""Paper-contract tests for clean/edited pair preparation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.prepare_edited_pairs.protocol import (
    CandidateToken,
    CharacterEdit,
    PairProtocolError,
    apply_paper_edits,
    build_aligned_words,
    eligible_candidates,
    order_candidates,
)
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PairPreparationRunError,
    PrepareEditedPairsConfig,
    PrepareEditedPairsResult,
    run_prepare_edited_pairs,
)
from typo_cot.experiments.prepare_edited_pairs.runtime import _tokenize_with_offsets
from typo_cot.models.wrapper import ModelWrapper


def test_catalog_marks_pair_preparation_as_implemented() -> None:
    assert get_experiment("prepare-edited-pairs").status == "implemented"


def test_pair_preparation_allows_the_documented_qwen_72b_scale_model() -> None:
    assert "Qwen/Qwen2.5-72B-Instruct" in ModelWrapper.get_allowed_models()


def test_cli_dispatches_experiment_specific_pair_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[PrepareEditedPairsConfig] = []

    def fake_run(config: PrepareEditedPairsConfig) -> PrepareEditedPairsResult:
        captured.append(config)
        return PrepareEditedPairsResult(
            pairs_path=config.output_dir / "pairs.jsonl",
            run_path=config.output_dir / "run.json",
            written=2,
        )

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
    assert capsys.readouterr().out.splitlines() == [
        "wrote 2 pair(s): results/pairs/pairs.jsonl",
        "run manifest: results/pairs/run.json",
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


@pytest.mark.parametrize("value", ("0", "5"))
def test_cli_rejects_edit_counts_outside_the_paper_protocol(value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
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
                value,
                "--output-dir",
                "results/pairs",
            ]
        )

    assert exc_info.value.code == 2


def test_eligible_candidates_follow_the_paper_span_rules() -> None:
    candidates = eligible_candidates(
        prompt_text="0123456789Alpha 42 (A) beta ?",
        token_texts=["<bos>", " Alpha", " 42", " (A)", " beta", "?"],
        relevances=[0.0, -0.8, 9.0, 5.0, 0.3, 2.0],
        offsets=[(0, 0), (10, 15), (16, 18), (19, 22), (23, 27), (28, 29)],
        editable_prompt_start=10,
        editable_prompt_end=29,
    )

    assert [(item.token_index, item.text, item.relevance) for item in candidates] == [
        (1, " Alpha", -0.8),
        (4, " beta", 0.3),
    ]


def test_bare_single_letter_words_are_not_treated_as_option_markers() -> None:
    candidates = eligible_candidates(
        prompt_text="a i (A) option",
        token_texts=[" a", " i", " (", "A", ")", " option"],
        relevances=[1.0] * 6,
        offsets=[(0, 1), (2, 3), (4, 5), (5, 6), (6, 7), (8, 14)],
        editable_prompt_start=0,
        editable_prompt_end=14,
    )

    assert [item.token_index for item in candidates] == [0, 1, 5]


@pytest.mark.parametrize("relevance", (float("nan"), float("inf"), float("-inf")))
def test_eligible_candidates_reject_nonfinite_relevance(relevance: float) -> None:
    with pytest.raises(PairProtocolError, match="non-finite"):
        eligible_candidates(
            prompt_text="word",
            token_texts=[" word"],
            relevances=[relevance],
            offsets=[(0, 4)],
            editable_prompt_start=0,
            editable_prompt_end=4,
        )


def test_runtime_offsets_exclude_tokenizer_prefix_whitespace() -> None:
    class PrefixWhitespaceTokenizer:
        def __call__(self, text: str, **_kwargs: object) -> dict[str, object]:
            assert text == " Alpha"
            return {"input_ids": [1], "offset_mapping": [(0, 6)]}

    _, offsets = _tokenize_with_offsets(PrefixWhitespaceTokenizer(), " Alpha")

    # The archived AttnLRP pipeline lstripped decoded token text before locating
    # it. Keeping that rule includes an item-initial token whose raw fast-tokenizer
    # offset also covers the separator immediately before the editable span.
    assert offsets == [(1, 6)]


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


def test_random_control_always_excludes_attribution_top_four_and_is_deterministic() -> None:
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
        num_edits=1,
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
        CandidateToken(9, " beta", 1.0, 17, 21, attribution_rank=1),
        CandidateToken(3, "A", 0.8, 10, 11, attribution_rank=2),
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
        editable_text="Aalpha beta",
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


def test_apply_paper_edits_rejects_a_candidate_without_an_attribution_rank() -> None:
    with pytest.raises(PairProtocolError, match="attribution_rank"):
        apply_paper_edits(
            editable_text="beta",
            editable_prompt_start=0,
            candidate_order=[CandidateToken(3, "beta", 1.0, 0, 4)],
            num_edits=1,
            seed=42,
            sample_id="sample-1",
            edit_token=lambda text, _seed: CharacterEdit(
                original=text,
                edited="beeta",
                operation="duplication",
                character_index=1,
                original_character="e",
                new_character="e",
            ),
        )


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
            edited="beeta",
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


def test_completed_resume_returns_without_accessing_runtime(tmp_path: Path) -> None:
    output_dir = tmp_path / "completed"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )
    expected = run_prepare_edited_pairs(config, runtime=_FakeRuntime())

    class RuntimeMustNotLoad(_FakeRuntime):
        def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
            raise AssertionError("completed resume must not load a model or dataset")

    resumed = run_prepare_edited_pairs(
        replace(config, resume=True),
        runtime=RuntimeMustNotLoad(),
    )

    assert resumed == expected


def test_completed_resume_accepts_equivalent_relative_and_absolute_output_paths(
    tmp_path: Path,
) -> None:
    absolute_output = tmp_path / "completed-path-form"
    relative_output = Path(os.path.relpath(absolute_output, start=Path.cwd()))
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=relative_output,
    )
    expected = run_prepare_edited_pairs(config, runtime=_FakeRuntime())

    class RuntimeMustNotLoad(_FakeRuntime):
        def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
            raise AssertionError("completed resume must not load a model or dataset")

    resumed = run_prepare_edited_pairs(
        replace(config, output_dir=absolute_output, resume=True),
        runtime=RuntimeMustNotLoad(),
    )

    assert resumed.written == expected.written
    assert resumed.pairs_path.resolve() == expected.pairs_path.resolve()
    assert resumed.run_path.resolve() == expected.run_path.resolve()


def test_runner_rejects_an_empty_benchmark_selection(tmp_path: Path) -> None:
    class EmptyRuntime(_FakeRuntime):
        def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
            return []

    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=tmp_path / "empty",
    )

    with pytest.raises(ValueError, match="no benchmark samples"):
        run_prepare_edited_pairs(config, runtime=EmptyRuntime())


def test_runner_rejects_resume_in_nonempty_directory_without_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "unrelated"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("not an experiment\n", encoding="utf-8")
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
        resume=True,
    )

    with pytest.raises(ValueError, match="run.json"):
        run_prepare_edited_pairs(config, runtime=_FakeRuntime())

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "not an experiment\n"
    assert not (output_dir / "pairs.jsonl").exists()


def test_runner_resumes_only_missing_records_after_an_item_failure(tmp_path: Path) -> None:
    class FailingRuntime(_FakeRuntime):
        def prepare_pair(
            self, sample: dict[str, str], config: PrepareEditedPairsConfig
        ) -> dict[str, object]:
            if sample["sample_id"] == "b":
                raise RuntimeError("injected failure")
            return super().prepare_pair(sample, config)

    output_dir = tmp_path / "resume"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )

    with pytest.raises(PairPreparationRunError, match="1 item"):
        run_prepare_edited_pairs(config, runtime=FailingRuntime())

    failed_manifest = json.loads((output_dir / "run.json").read_text())
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["counts"] == {"discovered": 2, "written": 1, "failed": 1}
    assert not (output_dir / "pairs.jsonl").exists()

    resumed_runtime = _FakeRuntime()
    result = run_prepare_edited_pairs(
        replace(config, resume=True),
        runtime=resumed_runtime,
    )

    rows = [json.loads(line) for line in result.pairs_path.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert json.loads((output_dir / "run.json").read_text())["status"] == "completed"


def test_runner_rejects_resume_when_runtime_provenance_changed(tmp_path: Path) -> None:
    class InitialRuntime(_FakeRuntime):
        def prepare_pair(
            self, sample: dict[str, str], config: PrepareEditedPairsConfig
        ) -> dict[str, object]:
            if sample["sample_id"] == "b":
                raise RuntimeError("injected failure")
            return super().prepare_pair(sample, config)

        def provenance(self) -> dict[str, str]:
            return {
                "model_revision": "revision-one",
                "dataset_records_sha256": "dataset-one",
            }

    class ChangedRuntime(_FakeRuntime):
        def provenance(self) -> dict[str, str]:
            return {
                "model_revision": "revision-two",
                "dataset_records_sha256": "dataset-one",
            }

    output_dir = tmp_path / "changed-provenance"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )
    with pytest.raises(PairPreparationRunError):
        run_prepare_edited_pairs(config, runtime=InitialRuntime())

    with pytest.raises(ValueError, match="provenance"):
        run_prepare_edited_pairs(
            replace(config, resume=True),
            runtime=ChangedRuntime(),
        )

    manifest = json.loads((output_dir / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["provenance"]["model_revision"] == "revision-one"


def test_readme_starts_from_the_complete_public_command() -> None:
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    for fragment in (
        "uv run --project projects/typo-cot --extra lrp typo-cot prepare-edited-pairs",
        "--model google/gemma-3-4b-it",
        "--benchmark gsm8k",
        "--targeting attribution-4",
        "--num-edits 4",
        "--output-dir results/prepare-edited-pairs",
        "pairs.jsonl",
        "run.json",
    ):
        assert fragment in readme
