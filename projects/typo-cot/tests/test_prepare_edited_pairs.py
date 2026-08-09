"""Paper-contract tests for clean/edited pair preparation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import typo_cot.cli as cli_module
import typo_cot.data.loader as loader_module
import typo_cot.experiments.prepare_edited_pairs.runtime as runtime_module
from typo_cot.cli import main
from typo_cot.data.cohorts import load_sample_id_cohort
from typo_cot.evaluation.extractor import create_extractor
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.prepare_edited_pairs.protocol import (
    CandidateToken,
    CharacterEdit,
    PairProtocolError,
    apply_paper_edits,
    build_aligned_words,
    eligible_candidates,
    order_candidates,
    seeded_character_edit,
)
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PairPreparationRunError,
    PrepareEditedPairsConfig,
    PrepareEditedPairsResult,
    _validate_preload_resume_provenance,
    run_prepare_edited_pairs,
)
from typo_cot.experiments.prepare_edited_pairs.runtime import (
    HuggingFacePairPreparationRuntime,
    _tokenize_with_offsets,
)
from typo_cot.models.wrapper import ModelWrapper


def test_catalog_marks_pair_preparation_as_implemented() -> None:
    assert get_experiment("prepare-edited-pairs").status == "implemented"


def test_pair_preparation_allows_the_documented_qwen_72b_scale_model() -> None:
    assert "Qwen/Qwen2.5-72B-Instruct" in ModelWrapper.get_allowed_models()


@pytest.mark.parametrize(
    ("model", "benchmark", "expected"),
    (
        ("Qwen/Qwen2.5-7B-Instruct", "mmlu", 100),
        ("qwen2.5-7b-instruct", "mmlu", 100),
        ("google/gemma-3-12b-it", "mmlu", 100),
        ("google/GEMMA-3-27B-IT", "mmlu", 100),
        ("meta-llama/Llama-3.1-70B-Instruct", "mmlu", 100),
        ("google/gemma-3-4b-it", "mmlu", 50),
        ("meta-llama/Llama-3.2-3B-Instruct", "mmlu", 50),
        ("Qwen/Qwen2.5-72B-Instruct", "mmlu", 100),
        ("google/gemma-3-4b-it", "mmlu-pro", 100),
        ("Qwen/Qwen2.5-7B-Instruct", "gsm8k", 50),
    ),
)
def test_paper_cohort_size_depends_on_model_and_benchmark(
    model: str,
    benchmark: str,
    expected: int,
) -> None:
    config = PrepareEditedPairsConfig(
        model=model,
        benchmark=benchmark,
        targeting="attribution-4",
        num_edits=4,
        output_dir=Path("unused"),
    )

    assert runtime_module._paper_samples_per_subset(config) == expected


def test_full_mmlu_paper_cohort_models_are_supported_model_basenames() -> None:
    allowed_basenames = {
        model.rsplit("/", 1)[-1].lower() for model in ModelWrapper.get_allowed_models()
    }

    assert runtime_module._MMLU_100_PER_SUBJECT_PAPER_MODELS <= allowed_basenames


def test_preload_provenance_freezes_the_paper_cohort_rule() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
    )
    config = PrepareEditedPairsConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark="mmlu",
        targeting="attribution-4",
        num_edits=4,
        output_dir=Path("unused"),
    )

    provenance = runtime_module.preload_provenance(config, torch_module=fake_torch)

    assert provenance["dataset_cohort_rule"] == "paper-model-benchmark-cohort/v1"
    assert provenance["dataset_samples_per_subset"] == 100
    assert provenance["generation_protocol"] == "explicit-greedy-generation/v1"
    assert provenance["generation_termination_protocol"] == (
        "effective-eos-vs-length-cap/v1"
    )


def test_preload_provenance_uses_null_when_subset_cap_is_not_applied() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
    )
    config = PrepareEditedPairsConfig(
        model="google/gemma-3-4b-it",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=Path("unused"),
    )

    provenance = runtime_module.preload_provenance(config, torch_module=fake_torch)

    assert provenance["dataset_samples_per_subset"] is None


def test_subset_cap_allowlist_does_not_expand_with_the_loader_default_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(runtime_module._DEFAULT_SAMPLES_PER_SUBSET, "gsm8k", 123)

    assert (
        runtime_module.paper_dataset_samples_per_subset(
            model="google/gemma-3-4b-it",
            benchmark="gsm8k",
        )
        is None
    )


def test_protocol_only_resume_rejects_incomplete_current_provenance() -> None:
    with pytest.raises(ValueError, match="missing required protocol provenance"):
        _validate_preload_resume_provenance(
            {"provenance": {"runtime": "fake"}},
            {"runtime": "fake"},
            protocol_only=True,
        )


def test_runtime_passes_the_frozen_cohort_size_to_the_dataset_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sample = SimpleNamespace(
        sample_id="sample",
        question="question",
        choices=["a", "b"],
        correct_answer="A",
        subset="subject",
    )

    class FakeLoader:
        def load(self) -> list[SimpleNamespace]:
            return [sample]

    def fake_create_loader(**kwargs: object) -> FakeLoader:
        captured.update(kwargs)
        return FakeLoader()

    monkeypatch.setattr(loader_module, "create_loader", fake_create_loader)
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.internal_benchmark = "mmlu"
    runtime._dataset_provenance = {}
    config = PrepareEditedPairsConfig(
        model="google/gemma-3-12b-it",
        benchmark="mmlu",
        targeting="attribution-4",
        num_edits=4,
        output_dir=Path("unused"),
    )

    assert runtime.load_samples(config) == [sample]
    assert captured["samples_per_subset"] == 100


def test_pair_runtime_generation_explicitly_freezes_greedy_decoding() -> None:
    calls: list[dict[str, object]] = []

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, _prompt: str, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }

        def decode(self, _token_ids: list[int], **_kwargs: object) -> str:
            return "The answer is 2."

    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.config = SimpleNamespace(max_new_tokens=512)
    runtime.tokenizer = Tokenizer()
    runtime.wrapper = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(
            generate=lambda **kwargs: calls.append(kwargs)
            or torch.tensor([[10, 11, 20, 99]])
        ),
    )
    runtime._torch = torch
    runtime.effective_eos_token_ids = (99,)

    prompt_ids, continuation_ids, continuation, termination = runtime._generate("prompt")

    assert prompt_ids == [10, 11]
    assert continuation_ids == [20, 99]
    assert continuation == "The answer is 2."
    assert termination == "eos"
    assert len(calls) == 1
    assert torch.equal(calls[0].pop("input_ids"), torch.tensor([[10, 11]]))
    assert torch.equal(
        calls[0].pop("attention_mask"),
        torch.ones(1, 2, dtype=torch.long),
    )
    assert calls[0] == {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
        "pad_token_id": 0,
        "eos_token_id": [99],
    }


def test_pair_runtime_answer_uses_empty_only_fallback_and_exact_correctness() -> None:
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.internal_benchmark = "gsm8k"
    runtime.extractor = create_extractor("gsm8k")

    answer = runtime._answer(
        "Reasoning complete.\nFinal answer: the total is $2.",
        "2",
    )

    assert answer == {
        "value": "2",
        "is_extracted": True,
        "is_correct": True,
        "method": "fallback:N2_answer_line",
        "primary_method": "no_match",
        "confidence": 1.0,
    }


def test_pair_runtime_disables_positional_fallback_at_the_generation_cap() -> None:
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.internal_benchmark = "gsm8k"
    runtime.extractor = create_extractor("gsm8k")
    text = "The computation is unfinished and currently reaches 2 dollars."

    before_cap = runtime._answer(
        text,
        "2",
        allow_positional=True,
    )
    at_cap = runtime._answer(
        text,
        "2",
        allow_positional=False,
    )

    assert before_cap["method"] == "fallback:N5_tail_number"
    assert before_cap["is_correct"] is True
    assert at_cap["method"] == "unextractable"
    assert at_cap["is_correct"] is False


def test_pair_runtime_treats_an_eos_at_the_generation_cap_as_complete() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, _prompt: str, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }

        def decode(self, _token_ids: list[int], **_kwargs: object) -> str:
            return "The computation is complete and the final total is 2 dollars."

    generated = [41] * 511 + [99]
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.config = SimpleNamespace(max_new_tokens=512)
    runtime.tokenizer = Tokenizer()
    runtime.wrapper = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(
            generate=lambda **_kwargs: torch.tensor([[10, 11, *generated]])
        ),
    )
    runtime._torch = torch
    runtime.effective_eos_token_ids = (99,)
    runtime.internal_benchmark = "gsm8k"
    runtime.extractor = create_extractor("gsm8k")

    _prompt_ids, continuation_ids, continuation, termination = runtime._generate(
        "prompt"
    )
    answer = runtime._answer(
        continuation,
        "2",
        allow_positional=termination == "eos",
    )

    assert len(continuation_ids) == 512
    assert continuation_ids[-1] == 99
    assert termination == "eos"
    assert answer["method"] == "fallback:N5_tail_number"
    assert answer["is_correct"] is True


def test_pair_runtime_treats_512_tokens_without_eos_as_length_cap() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, _prompt: str, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }

        def decode(self, _token_ids: list[int], **_kwargs: object) -> str:
            return "The computation is unfinished and currently reaches 2 dollars."

    generated = [41] * 512
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.config = SimpleNamespace(max_new_tokens=512)
    runtime.tokenizer = Tokenizer()
    runtime.wrapper = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(
            generate=lambda **_kwargs: torch.tensor([[10, 11, *generated]])
        ),
    )
    runtime._torch = torch
    runtime.effective_eos_token_ids = (99,)
    runtime.internal_benchmark = "gsm8k"
    runtime.extractor = create_extractor("gsm8k")

    _prompt_ids, continuation_ids, continuation, termination = runtime._generate(
        "prompt"
    )
    answer = runtime._answer(
        continuation,
        "2",
        allow_positional=termination == "eos",
    )

    assert len(continuation_ids) == 512
    assert termination == "length-cap"
    assert answer["method"] == "unextractable"
    assert answer["is_correct"] is False


def test_pair_runtime_keeps_the_math_extractor_native_comparator() -> None:
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.internal_benchmark = "math"
    runtime.extractor = create_extractor("math")

    answer = runtime._answer(r"Reasoning. \boxed{\frac{1}{2}}", r"\frac{1}{2}")

    assert answer == {
        "value": r"\frac{1}{2}",
        "is_extracted": True,
        "is_correct": True,
        "method": "primary:boxed",
        "primary_method": "boxed",
        "confidence": 0.9,
    }


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


def test_cli_forwards_an_explicit_sample_id_cohort(
    monkeypatch: pytest.MonkeyPatch,
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
                "google/gemma-3-12b-it",
                "--benchmark",
                "mmlu",
                "--targeting",
                "attribution-4",
                "--num-edits",
                "4",
                "--sample-ids",
                "data/cohort.json",
                "--output-dir",
                "results/pairs",
            ]
        )
        == 0
    )
    assert captured == [
        PrepareEditedPairsConfig(
            model="google/gemma-3-12b-it",
            benchmark="mmlu",
            targeting="attribution-4",
            num_edits=4,
            sample_ids=Path("data/cohort.json"),
            output_dir=Path("results/pairs"),
        )
    ]


def test_explicit_sample_id_cohort_cannot_be_combined_with_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_ids.*limit"):
        PrepareEditedPairsConfig(
            model="google/gemma-3-12b-it",
            benchmark="mmlu",
            targeting="attribution-4",
            num_edits=4,
            sample_ids=tmp_path / "cohort.json",
            limit=1,
            output_dir=tmp_path / "out",
        )


def _write_sample_id_cohort(
    path: Path,
    sample_ids: list[str],
    *,
    benchmark: str = "gsm8k",
    model: str = "test/model",
) -> Path:
    sample_ids_sha256 = hashlib.sha256(
        json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": "sample-id-cohort/v1",
        "paper_sha256": ("2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0"),
        "cohort_id": "fixture-cohort",
        "benchmark": benchmark,
        "selection": "fixture-order/v1",
        "provenance": "test-fixture",
        "sample_count": len(sample_ids),
        "sample_ids_sha256": sample_ids_sha256,
        "sample_ids": sample_ids,
        "model_samples_per_subset": {model: None},
        "model_selected_sample_counts": {model: len(sample_ids)},
        "model_selected_sample_ids_sha256": {
            model: hashlib.sha256(
                json.dumps(
                    sorted(sample_ids),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_builtin_runtime_leaves_cohort_selection_to_the_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        SimpleNamespace(
            sample_id=sample_id,
            question=f"question {sample_id}",
            choices=["a", "b"],
            correct_answer="A",
            subset="subject",
        )
        for sample_id in ("a", "b")
    ]

    class FakeLoader:
        def load(self) -> list[SimpleNamespace]:
            return samples

    monkeypatch.setattr(loader_module, "create_loader", lambda **_kwargs: FakeLoader())
    cohort_path = _write_sample_id_cohort(tmp_path / "cohort.json", ["b"])
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        sample_ids=cohort_path,
        output_dir=tmp_path / "out",
    )
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.internal_benchmark = "gsm8k"
    runtime._dataset_provenance = {}

    assert runtime.load_samples(config) == samples

    runtime.record_selected_samples([samples[1]])
    assert runtime._dataset_provenance["dataset_sample_count"] == 1


def test_sample_id_cohort_filters_before_pair_execution_and_is_manifested(
    tmp_path: Path,
) -> None:
    cohort_path = _write_sample_id_cohort(tmp_path / "cohort.json", ["b"])
    output_dir = tmp_path / "pairs"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        sample_ids=cohort_path,
        output_dir=output_dir,
    )

    run_prepare_edited_pairs(config, runtime=_FakeRuntime())

    rows = [json.loads(line) for line in (output_dir / "pairs.jsonl").read_text().splitlines()]
    manifest = json.loads((output_dir / "run.json").read_text())
    cohort = load_sample_id_cohort(cohort_path)
    assert [row["sample_id"] for row in rows] == ["b"]
    assert manifest["arguments"]["sample_ids"] == str(cohort_path.resolve())
    assert manifest["arguments"]["limit"] is None
    assert manifest["counts"] == {"discovered": 1, "written": 1, "failed": 0}
    assert manifest["provenance"]["dataset_cohort_rule"] == "explicit-sample-id-cohort/v1"
    assert manifest["provenance"]["sample_id_cohort"] == cohort.provenance_for("test/model")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"benchmark": "mmlu"}, "benchmark"),
        ({"sample_count": 2}, "sample_count"),
        ({"sample_ids_sha256": "0" * 64}, "SHA-256"),
        ({"sample_ids": ["b", "b"]}, "unique"),
    ),
)
def test_malformed_sample_id_cohort_fails_before_gpu_runtime_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, object],
    message: str,
) -> None:
    cohort_path = _write_sample_id_cohort(tmp_path / "cohort.json", ["b"])
    payload = json.loads(cohort_path.read_text(encoding="utf-8"))
    payload.update(mutation)
    cohort_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def runtime_must_not_initialize(config: PrepareEditedPairsConfig) -> None:
        raise AssertionError("invalid cohort must fail before GPU runtime initialization")

    monkeypatch.setattr(
        runtime_module,
        "HuggingFacePairPreparationRuntime",
        runtime_must_not_initialize,
    )
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        sample_ids=cohort_path,
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match=message):
        run_prepare_edited_pairs(config)


def test_sample_id_cohort_rejects_missing_expected_model_ids(tmp_path: Path) -> None:
    cohort_path = _write_sample_id_cohort(tmp_path / "cohort.json", ["missing"])
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        sample_ids=cohort_path,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="expected 1.*selected.*found 0"):
        run_prepare_edited_pairs(config, runtime=_FakeRuntime())


@pytest.mark.parametrize(
    "error",
    (
        FileExistsError("output directory is not empty"),
        ValueError("resume arguments do not match"),
        PairPreparationRunError("one or more items failed"),
    ),
    ids=("existing-output", "invalid-resume", "item-failure"),
)
def test_cli_reports_known_pair_preparation_failures_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fake_run(config: PrepareEditedPairsConfig) -> PrepareEditedPairsResult:
        raise error

    monkeypatch.setattr(cli_module, "run_prepare_edited_pairs", fake_run)

    exit_code = main(
        [
            "prepare-edited-pairs",
            "--model",
            "test/model",
            "--benchmark",
            "gsm8k",
            "--targeting",
            "attribution-4",
            "--num-edits",
            "4",
            "--output-dir",
            "results/pairs",
        ]
    )

    assert exit_code == 1
    assert capsys.readouterr().err == f"prepare-edited-pairs: error: {error}\n"


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
        choice_list_start=19,
    )

    assert [(item.token_index, item.text, item.relevance) for item in candidates] == [
        (1, " Alpha", -0.8),
        (4, " beta", 0.3),
    ]


def test_bare_single_letter_words_are_not_treated_as_option_markers() -> None:
    candidates = eligible_candidates(
        prompt_text="a i\n(A) option",
        token_texts=[" a", " i", " (", "A", ")", " option"],
        relevances=[1.0] * 6,
        offsets=[(0, 1), (2, 3), (4, 5), (5, 6), (6, 7), (8, 14)],
        editable_prompt_start=0,
        editable_prompt_end=14,
        choice_list_start=3,
    )

    assert [item.token_index for item in candidates] == [0, 1, 5]


def test_option_markers_are_filtered_only_inside_the_choice_list() -> None:
    candidates = eligible_candidates(
        prompt_text="A. point b: ratio\n(A) first (B) second",
        token_texts=[" A.", " point", " b:", " ratio", " (A)", " first", " (B)", " second"],
        relevances=[1.0] * 8,
        offsets=[
            (0, 2),
            (3, 8),
            (9, 11),
            (12, 17),
            (18, 21),
            (22, 27),
            (28, 31),
            (32, 38),
        ],
        editable_prompt_start=0,
        editable_prompt_end=38,
        choice_list_start=17,
    )

    assert [item.token_index for item in candidates] == [0, 1, 2, 3, 5, 7]


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


def test_attribution_targets_the_distribution_after_the_first_cot_token() -> None:
    captured: dict[str, object] = {}

    class FakeAnalyzer:
        def compute_relevance(
            self, input_ids: torch.Tensor, *, target_position: int
        ) -> torch.Tensor:
            captured["input_ids"] = input_ids.tolist()
            captured["target_position"] = target_position
            return torch.tensor([0.1, 0.2, 0.3, 0.4])

    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime._torch = torch
    runtime.wrapper = SimpleNamespace(device=torch.device("cpu"))
    runtime.analyzer = FakeAnalyzer()

    relevance = runtime._relevance_after_first_cot([10, 11], [12, 13])

    assert captured == {
        "input_ids": [[10, 11, 12, 13]],
        # Causal logits at the first generated token (index 2) are the
        # distribution immediately after that token, as specified in §3.1.
        "target_position": 2,
    }
    assert relevance == pytest.approx([0.1, 0.2])


class _SingleTokenPairTokenizer:
    def __call__(self, text: str, **_kwargs: object) -> dict[str, object]:
        if text == "1234":
            return {"input_ids": [1], "offset_mapping": [(0, len(text))]}
        return {
            "input_ids": [3, 4],
            "offset_mapping": [(0, len(text)), (0, len(text))],
        }

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return {1: "1234", 2: " step", 3: " alpha", 4: " alpha"}[token_ids[0]]


def _pair_runtime_for_prompt(prompt: str) -> tuple[HuggingFacePairPreparationRuntime, list[str]]:
    runtime = object.__new__(HuggingFacePairPreparationRuntime)
    runtime.tokenizer = _SingleTokenPairTokenizer()
    generated_prompts: list[str] = []
    prompt_result = SimpleNamespace(
        question_start_in_full=0,
        question_end_in_full=len(prompt),
        question_with_choices_end=len(prompt),
    )
    runtime._prompt = lambda _sample: (prompt_result, prompt)

    def generate(text: str) -> tuple[list[int], list[int], str, str]:
        generated_prompts.append(text)
        prompt_ids = runtime.tokenizer(text)["input_ids"]
        return list(prompt_ids), [2], f"continuation:{text}", "eos"

    runtime._generate = generate
    runtime._relevance_after_first_cot = lambda prompt_ids, _generated_ids: [
        1.0 - 0.5 * index for index in range(len(prompt_ids))
    ]
    runtime._answer = lambda continuation, _correct, **_kwargs: {
        "value": continuation,
        "is_extracted": True,
        "is_correct": False,
        "method": "fixture",
        "confidence": 1.0,
    }
    return runtime, generated_prompts


def test_runtime_retains_an_item_with_no_eligible_edit() -> None:
    runtime, generated_prompts = _pair_runtime_for_prompt("1234")
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=Path("unused"),
    )
    sample = SimpleNamespace(
        sample_id="math-only",
        question="1234",
        choices=None,
        correct_answer="4",
        subset=None,
    )

    record = runtime.prepare_pair(sample, config)

    assert generated_prompts == ["1234"]
    assert record["num_candidates"] == 0
    assert record["num_target_attempts"] == 0
    assert record["target_attempts"] == []
    assert record["num_aligned_words"] == 0
    assert record["aligned_words"] == []
    assert record["edited"]["prompt"] == record["clean"]["prompt"]
    assert record["edited"]["continuation"] == record["clean"]["continuation"]
    assert record["answer_changed"] is False


def test_runtime_retains_attempts_when_edits_have_no_final_text_difference() -> None:
    runtime, generated_prompts = _pair_runtime_for_prompt("alpha")
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=2,
        output_dir=Path("unused"),
    )
    sample = SimpleNamespace(
        sample_id="net-zero-109",
        question="alpha",
        choices=None,
        correct_answer="A",
        subset=None,
    )

    record = runtime.prepare_pair(sample, config)

    assert generated_prompts == ["alpha"]
    assert record["num_candidates"] == 2
    assert record["num_target_attempts"] == 2
    assert len(record["target_attempts"]) == 2
    assert {attempt["target_token_index"] for attempt in record["target_attempts"]} == {0, 1}
    assert [attempt["edited_token_text"] for attempt in record["target_attempts"]] == [
        "aopha",
        "alpha",
    ]
    assert record["num_aligned_words"] == 0
    assert record["aligned_words"] == []
    assert record["edited"]["prompt"] == record["clean"]["prompt"]
    assert record["answer_changed"] is False


def test_runtime_rejects_changed_text_without_an_aligned_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = _pair_runtime_for_prompt("alpha")
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=1,
        output_dir=Path("unused"),
    )
    sample = SimpleNamespace(
        sample_id="broken-alignment",
        question="alpha",
        choices=None,
        correct_answer="A",
        subset=None,
    )
    monkeypatch.setattr(runtime_module, "build_aligned_words", lambda **_kwargs: ())

    with pytest.raises(PairProtocolError, match="changed.*no alignable"):
        runtime.prepare_pair(sample, config)


@pytest.mark.parametrize(
    ("seed", "expected"),
    (
        (0, ("substitution", "fat", 0, "f")),
        (1, ("duplication", "caat", 1, "a")),
        (4, ("deletion", "at", 0, None)),
    ),
)
def test_seeded_character_edit_preserves_the_reported_seeded_operation(
    seed: int,
    expected: tuple[str, str, int, str | None],
) -> None:
    edit = seeded_character_edit("cat", seed)

    assert edit is not None
    assert (edit.operation, edit.edited, edit.character_index, edit.new_character) == expected


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
    assert manifest["outputs"] == {
        "pairs": {
            "path": "pairs.jsonl",
            "sha256": hashlib.sha256(result.pairs_path.read_bytes()).hexdigest(),
            "records": 2,
        }
    }
    assert result.pairs_path == output_dir / "pairs.jsonl"


def test_completed_resume_returns_without_loading_samples(tmp_path: Path) -> None:
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


def test_completed_resume_rejects_a_legacy_cohort_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "legacy-cohort"
    config = PrepareEditedPairsConfig(
        model="google/gemma-3-12b-it",
        benchmark="mmlu",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )
    run_prepare_edited_pairs(config, runtime=_FakeRuntime())
    manifest_path = output_dir / "run.json"
    manifest = json.loads(manifest_path.read_text())
    assert "dataset_cohort_rule" not in manifest["provenance"]

    monkeypatch.setattr(
        runtime_module,
        "preload_provenance",
        lambda config: {
            "model": config.model,
            "benchmark_dataset_loader": "mmlu",
            "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
            "dataset_samples_per_subset": 100,
            "random_seed_algorithm": "sha256-first-64-bits/v1",
            "generation_protocol": "explicit-greedy-generation/v1",
            "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
            "target_position": "maximum-logit-after-first-cot-token",
            "alignment": "actual-edited-word-final-token",
            "historical_compatibility_notes": [],
        },
    )

    def model_must_not_load(config: PrepareEditedPairsConfig) -> None:
        raise AssertionError("legacy cohort mismatch must fail before model loading")

    monkeypatch.setattr(
        runtime_module,
        "HuggingFacePairPreparationRuntime",
        model_must_not_load,
    )

    with pytest.raises(ValueError, match="resume provenance.*before model loading"):
        run_prepare_edited_pairs(replace(config, resume=True))


def test_completed_resume_rejects_legacy_token_count_termination_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "legacy-generation-termination"
    config = PrepareEditedPairsConfig(
        model="google/gemma-3-4b-it",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )
    current_protocol = {
        "model": config.model,
        "benchmark_dataset_loader": "gsm8k",
        "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
        "dataset_samples_per_subset": None,
        "random_seed_algorithm": "sha256-first-64-bits/v1",
        "generation_protocol": "explicit-greedy-generation/v1",
        "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
        "target_position": "maximum-logit-after-first-cot-token",
        "alignment": "actual-edited-word-final-token",
        "historical_compatibility_notes": [],
    }

    class LegacyRuntime(_FakeRuntime):
        def provenance(self) -> dict[str, object]:
            legacy = dict(current_protocol)
            legacy.pop("generation_termination_protocol")
            return legacy

    run_prepare_edited_pairs(config, runtime=LegacyRuntime())
    monkeypatch.setattr(
        runtime_module,
        "preload_provenance",
        lambda _config: current_protocol,
    )

    def model_must_not_load(config: PrepareEditedPairsConfig) -> None:
        raise AssertionError("legacy termination protocol must fail before model loading")

    monkeypatch.setattr(
        runtime_module,
        "HuggingFacePairPreparationRuntime",
        model_must_not_load,
    )

    with pytest.raises(ValueError, match="resume provenance.*before model loading"):
        run_prepare_edited_pairs(replace(config, resume=True))


def test_completed_resume_ignores_environment_only_provenance_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "different-environment"
    config = PrepareEditedPairsConfig(
        model="google/gemma-3-12b-it",
        benchmark="mmlu",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )

    protocol = {
        "model": config.model,
        "benchmark_dataset_loader": "mmlu",
        "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
        "dataset_samples_per_subset": 100,
        "random_seed_algorithm": "sha256-first-64-bits/v1",
        "generation_protocol": "explicit-greedy-generation/v1",
        "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
        "target_position": "maximum-logit-after-first-cot-token",
        "alignment": "actual-edited-word-final-token",
        "historical_compatibility_notes": [],
    }

    class InitialRuntime(_FakeRuntime):
        def provenance(self) -> dict[str, object]:
            return {**protocol, "gpu_names": ["old-gpu"]}

    expected = run_prepare_edited_pairs(config, runtime=InitialRuntime())
    monkeypatch.setattr(
        runtime_module,
        "preload_provenance",
        lambda config: {**protocol, "gpu_names": ["new-gpu"]},
    )

    def model_must_not_load(config: PrepareEditedPairsConfig) -> None:
        raise AssertionError("completed resume must not load the model")

    monkeypatch.setattr(
        runtime_module,
        "HuggingFacePairPreparationRuntime",
        model_must_not_load,
    )

    resumed = run_prepare_edited_pairs(replace(config, resume=True))

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


def test_completed_resume_rejects_a_missing_published_pairs_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing-pairs"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )
    run_prepare_edited_pairs(config, runtime=_FakeRuntime())
    (output_dir / "pairs.jsonl").unlink()

    class RuntimeMustNotLoad(_FakeRuntime):
        def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
            raise AssertionError("a completed run without pairs must fail before runtime access")

    with pytest.raises(ValueError, match="completed.*pairs.jsonl"):
        run_prepare_edited_pairs(
            replace(config, resume=True),
            runtime=RuntimeMustNotLoad(),
        )


def test_completed_resume_rejects_a_changed_published_pairs_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "changed-pairs"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )
    result = run_prepare_edited_pairs(config, runtime=_FakeRuntime())
    result.pairs_path.write_text(
        result.pairs_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    class RuntimeMustNotLoad(_FakeRuntime):
        def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
            raise AssertionError("changed completed output must fail before runtime access")

    with pytest.raises(ValueError, match="pairs.*SHA|hash|changed"):
        run_prepare_edited_pairs(
            replace(config, resume=True),
            runtime=RuntimeMustNotLoad(),
        )


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


@pytest.mark.parametrize("create_empty_directory", (False, True))
def test_runner_rejects_resume_without_a_manifest_before_runtime_access(
    tmp_path: Path,
    create_empty_directory: bool,
) -> None:
    output_dir = tmp_path / "missing-resume-run"
    if create_empty_directory:
        output_dir.mkdir()

    class RuntimeMustNotLoad(_FakeRuntime):
        def load_samples(self, config: PrepareEditedPairsConfig) -> list[dict[str, str]]:
            raise AssertionError("resume without run.json must fail before runtime access")

    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
        resume=True,
    )

    with pytest.raises(ValueError, match="cannot resume without.*run.json"):
        run_prepare_edited_pairs(config, runtime=RuntimeMustNotLoad())

    assert not (output_dir / "run.json").exists()
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


def test_runner_stays_running_while_processing_after_an_item_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "running-after-failure"
    observed_statuses: list[str] = []

    class ObservingRuntime(_FakeRuntime):
        def prepare_pair(
            self, sample: dict[str, str], config: PrepareEditedPairsConfig
        ) -> dict[str, object]:
            if sample["sample_id"] == "a":
                raise RuntimeError("injected failure")
            observed_statuses.append(json.loads((output_dir / "run.json").read_text())["status"])
            return super().prepare_pair(sample, config)

    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )

    with pytest.raises(PairPreparationRunError):
        run_prepare_edited_pairs(config, runtime=ObservingRuntime())

    assert observed_statuses == ["running"]
    assert json.loads((output_dir / "run.json").read_text())["status"] == "failed"


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


def test_runner_rejects_changed_environment_before_loading_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InitialRuntime(_FakeRuntime):
        def prepare_pair(
            self, sample: dict[str, str], config: PrepareEditedPairsConfig
        ) -> dict[str, object]:
            if sample["sample_id"] == "b":
                raise RuntimeError("injected failure")
            return super().prepare_pair(sample, config)

        def provenance(self) -> dict[str, str]:
            return {"python": "old", "model": "test/model"}

    output_dir = tmp_path / "preload-provenance"
    config = PrepareEditedPairsConfig(
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        num_edits=4,
        output_dir=output_dir,
    )
    with pytest.raises(PairPreparationRunError):
        run_prepare_edited_pairs(config, runtime=InitialRuntime())

    monkeypatch.setattr(
        runtime_module,
        "preload_provenance",
        lambda config: {"python": "new", "model": config.model},
        raising=False,
    )

    def model_must_not_load(config: PrepareEditedPairsConfig) -> None:
        raise AssertionError("environment mismatch must fail before model loading")

    monkeypatch.setattr(runtime_module, "HuggingFacePairPreparationRuntime", model_must_not_load)

    with pytest.raises(ValueError, match="provenance.*before model loading"):
        run_prepare_edited_pairs(replace(config, resume=True))


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
