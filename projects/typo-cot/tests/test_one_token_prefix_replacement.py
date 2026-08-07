"""Final-PDF contracts for the one-token clean-prefix diagnostic."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import typo_cot.cli as cli_module
import typo_cot.experiments.one_token_prefix_replacement as token_api
import typo_cot.experiments.one_token_prefix_replacement.runner as token_runner
import typo_cot.experiments.one_token_prefix_replacement.runtime as token_runtime
from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.clean_prefix_scan.source import load_extension_source

MODEL = "google/gemma-3-1b-it"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aligned_word() -> dict[str, object]:
    return {
        "word_index": 0,
        "clean_text": "clean",
        "edited_text": "clena",
        "clean_editable_span": {"start": 0, "end": 5},
        "edited_editable_span": {"start": 0, "end": 5},
        "clean_prompt_span": {"start": 0, "end": 5},
        "edited_prompt_span": {"start": 0, "end": 5},
        "target_ranks": [1],
        "target_token_indices": [1],
        "clean_token_indices": [1],
        "edited_token_indices": [1],
        "clean_final_token": 1,
        "edited_final_token": 1,
    }


def _pair(sample_id: str, *, targeting: str) -> dict[str, object]:
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": MODEL,
        "benchmark": "gsm8k",
        "targeting": targeting,
        "seed": 42,
        "num_edits_requested": 4,
        "num_target_attempts": 1,
        "num_aligned_words": 1,
        "gold_answer": "2",
        "clean": {
            "prompt": f"clean prompt {sample_id}",
            "prompt_token_count": 2,
            "continuation": "First reason. Then calculate. The answer is 2.",
            "answer": {
                "value": "2",
                "is_extracted": True,
                "is_correct": True,
                "method": "fixture",
                "primary_method": "fixture",
                "confidence": 1.0,
            },
        },
        "edited": {
            "prompt": f"edited prompt {sample_id}",
            "prompt_token_count": 2,
            "continuation": "Wrong reason. The answer is 3.",
            "answer": {
                "value": "3",
                "is_extracted": True,
                "is_correct": False,
                "method": "fixture",
                "primary_method": "fixture",
                "confidence": 1.0,
            },
        },
        "answer_changed": True,
        "target_attempts": [{"token_index": 1, "operation": "substitution"}],
        "aligned_words": [_aligned_word()],
    }


def _write_pair_source(root: Path, *, targeting: str, sample_id: str) -> Path:
    root.mkdir(parents=True)
    path = root / "pairs.jsonl"
    path.write_text(
        json.dumps(_pair(sample_id, targeting=targeting), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "run.json",
        {
            "schema_version": "prepare-edited-pairs-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "prepare-edited-pairs",
            "status": "completed",
            "arguments": {
                "model": MODEL,
                "benchmark": "gsm8k",
                "targeting": targeting,
                "num_edits": 4,
                "seed": 42,
                "max_new_tokens": 512,
                "gpu_id": "0",
                "limit": None,
                "output_dir": str(root.resolve()),
            },
            "counts": {"discovered": 1, "written": 1, "failed": 0},
            "failures": [],
            "decoding": {
                "strategy": "greedy",
                "dtype": "bfloat16",
                "padding_side": "left",
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
            },
            "provenance": {
                "model": MODEL,
                "model_revision": "source-revision",
                "dataset_sample_count": 1,
                "dataset_records_sha256": "d" * 64,
                "generation_protocol": "explicit-greedy-generation/v1",
            },
        },
    )
    return path


def _config(tmp_path: Path, *, resume: bool = False, adjacent: bool = True) -> object:
    attribution = _write_pair_source(
        tmp_path / "attribution", targeting="attribution-4", sample_id="sample-a"
    )
    random = _write_pair_source(tmp_path / "random", targeting="random-4", sample_id="sample-r")
    return token_api.OneTokenPrefixReplacementConfig(
        model=MODEL,
        benchmark="gsm8k",
        cohort="extension",
        fixed_window_run=None,
        pairs=(attribution, random),
        max_pairs=150,
        position_controls=("distant", "adjacent") if adjacent else ("distant",),
        output_dir=tmp_path / "output",
        gpu_id="0",
        limit=1,
        resume=resume,
    )


class _Runtime:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls: list[tuple[int, int]] = []

    def provenance(self) -> Mapping[str, object]:
        return {
            "runtime": "one-token-prefix-replacement-huggingface/v1",
            "python": "fixture",
            "torch": "fixture",
            "transformers": "fixture",
            "accelerate": "fixture",
            "numpy": "fixture",
            "requested_revision": "source-revision",
            "model_revision": "source-revision",
            "tokenizer_revision": "source-revision",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "cuda": "fixture",
            "gpu_name": "fixture-gpu",
            "cuda_visible_devices": "0",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_new_tokens": 512,
                "use_cache": True,
                "return_dict_in_generate": False,
                "output_scores": False,
                "padding_side": "left",
                "batch_size": 1,
            },
            "effective_eos_token_ids": [99],
            "effective_eos_token_ids_source": "model-generation-config",
            "answer_decoding": "generated-token-ids-only/v1",
            "answer_extraction": "primary-then-empty-only-fallback-cap-aware/v1",
            "pre_answer_boundary": "first-submitted-[Tt]he-answer-is/v1",
            "profile_input": "tokenized-full-clean-cot-under-clean-and-edited-prompts/v1",
            "generation_input": "clean-prompt-ids-plus-clean-cot-before-site-plus-one-token/v1",
            "token_admissibility": {
                "implementation": "submitted-producer-tokenizer-candidate-pool/v1",
                "model_logit_size": 1000,
                "tokenizer_vocab_entries": 900,
                "n_real_tokenizer_ids_in_logits": 900,
                "n_special_ids": 2,
                "n_marker_ids": 1,
                "n_admissible_ids": 897,
                "admissible_token_ids_sha256_algorithm": "sorted-decimal-lines/v1",
                "admissible_token_ids_sha256": "a" * 64,
                "marker_regex": (
                    r"^(?:<unused\d+>|<\|reserved_special_token_\d+\|>"
                    r"|\[control_\d+\]|\[unused\d+\])$"
                ),
            },
            "implementation_code_identity": token_runtime.implementation_code_identity(),
        }

    def prepare_pair(self, _pair: Mapping[str, object]) -> object:
        cot = tuple(range(10, 18))
        return token_api.OneTokenInputPlan(
            clean_prompt_ids=(1, 2),
            edited_prompt_ids=(3, 4),
            clean_full_ids=(1, 2, *cot),
            edited_full_ids=(3, 4, *cot),
            clean_cot_ids=cot,
        )

    def profile_pair(self, _plan: object) -> object:
        return token_api.OneTokenProfile(
            clean_to_edited_kl=(10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0),
            clean_token_rank_under_clean=(1, 1, 1, 1, 1, 1, 1, 1),
            clean_token_rank_under_edited=(1, 2, 2, 2, 2, 2, 2, 2),
            edited_top1_ids=tuple(range(100, 108)),
            edited_top1_is_admissible=(True,) * 8,
        )

    def generate_arm(
        self,
        plan: object,
        *,
        position: int,
        forced_token_id: int,
        gold_answer: str,
    ) -> object:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("fixture interruption")
        self.calls.append((position, forced_token_id))
        assert gold_answer == "2"
        assert plan.generation_input_ids(position, forced_token_id)[-1] == forced_token_id
        wrong = forced_token_id in {101, 106}
        value = "3" if wrong else "2"
        extraction = extract_with_fallback(
            f"The answer is {value}.", benchmark="gsm8k", correct_answer="2"
        )
        return token_api.OneTokenGeneration(
            token_ids=(99,),
            text=f"The answer is {value}.",
            value=extraction.value,
            is_extracted=extraction.is_extracted,
            is_correct=extraction.is_correct,
            method=extraction.method,
            primary_method=extraction.primary_method,
            stop_reason="eos_token",
            stop_token_id=99,
        )


class _NoRuntimeUse:
    def __getattribute__(self, name: str) -> object:
        if name.startswith("__"):
            return super().__getattribute__(name)
        raise AssertionError(f"completed resume touched runtime method {name}")


def test_catalog_exposes_the_implemented_operation_and_four_outputs() -> None:
    spec = get_experiment("one-token-prefix-replacement")

    assert spec.status == "implemented"
    assert spec.required_arguments == (
        "--model",
        "--benchmark",
        "--cohort",
        "--position-controls",
        "--output-dir",
    )
    assert spec.outputs == (
        "one_token_records.jsonl",
        "pair_status_records.jsonl",
        "one_token_summary.json",
        "run.json",
    )


def test_public_api_exposes_position_profile_runtime_and_runner_contracts() -> None:
    expected = {
        "POSITION_CONTROLS",
        "OneTokenInputPlan",
        "OneTokenProfile",
        "OneTokenGeneration",
        "OneTokenPrefixReplacementConfig",
        "OneTokenPrefixReplacementResult",
        "OneTokenPrefixReplacementRunError",
        "choose_distant_positions",
        "choose_adjacent_position",
        "run_one_token_prefix_replacement",
    }

    assert expected.issubset(set(token_api.__all__))
    signature = inspect.signature(token_api.OneTokenPrefixReplacementRuntime.generate_arm)
    assert list(signature.parameters) == [
        "self",
        "plan",
        "position",
        "forced_token_id",
        "gold_answer",
    ]


def test_runtime_code_identity_fingerprints_the_installed_python_source_tree() -> None:
    identity = token_runtime.implementation_code_identity()

    assert identity["algorithm"] == "one-token-executable-code-bundle-sha256/v1"
    assert isinstance(identity["python_file_count"], int)
    assert identity["python_file_count"] > 0
    assert isinstance(identity["sha256"], str)
    assert len(identity["sha256"]) == 64
    int(identity["sha256"], 16)


def test_selected_position_excludes_edited_top1_clean_tokens_and_breaks_ties_by_index() -> None:
    profile = token_api.OneTokenProfile(
        clean_to_edited_kl=(99.0, 8.0, 8.0, 2.0, 1.0),
        clean_token_rank_under_clean=(1, 1, 1, 1, 1),
        clean_token_rank_under_edited=(1, 2, 2, 2, 2),
        edited_top1_ids=(30, 31, 32, 33, 34),
        edited_top1_is_admissible=(True,) * 5,
    )

    selected = token_api.choose_distant_positions(profile, min_distance=3)

    assert selected.selected_position == 1
    assert selected.distant_position == 4
    assert selected.selected_edited_top1_id == 31
    assert selected.distant_edited_top1_id == 34


def test_distant_control_uses_median_low_kl_and_smallest_index_for_exact_ties() -> None:
    profile = token_api.OneTokenProfile(
        clean_to_edited_kl=(20.0, 10.0, 3.0, 7.0, 5.0, 3.0, 9.0, 1.0),
        clean_token_rank_under_clean=(1,) * 8,
        clean_token_rank_under_edited=(2,) * 8,
        edited_top1_ids=tuple(range(100, 108)),
        edited_top1_is_admissible=(True,) * 8,
    )

    selected = token_api.choose_distant_positions(profile, min_distance=3)

    assert selected.selected_position == 0
    # candidates 3..7 have KL [7,5,3,9,1], median-low is 5 at position 4
    assert selected.distant_position == 4


def test_adjacent_control_is_nearest_strictly_lower_kl_and_hash_tie_is_stable() -> None:
    kl = (1.0, 4.0, 9.0, 4.0, 1.0)
    first = token_api.choose_adjacent_position(
        kl,
        selected_position=2,
        tie_key="gemma1b_gsm8k|attribution-4|sample-a",
    )
    second = token_api.choose_adjacent_position(
        kl,
        selected_position=2,
        tie_key="gemma1b_gsm8k|attribution-4|sample-a",
    )

    assert first == second
    assert first in {1, 3}
    digest = hashlib.sha256(b"gemma1b_gsm8k|attribution-4|sample-a").hexdigest()
    prefer_right = int(digest, 16) % 2
    assert first == (3 if prefer_right else 1)


def test_input_plan_uses_direct_clean_prompt_prefix_and_one_forced_id() -> None:
    plan = token_api.OneTokenInputPlan(
        clean_prompt_ids=(1, 2),
        edited_prompt_ids=(3, 4, 5),
        clean_full_ids=(1, 2, 10, 11, 12),
        edited_full_ids=(3, 4, 5, 10, 11, 12),
        clean_cot_ids=(10, 11, 12),
    )

    assert plan.profile_clean_input_ids == (1, 2, 10, 11, 12)
    assert plan.profile_edited_input_ids == (3, 4, 5, 10, 11, 12)
    assert plan.generation_input_ids(1, 99) == (1, 2, 10, 99)
    assert plan.generation_input_ids(0, 99) == (1, 2, 99)

    with pytest.raises(ValueError, match="exact clean prompt boundary"):
        token_api.OneTokenInputPlan(
            clean_prompt_ids=(1, 2),
            edited_prompt_ids=(3, 4),
            clean_full_ids=(1, 9, 10),
            edited_full_ids=(3, 4, 10),
            clean_cot_ids=(10,),
        )


def test_adjacent_tie_key_uses_the_submitted_producer_identity_codes(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.setting_id == "gemma1b_gsm8k"
    assert (
        config.adjacent_tie_key(targeting="attribution-4", sample_id="gsm8k_00431")
        == "gemma1b_gsm8k|lxt4|gsm8k_00431"
    )
    assert (
        config.adjacent_tie_key(targeting="random-4", sample_id="gsm8k_00431")
        == "gemma1b_gsm8k|rnd4|gsm8k_00431"
    )


def _arm(
    correct: bool,
    *,
    noop: bool = False,
    token_id: int = 101,
    admissible: bool | None = True,
) -> dict[str, object]:
    return {
        "generation": {"is_correct": correct},
        "is_noop": noop,
        "forced_token_id": token_id,
        "token_is_admissible": admissible,
    }


def test_event_classifier_keeps_table10_factorial_and_adjacent_denominators_distinct() -> None:
    arms = {
        "selected_keep": _arm(True),
        "selected_from_selected": _arm(False),
        "selected_from_distant": _arm(True),
        "distant_keep": _arm(True),
        "distant_from_selected": _arm(False),
        "distant_from_distant": _arm(False),
        "adjacent_keep": _arm(True),
        "adjacent_from_selected": _arm(True),
    }

    events = token_runner.classify_one_token_events(
        arms,
        selected_before_control=True,
        adjacent_requested=True,
    )

    assert events["table10"] == {
        "eligible": True,
        "selected_correct_to_incorrect": True,
        "control_correct_to_incorrect": True,
    }
    assert events["distant_factorial"] == {
        "eligible": True,
        "selected_loss_events": 1,
        "control_loss_events": 2,
        "event_opportunities_per_position": 2,
        "selected_before_control": True,
    }
    assert events["adjacent"] == {
        "eligible": True,
        "selected_correct_to_incorrect": True,
        "control_correct_to_incorrect": False,
    }


def test_factorial_requires_all_four_non_noops_but_not_distinct_source_token_ids() -> None:
    arms = {
        "selected_keep": _arm(True),
        "selected_from_selected": _arm(False),
        "selected_from_distant": _arm(False),
        "distant_keep": _arm(True),
        "distant_from_selected": _arm(False),
        "distant_from_distant": _arm(False, noop=True),
    }

    noop = token_runner.classify_one_token_events(
        arms, selected_before_control=True, adjacent_requested=False
    )
    identical = token_runner.classify_one_token_events(
        {name: _arm(True) for name in arms},
        selected_before_control=True,
        adjacent_requested=False,
    )

    assert noop["distant_factorial"]["eligible"] is False
    assert identical["distant_factorial"]["eligible"] is True


def test_submitted_producer_factorial_adds_distinct_and_admissible_token_guards() -> None:
    arms = {
        "selected_keep": _arm(True, token_id=10, admissible=None),
        "selected_from_selected": _arm(False, token_id=101),
        "selected_from_distant": _arm(True, token_id=102),
        "distant_keep": _arm(True, token_id=20, admissible=None),
        "distant_from_selected": _arm(False, token_id=101),
        "distant_from_distant": _arm(False, token_id=102),
    }

    eligible = token_runner.classify_one_token_events(
        arms, selected_before_control=True, adjacent_requested=False
    )
    same_token = token_runner.classify_one_token_events(
        {
            **arms,
            "selected_from_distant": _arm(True, token_id=101),
            "distant_from_distant": _arm(False, token_id=101),
        },
        selected_before_control=True,
        adjacent_requested=False,
    )
    inadmissible = token_runner.classify_one_token_events(
        {
            **arms,
            "selected_from_distant": _arm(True, token_id=102, admissible=False),
            "distant_from_distant": _arm(False, token_id=102, admissible=False),
        },
        selected_before_control=True,
        adjacent_requested=False,
    )

    assert eligible["distant_factorial"]["eligible"] is True
    assert eligible["distant_factorial_submitted_producer"]["eligible"] is True
    assert same_token["distant_factorial"]["eligible"] is True
    assert same_token["distant_factorial_submitted_producer"] == {
        "eligible": False,
        "selected_loss_events": 0,
        "control_loss_events": 0,
        "event_opportunities_per_position": 2,
        "selected_before_control": True,
        "additional_guard": {
            "selected_and_control_tokens_distinct": False,
            "selected_token_admissible": True,
            "control_token_admissible": True,
        },
        "exclusion_reasons": ["selected-and-control-tokens-identical"],
    }
    assert inadmissible["distant_factorial_submitted_producer"]["eligible"] is False
    assert inadmissible["distant_factorial_submitted_producer"]["exclusion_reasons"] == [
        "control-token-not-admissible"
    ]


def test_aggregate_reports_paper_literal_and_submitted_producer_denominators() -> None:
    distinct = {
        "selected_keep": _arm(True, token_id=10, admissible=None),
        "selected_from_selected": _arm(False, token_id=101),
        "selected_from_distant": _arm(True, token_id=102),
        "distant_keep": _arm(True, token_id=20, admissible=None),
        "distant_from_selected": _arm(False, token_id=101),
        "distant_from_distant": _arm(False, token_id=102),
    }
    identical = {
        **distinct,
        "selected_from_distant": _arm(True, token_id=101),
        "distant_from_distant": _arm(False, token_id=101),
    }
    rows = [
        token_runner.classify_one_token_events(
            arms, selected_before_control=True, adjacent_requested=False
        )
        for arms in (distinct, identical)
    ]

    summary = token_runner.aggregate_one_token_events(rows, adjacent_requested=False)

    assert summary["distant_factorial"]["paired_eligible"] == 2
    assert summary["distant_factorial_submitted_producer"]["paired_eligible"] == 1
    assert summary["distant_factorial_submitted_producer"]["attrition_from_paper_literal"] == {
        "count": 1,
        "reasons": {"selected-and-control-tokens-identical": 1},
    }


def test_generation_contract_enforces_cap_and_effective_eos_semantics() -> None:
    common = {
        "text": "The answer is 2.",
        "value": "2",
        "is_extracted": True,
        "is_correct": True,
        "method": "fixture",
        "primary_method": "fixture",
    }
    with pytest.raises(ValueError, match="cannot exceed 512"):
        token_api.OneTokenGeneration(
            **common,
            token_ids=(1,) * 512 + (99,),
            stop_reason="eos_token",
            stop_token_id=99,
        )
    with pytest.raises(ValueError, match="is_extracted"):
        token_api.OneTokenGeneration(
            **{**common, "value": "", "is_extracted": True},
            token_ids=(99,),
            stop_reason="eos_token",
            stop_token_id=99,
        )

    continued = token_api.OneTokenGeneration(
        **common,
        token_ids=(99, 1, 99),
        stop_reason="eos_token",
        stop_token_id=99,
    )
    with pytest.raises(ValueError, match="continued after"):
        token_runner._validate_generation(
            continued,
            benchmark="gsm8k",
            gold_answer="2",
            eos_ids=(99,),
        )


def test_config_rejects_mixed_sources_and_nonpaper_adjacent_settings(tmp_path: Path) -> None:
    common = {
        "model": MODEL,
        "benchmark": "gsm8k",
        "output_dir": tmp_path / "output",
        "gpu_id": "0",
    }
    with pytest.raises(ValueError, match="primary"):
        token_api.OneTokenPrefixReplacementConfig(
            **common,
            cohort="primary",
            fixed_window_run=None,
            pairs=(tmp_path / "pairs.jsonl",),
            max_pairs=None,
            position_controls=("distant",),
        )
    with pytest.raises(ValueError, match="distant"):
        token_api.OneTokenPrefixReplacementConfig(
            **common,
            cohort="extension",
            fixed_window_run=None,
            pairs=(tmp_path / "a", tmp_path / "b"),
            max_pairs=150,
            position_controls=("adjacent",),
        )
    with pytest.raises(ValueError, match="prespecified"):
        token_api.OneTokenPrefixReplacementConfig(
            model="google/gemma-3-4b-it",
            benchmark="mmlu",
            cohort="extension",
            fixed_window_run=None,
            pairs=(tmp_path / "a", tmp_path / "b"),
            max_pairs=150,
            position_controls=("distant", "adjacent"),
            output_dir=tmp_path / "other",
            gpu_id="0",
        )
    with pytest.raises(ValueError, match="primary.*Gemma-3-4B/GSM8K"):
        token_api.OneTokenPrefixReplacementConfig(
            model=MODEL,
            benchmark="gsm8k",
            cohort="primary",
            fixed_window_run=tmp_path / "fixed",
            pairs=(),
            max_pairs=None,
            position_controls=("distant",),
            output_dir=tmp_path / "wrong-primary",
            gpu_id="0",
        )
    with pytest.raises(ValueError, match="max_pairs=150"):
        token_api.OneTokenPrefixReplacementConfig(
            model=MODEL,
            benchmark="gsm8k",
            cohort="extension",
            fixed_window_run=None,
            pairs=(tmp_path / "a" / "pairs.jsonl", tmp_path / "b" / "pairs.jsonl"),
            max_pairs=149,
            position_controls=("distant",),
            output_dir=tmp_path / "output-149",
            gpu_id="0",
        )
    with pytest.raises(ValueError, match="single non-negative integer"):
        token_api.OneTokenPrefixReplacementConfig(
            model=MODEL,
            benchmark="gsm8k",
            cohort="extension",
            fixed_window_run=None,
            pairs=(tmp_path / "a" / "pairs.jsonl", tmp_path / "b" / "pairs.jsonl"),
            max_pairs=150,
            position_controls=("distant",),
            output_dir=tmp_path / "output-gpus",
            gpu_id="0,1",
        )
    with pytest.raises(ValueError, match="separate from every source"):
        token_api.OneTokenPrefixReplacementConfig(
            model=MODEL,
            benchmark="gsm8k",
            cohort="extension",
            fixed_window_run=None,
            pairs=(tmp_path / "a" / "pairs.jsonl", tmp_path / "b" / "pairs.jsonl"),
            max_pairs=150,
            position_controls=("distant",),
            output_dir=tmp_path / "a" / "output",
            gpu_id="0",
        )


def test_runner_writes_exactly_four_outputs_and_separate_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime = _Runtime()

    result = token_api.run_one_token_prefix_replacement(config, runtime=runtime)
    names = sorted(path.name for path in config.output_dir.iterdir())
    records = [json.loads(line) for line in result.records_path.read_text().splitlines()]
    statuses = [
        json.loads(line) for line in result.pair_status_records_path.read_text().splitlines()
    ]
    summary = json.loads(result.summary_path.read_text())
    manifest = json.loads(result.run_path.read_text())

    assert names == [
        "one_token_records.jsonl",
        "one_token_summary.json",
        "pair_status_records.jsonl",
        "run.json",
    ]
    assert result.records == 1
    assert len(records) == 1
    assert len(statuses) == 2
    assert records[0]["positions"] == {"selected": 1, "distant": 6, "adjacent": 2}
    assert len(records[0]["arms"]) == 8
    assert summary["counts"]["source_pairs"] == 2
    assert summary["counts"]["selected_full"] == 2
    assert summary["counts"]["executed"] == 1
    assert summary["metrics"]["table10"]["paired_eligible"] == 1
    assert summary["metrics"]["distant_factorial"]["paired_eligible"] == 1
    assert summary["metrics"]["distant_factorial_submitted_producer"]["paired_eligible"] == 1
    assert summary["metrics"]["distant_factorial_submitted_producer"][
        "attrition_from_paper_literal"
    ] == {"count": 0, "reasons": {}}
    assert summary["metrics"]["adjacent"]["paired_eligible"] == 1
    assert records[0]["profile"]["edited_top1_is_admissible"] == [True] * 8
    assert records[0]["arms"][0]["token_is_admissible"] is None
    assert records[0]["arms"][1]["token_is_admissible"] is True
    assert summary["comparability"]["status"] == "partial-smoke-run"
    assert summary["comparability"]["limitations"] == [
        "selected-target-count-differs-from-paper",
        "limit-is-smoke-only",
    ]
    assert summary["comparability"]["requirements"]["paper_source_protocol"] is True
    assert summary["comparability"]["requirements"]["paper_source_cohort_identity"] is False
    assert "paper_source_cohort" not in summary["comparability"]["requirements"]
    assert manifest["status"] == "completed"
    assert manifest["comparability"] == summary["comparability"]
    assert manifest["plan"]["selected_full"] == [
        ["attribution-4", "sample-a"],
        ["random-4", "sample-r"],
    ]
    assert manifest["checkpoints"] == {}


def test_extension_selection_keeps_a_length_eligible_boundary_invalid_source_id(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    config = token_api.OneTokenPrefixReplacementConfig(**{**base.public_arguments(), "limit": None})

    class BoundaryInvalidFirst(_Runtime):
        def prepare_pair(self, pair: Mapping[str, object]) -> object:
            if pair.get("targeting") == "attribution-4":
                raise token_api.OneTokenBoundaryInvalid(
                    "synthetic exact-boundary failure",
                    clean_cot_ids=tuple(range(10, 18)),
                )
            return super().prepare_pair(pair)

    result = token_api.run_one_token_prefix_replacement(
        config,
        runtime=BoundaryInvalidFirst(),
    )
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    statuses = [
        json.loads(line)
        for line in result.pair_status_records_path.read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["plan"]["selected_full"] == [
        ["attribution-4", "sample-a"],
        ["random-4", "sample-r"],
    ]
    assert manifest["plan"]["eligible_case_count"] == 2
    assert result.selected_pairs == 2
    assert result.records == 1
    assert statuses[0]["execution_status"] == "invalid-boundary"
    assert statuses[0]["selected_full"] is True
    assert statuses[0]["selected_for_execution"] is True
    assert statuses[0]["exclusion_reason"] == "prompt-boundary-invalid"
    assert manifest["comparability"]["status"] == "partial-paper-protocol"
    assert manifest["comparability"]["requirements"]["selected_exact_boundary_valid_count"] == 1
    assert manifest["comparability"]["requirements"]["selected_exact_boundaries_all_valid"] is False
    assert "selected-target-has-invalid-exact-boundary" in manifest["comparability"]["limitations"]


def test_failed_run_retains_arm_checkpoint_and_resume_does_not_repeat_completed_arms(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    interrupted = _Runtime(fail_after=3)

    with pytest.raises(token_api.OneTokenPrefixReplacementRunError):
        token_api.run_one_token_prefix_replacement(config, runtime=interrupted)

    output = config.output_dir
    assert not (output / "one_token_records.jsonl").exists()
    checkpoint = next(
        (output / ".one-token-prefix-replacement-work" / "checkpoints").glob("*.json")
    )
    partial = json.loads(checkpoint.read_text())
    assert len(partial["arms"]) == 3
    failed_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    checkpoint_metadata = next(iter(failed_manifest["checkpoints"].values()))
    runtime_sha256 = hashlib.sha256(
        json.dumps(
            failed_manifest["runtime"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert checkpoint_metadata["runtime_provenance_sha256"] == runtime_sha256

    resumed_runtime = _Runtime()
    result = token_api.run_one_token_prefix_replacement(
        token_api.OneTokenPrefixReplacementConfig(
            **{**config.public_arguments(), "output_dir": output, "resume": True}
        ),
        runtime=resumed_runtime,
    )

    assert result.records == 1
    assert len(resumed_runtime.calls) == 5
    assert not (output / ".one-token-prefix-replacement-work").exists()


def test_resume_rejects_checkpoint_bytes_not_bound_by_the_failed_manifest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(token_api.OneTokenPrefixReplacementRunError):
        token_api.run_one_token_prefix_replacement(config, runtime=_Runtime(fail_after=3))

    checkpoint = next(
        (config.output_dir / ".one-token-prefix-replacement-work" / "checkpoints").glob("*.json")
    )
    checkpoint.write_text(
        checkpoint.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(token_api.OneTokenPrefixReplacementRunError, match="checkpoint SHA-256"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_Runtime())


def test_resume_rejects_an_equal_count_but_different_admissible_token_pool(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(token_api.OneTokenPrefixReplacementRunError):
        token_api.run_one_token_prefix_replacement(config, runtime=_Runtime(fail_after=3))

    class DifferentPoolRuntime(_Runtime):
        def provenance(self) -> Mapping[str, object]:
            value = dict(super().provenance())
            token_pool = dict(value["token_admissibility"])  # type: ignore[arg-type]
            token_pool["admissible_token_ids_sha256"] = "b" * 64
            value["token_admissibility"] = token_pool
            return value

    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match="runtime provenance"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=DifferentPoolRuntime())


def test_resume_validates_and_skips_a_registered_position_exclusion_checkpoint(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    config = token_api.OneTokenPrefixReplacementConfig(**{**base.public_arguments(), "limit": None})

    class ExcludeThenFail(_Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.profile_calls = 0

        def profile_pair(self, plan: object) -> object:
            self.profile_calls += 1
            if self.profile_calls == 1:
                return token_api.OneTokenProfile(
                    clean_to_edited_kl=(8.0,) * 8,
                    clean_token_rank_under_clean=(1,) * 8,
                    clean_token_rank_under_edited=(1,) * 8,
                    edited_top1_ids=tuple(range(100, 108)),
                    edited_top1_is_admissible=(True,) * 8,
                )
            raise RuntimeError("fixture profile interruption")

    with pytest.raises(token_api.OneTokenPrefixReplacementRunError):
        token_api.run_one_token_prefix_replacement(config, runtime=ExcludeThenFail())

    checkpoints = sorted(
        (config.output_dir / ".one-token-prefix-replacement-work" / "checkpoints").glob("*.json")
    )
    assert len(checkpoints) == 1
    excluded = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert excluded["position_exclusion_reason"] == (
        "no-position-with-clean-token-below-edited-top1"
    )
    assert excluded["profile_sha256"]
    assert excluded["input_plan_sha256"]

    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )
    runtime = _Runtime()
    result = token_api.run_one_token_prefix_replacement(resumed, runtime=runtime)

    assert result.records == 1
    assert len(runtime.calls) == 8


def test_completed_resume_reconstructs_every_output_without_loading_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    before = {
        path.name: _sha256(path)
        for path in (
            first.records_path,
            first.pair_status_records_path,
            first.summary_path,
            first.run_path,
        )
    }
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    second = token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())
    after = {
        path.name: _sha256(path)
        for path in (
            second.records_path,
            second.pair_status_records_path,
            second.summary_path,
            second.run_path,
        )
    }

    assert after == before


def test_source_drift_during_generation_prevents_publication(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class MutatingRuntime(_Runtime):
        changed = False

        def generate_arm(self, *args: object, **kwargs: object) -> object:
            result = super().generate_arm(*args, **kwargs)
            if not self.changed:
                config.pairs[0].write_text(
                    config.pairs[0].read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                self.changed = True
            return result

    with pytest.raises(token_api.OneTokenPrefixReplacementRunError, match="source snapshot"):
        token_api.run_one_token_prefix_replacement(config, runtime=MutatingRuntime())
    assert not (config.output_dir / "one_token_records.jsonl").exists()
    assert not (config.output_dir / "one_token_summary.json").exists()


def test_completed_output_tampering_is_rejected_before_runtime_access(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    result.records_path.write_text(
        result.records_path.read_text(encoding="utf-8").replace("sample-a", "forged"),
        encoding="utf-8",
    )
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match="output SHA-256"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())


def test_completed_resume_reports_a_missing_public_output_as_a_value_error(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    result.records_path.unlink()
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match="completed output is missing"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())


def test_completed_resume_rejects_rehashed_record_source_identity_tampering(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    record = json.loads(result.records_path.read_text(encoding="utf-8"))
    record["source"]["source_record_sha256"] = "f" * 64
    result.records_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    manifest["outputs"][result.records_path.name]["sha256"] = _sha256(result.records_path)
    manifest["outputs"][result.records_path.name]["bytes"] = result.records_path.stat().st_size
    _write_json(result.run_path, manifest)
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match="source identity"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())


def test_completed_resume_rejects_rehashed_status_semantic_tampering(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    statuses = [
        json.loads(line)
        for line in result.pair_status_records_path.read_text(encoding="utf-8").splitlines()
    ]
    statuses[0]["selected_for_execution"] = False
    result.pair_status_records_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in statuses
        ),
        encoding="utf-8",
    )
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    metadata = manifest["outputs"][result.pair_status_records_path.name]
    metadata["sha256"] = _sha256(result.pair_status_records_path)
    metadata["bytes"] = result.pair_status_records_path.stat().st_size
    _write_json(result.run_path, manifest)
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match="status semantics"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())


def test_completed_resume_rejects_a_forged_position_exclusion_detail(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    config = token_api.OneTokenPrefixReplacementConfig(**{**base.public_arguments(), "limit": None})

    class ExcludeFirst(_Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.profile_calls = 0

        def profile_pair(self, plan: object) -> object:
            self.profile_calls += 1
            if self.profile_calls == 1:
                return token_api.OneTokenProfile(
                    clean_to_edited_kl=(8.0,) * 8,
                    clean_token_rank_under_clean=(1,) * 8,
                    clean_token_rank_under_edited=(1,) * 8,
                    edited_top1_ids=tuple(range(100, 108)),
                    edited_top1_is_admissible=(True,) * 8,
                )
            return super().profile_pair(plan)

    result = token_api.run_one_token_prefix_replacement(config, runtime=ExcludeFirst())
    statuses = [
        json.loads(line)
        for line in result.pair_status_records_path.read_text(encoding="utf-8").splitlines()
    ]
    excluded = next(row for row in statuses if row["execution_status"] == "position-unavailable")
    assert excluded["position_exclusion_evidence"]["position_exclusion_reason"] == (
        "no-position-with-clean-token-below-edited-top1"
    )
    assert excluded["position_exclusion_evidence"]["profile_sha256"]
    excluded["adjacent_unavailable_reason"] = "forged"
    result.pair_status_records_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in statuses
        ),
        encoding="utf-8",
    )
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    metadata = manifest["outputs"][result.pair_status_records_path.name]
    metadata["sha256"] = _sha256(result.pair_status_records_path)
    metadata["bytes"] = result.pair_status_records_path.stat().st_size
    _write_json(result.run_path, manifest)
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match="adjacent availability"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())


def test_completed_resume_rejects_a_dropped_record_relabelled_as_position_excluded(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    statuses = [
        json.loads(line)
        for line in result.pair_status_records_path.read_text(encoding="utf-8").splitlines()
    ]
    completed = next(row for row in statuses if row["execution_status"] == "completed")
    completed.update(
        {
            "positions": {"selected": 0, "distant": None, "adjacent": None},
            "adjacent_available": False,
            "adjacent_unavailable_reason": "case-excluded-before-adjacent-position-selection",
            "execution_status": "position-unavailable",
            "exclusion_reason": "no-distant-lower-median-control",
            "record_sha256": None,
            "position_exclusion_evidence": None,
        }
    )
    result.records_path.write_text("", encoding="utf-8")
    result.pair_status_records_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in statuses
        ),
        encoding="utf-8",
    )
    source = token_runner.load_source_bundle(
        cohort="extension",
        model=config.model,
        benchmark=config.benchmark,
        fixed_window_run=None,
        pairs=config.pairs,
    )
    summary = token_runner._build_summary(
        config=config,
        source=source,
        plan=manifest["plan"],
        records=[],
        statuses=statuses,
    )
    _write_json(result.summary_path, summary)
    manifest["counts"]["records"] = 0
    for path in (
        result.records_path,
        result.pair_status_records_path,
        result.summary_path,
    ):
        manifest["outputs"][path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    _write_json(result.run_path, manifest)
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match="position-exclusion evidence"):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda run: run["plan"].__setitem__("cases_sha256", "f" * 64), "plan fingerprint"),
        (lambda run: run["runtime"].__setitem__("dtype", "float16"), "runtime"),
        (lambda run: run["runtime"].__setitem__("runtime", "forged"), "runtime"),
    ),
)
def test_completed_resume_rejects_manifest_semantic_tampering(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    config = _config(tmp_path)
    result = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    mutation(manifest)
    _write_json(result.run_path, manifest)
    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )

    with pytest.raises(ValueError, match=message):
        token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())


def test_late_source_validation_failure_removes_uncommitted_public_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    original = token_runner.validate_source_snapshot
    calls = 0

    def fail_after_publication(source: object) -> None:
        nonlocal calls
        calls += 1
        original(source)
        if calls == 3:
            raise ValueError("late source snapshot failure")

    monkeypatch.setattr(token_runner, "validate_source_snapshot", fail_after_publication)

    with pytest.raises(token_api.OneTokenPrefixReplacementRunError):
        token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    for name in (
        "one_token_records.jsonl",
        "pair_status_records.jsonl",
        "one_token_summary.json",
    ):
        assert not (config.output_dir / name).exists()


def test_completed_manifest_write_failure_keeps_resume_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    original = token_runner._write_json_atomic

    def fail_completed_manifest(path: Path, payload: object) -> None:
        if (
            path.name == "run.json"
            and isinstance(payload, Mapping)
            and payload.get("status") == "completed"
        ):
            raise OSError("fixture completed-manifest failure")
        original(path, payload)

    monkeypatch.setattr(token_runner, "_write_json_atomic", fail_completed_manifest)

    with pytest.raises(token_api.OneTokenPrefixReplacementRunError):
        token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    manifest = json.loads((config.output_dir / "run.json").read_text(encoding="utf-8"))
    checkpoint_dir = config.output_dir / ".one-token-prefix-replacement-work" / "checkpoints"
    assert manifest["status"] == "failed"
    assert manifest["checkpoints"]
    assert list(checkpoint_dir.glob("*.json"))


def test_post_commit_checkpoint_cleanup_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    def fail_cleanup(_path: Path) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(token_runner.shutil, "rmtree", fail_cleanup)
    result = token_api.run_one_token_prefix_replacement(config, runtime=_Runtime())
    before = {
        path.name: _sha256(path)
        for path in (
            result.records_path,
            result.pair_status_records_path,
            result.summary_path,
            result.run_path,
        )
    }
    manifest = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["failures"] == []

    resumed = token_api.OneTokenPrefixReplacementConfig(
        **{**config.public_arguments(), "output_dir": config.output_dir, "resume": True}
    )
    second = token_api.run_one_token_prefix_replacement(resumed, runtime=_NoRuntimeUse())
    after = {
        path.name: _sha256(path)
        for path in (
            second.records_path,
            second.pair_status_records_path,
            second.summary_path,
            second.run_path,
        )
    }
    assert after == before


def test_cli_parses_primary_extension_and_adjacent_source_contracts() -> None:
    parser = cli_module._parser()
    primary = parser.parse_args(
        [
            "one-token-prefix-replacement",
            "--model",
            "google/gemma-3-4b-it",
            "--benchmark",
            "gsm8k",
            "--cohort",
            "primary",
            "--fixed-window-run",
            "fixed",
            "--position-controls",
            "distant",
            "--output-dir",
            "output",
        ]
    )
    extension = parser.parse_args(
        [
            "one-token-prefix-replacement",
            "--model",
            MODEL,
            "--benchmark",
            "gsm8k",
            "--cohort",
            "extension",
            "--pairs",
            "attribution/pairs.jsonl",
            "random/pairs.jsonl",
            "--max-pairs",
            "150",
            "--position-controls",
            "distant",
            "adjacent",
            "--output-dir",
            "output",
        ]
    )

    assert primary.fixed_window_run == Path("fixed")
    assert primary.position_controls == ["distant"]
    assert extension.pairs == [
        Path("attribution/pairs.jsonl"),
        Path("random/pairs.jsonl"),
    ]
    assert extension.position_controls == ["distant", "adjacent"]


def test_cli_dispatches_the_exact_config_and_prints_all_four_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    output = tmp_path / "output"

    def fake_run(config: object) -> object:
        captured["config"] = config
        return SimpleNamespace(
            records=1,
            executed_pairs=1,
            records_path=output / "one_token_records.jsonl",
            pair_status_records_path=output / "pair_status_records.jsonl",
            summary_path=output / "one_token_summary.json",
            run_path=output / "run.json",
        )

    monkeypatch.setattr(cli_module, "run_one_token_prefix_replacement", fake_run)
    exit_code = cli_module.main(
        [
            "one-token-prefix-replacement",
            "--model",
            MODEL,
            "--benchmark",
            "gsm8k",
            "--cohort",
            "extension",
            "--pairs",
            str(tmp_path / "attribution" / "pairs.jsonl"),
            str(tmp_path / "random" / "pairs.jsonl"),
            "--max-pairs",
            "150",
            "--position-controls",
            "distant",
            "adjacent",
            "--gpu-id",
            "0",
            "--limit",
            "1",
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert isinstance(config, token_api.OneTokenPrefixReplacementConfig)
    assert config.position_controls == ("distant", "adjacent")
    assert config.limit == 1
    printed = capsys.readouterr().out
    for name in (
        "one_token_records.jsonl",
        "pair_status_records.jsonl",
        "one_token_summary.json",
        "run.json",
    ):
        assert name in printed


def test_cli_reports_source_io_errors_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_config: object) -> object:
        raise OSError("synthetic source read failure")

    monkeypatch.setattr(cli_module, "run_one_token_prefix_replacement", fail)
    exit_code = cli_module.main(
        [
            "one-token-prefix-replacement",
            "--model",
            MODEL,
            "--benchmark",
            "gsm8k",
            "--cohort",
            "extension",
            "--pairs",
            str(tmp_path / "attribution" / "pairs.jsonl"),
            str(tmp_path / "random" / "pairs.jsonl"),
            "--max-pairs",
            "150",
            "--position-controls",
            "distant",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == ("one-token-prefix-replacement: error: synthetic source read failure\n")
    assert "Traceback" not in captured.err


def test_summary_preserves_final_pdf_table_references_without_pooling_primary(
    tmp_path: Path,
) -> None:
    result = token_api.run_one_token_prefix_replacement(_config(tmp_path), runtime=_Runtime())
    reference = json.loads(result.summary_path.read_text())["historical_reference"]

    assert reference["table10_extension_aggregate"] == {
        "includes_primary": False,
        "paired_eligible": 1629,
        "selected": {"numerator": 492, "percent": 30.2},
        "control": {"numerator": 296, "percent": 18.2},
    }
    assert reference["table11_distant_pooled"] == {
        "includes_primary": False,
        "paired_eligible": 1575,
        "selected_percent": 28.3,
        "control_percent": 20.1,
        "difference_percentage_points": 8.2,
        "confidence_interval_95": [6.0, 10.4],
    }
    assert reference["table11_distant_submitted_producer_exact_counts"] == {
        "includes_primary": False,
        "paired_eligible": 1575,
        "event_opportunities_per_position": 3150,
        "selected": {"numerator": 892, "percent": 28.3},
        "control": {"numerator": 633, "percent": 20.1},
        "difference_percentage_points": 8.2,
    }
    assert reference["table11_adjacent_pooled"]["paired_eligible"] == 391
    assert reference["table11_distant_final_pdf_literal_reclassification"] == {
        "includes_primary": False,
        "paired_eligible": 1603,
        "event_opportunities_per_position": 3206,
        "selected": {"numerator": 912, "percent": 28.4},
        "control": {"numerator": 647, "percent": 20.2},
        "difference_percentage_points": 8.3,
        "submitted_producer_attrition": {
            "count": 28,
            "selected_and_control_tokens_identical": 28,
            "selected_token_not_admissible": 0,
            "control_token_not_admissible": 0,
        },
    }
    assert reference["figure5_worked_case"] == {
        "setting_id": "gemma4b_gsm8k",
        "targeting_code": "lxt4",
        "sample_id": "gsm8k_00556",
        "selected_position": 23,
        "distant_position": 60,
        "selected_kl": 8.785974,
        "clean_token_rank_under_clean": 1,
        "clean_token_rank_under_edited": 8,
        "clean_token_text": " thrice",
        "selected_edited_top1_text": " twice",
        "selected_keep_answer": "160",
        "selected_replacement_answer": "120",
        "distant_keep_answer": "160",
        "distant_replacement_answer": "160",
    }
    assert reference["table10_cells"]["gemma4b_gsm8k"]["selected"] == {
        "numerator": 41,
        "percent": 26.8,
    }

    protocol = json.loads(result.run_path.read_text())["protocol"]
    assert protocol["paper_defined"]["eligibility"]["exact_token_boundaries"] is True
    assert len(protocol["paper_defined"]["settings"]["all_cells"]) == 15
    assert len(protocol["paper_defined"]["settings"]["adjacent_cells"]) == 3
    assert protocol["legacy_backed"]["selection_stage_alignment"] == (
        "prompt-length-suffix-equality-before-exact-boundary-audit"
    )
    assert protocol["public_implementation"]["exact_boundary_audit"] == (
        "clean-and-edited-prompt-prefix-preservation-plus-shared-clean-cot-suffix"
    )


def test_documentation_calls_the_operation_a_supplementary_diagnostic() -> None:
    project = Path(__file__).resolve().parents[1]
    readme = (project / "README.md").read_text(encoding="utf-8")
    details = (project / "docs" / "one-token-prefix-replacement.md").read_text(encoding="utf-8")

    for text in (readme, details):
        assert "one-token-prefix-replacement" in text
        assert "--cohort primary" in text
        assert "--cohort extension" in text
        assert "--position-controls distant" in text
        assert "distant adjacent" in text
        assert "not typo repair" in text
        assert "legacy-backed" in text
        assert "one_token_records.jsonl" in text
        assert "pair_status_records.jsonl" in text
        assert "one_token_summary.json" in text
        assert "run.json" in text
        assert "distant_factorial_submitted_producer" in text

    assert "RQ3 result" not in details
    assert "1,629" in details
    assert "1,603" in details
    assert "1,575" in details
    assert "391" in details
    assert "short_setting_id|legacy_condition_code|sample_id" in details
    assert "gsm8k_00556" in details
    assert "setting-level GPU producer" in details


def test_extension_source_adapter_is_reused_instead_of_legacy_target_sets(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    source = load_extension_source(config.pairs, model=config.model, benchmark=config.benchmark)

    assert [case.key for case in source.cases] == [
        ("attribution-4", "sample-a"),
        ("random-4", "sample-r"),
    ]


def test_production_profile_runtime_uses_correct_prediction_rows_and_kl_direction() -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *, input_ids: object) -> object:
            self.calls += 1
            first = int(input_ids[0, 0])
            logits = torch.zeros((1, 4, 3), dtype=torch.float32)
            if first == 9:
                logits[0, 1] = torch.tensor([4.0, 1.0, 0.0])
                logits[0, 2] = torch.tensor([0.0, 3.0, 1.0])
            else:
                logits[0, 1] = torch.tensor([1.0, 3.0, 0.0])
                logits[0, 2] = torch.tensor([0.0, 1.0, 4.0])
            return SimpleNamespace(logits=logits)

    runtime = token_api.HuggingFaceOneTokenPrefixReplacementRuntime.__new__(
        token_api.HuggingFaceOneTokenPrefixReplacementRuntime
    )
    runtime._torch = torch
    runtime.device = torch.device("cpu")
    runtime.model = Model()
    runtime.admissible_token_ids = frozenset({1})
    plan = token_api.OneTokenInputPlan(
        clean_prompt_ids=(9, 8),
        edited_prompt_ids=(7, 6),
        clean_full_ids=(9, 8, 0, 1),
        edited_full_ids=(7, 6, 0, 1),
        clean_cot_ids=(0, 1),
    )

    profile = runtime.profile_pair(plan)
    clean_rows = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    edited_rows = torch.tensor([[1.0, 3.0, 0.0], [0.0, 1.0, 4.0]])
    clean_logp = torch.log_softmax(clean_rows, dim=-1)
    edited_logp = torch.log_softmax(edited_rows, dim=-1)
    expected_kl = (clean_logp.exp() * (clean_logp - edited_logp)).sum(dim=-1)

    assert runtime.model.calls == 2
    assert profile.clean_to_edited_kl == pytest.approx(tuple(expected_kl.tolist()))
    assert profile.clean_token_rank_under_clean == (1, 1)
    assert profile.clean_token_rank_under_edited == (2, 2)
    assert profile.edited_top1_ids == (1, 2)
    assert profile.edited_top1_is_admissible == (True, False)


def test_submitted_producer_token_pool_excludes_special_marker_and_oov_ids() -> None:
    class AddedToken:
        def __init__(self, *, special: bool) -> None:
            self.special = special

    class Tokenizer:
        all_special_ids = [2]
        added_tokens_decoder = {3: AddedToken(special=True), 4: AddedToken(special=False)}

        @staticmethod
        def get_vocab() -> dict[str, int]:
            return {
                "ordinary": 0,
                "<unused17>": 1,
                "<eos>": 2,
                "<added-special>": 3,
                "ordinary-added": 4,
                "outside-logits": 5,
            }

    admissible, stats = token_runtime._tokenizer_candidate_pool(Tokenizer(), model_logit_size=5)

    assert admissible == frozenset({0, 4})
    assert stats == {
        "implementation": "submitted-producer-tokenizer-candidate-pool/v1",
        "model_logit_size": 5,
        "tokenizer_vocab_entries": 6,
        "n_real_tokenizer_ids_in_logits": 5,
        "n_special_ids": 2,
        "n_marker_ids": 1,
        "n_admissible_ids": 2,
        "admissible_token_ids_sha256_algorithm": "sorted-decimal-lines/v1",
        "admissible_token_ids_sha256": hashlib.sha256(b"0\n4\n").hexdigest(),
        "marker_regex": (
            r"^(?:<unused\d+>|<\|reserved_special_token_\d+\|>"
            r"|\[control_\d+\]|\[unused\d+\])$"
        ),
    }


def test_production_prepare_pair_preserves_selection_eligible_boundary_failure() -> None:
    cot = tuple(range(10, 18))

    class Tokenizer:
        def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
            values = {
                "clean": [1, 2],
                "edited": [3, 4],
                "cleanReasoning. ": [1, 99, *cot],
                "editedReasoning. ": [3, 4, *cot],
            }
            return {"input_ids": values[text]}

    runtime = token_api.HuggingFaceOneTokenPrefixReplacementRuntime.__new__(
        token_api.HuggingFaceOneTokenPrefixReplacementRuntime
    )
    runtime.tokenizer = Tokenizer()
    pair = {
        "clean": {
            "prompt": "clean",
            "prompt_token_count": 2,
            "continuation": "Reasoning. The answer is 2.",
        },
        "edited": {"prompt": "edited", "prompt_token_count": 2},
    }

    with pytest.raises(token_api.OneTokenBoundaryInvalid) as raised:
        runtime.prepare_pair(pair)

    assert raised.value.clean_cot_ids == cot
    assert raised.value.cot_token_count == 8
    assert raised.value.eligible_length is True


def test_production_generation_runtime_passes_direct_ids_and_decodes_only_suffix() -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self) -> None:
            self.input_ids: list[int] | None = None
            self.arguments: dict[str, object] | None = None

        def generate(
            self, *, input_ids: object, attention_mask: object, **kwargs: object
        ) -> object:
            assert attention_mask.tolist() == [[1, 1, 1, 1]]
            self.input_ids = input_ids[0].tolist()
            self.arguments = kwargs
            return torch.tensor([self.input_ids + [2]], dtype=torch.long)

    class Tokenizer:
        pad_token_id = 0

        def __init__(self) -> None:
            self.decoded: list[int] | None = None

        def decode(self, token_ids: list[int], **kwargs: object) -> str:
            assert kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            self.decoded = token_ids
            return "The answer is 2."

    runtime = token_api.HuggingFaceOneTokenPrefixReplacementRuntime.__new__(
        token_api.HuggingFaceOneTokenPrefixReplacementRuntime
    )
    runtime._torch = torch
    runtime.device = torch.device("cpu")
    runtime.model = Model()
    runtime.tokenizer = Tokenizer()
    runtime.effective_eos_token_ids = (2,)
    runtime.config = SimpleNamespace(benchmark="gsm8k")
    plan = token_api.OneTokenInputPlan(
        clean_prompt_ids=(1, 2),
        edited_prompt_ids=(3, 4),
        clean_full_ids=(1, 2, 10, 11, 12),
        edited_full_ids=(3, 4, 10, 11, 12),
        clean_cot_ids=(10, 11, 12),
    )

    generation = runtime.generate_arm(
        plan,
        position=1,
        forced_token_id=99,
        gold_answer="2",
    )

    assert runtime.model.input_ids == [1, 2, 10, 99]
    assert runtime.tokenizer.decoded == [2]
    assert runtime.model.arguments["do_sample"] is False
    assert runtime.model.arguments["max_new_tokens"] == 512
    assert runtime.model.arguments["eos_token_id"] == [2]
    assert generation.token_ids == (2,)
    assert generation.is_correct is True
    assert generation.stop_reason == "eos_token"
