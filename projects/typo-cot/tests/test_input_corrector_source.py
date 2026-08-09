"""Fail-closed source contracts for the Appendix E input-corrector audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import typo_cot.experiments.input_corrector_audit.source as source_module
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.input_corrector_audit.source import (
    InputCorrectorSourceError,
    load_input_corrector_source,
    sha256_file,
)


MODEL = "google/gemma-3-1b-it"
PUBLIC_BENCHMARKS = ("gsm8k", "mmlu", "mmlu-pro", "arc", "csqa", "math-500")
_DELETE = object()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _answer(value: str, *, correct: bool) -> dict[str, object]:
    return {
        "value": value,
        "is_extracted": True,
        "is_correct": correct,
        "method": "primary:fixture",
        "primary_method": "fixture",
        "confidence": 1.0,
    }


def _pair_record(
    sample_id: str,
    *,
    model: str = MODEL,
    benchmark: str = "gsm8k",
) -> dict[str, object]:
    prefix = "FEWSHOT\nQuestion: "
    suffix = "\nAnswer:"
    clean_editable = "alpha beta"
    edited_editable = "alpga beta"
    clean_prompt = prefix + clean_editable + suffix
    edited_prompt = prefix + edited_editable + suffix
    editable_start = len(prefix)
    clean_editable_end = editable_start + len(clean_editable)
    edited_editable_end = editable_start + len(edited_editable)
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": model,
        "benchmark": benchmark,
        "targeting": "attribution-4",
        "seed": 42,
        "num_edits_requested": 4,
        "num_candidates": 4,
        "num_target_attempts": 1,
        "num_aligned_words": 1,
        "gold_answer": "2",
        "subset": None,
        "attribution_target": {
            "definition": "maximum-logit-after-first-cot-token",
            "position": 9,
            "first_cot_token_id": 123,
            "first_cot_token_text": " First",
            "context": "complete-clean-generation",
        },
        "clean": {
            "question": clean_editable,
            "choices": None,
            "editable_text": clean_editable,
            "editable_prompt_span": {
                "start": editable_start,
                "end": clean_editable_end,
            },
            "prompt": clean_prompt,
            "prompt_token_count": 12,
            "continuation": "Clean reasoning. The answer is 2.",
            "continuation_token_count": 8,
            "answer": _answer("2", correct=True),
        },
        "edited": {
            "editable_text": edited_editable,
            "editable_prompt_span": {
                "start": editable_start,
                "end": edited_editable_end,
            },
            "prompt": edited_prompt,
            "prompt_token_count": 12,
            "continuation": "Edited reasoning. The answer is 3.",
            "continuation_token_count": 8,
            "answer": _answer("3", correct=False),
        },
        "answer_changed": True,
        "excluded_attribution_tokens": [],
        "target_attempts": [
            {
                "selection_rank": 1,
                "attribution_rank": 1,
                "target_token_index": 4,
                "target_token_text": "alpha",
                "relevance": 1.0,
                "intended_prompt_span": {
                    "start": editable_start,
                    "end": editable_start + 5,
                },
                "intended_editable_span": {"start": 0, "end": 5},
                "landed_editable_span_before": {"start": 0, "end": 5},
                "landed_text_before": "alpha",
                "landed_origin_index": 0,
                "landed_on_intended_token": True,
                "intended_word_index": 0,
                "landed_word_index": 0,
                "operation": "substitution",
                "character_index": 3,
                "original_character": "h",
                "new_character": "g",
                "edited_token_text": "alpga",
            }
        ],
        "aligned_words": [
            {
                "word_index": 0,
                "clean_text": "alpha",
                "edited_text": "alpga",
                "clean_editable_span": {"start": 0, "end": 5},
                "edited_editable_span": {"start": 0, "end": 5},
                "clean_prompt_span": {
                    "start": editable_start,
                    "end": editable_start + 5,
                },
                "edited_prompt_span": {
                    "start": editable_start,
                    "end": editable_start + 5,
                },
                "target_ranks": [1],
                "target_token_indices": [4],
                "clean_token_indices": [4],
                "edited_token_indices": [4],
                "clean_final_token": 4,
                "edited_final_token": 4,
            }
        ],
    }


def _dataset_loader(benchmark: str) -> str:
    return {
        "gsm8k": "gsm8k",
        "mmlu": "mmlu",
        "mmlu-pro": "mmlu_pro",
        "arc": "arc",
        "csqa": "commonsense_qa",
        "math-500": "math",
    }.get(benchmark, benchmark)


def _manifest(
    directory: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    model: str = MODEL,
    benchmark: str = "gsm8k",
) -> dict[str, object]:
    dataset_identity = [
        {
            "sample_id": row["sample_id"],
            "question": row["clean"]["question"],
            "choices": row["clean"]["choices"],
            "correct_answer": row["gold_answer"],
            "subset": row["subset"],
        }
        for row in rows
    ]
    dataset_sha256 = hashlib.sha256(
        json.dumps(
            dataset_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    pairs_path = directory / "pairs.jsonl"
    return {
        "schema_version": "prepare-edited-pairs-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "prepare-edited-pairs",
        "status": "completed",
        "arguments": {
            "model": model,
            "benchmark": benchmark,
            "targeting": "attribution-4",
            "num_edits": 4,
            "seed": 42,
            "max_new_tokens": 512,
            "gpu_id": "0",
            "limit": None,
            "output_dir": str(directory.resolve()),
        },
        "counts": {
            "discovered": len(rows),
            "written": len(rows),
            "failed": 0,
        },
        "failures": [],
        "outputs": {
            "pairs": {
                "path": "pairs.jsonl",
                "sha256": sha256_file(pairs_path),
                "records": len(rows),
            }
        },
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
            "model": model,
            "model_revision": "1" * 40,
            "benchmark_dataset_loader": _dataset_loader(benchmark),
            "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
            "dataset_samples_per_subset": (
                50 if benchmark == "mmlu" else 100 if benchmark == "mmlu-pro" else None
            ),
            "dataset_sample_count": len(rows),
            "dataset_records_sha256": dataset_sha256,
            "random_seed_algorithm": "sha256-first-64-bits/v1",
            "generation_protocol": "explicit-greedy-generation/v1",
            "target_position": "maximum-logit-after-first-cot-token",
            "alignment": "actual-edited-word-final-token",
            "historical_compatibility_notes": [],
        },
    }


def _write_source(
    directory: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    model: str = MODEL,
    benchmark: str = "gsm8k",
) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    pairs_path = directory / "pairs.jsonl"
    run_path = directory / "run.json"
    _write_jsonl(pairs_path, rows)
    _write_json(
        run_path,
        _manifest(directory, rows, model=model, benchmark=benchmark),
    )
    return pairs_path, run_path


def _set_path(payload: dict[str, object], path: str, value: object) -> None:
    components = path.split(".")
    current: dict[str, object] = payload
    for component in components[:-1]:
        child = current[component]
        assert isinstance(child, dict)
        current = child
    if value is _DELETE:
        del current[components[-1]]
    else:
        current[components[-1]] = value


def _load(pairs_path: Path, *, model: str = MODEL, benchmark: str = "gsm8k") -> object:
    return load_input_corrector_source(
        pairs_path,
        model=model,
        benchmark=benchmark,
        require_paper_cohort_size=False,
    )


def test_loads_completed_unlimited_prepare_source_and_exposes_stable_hashes(
    tmp_path: Path,
) -> None:
    rows = [_pair_record("sample-000"), _pair_record("sample-001")]
    pairs_path, run_path = _write_source(tmp_path / "source", rows)

    source = _load(pairs_path)

    assert source is not None
    assert sha256_file(pairs_path) == hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    assert sha256_file(run_path) == hashlib.sha256(run_path.read_bytes()).hexdigest()
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    assert source.pairs_sha256 == manifest["outputs"]["pairs"]["sha256"]


def test_jsonl_loader_keeps_unicode_line_separators_inside_a_record(
    tmp_path: Path,
) -> None:
    row = _pair_record("sample-000")
    clean = row["clean"]
    assert isinstance(clean, dict)
    clean["continuation"] = "first segment\u2028second segment"
    pairs_path, _run_path = _write_source(tmp_path / "source", [row])

    source = _load(pairs_path)

    assert source.records[0]["clean"]["continuation"] == (
        "first segment\u2028second segment"
    )


@pytest.mark.parametrize("benchmark", PUBLIC_BENCHMARKS)
def test_accepts_every_public_benchmark_name(tmp_path: Path, benchmark: str) -> None:
    row = _pair_record("sample-000", benchmark=benchmark)
    pairs_path, _run_path = _write_source(
        tmp_path / benchmark,
        [row],
        benchmark=benchmark,
    )

    assert _load(pairs_path, benchmark=benchmark) is not None


@pytest.mark.parametrize("benchmark", PUBLIC_BENCHMARKS)
def test_default_public_loader_rejects_a_truncated_paper_cohort(
    tmp_path: Path,
    benchmark: str,
) -> None:
    row = _pair_record("sample-000", benchmark=benchmark)
    pairs_path, _run_path = _write_source(
        tmp_path / benchmark,
        [row],
        benchmark=benchmark,
    )

    with pytest.raises(InputCorrectorSourceError, match="paper source|records|contain"):
        load_input_corrector_source(
            pairs_path,
            model=MODEL,
            benchmark=benchmark,
        )


@pytest.mark.parametrize("benchmark", ["mmlu_pro", "commonsense_qa", "math", "unknown"])
def test_rejects_nonpublic_benchmark_aliases(tmp_path: Path, benchmark: str) -> None:
    row = _pair_record("sample-000", benchmark=benchmark)
    pairs_path, _run_path = _write_source(
        tmp_path / "source",
        [row],
        benchmark=benchmark,
    )

    with pytest.raises(InputCorrectorSourceError, match="benchmark"):
        _load(pairs_path, benchmark=benchmark)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "prepare-edited-pairs-run/v0", "schema"),
        ("paper_sha256", "0" * 64, "paper"),
        ("operation", "some-other-operation", "operation"),
        ("status", "running", "completed"),
        ("arguments.model", "other/model", "model"),
        ("arguments.benchmark", "mmlu", "benchmark"),
        ("arguments.targeting", "random-4", "attribution-4"),
        ("arguments.seed", 7, "seed"),
        ("arguments.seed", 42.0, "seed"),
        ("arguments.num_edits", 3, "four edits|num_edits"),
        ("arguments.num_edits", 4.0, "four edits|num_edits"),
        ("arguments.limit", 1, "unlimited|limit"),
        ("arguments.limit", _DELETE, "limit|unlimited"),
        ("arguments.max_new_tokens", 256, "512|max_new_tokens"),
        ("decoding.max_new_tokens", 256, "512|max_new_tokens"),
        ("provenance.model", "other/model", "model"),
        ("provenance.model_revision", "", "model_revision|revision"),
        ("provenance.model_revision", _DELETE, "model_revision|revision"),
        (
            "provenance.benchmark_dataset_loader",
            _DELETE,
            "benchmark_dataset_loader|dataset",
        ),
        ("provenance.dataset_cohort_rule", _DELETE, "dataset_cohort_rule|dataset"),
        (
            "provenance.dataset_cohort_rule",
            "explicit-sample-id-cohort/v1",
            "dataset_cohort_rule|paper.*cohort",
        ),
        ("provenance.dataset_samples_per_subset", 99, "samples_per_subset|subset"),
        ("provenance.random_seed_algorithm", "python-hash/v0", "random_seed_algorithm"),
        ("provenance.generation_protocol", "model-defaults/v0", "generation_protocol"),
        ("provenance.target_position", "last-token", "target_position"),
        ("provenance.alignment", "token-substring", "alignment"),
        ("provenance.dataset_records_sha256", "not-a-sha", "dataset_records_sha256|SHA"),
        ("provenance.dataset_records_sha256", _DELETE, "dataset_records_sha256|dataset"),
        ("provenance.dataset_sample_count", 2, "count|dataset_sample_count"),
        ("counts.discovered", 2, "count|discovered"),
        ("counts.written", 2, "count|written"),
        ("counts.failed", 1, "failure|failed"),
        ("failures", [{"sample_id": "failed"}], "failure"),
        ("outputs", _DELETE, "output|pairs|SHA"),
        ("outputs.pairs.path", "other.jsonl", "output|path|pairs"),
        ("outputs.pairs.sha256", "0" * 64, "output|pairs|SHA|hash"),
        ("outputs.pairs.records", 2, "output|record|count"),
    ],
)
def test_rejects_manifest_contract_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    pairs_path, run_path = _write_source(
        tmp_path / "source",
        [_pair_record("sample-000")],
    )
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    _set_path(manifest, field, value)
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSourceError, match=message):
        _load(pairs_path)


def test_rejects_an_unlimited_explicit_sample_id_cohort(tmp_path: Path) -> None:
    pairs_path, run_path = _write_source(
        tmp_path / "source",
        [_pair_record("sample-000")],
    )
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["arguments"]["sample_ids"] = str((tmp_path / "cohort.json").resolve())
    manifest["provenance"]["dataset_cohort_rule"] = "explicit-sample-id-cohort/v1"
    manifest["provenance"]["sample_id_cohort"] = {
        "sample_count": 1,
        "sample_ids_sha256": "1" * 64,
    }
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSourceError, match="sample_ids|explicit|paper.*cohort"):
        _load(pairs_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("relevance", 99.0), ("target_token_index", 9)),
)
def test_rejects_pairs_changed_after_the_completed_manifest_bound_them(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = _pair_record("sample-000")
    pairs_path, _run_path = _write_source(tmp_path / "source", [row])
    attempts = row["target_attempts"]
    assert isinstance(attempts, list) and isinstance(attempts[0], dict)
    attempts[0][field] = value
    _write_jsonl(pairs_path, [row])

    with pytest.raises(InputCorrectorSourceError, match="output|pairs|SHA|hash"):
        _load(pairs_path)


def test_rejects_an_unexpected_completed_output_inventory(tmp_path: Path) -> None:
    pairs_path, run_path = _write_source(
        tmp_path / "source",
        [_pair_record("sample-000")],
    )
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    manifest["outputs"]["unexpected"] = {"path": "unexpected.txt"}
    _write_json(run_path, manifest)

    with pytest.raises(InputCorrectorSourceError, match="output|inventory|pairs"):
        _load(pairs_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "prepare-edited-pairs/v0", "schema"),
        ("sample_id", "", "sample_id"),
        ("model", "other/model", "model"),
        ("benchmark", "mmlu", "benchmark"),
        ("targeting", "random-4", "attribution-4|targeting"),
        ("seed", 7, "seed"),
        ("seed", 42.0, "seed"),
        ("num_edits_requested", 3, "four edits|num_edits_requested"),
        ("num_edits_requested", 4.0, "four edits|num_edits_requested"),
        ("num_target_attempts", 2, "num_target_attempts|target_attempts"),
    ],
)
def test_rejects_pair_record_schema_or_identity_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    row = _pair_record("sample-000")
    _set_path(row, field, value)
    pairs_path, _run_path = _write_source(tmp_path / "source", [row])

    with pytest.raises(InputCorrectorSourceError, match=message):
        _load(pairs_path)


def test_accepts_a_consistent_zero_edit_record_without_inventing_a_word(
    tmp_path: Path,
) -> None:
    row = _pair_record("sample-000")
    row["num_target_attempts"] = 0
    row["num_aligned_words"] = 0
    row["target_attempts"] = []
    row["aligned_words"] = []
    row["edited"] = dict(row["clean"])
    row["answer_changed"] = False
    pairs_path, _run_path = _write_source(tmp_path / "source", [row])

    source = _load(pairs_path)

    assert source.records[0]["num_aligned_words"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clean.editable_text", "not-the-prompt-slice"),
        ("edited.editable_text", "not-the-prompt-slice"),
        ("clean.editable_prompt_span.start", 1),
        ("edited.editable_prompt_span.end", 10_000),
        ("clean.editable_prompt_span.start", True),
    ],
)
def test_rejects_editable_text_that_does_not_match_its_recorded_prompt_span(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = _pair_record("sample-000")
    _set_path(row, field, value)
    pairs_path, _run_path = _write_source(tmp_path / "source", [row])

    with pytest.raises(InputCorrectorSourceError, match="editable|span|prompt"):
        _load(pairs_path)


def test_rejects_clean_editable_text_that_disagrees_with_question_and_choices(
    tmp_path: Path,
) -> None:
    row = _pair_record("sample-000")
    clean = row["clean"]
    assert isinstance(clean, dict)
    clean["question"] = "a different source question"
    pairs_path, _run_path = _write_source(tmp_path / "source", [row])

    with pytest.raises(InputCorrectorSourceError, match="question|choices|editable|reference"):
        _load(pairs_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_aligned_words", 0),
        ("aligned_words", []),
        ("aligned_words", "not-a-list"),
        ("aligned_words.0.clean_editable_span.start", 1),
        ("aligned_words.0.edited_prompt_span.end", 10_000),
    ],
)
def test_rejects_invalid_aligned_word_count_or_spans(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = _pair_record("sample-000")
    if field.startswith("aligned_words.0."):
        _, _, remainder = field.partition("aligned_words.0.")
        aligned_words = row["aligned_words"]
        assert isinstance(aligned_words, list)
        word = aligned_words[0]
        assert isinstance(word, dict)
        _set_path(word, remainder, value)
    else:
        _set_path(row, field, value)
    pairs_path, _run_path = _write_source(tmp_path / "source", [row])

    with pytest.raises(InputCorrectorSourceError, match="aligned|span|word"):
        _load(pairs_path)


def test_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    rows = [_pair_record("sample-000"), _pair_record("sample-000")]
    pairs_path, _run_path = _write_source(tmp_path / "source", rows)

    with pytest.raises(InputCorrectorSourceError, match="sorted|duplicate|sample"):
        _load(pairs_path)


def test_rejects_out_of_order_sample_ids(tmp_path: Path) -> None:
    rows = [_pair_record("sample-001"), _pair_record("sample-000")]
    pairs_path, _run_path = _write_source(tmp_path / "source", rows)

    with pytest.raises(InputCorrectorSourceError, match="sorted|order|sample"):
        _load(pairs_path)


def test_rejects_empty_pairs_file(tmp_path: Path) -> None:
    pairs_path, _run_path = _write_source(tmp_path / "source", [])

    with pytest.raises(InputCorrectorSourceError, match="empty|record|count"):
        _load(pairs_path)


@pytest.mark.parametrize("target", ["run", "pairs"])
def test_rejects_duplicate_json_object_keys(tmp_path: Path, target: str) -> None:
    pairs_path, run_path = _write_source(
        tmp_path / "source",
        [_pair_record("sample-000")],
    )
    if target == "run":
        text = run_path.read_text(encoding="utf-8")
        assert text.count('"status": "completed"') == 1
        run_path.write_text(
            text.replace(
                '"status": "completed"',
                '"status": "completed", "status": "completed"',
                1,
            ),
            encoding="utf-8",
        )
    else:
        text = pairs_path.read_text(encoding="utf-8")
        assert text.count('"sample_id": "sample-000"') == 1
        pairs_path.write_text(
            text.replace(
                '"sample_id": "sample-000"',
                '"sample_id": "sample-000", "sample_id": "sample-000"',
                1,
            ),
            encoding="utf-8",
        )

    with pytest.raises(InputCorrectorSourceError, match="duplicate|JSON"):
        _load(pairs_path)


@pytest.mark.parametrize("target", ["run", "pairs"])
def test_rejects_nonfinite_json_constants(tmp_path: Path, target: str) -> None:
    pairs_path, run_path = _write_source(
        tmp_path / "source",
        [_pair_record("sample-000")],
    )
    if target == "run":
        text = run_path.read_text(encoding="utf-8")
        assert text.count('"dataset_sample_count": 1') == 1
        run_path.write_text(
            text.replace('"dataset_sample_count": 1', '"dataset_sample_count": NaN', 1),
            encoding="utf-8",
        )
    else:
        text = pairs_path.read_text(encoding="utf-8")
        assert text.count('"num_candidates": 4') == 1
        pairs_path.write_text(
            text.replace('"num_candidates": 4', '"num_candidates": NaN', 1),
            encoding="utf-8",
        )

    with pytest.raises(InputCorrectorSourceError, match="finite|constant|JSON|NaN"):
        _load(pairs_path)


def test_requires_existing_pairs_and_sibling_run_manifest(tmp_path: Path) -> None:
    missing_pairs = tmp_path / "missing" / "pairs.jsonl"
    with pytest.raises(InputCorrectorSourceError, match="pairs|file"):
        _load(missing_pairs)

    pairs_path, run_path = _write_source(
        tmp_path / "source",
        [_pair_record("sample-000")],
    )
    run_path.unlink()
    with pytest.raises(InputCorrectorSourceError, match="run.json|manifest"):
        _load(pairs_path)


@pytest.mark.parametrize("role", ["pairs", "run"])
def test_rechecks_source_hashes_after_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    pairs_path, run_path = _write_source(
        tmp_path / "source",
        [_pair_record("sample-000")],
    )
    target = pairs_path if role == "pairs" else run_path
    real_sha256_file = source_module.sha256_file
    calls = 0

    def mutate_before_recheck(path: Path) -> str:
        nonlocal calls
        if Path(path).resolve() == target.resolve():
            calls += 1
            if calls == 2:
                target.write_bytes(target.read_bytes() + b" ")
        return real_sha256_file(path)

    monkeypatch.setattr(source_module, "sha256_file", mutate_before_recheck)

    with pytest.raises(InputCorrectorSourceError, match="changed|hash|SHA-256"):
        _load(pairs_path)
    assert calls >= 2
