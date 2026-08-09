"""CPU-only contracts for the production Appendix E runtime adapters.

The adapters are tested through explicit corrector, wrapper, and torch seams.
No test loads a Hugging Face checkpoint, contacts the network, or requires a
physical CUDA device.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from typo_cot.experiments.input_corrector_audit.protocol import (
    CORRECTOR_MODELS,
    PROTOCOL_SHA256,
)
from typo_cot.experiments.input_corrector_audit.runtime import (
    HuggingFaceSamePromptRuntime,
    ProductionCorrectionRuntime,
)


MODEL = "google/gemma-3-1b-it"
SOURCE_REVISION = "1" * 40


def _config(
    *,
    corrector: str = "pyspellchecker",
    benchmark: str = "gsm8k",
    gpu_id: str = "1",
) -> SimpleNamespace:
    return SimpleNamespace(
        corrector=corrector,
        model=MODEL,
        benchmark=benchmark,
        gpu_id=gpu_id,
    )


class _CudaSeam:
    def __init__(self, *, device_count: int = 1) -> None:
        self._device_count = device_count
        self.empty_cache_calls = 0
        self.manual_seed_calls: list[int] = []

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self._device_count

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "fixture-gpu"

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(total_memory=123_456)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _TorchSeam:
    """Expose real CPU tensors behind a fake CUDA/provenance boundary."""

    def __init__(self, torch: Any, *, device_count: int = 1) -> None:
        self._torch = torch
        self.bfloat16 = torch.bfloat16
        self.cuda = _CudaSeam(device_count=device_count)
        self.version = SimpleNamespace(cuda="fixture-cuda")
        self.manual_seed_calls: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class _PlainCorrector:
    def __init__(self, corrected: str, *, revision: str | None) -> None:
        self.corrected = corrected
        self.closed = False
        self._model = (
            None
            if revision is None
            else SimpleNamespace(config=SimpleNamespace(_commit_hash=revision))
        )
        self._tokenizer = (
            None if revision is None else SimpleNamespace(init_kwargs={"_commit_hash": revision})
        )

    def correct(self, text: str) -> str:
        assert text == "teh"
        return self.corrected

    def resolved_revision(self) -> str | None:
        if self._model is None:
            return None
        return str(self._model.config._commit_hash)

    def close(self) -> None:
        self.closed = True


class _TaggedQwenCorrector(_PlainCorrector):
    def correct(self, text: str) -> str:  # pragma: no cover - wrong adapter path
        raise AssertionError("Qwen metadata would be lost by calling correct()")

    def correct_with_meta(self, text: str) -> tuple[str, dict[str, object]]:
        assert text == "teh"
        return self.corrected, {
            "parse_failed": True,
            "n_calls": 2,
            "raw_response": "format still invalid",
        }


@pytest.mark.parametrize(
    ("corrector_id", "corrected", "expected_parse_failed", "expected_calls"),
    (
        ("pyspellchecker", "the", False, 1),
        ("t5-large-spell", "the", False, 1),
        ("qwen2.5-7b-instruct", "teh", True, 2),
    ),
)
def test_correction_adapter_preserves_outcome_metadata_and_pinned_provenance(
    monkeypatch: pytest.MonkeyPatch,
    corrector_id: str,
    corrected: str,
    expected_parse_failed: bool,
    expected_calls: int,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    requested_revision = CORRECTOR_MODELS[corrector_id]["revision"]
    fake_corrector = (
        _TaggedQwenCorrector(corrected, revision=requested_revision)
        if corrector_id == "qwen2.5-7b-instruct"
        else _PlainCorrector(corrected, revision=requested_revision)
    )
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def factory(name: str, **kwargs: object) -> object:
        factory_calls.append((name, dict(kwargs)))
        return fake_corrector

    torch_seam = _TorchSeam(torch)
    runtime = ProductionCorrectionRuntime(
        _config(corrector=corrector_id),
        corrector_factory=factory,
        torch_module=torch_seam,
    )

    outcome = runtime.correct("teh")
    provenance = runtime.provenance()

    assert outcome.corrected_text == corrected
    assert outcome.parse_failed is expected_parse_failed
    assert outcome.n_calls == expected_calls
    assert outcome.raw_response == (
        "format still invalid" if corrector_id == "qwen2.5-7b-instruct" else None
    )
    assert factory_calls[0][0] == corrector_id
    if requested_revision is None:
        assert "revision" not in factory_calls[0][1]
    else:
        assert factory_calls[0][1]["revision"] == requested_revision
        assert factory_calls[0][1]["device"] == "cuda"
    assert provenance["corrector"] == corrector_id
    assert provenance["requested_revision"] == requested_revision
    assert provenance.get("model_revision") == requested_revision
    assert provenance.get("tokenizer_revision") == requested_revision
    assert provenance["protocol_sha256"] == PROTOCOL_SHA256

    runtime.close()
    assert fake_corrector.closed is True


def test_neural_correction_adapter_rejects_resolved_revision_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    corrector = _PlainCorrector("the", revision="9" * 40)

    runtime = ProductionCorrectionRuntime(
        _config(corrector="t5-large-spell"),
        corrector_factory=lambda *args, **kwargs: corrector,
        torch_module=_TorchSeam(torch),
    )

    with pytest.raises(ValueError, match="revision|pin|requested"):
        runtime.provenance()


def test_real_pyspellchecker_runtime_corrects_on_cpu_and_hashes_its_dictionary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    runtime = ProductionCorrectionRuntime(_config(corrector="pyspellchecker"))

    outcome = runtime.correct("teh speling fox")
    provenance = runtime.provenance()

    assert outcome.corrected_text == "the spelling fox"
    assert outcome.parse_failed is False
    assert provenance["device"] == "cpu"
    assert provenance["dictionary_language"] == "en"
    assert len(provenance["dictionary_sha256"]) == 64
    runtime.close()


def test_both_production_adapters_reject_a_conflicting_visible_gpu_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    correction_factory_calls: list[object] = []
    wrapper_factory_calls: list[object] = []

    def correction_factory(*args: object, **kwargs: object) -> object:
        correction_factory_calls.append((args, kwargs))
        raise AssertionError("corrector must not load after an environment conflict")

    def wrapper_factory(*args: object, **kwargs: object) -> object:
        wrapper_factory_calls.append((args, kwargs))
        raise AssertionError("evaluator must not load after an environment conflict")

    with pytest.raises(ValueError, match=r"CUDA_VISIBLE_DEVICES.*gpu-id|conflict"):
        ProductionCorrectionRuntime(
            _config(gpu_id="1"),
            corrector_factory=correction_factory,
            torch_module=_TorchSeam(torch),
        )
    with pytest.raises(ValueError, match=r"CUDA_VISIBLE_DEVICES.*gpu-id|conflict"):
        HuggingFaceSamePromptRuntime(
            _config(gpu_id="1"),
            revision=SOURCE_REVISION,
            wrapper_factory=wrapper_factory,
            torch_module=_TorchSeam(torch),
        )
    assert correction_factory_calls == []
    assert wrapper_factory_calls == []


def test_same_prompt_runtime_rejects_multi_gpu_visibility_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    calls: list[object] = []

    def wrapper_factory(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("multi-GPU evaluator must not load")

    with pytest.raises(ValueError, match=r"exactly one|single.*GPU|one.*GPU"):
        HuggingFaceSamePromptRuntime(
            _config(gpu_id="1,2"),
            revision=SOURCE_REVISION,
            wrapper_factory=wrapper_factory,
            torch_module=_TorchSeam(torch, device_count=2),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("prompts", "sample_ids", "gold_answers", "message"),
    (
        (("p", "p", "q"), ("a", "a", "b"), ("1", "1", "2"), "2 or 4|row"),
        (("p", "q"), ("a", "a"), ("1", "1"), "adjacent|duplicate|prompt"),
        (("p", "p"), ("a", "b"), ("1", "1"), "sample.*adjacent|sample.*duplicate"),
        (("p", "p"), ("a", "a"), ("1", "2"), "gold.*adjacent|gold.*duplicate"),
    ),
)
def test_same_prompt_runtime_validates_adjacent_duplicate_inputs_before_tokenization(
    prompts: tuple[str, ...],
    sample_ids: tuple[str, ...],
    gold_answers: tuple[str, ...],
    message: str,
) -> None:
    runtime = object.__new__(HuggingFaceSamePromptRuntime)
    runtime._closed = False

    with pytest.raises(ValueError, match=message):
        runtime.generate_duplicate_batch(
            prompts,
            sample_ids=sample_ids,
            gold_answers=gold_answers,
        )


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self, torch: Any, *, revision: str = SOURCE_REVISION) -> None:
        self._torch = torch
        self.padding_side = "right"
        self.init_kwargs = {"_commit_hash": revision}
        self.decode_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []
        self.decoded = {
            41: "Reasoning only. **2**",
            42: "Reasoning only. **3**",
            43: "Reasoning only. **4**",
            45: "Reasoning only. **4**",
        }

    @staticmethod
    def _ids(text: str) -> list[int]:
        return {"p": [11, 12], "long": [21, 22, 23]}[text]

    def __call__(self, text: object, **kwargs: object) -> dict[str, object]:
        if isinstance(text, str):
            assert kwargs["add_special_tokens"] is True
            return {"input_ids": self._ids(text)}
        assert isinstance(text, list)
        assert self.padding_side == "left"
        rows = [self._ids(value) for value in text]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": self._torch.tensor(padded, dtype=self._torch.long),
            "attention_mask": self._torch.tensor(masks, dtype=self._torch.long),
        }

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        ids = tuple(int(token) for token in token_ids)
        self.decode_calls.append((ids, dict(kwargs)))
        return self.decoded[ids[0]]


class _Model:
    def __init__(self, torch: Any, *, revision: str = SOURCE_REVISION) -> None:
        self._torch = torch
        self.config = SimpleNamespace(_commit_hash=revision)
        self.generation_config = SimpleNamespace(
            stop_strings=None,
            forced_eos_token_id=None,
            eos_token_id=99,
        )
        self.calls: list[dict[str, object]] = []
        self.eval_calls = 0
        self._parameter = torch.tensor(0)

    def eval(self) -> None:
        self.eval_calls += 1

    def parameters(self) -> object:
        return iter((self._parameter,))

    def generate(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        suffix = self._torch.tensor(
            (
                (41, 99, 0),
                (42, 99, 0),
                (43, 44, 99),
                (45, 46, 99),
            ),
            dtype=self._torch.long,
            device=input_ids.device,
        )
        return self._torch.cat((input_ids, suffix[: int(input_ids.shape[0])]), dim=1)


def _generation_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    benchmark: str = "gsm8k",
) -> tuple[
    HuggingFaceSamePromptRuntime,
    _Model,
    _Tokenizer,
    SimpleNamespace,
    _TorchSeam,
    list[dict[str, object]],
]:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch_seam = _TorchSeam(torch)
    model = _Model(torch)
    tokenizer = _Tokenizer(torch)
    wrapper = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        _model=model,
        _tokenizer=tokenizer,
    )
    factory_calls: list[dict[str, object]] = []

    def wrapper_factory(**kwargs: object) -> object:
        factory_calls.append(dict(kwargs))
        return wrapper

    runtime = HuggingFaceSamePromptRuntime(
        _config(benchmark=benchmark),
        revision=SOURCE_REVISION,
        wrapper_factory=wrapper_factory,
        torch_module=torch_seam,
    )
    return runtime, model, tokenizer, wrapper, torch_seam, factory_calls


def test_same_prompt_batch_slices_token_suffix_after_left_padding_and_falls_back_symmetrically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, model, tokenizer, _wrapper, _torch_seam, _factory_calls = _generation_runtime(
        monkeypatch
    )

    generations = runtime.generate_duplicate_batch(
        ("p", "p", "long", "long"),
        sample_ids=("a", "a", "b", "b"),
        gold_answers=("2", "2", "4", "4"),
    )

    assert [generation.sample_id for generation in generations] == ["a", "a", "b", "b"]
    assert [generation.token_ids for generation in generations] == [
        (41, 99),
        (42, 99),
        (43, 44, 99),
        (45, 46, 99),
    ]
    # Both adjacent arms need the final-paper empty-primary fallback. A
    # submitted-primary-only extractor would return empty strings here.
    assert [generation.extracted_answer for generation in generations] == ["2", "3", "4", "4"]
    assert [generation.is_extracted for generation in generations] == [True] * 4
    assert [generation.is_correct for generation in generations] == [True, False, True, True]
    assert [generation.method for generation in generations] == ["fallback:N3_bold"] * 4
    assert [generation.primary_method for generation in generations] == ["no_match"] * 4

    call = model.calls[0]
    assert tuple(tuple(row) for row in call["input_ids"].tolist()) == (
        (0, 11, 12),
        (0, 11, 12),
        (21, 22, 23),
        (21, 22, 23),
    )
    assert tuple(tuple(row) for row in call["attention_mask"].tolist()) == (
        (0, 1, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    )
    assert call["max_new_tokens"] == 512
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["num_return_sequences"] == 1
    assert call["temperature"] is None
    assert call["top_p"] is None
    assert call["top_k"] is None
    assert call["use_cache"] is True
    assert call["return_dict_in_generate"] is False
    assert call["output_scores"] is False
    assert call["pad_token_id"] == 0
    assert call["eos_token_id"] == [99]
    assert all(
        kwargs
        == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        for _ids, kwargs in tokenizer.decode_calls
    )


def test_same_prompt_runtime_maps_public_mmlu_pro_to_the_task_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, tokenizer, _wrapper, _torch_seam, _factory_calls = _generation_runtime(
        monkeypatch,
        benchmark="mmlu-pro",
    )
    tokenizer.decoded[41] = "Reasoning only. **(J)**"
    tokenizer.decoded[42] = "Reasoning only. **(J)**"

    generations = runtime.generate_duplicate_batch(
        ("p", "p"),
        sample_ids=("a", "a"),
        gold_answers=("J", "J"),
    )

    assert [generation.extracted_answer for generation in generations] == ["J", "J"]
    assert all(generation.is_correct for generation in generations)


def test_same_prompt_runtime_loads_the_source_revision_and_records_resolved_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, model, tokenizer, _wrapper, torch_seam, factory_calls = _generation_runtime(
        monkeypatch
    )

    provenance = runtime.provenance()

    assert factory_calls == [
        {
            "model_name": MODEL,
            "gpu_id": "1",
            "dtype": torch_seam.bfloat16,
            "wrap_for_lxt": False,
            "revision": SOURCE_REVISION,
        }
    ]
    assert model.eval_calls >= 1
    assert tokenizer.padding_side == "left"
    assert provenance["model"] == MODEL
    assert provenance["requested_revision"] == SOURCE_REVISION
    assert provenance["model_revision"] == SOURCE_REVISION
    assert provenance["tokenizer_revision"] == SOURCE_REVISION
    assert provenance["model_revision_source"] == "model-config-metadata"
    assert provenance["tokenizer_revision_source"] == "tokenizer-init-metadata"
    assert provenance["cuda_visible_devices"] == "1"
    assert provenance["gpu_name"] == "fixture-gpu"
    assert provenance["gpu_total_memory_bytes"] == 123_456
    assert provenance["protocol_sha256"] == PROTOCOL_SHA256
    assert provenance["generation"]["padding_side"] == "left"
    assert provenance["generation"]["max_new_tokens"] == 512
    assert provenance["answer_extraction"] == (
        "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1"
    )


@pytest.mark.parametrize("component", ("model", "tokenizer"))
def test_same_prompt_runtime_rejects_resolved_revision_drift(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    runtime, model, tokenizer, _wrapper, _torch_seam, _factory_calls = _generation_runtime(
        monkeypatch
    )
    if component == "model":
        model.config._commit_hash = "9" * 40
    else:
        tokenizer.init_kwargs["_commit_hash"] = "9" * 40

    with pytest.raises(ValueError, match="revision|pin|requested"):
        runtime.provenance()


def test_same_prompt_runtime_close_releases_all_model_references_and_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, _tokenizer, wrapper, torch_seam, _factory_calls = _generation_runtime(
        monkeypatch
    )

    runtime.close()

    assert runtime.model is None
    assert runtime.tokenizer is None
    assert wrapper._model is None
    assert wrapper._tokenizer is None
    assert torch_seam.cuda.empty_cache_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        runtime.generate_duplicate_batch(
            ("p", "p"),
            sample_ids=("a", "a"),
            gold_answers=("2", "2"),
        )
