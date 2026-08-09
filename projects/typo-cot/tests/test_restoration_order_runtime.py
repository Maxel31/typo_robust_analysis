"""CPU seams for the pinned, batched Table 13 generation runtime."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from typo_cot.experiments.restoration_order_accuracy.protocol import PROTOCOL_SHA256
from typo_cot.experiments.restoration_order_accuracy.runtime import (
    HuggingFaceRestorationRuntime,
)


MODEL = "google/gemma-3-4b-it"
REVISION = "1" * 40


class _Cuda:
    def __init__(self, count: int = 1) -> None:
        self.count = count
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self.count

    def manual_seed_all(self, seed: int) -> None:
        assert seed == 42

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "fixture-gpu"

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(total_memory=123_456)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _Torch:
    def __init__(self, torch: Any, count: int = 1) -> None:
        self._torch = torch
        self.bfloat16 = torch.bfloat16
        self.long = torch.long
        self.cuda = _Cuda(count)
        self.version = SimpleNamespace(cuda="fixture-cuda")

    def tensor(self, *args: object, **kwargs: object) -> object:
        return self._torch.tensor(*args, **kwargs)

    def cat(self, *args: object, **kwargs: object) -> object:
        return self._torch.cat(*args, **kwargs)

    def manual_seed(self, seed: int) -> None:
        assert seed == 42

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self, torch: Any, revision: str = REVISION) -> None:
        self.torch = torch
        self.padding_side = "right"
        self.init_kwargs = {"_commit_hash": revision}

    @staticmethod
    def _ids(text: str) -> list[int]:
        return {"p": [11, 12], "long": [21, 22, 23]}[text]

    def __call__(self, value: object, **kwargs: object) -> dict[str, object]:
        if isinstance(value, str):
            return {"input_ids": self._ids(value)}
        assert isinstance(value, list)
        assert self.padding_side == "left"
        rows = [self._ids(text) for text in value]
        width = max(map(len, rows))
        return {
            "input_ids": self.torch.tensor(
                [[0] * (width - len(row)) + row for row in rows]
            ),
            "attention_mask": self.torch.tensor(
                [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            ),
        }

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        return {41: "Reasoning only. **2**", 43: "Reasoning only. **4**"}[token_ids[0]]


class _Model:
    def __init__(self, torch: Any, revision: str = REVISION) -> None:
        self.torch = torch
        self.config = SimpleNamespace(_commit_hash=revision)
        self.generation_config = SimpleNamespace(
            stop_strings=None,
            forced_eos_token_id=None,
            eos_token_id=99,
            to_dict=lambda: {
                "eos_token_id": 99,
                "forced_eos_token_id": None,
                "stop_strings": None,
            },
        )
        self.parameter = torch.tensor(0)
        self.calls: list[dict[str, object]] = []

    def eval(self) -> None:
        return None

    def parameters(self) -> object:
        return iter((self.parameter,))

    def generate(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        inputs = kwargs["input_ids"]
        suffix = self.torch.tensor(((41, 99, 0), (43, 44, 99)))
        return self.torch.cat((inputs, suffix[: int(inputs.shape[0])]), dim=1)


def _config(gpu_id: str = "1", benchmark: str = "gsm8k") -> SimpleNamespace:
    return SimpleNamespace(
        model=MODEL,
        benchmark=benchmark,
        gpu_id=gpu_id,
        batch_size=8,
    )


def _runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[HuggingFaceRestorationRuntime, _Model, _Tokenizer, _Torch, list[dict[str, object]]]:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch_seam = _Torch(torch)
    model = _Model(torch)
    tokenizer = _Tokenizer(torch)
    wrapper = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        _model=model,
        _tokenizer=tokenizer,
    )
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return wrapper

    runtime = HuggingFaceRestorationRuntime(
        _config(),
        revision=REVISION,
        wrapper_factory=factory,
        torch_module=torch_seam,
    )
    return runtime, model, tokenizer, torch_seam, calls


def test_runtime_left_pads_one_arm_batch_and_uses_final_paper_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, model, _tokenizer, _torch, _calls = _runtime(monkeypatch)

    rows = runtime.generate_batch(
        ("p", "long"),
        sample_ids=("a", "b"),
        gold_answers=("2", "4"),
    )

    assert [row.sample_id for row in rows] == ["a", "b"]
    assert [row.token_ids for row in rows] == [(41, 99), (43, 44, 99)]
    assert [row.termination for row in rows] == ["eos", "eos"]
    assert [row.extracted_answer for row in rows] == ["2", "4"]
    assert all(row.is_correct for row in rows)
    assert [row.method for row in rows] == ["fallback:N3_bold"] * 2
    call = model.calls[0]
    assert tuple(tuple(row) for row in call["input_ids"].tolist()) == (
        (0, 11, 12),
        (21, 22, 23),
    )
    assert call["max_new_tokens"] == 512
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["eos_token_id"] == [99]


def test_runtime_loads_exact_source_revision_and_records_single_gpu_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, _tokenizer, torch_seam, calls = _runtime(monkeypatch)

    provenance = runtime.provenance()

    assert calls == [
        {
            "model_name": MODEL,
            "gpu_id": "1",
            "dtype": torch_seam.bfloat16,
            "wrap_for_lxt": False,
            "revision": REVISION,
        }
    ]
    assert provenance["protocol_sha256"] == PROTOCOL_SHA256
    assert provenance["model_revision"] == REVISION
    assert provenance["tokenizer_revision"] == REVISION
    assert provenance["cuda_visible_devices"] == "1"
    assert provenance["generation"]["batch_size"] == 8
    assert len(provenance["base_generation_config_sha256"]) == 64
    assert provenance["answer_extraction"] == (
        "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1"
    )


def test_runtime_rejects_conflicting_or_multiple_visible_gpus_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[object] = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="conflict|CUDA_VISIBLE_DEVICES"):
        HuggingFaceRestorationRuntime(
            _config("1"),
            revision=REVISION,
            wrapper_factory=lambda **kwargs: calls.append(kwargs),
            torch_module=_Torch(torch),
        )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    with pytest.raises(ValueError, match="one|single|GPU"):
        HuggingFaceRestorationRuntime(
            _config("1,2"),
            revision=REVISION,
            wrapper_factory=lambda **kwargs: calls.append(kwargs),
            torch_module=_Torch(torch, count=2),
        )
    assert calls == []


def test_runtime_rejects_invalid_batch_shape_before_tokenization() -> None:
    runtime = object.__new__(HuggingFaceRestorationRuntime)
    runtime._closed = False
    runtime.batch_size = 8
    with pytest.raises(ValueError, match="aligned|length|batch"):
        runtime.generate_batch(("p",), sample_ids=("a", "b"), gold_answers=("2",))
    with pytest.raises(ValueError, match="non-empty|batch"):
        runtime.generate_batch((), sample_ids=(), gold_answers=())


def test_runtime_close_releases_model_and_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, _tokenizer, torch_seam, _calls = _runtime(monkeypatch)
    wrapper = runtime.wrapper
    runtime.close()
    assert runtime.model is None
    assert runtime.tokenizer is None
    assert wrapper._model is None
    assert wrapper._tokenizer is None
    assert torch_seam.cuda.empty_cache_calls == 1


def test_gpu_code_identity_excludes_analysis_only_sources() -> None:
    from typo_cot.experiments.restoration_order_accuracy.analysis_integrity import (
        analysis_code_identity,
    )
    from typo_cot.experiments.restoration_order_accuracy.integrity import (
        implementation_code_identity,
    )

    producer_files = set(implementation_code_identity()["files"])
    analysis_files = set(analysis_code_identity()["files"])
    for name in ("aggregation.py", "statistics.py", "reference.py"):
        assert not any(path.endswith(name) for path in producer_files)
        assert any(path.endswith(name) for path in analysis_files)
