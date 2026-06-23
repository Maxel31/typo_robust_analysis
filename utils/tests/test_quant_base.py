"""量子化基盤のテスト。"""

from __future__ import annotations

import inspect

import pytest

from typo_utils.quant.base import QuantConfig, Quantizer
from typo_utils.quant.llm_compressor import (
    AWQQuantizer,
    GPTQQuantizer,
    SmoothQuantQuantizer,
    create_quantizer,
)
from typo_utils.quant.calibration import prepare_calibration_data


def test_quant_config_defaults():
    cfg = QuantConfig(method="gptq", bits=4)
    assert cfg.group_size == 128
    assert cfg.num_calibration_samples == 512


def test_quant_config_custom():
    cfg = QuantConfig(method="awq", bits=8, group_size=64, num_calibration_samples=256)
    assert cfg.group_size == 64
    assert cfg.num_calibration_samples == 256


def test_quantizer_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Quantizer()


def test_gptq_quantizer_is_quantizer():
    q = GPTQQuantizer()
    assert isinstance(q, Quantizer)
    assert callable(q.quantize)


def test_awq_quantizer_is_quantizer():
    q = AWQQuantizer()
    assert isinstance(q, Quantizer)


def test_smoothquant_quantizer_is_quantizer():
    q = SmoothQuantQuantizer()
    assert isinstance(q, Quantizer)


@pytest.mark.parametrize(
    "method,cls",
    [
        ("gptq", GPTQQuantizer),
        ("awq", AWQQuantizer),
        ("smoothquant", SmoothQuantQuantizer),
    ],
)
def test_create_quantizer_factory(method, cls):
    q = create_quantizer(method)
    assert isinstance(q, cls)


def test_create_quantizer_unknown_raises():
    with pytest.raises((ValueError, KeyError)):
        create_quantizer("nonexistent")


def test_quantizer_quantize_accepts_gpu_ids():
    q = GPTQQuantizer()
    sig = inspect.signature(q.quantize)
    assert "gpu_ids" in sig.parameters


def test_quantizer_quantize_accepts_config():
    q = GPTQQuantizer()
    sig = inspect.signature(q.quantize)
    assert "config" in sig.parameters


def test_prepare_calibration_data_callable():
    assert callable(prepare_calibration_data)
    sig = inspect.signature(prepare_calibration_data)
    assert "dataset_name" in sig.parameters
    assert "num_samples" in sig.parameters
