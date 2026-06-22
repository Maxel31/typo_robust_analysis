"""QEP 量子化のテスト。"""

from __future__ import annotations

import inspect

from typo_utils.quant.base import Quantizer
from typo_utils.quant.onecompression import QEPQuantizer
from typo_utils.quant.llm_compressor import create_quantizer


def test_qep_quantizer_is_quantizer():
    q = QEPQuantizer()
    assert isinstance(q, Quantizer)
    assert callable(q.quantize)


def test_create_quantizer_returns_qep():
    q = create_quantizer("qep")
    assert isinstance(q, QEPQuantizer)


def test_qep_quantize_accepts_gpu_ids():
    q = QEPQuantizer()
    sig = inspect.signature(q.quantize)
    assert "gpu_ids" in sig.parameters


def test_qep_quantize_accepts_config():
    q = QEPQuantizer()
    sig = inspect.signature(q.quantize)
    assert "config" in sig.parameters
