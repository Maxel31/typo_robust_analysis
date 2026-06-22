"""量子化スクリプトのテスト。"""

from __future__ import annotations

import subprocess
import sys


def test_prepare_calibration_data_help():
    result = subprocess.run(
        [sys.executable, "scripts/prepare_calibration_data.py", "--help"],
        capture_output=True,
        text=True,
        cwd="projects/quant-typo-neuron",
    )
    assert result.returncode == 0
    assert "calibration" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_quantize_model_help():
    result = subprocess.run(
        [sys.executable, "scripts/quantize_model.py", "--help"],
        capture_output=True,
        text=True,
        cwd="projects/quant-typo-neuron",
    )
    assert result.returncode == 0
    assert "model" in result.stdout.lower() or "usage" in result.stdout.lower()
