"""実験設定 + エントリーポイントのテスト。"""

from __future__ import annotations

import subprocess
import sys

from typo_utils.config import load_config


def test_base_eval_config_loads():
    cfg = load_config("projects/quant-typo-neuron/configs/base_eval.yaml")
    assert "model" in cfg
    assert "benchmark" in cfg
    assert "seed" in cfg
    assert "gpu_ids" in cfg


def test_typo_eval_config_loads():
    cfg = load_config("projects/quant-typo-neuron/configs/typo_eval.yaml")
    assert "typo" in cfg
    assert cfg.typo.type != "clean"


def test_quant_eval_config_loads():
    cfg = load_config("projects/quant-typo-neuron/configs/quant_eval.yaml")
    assert "model" in cfg
    assert cfg.model.get("quant_method") is not None


def test_run_eval_help():
    result = subprocess.run(
        [sys.executable, "experiments/run_eval.py", "--help"],
        capture_output=True,
        text=True,
        cwd="projects/quant-typo-neuron",
    )
    assert result.returncode == 0
    assert "config" in result.stdout.lower() or "usage" in result.stdout.lower()
