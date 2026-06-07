"""Tests for the --gpu-ids CLI option in experiments/quantization/weight_diff.py.

The experiments directory is not a package, so we load the script module
directly via importlib.util.  All tests run on CPU with no models required.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load experiments/quantization/weight_diff.py as a module via importlib.util
# so that the experiments dir does not need to be a package.
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "quantization"
    / "weight_diff.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("weight_diff_script", _SCRIPT)
    assert spec is not None, f"Could not create spec for {_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_mod = _load_script()
apply_gpu_ids = _mod.apply_gpu_ids


# ---------------------------------------------------------------------------
# Helper: save and restore CUDA_VISIBLE_DEVICES around each test
# ---------------------------------------------------------------------------

class _EnvGuard:
    """Context manager that saves and restores CUDA_VISIBLE_DEVICES."""

    KEY = "CUDA_VISIBLE_DEVICES"

    def __enter__(self):
        self._saved = os.environ.get(self.KEY)
        # Remove key so tests start from a clean slate
        os.environ.pop(self.KEY, None)
        return self

    def __exit__(self, *_):
        if self._saved is None:
            os.environ.pop(self.KEY, None)
        else:
            os.environ[self.KEY] = self._saved


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApplyGpuIds:
    def test_sets_cuda_visible_devices(self):
        """apply_gpu_ids('2,3') sets CUDA_VISIBLE_DEVICES to '2,3'."""
        with _EnvGuard():
            apply_gpu_ids("2,3")
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2,3"

    def test_none_leaves_env_unchanged(self):
        """apply_gpu_ids(None) does not modify CUDA_VISIBLE_DEVICES."""
        with _EnvGuard():
            # Env var is absent at this point
            apply_gpu_ids(None)
            assert "CUDA_VISIBLE_DEVICES" not in os.environ

    def test_none_does_not_overwrite_existing_value(self):
        """apply_gpu_ids(None) leaves a pre-existing CUDA_VISIBLE_DEVICES intact."""
        with _EnvGuard():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            apply_gpu_ids(None)
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_single_gpu_id(self):
        """apply_gpu_ids('0') sets CUDA_VISIBLE_DEVICES to '0'."""
        with _EnvGuard():
            apply_gpu_ids("0")
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_empty_string_leaves_env_unchanged(self):
        """apply_gpu_ids('') (falsy) does not set CUDA_VISIBLE_DEVICES."""
        with _EnvGuard():
            apply_gpu_ids("")
            assert "CUDA_VISIBLE_DEVICES" not in os.environ
