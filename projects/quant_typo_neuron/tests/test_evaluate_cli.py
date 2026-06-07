"""Tests for the --gpu-ids CLI option in evaluate.py.

Loads apply_gpu_ids via importlib.util so we can test it without triggering
any CUDA / torch / transformers imports.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load apply_gpu_ids from the script via importlib.util (CPU-safe, no torch)
# ---------------------------------------------------------------------------
_EVALUATE_PY = (
    Path(__file__).parent.parent
    / "experiments"
    / "robustness_evaluation"
    / "evaluate.py"
)


def _load_apply_gpu_ids():
    spec = importlib.util.spec_from_file_location("evaluate_script", _EVALUATE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.apply_gpu_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyGpuIds:
    """Unit tests for apply_gpu_ids."""

    def test_sets_cuda_visible_devices(self):
        """apply_gpu_ids('2,3') must set os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'."""
        apply_gpu_ids = _load_apply_gpu_ids()
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            apply_gpu_ids("2,3")
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
        finally:
            # Restore original environment
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def test_none_does_not_set_env(self):
        """apply_gpu_ids(None) must not modify CUDA_VISIBLE_DEVICES."""
        apply_gpu_ids = _load_apply_gpu_ids()
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            apply_gpu_ids(None)
            assert "CUDA_VISIBLE_DEVICES" not in os.environ
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def test_empty_string_does_not_set_env(self):
        """apply_gpu_ids('') (falsy) must not modify CUDA_VISIBLE_DEVICES."""
        apply_gpu_ids = _load_apply_gpu_ids()
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            apply_gpu_ids("")
            assert "CUDA_VISIBLE_DEVICES" not in os.environ
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def test_single_gpu_id(self):
        """apply_gpu_ids('0') sets CUDA_VISIBLE_DEVICES == '0'."""
        apply_gpu_ids = _load_apply_gpu_ids()
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            apply_gpu_ids("0")
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def test_none_preserves_existing_env(self):
        """apply_gpu_ids(None) leaves a pre-existing CUDA_VISIBLE_DEVICES unchanged."""
        apply_gpu_ids = _load_apply_gpu_ids()
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        try:
            apply_gpu_ids(None)
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
