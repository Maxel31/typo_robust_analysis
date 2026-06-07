"""Tests for the --gpu-ids CLI option in experiments/quantization/quantize.py.

CPU-only: does NOT import gptqmodel or torch.  The script is loaded via
importlib.util so that the experiments/ directory (which is not a package)
can be accessed without adding it to sys.path.

Instruction (quoted verbatim):
  "load `apply_gpu_ids` from the script via importlib.util (experiments dir
   is not a package) WITHOUT triggering heavy top-level imports — if the
   module top-level imports gptqmodel/torch, refactor the script so those
   are inside functions so the module is importable on CPU.
   assert apply_gpu_ids('2,3') sets os.environ['CUDA_VISIBLE_DEVICES']=='2,3';
   apply_gpu_ids(None) unchanged (save/restore env)."
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Load apply_gpu_ids from the script WITHOUT importing gptqmodel/torch
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).parent.parent
    / "experiments"
    / "quantization"
    / "quantize.py"
)


def _load_apply_gpu_ids():
    """Import only the quantize module via importlib.util and return apply_gpu_ids."""
    spec = importlib.util.spec_from_file_location("_quantize_script", _SCRIPT)
    assert spec is not None, f"Could not create spec for {_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.apply_gpu_ids


# Load once at collection time (no GPU / torch needed because heavy imports
# are all inside functions in the script).
apply_gpu_ids = _load_apply_gpu_ids()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_apply_gpu_ids_sets_env():
    """apply_gpu_ids('2,3') must set CUDA_VISIBLE_DEVICES to '2,3'."""
    old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        apply_gpu_ids("2,3")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def test_apply_gpu_ids_single_gpu():
    """apply_gpu_ids('0') must set CUDA_VISIBLE_DEVICES to '0'."""
    old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        apply_gpu_ids("0")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def test_apply_gpu_ids_none_leaves_env_unchanged():
    """apply_gpu_ids(None) must NOT modify CUDA_VISIBLE_DEVICES."""
    # Case 1: env var not set
    old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        apply_gpu_ids(None)
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    finally:
        if old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old

    # Case 2: env var already set — must remain unchanged
    os.environ["CUDA_VISIBLE_DEVICES"] = "99"
    try:
        apply_gpu_ids(None)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "99"
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def test_apply_gpu_ids_empty_string_leaves_env_unchanged():
    """apply_gpu_ids('') (falsy) must NOT modify CUDA_VISIBLE_DEVICES."""
    old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        apply_gpu_ids("")
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    finally:
        if old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
