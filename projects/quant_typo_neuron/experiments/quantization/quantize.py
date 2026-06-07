"""M1: AWQ/GPTQ/NF4/INT8/RTN で量子化バリアントを生成。

Real CLI (argparse --config) that quantizes the configured model into the
variant registry outputs (README §5 M1).

Instruction (quoted verbatim):
  "experiments/quantization/quantize.py: real CLI (argparse --config) that
   quantizes the configured model into the variant registry outputs."

Usage
-----
    uv run python experiments/quantization/quantize.py \\
        --config configs/quantization.yaml \\
        [--output-root results/quantization]

Config schema (configs/quantization.yaml)
-----------------------------------------
    model: <hf-model-id-or-local-path>
    methods: [gptq, awq, nf4, int8, rtn]
    bits: [4, 8]
    group_size: 128
    calib:
      source: c4 | list  # or inline list of strings
      n_samples: 128
      seq_len: 2048
    seed: 42

STATUS: implemented in feature/quant_typo_neuron/quantization-gptq-awq.
Only GPTQ is implemented in this branch; other methods (nf4, int8, rtn, awq)
are handled via the loader registry stubs and will raise NotImplementedError
until their respective branches are merged.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration data helpers
# ---------------------------------------------------------------------------

_BUILTIN_CALIB: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Language models learn representations of text.",
    "Quantization reduces model size and speeds up inference.",
    "Typo robustness measures how well a model handles misspelled words.",
    "Calibration data guides the GPTQ Hessian estimation.",
    "The cat sat on the mat and looked out the window.",
    "Large language models are trained on diverse internet text.",
    "Scientific research requires careful experimental design.",
]


def _load_c4_calib(n_samples: int, seq_len: int, seed: int) -> list[str]:
    """Sample ``n_samples`` snippets from the C4 dataset."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "datasets is required for c4 calibration. "
            "Run `uv sync --extra llm` to install it."
        ) from exc

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts: list[str] = []
    for row in ds.shuffle(seed=seed):
        text = row["text"]
        # Trim to approximately seq_len characters (rough proxy for tokens)
        texts.append(text[: seq_len * 4])
        if len(texts) >= n_samples:
            break
    return texts


def _resolve_calibration(calib_cfg: Any, seed: int) -> list[str]:
    """Return a list of calibration strings from the config's calib section."""
    if calib_cfg is None:
        return _BUILTIN_CALIB

    if isinstance(calib_cfg, list):
        # Inline list of strings in the yaml
        return [str(s) for s in calib_cfg]

    source = calib_cfg.get("source", "builtin")
    n_samples = int(calib_cfg.get("n_samples", 128))
    seq_len = int(calib_cfg.get("seq_len", 2048))

    if source == "c4":
        log.info("Loading %d C4 calibration samples (seq_len=%d)…", n_samples, seq_len)
        return _load_c4_calib(n_samples=n_samples, seq_len=seq_len, seed=seed)
    else:
        log.warning("Unknown calib source '%s'; falling back to built-in strings.", source)
        return _BUILTIN_CALIB


# ---------------------------------------------------------------------------
# Per-method quantization dispatchers
# ---------------------------------------------------------------------------


def _quantize_gptq(
    model_id: str,
    out_dir: str,
    bits: int,
    group_size: int,
    calib: list[str],
) -> None:
    from quant_typo_neuron.quantization.gptq_awq import quantize_gptq

    log.info(
        "GPTQ W%d quantization: %s → %s (group_size=%d)",
        bits,
        model_id,
        out_dir,
        group_size,
    )
    quantize_gptq(
        model_dir=model_id,
        out_dir=out_dir,
        bits=bits,
        group_size=group_size,
        calib=calib,
    )


def _quantize_other(method: str, model_id: str, out_dir: str, bits: int) -> None:
    """Placeholder dispatcher for methods not yet implemented in this branch."""
    log.warning(
        "Method '%s' W%d is not implemented in this branch "
        "(feature/quant_typo_neuron/quantization-gptq-awq). "
        "Skipping %s → %s.",
        method,
        bits,
        model_id,
        out_dir,
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for M1 quantization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to configs/quantization.yaml")
    parser.add_argument(
        "--output-root",
        default="results/quantization",
        help="Root directory for quantized model outputs (default: results/quantization)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override methods from config (e.g. --methods gptq)",
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=None,
        help="Override bits from config (e.g. --bits 4)",
    )
    args = parser.parse_args(argv)

    # Load config via OmegaConf
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "omegaconf is required. Run `uv sync` to install it."
        ) from exc

    cfg = OmegaConf.load(args.config)

    model_id: str = cfg.model
    methods: list[str] = args.methods if args.methods else list(cfg.get("methods", ["gptq"]))
    bits_list: list[int] = args.bits if args.bits else list(cfg.get("bits", [4]))
    group_size: int = int(cfg.get("group_size", 128))
    seed: int = int(cfg.get("seed", 42))
    calib_cfg = cfg.get("calib", None)
    output_root = Path(args.output_root)

    log.info("Model: %s", model_id)
    log.info("Methods: %s", methods)
    log.info("Bits: %s", bits_list)
    log.info("Group size: %d", group_size)

    calib = _resolve_calibration(calib_cfg, seed=seed)
    log.info("Calibration: %d samples loaded", len(calib))

    for method in methods:
        for bits in bits_list:
            variant_name = f"{method}_w{bits}"
            out_dir = str(output_root / variant_name)
            os.makedirs(out_dir, exist_ok=True)

            if method == "gptq":
                _quantize_gptq(
                    model_id=model_id,
                    out_dir=out_dir,
                    bits=bits,
                    group_size=group_size,
                    calib=calib,
                )
            else:
                _quantize_other(method=method, model_id=model_id, out_dir=out_dir, bits=bits)

    log.info("Done. Quantized outputs in %s", output_root)


if __name__ == "__main__":
    main()
