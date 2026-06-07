"""M1: ΔW = W_fp16 - dequant(W_q) を layer/row/col 単位で抽出し保存する。

CLI usage (README §5):
    uv run python experiments/quantization/weight_diff.py --config configs/quantization.yaml
    uv run python experiments/quantization/weight_diff.py --config configs/quantization.yaml \\
        --variant rtn_w4 --group-size 128

Output layout:
    results/quantization_weight_diff/<run>/
        <variant_name>/delta_w/<layer_name>.pt   — ΔW tensor saved with torch.save
        <variant_name>/stats.json                — per-layer frobenius + mean_abs summary
        stats_all.json                           — combined summary across all variants

``<run>`` is an ISO-8601 timestamp (YYYY-MM-DDTHH-MM-SS) generated at launch.

Only the ``rtn`` variant is supported without additional model files; other
methods (gptq, awq, nf4, int8) require quantized model checkpoints produced
by ``experiments/quantization/quantize.py`` and are loaded via
``typo_utils.quant.loader.load_variant``.

For the rtn variant, ΔW is computed purely from the FP16 model weights using
``quant_typo_neuron.quantization.weight_diff.rtn_reconstruction_diff``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def apply_gpu_ids(gpu_ids: str | None) -> None:
    """Set CUDA_VISIBLE_DEVICES if gpu_ids is truthy.

    Must be called before torch is imported so that CUDA device
    visibility is established before the CUDA runtime initialises.

    Args:
        gpu_ids: Comma-separated GPU ids, e.g. ``"2,3"``.  When *None*
            or an empty string the environment is left unchanged.
    """
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML config file, merging a ``_base_`` reference when present."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required. Install via: uv sync --extra llm"
        ) from exc

    cfg_path = Path(path)
    with cfg_path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    # Merge base config if referenced
    if "_base_" in cfg:
        base_path = cfg_path.parent / cfg.pop("_base_")
        with base_path.open() as f:
            base: dict[str, Any] = yaml.safe_load(f) or {}
        base.update(cfg)
        cfg = base

    return cfg


def _run_rtn(
    fp16_model: Any,
    bits: int,
    group_size: int,
    out_dir: Path,
) -> dict[str, dict[str, float]]:
    """Compute RTN ΔW for every Linear layer and save tensors to ``out_dir``."""
    import torch
    import torch.nn as nn

    from quant_typo_neuron.quantization.weight_diff import (
        diff_stats,
        rtn_reconstruction_diff,
    )

    delta_dir = out_dir / "delta_w"
    delta_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float]] = {}
    for name, mod in fp16_model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        weight = mod.weight.data.float()
        delta = rtn_reconstruction_diff(weight, bits=bits, group_size=group_size)
        stats = diff_stats(delta)
        # Save tensor (replace path separators for filesystem safety)
        safe_name = name.replace("/", "__")
        torch.save(delta, delta_dir / f"{safe_name}.pt")
        summary[name] = {
            "frobenius": float(stats["frobenius"]),
            "mean_abs": float(stats["mean_abs"]),
        }
        print(f"  {name}: frobenius={stats['frobenius']:.4f}  mean_abs={stats['mean_abs']:.4f}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="M1: Compute ΔW = W_fp16 − dequant(W_q) and save per-layer tensors."
    )
    p.add_argument("--config", required=True, help="Path to configs/quantization.yaml")
    p.add_argument(
        "--variant",
        default=None,
        help="Override quantization variant name (e.g. rtn_w4, rtn_w8). "
             "Defaults to all rtn variants derived from config bits list.",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Override model_id from config (HuggingFace model identifier).",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Override group_size from config.",
    )
    p.add_argument(
        "--gpu-ids",
        default=None,
        help=(
            "Comma-separated GPU ids, e.g. '2,3'. "
            "Sets CUDA_VISIBLE_DEVICES before importing torch."
        ),
    )
    args, _overrides = p.parse_known_args()

    # Apply GPU selection BEFORE any torch / model-loading imports so that
    # CUDA_VISIBLE_DEVICES is honoured by the CUDA runtime on initialisation.
    apply_gpu_ids(args.gpu_ids)

    cfg = _load_yaml(args.config)

    model_id: str = args.model_id or cfg.get("model", "")
    if not model_id:
        print("ERROR: model_id must be set via --model-id or the config 'model' key.", file=sys.stderr)
        sys.exit(1)

    group_size: int = args.group_size or int(cfg.get("group_size", 128))

    # Determine which variants to process
    if args.variant:
        variant_names = [args.variant]
    else:
        # Default: compute for rtn variants using bits from config
        bits_list: list[int] = cfg.get("bits", [4, 8])
        variant_names = [f"rtn_w{b}" for b in bits_list]

    # Run timestamp for output directory
    run_ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    results_root = Path(__file__).parents[2] / "results" / "quantization_weight_diff" / run_ts
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_ts}")
    print(f"Model: {model_id}")
    print(f"Variants: {variant_names}")
    print(f"Results: {results_root}")

    # Load FP16 model once (all rtn variants share it)
    try:
        import torch
        from transformers import AutoModelForCausalLM

        print(f"\nLoading FP16 model: {model_id} ...")
        fp16_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        fp16_model.eval()
    except Exception as exc:
        print(f"ERROR loading model '{model_id}': {exc}", file=sys.stderr)
        sys.exit(1)

    all_stats: dict[str, Any] = {}

    for variant_name in variant_names:
        from typo_utils.quant.loader import get_variant

        try:
            variant = get_variant(variant_name)
        except KeyError:
            print(f"WARNING: Unknown variant '{variant_name}', skipping.", file=sys.stderr)
            continue

        if variant.method != "rtn":
            print(
                f"WARNING: Variant '{variant_name}' uses method '{variant.method}'. "
                "Non-rtn methods require a pre-quantized checkpoint (not yet supported "
                "by this script). Skipping.",
                file=sys.stderr,
            )
            continue

        bits: int = variant.bits or 4
        var_gs: int = variant.group_size or group_size

        print(f"\n--- Variant: {variant_name} (bits={bits}, group_size={var_gs}) ---")
        out_dir = results_root / variant_name
        summary = _run_rtn(fp16_model, bits=bits, group_size=var_gs, out_dir=out_dir)
        all_stats[variant_name] = summary

        # Save per-variant stats
        stats_path = out_dir / "stats.json"
        with stats_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Stats saved: {stats_path}")

    # Save combined summary at run root
    combined_path = results_root / "stats_all.json"
    with combined_path.open("w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nCombined stats: {combined_path}")
    print("Done.")


if __name__ == "__main__":
    main()
