"""M0: responsibility scoring -> Delta_n -> typo neuron/head mask identification.

Faithful CLI port of Tsuji et al.'s typo_neurons.py and typo_heads.py __main__ blocks.

Usage
-----
    uv run --extra llm python experiments/neuron_identification/identify.py \\
        --config configs/neuron_identification.yaml \\
        [--gpu-ids 2,3] \\
        [dataset.data_size=100] \\
        [dataset.typo_num=1]

Outputs (under results/<model_slug>/)
--------------------------------------
    sorted_neurons.pkl   -- list of [orig, typo, split] sorted-neuron rankings
    sorted_heads.pkl     -- list of [orig, typo, split] sorted-head rankings
    neuron_mask.json     -- top neuron_top_frac mask  (layer -> [dims])
    head_mask.json       -- top head_top_frac mask    (layer -> [head_ids])
    delta.npz            -- diff arrays for typo/original/split neurons+heads

GPU note: --gpu-ids sets CUDA_VISIBLE_DEVICES *before* any torch import,
matching the project-wide GPU allocation policy (GPUs 2,3 permitted).
"""
from __future__ import annotations

import argparse
import os


def _apply_gpu_ids(gpu_ids: str) -> None:
    """Set CUDA_VISIBLE_DEVICES before any torch/CUDA initialisation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a configs/*.yaml file (supports _base_ inheritance).",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated CUDA device ids to expose (e.g. '2,3'). "
             "Applied via CUDA_VISIBLE_DEVICES *before* importing torch.",
    )
    args, overrides = parser.parse_known_args()

    # --- GPU: must happen before any torch import ---
    if args.gpu_ids is not None:
        _apply_gpu_ids(args.gpu_ids)

    # --- deferred heavy imports (after GPU env is set) ---
    import pickle
    from pathlib import Path

    import numpy as np

    from typo_utils.config import load_config
    from typo_utils.llm import LLM, load_model

    from quant_typo_neuron.data.wordnet_id import (
        create_dataset,
        load_dataset,
        load_original_data,
        save_dataset,
    )
    from quant_typo_neuron.neuron_identification.heads import find_attn
    from quant_typo_neuron.neuron_identification.scoring import (
        find_neurons,
        save_mask,
        top_fraction_mask,
    )

    # --- load config ---
    cfg = load_config(args.config, overrides if overrides else None)

    model_name: str = str(cfg.model)
    model_slug = model_name.split("/")[-1]

    # --- resolve hyperparams from config ---
    # data_size: number of dataset entries to score (defaults to 100)
    data_size = int(cfg.dataset.get("data_size", 100))
    # typo_num: may live under dataset.typo_num, dataset.typo_t, or root typo_num
    typo_num = int(cfg.dataset.get("typo_num", cfg.dataset.get("typo_t", 1)))
    # fraction thresholds: prefer root-level keys, fall back to dataset sub-keys
    neuron_top_frac = float(
        cfg.get("neuron_top_frac", cfg.dataset.get("neuron_top_frac", 0.005))
    )
    head_top_frac = float(
        cfg.get("head_top_frac", cfg.dataset.get("head_top_frac", 0.015))
    )

    # --- locate or build meaning_dataset ---
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / model_slug
    dataset_path = data_dir / "meaning_dataset.json"

    model_loaded = False

    if dataset_path.exists():
        print(f"Loading existing dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
    else:
        print(f"Dataset not found at {dataset_path} -- building on the fly.")
        original_data = load_original_data()
        # eager attention so head responsibility (attention weights) can be captured
        tokenizer, model = load_model(model_name, attn_implementation="eager")
        model.eval()
        model_loaded = True
        llm = LLM(model, tokenizer)
        dataset = create_dataset(llm, original_data, threshold=0)
        dataset = sorted(dataset, key=lambda x: x["prob"], reverse=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset, dataset_path)
        print(f"Saved {len(dataset)} entries to {dataset_path}")

    # Slice dataset to data_size
    dataset = dataset[:data_size]
    print(f"Using {len(dataset)} entries (data_size={data_size}).")

    # --- load model (if not already loaded above) ---
    if not model_loaded:
        # eager attention so head responsibility (attention weights) can be captured
        tokenizer, model = load_model(model_name, attn_implementation="eager")
        model.eval()

    # --- run find_neurons ---
    print("Running find_neurons ...")
    original_sorted_neurons, typo_sorted_neurons, splited_sorted_neurons = find_neurons(
        dataset,
        model,
        tokenizer,
        vs_org=False,
        typo_num=typo_num,
    )

    # --- run find_attn ---
    print("Running find_attn ...")
    original_sorted_attn, typo_sorted_attn, splited_sorted_attn = find_attn(
        dataset,
        model,
        tokenizer,
        vs_org=False,
        typo_num=typo_num,
    )

    # --- prepare output directory ---
    out_dir = project_root / "results" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save sorted neurons (pickle) ---
    neurons_path = out_dir / "sorted_neurons.pkl"
    with open(neurons_path, "wb") as f:
        pickle.dump(
            [original_sorted_neurons, typo_sorted_neurons, splited_sorted_neurons], f
        )
    print(f"Saved sorted neurons to {neurons_path}")

    # --- save sorted heads (pickle) ---
    heads_path = out_dir / "sorted_heads.pkl"
    with open(heads_path, "wb") as f:
        pickle.dump(
            [original_sorted_attn, typo_sorted_attn, splited_sorted_attn], f
        )
    print(f"Saved sorted heads to {heads_path}")

    # --- save neuron mask (top fraction) ---
    neuron_mask = top_fraction_mask(typo_sorted_neurons, neuron_top_frac)
    neuron_mask_path = out_dir / "neuron_mask.json"
    save_mask(neuron_mask, neuron_mask_path)
    print(f"Saved neuron mask ({len(neuron_mask)} layers) to {neuron_mask_path}")

    # --- save head mask (top fraction) ---
    head_mask = top_fraction_mask(typo_sorted_attn, head_top_frac)
    head_mask_path = out_dir / "head_mask.json"
    save_mask(head_mask, head_mask_path)
    print(f"Saved head mask ({len(head_mask)} layers) to {head_mask_path}")

    # --- save delta.npz (diff arrays for downstream analysis) ---
    typo_diffs = np.array([n["diff"] for n in typo_sorted_neurons])
    original_diffs = np.array([n["diff"] for n in original_sorted_neurons])
    splited_diffs = np.array([n["diff"] for n in splited_sorted_neurons])
    head_typo_diffs = np.array([h["diff"] for h in typo_sorted_attn])

    delta_path = out_dir / "delta.npz"
    np.savez(
        delta_path,
        typo_neuron_diffs=typo_diffs,
        original_neuron_diffs=original_diffs,
        splited_neuron_diffs=splited_diffs,
        typo_head_diffs=head_typo_diffs,
    )
    print(f"Saved delta arrays to {delta_path}")

    # --- print summary (faithful to reference __main__ output) ---
    print("\noriginal (top 5 neurons):")
    for entry in original_sorted_neurons[:5]:
        print(entry)
    print("\ntypo (top 5 neurons):")
    for entry in typo_sorted_neurons[:5]:
        print(entry)
    print("\nsplit (top 5 neurons):")
    for entry in splited_sorted_neurons[:5]:
        print(entry)

    print("\ntypo (top 5 heads):")
    for entry in typo_sorted_attn[:5]:
        print(entry)

    avg_head_diff = sum(h["diff"] for h in typo_sorted_attn) / len(typo_sorted_attn)
    max_head_diff = max(h["diff"] for h in typo_sorted_attn)
    min_head_diff = min(h["diff"] for h in typo_sorted_attn)
    print(
        f"\nHead diff stats -- mean: {avg_head_diff:.6f}  "
        f"max: {max_head_diff:.6f}  min: {min_head_diff:.6f}"
    )


if __name__ == "__main__":
    main()
