"""M0: Build the word-identification meaning_dataset (WordNet, per-model).

Faithful CLI port of Tsuji et al.'s create_dataset.py __main__ block.

Usage
-----
    uv run --extra llm python experiments/neuron_identification/build_dataset.py \\
        --config configs/neuron_identification.yaml \\
        [--gpu-ids 2,3] \\
        [dataset.n_samples=50] \\
        [dataset.max_data_size=5000]

Outputs
-------
    data/<model_slug>/meaning_dataset.json   (gitignored; model-specific)

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
    from pathlib import Path

    from typo_utils.config import load_config
    from typo_utils.llm import LLM, load_model

    from quant_typo_neuron.data.wordnet_id import (
        create_dataset,
        load_original_data,
        save_dataset,
    )

    # --- load config ---
    cfg = load_config(args.config, overrides if overrides else None)

    # --- resolve slice sizes ---
    # dataset.n_samples    : how many entries from original_data to consider
    # dataset.max_data_size: maximum number of *passing* entries to keep
    n_samples = int(cfg.dataset.get("n_samples", 62643))
    max_data_size = int(cfg.dataset.get("max_data_size", n_samples))

    model_name: str = str(cfg.model)
    model_slug = model_name.split("/")[-1]

    # --- load original data (vendored JSON) ---
    original_data = load_original_data()
    if n_samples < len(original_data):
        original_data = original_data[:n_samples]

    # --- load model ---
    tokenizer, model = load_model(model_name)
    model.eval()
    llm = LLM(model, tokenizer)

    # --- build dataset ---
    dataset = create_dataset(llm, original_data, threshold=0)

    # sort by descending probability (reference behaviour)
    dataset = sorted(dataset, key=lambda x: x["prob"], reverse=True)

    # optionally cap at max_data_size
    if max_data_size < len(dataset):
        dataset = dataset[:max_data_size]

    # --- save ---
    # data/ is gitignored; model-specific subdirectory
    out_dir = Path(__file__).parent.parent.parent / "data" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "meaning_dataset.json"
    save_dataset(dataset, out_path)
    print(f"Saved {len(dataset)} entries to {out_path}")


if __name__ == "__main__":
    main()
