"""M2: 4条件(clean/typo x fp16/Q)評価。項目単位0/1を items.jsonl に保存。

Usage
-----
    uv run python experiments/robustness_evaluation/evaluate.py \
        --config configs/robustness_evaluation.yaml

The config YAML keys (all optional except model_id / variant_name):

    model_id:       str    HuggingFace model identifier
    variant:        str    QuantVariant registry name (default "fp16")
    dataset:        str    Task name (default "gsm8k")
    split:          str    Dataset split (default "test")
    limit:          int    Truncate dataset to this many items (default: all)
    typo_types:     list   Typo types to iterate (default: all 4 real typos)
    eps_levels:     list   Eps values to iterate (default: [1])
    seeds:          list   Seeds to iterate (default: [0, 1, 2, 3, 4])
    device:         str    Torch device (default "cuda:0")
    max_new_tokens: int    Greedy generation limit (default 32)
    run_name:       str    Sub-directory under results/robustness_evaluation/
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def _load_config(path: str) -> dict:
    """Load a YAML or JSON config file into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        import yaml  # type: ignore[import]
        with p.open() as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        # Fallback to JSON if pyyaml is not installed
        with p.open() as fh:
            return json.load(fh)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="M2 robustness evaluation driver: produce items.jsonl"
    )
    parser.add_argument("--config", required=True, help="path to configs/*.yaml")
    args, overrides = parser.parse_known_args(argv)

    cfg = _load_config(args.config)

    # Apply CLI overrides: --key=value or --key value
    i = 0
    while i < len(overrides):
        tok = overrides[i]
        if tok.startswith("--"):
            if "=" in tok:
                k, v = tok[2:].split("=", 1)
            else:
                k = tok[2:]
                i += 1
                v = overrides[i] if i < len(overrides) else ""
            cfg[k] = v
        i += 1

    # ---------------------------------------------------------------------------
    # Config extraction with defaults
    # ---------------------------------------------------------------------------
    model_id: str = cfg.get("model_id", "")
    variant_name: str = cfg.get("variant", "fp16")
    dataset_name: str = cfg.get("dataset", "gsm8k")
    split: str = cfg.get("split", "test")
    limit: int | None = int(cfg["limit"]) if "limit" in cfg else None
    typo_types: list[str] = cfg.get(
        "typo_types", ["sub_keyboard", "insert", "delete", "transpose"]
    )
    eps_levels: list[int | float] = cfg.get("eps_levels", [1])
    seeds: list[int] = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    device: str = cfg.get("device", "cuda:0")
    max_new_tokens: int = int(cfg.get("max_new_tokens", 32))
    run_name: str = cfg.get(
        "run_name",
        datetime.datetime.now(datetime.timezone.utc).strftime("run_%Y%m%d_%H%M%S"),
    )

    # ---------------------------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------------------------
    results_dir = Path("results") / "robustness_evaluation" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    items_path = results_dir / "items.jsonl"
    config_path = results_dir / "config.json"

    # Save config snapshot for reproducibility
    with config_path.open("w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # Load quantization variant + model
    # ---------------------------------------------------------------------------
    from typo_utils.quant.loader import load_variant  # type: ignore[import]

    print(f"[evaluate] Loading variant '{variant_name}' for model '{model_id}'...", flush=True)
    model, variant = load_variant(variant_name, model_id=model_id or None)
    method = variant.method
    bit = variant.bits
    model_label = model_id or variant_name

    # ---------------------------------------------------------------------------
    # Build predict_fn
    # ---------------------------------------------------------------------------
    from transformers import AutoTokenizer  # type: ignore[import]

    from quant_typo_neuron.robustness_evaluation.runner import make_hf_predict

    print(f"[evaluate] Loading tokenizer for '{model_id}'...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    predict = make_hf_predict(model, tokenizer, device, max_new_tokens=max_new_tokens)

    # ---------------------------------------------------------------------------
    # Load task dataset
    # ---------------------------------------------------------------------------
    from quant_typo_neuron.data.tasks import load_task

    print(
        f"[evaluate] Loading task '{dataset_name}' split='{split}' limit={limit}...",
        flush=True,
    )
    items = load_task(dataset_name, split=split, limit=limit)
    print(f"[evaluate] Loaded {len(items)} items.", flush=True)

    # ---------------------------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------------------------
    from quant_typo_neuron.robustness_evaluation.runner import run_evaluation
    from quant_typo_neuron.robustness_evaluation.schema import write_items

    n_total = len(typo_types) * len(eps_levels) * len(seeds) * len(items)
    print(
        f"[evaluate] Running: {len(typo_types)} typo_types x "
        f"{len(eps_levels)} eps x {len(seeds)} seeds x {len(items)} items "
        f"= {n_total} records...",
        flush=True,
    )

    results = run_evaluation(
        items,
        predict,
        model=model_label,
        method=method,
        bit=bit,
        dataset=dataset_name,
        typo_types=typo_types,
        eps_levels=eps_levels,
        seeds=seeds,
    )

    # ---------------------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------------------
    write_items(items_path, results)
    print(f"[evaluate] Wrote {len(results)} records to {items_path}", flush=True)


if __name__ == "__main__":
    main()
