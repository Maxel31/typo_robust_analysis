# UCC-Inj reproduction

This independent project reproduces UCC-Inj experiment 6 without modifying the
existing typo-robustness training implementation.  The original experiment
compares the hidden representation of each clean GSM8K question to a copy with
one, two, or three random Unicode variation selectors appended after every
character.  These selectors are normally invisible but remain a tokenizer-level
perturbation.

For every clean/noisy pair, this implementation applies the model's chat
template, performs separate forwards, takes each hidden-state tensor at the
last non-padding model-input token, and calculates a cosine similarity per
layer.  The full run uses the 1,319 GSM8K `test` questions.  It writes a
per-example JSONL and a layer/noise-level aggregate; it never overwrites an
existing result directory.

## Install

```bash
uv sync --project projects/ucc-inj-reproduction --group dev
uv run --project projects/ucc-inj-reproduction --group dev pytest -q projects/ucc-inj-reproduction/tests
```

## GPU 0 or GPU 1

`CUDA_VISIBLE_DEVICES` remaps the physical device to logical `cuda:0`, so the
same configuration is used in both commands.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/ucc-inj-reproduction ucc-inj-reproduction exp6-cosine --config projects/ucc-inj-reproduction/configs/exp6_gsm8k.yaml --output-dir artifacts/ucc-inj/exp6-gpu0
CUDA_VISIBLE_DEVICES=1 uv run --project projects/ucc-inj-reproduction ucc-inj-reproduction exp6-cosine --config projects/ucc-inj-reproduction/configs/exp6_gsm8k.yaml --output-dir artifacts/ucc-inj/exp6-gpu1
```

For a non-destructive smoke test, add `--limit 2` and choose a new output
directory.  The resulting `per_example.jsonl` has exactly
`limit × len(noise_levels)` records.  For the full configuration this is
`1319 × 3 = 3957` records.

## Scope and fidelity

The noise construction is faithful to UCC-Inj's public
`datasets/gsm8k/main/encoder.py`: selector byte values are sampled independently
for every character, with exactly `noise_level` selectors per character.  Seeds
are derived from `(global seed, GSM8K example index, noise level)` so that the
same configuration is reproducible across Python processes and GPU assignment.

This is not a typo-robustness accuracy evaluation and does not make a semantic
denoising claim by itself.  The follow-up Linear Probe experiment will be added
on a separate branch and will use this project only after this exp6 interface is
reviewed and merged.
