# UCC-Inj exp6 protocol adaptation

This isolated project adapts the hidden-state comparison from UCC-Inj
experiment 6 to Gemma 3. It does not modify the typo-robustness training code.

The upstream experiment at
https://github.com/yifusuyi/UCC-Inj/tree/aec814fcbb4388fa6fa874fbcec08a3ab1f78190/exp6
uses `Qwen/Qwen3-30B-A3B` and checked-in
`problemset_encoded{0,1,2,3}.jsonl` files. This project instead uses
`google/gemma-3-4b-it`, loads a pinned GSM8K revision, and regenerates
variation-selector noise from stable per-example seeds. Its results are
therefore a controlled **protocol adaptation**, not a faithful reproduction of
the upstream layer curve.

## Measurement contract

For each GSM8K question and noise level, the implementation:

1. renders the same user-only chat template for clean and noisy text;
2. tokenizes the entire prompt with truncation disabled;
3. rejects an input before model inference if it exceeds the configured hard
   cap;
4. independently forwards clean and noisy inputs;
5. selects the last non-padding model-input token from each complete prompt;
6. requires the template suffix and terminal token ID to agree; and
7. reports cosine similarity for every hidden-state index.

Physical token indices may differ because variation selectors change
tokenization. The compared logical position remains the terminal token of the
complete templated input. Hidden-state index 0 is the embedding output; later
indices follow the model's returned `hidden_states` tuple.

Noise level 0 is mandatory. It is not implemented by reusing the clean result:
the identical text is encoded and forwarded a second time, and the run aborts
unless input IDs are identical and every layer's cosine is one within numerical
tolerance.

## Fail-closed behavior

The run aborts rather than emitting a partial scientific result when it sees:

- silent or explicit truncation;
- a complete input above `max_length`;
- clean/noisy template suffix or terminal-token disagreement;
- a broken level-0 identity control;
- zero-norm or non-finite hidden states/cosines; or
- a scope label other than `adaptation`.

`max_length` is a rejection cap, not a truncation length. The default
131,072-token cap matches the intended Gemma context scale, but the configured
model remains the final authority and may reject a smaller unsupported input.

## Provenance and outputs

Every record stores the example index, noise seed, clean/noisy UTF-8 hashes,
input-ID hashes, token counts, selected physical indices, terminal IDs, shared
terminal-suffix length, upstream commit, and cosine vector.

Each new output directory contains:

- `config.json`
- `provenance.json`
- `per_example.jsonl`
- `layer_summary.json`

The provenance manifest records the requested and resolved model/tokenizer
revision, chat-template hash, dataset revision/fingerprint and ordered cohort
hash, relevant source-file and lockfile hashes, and software/GPU versions.
JSON serialization rejects NaN and Infinity. Existing output directories are
never overwritten.

## Install and test

```bash
uv sync --project projects/ucc-inj-reproduction --group dev
uv run --project projects/ucc-inj-reproduction --group dev \
  pytest -q projects/ucc-inj-reproduction/tests
```

## Run

`CUDA_VISIBLE_DEVICES` remaps the selected physical GPU to logical
`cuda:0`. Choose a new output directory for every run.

```bash
CUDA_VISIBLE_DEVICES=5 uv run --project projects/ucc-inj-reproduction \
  ucc-inj-reproduction exp6-cosine \
  --config projects/ucc-inj-reproduction/configs/exp6_gsm8k.yaml \
  --output-dir artifacts/ucc-inj/exp6-gemma-adaptation
```

For a non-destructive smoke test, add `--limit 2`. The standard levels are
`[0, 1, 2, 3]`, so a full 1,319-question run produces
`1319 × 4 = 5276` records.

A faithful upstream reproduction would need the pinned Qwen model and the exact
checked-in encoded JSONL blobs. That mode is deliberately outside this PR.
