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

`max_length` is a rejection cap, not a truncation length. The 8,192-token
default bounds the memory required by all-layer hidden-state capture; complete
inputs above it are rejected before model inference.

## Provenance and outputs

Every record stores the example index, noise seed, clean/noisy UTF-8 hashes,
input-ID hashes, token counts, selected physical indices, terminal IDs, shared
terminal-suffix length, upstream commit, and cosine vector.

Each new output directory contains:

- `config.json`
- `provenance.json`
- `per_example.jsonl`
- `layer_summary.json`

The configuration pins Gemma to immutable Hugging Face commit
`093f9f388b31de276ce2de164bdc2081324b9767`. The runtime resolves that exact
commit with `snapshot_download`, verifies the returned snapshot directory's
commit, and loads both the model and tokenizer from that one local snapshot
with network fallback disabled. It never labels a merely requested revision as
resolved. The default GSM8K revision is likewise an immutable 40-character
commit and mutable dataset refs are rejected.

The provenance manifest records the verified snapshot commit, the
tokenizer-vocabulary and chat-template hashes, dataset revision/fingerprint and
ordered cohort hash, relevant source-file and lockfile hashes, and software/GPU
versions. Injected test runtimes that bypass the verified loader are explicitly
labeled unverified.
JSON serialization rejects NaN and Infinity. Empty cohorts and empty result
payloads are rejected. Existing output directories are detected before GPU
inference and are never overwritten.

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
