# Layerwise free-answer activation patching

## Paper contract

This operation reproduces the free-answer scan on the right of Figure 2 in the
final paper. It is intentionally separate from `layerwise-kl-patching`: the two
scans use different source samples, eligibility checks, denominators, and
readouts.

The paper grid is the Cartesian product of these four checkpoints and two
benchmarks:

- `google/gemma-3-4b-it`
- `meta-llama/Llama-3.2-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-3B-Instruct`
- `gsm8k` and `mmlu`

For each setting, the input anchor pool contains at most 300 prepared failures:
up to 150 Attribution-4 records and up to 150 Random-4 records, independently
ordered by the paper seed 42. The runtime regenerates the untreated clean and
edited continuations with greedy bfloat16 decoding and retains only cases whose
new clean answer is correct and whose new edited answer is wrong. This
`current-flip` set is fixed before interpreting patched outcomes and is reused
at every layer and in both directions.

The historical final-paper denominators and peak layers are acceptance
references, not values that a fresh public cohort is forced to equal:

| Benchmark | Model | n | Restoration peak | Induction peak |
|---|---|---:|---:|---:|
| GSM8K | Gemma-3-4B | 172 | L4 | L0 |
| GSM8K | Llama-3.2-3B | 220 | L2 | L0 |
| GSM8K | Mistral-7B | 179 | L1 | L0 |
| GSM8K | Qwen2.5-3B | 94 | L8 | L0 |
| MMLU | Gemma-3-4B | 209 | L5 | L0 |
| MMLU | Llama-3.2-3B | 226 | L0 | L0 |
| MMLU | Mistral-7B | 205 | L1 | L1 |
| MMLU | Qwen2.5-3B | 209 | L5 | L0 |

These values explain the paper's statements that all eight induction curves
and seven of eight restoration curves peak in L0–L5, with Qwen2.5-3B/GSM8K
restoration at L8 as the exception.

## Intervention and readout

At one decoder layer at a time, the complete block output at every aligned
edited-word final token is copied between the paired prompts. All positions are
patched simultaneously. L0 means the output of decoder block 0; later blocks
are recomputed. During generation the hook applies exactly once, on prompt
prefill. Decode steps use the resulting key/value cache and are not patched at
newly generated token positions.

- `clean-to-edited`: clean states are copied into the edited prompt;
  restoration is one when the patched answer equals the regenerated clean
  answer.
- `edited-to-clean`: edited states are copied into the clean prompt; induction
  is one when the patched answer differs from the regenerated clean answer.

Every layer independently generates at most 512 new tokens with greedy
decoding. The task-specific primary answer extractor runs first. Only an empty
primary result invokes the deterministic fallback. A still-empty result stays
in the denominator and contributes zero in either direction, following the
final PDF's explicit rule that unextractable answers are failures.
The final block is an audited structural no-op because an edited-word output at
that depth cannot propagate to the prompt-final generation position.

## Input requirements

Both `--attribution-pairs` and `--random-pairs` must name `pairs.jsonl` from
completed, unlimited `prepare-edited-pairs` runs. Their sibling `run.json`
manifests must agree on the final-paper SHA-256, model, benchmark, dataset
cohort, seed 42, four requested edits, 512-token generation cap, and exact
Hugging Face model revision. Each file must match its declared targeting arm.

The source revision pins both the model and tokenizer loaded for patching. The
runtime re-tokenizes every prompt and checks the recorded word-final token
coordinates against exact offset mappings before any intervention.

## Command

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot layerwise-answer-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --attribution-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --random-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/random-4/pairs.jsonl \
  --directions clean-to-edited edited-to-clean \
  --max-pairs 300 \
  --gpu-id 0 \
  --output-dir results/layerwise-answer-patching/gemma-3-4b-it/gsm8k
```

`--max-pairs 300` allocates 150 anchors to each targeting arm. Smaller positive
even values use the same balanced rule and are marked `partial-smoke-run`.
`--resume` is valid only for the same arguments, source hashes, paper protocol,
and runtime fingerprint.

## Outputs and resume behavior

The output directory contains:

- `answer_layer_records.jsonl`, one row per included pair, direction, and layer;
- `pair_status_records.jsonl`, one row per selected anchor with its baseline
  extraction and fixed-cohort status;
- `setting_summary.json`, the binary layer profiles, Wilson intervals, peaks,
  paired-bootstrap MCB sets, and population counts;
- `run.json`, the authoritative running/failed/completed manifest.

Per-anchor checkpoints are private work files until the full requested run
finishes. A pair failure marks `run.json` failed without publishing a partial
summary; `--resume` keeps valid expensive checkpoints and retries missing
pairs. A completed resume verifies every output SHA-256 and returns without
loading model weights.

Fresh `prepare-edited-pairs` outputs follow the final PDF while correcting
documented historical alignment and process-random hash defects. Consequently,
they reproduce the public protocol but do not by themselves claim the exact
historical Figure 2 IDs or rates; exact historical comparisons require the
released frozen source records.

Two discrepancies in those historical records are preserved as provenance,
not copied into the default paper-first protocol. The Qwen2.5-3B Figure 2
archives contain Attribution-4 only despite the caption saying both rules are
pooled. Also, the final plotting audit treated a still-unextractable
edited-to-clean continuation as an induced change, whereas the PDF says every
unextractable answer is a failure. The public runner therefore requires both
source arms and scores an unextractable result as zero in both directions. It
records both differences in `run.json`; a future exact-archive replay must be
explicitly labelled and must not silently replace this default.
