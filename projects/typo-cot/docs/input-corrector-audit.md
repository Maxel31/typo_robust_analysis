# Input-corrector audit

This document describes the public contract and runnable interface for the
input-correction audit. The final 19-page PDF defines the reported experiment;
this document explains how the public implementation reproduces it and records
recovered details. In particular, Appendix E and Table 12 define the three
correctors, the edited-word and prompt-level readouts, and the provenance
controls described below.

## Paper boundary

Table 12 reports the following historical values:

| Corrector ID | Paper label | Word | Exact clean | Same | Archive |
|---|---|---:|---:|---:|---:|
| `pyspellchecker` | Dictionary | .663 | 7,548 | 0 | 708 (9.38%) |
| `t5-large-spell` | T5-large | .886 | 21,306 | 0 | 1,874 (8.80%) |
| `qwen2.5-7b-instruct` | Qwen2.5 | .734 | 16,787 | 0 | 1,780 (10.60%) |
| Total | -- | -- | 45,641 | 0 | 4,362 (9.56%) |

`Word` is the equal-weight mean of 25 setting-level exact edited-word
restoration rates. `Exact clean` counts corrected prompts that are byte
identical to their clean prompts. `Same` compares duplicated identical prompts
inside one generation batch. The paper's `archive` column compares a separate
archived clean run; the paper explicitly states that this is neither a
corrector effect nor subtractable noise. The printed values are descriptive
historical references, not pass/fail thresholds for fresh outputs.

The core grid is five evaluator models by five tasks, with each of the three
correctors run independently:

```text
models:
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3

tasks:
  gsm8k
  mmlu
  mmlu-pro
  arc
  csqa
```

This gives 25 setting rates for each corrector and 75 completed setting runs in
the core summary. MATH-500 is not a 26th Table 12 setting. It is an optional
five-model collateral-change diagnostic for `t5-large-spell` and
`qwen2.5-7b-instruct`, for ten additional runs. The paper reports 2.562 and
.182 changed intact words per MATH-500 item for T5 and Qwen respectively; it
does not report a corresponding pyspellchecker MATH-500 result.

## Prepared-pair input

Every setting consumes `pairs.jsonl` from `prepare-edited-pairs`, together with
the sibling producer `run.json`. A paper-grid run must reject a source unless
all of the following hold:

- every record has `schema_version: prepare-edited-pairs/v1`;
- the sibling manifest has `schema_version: prepare-edited-pairs-run/v1` and a
  completed status with no failures;
- the source is an unlimited run, not a `--limit` smoke test;
- `targeting` is `attribution-4`, `seed` is 42, and
  `num_edits_requested` is 4;
- the source model and benchmark exactly match the command arguments;
- the completed item count agrees across the manifest and JSONL and equals the
  paper cohort: 1,319 GSM8K, 2,850 MMLU, 1,400 MMLU-Pro, 1,172 ARC, 1,221
  CSQA, or 500 MATH-500 records per model;
- no explicit `sample_ids` cohort is present, and the paper dataset-cohort,
  subset-cap, seeded-edit, generation, target-position, alignment, and greedy
  decoding provenance all match the frozen preparation contract;
- model revision, dataset fingerprint, and final-PDF fingerprint are present
  and valid; the completed manifest binds the exact `pairs.jsonl` path, record
  count, and SHA-256, while the loader recomputes and snapshots identities for
  both source files; and
- sample IDs, editable spans, prompt spans, prompt reconstruction, and
  `aligned_words` counts satisfy the prepared-pair contract.

The completed output inventory must contain only `pairs.jsonl`. This makes
changes to relevance, token positions, edit events, or any other pair bytes
detectable before a corrector or evaluator model is loaded. A legacy completed
source without the output binding must be regenerated with
`prepare-edited-pairs`.

Every source record remains in the setting run, including a record for which no
character change could be applied. `aligned_words` remains part of source
integrity validation, but it does not replace the submitted Table 12 word
metric. A zero-edit record may still enter item-level prompt and collateral
counts; it must not be silently removed or represented as one restored word.

The corrector receives only `edited.editable_text`. Its result is spliced into
the exact recorded edited prompt:

```text
edited.prompt[:edited.editable_prompt_span.start]
+ corrected_editable_text
+ edited.prompt[edited.editable_prompt_span.end:]
```

The runner never rebuilds a prompt from the current template implementation.
This preserves choices, demonstrations, whitespace, and all prompt bytes
outside the editable span.

## Corrector protocols and compute

The public IDs and implementations are:

- `pyspellchecker`: `pyspellchecker==0.9.0`, using its English dictionary on
  CPU. Alphabetic words and internal apostrophes are considered; one-character
  and known words are retained. A highest-frequency tie is resolved
  lexicographically so Python set iteration cannot change the result.
- `t5-large-spell`: `ai-forever/T5-large-spell` on GPU. The recovered producer
  uses the `grammar: ` prefix, corrects nonblank lines independently, preserves
  blank lines, truncates an input line at 512 tokens, and greedily generates at
  most 256 new tokens.
- `qwen2.5-7b-instruct`: `Qwen/Qwen2.5-7B-Instruct` on GPU. The recovered
  producer uses a conservative instruction that permits typo correction only,
  requires `<corrected>...</corrected>`, retries once with a format reminder,
  and returns the original text with an explicit parse-failure record if the
  second response cannot be parsed.

The PDF names the models and package version but does not print the exact T5
line protocol, Qwen instruction, retry protocol, generation caps, or resolved
Hugging Face revisions. Those recovered details are therefore labelled
`legacy-backed`, frozen behind versioned protocol and executable-code hashes,
and recorded in every run. Their literal prompts and decoding parameters are
frozen in the public implementation rather than existing only in an old
worktree or machine-local script.

Although pyspellchecker correction is CPU-only, the core end-to-end command
also performs the evaluator-model `Same` check and therefore uses a GPU after
correction. The runner checkpoints these as separate phases and closes the
correction runtime before evaluator generation. The pyspellchecker correction
phase does not load neural-corrector weights; evaluator generation still
requires a GPU. The summary builder is CPU-only: it loads no corrector or
evaluator model and uses no GPU.

## Word restoration and collateral changes

The submitted metric splits clean, edited, and corrected editable text on
whitespace and aligns clean against edited with
`difflib.SequenceMatcher(autojunk=False)`. Only equal-length `replace` spans
receive positional clean/edited word pairs. For each such pair, the runner asks
whether the corrected word at the same whitespace-word position exactly equals
its clean form. Unequal-length replacements, insertions, and deletions are
reported as unalignable instead of entering the Word denominator. The
setting-level rate is:

```text
restored actual edited words / actual edited words
```

The Table 12 `Word` value is the arithmetic mean of the 25 setting-level rates,
not a pooled word-level micro-average and not an average over item-level
percentages.

For compatibility with the submitted producer, a collateral change is an
equal-length `replace` between clean and corrected whitespace-word sequences
whose corrected-side position is not one of the clean-versus-edited change
positions. Insertions, deletions, unequal replacements, and coordinate drift
after a word-count change are not repaired or reinterpreted. This is a limited
alignment heuristic, not a general word edit distance. The optional MATH-500
readout is the total counted collateral changes divided by the number of source
items, first retained per setting and then reported with the same setting
provenance. It does not establish that a corrector is harmless on clean inputs;
the paper did not run that experiment.

## Exact-clean identity

The sole Exact-clean predicate is full-prompt UTF-8 byte identity:

```python
corrected_prompt.encode("utf-8") == clean_prompt.encode("utf-8")
```

No `strip`, whitespace collapsing, Unicode normalization, prompt-template
reconstruction, or token-ID equality may substitute for this predicate. The
record stores both prompt SHA-256 values even when they are equal. A
normalized-text restoration flag may be retained as a diagnostic, but it
cannot select the Exact-clean cohort or contribute to Table 12.

## Same-batch provenance control

For Exact-clean records, sorted source pairs are grouped two at a time. If the
two prompts are `p` and `q`, the evaluator runtime receives exactly one ordered
generation call:

```text
[p, p, q, q]
```

The last batch may contain one pair and therefore `[p, p]`. Rows `2i` and
`2i+1` are byte-identical duplicate arms for one source item. A pair generated
by two calls, two processes, or two reconstructed batches is not a `Same`
observation, even when its prompt hashes match.

The checkpoint unit is the complete generation batch, not one arm or one
sample. It stores the ordered duplicated sample IDs, source-record hashes, and
all returned generation rows. Resume validates the run-level source, protocol,
arguments, and executable code before reuse, and requires the recorded
evaluator provenance whenever Same checkpoints exist. A batch is either reused
whole or rerun whole; arms from different calls are never combined.

The legacy-backed submitted producer counts an extracted-answer difference
between duplicate arms as a `Same` event. Public records also retain generated
token IDs, decoded continuations, extraction results, correctness, and exact
generation-text identity so this interpretation can be audited. Both arms use
the same task extractor, and unextractable results remain explicit.

## Commands

The first loop prepares the complete 30-setting Attribution-4 source matrix;
the second runs the 75 core corrector settings. `GPU_ID` names the one physical
device exposed to each process and defaults to device 0.

```bash
GPU_ID="${GPU_ID:-0}"
MODELS=(
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
BENCHMARKS=(gsm8k mmlu mmlu-pro arc csqa)
SOURCE_BENCHMARKS=(gsm8k mmlu mmlu-pro arc csqa math-500)
CORRECTORS=(pyspellchecker t5-large-spell qwen2.5-7b-instruct)

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for BENCHMARK in "${SOURCE_BENCHMARKS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      uv run --project projects/typo-cot --extra lrp \
      typo-cot prepare-edited-pairs \
      --model "${MODEL}" \
      --benchmark "${BENCHMARK}" \
      --targeting attribution-4 \
      --num-edits 4 \
      --seed 42 \
      --max-new-tokens 512 \
      --gpu-id "${GPU_ID}" \
      --output-dir \
        "results/prepare-edited-pairs/${MODEL_SLUG}/${BENCHMARK}/attribution-4"
  done
done

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for BENCHMARK in "${BENCHMARKS[@]}"; do
    for CORRECTOR in "${CORRECTORS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      uv run --project projects/typo-cot --extra lrp \
        typo-cot input-corrector-audit \
        --corrector "${CORRECTOR}" \
        --model "${MODEL}" \
        --benchmark "${BENCHMARK}" \
        --pairs \
          "results/prepare-edited-pairs/${MODEL_SLUG}/${BENCHMARK}/attribution-4/pairs.jsonl" \
        --gpu-id "${GPU_ID}" \
        --output-dir \
          "results/input-corrector-audit/core/${CORRECTOR}/${MODEL_SLUG}/${BENCHMARK}"
    done
  done
done
```

The optional MATH-500 collateral audit uses the same producer command, the same
five evaluator models, and only the two neural correctors. Its output root is
kept separate from the core grid:

```bash
GPU_ID="${GPU_ID:-0}"
MODELS=(
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
CORRECTORS=(t5-large-spell qwen2.5-7b-instruct)

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for CORRECTOR in "${CORRECTORS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    uv run --project projects/typo-cot --extra lrp \
      typo-cot input-corrector-audit \
      --corrector "${CORRECTOR}" \
      --model "${MODEL}" \
      --benchmark math-500 \
      --pairs \
        "results/prepare-edited-pairs/${MODEL_SLUG}/math-500/attribution-4/pairs.jsonl" \
      --gpu-id "${GPU_ID}" \
      --output-dir \
        "results/input-corrector-audit/math-500/${CORRECTOR}/${MODEL_SLUG}"
  done
done
```

After all core runs complete, the CPU builder validates and aggregates them.
`--math-runs-root` is optional and requires the complete ten-run MATH-500 grid
when supplied:

```bash
uv run --project projects/typo-cot \
  typo-cot build-input-corrector-summary \
  --runs-root results/input-corrector-audit/core \
  --math-runs-root results/input-corrector-audit/math-500 \
  --output-dir results/input-corrector-summary
```

Omit `--math-runs-root` when reproducing Table 12 alone. `--limit N` selects a
sorted smoke prefix; a limited setting is labelled custom and the builder
rejects it as a paper-grid input.

## Setting outputs

Each `input-corrector-audit` setting publishes only after every source item and
required Same batch succeeds:

- `corrector_records.jsonl`, with source identity, corrected text, word-level
  restoration and collateral rows, Exact-clean identity, and Same-batch
  generation records;
- `corrector_audit_summary.json`, with integer denominators and setting-level
  rates recomputed from those records; and
- `run.json`, with arguments, status, protocol, paper/source/code/runtime
  provenance, progress, failures, revisions, and output hashes.

The CPU builder recursively discovers manifests and requires exactly one
completed unlimited run for every core model-task-corrector cell. It rejects
missing, duplicate, unexpected, limited, mixed-protocol, executable-code,
malformed, nonproduction-runtime, mixed-revision, cross-corrector cohort,
cross-model dataset, or output-inconsistent runs. It validates nested
generation/correction evidence and recomputes every count from
`corrector_records.jsonl`. It publishes:

- `input_corrector_summary.json`;
- `table12_input_correctors.csv`;
- `table12_input_correctors.md`;
- `table12_input_correctors.tex`; and
- `run.json`.

If the optional MATH-500 root is supplied, the summary additionally contains a
clearly separate collateral table. Those ten runs never enter the 25-setting
`Word` mean or the Table 12 Exact-clean and Same totals.

## Archive and source-pair comparisons

The current `prepare-edited-pairs` records contain clean continuations from
their own completed producer runs. Comparing a fresh corrected-input result
with that stored continuation can be useful, but it is reported as the fresh
`separate_source` / `same_batch_corrected_vs_source_pair_clean` diagnostic. It
uses the duplicate (second) arm from the same-batch pair as its current-run
endpoint. This is not a fresh reproduction of the published `archive` column
and must not be named, pooled, or rendered as that column.

The original archive lacks the complete public binding needed to establish the
same model revisions, dataset bytes, prompts, batch placement, runtime, code,
and item-level generation records. Without a separately published and
hash-verified archive artifact, the builder labels the fresh separate-source
column separately and retains the paper's 708/1,874/1,780 archive counts only
as descriptive reference metadata. It must never infer them from Same,
subtract them from a corrector result, or use them as correction-effect
estimates.

## Restart, hashes, and revisions

A new setting requires an empty output directory. Private correction
checkpoints and complete Same-batch checkpoints are written atomically in a
hidden work directory. A failed run does not publish a partial records file.
After fixing an external failure, rerun the identical command with `--resume`:

```bash
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  uv run --project projects/typo-cot --extra lrp \
  typo-cot input-corrector-audit \
  --corrector t5-large-spell \
  --model google/gemma-3-1b-it \
  --benchmark gsm8k \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
  --gpu-id "${GPU_ID}" \
  --output-dir \
    results/input-corrector-audit/core/t5-large-spell/gemma-3-1b-it/gsm8k \
  --resume
```

Resume validates the original arguments, paper and protocol hashes, source
identities, executable code, and every reused checkpoint's schema, sample
order, and source hashes. Whenever resume loads a correction or evaluator
runtime, its provenance must match the previously recorded runtime. A
checkpoint set that lacks its evaluator provenance is rejected. Completed
resume revalidates public bytes and returns without loading model weights.
Source files are hashed again immediately before publication so a mid-run input
mutation fails closed.

JSON hashing uses canonical, finite JSON and SHA-256, never Python's salted
`hash()`. Setting files use temporary writes, file and parent-directory
`fsync`, and atomic replacement. The CPU builder stages its complete output
directory and publishes it with one rename plus parent-directory `fsync`.
Producer and CPU-analysis code have separate executable identities, so a
rendering-only change does not alter the producer identity.

For neural correctors, `run.json` records requested and resolved
corrector-model and tokenizer revisions. Core records selected for `Same` also
record the evaluator model/tokenizer revisions separately; the MATH-only
diagnostic intentionally performs no evaluator generation. Runtime provenance
also records PyTorch, Transformers, Accelerate, dtype, decoding, CUDA
visibility, and hardware. For pyspellchecker it records the package and
dictionary identities. A historical cache revision must not be silently
treated as paper-specified merely because it is present on one machine.

## Interpretation limits

The audit measures recovery of deliberately selected one-character edits and
collateral changes under three specific correctors. It is not a ranking of
defenses, a natural-typo population estimate, or evidence that exact text
restoration is necessary for a correct answer. The paper also did not measure
same-run answer changes for nonidentical corrected prompts or false-correction
harm on clean inputs.

Fresh numerical differences can reflect resolved model revisions, runtime, or
hardware even when the frozen public protocol is followed. The public result is
therefore the hash-attested fresh run, while the final PDF remains the reference
for the submitted values and their interpretation.

## Prior implementation context

The earlier
[`4ldk/typo_neurons_and_heads`](https://github.com/4ldk/typo_neurons_and_heads)
repository is useful for the lineage of typo operators and terminology. It does
not define this paper's Table 12 cohort, corrector protocols, or reproduction
commands; where details differ, this implementation follows the final PDF and
records recovered producer details explicitly as `legacy-backed`.
