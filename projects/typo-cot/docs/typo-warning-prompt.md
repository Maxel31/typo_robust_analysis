# Typo-warning prompt audit

This command reproduces the Appendix E comparison of edited-input accuracy
with and without one explicit typo warning. The final PDF is the primary
experiment specification: it reports benchmark-level edited-input accuracy—the
GSM8K change from 60.1% to 54.1%, the MMLU change from 57.6% to 56.2%—and
significance only for GSM8K.

## Provenance boundary

The PDF directly specifies the warning comparison, the two tasks, the printed
accuracies, and the significance conclusion. It does not print the exact
instruction, model grid, item IDs, generation cap, batch layout, or pooling,
pairing, and exact-test implementation. The following details were
recovered from the submitted producer and are labelled
`legacy-backed`:

- models: `google/gemma-3-4b-it`,
  `meta-llama/Llama-3.2-3B-Instruct`, and
  `mistralai/Mistral-7B-Instruct-v0.3`;
- 300 selected Attribution-4 inputs per model-task setting; the historical
  selector sorted the Attribution-4/Random-4 intersection, shuffled with
  `random.Random(42)`, took 300 IDs, and sorted them again;
- one arm at a time in sorted batches of eight, with a final partial batch;
- greedy bfloat16 generation with left padding and at most 512 new tokens;
- the task-specific extractor used by the submitted producer, without the
  newer repository-level empty-answer fallback; and
- ID-paired outcomes, three-setting micro-pooling within each benchmark, and
  an exact two-sided McNemar/binomial test.

The producer did not record model revisions. The pinned revisions in the input
manifest were recovered from model-cache references in the submission
environment and are identified as such in every run. Fresh outputs are the
public reproduction results; the printed values remain descriptive comparison
values rather than pass/fail thresholds.

## Exact submitted inputs

Python's process-randomized `hash()` participated in the original typo seeds,
and its process salt was not recorded. Regenerating the same typo strings from
an arbitrary modern seed therefore cannot reproduce the submitted inputs. The
repository instead includes `data/submitted_input_edits.json`, an output-free
manifest derived from the archived producer inputs.

For each selected public-dataset item, the manifest stores only the information
needed to reconstruct the edited question or question-plus-choices and bind it
to the submitted prompt. It contains no generated continuation, extracted
answer, correctness event, or accuracy result. The loader:

1. loads GSM8K or MMLU at the manifest's pinned dataset revision;
2. validates the selected IDs, subsets, clean dataset records, and aggregate
   hashes;
3. applies the recorded edits while checking the expected text at every edit;
4. rebuilds the prompt through the public task template; and
5. validates the reconstructed no-warning and warning prompt identities.

The default command uses this bundled manifest. `--input-fixture` is an
advanced audit override; such a run is labelled custom unless its bytes match
the bundled artifact. The paper-summary builder accepts only the exact bundled
fixture and all 300 items for each of the six settings. `--limit N` executes a
sorted prefix for smoke testing and is never accepted as a paper-summary input.

## Warning intervention

The submitted instruction is:

```text
IMPORTANT: The question may contain typos or misspelled words caused by keyboard errors. Before solving, first silently correct any obvious typos to recover the intended wording, then reason step by step over the corrected text.
```

For GSM8K the insertion marker is `Now solve the following problem:`; for MMLU
it is `Now answer the following question:`. The marker must occur exactly once.
The `without-warning` prompt is the reconstructed submitted prompt byte for
byte. The `with-warning` prompt inserts `WARNING_TEXT + "\n\n"` immediately
before the marker. The implementation verifies both the exact inserted bytes
and the unchanged suffix, and fails closed on an ambiguous boundary.

## Generation and scoring

The runner processes every missing `without-warning` arm first and every
missing `with-warning` arm second. Within each arm, IDs are sorted and divided
into batches of eight. It uses greedy decoding (`do_sample=false`, one beam,
one return sequence, temperature/top-p/top-k unset), bfloat16 weights, left
padding, cache enabled, and a 512-new-token cap. Only newly generated token IDs
are decoded; EOS and capped termination are validated explicitly.

Scoring calls the same task extractor used by the submitted producer. It does
not call the repository's later `extract_with_fallback` wrapper when that
extractor returns an empty answer. Both warning arms use the same scoring path,
and unextractable generations remain in the common denominator as incorrect.

The runtime requires Python 3.12 or newer and the Appendix A package versions:
PyTorch 2.10.0, Transformers 4.57.6, and Accelerate 1.12.0. It rejects a
different runtime implementation or package version instead of labelling that
output as an exact paper-grid setting. GPU and CUDA details remain recorded
provenance rather than hardware acceptance criteria.

Each setting writes:

- `warning_prompt_records.jsonl`, containing both arm inputs, generated token
  IDs/text, extracted answers, correctness, and source/plan fingerprints;
- `warning_prompt_summary.json`, containing integer totals, paired events,
  accuracy changes, and the exact test; and
- `run.json`, containing arguments, protocol, input and executable-code hashes,
  runtime provenance, progress, failures, and output hashes.

## Pooled paper summary

`build-typo-warning-summary` recursively discovers completed setting manifests
under `--runs-root`. It requires exactly three models by two benchmarks, 300
paired records per setting, the exact bundled input artifact, unlimited runs,
and one common protocol/code identity. It revalidates source and output bytes,
reconstructs every correctness event from generated text, and never trusts a
stored percentage.

The per-setting summary is retained and byte-attested as a derived artifact,
but neither completed resume nor the paper builder reinterprets its stored
McNemar payload with the current statistics code. The builder computes all
publication metrics afresh from the semantically validated paired records.

CPU-only here means that the builder loads no model weights and uses no GPU. It
does reload GSM8K and MMLU at the pinned revisions to validate the exact source
records. The base install therefore includes `datasets`, and an uncached build
requires network access; a complete compatible Hugging Face dataset cache can
satisfy those reads offline.

For each benchmark it micro-pools three 300-item settings (`n=900`). With
`b10 = count(without-warning correct, with-warning wrong)` and
`b01 = count(without-warning wrong, with-warning correct)`, it reports the exact
two-sided binomial probability under `Binomial(b10+b01, 0.5)`; the probability
is 1 when there are no discordant pairs.

## Restart and publication

A new setting requires an empty output directory. Each completed sample-arm is
written as an atomic private checkpoint. `--resume` requires identical
arguments, protocol, submitted input, selected cohort, runtime identity,
generation/input/scoring Python-file hashes, and checkpoint hashes. CPU-only
aggregation and rendering files have a separate identity in the paper-summary
manifest, so changing them does not invalidate completed GPU generations. It
skips only validated sample-arms. A completed resume revalidates public outputs
and returns without loading model weights.

Final setting files are written atomically and private checkpoints are removed
only after complete validation. The CPU builder stages all artifacts beside the
destination and publishes them with one rename. If the subsequent parent
directory durability sync fails, it removes only the renamed directory whose
filesystem identity still matches its stage; a concurrent replacement is left
untouched and reported as an error.

## Interpretation limit

This audit covers one recovered English instruction and two tasks. It is not a
general evaluation of self-correction and is not a performance comparison with
activation patching. Benchmark-level pooling can also conceal model-level
heterogeneity, so per-setting rows remain part of every summary.
