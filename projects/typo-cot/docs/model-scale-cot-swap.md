# Model-scale CoT-swap

`model-scale-cot-swap` reproduces Appendix C/Table 9 of the final PDF. The PDF
fingerprint reported by `typo-cot experiments source` is the primary authority.
The exact loader-order ID selector and model-specific MMLU subset caps are
submitted-producer details retained as secondary provenance.

## Experiment boundary

Table 9 is one MMLU spot check, not evidence that every paper result
generalizes to large models. It uses Attribution-4, four requested character
edits, the complete A/B/C/D CoT-swap cells, greedy bfloat16 generation, a
512-token source-continuation cap, and a 16-token regenerated answer-span cap.

The exact model grid is:

| Family | Models |
|---|---|
| Gemma 3 | 1B, 4B, 12B, 27B |
| Llama | 3.2-1B, 3.2-3B, 3.1-70B |
| Mistral | 7B-Instruct-v0.3 |
| Qwen 2.5 | 72B-Instruct |

Each model is a separate resumable `prepare-edited-pairs` setting followed by
a separate resumable `cot-swap` setting. The final `model-scale-cot-swap`
command is a CPU-only artifact builder; it never loads model weights. The
copy-paste producer and builder commands are in the root README.

## Shared first-500 selector

The final PDF says that all rows use the same first 500 MMLU sample IDs. The
submitted selector was the first 500 IDs in the seed-42 loader order with 100
examples per subject. Its public protocol artifact is
`data/cohorts/model_scale_mmlu_first500.json` and includes exactly 500 unique
IDs plus a canonical list hash.

Pair preparation intersects that selector with the model's submitted MMLU
source cohort before any model call. The submitted cohort cap was 50 examples
per subject for Gemma-1B/4B, Llama-1B/3B, and Mistral-7B, so those settings
contain 250 selected IDs. Gemma-12B/27B, Llama-70B, and Qwen-72B used 100 per
subject and contain all 500. These counts explain producer coverage; neither
250 nor 500 is an outcome denominator.

The cohort file is a deterministic protocol input, not an archived result.
Its exact identities and cap split are not printed in the PDF, so manifests
label them `submitted-source-recovered`. A changed, reordered, duplicated, or
wrong-benchmark selector fails before GPU initialization or table publication.

## Denominators and events

For every model, the source selector is narrowed by the CoT-swap edit-validity
and template gates. Among successfully executed records:

- `n_s` contains records whose regenerated A answer is correct;
- Both is `B != A` over `n_s`;
- Question only is `C != A` over `n_s`;
- CoT only is `D != A` over `n_s`; and
- `n_B` contains the `n_s` records with `B != A`; restoration is `C == A`
  over `n_B`.

All comparisons use the task's canonical extracted-answer equality. The
builder recomputes events from the stored A/B/C/D answers, then checks the
producer event flags, status denominator flags, per-setting summary, and
integer counts. It does not compare raw answer strings or reuse rounded
percentages as computational input.

## Final-PDF reference

| Model | `n_s` | Both | Question only | CoT only | Restored / `n_B` |
|---|---:|---:|---:|---:|---:|
| Gemma-3-1B | 65 | 19 | 14 | 12 | 9 / 19 |
| Gemma-3-4B | 129 | 32 | 10 | 28 | 23 / 32 |
| Gemma-3-12B | 351 | 41 | 11 | 29 | 33 / 41 |
| Gemma-3-27B | 383 | 33 | 8 | 36 | 30 / 33 |
| Llama-3.2-1B | 119 | 49 | 24 | 29 | 30 / 49 |
| Llama-3.2-3B | 142 | 36 | 11 | 32 | 27 / 36 |
| Llama-3.1-70B | 411 | 35 | 2 | 33 | 33 / 35 |
| Mistral-7B | 137 | 28 | 8 | 25 | 24 / 28 |
| Qwen2.5-72B | 331 | 10 | 7 | 12 | 8 / 10 |

The rendered paper percentages are derived from these integers. Qwen-72B is
directional only because `n_B = 10`. Table 9 reports no confidence intervals,
hypothesis tests, or cross-family trend test, so the public builder adds none.

The text following Table 9 describes the Llama sequence: Both decreases
41.2% → 25.4% → 8.5%, while restoration rises 61.2% → 75.0% → 94.3%.
The builder records this literal comparison but does not promote it to an
inferential scale law.

## Validation boundary

Before publication, the builder requires:

- the canonical final-paper SHA-256 and exact cohort artifact;
- completed, unlimited, four-edit Attribution-4 pair preparations selected by
  that cohort, with seed 42, frozen greedy decoding, model revision, dataset
  fingerprint, source-cap provenance, sorted unique IDs, and zero failures;
- completed, unlimited CoT-swap runs with the exact public protocol and output
  registry, each cryptographically linked to its prepared source;
- status coverage matching every selected pair and executed records matching
  completed statuses; and
- the exact nine model settings for a complete paper-grid comparison.

Partial valid grids are rendered with explicit missing/unexpected settings but
are not compared as the complete Table 9 grid. Duplicate settings, malformed
JSON, duplicate keys, non-finite values, wrong hashes, stale cohort IDs,
limited runs, and contradictory answer/event/count data abort without creating
the destination.

## Hardware and resume

The public producers support one visible GPU and comma-separated model-parallel
GPU IDs. Small-model smoke and execution validation use physical GPU 0 only.
The 70B/72B bfloat16 checkpoints generally need more aggregate accelerator
memory than one 95-GB device; using multiple GPUs changes hardware placement,
not the generation, extraction, cohort, or event protocol. Every model remains
an independent resumable setting. Add `--resume` to the otherwise identical
producer command only when continuing its existing output directory.

## Outputs

A successful CPU build publishes exactly:

- `model_scale_records.jsonl`: one integer-event row per complete model;
- `model_scale_summary.json`: grid coverage, reference comparison, protocol,
  provenance, and limitations;
- `table9_model_scale.csv`: tidy machine-readable cells;
- `table9_model_scale.md` and `table9_model_scale.tex`: readable fragments; and
- `run.json`: input, implementation, protocol, and output fingerprints.
