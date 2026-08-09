# Patch and complete-text combination

This command reproduces the final paper's Table 2 as one descriptive two-by-two
comparison. The final PDF, identified by the SHA-256 returned by `typo-cot
experiments source`, is authoritative. The published cell totals are acceptance
references for the archived run, not required outcomes for newly prepared
public pairs.

| Fixed `[0,6)` patch | Clean pre-answer text | Published correct |
|---|---|---:|
| absent | none | 0/172 (0.0%) |
| present | none | 129/172 (75.0%) |
| absent | complete | 168/172 (97.7%) |
| present | complete | 171/172 (99.4%) |

The 0/172 cell is zero by selection: the primary cohort contains fresh
clean-correct, edited-wrong pairs. All four cells use those same pairs. The
patch-only failures are not used as a smaller denominator for either
complete-text condition.

## Command

Run the completed interface from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-text-combination \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --layers 0:6 \
  --gpu-id 0 \
  --output-dir results/patch-text-combination/gemma-3-4b-it/gsm8k
```

`--limit 1` is a smoke-test control and is recorded as partial. `--resume`
continues only the identical recorded run. The paper command requires exactly
Gemma-3-4B-IT, GSM8K, clean-to-edited direction, and the half-open decoder-block
window `[0,6)`.

## Reference-run contract

`--fixed-window-run` is the only cohort input. It must be a failure-free,
completed `fixed-window-answer-patching-run/v1` run that contains clean-to-edited
`0:6`. Before loading model weights, the command verifies:

- the final-paper fingerprint and fixed-window protocol;
- the recorded arguments, model, benchmark, model revision, and generation
  contract;
- every prepared-pair file and source manifest SHA-256;
- the source cohort order, pair identities, alignments, and regenerated
  baselines;
- every fixed-window public output SHA-256 and record count;
- one valid clean-to-edited `0:6` row for every included pair; and
- the reference summary against the pair-level records.

The no-text cells are reused from that verified reference rather than generated
again: `patch=absent` uses the untreated edited baseline and `patch=present`
uses the correct-coordinate `0:6` patched generation. This fixes the denominator
and makes the two commands auditable as one lineage.

The selected clean and edited baselines are nevertheless replayed under the new
runtime before complete-text generation. Every replayed token ID, decoded text,
answer, and extraction field must equal the reference. All selected baseline
replays finish before either complete-text cell is generated, so an environment
that cannot reproduce the no-text source cannot create a mixed four-cell table.
Because the supplied text comes from the prepared source rather than the replay,
the pair status and summary also record whether that stored clean continuation
is text-identical to the verified reference clean baseline; a difference is
reported rather than silently changing the legacy-backed text source.

## Complete-text boundary

For each pair, complete clean pre-answer text is derived from the prepared
source record's clean continuation. The public locator reproduces the submitted
legacy execution:

1. find every literal match of `[Tt]he answer is`;
2. cut immediately before the first match;
3. if there is no match, use the complete continuation; and
4. retain the item regardless of no-trigger, multiple-trigger, or residual-
   answer-fragment diagnostics.

The final PDF defines complete pre-answer text but does not state this character
locator, so `run.json` labels it as a legacy-backed implementation detail.
Every pair records whether a trigger was found, its count and first character
position, whether a known answer fragment remains, and the supplied character
length. Executed pairs additionally record the token length. The summary counts
no-trigger, multiple-trigger, empty-text, residual-fragment, and union-anomaly
pairs for both the full reference and the executed subset. These diagnostics do
not remove an item from the common denominator, but any anomaly prevents a
`fresh-paper-protocol-run` comparability label because the character locator is
not specified by the final PDF.

The edited prompt and complete text are concatenated as text and tokenized in
one call. The resulting IDs must begin with the exact edited prompt-only IDs;
separately tokenized ID arrays are never concatenated. Boundary instability is
a run failure, not a silent exclusion. The patch donor is captured from the
clean question-only prompt; causal attention makes those question-position
states independent of the later supplied text. The recipient is the edited
prompt plus complete clean text, and the clean edited-word-final residuals are
written to the aligned edited-word-final positions at every layer in `[0,6)`
during prefill. All later states are recomputed.

## Generation and scoring

The two complete-text cells are generated independently with bfloat16, left
padding, greedy decoding, and at most 512 new tokens. The primary GSM8K answer
extractor runs first and the deterministic fallback runs only when the primary
extractor is empty. Correctness is equality with the gold answer. An
unextractable continuation is incorrect and remains in the denominator.

`patch_text_summary.json` reports only the four per-cell success counts, common
total, and rates. It also records the cohort fingerprint and the published
historical values above. It does not calculate a confidence interval, hypothesis
test, factorial interaction, mediation quantity, bootstrap, or ranking because
the final paper reports none for Table 2.

## Outputs and restart behavior

- `patch_text_records.jsonl` contains exactly four ordered cell rows per selected
  pair, including source fingerprints, intervention identity, supplied-text
  metadata, generation tokens/text, extraction provenance, and gold
  correctness.
- `pair_status_records.jsonl` contains one row per pair in the complete reference
  denominator. It marks whether the pair was selected by `--limit` and, when
  selected, includes the verified baseline replay, clean-text boundary
  diagnostics, and exact input use.
- `patch_text_summary.json` contains descriptive cell totals, source-denominator
  composition, comparability labels, published reference metadata, and explicit
  interpretation limits.
- `run.json` contains arguments, input/output SHA-256 values, deterministic plan,
  paper and protocol fingerprints, model/tokenizer/runtime provenance, progress,
  checkpoint hashes and pair lineage, failures, and completion status.

Work is stored under `.patch-text-combination-work/baselines/` and
`.patch-text-combination-work/complete-text/`. Baselines are checkpointed
individually during the all-pair replay gate. Both newly generated complete-text
cells must then validate before a pair's complete-text checkpoint is registered.
On a runtime failure, verified earlier checkpoints remain, but no partial public
tables are published. `--resume` checks the original input, plan, runtime, and
checkpoint hashes before continuing at the first outstanding pair. A completed
resume verifies the manifest state, runtime and checkpoint lineage, every public
output hash, and the cross-file record semantics before returning without model
weights. Successful publication then attempts to remove the private work
directory. A cleanup error leaves redundant private checkpoints in place but
does not roll back or delete already durable completed outputs.

## Interpretation boundary

This is a same-pair descriptive comparison of interventions with different
sizes and timings. Complete text includes content immediately preceding the
answer, while the patch changes a few internal coordinates before generation.
The four rates do not identify an interaction, mediation path, direct or
indirect effect, necessity, sufficiency, mechanism, or deployable defense, and
they do not rank internal patching against text supply. A fresh public run is
labelled as a protocol reproduction only when the source run is complete, its
common denominator contains exactly 172 pairs and both targeting arms, prepared
clean continuations match the verified clean baselines, and the legacy-backed
boundary has no diagnostic anomaly. It does not claim the unpublished
historical 172 sample identities.
