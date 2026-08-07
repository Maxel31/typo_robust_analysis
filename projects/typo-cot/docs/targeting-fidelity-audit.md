# Targeting-fidelity audit contract

`targeting-fidelity-audit` is the CPU-only aggregation of the perturbation
checks reported in section 3.1 and Appendix A of the final paper. The PDF is
the primary source. It reports:

- four distinct edited words in 81.8% of Attribution-4 items over 42
  model--benchmark settings and 95.7% of Random-4 items;
- zero misplaced rank-1 Attribution-4 applications among 68,650 decidable
  attempts, with misses concentrated at application ranks 2--4;
- a 30.2% miss rate over 540,724 legacy-decidable edits pooled across
  Attribution-4 and Random-4 (7,589 legacy deletions were undecidable);
- a gold-option edit in 21.5% (3,501/16,316) of the 20-setting Attribution-4
  CoT-swap included multiple-choice cohort;
- substitution, duplication, and deletion as the three seeded Table 4
  operations.

These are reference values, not hard-coded acceptance thresholds. Fresh public
pair generation intentionally fixes several historical implementation defects
documented in [`prepare-edited-pairs.md`](prepare-edited-pairs.md), so the audit
records observed values without forcing them to equal archived results. In
particular, public v1 records a Boolean landing decision for every attempt,
whereas the legacy 30.2% excluded undecidable edits. The 21.5% gold-option
reference additionally requires a later CoT-swap inclusion condition and cannot
be recomputed from prepared-pair inputs alone.

The PDF is the source for the published rounded rates and the rank-1
`0/68,650` result. Exact numerators, denominators, the rank 2--4 breakdown, and
the 3,501/16,316 cohort count come from the frozen archival reanalysis. This
distinction is recorded per metric in `run.json`; exact archival counts are not
mislabelled as verbatim PDF values. The frozen reanalysis is an author-local
artifact and is intentionally not distributed in this public repository.
`run.json` records its stable artifact identifier and SHA-256 values instead of
advertising a repository path that does not exist.

## Command

Run from the repository root after all required four-edit cells have completed:

```bash
uv run --project projects/typo-cot typo-cot targeting-fidelity-audit \
  --pairs-root results/prepare-edited-pairs \
  --output-dir results/targeting-fidelity-audit
```

The command does not load a model or require the `lrp` extra. It recursively
discovers `pairs.jsonl` files under `--pairs-root`; every file must have a
sibling completed `run.json` using the paper fingerprint and
`prepare-edited-pairs-run/v1` schema. Every pair must use
`prepare-edited-pairs/v1`, request four edits, and agree with its source
manifest on model, benchmark, targeting condition, seed, and record count.
Duplicate pair identities are rejected. `--output-dir` must not already exist,
so a previous report cannot be partially overwritten. The default
`--expected-seed 42` enforces the paper run; changing it is supported only for
an explicitly labelled sensitivity audit and is recorded in `run.json`. The
command rejects partial individual runs, but intentionally permits an
explicitly partial grid; `run.json` reports observed setting coverage instead
of falsely claiming that all 42 paper settings were supplied.

The audit does not trust a pair's landing Boolean or operation label in
isolation. Starting from the clean editable text, it independently derives the
producer's cumulative-shift landing coordinates and the Table 4 edit selected
by SHA-256 from `(seed, sample_id, target_token_index, target_token_text)` while
tracking every character's clean-text origin. The replay must reproduce the
prompt/editable coordinates, landing origin and Boolean, operation and edited
text, changed-word spans, selection ranks, and target-token indices. The
per-item attribution target must name the maximum-logit distribution after the
first CoT token, use the complete clean generation, and place that token at the
recorded clean prompt length. Random-4 must exclude attribution ranks 1--4, or
all available candidates when fewer than four exist, and may target only ranks
above four. Target-token indices must lie within the recorded clean prompt
token count, and the complete edited prompt must equal the clean prompt with
exactly its editable span replaced. Each changed word's clean and edited token
index lists must be non-empty, strictly increasing, and in bounds, with the
recorded final token equal to the list's last entry. This CPU-only audit does
not reload the tokenizer, so these are structural coordinate checks rather
than an independent reconstruction of token-to-character overlap. Zero
attempts and zero final changed words are valid producer outcomes, including a
Random-4 item with no candidate remaining after the up-to-four exclusion. JSON
duplicate keys, non-finite numbers, out-of-order sample IDs, and inconsistent
producer protocol metadata are rejected before an output directory is
published.

## Metrics

The audit keeps the paper's denominators separate:

- `four_distinct_word_rate` is the number of items with four rows in
  `aligned_words` divided by all prepared items in the row;
- `zero_attempt_items` counts retained items with no applicable selected edit;
  `zero_aligned_word_items` counts items with no final changed word, and
  `attempted_but_zero_aligned_word_items` distinguishes edits that cancel back
  to the clean text. A zero-attempt item is not counted by
  `all_attempts_faithful_items` merely because `all([])` is vacuously true;
- `targeting_fidelity_rate` is attempts with
  `landed_on_intended_token=true` divided by all target attempts;
- `selection_rank_<r>_fidelity_rate` repeats that calculation for successful
  Attribution-4 application rank `r`, matching the legacy paper audit for
  ranks 1--4. The original candidate `attribution_rank` remains in each
  per-item attempt and is not conflated with this application rank;
- `prepared_pair_gold_option_edit_rate` is prepared multiple-choice items whose
  actual changed-word span overlaps the correct option divided by all prepared
  multiple-choice inputs. It is a pair-quality descriptor, not the paper's
  conditional 21.5% CoT-swap rate;
- operation counts count target attempts, not items or distinct words.

Gold-option membership is reconstructed from the frozen clean question,
ordered choices, gold letter, and exact editable text. It uses the actual
changed-word spans in `aligned_words`, not merely the intended target span, so
cumulative-offset targeting misses are classified according to what was
really edited.

## Outputs

`targeting_fidelity_records.jsonl` uses schema
`targeting-fidelity-record/v1` and provides the per-item evidence behind every
aggregate. `targeting_fidelity.csv` contains deterministic per-setting rows,
pooled rows for each targeting condition, and an all-input row; rates are
decimal fractions and undefined rates are empty. Its prepared-pair gold-option
columns are explicitly prefixed with `prepared_` to prevent comparison with the
later conditional cohort. `operation_counts.json` uses
schema `targeting-fidelity-operation-counts/v1` and contains the same operation
counts by setting, by targeting condition, and overall.

`run.json` uses schema `targeting-fidelity-audit-run/v1`. It records the
canonical paper SHA-256, resolved arguments, SHA-256 and record count for every
input pair file and manifest, output SHA-256 values, discovered setting counts,
and the Appendix A reference values above. Its `paper_comparison` block lists
missing and unexpected cells against the exact 42-setting/two-condition paper
grid and checks seed 42, the 512-token generation cap, archival per-cell item
counts, final-paper cohort rule and per-subset cap, a non-null model revision,
and the 68,660-item total in each arm. Attribution-4 and Random-4 inputs for a
setting must have identical ordered sample IDs, reconstructed dataset hashes,
model revision, protocol identifiers, historical compatibility notes, and
recorded Python/PyTorch/Transformers/Accelerate/LXT/Datasets versions, CUDA
version, and visible GPU names; normalized model aliases cannot hide duplicate
cells. These checks require paired-arm environment identity but do not require
one hard-coded public GPU model. A partial or mixed grid is `not_comparable`;
even a complete public-v1 grid is labelled `descriptive_only` for the legacy
landing rate because the two decidability protocols differ. The public output
directory is published only after all inputs validate and all four files are
complete.

Validation retains only one input cell at a time. Each validated record is
written immediately to a temporary per-cell JSONL spool while aggregate
counters are updated incrementally; ordered sample cohorts are retained as
full SHA-256 fingerprints rather than Python object lists. The final records
file is assembled by concatenating those spools in deterministic setting order.
The temporary workspace is removed on success or failure, and the completed
four-file output still uses an atomic directory rename. On Linux the publish
step uses `RENAME_NOREPLACE`, so even an empty destination created concurrently
is not overwritten. Consequently the audit does not retain all 137,320
paper-grid records in RAM before publication.
