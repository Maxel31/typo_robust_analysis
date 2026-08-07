# Targeting-fidelity audit contract

`targeting-fidelity-audit` is the CPU-only aggregation of the perturbation
checks reported in section 3.1 and Appendix A of the final paper. The PDF is
the primary source. It reports:

- four distinct edited words in 81.8% of Attribution-4 items over 42
  model--benchmark settings and 95.7% of Random-4 items;
- zero misplaced top-attribution targets among 68,650 such attempts and a
  30.2% miss rate over all Attribution-4 edits, concentrated at attribution
  ranks 2--4;
- a gold-option edit in 21.5% of multiple-choice items;
- substitution, duplication, and deletion as the three seeded Table 4
  operations.

These are reference values, not hard-coded acceptance thresholds. Fresh public
pair generation intentionally fixes several historical implementation defects
documented in [`prepare-edited-pairs.md`](prepare-edited-pairs.md), so the audit
records observed values and deltas without forcing them to equal archived
results.

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
so a previous report cannot be partially overwritten.

## Metrics

The audit keeps the paper's denominators separate:

- `four_distinct_word_rate` is the number of items with four rows in
  `aligned_words` divided by all prepared items in the row;
- `targeting_fidelity_rate` is attempts with
  `landed_on_intended_token=true` divided by all target attempts;
- `attribution_rank_<r>_fidelity_rate` repeats that attempt-level calculation
  for original AttnLRP rank `r`, for `r` from 1 through 4;
- `gold_option_edit_rate` is multiple-choice items whose actual changed-word
  span overlaps the correct option divided by all multiple-choice items;
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
decimal fractions and undefined rates are empty. `operation_counts.json` uses
schema `targeting-fidelity-operation-counts/v1` and contains the same operation
counts by setting, by targeting condition, and overall.

`run.json` uses schema `targeting-fidelity-audit-run/v1`. It records the
canonical paper SHA-256, resolved arguments, SHA-256 and record count for every
input pair file and manifest, output SHA-256 values, discovered setting counts,
and the Appendix A reference values above. The public output directory is
published only after all inputs validate and all four files are complete.
