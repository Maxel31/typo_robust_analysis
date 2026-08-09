# Edit-count sensitivity

`edit-count-sensitivity` reproduces Appendix C/Table 8 of the final PDF. The
canonical PDF fingerprint printed by `typo-cot experiments source` is the
authority. Archived scripts and values are comparison evidence only.

## Producer commands

Table 8 combines two analyses, so reproduction keeps their producers and
denominators separate.

For every accuracy model–benchmark setting, run `prepare-edited-pairs` under
Attribution-4 with `--num-edits 1`, `2`, and `4`. These completed runs already
contain clean and edited free-generation correctness for every source item.
They must use seed 42, greedy bfloat16 decoding, the 512-token cap, the complete
paper dataset cohort, and no `--limit`.

The PDF itself fixes the total (51) and the six benchmark names. To prevent a
different 51-cell collection from being mislabeled as Table 8, the exact
setting identities recovered from the submitted Table 8 source are also frozen:

| Model | Benchmarks |
|---|---|
| Gemma-3-1B, Gemma-3-4B, Gemma-3-12B | ARC, CSQA, GSM8K, MATH-500, MMLU, MMLU-Pro |
| Llama-3.2-1B, Llama-3.2-3B | ARC, CSQA, GSM8K, MATH-500, MMLU, MMLU-Pro |
| Mistral-7B | ARC, CSQA, GSM8K, MATH-500, MMLU, MMLU-Pro |
| Qwen2.5-0.5B, Qwen2.5-1.5B | ARC, CSQA, GSM8K, MATH-500, MMLU, MMLU-Pro |
| Qwen2.5-3B | GSM8K, MMLU, MMLU-Pro only |

This recovered identity list is secondary provenance, not a replacement for
the final PDF's methods or numeric results. The complete copy-paste producer
loop is in the root README.

For Gemma-3-4B, Llama-3.2-3B, and Mistral-7B crossed with GSM8K and MMLU, run
`cot-swap` from each prepared source with the matching
`--source-num-edits 1`, `2`, or `4`. The option defaults to four for the main
CoT-swap experiment. Values one and two bind the selected source count into the
protocol, checkpoint, records, summary, and manifest rather than changing the
four A/B/C/D cells or their extraction rules. Physical GPU 0 is the documented
reproduction device.

After the producers finish, the CPU-only command is:

```bash
uv run --project projects/typo-cot \
  typo-cot edit-count-sensitivity \
  --pairs-root results/edit-count-pairs \
  --cot-swap-runs-root results/edit-count-cot-swap \
  --edit-counts 1 2 4 \
  --output-dir results/edit-count-sensitivity
```

Both input roots are searched recursively, so their internal directory layout
does not define setting identity. Identity comes from verified manifests.
Non-Attribution runs, unrelated operations, edit counts outside the requested
set, and CoT-swap settings outside the paper's six-cell grid are ignored.
Duplicate applicable identities fail closed.

## Accuracy denominators

One complete accuracy setting has all three requested prepared sources. The
full-condition row is computed as follows:

- zero edits: `clean.answer.is_correct` from the lowest requested edit-count
  source;
- one, two, and four edits: `edited.answer.is_correct` from the corresponding
  source; and
- the reported 51-setting mean: the unweighted mean of those setting rates.

The matched row is different. Within every setting, sample IDs are intersected
over the clean, one-, two-, and four-edit conditions, and their integer correct
counts are then micro-summed across settings. For every shared ID, the clean
prompt, continuation, extracted answer, correctness, gold answer, and token
counts must agree exactly across its three independently prepared sources.
This prevents a changed clean rerun from being silently treated as one matched
condition.

The final-PDF grid has the exact 51 complete settings listed above. The PDF
reports 81,812 IDs in the matched pool and
clean accuracy above four-edit accuracy in all 51 settings. A fresh partial
grid is still rendered and labelled, but it cannot be declared the paper grid.

## CoT-swap restoration denominators

The four cells retain their original meanings:

- A: clean question and clean CoT;
- B: edited question and edited CoT;
- C: edited question and clean CoT; and
- D: clean question and edited CoT.

At each edit count separately, restoration conditions on a successfully
regenerated correct A for which B differs from A, and succeeds when C equals A.
Zero-edit restoration is undefined because no typo-induced B change exists.
The count-one, count-two, and count-four cohorts are not intersected. The
six-setting pooled rows micro-sum restored and denominator integers separately
at each count; no confidence interval, p-value, or cross-count paired claim is
added because Table 8 reports none.

## Validation boundary

The builder verifies before publishing:

- completed final-paper `prepare-edited-pairs/v1` manifests, full cohorts,
  seed, decoding, targeting, model revision, dataset fingerprint, record count,
  and strict sorted JSONL identity, with one model revision and dataset cohort
  shared across edit counts;
- completed unlimited `cot-swap-run/v1` manifests, requested edit count,
  canonical protocol hash, restoration definition, output registry and SHA-256
  values;
- exact source linkage in both the CoT-swap manifest and summary, plus each
  executed record's source-record hash;
- complete status coverage of the prepared cohort and exact agreement between
  completed status IDs, executed record IDs, denominator flags, and producer
  counts; and
- record-level restoration event logic and equality with the producer summary.

Malformed JSON, duplicate keys, non-finite values, blank JSONL rows, duplicate
settings, source mismatches, output tampering, and a restored event outside its
denominator abort without creating the output directory.

## Final-PDF reference values

The final PDF reports equal-setting accuracy of 52.1%, 50.0%, 48.3%, and 46.0%
at zero, one, two, and four edits. The corresponding 81,812-item matched rates
are 54.6%, 52.5%, 50.9%, and 48.8%.

The six-setting CoT-swap restoration totals are 811/908 (89.3%), 988/1,123
(88.0%), and 1,217/1,415 (86.0%) at one, two, and four edits. Per-setting
integer values and rates are stored in `edit_count_summary.json`; the rendered
table does not rely on rounded historical percentages as computational input.

Fresh runs are not forced to equal these values. The submitted historical
sample identities are unpublished, fresh pair preparation fixes documented
legacy seed/alignment defects, and the final PDF's symmetric answer fallback
conflicts with part of the old aggregation. Comparisons therefore require a
complete grid and use one-decimal percentage equality only as a descriptive
check; `historical_cohort_identity` remains false.

## Outputs

The destination must not exist. A successful run publishes atomically:

- `edit_count_records.jsonl`: one accuracy row per complete setting and one
  restoration row per complete six-grid setting, with integer counts and
  source hashes;
- `edit_count_summary.json`: the two aggregates, coverage, protocol,
  final-PDF references, comparison status, and limitations;
- `table8_edit_count.csv`: tidy machine-readable table cells;
- `table8_edit_count.md` and `table8_edit_count.tex`: readable fragments; and
- `run.json`: arguments, implementation identity, every producer hash, output
  hashes, and final comparability status.

The command imports no model, tokenizer, Torch, Transformers, or LRP runtime.
