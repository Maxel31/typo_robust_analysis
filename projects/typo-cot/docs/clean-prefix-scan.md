# Clean pre-answer token-prefix scan

`clean-prefix-scan` is the public runner for the final paper's RQ3 analysis
(§3.5, §4.3, Figure 3, Appendix D, and Table 10). The canonical PDF
fingerprint printed by `typo-cot experiments source` is authoritative. This
document distinguishes requirements stated by that PDF from implementation
details recovered from the submitted producer.

## Commands and cohort sources

The primary Gemma-3-4B/GSM8K scan uses the clean-to-edited denominator of a
completed, unlimited `fixed-window-answer-patching` run. The PDF states that
the two analyses use the same 172-pair historical cohort. A fresh public run
uses the exact hash-verified denominator of its referenced fixed-window run but
does not claim the unpublished historical sample identities.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot clean-prefix-scan \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cohort primary \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --relative-budgets 0 .02 .05 .08 .12 .16 .20 .25 .325 .40 .50 .65 .80 1 \
  --absolute-budgets 1 2 4 8 16 32 64 \
  --gpu-id 0 \
  --output-dir results/clean-prefix-scan/gemma-3-4b-it/gsm8k
```

Each extension instead consumes the completed, unlimited Attribution-4 and
Random-4 `prepare-edited-pairs/v1` files for one model/task setting. This is the
only public upstream operation that covers all fourteen extension settings
without importing CoT-swap's unrelated template and 16-token answer-span
cohort.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot clean-prefix-scan \
  --model google/gemma-3-1b-it \
  --benchmark gsm8k \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/random-4/pairs.jsonl \
  --max-pairs 150 \
  --relative-budgets 0 .02 .05 .08 .12 .16 .20 .25 .325 .40 .50 .65 .80 1 \
  --absolute-budgets 1 2 4 8 16 32 64 \
  --gpu-id 0 \
  --output-dir results/clean-prefix-scan/gemma-3-1b-it/gsm8k
```

`--fixed-window-run` is valid only with `--cohort primary`. `--pairs` and
`--max-pairs` are valid only with `--cohort extension`. `--limit` is applied
after the complete source cohort and deterministic target plan have been
fingerprinted, so a smoke run cannot change full-run selection.

## Final-PDF-defined estimand

Let `Qp` be the edited question, `Cc` the clean pre-answer text, and `L_C` its
token length. At budget `k`, the operation supplies the first `k` tokens of
`Cc` after `Qp` and freely generates a continuation. The relative budget rule
is `k = round(r * L_C)`. Point correctness means that the newly generated
answer matches the gold answer. An unextractable answer is incorrect and is
not removed.

A selected target enters every headline rate only when its scan is valid and
its newly generated `k=0` answer is wrong. On that one fixed denominator:

- point correctness is correctness at one requested relative budget;
- stable-through-later correctness is correctness there and at every longer
  distinct tested `k`;
- `k*` is the shortest tested `k` from which correctness remains true;
- the Table 10 `Short` event is `k* / L_C <= 0.2`;
- a trajectory is non-monotone when correctness changes at least twice between
  adjacent increasing distinct `k` values;
- `Full` is correctness at `k=L_C`.

At an absolute token budget `a`, exact point correctness is defined only for
rows with `L_C >= a`, and its applicability denominator therefore shrinks.
Stable recovery (`k* <= a`) remains defined for every row in the common
fresh-`k=0`-wrong cohort: a row with `L_C < a` is still a stable success when
its recovered suffix begins no later than `a`. The summary publishes both the
exact-point applicability count and the number of shorter rows so these two
denominators cannot be confused.

The PDF reports fourteen extension settings with 2,100 selected targets, 2,094
valid scans, and 1,858 fresh-`k=0` errors. Of those errors, 1,425 are correct at
complete text, 537 have `k*/L_C <= 0.2`, and 655 are non-monotone. These are
historical references, not acceptance targets for fresh corrected pair
preparation. The primary cell is reported separately: it must not be included
in the fourteen-setting aggregate.

Appendix D additionally prints primary absolute-prefix point-correct rates of
0.0, 8.7, 23.3, 24.4, 36.6, 44.8, 59.9, and 75.9 percent at
`k=0,1,2,4,8,16,32,64`. The summary retains these as printed percentages rather
than reverse-engineering integer numerators from rounded values. It also
retains the 1,858-row grid-sensitivity stable counts (537 full grid, 544
relative-only, 541 absolute-only, 545 half-relative plus absolute, and 561
sparse-relative plus absolute). The documented `k=1` legacy-exact-answer versus
canonical-gold discrepancy remains explicitly labelled and these references
are never acceptance targets for a fresh run.

Figure 3's confidence bands use 10,000 percentile bootstrap samples clustered
by `(benchmark, sample ID)` across the fourteen extensions. A single-setting
GPU runner emits the integer events needed by that later CPU artifact builder;
it does not invent a per-setting confidence interval or pool the primary cell.

## Exact token intervention

The prefix is an ID sequence, not decoded text. For each source pair, tokenize
all four strings with the pinned tokenizer revision:

1. the clean prompt `Qc`;
2. `Qc + Cc`;
3. the edited prompt `Qp`;
4. `Qp + Cc`.

The submitted selection code sliced each complete token sequence at the length
of its independently tokenized prompt and required the two remaining `Cc`
suffix sequences to be identical. It did not yet test either prompt prefix.
The scan then separately requires the complete edited input to begin with its
independently tokenized edited prompt IDs. A selected target that fails this
second check is scan-invalid and is not replaced; this two-stage detail accounts
for the historical 2,094 valid scans among 2,100 selected targets. The clean
prompt-prefix equality is not silently added as a new filter. If the common
suffix is `clean_cot_ids`, a valid budget `k` uses exactly:

```text
edited_prompt_ids + clean_cot_ids[:k]
```

Thus `k=0` is the exact edited prompt and `k=L_C` is the exact complete edited
prompt/clean-text encoding. Decoding a token prefix, appending it as text, and
retokenizing is forbidden because it can change the boundary. Only newly
generated token IDs are decoded for answer extraction; the fixed prefix is not
passed to the extractor, since an intermediate number in the supplied GSM8K
reasoning could otherwise be mistaken for the regenerated answer.

Generation is greedy, bfloat16, left-padded, cached, and capped at 512 new
tokens. Sampling controls, beam count, returned-sequence count, effective EOS
IDs, generated-ID-only decoding, stop reason, and cap termination are explicit
runtime provenance rather than inherited model defaults.

## Extension target selection

The public extension runner revalidates both prepared-pair manifests and every
record. The manifests must share the paper fingerprint, model, benchmark,
pinned model/tokenizer revision, and dataset fingerprint; use seed 42, four
requested edits, the 512-token explicit-greedy protocol, no source `--limit`,
and contain no failures.

Candidate records require a re-extracted correct clean answer, a re-extracted
wrong edited answer, at least one aligned edited word, a non-empty clean
pre-answer span, the submitted clean/edited suffix-alignment check, and
`8 <= L_C <= 512`. Stored correctness booleans are diagnostic only.
Attribution-4 and Random-4 remain separate arms throughout selection and
reporting. The edited prompt-prefix validity check occurs after selection, as
described above.

The PDF states that 150 IDs per extension are selected deterministically in
proportion to arm sizes from pools capped at 400. The submitted producer fills
in the unprinted details: sort by sample ID and cap each arm at 400, determine
arm quotas summing to `--max-pairs`, then take within an arm the indices
`floor(j * n_available / n_take)`. Selected targets that fail a final runtime
boundary check are retained as invalid statuses and are not replaced. This
algorithm is versioned as a legacy-backed selection detail.

## Paper-defined and legacy-backed details

The following are final-PDF-defined: first-`k` clean tokens under `Qp`, the
relative formula, gold correctness, fresh-`k=0` denominator, stable and
non-monotone definitions, 8--512 clean-text length filter, deterministic
150-target proportional sampling from capped pools, greedy bfloat16 left-
padded generation, 512-token cap, and cross-setting cluster bootstrap.

The following are compatible submitted-producer details not printed by the
PDF: the exact dense relative and absolute grids shown in the commands,
Python's ties-to-even `round`, the exact pre-answer trigger locator, batch size
one, per-arm interpretation of the 400 cap, quota tie handling, and the
systematic `floor` indices. Manifests and summaries label them `legacy-backed`.
Sensitivity arguments never rewrite those labels.

## Outputs

One model/task setting writes:

- `prefix_scan_records.jsonl`: one complete scan per valid selected target,
  containing source identity, token fingerprints, the distinct ordered `k`
  grid with absolute/relative origins, generated token IDs and text,
  extraction/stop provenance, correctness, stable flags, `k*`, and transition
  count;
- `pair_status_records.jsonl`: every upstream source pair with candidate,
  cap/quota/selection, token-boundary, execution, scan-validity, and fresh-
  `k=0` denominator statuses plus explicit reasons;
- `prefix_scan_summary.json`: selection funnel, exact integer numerator/
  denominator pairs for point/stable/full/short/non-monotone outcomes, source
  arm counts, absolute point-applicability diagnostics, termination/extraction
  diagnostics, comparability, and published references;
- `run.json`: arguments, paper/source fingerprints, frozen per-case plan rows,
  runtime provenance, live checkpoint or completed-output fingerprints,
  progress, failures, completion state, and paper-defined versus legacy-backed
  protocol fields.

Fresh public pair preparation fixes documented historical seed, extraction,
and alignment defects and cannot prove identity with unpublished Table 10 IDs.
A full supported setting is therefore a fresh final-PDF protocol run with
legacy-backed execution details, not an exact historical-table reproduction.

## Restart and mutation safety

Each selected `(targeting, sample_id)` owns one atomic checkpoint. Its ordered
points may grow one budget at a time, which avoids losing a long multi-budget
scan. Reuse requires exact source-record, source-manifest, full-plan, executed-
plan, budget-grid, runtime, model/tokenizer revision, and checkpoint hashes.
Every stored answer and correctness event is reconstructed before reuse;
unknown or extra fields are rejected.

A failed run retains valid checkpoints but publishes no partial JSONL or
summary. Crash-left checkpoint files can be adopted only after complete
semantic validation. Before publication and again before committing the final
completed manifest, every upstream file is rehashed. Source drift during GPU
work therefore leaves reusable checkpoints but no public result. A completed
run removes its private work directory and clears the final checkpoint registry
instead of retaining references to deleted files. Completed `--resume`
validates all output hashes; reconstructs every record, point stability flag,
status row, summary, plan/count relationship, EOS termination, and source
identity; and returns without loading model weights.
