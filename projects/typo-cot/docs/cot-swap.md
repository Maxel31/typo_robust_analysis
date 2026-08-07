# Complete pre-answer text CoT swap

`cot-swap` is the public implementation of the final paper's RQ2 complete-text
crossing (§3.4, §4.2, Table 1, and Appendix C). The final PDF identified by
`typo-cot experiments source` is authoritative. Legacy artifacts are consulted
only where the PDF does not define an implementation detail, and every such
detail is labelled below.

## Commands

Run one model, benchmark, and targeting arm on exactly one visible physical GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot cot-swap \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --targeting attribution-4 \
  --gpu-id 0 \
  --output-dir results/cot-swap/gemma-3-4b-it/gsm8k/attribution-4
```

Use `--targeting random-4` with the matching completed source for the parallel
control. Use `--limit 1` only for an explicitly partial smoke run. Continue an
interrupted or verify a completed identical run with `--resume`.

## Source contract

`--pairs` must name `pairs.jsonl` from a sibling completed
`prepare-edited-pairs-run/v1` manifest. Model, benchmark, targeting arm, paper
fingerprint, seed 42, requested four edits, generation protocol, source model
revision, and record counts must agree. Every JSONL record is parsed strictly,
must be sorted by unique sample ID, and must carry the requested setting plus
both prompt/continuation sides.

The source pair-preparation run must be unlimited and must contain its complete
recorded dataset cohort. `--limit` belongs to this downstream command and is
applied only after the full source is validated and the deterministic plan is
built. Source `pairs.jsonl`, source `run.json`, and each canonical record line
receive independent SHA-256 fingerprints.

## Four fixed-context cells

Let `Qc` and `Qp` be the stored clean and edited prompts, and let `Cc` and `Cp`
be the pre-answer portions of the freely generated clean and edited
continuations. One selected pair always defines all four cells:

| Cell | Fixed input | Reported comparison |
|---|---|---|
| A | `(Qc, Cc)` | clean baseline |
| B | `(Qp, Cp)` | both question and CoT changed |
| C | `(Qp, Cc)` | question-only change |
| D | `(Qc, Cp)` | CoT-only change |

The full cell input is formed as the exact recorded question-side prompt plus
the selected pre-answer text, then tokenized as one sequence. The submitted PDF
does not state whether archived token IDs or decoded text were reused; the
submitted implementation reconstructed text and retokenized it. The public
runner records this legacy-backed choice, the supplied-text SHA-256, character
and token counts, and the tokenized full-input SHA-256 for every cell.

There is no activation patch, prompt correction, or question repair in this
operation. In particular, C retains the edited question.

## Applied-edit validity

Applied-edit validity is checked before the answer-template filter. A record is
edit-valid only when its stored clean and edited prompts differ and it records
at least one applied target attempt. A zero-edit record has no typo-induced
contrast, so Appendix C/Table 8 makes restoration undefined for it. Such a
record is retained in `pair_status_records.jsonl` as `no-applied-edit`, but is
excluded before template eligibility and GPU generation.

Zero-edit exclusions and answer-template exclusions are counted separately.
Consequently, the PDF's historical 14.0% template-filter rate must not be used
as a target for, or silently combined with, the applied-edit exclusion count.

## Answer-template boundary

The final PDF reports an answer-template filter, says it removed 14.0% of
candidates, and warns that it disproportionately removed degenerate edited CoTs,
but it does not publish the exact locator. To reproduce the submitted
implementation, each stored continuation is cut immediately before its first
literal `The answer is` or `the answer is` match. Both clean and edited
continuations must pass these frozen legacy-backed checks:

- at least one trigger is present;
- exactly one trigger is present;
- its first character is not within the first 25% of the continuation;
- no earlier `Answer:`/`Answer=` or `Final Answer` fragment remains in the
  supplied prefix. A second `The answer is` is already rejected by the exact
  one-trigger check.

An edit-valid item failing either side is recorded in
`pair_status_records.jsonl` with all applicable reasons and is excluded before
GPU generation. The implementation does not tune the filter to obtain the
historical 14.0% rate. The exact prefix, boundary offset, trigger count, and
diagnostics are deterministically planned and fingerprinted before model
loading.

## Generation and extraction

For every template-eligible pair, A, B, C, and D are all regenerated from their
fixed contexts. Generation follows Appendix A exactly:

- bfloat16 model weights;
- left padding;
- greedy decoding (`do_sample=false`, one beam and one returned sequence);
- no temperature, top-p, or top-k sampling;
- cached generation;
- at most 16 new answer-span tokens.

The task-specific primary `.extract()` method is applied first, and every
non-empty primary answer is preserved. Only an empty primary result invokes the
deterministic fallback. The final PDF requires this ordering symmetrically for
A, B, C, and D, but its exact fallback regular expressions and capped-text
handling are not specified by the final PDF. The public implementation therefore
labels those rules as legacy-backed details. In particular, when generation
ends only at the 16-token cap, positional numeric fallback rules N4/N5 are
disabled so an unfinished calculation cannot be mistaken for a final answer;
boxed, answer-line, and bold fallback rules remain available. This cap gate
never replaces a non-empty primary result, including a primary result from a
looser task-extractor pattern.

An answer that remains unextractable is retained: it cannot enter the A-correct
denominator, and it fails an equality comparison in B, C, or D. Each cell
records EOS/cap termination, primary and final extraction methods, and the
setting summary cross-tabulates termination reason by extraction method.

The source model revision pins both model and tokenizer loading. Runtime
provenance records the resolved revisions, package versions, CUDA, GPU, dtype,
padding, effective EOS IDs, and exact generation arguments. One pair is
regenerated as one-pair, four-cell batch in A--D order. The answer text is
decoded only from generated token IDs with special tokens skipped and tokenizer
space cleanup disabled. The PDF specifies neither batch size nor this decode
mechanism; the submitted legacy runner used a default eight-row batch and a
full-output character slice. The public choices are versioned in runtime
provenance so `--resume` cannot mix the two execution shapes.

## Denominators and metrics

The denominator pipeline is ordered and auditable:

1. require an applied-edit-valid source record;
2. require both sides to pass the answer-template filter;
3. successfully regenerate all four cells;
4. place the pair in the common change-rate denominator only when regenerated A
   equals the gold answer;
5. report `both_changed` as `B != A`, `question_only_changed` as `C != A`, and
   `cot_only_changed` as `D != A` over that common denominator;
6. condition restoration further on `B != A`, succeeding when `C == A`.

All equality checks use non-empty canonical extracted answers. Rates are
reported as exact integer numerator/denominator pairs plus floating-point
fractions. No confidence interval, p-value, bootstrap, sign test, interaction,
mediation, or causal-effect estimate is added because the final paper reports
none for this table.

The later CPU artifact builder micro-sums integer setting counts within targeting
arm and task, then over tasks. Attribution-4 and Random-4 must never be pooled
together. A complete headline grid contains these five legacy-backed main model
identifiers across GSM8K, MMLU, MMLU-Pro, ARC, and CSQA:

- `google/gemma-3-1b-it`
- `google/gemma-3-4b-it`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

The PDF specifies five 1B--7B models but does not print this complete identifier
list in one place; the identifiers are recovered from the submitted setting
grid and recorded as implementation provenance.

This command covers that five-model headline grid. Appendix C/Table 9's separate
12B--72B MMLU scale ladder belongs to the catalogued `model-scale-cot-swap`
operation. It is not claimed as reproduced by this single-visible-GPU command;
in particular, the 70B/72B settings are not silently routed across additional
GPUs. The documented reproduction and smoke validation use physical GPU 0,
while the public argument accepts another explicitly selected single visible
GPU.

## Published references and a final-PDF conflict

The final PDF reports the task-pooled Attribution-4 values:

| Task | B-change denominator | Restoration | Question-only change | CoT-only change |
|---|---:|---:|---:|---:|
| GSM8K | 630 | 97.8% | 0.8% | 19.5% |
| MMLU | 1,716 | 76.8% | 7.9% | 21.1% |
| MMLU-Pro | 531 | 79.3% | 8.4% | 27.8% |
| ARC | 735 | 67.6% | 8.0% | 13.8% |
| CSQA | 1,022 | 67.2% | 10.8% | 18.8% |
| Pooled | 4,634 | 76.4% | 7.4% | 19.6% |

Appendix C gives the pooled counts as 19,550 A-correct cases, 4,634 B changes
(23.7%), and 3,539 restorations (76.4%). It also reports `CoT-only >
question-only` in 24/25 model-task cells for each targeting arm.

These are historical references, not values a fresh run is forced to match.
There is a material inconsistency in the submitted artifacts: Appendix A says
the deterministic fallback is applied symmetrically to every condition, while
the final aggregation that produced the printed headline retained the old
A-correct flag and rescued only later comparison cells. The public runner
follows the final-PDF symmetric rule and records this limitation; it never
labels numerical equality as proof of source identity.

Fresh pair preparation also fixes documented process-randomized seed,
Mistral-AttnLRP, and edited-word alignment defects. The public applied-edit gate
also makes zero-edit cases explicit before template filtering. Exact historical
rows require the frozen archived records, not merely the same model names, and
the fresh edit-valid cohort can differ from the legacy historical cohort.

## Outputs and restart behavior

One GPU setting writes:

- `cot_swap_records.jsonl`: one successful four-cell generation per executed
  pair, including fixed-input, EOS/cap termination, extraction provenance, and
  A-relative events;
- `pair_status_records.jsonl`: every planned source pair with template,
  selection, execution, and denominator status plus explicit reasons;
- `cot_swap_summary.json`: exact setting counts/rates, separate edit-validity
  and template exclusions, per-cell termination/extraction diagnostics,
  comparability metadata, and published references;
- `run.json`: arguments, paper/source/plan/runtime/output fingerprints,
  checkpoints, progress, failures, and comparability status.

Per-pair checkpoints bind the paper, source record, source run, pair file,
deterministic cell plan, and runtime fingerprints. Failed runs retain valid
checkpoints but publish no final records or summary. A non-completed resume
removes any crash-left public files before pending GPU work. If all selected
checkpoints already validate, it republishes without loading model weights.
Each checkpoint is durable immediately; the growing `run.json` registry is
flushed at power-of-two checkpoint counts, keeping total manifest serialization
linear while crash-left files are recovered by deterministic identity and full
semantic validation.
Successful finalization is committed by one atomic `run.json` replacement and
removes only the operation-owned work directory. A completed resume revalidates
arguments, source and output SHA-256 values, reconstructed checkpoint hashes,
every output's reconstructed semantics, schemas, record counts, and plan
fingerprint without loading model weights.

Complete pre-answer text contains near-answer content. The operation measures a
selected conditional upper bound, not population typo robustness, reasoning
faithfulness, a deployable defense, or a direct/indirect/total causal effect.
