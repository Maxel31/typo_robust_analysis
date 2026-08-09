# Edited-word restoration-order accuracy

This document defines the public reproduction contract for the edited-word
restoration-order experiment in Appendix E and Table 13 of the final 19-page
PDF. The experiment is an oracle diagnostic: it uses the known clean text and
the Attribution-4 relevance assigned to each edited word. It is not a fourth
input corrector, a deployable typo detector, or an estimate of correction
performance on naturally occurring typos.

## Paper boundary

The paper reports a pooled historical cohort of 1,582 items from three models
and two tasks. Starting from inputs selected because the archived clean answer
was correct and the archived four-edit answer was wrong, it freshly generated
the two endpoints and nine intermediate conditions shown below.

| Restoration order | Zero restored | One restored | Two restored | Three restored | All restored |
|---|---:|---:|---:|---:|---:|
| High relevance first | 12.0% | 42.3% | 56.1% | 68.0% | 88.9% |
| Seeded random | 12.0% | 37.9% | 52.0% | 64.8% | 88.9% |
| Low relevance first | 12.0% | 37.0% | 48.9% | 59.9% | 88.9% |

For high-relevance-first versus seeded-random restoration, the reported exact
paired p values at one, two, and three restored words are `.0018`, `.0053`,
and `.019`, respectively. These percentages, p values, and the historical
`n=1,582` are descriptive comparison values from the submitted experiment.
They are not pass/fail thresholds for a fresh public run.

The complete public grid is exactly:

```text
models:
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3

benchmarks:
  gsm8k
  mmlu
```

The public runner follows the final PDF when a recovered implementation detail
conflicts with it. In particular, the final PDF specifies the registered
task-specific extractor followed by the deterministic fallback only when the
primary result is empty; a still-unextractable answer is incorrect. The
submitted Table 13 generation script called only the primary extractor. The
public pipeline requires source outcomes produced with the final-PDF rule and
applies that same empty-only fallback to all eleven fresh conditions. It
records the historical primary-only discrepancy in provenance. This
difference, together with pinned current model revisions and dataset-content
identities, is another reason that the printed historical values cannot be
fresh-run acceptance criteria. The public command is therefore a fresh
final-PDF protocol replication; it does not claim to reconstruct the private
historical membership byte for byte.

## Prepared source and frozen cohort

Each setting consumes the complete `pairs.jsonl` produced by
`prepare-edited-pairs`, plus its sibling `run.json`. A paper-grid run rejects a
source unless all of the following are true:

- the producer completed without failures and was not a `--limit` smoke run;
- model and benchmark equal the consumer arguments exactly;
- targeting is `attribution-4`, `num_edits_requested` is 4, and the seed is
  42;
- the source contains all 1,319 GSM8K or all 2,850 MMLU records for that model;
- prompt, dataset, model-revision, relevance, edit, generation, answer
  extraction, and final-PDF provenance satisfy the frozen pair-preparation
  contract; and
- the manifest and JSONL bytes match their recorded SHA-256 identities and all
  per-record prompt spans, edit events, and source outcomes pass integrity
  validation.

The 1,319 and 2,850 source-record counts above identify the pinned public
GSM8K and MMLU dataset cohorts used by this reproduction. They are public
input-contract checks, not Table 13 result counts printed in the paper.

The setting cohort is chosen once from those source outcomes. A record is a
candidate exactly when its source clean outcome is correct and its source
Attribution-4 edited outcome is wrong. Here, “four-edit” denotes the arm that
requested four edits; the realized number of valid edit events remains an
explicit per-item value.

Submitted-compatible edit separation is then applied to each candidate. Items
that cannot satisfy that reconstruction contract are excluded before any new
condition is generated. The resulting model/task/sample identities and their
ordered edit plans form one immutable cohort fingerprint.

The runner freshly generates the no-restoration and full-restoration endpoints
along with the intermediate conditions. It must never select, discard, or
reweight an item using those fresh `k=0` or `k=4` outcomes. In particular, a
fresh edited endpoint becoming correct or a fresh clean endpoint becoming
wrong does not change the denominator. This preserves the paper's source-based
selection while measuring endpoint drift instead of hiding it.

## Submitted edit-group reconstruction

The final PDF defines the experimental comparison but does not fully specify
the character grouping or per-item random-key derivation. The compatibility
rules in this section were recovered from the submitted experiment code, then
reimplemented and tested here; they are not presented as additional facts
printed in the PDF.

Restoration acts only inside the recorded editable region. The clean editable
composite follows the submitted producer's exact convention:

- GSM8K uses the question text alone; and
- MMLU uses the question, one newline, and space-separated options formatted
  as `(A) ... (B) ...` in dataset order.

The loader verifies this composite against the prepared pair rather than
silently rebuilding a prompt from the current template. It compares the clean
and edited editable strings with
`difflib.SequenceMatcher(autojunk=False)`. Consecutive non-`equal` opcodes are
merged into one edit group until an `equal` block is reached. Groups are in
left-to-right text order.

The number of edit groups must be at least one and must exactly equal the
number of stored realized edit events. Stored events are sorted by their
source token index, and group `i` is matched to event `i`; this historical
mapping is not repaired using a new tokenizer alignment. A mismatch is a
recorded source exclusion, not permission to merge, split, or guess groups.

For a selected group, reconstruction copies its exact clean character segment.
For an unselected group, it copies the exact edited segment. Equal blocks are
copied unchanged. The following endpoint checks are mandatory:

```text
restore no groups  == recorded edited editable text
restore all groups == recorded clean editable text
```

The reconstructed editable text is spliced through the recorded prompt span.
The no-group prompt must equal the recorded edited prompt byte for byte, and
the all-group prompt must equal the recorded clean prompt byte for byte. Text
normalization, retokenized approximate replacement, and prompt-template
reconstruction cannot substitute for those checks.

For any source item with `n` realized groups, a budget at or above `n` means
the exact clean endpoint. Such an item is not dropped or padded with invented
edits merely because fewer than four valid edit events were realized.

## Restoration plans

The public order IDs are `high-relevance-first`, `seeded-random`, and
`low-relevance-first`. Every plan is a prefix of one fixed permutation, so the
set restored at budget `k` is nested within the set restored at budget `k+1`.

- `high-relevance-first` sorts by descending absolute AttnLRP relevance.
- `low-relevance-first` sorts by ascending absolute AttnLRP relevance.
- Relevance ties are resolved by the stored left-to-right event order.
- `seeded-random` uses the submitted deterministic per-item permutation below.

The submitted random-order compatibility algorithm is frozen exactly:

```python
digest = hashlib.md5(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
rng = random.Random(int(digest, 16))
order = list(range(realized_edit_count))
rng.shuffle(order)
```

The first `k` indices are restored. The default and paper seed is 42. MD5 is
used here only to preserve the submitted random row and paired p values; it is
not used as a security primitive or an artifact-integrity hash. Replacing this
derivation with SHA-256, NumPy RNG, Python's process-randomized `hash()`, or a
single grid-wide RNG stream defines a different experiment. Artifact and
provenance integrity continue to use SHA-256.

Budgets are `0 1 2 3 4`. Budget zero and budget four are shared endpoints,
not separately generated copies for each order. There are therefore eleven
fresh conditions per retained item: two endpoints plus three orders at each of
budgets one, two, and three.

## Generation and answer scoring

For each of the eleven conditions, sample identities are sorted and processed
in batches of eight, with a final partial batch. A failed batch call is retried
one prompt at a time under the same decoding and scoring protocol and that
fallback path is recorded. If any item still fails, the setting remains
incomplete and cannot enter the table builder; downstream pairwise deletion is
not allowed.

Generation uses the paper protocol:

- the model is loaded in bfloat16 with left padding;
- decoding is greedy with `do_sample=false`, one beam, and one returned
  sequence;
- temperature, top-p, and top-k are unset rather than assigned sampling
  values;
- at most 512 new tokens are generated, and only new token IDs are decoded;
- EOS and length-cap termination are recorded; and
- GSM8K uses its frozen 8-shot prompt and MMLU its frozen 5-shot prompt.

Every condition uses the same task-specific primary extractor and the same
empty-primary-only deterministic fallback. A still-empty extraction is marked
unextractable and incorrect while remaining in the common cohort. Correctness
uses the benchmark's exact canonical answer comparison; substring matching or
ad hoc numeric/letter recovery is not permitted.

## Runnable commands

From the repository root, prepare the complete matching Attribution-4 source
when it does not already exist:

```bash
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
uv run --project projects/typo-cot --extra lrp \
  typo-cot prepare-edited-pairs \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --targeting attribution-4 \
  --num-edits 4 \
  --seed 42 \
  --max-new-tokens 512 \
  --gpu-id "${GPU_ID}" \
  --output-dir \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4
```

Run one model/task setting:

```bash
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
uv run --project projects/typo-cot --extra lrp \
  typo-cot restoration-order-accuracy \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --orders high-relevance-first seeded-random low-relevance-first \
  --budgets 0 1 2 3 4 \
  --seed 42 \
  --batch-size 8 \
  --gpu-id "${GPU_ID}" \
  --output-dir \
    results/restoration-order-accuracy/gemma-3-4b-it/gsm8k
```

Repeat that setting command for exactly the three-model by two-benchmark grid,
then build a fresh Table 13 protocol replication without loading model weights:

```bash
uv run --project projects/typo-cot \
  typo-cot build-restoration-order-table \
  --runs-root results/restoration-order-accuracy \
  --output-dir results/restoration-order-table
```

`--limit N` is a labelled smoke run and is rejected by the complete-grid
builder. The producer rejects partial or reordered order/budget grids rather
than treating them as custom Table 13 runs.

## Outputs and paired statistics

Each completed setting publishes:

- `restoration_order_records.jsonl`, with the fixed source identity, realized
  groups, relevance and random plans, the shared prompt-prefix/suffix identity,
  all eleven exact prompts, generated token IDs/text, extraction provenance,
  EOS-versus-length-cap termination, singleton-retry provenance, and
  correctness events;
- `restoration_order_summary.json`, with setting-level integer totals and
  descriptive accuracies; and
- `run.json`, with arguments, source/cohort/plan identities, protocol and code
  identities, runtime provenance, progress, failures, and output hashes.

The CPU builder requires exactly six completed settings, no duplicates or
extras, and one compatible paper protocol. It verifies every manifest and
output hash. It also reopens each setting's original `pairs.jsonl` and verifies
the source manifest, selected records, gold answers, endpoint prompt spans,
edit events, and source-record hashes before reconstructing each canonical
relevance/random plan. It checks that every condition has the same exact
prompt context, recomputes correctness and extraction counts from records, and
requires the same eleven conditions for every selected model/task/sample
identity. Stored percentages and stored p values are never trusted as
aggregation inputs.

Every rendered fresh result states its pooled `n` and labels itself as a
final-PDF protocol replication. The JSON also carries only the rounded cohort,
accuracy, and p-value references printed in the final PDF. Exact legacy
numerators, discordant counts, and per-setting membership remain part of the
local audit archive rather than being represented as PDF-published facts.

Publication accuracy is a pooled item-level micro-average. The builder
concatenates the six setting cohorts by full model/task/sample identity and
computes `correct / n` for each condition. It does not give each model/task
setting equal weight. The same benchmark sample evaluated under different
models remains a different paired identity and is not collapsed.

For each budget `k` in 1, 2, and 3, let:

```text
b10 = high-relevance-first correct and seeded-random wrong
b01 = high-relevance-first wrong and seeded-random correct
```

The reported p value is the two-sided exact binomial probability under
`Binomial(b10 + b01, 0.5)`, equivalent to exact paired McNemar inference. It is
one when there are no discordant pairs. The three tests are descriptive and
unadjusted, matching the paper. Low-relevance-first accuracies are reported as
the third curve; any additional high-versus-low test is diagnostic and cannot
replace the paper's high-versus-random comparison.

The builder publishes `restoration_order_table.json`,
`table13_restoration_order.csv`, `table13_restoration_order.md`,
`table13_restoration_order.tex`, and `run.json`. Integer numerators,
denominators, discordant counts, and all per-setting rows remain available so
rounded percentages cannot conceal an identity or pooling error.

## Provenance, restart, and atomic publication

The setting manifest records the final-PDF fingerprint, exact source paths and
SHA-256 hashes, source producer identity, the model revision and dataset-content
identity, prompt and extraction protocols, selected IDs, cohort and
restoration-plan hashes, the legacy random algorithm ID and seed, explicit
decoding arguments, base/effective generation-config hashes, package versions,
runtime/CUDA/GPU details, termination/retry events, and hashes of the
generation, reconstruction, scoring, and integrity code. A changed source
file, source manifest, source outcome, edit plan, executable source, or runtime
identity fails closed instead of being silently combined with old generations.

A new run requires that the final output directory does not yet exist. An
inter-process lock serializes invocations for that output identity. Running and
failed manifests plus checkpoints live in a hidden sibling directory with a
random run-ID owner record; they are never placed in the final output. Cleanup
and resume first verify that owner record against the run manifest and output
identity. Each complete condition batch is written as an atomic private
checkpoint. `--resume` requires identical
arguments, protocol, source, cohort, plan, generation/scoring code, runtime
identity, and checkpoint hashes; it skips only a fully validated batch. A
partial batch is regenerated as a unit, so records from two executions are not
silently mixed. A completed resume validates the public outputs and returns
without loading model weights.

The three setting outputs, including the completed manifest that commits their
hashes, are assembled in a second private sibling directory only after every
retained identity has all eleven conditions. That complete directory is
published with one Linux `renameat2(..., RENAME_NOREPLACE)` operation, so even
an empty destination that appears concurrently is never replaced. A commit
failure leaves the resumable work private; checkpoint cleanup after a
successful commit cannot turn the completed public run back into a failed run.
The table builder likewise keeps
aggregation/rendering code identity separate from the GPU-generation identity,
uses its own destination lock, stages all five artifacts beside the destination,
and publishes one completed directory without replacing an existing result.
Cleanup is limited to private paths owned by both the output identity and run
ID.

## Interpretation limit

The experiment shows how much answer accuracy can be recovered when an oracle
reveals both the intended clean substrings and their attribution ranking. It
does not show that attribution alone can locate or repair an unknown typo in a
new prompt. The clean-correct/edit-wrong source selection also conditions on a
specific failure mode, so the curve is not an unconditional benchmark
accuracy curve. Fresh results should therefore be reported with their exact
cohort and endpoint drift, with the paper values retained only as historical
references.

## Prior-work implementation lineage

The broader idea of ranking input tokens by importance and applying a
character-level typo follows Tsuji et al., *Investigating Neurons and Heads in
Transformer-based LLMs for Typographical Errors* (EMNLP 2025). We inspected
the authors' public
[`typo_neurons_and_heads`](https://github.com/4ldk/typo_neurons_and_heads)
repository at `develop` commit
[`182a77cd3b4cf0cf38733653e4e3feca98f2fc43`](https://github.com/4ldk/typo_neurons_and_heads/commit/182a77cd3b4cf0cf38733653e4e3feca98f2fc43).
That Apache-2.0 repository contains no partial-restoration or Table 13
restoration-order implementation. Its relevance target, typo operator, task,
evaluation, and process-global randomization also differ from this paper's
final protocol. This package therefore treats it as conceptual lineage only:
no code or data from that upstream repository is copied. The public code here
is independently written with the final PDF as the experiment authority; the
explicitly labelled `difflib`, MD5-key, and batch-order compatibility details
come from the recovered submitted experiment code where the PDF is silent.
