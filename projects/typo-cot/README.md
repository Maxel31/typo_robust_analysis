# typo-cot reproduction package

This package contains the public reproduction interface for **“Edited-Word
Activation Patching Reverses Selected Typo-Induced Answer Changes after
Tokenization.”**

The final paper is the primary experimental specification. Its canonical
SHA-256 is available from the catalog's single source of truth:

```bash
uv run --project projects/typo-cot typo-cot experiments source
```

See
[`docs/paper-experiments.md`](docs/paper-experiments.md) for the transcribed
operation matrix, denominators, target directory layout, and one-command-per-
experiment interface.

## Setup

From the repository root:

```bash
uv sync --project projects/typo-cot
```

Pair preparation and activation-patching commands additionally need the
paper-locked GPU/LRP dependencies:

```bash
uv sync --project projects/typo-cot --extra lrp
```

## Available commands

The experiment catalog is implemented and does not require a GPU:

```bash
uv run --project projects/typo-cot typo-cot experiments list
uv run --project projects/typo-cot typo-cot experiments list --format json
uv run --project projects/typo-cot typo-cot experiments show cot-swap
uv run --project projects/typo-cot typo-cot experiments show clean-prefix-scan --format json
```

Each catalog entry includes its stable `target_command`, paper section, required
operation-specific arguments, cohort, intervention, readout, outputs, compute
class, and implementation status. Direct experiment runners are added in
separate reviewed PRs; only entries marked `implemented` are runnable.

## Prepare clean/edited pairs

`prepare-edited-pairs` performs the paper's input-preparation operation: greedy
clean generation, first-CoT-token AttnLRP targeting, up to four seeded
single-character edits, greedy edited generation, deterministic answer
extraction, and edited-word-final token alignment. Run it separately for each
model, benchmark, and targeting condition:

```bash
uv run --project projects/typo-cot --extra lrp typo-cot prepare-edited-pairs \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --targeting attribution-4 \
  --num-edits 4 \
  --output-dir results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4
```

Use `--targeting random-4` for the within-item control sampled after excluding
the four Attribution-4 tokens. `--num-edits` selects how many of the sampled
targets to edit (1–4). The paper defaults are `--seed 42` and
`--max-new-tokens 512`; `--limit 1` is available for a GPU smoke run. A stopped
run can be continued with `--resume`.
Both generations explicitly freeze `do_sample=false`, one beam, one returned
sequence, no temperature/top-p/top-k sampling, cached decoding, and the requested
token cap; model defaults cannot silently change the cohort. The primary answer
extractor runs first, the deterministic fallback runs only for an empty result,
and GSM8K correctness uses exact canonical strings rather than floating-point
tolerance. MATH-500 retains its task-native symbolic comparator.
The command writes:

- `pairs.jsonl`: versioned per-item clean/edited generations, target-attempt
  provenance, actual distinct edited-word spans, and clean/edited word-final
  token indices;
- `run.json`: frozen arguments, environment and dataset provenance, progress,
  failures, and record counts.

The target attempts and aligned words are deliberately separate. AttnLRP ranks
tokens, and the final paper reports that some lower-ranked attempts land outside
their intended token or share a word. The pair format preserves that behavior
instead of silently converting targeting into a word-level selector.
Every selected dataset item is retained in `pairs.jsonl`, including an item for
which no eligible character edit exists. Such a record has empty
`target_attempts` and `aligned_words`, identical clean/edited text and generation,
and `answer_changed: false`. It remains in population-level targeting-fidelity
denominators but is ineligible for a later edited-word activation patch.
The complete field contract and coordinate conventions are documented in
[`docs/prepare-edited-pairs.md`](docs/prepare-edited-pairs.md).

Dataset cohort size follows the final paper setting rather than one global
MMLU cap. MMLU uses 100 examples per subject (5,700 items) for
Qwen2.5-7B-Instruct, Gemma-3-12B-IT, and Gemma-3-27B-IT, and 50 per subject
(2,850 items) for the other paper models. MMLU-Pro uses 100 per subject for
every model. The selected cohort size and versioned selection rule are recorded
in `run.json` provenance.

## Audit targeting fidelity

After preparing every four-edit model/benchmark/targeting cell, aggregate the
paper's Appendix A input-quality checks in one CPU-only operation:

```bash
uv run --project projects/typo-cot typo-cot targeting-fidelity-audit \
  --pairs-root results/prepare-edited-pairs \
  --output-dir results/targeting-fidelity-audit
```

The input root may contain any directory layout; the audit discovers completed
`prepare-edited-pairs/v1` outputs recursively and reads setting identity from
their records and manifests rather than from machine-specific path names. It
rejects partial, duplicate, mixed, or non-four-edit inputs instead of silently
changing the Appendix A denominator. It expects the paper seed 42 by default;
use `--expected-seed` only for a separately labelled sensitivity run. The
audit independently replays cumulative landing offsets and every SHA-seeded
character edit instead of trusting recorded landing/operation flags. The
command writes:

- `targeting_fidelity_records.jsonl`: one validation/audit row per prepared
  item;
- `targeting_fidelity.csv`: per-setting and pooled target-landing, distinct-word,
  zero-attempt/zero-aligned-word counts, and prepared-pair gold-option rates,
  including Attribution ranks 1--4 separately;
- `operation_counts.json`: substitution, duplication, and deletion counts by
  setting and targeting condition;
- `run.json`: input/output hashes, arguments, paper fingerprint, counts, and
  the reported Appendix A reference values used for comparison.

Rates use items or edit attempts exactly as named by each column. In
particular, the four-distinct-word rate is item-level, target fidelity is
attempt-level, and the prepared-pair gold-option rate is restricted to
multiple-choice inputs. Items with no successful target attempt remain in the
item denominator and are not vacuously classified as all-attempts-faithful;
attempted edits that cancel to the clean text are reported separately. The
paper's 21.5% gold-option value uses the later
Attribution-4 CoT-swap included cohort, so this pair-only command records that
reference as not directly computable rather than comparing unlike denominators.
`run.json` permits a `descriptive_only` paper comparison only after checking
the exact 42-setting grid, archival per-cell counts, paired-arm
sample/provenance identity, paper cohort rule and subset caps, pinned model
revisions, seed 42, and the 512-token generation cap;
otherwise its status is `not_comparable`.
See
[`docs/targeting-fidelity-audit.md`](docs/targeting-fidelity-audit.md) for the
schemas and paper comparison rules.

## Scan layerwise first-CoT-token KL patches

`layerwise-kl-patching` implements the paper's distributional RQ1 scan for one
model, benchmark, and targeting condition. It selects the stored freely
generated cases whose clean answer is correct and edited answer is wrong, then
copies one complete decoder block's residual output at every aligned edited-word
final token. Run every layer in both directions with physical GPU 0 as follows:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot layerwise-kl-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --targeting attribution-4 \
  --directions clean-to-edited edited-to-clean \
  --gpu-id 0 \
  --output-dir results/layerwise-kl-patching/gemma-3-4b-it/gsm8k/attribution-4
```

The readout is the prompt-final next-token distribution that predicts the first
CoT token. It is not pair preparation's adjacent AttnLRP target after observing
that token. For `clean-to-edited`, the normalized score is
`1 - KL(clean || patched-edited) / KL(clean || edited)`; the reciprocal
direction swaps clean and edited everywhere. Untreated denominators must be
finite and greater than `1e-9`. A pair and direction enter a summary only when
all decoder layers have finite patched KL values; negative normalized values
remain negative.

The command writes:

- `layer_records.jsonl`: included long-form records ordered by sample,
  direction, and layer;
- `pair_status_records.jsonl`: every selected pair/direction and its inclusion
  or explicit exclusion reason;
- `setting_summary.json`: across-pair layer medians, peak and Hsu MCB layer set,
  plus paper-defined early/middle/late summaries;
- `run.json`: arguments, paper/input/output fingerprints, protocol and runtime
  provenance, progress, failures, and comparability status.

Use `--limit 1` only for a labelled GPU smoke run. It is always marked partial
and is not comparable with the paper. An interrupted identical run can continue
with `--resume`; public outputs are finalized only after all selected pairs have
complete checkpoints. The command validates a completed four-edit, seed-42
`prepare-edited-pairs/v1` input before loading model weights. The source must
be an unlimited run with the paper's 512-token generation cap; its recorded
Hugging Face commit pins both the model and tokenizer used for patching.

This command produces one setting summary. The paper's `.639/.410/.111`
headline additionally macro-averages 30 setting summaries with equal setting
weight after excluding the two flagged small MATH cells; that cross-setting
artifact step is intentionally separate. Fresh public pair generation follows
the final PDF but fixes documented historical alignment and extraction defects,
so exact Table 5 row counts require the corresponding frozen source IDs rather
than an undocumented recreation of old teacher-forcing gates. See
[`docs/layerwise-kl-patching.md`](docs/layerwise-kl-patching.md) for the complete
contract.

## Scan layerwise free-answer patches

`layerwise-answer-patching` implements the separate free-generation scan shown
on the right of Figure 2. It pools the two target-selection arms without losing
their provenance: up to 150 seed-42 shuffled clean-correct/edited-wrong anchor
pairs are selected from each completed pair-preparation run, for at most 300
anchors in one model/benchmark setting. The command then greedily regenerates
both untreated answers in the patching process and freezes the pairs that are
still clean-correct and edited-wrong as one denominator for every layer and
both directions.

Run one of the paper's eight settings on physical GPU 0 as follows:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot layerwise-answer-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --attribution-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --random-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/random-4/pairs.jsonl \
  --directions clean-to-edited edited-to-clean \
  --max-pairs 300 \
  --gpu-id 0 \
  --output-dir results/layerwise-answer-patching/gemma-3-4b-it/gsm8k
```

Each layer starts an independent greedy continuation of at most 512 tokens.
Only the initial prompt prefill is patched; later decode steps consume the
patched key/value cache. `clean-to-edited` succeeds when the patched edited
answer returns to the regenerated clean answer. `edited-to-clean` succeeds
when the patched clean answer changes from that answer. The task-specific
primary extractor is always tried first, and the paper's deterministic
fallback is invoked only for an empty primary result. An answer that remains
unextractable is a failed readout in either direction, so it contributes zero
without being removed from the fixed denominator.

The command writes:

- `answer_layer_records.jsonl`: per-pair, per-direction, per-layer generated
  answers and binary events;
- `pair_status_records.jsonl`: every selected anchor and its fixed-cohort
  inclusion or exclusion reason;
- `setting_summary.json`: fixed denominator, layer profiles, peaks, Wilson
  intervals, and paired binary MCB sets;
- `run.json`: arguments, source/model/output fingerprints, protocol, progress,
  failures, and comparability status.

`--max-pairs 300` is the paper run. Smaller values are labelled partial smoke
runs. An interrupted identical run can continue with `--resume`; a completed
resume validates hashes and returns without loading model weights. Fresh public
pair preparation follows the final-PDF protocol but does not claim to recreate
the historical Figure 2 sample IDs. See
[`docs/layerwise-answer-patching.md`](docs/layerwise-answer-patching.md) for the
schemas and the eight-setting acceptance values.

`run.json` labels a run `fresh-paper-protocol-reproduction` only when it uses
one of the four Figure 2 models on GSM8K or MMLU, requests both directions and
300 anchors, and both targeting arms contribute to the selected and regenerated
fixed cohorts. Every unmet condition is listed under `comparability.limitations`.

## Patch fixed layer windows and regenerate answers

`fixed-window-answer-patching` implements the answer intervention reported in
Table 6. It pools up to 150 stored clean-correct/edited-wrong anchors from each
targeting arm, greedily regenerates both untreated answers, then applies all six
complete decoder-block output patches in `[0,6)` during one prompt prefill. Run
one of the six planned Gemma/Llama/Mistral by GSM8K/MMLU settings on physical
GPU 0 as follows:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot fixed-window-answer-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/random-4/pairs.jsonl \
  --layers 0:6 \
  --directions clean-to-edited edited-to-clean \
  --gpu-id 0 \
  --output-dir results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k
```

The two directions intentionally use different fixed denominators. Restoration
(`clean-to-edited`) includes regenerated clean-correct/edited-wrong pairs and
succeeds when the patched edited answer equals the regenerated clean answer.
Reciprocal induction (`edited-to-clean`) includes every regenerated
clean-correct anchor, even when its regenerated edited answer is also correct,
and succeeds only when an extracted patched answer differs from the regenerated
clean answer. An unextractable patched answer is a failed readout in either
direction and contributes zero without leaving that direction's denominator.

The prespecified Table 7 MMLU-Pro depth comparison is the same operation with
two independent six-layer windows on the same anchors:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot fixed-window-answer-patching \
  --model Qwen/Qwen2.5-3B-Instruct \
  --benchmark mmlu-pro \
  --pairs results/prepare-edited-pairs/Qwen2.5-3B-Instruct/mmlu-pro/attribution-4/pairs.jsonl \
  --layers 0:6 6:12 \
  --directions clean-to-edited \
  --gpu-id 0 \
  --output-dir results/fixed-window-answer-patching/Qwen2.5-3B-Instruct/mmlu-pro
```

`--pairs` accepts one or both completed targeting arms. Two arms are capped at
150 selected anchors each; one arm is capped at 300. `--layers` accepts one or
more non-overlapping half-open decoder-block windows. `--limit 1` is available
only for a labelled smoke run. An interrupted identical run can continue with
`--resume`; a completed resume validates all hashes before model loading.

The command writes:

- `fixed_window_records.jsonl`: per-pair, per-direction, per-window generations
  and binary events;
- `pair_status_records.jsonl`: every selected anchor and its direction-specific
  denominator inclusion or exclusion reason;
- `setting_summary.json`: denominator and event counts by direction/window,
  Wilson intervals, targeting-arm breakdowns, and the 10,000-resample paired
  percentile comparison for the two MMLU-Pro windows;
- `run.json`: arguments, paper/input/model/output fingerprints, protocol,
  checkpoints, progress, failures, and comparability status.

A structurally complete fresh Table 6 run is labelled
`fresh-paper-protocol-run`; the analogous Table 7 execution is labelled
`fresh-prespecified-mmlu-pro-window-run`. These labels mean that the published
protocol shape was executed on a fresh public cohort, not that unpublished
historical sample IDs or published denominators were recreated.

The final PDF is authoritative where the historical artifacts disagree with its
stated protocol. In particular, the paper says an unextractable answer is a
failed intervention readout, while the historical Table 6 induction aggregate
counted some unextractable patched answers as changes. Fresh runs therefore
report the paper-defined extracted-answer event and retain the published Table 6
values only as historical reference metadata, not as forced acceptance targets.
The offset and cross-item donor comparisons are a separate
`patch-coordinate-controls` operation. See
[`docs/fixed-window-answer-patching.md`](docs/fixed-window-answer-patching.md)
for the complete selection, schema, comparison, and resume contract.

## Compare patch coordinates and donor content

`patch-coordinate-controls` reproduces the primary-coordinate rows of Table 7
for Gemma-3-4B on GSM8K. It takes a completed
`fixed-window-answer-patching` run, keeps its exact clean-to-edited denominator
and correct-coordinate endpoint, and evaluates a same-item +2-token coordinate
control and a matched cross-item donor on those same pairs. The identity
self-copy arm is an execution invariant: copying each edited state back onto
itself must leave the greedy generation token-identical to the untreated
edited baseline.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-coordinate-controls \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --layers 0:6 \
  --controls correct offset-2 cross-item self-copy \
  --gpu-id 0 \
  --output-dir results/patch-coordinate-controls/gemma-3-4b-it/gsm8k
```

The offset arm moves both the clean donor coordinate and edited write
coordinate two tokens past each aligned edited-word endpoint, retaining only
coordinate pairs that remain valid and are not edited-word endpoints. The
cross-item arm keeps the recipient coordinates fixed but uses a different
clean donor with the same targeting arm and number of aligned words. Donors
are assigned deterministically by a cyclic shift of sorted sample IDs within
each matching stratum, so no random seed silently changes the comparison.

The referenced run must be complete, include `clean-to-edited` and `[0,6)`,
and pass its recorded input and output SHA-256 checks before model weights are
loaded. All requested arms use residual block outputs over `[0,6)`, greedy
decoding, and one paired denominator. Restoration means that the extracted
patched answer equals the regenerated clean answer; an unextractable answer is
a failure and remains in the denominator. The summary reports arm rates and
the paper's two-sided exact McNemar comparisons of `correct` against
`offset-2` and `cross-item`. The published 129/172, 44/172, and 42/172 values
remain historical reference metadata: a run from newly prepared public pairs
is labelled as a fresh protocol reproduction and does not claim the
unpublished historical sample IDs.

The command writes `coordinate_control_records.jsonl`,
`pair_status_records.jsonl`, `coordinate_control_summary.json`, and `run.json`.
Use `--resume` to validate and continue an interrupted identical run. A smaller
`--limit` is labelled as a smoke run and limits recipients only after the full
reference cohort has fixed the donor map. See
[`docs/patch-coordinate-controls.md`](docs/patch-coordinate-controls.md) for
the complete reference, schema, statistic, and restart contract.

## Tests

```bash
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_paper_experiment_catalog.py
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_targeting_fidelity_audit.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_layerwise_kl_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_layerwise_answer_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_fixed_window_answer_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_patch_coordinate_controls.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests
```

The contract tests enforce the final-PDF fingerprint, complete operation list,
descriptive names, unique command slugs, CLI JSON schema, and documentation
coverage. The full suite also exercises pair preparation and its model, dataset,
prompt, answer-extraction, and AttnLRP adapters without downloading model
weights.

## Repository scope

Only the public experiment catalog, implemented runners, their runtime
dependencies, and tests are tracked here. Intermediate Exp1–20 scripts,
machine-specific configuration, and archived outputs are deliberately excluded
because they are not the paper's reproduction interface.
