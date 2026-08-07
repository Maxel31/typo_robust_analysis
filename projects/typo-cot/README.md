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
The final paper explicitly labels both the offset and cross-item comparisons
as post-hoc controls; public protocol and summary metadata record
`design_status: post-hoc` for both arms.

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

## Compare patch write positions

`patch-position-controls` reproduces the Appendix B reachability scan for the
Gemma-3-4B/GSM8K Attribution-4 layerwise-KL cohort. It consumes a completed
`layerwise-kl-patching` run rather than a separately sampled pair file: the
referenced run fixes the exact clean-to-edited included IDs, untreated KL
denominators, and edited-word layer profile before either alternative position
is evaluated.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-position-controls \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --layerwise-kl-run \
    results/layerwise-kl-patching/gemma-3-4b-it/gsm8k/attribution-4 \
  --positions edited-word prompt-final question-final \
  --gpu-id 0 \
  --output-dir results/patch-position-controls/gemma-3-4b-it/gsm8k
```

The canonical direction is clean-to-edited only. For `edited-word`, the
command carries forward the referenced run's clean edited-word-final states,
aligned edited destinations, and complete layer profile without silently
forming a new denominator. `prompt-final` copies the clean prompt's last token
state to the edited prompt's last token. `question-final` copies between the
last tokens overlapping each run's recorded editable GSM8K question span. The
latter locator is an implementation detail recovered from the legacy runner;
the final PDF names the position but does not specify its token locator.

Every requested alternative arm must finish every decoder layer for every
referenced included ID. Scores use the same source denominator:
`1 - KL(clean || patched-edited) / KL(clean || edited)`. The summary reports
the common cohort fingerprint, layerwise across-pair medians, and every layer
tied at the maximum. It does not invent a three-arm difference test, CI, or
alternative-position MCB result absent from the final paper; Table 5's
edited-word MCB remains source-profile metadata. The analysis is
recorded as `post_hoc_exploratory_descriptive` and maps intervention
reachability, not information movement or an encoding mechanism.

The command writes `position_control_records.jsonl`,
`pair_status_records.jsonl`, `position_control_summary.json`, and `run.json`.
A fresh public upstream run is labelled as a fresh protocol reproduction; it
does not claim the undocumented historical IDs merely because the paper's
legacy row had 109 pairs. `--limit 1` produces a labelled smoke run, and
`--resume` validates the original input, plan, runtime, checkpoints, and final
output hashes before reuse. See
[`docs/patch-position-controls.md`](docs/patch-position-controls.md) for the
published acceptance values, schemas, last-layer behavior, and restart
contract.

## Cross the fixed patch with complete clean text

`patch-text-combination` reproduces Table 2's descriptive two-by-two on one
paired denominator. It consumes the completed Gemma-3-4B/GSM8K
`fixed-window-answer-patching` run that already fixes the clean-correct,
edited-wrong cohort, its untreated generations, and the clean-to-edited
`[0,6)` patched generations. One invocation always emits all four conditions;
there is no option to run cells on different pair sets.

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

The no-text cells are carried forward from the hash-verified reference:
untreated edited generation for patch-absent and its `[0,6)` clean-to-edited
generation for patch-present. The command then supplies the complete clean
pre-answer text under the edited question and generates the two complete-text
cells without and with the same patch. Before those cells are generated, every
selected clean and edited baseline is replayed and must match the reference
token-for-token; complete-text generation starts only after the whole replay
gate passes. Correctness is always against the GSM8K gold answer; an
unextractable answer is incorrect and remains in the shared denominator.

The final PDF specifies complete pre-answer text but not its exact character
locator. To reproduce the submitted implementation, this command takes the
prepared clean continuation and cuts immediately before the first literal
`The answer is` or `the answer is`; if neither occurs, it supplies the entire
continuation. It records the boundary diagnostics and requires tokenizing the
edited prompt plus supplied text to preserve the exact edited prompt-token
prefix. These are labelled legacy-backed implementation details, not additional
claims from the paper.

The historical Table 2 values—0/172, 129/172, 168/172, and 171/172—are stored
only as published reference metadata. A fresh run reports its actual source
denominator and does not claim the unpublished historical IDs; a denominator
other than 172 is explicitly labelled as only a partial protocol run. The
summary is descriptive: it reports four gold-correct counts and rates, without an
interaction, mediation, necessity, sufficiency, or ranking estimate. Complete
text includes near-answer content.

The command writes `patch_text_records.jsonl`, `pair_status_records.jsonl`,
`patch_text_summary.json`, and `run.json`. `--limit 1` produces a labelled GPU
smoke run. `--resume` verifies the source, deterministic plan, runtime,
checkpoints, and completed output hashes before reuse. See
[`docs/patch-text-combination.md`](docs/patch-text-combination.md) for the full
cell, boundary, provenance, and restart contracts.

## Cross clean and edited questions with complete pre-answer text

`cot-swap` implements the final paper's descriptive RQ2 four-cell crossing for
one model, benchmark, and targeting arm. It consumes a completed
`prepare-edited-pairs` output so the same freely generated clean and edited
continuations supply both sides of the crossing:

```text
A = clean question  + clean pre-answer text
B = edited question + edited pre-answer text
C = edited question + clean pre-answer text
D = clean question  + edited pre-answer text
```

Each supplied pre-answer sequence is fixed context. Only the answer span is
regenerated, greedily, for at most 16 tokens. Run each model/benchmark/targeting
cell separately on physical GPU 0:

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

Use the same command with `--targeting random-4` and its corresponding pair
file for the control arm. `--limit 1` is available only for a labelled smoke
run. `--resume` validates the original source, deterministic four-cell plan,
runtime, checkpoints, and finalized output hashes before reusing them.

The denominator pipeline first requires applied-edit validity, then both
answer-template boundaries, successful A--D execution, and finally a correct
regenerated A answer. Zero-edit records are reported separately because their
typo-induced restoration is undefined; they are not folded into the historical
14.0% template exclusion. B, C, and D are compared with the extracted A answer,
not directly with gold. Restoration further conditions on `B != A` and succeeds
when `C == A`. An unextractable B, C, or D answer is a failed equality and
remains in the relevant denominator.

The task-specific primary `.extract()` runs first and every non-empty result is
preserved. The fallback runs only for an empty primary, symmetrically in A, B,
C, and D. Its exact regexes and the cap-aware N4/N5 gate are legacy-backed
details not specified by the final PDF. EOS versus 16-token-cap termination and
extraction method are recorded per cell. The versioned runtime uses a one-pair,
four-cell batch and decodes generated token IDs only, preventing `--resume`
from mixing a different batch/decode policy.

The final PDF reports the historical Attribution-4 pool as 19,550 A-correct
cases, 4,634 B changes, and 3,539 C restorations (76.4%). Those values are
reference metadata, not forced acceptance targets. The archived aggregation
used for the printed values did not rescue empty A answers, whereas Appendix A
states that the fallback applies to every condition. This public command follows
the final-PDF rule and labels fresh or partial provenance explicitly; it does not
silently reproduce the asymmetric historical calculation.

The command writes `cot_swap_records.jsonl`, `pair_status_records.jsonl`,
`cot_swap_summary.json`, and `run.json`. This PR deliberately emits one setting
summary; the five-model task pools for Table 1 are a separate CPU artifact-
building operation so raw GPU settings cannot be silently mixed. Complete text
contains answer-near content, so these crossed conditions are an upper-bound
diagnostic rather than direct, indirect, total, mediation, or deployable repair
effects. See [`docs/cot-swap.md`](docs/cot-swap.md) for the boundary rule,
schemas, published references, and comparability contract.

This command implements the five 1B--7B headline models. The separate
Appendix C/Table 9 12B--72B ladder remains the catalogued
`model-scale-cot-swap` operation; it is not represented as completed by this
single-visible-GPU runner. The documented command and repository validation use
physical GPU 0; `--gpu-id` can explicitly select another single visible device.

## Delete the final non-empty line from clean pre-answer text

`answer-line-deletion` implements the final paper's RQ2 content/format control
for Gemma-3-4B, Llama-3.2-3B, and Mistral-7B on GSM8K and MMLU. It consumes one
completed, unlimited Random-4 `cot-swap` run, selects its regenerated-A-correct
`B != A` cases by sample ID, and applies the submitted control cap of 150 cases
per model/task setting:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot answer-line-deletion \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cot-swap-run results/cot-swap/gemma-3-4b-it/gsm8k/random-4 \
  --max-pairs 150 \
  --gpu-id 0 \
  --output-dir results/answer-line-deletion/gemma-3-4b-it/gsm8k
```

For every selected pair, the command reruns two fixed-context conditions in one
paired batch: the edited question with complete clean pre-answer text, and the
same input after deleting the clean text's final non-empty line. The deletion
leaves one newline when earlier text remains, so generation resumes at the
truncation boundary. If there is only one non-empty line, the supplied prefix
becomes empty; outputs report and stratify this diagnostic. The model generates
at most 16 continuation tokens from each fixed boundary, and an answer is
extracted from only those new token IDs using the final PDF's greedy bfloat16
left-padded protocol. Restoration in each arm is
equality with the source run's clean A answer; unextractable answers are retained
as failures.

`--limit 1` is a labelled smoke run and is applied only after validating and
constructing the full deterministic capped cohort. `--resume` verifies the
upstream run, prepared pairs, plan, runtime, pair-atomic checkpoints, and public
output hashes before reuse. The command writes
`answer_line_deletion_records.jsonl`, `pair_status_records.jsonl`,
`answer_line_deletion_summary.json`, and `run.json`.

Table 1 reports pooled baseline-to-deletion restoration of 95.2% to 48.9%
(`n=333`) on GSM8K and 82.2% to 29.1% (`n=450`) on MMLU. The archived producer
of those printed values generated up to 256 answer tokens, conflicting with the
final PDF's global 16-token CoT-swap rule. The public command follows the final
PDF and records this as a historical-comparability limitation; it does not tune
fresh results to the printed percentages. Deleting the last line also truncates
the text mid-flow, so the contrast combines near-answer content removal with
format disruption; in the archived Table 1 cohort it removed the entire prefix
for 179/333 GSM8K and 334/450 MMLU cases. It is therefore not a reasoning-only,
local-line-only, or mediation effect. See
[`docs/answer-line-deletion.md`](docs/answer-line-deletion.md) for the exact
source, selection, deletion, schema, and restart contracts.

## Scan clean pre-answer token prefixes

`clean-prefix-scan` implements the final paper's RQ3 text intervention. Under
the edited question, it supplies the first `k` tokenizer IDs of the clean
pre-answer text and independently regenerates an answer at every requested
absolute or relative budget. It has two explicit cohort modes because the PDF
uses the fixed-window denominator for its primary Gemma-3-4B/GSM8K cell but a
separate deterministic 150-target sample in each of fourteen extensions.

Run the primary cell from its completed, unlimited fixed-window result:

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

Run one extension from its two completed, unlimited pair preparations:

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

The relative rule `k = round(r * L_C)`, gold-answer correctness, the common
fresh-`k=0`-wrong denominator, stable-through-later correctness, and the
two-transition non-monotonicity definition come from the final PDF. The dense
budget values shown above, Python ties-to-even rounding, per-arm 400 cap,
proportional/systematic extension selection, batch size one, and exact
pre-answer locator are submitted-producer details not printed by the PDF. They
are versioned as `legacy-backed`, never silently promoted to paper-defined
requirements.

The operation tokenizes the complete clean text after both the clean and edited
prompts. Matching clean/edited suffix IDs reproduce the submitted selection
check; a second scan-time check requires the complete edited input to preserve
the separately tokenized edited prompt. A selected target that fails that
second check remains an invalid status and is not replaced. Valid points pass
`edited_prompt_ids + clean_cot_ids[:k]` directly to the model, without decoding
and retokenizing the prefix. Only newly generated IDs are decoded for answer
extraction. Unextractable generations are wrong outcomes and stay in the
denominator. `--limit 1` is a labelled GPU smoke run; `--resume` continues
hash-bound point checkpoints. Absolute point correctness reports the smaller
`L_C >= k` applicability count, while absolute stable recovery (`k* <= k`)
keeps the common fresh-`k=0`-wrong denominator, including shorter clean CoTs.
After successful publication the private checkpoint directory is removed and
the completed manifest clears its checkpoint registry; completed `--resume`
reconstructs records, statuses, and the summary without loading model weights.

Each setting writes `prefix_scan_records.jsonl`,
`pair_status_records.jsonl`, `prefix_scan_summary.json`, and `run.json`.
Figure 3's 14-setting cluster bootstrap is a later CPU artifact-building step;
the primary cell is not pooled into that extension aggregate. See
[`docs/clean-prefix-scan.md`](docs/clean-prefix-scan.md) for the full source,
selection, metric, schema, and restart contract.

## Replace one clean pre-answer token and regenerate

`one-token-prefix-replacement` implements the supplementary diagnostic in
Appendix D and Tables 10--11. Under the clean question, it supplies the clean
pre-answer token IDs before a selected position, forces either the observed
clean token or a typo-context top-1 token at that position, and freely
regenerates the answer. This measures answer sensitivity under the clean
question. It is not typo repair, a prefix-cut rule, or an RQ3 answer.

Run the primary Gemma-3-4B/GSM8K cell from the same fixed 172-pair source used
by the primary clean-prefix scan:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cohort primary \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --position-controls distant \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-4b-it/gsm8k
```

Run one of the fourteen extensions from the same deterministic 150-target
selection used by `clean-prefix-scan`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-1b-it \
  --benchmark mmlu \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/mmlu/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/mmlu/random-4/pairs.jsonl \
  --max-pairs 150 \
  --position-controls distant \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-1b-it/mmlu
```

The prespecified adjacent-position check applies only to Gemma-3-1B/GSM8K,
Llama-3.2-3B/ARC, and Mistral-7B/MMLU. Add it to the corresponding extension
run; the distant arms are retained so both paper tables come from one
hash-bound record set:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-1b-it \
  --benchmark gsm8k \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/random-4/pairs.jsonl \
  --max-pairs 150 \
  --position-controls distant adjacent \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-1b-it/gsm8k
```

Candidate position `P` maximizes clean-versus-edited next-token KL where the
clean token is not top-1 under the edited-question context. The distant
control `C` is a lower-median-KL candidate at least three tokens from `P`.
With `distant`, the runner generates both local substitutions needed by Table
10 and all four crossings of the `P`- and `C`-derived token identities needed
by Table 11. `adjacent` additionally applies the `P`-derived token at the
nearest lower-KL position. The exact median-low and adjacent tie rules are
submitted-producer details and are labelled `legacy-backed`; neither position
rule uses generated-answer outcomes.

Every arm uses direct token IDs, greedy bfloat16 generation, batch size one,
left padding, and a 512-new-token cap. Correctness is against the canonical
gold answer; an unextractable continuation is incorrect. Table 10, distant
factorial, and adjacent metrics keep their distinct paired-eligibility rules.
The summary reports the final-PDF-literal four-nonnoop factorial as
`distant_factorial` and the stricter submitted-producer distinct/admissible
variant as `distant_factorial_submitted_producer`, including reason-wise
attrition. On the frozen historical extensions these contain 1,603 and 1,575
pairs respectively; the 28-record difference is entirely `b_P == b_C`.
The primary cell is reported separately and must not enter the fourteen-
extension aggregate. `--limit 1` is a labelled GPU smoke run and `--resume`
continues hash-bound per-arm checkpoints.

`run.json` and `one_token_summary.json` label a limit-truncated execution
`partial-smoke-run`. An unlimited setting is labelled
`fresh-paper-protocol-run` only when it retains the paper-sized 172-target
primary or 150-target extension plan and includes the prespecified adjacent
control for the three Appendix-D settings, with every selected target passing
the exact-boundary audit; otherwise every mismatch is listed under
`comparability.limitations`. Fresh preparation follows the paper source
protocol but is not labelled as proof of byte-identical historical membership.

Each setting writes `one_token_records.jsonl`, `pair_status_records.jsonl`,
`one_token_summary.json`, and `run.json`. See
[`docs/one-token-prefix-replacement.md`](docs/one-token-prefix-replacement.md)
for the exact source, token-position, arm, denominator, historical-reference,
and restart contracts. The fourteen-setting clustered intervals and table
assembly consume these integer events in the later CPU artifact-building step;
the single-setting GPU command does not silently pool the primary cell.

## Tests

```bash
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_paper_experiment_catalog.py
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_targeting_fidelity_audit.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_layerwise_kl_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_layerwise_answer_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_fixed_window_answer_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_patch_coordinate_controls.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_patch_position_controls.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_patch_text_combination.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_cot_swap.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_answer_line_deletion.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_clean_prefix_scan.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_one_token_prefix_replacement.py
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
