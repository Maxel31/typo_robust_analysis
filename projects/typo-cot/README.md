# typo-cot reproduction package

[English](README.md) | [日本語](README.ja.md)

This package contains the public reproduction interface for **“Edited-Word
Activation Patching Reverses Selected Typo-Induced Answer Changes after
Tokenization.”**

Use the setup and operation-specific commands in this guide to reproduce the
paper. The final paper defines the experimental protocol and reported results;
the catalog prints its SHA-256 for optional local-PDF integrity checks:

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

Some paper models require gated Hugging Face access. Before running GPU
experiments, sign in on the
[`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) and
[`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
model pages, review and accept their access conditions, and authenticate the
approved account locally (or provide its read token through `HF_TOKEN`):

```bash
uv run --project projects/typo-cot --extra lrp hf auth login
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

## Implemented ARR manifest and coordinate controls

`build-rebuttal-manifest` is implemented and CPU-only. It accepts the twelve
completed `prepare-edited-pairs` sources and six completed fixed-window runs
that reproduce the paper's six-setting reference. It fails closed unless their
schemas, hashes, model revisions, selected-anchor audit, uncapped harm cohort,
explicit alignment-ineligible coverage, and the paper's
1,241-pair/800-restoration totals agree. It does not run a model or inspect any
new intervention result.

```bash
PAIR_ROOT=projects/typo-cot/results/prepare-edited-pairs
FIXED_ROOT=projects/typo-cot/results/fixed-window-answer-patching
REBUTTAL_ROOT=projects/typo-cot/results/rebuttal

uv run --project projects/typo-cot typo-cot build-rebuttal-manifest \
  --prepared-pairs-root "${PAIR_ROOT}" \
  --fixed-window-root "${FIXED_ROOT}" \
  --output-dir "${REBUTTAL_ROOT}/manifest"
```

The command writes `pair_manifest.jsonl`, `cohort_ids.json`,
`source_audit.json`, and `run.json`. These generated artifacts remain local and
are inputs to every result-producing ARR command below.

`six-setting-patch-controls` is implemented and GPU-only. It reuses the
hash-validated correct-coordinate outcomes from the six fixed-window runs and
generates the prospective strict offset-2 and deterministic cross-item arms.
The primary analysis uses only each setting's common-valid pairs, runs 12 exact
McNemar tests with one Holm family, and computes paired and equal-setting nested
bootstrap intervals. Use exactly one physical GPU; `--limit-per-setting` is a
non-confirmatory smoke-test option and `--resume` verifies and reuses completed
pair checkpoints.

The reused correct arm was produced with positional fallback enabled after an
empty primary extraction, including at the generation length cap. The new
offset and cross-item arms deliberately use that same rule for symmetric paired
scoring. This corrects the initial runner contract before any confirmatory
six-setting result generation; it affects only length-capped continuations with
an empty primary extraction, and termination remains recorded.

```bash
GPU_ID=0
FIXED_ROOT=projects/typo-cot/results/fixed-window-answer-patching
REBUTTAL_ROOT=projects/typo-cot/results/rebuttal

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project projects/typo-cot --extra lrp \
  typo-cot six-setting-patch-controls \
  --config projects/typo-cot/configs/rebuttal/six-setting-patch-controls.yaml \
  --manifest "${REBUTTAL_ROOT}/manifest/pair_manifest.jsonl" \
  --fixed-window-root "${FIXED_ROOT}" \
  --gpu-id "${GPU_ID}" \
  --output-dir "${REBUTTAL_ROOT}/six-setting-patch-controls"
```

The command writes `control_records.jsonl`, `pair_status_records.jsonl`,
`six_setting_control_table.csv`, `common_denominator_flow.csv`,
`multiplicity_table.csv`, `macro_average.json`,
`risk_difference_forest.svg`, and a hash-bound `run.json`.

`source-write-coordinate-grid` is implemented and GPU-only. It separates donor
content from write location over the primary Gemma/GSM8K and prespecified
replication Mistral/MMLU cohorts. The common-valid denominator requires the
complete correct and strict offset coordinate plans. It reuses fixed `E->E`
events and generates `E->O`, `O->E`, and `O->O`, then reports Cochran's Q and
the two prespecified paired contrasts per cohort with one Holm family.
All four arms use the fixed-window producer's extraction contract, including
its empty-primary positional fallback, so capped continuations are scored
symmetrically; termination remains recorded for every new generation.
`--limit-per-cohort` is available only for non-confirmatory smoke tests.

```bash
GPU_ID=0
FIXED_ROOT=projects/typo-cot/results/fixed-window-answer-patching
REBUTTAL_ROOT=projects/typo-cot/results/rebuttal

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project projects/typo-cot --extra lrp \
  typo-cot source-write-coordinate-grid \
  --config projects/typo-cot/configs/rebuttal/source-write-coordinate-grid.yaml \
  --manifest "${REBUTTAL_ROOT}/manifest/pair_manifest.jsonl" \
  --fixed-window-root "${FIXED_ROOT}" \
  --cohorts primary replication \
  --gpu-id "${GPU_ID}" \
  --output-dir "${REBUTTAL_ROOT}/source-write-coordinate-grid"
```

The command writes `source_write_grid_records.jsonl`,
`pair_status_records.jsonl`, `source_write_grid_table.csv`,
`source_write_contrasts.csv`, and a hash-bound `run.json`.

`multitoken-kl-readout` is implemented and GPU-only. For every restoration
pair in the six-setting manifest, it first tokenizes the stored clean
continuation. A pair with fewer than 16 continuation tokens is recorded as
`clean_continuation_lt_16`; a pair whose prompt token IDs are not an exact
prefix after appending the continuation is recorded as
`clean_prompt_not_exact_token_prefix`. Both checks happen before any model
forward. These pairs are excluded from the readout denominator while remaining
in the 1,241-pair audit trail. The output reports `n_target_available` per
setting and `target_available_pairs` overall.
For every available pair, the command teacher-forces the same first 16 token
IDs after the clean, typo, and patched-typo prompts. The patch copies the
edited-word state over layers `[0,6)`. The primary per-pair score compares the
mean `KL(clean || patched)` with `KL(clean || typo)` over tokens 2--16,
deliberately excluding the first CoT token used by the original targeting
metric. Secondary outputs cover tokens 2--4, tokens 2--8, token-wise raw KL
reduction, and the paired first-token versus tokens 2--16 difference. Near-zero untreated
denominators are excluded according to the frozen analysis plan; negative
restoration values are retained. `--limit-per-setting` is available only for
non-confirmatory smoke tests, and `--resume` verifies input-content-addressed,
hash-bound pair checkpoints before reuse.
Per-token diagnostic labels use the tokenizer's native vocabulary pieces,
rather than independently decoding byte fragments as if each were text.

```bash
GPU_ID=0
REBUTTAL_ROOT=projects/typo-cot/results/rebuttal

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project projects/typo-cot --extra lrp \
  typo-cot multitoken-kl-readout \
  --config projects/typo-cot/configs/rebuttal/multitoken-kl-readout.yaml \
  --manifest "${REBUTTAL_ROOT}/manifest/pair_manifest.jsonl" \
  --teacher-forced-tokens 16 \
  --primary-token-range 2:16 \
  --gpu-id "${GPU_ID}" \
  --output-dir "${REBUTTAL_ROOT}/multitoken-kl-readout"
```

The command writes `multitoken_kl_records.jsonl`, `setting_metrics.csv`,
`token_position_trajectory.csv`, `token_position_trajectory.svg`,
`multitoken_summary.json`, and a hash-bound `run.json`.

## Frozen interfaces for remaining ARR additions

The following four commands are `interface-frozen` and **not yet runnable**.
They were written before implementation so every additional experiment has its
own operation, arguments, inputs, and output directory. The statistical and
cohort contracts are fixed in
[`docs/rebuttal_analysis_plan_v1.md`](docs/rebuttal_analysis_plan_v1.md).
`interface-frozen` is a prose-only pre-implementation label, not a third
experiment-catalog status. Such commands are deliberately absent from the CLI
and `experiments list`; each command's implementation PR registers it directly
as an `implemented` operation after its contract tests pass.

```bash
GPU_ID=0
REBUTTAL_ROOT=projects/typo-cot/results/rebuttal

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-harm-audit \
  --config projects/typo-cot/configs/rebuttal/patch-harm-audit.yaml \
  --manifest "${REBUTTAL_ROOT}/manifest/pair_manifest.jsonl" \
  --cohort clean-correct-typo-correct \
  --gpu-id "${GPU_ID}" \
  --output-dir "${REBUTTAL_ROOT}/patch-harm-audit"

uv run --project projects/typo-cot typo-cot tokenization-severity-analysis \
  --config projects/typo-cot/configs/rebuttal/tokenization-severity-analysis.yaml \
  --manifest "${REBUTTAL_ROOT}/manifest/pair_manifest.jsonl" \
  --controls-run "${REBUTTAL_ROOT}/six-setting-patch-controls" \
  --output-dir "${REBUTTAL_ROOT}/tokenization-severity-analysis"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project projects/typo-cot --extra lrp \
  typo-cot subword-position-patching \
  --config projects/typo-cot/configs/rebuttal/subword-position-patching.yaml \
  --manifest "${REBUTTAL_ROOT}/manifest/pair_manifest.jsonl" \
  --modes first final all \
  --token-count-policy equal-count-primary \
  --gpu-id "${GPU_ID}" \
  --output-dir "${REBUTTAL_ROOT}/subword-position-patching"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project projects/typo-cot --extra lrp \
  typo-cot held-out-window-evaluation \
  --config projects/typo-cot/configs/rebuttal/held-out-window-evaluation.yaml \
  --manifest "${REBUTTAL_ROOT}/manifest/pair_manifest.jsonl" \
  --cohort-ids "${REBUTTAL_ROOT}/manifest/cohort_ids.json" \
  --gpu-id "${GPU_ID}" \
  --output-dir "${REBUTTAL_ROOT}/held-out-window-evaluation"
```

## Frozen interfaces for typo-robustness training

Training uses a separately locked project so adding PEFT does not change the
paper reproduction environment. The data roles, leakage controls, hierarchical
layer/component localization, losses, baselines, and pre-PR empirical gate are
fixed in
[`docs/robustness_training_plan_v1.md`](docs/robustness_training_plan_v1.md).
These `interface-frozen` commands are also **not yet runnable**. In particular,
the training implementation is not opened as a PR until held-out evaluation
shows improved typo robustness with clean performance preserved.

```bash
GPU_ID=0
TRAIN_PROJECT=projects/typo-robust-training
TRAIN_ROOT=projects/typo-robust-training/results

uv sync --project "${TRAIN_PROJECT}" --locked

uv run --project "${TRAIN_PROJECT}" --locked typo-cot build-robustness-training-data \
  --config "${TRAIN_PROJECT}/configs/gemma4b-sanity.yaml" \
  --output-dir "${TRAIN_ROOT}/data/gemma4b-sanity"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot select-distillation-layers \
  --config "${TRAIN_PROJECT}/configs/gemma4b-layer-selection.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --tasks gsm8k mmlu arc \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/localization/layers"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot localize-robustness-components \
  --config "${TRAIN_PROJECT}/configs/gemma4b-component-localization.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --components mlp-neuron attention-head \
  --causal-readouts answer multitoken-kl \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/localization/components"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-noisy-language-model \
  --config "${TRAIN_PROJECT}/configs/baselines/noisy-language-model.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/noisy-language-model/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-output-matching \
  --config "${TRAIN_PROJECT}/configs/baselines/output-matching.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/output-matching/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-global-state-alignment \
  --config "${TRAIN_PROJECT}/configs/baselines/global-state-alignment.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/global-state-alignment/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-localized-state-distillation \
  --config "${TRAIN_PROJECT}/configs/gemma4b-targeted-lora.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --component-selection "${TRAIN_ROOT}/localization/components/component_selection.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/localized-state-distillation/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot evaluate-typo-robustness \
  --config "${TRAIN_PROJECT}/configs/gemma4b-evaluation.yaml" \
  --data-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/evaluation_manifest.json" \
  --base-model google/gemma-3-4b-it \
  --checkpoints "${TRAIN_ROOT}/training" \
  --splits same-task unseen-task unseen-typo \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/evaluation/gemma4b"
```

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
Each clean and edited arm records whether generation ended at an effective EOS
or only at the length cap. An EOS in the 512th generated position is still an
EOS completion, so positional fallback remains available; only 512 generated
tokens with no effective EOS are `length-cap` and disable that fallback. Runs
created before `generation_termination_protocol` was recorded must be regenerated
instead of resumed or reused downstream.
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

## Build the one-token paper tables

After the fifteen setting-level `one-token-prefix-replacement` runs finish,
build the Appendix D tables and the reproducible part of Figure 5 on CPU:

```bash
uv run --project projects/typo-cot \
  typo-cot build-one-token-tables \
  --runs-root results/one-token-prefix-replacement \
  --output-dir results/one-token-tables
```

The command recursively discovers producer `run.json` files below
`--runs-root`, verifies their paper/protocol identities and output checksums,
then recomputes every rate from the integer events in the JSONL records. It
never imports model, tokenizer, `torch`, or `transformers` code. Inputs must be
unlimited `fresh-paper-protocol-run` outputs from the expected primary or
fourteen extension settings; unexpected, duplicate, mixed-identity, or
tampered runs fail closed. Missing expected settings remain visible in the
coverage report, and a paper-labelled pooled estimate is omitted whenever its
required grid is incomplete.

The output directory must not already exist. It is published atomically with:

- `one_token_tables.json`: machine-readable cells, pools, inference metadata,
  historical references, and comparability decisions;
- `table10_one_token.csv`: the per-setting Table 10 token columns and the
  fourteen-extension aggregate when complete;
- `table11_position_controls.csv`: distant and adjacent position controls;
- `one_token_tables.md` and `one_token_tables.tex`: deterministic readable
  table fragments;
- `figure5_validation.json`: validation of the one-token panel fields present
  in producer records, with unavailable token text marked unverifiable; and
- `run.json`: input/output hashes and the frozen analysis protocol.

For the distant Table 11 analysis, the output keeps both the PDF-literal
four-non-noop denominator and the stricter submitted-producer denominator.
Only the latter is compared with the printed historical row. The primary
Gemma-3-4B/GSM8K setting is always reported separately and is never pooled
into the fourteen extensions. Adjacent controls use only the three
prespecified settings. Cluster keys, resampling counts, seed derivation, and
the Figure 5 field-by-field scope are documented in
[`docs/build-one-token-tables.md`](docs/build-one-token-tables.md).

## Measure sensitivity to the number of edits

`edit-count-sensitivity` builds Appendix C/Table 8 from fresh, verified
setting-level artifacts. The accuracy and CoT-swap parts have different grids
and denominators, so their GPU producers remain explicit rather than being
hidden inside the CPU aggregation command.

First run `prepare-edited-pairs` with Attribution-4 separately for one, two,
and four requested edits in the exact 51-setting accuracy grid:

```bash
ACCURACY_FULL_MODELS=(
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  google/gemma-3-12b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  Qwen/Qwen2.5-0.5B-Instruct
  Qwen/Qwen2.5-1.5B-Instruct
)
ACCURACY_BENCHMARKS=(arc csqa gsm8k math-500 mmlu mmlu-pro)

prepare_edit_count_setting() {
  local MODEL="$1"
  local BENCHMARK="$2"
  local MODEL_SLUG="${MODEL##*/}"
  for EDIT_COUNT in 1 2 4; do
    CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
      typo-cot prepare-edited-pairs \
      --model "${MODEL}" \
      --benchmark "${BENCHMARK}" \
      --targeting attribution-4 \
      --num-edits "${EDIT_COUNT}" \
      --gpu-id 0 \
      --output-dir \
        "results/edit-count-pairs/${MODEL_SLUG}/${BENCHMARK}/${EDIT_COUNT}"
  done
}

for MODEL in "${ACCURACY_FULL_MODELS[@]}"; do
  for BENCHMARK in "${ACCURACY_BENCHMARKS[@]}"; do
    prepare_edit_count_setting "${MODEL}" "${BENCHMARK}"
  done
done
for BENCHMARK in gsm8k mmlu mmlu-pro; do
  prepare_edit_count_setting Qwen/Qwen2.5-3B-Instruct "${BENCHMARK}"
done
```

For the six restoration settings—Gemma-3-4B, Llama-3.2-3B, and Mistral-7B
crossed with GSM8K and MMLU—run the same four-cell CoT swap from each edit-count
source. `--source-num-edits` defaults to four for the main CoT-swap experiment;
the explicit value below labels the Table 8 sensitivity protocol:

```bash
RESTORATION_MODELS=(
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
for MODEL in "${RESTORATION_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  for BENCHMARK in gsm8k mmlu; do
    for EDIT_COUNT in 1 2 4; do
      CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
        typo-cot cot-swap \
        --model "${MODEL}" \
        --benchmark "${BENCHMARK}" \
        --pairs \
          "results/edit-count-pairs/${MODEL_SLUG}/${BENCHMARK}/${EDIT_COUNT}/pairs.jsonl" \
        --targeting attribution-4 \
        --source-num-edits "${EDIT_COUNT}" \
        --gpu-id 0 \
        --output-dir \
          "results/edit-count-cot-swap/${MODEL_SLUG}/${BENCHMARK}/${EDIT_COUNT}"
    done
  done
done
```

After all producers finish, build Table 8 on CPU:

```bash
uv run --project projects/typo-cot \
  typo-cot edit-count-sensitivity \
  --pairs-root results/edit-count-pairs \
  --cot-swap-runs-root results/edit-count-cot-swap \
  --edit-counts 1 2 4 \
  --output-dir results/edit-count-sensitivity
```

The builder recursively discovers completed producer manifests and verifies the
paper fingerprint, protocol, setting identity, source and output hashes, full
unlimited cohorts, and record-level integer events. Accuracy uses complete
Attribution-4 model–benchmark settings. Its equal-setting row keeps each
condition's full denominator, while its matched row intersects sample IDs over
clean, one-, two-, and four-edit conditions. Clean answers must agree across
the three independently prepared sources. CoT restoration is undefined at zero
edits and conditions separately at each edit count on a regenerated correct A
whose B answer changes; those three denominators are never intersected.

The final-PDF grid is complete only with the exact 51 accuracy settings above
and all eighteen CoT-swap runs (six settings by three edit counts). The PDF
specifies the count and six benchmarks; setting identities are recovered from
the submitted Table 8 source and frozen only to prevent a different 51-cell
grid from being mislabeled. Partial valid inputs remain
auditable, but the corresponding paper-pooled comparison is omitted. The
output directory must not already exist and is published atomically with
`edit_count_records.jsonl`, `edit_count_summary.json`,
`table8_edit_count.csv`, `table8_edit_count.md`, `table8_edit_count.tex`, and
`run.json`. See
[`docs/edit-count-sensitivity.md`](docs/edit-count-sensitivity.md) for the
denominator, validation, historical-reference, and comparability contracts.

## Compare complete-CoT swaps across model scale

`model-scale-cot-swap` builds Appendix C/Table 9 from independently resumable
MMLU Attribution-4 producer runs. The paper fixes nine models and applies one
shared selector containing the first 500 seed-42 MMLU loader IDs. The selector
is versioned at
`projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json`; it contains
dataset IDs, the exact model-specific selected-ID set hashes, and protocol
metadata, not historical model outputs.

Run pair preparation and CoT swap separately for each model. The commands
below default to physical GPU 0. Set `MODEL_SCALE_GPU_IDS` to a comma-separated
set only when a 70B/72B model requires model sharding on the reproduction
machine; each setting still produces its own manifest and resumable output.

```bash
MODEL_SCALE_GPU_IDS="${MODEL_SCALE_GPU_IDS:-0}"
MODEL_SCALE_MODELS=(
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  google/gemma-3-12b-it
  google/gemma-3-27b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-70B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  Qwen/Qwen2.5-72B-Instruct
)
MODEL_SCALE_COHORT=projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json

for MODEL in "${MODEL_SCALE_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  CUDA_VISIBLE_DEVICES="${MODEL_SCALE_GPU_IDS}" \
    uv run --project projects/typo-cot --extra lrp \
    typo-cot prepare-edited-pairs \
    --model "${MODEL}" \
    --benchmark mmlu \
    --targeting attribution-4 \
    --num-edits 4 \
    --sample-ids "${MODEL_SCALE_COHORT}" \
    --gpu-id "${MODEL_SCALE_GPU_IDS}" \
    --output-dir "results/model-scale-pairs/${MODEL_SLUG}"

  CUDA_VISIBLE_DEVICES="${MODEL_SCALE_GPU_IDS}" \
    uv run --project projects/typo-cot --extra lrp \
    typo-cot cot-swap \
    --model "${MODEL}" \
    --benchmark mmlu \
    --pairs "results/model-scale-pairs/${MODEL_SLUG}/pairs.jsonl" \
    --targeting attribution-4 \
    --gpu-id "${MODEL_SCALE_GPU_IDS}" \
    --output-dir "results/model-scale-cot-swap-runs/${MODEL_SLUG}"
done
```

If either producer is interrupted, rerun that model's identical command with
`--resume`; a new output directory must be started without that flag.

After all nine producer settings complete, build Table 9 on CPU:

```bash
uv run --project projects/typo-cot \
  typo-cot model-scale-cot-swap \
  --pairs-root results/model-scale-pairs \
  --cot-swap-runs-root results/model-scale-cot-swap-runs \
  --cohort projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json \
  --output-dir results/model-scale-cot-swap
```

The shared selector is intersected with each model's final-paper MMLU source
cohort before inference. The submitted setup used 50 examples per subject for
five smaller-model settings and 100 for Gemma-12B/27B and the 70B/72B scale
checks; consequently the selector retains 250 or 500 source IDs respectively.
This model-specific cap and exact ID-set identity are recovered producer
details, not claims added to the PDF. Pair preparation, CoT swap, and the CPU
builder each verify those details. The builder also verifies all source and
output hashes, the exact nine-setting grid, binds every gold answer to its
prepared pair, and recomputes A/B/C/D correctness and event semantics.

`n_s` is the number of executed, regenerated-A-correct pairs and is the common
denominator for Both, Question only, and CoT only. Restoration uses only the
`n_B` subset for which B differs from A. The output directory is published
atomically with `model_scale_records.jsonl`, `model_scale_summary.json`,
`table9_model_scale.csv`, `table9_model_scale.md`, `table9_model_scale.tex`, and
`run.json`. Fresh results are compared descriptively with the final-PDF table;
Qwen2.5-72B remains directional because its published `n_B` is only 10. See
[`docs/model-scale-cot-swap.md`](docs/model-scale-cot-swap.md) for the full
cohort, denominator, validation, and hardware contracts.

## Compare edited-input accuracy with and without a typo warning

`typo-warning-prompt` implements the Appendix E audit behind the reported
GSM8K change from 60.1% to 54.1% and MMLU change from 57.6% to 56.2%. It
regenerates the same edited Attribution-4 question twice, once with the
submitted warning disabled and once with it inserted immediately before the
task's final “Now solve/answer” marker. The edited question, answer choices,
few-shot examples, and all text after that marker remain byte-identical.

Run the six submitted model-task settings. The repository includes an
output-free edit manifest that reconstructs the exact 300 submitted
Attribution-4 inputs in each setting from pinned public benchmark records. It
contains no archived generations, extracted answers, correctness labels, or
accuracy results.

```bash
WARNING_MODELS=(
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
WARNING_BENCHMARKS=(gsm8k mmlu)

for MODEL in "${WARNING_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  for BENCHMARK in "${WARNING_BENCHMARKS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
      typo-cot typo-warning-prompt \
      --model "${MODEL}" \
      --benchmark "${BENCHMARK}" \
      --gpu-id 0 \
      --output-dir \
        "results/typo-warning-prompt/${MODEL_SLUG}/${BENCHMARK}"
  done
done

uv run --project projects/typo-cot \
  typo-cot build-typo-warning-summary \
  --runs-root results/typo-warning-prompt \
  --output-dir results/typo-warning-summary
```

Add `--resume` only when continuing an interrupted setting command. `--limit 1`
labels a smoke run and is not accepted by the paper-summary builder. Each
setting publishes `warning_prompt_records.jsonl`,
`warning_prompt_summary.json`, and `run.json`. The CPU builder validates the
complete six-setting grid, the exact submitted input-manifest identity, all
source/output hashes, the paired ID sets,
and the two arm outcomes before pooling the three 300-item settings within each
benchmark. It writes `typo_warning_summary.json`,
`typo_warning_summary.csv`, `typo_warning_summary.md`,
`typo_warning_summary.tex`, and `run.json`; the p-value is the exact two-sided
McNemar/binomial test over discordant paired outcomes.

Per-setting summaries remain byte-attested derived artifacts, but the builder
does not use their stored metrics. It recomputes publication statistics from
the validated paired records, so CPU-analysis changes do not require another
model-generation run.

“CPU builder” means that summary construction loads no model weights and needs
no GPU. It still reopens GSM8K and MMLU at their pinned revisions to verify the
submitted inputs, so the base installation includes `datasets` and the first
uncached build requires network access. A complete compatible Hugging Face
dataset cache can satisfy the same reads offline.

The PDF defines the warning comparison, two tasks, printed accuracies, and
significance conclusion. The exact English instruction, three-model grid,
seed-42 shared-ID shuffle, 300-item cohort, insertion boundary, same-arm batches
of eight, task-specific submitted answer extractor, and 512-token greedy
generation are recovered from the submitted producer and labelled
`legacy-backed`. The public runner follows those recovered details and writes
one restartable checkpoint per sample and arm. Recovered model revisions are
identified from the submission environment's cache because the producer did
not record them. Fresh outputs are the public reproduction result; printed
percentages are descriptive historical references, not acceptance targets. As
the paper cautions, this one instruction over two tasks is not a general
self-correction evaluation and is not a performance comparison with activation
patching. See
[`docs/typo-warning-prompt.md`](docs/typo-warning-prompt.md) for the complete
prompt, selection, validation, schema, and restart contracts.

## Audit input correctors and build Table 12

`input-corrector-audit` applies one explicitly selected corrector to one
model-task Attribution-4 input set. The core experiment is the complete grid
of five evaluation models, five tasks, and three correctors. Each source must
be a completed, unlimited `prepare-edited-pairs` run with seed 42, four edits,
and `attribution-4` targeting; the runner validates the sibling `run.json`
before loading a corrector.

The first loop below prepares all 30 Attribution-4 model-task sources with the
paper counts (1,319/2,850/1,400/1,172/1,221 core items and 500 MATH-500 items
per model). The next loop runs the 75 core corrector settings and builds the
table. The optional MATH-500 loop reproduces the Appendix E collateral-change
diagnostic for the T5 and Qwen correctors; those ten runs never enter the
25-setting Table 12 mean.

```bash
GPU_ID="${GPU_ID:-0}"
INPUT_CORRECTOR_MODELS=(
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
INPUT_CORRECTOR_BENCHMARKS=(gsm8k mmlu mmlu-pro arc csqa)
INPUT_CORRECTORS=(pyspellchecker t5-large-spell qwen2.5-7b-instruct)
INPUT_CORRECTOR_SOURCE_BENCHMARKS=(gsm8k mmlu mmlu-pro arc csqa math-500)

# Prepare the complete Attribution-4 source matrix consumed below.
for MODEL in "${INPUT_CORRECTOR_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for BENCHMARK in "${INPUT_CORRECTOR_SOURCE_BENCHMARKS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      uv run --project projects/typo-cot --extra lrp \
      typo-cot prepare-edited-pairs \
      --model "${MODEL}" \
      --benchmark "${BENCHMARK}" \
      --targeting attribution-4 \
      --num-edits 4 \
      --seed 42 \
      --max-new-tokens 512 \
      --gpu-id "${GPU_ID}" \
      --output-dir \
        "results/prepare-edited-pairs/${MODEL_SLUG}/${BENCHMARK}/attribution-4"
  done
done

# Run the 75 core corrector settings.
for MODEL in "${INPUT_CORRECTOR_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for BENCHMARK in "${INPUT_CORRECTOR_BENCHMARKS[@]}"; do
    for CORRECTOR in "${INPUT_CORRECTORS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        uv run --project projects/typo-cot --extra lrp \
        typo-cot input-corrector-audit \
        --corrector "${CORRECTOR}" \
        --model "${MODEL}" \
        --benchmark "${BENCHMARK}" \
        --pairs \
          "results/prepare-edited-pairs/${MODEL_SLUG}/${BENCHMARK}/attribution-4/pairs.jsonl" \
        --gpu-id "${GPU_ID}" \
        --output-dir \
          "results/input-corrector-audit/core/${CORRECTOR}/${MODEL_SLUG}/${BENCHMARK}"
    done
  done
done

INPUT_CORRECTOR_MATH_CORRECTORS=(t5-large-spell qwen2.5-7b-instruct)
# Run the optional ten-setting MATH-500 diagnostic.
for MODEL in "${INPUT_CORRECTOR_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for CORRECTOR in "${INPUT_CORRECTOR_MATH_CORRECTORS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      uv run --project projects/typo-cot --extra lrp \
      typo-cot input-corrector-audit \
      --corrector "${CORRECTOR}" \
      --model "${MODEL}" \
      --benchmark math-500 \
      --pairs \
        "results/prepare-edited-pairs/${MODEL_SLUG}/math-500/attribution-4/pairs.jsonl" \
      --gpu-id "${GPU_ID}" \
      --output-dir \
        "results/input-corrector-audit/math-500/${CORRECTOR}/${MODEL_SLUG}"
  done
done

# Build Table 12 and the optional diagnostic summary on CPU.
uv run --project projects/typo-cot \
  typo-cot build-input-corrector-summary \
  --runs-root results/input-corrector-audit/core \
  --math-runs-root results/input-corrector-audit/math-500 \
  --output-dir results/input-corrector-summary
```

The `Word` value is the equally weighted mean of 25 within-setting exact
restoration rates, not a pooled word ratio. `Exact clean` compares the final
clean and corrected prompts byte for byte, including few-shot text and
whitespace. For those exact prompts, `Same` generates adjacent duplicate rows
in one call, using `[p, p, q, q]` batches. A whitespace-normalized restoration
flag is retained only as a diagnostic and cannot define `Exact clean`.

Each setting writes `corrector_records.jsonl`,
`corrector_audit_summary.json`, and `run.json`. The builder writes
`input_corrector_summary.json`, `table12_input_correctors.csv`,
`table12_input_correctors.md`, `table12_input_correctors.tex`, and `run.json`.
The paper's `archive` column compared a corrected-generation run with a
separately archived clean run. The fresh source-pair comparison is reported
separately, but it is not substituted for the published archive count and is
not interpreted as a corrector effect or subtractable noise. Add `--resume`
only to continue an interrupted setting; `--limit 1` is a labelled smoke run
and the complete-grid builder rejects it. See
[`docs/input-corrector-audit.md`](docs/input-corrector-audit.md) for the exact
corrector prompts, alignment metric, provenance, validation, and restart
contracts. If the optional MATH-500 loop is skipped, omit
`--math-runs-root` from the builder command as well.

## Compare edited-word restoration orders with the Table 13 protocol

`restoration-order-accuracy` is the Appendix E oracle diagnostic behind
Table 13. It does not run a fourth input corrector. Starting from the
Attribution-4 source outcome, it selects clean-correct/four-edit-wrong items,
restores known clean substrings in high-relevance-first, seeded-random, or
low-relevance-first order, and freshly regenerates the answer at equal budgets.
This needs the paired clean input and AttnLRP relevance and is therefore an
analysis upper bound, not a deployable typo detector or correction method. The
public command produces a fresh final-PDF protocol replication; it does not
claim to recover the private archived cohort byte for byte.

Run the complete three-model by two-task grid below. The source-preparation
loop is needed when these six completed runs were not already created for the
Table 12 audit.

```bash
GPU_ID="${GPU_ID:-0}"
RESTORATION_MODELS=(
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
RESTORATION_BENCHMARKS=(gsm8k mmlu)

# Prepare the six complete Attribution-4 sources if they are not already present.
for MODEL in "${RESTORATION_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for BENCHMARK in "${RESTORATION_BENCHMARKS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      uv run --project projects/typo-cot --extra lrp \
      typo-cot prepare-edited-pairs \
      --model "${MODEL}" \
      --benchmark "${BENCHMARK}" \
      --targeting attribution-4 \
      --num-edits 4 \
      --seed 42 \
      --max-new-tokens 512 \
      --gpu-id "${GPU_ID}" \
      --output-dir \
        "results/prepare-edited-pairs/${MODEL_SLUG}/${BENCHMARK}/attribution-4"
  done
done

# Generate both shared endpoints and all nine intermediate conditions.
for MODEL in "${RESTORATION_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  MODEL_SLUG="${MODEL_SLUG,,}"
  for BENCHMARK in "${RESTORATION_BENCHMARKS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      uv run --project projects/typo-cot --extra lrp \
      typo-cot restoration-order-accuracy \
      --model "${MODEL}" \
      --benchmark "${BENCHMARK}" \
      --pairs \
        "results/prepare-edited-pairs/${MODEL_SLUG}/${BENCHMARK}/attribution-4/pairs.jsonl" \
      --orders high-relevance-first seeded-random low-relevance-first \
      --budgets 0 1 2 3 4 \
      --seed 42 \
      --batch-size 8 \
      --gpu-id "${GPU_ID}" \
      --output-dir \
        "results/restoration-order-accuracy/${MODEL_SLUG}/${BENCHMARK}"
  done
done

# Validate the six settings and build a fresh Table 13 protocol replication on CPU.
uv run --project projects/typo-cot \
  typo-cot build-restoration-order-table \
  --runs-root results/restoration-order-accuracy \
  --output-dir results/restoration-order-table
```

The cohort is frozen from the source clean/four-edit outcomes before any of
the eleven new conditions are generated. It is never re-filtered using the new
`k=0` or `k=4` answers. The CPU builder pairs every condition by full
model/task/sample identity, pools items rather than equally weighting six
settings, and computes the reported high-first-versus-random p values with a
two-sided exact McNemar/binomial test. Those three tests are descriptive and
unadjusted, as in the paper.

Before source selection, both stored endpoint continuations are rescored with
the final-PDF primary-then-empty-only-fallback rule. A stale or primary-only
stored answer is rejected instead of silently changing the cohort denominator.
The completed producer manifest also binds the exact `pairs.jsonl` bytes and
record count. Sources made before that binding or explicit EOS-versus-length-cap
termination was introduced must be regenerated with `prepare-edited-pairs`;
editing relevance or token-position metadata after completion is rejected
before model loading.

For every retained item, the runner rebuilds the full clean prompt from its
question, choices, and subject. It first checks the current GSM8K 8-shot or
MMLU 5-shot template against the archived `*_cot_v1` probe SHA-256, then
requires the rebuilt UTF-8 bytes and editable span to match the source. The
six-setting builder additionally requires one model revision across both tasks
for each model, and one dataset plus ordered-sample identity across all three
models for each task.

The PDF reports 1,582 archived-selected items, endpoint accuracies 12.0% and
88.9%, and the three intermediate rows. Fresh public preparation follows the
same paper protocol but does not prove byte-identical membership in that
private archive, so printed values are retained as historical references rather
than acceptance targets. The submitted producer's stable seed-42 random order
is versioned unchanged because replacing its key derivation would change the
random row and paired p values. The public schema records each source item's
realized edit count and makes budgets at or above that count equal to the clean
endpoint instead of silently dropping or padding that item.

For submitted grouping compatibility, restoration uses contiguous
`difflib` character edit groups. The paper describes these units as edited
words, but one group is not guaranteed to correspond one-to-one with a
whitespace-delimited word; the public records expose both the realized groups
and their source events instead of hiding that distinction.

Each setting writes `restoration_order_records.jsonl`,
`restoration_order_summary.json`, and `run.json`. The builder writes
`restoration_order_table.json`, `table13_restoration_order.csv`,
`table13_restoration_order.md`, `table13_restoration_order.tex`, and
`run.json`. Every rendered result labels its fresh pooled cohort size;
Markdown and LaTeX also state the historical PDF cohort size. A setting's
final output directory appears
only after all artifacts commit atomically; add `--resume` only for an
interrupted private work directory. `--limit 1` is a labelled GPU smoke run
and is rejected by the complete-grid builder. See
[`docs/restoration-order-accuracy.md`](docs/restoration-order-accuracy.md) for
the source-selection, reconstruction, batching, provenance, inference, and
restart contracts.

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
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_build_one_token_tables_*.py
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_edit_count_sensitivity.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_typo_warning_prompt.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_input_corrector_*.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_restoration_order_*.py
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
