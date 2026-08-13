# Typo-robustness training

This project trains and evaluates typo-robust adapters without changing the
locked environment used to reproduce the submitted activation-patching paper.
The method is prospective. Implementation is reviewed in feature-scoped pull
requests, while generated data, checkpoints, and experimental results remain
local artifacts until the frozen evaluation protocol permits a claim.

The scientific order is fixed as:

```text
training/evaluation data -> generic-text causal layer localization
                         -> adapter training -> held-out evaluation
```

The confirmatory target is a model-specific residual-stream window selected by
joint Activation Patching on generic text. Selection uses only multi-token KL
restoration; downstream answers, clean-harm scores, neuron/head screening, and
training outcomes cannot alter the target. Neuron/head localization remains an
exploratory negative-result analysis rather than part of the proposed method.

## Environment

Run commands from the repository root. The training project owns its lockfile
and adds training-only dependencies such as PEFT. `projects/typo-cot/uv.lock`
and its paper-reproduction environment are not refreshed.

```bash
TRAIN_PROJECT=projects/typo-robust-training
TRAIN_ROOT=projects/typo-robust-training/results
GPU_SELECT=5
GPU_VALIDATE=6
GPU_ID=5  # exploratory commands below
WANDB_PROJECT=typo-robustness-training
# Before training, provide WANDB_API_KEY through a secret manager or the environment.
# Optional: export WANDB_ENTITY=<team-or-user-entity>

uv sync --project "${TRAIN_PROJECT}" --locked
```

All released commands take one scientific operation. The current experiment
uses physical GPUs 5 and 6; a public clone may select any available devices.

## 1. Build leakage-resistant training and evaluation data

Before the first build, obtain GitHub Typo Corpus v1.0.0 under its
origin-repository terms. It is not redistributed by this repository. The
approval file must be a JSON object mapping each repository URL that may be
used to its verified license; unlisted repositories are dropped. Dolma is
streamed from the URL inventory at its pinned dataset revision. A locally
obtained JSONL or JSONL.GZ Dolma sample may be supplied as an optional cache.

```bash
export TYPO_GITHUB_CORPUS_PATH=/absolute/path/to/github-typo-corpus.v1.0.0.jsonl.gz
export TYPO_GITHUB_APPROVED_REPOSITORIES=/absolute/path/to/github-typo-approved-repositories.json
# Optional: export TYPO_DOLMA_CORPUS_PATH=/absolute/path/to/dolma-v1_5-sample.jsonl.gz
```

The builder records SHA-256 digests of the natural inputs and either the local
Dolma archive or its pinned URL inventory, together with the selected shard
URLs. A missing required file, unapproved natural-typo repository, malformed
record, or source revision drift fails the run visibly rather than silently
changing the mixture.

```bash
uv run --project "${TRAIN_PROJECT}" --locked typo-cot build-robustness-training-data \
  --config "${TRAIN_PROJECT}/configs/gemma4b-sanity.yaml" \
  --output-dir "${TRAIN_ROOT}/data/gemma4b-sanity"
```

The default sanity mixture is measured by non-padding student tokens: 85%
FineWeb-Edu, 10% public GSM8K/MMLU/ARC-Challenge train or development data, and
at most 5% natural pairs from training repositories in GitHub Typo Corpus.
FineWeb-Edu document groups and natural-typo repositories are split before
sampling. Dolma, MMLU-Pro, MATH-500, CommonsenseQA, held-out natural
repositories, and adjacent transpositions never tune the v1 method.

For long FineWeb-Edu and Dolma documents, the builder first chooses one
content-hash-stable 8,192-character window, then keeps a whole-word prefix that
fits the pinned tokenizer's 512-token limit. It records the original length,
window boundaries, retained character count, and exact token count. This keeps
typo targets inside the sequence seen by the model and bounds deduplication
memory without choosing windows from evaluation outcomes.

Training records retain clean text and generate substitution, deletion,
insertion, duplication, and keyboard-neighbor typos deterministically on the
fly. The five training operations are sampled uniformly in v1; the general
substitution character is sampled from training-repository-only natural typo
statistics. Fixed tune, pre-PR-gate, and final-test typo pairs are materialized and
content-hashed. The final-test identities remain sealed until all methods,
hyperparameters, stopping rules, and the passing pre-PR checkpoint are frozen.

The command writes:

- `training_sources.jsonl`: ordered clean training records and source metadata;
- `typo_statistics.json`: character-edit statistics derived only from natural
  pairs in training repositories;
- `diagnostic_manifest.jsonl`: fixed GSM8K/MMLU/ARC train/dev clean-typo
  localization pairs;
- `tune_manifest.jsonl`: fixed iteration-only evaluation pairs;
- `pre_pr_gate_manifest.jsonl`: fixed one-use pre-PR gate pairs;
- `final_test_manifest.jsonl`: sealed final paper-test identities;
- `evaluation_manifest.json`: split roles, hashes, typo-operation inventory,
  and benchmark/source revisions;
- `decontamination_report.json`: exact/near-duplicate and task-denylist audit;
- `run.json`: arguments, environment, counts, hashes, and completion status.

Corpus text, generated pairs, checkpoints, and run outputs are local artifacts
under `results/` and are never committed.

## 2. Freeze and select the confirmatory causal window

The selector first freezes 200 selection and 200 validation FineWeb-Edu pairs
that are disjoint from training and every evaluation tier. Each document gets
one deterministic typo from the paper's keyboard-neighbor substitution,
deletion, and duplication operations. For every contiguous window of width
`max(1, floor(L / 6 + 0.5))`, it jointly patches complete decoder-block
residual outputs at the edited-word-final token. The greatest median pairwise
multi-token KL restoration over tokens 2--16 wins; exact ties select the
shallower window. Independent validation must have a bootstrap 95% confidence
interval whose lower bound is above zero.

```bash
LOCALIZATION_ROOT="${TRAIN_ROOT}/localization/generic-joint-window-v1"

uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot freeze-generic-localization-pairs \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-generic-joint-window.yaml" \
  --exclude-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --output-dir "${LOCALIZATION_ROOT}/pairs"

CUDA_VISIBLE_DEVICES="${GPU_SELECT}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot select-generic-joint-patch-window \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-generic-joint-window.yaml" \
  --selection-manifest "${LOCALIZATION_ROOT}/pairs/selection_manifest.jsonl" \
  --gpu-id "${GPU_SELECT}" \
  --output-dir "${LOCALIZATION_ROOT}/selection" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_VALIDATE}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot validate-generic-joint-patch-window \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-generic-joint-window.yaml" \
  --validation-manifest "${LOCALIZATION_ROOT}/pairs/validation_manifest.jsonl" \
  --window-selection "${LOCALIZATION_ROOT}/selection/window_selection.json" \
  --gpu-id "${GPU_VALIDATE}" \
  --output-dir "${LOCALIZATION_ROOT}/validation" \
  --resume
```

Pair identities, exclusions, realized typos, source/model revisions, per-pair
KL denominators, scans, and output hashes are bound into run manifests. A model
whose independent validation fails is not eligible for localized-state
training. Failed validation retains its audit artifacts and exits nonzero.

### Exploratory reasoning-task selector

The earlier exploratory selector remains available to reproduce the negative
component-localization study, but it does not choose the confirmatory target.

```bash
CUDA_VISIBLE_DEVICES="${GPU_SELECT}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot select-distillation-layers \
  --config "${TRAIN_PROJECT}/configs/gemma4b-layer-selection.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --tasks gsm8k mmlu arc \
  --gpu-id "${GPU_SELECT}" \
  --output-dir "${TRAIN_ROOT}/localization/layers" \
  --resume
```

This command records the historical composite-score window using diagnostic
reasoning data. Its answer and harm terms are not used by the confirmatory
generic-text selector above.

## 3. Reproduce the exploratory neuron/head causal analysis

This command reproduces the component-level study that preceded the current
residual-window method. It is an ablation and negative-result audit: its output
must not choose or modify the confirmatory training target.

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot localize-robustness-components \
  --config "${TRAIN_PROJECT}/configs/gemma4b-component-localization.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --components mlp-neuron attention-head \
  --causal-readouts answer multitoken-kl \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/localization/components" \
  --resume
```

Activation difference and gradient attribution only shortlist components.
`component_selection.json` contains only candidates whose clean-to-typo causal
patch has a beneficial direction on at least two tasks and passes the frozen
clean-harm rule. This historical selection artifact is retained for analysis;
the proposed adapter uses the independently validated generic-text residual
window from Section 2.

## 4. Train baseline adapters and reproduce the historical Cycle 1 ablation

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-noisy-language-model \
  --config "${TRAIN_PROJECT}/configs/baselines/noisy-language-model.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/noisy-language-model/seed-42" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-output-matching \
  --config "${TRAIN_PROJECT}/configs/baselines/output-matching.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/output-matching/seed-42" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-global-state-alignment \
  --config "${TRAIN_PROJECT}/configs/baselines/global-state-alignment.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/global-state-alignment/seed-42" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-localized-state-distillation \
  --config "${TRAIN_PROJECT}/configs/ablations/gemma4b-component-state-cycle1.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --component-selection "${TRAIN_ROOT}/localization/components/component_selection.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/localized-state-distillation/seed-42" \
  --resume
```

The last command is retained solely to reproduce the failed component-level
Cycle 1 ablation. Its relative-MSE objective and component-selected LoRA are
not the confirmatory method. The bounded residual-window objective that
consumes the Section 2 artifacts is published as a separate feature-scoped
change, so the historical failure cannot silently select the active target.

The commands above reproduce the version-1 pilot and must not be reported as
the confirmatory comparison. The later bounded residual-window training
feature supplies its matched output-only baseline and controls for seeds 42,
43, and 44. In every teacher/student condition, the teacher receives clean
input, stays frozen, and is never activation-patched; only declared student
LoRA parameters may change.

Every public training command requires online W&B tracking. Supply the API key
only through `WANDB_API_KEY`; `WANDB_ENTITY` is optional. At each completed
optimizer step the run uploads aggregate total/component losses, learning
rate, gradient norm, token throughput, current GPU memory, and GPU-memory peak
since training start. Raw corpus text, prompts, record IDs, the API key, and
checkpoint contents are never sent.
`wandb_run.json` stores only non-secret run identity, scientific
bindings/presentation, URL, status, and the resume boundary so `--resume`
continues the same W&B run without duplicating the loss curve.

W&B names expose the scientific role directly; opaque arm abbreviations are
not used. The adapter configs published in this feature are the version-1
Cycle 1 reproduction suite, so their historical status is explicit:

| Role shown in W&B | Operation | Meaning |
|---|---|---|
| `Historical baseline` | `Noisy-language-model training` | Cycle 1 ordinary causal-language-model baseline on noisy text |
| `Historical pilot` | `Output/answer/clean-loss training` | Cycle 1 multi-loss output-matching pilot |
| `Historical control` | `Global relative-MSE state alignment` | Cycle 1 all-layer/all-token state control |
| `Historical ablation` | `Component-level relative-MSE state distillation` | Failed Cycle 1 neuron/head experiment; not the confirmatory method |

The suffix records state layers, model, optimizer-step budget, and seed. All
of these version-1 runs are placed in the separate `Historical Cycle 1` group,
so they cannot be mistaken for the bounded residual-window comparison, whose
config and W&B mapping are introduced with that later training feature.

### Confirmatory Cycle 3 training and controls

Cycle 3 holds the frozen self-teacher, training stream, all-linear LoRA
capacity, optimizer, exact clean:noisy alternation, and 10M student-token
budget constant. The proposed condition differs from output-distribution
matching only by a bounded residual-state cosine loss at the independently
selected causal window. The random-window and all-layer controls change only
the state-loss layer scope.

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-random-window-state-distillation \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-random-window-10m.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-cycle3-64m" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${TRAIN_ROOT}/evaluation-data/robustness-v1" \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/random-window-state-distillation/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-global-state-alignment \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-all-layer-state-10m.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-cycle3-64m" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${TRAIN_ROOT}/evaluation-data/robustness-v1" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/all-layer-state-distillation/seed-42"
```

Run the two controls serially when only one GPU is available. Add `--resume`
only when the same output directory already contains an exact compatible
checkpoint. W&B uses descriptive names: `Control · Random-window state
distillation` and `Control · All-layer state distillation`.

## 5. Freeze the independent evaluation study

Freeze the exact clean/typo texts before comparing any adapter. The source
config must be the byte-identical v3 config used to build the exclusion data;
this binds every evaluation item to the training, diagnostic, and tune IDs it
must exclude. This step runs no model and reveals no model output.

```bash
EVALUATION_DATA="${TRAIN_ROOT}/evaluation-data/robustness-v1"
SOURCE_CONFIG="${TRAIN_PROJECT}/configs/cycle3/gemma4b-data-64m.yaml"

uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot freeze-robustness-evaluation \
  --protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --source-config "${SOURCE_CONFIG}" \
  --exclude-data "${TRAIN_ROOT}/data/gemma4b-cycle3-64m" \
  --output-dir "${EVALUATION_DATA}"
```

The command writes hash-bound `tune`, one-use `pre_pr_gate`, and one-use
`final_test` task and corpus manifests. Exact typo strings are shared by Base,
the output-distribution-matching baseline, and every proposed adapter. Task
IDs, corpus groups, natural-typo repositories, and corrected words are
disjoint across training/tune/sealed roles. The committed protocol and its
gate definitions are described in
[`../typo-cot/docs/robustness_evaluation_protocol_v1.md`](../typo-cot/docs/robustness_evaluation_protocol_v1.md).

## 6. Evaluate held-out robustness

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot evaluate-typo-robustness \
  --config "${TRAIN_PROJECT}/configs/gemma4b-evaluation.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --evaluation-role tune \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --checkpoint "${TRAIN_ROOT}/training/localized-state-distillation/seed-42/adapter" \
  --splits same-task unseen-task unseen-content unseen-typo \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/evaluation/tune/targeted-seed-42"
```

`base` is always evaluated automatically. Repeat `--checkpoint` to compare
multiple completed adapters; each path must contain the hash-bound
`training_runtime.json` written by its training command. Use only `tune` while
changing the method. After all hyperparameters and the checkpoint are frozen,
run the same command once with `--evaluation-role pre-pr-gate` and
`--confirm-sealed-role`; use `final-test` only after the passing pre-PR
checkpoint is frozen. Sealed-role access is recorded next to the immutable data
artifacts and cannot be silently repeated.

The generic-text window must include its independent passing validation
artifact. The evaluator binds both files into the run identity; it will not
silently fall back to the historical reasoning-task selector.

The command above starts a fresh evaluation. Add `--resume` only after that
evaluation output directory contains its existing `run.json` and pair
checkpoints.

The report includes clean/typo accuracy, wrong-to-right, right-to-wrong, net
accuracy, clean harm, multi-token KL, additional paired-patch gain,
tokenization strata, unseen tasks, unseen operations, and natural typos. The
frozen pre-PR gate requires a primary random-2 typo improvement of at least +2
accuracy points over Base with a 95% confidence-interval lower bound above
zero, a clean macro decrease of at most 1 point, no task decrease of 3 points
or more, a held-out clean perplexity ratio at most 1.02, and no material natural
typo degradation. The mechanistic patch audit is reported but is not a blocking
gate. Confirmatory claims additionally require three training seeds; failed
attempts are retained locally and summarized in the eventual PR.

The full frozen protocol is in
[`../typo-cot/docs/robustness_training_plan_v1.md`](../typo-cot/docs/robustness_training_plan_v1.md).
