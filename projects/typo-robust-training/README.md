# Typo-robustness training

This project trains and evaluates typo-robust adapters without changing the
locked environment used to reproduce the submitted activation-patching paper.
The method is prospective. Implementation is reviewed in feature-scoped pull
requests, while generated data, checkpoints, and experimental results remain
local artifacts until the frozen evaluation protocol permits a claim.

The confirmatory Cycle 3 order is fixed as:

```text
64M training data -> frozen evaluation text -> generic-text causal localization
                  -> baseline/proposal/control training -> held-out evaluation
```

The confirmatory target is a model-specific residual-stream window selected by
joint Activation Patching on generic text. Selection uses only multi-token KL
restoration; downstream answers, clean-harm scores, neuron/head screening, and
training outcomes cannot alter the target. Neuron/head localization remains an
exploratory negative-result analysis rather than part of the proposed method.

Detailed Japanese design notes:

- [Current three-step proposal](docs/current_proposal_three_step_method_v1.ja.md)
- [SAE diagnostics and successor proposals](docs/sae_and_successor_proposals_v1.ja.md)

The first note distinguishes the frozen current method from historical
ablations. The second distinguishes the parallel SAE diagnostic track from
future methods that are not yet authorized for training.

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

### Freeze the exact tokenizer before any scientific runtime

Probe, training, and evaluation processes all require the same frozen
tokenizer manifest. Run this command from a clean checkout at the exact code
revision that will execute the experiment:

```bash
MODEL_ID=mistralai/Mistral-7B-v0.1
MODEL_REVISION=7231864981174d9bee8c7687c24c8344414eae6b
CODE_REVISION="$(git rev-parse HEAD)"
TOKENIZER_FREEZE="${TRAIN_ROOT}/tokenizers/mistral-7b-v0.1"

uv run --project "${TRAIN_PROJECT}" --locked typo-cot freeze-tokenizer-attestation \
  --model "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --code-revision "${CODE_REVISION}" \
  --output-dir "${TOKENIZER_FREEZE}"

export TYPO_COT_TOKENIZER_ATTESTATION_MANIFEST="${TOKENIZER_FREEZE}/tokenizer-attestation.json"
```

The consumer file keeps the shared
`typo-cot-tokenizer-snapshot-attestation/v1` schema. It inventories and hashes
the tokenizer/config assets at the exact Hub commit. The adjacent freeze-run
manifest binds that file to the provider identity, complete source-tree
attestation, and runtime package versions; both files are deterministic and
closed-world validated. Branch names such as `main` are rejected.

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

### Build the confirmatory Cycle 3 training stream

The sanity build above is not the confirmatory training producer. Build the
full, hash-bound 64M-token clean source inventory before freezing evaluation,
localizing the causal window, or starting any Cycle 3 adapter. This is a CPU
data-preparation command; it does not run a model or open a sealed role.

```bash
CYCLE3_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m"

uv run --project "${TRAIN_PROJECT}" --locked typo-cot build-robustness-training-data \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-data-64m.yaml" \
  --output-dir "${CYCLE3_DATA}"
```

## 2. Freeze the independent evaluation study

Freeze the exact clean/typo texts after the Cycle 3 training inventory exists
and before any training command consumes monitor data. The source config must
be the byte-identical v3 config used to build the exclusion data; this binds
every evaluation item to the training, diagnostic, and tune IDs it must
exclude. This step runs no model and reveals no model output.

```bash
EVALUATION_DATA="${TRAIN_ROOT}/evaluation-data/robustness-v1"
SOURCE_CONFIG="${TRAIN_PROJECT}/configs/cycle3/gemma4b-data-64m.yaml"

uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot freeze-robustness-evaluation \
  --protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --source-config "${SOURCE_CONFIG}" \
  --exclude-data "${CYCLE3_DATA}" \
  --output-dir "${EVALUATION_DATA}"
```

The command writes hash-bound `tune`, one-use `pre_pr_gate`, and one-use
`final_test` task and corpus manifests. Exact typo strings are shared by Base,
the output-distribution-matching baseline, and every proposed adapter. Task
IDs, corpus groups, natural-typo repositories, and corrected words are
disjoint across training/tune/sealed roles. The committed protocol and its
gate definitions are described in
[`../typo-cot/docs/robustness_evaluation_protocol_v1.md`](../typo-cot/docs/robustness_evaluation_protocol_v1.md).

## 3. Freeze and select the confirmatory causal window

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
  --exclude-data "${CYCLE3_DATA}" \
  --exclude-data "${EVALUATION_DATA}" \
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

## 4. Reproduce the exploratory neuron/head causal analysis

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
window from Section 3.

## 5. Train adapters and controls

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
consumes the Section 3 artifacts is published as a separate feature-scoped
change, so the historical failure cannot silently select the active target.

The commands above reproduce the version-1 pilot and must not be reported as
the confirmatory comparison. The later bounded residual-window training
feature supplies proposed-method seeds 42, 43, and 44; its matched output-only
baseline and scope controls remain frozen at seed 42. In every teacher/student
condition, the teacher receives clean input, stays frozen, and is never
activation-patched; only declared student LoRA parameters may change.

Every public training command requires online W&B tracking. Supply the API key
only through `WANDB_API_KEY`; `WANDB_ENTITY` is optional. At each completed
optimizer step the run uploads aggregate total/component losses, learning
rate, gradient norm, token throughput, current GPU memory, and GPU-memory peak
since training start. Raw corpus text, prompts, record IDs, the API key, and
checkpoint contents are never sent.
`wandb_run.json` stores only non-secret run identity, scientific
bindings/presentation, URL, status, and the resume boundary so `--resume`
continues the same W&B run without duplicating the loss curve.

W&B names avoid opaque arm abbreviations. The version-1 Cycle 1 reproduction
suite uses the broad `Legacy` role and distinguishes conditions by the exact
operation name, job type, and condition tag:

| Role shown in W&B | Operation | Meaning |
|---|---|---|
| `Legacy` | `Noisy language-model training` | Cycle 1 ordinary causal-language-model baseline on noisy text |
| `Legacy` | `Output/answer/clean-loss pilot` | Cycle 1 multi-loss output-matching pilot |
| `Legacy` | `Global relative-MSE state pilot` | Cycle 1 all-layer/all-token state control |
| `Legacy` | `Component-level state distillation` | Failed Cycle 1 neuron/head experiment; not the confirmatory method |

The suffix records state layers, model, optimizer-step budget, and seed. These
version-1 runs use the runtime group `Cycle 1 · <model> · <budget>`; their
operation, job type, and condition tags keep them distinct from the bounded
residual-window comparison.

### Confirmatory Cycle 3 training and controls

Cycle 3 holds the frozen self-teacher, training stream, all-linear LoRA
capacity, optimizer, exact clean:noisy alternation, and 10M student-token
budget constant. The proposed condition differs from the output-distribution
matching baseline only by a bounded residual-state cosine loss at the
independently selected causal window. The random-window and all-layer controls
change only the state-loss layer scope.

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-output-matching \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-output-matching-10m.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${EVALUATION_DATA}" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/output-matching/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-localized-state-distillation \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-causal-window-10m.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${EVALUATION_DATA}" \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-random-window-state-distillation \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-random-window-10m.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${EVALUATION_DATA}" \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/random-window-state-distillation/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-global-state-alignment \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-all-layer-state-10m.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${EVALUATION_DATA}" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/all-layer-state-distillation/seed-42"
```

Run the four conditions serially when only one GPU is available. Add `--resume`
only when the same output directory already contains an exact compatible
checkpoint. W&B uses descriptive names beginning with `Kojima-inspired baseline`,
`Proposed method`, `Random-window control`, and `All-layer control`; each name
also includes the operation, layer range, model, token budget, and seed.

## 6. Evaluate held-out robustness

The five-arm, hidden-state-loss-free factorial based on the Linear Probe boundary
and the Activation-Patching downstream horizon is specified in
[`docs/state_free_probe_factorial_v1.ja.md`](docs/state_free_probe_factorial_v1.ja.md).

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot evaluate-typo-robustness \
  --config "${TRAIN_PROJECT}/configs/gemma4b-evaluation.yaml" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-data "${EVALUATION_DATA}" \
  --evaluation-role tune \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/output-matching/seed-42/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-42/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/random-window-state-distillation/seed-42/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/all-layer-state-distillation/seed-42/adapter" \
  --splits same-task unseen-task unseen-content unseen-typo \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/evaluation/tune/cycle3-comparison-seed-42"
```

`base` is always evaluated automatically, so the example compares five model
conditions on identical pairs. Each checkpoint path must contain the hash-bound
`training_runtime.json` written by its training command. This first command is
the repeatable seed-42 `tune` comparison; it is not a valid sealed-role
checkpoint inventory.

Only after the tune result freezes the method and hyperparameters, produce the
remaining proposed-method seeds required by the frozen evaluation protocol:

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-localized-state-distillation \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-causal-window-10m.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${EVALUATION_DATA}" \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --seed 43 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-43"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-localized-state-distillation \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-causal-window-10m.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${EVALUATION_DATA}" \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --seed 44 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-44"
```

The one-use pre-PR gate is executable only after both producers above complete.
It must receive the complete proposed-method seed inventory 42/43/44; the
matched baseline and scope controls remain the frozen seed-42 checkpoints:

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot evaluate-typo-robustness \
  --config "${TRAIN_PROJECT}/configs/gemma4b-evaluation.yaml" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --training-data "${CYCLE3_DATA}" \
  --evaluation-data "${EVALUATION_DATA}" \
  --evaluation-role pre-pr-gate \
  --confirm-sealed-role \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/output-matching/seed-42/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-42/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-43/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-44/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/random-window-state-distillation/seed-42/adapter" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/all-layer-state-distillation/seed-42/adapter" \
  --splits same-task unseen-task unseen-content unseen-typo \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/evaluation/pre-pr-gate/cycle3-comparison"
```

Use `final-test` only after the passing pre-PR checkpoint inventory is frozen,
by rerunning this complete-inventory command with `--evaluation-role
final-test` and a new final-test output directory. Sealed-role access is
recorded next to the immutable data artifacts and cannot be silently repeated.

The generic-text window must include its independent passing validation
artifact. The evaluator binds both files into the run identity; it will not
silently fall back to the historical reasoning-task selector.

The command above starts a fresh evaluation. Add `--resume` only after that
evaluation output directory contains its existing `run.json` and pair
checkpoints.

The report includes clean/typo accuracy, wrong-to-right, right-to-wrong, net
accuracy, clean harm, multi-token KL, additional paired-patch gain,
tokenization strata, unseen tasks, unseen operations, held-out corpus PPL and
Base-to-adapter clean KL, and natural typos. The
frozen pre-PR gate requires a primary random-2 typo improvement of at least +2
accuracy points over Base with a 95% confidence-interval lower bound above
zero, a clean macro decrease of at most 1 point, no task decrease of 3 points
or more, a held-out clean perplexity ratio at most 1.02, clean forward KL at
most 0.03 nats/token, and no material natural typo accuracy or held-out-pair KL
degradation. The mechanistic patch audit is reported but is not a blocking gate.
Confirmatory claims additionally require three training seeds; failed attempts
are retained locally and summarized in the eventual PR.

The full frozen protocol is in
[`../typo-cot/docs/robustness_training_plan_v1.md`](../typo-cot/docs/robustness_training_plan_v1.md).

## 7. Run the parallel SAE diagnostic track (GPU 0 only)

The SAE track is diagnostic/future-study work. It must not modify or interrupt
the protected GPU 5/6 output-matching and causal-window runs. If frozen
robustness evaluation and this track compete for resources, stop scheduling new
SAE work and give the frozen evaluation priority.

First extend the remaining clean FineWeb-Edu stream to the preregistered source
budget. The builder requires all frozen evaluation and localization roles,
rejects exact identity/content overlap, and applies the frozen character-5gram
near-duplicate check. This is a separate data-preparation command and uses no
GPU:

This section cannot run from repository artifacts alone. Before executing it,
an operator must provision an absolute external artifact directory as
`SAE_ROOT` and supply a separately reviewed absolute `ROOTED_REGISTRY` that
contains `wp2_project_root` and `wp2_project_root_identity`. The checked-in
legacy registry-v1 is evidence only. Mixing it with the reviewed registry
changes the bound preregistration SHA, so validation rejects that training run.
Every block fails before data preparation or GPU execution when its external
contract or command-specific preceding artifact is invalid or missing. Each SAE
command block below runs in its own subshell: GPU 0 and the SAE W&B project are
local to that block, every `exit 2` exits only the subshell, and the caller's
`GPU_ID`, `WANDB_PROJECT`, and `EVALUATION_DATA` remain unchanged.

```bash
(
: "${SAE_ROOT:?Set SAE_ROOT to a provisioned absolute external artifact directory}"
: "${ROOTED_REGISTRY:?Set ROOTED_REGISTRY to the separately reviewed absolute registry path}"
case "${SAE_ROOT}" in /*) ;; *) echo "SAE_ROOT must be absolute" >&2; exit 2 ;; esac
case "${ROOTED_REGISTRY}" in /*) ;; *) echo "ROOTED_REGISTRY must be absolute" >&2; exit 2 ;; esac
[ -d "${SAE_ROOT}" ] || { echo "SAE_ROOT does not exist: ${SAE_ROOT}" >&2; exit 2; }
[ -f "${ROOTED_REGISTRY}" ] || { echo "reviewed ROOTED_REGISTRY does not exist: ${ROOTED_REGISTRY}" >&2; exit 2; }

SAE_TRAINING_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m/training_sources.jsonl"
SAE_EVALUATION_DATA="${TRAIN_ROOT}/evaluation-data/robustness-v1"
SAE_LOCALIZATION_DATA="${TRAIN_ROOT}/localization/generic-joint-window-v1/pairs"
for REQUIRED_INPUT in "${SAE_TRAINING_DATA}" "${SAE_EVALUATION_DATA}" "${SAE_LOCALIZATION_DATA}"; do
  [ -e "${REQUIRED_INPUT}" ] || { echo "required prior artifact does not exist: ${REQUIRED_INPUT}" >&2; exit 2; }
done

uv run --project "${TRAIN_PROJECT}" --locked typo-cot build-sae-clean-corpus \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${ROOTED_REGISTRY}" \
  --existing-data "${SAE_TRAINING_DATA}" \
  --exclude-data "${SAE_EVALUATION_DATA}" \
  --exclude-data "${SAE_LOCALIZATION_DATA}" \
  --training-budget minimum \
  --output-dir "${SAE_ROOT}/clean-corpus"
)
```

Then calibrate the three preregistered L1 coefficients on the same combined
clean stream:

```bash
(
: "${SAE_ROOT:?Set SAE_ROOT to a provisioned absolute external artifact directory}"
: "${ROOTED_REGISTRY:?Set ROOTED_REGISTRY to the separately reviewed absolute registry path}"
case "${SAE_ROOT}" in /*) ;; *) echo "SAE_ROOT must be absolute" >&2; exit 2 ;; esac
case "${ROOTED_REGISTRY}" in /*) ;; *) echo "ROOTED_REGISTRY must be absolute" >&2; exit 2 ;; esac
[ -d "${SAE_ROOT}" ] || { echo "SAE_ROOT does not exist: ${SAE_ROOT}" >&2; exit 2; }
[ -f "${ROOTED_REGISTRY}" ] || { echo "reviewed ROOTED_REGISTRY does not exist: ${ROOTED_REGISTRY}" >&2; exit 2; }
SAE_GPU_ID="0"
SAE_WANDB_PROJECT="typo-robustness-sae"
SAE_TRAINING_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m/training_sources.jsonl"
SAE_SUPPLEMENT_DATA="${SAE_ROOT}/clean-corpus/sae_clean_supplement.jsonl"
for REQUIRED_INPUT in "${SAE_TRAINING_DATA}" "${SAE_SUPPLEMENT_DATA}"; do
  [ -f "${REQUIRED_INPUT}" ] || { echo "required SAE input file does not exist: ${REQUIRED_INPUT}" >&2; exit 2; }
done
CUDA_VISIBLE_DEVICES="${SAE_GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot calibrate-sparse-autoencoder-l1 \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${ROOTED_REGISTRY}" \
  --training-data "${SAE_TRAINING_DATA}" \
  --training-data "${SAE_SUPPLEMENT_DATA}" \
  --gpu-id "${SAE_GPU_ID}" \
  --wandb-project "${SAE_WANDB_PROJECT}" \
  --output-dir "${SAE_ROOT}/l1-calibration"
)
```

Train the two layer-5 initialization seeds and the layer-20 SAE. The
command refuses typo/task records and writes the eligible record-ID SHA-256
before the first model forward. Add another decontaminated clean FineWeb-Edu
manifest by repeating `--training-data` when needed to reach at least 100M
unique source tokens.

```bash
(
: "${SAE_ROOT:?Set SAE_ROOT to a provisioned absolute external artifact directory}"
: "${ROOTED_REGISTRY:?Set ROOTED_REGISTRY to the separately reviewed absolute registry path}"
case "${SAE_ROOT}" in /*) ;; *) echo "SAE_ROOT must be absolute" >&2; exit 2 ;; esac
case "${ROOTED_REGISTRY}" in /*) ;; *) echo "ROOTED_REGISTRY must be absolute" >&2; exit 2 ;; esac
[ -d "${SAE_ROOT}" ] || { echo "SAE_ROOT does not exist: ${SAE_ROOT}" >&2; exit 2; }
[ -f "${ROOTED_REGISTRY}" ] || { echo "reviewed ROOTED_REGISTRY does not exist: ${ROOTED_REGISTRY}" >&2; exit 2; }
SAE_GPU_ID="0"
SAE_WANDB_PROJECT="typo-robustness-sae"
SAE_TRAINING_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m/training_sources.jsonl"
SAE_SUPPLEMENT_DATA="${SAE_ROOT}/clean-corpus/sae_clean_supplement.jsonl"
SAE_L1_SELECTION="${SAE_ROOT}/l1-calibration/l1_selection.json"
for REQUIRED_INPUT in "${SAE_TRAINING_DATA}" "${SAE_SUPPLEMENT_DATA}" "${SAE_L1_SELECTION}"; do
  [ -f "${REQUIRED_INPUT}" ] || { echo "required SAE input file does not exist: ${REQUIRED_INPUT}" >&2; exit 2; }
done
CUDA_VISIBLE_DEVICES="${SAE_GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-sparse-autoencoders \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${ROOTED_REGISTRY}" \
  --training-data "${SAE_TRAINING_DATA}" \
  --training-data "${SAE_SUPPLEMENT_DATA}" \
  --l1-selection "${SAE_L1_SELECTION}" \
  --gpu-id "${SAE_GPU_ID}" \
  --wandb-project "${SAE_WANDB_PROJECT}" \
  --output-dir "${SAE_ROOT}/training"
)
```

The frozen 10M-token activation subsample stores four bfloat16 residual streams
and requires about 205 GB of disk. A 1M-token shuffle buffer can temporarily use
more than 41 GB of host RAM while it is joined and permuted. Before starting,
the operator must verify that the provisioned `SAE_ROOT` has at least 220 GB
free and the host has at least 48 GB available RAM.

Finally compute held-in firing probabilities, reconstruction-error scale, and
the frozen WP-2 acceptance report. This command evaluates only clean LM data;
it does not run task accuracy or open any evaluation tier.

```bash
(
: "${SAE_ROOT:?Set SAE_ROOT to a provisioned absolute external artifact directory}"
: "${ROOTED_REGISTRY:?Set ROOTED_REGISTRY to the separately reviewed absolute registry path}"
case "${SAE_ROOT}" in /*) ;; *) echo "SAE_ROOT must be absolute" >&2; exit 2 ;; esac
case "${ROOTED_REGISTRY}" in /*) ;; *) echo "ROOTED_REGISTRY must be absolute" >&2; exit 2 ;; esac
[ -d "${SAE_ROOT}" ] || { echo "SAE_ROOT does not exist: ${SAE_ROOT}" >&2; exit 2; }
[ -f "${ROOTED_REGISTRY}" ] || { echo "reviewed ROOTED_REGISTRY does not exist: ${ROOTED_REGISTRY}" >&2; exit 2; }
SAE_GPU_ID="0"
SAE_TRAINING_DATA="${TRAIN_ROOT}/data/gemma4b-cycle3-64m/training_sources.jsonl"
SAE_SUPPLEMENT_DATA="${SAE_ROOT}/clean-corpus/sae_clean_supplement.jsonl"
SAE_CHECKPOINT_DIR="${SAE_ROOT}/training"
for REQUIRED_INPUT in "${SAE_TRAINING_DATA}" "${SAE_SUPPLEMENT_DATA}"; do
  [ -f "${REQUIRED_INPUT}" ] || { echo "required SAE input file does not exist: ${REQUIRED_INPUT}" >&2; exit 2; }
done
[ -d "${SAE_CHECKPOINT_DIR}" ] || { echo "SAE checkpoint directory does not exist: ${SAE_CHECKPOINT_DIR}" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="${SAE_GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot validate-sparse-autoencoders \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${ROOTED_REGISTRY}" \
  --validation-data "${SAE_TRAINING_DATA}" \
  --validation-data "${SAE_SUPPLEMENT_DATA}" \
  --checkpoint-dir "${SAE_CHECKPOINT_DIR}" \
  --gpu-id "${SAE_GPU_ID}" \
  --output-dir "${SAE_ROOT}/validation"
)
```

The first WP-2 validation reserves its attempt with an exclusive project-root
file before creating output or initializing GPU/runtime work, then records its
immutable failed-bundle lineage in `${SAE_ROOT}/wp2_attempts.json`. A newly
reviewed initial v1 preregistration must declare an absolute `wp2_project_root`
and its `wp2_project_root_identity` (`device`/`inode`), and that directory must
already exist with the registered identity. This is deliberately a
machine-local execution contract, just like the absolute path. The checked-in
legacy v1 preregistration
is intentionally byte-identical to the artifact used by the completed initial
run: it remains readable as retry-lineage evidence, but cannot start another
initial validation. The exclusive record is file-fsynced and its parent
directory is then fsynced; a crash after reservation consumes that attempt and
there is no implicit validation resume.

A retry remains prohibited until a separately reviewed authorization binds the
original config, preregistration, training, validation, acceptance, and ledger
hashes to exactly one amended config and preregistration hash. The amended
preregistration uses schema v2 and contains the same absolute
`wp2_project_root` plus a `wp2_retry` lineage marker. Its
`initial_attempt_ledger_path` must be exactly
`wp2_project_root/wp2_attempts.json`. That reviewed marker, not a CLI option,
automatically and unavoidably selects retry mode for both training and
validation; the authorization in turn binds the full v2 preregistration
SHA-256. CLI omission therefore cannot fall back to an initial validation path.
The initial validation output itself may be outside the project root because
the ledger binds its absolute path and the authorization binds every artifact
hash; relocating or copying that output does not create a new retry budget.

Retry training creates `${SAE_ROOT}/wp2_retry_claim.json` with an exclusive
filesystem claim before runtime or model initialization. Exact resume compares
that claim before creating or rewriting any output artifact, including
`source_registry.json`. The claim binds the ordered manifest paths and raw
hashes, the canonical values actually loaded into the reserved/eligible source
stream, the loaded layer-to-L1 mapping, the reviewed implementation closure,
and the effective W&B project/entity. The project root also holds a nonblocking
process-lifetime lock, so only one fresh/resumed retry can reach runtime.
Exclusive claim/reservation/completion records open the already canonical,
pre-existing parent path with `O_DIRECTORY|O_NOFOLLOW`; their literal final basename is created with
`O_EXCL|O_NOFOLLOW`, so a final-component symlink cannot redirect or preserve a
budget slot. The reviewed preregistration bytes pin the project root's
`st_dev/st_ino` across invocations;
the lease and every authority read/write compare the opened directory against
that identity. A renamed root replaced by either a symlink or a new regular
directory therefore fails closed.
Training and validation output paths likewise reject final symlinks.
Existing claim/reservation/completion authorities are read through the same
anchored parent with a nonblocking `O_NOFOLLOW` descriptor and must be
same-owner, singly linked regular files; equal-payload aliases are not accepted.
On retry `--resume`, the training-completion authority is checked before
runtime/model initialization. If its exact hash-bound run and model artifacts
still exist, the completed result is returned without training; if any recorded
output is missing or differs, resume fails closed and cannot spend the retry again.

If the process dies after the claim but before a training checkpoint, exact
`--resume` is allowed only when the output is absent/empty or contains the exact
canonical source registry (plus an abandoned atomic checkpoint temporary, or a
source-registry temporary with the product's canonical positive ASCII PID name,
same-user singly linked regular non-symlink type, and bytes that prefix the
expected canonical registry).
Source-registry temporaries are non-authoritative
and are never loaded as the registry during recovery.
Every other orphan artifact fails closed. After runtime, model, and optimizer
initialization, training fsyncs an atomic cursor-zero checkpoint before W&B,
activation collection, or an optimizer step. W&B likewise persists a
claim-derived run-ID intent before `wandb.init`, so recovery reuses that identity.
Ordinary non-retry resume still requires its pre-existing checkpoint.

Retry validation reserves its sole slot before any output
or GPU/runtime work, accepts only the exact claimed training `run.json`, and
rechecks that SHA immediately before model loading and again before recording
completion. Its immutable completion records attempt 2, pass/fail, the parent
ledger, training run, reservation, validation, and acceptance hashes. Changing
an output parent cannot reset either project-global reservation. This repository
does not yet contain a scientific retry authorization, v2 retry registry, or
amended retry values; the lineage mechanism does not itself authorize a retrain.

Append `--resume` to a calibration/training command only after its own
hash-bound checkpoint exists, except for the exact reviewed retry claim-only
recovery described above. The frozen method and gates are documented in
[`docs/sae_track_plan_v1.md`](docs/sae_track_plan_v1.md).
