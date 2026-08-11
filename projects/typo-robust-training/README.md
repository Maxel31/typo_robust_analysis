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

## 3. Causally localize neurons and attention heads

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
clean-harm rule.

## 4. Train separate baselines and the proposed adapter

```bash
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
```

Repeat every condition with seeds 42, 43, and 44. The teacher receives clean
input, stays frozen, and is never activation-patched. The student receives typo
input; only declared LoRA parameters may change. All conditions share token
accounting, checkpoint/resume, and clean-preservation evaluation.

## 5. Evaluate held-out robustness

```bash
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

The report includes clean/typo accuracy, wrong-to-right, right-to-wrong, net
accuracy, clean harm, multi-token KL, additional paired-patch gain,
tokenization strata, unseen tasks, unseen operations, and natural typos. The
minimum pre-PR gate is +3 typo-accuracy points, at most -1 clean-accuracy point,
wrong-to-right greater than right-to-wrong, positive unseen transfer, and the
same improvement direction for at least two of seeds 42/43/44. Failed attempts
are retained locally and summarized in the eventual PR.

The full frozen protocol is in
[`../typo-cot/docs/robustness_training_plan_v1.md`](../typo-cot/docs/robustness_training_plan_v1.md).
