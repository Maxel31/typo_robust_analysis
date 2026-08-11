# Typo-robustness training plan v1

Status: **prospective, interface-frozen, and not a result of the submitted
paper**.

Working method name: **intervention-guided localized state distillation**.
This track asks whether a model can learn to process typo input without needing
a paired clean run at inference. The submitted paper supplies diagnostic
evidence about useful edited-word coordinates; it does not establish that the
training method works.

The development order is deliberately hierarchical:

```text
layer -> component -> training
```

First locate a layer window on diagnostic data, then causally validate MLP
neuron and attention head candidates inside that window, and only then
concentrate the state loss on components that replicate across tasks. The first
implementation updates layer-scoped LoRA adapters; it does not directly mutate
only the selected weight rows.

## Pull-request evidence gate

There will be **no training pull request** merely because the trainer runs or
the loss decreases. Training code, configs, checkpoints, and iteration logs
remain on a local descriptive branch until a frozen held-out evaluation shows
typo robustness above the base model while preserving clean performance. Failed
attempts remain in local run records and must be summarized when the eventual
PR is opened.

Development may tune only on the train/tune splits. Once a candidate passes the
tune gates, its config and checkpoint hash are frozen and evaluated on the
untouched pre-PR gate split. If that check fails, the failed gate result is
retained; a new protocol version must predeclare another untouched gate split
before further tuning. The final paper-test partitions remain untouched until
the method and all comparison conditions are frozen.

The minimum pre-PR evidence is:

- typo accuracy at least 3 percentage points above Base;
- clean accuracy loss no larger than 1 percentage point;
- wrong-to-right greater than right-to-wrong;
- positive mean transfer on unseen tasks and unseen typo conditions;
- at least two of three seeds (42, 43, 44) agree in direction;
- no data-integrity, gradient-scope, resume, or leakage test failure.

These are engineering progression gates, not significance thresholds or a
license to omit negative results.

## Environment boundary

Rebuttal inference keeps the submitted-paper environment unchanged. Training
uses a separately locked project at `projects/typo-robust-training`, whose
environment adds PEFT and training-only dependencies while consuming the local
`typo-cot` package. The rebuttal lock must not be refreshed as a side effect of
training setup.

The initial model is `google/gemma-3-4b-it`. Llama-3.2-3B and Mistral-7B are
extensions only after the Gemma pilot passes the pre-PR evidence gate. All GPU
validation for this implementation uses physical GPU 1 and records logical
`cuda:0`, `CUDA_VISIBLE_DEVICES=1`, package versions, peak VRAM, throughput,
tokens, wall time, and GPU-hours.

## Dataset roles and licensing checks

Dataset diversity is part of the hypothesis, not an afterthought. Each source
has one prespecified role.

| Source | Role before the MVP gate | Isolation rule |
|---|---|---|
| FineWeb-Edu | Primary broad clean corpus for sanity and MVP training | Documents are split by stable content ID before sampling; held-out documents never enter training |
| GSM8K train, MMLU public train/dev, ARC-Challenge train | 10--20% reasoning/instruction mixture and layer/component diagnostics | Official test portions are excluded; sample IDs are disjoint from evaluation |
| GitHub Typo Corpus | Small natural-pair mixture, empirical operation statistics, and natural-typo evaluation | Split by repository before extracting edits; adjacent transpositions are removed from train/tune and train-derived statistics; held-out repositories cannot influence the generator |
| Dolma | Unseen-domain evaluation for the initial MVP | No Dolma document enters MVP training or tuning |
| CoT Collection | Optional reasoning supplement after a license/terms review | Disabled by default; evaluation-task families and near duplicates are removed |

FineWeb-Edu is chosen because its released 1.3T-token educational subset is
broad enough to sample without downloading the full corpus and its published
evaluation includes knowledge/reasoning-intensive benchmarks. Dolma provides a
different mixture of web, papers, code, books, social material, and
encyclopedic text, making it useful as an unseen-domain check rather than a
second source silently mixed into the pilot.

GitHub Typo Corpus contains real correction histories but is GitHub-domain
biased. Its English records are partitioned by repository: 70% generator/train,
10% tune, and 20% held-out natural typo. At most 5% of MVP training sequences
may be natural clean/typo pairs from train repositories. Only the train split
may estimate operation, character, or edit-distance frequencies. Tune and test
repositories remain opaque to generation.

Adjacent transpositions are excluded from every training and tuning source,
including the natural-pair mixture, typo-frequency estimation, generator tuning,
and optional reasoning data. They may occur only in untouched gate/test records,
where synthetic and natural-transposition results are reported separately. This
keeps the v1 unseen-operation label valid rather than limiting it to the
synthetic source.

The CoT Collection is not enabled in the default reproducible configuration:
its repository describes non-commercial/terms restrictions that require an
explicit use review. If enabled in a labelled research-only run, the exact
dataset revision, approved terms, task denylist, and retained IDs must be
recorded. The benchmark train/dev mixture remains the default reasoning source.

Dataset and paper references used to fix these roles:

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Dolma paper](https://arxiv.org/abs/2402.00159)
- [GitHub Typo Corpus paper](https://aclanthology.org/2020.lrec-1.835/)
- [CoT Collection paper](https://aclanthology.org/2023.emnlp-main.782/)
- [Kojima et al. noisy-text continual pre-training](https://aclanthology.org/2025.tacl-1.38/)

Every downloaded dataset uses a pinned revision when the provider exposes one.
The run manifest records license metadata, source revision, split, streaming
shuffle parameters, raw IDs, and content hashes. No downloaded corpus or
generated training text is committed.

## Data stages and token budgets

`build-robustness-training-data` creates split manifests and fixed evaluation
pairs; training perturbations remain on-the-fly.

| Stage | Token budget | Training mixture | Purpose |
|---|---:|---|---|
| Unit fixture | under 10k | hand-authored licensed fixtures | alignment, masks, and deterministic replay |
| Sanity | 1--2M | 85% FineWeb-Edu, 10% benchmark train/dev, 5% natural train repositories | verify loss, gradient, memory, and throughput |
| MVP | 10M | 80% FineWeb-Edu, 15% benchmark train/dev, up to 5% natural train repositories | compare Base, three baselines, and Proposed |
| Main | 50--100M | fixed only after the MVP gate | three-seed transfer experiment |
| Scale | 100--300M | fixed only after Main evidence | model/domain extension |

Mixture percentages are measured by non-padding student tokens. Each training
batch also contains the teacher's clean counterpart, and the student performs a
clean-preservation pass. In addition, 10% of scheduled student examples are
explicit no-op clean pairs. Thus clean examples remain present even when every
noisy pair has a clean teacher.

The first 100 sanity steps record throughput, VRAM, every loss component,
gradient norms by layer/module, and the effective non-padding token count.
Those measurements determine the 50M-token GPU-hour estimate; they do not
authorize scale-up before the MVP gate.

## Leakage-resistant split construction

Generalization is evaluated along three independent axes:

- **unseen content**: documents and near-duplicate clusters absent from train;
- **unseen task**: MMLU-Pro, MATH-500, and CommonsenseQA are excluded from
  localization, generator tuning, and training;
- **unseen typo**: held-out natural repositories and typo operations absent
  from training.

All source text is normalized only for duplicate detection, then assigned by a
stable content identifier. Exact hashes and MinHash/character-shingle near-
duplicate groups are split atomically. Evaluation prompts and answers form a
denylist against FineWeb-Edu and optional reasoning data. The decontamination
report records exact matches, near matches, excluded task families, and counts
before and after filtering.

GSM8K, MMLU, and ARC diagnostic records use only their public train/development
portions. Their tests provide same-task unseen evaluation. MMLU-Pro, MATH-500,
and CommonsenseQA provide unseen-task evaluation and never select layers,
components, hyperparameters, or stopping steps.

## Synthetic and natural typo construction

Training perturbations operate on eligible words containing alphabetic
characters, at least two letters, and no URL, email, code-like identifier, or
special-symbol span. Target selection mixes uniform eligible words and a fixed
content-word preference; AttnLRP is not run over the web corpus.

The training generator supports:

- keyboard-neighbor substitution;
- deletion;
- insertion of an adjacent-key or repeated character;
- duplication;
- a general substitution sampled from train-only natural statistics.

One, two, and three-to-four edits have probabilities 0.50, 0.30, and 0.20.
Edits target distinct original words unless a record contains too few eligible
words. The paper's substitution/deletion/duplication family is always
represented; insertion broadens training diversity. Adjacent transposition is
held out as the synthetic unseen-operation test in v1.

The generator emits both clean and typo character spans and retokenizes both
sides. It records every clean/typo token span plus first/final subword indices.
For state supervision, the default coordinate is the word-final token. Invalid
or ambiguous alignment is rejected rather than guessed in the primary data.

Training edits change on each epoch using a counter-based per-record RNG keyed
by `(global seed, epoch, source ID)`. Worker count and resume boundaries cannot
change the sampled sequence. Validation, gate, and test typo manifests are
materialized once and content-hashed.

## Layer selection

`select-distillation-layers` uses 200--300 diagnostic examples per task from
GSM8K, MMLU, and ARC-Challenge train/dev. It runs every single-layer edited-word
patch and computes multi-token KL restoration, answer restoration, and harm.
Before any patch outcome is read, a record is KL-eligible only when its finite
untreated mean `KL(clean || typo)` over tokens 2--16 is greater than **1e-6 nats**.
Eligibility is computed once from unpatched logits and is identical for every
layer. `R_KL_2:16,l` is the ratio of cohort-level mean patched and untreated KL,
not a mean of per-record ratios. Ineligible records remain in answer/harm terms
and in the audit output but never receive a normalized KL value. A task with
fewer than 50 KL-eligible records or less than 80% eligibility makes selection
fail closed; no epsilon substitution or fallback layer metric is allowed.
This `1e-6` selection threshold is intentionally stricter than the rebuttal
readout's `1e-9` finite-denominator guard. The rebuttal estimates a fixed
prespecified intervention and keeps small finite effects visible; layer
selection is adaptive and determines every downstream training target, so it
screens near-degenerate records whose normalized score could steer selection.
Implementations must preserve these role-specific thresholds rather than
silently harmonizing them.
For layer `l`:

```text
s_l = R_KL_2:16,l + beta * R_answer,l - gamma * H_l
```

`R_answer,l` is the patched-correct rate among records whose unpatched clean
answer is correct and unpatched typo answer is wrong. `H_l` is the
patched-wrong rate among records whose clean and unpatched typo answers are
both correct. Each answer cohort must contain at least ten records per task;
otherwise selection fails closed. The default pilot freezes `beta=0.5` and
`gamma=1.0`, so one point of right-to-wrong harm is never worth less than the
same answer-restoration gain. A width-six candidate receives the arithmetic
mean of its six layer scores. Tasks are equally weighted, and ties are resolved
by the smallest window start. Bootstrap resampling is stratified within task by
KL eligibility and the two unpatched answer-cohort labels, preserving every
scientific denominator.

`beta`, `gamma`, the contiguous-window width, task macro-averaging, and tie
breaking are fixed in config before outputs are inspected. The selected window
must be written to `layer_selection.json` with per-task scores and bootstrap
uncertainty. `[0,6)` is a Gemma pilot candidate from the paper, not a universal
default for other models.

## Component localization

`localize-robustness-components` examines only the selected layers and edited-
word positions. It first screens candidates by clean/typo activation difference
and gradient-based attribution, then causally patches the shortlisted MLP
neuron activations and attention head outputs. Screening alone never labels a
component causal.

Candidates receive answer-restoration and multi-token-KL restoration scores on
each diagnostic task. A component is eligible for training only if it has the
same beneficial direction on at least two tasks and no prespecified clean-harm
violation. The output records the complete candidate universe, shortlist rule,
causal validation denominator, task scores, selected set, and normalized
non-negative component weights.

The selected component set controls where state loss is measured. LoRA remains
on the containing selected layers because one SwiGLU MLP channel spans
`gate_proj`, `up_proj`, and `down_proj`, and direct row-only updates would
confound localization with a more fragile optimizer. Direct component-weight
updates and sparse-feature/SAE supervision are separately labelled later
ablations, used only if neuron/head selection proves unstable.

## Teacher, student, and trainable scope

The teacher is the frozen base model on clean input. It performs no activation
patch and receives no gradient. The student starts from the same base model,
sees typo input, and trains LoRA adapters. The default targeted adapters cover
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`
in selected layers with rank 16, alpha 32, and dropout 0.05.

Comparisons separate the state-measurement scope from parameter placement:

| Condition | State loss | LoRA placement |
|---|---|---|
| Targeted component | selected neurons/heads | selected containing layers |
| Targeted layer | all hidden dimensions at edited positions | selected layers |
| All-layer adapter | selected components | all layers, rank matched by trainable count |
| Middle-control | selected components | frozen-width middle layers |
| Output-only | none | selected layers |

All base weights stay frozen. Tests must prove that teacher parameters and
student base weights have no gradients and only intended adapter parameters
change.

## Losses

The total loss is:

```text
L = lambda_answer * L_answer
  + lambda_output * L_output
  + lambda_state * L_state
  + lambda_clean * L_clean
```

`L_answer` is answer-token cross entropy on reasoning examples only.
`L_output` matches clean-teacher and typo-student next-token distributions on
aligned non-edited positions; tokens belonging to a noised word are excluded
from optimization. `L_clean` matches the base teacher and student on clean
input. `L_state` compares selected edited-position activations:

```text
L_state = sum_(l,j,t in S) w_lj * distance(a_student_typo, stopgrad(a_teacher_clean))
```

Cosine distance is the layer-level default; component-level MLP activations use
normalized squared error unless the frozen config selects cosine. Initial
weights are answer 1.0, output 1.0, state 0.5, and clean 0.5. Only
`lambda_state` in {0.1, 0.5, 1.0} and `lambda_clean` in {0.25, 0.5, 1.0} form
the first prespecified sweep.

The optimizer starts with bf16, sequence length 512, AdamW, learning rate
2e-4, weight decay 0.01, 3% warmup, cosine decay, gradient checkpointing,
micro-batch 1--2, global batch 32--64, and three seeds 42/43/44. Any change is
versioned and justified by tune metrics, not test outcomes.

## Separate training commands and baselines

Each scientific condition has a separate public command:

1. `build-robustness-training-data`
2. `select-distillation-layers`
3. `localize-robustness-components`
4. `train-noisy-language-model`
5. `train-output-matching`
6. `train-global-state-alignment`
7. `train-localized-state-distillation`
8. `evaluate-typo-robustness`

`train-noisy-language-model` performs ordinary noisy next-token training and
also exposes the separately labelled Kojima-style noised-word mask plus word-
distribution matching variant. `train-output-matching` has no state loss.
`train-global-state-alignment` aligns all configured layers/tokens.
`train-localized-state-distillation` is Proposed and consumes immutable layer
and component selection artifacts. Training commands share checkpoint/resume,
token accounting, logging, and gradient-scope validation but never infer their
scientific condition from an output path.

Required ablations remove state, output, or clean loss; use random/middle/late
layers; use random positions; align all tokens; compare targeted/all-layer LoRA;
and train at one/two/four typo intensities. Random selections are materialized
before training and paired across seeds.

## Evaluation

`evaluate-typo-robustness` evaluates Base, noisy LM, output matching, global
state alignment, and localized state distillation with identical decoding and
cohorts. It reports:

- clean and typo accuracy;
- wrong-to-right and right-to-wrong transitions;
- net accuracy change and clean-input harm;
- clean/typo multi-token KL gap;
- additional patch gain from the frozen paired `[0,6)` audit;
- tokenization and edit-count strata;
- unseen content, unseen task, unseen typo operation, and natural typo results;
- three-seed estimates, intervals, training tokens, GPU-hours, and trainable
  parameter counts.

An accepted model should reduce additional patch gain by at least 30% relative
to Base, indicating less dependence on the diagnostic clean-state transplant.
This is secondary to the accuracy and harm gates and cannot rescue a model that
damages clean performance.

The paired patch audit never mixes model states. For Base, both the clean donor
run and typo recipient run use the frozen base model with no adapter. For every
trained condition, both runs use the same evaluated checkpoint with that
condition's adapters enabled; the clean donor activation is captured from that
adapted model and written into its own adapted typo run. Adapter-disabled or
base-donor-to-adapted-recipient variants are optional diagnostics, are labelled
separately, and cannot satisfy the 30% acceptance gate.

## Test requirements

Before any long training run, tests cover one-token/multi-subword alignment,
token-count increase/decrease, punctuation adjacency, left padding, deterministic
on-the-fly replay, resume sample order, and held-out split disjointness. Loss
tests cover clean-equals-typo near-zero state loss, edited-position masks,
component masks, noised-word output masking, and zero teacher gradients.
Parameter tests cover exact LoRA layer/module placement and frozen base weights.

GPU smoke tests use physical GPU 1 only. A 100-pair hidden-state extraction and
100-step sanity train must demonstrate finite losses, non-zero adapter
gradients, unchanged teacher/base parameters, a byte-valid resumable
checkpoint, and reproducible post-resume sample order before the MVP run.
