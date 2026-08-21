# Typo-robustness evaluation protocol v1.4

Status: **endpoints frozen before training cycle 2; v1.4 is a pre-opening
executable-scope amendment with no model-output access**. It removes declarations
that had no implementation in the frozen evaluator; the confirmatory endpoints
and all implemented secondary endpoints are unchanged.

This document is the evaluation contract for typo-robustness training. Training
losses, data mixtures, adapter placement, and stopping rules may change between
cycles. The item identities, realized typo texts, prompts, decoding, answer
extraction, endpoints, statistics, thresholds, and opening rules below may not.
Any scientific change requires a new protocol version and a complete rerun of
Base and every compared adapter under that version.

## 1. Confirmatory estimands

Every task item is evaluated as the same paired 2 by 2 matrix:

| | clean input | frozen typo input |
|---|---:|---:|
| Base | `B_clean` | `B_typo` |
| Adapter | `A_clean` | `A_typo` |

The only two confirmatory endpoints are:

1. clean non-inferiority, `Delta_clean = acc(A_clean) - acc(B_clean)`;
2. typo superiority, `Delta_typo = acc(A_typo) - acc(B_typo)` under the frozen
   random two-edit condition.

Both must pass. A reduced clean--typo gap is never sufficient because lowering
clean accuracy also reduces that gap. Every comparison reports the full paired
transition table (wrong-to-right, right-to-wrong, wrong-to-wrong, and
right-to-right); net accuracy is never reported alone. The confirmatory sample
is an unconditional fixed population sample, not a Base-selected
clean-correct/typo-wrong cohort. Such a flip cohort is secondary only.

## 2. Data tiers and opening rules

| Tier | Purpose | Access |
|---|---|---|
| monitor | training-side clean/output KL and numerical safety | schedule is owned by the training run; this evaluation contract only forbids task accuracy |
| tune | cycle and ablation selection | repeatable; all accesses logged; known to be optimizable |
| pre-PR gate | one confirmatory check of a frozen candidate | opened once after arm, three seeds, config, and checkpoint hashes are committed |
| final test | paper headline result | opened exactly once after every model, arm, seed, and analysis decision is frozen and the pre-PR gate passed |

The final test is not opened once per model or once per cycle. All confirmatory
arms are evaluated in the same opening. A failed pre-PR gate is retained; a
later confirmatory attempt requires a prospectively declared protocol version
and a new untouched gate pool. Resume of the exact hash-bound run is allowed
and is not a new opening.

The tune task pool contains 500 records from public training/development
portions, disjoint from localization and training. It does not contain
MMLU-Pro, MATH-500, or CommonsenseQA. It is accompanied by 200 held-out
FineWeb-Edu documents, 100 held-out natural clean/typo language-model pairs,
and 100 natural-injection task pairs. The latter two use the same tune-only
repository/word partition but are distinct evaluation rows.

The pre-PR gate contains 500 records each from GSM8K, MMLU, ARC-Challenge,
MMLU-Pro, and CommonsenseQA (2,500 task items). MATH-500 is reserved for the
final test because its released test set has only 500 records.

The final test contains 500 records each from GSM8K, MMLU, ARC-Challenge,
MMLU-Pro, and CommonsenseQA, plus all 440 MATH-500 records that support the
unchanged four-distinct-word severity condition (2,940 task items). MMLU and
MMLU-Pro are stratified by subject/category; MATH-500 is stratified by subject
and level.
Selections are deterministic from source ID, namespace, and seed 42. All pools
are mutually disjoint and exclude localization IDs, training IDs, near-
duplicate groups, and the cycle-1 pilot records.

## 3. Frozen typo conditions

The primary condition is `random-2`: exactly two distinct eligible question
words, uniformly selected without attribution, receive operations drawn from
the submitted paper's Table 4 family (QWERTY-neighbor substitution, deletion,
or duplication). One realized typo text per item is frozen with seed 42.

Secondary conditions are:

- `random-1` and `random-4`, on a prespecified stratified 1,200-item subset, for
  a severity curve;
- `transposition-2`, on 500 items, as an edit-operation held out from training;
- `natural-injection`, one real misspelling on 500 task items, using an evaluation-only dictionary
  derived from held-out licensed GitHub repositories;
- held-out natural clean/typo language-model pairs (1,000 in final test).

Secondary conditions cannot replace or be pooled into the confirmatory
`random-2` endpoint. Model-specific Attribution-4 remains a possible separately
preregistered stress diagnostic, but it is not declared by this model-independent
frozen battery and cannot affect its gates.

Only the question is edited; few-shot demonstrations remain clean. Eligible
targets are alphabetic words of at least three letters. URL, email, identifier,
number, mathematical-expression, option-label, and gold-answer spans are
ineligible. For multiple-choice tasks the answer-option text itself is not
edited. If an item has too few eligible distinct words, it is replaced by the
next item in the deterministic order before the item list is frozen; evaluation
never silently changes its denominator.

Every realized pair is serialized as canonical JSONL. The registry records its
SHA-256 digest, source revision, generator version, condition, seed, item count,
and coverage. The frozen realized text is the canonical evaluation artifact and
is never regenerated due to a library or environment update.

## 4. Generation and extraction

The submitted-paper protocol is reused: task-specific clean few-shot CoT
templates, greedy decoding, bfloat16 inference, at most 512 new tokens, and the
same task-specific answer extractors with deterministic fallback. GSM8K uses
8-shot; MMLU, MMLU-Pro, ARC-Challenge, and CommonsenseQA use 5-shot; MATH-500
uses 4-shot. Unextractable output is incorrect in every arm and input
condition. Chat template and effective EOS handling are fixed per Base model
and shared by all of its adapters.

Before every evaluation, a no-op regression must show that perturbation
severity zero is byte-identical to the clean question. Base and all adapter
seeds read the identical frozen pair files.

## 5. Corpus preservation and natural typo checks

The final clean-preservation battery also contains 1,000 group-disjoint
FineWeb-Edu documents and 1,000 unseen-domain Dolma documents. Report
teacher-forced PPL ratio `PPL(Adapter) / PPL(Base)` and clean forward
`KL(Base || Adapter)` median and p95. Held-out natural repositories are split
by repository, never by edit. Natural-injection dictionary entries are
disjoint by corrected word between training and evaluation.

Corpus likelihoods use the tokenizer's right-truncated first 512 tokens of
each frozen document. Natural-pair KL uses exact unchanged character spans to
align non-edited next-token coordinates; edited-word targets are excluded.
This token cap and alignment rule are shared by Base and every adapter and may
not change between cycles.

Natural evaluation is a key secondary non-degradation check, not a third
confirmatory endpoint. Dataset licensing and achieved injection coverage are
reported. No ad-hoc character probe is a gate in v1; the three untouched task
families and corpus preservation battery are the prespecified clean-harm
guards.

## 6. Statistical contract and gates

All accuracy intervals use a paired, task-stratified bootstrap with 10,000
replicates and seed 42. Task macro averages weight tasks equally. The final
three-seed estimate uses a hierarchical bootstrap over learning seed and item;
every seed is also reported separately. Exact McNemar tests and the complete
transition counts are reported for each arm pair.

The machine-readable report records both a two-sided 95% interval and the
prespecified one-sided 95% lower bound. Gate comparisons use the latter;
two-sided intervals remain the descriptive uncertainty summary. The
hierarchical estimate, rather than requiring every individual seed interval to
pass, is the confirmatory estimate. The separate directional rule still
requires non-negative clean change and positive typo change in at least two of
three seeds.

The two primary hypotheses form an intersection-union decision: both clean
non-inferiority and typo superiority must pass, so no multiplicity correction
is applied between them. Secondary intervals are descriptive; any confirmatory
secondary claims require Holm correction declared before opening.

For a model/candidate to pass:

- clean macro: estimate at least -1.0 percentage point and one-sided 95% lower
  confidence bound above -1.0 point versus Base;
- task collapse guard: every task's clean point estimate is above -3.0 points;
- typo `random-2` macro: estimate at least +2.0 points and 95% lower confidence
  bound above 0 versus Base;
- clean PPL ratio at most 1.02;
- clean forward `KL(Base || Adapter)` median at most 0.03 nats/token;
- at least two of three seeds have non-negative clean change and positive typo
  change.

For an incremental localized-state claim, the proposed arm must additionally
be clean-non-inferior to the output-matching baseline and have a positive lower
confidence bound for typo accuracy versus that baseline. A causal-localization
claim additionally requires the proposed causal window to outperform a frozen
same-width random middle/late window and to be non-inferior to full-layer state
alignment under identical data order, LoRA capacity, and token budget. Without
those controls, the result is described as state-alignment improvement rather
than evidence that causal target selection mattered. Natural injection
must have a point change of at least -1.0 point and lower bound above -2.0
points; held-out natural KL must not expand. These are key secondary gates.

`KL(Base || Adapter) <= 0.03` nats/token is a frozen safety gate. State distance,
state loss, and paired-patch gain are mechanistic diagnostics and never block a
behaviorally successful model. The held-out natural-pair clean--typo KL gap is
the separately declared key-secondary non-degradation gate above.

## 7. Mechanistic audit

On a fixed 500-pair subset, apply the submitted paper's paired-patching audit:
first-CoT-token clean--typo KL, additional restoration from the frozen model-
specific early residual-stream window, their Base-to-Adapter change, and all
answer transitions/clean harm. The donor is the evaluated model's own clean
run. A reduced additional patch gain supports internalization; unchanged gain
with improved accuracy is reported as later-stage compensation. Neither result
changes the primary pass/fail decision.

## 8. Change control and reporting

Version 1.1 was a prospective source-capacity amendment made before cycle-2
training and before any model outcome was evaluated. Its initial audit reported
that 34 of the released 500 MATH-500 records had fewer than four eligible words
and consequently declared a 466-record population. Version 1.3 below documents
why that audit did not enforce the full distinct-word eligibility contract.

The v1.1 pre-training source-capacity audit reported: GSM8K 1,319/1,000 eligible/needed,
MMLU 13,667/1,000, ARC-Challenge 1,156/1,000, MMLU-Pro 11,792/1,000,
MATH-500 466/466, and CommonsenseQA 1,213/1,000. The MATH-500 value is
superseded by the corrected v1.3 census; the other task pools retain a
deterministic unused margin.

Version 1.2 separates two prespecified natural-typo estimands that v1.1's first
implementation had incorrectly coupled. Natural LM pairs test unseen
repositories and therefore use repository-disjoint rows. Natural injection
tests unseen corrected words and therefore uses a deterministic corrected-word
partition: 60% training, 10% tune, 10% pre-PR, and 20% final. Training excludes
every record whose corrected word belongs to an evaluation role, regardless of
repository. The first v1.1 materialization attempt failed before writing a
registry because its coupled tune dictionary covered 0/100 task items. This
amendment was made without model inference or outcome access and preserves all
endpoints, typo operations, thresholds, and opening rules.

Version 1.3 is a prospective correctness amendment made before any frozen
evaluation role was opened or any model output was read. A falsification test
showed that the v1.1 capacity audit counted eligible word occurrences rather
than case-insensitive distinct lexical words and did not apply the complete
mathematical-expression and gold-answer exclusions. Re-running a full census
against the pinned MATH-500 revision with the already declared eligibility
rules yields 440/500 records with at least four eligible distinct words; all
440 also support two distinct adjacent-transposition targets. The previous
v1.2 artifacts are invalidated rather than opened. Version 1.3 freezes those
440 records, records the task-level source/exclusion/eligibility census in the
registry, and does not weaken any edit rule, endpoint, threshold, or opening
condition. No accuracy, KL, likelihood, generation, or patching result informed
this amendment.

`registry.json` binds protocol/config/source hashes, item and pair files,
prompt/extractor versions, opening records, and reports. A pre-opening memo
binds the arm IDs, three training seeds, config hashes, checkpoint hashes,
model revisions, and hypotheses. Critical bugs are logged; corrected results
and affected earlier results are both retained. Primary definitions,
thresholds, sample sizes, item identities, opening rules, or typo generation
cannot be amended retroactively.

The fixed report includes Base and every arm/seed; task and macro clean/typo
accuracy; four-way transitions; PPL/KL; all secondary typo conditions; natural
coverage; tokenization strata; and the mechanistic audit. Negative results use
the same table and are not omitted.
