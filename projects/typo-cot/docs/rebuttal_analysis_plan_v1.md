# ARR rebuttal analysis plan v1

Status: **manifest, six-setting controls, and source/write grid implementations
complete; remaining result-producing interfaces are frozen; no full-cohort new
intervention outcome has been inspected**.

This plan fixed the additional analyses requested for the ARR rebuttal before
their result-generating commands were implemented. The submitted 19-page PDF
is the reference for the existing experiment, cohort definitions, and reported
numbers. Print its recorded fingerprint with:

```bash
uv run --project projects/typo-cot typo-cot experiments source
```

The additions below test limitations stated in that PDF. They do not silently
redefine the submitted experiment. Any departure needed after implementation
must create a versioned v2 plan and must report the v1 result rather than
overwriting it.

## Existing evidence and fixed settings

The paper reports that an edited-word patch over residual layers `[0,6)`
restored **800/1,241** selected clean-correct/typo-wrong failures in six planned
settings. The exact setting denominators are frozen as follows.

| Task | Model | Paper denominator |
|---|---|---:|
| GSM8K | Gemma-3-4B-IT | 172 |
| GSM8K | Llama-3.2-3B-Instruct | 232 |
| GSM8K | Mistral-7B-Instruct-v0.3 | 197 |
| MMLU | Gemma-3-4B-IT | 209 |
| MMLU | Llama-3.2-3B-Instruct | 226 |
| MMLU | Mistral-7B-Instruct-v0.3 | 205 |

The primary 172-pair Gemma/GSM8K control reports 129 correct-coordinate,
44 offset, and 42 cross-item restorations. The paper also states three limits
that motivate this plan: answer-level position/donor controls cover only that
primary cohort, `[0,6)` was selected data-adaptively, and each clean-to-typo
intervention requires a paired clean run.

The manifest builder accepts a freshly regenerated source only when all frozen
producer contracts and the exact six-setting 1,241/800 reference totals are
reproduced. It is not a generic cohort builder. A source with different counts
must use a separately versioned analysis and must never be pooled with, or
presented as identical to, the rebuttal cohort.

## Sample-ID and provenance freeze

`build-rebuttal-manifest` is the only command allowed to select records. Before
any new model forward or generation, it writes:

- `pair_manifest.jsonl`, containing every validated pair and its aligned edited
  spans;
- `cohort_ids.json`, containing the ordered stable pair IDs for every setting
  and analysis cohort (each pair record retains its human-readable sample ID);
- `source_audit.json`, containing source/output hashes, source schema versions,
  model/tokenizer revisions, and every exclusion;
- `run.json`, containing arguments, environment, completion state, and output
  hashes.

The exact pair identities therefore live in the versioned `cohort_ids.json`
artifact, not in prose or a directory-name convention. Its SHA-256 is copied
into every downstream run. The manifest implementation PR must be merged before
GPU work, and that PR must include the frozen selection rules and tests;
generated records remain local experiment artifacts and are not committed.

The six-setting restoration cohort is the fixed-window run's ordered,
regenerated clean-to-typo denominator. The manifest separately records every
anchor selected by that run and every baseline exclusion between the selected
anchor pool and the paper denominator. The harm cohort is selected
independently from validated prepared pairs with at least one aligned edited
word, `clean_correct == true`, and `typo_correct == true`; it is exhaustive and
uncapped under that stored prepared-run definition.

The full clean-correct population in the prepared sources is retained as
`full_clean_correct`, then partitioned into `patch_eligible_clean_correct` (at
least one aligned edited word) and `alignment_ineligible_clean_correct` (no
aligned edited word). Prepared, patch-eligible clean-correct/typo-wrong records
outside the paper's regenerated denominator are listed explicitly rather than
silently assigned to either conditional cohort. Therefore restoration plus harm
is a disjoint **repair--harm composite**, but is not claimed to exhaust a
population unless both the outside and alignment-ineligible sets are empty. No
outcome-dependent sampling is allowed.

Every aligned edit records clean and typo character spans, token spans, and
word-final token coordinates. A record is invalid for an arm only under that
arm's prespecified coordinate rule; invalid records and reasons stay in the
status output.

## Frozen command order

The command names describe operations rather than paper question or priority
labels. The stable execution order is:

1. `build-rebuttal-manifest`
2. `six-setting-patch-controls`
3. `source-write-coordinate-grid`
4. `multitoken-kl-readout`
5. `patch-harm-audit`
6. `tokenization-severity-analysis`
7. `subword-position-patching`
8. `held-out-window-evaluation`

Each command owns a configuration, output directory, per-item records, and
`run.json`. A command may consume an earlier run only after validating its
schema, completion flag, arguments, model/tokenizer provenance, and hashes.

## Six-setting patch controls

`six-setting-patch-controls` extends the answer-level specificity controls to
all six settings. Each item has three arms over `[0,6)` and greedy generation:

1. **correct**: same-item clean edited-word-final states are written to the
   aligned typo edited-word-final coordinates;
2. **offset-2**: both clean donor and typo write coordinates move two tokens
   after the corresponding edited-word endpoint;
3. **cross-item**: another item's clean edited-word-final states are written to
   the recipient's original typo coordinates.

The reused correct arm was produced under a primary-then-empty-only positional
fallback that remains enabled at the generation length cap. The prospectively
generated offset and cross-item arms use the same extraction rule so paired
events are comparable. Before any confirmatory six-setting result generation,
this corrects the initial internal runtime contract that enabled positional
fallback only after EOS. The impact is limited to length-capped continuations
whose primary extractor is empty; EOS versus length-cap termination is still
stored for every new generation.

Every offset coordinate must remain in the stored prompt interior (excluding
the first and final prompt tokens), outside every edited-token span, and valid
in both donor and recipient prompts. If any aligned word fails this rule, the
complete offset arm is invalid; individual coordinates are never dropped. This
prospective confirmatory rule is intentionally stricter than the submitted
primary control's historical per-coordinate filtering and preserves equal
intervention cardinality between correct and offset arms. Cross-item donors are
assigned by a deterministic cyclic derangement after sorting pair identities
within `(task, model, target_rule, number_of_aligned_words)`. Singleton strata
are invalid; a donor may never equal its recipient.

The primary comparison uses only the **common-valid** subset on which all three
arms are valid. Each setting reports `n_original`, `n_offset_valid`,
`n_cross_item_valid`, and `n_common_valid`; arm-specific descriptive rates may
be shown separately but may not replace the paired denominator.

For each setting and each of correct-vs-offset and correct-vs-cross-item, report
the restoration rate, paired risk difference, pair-bootstrap 95% interval, and
two-sided exact McNemar test. Apply Holm adjustment jointly to the 12
setting-level tests. Also report an equal-setting macro-average with a nested
bootstrap that first resamples settings and then pairs within settings.

Required outputs are per-item records, a six-setting control table, a
common-denominator flow table, a risk-difference forest plot, and a complete
multiplicity table.

## Source/write coordinate grid

`source-write-coordinate-grid` separates donor content from write location.
Let `E` be the edited-word-final coordinate and `O` the valid +2 offset. The
four paired arms are `E->E`, `E->O`, `O->E`, and `O->O`.

The primary cohort is Gemma-3-4B/GSM8K; the prespecified replication cohort is
Mistral-7B/MMLU. All four arms use a shared common-valid denominator. The two
primary contrasts are `E->E` versus `E->O` (write location) and `E->E` versus
`O->E` (donor source). Report a four-arm Cochran's Q, two-sided exact McNemar
tests for prespecified pairwise contrasts, Holm adjustment, risk differences,
and pair-bootstrap 95% intervals.

Because `E->E` is reused from the fixed-window producer, all four arms use that
producer's primary-then-empty-only positional fallback, including at the
generation length cap. New generations additionally retain explicit EOS versus
length-cap termination provenance.

Interpretation is fixed before results:

- only `E->E` high: the source/write combination is specific;
- `O->E` also high: write location dominates and donor specificity is weak;
- `E->O` also high: donor content matters but position specificity is weak;
- all arms close: the present specificity interpretation must be reconsidered.

## Multi-token distributional readout

`multitoken-kl-readout` evaluates all 1,241 restoration pairs without free
generation. It teacher-forces the same clean continuation tokens `y_1..y_16`
after the clean question, typo question, and `[0,6)` patched typo question.
At each position it stores raw `KL(clean || typo)` and
`KL(clean || patched)` in float64 after model logits are materialized.

The primary metric excludes the adjacent first token:

```text
R_{2:16} = 1 - mean(KL_patch[t], t=2..16) / mean(KL_typo[t], t=2..16)
```

Untreated denominators must be finite and greater than `1e-9`. Secondary
metrics are `R_{2:4}`, `R_{2:8}`, each token's raw KL reduction, and the paired
difference between token 1 and tokens 2--16. Report setting estimates,
pair-bootstrap intervals, and a position 1--16 trajectory. Cache files are
content-addressed and may be reused only after exact source/model validation.

## Correct-answer harm audit

`patch-harm-audit` applies the correct-coordinate `[0,6)` clean-to-typo patch to
the fixed clean-correct/typo-correct cohort. It reports preservation,
right-to-wrong harm, any extracted-answer change, and unextractable patched
answers without removing them from the denominator.

The required table contains, per setting, `n_typo_correct`, `preserve`, `harm`,
and `answer_changed`. A combined audit joins the complete harm records with the
complete 1,241-pair restoration partition and reports wrong-to-right,
right-to-wrong, and their conditional count/rate balance. It must be labelled a
**repair--harm composite**, not population net accuracy, whenever the manifest
reports prepared typo-wrong records outside the regenerated paper denominator.
A population net-accuracy claim requires a separately preregistered patch run
over those uncovered records under the same regenerated-baseline and coordinate
rules. Until this audit is complete, all prose must call 800/1,241 a conditional
recovery result on selected clean-correct/typo-wrong failures.

## Low-compute and held-out analyses

`tokenization-severity-analysis` consumes completed three-arm records and adds
no model inference. Prespecified strata are unchanged/changed subtoken count,
typo-side fragmentation increase, edit count 1/2/3--4, and clean edited word
being single-token/multi-token. It reports all three arm rates and denominators
for every stratum; empty or tiny cells are shown, not silently merged.

`subword-position-patching` compares first, final, and all aligned subwords on
the primary 172-pair cohort. The primary analysis is the equal-subtoken-count
subset. A separately labelled secondary analysis may use an explicit monotone
alignment heuristic when counts differ; it cannot be pooled with the primary.

`held-out-window-evaluation` selects a contiguous six-layer window using only a
diagnostic split and evaluates it once on a disjoint sample-ID split. A single
invocation reads both ordered split lists from the manifest's
`cohort_ids.json`; it does not consume an outcome-bearing `--selection-run`.
It completes and hashes `window_selection.json` before loading any held-out
record, then writes held-out per-item records and the final comparison. Split
IDs, the window scoring rule, tie breaking, and the selected window are thus
recorded before held-out outcomes are read. `[0,6)` remains the paper's
historical, data-adaptive reference and is not relabelled prespecified.

## Exclusions, failures, and multiplicity

An invalid arm is a planned exclusion; model/runtime/error failures are not.
Failures remain visible in `pair_status_records.jsonl` and `run.json`, and a
run with unresolved failures cannot publish a final table. Unextractable
patched answers are failed intervention readouts and count as zero success.

Bootstrap uses 10,000 deterministic resamples and seed 42. Resampling preserves
pairing and, for pooled analyses, clustering by setting and sample ID. Exact
tests are used for paired binary outcomes. Every exploratory contrast is
labelled post-hoc and kept out of the confirmatory Holm family.

## Negative-result claim rule

This **negative-result claim rule** is binding:

- a correct-versus-control contrast is "clearly separated" only when its paired
  risk difference is at least **10 percentage points** and its two-sided exact
  McNemar test has **Holm-adjusted** `p < 0.05` in the frozen 12-test family;
- a control "approaches correct" whenever either part of that rule fails; this
  definition is applied mechanically rather than after inspecting rates;
- if both controls are clearly separated in at least five of six settings, the
  rebuttal may describe cross-setting specificity while naming every
  non-supporting setting;
- otherwise, if at least one setting supports both contrasts, reversibility may
  remain cross-setting but specificity must be called setting-dependent; if no
  setting supports both, no specificity claim is retained;
- if either control approaches correct in any setting, the Introduction,
  contribution list, abstract, and rebuttal cannot make an unconditional
  all-setting specificity claim;
- if multi-token effects vanish after token 1, the distributional claim is
  restricted to the adjacent readout;
- if harm offsets repair, no net robustness benefit may be claimed;
- if the held-out window does not reproduce the early advantage, `[0,6)` is
  described only as a data-adaptive result.

No offset, window, subset, donor match, token range, or claim threshold may be
changed after inspecting its outcome under this plan.

## Review and execution policy

Each operation is implemented test-first on its own descriptive branch and PR
against `develop`. The next operation starts only after actionable review on
the current PR is resolved and the PR is merged. CPU contract tests precede
each GPU smoke. Project validation runs use only physical GPU 3, expressed as
matching `CUDA_VISIBLE_DEVICES=3` and `--gpu-id 3`; public examples retain a
user-selectable GPU variable.
