# ARR rebuttal analysis plan v1

Status: **interface-frozen; no new outcome has been inspected**.

This plan fixes the additional analyses requested for the ARR rebuttal before
their result-generating commands are implemented. The submitted 19-page PDF is
the reference for the existing experiment, cohort definitions, and reported
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

The public implementation may be run on a freshly prepared cohort, while the
rebuttal analysis uses the exact archived paper cohort when its source records
pass integrity checks. Fresh and historical cohorts must never be pooled or
presented as identical.

## Sample-ID and provenance freeze

`build-rebuttal-manifest` is the only command allowed to select records. Before
any new model forward or generation, it writes:

- `pair_manifest.jsonl`, containing every validated pair and its aligned edited
  spans;
- `cohort_ids.json`, containing the ordered sample IDs for every setting and
  analysis cohort;
- `source_audit.json`, containing source/output hashes, source schema versions,
  model/tokenizer revisions, and every exclusion;
- `run.json`, containing arguments, environment, completion state, and output
  hashes.

The exact sample IDs therefore live in the versioned `cohort_ids.json` artifact,
not in prose or a directory-name convention. Its SHA-256 is copied into every
downstream run. The manifest implementation PR must be merged before GPU work,
and that PR must include the frozen selection rules and tests; generated records
remain local experiment artifacts and are not committed.

The six-setting restoration cohort is the fixed-window run's ordered
clean-to-typo denominator. The harm cohort is selected independently from
validated prepared pairs with `clean_correct == true` and
`typo_correct == true`. If a harm setting exceeds 500 records, the first 500 in
canonical sample-ID order are retained. No outcome-dependent random sampling is
allowed.

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

An offset coordinate must be within the question portion, outside every edited
token span, and valid in both donor and recipient prompts. Cross-item donors are
assigned by a deterministic cyclic derangement after sorting sample IDs within
`(task, model, target_rule, number_of_aligned_words)`. Singleton strata are
invalid; a donor may never equal its recipient.

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
and `answer_changed`. A combined accuracy audit joins these records with the
failure cohort and reports wrong-to-right, right-to-wrong, and net accuracy.
Until this audit is complete, all prose must call 800/1,241 a conditional
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
diagnostic split and evaluates it once on a disjoint sample-ID split. Split IDs,
the window scoring rule, tie breaking, and the selected window are recorded
before held-out outcomes are read. `[0,6)` remains the paper's historical,
data-adaptive reference and is not relabelled prespecified.

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

- if correct exceeds both controls in at least five of six settings, the
  rebuttal may describe cross-setting specificity;
- if effects vary materially, reversibility may remain cross-setting but
  specificity must be called setting-dependent;
- if either control approaches correct in any setting, the Introduction,
  contribution list, abstract, and rebuttal must weaken the specificity claim;
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
