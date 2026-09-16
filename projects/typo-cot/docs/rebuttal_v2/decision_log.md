# Rebuttal v2 decision log

This log records a reviewer-requested, post-hoc audit. It is not a preregistration
made before seeing the submitted paper's results. The scientific contract is
[README.md](README.md), with field definitions in [schemas.md](schemas.md).

## Initial contract freeze

- Protocol schema: `rebuttal-protocol/v2`.
- Protocol revision: `2.0.0`.
- Initial contract recorded at: `2026-09-16T06:47:07Z`.
- Code baseline reviewed: `f3c91f8e662a2b60cec30cc84ca62ba715d5469a`.
- Source documents: user-supplied `Rebuttal_README_v2.md` and
  `Rebuttal_PR_plan_v2.md`; these working drafts are not runtime inputs.
- Scope of PR-00: documentation, schemas and
  [protocol.json](../../configs/rebuttal_v2/protocol.json).
- Freeze identity: protocol revision, canonical protocol SHA-256, and the Git
  commit containing these documents. Generated artifacts must record all three.
  The canonical hash is computed as defined in schemas.md; it is not a hash of
  a pretty-printed JSON file or of a document containing its own hash.

The initial record precedes v2 implementation and formal generation. PR review
may correct the draft contract before any run. Every correction remains in Git
history. Once artifacts exist, incompatible schema changes require a new schema
version. Changes to cohort selection, parser, donor policy, window or estimand
require a new protocol revision and a new run; previous artifacts are retained.
Any decision informed by new outcomes must explicitly record that fact.

## Known outcomes and unavailable evidence

The supplied design reports the following historical values. They are reference
metadata only, never v2 success-count acceptance constraints:

| Historical reference | Value | Use in v2 |
|---|---|---|
| Gemma / GSM8K main cohort | 172 pairs | Audit recovery and missing exact IDs; never fill missing pairs |
| Gemma correct / offset / cross | 129 / 44 / 42 successes of 172 | Historical comparison panel only |
| Table 6 pooled result | 800 successes of 1,241 | Separate historical reference, not a new population rate |
| Qwen / MMLU-Pro Table 7 | n=97; early 52.6%, middle 55.7% | Reason to retain the non-early-favouring setting |
| Target-token miss / gold-option edit | 30.2% / 21.5% | References with their original cohort scope, not R2 labels |

The design quotes PDF SHA-256
`2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0`,
which matches the existing catalog. PR-00 does not claim a fresh PDF comparison.
Review originals, archived exact cohort inventories, raw all-arm generations,
historical revisions, donor bank and RQ2 prefixes have not been supplied through
the v2 intake. Their absence is not evidence of zero success or of a reproduced
cohort. No new model outcome was used to choose this contract.

## Decisions made in PR-00

| Decision | Reason and consequence |
|---|---|
| Keep v1 parser, schemas, historical validators and results | Existing validators re-extract and require historical counts; changing them would alter old artifact meaning |
| Add a separate v2 package and CLI namespace | Outcome-independent intake and fresh experiments require different acceptance conditions |
| R0/R1/R2/R3 first; R4 next; R5 optional | Prioritize archive, extraction and semantic evidence before extending GPU work |
| R3 uses the original Gemma/GSM8K 172; R4 the original Qwen/MMLU-Pro 97 and two windows | Do not replace missing inputs with favourable settings or select a new cohort to reproduce a count |
| Preserve unknown historical provenance | A fresh immutable revision is recorded separately, not asserted to be the submitted revision |
| Define extraction independently of gold | The final-line grammar and counterexamples in README section 18 are frozen before v2 outcome scoring |
| Independently reconstruct actual spans and token endpoints | Producer replay alone cannot validate submitted-text targeting or semantic preservation |
| Keep semantic labels separate from baseline job eligibility | Non-preserved and unassessed items still belong to the original cohort's behavioural coverage |
| Strict all-or-nothing +2 offset | Do not change intervention multiplicity by silently dropping invalid coordinates |
| Fixed group-disjoint donor bank, deterministic one-to-one assignment without replacement | Avoid cyclic within-cohort donors; shortages affect cross only; donor reuse requires a later revision |
| Fresh clean/typo/self/correct/offset/cross within one runtime | Do not compare historical correct generations to fresh controls as the primary experiment |
| Group bootstrap and separate control-valid denominators | Keep repeated perturbations of the same original problem together and retain valid offset-only evidence |
| R3 Holm family remains two comparisons | Missing cross contributes p=1 only in multiplicity correction, not as an observed result |
| Separate worker telemetry from scientific configuration | Different physical GPU numbers of the same device model must not invalidate compatible shards |
| Add explicit runtime-lock and upstream audit-run CLI references | Make artifact discovery and CPU-only reporting possible without inferring files from directories |
| Keep real-model smoke distinct from CPU acceptance fixtures | Toy-model correctness cannot establish backend/cache/self-copy behaviour on the actual models |

The user approved sequential feature PRs and the branch prefix
`ARR2026_August_Rebuttal/`. Each PR starts from the then-current main; all
review findings must be addressed and the previous PR incorporated into main
before the next feature begins. PR-00 updates the existing public-config
allowlist for protocol.json without adding runtime stubs or documentary
string-matching tests.

## Status at PR-00

| Dimension | Status |
|---|---|
| Protocol documentation | Defined in this PR; subject to PR review |
| v2 command implementation | Not started; no executable command registered |
| v2 CPU acceptance fixtures | Not implemented |
| Real-model smoke | Not run |
| Formal R0–R5 artifacts/results | Not run |
| Human semantic annotation | Not performed |
| Existing regression tests | Record actual commands and outcomes in the PR validation section |

These statuses refer to v2 scientific implementation. Passing repository
documentation/layout regression checks does not change the CPU-fixture,
real-model-smoke or formal-results status.

## Publication constraint

The [ARR author-response guidance](https://aclrollingreview.org/authors#author-response)
was checked on 2026-09-16. It allows limited additional experiments directly
answering reviewer questions and requires text-only responses without external
links. These repository documents are development artifacts. The cycle-specific
instructions must still be checked before submitting the response.

## PR-00 review clarifications (2026-09-16)

The first external review of PR #180 identified five schema/documentation
inconsistencies before any v2 runtime implementation or observation:

- Make section 8's donor rule agree with the already-fixed no-replacement policy.
- Separate uniquely identified execution attempts from a job's one canonical
  successful generation so retry history does not create duplicate output IDs.
- Represent missing generation/parser inputs in scoring coverage rather than
  inventing an extraction result.
- Store independently audited positions only in downstream input-audit records;
  never mutate the intake manifest to insert later audit results.
- Explicitly allow an absent donor-bank hash in plan rows, using canonical null
  rather than a placeholder digest.

The subsequent Claude review identified that R5 was listed as selectable without
its cap-specific settings. Remove R5 from v2.0.0's selectable optional experiments
and embedded JSON; its design remains in README §10 for a dedicated optional
protocol revision in PR-07 after the required work. R3/R4 settings are unchanged.
This correction changes the canonical protocol hash; the earlier draft hash
remains in the PR review history, and no runs/artifacts were produced under it.

Additional review clarifications make unknown group IDs an explicit global
planning preflight failure (CPU audits remain available), distinguish typed
configuration fingerprints from domain-separated record IDs without changing
the hash algorithm, and make schemas.md the sole normative field definition.
Parser coverage never triggers an outcome-dependent switch of the primary
parser, denominator or human-scoring column. Report coverage by arm on fixed
cohorts and restrict claims to fixed-parser success; without complete independent
human scoring, parser-independent semantic claims are inconclusive.

These are pre-implementation review corrections without new model observations.
The review and corrective commits remain part of the initial PR-00 freeze history.

## PR-01 implementation (2026-09-16)

PR-00 was merged in `b63d4251c93f5ff11cd60841bfe739f20028724c` before PR-01
started from main. PR-01 implements only the CPU archive intake and its CLI.
The scientific protocol remains revision `2.0.0` with canonical SHA-256
`d13794e9d29f1c1750c66f996475ba05f7932e239768963ef1014b562868e70b`.

The initial versioned source adapters are documented in [intake.md](intake.md).
They explicitly serialize the PR-00 source contract; unknown layouts are reported
without guessing or converting legacy artifacts in place. Saved scoring metadata
is retained and summarized as archived judgments, never rerun or used as an
acceptance quota. Missing evidence and structural non-applicability stay distinct.

Independent CPU fixtures cover identity/hash integrity, exact missing IDs,
source-kind separation, aliases, absent raw generation, zero-row metadata, and
128 recorded successes against a historical reference of 129. They are synthetic
fixtures, not evidence that the original 172/97-item archives were recovered.
Existing v1 entry points, extractors and acceptance rules are unchanged.

Implementation and CPU fixtures are available for R0 only. Real-model smoke and
formal model results remain not run; R1 onward are not implemented by this PR.
The optional model-dependent test suite requires torch, which is not installed
in the CPU-only validation environment. No GPU test is claimed as passed.

PR-01 review corrections bind every expected and observed generation identity to
one slot, validate the frozen historical reference metadata without enforcing
observed counts, and retain excluded extras without treating them as members.
Alias identity is derived from the same captured bytes whose hash is recorded.
Observed stopping flags take precedence over unsupported raw labels, and explicit
null archived metadata remains unknown rather than being inherited. CSV text
escaping changes only the export view; canonical JSON/JSONL identities are intact.
These corrections do not use new model outcomes or change the scientific protocol.
