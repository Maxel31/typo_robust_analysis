# R2 input and blinded semantic audit (PR-03)

Implementation boundary fixed before any archive outcomes or human judgments
are inspected. The scientific protocol remains revision `2.0.0`; the algorithms
below concretize the PR-00 contract rather than changing cohorts or interventions.
Implementation and synthetic CPU fixtures are available. Actual human judgments,
real-model smoke, and formal results have not been completed. Command execution
may be complete while the whole multi-stage R2 experiment remains partial.

## Independent text and token coordinates

Text is never normalized or rewritten. Character offsets are half-open Unicode
code point spans in the saved query; prompt coordinates add only the explicitly
saved, verified query span. Repeated words or few-shot examples never justify a
first-substring match.

The difference algorithm uses unit-cost Levenshtein distance. A deterministic
representative path prefers substitution, deletion, then insertion on ties, but
multiple minimum-cost paths are reported as ambiguous, not as verified alignment.
Adjacent non-equal operations form edit spans; all disjoint edits are retained.
No net change is explicit and does not prove the absence of attempted/cancelled
historical edits. Limits are 4,000,000 DP cells and 10,000 code points per side;
larger inputs report audit unavailability instead of invented empty differences.

Words are maximal Unicode letter/number/combining-mark runs containing a letter
or number. ASCII apostrophe and U+2019 are internal only between such characters.
The Unicode database version and algorithm version are recorded. Zero-width
edits map only to a containing word or a single unambiguous touching word.
Non-unique or multiword correspondence is not assigned an endpoint. Multiple
edits within the same pair of words retain their edit records and share a single
deduplicated word correspondence. Archived token positions cannot resolve an
independently ambiguous diff.

Tokenization uses the complete, already rendered exact prompt. All subtokens
overlapping each changed word are retained; the write endpoint is their last
token, not the first subtoken or the first matching word elsewhere in the prompt.
Special tokens remain in the full token-ID sequence. Complete prompt token IDs
and canonical JSON hashes are saved independently for clean and typo prompts.
Only an audit with `alignment_status="aligned"` supplies word spans to endpoint
tokenization. One-to-many or otherwise unsupported
correspondence still tokenizes the complete prompt with no word spans, preserving
its token IDs and hash but producing no word endpoints; it is never deduplicated
or salvaged, and its invalid/unknown state is retained in the input-audit records.
Unknown alignment remains separate from archived-tokenizer agreement.
`archived_revision_match` compares revisions alone;
`matches_archived_tokenizer` is true only when archived tokenizer ID, revision,
and tokenizer-file hash are all present and agree. Known mismatch is false;
insufficient evidence is null. An archived identity value that is not a nonblank
string is insufficient evidence, not a mismatch. Known nonblank strings compare
exactly as archived, without stripping, coercion, or normalization; any unequal
known value makes the result false. The comparison uses explicit archived runtime
keys `tokenizer_id` and `tokenizer_sha256` in addition to the saved tokenizer
revision; it does not certify historical wrapper or special-token insertion policy.

## Pinned local tokenizer lock

The CPU dependency is `tokenizers==0.22.2`, already present in the repository
dependency lock for the optional model runtime. R2 does not import torch or
download model weights. The lock uses schema `rebuttal-tokenizer-lock/v2`, with
`models` keyed by exact manifest model ID. Each entry has:

- `tokenizer_id` and immutable 40-hex `revision`;
- `tokenizer_files: {"tokenizer.json": ArtifactRef}`;
- `library: {"name": "tokenizers", "version": "0.22.2"}`;
- `special_token_ids`, mapping names to integer IDs or null;
- `chat_template`, null or an object containing exact `text` and its `sha256`;
- `add_special_tokens`, boolean;
- `normalization`, the exact serialized tokenizer normalizer or null;
- `padding_side`, `left` or `right`;
- `offset_convention: "unicode-codepoint-half-open"`.

The tokenizer bytes and descriptors are verified before encoding. The complete
set of non-null declared special-token IDs must equal the tokenizer's actual
special-token IDs; named roles may share an ID but cannot designate ordinary
vocabulary entries as special. The complete prompt is already rendered: a chat
template is recorded but never applied again.
Padding/truncation must not shorten or alter the saved complete prompt sequence.
A missing model entry gives unknown token alignment while independent text audit
can still run. A broken declared hash or contradictory lock is an integrity error.

## Explicit region and archived-coordinate adapters

When available, `prompt_regions` is an object with `clean` and `typo` arrays.
Each region has `kind`, `start`, and `end`, using full-prompt code points;
`option_label` may additionally bind an option to its presented label. Kinds are
`stem`, `option-content`, `option-label`, `instructions`, `context`, and `entity`.
Missing or unsupported layouts produce unknown classification; they are never
inferred by searching for the first repeated query. Context exported to raters
comes only from explicitly annotated current-question regions, never the entire
few-shot prompt.
If declared regions remain but their exact full prompt is unavailable, ancillary
arrays stay empty and private coverage records a side-specific unavailable reason;
the available typo query remains selected instead of aborting the entire batch.
Malformed region coordinates in an available prompt are still integrity errors.

Risk flags separately describe stem, option content, option label, gold-option
membership, numbers, quantities, units, negations, and explicit entity regions.
Unknown is null, not false. Finite lexical diagnostics have a recorded version
and do not claim complete semantic detection. An option edit or risk flag is
not a semantic label and never automatically means `task_changed`.

Archived nested metadata has no universal layout in PR-01. R2 therefore interprets
only explicit coordinate adapters; unrecognized layouts remain unknown:

- An intended-token entry in `intended_edits` uses
  `audit_adapter="rebuttal-intended-token/v2"`, `clean_token_index`,
  `actual_clean_span`, and `actual_typo_span`. The two query spans must identify
  an independently reconstructed edit. No implicit list-index pairing is used.
- An archived coordinate entry in `archived_token_positions` uses
  `audit_adapter="rebuttal-word-endpoints/v2"`, `clean_endpoint`, and
  `typo_endpoint`, both zero-based full-prompt positions.

Intended-token hit and archived-coordinate agreement are separate from independent
alignment eligibility. A targeting miss does not erase a valid actual edited-word
endpoint. Token hit compares the actual edit's character span to the intended
token's full-prompt offset; membership in the same word is a separate `word_hit`
diagnostic. An insertion exactly on the intended token boundary is unknown, not
automatically a hit. These adapters expose provenance; they do not rescue ambiguous alignment.

## Two-stage semantic review

Stage A selects all recovered verified original members of R3/R4, regardless of
alignment, risk, parser results, or patch outcomes. Missing original IDs and
unavailable questions remain in private coverage. Extra/unknown membership is
reported separately rather than used to refill the original cohort.

Random opaque IDs map privately to source pairs. Reviewers receive only exact
typo query, explicitly necessary typo context/instructions/options, and their
review form. Do not distribute the directory containing private mapping, model,
gold, clean text, or results. Stage A responses contain `blind_id`, `rater_id`,
`timestamp`, `interpretations`, `ambiguity` (`unique`, `ambiguous`, `unassessable`),
and `rationale` under their versioned schema.

Two independent raters are the default. A single-rater audit requires an explicit
exception reason and never reports inter-rater agreement. All required Stage A
keys must be completed and hash-locked for the entire frozen batch before Stage B
exports any clean text. Missing, duplicate, unknown, or empty keys fail validation.
An explicit unassessable judgment counts as completed; an absent row does not.

Stage B adds clean material and may reveal task gold, while model, arm, and patch
outcomes stay hidden. It binds each rater's own immutable Stage A response and
the Stage A lock hash. Legal labels are `unique_preserved`, `ambiguous`,
`task_changed`, and `unassessable`. Independent ratings and any adjudication are
retained with references. Disagreements require an identified, timestamped
adjudication and reason. Incomplete returned A/B inventories are rejected with
the exact missing keys; no semantic label is fabricated for an unreviewed pair.
Adjudication is accepted only for disagreements among the required independent
raters. Unanimous and single-rater items reject even a redundant adjudication;
their independent label cannot be overridden by an unnecessary adjudication.
If the verified clean query is missing or whitespace-only, Stage A selection is
unchanged and coverage records `clean_query_unavailable`. Stage B allows only
`unassessable` for that item, and import requires each rater's explicit matching
judgment. Missing rows still fail: unassessability is never silently generated,
and absent clean evidence cannot establish semantic preservation or task change.

The import retains finalized labels, including `unassessable`, in
`semantic_labels.jsonl`. Their counts are reported in
`semantic_strata_table.csv` under dimension `semantic_label` and stratum
`unassessable`. `annotation_agreement.json` reports rater count,
pre-adjudication agreement, and adjudications; it does not duplicate semantic
stratum counts. Source/annotation coverage remains in `annotation_coverage.json`.
Semantic eligibility is derived only from the finalized labels, never from answer
success or patch effects.

## Commands and artifacts

From the repository root (paths refer to user-supplied inputs):

```bash
uv run --project projects/typo-cot typo-cot rebuttal-v2 input-audit \
  --manifest /path/to/intake/pair_manifest.jsonl \
  --tokenizer-lock /path/to/tokenizer_lock.json --output-dir /path/to/input-audit

uv run --project projects/typo-cot typo-cot rebuttal-v2 annotation-export \
  --audit /path/to/input-audit/input_audit_records.jsonl --stage a \
  --raters reader-one reader-two --output-dir /path/to/stage-a

uv run --project projects/typo-cot typo-cot rebuttal-v2 annotation-export \
  --audit /path/to/input-audit/input_audit_records.jsonl --stage b \
  --stage-a-labels /path/to/returned-a.jsonl \
  --annotation-batch /path/to/stage-a/annotation_batch.json --output-dir /path/to/stage-b

uv run --project projects/typo-cot typo-cot rebuttal-v2 annotation-import \
  --audit /path/to/input-audit/input_audit_records.jsonl --labels /path/to/returned-b.jsonl \
  --annotation-batch /path/to/stage-b/annotation_batch.json --output-dir /path/to/semantic
```

Default placeholder rater IDs are `rater-1` and `rater-2`; set actual identities
before Stage A export. A single ID requires `--single-rater-reason` at that stage.
Stage B takes the frozen rater list from the batch, not new CLI settings.

| Stage | Main output artifacts |
|---|---|
| Input audit | `input_audit_records.jsonl`, required sibling `input_audit_records.meta.json`, `input_audit_summary.json`, `cohort_flow.csv` |
| Stage A | Reviewer `annotation_stage_a.jsonl`; private `annotation_mapping.jsonl`, `annotation_batch.json`, `annotation_coverage.json` |
| Stage B | Reviewer `annotation_stage_b.jsonl`; immutable `stage_a_lock.json` and a new private `annotation_batch.json` |
| Import | `semantic_labels.jsonl`, `annotation_agreement.json`, `annotation_coverage.json`, `cohort_flow.csv`, `semantic_strata_table.csv` |

Stage B copies the private mapping and coverage artifacts, but it does not copy
the returned Stage A ratings file or every upstream evidence artifact. Its batch
and lock retain hash-bound references, often absolute paths, to those originals.
Keep every referenced file immutable and available through import and later
consumers. This is not a self-contained export: relocating files without updating
the complete reference tree and preserving its hash bindings invalidates it.

Every stage writes `run.json` and `protocol.json`, verifies transitive input hashes
before atomic publication, and requires an absent or empty output directory.
Input-audit records retain all manifest pairs, including extras and unknowns;
each table keeps cohort/membership strata separate. Later planning sets
`C_correct`, `C_offset`, `C_cross`, `C_three`, and `C_fresh_failure` are not inferred
from these CPU tables.

Returned A forms use `rebuttal-annotation-stage-a-rating/v2`. Returned B forms
use `rebuttal-annotation-stage-b-rating/v2` with `blind_id`, `rater_id`, `timestamp`,
`stage_a_lock_sha256`, `label`, and `rationale`. Adjudication rows in the same B
JSONL use `rebuttal-annotation-adjudication/v2` with `blind_id`, `adjudicator_id`,
`timestamp`, `stage_a_lock_sha256`, `final_label`, and `disagreement_reason`.
Every form includes its `schema_version`. The shared B export has only a hash
reference to each rater's A response, not another rater's interpretations.
