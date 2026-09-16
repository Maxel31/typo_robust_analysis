# Rebuttal v2 artifact and CLI contracts

Status: protocol revision `2.0.0`. PR-01 implements R0 intake, PR-02 implements
R1 answer audit, PR-03 implements R2 input audit and two-stage annotations, and
PR-04 implements CPU-only global planning. Generation and reduction contracts
remain specifications.
No real-model smoke or formal results are available. Scientific definitions, parser grammar, cohorts,
and acceptance cases are in [README.md](README.md). Machine settings are in
[protocol.json](../../configs/rebuttal_v2/protocol.json).

## 1. Common representation and identity

Artifacts use UTF-8 JSON or JSONL, with one object per nonempty JSONL line. Reject
duplicate JSON keys, nonfinite numbers, duplicate record IDs, and a schema version
the reader does not support. Each artifact/row has `schema_version` using the
`rebuttal-<artifact>/v2` namespace; nested records inherit the containing schema.
Required keys must be present. Only explicitly nullable fields may be `null`;
an empty list means known empty, not unknown. Extensions need a documented schema
revision rather than silently ignored fields. CSV exports have headers and retain
IDs linking to their machine-readable source rows. CSV `NA` means unavailable,
never zero or false.

Timestamps are UTC RFC 3339 strings ending in `Z`, including seconds. Dates are
`YYYY-MM-DD`; local-time strings are not accepted as freeze timestamps. Exact text
is preserved without stripping, newline conversion, Unicode normalization, or
case conversion. Character coordinates are zero-based Unicode code-point offsets
and half-open intervals `[start,end)`, never UTF-8 byte or UTF-16 offsets. Token
positions are zero-based positions in the unpadded, fully rendered prompt token
sequence, including special tokens. Runtime padding translation is recorded
separately. Layer windows are half-open decoder-block index intervals.

`sha256` means 64 lowercase hexadecimal characters. File hashes cover exact file
bytes; text hashes cover UTF-8 bytes of the exact string. Canonical object hashing
uses Python-compatible JSON with `sort_keys=True`, `ensure_ascii=False`,
`separators=(",", ":")`, `allow_nan=False`, encoded as UTF-8 without a trailing
newline. Schema-typed integers remain integers; identity payloads contain only
strings, integers, booleans, lists, objects, and null. Other finite numbers use
JSON's round-trip representation, with `-0.0` normalized to `0.0`. Every record
identity listed below hashes `domain + "\n" + canonical_json(payload)`; the domain and payload
are retained with the artifact. Hashing does not establish scientific validity.
Configuration hashes (`protocol_sha256`, `generation_config_sha256`, and
`scientific_config_sha256`) hash the canonical JSON of their specified object
without a domain prefix; they differ from hashes of pretty-printed file bytes.
Configuration hashes are typed content fingerprints, compared only within their
named field and supported schema, never interchangeable untyped record IDs.

`ArtifactRef = {path, sha256}` identifies an existing immutable local file.
Relative paths resolve against the JSON/JSONL file containing the reference;
moving an artifact requires preserving its reference tree or explicitly updating
references and hashes. `RecordRef = {artifact: ArtifactRef, record_id}` identifies
a unique row within that file. No row-number-only or mutable URL references are
accepted for frozen inputs. Missing files are represented in the source index,
not by a fabricated hash. All consumers revalidate referenced bytes before use.

| Identity | Domain and canonical payload | Invariant |
|---|---|---|
| `original_problem_group_id` | `rebuttal-original-problem/v2`; `{dataset_id, split, original_problem_id}` | Independent of model, target rule, perturbation, source kind, and outcome |
| `pair_id` | `rebuttal-pair/v2`; `{source_namespace, source_pair_key}` | Same acquisition identity across cohorts and all generated arms |
| `generation_id` | `rebuttal-generation/v2`; `{source_namespace, source_generation_key}` for archive; `{plan_id, scientific_config_sha256}` for fresh | Parser/scoring changes do not change generation identity |
| `attempt_id` | `rebuttal-generation-attempt/v2`; `{generation_id, attempt_number}` | Fresh execution attempts are distinct from the canonical job output; attempt numbers start at 1 and never repeat for a generation ID |
| `plan_id` | `rebuttal-plan-row/v2`; scientific planning payload specified in §7 | Independent of worker, shard count, and execution outcome |

`dataset_id` names a stable dataset namespace with stable original ID semantics;
`dataset_revision` is separately nullable provenance. Unknown revision alone does
not erase known original grouping. Reissued or ambiguous IDs across snapshots
require an explicit, hashed identity-alias mapping. Unknown original problem
identity gives `original_problem_group_id=null` with a reason, never a fabricated
group. Such records remain available for CPU source/answer audits, but cannot
enter donor matching, formal global sharding, or grouped inference.

An identity-alias artifact (`rebuttal-identity-aliases/v2`) contains
`pair_aliases` and `group_aliases` arrays. Every row is
`{alias, canonical, evidence_ref}`; the first two fields contain the complete
corresponding identity payload from the table above, and `evidence_ref` is an
`ArtifactRef` supporting equivalence. Resolve aliases before hashing while
preserving original payloads in provenance. Duplicate alias keys, cycles,
conflicting canonical targets and unequal exact texts for aliased pair identities
are errors. Missing evidence cannot be replaced with a text-similarity match.

`source_namespace` identifies an acquisition event, never an absolute file path
or machine. `source_pair_key` is the archived pair ID when available, otherwise
a stable producer `source_record_id`; the key origin is recorded. Intake verifies
that each `(source_namespace, source_pair_key)` has one identity, one set of exact
text hashes, and consistent task/model/target/perturbation metadata, including
across cohorts and files. A conflicting record is an input error; changing the
file hash must not turn it into a new pair. Different sources are linked only by
explicit identity aliases. Matching texts alone do not establish identity.

## 2. Status vocabularies

Status dimensions are separate fields, not interchangeable labels:

| Field | Values | Meaning |
|---|---|---|
| `availability` | `available`, `missing`, `unknown`, `not_applicable` | Present and verified; expected but absent; existence/identity unresolved; structurally unused |
| `coverage_status` | `complete`, `partial`, `unknown` | Recovery of an explicitly enumerated source cohort; count agreement alone is insufficient |
| `eligibility` | `valid`, `invalid`, `unknown`, `not_available` | Passed prespecified post-hoc protocol conditions; evaluated and failed; necessary evidence unresolved; optional input not provided |
| `runtime_status` | `pending`, `running`, `complete`, `failed` | Execution status for a planned valid job; model answer correctness is unrelated |
| `termination` | `eos`, `length-cap`, `unknown` | Observed generation stopping reason, with supporting metadata; unsupported archive labels remain raw metadata and imply unknown only if no supported stopping observation is available |
| `extraction_status` | `extracted`, `unextractable`, `ambiguous` | Gold-independent parser result on present output; reasons retain quoted/negated/unsupported syntax |
| `score_status` | `scored`, `not_scored` | Gold comparison available, or an executed extraction cannot be scored because gold is missing |
| `run_status` | `complete`, `partial`, `failed` | Command execution/artifact completeness; low answer success never fails a run |
| `experiment_status` | `not_run`, `partial`, `complete`, `invalid` | Scientific execution coverage, including runtime-integrity rejection |
| Claims `status` | `supported`, `not_supported`, `inconclusive`, `not_run` | Evidence for the precisely scoped claim |

Every non-success/non-available state includes stable `reason_codes` and a
human-readable explanation. Missing donor bank gives cross `not_available`;
a supplied bank with no legal match gives cross `invalid`. Missing provenance
gives `unknown`, not evidence of a failed condition. Runtime OOM, missing source
bytes, and hook violations are execution failures, not model errors. Extraction
failure from an otherwise complete generation is a scored failure under that
parser and stays in the denominator. `gold_match=null` is never treated as false.

## 3. Archive index, normalized sources, and intake

`archive_index.json` (`rebuttal-archive-index/v2`) has `created_at`,
`source_namespace`, `sources`, `cohorts`, `identity_aliases_ref` (nullable), and
`notes`. It is an inventory supplied by the data owner, not a list newly selected
by the current parser. Each `sources` entry contains:

| Fields | Contract |
|---|---|
| `source_id`, `source_kind`, `role` | Unique stable source ID; kind is `submitted-archive`, `public-regeneration`, or `fresh-audit`; role enum is `pairs`, `generations`, `prompts`, `cohort-ids`, `runtime-metadata`, `donor-bank`, `rq2-prefixes` |
| `path`, `expected_sha256` | Nullable if location/hash is unknown; a present source is read and its observed hash recorded |
| `format`, `adapter_version` | Declared source format and normalization adapter; absent adapter is reported unsupported |
| `source_commit`, `model_revision`, `tokenizer_revision`, `prompt_template_revision`, `generation_config` | Nullable archived provenance; each unknown has a reason |
| `availability`, `reason_codes` | Source lookup result, never inferred from reported success totals |

A source whose supplied hash disagrees with observed bytes is rejected; a source
with no historical hash can receive an observed hash but retains
`historical_hash_status=unknown`. Current revisions never fill archived unknowns.

Each cohort entry contains `cohort_id`, `setting`, `historical_n_reference`,
`expected_ids_ref` (nullable), `id_namespace`, and `membership_provenance_ref`
(nullable). The ID artifact enumerates the archived `source_pair_key` values,
independent of recovered record availability and new correctness judgments.
It is a JSON object with `schema_version="rebuttal-cohort-ids/v2"`, `cohort_id`,
`source_namespace`, and a unique nonempty-string array `source_pair_keys`.
For R3/R4 the historical count references are 172/97, not quotas to fill. If the
ID list is absent, `coverage_status=unknown`; report recovered count and count
shortfall separately without claiming to identify missing pairs. If the list is
known, retain exact missing IDs, extra IDs, duplicates, and conflicting records.
Extra records cannot silently enter the archived cohort. Count agreement cannot
establish exact membership. No missing pair is replaced with a newly sampled one.

Normalization preserves each original source record and its `RecordRef`. The
common pair payload is `source_record_id`, `source_pair_key`, `historical_pair_id`
(nullable), `cohort_ids`, dataset namespace/split/original ID (nullable only when
unknown), task/model identity, target rule and perturbation identity (nullable
only when unknown), exact texts/prompts, gold, saved edits/coordinates, and raw
generation references. Unsupported source layouts are reported; aggregate counts
are never expanded into invented per-item generations.

PR-01's concrete source adapter versions and input serialization are defined in
[intake.md](intake.md). They make the source-side normalization contract explicit;
they do not reinterpret v1 manifests as v2 or add a new scientific protocol.

Intake writes `source_audit.json`, `archive_inventory.jsonl`,
`pair_manifest.jsonl`, `pair_manifest.meta.json`, `archive_generation_records.jsonl`,
`historical_count_comparison.csv`, `missing_inputs.csv`, and `run.json`.
The source audit identifies which subsequent commands/tables each missing input
blocks. Inventory records requested and observed hashes and source provenance.
Historical counts are source-attributed reference columns separate from newly
measured counts. A successful inventory with missing records may have a complete
command run and partial/unknown cohort coverage; it must preserve both statuses.

## 4. Pair manifest, raw generation links, and tokenizer lock

Each `pair_manifest.jsonl` row (`rebuttal-pair-manifest/v2`) has:

| Fields | Contract |
|---|---|
| `protocol_ref`, `protocol_sha256`, `pair_id`, `source_namespace`, `source_pair_key`, `source_pair_key_kind` | Immutable protocol and identity; kind distinguishes historical ID from producer ID |
| `historical_pair_id`, `original_problem_group_id`, `dataset_revision` | Nullable provenance/group fields under §1 rules |
| `dataset_id`, `split`, `original_problem_id`, `cohort_membership` | Namespace components (nullable when unknown); membership contains cohort ID, known/unknown membership, and source reference |
| `source_kind`, `source_ref`, `source_sha256`, `source_record_id` | Original source provenance; source hash agrees with the referenced source |
| `pair_identity`, `group_identity`, `source_identity_provenance`, `additional_source_refs` | Original/canonical identity payloads and alias evidence, all acquisition identity claims, and additional source `RecordRef`s; both arrays are always present |
| `task`, `model_id`, `target_rule`, `perturbation_id` | Task/model required for normalized usable pair records; unknown target/perturbation allowed with reasons |
| `model_revision`, `tokenizer_revision`, `archived_runtime` | Nullable archived values only; not overwritten by a fresh runtime lock |
| `clean_text`, `typo_text`, `clean_text_sha256`, `typo_text_sha256` | Exact inline strings and hashes; each side may be null when genuinely missing, with availability/reasons |
| `clean_prompt`, `typo_prompt`, `prompt_template_revision` | Exact complete rendered prompt text or `ArtifactRef` plus text hash; nullable with reasons, no reconstruction passed off as archive |
| `clean_query_span`, `typo_query_span`, `prompt_regions` | Explicit full-prompt coordinates for query, context, options and few-shot regions; nullable until independently resolved |
| `gold`, `canonical_gold`, `gold_canonicalization_version`, `valid_labels` | Nullable if absent; numeric gold uses exact rational/decimal representation, choice labels are item-specific |
| `intended_edits`, `archived_actual_edits`, `archived_token_positions` | Nullable archived evidence; no substitution of independently inferred edits |
| `archived_generations` | References described below, including all available arms/windows and missing expected arms |
| `availability`, `reason_codes` | Structured per-component availability and remaining unknowns |

For each text/prompt payload exactly one of inline text or a text-file reference
is used when available. Both are null only when missing/unknown. Inline text and
full prompt remain distinct: query text is not assumed to equal the complete
few-shot prompt. Consumers verify prompt hash and explicit query-span contents;
repeated substrings do not justify choosing the first occurrence. Prompt
regeneration creates separately labeled fresh provenance and requires a declared
protocol change if it changes the exact archived prompt contract.

Each `cohort_membership` entry is `{cohort_id, membership, source_ref}`;
`membership` is `member`, `extra`, or `unknown`. Its `source_ref` is always an
`ArtifactRef`, selecting the expected-ID artifact, the explicit membership
evidence artifact, or an acquisition artifact actually claiming that cohort in
that priority order. Multiple claimants are ordered by canonical JSON of their
artifact references; the representative pair source need not be a claimant. The
pair-level `source_ref` remains a `RecordRef` identifying its exact source row;
do not interchange the two reference types.

`source_identity_provenance` is a nonempty array of acquisition claims. Each entry
contains `source_ref` (`RecordRef`), `pair_identity`, `group_identity`,
`cohort_ids` (sorted unique original membership claims),
`historical_pair_id` (nullable), `source_pair_key_kind`, and
`source_pair_key_kind_explicit` (boolean). Each identity object retains its
`domain`, canonical `payload`, `original_payload`, and `alias_evidence_refs`.
All acquisition claims and alias evidence survive deduplication, ordered by their
canonical JSON. Within one original acquisition identity, contradictory known
historical IDs or explicit key kinds are errors; null historical ID plus a known
ID resolves to that known ID while both original claims remain in provenance.
Other scientific unknown/known metadata is not merged. An inferred key kind is
recomputed from the resolved historical ID unless an explicit kind is available.

Different original pair identities can coalesce only with verified aliases.
The top-level identity/provenance fields summarize the original identity matching
the canonical pair identity if present, otherwise the lexicographically first
canonical-JSON original identity. Within it, prefer an acquisition carrying the
known historical ID, then a canonical original group identity, then canonical-JSON
order. The other source rows remain as unique RecordRefs in
`additional_source_refs`, excluding the representative reference (empty for a
single source row). All acquisition declarations still remain in provenance and
inventory. This representative is not selected by source input order.
Exact-text equality is checked by each component's hash before deduplication.
For each text/prompt component, its inline/reference representation is selected
independently and deterministically: prefer an available inline string, otherwise
the artifact reference first in canonical-JSON order. Keep inline and reference
mutually exclusive. All original acquisition references remain available even
if this representation comes from a different acquisition than the identity
representative. Equivalent payload storage choices cannot make source order
change the normalized manifest bytes.
Different `source_kind` values remain incompatible for one normalized pair; use
separately namespaced intake runs for archived and regenerated acquisition events.

`archived_generations` entries contain `generation_id`, `arm`, `window`,
`availability`, `record_ref`, and `reason_codes`. Available records point into
`archive_generation_records.jsonl` (`rebuttal-archive-generation/v2`), which
contains `generation_id`, `pair_id`, source reference, arm/window, exact `raw_text`
or `raw_text_ref` and `raw_text_sha256`, optional `generated_token_ids`, optional
recorded stopping metadata, `termination`, and nullable archived runtime/parser
provenance. Archive self output may be structurally absent; never invent it.
`answer-audit --manifest` can therefore discover and verify every archived raw
generation without an implicit results directory or extra CLI argument. Missing
raw text produces coverage/missing rows, not fabricated scores.

`pair_manifest.meta.json` (`rebuttal-pair-manifest-metadata/v2`) is the required
sibling of `pair_manifest.jsonl`; it contains manifest, protocol, source-audit and
inventory `ArtifactRef`s and `protocol_sha256`. It binds the protocol and source
coverage even when the manifest has zero rows because no archived pairs were
recovered. It never references intake `run.json`, which instead binds these
outputs; this avoids a provenance-reference cycle. Manifest-only consumers
validate this companion first and then the row-level generation/source links.

`tokenizer_lock.json` (`rebuttal-tokenizer-lock/v2`) has a `models` object keyed
by exact model ID (for example `google/gemma-3-4b-it`). Each value contains
tokenizer ID, immutable revision, tokenizer-file hashes, library version,
special-token IDs, chat-template text/hash (if used), special-token insertion
policy, normalization settings, offset convention, and `padding_side`. Every
field affecting rendered tokenization is fixed. A multi-model manifest uses its
matching entry; an absent model entry makes that pair's token audit `unknown`
while retaining independent text audit results. The lock describes the tokenizer
used for independent audit; `matches_archived_tokenizer` is true/false/unknown
separately. For each archived tokenizer ID, revision, and file-hash value, a
non-string or blank value contributes unknown identity evidence. Any known
nonempty unequal string makes the aggregate result false; otherwise incomplete
evidence makes it null. Known values are compared exactly as stored without
stripping, coercion, or normalization. Unknown archived revision permits a new
pinned-tokenizer audit but prevents claiming it verified the unknown historical
tokenizer. No floating `main` revision qualifies as an immutable lock.

## 5. Answer audit and input audit

PR-02's concrete R1 artifacts, parser scoring contracts, availability handling,
and frozen blind-sampling policy are defined in [answer_audit.md](answer_audit.md).
The primary score uses exact canonical gold; the explicitly labeled public v1
proxy reproduces its unchanged raw-gold comparison and additionally reports an
auxiliary canonical comparison. These are distinct scoring rules, not an
unidentified historical parser or a fresh-baseline experiment.

`answer_audit_records.jsonl` (`rebuttal-answer-audit/v2`) joins `generation_id`
and verified generation `RecordRef` to `pair_id`, task/valid labels, parser ID,
parser code hash/version, extraction status, raw extracted value, canonical value,
candidate spans, reason codes, termination, score status, gold reference and
`gold_match`. Numeric canonical values use reduced integer numerator/positive
denominator strings; decimal/exponent spellings compare without binary floats.
Extractors receive text/task/labels/termination only. Gold comparison is a
separate operation, so rescoring cannot change a saved generation.
Audit/score rows require present complete raw output and an executed parser.
Absent archive text, failed or unfinished fresh execution, and an unavailable
selected parser produce no audit/score row; they are tracked in
`scoring_coverage.jsonl` as specified in §8. For present output with an executed
parser, unavailable gold gives `score_status=not_scored`, `gold_match=null` while
preserving the nonnull extraction status/result. Do not invent an `unextractable`
model output when extraction could not run.

The old parser is `archive_parser` only when its historical code/version is
identified; otherwise identify `public_v1_proxy` with its actual code hash. Run
both parsers on the identical raw generation for every available arm. Store
cohort membership before scoring and emit baseline transitions separately from
fixed-cohort scoring comparisons. The primary-parser fresh-failure ID list is
frozen once for sensitivity analysis; parser-specific failure IDs have distinct
artifact names and cannot replace it.

Answer-review exports allow only `schema_version`, opaque `blind_id`, and `raw_text` alongside the
annotation form. A private mapping stores generation/pair/arm/model/gold/parser
details and sampling strata. The selection policy, per-stratum hash ordering,
fixed sample caps and unaudited counts are frozen before reviewer judgments;
agreement samples are included. No gold or correctness labels are in the reviewer
view. Imported judgments describe answer presence/content before gold comparison.

`input_audit_records.jsonl` (`rebuttal-input-audit/v2`) contains pair/manifest
references, tokenizer lock/hash, exact-text hashes, independent diff and alignment
algorithm versions, edit spans on both sides, changed-word spans, full-prompt
spans, token spans/endpoints, intended-token-hit status, archived-coordinate
agreement, risk flags, and eligibility/reasons per audit dimension. Store each
disjoint edit; preserve the original up-to-four-edit intervention. Token spans
include all subtokens overlapping edited words, so strict offset exclusion cannot
miss a nonfinal subtoken. A word endpoint is the last token of that word, not
the first occurrence of matching token text. Many edits within a word retain
their edit records but produce one explicitly deduplicated write endpoint.
Each side also records its complete prompt token IDs and
`prompt_token_ids_sha256` (canonical JSON array hash) under the selected model's
tokenizer-lock entry. These are checked against the fresh runtime before using
token endpoints, independently of historical-coordinate agreement.

Diff ambiguity, cancellation/no net change, unavailable region annotations, and
unreconstructable endpoints remain explicit states. Do not resolve ambiguity by
testing which alignment improves patch outcomes. Any deterministic tie-breaking
or word-segmentation algorithm is versioned and fixed by PR-03 before real
outcome analysis. Risk flags separately identify stem, option content, option
label, gold-option membership, number, quantity, unit, negation and entity edits;
unknown classification is not false. They do not set semantic labels.

## 6. Blinded annotations and semantic labels

PR-03 concretizes the versioned difference/word-segmentation algorithms, local
tokenizer lock, archived-coordinate adapters, annotation forms and outputs in
[input_audit.md](input_audit.md). Input audit requires sibling
`input_audit_records.meta.json` (`rebuttal-input-audit-metadata/v2`) binding its
record file, manifest and manifest metadata, tokenizer lock, protocol and algorithm
version. Annotation commands verify this chain, including an empty audit.
Command-stage completion does not claim completion of the whole R2 experiment.

Annotation metadata carries batch ID, schema version, frozen selection manifest
hash and stage. Private `annotation_mapping.jsonl` maps a random opaque `blind_id`
to pair ID and source references. Blind IDs must not encode pair IDs, labels,
setting, or outcome. Mapping and audit provenance stay outside reviewer exports.

The Stage A export `annotation_stage_a.jsonl` is an allowlist: `schema_version`, `blind_id`,
`stage="a"`, typo query,
necessary typo context, task instructions and presented options. The label form
adds `rater_id`, `timestamp`, `interpretations`, `ambiguity`, and `rationale`.
No clean text, intended edit, target rule, model, arm, gold, answer generation,
patch outcome, internal path, or private ID enters the export. Necessary context
comes from explicit prompt regions, not the entire few-shot prompt; exemplars
containing answers are not included. The actual question's options stay visible.

`annotation_batch.json` privately records required raters (normally two), chosen
single-rater exception if needed, selected blind IDs, private mapping reference,
instructions version, and export hashes. Stage B requires a lock over completed
Stage A rows for all required `(blind_id,rater_id)` keys in the frozen batch.
`stage_a_lock.json` contains those keys, input-label hash, mapping hash and lock
timestamp. Empty, duplicate, unknown or missing keys fail the lock. An explicit
unassessable judgment counts as completed; a missing judgment does not. Stage A
must be locked for the whole batch before any Stage B material is exported;
raters cannot revise Stage A after seeing clean text.

Stage B export `annotation_stage_b.jsonl` adds clean query/context, the fixed
Stage A response reference and the
semantic-assessment form. It may expose task gold for assessing gold preservation
only after Stage A lock; model, arm and patch outputs remain hidden. Rows include
blind ID, rater ID, timestamp, stage A lock hash, label and rationale. The label
set is `unique_preserved`, `ambiguous`, `task_changed`, `unassessable`.

Adjudication retains all independent A/B judgments, adjudicator ID/timestamp,
disagreement reason, and final label. Import verifies selected IDs, complete
required ratings, lock hashes, unique keys, legal labels and mapping integrity.
Adjudication requires disagreement among the required independent raters;
unanimous and single-rater items reject adjudication rows.
Missing/blank clean queries do not change Stage A selection. They restrict Stage
B `allowed_labels` and every imported rating to explicit `unassessable`, with a
private coverage reason; import never fabricates that judgment for a missing row.
Missing annotation records are reported as missing, never silently labeled
`unassessable`. `semantic_labels.jsonl` holds pair ID, final label, rating mode,
source/judgment references and adjudication status; no patch results are inputs.
`annotation_agreement.json` reports rater count, missing judgments, pre-adjudication
agreement and adjudicated counts. Single-rater mode must not claim inter-rater
agreement.

For Stage B export and annotation import, `--annotation-batch` explicitly names
the private batch JSON rather than relying on a returned label file's directory.
Stage B validates the complete returned A ID/rater inventory against the supplied
Stage A batch, hashes that labels file, and writes an immutable A lock plus a
new Stage B `annotation_batch.json`. The new batch references the prior batch,
private mapping, returned A judgments, A lock and B export by verified hashes.
Only the mapping and coverage are copied into the Stage B directory; the returned
Stage A file and other upstream evidence remain external hash-bound references,
often with absolute paths. The Stage B directory is therefore not a self-contained
export. Original referenced files must remain immutable and available through
import and later consumers; relocation is valid only when the full reference tree
is updated consistently without breaking any recorded hash binding.
Import takes this Stage B batch plus returned independent B/adjudication records,
validates their IDs/rater inventory and referenced A lock, and retains all A/B
judgments and adjudication with hashes in its output metadata. Raters return
only review forms; they do not construct metadata packages or see private refs.

## 7. Runtime lock, donor bank, and global plan

`runtime_lock.json` (`rebuttal-runtime-lock/v2`) has `settings` keyed by setting
ID (`R3`, `R4`). Each selected entry pins model/tokenizer IDs and immutable
revisions/file hashes, complete tokenizer policy, dtype, quantization, attention
backend, cache behavior, generation parameters, effective EOS IDs, library
versions, implementation commit/tree hash, and `device_identity` containing
device type/model/compute capability. `compute_settings` includes relevant
CUDA/driver versions and deterministic/precision settings. `prompt_sha256` and
`prompt_token_ids_sha256` are maps from pair ID to clean/typo hashes for that
setting; donor prompt/token hashes are likewise bound when cross is planned.
The protocol's `runtime_lock_required` list applies to every selected setting
entry. Historical revision unknowns stay in the manifest. Resolve this lock
before GPU forward; observed runtime must match the selected setting's entry.

The top-level field set is exactly `schema_version`, `settings`, and optional
`provenance`. Only `provenance` is non-scientific. Each selected setting entry has
exactly `model_id`, `model_revision`, `model_files`, `tokenizer_id`,
`tokenizer_revision`, `tokenizer_policy`, `generation`, `attention_backend`,
`cache_behavior`, `effective_eos_ids`, `library_versions`, `code_commit`,
`code_tree_sha256`, `device_identity`, `compute_settings`, `prompt_sha256`,
`prompt_token_ids_sha256`, `donor_prompt_sha256`, and
`donor_prompt_token_ids_sha256`. `model_files` is a nonempty map from safe relative
model filenames to declared SHA-256 digests. Planning does not read model weight
bytes, so PR-05 must verify every actually loaded weight/config byte against these
declarations before GPU execution; a CPU-valid plan is not proof of the actual
model runtime. `tokenizer_policy` has exactly `tokenizer_files`, `library`,
`special_token_ids`, `chat_template`, `add_special_tokens`, `normalization`,
`padding_side`, and `offset_convention`. Its tokenizer ArtifactRefs are resolved
relative to the runtime-lock file and their bytes are verified.

`generation` contains every frozen protocol generation key with typed-equal
values and may add explicit resolved JSON controls needed by the runtime.
`cache_behavior` includes typed boolean `use_cache`; `effective_eos_ids` is a
nonempty unique integer array. `library_versions` includes `tokenizers`,
`transformers`, and `torch`. `device_identity` is exactly `type`, `model`, and
`compute_capability`; `compute_settings` includes `cuda_version`, `driver_version`,
`deterministic_algorithms`, and `allow_tf32` plus any other explicitly resolved
precision/determinism settings. CUDA versions and capability are required for a
CUDA declaration. Physical GPU index/UUID/serial, host, PID, shard values,
timestamps, and descriptive provenance do not belong in scientific setting
entries.

Formal planning requires equality between the input-audit tokenizer policy/entry
and the relevant fresh runtime tokenizer policy, and verifies every exact prompt
token-ID hash before accepting endpoints. Donors must pass the same equality and
prompt-token checks. A mismatch stops formal planning for the affected setting;
it is not cured by matching token counts or by trusting archived coordinates.
Missing lock entries stop formal planning, while CPU audit reports may retain
unknown token alignment. Historical tokenizer agreement remains a separate field.
The runtime tokenizer policy must be typed-equal to the PR-03 audit descriptor
after removing only ArtifactRef locations; equal token counts are not enough.
For every selected prompt whose exact text is known, the declared full-prompt text
and complete token-ID hashes are verified even if word alignment is invalid, so
alignment-invalid pairs can still retain independently eligible baselines. A known
empty token-ID array is hashed and then makes the corresponding arm invalid. An
absent full prompt remains unknown for that arm. Missing or mismatching runtime
hashes/policies are global preflight blockers. This check does not assert that a
fresh lock matches an unknown historical tokenizer revision.

The scientific hash projection is `{protocol_sha256, settings}` with only selected
settings, sorted by setting ID. Each setting includes model/tokenizer IDs and
immutable revision/file hashes, tokenizer policy, code commit/tree hash, library
versions, dtype/quantization, attention backend, cache behavior, generation
parameters, effective EOS IDs, device identity, compute settings and prompt text/
token-ID hash maps listed above. Exclude locations/paths, descriptive provenance,
timestamps, host/PID, GPU indices/UUIDs/serials, shard assignments and shard count.
Hash this projection rather than the runtime-lock file bytes. Runtime identity
therefore includes device model while compatible workers share one scientific
hash for the full selected setting configuration.

`donor_bank.jsonl` (`rebuttal-donor-bank/v2`) supplies donor pair/group identity,
task/model/target rule, verified clean prompt reference/hash, aligned edited-word
count and ordered clean endpoints, tokenizer lock, and provenance. All donor
original-problem groups are disjoint from all recipients; recipient variants
under other target rules/models are also excluded. Within matching strata,
order recipients and donors by seed-42 domain-separated identity hashes, breaking
ties by ID, and assign one-to-one without replacement. Fix that assignment once
globally. Remaining recipients are invalid when the bank is too small. Missing
bank marks cross `not_available`. No cyclic or reuse fallback occurs in v2.0.0;
a reuse policy requires a protocol revision and fixed-bank inference disclosure.
The ordering domain is `rebuttal-donor-order/v2`; the payload is
`{seed:42, role:"recipient"|"donor", stratum:{model_id,task,target_rule,aligned_word_count}, pair_id}`.
Compare full hexadecimal digests lexicographically, then `pair_id` for ties.
Zip the ordered recipient and donor lists in each exact stratum. Endpoint/state
order within a pair follows ascending actual edited-word position in its prompt.

Each donor row has exactly `schema_version`, `pair_id`,
`original_problem_group_id`, `task`, `model_id`, `target_rule`,
`clean_prompt_ref`, `clean_prompt_sha256`, `aligned_word_count`,
`clean_endpoints`, `tokenizer_lock_ref`, `input_audit_ref`, and `provenance`.
The prompt and tokenizer-lock fields are ArtifactRefs. `input_audit_ref` is a
RecordRef whose record ID is the verified `input_audit_id`; `provenance` has
exactly `manifest_ref` and `source_ref`, both RecordRefs copied by identity/hash
from the verified audit/manifest chain. `aligned_word_count` is at least one and
equals the strictly ordered `clean_endpoints` length and the independent audit's
aligned-edit count. All prompt bytes, endpoints, group/target/model/task identity,
tokenizer descriptor, and reference hashes are revalidated through the complete
PR-03 chain. Donors may be independently acquired extra pairs and need not be
members of a recipient historical cohort. Distinct donor pair IDs may share one
donor original-problem group; the rule forbids overlap with every selected
recipient group, not mutual disjointness inside the bank. Duplicate donor pair IDs
are invalid.
Provenance RecordRefs may use a different existing location only when artifact
bytes/hash and record ID equal the verified reference. The authoritative pair
still comes from the valid audit/manifest chain; a content alias does not permit
partial relocation of a broken reference graph.

`plan.jsonl` (`rebuttal-plan/v2`) contains one row for every intended
`(pair_id,window,arm)` including ineligible arms. Fields are `plan_id`, `pair_id`,
`original_problem_group_id`, `setting`, `window`, `arm`, `source_pair_id`,
`source_prompt_ref`, `recipient_prompt_ref`, `source_positions`, `write_positions`,
`source_tensor_site`, `write_tensor_site`, `application_phase`,
`expected_applications`, `generation_config_sha256`, `scientific_config_sha256`,
`donor_bank_sha256`, `eligibility`, `reason_codes`, and semantic-analysis labels.
The scientific planning payload for `plan_id` includes these fields except the
ID itself and semantic labels, plus verified input-audit/manifest/protocol hashes.
Reference payloads contribute content hashes and record IDs, not filesystem paths.
Shard allocation and mutable execution state are not part of the payload.

The exact row field set is the preceding list plus `schema_version`,
`protocol_sha256`, `manifest_sha256`, `input_audit_sha256`, `semantic_label`,
`semantic_status`, and `semantic_reason_codes`. Eligibility is one of `valid`,
`invalid`, `unknown`, or `not_available`; evidence missingness remains distinct
from a known-invalid operation. A prompt reference, when present, is exactly
`{artifact:{path,sha256},record_id,field}` where `field` is `clean_prompt` or
`typo_prompt`. It points to the verified manifest record even when the original
prompt representation was inline, preserving the source representation without
copying prompt text into a plan row.

`plan_id` uses domain `rebuttal-plan-row/v2`. Its payload contains every row field
except `plan_id` and the three `semantic_*` fields; ArtifactRef paths are removed
while their SHA-256 values remain. `generation_config_sha256` hashes exactly
`{generation: runtime_entry.generation, effective_eos_ids: ...}`. Semantic labels
are analysis metadata and never alter plan identity or job eligibility.

All listed fields are present even on ineligible rows. `donor_bank_sha256` is
nullable: with no supplied bank it is JSON `null` on every row and in metadata,
including retained cross `not_available` rows. The canonical planning payload
contains that literal null, not an omitted key, empty string or invented digest
of a nonexistent bank. With a supplied bank, use its verified file hash.
Baseline `source_pair_id`, `source_prompt_ref`, tensor sites,
`application_phase` and `expected_applications` are null, and its source/write
positions are known-empty arrays. Valid patch rows require nonnull source and
recipient prompt refs, source pair ID, full coordinate arrays, sites, phase and
application counts. For ineligible patch rows, unresolved source identity,
prompt refs or coordinate arrays may be null with per-field reason codes; retain
known values even when a different prerequisite fails. Fixed window/sites/phase
and expected applications remain populated when known from the protocol, but
may be null if the corresponding operation cannot be specified. An unavailable
recipient prompt is null, never a fabricated reference. Such rows never become
generation jobs. Generation/scientific config hashes and original group IDs must
still be resolved for a formal frozen plan; unresolved global preflight failures
are reported before producing a runnable plan.
A null original-problem group ID on any selected pair, including a clean/typo
baseline, fails global preflight: report all blocking pair IDs and produce no
runnable plan, expected generation grid, or shard assignment. These records
remain available to CPU source/answer audits; do not silently drop them to pass
preflight.

After global preflight passes, baseline clean/typo have `window=null`, empty
positions and no patch source/site; they are each planned once for every
text/runtime-eligible archived pair.
Self/correct require alignment; offset requires all corresponding source and
write endpoints shifted by +2 to be legal; cross additionally requires the fixed
donor. If any offset violates a prompt boundary, any edited-word token span, or
write uniqueness, the entire offset arm is invalid. No endpoint is dropped to
salvage a partially legal arm. Both source and write endpoints move by +2; the
first and last prompt tokens are forbidden, and shifted endpoints may not land on
any token belonging to any edited word. Invalid offset rows retain the complete
candidate coordinate arrays. Patch rows have the fixed window, complete block
output sites, `application_phase="prefill-only"`, one expected application per
selected layer and zero at decode. Semantic labels define analysis subsets and
never remove otherwise eligible baseline/patch generation jobs.

Selection admits only verified `member` records of declared selected cohorts;
`extra` and `unknown` records are excluded and reported. R3 may plan the recovered
verified subset when declared original IDs are missing, but coverage must not call
that the complete original 172. An available R3 full prompt and complete audited
token IDs can keep its baseline eligible when separate exact query text is absent;
alignment and analysis missingness remain explicit. R4 requires all declared
frozen original IDs and nonnull, nonblank clean/typo query texts and full prompts;
it never fills a
shortfall or uses a historical success/count quota. A null group on any selected
pair, donor overlap, tokenizer/runtime mismatch, or other global blocker prevents
every runnable plan row. Missing donor bank is not a global blocker: cross is
`not_available` and other arms continue. A present but empty valid bank makes
matching cross rows invalid for shortage.

`plan.meta.json` (`rebuttal-plan-metadata/v2`) is a required sibling of
`plan.jsonl`. It stores plan file hash, protocol/manifest/input-audit/semantic-label
and runtime-lock `ArtifactRef`s, nullable donor-bank reference/hash, selected
experiments, full intended row counts, valid expected generation IDs, invalid and
unavailable IDs/reasons, and freeze timestamp. All dependencies are transitively
hashed. No reference cycle is allowed: the metadata references the plan, while
plan rows bind upstream inputs rather than their own metadata hash.

Its exact fields are `schema_version`, `freeze_timestamp`, `plan_ref`,
`protocol_ref`, `protocol_sha256`, `manifest_ref`, `manifest_metadata_ref`,
`input_audit_ref`, `input_audit_metadata_ref`, `semantic_labels_ref`,
`semantic_labels_run_ref`, `runtime_lock_ref`, `donor_bank_ref`,
`donor_bank_sha256`, `experiments`, `scientific_config_sha256`,
`intended_row_count`, `row_counts`, `expected_generation_ids`,
`ineligible_plan_rows`, `preflight_ref`, `coverage_ref`,
`donor_assignment_ref`, `expected_generations_ref`, and `shard_assignment_ref`.
`row_counts` is sorted by setting/arm/eligibility and carries `count`;
`ineligible_plan_rows` retains `plan_id`, eligibility (including unknown and
not-available), and reasons. Own output ArtifactRefs are relative; verified
upstream references remain absolute. With no bank, `donor_bank_ref` and
`donor_bank_sha256` are literal null everywhere rather than a digest of an absent
file.

`expected_generations.jsonl` has schema
`rebuttal-expected-generation/v2` and exactly `schema_version`, `generation_id`,
`plan_id`, `pair_id`, `setting`, `arm`, `window`, and
`scientific_config_sha256`, for valid rows only. `generation_id` uses the existing
`rebuttal-generation/v2` domain over
`{plan_id,scientific_config_sha256}`. `preflight.json` records schema, status,
sorted blockers, blocking pair IDs and experiments. `planning_coverage.json`
retains selected IDs, per-pair and source-cohort coverage, and counts. A logical
global preflight failure publishes only `preflight.json`,
`planning_coverage.json`, `protocol.json`, and `run.json`, returns status 1, and
does not publish a plan/grid/shard assignment. Structurally malformed or
hash-invalid inputs raise before publication.

Formal plans require resolved original-problem group identity before sharding.
`shard_assignment.json` is separately generated for `num_shards` in 1..6. Map
each group by the integer value of SHA-256 over
`"rebuttal-shard/v2\n" + original_problem_group_id` modulo `num_shards`, keeping
all variants/arms/windows together. Changing shard count changes only this
allocation, not scientific plan rows, donor assignment or cohorts. Donor input
may reside outside the recipient shard. Shard workers never replan.

The passing output set is `plan.jsonl`, `plan.meta.json`, `preflight.json`,
`planning_coverage.json`, `donor_assignment.json`,
`expected_generations.jsonl`, `shard_assignment.json`, `protocol.json`, and
`run.json`. Publication is atomic into an absent or empty output directory and
does not replace existing artifacts. `load_plan` treats `plan.meta.json` as the
scientific root and revalidates every metadata-bound scientific output and
immutable upstream reference by reconstructing the expected global plan; the
operational planning `run.json` is not part of that read-back root. Historical
producer code-identity references are validated as recorded provenance and are
not compared with upgraded current code. `load_plan` is never a shard-local
replanner. The planning run uses
`command="plan"`, `mode="cpu-planning"`, `experiment_id=null`, selected
`experiment_ids`, `experiment_status="not_run"`, and
`planning_status="complete"|"blocked"`. Its expected/completed IDs mean frozen
plan rows, not executed model jobs; `model_execution_performed` is false.

## 8. Fresh generations, scores, checkpoints, and run metadata

`generation_records.jsonl` (`rebuttal-generation-record/v2`) contains
`generation_id`, `completed_attempt_id`, `plan_id`, `pair_id`, arm/window, generated token IDs, exact decoded
generated suffix and text hash, effective EOS IDs, generated length, termination
and supporting stop metadata, runtime status, scientific config hash, worker
telemetry reference, source/write positions, hook counts by layer/phase, and
integrity diagnostics. Decode only the generated suffix, not the prompt. EOS on
token 512 is `eos`; length 512 without stopping evidence is not sufficient to
infer `length-cap` for an archive. Unsupported archive stop labels are preserved
as raw metadata; without supported observed stop metadata, `termination=unknown`.
They do not erase a stopping reason established by supported observation flags.
Fresh canonical generation records
have `runtime_status=complete`; there is at most one row per `generation_id`,
referencing its successful attempt. Failure history is never appended as duplicate
generation rows or decoded/scored as a complete model answer.

`attempt_records.jsonl` (`rebuttal-generation-attempt/v2`) has one row per unique
`attempt_id`, with `generation_id`, `attempt_number`, plan/pair identity, runtime
status, start/end times, worker telemetry, reason/exception metadata, and nullable
partial-output diagnostics. The pair checkpoint has a single writer and reserves
the next monotonic attempt number durably before execution; it never reuses a
number after interruption or resume. Running attempts transition atomically to
terminal complete/failed records under the same attempt ID, rather than appending
duplicate IDs. Resume records an interrupted attempt as failed with that reason
before reserving a new attempt; it must not retry a still-live writer's attempt.
Failed attempts remain in this history. A completed attempt and the canonical
generation record are published through the same atomic pair-checkpoint update;
recovery finishes an interrupted publication without generating a second output.
Once canonical output exists, resume reuses it and cannot replace it with another
attempt based on the extracted answer.

Scoring lives in separate `score_records.jsonl`, using the §5 answer-audit shape
and generation references, one row per generation/parser version. Gold match and
archive/audit parses are views joined to generation records, not mutable fields
inside immutable raw output. Complete output with unextractable/ambiguous answer
and available gold is scored with `gold_match=false`. Missing gold after an
executed parser uses `score_status=not_scored`, `gold_match=null` and explicit
reasons while preserving the actual extraction result. Missing/failed outputs
and unavailable selected parsers have no audit/score row.

`scoring_coverage.jsonl` (`rebuttal-scoring-coverage/v2`) has one entry per expected
source slot or planned valid job and parser. It contains `coverage_id`, nullable
`plan_id`, nullable `source_slot` (`{pair_id, arm, window}` for archive), verified
plan/manifest reference, requested parser ID and nullable resolved parser version,
nullable known `generation_id`,
nullable generation/score references, and coverage reasons. `coverage_id` hashes
domain `rebuttal-scoring-coverage/v2` with
`{plan_id, source_slot, parser_id, parser_version}`; exactly one of plan ID and
source slot is nonnull. `scoring_status` is `scored`, `not_scored`, `output_missing`,
`execution_pending`, `execution_failed`, or `parser_unavailable`. Absent/failed
output references and unavailable score references stay null, and no extraction
fields are invented. This artifact accounts for the full
expected grid without treating runtime failures as parser failures or model errors.

`scientific_config_sha256` hashes the explicit projection in §7, including device
model and code. `worker_telemetry` records host, PID, physical GPU index,
UUID/serial, driver diagnostics, times and resource usage separately. Same-model
GPUs with different physical indices/UUIDs share scientific identity when their
compute configuration agrees. A device-model/backend/dtype/code mismatch against
the selected setting's lock requires a separate run and cannot be pooled as one
locked experiment.

A pair checkpoint contains the frozen complete expected set of that pair's valid
arms/windows, canonical successful outputs, attempt history and durable counters,
and input/protocol/plan/code/
scientific-runtime hashes. Write atomically; incomplete pairs remain incomplete.
Resume checks every hash and keeps already completed arms and failure history;
it may finish missing/retry failed execution jobs without selecting by answer
correctness. No arm regeneration is justified by an inconvenient model answer.
Pair completion requires all planned valid jobs, including shared R4 baselines.
Smoke uses an independent plan/run namespace and never enters formal reduction.

Every command writes `run.json` (`rebuttal-run/v2`) with command/mode, start/end
times, invocation arguments, code identity, input/config/output `ArtifactRef`s,
run status, expected/completed/failed/missing IDs, and reasons. GPU runs additionally
bind the global plan, scientific runtime and shard allocation/worker telemetry.
No observation-dependent success-rate acceptance condition exists. Hook or
self-copy integrity violations invalidate scientific release for the affected
runtime even when output files are complete; they are not counted as model harm.

## 9. Reduction, reports, and required command inputs

`report.json` (`rebuttal-report/v2`) includes verified input/run references,
protocol/parser identities, mode (`cpu-audit` or `fresh-with-audit`), source coverage,
experiment/table statuses, fixed cohort ID artifact references, planned and
observed execution grid, missing/duplicate/unexpected rows, and table references.
Unknown/invalid plans do not enter the expected generation grid; valid jobs with
missing outputs are runtime missingness. Duplicate or unexpected generation IDs
are contract errors, never an opportunity to choose the favorable row.

Each estimate stores setting/window/parser, named cohort and IDs/hash, numerator,
denominator, expected denominator, runtime-complete denominator, effect in
percentage points, CI method/level, bootstrap seed/replicates/cluster key,
discordant counts, optional test eligibility and limitations. Preserve full
precision in JSON. Empty subsets use `effect=null`, `ci=null`, `n=0`, not 0%.
Partial runtime subsets are explicitly descriptive and do not masquerade as the
complete planned estimand. Rates under a fixed parser retain complete
unextractable/ambiguous outputs in the denominator.

Use C_offset for correct−offset, C_cross for correct−cross, and C_three for
direct three-arm rate tables. Freeze primary-parser C_fresh_failure IDs for
sensitivity scoring; separately report parser-specific baseline transitions.
Bootstrap original-problem groups 10,000 times with seed42, retaining all variants
and arms together; each replicate computes the pooled pair-weighted rate
difference. Use percentile 95% intervals with linear quantiles. R3 has a fixed
two-comparison Holm family; an unavailable contrast has no reported p-value and
contributes p=1 only to adjustment. Optional McNemar requires independent pairs;
R4/R5 are separate diagnostics. Donor inference is conditional on the fixed bank.

`claims.jsonl` contains `claim`, `supporting_table`, `n`, `effect`, `ci`, `scope`,
`limitation`, `status` and evidence references. Explicitly unexecuted R4/R5 experiments produce
`not_run`, not empty success tables. CPU-only reporting creates the available
T0–T3 tables and archived-results panel with explicit provenance. Fresh T4 and
optional T5/T6 are `not_run` only when no execution occurred. Incomplete executions
have `partial` experiment/table status: any observed-subset estimates are labeled
descriptive and the complete planned primary estimate is unavailable. Integrity
violations use `invalid` with reasons. A CPU report does not require a GPU plan
or runtime lock; it must not infer absence of execution from absent input alone,
and reports unavailable execution evidence as unknown coverage.

The following contracts are planned CLI interfaces; none is implemented in PR-00.
Each command requires `--output-dir` and validates transitive artifact references.
CPU imports must not initialize a GPU model.

| Command | Required explicit input flags | Additional/conditional inputs |
|---|---|---|
| `intake` | `--archive-index`, `--protocol` | Missing sources allowed as inventory findings |
| `answer-audit` | `--manifest` | Required sibling manifest metadata provides protocol/coverage even for zero rows; raw generation refs in rows |
| `input-audit` | `--manifest`, `--tokenizer-lock` | Pinned audit tokenizer may differ from unknown archived tokenizer; difference reported |
| `annotation-export --stage a` | `--audit` | Writes private batch/mapping and allowlisted reviewer export |
| `annotation-export --stage b` | `--audit`, `--stage-a-labels`, `--annotation-batch` | Stage A batch binds selected IDs/raters and mapping; freeze complete A labels before export |
| `annotation-import` | `--audit`, `--labels`, `--annotation-batch` | Stage B batch references private mapping, prior judgments and A lock; retain A/B/adjudication sources |
| `plan` | `--manifest`, `--input-audit`, `--labels`, `--runtime-lock`, `--protocol`, `--experiments` | `--donor-bank` optional; missing bank disables cross only |
| `run` | `--plan`, `--shard-index`, `--num-shards` | Required sibling `plan.meta.json`; `--smoke` uses separate smoke plan; `--resume` checks pair checkpoints |
| `reduce` (CPU) | `--intake-run` | `--answer-audit-run`, `--input-audit-run`, `--semantic-audit-run` add available evidence; missing stages reported |
| `reduce` (fresh) | CPU inputs plus `--plan`, `--shard-root` | Both GPU flags required together; verify global plan and every discovered shard run/output hash |

Artifact discovery is explicit: audit rows reference their manifest; manifest
metadata binds protocol/source coverage and rows reference raw sources;
annotation inputs name their private batch and lock artifacts; plan metadata
references all frozen upstream artifacts;
reduction follows only supplied run manifests and their verified references.
Directory-name guesses must not silently select another run or parser version.
If required data is unknown, commands emit coverage/preflight findings and do not
advertise unsupported tables, formal GPU readiness, or exact historical replication.
