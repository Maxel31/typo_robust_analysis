# R3/R4 global planning (PR-04)

`typo-cot rebuttal-v2 plan` is CPU-only. It verifies the complete PR-01/PR-03
reference chain, freezes arm coordinates and independent donor assignments, and
publishes one global plan before sharding. It does not import a model framework,
load model weights, execute an arm, score a generation, or reduce results. The
scientific protocol remains revision `2.0.0`; PR-04 concretizes its planning
contract without adding a cohort, arm, window, or outcome-dependent criterion.

Implementation and synthetic CPU acceptance fixtures are available. The original
172/97-item archives have not been recovered here, no human labels have been
collected, and no real-model smoke or formal experiment has run. A CPU-valid
runtime declaration is not evidence that an actual GPU process loaded the
declared model bytes.

## Run

From the repository root:

```bash
uv run --project projects/typo-cot typo-cot rebuttal-v2 plan \
  --manifest /path/to/intake/pair_manifest.jsonl \
  --input-audit /path/to/input-audit/input_audit_records.jsonl \
  --labels /path/to/semantic-audit/semantic_labels.jsonl \
  --runtime-lock /path/to/runtime_lock.json \
  --protocol projects/typo-cot/configs/rebuttal_v2/protocol.json \
  --experiments R3 \
  --donor-bank /path/to/donor_bank.jsonl \
  --num-shards 1 \
  --output-dir /path/to/plan
```

`--experiments` accepts `R3`, `R4`, or both; input order is canonicalized by
setting ID. `--donor-bank` is optional. Without it, cross rows remain in the plan
as `not_available` while baseline, self, correct, and offset planning continues.
`--num-shards` defaults to 1 and must be in 1..6. Changing it changes only
`shard_assignment.json`, not plan rows, IDs, donor assignment, or the expected
generation grid. Use an absent or empty output directory; publication is atomic
and never replaces an existing artifact set.

The required siblings and transitive references are part of each input:

- the manifest requires `pair_manifest.meta.json` and its source/protocol tree;
- the input audit requires `input_audit_records.meta.json`, its exact manifest,
  and the PR-03 tokenizer lock;
- semantic labels require the sibling annotation-import `run.json` and complete
  human Stage A/B/adjudication provenance;
- the runtime lock verifies local tokenizer ArtifactRef bytes and complete prompt
  declarations; and
- a donor bank, when supplied, references complete independently verified PR-03
  input-audit/manifest/source chains.

Historical producer code references are validated as historical declarations;
they are not compared with upgraded current source files. Every consumed input
remains immutable and available at its recorded location. The artifact graph is
not a self-contained directory that can be moved one file at a time.

## Runtime lock

The normative field contract is [schemas.md §7](schemas.md#7-runtime-lock-donor-bank-and-global-plan).
The illustrative shape below omits no required setting keys, but every placeholder
must be replaced with the real typed value. SHA strings stand for 64 lowercase
hex characters and revisions/`code_commit` for immutable 40-hex commits.

```json
{
  "schema_version": "rebuttal-runtime-lock/v2",
  "settings": {
    "R3": {
      "model_id": "google/gemma-3-4b-it",
      "model_revision": "<40-hex-commit>",
      "model_files": {
        "config.json": "<sha256>",
        "model-00001-of-00002.safetensors": "<sha256>",
        "model-00002-of-00002.safetensors": "<sha256>"
      },
      "tokenizer_id": "google/gemma-3-4b-it",
      "tokenizer_revision": "<40-hex-commit>",
      "tokenizer_policy": {
        "tokenizer_files": {
          "tokenizer.json": {"path": "tokenizer.json", "sha256": "<sha256>"}
        },
        "library": {"name": "tokenizers", "version": "0.22.2"},
        "special_token_ids": {"eos_token_id": 1},
        "chat_template": null,
        "add_special_tokens": false,
        "normalization": null,
        "padding_side": "left",
        "offset_convention": "unicode-codepoint-half-open"
      },
      "generation": {
        "dtype": "bfloat16",
        "quantization": false,
        "do_sample": false,
        "num_beams": 1,
        "max_new_tokens": 512,
        "padding_side": "left",
        "batch_size": 1
      },
      "attention_backend": "sdpa",
      "cache_behavior": {"use_cache": true, "implementation": "dynamic"},
      "effective_eos_ids": [1],
      "library_versions": {
        "tokenizers": "0.22.2",
        "transformers": "<version>",
        "torch": "<version>"
      },
      "code_commit": "<40-hex-commit>",
      "code_tree_sha256": "<sha256>",
      "device_identity": {
        "type": "cuda",
        "model": "<device-model>",
        "compute_capability": "<major.minor>"
      },
      "compute_settings": {
        "cuda_version": "<version>",
        "driver_version": "<version>",
        "deterministic_algorithms": true,
        "allow_tf32": false,
        "matmul_precision": "highest"
      },
      "prompt_sha256": {
        "<recipient-pair-id>": {"clean": "<sha256>", "typo": "<sha256>"}
      },
      "prompt_token_ids_sha256": {
        "<recipient-pair-id>": {"clean": "<sha256>", "typo": "<sha256>"}
      },
      "donor_prompt_sha256": {"<donor-pair-id>": "<sha256>"},
      "donor_prompt_token_ids_sha256": {"<donor-pair-id>": "<sha256>"}
    }
  },
  "provenance": {"notes": "non-scientific operator notes only"}
}
```

Top-level scientific keys are closed. Physical GPU indices, UUIDs/serials, host,
PID, timestamps and shard values do not enter a setting. Device model and CUDA/
driver/precision settings do enter the scientific hash. Tokenizer ArtifactRef
locations are excluded from that hash while their bytes and digests remain bound.
The runtime tokenizer policy must exactly match the PR-03 audit policy after only
those locations are removed.

`model_files` contains safe relative filenames and digest declarations. PR-04
does not read weights. Before PR-05 performs any forward pass, it must verify all
actually loaded model weights and configuration bytes against these declarations.
Matching a CPU plan does not establish actual model/runtime compatibility.

For known prompts, text and complete token-ID hashes are required even when word
alignment is invalid; this preserves baseline planning independently of patch
eligibility. A known empty ID array is hashed and the corresponding arm is
invalid. An unavailable prompt is `unknown`. Equal token counts do not substitute
for exact hashes. Fresh runtime identity does not fill an unknown archived
revision or claim historical agreement.

## Manual donor-bank assembly

PR-04 validates and assigns a supplied bank; it does not search a dataset or
construct the candidate bank. Assemble that bank before looking at gold, answer
correctness, patch success, or any other outcome:

1. Acquire candidate pairs independently of the selected recipient outcomes.
2. Run normal PR-01 intake and PR-03 input audit on those pairs with documented
   provenance. They may be verified extra pairs and need not belong to the
   recipient historical cohort.
3. Keep only rows with known nonnull original-problem group and target rule, valid
   independent alignment, a verified clean full prompt, and at least one aligned
   word.
4. Exclude every original-problem group used by any selected recipient, including
   variants under another model or target rule. Do not merely filter the current
   matching stratum.
5. Write the exact raw clean-prompt bytes to the referenced prompt artifact and
   construct the row below from verified pair/audit fields. Resolve every path
   relative to the bank file.

```json
{
  "schema_version": "rebuttal-donor-bank/v2",
  "pair_id": "<verified-donor-pair-id>",
  "original_problem_group_id": "<verified-group-id>",
  "task": "gsm8k",
  "model_id": "google/gemma-3-4b-it",
  "target_rule": "<verified-target-rule>",
  "clean_prompt_ref": {"path": "prompts/donor.txt", "sha256": "<sha256>"},
  "clean_prompt_sha256": "<same-sha256>",
  "aligned_word_count": 1,
  "clean_endpoints": [37],
  "tokenizer_lock_ref": {"path": "audit/tokenizer_lock.json", "sha256": "<sha256>"},
  "input_audit_ref": {
    "artifact": {"path": "audit/input_audit_records.jsonl", "sha256": "<sha256>"},
    "record_id": "<verified-input-audit-id>"
  },
  "provenance": {
    "manifest_ref": {
      "artifact": {"path": "intake/pair_manifest.jsonl", "sha256": "<sha256>"},
      "record_id": "<verified-donor-pair-id>"
    },
    "source_ref": {
      "artifact": {"path": "source/pairs.jsonl", "sha256": "<sha256>"},
      "record_id": "<verified-source-record-id>"
    }
  }
}
```

A provenance RecordRef may use a different existing path only when its artifact
bytes/hash and record ID equal the verified reference. The authoritative pair is
still returned from the fully validated audit/manifest chain; this content alias
does not permit a broken reference graph to be relocated one file at a time.

Duplicate donor pair IDs are rejected. Distinct donor pairs may share a donor
group; v2 forbids overlap between donor groups and all selected recipient groups,
not mutual disjointness within the bank. Assignment is independently hash-ordered
inside exact `(model_id, task, target_rule, aligned_word_count)` strata and is
one-to-one without replacement. A supplied empty/short bank produces invalid
cross rows for shortage. There is no cyclic or reuse fallback.

## Cohort and arm construction

Only verified `member` rows of the selected archived cohort enter planning.
`extra` and `unknown` memberships remain in coverage and are not selected. R3 may
plan a recovered verified subset when original IDs remain missing, but neither
the plan nor documentation may call it complete recovery of the original 172.
When an R3 full prompt and its complete audited token IDs are available, its
baseline may remain eligible even if the separate exact query text is missing;
alignment-dependent arms and analysis retain the resulting unknown evidence.
R4 requires all declared frozen original IDs and nonblank exact `clean_text`,
`typo_text`, `clean_prompt`, and `typo_prompt`; it does not replace a missing item
or enforce a historical success/count quota.

A null original-problem group on any selected pair, recipient/donor group or pair
overlap, unresolved runtime identity, tokenizer mismatch, or required hash
mismatch blocks the entire global plan. A logical block publishes coverage and
preflight only. Structurally malformed or hash-invalid input raises without
publication.

After preflight, clean and typo each have one row with `window=null`. For each
frozen window there is one self, correct, offset and cross row: six rows per R3
pair and ten per R4 pair. Ineligible/unknown/not-available arms remain explicit
rows, while `expected_generations.jsonl` contains valid rows only. Semantic labels
are analysis metadata and never remove baseline or patch jobs.

Offset moves every clean source and typo write endpoint by +2. Every shifted
endpoint must be inside the prompt excluding first/last tokens, outside every
token of every edited word, and unique on its side. One failure invalidates the
whole arm; no endpoint is dropped, and the full candidate arrays remain recorded.

Donor order uses domain `rebuttal-donor-order/v2`, seed 42, role, exact stratum,
and pair ID. Full hex digests and then IDs are sorted independently before
zipping. `plan_id` uses `rebuttal-plan-row/v2`; semantic fields and ArtifactRef
paths are excluded, but content hashes remain. Sharding uses the frozen group hash
and occurs only after the global plan and donor assignment are fixed.

## Eligibility and reason codes

Statuses have operational, not outcome, meanings:

- `valid`: all known prerequisites for this arm are satisfied;
- `invalid`: evidence is present and proves the arm cannot follow the contract;
- `unknown`: required evidence is absent or unresolved, without asserting
  invalidity; and
- `not_available`: the optional donor bank was not supplied, so only cross is not
  available and other arms continue.

Common reason-code explanations follow. The JSON keeps codes, not a new prose
field:

| Code or suffix | Meaning |
|---|---|
| `source_cohort_not_declared`, `source_cohort_expected_ids_unknown` | Selected cohort declaration or exact-ID evidence is missing. |
| `r4_declared_original_ids_not_fully_recovered`, `r4_*_unavailable` | R4 lacks a frozen original ID or one of its required exact text/prompt components. |
| `original_problem_group_id_unknown` | A selected pair cannot be safely grouped or sharded. |
| `*_prompt_unknown` | Exact full prompt evidence is absent; arm eligibility remains unknown. |
| `*_prompt_blank`, `*_prompt_token_sequence_empty` | Known prompt or complete token sequence is unusable, so the arm is invalid. |
| `input_alignment_unknown`, `input_alignment_invalid`, `no_aligned_edits` | Independent PR-03 coordinates are unavailable, known invalid, or empty. |
| `*_offset_outside_prompt_interior` | A +2 endpoint reaches the forbidden first/last token or leaves the prompt. |
| `*_offset_overlaps_edited_token` | A +2 endpoint lands on any independently edited-word token. |
| `duplicate_source_positions`, `duplicate_write_positions` | Shifted positions collapse; the whole offset arm is invalid. |
| `target_rule_unknown`, `aligned_word_count_zero` | Recipient donor stratum cannot be known without inventing metadata. |
| `donor_bank_insufficient` | A bank exists but the exact stratum has too few unused donors. |
| `donor_bank_not_available` | No bank was supplied; cross has `not_available` precedence. |
| `donor_pair_id_overlaps_selected_recipient`, `donor_original_problem_group_overlaps_selected_recipients` | Bank independence fails global preflight. |
| `input_audit_tokenizer_descriptor_*`, `donor_tokenizer_descriptor_*` | Runtime and independently audited tokenizer policies are missing or differ. |
| `*_runtime_prompt_sha256_*`, `*_runtime_prompt_token_ids_sha256_*` | Runtime text/token declarations are missing or differ from verified full prompts. |
| `source_pair_id_unresolved`, `*_prompt_ref_unresolved`, `*_positions_unresolved` | An ineligible patch row preserves a specifically unresolved source/reference/coordinate. |
| `semantic_label_not_available` | Analysis label is absent; this does not change generation eligibility. |

## Rows and outputs

The exact normative fields are in [schemas.md §7](schemas.md#7-runtime-lock-donor-bank-and-global-plan).
A `rebuttal-plan/v2` row has exactly:

```text
schema_version, plan_id, pair_id, original_problem_group_id, setting, window,
arm, source_pair_id, source_prompt_ref, recipient_prompt_ref, source_positions,
write_positions, source_tensor_site, write_tensor_site, application_phase,
expected_applications, generation_config_sha256, scientific_config_sha256,
donor_bank_sha256, protocol_sha256, manifest_sha256, input_audit_sha256,
eligibility, reason_codes, semantic_label, semantic_status,
semantic_reason_codes
```

A nonnull PromptRef is
`{artifact:{path,sha256},record_id,field}` with `field` equal to `clean_prompt` or
`typo_prompt`. Baselines have null source identity/sites/phase/applications and
empty coordinate arrays. Patch rows use the complete-decoder-block-output site,
`prefill-only`, and per-layer application maps such as
`{"prefill":{"0":1,...},"decode":{"0":0,...}}` for the frozen window.

`plan.meta.json` has exactly:

```text
schema_version, freeze_timestamp, plan_ref, protocol_ref, protocol_sha256,
manifest_ref, manifest_metadata_ref, input_audit_ref,
input_audit_metadata_ref, semantic_labels_ref, semantic_labels_run_ref,
runtime_lock_ref, donor_bank_ref, donor_bank_sha256, experiments,
scientific_config_sha256, intended_row_count, row_counts,
expected_generation_ids, ineligible_plan_rows, preflight_ref, coverage_ref,
donor_assignment_ref, expected_generations_ref, shard_assignment_ref
```

`rebuttal-expected-generation/v2` rows have exactly `schema_version`,
`generation_id`, `plan_id`, `pair_id`, `setting`, `arm`, `window`, and
`scientific_config_sha256`. `preflight.json` has `schema_version`, `status`,
`blockers`, `blocking_pair_ids`, and `experiments`. `planning_coverage.json` has
`schema_version`, `selected_pair_ids`, `pair_coverage`,
`source_cohort_coverage`, and `counts`. `shard_assignment.json` has
`schema_version`, `num_shards`, and group rows containing
`original_problem_group_id`, `shard_index`, sorted `pair_ids`, and sorted
`plan_ids`.

A passed plan directory contains:

- `plan.jsonl` and required `plan.meta.json`;
- `preflight.json` and `planning_coverage.json`;
- `donor_assignment.json` (including the explicit unavailable state when no bank
  exists);
- valid-only `expected_generations.jsonl`;
- `shard_assignment.json`;
- the exact `protocol.json`; and
- `run.json`.

With no donor bank, `donor_bank_sha256` is literal JSON null in every row and
metadata, while `donor_bank_ref` is null in metadata. The metadata binds every
upstream audit/run reference, the global
scientific hash, sorted row counts, every expected generation ID, and all
non-valid rows/reasons. It references the plan; plan rows never reference their
own metadata, avoiding a cycle.

On logical preflight failure, only `preflight.json`,
`planning_coverage.json`, `protocol.json`, and `run.json` are published and the
command exits 1. Only the diagnostic report completed; `run.json` records
`planning_status="blocked"`, `run_status="failed"`, and
`experiment_status="not_run"`, not a model result. A passing run has
`planning_status="complete"`; expected/completed IDs denote frozen plan rows, not
executed generation jobs. `model_execution_performed` is always false.

`load_plan` is read-only. It uses `plan.meta.json` as the scientific root, verifies
every metadata-bound scientific output and immutable upstream reference,
reconstructs the same global plan, and compares the canonical rows, metadata,
expected grid, donor assignment, and shard assignment. The operational planning
`run.json` is not a required read-back input. Historical producer code identity
is preserved as provenance rather than compared with upgraded current code.
Workers must consume that frozen global result. They do not select a shard-local
cohort, reassign donors, or replan after observing execution state.
