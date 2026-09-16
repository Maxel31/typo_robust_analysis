# R0 archive intake (PR-01)

`typo-cot rebuttal-v2 intake` is CPU-only. It verifies archived inputs and produces
a new v2 manifest without requiring any historical success count. It does not
generate answers, reconstruct historical prompts, tokenize inputs, rescore
answers, or claim that the archived cohort was recovered merely because counts
agree. Other `rebuttal-v2` commands are not yet implemented.

## Run

From the repository root:

```bash
uv run --project projects/typo-cot typo-cot rebuttal-v2 intake \
  --archive-index /path/to/archive_index.json \
  --protocol projects/typo-cot/configs/rebuttal_v2/protocol.json \
  --output-dir /path/to/new-intake
```

Use a new or empty output directory. Existing results are never overwritten.
The index follows [schemas.md §3](schemas.md#3-archive-index-normalized-sources-and-intake).
Its source paths and artifact references resolve relative to the file containing
them. Input SHA-256 mismatches, conflicting identities, malformed JSON and
unsupported protocol revisions are errors. Missing sources and unsupported
source layouts are reported as missing/unsupported coverage, not fabricated data.

A successfully completed inventory can have `run_status=complete` while a cohort
has `coverage_status=partial` or `unknown`. Command success does not mean that the
archive, subsequent audits, or model experiments are complete.

## Explicit adapters

The initial adapters use UTF-8 JSONL. Every nonempty line is an object with a
unique `source_record_id`. Duplicate keys and nonfinite JSON numbers are rejected.
The original file is retained as an immutable, hash-verified source reference.
Unknown adapter names are not guessed from filenames or field similarity.

| Source role | `format` | `adapter_version` and row `schema_version` |
|---|---|---|
| `pairs` | `jsonl` | `rebuttal-source-pair/v2` |
| `generations` | `jsonl` | `rebuttal-source-generation/v2` |

Source entries still require the provenance, availability and reason fields in
schemas.md, including explicit nulls for unknown source/model/tokenizer revisions.
Source kinds remain `submitted-archive`, `public-regeneration`, or `fresh-audit`;
the adapter never converts one kind into another. Other source roles can be
inventoried without pretending that they are pair or generation records.

### Pair rows

Required fields are `schema_version`, `source_record_id`, `source_pair_key`,
`task`, and `model_id`. Example (illustrative data, not an original paper item):

```json
{
  "schema_version": "rebuttal-source-pair/v2",
  "source_record_id": "record-001",
  "source_pair_key": "archived-pair-001",
  "historical_pair_id": "archived-pair-001",
  "cohort_ids": ["R3"],
  "dataset_id": "gsm8k",
  "split": "test",
  "original_problem_id": "original-001",
  "task": "gsm8k",
  "model_id": "google/gemma-3-4b-it",
  "target_rule": "max-logit",
  "perturbation_id": "edit-001",
  "clean_text": "How many apples?",
  "typo_text": "How many aplpes?"
}
```

Optional archived fields correspond to the manifest contract: dataset/model/
tokenizer revisions, exact text and prompts, query spans and prompt regions,
gold and its canonicalization provenance, valid choice labels, saved edits and
token positions, and archived runtime metadata. Omitted archived evidence becomes
null with availability reasons, not a reconstructed historical value. Query text
does not automatically become a full prompt. Text whitespace, Unicode, and line
endings are preserved exactly.

Supplied `canonical_gold` is checked without inferring an answer. For GSM8K it
uses `{numerator, denominator}` with integer strings in reduced form and a
positive denominator. For choice tasks it is a label string, checked against
`valid_labels` when that set is known; the initial choice validator supports
MMLU-Pro. Other tasks may retain raw gold but cannot declare an unsupported
canonical gold representation. A supplied canonical value also requires
its `gold_canonicalization_version`; missing gold stays missing.

Text/prompt payloads may use an inline string or a corresponding `*_ref` with
`{path, sha256}`, not both. A supplied `*_sha256` must agree with the exact UTF-8
text. An explicit query span must match the query at that position in the full
prompt; intake does not choose the first repeated occurrence.

`expected_generations` can enumerate archived slots as `{arm, window}` objects
with an optional `source_generation_key`. Missing slots remain missing. If the
archive did not enumerate its generation grid, intake does not infer a complete
historical grid from the current six-arm fresh protocol or aggregate counts.
Omission/null means an unknown grid; an explicit empty array means known empty.

### Generation rows

Required identity fields are `schema_version`, `source_record_id`,
`source_generation_key`, `source_pair_key`, `arm`, and `window` (null for baseline
outputs, a two-integer half-open interval for patch outputs). Example:

```json
{
  "schema_version": "rebuttal-source-generation/v2",
  "source_record_id": "generation-record-001",
  "source_generation_key": "archived-generation-001",
  "source_pair_key": "archived-pair-001",
  "arm": "correct",
  "window": [0, 6],
  "raw_text": "The answer is 42."
}
```

Use `raw_text` or `raw_text_ref` for actual saved output. Missing output is not
expanded from a count or converted into an empty answer. Saved token IDs and stop
metadata are optional. Unsupported/unknown stopping reasons remain unknown;
output length alone does not establish a historical token-cap termination.
For this adapter, `stopping_metadata.eos_observed=true` establishes EOS, including
EOS at the cap. A length-cap observation requires `eos_observed=false` and
`length_cap_reached=true`. Other saved stop metadata remains raw evidence and
does not establish either stopping reason automatically.

Optional `archived_scoring` preserves the old parser's recorded judgments as
source-attributed metadata, not newly computed truth. It cannot alter pair
membership, eligibility, or the identities of raw generations. Cohort
`historical_counts` are likewise reference metadata, never acceptance quotas.
Generation scoring metadata allows `parser_id`, `parser_version` (strings or
null), and `gold_match` (boolean or null). It is summarized only for present raw
outputs, separately by source kind, arm, window and declared parser. These are
descriptive counts of saved judgments, not R1 re-extraction results. Pair-level
`archived_scoring` remains opaque metadata and is not used to manufacture counts.

## Identity, membership, and outputs

Pair identity uses acquisition namespace and source pair key, independently of
filename, answer correctness and text hash. The same identity with contradictory
texts is rejected. Original-problem grouping ignores target rule and model; an
unknown original ID stays null. Hash-verified explicit aliases are required to
link acquisition identities. Text similarity is not identity evidence.
Consistent duplicate acquisitions retain their additional source references and
are explicitly reported. Contradictory scientific metadata is rejected; source
order cannot select the preferred gold, revision or archived editing evidence.

An `expected_ids_ref` points to the explicitly enumerated archived ID artifact
defined in schemas.md. Known missing IDs and extra records are reported without
substitution. Without an ID list, coverage is unknown even when the recovered
count equals 172 or 97. This command never selects new examples to fill a quota.
Known R3/R4 membership also requires the pair's task/model to match the frozen
setting. Other historical setting IDs remain explicitly unvalidated rather than
being silently interpreted as one of these experiments.

The output set is `source_audit.json`, `archive_inventory.jsonl`,
`pair_manifest.jsonl`, `pair_manifest.meta.json`,
`archive_generation_records.jsonl`, `historical_count_comparison.csv`,
`missing_inputs.csv`, and `run.json`. Historical counts and observed source counts
remain separately labeled. No source aggregates are expanded into per-item rows.
The manifest metadata binds protocol and source coverage even with zero rows.
Run metadata references the outputs, never the reverse, to avoid reference cycles.
An exact `protocol.json` snapshot is copied into the output set. The run records
the canonical protocol hash/revision and the implementing modules' content hashes,
plus the Git commit and dirty state when Git identity is available. Outside a Git
checkout those Git fields are null with an explicit reason, not invented values.

The original submitted data has not been recovered by implementing this command.
CPU fixtures are synthetic and are not formal R0 results. Real-model smoke and
formal R3/R4 results remain not run.
