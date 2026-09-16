# R1 saved-generation answer audit (PR-02)

`typo-cot rebuttal-v2 answer-audit` is CPU-only. It compares two versioned parsers
on the same saved text, without changing archived generations, cohort membership,
the intake manifest, or the public v1 parser. Implementation and synthetic CPU
acceptance fixtures are available; real-model smoke and formal results are not.

## Invocation and trusted inputs

From the repository root:

```bash
uv run --project projects/typo-cot typo-cot rebuttal-v2 answer-audit \
  --manifest /path/to/intake/pair_manifest.jsonl \
  --output-dir /path/to/new-answer-audit
```

Supply a manifest produced by [intake](intake.md), with its required sibling
`pair_manifest.meta.json`. Metadata is required even for a zero-row manifest:
it binds the protocol, source audit, inventory, and original cohort coverage.
Keep the referenced source artifacts accessible. References and text hashes are
verified using captured bytes; inputs are checked again before publication.
Missing source declarations can remain explicit unknowns, but a missing or
modified artifact advertised by a hash is an integrity error, not a model error.
This is a trusted local artifact workflow, not a sandbox for untrusted paths.

The output directory must be absent or empty. Publication is atomic and does not
overwrite a completed audit. The command returns JSON run metadata on stdout;
invalid input or publication failure returns a nonzero status and stderr error.
No model weights, tokenizer download, torch, or GPU are required.

## Frozen parser and scoring contracts

The scientific grammar remains [README §18](README.md#18-pr-00で固定する実装境界と受入例).
`explicit_answer_v2`, version `2.0.0`, accepts a complete explicit answer on the
last nonempty LF-delimited line. It checks earlier complete, nonquoted explicit
answers for contradictions. Numeric values are exact reduced fractions, not
binary floats; choice labels must belong to the item's supplied valid labels.
Candidate offsets address Unicode code points in the original text. Extraction
receives no gold. A separate comparison uses the supplied canonical gold.

The second parser is named `public_v1_proxy`, version
`public-extractor-fallback/v1`. It executes the unchanged public extractor and
uses its fallback only after an empty extraction. Its positional fallback is
disabled for recorded `length-cap` outputs. The primary extraction itself is
not silently repaired. The actual hashes of `extractor.py` and `fallback.py`
are saved; archive metadata naming a parser does not prove the historical code
has been recovered, so this implementation never labels the proxy `archive_parser`.

The proxy's main `score_status` and `gold_match` reproduce public v1
`fallback.answers_equal` using the archived raw gold string. Missing or nonstring
raw gold is not reconstructed from canonical gold. Auxiliary
`canonical_score_status` and `canonical_gold_match` compare the proxy's fully
canonicalizable extraction with the same exact canonical gold as the primary
parser. This auxiliary comparison helps separate extraction differences from
scoring-rule differences; it does not replace the legacy result. The parser
descriptors and transition rows identify their distinct scoring rules.

Stopping is an independent recorded dimension: a complete answer at the length
cap may be extracted; EOS at the cap remains `eos`; unknown stopping is not
inferred from the text length. The primary parser does not change its grammar
according to stopping or gold.

## Availability and denominators

Each known pair/arm/window slot has one coverage row per selected parser.
Clean/typo baselines retain `window=null` and are not replicated for R4's two
patch windows. Unknown expected grids are listed separately, not assumed empty.

| Situation | Audit row | Coverage / score |
|---|---|---|
| Raw output absent | None | `output_missing` |
| Parser cannot execute | None for that parser | `parser_unavailable` |
| Output parsed, required gold unavailable | Present | `not_scored`, `gold_match=null` |
| Output ambiguous/unextractable, valid gold present | Present | `scored`, `gold_match=false` |
| Extracted answer with valid gold | Present | `scored`, boolean comparison |

Operational limits for the primary parser are 1,000,000 text code points,
1,000 digits per numeric spelling, and absolute exponent 1,000. Valid numeric
syntax beyond these limits, or host numeric-conversion inability, reports parser
unavailability instead of a failed answer. Missing choice labels likewise make
the primary parser unavailable; the public proxy can still execute its own
task-wide extraction contract. Neither these limits nor low extraction coverage
justify changing the frozen grammar or selecting a different cohort.

Counts are partitioned by saved cohort/membership, task/model, source kind,
arm/window, expected-slot membership, and parser. Members, extras, and unknown
membership remain distinct. Rates explicitly give their denominators: scored
outputs for answer success, executed parsers for extraction statuses, present
raw outputs for stopping, and comparable patch/typo scores for recovery.
Primary-minus-proxy success differences use only slots with both scores present,
with unknown counts reported separately. A full-original-cohort success rate is
null unless that exact original cohort has complete, scored expected-slot coverage.
Missing original IDs are retained from the source audit and never replaced by
extra recovered examples. Per-pair baseline transitions explicitly retain
`unknown` when either score is unavailable. Failure-ID artifacts refer to
**archived baselines**, not the fresh baseline list that the later runner must
freeze independently. The main failure-ID lists contain original members;
extra and unverified eligible IDs are separate, with cohort-specific states kept.

## Published artifacts

| Artifact | Purpose |
|---|---|
| `answer_audit_records.jsonl` | Verified generation/manifest references, extraction, spans, stopping, parser identity and scores |
| `scoring_coverage.jsonl` | Every known slot/parser, including unavailable output or parser; reference to score when present |
| `parser_transition_table.json` / `.csv` | Old/new results on each identical saved generation |
| `baseline_transition_table.json` / `.csv` | Per-pair archived clean-correct/typo-incorrect eligibility transitions |
| `primary_parser_archived_baseline_failure_ids.json` | Primary archived eligibility IDs and states |
| `public_v1_proxy_archived_baseline_failure_ids.json` | Separately named proxy archived eligibility IDs and states |
| `parser_disagreement_blind.jsonl` | Reviewer-safe raw-output sample |
| `parser_disagreement_private.jsonl` | Private sample mapping and parser/context metadata |
| `blind_sampling_policy.json` | Fixed selection policy, stratum sizes, selected and unaudited counts |
| `parser_audit_summary.json` | Extraction/scoring/stopping counts, recovery comparisons, source coverage and claim limits |
| `run.json`, `protocol.json` | Invocation, code/input/output hashes, execution status, and unchanged protocol |

CSV presentation escapes spreadsheet formula prefixes and uses `NA` for null;
JSON retains exact structured values. A complete CPU run means the requested
audit executed, not that source recovery, human scoring, or model experiments
are complete. Input and output hashes allow the comparison to be reproduced
without overwriting any historical table.

## Blinded review export

The selection policy is fixed before human judgments: seed 42, SHA-256 ordering
within extraction-disagreement/doubt strata, up to 32 examples per stratum, and
up to 8 extraction-agreement controls. Selection never uses gold, correctness,
or human labels. The policy records both selected and unaudited counts.

Only `schema_version`, opaque `blind_id`, and the exact `raw_text` enter the
reviewer export. Give reviewers only `parser_disagreement_blind.jsonl`, not the
directory containing private mappings or score reports. Context that a model
itself wrote remains in its raw text; no extra gold/model/arm metadata is added.
This PR exports samples but does not import or adjudicate human annotations.
Until independent human scoring covers the necessary cases, parser-independent
semantic claims remain `inconclusive`; automated correctness means success
under the identified extraction and scoring rule.
