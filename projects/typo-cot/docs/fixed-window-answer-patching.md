# Fixed-window free-answer patching

This document defines the public contract for `fixed-window-answer-patching`,
the multi-layer intervention reported in Tables 6 and 7 of the final PDF. The
single-layer Figure 2 answer scan remains a separate
`layerwise-answer-patching` command.

## Source and anchor selection

Every `--pairs` value must be `pairs.jsonl` from a completed, unlimited
`prepare-edited-pairs/v1` run with the same model, benchmark, dataset
fingerprint, pinned model revision, paper SHA-256, seed 42, four requested edits,
512-token generation cap, and `explicit-greedy-generation/v1` manifest. One
Attribution-4 or Random-4 source permits up to 300 anchors. When both sources are
supplied, each arm is independently sorted by sample ID, shuffled with seed 42,
and capped at 150 before pooling. A `--limit` smoke run takes a round-robin
prefix of the two selected arms.

An anchor's stored clean and edited continuations are re-extracted using the
final-PDF primary-then-empty-only-fallback rule. It is eligible only when that
audit finds clean-correct/edited-wrong and at least one aligned edited word; old
stored correctness booleans are never trusted for selection. The runner
deliberately does not exclude an edited word at the final prompt token: unlike a
final-block-only patch, an early multi-layer patch is followed by downstream
recomputation and is not structurally a no-op at that coordinate.

Fresh untreated continuations are generated after selection. They define two
different fixed cohorts:

- `clean-to-edited` restoration includes fresh clean-correct and fresh
  edited-wrong anchors;
- `edited-to-clean` induction includes every fresh clean-correct anchor,
  irrespective of the fresh edited answer.

The same direction-specific cohort is retained for every requested window.

## Intervention

`START:STOP` denotes a half-open range of zero-based complete decoder blocks.
For each window and direction, donor block outputs are captured from an
independent untreated prompt forward. Hooks for every block in the window are
then installed simultaneously for one recipient generation. At each aligned
edited word's final token, the complete donor block output overwrites the
recipient block output. All other token positions remain unchanged. Hooks apply
once during prompt prefill; cached one-token decoding steps are not patched.

Each window starts an independent greedy, bfloat16, left-padded continuation of
at most 512 tokens. The primary task extractor runs first. The deterministic
fallback runs only when the primary result is empty.

Restoration is true only when an extracted patched edited answer canonically
equals the fresh clean answer. Induction is true only when an extracted patched
clean answer canonically differs from the fresh clean answer. An unextractable
patched answer is false in both directions and stays in the denominator.

## Paper settings

Table 6 uses `[0,6)` and both directions for Gemma-3-4B, Llama-3.2-3B, and
Mistral-7B on GSM8K and MMLU. The six published setting totals are restoration
`800/1,241` and induction `871/1,458`; their micro-pooled row is post hoc.

The prespecified Table 7 MMLU-Pro comparison uses Attribution-4,
`clean-to-edited`, and the paired windows `[0,6)` and `[6,12)`:

| Model | Pairs | `[0,6)` | `[6,12)` | Difference | Published 95% CI |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B | 97 | 51 | 54 | -3.1 pp | [-14.4, 8.2] pp |
| Mistral-7B | 120 | 89 | 67 | +18.3 pp | [8.3, 28.3] pp |

Fresh window differences use a pair-resampled percentile bootstrap with 10,000
replicates and seed 42. Both event columns are indexed by the same ordered pair
cohort before resampling.

## Output schemas

`fixed_window_records.jsonl` contains one row per included
pair-direction-window:

- model, benchmark, targeting arm, sample ID, source-record SHA-256;
- direction, half-open window bounds, model layer count, aligned position count;
- fresh clean/edited baseline generations;
- patched generation, extraction provenance, binary event, and whether its
  token IDs are identical to the untreated recipient.

`pair_status_records.jsonl` contains every selected anchor, including those
excluded after fresh regeneration. Its `direction_status` object records a
separate inclusion flag and reason for each requested direction, together with
the fresh baselines.

`setting_summary.json` contains source, selected, and direction-denominator
counts; per-arm counts; explicit upstream exclusions; success counts, rates,
and descriptive Wilson intervals by direction/window; the paired MMLU-Pro
comparison when applicable; and published values as historical reference
metadata.

`run.json` binds the command arguments, protocol, source files, source manifests,
model/runtime provenance (including NumPy for bootstrap reproduction),
checkpoints, outputs, and final-paper artifact with SHA-256 fingerprints. A run
is labelled as one of:

- `fresh-paper-protocol-run` for a planned Table 6 configuration on a fresh
  public cohort;
- `fresh-prespecified-mmlu-pro-window-run` for a planned Table 7 configuration
  on a fresh public cohort;
- `partial-smoke-run`, `partial-paper-protocol`, or `non-paper-setting` with
  explicit limitations otherwise.

Fresh prepared pairs do not recreate unpublished historical sample IDs, so the
runner never claims exact historical-cohort identity.

## Historical discrepancies

The final PDF is the authoritative protocol. Preserved historical artifacts are
kept outside the public repository and reveal three differences that prevent
published counts from serving as forced acceptance tests:

1. the original Table 6 coordinate locator could silently match a repeated word
   rather than the aligned edited word; public pairs retain and revalidate actual
   tokenizer offsets and word-final coordinates;
2. historical Table 6 induction counted some unextractable patched answers as
   incorrect changes; the PDF says unextractable intervention answers are
   failures, which the public runner follows;
3. the historical MMLU-Pro analyzer used 100,000 bootstrap replicates and a
   different seed, while the final PDF specifies 10,000 replicates and seed 42.

The published values and these discrepancies are recorded in
`setting_summary.json`; neither silently changes the fresh event definition.

## Resume safety

Each selected targeting/sample identity has one atomic checkpoint. Reuse
requires an exact source-record fingerprint, runtime fingerprint, layer count,
ordered window grid, direction set, baseline schema, event recomputation, and
checkpoint SHA-256 match. A failed run retains valid checkpoints but removes all
partially finalized public tables. `--resume` rejects changed arguments or
inputs. For a completed run it validates all output hashes and returns before
loading model weights.
