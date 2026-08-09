# Patch-position reachability controls

This command implements the final paper's RQ1 position scan in §4.1 and
Appendix B. The final PDF identified by `typo-cot experiments source` is the
protocol authority. Historical scripts and raw results are used only to fill
in an unpublished token-locator detail and to explain a legacy hook-order
defect.

## Completed command

Run the published Gemma-3-4B/GSM8K Attribution-4 setting on physical GPU 0
after completing its `layerwise-kl-patching` command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-position-controls \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --layerwise-kl-run \
    results/layerwise-kl-patching/gemma-3-4b-it/gsm8k/attribution-4 \
  --positions edited-word prompt-final question-final \
  --gpu-id 0 \
  --output-dir results/patch-position-controls/gemma-3-4b-it/gsm8k
```

Use `--limit 1` only for a labelled smoke run. Continue an interrupted
identical run with `--resume`; `--resume` without the original `run.json` is an
error, as is starting without `--resume` in a non-empty output directory.

## Cohort and source boundary

The published denominator is the Table 5 Gemma-3-4B/GSM8K Attribution-4
layerwise-KL restoration cohort (`n=109`). It is not a subset of the separate
172-pair fixed-window free-answer cohort. Appendix B's “same 109 pairs” means
that all three position arms use the same included complete-grid IDs and
untreated KL denominators.

The public command therefore accepts `--layerwise-kl-run`, not a fabricated
`common-109.jsonl`. Before loading model weights, it requires and verifies:

- a completed `layerwise-kl-patching/v1` Gemma-3-4B/GSM8K Attribution-4 run;
- an unlimited source run containing the canonical `clean-to-edited` direction;
- all recorded source-output SHA-256 values;
- one unique status row and one complete finite layer grid per included ID;
- source pair-record fingerprints, model/tokenizer revision, the final-paper
  fingerprint, denominator threshold, and prompt-final readout protocol.

The referenced included IDs, denominators, and edited-word records become the
fixed source arm. Only the two alternative positions require new patched model
forwards. A fresh upstream pair-preparation and layerwise scan can follow the
paper protocol without recovering all undocumented historical eligibility
gates. Consequently, `run.json` distinguishes a fresh protocol reproduction
from exact historical-ID provenance and reports the actual common count rather
than forcing it to 109.

## Intervention and readout

Every intervention copies the residual-stream output after one complete
decoder block, one layer at a time, from the clean prompt into the edited
prompt. The input text and tokens remain unchanged. The arms are:

| Position | Clean donor coordinates | Edited write coordinates |
|---|---|---|
| `edited-word` | Every aligned edited word's final token | Every corresponding edited word's final token |
| `prompt-final` | The final non-padding prompt token | The final non-padding prompt token |
| `question-final` | The last token overlapping the recorded editable GSM8K question span | The last token overlapping the recorded edited question span |

The final PDF explicitly names the three positions and says that the separate
scan varies the write position. It does not define the exact question-final
token locator. The published setting is GSM8K, where the prepared pair's
editable span is the question; using its final overlapping token is the
portable equivalent of the legacy runner's independent question-string
`rfind` and offset-map lookup. Both source and destination coordinates are
recorded per pair rather than inferred from sequence length during aggregation.

The only canonical direction for this position analysis is
`clean-to-edited`. Although §3.3 defines the reciprocal intervention for the
general RQ1 method, the final PDF publishes no reciprocal position profile.
The readout is the prompt-final next-token distribution predicting the first
CoT token, and the score for every arm is

```text
1 - KL(p_clean || p_patched-edited) / KL(p_clean || p_edited)
```

The referenced untreated denominator must be finite and greater than `1e-9`.
Every requested arm must have one finite value for every decoder layer and
every common ID; negative restoration remains negative. No arm may silently
drop a pair, because doing so would violate the common-cohort comparison.

## Aggregation and published checks

For each position and layer, the canonical profile is the across-pair median
of normalized KL restoration. The summary reports the maximum value and all
exactly tied peak layers. The final PDF reports no confidence intervals or
arm-difference test for the new three-position comparison, and no MCB set for
either alternative position; this operation does not invent them. Table 5's
edited-word MCB remains explicitly labelled source-profile metadata.

The rounded published checks are:

| Position | Published result |
|---|---|
| `edited-word` | peak L2 = `.751` on 109 pairs (Table 5; MCB `8*`) |
| `prompt-final` | L0 = `.001`, L16 = `.25`, L32 = `.98` |
| `question-final` | peak across all layers = `.011` (peak layer unpublished) |

These values are reference metadata, not hard-coded replacements for a fresh
run. Their interpretation is reachability: the specified write reaches the
readout early from edited-word positions, late from prompt-final, and
effectively never from question-final. The crossover does not trace
information moving between positions and does not locate an encoding
mechanism. The analysis is post-hoc, exploratory, and descriptive.

## Correct last-layer behavior

The legacy prompt-final raw output reports zero restoration at L33 because a
readout hook was registered before the patch hook and captured the pre-patch
final-block hidden state. The final paper does not publish L33. The public
runtime reads the model's logits after the patch hook has modified the block
output, matching the intervention defined in §3.3.

For a model whose last decoder block is L-1:

- a `prompt-final` patch at L-1 reaches the final norm/language-model head and
  should reproduce the clean prompt-final state up to numerical tolerance;
- an `edited-word` or `question-final` patch at L-1 cannot flow backward or
  through another decoder block to the prompt-final causal readout, so its
  lack of restoration is structural.

Unit tests exercise this distinction with a causal toy decoder and verify that
the production runtime reads post-patch logits rather than a competing hook's
pre-patch capture.

## Outputs and restart semantics

`position_control_records.jsonl` contains one row per common pair, requested
position, and decoder layer. Each row records the fixed denominator, patched
KL, normalized score, source record fingerprint, source/destination token
coordinates, locator, layer index, layer count, and whether the row was
carried from the verified edited-word reference or freshly executed.

`pair_status_records.jsonl` contains one row per selected common pair with its
source status fingerprint, denominator, all position coordinates, complete
grid checks, and runtime/reference denominator comparison. The summary and
manifest also retain upstream exclusion counts from the referenced layerwise
run; exclusions are never relabelled as position-specific failures.

`position_control_summary.json` contains the common-cohort count and ID
fingerprint, actual versus published cohort provenance, each position's
layerwise median profile and peak, protocol, design status, published rounded
references, and the boundary on interpretation.

`run.json` records arguments, final-paper fingerprint, reference run and every
source/output SHA-256, model/tokenizer/runtime provenance, deterministic plan,
progress, checkpoint registry, failures, completion state, and comparability.
Pair-atomic checkpoints cover both alternative positions so a pair cannot be
half-published. A failed run publishes no partial result tables. A completed
resume verifies all public hashes and returns before loading model weights.
Unregistered work files are removed against the reconstructed plan before
reuse.
