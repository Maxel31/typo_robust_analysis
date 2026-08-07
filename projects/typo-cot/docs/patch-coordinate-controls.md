# Patch-coordinate controls

This document defines the public contract for `patch-coordinate-controls`, the
answer-level primary-coordinate comparison in final-PDF §3.3, §4.1, Appendix B,
and Table 7. The final PDF, identified by the SHA-256 printed by
`typo-cot experiments source`, is authoritative.

## Scope

The published comparison uses the same 172 Gemma-3-4B/GSM8K
clean-correct/edited-wrong pairs for all three answer arms:

- correct edited-word coordinates: 129/172 restored (75.0%);
- same-item +2-token coordinates: 44/172 restored (25.6%);
- matched cross-item clean donor: 42/172 restored (24.4%).

The final paper explicitly classifies the offset and cross-item comparisons as
post-hoc controls. Their protocol and historical-reference metadata therefore
record `design_status: post-hoc`; this implementation does not present them as
prespecified confirmatory analyses.

The exact historical sample IDs and model revision were not published. In
addition, the archived historical runner used an older substring locator whose
coordinates disagree with the corrected public aligner on many recoverable
records. The values above are therefore historical reference metadata, not
acceptance targets for a newly prepared public cohort. A complete public run is
labelled `fresh-primary-coordinate-control-run` and explicitly records
`historical_cohort_identity: false`.

Identity self-copy is not a fourth published Table 7 answer row. The final PDF
reports it as an implementation control for the patching machinery. This
command includes it as a diagnostic extension and fails the run if the copied
generation is not token-identical to the untreated edited baseline.

## Prerequisite and command

Run `fixed-window-answer-patching` first. The coordinate command consumes that
completed output directory rather than selecting the source pairs again:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-coordinate-controls \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --layers 0:6 \
  --controls correct offset-2 cross-item self-copy \
  --gpu-id 0 \
  --output-dir results/patch-coordinate-controls/gemma-3-4b-it/gsm8k
```

The reference denominator is exactly the ordered set of
`pair_status_records.jsonl` rows whose
`direction_status.clean-to-edited.included` value is true. The `correct` arm is
the matching `clean-to-edited`, `0:6` row from
`fixed_window_records.jsonl`; it is not regenerated a second time.

Before loading model weights, the command validates:

- the completed fixed-window run schema, final-paper fingerprint, protocol,
  model, benchmark, direction, and window;
- all three fixed-window output hashes and record counts;
- every prepared-pair file and sibling run hash recorded by the reference;
- the selected-pair order, source-record hashes, regenerated baselines,
  direction inclusion flags, correct-patch events, and summary aggregates;
- both targeting arms in the prepared sources and in the actual included
  clean-to-edited denominator;
- every +2 coordinate plan and the full cross-item donor map.

After model loading but before generation, the decoder depth and normalized
runtime provenance must also equal the fixed-window reference. The comparison
includes package versions, model/tokenizer revisions, adapter, dtype, device,
GPU, and generation/extraction settings; only operation-specific class labels
are ignored. Thus the referenced `correct` arm cannot be paired with controls
from a materially different execution environment merely because their
untreated answer tokens happen to match.

`--limit N` is a smoke-only option. It limits recipients after the full
reference denominator has fixed the donor map, so the donor for a retained
recipient does not change when `N` changes.

## Interventions

All arms use complete decoder-block residual outputs at layers `[0,6)`, patch
only during prompt prefill, and then freely generate with the same greedy,
bfloat16, left-padded, 512-token protocol as the reference.

`correct`
: Copy every aligned clean edited-word final-token state to the corresponding
  original edited coordinate in the same item. The verified fixed-window row
  supplies this endpoint.

`offset-2`
: Add two to both coordinates in each aligned clean/edited endpoint pair. Keep
  a pair only when both shifted positions are inside
  `[1, prompt_length - 1)` and neither shifted position is an original
  edited-word endpoint. A pair may retain fewer coordinates, but every
  reference item must retain at least one; otherwise the run fails before GPU
  work rather than silently changing the paired denominator.

`cross-item`
: Keep the recipient's original edited coordinates. Within each
  `(targeting arm, aligned-word count)` stratum, sort sample IDs and assign the
  next ID cyclically as the clean donor. This is deterministic, non-self, and
  preserves both the targeting condition and source/destination row count. A
  singleton stratum is a preflight error.

`self-copy`
: Capture the edited recipient's own states and write them back to the same
  edited coordinates. Its complete generated token sequence must equal the
  referenced untreated edited sequence.

The restoration event is true only when an extracted patched answer equals the
regenerated clean answer under exact task-aware comparison. The task extractor
is tried first and the deterministic fallback only when the primary extractor
is empty. An answer that remains unextractable is false and stays in the
denominator.

## Statistics

The summary computes arm rates and Wilson intervals as descriptive fresh-run
statistics. The two headline comparisons are `correct` versus `offset-2` and
`correct` versus `cross-item` on the same ordered event vectors. Each uses a
two-sided exact McNemar conditional-binomial test with no asymptotic or SciPy
dependency. The historical discordant counts reproduce:

- correct versus offset: 92 correct-only, 7 offset-only,
  `p = 5.0749031687767575e-20`;
- correct versus cross-item: 98 correct-only, 11 cross-only,
  `p = 1.3281749873159826e-18`.

Self-copy is reported under `self_copy_integrity`, outside the published
Table 7 comparison map.

## Outputs and restart safety

The command writes:

- `coordinate_control_records.jsonl`: one ordered row per executed pair and
  requested arm, including the baseline, generation, binary event, donor, and
  exact source/write coordinates;
- `pair_status_records.jsonl`: every pair in the full reference denominator,
  its coordinate/donor plan, limit selection status, and baseline replay;
- `coordinate_control_summary.json`: population, arm rates, exact paired
  tests, self-copy integrity, protocol, and historical references;
- `run.json`: arguments, paper/reference/source/runtime/plan fingerprints,
  progress, checkpoint registry, failures, output hashes, and comparability.

Baseline replay and control generation use separate pair-atomic checkpoint
directories. Every selected baseline must match the reference exactly before
any control is generated. A failed run publishes no partial result tables;
rerun the identical command with `--resume`. A completed resume validates the
three public output hashes and returns before loading model weights.

SHA-256 values are content fingerprints, not random seeds. They make a changed
paper, source, donor map, checkpoint, or output detectable; they do not choose
samples or claim that two scientifically different artifacts are equivalent.
Donor assignment is instead fixed directly by sorted sample IDs and cyclic
shift, while upstream pair preparation records its own explicit seed policy.
