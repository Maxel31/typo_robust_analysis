# Layerwise edited-word KL patching contract

This command implements the final paper's single-decoder-layer distributional
scan (§3.3, §4.1, Appendix B, Figures 2 and 4, and Table 5). The submitted PDF
identified by `typo-cot experiments source` is authoritative.

## One-setting command

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot layerwise-kl-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --targeting attribution-4 \
  --directions clean-to-edited edited-to-clean \
  --gpu-id 0 \
  --output-dir results/layerwise-kl-patching/gemma-3-4b-it/gsm8k/attribution-4
```

`--directions` accepts either or both named directions without duplicates. The
paper run uses both. `--limit 1` is a partial smoke test, and `--resume` continues
only the exact interrupted run. The public runner scans all decoder layers; it
does not expose a layer subset that could silently change the complete-grid
denominator.

## Input cohort and validation

`--pairs` must be `pairs.jsonl` beside a completed
`prepare-edited-pairs-run/v1` `run.json`. Before model loading, the command
checks the paper fingerprint, model, benchmark, targeting rule, seed 42,
four-edit condition, source record count, sorted unique sample IDs, and strict
JSON syntax. Duplicate keys and non-standard `NaN`/`Infinity` constants are
errors rather than values.

The source must be a full preparation run (`limit: null`) using the paper's
512-token generation cap. Its recorded Hugging Face model commit is used to
load both the model and tokenizer, and both resolved revisions are checked
before any pair is scanned. A moving repository default therefore cannot
silently change the failure cohort or patched activations.

The RQ1 failure cohort first requires the freely generated clean answer to be
correct and the freely generated edited answer to be wrong. Each runnable pair
also needs at least one aligned changed word. At runtime the tokenizer must
reproduce both recorded prompt token counts, and every aligned word's recorded
final-token coordinate must remain in bounds and be the final entry of its
token-index list. Empty alignment is an explicit scientific exclusion, not a
run failure.

The final PDF specifies this selected clean-correct/edited-wrong cohort and the
Appendix B KL/grid filters. Historical Table 5 construction additionally lost
some cases through answer-trigger and teacher-forced-divergence preparation
that is unnecessary for a prompt-final KL readout and is not enumerated in the
PDF. The public runner does not invent those legacy gates. A comparison that
claims exact historical row identity must supply and fingerprint the frozen
source cohort; a fresh run is labelled as a fresh paper-protocol reproduction.

## Intervention and readout

For every aligned changed word, let `t` be its clean final-token index and
`pi(t)` its edited final-token index. At decoder layer `l`:

- `clean-to-edited` writes the clean block output at `t` into the edited run at
  `pi(t)`;
- `edited-to-clean` writes the edited block output at `pi(t)` into the clean run
  at `t`.

Layer means one complete Transformer decoder block, including its attention and
MLP sublayers. `L0` is block 0's output, not an embedding. Only the aligned
recipient positions are overwritten, the input tokens remain fixed, and all
later blocks recompute normally. Donor states for all layers are captured from
an untreated prompt-only forward pass. Patching does not require AttnLRP model
rewrites; the `lrp` extra supplies the paper's locked model environment shared
with pair preparation.

The clean and edited baseline distributions are `softmax(logits[:, -1, :])`
from prompt-only forwards. They predict the first CoT token. This differs from
the pair-preparation attribution target, which observes the first CoT token and
attributes the adjacent next-token prediction.

For clean-to-edited restoration:

```text
R(l) = 1 - KL(p_clean || p_patched_edited(l))
             / KL(p_clean || p_edited)
```

Edited-to-clean induction swaps clean and edited in both numerator and
denominator and reads the patched clean run. A score of 1 is complete
distributional restoration, 0 is no improvement over the untreated recipient,
and a negative value means the patch moved farther from the reference. Scores
are never clipped. Direction-specific untreated KL must be finite and strictly
greater than `1e-9`.

## Complete grids and setting statistics

Validity is assessed independently for each pair and direction. An included
grid contains exactly layers `0..L-1`, with a finite patched KL and normalized
score at every layer. Missing, duplicate, or non-finite cells exclude the whole
pair/direction from that direction's profile.

For each layer, `setting_summary.json` reports the median normalized score over
included pairs. The peak is the lowest layer attaining the maximum median;
every exact tied layer is retained separately. Table 5's relative depth is
`layer / L`, not `layer / (L - 1)`.

The Hsu multiple-comparisons-with-the-best set uses 2,000 paired bootstrap
resamples with seed 42. For each layer it bootstraps the median within-pair
difference from the observed peak layer using one shared sample-index sequence;
the layer is retained when the one-sided 95% upper boundary is non-negative.

Depth-third membership uses the layer center `(layer + 0.5) / L`: early is
below `1/3`, middle is from `1/3` to below `2/3`, and late is at least `2/3`.
The command takes an arithmetic mean across layers within each pair and third,
then the median across pairs within the setting. The later paper-artifact step
macro-averages those setting values without weighting settings. The headline
uses 30 settings and 7,919 frozen pairs after excluding Gemma-3-4B/MATH
(`n=27`) and Qwen2.5-3B/MATH (`n=13`); all 32 settings are the sensitivity.

## Outputs and recovery

`layer_records.jsonl` uses schema `layerwise-kl-patching-layer/v1` and is sorted
by sample ID, canonical direction order, and layer. Each row records setting
identity, source-record fingerprint, direction, layer and both relative-depth
coordinates, aligned position count, untreated denominator KL, patched
numerator KL, and normalized score.

`pair_status_records.jsonl` uses schema
`layerwise-kl-patching-pair-status/v1`. It makes every selected
pair/direction's `included` or `excluded` state auditable, including denominator
and complete-grid exclusion reasons. Upstream non-failure-cohort and unaligned
records are counted in `setting_summary.json` but do not create artificial
directional KL cells.

`setting_summary.json` records the direction-specific population flow, layer
profile, peak/MCB result, and depth-thirds result. `run.json` records the exact
arguments, input and output SHA-256 hashes, paper fingerprint, protocol
constants, model/tokenizer revision, decoder-layer adapter and count, dependency
and CUDA/GPU provenance, timestamps, progress/failures, and whether the run is
full or partial.

Pair checkpoints are written atomically below a hidden work directory. A
runtime exception leaves `run.json` failed and does not publish partial final
JSONL files. `--resume` requires identical arguments, source hash, paper
contract, protocol fingerprint, and runtime provenance. A completed resume
verifies final output hashes and exits without reloading the model.
