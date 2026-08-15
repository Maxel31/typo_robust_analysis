# SAE diagnostic track v1

This track is parallel to, and must not alter, the frozen 10M-token robustness
comparison. GPU 5/6 jobs are protected. SAE work uses the preregistered dedicated GPU and yields whenever
the frozen robustness evaluation needs the device or operator attention.

## Scope and order

1. Finish the protected 10M-token output-matching and causal-window runs without
   changing, restarting, or evaluating task accuracy during training.
2. Calibrate and train ReLU+L1 sparse autoencoders on clean FineWeb-Edu only:
   layer 5 with initialization seeds 42 and 43, and layer 20 with seed 42.
3. Apply the preregistered WP-2 acceptance gates before any SAE-based diagnosis.
4. Only accepted SAEs may be used for the Base patch/L0 diagnostic, retrospective
   checkpoint analysis, and preregistered feature causal kill test.
5. A feature-targeted training arm remains a document-only future study until
   the existing decision tree is complete and the kill test passes.

## Frozen SAE definition

For a complete decoder-block residual output `x`, the SAE is

`z = ReLU(Wx + b)` and `x_hat = Dz`.

The optimization objective is mean per-coordinate reconstruction squared error
plus `lambda_L1 * mean(sum(z))`. The decoder directions (columns of `D`) are
renormalized to unit L2 norm after every optimizer step. The expansion factor is
16, so Gemma's 2,560-dimensional residual stream maps to 40,960 unconstrained
ReLU features. No top-k constraint is used because L0 and the registered energy
score require an unconstrained firing count.

Three L1 candidates are compared on the same calibration stream. Selection first
requires median L0 in [30, 150], then chooses the lowest
FVU; deterministic coefficient order breaks an exact tie. This selection cannot
use typo inputs or downstream behavior.

### One-time calibration amendment (2026-08-15)

The original one-million-token calibration completed all 489 optimizer steps on
clean data, but none of the registered coefficients `[0.0001, 0.0003, 0.001]`
entered the frozen median-L0 interval. At the strongest coefficient, median L0
was 4,874 at layer 5 and 3,856 at layer 20, still more than 25 times the upper
gate. Before any WP-2, WP-3, or WP-5 result was observed, the one adjustment
allowed by the preregistration was consumed: the grid is now the fixed
log-spaced bracket `[0.01, 0.1, 1.0]`. A falsification-first review then showed
that 489 optimizer steps were insufficient for the encoder bias to reach the
sparse regime: on the product optimizer path, a 1,000-fold coefficient increase
changed synthetic median L0 by only 1.2%, while ten times as many steps changed
it materially. The same pre-registered lambda/token amendment therefore raises
the calibration budget from 1M to 10M activation tokens. Training is streamed in
the existing deterministic 1M-token buffers, and final statistics are computed
by replaying the identical 10M-token activation stream after optimization. The
selection rule, clean-only data, and all WP-2/WP-5 gates remain unchanged. The
failed W&B run ID and all six observed L0 values are recorded in the registry;
no further coefficient or token adjustment remains authorized.

GPU 1 subsequently became occupied by an unrelated workload. The operator
assigned the waiting retry to GPU 0. This operational reassignment changes no
data, model, loss, threshold, or evaluation setting and is bound in the same
registry.

## Data separation

Only `clean`, `fineweb_edu`, `train` records with no task or answer are accepted.
The input artifact already excludes all frozen evaluation and localization IDs.
The first 30,000 records in the exact seed-42 epoch-0 training order are reserved
for the protected 10M-token runs, exceeding their observed source consumption.
Their complement is the initial SAE stream. The selected record-ID list and its
SHA-256 digest are written before activation collection. Additional FineWeb-Edu
must pass the same record/group/normalized-content decontamination before it can
raise the unique clean budget to at least 100M tokens.

The supplement builder replays the pinned, unshuffled FineWeb-Edu stream and
requires the frozen tune, pre-PR, final-test, localization-selection, and
localization-validation manifests as exclusions. It rejects record/source/group
identity overlap, normalized-content overlap, and character-5gram MinHash
candidates whose exact Jaccard similarity is at least 0.99. The minimum build
budget includes 100M training activations, 10M held-in statistics activations,
and 200 untouched splice documents.

### Pre-run data amendment (2026-08-15)

The first corpus-build attempt stopped before writing an artifact or running a
model because the protected source manifest contains 131 normalized-content
duplicates outside or across the reserved prefix. The source manifest hash and
the exact 30,000-record protected prefix remain unchanged. Every protected-prefix
row remains an exclusion anchor. Eligible rows are traversed in the already
frozen `sae-clean-source-order/v1`; a row is omitted when its normalized content
already occurs in the protected prefix or an earlier eligible row. This leaves
126,901 initial eligible records and 51,735,090 source tokens; fresh FineWeb-Edu
supplements the difference to the unchanged token budget. No acceptance gate,
kill-test threshold, evaluation item, or model setting changed.

A subsequent calibration-input check stopped before runtime initialization after
finding eight protected record IDs in the first supplement. Those documents had
been omitted from the eligible stream as exact normalized-content duplicates,
then replayed from FineWeb-Edu with different segmentation. The supplement
builder therefore retains every raw protected record/source/group identity as an
exclusion, even when the row itself is omitted from eligible training. The first
supplement is quarantined and rebuilt; no model forward had started. Derived
eligible values are now checked mechanically against this preregistration at
corpus-build, calibration, training, and validation input loading.

## Preregistered acceptance and kill-test rules

WP-2 requires FVU <= 0.35, median L0 in [30, 150], dead-feature rate <= 20%
where dead means `p_i < 1e-5`, median splice KL <= 0.15 nats/token, and saved
`p_i` plus median reconstruction error `s`. One documented lambda/token adjustment
is allowed; a second failure stops the track.

WP-5 is not authorized until two accepted layer-5 seeds exist. Its thresholds
are frozen at `median(R_z) >= 0.5 * median(R_full)` and
`median(R_sup) >= 0.25 * median(R_full)`, with direction replication across the
two seeds. Until that test passes, this project makes no component- or
feature-localization claim.
