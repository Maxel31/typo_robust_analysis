# Build the one-token paper tables

`build-one-token-tables` is the CPU-only artifact builder for the supplementary
one-token diagnostic in Appendix D, Tables 10--11, and the one-token panel of
Figure 5. The final submitted PDF, identified by the canonical SHA-256 printed
by `typo-cot experiments source`, is the highest-priority specification. The
builder consumes completed public `one-token-prefix-replacement` runs; it does
not generate answers, load model weights, or silently import archived results.

The diagnostic measures clean-question answer sensitivity to one forced token.
It is not typo repair, activation-patching mediation, or a causal estimate of
the effect of KL divergence.

## Command

Run the builder from the repository root after producing the setting-level
outputs:

```bash
uv run --project projects/typo-cot \
  typo-cot build-one-token-tables \
  --runs-root results/one-token-prefix-replacement \
  --output-dir results/one-token-tables
```

The command is intentionally CPU-only. It must not import `torch`,
`transformers`, model wrappers, or tokenizer implementations. `--runs-root`
may use any directory layout: setting identity comes from each verified
manifest and record, never from a path name. `--output-dir` must not already
exist and is published atomically only after all requested artifacts have been
rendered and hashed.

## Required setting grid

The paper has fifteen model-by-task cells: five models crossed with GSM8K,
MMLU, and ARC-Challenge.

| Public model ID | GSM8K | MMLU | ARC-C |
|---|---:|---:|---:|
| `google/gemma-3-1b-it` | extension + adjacent | extension | extension |
| `google/gemma-3-4b-it` | primary | extension | extension |
| `meta-llama/Llama-3.2-1B-Instruct` | extension | extension | extension |
| `meta-llama/Llama-3.2-3B-Instruct` | extension | extension | extension + adjacent |
| `mistralai/Mistral-7B-Instruct-v0.3` | extension | extension + adjacent | extension |

Gemma-3-4B/GSM8K is the separately reported frozen 172-pair primary cell. The
other fourteen cells are deterministic 150-target extensions. The primary
cell must never enter an extension pool. The adjacent comparison is defined
only for the three cells marked above:

- Gemma-3-1B/GSM8K;
- Llama-3.2-3B/ARC-Challenge; and
- Mistral-7B/MMLU.

The builder recursively discovers `run.json` files, then accepts at most one
run for each expected model/benchmark identity. An accepted input must be a
completed, unlimited `fresh-paper-protocol-run` with the canonical paper and
one-token protocol identities, verified public-output SHA-256 values, the
paper cohort type, and the complete frozen plan. The primary plan must contain
172 targets; every extension plan must contain 150 targets. Each of the three
adjacent cells must have requested both `distant` and `adjacent` controls.

An unexpected setting, duplicate identity, `--limit` smoke run, incomplete
run, mixed paper/protocol identity, malformed record, manifest/record
disagreement, checksum mismatch, or output mutation is an error. The command
does not choose whichever duplicate happens to be encountered first and does
not repair an invalid grid by dropping records.

## Table 10 token columns

Table 10 calls its common one-token denominator `n1`. For each record, let
`P_keep` and `C_keep` force the clean token at the selected maximum-KL position
`P` and distant lower-median-KL control `C`. Let `P_bP` force the edited-context
top-1 token derived at `P` back at `P`, and let `C_bC` analogously use the token
derived at `C` back at `C`.

A record enters `n1` exactly when:

1. `P_keep` is correct;
2. `C_keep` is correct;
3. `P_bP` is not a no-op; and
4. `C_bC` is not a no-op.

On this one common denominator:

```text
Sel. = count(P_bP is incorrect) / n1
Ctl. = count(C_bC is incorrect) / n1
```

Because both keep arms are required to be correct, these are
correct-to-incorrect rates, not arbitrary answer-change rates. An
unextractable replacement answer is an incorrect outcome and remains in the
denominator. Eligibility is not recalculated separately for the two columns.
Bold `Sel.` in the PDF means only `Sel. > Ctl.`; it is not a significance
marker.

The CSV contains each available setting cell. Its paper-labelled aggregate is
emitted only when all fourteen extension cells are present and excludes the
primary cell by construction. The clean-prefix `n0`, `Full`, `Short`, and
`Nonmono.` columns in the printed Table 10 come from a different producer and
different denominator; this command does not recreate or join those columns.
Consequently `table10_one_token.csv` is explicitly the one-token portion of
Table 10, not a claim that the two Table 10 protocols share a denominator.

## Table 11 distant factorial definitions

The six distant arms cross two intervention positions with two replacement
token identities. Write `H_xy` for an incorrect outcome at position `x` using
the edited-context top-1 token derived at `y`, with `x,y` in `{P,C}`. Both
factorial outputs use:

```text
Loss_P = (H_PP + H_PC) / (2 n)
Loss_C = (H_CP + H_CC) / (2 n)
Delta  = Loss_P - Loss_C
```

`Delta` is reported in percentage points. `n` counts records; each loss rate
has `2*n` event opportunities. The extension pool is also split by the stored
token order `P<C` (selected position before control) and `P>C` (selected
position after control). Exact integer event counts and denominators are the
source of truth; displayed percentages are rounded only when rendering.

### Final-PDF-literal denominator

`distant_factorial` implements the final PDF's prose literally. A record is
eligible when both keep arms are correct and all four semantic replacements
`P_bP`, `P_bC`, `C_bP`, and `C_bC` are non-noops. The PDF does not require the
two source token IDs to differ and does not impose the submitted producer's
special-token admissibility guard. Therefore equal `b_P` and `b_C` IDs remain
eligible if all four recorded semantic arms are non-noops.

### Submitted-producer-compatible denominator

`distant_factorial_submitted_producer` starts from the PDF-literal set and
additionally requires:

- `b_P != b_C`;
- `b_P` is in the submitted producer's admissible token-ID pool; and
- `b_C` is in that pool.

This is the operationalization that produced the printed Table 11 row. It is
not identical to the literal denominator stated in the final PDF. The output
must show both results side by side, name the additional exclusions by reason,
and compare the printed historical row only with the submitted-producer
result. It must never label the printed `n=1,575` as PDF-literal.

On the archived submitted records, the distinction is:

| Classification | `n` | P events / opportunities | C events / opportunities | Loss P/C | Delta |
|---|---:|---:|---:|---:|---:|
| final-PDF literal reclassification | 1,603 | 912 / 3,206 | 647 / 3,206 | 28.4% / 20.2% | +8.3 pp |
| submitted producer / printed row | 1,575 | 892 / 3,150 | 633 / 3,150 | 28.3% / 20.1% | +8.2 pp |

All 28 archived additional exclusions had `b_P == b_C`; none was caused by an
inadmissible source token. These values audit the historical archive. They are
references, not pass/fail acceptance values for a newly generated public
cohort.

## Table 11 adjacent same-token control

The adjacent analysis uses the same `b_P` token at `P` and at the selected
adjacent lower-KL position `A`. Its common paired denominator requires:

1. `P_keep` and `A_keep` are both correct; and
2. `P_bP` and `A_bP` are both non-noops.

The two rates are `count(P_bP incorrect)/n` and
`count(A_bP incorrect)/n`. Adjacent pooling is permitted only after all three
prespecified adjacent settings are present. No other setting may be added to
that pool. The position rule and its outcome-blind SHA-256 side tie-break are
producer-compatibility details; they do not randomize semantic role and do not
turn the contrast into a causal estimate of divergence.

## Clustered inference

The final PDF specifies that intervals and tests cluster by
`(benchmark, sample_id)`. It does not print the estimator family, bootstrap
draw count, seed derivation, or test statistic. The builder therefore labels
the following frozen procedures as `submitted-producer-compatible`, not
`paper-defined`.

The distant-factorial rows use the submitted analysis's intercept-only
cluster-robust OLS interval. The point estimate is the record-level mean of
`(selected_loss_events - control_loss_events) / 2`. Its CR1 sandwich variance
uses cluster-summed residuals, the `G/(G-1)` small-sample correction, and a
Student-t critical value with `G-1` degrees of freedom. This is the procedure
that reproduces the printed distant intervals `[6.0, 10.4]`,
`[13.0, 18.4]`, and `[-10.0, -3.0]` after percentage-point rounding.

The adjacent rows use a percentile cluster bootstrap with 50,000 draws:

- sorted unique `(benchmark, sample_id)` clusters are sampled with replacement,
  drawing the original number of clusters per replicate;
- every record belonging to a sampled cluster is retained, including repeated
  model/targeting records sharing that benchmark and sample ID;
- the bootstrap statistic is the paired selected-minus-adjacent-control event;
- base seed 42 is domain-separated by the analysis label as
  `SHA256("42|" + label)`, taking the first 64 bits and reducing modulo
  `2**63 - 1`; and
- the 2.5th and 97.5th percentiles are reported without substituting a
  record-level interval.

The stable submitted labels are `pooled` for the three-setting row and
`setting:<setting_id>` for each of its three cells. They are part of the RNG
protocol rather than display text: renaming one would deliberately select a
different bootstrap stream.

Two-sided tests, when emitted in the machine-readable artifact, use the
submitted analysis's Monte Carlo cluster sign-flip test: contributions are
first summed within `(benchmark, sample_id)`, then cluster signs are flipped
for 200,000 draws with seed 42, and the p-value uses the plus-one correction
`(extreme + 1)/(B + 1)`. These p-values are secondary metadata; Table 10 bold
formatting must not be derived from them. An exact paired McNemar value may be
recorded for the adjacent comparison only as descriptive metadata, never as a
replacement for the paper's clustered interval.

The analysis artifact records the cluster key, cluster counts, repeated-cluster
diagnostics, statistic, estimator-specific correction or number of draws,
base and derived seeds where applicable, interval method, and test method. A
PDF-literal interval computed for a fresh run is a new, clearly labelled
reanalysis; the PDF contains no historical literal `n=1,603` interval to
match.

## Figure 5 validation boundary

The worked one-token case is identified before inspecting outcomes as:

```text
model      google/gemma-3-4b-it
benchmark  gsm8k
targeting  attribution-4 (legacy code lxt4)
sample_id  gsm8k_00556
```

From a verified one-token record, the builder can validate:

- setting, targeting condition, and sample ID;
- clean-CoT token count;
- `P=23` and distant `C=60`;
- `KL[P]` (the printed value is 8.79; the historical machine value is
  8.785974);
- the clean-reference token's ranks at `P`, 1 under the clean context and 8
  under the edited context;
- the selected keep/replacement extracted answers, `160/120`;
- the distant keep/diagonal-replacement extracted answers, `160/160`; and
- the relevant correctness, no-op, token-ID, and provenance fields.

The public record stores token IDs but deliberately does not store decoded
token strings. Since this CPU builder does not load the tokenizer, the printed
strings `" thrice"` and `" twice"` are reported as historical references with
status `unverifiable-from-one-token-record`, not guessed from IDs. The builder
also cannot validate from this producer alone:

- the displayed clean and typo questions or the red character edits;
- the displayed clean and typo CoTs;
- the clean-prefix transition at `k=23` and `k=24`, or stable boundary
  `24/61`; or
- the fixed-window `[0,6)` free-answer and fixed-context rank/KL patching
  results, including the printed patched KL `0.00034`.

Those fields belong to pair preparation, `clean-prefix-scan`, and
`fixed-window-answer-patching`. `figure5_validation.json` names the required
producer for each unavailable field. It must not present a copied caption value
as if it had been revalidated. Figure 5 remains an illustrative worked case,
not aggregate or mediation evidence.

## Partial coverage and fail-closed behavior

Missing expected settings and invalid supplied settings are different states.
A root containing a valid subset may produce setting-level descriptive rows
and a coverage report. Every absent cell is named explicitly. Paper-labelled
pools and their inference are omitted as follows:

- the Table 10 extension aggregate and distant Table 11 pools require all
  fourteen extensions;
- the adjacent pooled row requires all three adjacent settings;
- the complete Table 10 token grid requires all fifteen settings; and
- the primary cell may be shown separately whenever its one valid run is
  present.

At least one valid producer `run.json` is required. An existing but empty
`--runs-root` is treated as a likely path/configuration error rather than a
successful artifact with fifteen missing settings.

No available cells are silently pooled under a paper label when a required
cell is missing. A partial artifact records `not_comparable` with missing
identities; it does not fill cells from historical constants. Conversely, any
discovered run that claims an expected identity but fails schema, completion,
protocol, cohort, plan-size, adjacent-coverage, record-count, or checksum
validation aborts the build. Partial coverage is not a mechanism for ignoring
bad input.

## Outputs

A successful invocation writes exactly:

- `one_token_tables.json`: machine-readable coverage, per-setting integer
  counts, both factorial classifications, extension and adjacent pools when
  permitted, inference metadata, historical references, and field-level
  comparability decisions;
- `table10_one_token.csv`: available per-setting `n1`, selected, and control
  token results, plus the fourteen-extension aggregate only when complete;
- `table11_position_controls.csv`: separately labelled PDF-literal distant,
  submitted-producer distant, order strata, and adjacent results;
- `one_token_tables.md`: deterministic human-readable tables, denominator
  notes, coverage, and comparison statuses;
- `one_token_tables.tex`: deterministic table fragments with the same numeric
  content and explicit denominator labels;
- `figure5_validation.json`: the worked-case identity, computed checks,
  historical references, unavailable fields, and their required producers;
  and
- `run.json`: resolved arguments, canonical paper and analysis-protocol
  fingerprints, sorted input inventory and SHA-256 values, coverage decision,
  inference constants, and every output SHA-256.

All JSON is emitted with stable key ordering, CSV rows use the frozen setting
order, and Markdown/LaTeX rendering is derived from the same integer analysis
object. Rendered files are never independently recomputed.

## Historical-reference contract

The final PDF's printed values and the archived literal reclassification are
embedded only as labelled references. For a fresh public cohort the builder
reports `match`, `differs`, or `not_comparable` field by field; numerical
difference is not a runtime failure. This distinction is necessary because the
public preparation follows the paper protocol but does not prove byte-identical
membership in the unpublished historical cohort.

The most important printed Table 11 references are:

| Row | `n` | Loss P/C | Delta [95% CI] |
|---|---:|---:|---:|
| distant pooled, 14 extensions | 1,575 | 28.3% / 20.1% | +8.2 [6.0, 10.4] pp |
| distant `P<C` | 1,044 | 32.3% / 16.6% | +15.7 [13.0, 18.4] pp |
| distant `P>C` | 531 | 20.5% / 27.0% | -6.5 [-10.0, -3.0] pp |
| adjacent pooled, 3 settings | 391 | 31.7% / 28.6% | +3.1 [-1.8, 7.9] pp |

The distant printed rows use the submitted-producer-compatible denominator.
Every printed adjacent interval includes zero. Historical Table 10 cell values,
the fourteen-extension `n1=1,629` aggregate (`492/1,629` selected and
`296/1,629` control), and the primary `n1=153` result (`41/153` selected and
`23/153` control) follow the same reference-only rule. They may diagnose a
reproduction difference, but they may not replace newly computed events or be
used to make a fresh run pass.
