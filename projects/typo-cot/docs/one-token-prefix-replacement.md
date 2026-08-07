# One-token clean-prefix replacement diagnostic

`one-token-prefix-replacement` is the public runner for the supplementary
diagnostic in the final paper's Appendix D, Tables 10--11, and Figure 5. The
canonical PDF fingerprint printed by `typo-cot experiments source` is
authoritative. This operation is deliberately outside RQ3's reported result:
it measures clean-question answer sensitivity to one forced token. It is not typo repair
and does not identify where a clean prefix should end.

## Commands and source cohorts

The primary cell reuses the exact unlimited Gemma-3-4B/GSM8K clean-to-edited
denominator of a completed `[0,6)` fixed-window run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cohort primary \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --position-controls distant \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-4b-it/gsm8k
```

Each extension consumes one completed, unlimited Attribution-4 preparation
and one completed, unlimited Random-4 preparation. It applies the same
correct-clean/wrong-edited and aligned-edit source filter, shared clean-CoT
suffix check, 8--512-token length rule, per-arm 400 cap, proportional quota,
and deterministic systematic selection as `clean-prefix-scan`. The paper also
requires exact token boundaries. To preserve the submitted cohort selection,
the public runner freezes selection-eligible IDs using the submitted
prompt-length suffix rule, then applies its stronger exact-boundary audit to
those IDs. A selected boundary failure remains `invalid-boundary`; it is never
silently replaced by a later source pair:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-1b-it \
  --benchmark mmlu \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/mmlu/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/mmlu/random-4/pairs.jsonl \
  --max-pairs 150 \
  --position-controls distant \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-1b-it/mmlu
```

The PDF's adjacent comparison is restricted to three prespecified extension
settings: Gemma-3-1B/GSM8K, Llama-3.2-3B/ARC, and Mistral-7B/MMLU. For one of
those settings, request both position controls in one run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-1b-it \
  --benchmark gsm8k \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/random-4/pairs.jsonl \
  --max-pairs 150 \
  --position-controls distant adjacent \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-1b-it/gsm8k
```

`--fixed-window-run` is valid only for `--cohort primary`. `--pairs` and
`--max-pairs` are valid only for `--cohort extension`. A public reproduction
uses the hash-verified cohort selected from those inputs; it does not import an
unversioned legacy target-set file. Final-PDF extension runs require
`--max-pairs 150`. `--limit` is applied only after the full source eligibility,
boundary-audit, and extension-selection plan is fingerprinted. Exact-boundary
validity does not backfill the submitted selection. Position profiles and arm
plans are then frozen per executed target before any answer generation.

## Paper-defined intervention

For one source pair, let `Q_c` and `Q_p` be the clean and edited question
prompts, and let `C_c=(c_0,...,c_(L_C-1))` be the clean pre-answer token IDs.
At every token position `t`, run teacher-forced forward passes over the same
`C_c` under `Q_c` and `Q_p`. Let:

- `KL[t]` be `KL(p_clean[t] || p_edited[t])` for the next-token distributions;
- `rank_edited[t]` be the one-indexed rank of `c_t` under the edited context;
- `b_t` be the edited-context top-1 token at `t`.

Candidate positions satisfy `rank_edited[t] > 1`. The selected position is

```text
P = argmax KL[t] over candidate t.
```

The distant control `C` is the lower-median-KL candidate with
`abs(C-P) >= 3`. The final PDF defines the candidate, maximum-divergence,
lower-median, and minimum-distance rules. Exact-tie handling is not printed;
the submitted producer uses the smallest token index.

Every answer arm is a separate generation pass under the clean question. At a
position `t`, it sends exactly

```text
clean_prompt_ids + clean_cot_ids[:t] + [forced_token_id]
```

to the model, then decodes only newly generated token IDs. A keep arm forces
`c_t`; a replacement arm forces a selected edited-context top-1 token. The
prefix is never decoded and retokenized. The forced token itself is fixed
context and is not passed to answer extraction.

## Distant arms and Table 10

Requesting `--position-controls distant` generates these six arms:

| Position | Keep | Replace with `b_P` | Replace with `b_C` |
|---|---|---|---|
| selected `P` | `P_keep` | `P_bP` | `P_bC` |
| distant `C` | `C_keep` | `C_bP` | `C_bC` |

Table 10 uses only the diagonal replacement identities `P_bP` and `C_bC`.
The final PDF prints a paired-eligible denominator `n1`; the submitted producer
operationalizes it by requiring both keep arms to be correct and both diagonal
replacements to be non-noops. On that common set:

- `Sel.` is `P_keep` correct and `P_bP` incorrect;
- `Ctl.` is `C_keep` correct and `C_bC` incorrect.

Correctness always means equality with the benchmark's canonical gold answer.
Unextractable output is incorrect and remains an outcome. The two rates are
correct-to-incorrect events, not arbitrary answer-change rates.

## Distant factorial control and Table 11

The same six arms also form the two-position by two-source-token crossing. Let
`H_xy` be the correct-to-incorrect event at position `x` using token identity
derived at `y`, with `x,y` in `{P,C}`. The common distant-factorial denominator
requires both keeps to be correct and all four substitutions to be non-noops.
The final PDF does not add a distinct-token-ID or special-token filter. Equal
numeric `b_P` and `b_C` IDs therefore remain eligible when all four semantic
arms are non-noops. A submitted post-hoc producer additionally required
`b_P != b_C` and admissible source-token IDs. Its admissible pool is the set of
tokenizer vocabulary IDs inside the model's output-logit dimension, minus all
standard and added special IDs and marker tokens matching
`<unusedN>`, `<|reserved_special_token_N|>`, `[control_N]`, or `[unusedN]`.

The public output keeps these definitions separate:

- `distant_factorial` is the primary, final-PDF-literal denominator;
- `distant_factorial_submitted_producer` adds the historical distinct/admissible
  guard and reports its attrition from the literal denominator by reason.

Reclassifying the frozen fourteen-extension records gives `n=1,603` under the
PDF-literal rule, with selected/control event counts `912/3,206` and
`647/3,206` (`28.4%/20.2%`, difference `8.3` percentage points). The submitted
producer gives the printed `n=1,575`; all 28 additional exclusions have
`b_P == b_C`, while zero are caused by token inadmissibility. These are
historical audit values, not acceptance targets for a fresh public cohort.
Its exact selected/control counts are `892/3,150` and `633/3,150`.

The reported position-marginal rates are

```text
Loss_P = (H_PP + H_PC) / 2
Loss_C = (H_CP + H_CC) / 2
Delta  = Loss_P - Loss_C.
```

The final PDF reports the fourteen extensions only. It clusters intervals by
`(benchmark, sample_id)` and separately stratifies records by whether `P<C` or
`P>C`. The primary 172-pair source is descriptive and must not enter this
aggregate.

## Adjacent same-token control

With `--position-controls adjacent`, the selected position and `b_P` are
unchanged. The additional control is the nearest non-`P` position whose KL is
strictly below `KL[P]`. The same token `b_P` is forced at both positions:

| Position | Keep | Replace with `b_P` |
|---|---|---|
| selected `P` | reused `P_keep` | reused `P_bP` |
| adjacent `A` | `A_keep` | `A_bP` |

The paired denominator requires both keeps correct and both replacements
non-noop. The PDF calls this an adjacent lower-KL comparison and says it was
fixed before outcome generation. The submitted producer supplies the exact
unprinted rule: if equally near candidates occur on both sides, one SHA-256
bit of `short_setting_id|legacy_condition_code|sample_id` chooses the preferred
side, followed by the smallest token-index tie-break. The condition code is
`lxt4` for Attribution-4 and `rnd4` for Random-4. This outcome-blind hash tie-break is
`legacy-backed`; it does not turn the comparison into a randomized causal
experiment.

## Generation and extraction

The final-PDF protocol uses greedy bfloat16 decoding, left padding, and at most
512 newly generated tokens. The submitted producer fixes batch size one,
`do_sample=False`, no sampling temperature, and seed 42. The public runtime
records the complete generation arguments, physical visibility and logical
device, pinned model/tokenizer revision, effective EOS IDs, stop reason, and
whether the cap was reached.

Only newly generated IDs are decoded. The benchmark-specific primary
extractor runs first; the deterministic fallback is permitted only for an
empty primary result and is applied identically to every arm. Extraction
failure is a wrong outcome, not a reason to remove the record.

## Historical references

Table 10 contains the following paired denominators and percent rates. They
are retained as labelled references, not hard-coded acceptance targets for a
fresh run with corrected public pair preparation.

| Model | Task | `n1` | Sel. | Ctl. |
|---|---:|---:|---:|---:|
| Gemma 1B | GSM8K | 120 | 38.3 | 29.2 |
| Gemma 1B | MMLU | 119 | 27.7 | 22.7 |
| Gemma 1B | ARC-C | 118 | 18.6 | 10.2 |
| Gemma 4B | GSM8K primary | 153 | 26.8 | 15.0 |
| Gemma 4B | MMLU | 125 | 18.4 | 15.2 |
| Gemma 4B | ARC-C | 125 | 16.8 | 11.2 |
| Llama 1B | GSM8K | 106 | 47.2 | 30.2 |
| Llama 1B | MMLU | 107 | 37.4 | 22.4 |
| Llama 1B | ARC-C | 106 | 22.6 | 17.9 |
| Llama 3B | GSM8K | 113 | 38.1 | 19.5 |
| Llama 3B | MMLU | 119 | 37.0 | 15.1 |
| Llama 3B | ARC-C | 121 | 25.6 | 14.0 |
| Mistral 7B | GSM8K | 103 | 38.8 | 24.3 |
| Mistral 7B | MMLU | 119 | 31.1 | 14.3 |
| Mistral 7B | ARC-C | 128 | 29.7 | 11.7 |
| fourteen-extension aggregate | -- | 1,629 | 30.2 | 18.2 |

Table 11 reports distant pooled `n=1,575`, Loss P/C `28.3/20.1`, and
`+8.2 [6.0,10.4]` percentage points. The `P<C` stratum is `n=1,044`,
`32.3/16.6`, `+15.7 [13.0,18.4]`; the `P>C` stratum is `n=531`,
`20.5/27.0`, `-6.5 [-10.0,-3.0]`.

The adjacent pooled row is `n=391`, `31.7/28.6`,
`+3.1 [-1.8,7.9]`. Its three cells are Gemma-1B/GSM8K (`n=127`,
`37.0/38.6`), Llama-3B/ARC (`n=133`, `24.8/21.8`), and
Mistral-7B/MMLU (`n=131`, `33.6/26.0`). Every adjacent confidence interval
includes zero.

The `n=1,575` row is the submitted-producer-compatible denominator. The same
stored records reclassified under the final PDF's literal text give `n=1,603`,
Loss P/C `28.4/20.2`, and `+8.3` percentage points. A later table builder must
label both instead of presenting either operationalization as the other.

Figure 5's machine-readable worked-case reference is Gemma-4B/GSM8K,
Attribution-4 (`lxt4`), `gsm8k_00556`: `P=23`, `C=60`, `KL[P]=8.785974`,
clean-token ranks `1` (clean) and `8` (edited), clean token `" thrice"`, and
selected edited top-1 token `" twice"`. The selected keep/replacement answers
are `160/120`; the distant keep/replacement answers are `160/160`.

## Paper-defined and legacy-backed details

The final PDF defines the clean-question intervention, candidate rank rule,
maximum-KL selected position, distant lower-median control at distance at
least three, local top-1 replacements, correct-to-wrong endpoint, paired
eligibility, distant four-arm and adjacent same-token comparisons, the 15
source cells, extension-only pooling, cluster key, and greedy bfloat16
512-token generation.

The final PDF requires exact token boundaries. The submitted selection stage
operationalized alignment as equality of the clean pre-answer suffix obtained
after slicing each full input by its separately tokenized prompt length. The
public implementation preserves that rule for cohort membership, then audits
that both full inputs preserve their exact prompt-ID prefixes and share the
same clean-CoT suffix. The manifest distinguishes these two stages explicitly.

Zero-based position storage, smallest-index maximum/median tie handling,
median-low implementation,
adjacent nearest-lower rule and SHA side tie-break, batch size one, and detailed
generator/extractor kwargs are compatible submitted-producer details not
printed by the PDF. Manifests label them
`legacy-backed` rather than silently promoting them to paper-defined claims.

## Outputs

One setting writes exactly four public files:

- `one_token_records.jsonl`: one completed intervention record per executable
  target, including the byte-bound source identity, aligned token plan,
  position profile, chosen positions and tokens, every requested arm, and all
  denominator/event flags;
- `pair_status_records.jsonl`: every source pair with source eligibility,
  cap/quota/selection, token alignment, position availability, execution, and
  explicit exclusion reason. A position-excluded selected case embeds its
  hash-bound input plan and profile so completed resume does not trust a
  self-declared exclusion;
- `one_token_summary.json`: source and selection funnels, arm termination and
  extraction diagnostics, separate Table 10/PDF-literal factorial/submitted-
  producer factorial/adjacent integer numerator-denominator metrics, attrition,
  and labelled final-PDF references;
- `run.json`: arguments, final-PDF fingerprint, frozen full plan, source and
  runtime fingerprints, checkpoint registry, failures, output hashes, and
  completion state.

This operation is the setting-level GPU producer only. Cross-setting Table 10,
Table 11, clustered intervals, strata, and Figure 5 validation belong to a
separately versioned CPU artifact-building operation. The producer emits the
integer events, both factorial definitions, and cluster keys needed by that
step; it does not pool the primary cell or invent a per-setting headline test.

Both the run manifest and summary carry an explicit comparability label.
`--limit` always produces `partial-smoke-run`. An unlimited run is
`fresh-paper-protocol-run` only with the paper-sized 172-target primary or
150-target extension plan and, in the three prespecified settings, both the
distant and adjacent controls. Every selected target must also pass the exact
boundary audit. Any short source plan, invalid selected boundary, or omitted
adjacent comparison is retained as a named limitation rather than being
presented as a complete paper-shaped setting. Fresh public preparation follows
the paper source protocol but does not prove byte-identical historical cohort
membership; the two facts are separate manifest fields.

## Restart and mutation safety

The full source cohort, extension sample, and aligned input-plan hashes are
fingerprinted before `--limit` affects execution. For each executed target,
the position profile and requested arm plan are checkpointed before its first
answer arm is generated. Each selected target owns an atomic checkpoint that
may grow one arm at a time. Reuse requires exact source, protocol, model/tokenizer
revision, plan, profile, arm, generation, extraction, and checkpoint hashes.
The runtime also fingerprints the transitive Python code bundle that can affect
this producer and the exact sorted admissible-token-ID set. Each private
checkpoint registry entry is bound to that runtime-provenance SHA, preventing
resume from mixing results across code or equal-sized-but-different token pools.

A failed run retains valid private checkpoints but publishes no partial JSONL
or summary. Every upstream file is rehashed before publication. Source drift
during GPU work therefore prevents public outputs. A completed run removes
its private checkpoint directory and stores hashes for the three public data files;
completed `--resume` reconstructs records, statuses, summaries, position and
eligibility metrics, stop metadata, source identity, and every position
exclusion from public evidence without loading model weights.
