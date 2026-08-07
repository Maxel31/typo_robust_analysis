# Paper-aligned experiment contract

## Canonical source and naming

This document is transcribed from the final 19-page PDF:

- Title: *Edited-Word Activation Patching Reverses Selected Typo-Induced Answer Changes after Tokenization*
- Filename: `Edited-Word Activation Patching Reverses Selected Typo-Induced Answer Changes after Tokenization.pdf`

Read the canonical SHA-256 from the catalog's single source of truth:

```bash
uv run --project projects/typo-cot typo-cot experiments source
```

The PDF is authoritative. Old experiment numbers, worktree READMEs, analysis
notes, archived outputs, and implementation comments are secondary evidence.
When they conflict, implementation must follow the PDF and record the mismatch
in provenance rather than silently preserving the old behavior.

The PDF itself is intentionally not added by this contract PR; the fingerprint
anchors the user-supplied final submission without assuming publication rights
or a permanent public URL. Once the paper artifact is public, its stable URL can
be added without changing the experiment contract.

Public identifiers describe operations. RQ1/RQ2/RQ3 remain only in the
`paper_question` metadata so a reader does not need the paper's question
numbering to understand a branch, directory, or command.

## Frozen shared setup

| Item | Paper specification |
|---|---|
| Tasks | GSM8K, MATH-500, MMLU, MMLU-Pro, ARC, and CSQA |
| Base patching families | Gemma-3-4B, Llama-3.2-3B, Mistral-7B, and Qwen2.5-3B |
| Extended layer scans | Ten models from 0.5B to 27B |
| CoT-swap main pool | Five 1B–7B models × five tasks (25 settings) |
| CoT-swap scale check | MMLU, including Gemma 12B/27B, Llama 70B, and Qwen 72B |
| Decoding | Greedy, bfloat16, left padding, at most 512 generated tokens |
| CoT-swap answer span | At most 16 generated tokens |
| Few-shot prompts | GSM8K 8; MMLU/MMLU-Pro/ARC/CSQA 5; MATH-500 4 |
| Random seed | 42 |
| Answer extraction | Task-specific deterministic extraction; unextractable is failure |
| Paper environment | Python ≥3.12; PyTorch 2.10.0; Transformers 4.57.6; Accelerate 1.12.0; LXT 2.1 |
| Logged extension hardware | NVIDIA RTX PRO 6000 Blackwell Max-Q, 95 GB |

For every item, the perturbation procedure attempts one character edit in each
of up to four eligible word spans. Attribution-4 selects the four largest
absolute AttnLRP relevances to the maximum logit immediately after the first CoT
token. Random-4 samples within the item after excluding those top-ranked spans;
it is not a population-random natural-typo condition. At each target, seeded
random order chooses the first applicable operation:

1. substitute a lowercase QWERTY neighbor while preserving case;
2. duplicate the selected character;
3. delete the selected character unless the word has one character.

Clean and edited spans align at each edited word's final token. This final-token
rule is mandatory for all activation transfer and must not be replaced by the
older substring-based alignment.

## Experiment inventory

`uv run typo-cot experiments list` is the machine-readable source for status.
The direct commands below are the stable target interface. During the staged
refactor, a direct command is executable only when its catalog status is
`implemented`.

| Operation | Paper link | Core denominator/readout |
|---|---|---|
| `prepare-edited-pairs` | §3.1, Appendix A, Table 4 | Versioned clean/edited pair and alignment records |
| `targeting-fidelity-audit` | §3.1, Appendix A | Edit landing, gold-option, and operation counts |
| `layerwise-kl-patching` | §3.3, §4.1, Appendix B | Selected clean-correct/edited-wrong pairs; complete finite grids with untreated KL > 1e-9; normalized first-CoT-token KL |
| `layerwise-answer-patching` | §3.2–3.3, §4.1 | Separate eight-setting free-generation scans; at most 300 pooled anchors are rechecked into one fixed n=94–226 denominator per setting |
| `fixed-window-answer-patching` | §3.3, §4.1, Appendix B | Frozen [0,6) answer patch; [6,12) prespecified MMLU-Pro comparison |
| `patch-coordinate-controls` | §3.3, §4.1, Appendix B | Primary 172 pairs; correct, +2 offset, cross-item, and identity controls |
| `patch-position-controls` | §3.3, §4.1, Table 5, Appendix B | Gemma-3-4B/GSM8K Attribution-4 layerwise-KL cohort (published n=109), held common across edited-word, prompt-final, and question-final positions |
| `patch-text-combination` | §3.5, §4.1, Table 2 | Descriptive patch absent/present × zero/full clean text on 172 pairs |
| `cot-swap` | §3.4, §4.2, Appendix C | Clean-correct A denominator; restoration conditions on B≠A |
| `answer-line-deletion` | §3.4, §4.2, Table 1 | Same eligible CoT-swap cases after final answer-line deletion |
| `clean-prefix-scan` | §3.5, §4.3, Appendix D | Valid scans whose fresh k=0 rerun is wrong; point/stable correctness |
| `one-token-prefix-replacement` | §3.5, §4.3, Appendix D | Eligible selected/control replacements; correct-to-wrong rate |
| `edit-count-sensitivity` | Appendix C, Table 8 | Accuracy and conditional CoT-swap restoration at 1/2/4 edits |
| `model-scale-cot-swap` | Appendix C, Table 9 | Same first 500 MMLU IDs across the model scale ladder |
| `typo-warning-prompt` | §4.3, Appendix E | Paired edited-input accuracy with/without the warning |
| `input-corrector-audit` | §4.3, Appendix E, Table 12 | Edited-word restoration, intact-word changes, and provenance controls |
| `restoration-order-accuracy` | §4.3, Appendix E, Table 13 | Accuracy after restoring 1/2/3 words in three orders |

These rates are not interchangeable. In particular, layerwise KL, layerwise
answer generation, fixed-window answer generation, complete-text CoT swap, and
clean-prefix scans have distinct cohorts and denominators. Reporting code must
never pool them into stages of a single causal path.

### Headline cohort invariants

These counts are acceptance checks for later runner and aggregation PRs. A
reproduction may expose additional sensitivity rows, but it must not silently
substitute one row's denominator for another. The source column points to the
submitted PDF page and labeled table, figure, or appendix paragraph used for
the transcription; the PDF fingerprint above identifies that exact artifact.

| Analysis | Paper cohort invariant | Final-PDF source |
|---|---|---|
| Layerwise KL patching | 32 completed settings; 30 headline settings after excluding MATH cells with n=13 and n=27; 7,919 retained pairs | p. 12, Table 3; p. 13, Appendix B |
| Layerwise answer patching | Eight settings (four base models × GSM8K/MMLU), with n=94–226 per curve | p. 6, Figure 2; p. 12, Table 3 |
| Fixed-window answer patching | Six planned settings; published restoration n=1,241 with 800 successes; published reciprocal induction n=1,458 with 871 reported changes | p. 6, §4.1; p. 15, Table 6 |
| Primary coordinate controls | The same 172 Gemma-3-4B/GSM8K pairs: correct coordinates 129, offset 44, cross-item donor 42 | p. 6, §4.1; p. 15, Table 7 |
| Position reachability | The same 109 pairs for all three patch positions | p. 13, Appendix B |
| Patch/text crossing | The same 172 pairs in all cells: 0, 129, 168, and 171 correct | p. 7, Table 2 |
| Prespecified MMLU-Pro windows | Qwen2.5-3B n=97 and Mistral-7B n=120 | p. 6, §4.1; p. 15, Table 7 |
| Complete-text CoT swap | 19,550 clean-correct cases; 4,634 B changes; 3,539 B-to-C restorations | p. 12, Table 3; p. 13, Appendix C |
| Answer-line deletion | GSM8K n=333 and MMLU n=450 in the three-model controls | p. 7, Table 1; p. 12, Table 3 |
| Clean-prefix extensions | 2,100 deterministic targets from 5,918 capped candidates; 2,094 valid scans; 1,858 fresh k=0 errors | p. 12, Table 3 and Appendix A; p. 17, Table 10 |
| One-token diagnostic | Primary n=153 and extensions n=1,629; distant common four-arm subset n=1,575; adjacent subset n=391 | p. 17, Table 10; p. 18, Table 11 |

The fixed-window counts in this table are publication references, not an
acceptance requirement for a fresh run. Preserved historical records show that
the published induction aggregate treated some unextractable patched answers as
incorrect changes, whereas the final PDF explicitly defines an unextractable
answer as a failed intervention readout. The public runner follows the latter
rule, records this historical discrepancy, and never counts an unextractable
patched answer as restoration or induction.

For layerwise KL patching, the normalized restoration readout is

```text
1 - KL(p_clean || p_patched) / KL(p_clean || p_edited)
```

and the reciprocal induction calculation swaps the clean and edited roles.
Only finite untreated denominators greater than `1e-9` and complete finite layer
grids are valid. Early/middle/late summaries first aggregate within a pair and
setting, then macro-average settings with equal weight.

## Stable command shapes

The paths are examples of the intended public layout. Each operation owns its
arguments and output directory; no machine-specific archive or worktree path is
implicit. The examples assume `cd projects/typo-cot`; from the repository root,
add `--project projects/typo-cot` immediately after `uv run`. Commands that use
AttnLRP or activation patching also require `--extra lrp` before `typo-cot`.

### Pair preparation and input audits

```bash
uv run --extra lrp typo-cot prepare-edited-pairs \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --targeting attribution-4 --num-edits 4 \
  --output-dir results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4

uv run typo-cot targeting-fidelity-audit \
  --pairs-root results/prepare-edited-pairs \
  --output-dir results/targeting-fidelity-audit
```

### Edited-word activation patching

```bash
uv run --extra lrp typo-cot layerwise-kl-patching \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --targeting attribution-4 \
  --directions clean-to-edited edited-to-clean \
  --gpu-id 0 \
  --output-dir results/layerwise-kl-patching/gemma-3-4b-it/gsm8k

uv run --extra lrp typo-cot layerwise-answer-patching \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --attribution-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --random-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/random-4/pairs.jsonl \
  --directions clean-to-edited edited-to-clean --max-pairs 300 \
  --gpu-id 0 \
  --output-dir results/layerwise-answer-patching/gemma-3-4b-it/gsm8k

uv run --extra lrp typo-cot fixed-window-answer-patching \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --pairs \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/random-4/pairs.jsonl \
  --layers 0:6 --directions clean-to-edited edited-to-clean \
  --gpu-id 0 \
  --output-dir results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot patch-coordinate-controls \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --layers 0:6 \
  --controls correct offset-2 cross-item self-copy \
  --gpu-id 0 \
  --output-dir results/patch-coordinate-controls/gemma-3-4b-it/gsm8k

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot patch-position-controls \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --layerwise-kl-run \
    results/layerwise-kl-patching/gemma-3-4b-it/gsm8k/attribution-4 \
  --positions edited-word prompt-final question-final \
  --gpu-id 0 \
  --output-dir results/patch-position-controls/gemma-3-4b-it/gsm8k

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot patch-text-combination \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --layers 0:6 \
  --gpu-id 0 \
  --output-dir results/patch-text-combination/gemma-3-4b-it/gsm8k
```

### Complete-text and prefix interventions

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot cot-swap \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --targeting attribution-4 --gpu-id 0 \
  --output-dir results/cot-swap/gemma-3-4b-it/gsm8k/attribution-4

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot answer-line-deletion \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --cot-swap-run results/cot-swap/gemma-3-4b-it/gsm8k/random-4 \
  --max-pairs 150 --gpu-id 0 \
  --output-dir results/answer-line-deletion/gemma-3-4b-it/gsm8k

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot clean-prefix-scan \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --cohort primary \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --relative-budgets 0 .02 .05 .08 .12 .16 .20 .25 .325 .40 .50 .65 .80 1 \
  --absolute-budgets 1 2 4 8 16 32 64 \
  --gpu-id 0 \
  --output-dir results/clean-prefix-scan/gemma-3-4b-it/gsm8k

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot clean-prefix-scan \
  --model google/gemma-3-1b-it --benchmark gsm8k \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/random-4/pairs.jsonl \
  --max-pairs 150 \
  --relative-budgets 0 .02 .05 .08 .12 .16 .20 .25 .325 .40 .50 .65 .80 1 \
  --absolute-budgets 1 2 4 8 16 32 64 \
  --gpu-id 0 \
  --output-dir results/clean-prefix-scan/gemma-3-1b-it/gsm8k

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot one-token-prefix-replacement \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --cohort primary \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --position-controls distant --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-4b-it/gsm8k

CUDA_VISIBLE_DEVICES=0 \
uv run --project projects/typo-cot --extra lrp typo-cot one-token-prefix-replacement \
  --model google/gemma-3-1b-it --benchmark gsm8k \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/random-4/pairs.jsonl \
  --max-pairs 150 --position-controls distant adjacent --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-1b-it/gsm8k
```

The prefix grid is the union of the shown absolute values and
`round(relative_budget * clean_cot_length)`. Relative point correctness and
stable-through-all-later correctness use the same fresh-k=0-wrong denominator.
At an absolute budget `a`, exact point correctness uses only rows with
`L_C >= a`, while stable recovery (`k* <= a`) keeps the common fresh-k=0-wrong
denominator. Non-monotonicity means at least two correctness transitions across
adjacent tested budgets. The primary source is the hash-verified clean-to-edited
denominator of the referenced fixed-window run. Extensions instead revalidate
the completed Attribution-4 and Random-4 pair preparations and deterministically
select at most 150 targets without importing CoT-swap's different template and
answer-span cohort.

The relative rule, denominator, and outcome definitions are paper-defined. The
exact dense budget values, Python ties-to-even rounding, per-arm 400 cap,
proportional/systematic selection, batch size one, and pre-answer locator are
submitted-producer details that remain explicitly `legacy-backed`. The command
writes `prefix_scan_records.jsonl`, `pair_status_records.jsonl`,
`prefix_scan_summary.json`, and `run.json`; the later paper-artifact builder,
not this setting runner, performs Figure 3's 14-setting clustered bootstrap.

### Sensitivity, scale, and input correction

```bash
uv run typo-cot edit-count-sensitivity \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --edit-counts 1 2 4 --pairs-root data/cohorts/edit-count \
  --output-dir results/edit-count-sensitivity/gemma-3-4b-it/gsm8k

uv run typo-cot model-scale-cot-swap \
  --models google/gemma-3-1b-it google/gemma-3-4b-it \
    google/gemma-3-12b-it google/gemma-3-27b-it \
    meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
    meta-llama/Llama-3.1-70B-Instruct mistralai/Mistral-7B-Instruct-v0.3 \
    Qwen/Qwen2.5-72B-Instruct \
  --benchmark mmlu --pairs-root data/cohorts/model-scale-cot-swap \
  --output-dir results/model-scale-cot-swap/mmlu

uv run typo-cot typo-warning-prompt \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --pairs data/cohorts/input-correction/gemma-3-4b-it_gsm8k.jsonl \
  --output-dir results/typo-warning-prompt/gemma-3-4b-it/gsm8k

uv run typo-cot input-corrector-audit \
  --corrector pyspellchecker --model google/gemma-3-4b-it --benchmark gsm8k \
  --pairs data/cohorts/input-correction/gemma-3-4b-it_gsm8k.jsonl \
  --output-dir results/input-corrector-audit/pyspellchecker/gemma-3-4b-it/gsm8k

uv run typo-cot restoration-order-accuracy \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --pairs data/cohorts/restoration-order/gemma-3-4b-it_gsm8k.jsonl \
  --order high-relevance seeded-random low-relevance --budgets 1 2 3 \
  --output-dir results/restoration-order-accuracy/gemma-3-4b-it/gsm8k
```

## Target package layout

Each operation moves behind an importable module with a thin CLI adapter. Shared
model loading, deterministic generation, alignment, schemas, resumable record
I/O, and provenance live under `common`; operations must not import another git
worktree or rely on an absolute local path.

```text
projects/typo-cot/
├── configs/
│   ├── paper/                    # frozen model/task/grid manifests
│   └── smoke/                    # tiny public validation manifests
├── data/
│   ├── cohorts/                  # released IDs/manifests, not model caches
│   └── fixtures/                 # small test inputs
├── docs/
│   ├── paper-experiments.md      # this contract
│   └── data_provenance.md
├── src/typo_cot/
│   ├── cli.py
│   └── experiments/
│       ├── common/               # schemas, seeds, generation, alignment, I/O
│       ├── prepare_edited_pairs/
│       ├── layerwise_kl_patching/
│       ├── layerwise_answer_patching/
│       ├── fixed_window_answer_patching/
│       ├── cot_swap/
│       ├── clean_prefix_scan/
│       ├── one_token_prefix_replacement/
│       └── ...                   # remaining operation-named modules
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
└── results/                      # ignored runtime outputs, one dir per operation
```

Every output directory must contain per-item records plus a run manifest with
the command, resolved arguments, input hashes, model revision, dependency
versions, seed, timestamps, and completion state. Interrupted runs must resume
without duplicating records.

## Branch and review sequence

Branches and PRs are also named for operations, for example:

1. `agent/paper-experiment-contract`
2. `agent/prepare-edited-pairs`
3. `agent/layerwise-kl-patching`
4. `agent/layerwise-answer-patching`
5. `agent/fixed-window-answer-patching`
6. `agent/cot-swap`
7. `agent/clean-prefix-scan`
8. the remaining control and appendix operations, one per branch
9. `agent/build-paper-artifacts`
10. `agent/remove-legacy-experiment-code`

Each PR targets `develop`. Tests are written and observed failing before the
implementation, then unit tests, CPU integration tests, and a proportionate GPU
smoke run must pass. No next operation starts until CI and all actionable review
threads for the current PR are clear.
