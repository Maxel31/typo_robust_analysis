# typo-cot reproduction package

This package contains the public reproduction interface for **“Edited-Word
Activation Patching Reverses Selected Typo-Induced Answer Changes after
Tokenization.”**

The final paper is the primary experimental specification. Its canonical
SHA-256 is available from the catalog's single source of truth:

```bash
uv run --project projects/typo-cot typo-cot experiments source
```

See
[`docs/paper-experiments.md`](docs/paper-experiments.md) for the transcribed
operation matrix, denominators, target directory layout, and one-command-per-
experiment interface.

## Setup

From the repository root:

```bash
uv sync --project projects/typo-cot
```

Pair preparation additionally needs the GPU/LRP dependencies:

```bash
uv sync --project projects/typo-cot --extra lrp
```

## Available commands

The experiment catalog is implemented and does not require a GPU:

```bash
uv run --project projects/typo-cot typo-cot experiments list
uv run --project projects/typo-cot typo-cot experiments list --format json
uv run --project projects/typo-cot typo-cot experiments show cot-swap
uv run --project projects/typo-cot typo-cot experiments show clean-prefix-scan --format json
```

Each catalog entry includes its stable `target_command`, paper section, required
operation-specific arguments, cohort, intervention, readout, outputs, compute
class, and implementation status. Direct experiment runners are added in
separate reviewed PRs; only entries marked `implemented` are runnable.

## Prepare clean/edited pairs

`prepare-edited-pairs` performs the paper's input-preparation operation: greedy
clean generation, first-CoT-token AttnLRP targeting, up to four seeded
single-character edits, greedy edited generation, deterministic answer
extraction, and edited-word-final token alignment. Run it separately for each
model, benchmark, and targeting condition:

```bash
uv run --project projects/typo-cot --extra lrp typo-cot prepare-edited-pairs \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --targeting attribution-4 \
  --num-edits 4 \
  --output-dir results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4
```

Use `--targeting random-4` for the within-item control sampled after excluding
the four Attribution-4 tokens. `--num-edits` selects how many of the sampled
targets to edit (1–4). The paper defaults are `--seed 42` and
`--max-new-tokens 512`; `--limit 1` is available for a GPU smoke run. A stopped
run can be continued with `--resume`.
The command writes:

- `pairs.jsonl`: versioned per-item clean/edited generations, target-attempt
  provenance, actual distinct edited-word spans, and clean/edited word-final
  token indices;
- `run.json`: frozen arguments, environment and dataset provenance, progress,
  failures, and record counts.

The target attempts and aligned words are deliberately separate. AttnLRP ranks
tokens, and the final paper reports that some lower-ranked attempts land outside
their intended token or share a word. The pair format preserves that behavior
instead of silently converting targeting into a word-level selector.
Every selected dataset item is retained in `pairs.jsonl`, including an item for
which no eligible character edit exists. Such a record has empty
`target_attempts` and `aligned_words`, identical clean/edited text and generation,
and `answer_changed: false`. It remains in population-level targeting-fidelity
denominators but is ineligible for a later edited-word activation patch.
The complete field contract and coordinate conventions are documented in
[`docs/prepare-edited-pairs.md`](docs/prepare-edited-pairs.md).

Dataset cohort size follows the final paper setting rather than one global
MMLU cap. MMLU uses 100 examples per subject (5,700 items) for
Qwen2.5-7B-Instruct, Gemma-3-12B-IT, and Gemma-3-27B-IT, and 50 per subject
(2,850 items) for the other paper models. MMLU-Pro uses 100 per subject for
every model. The selected cohort size and versioned selection rule are recorded
in `run.json` provenance.

## Audit targeting fidelity

After preparing every four-edit model/benchmark/targeting cell, aggregate the
paper's Appendix A input-quality checks in one CPU-only operation:

```bash
uv run --project projects/typo-cot typo-cot targeting-fidelity-audit \
  --pairs-root results/prepare-edited-pairs \
  --output-dir results/targeting-fidelity-audit
```

The input root may contain any directory layout; the audit discovers completed
`prepare-edited-pairs/v1` outputs recursively and reads setting identity from
their records and manifests rather than from machine-specific path names. It
rejects partial, duplicate, mixed, or non-four-edit inputs instead of silently
changing the Appendix A denominator. It expects the paper seed 42 by default;
use `--expected-seed` only for a separately labelled sensitivity run. The
audit independently replays cumulative landing offsets and every SHA-seeded
character edit instead of trusting recorded landing/operation flags. The
command writes:

- `targeting_fidelity_records.jsonl`: one validation/audit row per prepared
  item;
- `targeting_fidelity.csv`: per-setting and pooled target-landing, distinct-word,
  and prepared-pair gold-option rates, including Attribution ranks 1--4
  separately;
- `operation_counts.json`: substitution, duplication, and deletion counts by
  setting and targeting condition;
- `run.json`: input/output hashes, arguments, paper fingerprint, counts, and
  the reported Appendix A reference values used for comparison.

Rates use items or edit attempts exactly as named by each column. In
particular, the four-distinct-word rate is item-level, target fidelity is
attempt-level, and the prepared-pair gold-option rate is restricted to
multiple-choice inputs. The paper's 21.5% gold-option value uses the later
Attribution-4 CoT-swap included cohort, so this pair-only command records that
reference as not directly computable rather than comparing unlike denominators.
`run.json` permits a `descriptive_only` paper comparison only after checking
the exact 42-setting grid, archival per-cell counts, paired-arm
sample/provenance identity, seed 42, and the 512-token generation cap;
otherwise its status is `not_comparable`.
See
[`docs/targeting-fidelity-audit.md`](docs/targeting-fidelity-audit.md) for the
schemas and paper comparison rules.

## Tests

```bash
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_paper_experiment_catalog.py
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_targeting_fidelity_audit.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests
```

The contract tests enforce the final-PDF fingerprint, complete operation list,
descriptive names, unique command slugs, CLI JSON schema, and documentation
coverage. The full suite also exercises pair preparation and its model, dataset,
prompt, answer-extraction, and AttnLRP adapters without downloading model
weights.

## Repository scope

Only the public experiment catalog, implemented runners, their runtime
dependencies, and tests are tracked here. Intermediate Exp1–20 scripts,
machine-specific configuration, and archived outputs are deliberately excluded
because they are not the paper's reproduction interface.
