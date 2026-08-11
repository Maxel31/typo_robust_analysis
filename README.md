# Edited-Word Activation Patching

[English](README.md) | [日本語](README.ja.md)

Reproduction code for **“Edited-Word Activation Patching Reverses Selected
Typo-Induced Answer Changes after Tokenization.”**

The public reproduction package is in [`projects/typo-cot`](projects/typo-cot).
Experiment names describe the operation they perform—for example,
`layerwise-kl-patching`, `cot-swap`, and `clean-prefix-scan`. Paper labels such
as RQ1 are used only as cross-references.

## Paper and reproduction

Start with the setup and per-experiment commands below. The final 19-page paper
defines the experimental design, cohorts, denominators, and reported results
that those commands reproduce. The catalog prints its SHA-256 so that a local
copy of the paper can be checked when needed:

```bash
uv run --project projects/typo-cot typo-cot experiments source
```

To verify a local copy of the paper, compare these two command outputs:

```bash
sha256sum "/path/to/Edited-Word Activation Patching Reverses Selected Typo-Induced Answer Changes after Tokenization.pdf"
uv run --project projects/typo-cot typo-cot experiments source
```

The hash is an integrity check, not an extra input to the experiments. If an
old script, branch, result note, or README disagrees with the final paper, use
the paper's specification. The transcribed experiment contract is documented in
[`paper-experiments.md`](projects/typo-cot/docs/paper-experiments.md).

## Quick start

```bash
git clone https://github.com/Maxel31/typo_robust_analysis.git
cd typo_robust_analysis
git switch develop

uv sync --project projects/typo-cot
uv run --project projects/typo-cot typo-cot experiments list
uv run --project projects/typo-cot typo-cot experiments show layerwise-kl-patching
```

All paper operations in the catalog are implemented. `experiments list` reports
the compute class, required arguments, outputs, and implementation status for
each operation.

GPU experiments use the paper-locked `lrp` environment. CPU-only catalog
inspection and contract tests require no model download.

## ARR additions and robustness training

The command interfaces for the ARR rebuttal additions and prospective
typo-robustness training are frozen before implementation in
[`rebuttal_analysis_plan_v1.md`](projects/typo-cot/docs/rebuttal_analysis_plan_v1.md)
and
[`robustness_training_plan_v1.md`](projects/typo-cot/docs/robustness_training_plan_v1.md).
The package README shows one descriptive command per experiment and labels
these future commands `interface-frozen` until their reviewed implementation is
merged. Training is published only after held-out evaluation demonstrates a
robustness improvement while preserving clean performance.

## Repository map

```text
.
├── .github/workflows/              # automated PR review
├── README.md
├── README.ja.md
├── pyproject.toml                  # uv workspace
├── uv.lock                         # locked reproduction environment
└── projects/typo-cot/
    ├── README.md                   # package setup and current commands
    ├── README.ja.md                # Japanese setup and current commands
    ├── docs/                       # paper contract and provenance
    ├── results/                    # ignored local outputs (.gitkeep only)
    ├── src/typo_cot/               # importable implementation
    └── tests/                      # CPU unit/contract tests
```

## Repository scope

The root `uv` workspace contains only the public reproduction package. Model
and dataset caches, generated results, the supplied paper PDF, and the local
archive used to audit historical figure/table sources are intentionally not
tracked. Pull requests are reviewed by
`.github/workflows/claude-code-review.yml`.

Changes are developed as one operation per branch and pull request against
`develop`. CI and actionable review feedback are resolved before the next
operation is merged.
