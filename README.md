# Edited-Word Activation Patching

Reproduction code for **“Edited-Word Activation Patching Reverses Selected
Typo-Induced Answer Changes after Tokenization.”**

The public reproduction package is in [`projects/typo-cot`](projects/typo-cot).
Experiment names describe the operation they perform—for example,
`layerwise-kl-patching`, `cot-swap`, and `clean-prefix-scan`. Paper labels such
as RQ1 are used only as cross-references.

## Source of truth

The final 19-page paper PDF is the primary source for experimental design,
cohorts, denominators, and reported evidence. Its SHA-256 is:

```text
2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0
```

If an old script, branch, result note, or README disagrees with that PDF, the
PDF wins. The transcribed experiment contract is documented in
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

The refactor is being published operation by operation. `catalogued` means the
paper contract and future command are fixed but the public runner is not yet
available; `implemented` means the command can run. The catalog reports this
status explicitly, so the README does not imply that unfinished runners work.

GPU experiments will require the locked `lrp` environment after its dedicated
environment PR lands. CPU-only catalog inspection and contract tests require no
model download.

## Repository map

```text
.
├── README.md
├── pyproject.toml                 # uv workspace
├── uv.lock                        # shared environment lock
├── projects/typo-cot/
│   ├── README.md                  # package setup and current commands
│   ├── configs/                   # versioned experiment configuration
│   ├── docs/                      # paper contract and provenance
│   ├── src/typo_cot/              # importable implementation
│   └── tests/                     # CPU unit/contract tests
└── utils/                         # shared utilities used by typo-cot
```

Changes are developed as one operation per branch and pull request, always
against `develop`. A subsequent operation starts only after CI and all
actionable review feedback on the current PR have been resolved.
