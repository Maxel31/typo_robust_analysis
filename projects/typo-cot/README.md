# typo-cot reproduction package

This package contains the public reproduction interface for **“Edited-Word
Activation Patching Reverses Selected Typo-Induced Answer Changes after
Tokenization.”**

The final paper, identified by SHA-256
`2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0`,
is the primary experimental specification. See
[`docs/paper-experiments.md`](docs/paper-experiments.md) for the transcribed
operation matrix, denominators, target directory layout, and one-command-per-
experiment interface.

## Setup

From the repository root:

```bash
uv sync --project projects/typo-cot
```

GPU runners will use the separately locked LRP environment. Until the
environment-lock PR is merged, do not treat the old optional dependency pins as
the final paper environment.

## Available commands

The experiment catalog is implemented and does not require a GPU:

```bash
uv run --project projects/typo-cot typo-cot experiments list
uv run --project projects/typo-cot typo-cot experiments list --format json
uv run --project projects/typo-cot typo-cot experiments show cot-swap
uv run --project projects/typo-cot typo-cot experiments show clean-prefix-scan --format json
```

Each catalog entry includes its stable operation command, paper section,
required operation-specific arguments, cohort, intervention, readout, outputs,
compute class, and implementation status. Direct experiment runners are added
in separate reviewed PRs; only entries marked `implemented` are runnable.

## Tests

```bash
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_paper_experiment_catalog.py
```

The contract tests enforce the final-PDF fingerprint, complete operation list,
descriptive names, unique command slugs, CLI JSON schema, and documentation
coverage.

## Historical material

The former Step0/Exp1–20 development README is retained at
[`docs/legacy-development-readme.md`](docs/legacy-development-readme.md) for
provenance only. It describes intermediate experiments and machine-specific
paths and is not a reproduction specification.
