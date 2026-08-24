# Mistral state-free factorial, 64M-token production contract

This document defines the production five-arm comparison identified by
`mistral-state-free-probe-factorial/v1`. It is intentionally separate from the
faithful Kojima reproduction and from the earlier Gemma pilot.

## Scientific question and five arms

The experiment crosses two interpretable axes while holding every other input
fixed:

1. **LoRA placement**: every decoder layer, the suffix beginning at the frozen
   Linear Probe transition, or a count-matched deterministic random layer set.
2. **Output supervision**: every aligned non-edited next-token coordinate, or
   only offsets +2 through +16 downstream of an edited-word-final coordinate on
   noisy rows (clean rows still use all aligned targets).

The resulting conditions are:

| condition | LoRA placement | noisy output scope |
|---|---|---|
| `factorial-all-layers-all-tokens` | all 32 decoder layers | aligned non-edited tokens |
| `factorial-all-layers-downstream-horizon` | all 32 decoder layers | downstream +2..+16 |
| `factorial-probe-suffix-all-tokens` | frozen probe-transition suffix | aligned non-edited tokens |
| `factorial-probe-suffix-downstream-horizon` | frozen probe-transition suffix | downstream +2..+16 |
| `factorial-random-layers-downstream-horizon` | deterministic count-matched random layers | downstream +2..+16 |

All five are **state-free**: output KL has weight 1 and noisy-LM, answer, state,
and independent-clean losses have weight 0. A state loss, calibration value, or
state target in any arm is a protocol violation.

## Frozen model and optimizer recipe

- Model: `mistralai/Mistral-7B-v0.1`
- Exact revision: `7231864981174d9bee8c7687c24c8344414eae6b`
- Context: 8,192 student tokens per micro-step
- LoRA: rank 16, alpha 8, dropout 0, bias none
- LoRA modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`; embedding and LM head are deliberately excluded
- AdamW: learning rate `1e-4`, weight decay `0.01`, zero warmup, constant
  schedule (`constant-with-warmup` with a warmup ratio of zero)
- Micro-batch 1, gradient accumulation 8, optimizer steps 1,000
- Exact budget: `8192 * 1 * 8 * 1000 = 65,536,000` student tokens
- Seeds: 42, 43, and 44
- Common-coordinate initialization:
  `sha256-layer-keyed-kaiming-a-zero-b/v1`

The layer-keyed initializer makes the initial LoRA values at every shared
layer/module coordinate identical across arms for one seed. It does not make
non-shared coordinates exist.

## Data contract

Each seed starts from that seed's exact pinned FineWeb packing artifact produced
for the faithful Kojima run. The factorial preparation command copies the
closed-world packed-source artifact and creates its own encoding/noise wrapper;
it never routes through faithful Kojima noise or masking.

The wrapper freezes:

- 8,000 usable 8,192-token pairs, hence exactly 65,536,000 student tokens;
- exact alternating clean/noisy rows, starting with clean (4,000 each);
- three equal-probability typo operations: QWERTY-neighbor substitution,
  deletion, and duplication;
- the edit-count distribution: 0.50 one edit, 0.30 two edits, and 0.20 three or
  four edits;
- every concrete typo, rejected attempt, replacement decision, and source
  attempt index;
- the tokenizer snapshot attestation, packed-parent hashes, realized operation
  counts, and realized edit-count counts.

Preparation occurs in a sibling temporary directory. The complete artifact is
hash-validated and round-tripped before one atomic rename publishes it. Root or
in-tree symlinks, partial output, rehashed source substitutions, invalid skip
ledgers, and tokenizer-attestation substitutions fail closed.

For a fixed seed, every arm must receive the same prepared directory. Re-running
preparation against the same packed parent and frozen tokenizer produces
byte-identical files. Training consumes `pairs.jsonl` sequentially and refuses
to skip, repeat, or regenerate a pair.

## Version boundaries

The condition labels deliberately match the Gemma pilot because their two axes
have the same meanings. Their protocol identities do not:

| experiment | config schema | method identity | data runtime |
|---|---|---|---|
| Gemma 10M pilot | `robustness-adapter-training-config/v7` | legacy probe-factorial identity | generic 512-token stream |
| Mistral 64M production | `robustness-adapter-training-config/v8` | `mistral-state-free-probe-factorial/v1` | attested precomputed 8,000-pair stream |

Schema number alone must never select a runtime. Loaders dispatch the Mistral
path only when schema, exact method identity, and factorial condition all match.
Evaluation provenance must likewise bind the Mistral model, revision, and method
identity rather than interpreting a v8 adapter as the Gemma v7 pilot.

The faithful Kojima condition remains a separate v7 method. It includes
embedding/LM-head adapters, the public four-operation noise process, its public
masking/teacher semantics, and its own packed-stream runtime. The factorial is
not described as faithful Kojima; a direct comparison runs both methods.

## Preparation and materialization

First freeze and export the exact tokenizer manifest required by every
scientific entry point:

```bash
export TYPO_COT_TOKENIZER_ATTESTATION_MANIFEST=/abs/path/tokenizer-attestation.json
```

For each seed in `42 43 44`, prepare the pinned parent and then the factorial
wrapper:

```bash
uv run --project projects/typo-robust-training typo-cot \
  prepare-kojima-faithful-data \
  --seed 42 \
  --output-dir /abs/data/kojima-packed/seed-42

uv run --project projects/typo-robust-training typo-cot \
  prepare-mistral-factorial-data \
  --seed 42 \
  --packed-source-dir /abs/data/kojima-packed/seed-42 \
  --output-dir /abs/data/mistral-factorial/seed-42
```

Bind one Mistral Linear Probe transition artifact into all five configs:

```bash
uv run --project projects/typo-robust-training typo-cot \
  materialize-probe-output-factorial-configs \
  --template projects/typo-robust-training/configs/proposals/mistral7b-v01-probe-output-factorial-64m.template.yaml \
  --probe-selection /abs/evidence/mistral-probe-transition.json \
  --output-dir /abs/configs/mistral-factorial-64m
```

Run each condition with its matching command, config, seed-specific data
directory, frozen T0 protocol, and monitor data. For example:

```bash
uv run --project projects/typo-robust-training typo-cot \
  train-factorial-probe-suffix-downstream-horizon \
  --config /abs/configs/mistral-factorial-64m/factorial-probe-suffix-downstream-horizon.json \
  --training-data /abs/data/mistral-factorial/seed-42 \
  --probe-selection /abs/evidence/mistral-probe-transition.json \
  --seed 42 --gpu-id 0 \
  --evaluation-protocol /abs/evaluation/protocol.json \
  --monitor-data /abs/evaluation/tune-monitor \
  --output-dir /abs/runs/mistral-factorial/seed-42/probe-suffix-horizon
```

The runtime validates the full 64M accounting, frozen data and tokenizer
attestations, method evidence, LoRA target modules, objective, and layer policy
before allocating the production training stream.
