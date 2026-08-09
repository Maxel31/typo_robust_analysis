# Final answer-line deletion control

`answer-line-deletion` is the public implementation of the final paper's RQ2
control (§3.4, §4.2, and Table 1). The PDF fingerprint reported by `typo-cot
experiments source` is authoritative. Submitted code and archived outputs are
used only for details the PDF does not define, and those details are labelled
below.

## Command

Run one of the three control models on one of the two control tasks using
physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot answer-line-deletion \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cot-swap-run results/cot-swap/gemma-3-4b-it/gsm8k/random-4 \
  --max-pairs 150 \
  --gpu-id 0 \
  --output-dir results/answer-line-deletion/gemma-3-4b-it/gsm8k
```

The other paper control models are
`meta-llama/Llama-3.2-3B-Instruct` and
`mistralai/Mistral-7B-Instruct-v0.3`; the other task is `mmlu`. Use `--limit 1`
only for an explicitly partial smoke run. Continue an interrupted identical run
with `--resume`.

## Source and fixed cohort

`--cot-swap-run` must be a completed, unlimited `cot-swap-run/v1` output for the
same model and benchmark, using Random-4, seed 42, four requested edits, the
canonical paper fingerprint, and the reviewed 16-token four-cell protocol. Its
three public outputs and every recorded semantic relationship are revalidated.
The referenced prepared-pair file and manifest are then loaded by their recorded
paths and SHA-256 values so the exact edited prompt and clean continuation can
be reconstructed. The validated source tree must remain at its recorded paths;
this operation intentionally offers no path override that could weaken the
upstream identity contract.

The deterministic source cohort contains records for which regenerated A is
correct and B differs from A under the public task-canonical equality. Records
are ordered by unique sample ID and the
first `--max-pairs` are retained. The submitted control used Random-4 and a cap
of 150 per model/task; neither detail is printed in the PDF, so both are
legacy-backed and recorded as such. A complete paper-protocol setting therefore
requires `--max-pairs 150`; another positive value is a labelled sensitivity
run. `--limit` is applied only after the complete capped plan is fingerprinted.
The archived GSM8K producer instead used raw stripped-string inequality for the
B-change gate; ten of its 333 rows were only formatting differences such as
`5` versus `5.00`. This is another reason a fresh canonical cohort need not have
the historical size.

## Paired text intervention

The source pair's clean continuation is cut using the same reviewed CoT-swap
answer-template boundary. Let `Cc` be that clean pre-answer text and `Qc`/`Qp`
the clean/edited prompts. Each selected pair creates two conditions:

| Arm | Fixed input | Readout |
|---|---|---|
| `complete` | `Qp + Cc` | source-control baseline |
| `answer-line-deleted` | `Qp + strip_last_nonempty_line(Cc)` | deletion control |

The submitted producer calls this `last_line`: trailing blank lines are ignored,
the final non-empty line is removed in full, and the retained prefix ends in one
newline when any earlier non-whitespace text remains. A single-line prefix
therefore becomes empty. The archived Table 1 cohort became empty in 179/333
GSM8K and 334/450 MMLU cases, so `prefix_became_empty` and separate empty/non-empty
strata are mandatory outputs. This exact character rule is legacy-backed; the PDF
specifies deletion of condition C's final answer line but does not publish an
algorithm for locating it.

Both arms are tokenized from their complete text in one two-row left-padded
batch. The edited prompt string remains exact, and token-prefix stability across
the concatenation boundary is recorded as a diagnostic. Only generated token IDs
are decoded. Generation is greedy with bfloat16
weights, explicit EOS IDs, no sampling, and at most 16 answer tokens. The
task-specific primary extractor runs first; only an empty primary invokes the
same deterministic, cap-aware fallback used symmetrically by `cot-swap`.

Each arm is restored when its non-empty canonical answer equals the source
record's regenerated clean A answer. Because cohort membership already requires
A correctness, this is also a correct-answer outcome, while preserving the
paper's A-relative CoT-swap estimand. Unextractable output remains in the
denominator as a failed restoration.

## Published reference and protocol conflict

Table 1 prints the following three-model pooled controls:

| Task | n | Complete | Answer line deleted |
|---|---:|---:|---:|
| GSM8K | 333 | 95.2% | 48.9% |
| MMLU | 450 | 82.2% | 29.1% |

The preserved producer artifacts identify these exact counts as
317/333→163/333 and 370/450→131/450. They also record generation with up to 256
new tokens. That conflicts with Appendix A's statement that CoT-swap answer
spans use at most 16 tokens. The public operation follows the final-PDF protocol
and stores the printed values only as historical reference metadata. Exact
historical rows additionally depend on frozen exploratory records and extraction
behavior, so numerical equality is never treated as proof of source identity.
The archived 16-token run retained the same historical IDs but yielded
315/333→14/333 on GSM8K and 369/450→54/450 on MMLU; its deleted arm also had
262/333 and 232/450 unextractable answers, respectively. Those artifacts do not
store raw generations or stop reasons, so they show a strong token-cap dependence
without proving which individual rows hit the cap.

This command publishes one model/task setting. It does not silently combine six
runs into the historical three-model micro-pool; the later paper-artifact build
must sum the integer numerators and denominators from the three model summaries
for each task and must retain the protocol/comparability labels.

The PDF itself warns that line deletion truncates the supplied text mid-flow.
The observed drop therefore combines removal of answer-near content with format
disruption and, for single-line prefixes, removal of all supplied clean text. It
cannot isolate reasoning content or be interpreted as a direct,
indirect, total, interaction, or mediation effect.

## Outputs and restart boundary

One setting writes:

- `answer_line_deletion_records.jsonl`: one successful paired generation per
  selected case, including exact text/token fingerprints, stop/extraction
  provenance, and both restoration events;
- `pair_status_records.jsonl`: every source CoT-swap record with cohort,
  cap/limit selection, execution, and exclusion status;
- `answer_line_deletion_summary.json`: integer counts and rates for both arms,
  the deleted-minus-complete rate difference, paired transitions, empty/non-empty
  prefix strata (including extraction and stop diagnostics), and published
  references;
- `run.json`: arguments, paper/upstream/prepared-pair/plan/runtime fingerprints,
  pair-atomic checkpoints, failures, outputs, and comparability status.

Each pair checkpoint binds the exact upstream record, prepared pair, text plan,
and runtime. Failed runs publish no partial public tables. A valid orphan left
after an atomic checkpoint rename is adopted on `--resume`; completed output
hashes and reconstructed semantics are checked without loading model weights.
