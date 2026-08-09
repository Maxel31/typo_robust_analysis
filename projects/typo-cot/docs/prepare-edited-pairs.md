# Clean/edited pair record contract

`prepare-edited-pairs` writes newline-delimited
`prepare-edited-pairs/v1` records. This schema is the handoff between input
generation and every later text-swap or activation-patching operation.

## Coordinate conventions

- Character spans are zero-based, half-open `[start, end)` offsets in the
  stored string named by the field.
- Token indices are zero-based indices from the model tokenizer with special
  tokens enabled. Width-zero special-token offsets never align to a word.
- `clean_final_token` and `edited_final_token` are the final overlapping tokens
  of the complete whitespace-delimited word that actually changed. They are
  not merely the ranked subword token and are not found by substring search.
- Arrays retain target-selection order. Pair records themselves are sorted by
  `sample_id` before `pairs.jsonl` is finalized.

## Pair record

Each line contains these top-level groups:

| Field | Meaning |
|---|---|
| `schema_version` | Always `prepare-edited-pairs/v1`. |
| `sample_id`, `model`, `benchmark`, `subset` | Stable item and setting identity. |
| `targeting`, `seed`, `num_edits_requested` | Frozen edit condition. |
| `attribution_target` | First-CoT-token identity and the selected maximum-logit target position. |
| `clean`, `edited` | Exact prompts, editable text, generated continuation, primary/fallback extraction provenance, exact canonical correctness, and token counts. |
| `target_attempts` | Ranked AttnLRP/token attempts and edit-landing provenance. |
| `aligned_words` | Deduplicated actual edited words and clean/edited word-final token coordinates. |
| `excluded_attribution_tokens` | Attribution top four excluded before within-item `random-4` sampling. Empty for `attribution-4`. |
| `answer_changed` | Whether the two extracted answer strings differ. |

`target_attempts` records both the intended token span and the span reached by
the paper run's cumulative offset rule. `landed_on_intended_token` is therefore
an observed property, not an assertion. `intended_word_index` and
`landed_word_index` are whitespace-word ordinals in the editable item text.

For AttnLRP, the target position is the first generated CoT token's index. In a
causal LM, logits at that index are the distribution after observing the first
CoT token (predicting the following token). This implements the final paper's
“maximum logit immediately after the first CoT token”; it is intentionally not
the prompt-final logits that predict the first CoT token itself.

`aligned_words` contains one row per distinct changed `landed_word_index`.
Multiple target ranks may map to the same row. `target_ranks` preserves those
links, while `clean_token_indices`, `edited_token_indices`, and their final
indices provide the coordinates later patching commands must use.

Every selected dataset item produces a record, even when no character edit can
be applied or the attempted edits leave no final text difference. The existing
arrays and counts represent these cases without a schema exception:

- no applicable edit: `num_target_attempts: 0`, `target_attempts: []`,
  `num_aligned_words: 0`, and `aligned_words: []`;
- attempted but no final changed word: non-empty `target_attempts` with
  `num_aligned_words: 0` and `aligned_words: []`.

When the final edited prompt equals the clean prompt, the deterministic clean
generation and answer are reused and `answer_changed` is false. These records
remain in the paper's item-level targeting-fidelity denominator. Downstream
activation-patching commands must instead require at least one aligned word and
record that exclusion explicitly.

## Paper cohort sizing

The dataset loader derives its per-subset cap from both benchmark and model so
the completed 42-setting grid has the final paper's denominators:

| Benchmark/model setting | Per-subset cap | Completed items |
|---|---:|---:|
| MMLU with Qwen2.5-7B-Instruct or Gemma-3-12B/27B-IT | 100 | 5,700 |
| MMLU with every other paper model | 50 | 2,850 |
| MMLU-Pro with every paper model | 100 | 1,400 |

Other benchmark loaders retain their complete paper cohorts. Model matching is
case-insensitive and uses the final component of a Hugging Face model ID, so
`Qwen/Qwen2.5-7B-Instruct` and `Qwen2.5-7B-Instruct` select the same rule. The
manifest records the rule version and chosen per-subset cap; resume validation
therefore rejects a checkpoint created under a different cohort rule. For
benchmarks whose loaders do not apply a per-subset cap, the provenance value is
`null` rather than a misleading default of 50.

## Run manifest and recovery

`run.json` uses schema `prepare-edited-pairs-run/v1`. It records the canonical
paper fingerprint, full arguments, greedy/bfloat16/left-padding settings,
package and hardware provenance, model revision, a SHA-256 fingerprint of the
loaded benchmark records, progress counts, and per-item failures. A completed
manifest also declares exactly one output, binding `pairs.jsonl` by path,
record count, and SHA-256 computed after its final atomic publication.

Greedy generation is fully explicit: `do_sample=false`, `num_beams=1`,
`num_return_sequences=1`, `temperature/top_p/top_k=null`, `use_cache=true`, and
no score-return mode. The manifest records these values and
`generation_protocol: explicit-greedy-generation/v1`, so downstream patching and
resume validation reject runs whose model defaults could have changed the
answers. The task extractor is attempted first; only an empty result invokes the
final-paper fallback for its registered benchmark. GSM8K numeric correctness is
exact after canonicalization and never passes through binary floating-point
comparison; MATH-500 retains its native symbolic normalizer.

Records are checkpointed individually in a hidden work directory. A failed run
does not publish a partial `pairs.jsonl`; after the cause is fixed, rerun the
identical command with `--resume`. The writer validates all frozen arguments,
model/dataset/environment provenance, and protocol identifiers before it skips
completed records. An incomplete resume checks the current environment and
protocol before model loading, then model revision and dataset fingerprints
after discovery but before any checkpoint is reused. A completed run performs
no new computation, so its fast resume checks only static protocol/cohort
identity and permits a different GPU or package environment. The manifest remains `running`
while pending items are processed and becomes `failed` only after the run ends.
It atomically publishes `pairs.jsonl` only when every selected item succeeds.
If that published file is later removed or its bytes differ from the completed
manifest, resuming reports an error instead of silently accepting or
regenerating it. Completed manifests created before this output identity was
introduced must be regenerated. Reusing a non-empty output directory without
`--resume` is rejected. Conversely, `--resume` always requires an existing
`run.json`; a missing or empty output directory is not silently treated as a
new run.

## Historical implementation differences

The final paper is the protocol authority. Five behaviors in the exploratory
code were not portable or did not implement that protocol exactly, so fresh
outputs intentionally differ from some archived machine-local artifacts:

- Per-item seeds previously used Python's process-randomized `hash()`. This
  command uses the versioned SHA-256 derivation recorded as
  `random_seed_algorithm` in `run.json`.
- Mistral previously routed LXT rules to Llama classes, leaving Mistral layers
  unpatched. The public implementation targets Mistral's own MLP, RMSNorm, and
  attention definitions so the requested AttnLRP method is actually applied.
- Historical patching scripts sometimes found a ranked token substring. The
  final paper specifies the last token of the complete word that actually
  changed; `aligned_words` stores those exact clean and edited coordinates.
- The exploratory option-label filter discarded every bare one-letter token
  from A through J, including ordinary words such as `a` and variables such as
  `i`, and applied the punctuated-letter pattern throughout the question. The
  public implementation excludes the parenthesized markers actually emitted by
  the paper prompts, such as `(A)`, only inside the choice-list region using the
  prompt's recorded question boundary. Question-body labels such as `A.` and
  `b:` remain eligible, while split `(A)` tokens are excluded as one marker.
- ARC can encode its answer key as a numeric source label while the public
  prompt relabels choices as `(A)`–`(D)`. The loader now maps through the
  source choice order so stored gold answers and extracted letters use the
  same coordinate system.

These differences are also listed under `historical_compatibility_notes` in
the run provenance. Do not combine newly generated pairs with archived pairs
without an explicit compatibility analysis.
