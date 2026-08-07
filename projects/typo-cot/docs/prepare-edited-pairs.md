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
| `clean`, `edited` | Exact prompts, editable text, generated continuation, deterministic extraction, correctness, and token counts. |
| `target_attempts` | Ranked AttnLRP/token attempts and edit-landing provenance. |
| `aligned_words` | Deduplicated actual edited words and clean/edited word-final token coordinates. |
| `excluded_attribution_tokens` | Attribution top four excluded before within-item `random-4` sampling. Empty for `attribution-4`. |
| `answer_changed` | Whether the two extracted answer strings differ. |

`target_attempts` records both the intended token span and the span reached by
the paper run's cumulative offset rule. `landed_on_intended_token` is therefore
an observed property, not an assertion. `intended_word_index` and
`landed_word_index` are whitespace-word ordinals in the editable item text.

`aligned_words` contains one row per distinct changed `landed_word_index`.
Multiple target ranks may map to the same row. `target_ranks` preserves those
links, while `clean_token_indices`, `edited_token_indices`, and their final
indices provide the coordinates later patching commands must use.

## Run manifest and recovery

`run.json` uses schema `prepare-edited-pairs-run/v1`. It records the canonical
paper fingerprint, full arguments, greedy/bfloat16/left-padding settings,
package and hardware provenance, model revision, a SHA-256 fingerprint of the
loaded benchmark records, progress counts, and per-item failures.

Records are checkpointed individually in a hidden work directory. A failed run
does not publish a partial `pairs.jsonl`; after the cause is fixed, rerun the
identical command with `--resume`. The writer validates all frozen arguments,
skips completed records, and atomically publishes `pairs.jsonl` only when every
selected item succeeds. Reusing a non-empty output directory without
`--resume` is rejected.

## Historical implementation differences

The final paper is the protocol authority. Four behaviors in the exploratory
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
  `i`. The public implementation excludes only contextual markers such as
  `(A)`, `A.`, `A)`, or `A:`, even when the tokenizer splits the marker.

These differences are also listed under `historical_compatibility_notes` in
the run provenance. Do not combine newly generated pairs with archived pairs
without an explicit compatibility analysis.
