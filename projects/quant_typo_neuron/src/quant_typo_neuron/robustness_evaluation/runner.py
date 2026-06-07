"""M2 evaluation driver: 4-condition robustness evaluation (clean/typo x fp16/Q).

Public API
----------
evaluate_cell(items, predict, typo_type, eps, seed, *, model, method, bit, dataset)
    -> list[ItemResult]
    Evaluate one cell (one typo_type x eps x seed combination) over all items.
    Returns one ItemResult per TaskItem with item-level 0/1 correctness.

run_evaluation(items, predict, *, model, method, bit, dataset,
               typo_types, eps_levels, seeds) -> list[ItemResult]
    Loop over typo_type x eps x seed combinations, accumulate ItemResults.

make_hf_predict(model, tokenizer, device) -> PredictFn
    Build a predict_fn that runs greedy generation on GPU.

PredictFn type
--------------
A ``PredictFn`` is a callable with signature::

    (prompt: str) -> str | tuple[str, float]

When it returns a plain ``str``, confidence defaults to ``1.0``.
When it returns ``(text, conf)``, ``conf`` is stored on ItemResult.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence, Union

from quant_typo_neuron.contracts import ItemResult
from quant_typo_neuron.data.tasks import TaskItem, apply_typo_to_prompts

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

PredictFn = Callable[[str], Union[str, tuple[str, float]]]

__all__ = ["evaluate_cell", "run_evaluation", "make_hf_predict", "PredictFn"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_predict(predict: PredictFn, prompt: str) -> tuple[str, float]:
    """Call predict_fn and normalise to (text, confidence).

    If predict returns a plain ``str``, confidence is set to ``1.0``.
    If predict returns ``(str, float)``, both are forwarded unchanged.
    """
    result = predict(prompt)
    if isinstance(result, tuple):
        text, conf = result
        return str(text), float(conf)
    return str(result), 1.0


def _is_correct(prediction: str, answer: str) -> int:
    """Return 1 if prediction matches answer (strip + case-insensitive), else 0."""
    return int(prediction.strip().lower() == answer.strip().lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_cell(
    items: Sequence[TaskItem],
    predict: PredictFn,
    typo_type: str,
    eps: int | float,
    seed: int,
    *,
    model: str,
    method: str,
    bit: int | None,
    dataset: str,
) -> list[ItemResult]:
    """Evaluate one cell (one typo_type x eps x seed combination) over all items.

    For each :class:`~quant_typo_neuron.data.tasks.TaskItem`:
    1. Get clean prediction from the original prompt.
    2. Perturb the prompt with :func:`apply_typo_to_prompts` and get typo prediction.
    3. Compute ``correct_clean`` and ``correct_typo`` as 0/1 vs ``item.answer``.
    4. Store ``conf`` from the predict_fn (defaults to 1.0 for plain-text predictors).

    Parameters
    ----------
    items:
        Source task items (unperturbed).
    predict:
        Callable ``(prompt: str) -> str | (str, float)``.
    typo_type:
        One of the real-typo types (``"sub_keyboard"``, ``"insert"``,
        ``"delete"``, ``"transpose"``).
    eps:
        Perturbation intensity -- integer 1 (one char) or float ratio.
    seed:
        Random seed for the typo generator (reproducibility).
    model:
        Model identifier string (stored in ItemResult).
    method:
        Quantization method (``"fp16"`` | ``"gptq"`` | ...).
    bit:
        Bit width (``None`` for fp16).
    dataset:
        Dataset name (stored in ItemResult).

    Returns
    -------
    list[ItemResult]
        One record per item; ``correct_clean`` and ``correct_typo`` are raw 0/1.
    """
    # Apply typo to all prompts in one pass (deterministic with seed)
    typo_items = apply_typo_to_prompts(items, typo_type=typo_type, eps=eps, seed=seed)

    results: list[ItemResult] = []
    for item, typo_item in zip(items, typo_items):
        # Clean prediction
        clean_text, conf = _call_predict(predict, item.prompt)
        correct_clean = _is_correct(clean_text, item.answer)

        # Typo'd prediction
        typo_text, _ = _call_predict(predict, typo_item.prompt)
        correct_typo = _is_correct(typo_text, item.answer)

        results.append(
            ItemResult(
                model=model,
                method=method,
                bit=bit,
                typo_type=typo_type,
                eps=eps,
                dataset=dataset,
                seed=seed,
                item_id=item.item_id,
                correct_clean=correct_clean,
                correct_typo=correct_typo,
                conf=conf,
            )
        )

    return results


def run_evaluation(
    items: Sequence[TaskItem],
    predict: PredictFn,
    *,
    model: str,
    method: str,
    bit: int | None,
    dataset: str,
    typo_types: Sequence[str],
    eps_levels: Sequence[int | float],
    seeds: Sequence[int],
) -> list[ItemResult]:
    """Loop over typo_type x eps x seed combinations and accumulate ItemResults.

    For each combination in ``typo_types x eps_levels x seeds``:
    - Call :func:`evaluate_cell` over all items.
    - Append results to the accumulator.

    Parameters
    ----------
    items:
        Source task items (unperturbed).
    predict:
        Callable ``(prompt: str) -> str | (str, float)``.
    model:
        Model identifier string (stored in each ItemResult).
    method:
        Quantization method.
    bit:
        Bit width (``None`` for fp16).
    dataset:
        Dataset name.
    typo_types:
        Sequence of typo type strings to iterate over.
    eps_levels:
        Sequence of eps values to iterate over.
    seeds:
        Sequence of random seeds; per README ``seeds x 5`` is standard.

    Returns
    -------
    list[ItemResult]
        Accumulated records:
        ``len(typo_types) x len(eps_levels) x len(seeds) x len(items)`` total.
    """
    all_results: list[ItemResult] = []

    for typo_type in typo_types:
        for eps in eps_levels:
            for seed in seeds:
                cell_results = evaluate_cell(
                    items, predict, typo_type, eps, seed,
                    model=model, method=method, bit=bit, dataset=dataset,
                )
                all_results.extend(cell_results)

    return all_results


def make_hf_predict(
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int = 32,
) -> PredictFn:
    """Build a predict_fn backed by greedy HuggingFace generation.

    The returned callable takes a prompt string, tokenises it, runs greedy
    generation on ``device``, decodes the **new** tokens only, and returns the
    decoded string.  Confidence is always ``1.0`` (greedy => no soft scores).

    Parameters
    ----------
    model:
        A HuggingFace ``PreTrainedModel`` already moved to ``device``.
    tokenizer:
        A HuggingFace tokenizer.
    device:
        Target device string (e.g. ``"cuda:0"``).
    max_new_tokens:
        Maximum number of tokens to generate (default 32).

    Returns
    -------
    PredictFn
        Callable ``(prompt: str) -> str``.
    """
    import torch

    def _predict(prompt: str) -> str:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )

        # Decode only newly generated tokens
        new_ids = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return _predict
