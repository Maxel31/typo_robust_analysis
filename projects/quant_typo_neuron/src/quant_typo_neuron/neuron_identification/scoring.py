"""M0: neuron responsibility scoring — faithful port of Tsuji et al. typo_neurons.py.

Shared imports (do NOT reimplement):
- typo_utils.neurons : get_acts, get_rank, NeuronMask
- quant_typo_neuron.data.wordnet_id : make_prompt, add_typo_to_data
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "get_averaged_act",
    "find_neurons",
    "top_fraction_mask",
    "save_mask",
    "load_mask",
]


# ---------------------------------------------------------------------------
# get_averaged_act — faithful port of typo_neurons.py get_averaged_act
# ---------------------------------------------------------------------------

def get_averaged_act(data, model, tokenizer, typo_num: int = 1, use_data_typo: bool = False):
    """Compute mean per-neuron activation for original, typo, and split variants.

    Faithful port of ``get_averaged_act`` in Tsuji et al. typo_neurons.py.

    For each item in ``data``:
      1. ``add_typo_to_data`` to build span bookkeeping.
      2. ``make_prompt`` to assemble original / typo / split input_ids.
      3. ``get_acts`` to collect FFN activations.
      4. Sum activations over the variant spans + the answer span.
    Divide accumulated sums by their respective point counts.

    Parameters
    ----------
    data:
        Iterable of dataset entries (dicts with ``importance_rank``,
        ``meaning_ids``, ``word``, etc.).
    model:
        Loaded HuggingFace causal-LM (on device).
    tokenizer:
        Corresponding tokenizer.
    typo_num:
        Number of tokens to corrupt per item (passed to ``add_typo_to_data``).
    use_data_typo:
        If True, use pre-computed typo stored in each data entry.

    Returns
    -------
    (original_acts, typo_acts, splited_acts)
        Each is a float32 tensor of shape ``[num_layers, d_ff]``.
    """
    import torch
    from tqdm import tqdm

    from quant_typo_neuron.data.wordnet_id import add_typo_to_data, make_prompt
    from typo_utils.neurons import get_acts

    original_acts = None
    typo_acts = None
    splited_acts = None

    original_total_points = 0
    variant_total_points = 0

    for line in tqdm(data):
        line = add_typo_to_data(
            line,
            tokenizer,
            typo_type="insert_char",
            typo_num=typo_num,
            use_data_typo=use_data_typo,
        )
        original_inputs = make_prompt(line["original_ids"], tokenizer, word=line["word"])
        typo_inputs = make_prompt(line["typo_ids"], tokenizer, word=line["word"])
        splited_inputs = make_prompt(line["splited_ids"], tokenizer, word=line["word"])

        # --- original acts ---
        oa = []
        acts = get_acts(model, original_inputs)
        # cast to float32 for bf16/fp16 safety (before any numpy or accumulation)
        acts = acts.float()
        for s, ori_e in zip(line["start_index"], line["original_end_index"]):
            original_total_points += ori_e - s
            oa.append(acts[s:ori_e])
        original_total_points -= line["word_start_index"]  # subtract (negative index)
        oa.append(acts[line["word_start_index"]:])
        oa = torch.concatenate(oa, dim=0).sum(dim=0)
        if original_acts is None:
            original_acts = oa
        else:
            original_acts += oa

        # --- typo acts ---
        ta = []
        acts = get_acts(model, typo_inputs)
        acts = acts.float()
        for s, var_e in zip(line["start_index"], line["variant_end_index"]):
            variant_total_points += var_e - s
            ta.append(acts[s:var_e])
        variant_total_points -= line["word_start_index"]  # subtract (negative index)
        ta.append(acts[line["word_start_index"]:])
        ta = torch.concatenate(ta, dim=0).sum(dim=0)
        if typo_acts is None:
            typo_acts = ta
        else:
            typo_acts += ta

        # --- split acts (same span bookkeeping as typo) ---
        sa = []
        acts = get_acts(model, splited_inputs)
        acts = acts.float()
        for s, var_e in zip(line["start_index"], line["variant_end_index"]):
            sa.append(acts[s:var_e])
        sa.append(acts[line["word_start_index"]:])
        sa = torch.concatenate(sa, dim=0).sum(dim=0)
        if splited_acts is None:
            splited_acts = sa
        else:
            splited_acts += sa

    original_acts = original_acts / original_total_points
    typo_acts = typo_acts / variant_total_points
    splited_acts = splited_acts / variant_total_points

    return original_acts, typo_acts, splited_acts


# ---------------------------------------------------------------------------
# find_neurons — faithful port of typo_neurons.py find_neurons
# ---------------------------------------------------------------------------

def find_neurons(
    data,
    model,
    tokenizer,
    vs_org: bool = False,
    typo_num: int = 1,
    use_data_typo: bool = False,
):
    """Rank neurons by their differential activation (Δ = main − baseline).

    Faithful port of ``find_neurons`` in Tsuji et al. typo_neurons.py.

    Parameters
    ----------
    data:
        Iterable of dataset entries.
    model:
        Loaded HuggingFace causal-LM.
    tokenizer:
        Corresponding tokenizer.
    vs_org:
        If True, compare typo/split against *only* original (single tensor
        baseline instead of max over two tensors).
    typo_num:
        Passed to ``get_averaged_act``.
    use_data_typo:
        Passed to ``get_averaged_act``.

    Returns
    -------
    (original_sorted_neurons, typo_sorted_neurons, splited_sorted_neurons)
        Each is a list of dicts from ``get_rank``.
    """
    from typo_utils.neurons import get_rank

    original_acts, typo_acts, splited_acts = get_averaged_act(
        data, model, tokenizer, typo_num=typo_num, use_data_typo=use_data_typo
    )

    original_sorted_neurons = get_rank(original_acts, [typo_acts, splited_acts])

    sub_acts = original_acts if vs_org else [original_acts, splited_acts]
    typo_sorted_neurons = get_rank(typo_acts, sub_acts)

    sub_acts = original_acts if vs_org else [original_acts, typo_acts]
    splited_sorted_neurons = get_rank(splited_acts, sub_acts)

    return original_sorted_neurons, typo_sorted_neurons, splited_sorted_neurons


# ---------------------------------------------------------------------------
# top_fraction_mask — select top-frac neurons from a ranked list
# ---------------------------------------------------------------------------

def top_fraction_mask(sorted_neurons: list, frac: float):
    """Return a NeuronMask for the top fraction of the ranked neuron list.

    Parameters
    ----------
    sorted_neurons:
        Output of ``find_neurons`` (list of dicts with ``"position"`` key).
    frac:
        Fraction of the full list to include (e.g. 0.005 = top 0.5%).

    Returns
    -------
    NeuronMask  (dict[int, list[int]])
        ``{layer_idx: [neuron_dim, ...]}``.
    """
    k = max(1, int(len(sorted_neurons) * frac))
    top = sorted_neurons[:k]
    mask: dict[int, list[int]] = {}
    for entry in top:
        layer, dim = entry["position"]
        if layer not in mask:
            mask[layer] = []
        mask[layer].append(dim)
    return mask


# ---------------------------------------------------------------------------
# save_mask / load_mask — JSON, int-key safe
# ---------------------------------------------------------------------------

def save_mask(mask, path) -> None:
    """Save a NeuronMask to JSON (layer keys stored as strings for JSON compat).

    Parameters
    ----------
    mask:
        ``dict[int, list[int]]`` neuron mask.
    path:
        Output file path (str or Path).
    """
    serializable = {str(k): v for k, v in mask.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_mask(path) -> dict:
    """Load a NeuronMask from JSON, restoring int keys.

    Parameters
    ----------
    path:
        Path to a JSON file written by ``save_mask``.

    Returns
    -------
    dict[int, list[int]]
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
