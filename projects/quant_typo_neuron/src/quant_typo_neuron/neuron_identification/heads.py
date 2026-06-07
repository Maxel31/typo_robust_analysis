"""M0: attention-head responsibility scoring — faithful port of Tsuji et al. typo_heads.py.

Head responsibility = attention ENTROPY ranked by Delta (main - baseline).

Shared imports (do NOT reimplement):
- typo_utils.neurons : get_rank
- quant_typo_neuron.data.wordnet_id : make_prompt, add_typo_to_data
- quant_typo_neuron.neuron_identification.scoring : top_fraction_mask (re-exported)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "AttentionInspector",
    "get_attn",
    "get_uni_distribution",
    "get_entropy",
    "get_entropies",
    "find_attn",
    "top_fraction_mask",
]


# ---------------------------------------------------------------------------
# AttentionInspector — faithful port of typo_heads.py AttentionInspector
# ---------------------------------------------------------------------------

class AttentionInspector:
    """Forward hook that accumulates attention weights from a self-attention layer.

    Faithful port of ``AttentionInspector`` in Tsuji et al. typo_heads.py.

    The hook captures ``output[1]`` (the attention weight tensor) produced when
    the model is called with ``output_attentions=True``.
    """

    def __init__(self, targetLayer) -> None:
        self.attention_weights: list = []
        self.featureHandle = targetLayer.register_forward_hook(self.feature)

    def feature(self, model, input, output) -> None:
        self.attention_weights.append(output[1].detach().cpu())

    def release(self) -> None:
        self.featureHandle.remove()


# ---------------------------------------------------------------------------
# get_attn — faithful port of typo_heads.py get_attn
# ---------------------------------------------------------------------------

def get_attn(model, input_ids):
    """Collect per-layer attention weight tensors via forward hooks.

    Faithful port of ``get_attn`` in Tsuji et al. typo_heads.py.

    Hooks ``self_attn`` (Gemma/Gemma2/Llama/Qwen2/Phi), ``self_attn``
    (GPTNeoX) for each layer and stacks the results.

    Note: some transformers versions with SDPA attention return ``None`` for
    ``output[1]`` in the hook. In that case, the implementation falls back
    to using ``model_output.attentions`` directly, which is equivalent.

    Parameters
    ----------
    model:
        A HuggingFace causal-LM supporting ``output_attentions=True``.
    input_ids:
        LongTensor of shape ``[1, seq_len]``.

    Returns
    -------
    torch.Tensor
        Shape ``[num_layers, num_heads, seq_len, seq_len]``.
        (Each layer inspector accumulates one ``[1, heads, seq, seq]`` tensor;
        they are concatenated along dim=0.)
    """
    import torch

    model.eval()
    with torch.no_grad():
        model_type = str(type(model))
        if (
            "GemmaForCausalLM" in model_type
            or "LlamaForCausalLM" in model_type
            or "Gemma2ForCausalLM" in model_type
            or "Qwen2ForCausalLM" in model_type
        ):
            AttnInspectors = [
                AttentionInspector(layer.self_attn) for layer in model.model.layers
            ]
        elif "GPTNeoXForCausalLM" in model_type:
            AttnInspectors = [
                AttentionInspector(layer.self_attn) for layer in model.gpt_neox.layers
            ]
        elif "PhiForCausalLM" in model_type:
            AttnInspectors = [
                AttentionInspector(layer.self_attn) for layer in model.model.layers
            ]
        else:
            # Fallback: try model.model.layers[i].self_attn
            try:
                AttnInspectors = [
                    AttentionInspector(layer.self_attn) for layer in model.model.layers
                ]
            except AttributeError:
                raise ValueError(
                    f"Model type '{model_type}' is not supported by get_attn. "
                    "Supported: Gemma, Gemma2, Llama, Qwen2, GPTNeoX, Phi."
                )

        input_ids = input_ids.to(model.device)
        model_output = model(input_ids, output_attentions=True, use_cache=False)

        for inspector in AttnInspectors:
            inspector.release()

        # Check if hooks captured actual attention weights.
        # Some transformers versions with SDPA return None in output[1] of the
        # self_attn hook, so fall back to model_output.attentions if needed.
        hook_has_weights = (
            len(AttnInspectors) > 0
            and len(AttnInspectors[0].attention_weights) > 0
            and AttnInspectors[0].attention_weights[0] is not None
        )

        if hook_has_weights:
            # Primary path: faithful to reference (hook-based)
            attn = torch.cat(
                [
                    torch.cat(inspector.attention_weights, dim=1)
                    for inspector in AttnInspectors
                ],
                dim=0,
            )
        else:
            # Fallback: use model output attentions directly
            # model_output.attentions is a tuple of [1, heads, seq, seq] tensors
            assert model_output.attentions is not None and len(model_output.attentions) > 0, (
                "output_attentions=True returned no attention weights. "
                "Try loading the model with attn_implementation='eager'."
            )
            attn = torch.cat(
                [a.detach().cpu() for a in model_output.attentions],
                dim=0,
            )

    return attn


# ---------------------------------------------------------------------------
# get_uni_distribution — faithful port of typo_heads.py get_uni_distribution
# ---------------------------------------------------------------------------

def get_uni_distribution(x):
    """Construct the causal uniform attention distribution for entropy normalisation.

    Faithful port of ``get_uni_distribution`` in Tsuji et al. typo_heads.py.

    For position i, the uniform distribution assigns weight 1/(i+1) to each of
    the i+1 preceding tokens (lower-triangular pattern).

    Parameters
    ----------
    x:
        Attention tensor of shape ``[..., seq_len, seq_len]``.

    Returns
    -------
    torch.Tensor
        Same shape as ``x``, lower-triangular, rows scaled by 1/(position+1).
    """
    import torch

    row_values = 1 / torch.arange(1, x.shape[-1] + 1, dtype=torch.float32)
    matrix = row_values.unsqueeze(1).expand_as(x)
    distribution = torch.tril(matrix)
    return distribution


# ---------------------------------------------------------------------------
# get_entropy — faithful port of typo_heads.py get_entropy
# ---------------------------------------------------------------------------

def get_entropy(attention_map):
    """Compute normalised attention entropy (1 - H_norm) for each head.

    Faithful port of ``get_entropy`` in Tsuji et al. typo_heads.py.

    entropy           = -sum(p * log2(p))          (per query position)
    uni_entropy       = -sum(u * log2(u))           (per query position, uniform causal)
    normed_entropy    = (entropy + eps) / (uni_entropy + eps)
    score             = 1 - normed_entropy     (high = peaked / focused)
    final             = mean over query positions

    Parameters
    ----------
    attention_map:
        Float tensor of shape ``[num_layers, num_heads, seq_len, seq_len]``.

    Returns
    -------
    torch.Tensor
        Shape ``[num_layers, num_heads]``.
    """
    import torch

    entropy = -torch.sum(attention_map * torch.log2(attention_map + 1e-9), dim=-1)

    uni_dist = get_uni_distribution(attention_map)
    uni_entropy = -torch.sum(uni_dist * torch.log2(uni_dist + 1e-9), dim=-1)

    normed_entropy = (entropy + 1e-9) / (uni_entropy + 1e-9)
    score = 1 - normed_entropy

    return score.mean(dim=-1)


# ---------------------------------------------------------------------------
# get_entropies — faithful port of typo_heads.py get_entropies
# ---------------------------------------------------------------------------

def get_entropies(data, model, tokenizer, typo_num: int = 1, use_data_typo: bool = False):
    """Compute mean per-head entropy score for original, typo, and split variants.

    Faithful port of ``get_entropies`` in Tsuji et al. typo_heads.py.

    For each item in ``data``:
      1. ``add_typo_to_data`` to build span bookkeeping.
      2. ``make_prompt`` to assemble original / typo / split input_ids.
      3. ``get_attn`` + ``get_entropy`` for each variant.
    Accumulate and divide by len(data).

    Parameters
    ----------
    data:
        Iterable of dataset entries.
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
    (original_entropy, typo_entropy, splited_entropy)
        Each is a float tensor of shape ``[num_layers, num_heads]``.
    """
    from tqdm import tqdm

    from quant_typo_neuron.data.wordnet_id import add_typo_to_data, make_prompt

    original_entropy = None
    typo_entropy = None
    splited_entropy = None

    data = list(data)  # materialise so len() is available

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

        o_attn = get_attn(model, original_inputs)
        o_ent = get_entropy(o_attn)
        if original_entropy is None:
            original_entropy = o_ent
        else:
            original_entropy += o_ent

        t_attn = get_attn(model, typo_inputs)
        t_ent = get_entropy(t_attn)
        if typo_entropy is None:
            typo_entropy = t_ent
        else:
            typo_entropy += t_ent

        s_attn = get_attn(model, splited_inputs)
        s_ent = get_entropy(s_attn)
        if splited_entropy is None:
            splited_entropy = s_ent
        else:
            splited_entropy += s_ent

    n = len(data)
    return (
        original_entropy / n,
        typo_entropy / n,
        splited_entropy / n,
    )


# ---------------------------------------------------------------------------
# find_attn — faithful port of typo_heads.py find_attn
# ---------------------------------------------------------------------------

def find_attn(
    data,
    model,
    tokenizer,
    vs_org: bool = False,
    typo_num: int = 1,
    use_data_typo: bool = False,
):
    """Rank attention heads by their differential entropy score (Delta = main - baseline).

    Faithful port of ``find_attn`` in Tsuji et al. typo_heads.py.

    Parameters
    ----------
    data:
        Iterable of dataset entries.
    model:
        Loaded HuggingFace causal-LM.
    tokenizer:
        Corresponding tokenizer.
    vs_org:
        If True, compare typo/split against *only* original.
    typo_num:
        Passed to ``get_entropies``.
    use_data_typo:
        Passed to ``get_entropies``.

    Returns
    -------
    (original_sorted_attn, typo_sorted_attn, splited_sorted_attn)
        Each is a list of dicts from ``get_rank``.
    """
    from typo_utils.neurons import get_rank

    original_attn, typo_attn, splited_attn = get_entropies(
        data, model, tokenizer, typo_num=typo_num, use_data_typo=use_data_typo
    )

    original_sorted_attn = get_rank(original_attn, [typo_attn, splited_attn])

    sub_attn = original_attn if vs_org else [original_attn, splited_attn]
    typo_sorted_attn = get_rank(typo_attn, sub_attn)

    sub_attn = original_attn if vs_org else [original_attn, typo_attn]
    splited_sorted_attn = get_rank(splited_attn, sub_attn)

    return original_sorted_attn, typo_sorted_attn, splited_sorted_attn


# ---------------------------------------------------------------------------
# top_fraction_mask re-export (convenient for head masks too)
# ---------------------------------------------------------------------------

# Re-export from scoring so callers can import from either module
from quant_typo_neuron.neuron_identification.scoring import top_fraction_mask  # noqa: E402, F401
