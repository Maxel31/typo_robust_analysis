"""FFN activation hooks, intervention classes, and neuron-ranking utilities.

Faithful port of the neuron-analysis utilities from Tsuji et al.'s
reference implementation:
- ``get_acts``             from typo_neurons.py  (OutputInspector / get_acts)
- ``Deactivator``          from utils.py
- ``Activator``            from utils.py
- ``HeadDeactivator``      from utils.py
- ``convertNeuronsToDict`` from utils.py
- ``get_rank``             from utils.py

The module is importable on CPU-only machines; torch is imported lazily
inside functions and class methods.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NeuronIndex = tuple[int, int]      # (layer, neuron_dim)
NeuronMask = dict[int, list[int]]  # layer -> list of intermediate dims


# ---------------------------------------------------------------------------
# OutputInspector  (inner helper for get_acts — from typo_neurons.py)
# ---------------------------------------------------------------------------

class OutputInspector:
    """Forward hook that accumulates outputs from a target layer.

    Faithful port of ``OutputInspector`` in Tsuji et al. typo_neurons.py.
    """

    def __init__(self, targetLayer) -> None:
        self.layerOutputs: list = []
        self.featureHandle = targetLayer.register_forward_hook(self.feature)

    def feature(self, model, input, output) -> None:
        self.layerOutputs.append(output.detach().cpu())

    def release(self) -> None:
        self.featureHandle.remove()


# ---------------------------------------------------------------------------
# get_acts  (from typo_neurons.py)
# ---------------------------------------------------------------------------

def get_acts(model, input_ids):
    """Collect FFN activation-function outputs for every decoder layer.

    Hooks ``mlp.act_fn`` (Gemma/Gemma2/Llama/Qwen2), ``mlp.act``
    (GPTNeoX), or ``mlp.activation_fn`` (Phi) for each layer and stacks
    the results into a single tensor.

    Faithful port of ``get_acts`` in Tsuji et al. typo_neurons.py.

    Parameters
    ----------
    model:
        A HuggingFace causal-LM.
    input_ids:
        LongTensor of shape ``[1, seq_len]``.

    Returns
    -------
    torch.Tensor
        Shape ``[seq_len, num_layers, d_ff]`` (after transpose from
        ``[num_layers, seq_len, d_ff]``).
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
            actInspectors = [
                OutputInspector(layer.mlp.act_fn)
                for layer in model.model.layers
            ]
        elif "GPTNeoXForCausalLM" in model_type:
            actInspectors = [
                OutputInspector(layer.mlp.act)
                for layer in model.gpt_neox.layers
            ]
        elif "PhiForCausalLM" in model_type:
            actInspectors = [
                OutputInspector(layer.mlp.activation_fn)
                for layer in model.model.layers
            ]
        else:
            # Fallback: try model.model.layers[i].mlp.act_fn
            # (supports synthetic/test models that expose this attribute)
            try:
                actInspectors = [
                    OutputInspector(layer.mlp.act_fn)
                    for layer in model.model.layers
                ]
            except AttributeError:
                raise ValueError(
                    f"Model type '{model_type}' is not supported by get_acts. "
                    "Supported: Gemma, Gemma2, Llama, Qwen2, GPTNeoX, Phi."
                )

        input_ids = input_ids.to(model.device)
        _ = model(input_ids, use_cache=False)

        for actInspector in actInspectors:
            actInspector.release()

        # Each inspector.layerOutputs: list of [1, seq, d_ff] tensors
        # cat along dim=1 in case multiple calls; then cat layers along dim=0
        # -> [n_layers, seq, d_ff]; transpose -> [seq, n_layers, d_ff]
        acts = torch.cat(
            [
                torch.cat(actInspector.layerOutputs, dim=1)
                for actInspector in actInspectors
            ],
            dim=0,
        ).transpose(0, 1)

    return acts


# ---------------------------------------------------------------------------
# Deactivator  (from utils.py)
# ---------------------------------------------------------------------------

class Deactivator:
    """Forward hook that zeroes specified neuron dimensions in a layer's output.

    Faithful port of ``Deactivator`` in Tsuji et al. utils.py.

    Parameters
    ----------
    targetLayer:
        The module to hook (e.g. ``layer.mlp.act_fn``).
    neuronIds:
        Indices of neurons to zero out.
    mode:
        ``"last"`` — zero only the last sequence position.
        ``"all"``  — zero all sequence positions.
        ``"lastN"`` — zero the last ``lastN`` positions.
    lastN:
        Used when ``mode="lastN"``.
    """

    def __init__(self, targetLayer, neuronIds, mode: str, lastN: int = 0) -> None:
        self.neuronIds = neuronIds
        assert mode in ["last", "all", "lastN"], "mode should be last or all"
        self.mode = mode
        self.lastN = lastN
        self.outputHandle = targetLayer.register_forward_hook(self.deactivate)

    def deactivate(self, model, input, output):
        if self.mode == "last":
            output[0, -1, self.neuronIds] *= 0
        elif self.mode == "all":
            output[0, :, self.neuronIds] *= 0
        elif self.mode == "lastN":
            output[0, -self.lastN :, self.neuronIds] *= 0
        else:
            print(f"{self.mode=} cannot be recognized")
        return output

    def release(self) -> None:
        self.outputHandle.remove()


# ---------------------------------------------------------------------------
# Activator  (from utils.py)
# ---------------------------------------------------------------------------

class Activator:
    """Forward hook that adds 1 to specified neuron dimensions in a layer's output.

    Faithful port of ``Activator`` in Tsuji et al. utils.py.

    Parameters
    ----------
    targetLayer:
        The module to hook (e.g. ``layer.mlp.act_fn``).
    neuronIds:
        Indices of neurons to boost.
    mode:
        ``"last"`` / ``"all"`` / ``"lastN"``.
    lastN:
        Used when ``mode="lastN"``.
    """

    def __init__(self, targetLayer, neuronIds, mode: str, lastN: int = 0) -> None:
        self.neuronIds = neuronIds
        assert mode in ["last", "all", "lastN"], "mode should be last or all"
        self.mode = mode
        self.lastN = lastN
        self.outputHandle = targetLayer.register_forward_hook(self.activate)

    def activate(self, model, input, output):
        if self.mode == "last":
            output[0, -1, self.neuronIds] += 1
        elif self.mode == "all":
            output[0, :, self.neuronIds] += 1
        elif self.mode == "lastN":
            output[0, -self.lastN :, self.neuronIds] += 1
        else:
            print(f"{self.mode=} cannot be recognized")
        return output

    def release(self) -> None:
        self.outputHandle.remove()


# ---------------------------------------------------------------------------
# HeadDeactivator  (from utils.py)
# ---------------------------------------------------------------------------

def _repeat_kv(hidden_states, n_rep: int):
    """Expand key/value heads for grouped-query attention."""
    import torch
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HeadDeactivator:
    """Forward hook that zeroes specified attention heads in a self-attention layer.

    Faithful port of ``HeadDeactivator`` in Tsuji et al. utils.py.

    Parameters
    ----------
    targetLayer:
        The self-attention module to hook.
    head_ids:
        Indices of attention heads to zero out.
    mode:
        ``"last"`` / ``"all"`` / ``"lastN"`` / ``"None"``.
    lastN:
        Used when ``mode="lastN"``.
    """

    def __init__(self, targetLayer, head_ids, mode: str, lastN: int = 0) -> None:
        self.head_ids = head_ids
        assert mode in ["last", "all", "lastN", "None"], "mode should be last or all"
        self.mode = mode
        self.lastN = lastN
        self.outputHandle = targetLayer.register_forward_hook(
            self.deactivate, with_kwargs=True
        )

    def deactivate(self, model, args, kwargs, output):
        import torch
        hidden_states = kwargs["hidden_states"]
        bsz, q_len, _ = hidden_states.size()
        value_states = model.v_proj(hidden_states)
        value_states = value_states.view(
            bsz, q_len, model.num_key_value_heads, model.head_dim
        ).transpose(1, 2)
        value_states = _repeat_kv(value_states, model.num_key_value_groups)

        attn_weight = output[1]
        n = attn_weight.size(2)
        attn_weight = attn_weight[:, :, :, :n]
        attn_output = torch.matmul(attn_weight, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.mode == "last":
            attn_output[:, -1, self.head_ids, :] = 0
        elif self.mode == "all":
            attn_output[:, :, self.head_ids, :] = 0
        elif self.mode == "lastN":
            attn_output[:, : -self.lastN, self.head_ids, :] = 0
        elif self.mode == "None":
            pass
        else:
            print(f"{self.mode=} cannot be recognized")

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = model.o_proj(attn_output)

        output = attn_output, output[1], output[2]
        return output

    def release(self) -> None:
        self.outputHandle.remove()


# ---------------------------------------------------------------------------
# convertNeuronsToDict  (from utils.py)
# ---------------------------------------------------------------------------

def convertNeuronsToDict(neurons) -> NeuronMask:
    """Convert a list of ``(layer, neuron)`` pairs into a layer-keyed dict.

    Faithful port of ``convertNeuronsToDict`` in Tsuji et al. utils.py.

    Parameters
    ----------
    neurons:
        Iterable of ``(layer_idx, neuron_idx)`` tuples.

    Returns
    -------
    dict[int, list[int]]
        ``{layer_idx: [neuron_idx, ...]}``.
    """
    layer2neurons: dict[int, list[int]] = {}
    for fn in neurons:
        i, j = fn
        if i not in layer2neurons:
            layer2neurons[i] = []
        layer2neurons[i].append(j)
    return layer2neurons


# ---------------------------------------------------------------------------
# get_rank  (from utils.py)
# ---------------------------------------------------------------------------

def get_rank(main_val, sub_val, top: str = "max") -> list:
    """Rank neurons by their differential activation (main − sub).

    Faithful port of ``get_rank`` in Tsuji et al. utils.py.

    Parameters
    ----------
    main_val:
        Tensor of shape ``[num_layers, d_ff]`` for the primary condition.
    sub_val:
        Either a single tensor of the same shape, or a list of such tensors.
        When a list, the element-wise max (``top="max"``) or min
        (``top="min"``) across all tensors is used as the baseline.
    top:
        ``"max"`` — rank by highest diff first (main dominates over subs).
        ``"min"`` — rank by lowest diff first.

    Returns
    -------
    list[dict]
        Each entry: ``{"position": (layer, neuron), "main_val": float,
        "sub_val": float, "diff": float}``, sorted by ``diff`` descending
        (``top="max"``) or ascending (``top="min"``).
    """
    import torch

    assert top in ["max", "min"]

    if not isinstance(sub_val, torch.Tensor):
        if top == "max":
            sub_val = torch.stack(sub_val).max(dim=0).values
        else:
            sub_val = torch.stack(sub_val).min(dim=0).values

    diff = main_val - sub_val
    descending = top == "max"
    ranks = torch.argsort(diff.flatten(), descending=descending)
    num_per_layer = diff.shape[1]
    sorted_val = []
    for r in ranks:
        position = (int(r // num_per_layer), int(r % num_per_layer))
        info = {
            "position": position,
            "main_val": main_val[position].tolist(),
            "sub_val": sub_val[position].tolist(),
            "diff": diff[position].tolist(),
        }
        sorted_val.append(info)

    return sorted_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "NeuronIndex",
    "NeuronMask",
    "OutputInspector",
    "get_acts",
    "Deactivator",
    "Activator",
    "HeadDeactivator",
    "convertNeuronsToDict",
    "get_rank",
]
