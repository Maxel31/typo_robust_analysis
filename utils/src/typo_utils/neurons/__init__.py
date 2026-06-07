"""Neuron-level analysis utilities (FFN intermediate activations, masks).

Shared by M0 (typo-neuron identification) and M4 (activation shift).
"""
from typo_utils.neurons.hooks import (
    FFNActivationHook,
    NeuronIndex,
    NeuronMask,
    collect_ffn_activations,
)

__all__ = ["NeuronIndex", "NeuronMask", "FFNActivationHook", "collect_ffn_activations"]
