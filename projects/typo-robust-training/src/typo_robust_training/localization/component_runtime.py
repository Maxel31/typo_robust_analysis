"""GPU runtime for approximate screening and component-level causal patches."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from typo_robust_training.localization.component_causal import ComponentCausalObservation
from typo_robust_training.localization.component_config import (
    ComponentLocalizationProtocol,
)
from typo_robust_training.localization.component_patching import (
    ComponentInputPatch,
    capture_module_inputs,
)
from typo_robust_training.localization.component_screening import ComponentScreenMetric
from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.localization.prompting import (
    PromptSide,
    build_diagnostic_prompts,
    word_final_token_positions,
)
from typo_robust_training.localization.records import LayerScan


_MAX_NEW_TOKENS = 512


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def component_statistics(
    *,
    kind: str,
    clean: Any,
    typo: Any,
    gradient: Any,
    attention_head_dim: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute activation-distance and first-order patch estimates per component."""

    import torch

    if kind not in {"mlp-neuron", "attention-head"}:
        raise ValueError("component statistics kind is unsupported")
    if not all(isinstance(value, torch.Tensor) for value in (clean, typo, gradient)):
        raise TypeError("component statistics inputs must be torch tensors")
    if clean.ndim != 2 or clean.shape != typo.shape or clean.shape != gradient.shape:
        raise ValueError(
            "component statistics tensors must have identical [positions, features] shape"
        )
    if int(clean.shape[0]) == 0 or int(clean.shape[1]) == 0:
        raise ValueError("component statistics tensors must be non-empty")
    if (
        isinstance(attention_head_dim, bool)
        or not isinstance(attention_head_dim, int)
        or attention_head_dim <= 0
    ):
        raise ValueError("attention_head_dim must be positive")
    delta = clean.float() - typo.float()
    gradients = gradient.float()
    if kind == "mlp-neuron":
        activation = delta.abs().mean(dim=0)
        attribution = -(gradients * delta).mean(dim=0)
    else:
        if int(delta.shape[1]) % attention_head_dim:
            raise ValueError("attention feature dimension is not divisible by head_dim")
        heads = int(delta.shape[1]) // attention_head_dim
        delta = delta.reshape(int(delta.shape[0]), heads, attention_head_dim)
        gradients = gradients.reshape(int(gradients.shape[0]), heads, attention_head_dim)
        activation = torch.linalg.vector_norm(delta, dim=-1).mean(dim=0)
        attribution = -(gradients * delta).sum(dim=-1).mean(dim=0)
    activation_values = tuple(float(value) for value in activation.detach().cpu().tolist())
    attribution_values = tuple(float(value) for value in attribution.detach().cpu().tolist())
    if any(not math.isfinite(value) for value in (*activation_values, *attribution_values)):
        raise ValueError("component statistics produced non-finite values")
    return activation_values, attribution_values


class _GradientInputCapture:
    """Retain gradients at several module inputs without severing their graph."""

    def __init__(self, modules: Sequence[Any]) -> None:
        self.modules = tuple(modules)
        if not self.modules:
            raise ValueError("gradient component modules must not be empty")
        self.values: list[Any | None] = [None] * len(self.modules)
        self.handles: list[Any] = []

    def __enter__(self) -> _GradientInputCapture:
        def make_hook(index: int):
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                if not inputs or not hasattr(inputs[0], "retain_grad"):
                    raise TypeError("gradient component input must start with a tensor")
                hidden = inputs[0]
                if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
                    raise ValueError("gradient component input must be [1, sequence, features]")
                if self.values[index] is not None:
                    raise RuntimeError("gradient component module ran more than once")
                hidden.retain_grad()
                self.values[index] = hidden

            return hook

        for index, module in enumerate(self.modules):
            self.handles.append(module.register_forward_pre_hook(make_hook(index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc_type is None and any(value is None for value in self.values):
            raise RuntimeError("gradient forward skipped a selected component module")
        return False


class HuggingFaceComponentLocalizationRuntime:
    """Screen within selected layers, then patch one candidate at a time."""

    def __init__(self, *, protocol: ComponentLocalizationProtocol, gpu_id: str) -> None:
        if not isinstance(protocol, ComponentLocalizationProtocol):
            raise TypeError("protocol must be ComponentLocalizationProtocol")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("component localization requires exactly one requested CUDA GPU")
        from typo_cot.evaluation.generation import resolve_effective_eos_token_ids
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.wrapper import create_model_wrapper

        self.protocol = protocol
        self.gpu_id = gpu_id
        self._torch = torch
        self.wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self.model = self.wrapper.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = self.wrapper.tokenizer
        self.tokenizer.padding_side = "left"
        self.layers = find_decoder_layers(self.model)
        self.device = next(self.model.parameters()).device
        self.effective_eos_token_ids, self.eos_source = resolve_effective_eos_token_ids(
            generation_config=self.model.generation_config,
            tokenizer=self.tokenizer,
            operation="localize-robustness-components",
        )
        self._validate_architecture()

    def _validate_architecture(self) -> None:
        if len(self.layers) != self.protocol.decoder_layers:
            raise ValueError("component model decoder layer count differs from config")
        text_config = getattr(self.model.config, "text_config", self.model.config)
        expected = {
            "hidden_size": self.protocol.hidden_size,
            "intermediate_size": self.protocol.mlp_intermediate_size,
            "num_attention_heads": self.protocol.attention_heads,
            "head_dim": self.protocol.attention_head_dim,
        }
        for field, value in expected.items():
            if getattr(text_config, field, None) != value:
                raise ValueError(f"component model {field} differs from config")
        for index, layer in enumerate(self.layers):
            mlp = getattr(getattr(layer, "mlp", None), "down_proj", None)
            attention = getattr(getattr(layer, "self_attn", None), "o_proj", None)
            if mlp is None or attention is None:
                raise ValueError(f"decoder layer {index} lacks a frozen component site")
            if mlp.in_features != self.protocol.mlp_intermediate_size:
                raise ValueError(f"decoder layer {index} MLP component width differs")
            expected_attention = self.protocol.attention_heads * self.protocol.attention_head_dim
            if attention.in_features != expected_attention:
                raise ValueError(f"decoder layer {index} attention component width differs")

    def _module(self, component: ComponentRef) -> Any:
        layer = self.layers[component.layer]
        return layer.mlp.down_proj if component.kind == "mlp-neuron" else layer.self_attn.o_proj

    def _modules(self, *, kind: str, layers: tuple[int, ...]) -> tuple[Any, ...]:
        return tuple(self._module(ComponentRef(kind, layer, 0)) for layer in layers)

    def _encode(self, side: PromptSide) -> tuple[Any, Any, tuple[int, ...]]:
        positions = word_final_token_positions(
            self.tokenizer,
            text=side.text,
            spans=side.spans,
        )
        encoded = self.tokenizer(
            side.text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("component prompt tokenizer must return one sequence")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
            raise ValueError("component prompt must be unpadded and fully attended")
        return input_ids.to(self.device), attention_mask.to(self.device), positions

    def _validate_scan(self, prompts: Any, scan: LayerScan) -> None:
        if scan.record_id != prompts.record_id or scan.task != prompts.task:
            raise ValueError("component runtime layer-scan identity differs")
        audit = scan.audit
        for side in ("clean", "typo"):
            expected = audit.get(f"{side}_prompt_sha256")
            if expected is not None:
                actual = hashlib.sha256(getattr(prompts, side).text.encode()).hexdigest()
                if expected != actual:
                    raise ValueError(f"component runtime {side} prompt hash differs")

    def _append_targets(self, ids: Any, mask: Any, targets: tuple[int, ...]) -> tuple[Any, Any]:
        if len(targets) != 16:
            raise ValueError("component runtime requires sixteen frozen clean targets")
        prefix = self._torch.tensor([targets[:-1]], dtype=ids.dtype, device=ids.device)
        return (
            self._torch.cat((ids, prefix), dim=1),
            self._torch.cat((mask, self._torch.ones_like(prefix, dtype=mask.dtype)), dim=1),
        )

    @staticmethod
    def _target_logits(output: Any, *, prompt_tokens: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("component model output must contain rank-three logits")
        selected = logits[0, prompt_tokens - 1 : prompt_tokens + 15, :]
        if int(selected.shape[0]) != 16:
            raise ValueError("component output does not cover sixteen targets")
        return selected

    def _kl_trajectory(self, reference: Any, comparison: Any) -> Any:
        reference_log = self._torch.log_softmax(reference.float(), dim=-1)
        comparison_log = self._torch.log_softmax(comparison.float(), dim=-1)
        return (reference_log.exp() * (reference_log - comparison_log)).sum(dim=-1).clamp_min(0.0)

    def _prompt_capture(
        self,
        *,
        ids: Any,
        mask: Any,
        positions: tuple[int, ...],
        layers: tuple[int, ...],
    ) -> dict[tuple[str, int], Any]:
        modules = tuple(
            module
            for kind in ("mlp-neuron", "attention-head")
            for module in self._modules(kind=kind, layers=layers)
        )
        values = capture_module_inputs(
            modules,
            positions=positions,
            forward=lambda: self.model(input_ids=ids, attention_mask=mask, use_cache=False),
        )
        result: dict[tuple[str, int], Any] = {}
        cursor = 0
        for kind in ("mlp-neuron", "attention-head"):
            for layer in layers:
                result[(kind, layer)] = values[cursor]
                cursor += 1
        return result

    def _gradient_values(
        self,
        *,
        ids: Any,
        mask: Any,
        prompt_tokens: int,
        positions: tuple[int, ...],
        targets: tuple[int, ...],
        reference_logits: Any,
        kind: str,
        layers: tuple[int, ...],
    ) -> dict[int, Any]:
        full_ids, full_mask = self._append_targets(ids, mask, targets)
        modules = self._modules(kind=kind, layers=layers)
        embeddings = self.model.get_input_embeddings()(full_ids).detach().requires_grad_(True)
        with self._torch.enable_grad(), _GradientInputCapture(modules) as capture:
            output = self.model(
                inputs_embeds=embeddings,
                attention_mask=full_mask,
                use_cache=False,
            )
            logits = self._target_logits(output, prompt_tokens=prompt_tokens)
            objective = self._kl_trajectory(reference_logits.detach(), logits)[1:16].mean()
            objective.backward()
        results: dict[int, Any] = {}
        for layer, hidden in zip(layers, capture.values, strict=True):
            if hidden is None or hidden.grad is None:
                raise RuntimeError("component input gradient is unavailable")
            results[layer] = hidden.grad[0, positions, :].detach().float().cpu()
        return results

    def screen_pair(
        self,
        record: dict[str, object],
        layer_scan: LayerScan,
        selected_layers: tuple[int, ...],
    ) -> tuple[ComponentScreenMetric, ...]:
        """Return full component arrays for one KL-eligible screening record."""

        if not layer_scan.untreated_kl_2_16:
            return ()
        prompts = build_diagnostic_prompts(record)
        self._validate_scan(prompts, layer_scan)
        clean_ids, clean_mask, clean_positions = self._encode(prompts.clean)
        typo_ids, typo_mask, typo_positions = self._encode(prompts.typo)
        if len(clean_positions) != len(typo_positions):
            raise ValueError("component screen alignment cardinalities differ")
        targets = layer_scan.target_token_ids
        clean_full = self._append_targets(clean_ids, clean_mask, targets)
        with self._torch.no_grad():
            clean_output = self.model(
                input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False
            )
            reference_logits = self._target_logits(
                clean_output, prompt_tokens=int(clean_ids.shape[1])
            ).detach()
            del clean_output
            clean_values = self._prompt_capture(
                ids=clean_ids,
                mask=clean_mask,
                positions=clean_positions,
                layers=selected_layers,
            )
            typo_values = self._prompt_capture(
                ids=typo_ids,
                mask=typo_mask,
                positions=typo_positions,
                layers=selected_layers,
            )
        metrics: list[ComponentScreenMetric] = []
        for kind in ("mlp-neuron", "attention-head"):
            gradients = self._gradient_values(
                ids=typo_ids,
                mask=typo_mask,
                prompt_tokens=int(typo_ids.shape[1]),
                positions=typo_positions,
                targets=targets,
                reference_logits=reference_logits,
                kind=kind,
                layers=selected_layers,
            )
            for layer in selected_layers:
                activation, attribution = component_statistics(
                    kind=kind,
                    clean=clean_values[(kind, layer)],
                    typo=typo_values[(kind, layer)],
                    gradient=gradients[layer],
                    attention_head_dim=self.protocol.attention_head_dim,
                )
                for index, (difference, effect) in enumerate(
                    zip(activation, attribution, strict=True)
                ):
                    metrics.append(
                        ComponentScreenMetric(
                            component=ComponentRef(kind, layer, index),
                            task=prompts.task,
                            records=1,
                            activation_difference=difference,
                            gradient_attribution=effect,
                        )
                    )
        return tuple(metrics)

    def _generate(
        self,
        *,
        ids: Any,
        mask: Any,
        answer: str,
        task: str,
        field: str,
        patch: Any,
    ) -> dict[str, object]:
        from typo_cot.evaluation.fallback import extract_with_fallback
        from typo_cot.evaluation.generation import classify_generated_token_ids

        context = patch if patch is not None else nullcontext()
        with context:
            output = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                return_dict_in_generate=False,
                output_scores=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.effective_eos_token_ids),
            )
        raw = output[0, int(ids.shape[1]) :].detach().cpu().tolist()
        token_ids, termination = classify_generated_token_ids(
            raw,
            effective_eos_token_ids=self.effective_eos_token_ids,
            max_new_tokens=_MAX_NEW_TOKENS,
            field=field,
        )
        text = self.tokenizer.decode(
            list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        extraction = extract_with_fallback(
            text,
            benchmark=task,
            correct_answer=answer,
            allow_positional=termination == "eos",
        )
        return {
            "token_ids": list(token_ids),
            "text": text,
            "termination": termination,
            "value": extraction.value,
            "is_correct": extraction.is_correct,
            "method": extraction.method,
            "primary_method": extraction.primary_method,
        }

    def _donor_slice(self, component: ComponentRef, values: Any) -> Any:
        if component.kind == "mlp-neuron":
            return values[:, component.index : component.index + 1]
        start = component.index * self.protocol.attention_head_dim
        return values[:, start : start + self.protocol.attention_head_dim]

    def causal_pair(
        self,
        record: dict[str, object],
        layer_scan: LayerScan,
        candidates: tuple[ComponentRef, ...],
    ) -> tuple[ComponentCausalObservation, ...]:
        """Patch each shortlisted component on one disjoint validation record."""

        prompts = build_diagnostic_prompts(record)
        self._validate_scan(prompts, layer_scan)
        clean_ids, clean_mask, clean_positions = self._encode(prompts.clean)
        typo_ids, typo_mask, typo_positions = self._encode(prompts.typo)
        if len(clean_positions) != len(typo_positions):
            raise ValueError("component causal alignment cardinalities differ")
        layers = tuple(sorted({component.layer for component in candidates}))
        valid_kl = bool(layer_scan.untreated_kl_2_16)
        reference_logits = None
        typo_full = None
        with self._torch.inference_mode():
            if valid_kl:
                clean_full = self._append_targets(
                    clean_ids, clean_mask, layer_scan.target_token_ids
                )
                typo_full = self._append_targets(typo_ids, typo_mask, layer_scan.target_token_ids)
                clean_output = self.model(
                    input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False
                )
                reference_logits = self._target_logits(
                    clean_output, prompt_tokens=int(clean_ids.shape[1])
                ).detach()
                del clean_output
            donors = self._prompt_capture(
                ids=clean_ids,
                mask=clean_mask,
                positions=clean_positions,
                layers=layers,
            )
            observations: list[ComponentCausalObservation] = []
            for component in candidates:
                donor = self._donor_slice(component, donors[(component.kind, component.layer)])
                patched_trajectory: tuple[float, ...] = ()
                patched_mean: float | None = None
                if valid_kl:
                    if typo_full is None or reference_logits is None:
                        raise RuntimeError("valid component KL is missing frozen targets")
                    with ComponentInputPatch(
                        self._module(component),
                        component=component,
                        positions=typo_positions,
                        donor_values=donor,
                        attention_head_dim=self.protocol.attention_head_dim,
                    ):
                        output = self.model(
                            input_ids=typo_full[0],
                            attention_mask=typo_full[1],
                            use_cache=False,
                        )
                    logits = self._target_logits(output, prompt_tokens=int(typo_ids.shape[1]))
                    trajectory = self._kl_trajectory(reference_logits, logits).detach().cpu()
                    patched_trajectory = tuple(float(value) for value in trajectory[1:16].tolist())
                    patched_mean = sum(patched_trajectory) / len(patched_trajectory)
                    del output
                patched_answer = None
                patched_correct = None
                if layer_scan.clean_correct:
                    patched_answer = self._generate(
                        ids=typo_ids,
                        mask=typo_mask,
                        answer=prompts.answer,
                        task=prompts.task,
                        field=f"{prompts.record_id}:{component.identifier}",
                        patch=ComponentInputPatch(
                            self._module(component),
                            component=component,
                            positions=typo_positions,
                            donor_values=donor,
                            attention_head_dim=self.protocol.attention_head_dim,
                        ),
                    )
                    patched_correct = bool(patched_answer["is_correct"])
                observations.append(
                    ComponentCausalObservation(
                        record_id=prompts.record_id,
                        task=prompts.task,
                        component=component,
                        untreated_mean_kl=layer_scan.untreated_mean_kl,
                        patched_mean_kl=patched_mean,
                        clean_correct=layer_scan.clean_correct,
                        typo_correct=layer_scan.typo_correct,
                        patched_correct=patched_correct,
                        audit={
                            "patched_kl_2_16": list(patched_trajectory),
                            "patched_answer": patched_answer,
                        },
                    )
                )
        return tuple(observations)

    def provenance(self) -> dict[str, object]:
        torch = self._torch
        return {
            "runtime": "HuggingFaceComponentLocalizationRuntime/v1",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "model": self.protocol.model,
            "requested_revision": self.protocol.model_revision,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "num_decoder_layers": len(self.layers),
            "dtype": self.protocol.dtype,
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.eos_source,
            "mlp_site": self.protocol.mlp_site,
            "attention_site": self.protocol.attention_site,
            "screen_gradient_source": "inputs-embeds-leaf-with-frozen-model-parameters/v1",
            "screening_status": "approximate-not-causal/v1",
            "causal_patch": "one-component-clean-to-typo-prefill/v1",
        }


__all__ = ["HuggingFaceComponentLocalizationRuntime", "component_statistics"]
