"""GPU runtime and exact rank-projected block-output interventions."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

import numpy as np

from typo_robust_training.localization.corpus_targets import clean_corpus_targets
from typo_robust_training.localization.prompting import word_final_token_positions
from typo_robust_training.probe.artifacts import ProbeTransitionArtifact
from typo_robust_training.probe.subspace import SemanticProbeSubspace
from typo_robust_training.probe.subspace_kill_config import SemanticSubspaceKillProtocol
from typo_robust_training.probe.subspace_kill_scoring import SubspaceKillScoreRow


def _hidden(output: object) -> Any:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not hasattr(value, "ndim") or value.ndim != 3 or int(value.shape[0]) != 1:
        raise ValueError("semantic patch block output must be [1, sequence, hidden]")
    return value


@contextmanager
def block_output_subspace_patch(
    layer: Any,
    *,
    position: int,
    clean_donor: Any,
    row_basis: Any | None,
) -> Iterator[None]:
    """Replace a full state or only the ``Q.T Q`` component at one token."""

    import torch

    if not isinstance(layer, torch.nn.Module):
        raise TypeError("semantic patch layer must be a torch module")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("semantic patch token position differs")
    if (
        not isinstance(clean_donor, torch.Tensor)
        or clean_donor.ndim != 1
        or int(clean_donor.numel()) == 0
        or not bool(torch.isfinite(clean_donor).all())
    ):
        raise ValueError("semantic patch clean donor differs")
    basis = None
    if row_basis is not None:
        if (
            not isinstance(row_basis, torch.Tensor)
            or row_basis.ndim != 2
            or int(row_basis.shape[1]) != int(clean_donor.numel())
            or int(row_basis.shape[0]) <= 0
            or row_basis.requires_grad
            or not bool(torch.isfinite(row_basis).all())
        ):
            raise ValueError("semantic patch row basis differs")
        gram = row_basis.double() @ row_basis.double().T
        if not torch.allclose(
            gram,
            torch.eye(int(row_basis.shape[0]), dtype=torch.float64, device=gram.device),
            atol=1e-8,
            rtol=1e-8,
        ):
            raise ValueError("semantic patch row basis is not orthonormal")
        basis = row_basis
    calls = 0

    def patch(_module: Any, _inputs: Any, output: Any) -> Any:
        nonlocal calls
        calls += 1
        hidden = _hidden(output)
        if calls != 1 or position >= int(hidden.shape[1]) or int(hidden.shape[2]) != int(
            clean_donor.numel()
        ):
            raise RuntimeError("semantic patch ran at the wrong count or coordinate")
        changed = hidden.clone()
        current = changed[0, position].float()
        donor = clean_donor.detach().to(device=hidden.device, dtype=torch.float32)
        if basis is None:
            replacement = donor
        else:
            q = basis.detach().to(device=hidden.device, dtype=torch.float32)
            delta = donor - current
            replacement = current + (delta @ q.T) @ q
        changed[0, position] = replacement.to(dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        if isinstance(output, list):
            return [changed, *output[1:]]
        return changed

    handle = layer.register_forward_hook(patch)
    try:
        yield
    finally:
        handle.remove()
    if calls != 1:
        raise RuntimeError("semantic patch did not run exactly once")


class HuggingFaceSemanticSubspaceKillRuntime:
    """Run all preregistered single-layer interventions on frozen generic text."""

    def __init__(
        self,
        *,
        protocol: SemanticSubspaceKillProtocol,
        parent: ProbeTransitionArtifact,
        semantic_by_seed: Mapping[int, SemanticProbeSubspace],
        pca_basis: np.ndarray,
        random_basis: np.ndarray,
        complement_by_seed: Mapping[int, np.ndarray],
        gpu_id: str,
    ) -> None:
        if not isinstance(protocol, SemanticSubspaceKillProtocol) or not isinstance(
            parent, ProbeTransitionArtifact
        ):
            raise TypeError("semantic kill runtime requires validated protocol and parent")
        if (
            parent.artifact_sha256 != protocol.parent_artifact_sha256
            or parent.model != protocol.model
            or parent.model_revision != protocol.model_revision
            or parent.code_revision != protocol.code_revision
            or set(semantic_by_seed) != set(parent.probe_seeds)
            or set(complement_by_seed) != set(parent.probe_seeds)
        ):
            raise ValueError("semantic kill runtime evidence identity differs")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError("CUDA_VISIBLE_DEVICES conflicts with semantic kill GPU")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("semantic kill runtime requires exactly one requested GPU")
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.wrapper import create_model_wrapper

        wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self.model = wrapper.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = wrapper.tokenizer
        if getattr(self.tokenizer, "is_fast", False) is not True:
            raise ValueError("semantic kill runtime requires a fast tokenizer")
        self.layers = tuple(find_decoder_layers(self.model))
        if len(self.layers) != parent.decoder_layers:
            raise ValueError("semantic kill runtime decoder layers differ")
        self.device = next(self.model.parameters()).device
        self.protocol = protocol
        self.parent = parent
        self._torch = torch
        self._semantic = {
            seed: torch.tensor(
                semantic_by_seed[seed].basis,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            for seed in parent.probe_seeds
        }
        self._pca = torch.tensor(
            pca_basis, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._random = torch.tensor(
            random_basis, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._complement = {
            seed: torch.tensor(
                complement_by_seed[seed],
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            for seed in parent.probe_seeds
        }

    @staticmethod
    def _span(record: Mapping[str, object], side: str) -> tuple[int, int]:
        value = record.get(f"{side}_word_char_span")
        text = record.get(f"{side}_text")
        if (
            not isinstance(text, str)
            or not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            or not 0 <= value[0] < value[1] <= len(text)
        ):
            raise ValueError("semantic kill runtime word span differs")
        return value[0], value[1]

    def _tokenize(self, text: str) -> tuple[Any, Any, tuple[int, ...]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if ids.ndim != 2 or ids.shape != mask.shape or int(ids.shape[0]) != 1 or not bool(
            mask.all()
        ):
            raise ValueError("semantic kill tokenizer output differs")
        return ids.to(self.device), mask.to(self.device), tuple(int(value) for value in ids[0])

    def _append_targets(self, ids: Any, mask: Any, targets: tuple[int, ...]) -> tuple[Any, Any]:
        prefix = self._torch.tensor([targets[:-1]], dtype=ids.dtype, device=ids.device)
        return (
            self._torch.cat((ids, prefix), dim=1),
            self._torch.cat((mask, self._torch.ones_like(prefix)), dim=1),
        )

    @staticmethod
    def _logits(output: Any, *, prompt_tokens: int) -> Any:
        logits = output.logits[0, prompt_tokens - 1 : prompt_tokens + 15]
        if logits.ndim != 2 or int(logits.shape[0]) != 16:
            raise ValueError("semantic kill readout does not cover sixteen targets")
        return logits

    def _kl(self, clean: Any, other: Any) -> tuple[float, ...]:
        clean_log = self._torch.log_softmax(clean.double(), dim=-1)
        other_log = self._torch.log_softmax(other.double(), dim=-1)
        values = (clean_log.exp() * (clean_log - other_log)).sum(dim=-1).clamp_min(0.0)
        return tuple(float(value) for value in values[1:].cpu())

    def _capture_donor(self, ids: Any, mask: Any, *, position: int) -> Any:
        captured: list[Any] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(_hidden(output)[0, position].detach().float())

        handle = self.layers[self.parent.selected_transition_layer].register_forward_hook(capture)
        try:
            self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("semantic kill runtime did not capture exactly one donor")
        return captured[0]

    def scan_pair_all_seeds(
        self, record: Mapping[str, object]
    ) -> Mapping[int, SubspaceKillScoreRow]:
        """Return seed-specific raw KL rows while sharing invariant interventions."""

        clean = record.get("clean_text")
        typo = record.get("typo_text")
        pair_id = record.get("pair_id")
        group = record.get("source_group_sha256")
        if not all(isinstance(value, str) and value for value in (clean, typo, pair_id, group)):
            raise ValueError("semantic kill runtime record identity differs")
        clean_span, typo_span = self._span(record, "clean"), self._span(record, "typo")
        clean_prefix, typo_prefix = clean[: clean_span[1]], typo[: typo_span[1]]
        clean_ids, clean_mask, clean_tokens = self._tokenize(clean_prefix)
        typo_ids, typo_mask, typo_tokens = self._tokenize(typo_prefix)
        _full_ids, _full_mask, full_clean_tokens = self._tokenize(clean)
        clean_position = word_final_token_positions(
            self.tokenizer, text=clean_prefix, spans=(clean_span,)
        )[0]
        typo_position = word_final_token_positions(
            self.tokenizer, text=typo_prefix, spans=(typo_span,)
        )[0]
        if (
            clean_position != record.get("clean_word_final_token")
            or typo_position != record.get("typo_word_final_token")
            or typo_position != len(typo_tokens) - 1
        ):
            raise ValueError("semantic kill runtime registered token coordinate differs")
        targets, reason = clean_corpus_targets(
            full_clean_token_ids=full_clean_tokens,
            clean_prompt_token_ids=clean_tokens,
            final_edited_token=clean_position,
            count=16,
        )
        if reason is not None:
            return MappingProxyType(
                {
                    seed: SubspaceKillScoreRow(
                        pair_id=pair_id,
                        source_group_sha256=group,
                        transition_layer=self.parent.selected_transition_layer,
                        clean_word_final_token=clean_position,
                        typo_word_final_token=typo_position,
                        untreated_kl_2_16=(),
                        patched_kl_2_16={},
                        invalid_reason=reason,
                    )
                    for seed in self.parent.probe_seeds
                }
            )
        clean_full = self._append_targets(clean_ids, clean_mask, targets)
        typo_full = self._append_targets(typo_ids, typo_mask, targets)
        layer = self.layers[self.parent.selected_transition_layer]
        with self._torch.inference_mode():
            reference = self._logits(
                self.model(input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False),
                prompt_tokens=len(clean_tokens),
            ).detach()
            untreated = self._kl(
                reference,
                self._logits(
                    self.model(
                        input_ids=typo_full[0], attention_mask=typo_full[1], use_cache=False
                    ),
                    prompt_tokens=len(typo_tokens),
                ),
            )
            donor = self._capture_donor(clean_ids, clean_mask, position=clean_position)

            def run(basis: Any | None) -> tuple[float, ...]:
                with block_output_subspace_patch(
                    layer,
                    position=typo_position,
                    clean_donor=donor,
                    row_basis=basis,
                ):
                    output = self.model(
                        input_ids=typo_full[0], attention_mask=typo_full[1], use_cache=False
                    )
                return self._kl(
                    reference, self._logits(output, prompt_tokens=len(typo_tokens))
                )

            common = {
                "full-state": run(None),
                "clean-fit-pca-rank16": run(self._pca),
                "deterministic-haar-random-rank16": run(self._random),
            }
            return MappingProxyType(
                {
                    seed: SubspaceKillScoreRow(
                        pair_id=pair_id,
                        source_group_sha256=group,
                        transition_layer=self.parent.selected_transition_layer,
                        clean_word_final_token=clean_position,
                        typo_word_final_token=typo_position,
                        untreated_kl_2_16=untreated,
                        patched_kl_2_16={
                            **common,
                            "semantic-rank16": run(self._semantic[seed]),
                            "semantic-complement-rank16": run(self._complement[seed]),
                        },
                    )
                    for seed in self.parent.probe_seeds
                }
            )

    def provenance(self) -> Mapping[str, object]:
        return {
            "schema_version": "probe-semantic-subspace-kill-runtime/v1",
            "runtime": "HuggingFaceSemanticSubspaceKillRuntime/v1",
            "model": self.protocol.model,
            "model_revision": self.protocol.model_revision,
            "code_revision": self.protocol.code_revision,
            "transition_layer": self.parent.selected_transition_layer,
            "hook_site": self.protocol.hook_site,
            "coordinate": self.protocol.coordinate,
            "operators": list(self.protocol.operators),
            "random_basis_seed": self.protocol.random_basis_seed,
            "complement_basis_seed": self.protocol.complement_basis_seed,
            "teacher_forced_offsets": list(self.protocol.readout_offsets),
        }


__all__ = ["HuggingFaceSemanticSubspaceKillRuntime", "block_output_subspace_patch"]
