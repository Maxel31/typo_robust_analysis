"""Read-only Hugging Face runtime for the transition-layer causal gate."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from typo_robust_training.localization.corpus_targets import clean_corpus_targets
from typo_robust_training.localization.prompting import word_final_token_positions
from typo_robust_training.state_gate.artifacts import SingleLayerGateRecord
from typo_robust_training.state_gate.config import SingleLayerGateProtocol


_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_TYPO_COT_RUNTIME_MODULES = (
    "typo_cot",
    "typo_cot.experiments.layerwise_kl_patching.patching",
    "typo_cot.models.tokenizer_attestation",
    "typo_cot.models.wrapper",
)


def _checkout_source_attestation() -> tuple[str, str]:
    """Attest both executing source trees and return HEAD plus their tree digest."""

    module_path = Path(__file__).resolve()
    dependency_paths: list[Path] = []
    for module_name in _REQUIRED_TYPO_COT_RUNTIME_MODULES:
        try:
            loaded_module = importlib.import_module(module_name)
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "single-layer gate runtime cannot locate a required typo_cot source module"
            ) from exc
        dependency_file = getattr(loaded_module, "__file__", None)
        if not isinstance(dependency_file, str) or not dependency_file:
            raise RuntimeError(
                "single-layer gate runtime cannot locate a required typo_cot source module"
            )
        dependency_paths.append(Path(dependency_file).resolve())
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=module_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise RuntimeError("single-layer gate runtime cannot locate its git checkout")
    checkout_root = Path(root_result.stdout.strip()).resolve()
    try:
        module_relative = module_path.relative_to(checkout_root)
        dependency_relatives = tuple(
            dependency_path.relative_to(checkout_root) for dependency_path in dependency_paths
        )
        package_relatives = (
            module_path.parents[1].relative_to(checkout_root),
            Path("projects/typo-cot/src/typo_cot"),
        )
    except ValueError as exc:
        raise RuntimeError(
            "single-layer gate runtime or local dependency is outside the attested checkout"
        ) from exc
    expected_package_relatives = (
        Path("projects/typo-robust-training/src/typo_robust_training"),
        Path("projects/typo-cot/src/typo_cot"),
    )
    if package_relatives != expected_package_relatives:
        raise RuntimeError(
            "single-layer gate runtime source roots differ from the required checkout layout"
        )

    dependency_root = checkout_root / expected_package_relatives[1]
    if any(not path.is_relative_to(dependency_root) for path in dependency_paths):
        raise RuntimeError(
            "single-layer gate runtime or local dependency is outside the attested checkout"
        )

    for source_relative in (module_relative, *dependency_relatives):
        tracked_result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", source_relative.as_posix()],
            cwd=checkout_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked_result.returncode != 0:
            raise RuntimeError(
                "single-layer gate runtime source is not tracked by the attested checkout"
            )
    dirty_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *(relative.as_posix() for relative in package_relatives),
        ],
        cwd=checkout_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty_result.returncode != 0 or dirty_result.stdout.strip():
        raise RuntimeError("single-layer gate runtime source tree is not clean")
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout_root,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = head_result.stdout.strip()
    if head_result.returncode != 0 or _REVISION.fullmatch(revision) is None:
        raise RuntimeError(
            "single-layer gate runtime cannot attest the executing code revision"
        )

    tree_entries: list[tuple[str, str]] = []
    for relative in package_relatives:
        tree_result = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative.as_posix()}"],
            cwd=checkout_root,
            check=False,
            capture_output=True,
            text=True,
        )
        tree_revision = tree_result.stdout.strip()
        if tree_result.returncode != 0 or _REVISION.fullmatch(tree_revision) is None:
            raise RuntimeError(
                "single-layer gate runtime cannot attest a required source tree"
            )
        tree_entries.append((relative.as_posix(), tree_revision))
    digest = hashlib.sha256()
    for relative, tree_revision in sorted(tree_entries):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tree_revision.encode("ascii"))
        digest.update(b"\0")
    source_tree_sha256 = digest.hexdigest()
    if _SHA256.fullmatch(source_tree_sha256) is None:  # pragma: no cover - hashlib invariant
        raise RuntimeError("single-layer gate source tree digest is invalid")
    return revision, source_tree_sha256


def _checkout_code_revision() -> str:
    """Return the revision only after both source trees pass full attestation."""

    return _checkout_source_attestation()[0]


def _require_exact_model_revision(
    *,
    model_config: object,
    tokenizer: object,
    expected: str,
) -> tuple[str, str]:
    """Require independently observable, exact model and tokenizer revisions."""

    model_candidates: list[str] = []
    for config in (model_config, getattr(model_config, "text_config", None)):
        revision = getattr(config, "_commit_hash", None)
        if isinstance(revision, str) and _REVISION.fullmatch(revision) is not None:
            model_candidates.append(revision)
    if not model_candidates:
        raise ValueError("single-layer gate loaded model revision is not observable")
    if any(revision != expected for revision in model_candidates):
        raise ValueError(
            "single-layer gate loaded model revision differs from preregistration"
        )
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    tokenizer_revision = (
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, Mapping) else None
    )
    if (
        not isinstance(tokenizer_revision, str)
        or _REVISION.fullmatch(tokenizer_revision) is None
    ):
        raise ValueError("single-layer gate loaded tokenizer revision is not observable")
    if tokenizer_revision != expected:
        raise ValueError(
            "single-layer gate loaded tokenizer revision differs from preregistration"
        )
    return model_candidates[0], tokenizer_revision


def _inflation_bucket(delta: int) -> str:
    if delta <= -2:
        return "minus-two-or-more"
    if delta == -1:
        return "minus-one"
    if delta == 0:
        return "same"
    if delta == 1:
        return "plus-one"
    return "plus-two-or-more"


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


class HuggingFaceSingleLayerGateProvider:
    """Patch exactly the parent transition layer and its frozen controls."""

    def __init__(self, *, protocol: SingleLayerGateProtocol, gpu_id: str) -> None:
        if not isinstance(protocol, SingleLayerGateProtocol):
            raise TypeError("single-layer gate runtime requires its validated protocol")
        code_revision, source_tree_sha256 = _checkout_source_attestation()
        if code_revision != protocol.code_revision:
            raise ValueError(
                "single-layer gate executing code revision differs from preregistration"
            )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError("CUDA_VISIBLE_DEVICES conflicts with the requested gate GPU")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("single-layer gate requires exactly one requested CUDA GPU")
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.tokenizer_attestation import require_frozen_tokenizer_attestation
        from typo_cot.models.wrapper import create_model_wrapper

        wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self.protocol = protocol
        self.gpu_id = gpu_id
        self.model_id = protocol.model
        self.model_revision = protocol.model_revision
        self.source_tree_sha256 = source_tree_sha256
        self.code_revision = code_revision
        self.decoder_layers = protocol.decoder_layers
        self.base_model_frozen = True
        self._torch = torch
        self.model = wrapper.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = wrapper.tokenizer
        self.tokenizer_snapshot_attestation = require_frozen_tokenizer_attestation(
            wrapper,
            expected_model=protocol.model,
            expected_revision=protocol.model_revision,
        )
        if getattr(self.tokenizer, "is_fast", False) is not True:
            raise ValueError("single-layer gate requires a fast tokenizer")
        self.layers = tuple(find_decoder_layers(self.model))
        self.device = next(self.model.parameters()).device
        if len(self.layers) != protocol.decoder_layers or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise ValueError("single-layer gate model identity or freeze boundary differs")
        model_revision, tokenizer_revision = _require_exact_model_revision(
            model_config=self.model.config,
            tokenizer=self.tokenizer,
            expected=protocol.model_revision,
        )
        # The clean donor (teacher) and typo recipient (student) are two roles of
        # this same independently attested, frozen model instance.
        self.teacher_revision = model_revision
        self.student_revision = model_revision
        self.tokenizer_revision = tokenizer_revision

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(text, add_special_tokens=True)
        if not isinstance(encoded, Mapping):
            raise ValueError("single-layer gate tokenizer returned no token mapping")
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("single-layer gate tokenizer returned no input ids")
        if hasattr(input_ids, "ndim"):
            if input_ids.ndim == 2 and int(input_ids.shape[0]) == 1:
                return int(input_ids.shape[1])
            if input_ids.ndim == 1:
                return int(input_ids.shape[0])
            raise ValueError("single-layer gate tokenizer token shape differs")
        if (
            isinstance(input_ids, Sequence)
            and input_ids
            and isinstance(input_ids[0], Sequence)
        ):
            if len(input_ids) != 1:
                raise ValueError("single-layer gate tokenizer returned a batch")
            return len(input_ids[0])
        if isinstance(input_ids, Sequence):
            return len(input_ids)
        raise ValueError("single-layer gate tokenizer input ids differ")

    def token_inflation_bucket(self, record: SingleLayerGateRecord) -> str:
        """Recompute the frozen stratum with the actual bound tokenizer."""

        return _inflation_bucket(
            self._token_count(record.typo_text) - self._token_count(record.clean_text)
        )

    def _tokenize(self, text: str) -> tuple[Any, Any, tuple[int, ...], tuple[tuple[int, int], ...]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("single-layer gate tokenizer must return one mapping")
        ids = encoded.get("input_ids")
        mask = encoded.get("attention_mask")
        offsets = encoded.get("offset_mapping")
        if ids is None or ids.ndim != 2 or int(ids.shape[0]) != 1:
            raise ValueError("single-layer gate tokenizer must return one input sequence")
        if mask is None:
            mask = self._torch.ones_like(ids)
        if offsets is None or offsets.ndim != 3 or tuple(offsets.shape[:2]) != tuple(ids.shape):
            raise ValueError("single-layer gate tokenizer must return exact offsets")
        if mask.shape != ids.shape or not bool(mask.all()):
            raise ValueError("single-layer gate input must be unpadded")
        return (
            ids.to(self.device),
            mask.to(self.device),
            tuple(int(value) for value in ids[0].tolist()),
            tuple((int(row[0]), int(row[1])) for row in offsets[0].tolist()),
        )

    def _append_targets(self, ids: Any, mask: Any, targets: tuple[int, ...]) -> tuple[Any, Any]:
        prefix = self._torch.tensor([targets[:-1]], dtype=ids.dtype, device=ids.device)
        return (
            self._torch.cat((ids, prefix), dim=1),
            self._torch.cat((mask, self._torch.ones_like(prefix)), dim=1),
        )

    def _target_logits(self, output: Any, *, prompt_tokens: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("single-layer gate model output lacks rank-three logits")
        selected = logits[0, prompt_tokens - 1 : prompt_tokens + 15, :]
        if int(selected.shape[0]) != 16:
            raise ValueError("single-layer gate output lacks sixteen target logits")
        return selected.detach()

    def _kl(self, reference: Any, comparison: Any) -> tuple[float, ...]:
        reference_log = self._torch.log_softmax(reference.double(), dim=-1)
        comparison_log = self._torch.log_softmax(comparison.double(), dim=-1)
        values = (reference_log.exp() * (reference_log - comparison_log)).sum(dim=-1).clamp_min(0.0)
        return tuple(float(value) for value in values[1:16].cpu().tolist())

    def _donor(self, ids: Any, mask: Any, *, layer: int, position: int) -> Any:
        from typo_cot.experiments.layerwise_kl_patching.patching import capture_block_outputs

        donors = capture_block_outputs(
            self.layers,
            positions=(position,),
            forward=lambda: self.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
            ),
        )
        return donors[layer]

    def _patched(
        self,
        *,
        ids: Any,
        mask: Any,
        layer: int,
        position: int,
        donor: Any,
        prompt_tokens: int,
        reference: Any,
    ) -> tuple[float, ...]:
        from typo_cot.experiments.layerwise_kl_patching.patching import BlockOutputPatch

        with BlockOutputPatch(
            self.layers,
            layer_index=layer,
            positions=(position,),
            donor_values=donor,
        ):
            output = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        return self._kl(reference, self._target_logits(output, prompt_tokens=prompt_tokens))

    def _offset_patched(
        self,
        *,
        clean_ids: Any,
        clean_mask: Any,
        typo_ids: Any,
        typo_mask: Any,
        layer: int,
        clean_position: int,
        typo_position: int,
        prompt_tokens: int,
        reference: Any,
    ) -> tuple[float, ...]:
        """Apply the +2 control using matching +2 donor and recipient coordinates."""

        offset = self.protocol.offset_control_tokens
        donor = self._donor(
            clean_ids,
            clean_mask,
            layer=layer,
            position=clean_position + offset,
        )
        return self._patched(
            ids=typo_ids,
            mask=typo_mask,
            layer=layer,
            position=typo_position + offset,
            donor=donor,
            prompt_tokens=prompt_tokens,
            reference=reference,
        )

    def scan(
        self,
        records: Sequence[SingleLayerGateRecord],
        *,
        donor_plan: Mapping[str, str],
        transition_layer: int,
    ) -> Sequence[Mapping[str, object]]:
        if not 1 <= transition_layer < len(self.layers):
            raise ValueError("single-layer gate transition lies outside the decoder")
        prepared: dict[str, dict[str, object]] = {}
        with self._torch.inference_mode():
            for record in records:
                clean_prefix = record.clean_text[: record.clean_word_char_span[1]]
                typo_prefix = record.typo_text[: record.typo_word_char_span[1]]
                clean_ids, clean_mask, clean_tokens, clean_offsets = self._tokenize(clean_prefix)
                typo_ids, typo_mask, typo_tokens, typo_offsets = self._tokenize(typo_prefix)
                _full_ids, _full_mask, full_clean_tokens, _full_offsets = self._tokenize(record.clean_text)
                clean_position = word_final_token_positions(
                    self.tokenizer,
                    text=clean_prefix,
                    spans=(record.clean_word_char_span,),
                )[0]
                typo_position = word_final_token_positions(
                    self.tokenizer,
                    text=typo_prefix,
                    spans=(record.typo_word_char_span,),
                )[0]
                if clean_position != len(clean_tokens) - 1 or typo_position != len(typo_tokens) - 1:
                    raise ValueError("single-layer gate prefix must end at edited-word-final token")
                targets, reason = clean_corpus_targets(
                    full_clean_token_ids=full_clean_tokens,
                    clean_prompt_token_ids=clean_tokens,
                    final_edited_token=clean_position,
                    count=self.protocol.teacher_forced_tokens,
                )
                prepared[record.pair_id] = {
                    "record": record,
                    "clean_ids": clean_ids,
                    "clean_mask": clean_mask,
                    "typo_ids": typo_ids,
                    "typo_mask": typo_mask,
                    "clean_tokens": clean_tokens,
                    "typo_tokens": typo_tokens,
                    "clean_offsets": clean_offsets,
                    "typo_offsets": typo_offsets,
                    "clean_position": clean_position,
                    "typo_position": typo_position,
                    "targets": targets,
                    "invalid_reason": reason,
                    "clean_donor": self._donor(
                        clean_ids,
                        clean_mask,
                        layer=transition_layer,
                        position=clean_position,
                    ),
                    "typo_donor": self._donor(
                        typo_ids,
                        typo_mask,
                        layer=transition_layer,
                        position=typo_position,
                    ),
                }
            output_rows: list[dict[str, object]] = []
            for record in records:
                item = prepared[record.pair_id]
                donor_item = prepared[donor_plan[record.pair_id]]
                targets = item["targets"]
                reason = item["invalid_reason"]
                clean_position = int(item["clean_position"])
                typo_position = int(item["typo_position"])
                base = {
                    "pair_id": record.pair_id,
                    "source_group_sha256": record.source_group_sha256,
                    "stratum": record.stratum,
                    "transition_layer": transition_layer,
                    "clean_word_final_token": clean_position,
                    "typo_word_final_token": typo_position,
                    "offset_donor_clean_token": (
                        clean_position + self.protocol.offset_control_tokens
                    ),
                    "offset_patch_token": typo_position + self.protocol.offset_control_tokens,
                    "cross_donor_pair_id": donor_plan[record.pair_id],
                    "cross_donor_clean_word_final_token": int(donor_item["clean_position"]),
                    "cross_donor_clean_prompt_offsets": [
                        list(row) for row in donor_item["clean_offsets"]
                    ],
                    "clean_prompt_offsets": [list(row) for row in item["clean_offsets"]],
                    "typo_prompt_offsets": [list(row) for row in item["typo_offsets"]],
                }
                if reason is not None:
                    output_rows.append(
                        {
                            **base,
                            "target_token_ids": [],
                            "untreated_kl_2_16": [],
                            "correct_kl_2_16": [],
                            "offset_kl_2_16": [],
                            "cross_kl_2_16": [],
                            "self_copy_kl_2_16": [],
                            "invalid_reason": reason,
                        }
                    )
                    continue
                clean_full = self._append_targets(item["clean_ids"], item["clean_mask"], targets)
                typo_full = self._append_targets(item["typo_ids"], item["typo_mask"], targets)
                clean_output = self.model(
                    input_ids=clean_full[0], attention_mask=clean_full[1], use_cache=False
                )
                reference = self._target_logits(
                    clean_output, prompt_tokens=len(item["clean_tokens"])
                )
                typo_output = self.model(
                    input_ids=typo_full[0], attention_mask=typo_full[1], use_cache=False
                )
                untreated = self._kl(
                    reference,
                    self._target_logits(typo_output, prompt_tokens=len(item["typo_tokens"])),
                )
                if sum(untreated) / len(untreated) <= self.protocol.denominator_min_exclusive:
                    output_rows.append(
                        {
                            **base,
                            "target_token_ids": [],
                            "untreated_kl_2_16": [],
                            "correct_kl_2_16": [],
                            "offset_kl_2_16": [],
                            "cross_kl_2_16": [],
                            "self_copy_kl_2_16": [],
                            "invalid_reason": "untreated-kl-at-or-below-1e-9",
                        }
                    )
                    continue
                patch_common = {
                    "ids": typo_full[0],
                    "mask": typo_full[1],
                    "layer": transition_layer,
                    "prompt_tokens": len(item["typo_tokens"]),
                    "reference": reference,
                }
                output_rows.append(
                    {
                        **base,
                        "target_token_ids": list(targets),
                        "untreated_kl_2_16": list(untreated),
                        "correct_kl_2_16": list(
                            self._patched(
                                **patch_common,
                                position=typo_position,
                                donor=item["clean_donor"],
                            )
                        ),
                        "offset_kl_2_16": list(
                            self._offset_patched(
                                clean_ids=clean_full[0],
                                clean_mask=clean_full[1],
                                typo_ids=typo_full[0],
                                typo_mask=typo_full[1],
                                layer=transition_layer,
                                clean_position=clean_position,
                                typo_position=typo_position,
                                prompt_tokens=len(item["typo_tokens"]),
                                reference=reference,
                            )
                        ),
                        "cross_kl_2_16": list(
                            self._patched(
                                **patch_common,
                                position=typo_position,
                                donor=donor_item["clean_donor"],
                            )
                        ),
                        "self_copy_kl_2_16": list(
                            self._patched(
                                **patch_common,
                                position=typo_position,
                                donor=item["typo_donor"],
                            )
                        ),
                        "invalid_reason": None,
                    }
                )
        return tuple(output_rows)

    def provenance(self) -> Mapping[str, object]:
        torch = self._torch
        return {
            "schema_version": "single-layer-gate-runtime/v2",
            "provider": "hugging-face-single-layer-gate/v1",
            "model": self.model_id,
            "model_revision": self.model_revision,
            "teacher_revision": self.teacher_revision,
            "student_revision": self.student_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_snapshot_attestation": (
                self.tokenizer_snapshot_attestation.provenance_dict()
            ),
            "code_revision": self.code_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "decoder_layers": len(self.layers),
            "dtype": "bfloat16",
            "hook_site": "complete-decoder-block-residual-output",
            "coordinate": "edited-word-final-token/v1",
            "readout": "teacher-forced-tokens-2-through-16-inclusive/v1",
            "base_model_frozen": True,
            "packages": {
                "python": platform.python_version(),
                "torch": _version("torch"),
                "transformers": _version("transformers"),
                "typo-cot": _version("typo-cot"),
            },
            "hardware": {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        }


__all__ = ["HuggingFaceSingleLayerGateProvider"]
