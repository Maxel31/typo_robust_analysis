"""Controlled adaptation of UCC-Inj experiment 6.

The upstream experiment uses Qwen/Qwen3-30B-A3B and checked-in noisy
GSM8K files. This module intentionally implements a reproducibility-hardened
Gemma protocol adaptation. It never silently truncates either side of a
clean/noisy pair.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .noise import inject_variation_selector_noise

ADAPTATION_SCOPE = "adaptation"
COMPARISON_POSITION = "last_non_padding_model_input_token"
UPSTREAM_REPOSITORY = "https://github.com/yifusuyi/UCC-Inj"
UPSTREAM_COMMIT = "aec814fcbb4388fa6fa874fbcec08a3ab1f78190"
UPSTREAM_MODEL = "Qwen/Qwen3-30B-A3B"
UPSTREAM_EXPERIMENT_SCRIPT = "exp6/test_similarity_with_clean_text.py"
UPSTREAM_NOISE_ENCODER = "datasets/gsm8k/main/encoder.py"
DEFAULT_GEMMA_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
DEFAULT_GSM8K_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"


class InputTooLongError(ValueError):
    """Raised before inference when a complete model input exceeds the hard cap."""


def _is_full_commit_sha(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return len(lowered) == 40 and all(character in "0123456789abcdef" for character in lowered)


def _validate_huggingface_repo_id(*, field: str, value: str) -> None:
    """Reject local paths and ambiguous Hub sources before snapshot resolution."""
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    parts = value.split("/")
    if (
        len(parts) != 2
        or any(
            not part
            or part in {".", ".."}
            or any(character not in allowed for character in part)
            for part in parts
        )
    ):
        raise ValueError(f"{field} must be a Hugging Face Hub repository ID in owner/name form")


@dataclass(frozen=True)
class Exp6Config:
    model: str
    protocol_scope: str = ADAPTATION_SCOPE
    model_revision: str = DEFAULT_GEMMA_REVISION
    dataset: str = "openai/gsm8k"
    dataset_revision: str = DEFAULT_GSM8K_REVISION
    dataset_config: str = "main"
    split: str = "test"
    question_field: str = "question"
    seed: int = 42
    noise_levels: tuple[int, ...] = (0, 1, 2, 3)
    limit: int | None = None
    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    add_generation_prompt: bool = False

    def validate(self) -> None:
        string_fields = (
            ("protocol_scope", self.protocol_scope),
            ("model", self.model),
            ("model_revision", self.model_revision),
            ("dataset", self.dataset),
            ("dataset_revision", self.dataset_revision),
            ("dataset_config", self.dataset_config),
            ("split", self.split),
            ("question_field", self.question_field),
            ("device", self.device),
            ("dtype", self.dtype),
        )
        for field, value in string_fields:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if self.protocol_scope != ADAPTATION_SCOPE:
            raise ValueError(
                "this implementation only supports protocol_scope='adaptation'; "
                "a faithful UCC-Inj reproduction requires the pinned Qwen/data artifacts"
            )
        _validate_huggingface_repo_id(field="model", value=self.model)
        _validate_huggingface_repo_id(field="dataset", value=self.dataset)
        if not _is_full_commit_sha(self.model_revision):
            raise ValueError("model_revision must be a full immutable 40-character commit SHA")
        if not _is_full_commit_sha(self.dataset_revision):
            raise ValueError("dataset_revision must be a full immutable 40-character commit SHA")
        if (
            not isinstance(self.noise_levels, tuple)
            or not self.noise_levels
            or any(
                not isinstance(level, int) or isinstance(level, bool) or level < 0
                for level in self.noise_levels
            )
        ):
            raise ValueError("noise_levels must contain one or more non-negative integers")
        if 0 not in self.noise_levels:
            raise ValueError("noise_levels must include the independently executed level-0 control")
        if len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("noise_levels must not contain duplicates")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.limit is not None and (
            not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer when supplied")
        if (
            not isinstance(self.max_length, int)
            or isinstance(self.max_length, bool)
            or self.max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer")
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code must remain disabled for this protocol adaptation")
        if not isinstance(self.add_generation_prompt, bool):
            raise ValueError("add_generation_prompt must be a boolean")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be one of: bfloat16, float16, float32")


@dataclass(frozen=True)
class ExtractedStates:
    hidden_states: np.ndarray
    input_ids: tuple[int, ...]
    input_token_count: int
    token_index: int
    terminal_token_id: int
    input_ids_sha256: str
    template_suffix_sha256: str
    logical_position: str = COMPARISON_POSITION
    truncated: bool = False


@dataclass(frozen=True)
class _QuestionCohort:
    questions: tuple[str, ...]
    provenance: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_ordered_texts(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256_token_ids(values: Sequence[int]) -> str:
    array = np.asarray(tuple(values), dtype="<i8")
    return _sha256_bytes(array.tobytes(order="C"))


def _tokenizer_vocab_sha256(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.get_vocab() must return a non-empty mapping")
    if any(not isinstance(token, str) or not isinstance(token_id, int) for token, token_id in vocab.items()):
        raise ValueError("tokenizer vocabulary must map strings to integer IDs")
    serialised = json.dumps(
        vocab,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(serialised)


def _template_suffix_sha256(
    tokenizer: Any,
    content: str,
    *,
    add_generation_prompt: bool,
) -> str:
    """Hash the rendered suffix without assuming templates preserve content verbatim."""
    boundary = f"__UCC_INJ_BOUNDARY_{_sha256_text(content)[:24]}__"
    if boundary in content:
        raise ValueError("content unexpectedly contains the template boundary marker")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content + boundary}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer chat template must render to text")
    first = rendered.find(boundary)
    if first < 0:
        raise ValueError("chat template did not preserve the boundary marker")
    if rendered.find(boundary, first + len(boundary)) >= 0:
        raise ValueError("chat template rendered the boundary marker more than once")
    return _sha256_text(rendered[first + len(boundary) :])


def stable_example_seed(*, root_seed: int, example_index: int, noise_level: int) -> int:
    """Derive a cross-process-stable seed; Python's hash is intentionally avoided."""
    payload = f"ucc-inj-exp6/v1:{root_seed}:{example_index}:{noise_level}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def last_non_padding_index(attention_mask: Any) -> int:
    """Return the final valid token index for a single-example attention mask."""
    values = np.asarray(attention_mask, dtype=np.int64).reshape(-1)
    valid = np.flatnonzero(values)
    if valid.size == 0:
        raise ValueError("attention mask has no non-padding tokens")
    return int(valid[-1])


def common_suffix_token_count(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of identical terminal token IDs in two complete inputs."""
    count = 0
    for left_value, right_value in zip(reversed(left), reversed(right), strict=False):
        if left_value != right_value:
            break
        count += 1
    return count


def cosine_per_layer(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """Compute finite cosine similarities for matching layer-by-hidden arrays."""
    clean_values = np.asarray(clean, dtype=np.float64)
    noisy_values = np.asarray(noisy, dtype=np.float64)
    if clean_values.shape != noisy_values.shape or clean_values.ndim != 2:
        raise ValueError("clean and noisy states must both have shape [layers, hidden]")
    if not np.all(np.isfinite(clean_values)) or not np.all(np.isfinite(noisy_values)):
        raise ValueError("hidden states contain non-finite values")
    denominator = np.linalg.norm(clean_values, axis=1) * np.linalg.norm(noisy_values, axis=1)
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise ValueError("cosine similarity is undefined for zero or non-finite norms")
    result = np.sum(clean_values * noisy_values, axis=1) / denominator
    if not np.all(np.isfinite(result)):
        raise ValueError("cosine similarity produced non-finite values")
    return result


def cosine_for_extracted_pair(clean: ExtractedStates, noisy: ExtractedStates) -> np.ndarray:
    """Compare the same logical endpoint of two complete, untruncated inputs."""
    if clean.truncated or noisy.truncated:
        raise ValueError("truncated inputs are not valid for the exp6 comparison")
    if (
        clean.logical_position != COMPARISON_POSITION
        or noisy.logical_position != COMPARISON_POSITION
    ):
        raise ValueError(
            "clean and noisy states must target the same logical position: "
            f"{COMPARISON_POSITION}"
        )
    if clean.template_suffix_sha256 != noisy.template_suffix_sha256:
        raise ValueError("clean and noisy chat-template suffixes differ")
    if clean.terminal_token_id != noisy.terminal_token_id:
        raise ValueError("clean and noisy complete inputs end at different terminal token IDs")
    if common_suffix_token_count(clean.input_ids, noisy.input_ids) == 0:
        raise ValueError("clean and noisy complete inputs have no shared terminal token suffix")
    return cosine_per_layer(clean.hidden_states, noisy.hidden_states)


def summarise_layer_cosines(records: Iterable[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Aggregate records into one row per noise level and hidden-state index."""
    grouped: dict[tuple[int, int], list[float]] = {}
    for record in records:
        level = int(record["noise_level"])
        for hidden_state_index, value in enumerate(record["cosine_similarity"]):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("records contain a non-finite cosine similarity")
            grouped.setdefault((level, hidden_state_index), []).append(numeric)
    return [
        {
            "noise_level": level,
            "hidden_state_index": hidden_state_index,
            "n": len(values),
            "mean_cosine": float(np.mean(values)),
            "median_cosine": float(np.median(values)),
            "std_cosine": float(np.std(values, ddof=0)),
        }
        for (level, hidden_state_index), values in sorted(grouped.items())
    ]


class HiddenStateExtractor:
    """Extract endpoint hidden states from complete prompts without truncation."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        config: Exp6Config,
        torch_module: Any,
        resolved_snapshot_commit: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.torch = torch_module
        self.resolved_snapshot_commit = resolved_snapshot_commit

    def states(self, text: str) -> ExtractedStates:
        messages = [{"role": "user", "content": text}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.config.add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
            truncation=False,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("tokenizer output must include input_ids and attention_mask")
        input_values = encoded["input_ids"][0].detach().cpu().numpy().reshape(-1)
        mask_values = encoded["attention_mask"][0].detach().cpu().numpy().reshape(-1)
        if input_values.size != mask_values.size:
            raise ValueError("input_ids and attention_mask lengths differ")
        valid_positions = np.flatnonzero(mask_values)
        if valid_positions.size == 0:
            raise ValueError("attention mask has no non-padding tokens")
        input_ids = tuple(int(input_values[position]) for position in valid_positions)
        if len(input_ids) > self.config.max_length:
            raise InputTooLongError(
                f"complete model input has {len(input_ids)} tokens, "
                f"exceeding max_length={self.config.max_length}; refusing to truncate"
            )
        index = last_non_padding_index(mask_values)
        device_encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model(**device_encoded, output_hidden_states=True, use_cache=False)
        hidden_states = np.stack(
            [hidden[0, index, :].float().cpu().numpy() for hidden in output.hidden_states],
            axis=0,
        )
        return ExtractedStates(
            hidden_states=hidden_states,
            input_ids=input_ids,
            input_token_count=len(input_ids),
            token_index=index,
            terminal_token_id=input_ids[-1],
            input_ids_sha256=_sha256_token_ids(input_ids),
            template_suffix_sha256=_template_suffix_sha256(
                self.tokenizer,
                text,
                add_generation_prompt=self.config.add_generation_prompt,
            ),
        )

    def runtime_provenance(self) -> dict[str, Any]:
        model_config = getattr(self.model, "config", None)
        config_dict = model_config.to_dict() if hasattr(model_config, "to_dict") else {}
        cuda = getattr(self.torch, "cuda", None)
        gpu_name: str | None = None
        if self.config.device.startswith("cuda") and cuda is not None and cuda.is_available():
            gpu_name = str(cuda.get_device_name(self.config.device))
        return {
            "model": {
                "requested_id": self.config.model,
                "requested_revision": self.config.model_revision,
                "resolved_name_or_path": str(getattr(model_config, "_name_or_path", self.config.model)),
                "resolved_commit": self.resolved_snapshot_commit,
                "snapshot_binding": (
                    "verified_huggingface_snapshot_directory"
                    if self.resolved_snapshot_commit is not None
                    else "unverified_injected_runtime"
                ),
                "loader_reported_commit": getattr(model_config, "_commit_hash", None),
                "class": type(self.model).__name__,
                "config_sha256": _sha256_text(
                    json.dumps(config_dict, ensure_ascii=False, sort_keys=True, allow_nan=False)
                ),
            },
            "tokenizer": {
                "requested_id": self.config.model,
                "requested_revision": self.config.model_revision,
                "resolved_name_or_path": str(getattr(self.tokenizer, "name_or_path", self.config.model)),
                "resolved_commit": self.resolved_snapshot_commit,
                "snapshot_binding": (
                    "verified_huggingface_snapshot_directory"
                    if self.resolved_snapshot_commit is not None
                    else "unverified_injected_runtime"
                ),
                "class": type(self.tokenizer).__name__,
                "vocab_sha256": _tokenizer_vocab_sha256(self.tokenizer),
                "chat_template_sha256": _sha256_text(
                    str(getattr(self.tokenizer, "chat_template", ""))
                ),
                "vocab_size": getattr(self.tokenizer, "vocab_size", None),
                "padding_side": getattr(self.tokenizer, "padding_side", None),
                "truncation_side": getattr(self.tokenizer, "truncation_side", None),
                "model_max_length": getattr(self.tokenizer, "model_max_length", None),
                "add_generation_prompt": self.config.add_generation_prompt,
                "truncation_policy": "reject",
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": str(getattr(self.torch, "__version__", "unknown")),
                "transformers": importlib.metadata.version("transformers"),
                "datasets": importlib.metadata.version("datasets"),
                "cuda": str(getattr(getattr(self.torch, "version", None), "cuda", None)),
                "gpu": gpu_name,
            },
        }


def _load_runtime(config: Exp6Config) -> HiddenStateExtractor:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config.validate()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    snapshot_path = Path(
        snapshot_download(repo_id=config.model, revision=config.model_revision)
    ).resolve()
    resolved_commit = snapshot_path.name.lower()
    if resolved_commit != config.model_revision.lower():
        raise RuntimeError(
            "Hugging Face snapshot resolved a different commit: "
            f"requested={config.model_revision}, resolved={resolved_commit}"
        )

    dtype = getattr(torch, config.dtype)
    local_arguments = {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path), trust_remote_code=False, **local_arguments
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_path),
        dtype=dtype,
        trust_remote_code=False,
        **local_arguments,
    ).to(config.device).eval()
    loader_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    if loader_commit is not None and loader_commit.lower() != resolved_commit:
        raise RuntimeError(
            "model loader reported a different commit from the verified snapshot: "
            f"snapshot={resolved_commit}, loader={loader_commit}"
        )
    return HiddenStateExtractor(
        tokenizer=tokenizer,
        model=model,
        config=config,
        torch_module=torch,
        resolved_snapshot_commit=resolved_commit,
    )


def _load_question_cohort(config: Exp6Config) -> _QuestionCohort:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    config.validate()
    snapshot_path = Path(
        snapshot_download(
            repo_id=config.dataset,
            repo_type="dataset",
            revision=config.dataset_revision,
        )
    ).resolve()
    resolved_commit = snapshot_path.name.lower()
    if resolved_commit != config.dataset_revision.lower():
        raise RuntimeError(
            "Hugging Face dataset snapshot resolved a different commit: "
            f"requested={config.dataset_revision}, resolved={resolved_commit}"
        )

    dataset = load_dataset(
        str(snapshot_path),
        config.dataset_config,
        split=config.split,
    )
    available_rows = len(dataset)
    rows = dataset if config.limit is None else dataset.select(range(min(config.limit, available_rows)))
    questions = tuple(str(row[config.question_field]) for row in rows)
    info = getattr(dataset, "info", None)
    return _QuestionCohort(
        questions=questions,
        provenance={
            "requested_id": config.dataset,
            "requested_config": config.dataset_config,
            "requested_split": config.split,
            "requested_revision": config.dataset_revision,
            "resolved_commit": resolved_commit,
            "snapshot_binding": "verified_huggingface_snapshot_directory",
            "fingerprint": getattr(rows, "_fingerprint", None),
            "builder_name": getattr(info, "builder_name", None),
            "config_name": getattr(info, "config_name", None),
            "version": str(getattr(info, "version", None)),
            "available_num_rows": available_rows,
            "selected_num_rows": len(questions),
            "selected_questions_sha256": _sha256_ordered_texts(questions),
        },
    )


def load_gsm8k_questions(config: Exp6Config) -> list[str]:
    """Load the configured GSM8K cohort."""
    return list(_load_question_cohort(config).questions)


def _implementation_provenance() -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parents[1]
    files = {
        "exp6.py": package_dir / "exp6.py",
        "noise.py": package_dir / "noise.py",
        "cli.py": package_dir / "cli.py",
        "uv.lock": project_dir / "uv.lock",
    }
    return {
        "source_sha256": {
            name: _sha256_bytes(path.read_bytes()) if path.is_file() else None
            for name, path in files.items()
        },
        "git_commit_environment": os.environ.get("GITHUB_SHA")
        or os.environ.get("TYPO_ROBUST_GIT_COMMIT"),
    }


def run_exp6(
    config: Exp6Config,
    *,
    questions: Sequence[str] | None = None,
    extractor: HiddenStateExtractor | None = None,
    progress: Callable[[Iterable[Any]], Iterable[Any]] = lambda values: values,
) -> tuple[list[dict[str, Any]], list[dict[str, float | int]], dict[str, Any]]:
    """Run the adaptation and return records, summary, and frozen provenance."""
    config.validate()
    if questions is None:
        cohort = _load_question_cohort(config)
    else:
        selected = list(questions)
        if config.limit is not None:
            selected = selected[: config.limit]
        injected = tuple(str(question) for question in selected)
        cohort = _QuestionCohort(
            questions=injected,
            provenance={
                "source": "injected",
                "selected_num_rows": len(injected),
                "selected_questions_sha256": _sha256_ordered_texts(injected),
            },
        )
    if not cohort.questions:
        raise ValueError("question cohort is empty; refusing to emit a successful empty run")
    active_extractor = extractor if extractor is not None else _load_runtime(config)
    runtime_provenance = (
        active_extractor.runtime_provenance()
        if hasattr(active_extractor, "runtime_provenance")
        else {"extractor": type(active_extractor).__name__, "injected": True}
    )
    records: list[dict[str, Any]] = []
    for example_index, question in enumerate(progress(cohort.questions)):
        clean_states = active_extractor.states(question)
        for level in config.noise_levels:
            noise_seed = stable_example_seed(
                root_seed=config.seed, example_index=example_index, noise_level=level
            )
            noisy_question = inject_variation_selector_noise(
                question, noise_level=level, seed=noise_seed
            )
            noisy_states = active_extractor.states(noisy_question)
            if level == 0:
                if noisy_question != question:
                    raise RuntimeError("level-0 control changed the input text")
                if clean_states.input_ids != noisy_states.input_ids:
                    raise RuntimeError("level-0 control changed tokenization")
            else:
                if noisy_question == question:
                    raise RuntimeError("nonzero noise level did not change the input text")
                if clean_states.input_ids == noisy_states.input_ids:
                    raise RuntimeError("nonzero noise was erased before reaching the model")
            similarities = cosine_for_extracted_pair(clean_states, noisy_states)
            if level == 0 and not np.allclose(similarities, 1.0, rtol=1e-6, atol=1e-6):
                raise RuntimeError("level-0 control did not produce unit cosine similarity")
            records.append(
                {
                    "schema_version": "ucc-inj-exp6/v2",
                    "protocol_scope": config.protocol_scope,
                    "upstream_commit": UPSTREAM_COMMIT,
                    "example_index": example_index,
                    "noise_level": level,
                    "noise_seed": noise_seed,
                    "comparison_position": COMPARISON_POSITION,
                    "clean_text_sha256": _sha256_text(question),
                    "noisy_text_sha256": _sha256_text(noisy_question),
                    "clean_input_tokens": clean_states.input_token_count,
                    "noisy_input_tokens": noisy_states.input_token_count,
                    "clean_token_index": clean_states.token_index,
                    "noisy_token_index": noisy_states.token_index,
                    "clean_terminal_token_id": clean_states.terminal_token_id,
                    "noisy_terminal_token_id": noisy_states.terminal_token_id,
                    "clean_input_ids_sha256": clean_states.input_ids_sha256,
                    "noisy_input_ids_sha256": noisy_states.input_ids_sha256,
                    "template_suffix_sha256": clean_states.template_suffix_sha256,
                    "common_terminal_suffix_tokens": common_suffix_token_count(
                        clean_states.input_ids, noisy_states.input_ids
                    ),
                    "truncated": False,
                    "cosine_similarity": [float(value) for value in similarities],
                }
            )
    expected_records = len(cohort.questions) * len(config.noise_levels)
    if len(records) != expected_records:
        raise RuntimeError(
            f"incomplete run: expected {expected_records} records, produced {len(records)}"
        )
    level_zero_records = sum(int(record["noise_level"]) == 0 for record in records)
    if level_zero_records != len(cohort.questions):
        raise RuntimeError("level-0 control count does not match the question cohort")
    provenance = {
        "schema_version": "ucc-inj-exp6-provenance/v2",
        "protocol": {
            "scope": config.protocol_scope,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_model": UPSTREAM_MODEL,
            "upstream_experiment_script": UPSTREAM_EXPERIMENT_SCRIPT,
            "upstream_noise_encoder": UPSTREAM_NOISE_ENCODER,
            "adaptation_model": config.model,
        },
        "resolved": {
            **runtime_provenance,
            "dataset": cohort.provenance,
            "implementation": _implementation_provenance(),
        },
    }
    return records, summarise_layer_cosines(records), provenance


def write_exp6_results(
    *,
    output_dir: Path,
    config: Exp6Config,
    records: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, float | int]],
    provenance: dict[str, Any],
    reserved_output_dir: bool = False,
) -> None:
    """Write a complete, strict-JSON result directory without overwriting runs."""
    if not records:
        raise ValueError("refusing to write an empty scientific result")
    config_payload = json.dumps(
        asdict(config), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    provenance_payload = json.dumps(
        provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    records_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    summary_payload = json.dumps(
        list(summary), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if reserved_output_dir:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise FileNotFoundError("reserved output directory is no longer a real directory")
        if any(output_dir.iterdir()):
            raise FileExistsError("reserved output directory is no longer empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(config_payload, encoding="utf-8")
    (output_dir / "provenance.json").write_text(provenance_payload, encoding="utf-8")
    (output_dir / "per_example.jsonl").write_text(records_payload, encoding="utf-8")
    (output_dir / "layer_summary.json").write_text(summary_payload, encoding="utf-8")
