from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import pytest
from typo_cot.models.tokenizer_attestation import (
    TOKENIZER_ASSET_FILENAMES,
    TokenizerAssetAttestation,
    TokenizerSnapshotAttestation,
)

from typo_robust_training.training.config import (
    is_kojima_faithful_protocol,
    is_mistral_factorial_protocol,
    load_adapter_training_config,
)
from typo_robust_training.training.kojima_faithful import (
    MAX_SEQUENCE_LENGTH,
    MISTRAL_REVISION,
    PrepareKojimaFaithfulDataConfig,
    prepare_kojima_faithful_data,
)
from typo_robust_training.training.methods import (
    PROBE_FACTORIAL_CONDITIONS,
    ProbeTransitionTrainingEvidence,
    materialize_probe_output_factorial_configs,
)
from typo_robust_training.training.mistral_factorial import (
    FACTORIAL_EDIT_COUNTS,
    FACTORIAL_METHOD_IDENTITY,
    FACTORIAL_OPERATIONS,
    PrepareMistralFactorialDataConfig,
    load_mistral_factorial_data_bundle,
    prepare_mistral_factorial_data,
)
from typo_robust_training.training.runner import (
    TrainingMicroStepResult,
    validate_micro_step_student_tokens,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "configs/proposals/mistral7b-v01-probe-output-factorial-64m.template.yaml"


class _Provider:
    source_file_sha256 = "a" * 64

    def texts(self) -> list[str]:
        return [f"document{i} " + ("alpha beta gamma delta " * 700) for i in range(40)]


class _FastCharacterTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    is_fast = True
    init_kwargs = {"_commit_hash": MISTRAL_REVISION}

    def __init__(self) -> None:
        alphabet = ["<s>", "</s>"] + [chr(value) for value in range(1, 128)]
        self._token_to_id = {token: index for index, token in enumerate(alphabet)}
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.bos_token, cursor):
                tokens.append(self.bos_token)
                cursor += len(self.bos_token)
            elif text.startswith(self.eos_token, cursor):
                tokens.append(self.eos_token)
                cursor += len(self.eos_token)
            else:
                tokens.append(text[cursor])
                cursor += 1
        return tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._token_to_id[token] for token in tokens]

    def decode(self, ids: list[int] | tuple[int, ...], *, add_special_tokens: bool = False) -> str:
        del add_special_tokens
        return "".join(self._id_to_token[int(value)] for value in ids)

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        maximum = int(kwargs.get("max_length", MAX_SEQUENCE_LENGTH))
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.bos_token, cursor):
                tokens.append(self.bos_token)
                offsets.append((0, 0))
                cursor += len(self.bos_token)
            else:
                tokens.append(text[cursor])
                offsets.append((cursor, cursor + 1))
                cursor += 1
        ids = self.convert_tokens_to_ids(tokens)[:maximum]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets[:maximum],
        }


def _tokenizer_attestation() -> TokenizerSnapshotAttestation:
    return TokenizerSnapshotAttestation(
        model_name="mistralai/Mistral-7B-v0.1",
        requested_revision=MISTRAL_REVISION,
        observed_commit=MISTRAL_REVISION,
        assets=tuple(
            TokenizerAssetAttestation(
                filename,
                filename == "tokenizer_config.json",
                "d" * 64 if filename == "tokenizer_config.json" else None,
            )
            for filename in TOKENIZER_ASSET_FILENAMES
        ),
        tokenizer_fingerprint_sha256="e" * 64,
        transformers_version=importlib.metadata.version("transformers"),
        tokenizers_version=importlib.metadata.version("tokenizers"),
        source_manifest_sha256="f" * 64,
    )


def _patch_attested_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: _FastCharacterTokenizer,
) -> None:
    attestation = _tokenizer_attestation()
    monkeypatch.setattr(
        "typo_robust_training.training.mistral_factorial._load_attested_factorial_tokenizer",
        lambda: (tokenizer, attestation.provenance_dict()),
    )


def _evidence() -> ProbeTransitionTrainingEvidence:
    return ProbeTransitionTrainingEvidence(
        model="mistralai/Mistral-7B-v0.1",
        model_revision=MISTRAL_REVISION,
        decoder_layers=32,
        selected_transition_layer=9,
        evidence_sha256="b" * 64,
    )


def _materialize_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_training_evidence",
        lambda *_args, **_kwargs: _evidence(),
    )
    evidence = tmp_path / "probe.json"
    evidence.write_text("fixture", encoding="utf-8")
    return dict(
        materialize_probe_output_factorial_configs(
            TEMPLATE,
            evidence_path=evidence,
            output_dir=tmp_path / "arms",
        )
    )


def _rehash_factorial(root: Path) -> None:
    manifest_path = root / "manifest.json"
    run_path = root / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in tuple(manifest["artifacts"]):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        manifest["artifacts"][name]["sha256"] = digest
        if name.startswith("packed_source/"):
            manifest["parent"]["artifacts"][name] = digest
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["manifest_sha256"] = manifest_sha
    run["outputs"]["manifest.json"]["sha256"] = manifest_sha
    for name in tuple(manifest["artifacts"]):
        run["outputs"][name]["sha256"] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run, sort_keys=True, indent=2) + "\n")


def _prepare_tiny_packed_source(
    root: Path,
    *,
    seed: int,
    tokenizer: _FastCharacterTokenizer,
) -> None:
    prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(
            seed=seed,
            output_dir=root,
            packed_examples=4,
        ),
        provider=_Provider(),
        tokenizer=tokenizer,
    )


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_mistral_factorial_materializer_freezes_exact_64m_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocols = _materialize_configs(tmp_path, monkeypatch)

    assert tuple(protocols) == PROBE_FACTORIAL_CONDITIONS
    for protocol in protocols.values():
        assert protocol.schema_version == "robustness-adapter-training-config/v8"
        assert protocol.method_identity == FACTORIAL_METHOD_IDENTITY
        assert protocol.model_revision == MISTRAL_REVISION
        assert protocol.max_sequence_length == 8192
        assert protocol.gradient_accumulation_steps == 8
        assert protocol.max_optimizer_steps == 1000
        assert protocol.max_student_tokens == 65_536_000
        assert protocol.seed_inventory == (42, 43, 44)
        assert protocol.lora_target_modules == (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        assert "embed_tokens" not in protocol.lora_target_modules
        assert "lm_head" not in protocol.lora_target_modules
        assert protocol.loss_weights == {
            "noisy_language_model": 0.0,
            "answer": 0.0,
            "output": 1.0,
            "state": 0.0,
            "clean": 0.0,
        }
        assert is_mistral_factorial_protocol(protocol)
        assert not is_kojima_faithful_protocol(protocol)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "revision", "c" * 40),
        ("sequence", "max_length", 4096),
        ("sequence", "training_corpus_revision", "c" * 40),
        ("optimization", "max_student_tokens", 65_535_999),
        ("optimization", "gradient_accumulation_steps", 4),
        ("objective", "state_scope", "some-state"),
        (
            "objective",
            "weights",
            {
                "noisy_language_model": 0,
                "answer": 0,
                "output": 1,
                "state": 1,
                "clean": 0,
            },
        ),
        ("adapter", "target_modules", ["q_proj"]),
    ],
)
def test_mistral_factorial_config_rejects_provenance_loss_and_budget_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: object,
) -> None:
    _materialize_configs(tmp_path, monkeypatch)
    path = tmp_path / "arms" / "factorial-all-layers-all-tokens.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[section][field] = value
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError):
        load_adapter_training_config(path)


def test_schema_number_cannot_route_factorial_into_kojima_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocols = _materialize_configs(tmp_path, monkeypatch)
    protocol = protocols["factorial-all-layers-all-tokens"]

    assert is_mistral_factorial_protocol(protocol)
    assert not is_kojima_faithful_protocol(protocol)
    assert not is_mistral_factorial_protocol(
        replace(protocol, condition="kojima-faithful-output-matching")
    )


def test_mistral_factorial_method_identity_cannot_be_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _materialize_configs(tmp_path, monkeypatch)
    path = tmp_path / "arms" / "factorial-all-layers-all-tokens.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["method_identity"] = "probe-output-factorial/v1"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="method_identity"):
        load_adapter_training_config(path)


def test_factorial_data_freezes_prevalidated_alternating_pairs_and_three_ops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FastCharacterTokenizer()
    _patch_attested_tokenizer(monkeypatch, tokenizer)
    packed = tmp_path / "packed"
    prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(
            seed=42,
            output_dir=packed,
            packed_examples=4,
        ),
        provider=_Provider(),
        tokenizer=tokenizer,
    )
    output = tmp_path / "factorial"
    result = prepare_mistral_factorial_data(
        PrepareMistralFactorialDataConfig(
            seed=42,
            packed_source_dir=packed,
            output_dir=output,
            target_usable_examples=2,
        ),
    )
    bundle = load_mistral_factorial_data_bundle(output, seed=42)

    assert result.student_tokens == 2 * MAX_SEQUENCE_LENGTH
    assert [pair.is_noop for pair in bundle.pairs] == [True, False]
    assert bundle.pairs[0].clean_text == bundle.pairs[0].typo_text
    assert bundle.pairs[1].clean_text != bundle.pairs[1].typo_text
    assert all(edit.operation in FACTORIAL_OPERATIONS for edit in bundle.pairs[1].edits)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["noise"]["operations"] == dict(FACTORIAL_OPERATIONS)
    assert manifest["noise"]["edit_count_distribution"] == dict(FACTORIAL_EDIT_COUNTS)
    assert manifest["pairing"]["realized_clean_examples"] == 1
    assert manifest["pairing"]["realized_noisy_examples"] == 1
    assert sum(manifest["noise"]["realized_operation_counts"].values()) == len(
        bundle.pairs[1].edits
    )
    assert sum(manifest["noise"]["realized_edit_count_counts"].values()) == 1
    assert bundle.tokenizer_snapshot_attestation == _tokenizer_attestation().provenance_dict()
    assert manifest["packing"]["target_student_tokens"] == 2 * MAX_SEQUENCE_LENGTH


def test_factorial_data_rejects_rehashed_pair_or_skip_desynchronization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FastCharacterTokenizer()
    _patch_attested_tokenizer(monkeypatch, tokenizer)
    packed = tmp_path / "packed"
    prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(seed=43, output_dir=packed, packed_examples=4),
        provider=_Provider(),
        tokenizer=tokenizer,
    )
    output = tmp_path / "factorial"
    prepare_mistral_factorial_data(
        PrepareMistralFactorialDataConfig(
            seed=43,
            packed_source_dir=packed,
            output_dir=output,
            target_usable_examples=2,
        ),
    )
    pair_path = output / "pairs.jsonl"
    rows = [json.loads(line) for line in pair_path.read_text().splitlines()]
    rows[1]["attempt_index"] = rows[0]["attempt_index"]
    unsigned = {key: value for key, value in rows[1].items() if key != "pair_sha256"}
    rows[1]["pair_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    pair_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    )
    _rehash_factorial(output)

    with pytest.raises(ValueError, match="pair identity|attempt order|skip/replacement"):
        load_mistral_factorial_data_bundle(output, seed=43)


def test_factorial_data_is_byte_identical_when_reprepared_for_the_same_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FastCharacterTokenizer()
    _patch_attested_tokenizer(monkeypatch, tokenizer)
    packed = tmp_path / "packed"
    _prepare_tiny_packed_source(packed, seed=44, tokenizer=tokenizer)

    for name in ("arm-a", "arm-b"):
        prepare_mistral_factorial_data(
            PrepareMistralFactorialDataConfig(
                seed=44,
                packed_source_dir=packed,
                output_dir=tmp_path / name,
                target_usable_examples=2,
            )
        )

    assert _artifact_bytes(tmp_path / "arm-a") == _artifact_bytes(tmp_path / "arm-b")


def test_factorial_loader_rejects_parent_and_attestation_rehash_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FastCharacterTokenizer()
    _patch_attested_tokenizer(monkeypatch, tokenizer)
    packed = tmp_path / "packed"
    _prepare_tiny_packed_source(packed, seed=42, tokenizer=tokenizer)
    parent_attack = tmp_path / "parent-attack"
    attestation_attack = tmp_path / "attestation-attack"
    for output in (parent_attack, attestation_attack):
        prepare_mistral_factorial_data(
            PrepareMistralFactorialDataConfig(
                seed=42,
                packed_source_dir=packed,
                output_dir=output,
                target_usable_examples=2,
            )
        )

    copied_run = parent_attack / "packed_source" / "run.json"
    copied_payload = json.loads(copied_run.read_text(encoding="utf-8"))
    copied_payload["seed"] = 43
    copied_run.write_text(json.dumps(copied_payload, sort_keys=True, indent=2) + "\n")
    _rehash_factorial(parent_attack)
    with pytest.raises(ValueError, match="Kojima data run/seed identity"):
        load_mistral_factorial_data_bundle(parent_attack, seed=42)

    manifest_path = attestation_attack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["tokenizer"]["snapshot_attestation"]
    provenance["observed_commit"] = "c" * 40
    payload = {
        key: value
        for key, value in provenance.items()
        if key not in {"attestation_sha256", "manifest_file_sha256"}
    }
    provenance["attestation_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    _rehash_factorial(attestation_attack)
    with pytest.raises(ValueError, match="attestation provenance identity"):
        load_mistral_factorial_data_bundle(attestation_attack, seed=42)


def test_factorial_loader_rejects_rehashed_invalid_skip_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FastCharacterTokenizer()
    _patch_attested_tokenizer(monkeypatch, tokenizer)
    packed = tmp_path / "packed"
    _prepare_tiny_packed_source(packed, seed=43, tokenizer=tokenizer)
    from typo_robust_training.training import mistral_factorial as module

    real_encode = module.encode_training_pair
    calls = 0

    def reject_first(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("frozen fixture rejection")
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(module, "encode_training_pair", reject_first)
    output = tmp_path / "factorial"
    prepare_mistral_factorial_data(
        PrepareMistralFactorialDataConfig(
            seed=43,
            packed_source_dir=packed,
            output_dir=output,
            target_usable_examples=2,
        )
    )
    skip_path = output / "skips.jsonl"
    rows = [json.loads(line) for line in skip_path.read_text().splitlines()]
    assert len(rows) == 1
    rows[0]["intended_is_noop"] = not rows[0]["intended_is_noop"]
    skip_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    )
    _rehash_factorial(output)

    with pytest.raises(ValueError, match="skip ledger differs from its source attempt"):
        load_mistral_factorial_data_bundle(output, seed=43)


def test_factorial_preparation_is_atomic_and_rejects_symlink_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FastCharacterTokenizer()
    packed = tmp_path / "packed"
    _prepare_tiny_packed_source(packed, seed=44, tokenizer=tokenizer)
    linked = tmp_path / "packed-link"
    linked.symlink_to(packed, target_is_directory=True)
    with pytest.raises(ValueError, match="root cannot be a symlink"):
        prepare_mistral_factorial_data(
            PrepareMistralFactorialDataConfig(
                seed=44,
                packed_source_dir=linked,
                output_dir=tmp_path / "linked-output",
                target_usable_examples=2,
            )
        )

    in_tree_link = packed / "unregistered-link"
    in_tree_link.symlink_to(packed / "manifest.json")
    with pytest.raises(ValueError, match="tree contains a symlink"):
        prepare_mistral_factorial_data(
            PrepareMistralFactorialDataConfig(
                seed=44,
                packed_source_dir=packed,
                output_dir=tmp_path / "tree-link-output",
                target_usable_examples=2,
            )
        )
    in_tree_link.unlink()

    class _ShortTokenizer(_FastCharacterTokenizer):
        def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
            kwargs["max_length"] = MAX_SEQUENCE_LENGTH - 1
            return super().__call__(text, **kwargs)

    _patch_attested_tokenizer(monkeypatch, _ShortTokenizer())
    output = tmp_path / "failed-output"
    with pytest.raises(ValueError, match="exhausted"):
        prepare_mistral_factorial_data(
            PrepareMistralFactorialDataConfig(
                seed=44,
                packed_source_dir=packed,
                output_dir=output,
                target_usable_examples=2,
            )
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed-output.tmp-*"))


def test_mistral_factorial_microstep_must_fill_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _materialize_configs(tmp_path, monkeypatch)["factorial-all-layers-all-tokens"]
    with pytest.raises(ValueError, match="fill the 8192-token context"):
        validate_micro_step_student_tokens(
            protocol,
            TrainingMicroStepResult(
                total_loss=1.0,
                losses={"output": 1.0},
                student_tokens=8191,
            ),
        )
