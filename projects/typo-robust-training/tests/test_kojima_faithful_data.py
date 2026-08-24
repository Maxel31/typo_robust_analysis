"""Adversarial tests for the real Kojima packing/noise/alignment path."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from typo_robust_training.data.records import TypoEdit
from typo_robust_training.training.kojima_faithful import (
    FINEWEB_REVISION,
    KojimaFaithfulDataBundle,
    KojimaFaithfulNoiseGenerator,
    MAX_SEQUENCE_LENGTH,
    MISTRAL_REVISION,
    PACKED_EXAMPLES,
    TARGET_USABLE_EXAMPLES,
    UnusableKojimaFaithfulPairError,
    PrepareKojimaFaithfulDataConfig,
    _noise_word,
    encode_kojima_faithful_pair,
    load_kojima_faithful_data_bundle,
    prepare_kojima_faithful_data,
)
from typo_robust_training.training.checkpoint import TrainingCursor
from typo_robust_training.training.losses import aligned_soft_cross_entropy
from typo_robust_training.training.pairs import TrainingPair, TrainingSource
from typo_robust_training.training.runner import _next_usable_training_pair


class CharacterTokenizer:
    """A reversible tokenizer with one multi-character BOS token."""

    bos_token = "<s>"
    eos_token = "</s>"
    init_kwargs = {"_commit_hash": MISTRAL_REVISION}

    def __init__(self) -> None:
        alphabet = ["<s>", "</s>"] + [chr(value) for value in range(1, 128)]
        self._token_to_id = {token: index for index, token in enumerate(alphabet)}
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        cursor = 0
        for special in (self.bos_token, self.eos_token):
            if special not in self._token_to_id:
                raise AssertionError
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


class StaticProvider:
    source_file_sha256 = "a" * 64

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def texts(self) -> list[str]:
        return self._texts


def _long_documents(count: int = 32) -> list[str]:
    return [f"document{i} " + ("alpha beta gamma " * 650) for i in range(count)]


def _rehash_artifacts(root: Path) -> None:
    packed = root / "packed_sources.jsonl"
    manifest_path = root / "manifest.json"
    run_path = root / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed_sha = hashlib.sha256(packed.read_bytes()).hexdigest()
    manifest["artifacts"]["packed_sources.jsonl"]["sha256"] = packed_sha
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["manifest_sha256"] = manifest_sha
    run["outputs"]["manifest.json"]["sha256"] = manifest_sha
    run["outputs"]["packed_sources.jsonl"]["sha256"] = packed_sha
    run_path.write_text(json.dumps(run, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def test_production_inventory_has_upstream_attempt_headroom() -> None:
    assert PACKED_EXAMPLES == 8_800
    assert TARGET_USABLE_EXAMPLES == 8_000


def test_builder_freezes_pinned_source_packing_and_nbsp_canonicalization(
    tmp_path: Path,
) -> None:
    tokenizer = CharacterTokenizer()
    documents = _long_documents()
    documents[3] += "\xa0"
    result = prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(
            seed=1,
            output_dir=tmp_path / "data",
            packed_examples=2,
        ),
        provider=StaticProvider(documents),
        tokenizer=tokenizer,
    )
    assert result.student_tokens == 2 * MAX_SEQUENCE_LENGTH
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["revision"] == FINEWEB_REVISION
    assert manifest["dataset"]["data_file_sha256"] == "a" * 64
    assert manifest["tokenizer"]["revision"] == MISTRAL_REVISION
    assert manifest["packing"]["overfill_tokens"] == 500
    assert manifest["packing"]["upstream_validation_documents_skipped"] == 20
    assert (
        manifest["reproduction_departure"]["row_order"]["upstream_unpinned_dependency"]
        == "torch"
    )
    assert manifest["packing"]["packed_attempts"] == 2
    assert manifest["packing"]["target_usable_examples"] == 2
    assert "\xa0" not in result.packed_sources_path.read_text(encoding="utf-8")


def test_loader_rejects_rehashed_source_revision_tampering(tmp_path: Path) -> None:
    root = tmp_path / "data"
    prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(seed=42, output_dir=root, packed_examples=2),
        provider=StaticProvider(_long_documents()),
        tokenizer=CharacterTokenizer(),
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["revision"] = "b" * 40
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _rehash_artifacts(root)
    with pytest.raises(ValueError, match="source/packing/noise identity"):
        load_kojima_faithful_data_bundle(root, seed=42)


def test_loader_rejects_rehashed_packed_row_reordering(tmp_path: Path) -> None:
    root = tmp_path / "data"
    prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(seed=43, output_dir=root, packed_examples=2),
        provider=StaticProvider(_long_documents()),
        tokenizer=CharacterTokenizer(),
    )
    path = root / "packed_sources.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    _rehash_artifacts(root)
    with pytest.raises(ValueError, match="identity/order"):
        load_kojima_faithful_data_bundle(root, seed=43)


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        ("noise", lambda value: value.__setitem__("operation_probability", 0.2)),
        ("runtime_versions", lambda value: value.__setitem__("torch", "unpinned")),
        (
            "reproduction_departure",
            lambda value: value["attempt_boundary"].__setitem__(
                "resolution", "silently-changed"
            ),
        ),
    ],
)
def test_loader_rejects_self_rehashed_closed_world_manifest_tampering(
    tmp_path: Path,
    section: str,
    mutation: object,
) -> None:
    root = tmp_path / section
    prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(seed=42, output_dir=root, packed_examples=2),
        provider=StaticProvider(_long_documents()),
        tokenizer=CharacterTokenizer(),
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(manifest[section])
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _rehash_artifacts(root)
    with pytest.raises(ValueError, match="source/packing/noise identity"):
        load_kojima_faithful_data_bundle(root, seed=42)


def test_loader_recomputes_record_id_after_self_rehashed_row_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    prepare_kojima_faithful_data(
        PrepareKojimaFaithfulDataConfig(seed=44, output_dir=root, packed_examples=2),
        provider=StaticProvider(_long_documents()),
        tokenizer=CharacterTokenizer(),
    )
    packed_path = root / "packed_sources.jsonl"
    rows = [json.loads(line) for line in packed_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["record_id"] = "d" * 64
    packed_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packing"]["source_order_sha256"] = _canonical_sha(
        [
            {
                "record_id": row["record_id"],
                "raw_row_indices": row["raw_row_indices"],
            }
            for row in rows
        ]
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _rehash_artifacts(root)
    with pytest.raises(ValueError, match="record_id is not derivable"):
        load_kojima_faithful_data_bundle(root, seed=44)


def _packed_source(tokenizer: CharacterTokenizer) -> TrainingSource:
    documents = tuple("alpha beta gamma delta\n" * 55 for _ in range(8))
    clean = tokenizer.bos_token + tokenizer.bos_token.join(documents)
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(clean))[:MAX_SEQUENCE_LENGTH]
    assert len(ids) == MAX_SEQUENCE_LENGTH
    token_sha = hashlib.sha256(
        b"".join(int(token).to_bytes(4, "big", signed=False) for token in ids)
    ).hexdigest()
    return TrainingSource(
        kind="clean",
        record_id="c" * 64,
        source="HuggingFaceFW/fineweb",
        source_revision=FINEWEB_REVISION,
        source_split="sample/10BT/000_00000.parquet",
        source_id="packed-000000",
        group_id="c" * 64,
        clean_text=clean,
        typo_text=None,
        task=None,
        answer=None,
        operation=None,
        metadata={
            "kojima_faithful": True,
            "packed_index": 0,
            "clean_prefix_token_ids_sha256": token_sha,
            "bos_token": tokenizer.bos_token,
            "raw_row_indices": (21,),
        },
        token_count=MAX_SEQUENCE_LENGTH,
    )


class _ReplayNoiseGenerator:
    def materialize(self, source: TrainingSource, *, epoch: int) -> TrainingPair:
        draw = random.random()
        return TrainingPair(
            record_id=source.record_id,
            clean_text=source.clean_text,
            typo_text=source.clean_text,
            task=None,
            answer=None,
            metadata={**source.metadata, "rng_draw": draw},
            edits=(),
            is_noop=True,
            epoch=epoch,
        )


def _runner_bundle(tokenizer: CharacterTokenizer, count: int) -> KojimaFaithfulDataBundle:
    template = _packed_source(tokenizer)
    sources = tuple(
        replace(
            template,
            record_id=f"{index + 1:064x}",
            group_id=f"{index + 1:064x}",
            source_id=f"packed-{index:06d}",
        )
        for index in range(count)
    )
    return KojimaFaithfulDataBundle(
        root=Path("/frozen"),
        sources=sources,
        generator=_ReplayNoiseGenerator(),  # type: ignore[arg-type]
        data_identity_sha256="a" * 64,
        training_data_sha256="a" * 64,
        artifact_sha256={},
        seed=42,
        source_file_sha256="b" * 64,
        source_revision=FINEWEB_REVISION,
        source_order_sha256="c" * 64,
        packing_policy="test",
        rng_policy="test",
        packed_attempts=count,
        target_usable_examples=max(1, count - 1),
    )


def test_unusable_attempt_advances_source_not_microstep_and_replays() -> None:
    bundle = _runner_bundle(CharacterTokenizer(), 3)
    first_record_id = bundle.sources[0].record_id

    class Runtime:
        @staticmethod
        def pair_is_usable(pair: TrainingPair) -> bool:
            return pair.record_id != first_record_id

    protocol = SimpleNamespace(condition="kojima-faithful-output-matching")
    cursor = TrainingCursor(0, 0, 7, 2, 16_384)
    random.seed(123)
    rng_state = random.getstate()
    pair, epoch, advanced = _next_usable_training_pair(
        bundle=bundle,
        cursor=cursor,
        seed=42,
        protocol=protocol,  # type: ignore[arg-type]
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    assert pair.record_id == bundle.sources[1].record_id
    assert epoch == 0
    assert advanced.source_index == 2
    assert advanced.micro_steps == cursor.micro_steps + 1
    assert advanced.student_tokens == cursor.student_tokens

    random.setstate(rng_state)
    replay_pair, replay_epoch, replay_cursor = _next_usable_training_pair(
        bundle=bundle,
        cursor=cursor,
        seed=42,
        protocol=protocol,  # type: ignore[arg-type]
        runtime=Runtime(),  # type: ignore[arg-type]
    )
    assert (replay_pair, replay_epoch, replay_cursor) == (pair, epoch, advanced)


def test_unusable_attempt_stream_exhaustion_fails_closed() -> None:
    bundle = _runner_bundle(CharacterTokenizer(), 2)

    class Runtime:
        @staticmethod
        def pair_is_usable(pair: TrainingPair) -> bool:
            del pair
            return False

    with pytest.raises(ValueError, match="exhausted before the next usable pair"):
        _next_usable_training_pair(
            bundle=bundle,
            cursor=TrainingCursor(0, 0, 0, 0, 0),
            seed=42,
            protocol=SimpleNamespace(condition="kojima-faithful-output-matching"),  # type: ignore[arg-type]
            runtime=Runtime(),  # type: ignore[arg-type]
        )


def test_valid_position_cardinality_mismatch_is_skippable_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharacterTokenizer()
    source = _packed_source(tokenizer)
    pair = TrainingPair(
        record_id=source.record_id,
        clean_text=source.clean_text,
        typo_text=source.clean_text,
        task=None,
        answer=None,
        metadata=source.metadata,
        edits=(),
        is_noop=True,
        epoch=0,
    )
    ids = tuple(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source.clean_text))[
            :MAX_SEQUENCE_LENGTH
        ]
    )
    calls = 0

    def mismatched_positions(*args: object, **kwargs: object):
        del args, kwargs
        nonlocal calls
        calls += 1
        positions = (0, 1) if calls == 1 else (0,)
        return ids, positions, len(positions)

    monkeypatch.setattr(
        "typo_robust_training.training.kojima_faithful._valid_target_positions",
        mismatched_positions,
    )
    with pytest.raises(UnusableKojimaFaithfulPairError):
        encode_kojima_faithful_pair(pair, tokenizer=tokenizer)


def test_noise_rng_state_replay_and_encoding_exclude_edited_tokens() -> None:
    tokenizer = CharacterTokenizer()
    source = _packed_source(tokenizer)
    generator = KojimaFaithfulNoiseGenerator(seed=42, bos_token=tokenizer.bos_token)
    random.seed(42)
    state = random.getstate()
    first = generator.materialize(source, epoch=0)
    random.setstate(state)
    replay = generator.materialize(source, epoch=0)
    assert first == replay
    assert first.edits
    start = source.clean_text.index("alpha")
    controlled = replace(
        first,
        typo_text=(
            source.clean_text[:start] + "xlpha" + source.clean_text[start + len("alpha") :]
        ),
        edits=(
            TypoEdit(
                operation="random-replace",
                clean_word="alpha",
                typo_word="xlpha",
                clean_char_span=(start, start + 5),
                typo_char_span=(start, start + 5),
            ),
        ),
        is_noop=False,
    )
    encoding = encode_kojima_faithful_pair(controlled, tokenizer=tokenizer)
    assert encoding.student_tokens == MAX_SEQUENCE_LENGTH
    assert len(encoding.output_logit_pairs) < MAX_SEQUENCE_LENGTH - 1
    assert len({left for left, _right in encoding.output_logit_pairs}) == len(
        encoding.output_logit_pairs
    )
    with pytest.raises(ValueError, match="must not repeat"):
        generator.materialize(source, epoch=1)


@pytest.mark.parametrize(
    ("seed", "expected"),
    [(2, "delete"), (1, "swap"), (5, "addition"), (9, "random-replace")],
)
def test_four_public_operations_are_reachable(seed: int, expected: str) -> None:
    random.seed(seed)
    noisy, operation = _noise_word("alpha")
    assert operation == expected
    assert isinstance(noisy, str) and noisy


def test_encoding_rejects_prepared_token_hash_drift() -> None:
    tokenizer = CharacterTokenizer()
    source = _packed_source(tokenizer)
    generator = KojimaFaithfulNoiseGenerator(seed=44, bos_token=tokenizer.bos_token)
    random.seed(44)
    pair = generator.materialize(source, epoch=0)
    pair = replace(pair, metadata={**pair.metadata, "clean_prefix_token_ids_sha256": "0" * 64})
    with pytest.raises(ValueError, match="prefix token hash"):
        encode_kojima_faithful_pair(pair, tokenizer=tokenizer)


def test_soft_cross_entropy_matches_public_sum_over_global_target_denominator() -> None:
    teacher = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]], requires_grad=True)
    pairs = ((0, 1), (2, 2))
    actual = aligned_soft_cross_entropy(
        teacher,
        student,
        logit_pairs=pairs,
        temperature=1.0,
    )
    teacher_probability = torch.softmax(teacher[0, [0, 2]], dim=-1)
    student_log = torch.log_softmax(student[0, [1, 2]], dim=-1)
    expected = -(teacher_probability * student_log).sum() / len(pairs)
    torch.testing.assert_close(actual, expected)


def test_tokenizer_revision_is_not_inferred_from_model_name(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.init_kwargs = {"_commit_hash": "d" * 40}
    with pytest.raises(ValueError, match="tokenizer revision"):
        prepare_kojima_faithful_data(
            PrepareKojimaFaithfulDataConfig(
                seed=1,
                output_dir=tmp_path / "data",
                packed_examples=1,
            ),
            provider=StaticProvider(_long_documents()),
            tokenizer=tokenizer,
        )
