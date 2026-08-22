from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_probe_transition_artifact import _bundle as _parent_bundle
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.probe import load_probe_transition_artifact
from typo_robust_training.state_gate.artifacts import (
    SingleLayerGateRecord,
    deterministic_cross_item_donor_plan,
    load_single_layer_gate_artifact,
)
from typo_robust_training.state_gate.config import load_single_layer_gate_config
from typo_robust_training.state_gate.producer import produce_single_layer_gate_artifact
from typo_robust_training.state_gate.runtime import (
    HuggingFaceSingleLayerGateProvider,
    _checkout_code_revision,
    _checkout_source_attestation,
    _require_exact_model_revision,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _gate_inputs(tmp_path: Path) -> tuple[dict[str, Path], object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    parent_path, _parent_payload, parent_files = _parent_bundle(parent_root)
    parent = load_probe_transition_artifact(parent_path)
    rows: list[dict[str, object]] = []
    operations = (
        ("keyboard-neighbor-substitution", "slpha", "same"),
        ("deletion", "apha", "minus-one"),
        ("duplication", "allpha", "plus-one"),
    )
    counts: dict[str, int] = {}
    for index in range(200):
        edit, typo_word, bucket = operations[index % 3]
        clean_word = "alpha"
        clean = f"Generic FineWeb document {index} contains {clean_word} and enough continuation words for deterministic testing."
        start = clean.index(clean_word)
        typo = clean[:start] + typo_word + clean[start + len(clean_word) :]
        stratum = f"{edit}|1|{bucket}"
        counts[stratum] = counts.get(stratum, 0) + 1
        rows.append(
            {
                "record_id": f"gate-{index:03d}",
                "pair_id": f"gate-pair-{index:03d}",
                "source_group_sha256": _sha(f"gate-group-{index}"),
                "parent_source_sha256": _sha(f"gate-parent-{index}"),
                "normalized_clean_sha256": normalized_content_sha256(clean),
                "normalized_noisy_sha256": normalized_content_sha256(typo),
                "clean_text": clean,
                "typo_text": typo,
                "clean_word_char_span": [start, start + len(clean_word)],
                "typo_word_char_span": [start, start + len(typo_word)],
                "edit_type": edit,
                "edit_count": 1,
                "token_inflation_bucket": bucket,
            }
        )
    cohort = _write(
        tmp_path / "gate-cohort.json",
        {
            "schema_version": "typo-single-layer-gate-cohort/v1",
            "role": "independent-generic-fineweb",
            "records": rows,
        },
    )
    protected = parent_files["protected"]
    # The deterministic donor is a cyclic shift because every source group is unique.
    donor = _write(
        tmp_path / "donor-plan.json",
        {
            "schema_version": "typo-cross-item-donor-plan/v1",
            "rule": "first-valid-cyclic-source-group-derangement/v1",
            "records": [
                {
                    "pair_id": row["pair_id"],
                    "donor_pair_id": rows[(index + 1) % len(rows)]["pair_id"],
                }
                for index, row in enumerate(rows)
            ],
        },
    )
    runtime_value = {
        "schema_version": "single-layer-gate-runtime/v2",
        "provider": "hugging-face-single-layer-gate/v1",
        "model": parent.model,
        "model_revision": parent.model_revision,
        "teacher_revision": parent.model_revision,
        "student_revision": parent.model_revision,
        "tokenizer_revision": parent.model_revision,
        "code_revision": "b" * 40,
        "source_tree_sha256": "d" * 64,
        "decoder_layers": parent.decoder_layers,
        "dtype": "bfloat16",
        "hook_site": "complete-decoder-block-residual-output",
        "coordinate": "edited-word-final-token/v1",
        "readout": "teacher-forced-tokens-2-through-16-inclusive/v1",
        "base_model_frozen": True,
        "packages": {"python": "test", "torch": "test", "transformers": "test"},
        "hardware": {"gpu_name": "fake", "cuda": "fake"},
    }
    runtime = _write(tmp_path / "runtime.json", runtime_value)
    input_hashes = {
        "parent_probe_artifact_sha256": parent.artifact_sha256,
        "cohort_manifest_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        "protected_registry_sha256": hashlib.sha256(protected.read_bytes()).hexdigest(),
        "donor_plan_sha256": hashlib.sha256(donor.read_bytes()).hexdigest(),
        "runtime_manifest_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
    }
    config = _write(
        tmp_path / "gate-config.json",
        {
            "schema_version": "probe-transition-single-layer-gate-config/v1",
            "model": {
                "id": parent.model,
                "revision": parent.model_revision,
                "code_revision": "b" * 40,
                "decoder_layers": parent.decoder_layers,
                "dtype": "bfloat16",
            },
            "inputs": input_hashes,
            "cohort": {
                "records": 200,
                "minimum_valid_records": 180,
                "stratum_counts": counts,
                "minimum_valid_per_stratum": {key: 60 for key in counts},
            },
            "intervention": {
                "hook_site": "complete-decoder-block-residual-output",
                "coordinate": "edited-word-final-token/v1",
                "direction": "clean-to-typo",
                "layer_source": "parent-probe-selected-transition-layer/v1",
                "offset_control_tokens": 2,
                "cross_item_rule": "first-valid-cyclic-source-group-derangement/v1",
                "self_copy_control": "typo-to-identical-typo-coordinate/v1",
                "teacher_forced_tokens": 16,
                "readout_offsets": [2, 16],
                "denominator_min_exclusive": 1e-9,
            },
            "gate": {
                "estimator": "source-group-equal-mean-pairwise-restoration/v1",
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 20260813,
                "confidence": 0.95,
                "bootstrap_unit": "source-group",
                "minimum_correct_ci_lower": 0.0,
                "minimum_correct_minus_offset_ci_lower": 0.0,
                "minimum_correct_minus_cross_ci_lower": 0.0,
                "maximum_absolute_self_copy_restoration": 1e-5,
            },
        },
    )
    return {
        "parent": parent_path,
        "cohort": cohort,
        "protected": protected,
        "donor": donor,
        "runtime": runtime,
        "config": config,
    }, runtime_value


class _FakeProvider:
    def __init__(self, provenance: object) -> None:
        self._provenance = provenance
        assert isinstance(provenance, Mapping)
        self.model_id = str(provenance["model"])
        self.model_revision = str(provenance["model_revision"])
        self.teacher_revision = str(provenance["teacher_revision"])
        self.student_revision = str(provenance["student_revision"])
        self.tokenizer_revision = str(provenance["tokenizer_revision"])
        self.code_revision = str(provenance["code_revision"])
        self.source_tree_sha256 = str(provenance["source_tree_sha256"])
        self.decoder_layers = int(provenance["decoder_layers"])
        self.base_model_frozen = bool(provenance["base_model_frozen"])
        self.scan_calls = 0

    def provenance(self) -> Mapping[str, object]:
        assert isinstance(self._provenance, Mapping)
        return self._provenance

    def token_inflation_bucket(self, record: SingleLayerGateRecord) -> str:
        return {
            "keyboard-neighbor-substitution": "same",
            "deletion": "minus-one",
            "duplication": "plus-one",
        }[record.edit_type]

    def scan(
        self,
        records: Sequence[SingleLayerGateRecord],
        *,
        donor_plan: Mapping[str, str],
        transition_layer: int,
    ) -> Sequence[Mapping[str, object]]:
        self.scan_calls += 1
        by_pair = {record.pair_id: record for record in records}
        output = []
        for record in records:
            donor = by_pair[donor_plan[record.pair_id]]
            clean_start, clean_stop = record.clean_word_char_span
            typo_start, typo_stop = record.typo_word_char_span
            output.append(
                {
                    "pair_id": record.pair_id,
                    "source_group_sha256": record.source_group_sha256,
                    "stratum": record.stratum,
                    "transition_layer": transition_layer,
                    "clean_word_final_token": 2,
                    "typo_word_final_token": 2,
                    "offset_donor_clean_token": 4,
                    "offset_patch_token": 4,
                    "cross_donor_pair_id": donor.pair_id,
                    "cross_donor_clean_word_final_token": 2,
                    "cross_donor_clean_prompt_offsets": [
                        [0, 0],
                        [0, donor.clean_word_char_span[0]],
                        list(donor.clean_word_char_span),
                    ],
                    "clean_prompt_offsets": [[0, 0], [0, clean_start], [clean_start, clean_stop]],
                    "typo_prompt_offsets": [[0, 0], [0, typo_start], [typo_start, typo_stop]],
                    "target_token_ids": list(range(16)),
                    "untreated_kl_2_16": [1.0] * 15,
                    "correct_kl_2_16": [0.2] * 15,
                    "offset_kl_2_16": [0.8] * 15,
                    "cross_kl_2_16": [0.9] * 15,
                    "self_copy_kl_2_16": [1.0] * 15,
                    "invalid_reason": None,
                }
            )
        return output


def _produce(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    inputs, provenance = _gate_inputs(tmp_path)
    output = tmp_path / "gate-output"
    produce_single_layer_gate_artifact(
        config_path=inputs["config"],
        parent_probe_artifact_path=inputs["parent"],
        cohort_manifest_path=inputs["cohort"],
        protected_registry_path=inputs["protected"],
        donor_plan_path=inputs["donor"],
        runtime_manifest_path=inputs["runtime"],
        output_dir=output,
        provider=_FakeProvider(provenance),
    )
    return output / "single-layer-gate.json", inputs


def test_fake_provider_e2e_recomputes_passed_gate(tmp_path: Path) -> None:
    artifact_path, _inputs = _produce(tmp_path)

    artifact = load_single_layer_gate_artifact(artifact_path)

    assert artifact.selected_transition_layer == 2
    assert artifact.valid_records == 200
    assert artifact.scores["correct"].ci_lower > 0
    assert artifact.scores["correct_minus_offset"].ci_lower > 0
    assert artifact.scores["correct_minus_cross"].ci_lower > 0


def test_offset_control_spies_matching_clean_and_typo_plus_two_coordinates() -> None:
    provider = object.__new__(HuggingFaceSingleLayerGateProvider)
    provider.protocol = SimpleNamespace(offset_control_tokens=2)
    observed: list[tuple[str, object, int]] = []

    def donor(ids: object, _mask: object, *, layer: int, position: int) -> str:
        observed.append(("donor", ids, position))
        assert layer == 7
        return "clean-plus-two-state"

    def patched(**kwargs: object) -> tuple[float, ...]:
        observed.append(("recipient", kwargs["ids"], int(kwargs["position"])))
        assert kwargs["donor"] == "clean-plus-two-state"
        return (0.0,) * 15

    provider._donor = donor  # type: ignore[method-assign]
    provider._patched = patched  # type: ignore[method-assign]
    provider._offset_patched(
        clean_ids="clean-full",
        clean_mask="clean-mask",
        typo_ids="typo-full",
        typo_mask="typo-mask",
        layer=7,
        clean_position=11,
        typo_position=14,
        prompt_tokens=15,
        reference="reference",
    )

    assert observed == [
        ("donor", "clean-full", 13),
        ("recipient", "typo-full", 16),
    ]


def _mutate_raw(
    artifact_path: Path,
    mutator: object,
) -> None:
    payload = json.loads(artifact_path.read_text())
    raw_ref = payload["references"]["raw_kl"]
    raw_path = artifact_path.parent / raw_ref["relative_path"]
    raw = json.loads(raw_path.read_text())
    assert callable(mutator)
    mutator(raw)
    _write(raw_path, raw)
    raw_ref["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    _write(artifact_path, payload)


def test_gate_rejects_wrong_layer_or_word_final_coordinate(tmp_path: Path) -> None:
    artifact_path, _inputs = _produce(tmp_path)
    _mutate_raw(
        artifact_path,
        lambda raw: raw["records"][0].update({"transition_layer": 3}),
    )
    with pytest.raises(ValueError, match="wrong layer"):
        load_single_layer_gate_artifact(artifact_path)

    artifact_path, _inputs = _produce(tmp_path / "second")
    _mutate_raw(
        artifact_path,
        lambda raw: raw["records"][0].update({"typo_word_final_token": 1}),
    )
    with pytest.raises(ValueError, match="non-word-final"):
        load_single_layer_gate_artifact(artifact_path)

    artifact_path, _inputs = _produce(tmp_path / "cross-token")
    _mutate_raw(
        artifact_path,
        lambda raw: raw["records"][0].update(
            {"cross_donor_clean_word_final_token": 1}
        ),
    )
    with pytest.raises(ValueError, match="cross donor patched a non-word-final"):
        load_single_layer_gate_artifact(artifact_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("offset_donor_clean_token", 3, r"offset donor differs from clean \+2"),
        ("offset_patch_token", 3, r"offset control differs from \+2"),
    ],
)
def test_gate_rejects_wrong_offset_donor_or_recipient_coordinate(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    artifact_path, _inputs = _produce(tmp_path)
    _mutate_raw(
        artifact_path,
        lambda raw: raw["records"][0].update({field: value}),
    )

    with pytest.raises(ValueError, match=message):
        load_single_layer_gate_artifact(artifact_path)


@pytest.mark.parametrize("control", ["offset", "cross"])
def test_gate_rejects_fabricated_pass_when_correct_does_not_beat_control(
    tmp_path: Path,
    control: str,
) -> None:
    artifact_path, _inputs = _produce(tmp_path)

    def mutate(raw: dict[str, object]) -> None:
        for row in raw["records"]:  # type: ignore[index]
            row[f"{control}_kl_2_16"] = [0.2] * 15

    _mutate_raw(artifact_path, mutate)
    with pytest.raises(ValueError, match="did not pass recomputation"):
        load_single_layer_gate_artifact(artifact_path)


def test_gate_rejects_non_identity_self_copy_and_parent_tamper(tmp_path: Path) -> None:
    artifact_path, _inputs = _produce(tmp_path)

    def mutate(raw: dict[str, object]) -> None:
        raw["records"][0]["self_copy_kl_2_16"] = [0.5] * 15  # type: ignore[index]

    _mutate_raw(artifact_path, mutate)
    with pytest.raises(ValueError, match="did not pass recomputation"):
        load_single_layer_gate_artifact(artifact_path)

    artifact_path, _inputs = _produce(tmp_path / "parent-tamper")
    payload = json.loads(artifact_path.read_text())
    parent_path = artifact_path.parent / payload["references"]["parent_probe_artifact"]["relative_path"]
    parent_path.write_bytes(parent_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash-mismatched"):
        load_single_layer_gate_artifact(artifact_path)


def test_gate_rejects_symlink_or_one_byte_raw_tamper(tmp_path: Path) -> None:
    artifact_path, _inputs = _produce(tmp_path)
    link = tmp_path / "gate-link.json"
    link.symlink_to(artifact_path)
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_single_layer_gate_artifact(link)

    payload = json.loads(artifact_path.read_text())
    raw_path = artifact_path.parent / payload["references"]["raw_kl"]["relative_path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash-mismatched"):
        load_single_layer_gate_artifact(artifact_path)


def test_gate_rejects_symlinked_referenced_file(tmp_path: Path) -> None:
    artifact_path, _inputs = _produce(tmp_path)
    payload = json.loads(artifact_path.read_text())
    raw_path = artifact_path.parent / payload["references"]["raw_kl"]["relative_path"]
    regular_copy = raw_path.with_name("raw-kl-regular.json")
    raw_path.rename(regular_copy)
    raw_path.symlink_to(regular_copy.name)

    with pytest.raises(ValueError, match="reference is a symlink"):
        load_single_layer_gate_artifact(artifact_path)


def test_gate_config_rejects_relaxed_causal_threshold(tmp_path: Path) -> None:
    inputs, _provenance = _gate_inputs(tmp_path)
    payload = json.loads(inputs["config"].read_text())
    payload["gate"]["minimum_correct_ci_lower"] = -1.0
    _write(inputs["config"], payload)

    with pytest.raises(ValueError, match="thresholds must all equal zero"):
        load_single_layer_gate_config(inputs["config"])


def _rewrite_cohort_and_rebind_config(
    inputs: Mapping[str, Path],
    cohort: object,
) -> None:
    _write(inputs["cohort"], cohort)
    config = json.loads(inputs["config"].read_text())
    config["inputs"]["cohort_manifest_sha256"] = hashlib.sha256(
        inputs["cohort"].read_bytes()
    ).hexdigest()
    _write(inputs["config"], config)


def test_gate_rejects_duplicate_normalized_pair_under_fresh_ids_and_groups(
    tmp_path: Path,
) -> None:
    inputs, provenance = _gate_inputs(tmp_path)
    cohort = json.loads(inputs["cohort"].read_text())
    original = cohort["records"][0]
    duplicate = cohort["records"][1]
    for field in (
        "normalized_clean_sha256",
        "normalized_noisy_sha256",
        "clean_text",
        "typo_text",
        "clean_word_char_span",
        "typo_word_char_span",
        "edit_type",
        "edit_count",
        "token_inflation_bucket",
    ):
        duplicate[field] = original[field]
    _rewrite_cohort_and_rebind_config(inputs, cohort)

    with pytest.raises(ValueError, match="normalized clean/noisy content must be unique"):
        produce_single_layer_gate_artifact(
            config_path=inputs["config"],
            parent_probe_artifact_path=inputs["parent"],
            cohort_manifest_path=inputs["cohort"],
            protected_registry_path=inputs["protected"],
            donor_plan_path=inputs["donor"],
            runtime_manifest_path=inputs["runtime"],
            output_dir=tmp_path / "rejected-output",
            provider=_FakeProvider(provenance),
        )


def test_gate_rejects_parent_source_split_across_bootstrap_groups(
    tmp_path: Path,
) -> None:
    inputs, provenance = _gate_inputs(tmp_path)
    cohort = json.loads(inputs["cohort"].read_text())
    cohort["records"][1]["parent_source_sha256"] = cohort["records"][0][
        "parent_source_sha256"
    ]
    _rewrite_cohort_and_rebind_config(inputs, cohort)

    with pytest.raises(ValueError, match="parent source maps to multiple bootstrap groups"):
        produce_single_layer_gate_artifact(
            config_path=inputs["config"],
            parent_probe_artifact_path=inputs["parent"],
            cohort_manifest_path=inputs["cohort"],
            protected_registry_path=inputs["protected"],
            donor_plan_path=inputs["donor"],
            runtime_manifest_path=inputs["runtime"],
            output_dir=tmp_path / "rejected-output",
            provider=_FakeProvider(provenance),
        )


def test_gate_recomputes_token_inflation_with_bound_runtime_before_scan(
    tmp_path: Path,
) -> None:
    inputs, provenance = _gate_inputs(tmp_path)

    class WrongTokenizerBucketProvider(_FakeProvider):
        def token_inflation_bucket(self, _record: SingleLayerGateRecord) -> str:
            return "plus-two-or-more"

    provider = WrongTokenizerBucketProvider(provenance)
    with pytest.raises(ValueError, match="differs from runtime tokenizer"):
        produce_single_layer_gate_artifact(
            config_path=inputs["config"],
            parent_probe_artifact_path=inputs["parent"],
            cohort_manifest_path=inputs["cohort"],
            protected_registry_path=inputs["protected"],
            donor_plan_path=inputs["donor"],
            runtime_manifest_path=inputs["runtime"],
            output_dir=tmp_path / "rejected-output",
            provider=provider,
        )
    assert provider.scan_calls == 0


def test_hugging_face_gate_provider_derives_bucket_from_its_bound_tokenizer() -> None:
    provider = object.__new__(HuggingFaceSingleLayerGateProvider)

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> object:
            assert add_special_tokens is True
            lengths = {"alpha": 3, "allpha": 4}
            return {"input_ids": list(range(lengths[text]))}

    provider.tokenizer = Tokenizer()
    record = SingleLayerGateRecord(
        record_id="record",
        pair_id="pair",
        source_group_sha256="a" * 64,
        parent_source_sha256="b" * 64,
        normalized_clean_sha256="c" * 64,
        normalized_noisy_sha256="d" * 64,
        clean_text="alpha",
        typo_text="allpha",
        clean_word_char_span=(0, 5),
        typo_word_char_span=(0, 6),
        edit_type="duplication",
        edit_count=1,
        token_inflation_bucket="plus-one",
    )

    assert provider.token_inflation_bucket(record) == "plus-one"


@pytest.mark.parametrize(
    "field",
    [
        "model_revision",
        "teacher_revision",
        "student_revision",
        "tokenizer_revision",
        "code_revision",
        "source_tree_sha256",
    ],
)
def test_gate_rejects_unattested_provider_identity_before_scan(
    tmp_path: Path,
    field: str,
) -> None:
    inputs, provenance = _gate_inputs(tmp_path)
    provider = _FakeProvider(provenance)
    setattr(provider, field, "c" * (64 if field == "source_tree_sha256" else 40))

    with pytest.raises(ValueError, match="identity or freeze contract"):
        produce_single_layer_gate_artifact(
            config_path=inputs["config"],
            parent_probe_artifact_path=inputs["parent"],
            cohort_manifest_path=inputs["cohort"],
            protected_registry_path=inputs["protected"],
            donor_plan_path=inputs["donor"],
            runtime_manifest_path=inputs["runtime"],
            output_dir=tmp_path / "rejected-output",
            provider=provider,
        )
    assert provider.scan_calls == 0


def test_gate_runtime_rejects_dirty_executing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent

    def fake_run(args: list[str], **_kwargs: object) -> object:
        operation = tuple(args[1:])
        if operation == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{checkout}\n")
        if operation[0] == "ls-files":
            return SimpleNamespace(returncode=0, stdout="tracked\n")
        if operation[0] == "status":
            return SimpleNamespace(returncode=0, stdout=" M state_gate/runtime.py\n")
        return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")

    monkeypatch.setattr(
        "typo_robust_training.state_gate.runtime.subprocess.run",
        fake_run,
    )
    with pytest.raises(RuntimeError, match="source tree is not clean"):
        _checkout_code_revision()


def test_gate_runtime_rejects_dirty_typo_cot_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent

    def fake_run(args: list[str], **_kwargs: object) -> object:
        operation = tuple(args[1:])
        if operation == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{checkout}\n")
        if operation[0] == "ls-files":
            return SimpleNamespace(returncode=0, stdout="tracked\n")
        if operation[0] == "status":
            assert any("projects/typo-cot/src/typo_cot" in value for value in args)
            return SimpleNamespace(
                returncode=0,
                stdout=" M projects/typo-cot/src/typo_cot/models/wrapper.py\n",
            )
        return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")

    monkeypatch.setattr(
        "typo_robust_training.state_gate.runtime.subprocess.run",
        fake_run,
    )
    with pytest.raises(RuntimeError, match="source tree is not clean"):
        _checkout_code_revision()


def test_gate_runtime_rejects_untracked_typo_cot_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent

    def fake_run(args: list[str], **_kwargs: object) -> object:
        operation = tuple(args[1:])
        if operation == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{checkout}\n")
        if operation[0] == "ls-files":
            dependency = any("projects/typo-cot/src/typo_cot" in value for value in args)
            return SimpleNamespace(returncode=1 if dependency else 0, stdout="")
        raise AssertionError("attestation continued after an untracked dependency")

    monkeypatch.setattr(
        "typo_robust_training.state_gate.runtime.subprocess.run",
        fake_run,
    )
    with pytest.raises(RuntimeError, match="not tracked"):
        _checkout_code_revision()


@pytest.mark.parametrize("origin", [None, "/tmp/unattested/typo_cot/__init__.py"])
def test_gate_runtime_rejects_unknown_or_outside_dependency_source(
    monkeypatch: pytest.MonkeyPatch,
    origin: str | None,
) -> None:
    checkout = Path(__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent
    monkeypatch.setattr(
        "typo_robust_training.state_gate.runtime.importlib.util.find_spec",
        lambda _name: SimpleNamespace(origin=origin) if origin is not None else None,
    )
    monkeypatch.setattr(
        "typo_robust_training.state_gate.runtime.subprocess.run",
        lambda _args, **_kwargs: SimpleNamespace(returncode=0, stdout=f"{checkout}\n"),
    )
    with pytest.raises(RuntimeError, match="dependency source|outside"):
        _checkout_code_revision()


def test_gate_source_tree_digest_binds_both_required_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent
    robust_tree = ["b" * 40]
    dependency_tree = ["c" * 40]

    def fake_run(args: list[str], **_kwargs: object) -> object:
        operation = tuple(args[1:])
        if operation == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{checkout}\n")
        if operation[0] == "ls-files" or operation[0] == "status":
            return SimpleNamespace(returncode=0, stdout="")
        if operation == ("rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")
        if operation[0] == "rev-parse" and "projects/typo-cot" in operation[1]:
            return SimpleNamespace(returncode=0, stdout=f"{dependency_tree[0]}\n")
        if operation[0] == "rev-parse" and "projects/typo-robust-training" in operation[1]:
            return SimpleNamespace(returncode=0, stdout=f"{robust_tree[0]}\n")
        raise AssertionError(f"unexpected git operation: {operation}")

    monkeypatch.setattr(
        "typo_robust_training.state_gate.runtime.subprocess.run",
        fake_run,
    )
    first_revision, first_digest = _checkout_source_attestation()
    dependency_tree[0] = "d" * 40
    second_revision, second_digest = _checkout_source_attestation()
    robust_tree[0] = "e" * 40
    third_revision, third_digest = _checkout_source_attestation()
    assert first_revision == second_revision == third_revision == "a" * 40
    assert first_digest != second_digest
    assert second_digest != third_digest


def test_gate_runtime_requires_executing_module_to_be_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent

    def fake_run(args: list[str], **_kwargs: object) -> object:
        operation = tuple(args[1:])
        if operation == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{checkout}\n")
        if operation[0] == "ls-files":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError("attestation continued after an untracked runtime")

    monkeypatch.setattr(
        "typo_robust_training.state_gate.runtime.subprocess.run",
        fake_run,
    )
    with pytest.raises(RuntimeError, match="not tracked"):
        _checkout_code_revision()


def test_gate_model_revision_must_be_observable_from_model_independently() -> None:
    config = SimpleNamespace(_commit_hash=None, text_config=None)
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "a" * 40})
    with pytest.raises(ValueError, match="model revision is not observable"):
        _require_exact_model_revision(
            model_config=config,
            tokenizer=tokenizer,
            expected="a" * 40,
        )
    config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="model revision differs"):
        _require_exact_model_revision(
            model_config=config,
            tokenizer=tokenizer,
            expected="a" * 40,
        )


@pytest.mark.parametrize(
    "tokenizer",
    [
        SimpleNamespace(),
        SimpleNamespace(init_kwargs=None),
        SimpleNamespace(init_kwargs=[]),
        SimpleNamespace(init_kwargs={}),
        SimpleNamespace(init_kwargs={"_commit_hash": None}),
        SimpleNamespace(init_kwargs={"_commit_hash": ""}),
        SimpleNamespace(init_kwargs={"_commit_hash": "not-a-revision"}),
    ],
)
def test_gate_tokenizer_revision_must_be_observable(tokenizer: object) -> None:
    config = SimpleNamespace(_commit_hash="a" * 40, text_config=None)
    with pytest.raises(ValueError, match="tokenizer revision is not observable"):
        _require_exact_model_revision(
            model_config=config,
            tokenizer=tokenizer,
            expected="a" * 40,
        )


def test_gate_tokenizer_revision_must_match_exactly() -> None:
    config = SimpleNamespace(_commit_hash="a" * 40, text_config=None)
    with pytest.raises(ValueError, match="tokenizer revision differs"):
        _require_exact_model_revision(
            model_config=config,
            tokenizer=SimpleNamespace(init_kwargs={"_commit_hash": "b" * 40}),
            expected="a" * 40,
        )


def test_gate_rejects_conflicting_top_level_and_text_model_revisions() -> None:
    config = SimpleNamespace(
        _commit_hash="a" * 40,
        text_config=SimpleNamespace(_commit_hash="b" * 40),
    )
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "a" * 40})
    with pytest.raises(ValueError, match="model revision differs"):
        _require_exact_model_revision(
            model_config=config,
            tokenizer=tokenizer,
            expected="a" * 40,
        )


@pytest.mark.parametrize("parent_role", ["fit", "selection", "validation", "protected"])
def test_gate_rejects_overlap_with_every_parent_or_protected_identity(
    tmp_path: Path,
    parent_role: str,
) -> None:
    inputs, provenance = _gate_inputs(tmp_path)
    parent = load_probe_transition_artifact(inputs["parent"])
    cohort = json.loads(inputs["cohort"].read_text())
    cohort["records"][0]["source_group_sha256"] = next(
        iter(getattr(parent.identity_inventory, parent_role))
    )
    _write(inputs["cohort"], cohort)
    config = json.loads(inputs["config"].read_text())
    config["inputs"]["cohort_manifest_sha256"] = hashlib.sha256(
        inputs["cohort"].read_bytes()
    ).hexdigest()
    _write(inputs["config"], config)

    with pytest.raises(ValueError, match="overlaps parent or protected"):
        produce_single_layer_gate_artifact(
            config_path=inputs["config"],
            parent_probe_artifact_path=inputs["parent"],
            cohort_manifest_path=inputs["cohort"],
            protected_registry_path=inputs["protected"],
            donor_plan_path=inputs["donor"],
            runtime_manifest_path=inputs["runtime"],
            output_dir=tmp_path / "rejected-output",
            provider=_FakeProvider(provenance),
        )


def test_donor_plan_requires_source_group_derangement() -> None:
    row = SingleLayerGateRecord(
        record_id="a",
        pair_id="a",
        source_group_sha256="a" * 64,
        parent_source_sha256="b" * 64,
        normalized_clean_sha256="c" * 64,
        normalized_noisy_sha256="d" * 64,
        clean_text="alpha",
        typo_text="slpha",
        clean_word_char_span=(0, 5),
        typo_word_char_span=(0, 5),
        edit_type="keyboard-neighbor-substitution",
        edit_count=1,
        token_inflation_bucket="same",
    )
    with pytest.raises(ValueError, match="at least two"):
        deterministic_cross_item_donor_plan((row,))
