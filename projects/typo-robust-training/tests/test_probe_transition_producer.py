from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors import safe_open

from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.integrity import sha256_file
from typo_robust_training.cli import register_commands
from typo_robust_training.probe.artifacts import load_probe_transition_artifact
from typo_robust_training.probe import artifacts as probe_artifacts
from typo_robust_training.probe.config import load_probe_producer_config
from typo_robust_training.probe.producer import (
    ProbeCohortRecord,
    ProbeTransitionProducerRunConfig,
    run_select_probe_transition,
)
from typo_robust_training.probe.runtime import (
    _checkout_code_revision,
    _inflation_bucket,
    _require_exact_model_revision,
)
from typo_robust_training.probe import runtime as probe_runtime


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(role: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    labels = ("alpha", "beta")
    typos = {
        0: ("slpha", "apha", "allpha"),
        1: ("neta", "eta", "betaa"),
    }
    operations = (
        "keyboard-neighbor-substitution",
        "deletion",
        "duplication",
    )
    buckets = ("same", "minus-one", "plus-one")
    for class_id, word in enumerate(labels):
        for repetition in range(3):
            identity = f"{role}-{class_id}-{repetition}"
            clean_text = f"The token {word} appears in context {identity}."
            start = clean_text.index(word)
            row: dict[str, object] = {
                "record_id": identity,
                "source_group_sha256": _sha(f"group-{identity}"),
                "parent_source_sha256": _sha(f"parent-{identity}"),
                "normalized_clean_sha256": normalized_content_sha256(clean_text),
                "class_id": class_id,
                "clean_text": clean_text,
                "clean_word_char_span": [start, start + len(word)],
            }
            if role != "fit":
                typo_word = typos[class_id][repetition]
                typo_text = clean_text[:start] + typo_word + clean_text[start + len(word) :]
                row.update(
                    {
                        "pair_id": f"pair-{identity}",
                        "normalized_noisy_sha256": normalized_content_sha256(typo_text),
                        "edit_type": operations[repetition],
                        "edit_count": 1,
                        "token_inflation_bucket": buckets[repetition],
                        "typo_text": typo_text,
                        "typo_word_char_span": [start, start + len(typo_word)],
                    }
                )
            rows.append(row)
    return {"schema_version": "typo-probe-cohort/v2", "role": role, "records": rows}


def _files(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {
        "classes": _write_json(
            tmp_path / "classes.json",
            {
                "schema_version": "typo-word-identity-classes/v1",
                "classes": [
                    {"class_id": 0, "label": "alpha"},
                    {"class_id": 1, "label": "beta"},
                ],
            },
        ),
        "fit": _write_json(tmp_path / "fit.json", _manifest("fit")),
        "selection": _write_json(
            tmp_path / "selection.json", _manifest("selection")
        ),
        "validation": _write_json(
            tmp_path / "validation.json", _manifest("validation")
        ),
        "protected": _write_json(
            tmp_path / "protected.json",
            {
                "schema_version": "typo-protected-split-registry/v1",
                "registries": [
                    {
                        "tier": tier,
                        "source_group_sha256": [_sha(f"protected-group-{tier}")],
                        "parent_source_sha256": [_sha(f"protected-parent-{tier}")],
                        "normalized_content_sha256": [_sha(f"protected-content-{tier}")],
                    }
                    for tier in ("training", "localization", "tune", "pre-pr", "sealed")
                ],
            },
        ),
    }
    config = {
        "schema_version": "typo-linear-probe-producer-config/v2",
        "model": {
            "id": "google/gemma-3-4b-it",
            "revision": "a" * 40,
            "code_revision": "b" * 40,
            "decoder_layers": 4,
            "hidden_size": 2,
            "dtype": "bfloat16",
        },
        "inputs": {
            "class_inventory_sha256": sha256_file(files["classes"]),
            "fit_manifest_sha256": sha256_file(files["fit"]),
            "selection_manifest_sha256": sha256_file(files["selection"]),
            "validation_manifest_sha256": sha256_file(files["validation"]),
            "protected_registry_sha256": sha256_file(files["protected"]),
        },
        "cohorts": {
            "records_per_class": {"fit": 3, "selection": 3, "validation": 3},
            "min_source_groups_per_class": {
                "fit": 2,
                "selection": 2,
                "validation": 2,
            },
            "stratum_counts": {
                role: {
                    "keyboard-neighbor-substitution|1|same": 2,
                    "deletion|1|minus-one": 2,
                    "duplication|1|plus-one": 2,
                }
                for role in ("selection", "validation")
            },
        },
        "probe": {
            "seeds": [42, 43],
            "optimizer": "adamw",
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "epochs": 50,
            "batch_size": 6,
            "hook_site": "complete-decoder-block-residual-output",
            "coordinate": "edited-word-final-token/v1",
        },
        "selection": {
            "metric": "largest-group-mean-paired-noise-penalty-drop/v2",
            "rule": "min-argmax-over-layers-one-through-last/v1",
            "tie_break": "smallest-layer/v1",
            "stability_rule": (
                "selection-exact-and-validation-within-one-layer-for-both-seeds/v1"
            ),
            "validation_rule": "group-bootstrap-95pct-lower-positive-for-both-seeds/v1",
            "bootstrap": {
                "resamples": 10_000,
                "seed": 1729,
                "confidence": 0.95,
                "unit": "source-group",
            },
        },
    }
    files["config"] = _write_json(tmp_path / "config.json", config)
    return files


class _FakeProvider:
    model = "google/gemma-3-4b-it"
    model_revision = "a" * 40
    code_revision = "b" * 40
    decoder_layers = 4
    hidden_size = 2
    base_model_frozen = True

    def activations(
        self,
        records: tuple[ProbeCohortRecord, ...],
        *,
        side: str,
    ) -> np.ndarray:
        values = np.empty((len(records), 4, 2), dtype=np.float32)
        for index, record in enumerate(records):
            clean = np.asarray((3.0, -3.0) if record.class_id == 0 else (-3.0, 3.0))
            values[index] = clean
            if side == "typo":
                values[index, :2] = -clean
            elif side != "clean":
                raise ValueError("unexpected side")
        return values

    def provenance(self) -> dict[str, object]:
        return {
            "provider": "fake-test-provider/v1",
            "model": self.model,
            "model_revision": self.model_revision,
            "base_model_frozen": True,
            "code_revision": self.code_revision,
        }

    def token_inflation_bucket(self, record: ProbeCohortRecord) -> str:
        return {
            "keyboard-neighbor-substitution": "same",
            "deletion": "minus-one",
            "duplication": "plus-one",
        }[record.edit_type]


def _run_config(files: dict[str, Path], output_dir: Path) -> ProbeTransitionProducerRunConfig:
    return ProbeTransitionProducerRunConfig(
        config_path=files["config"],
        class_inventory_path=files["classes"],
        fit_manifest_path=files["fit"],
        selection_manifest_path=files["selection"],
        validation_manifest_path=files["validation"],
        protected_registry_path=files["protected"],
        gpu_id="0",
        output_dir=output_dir,
    )


def test_producer_derives_transition_and_binds_every_output(tmp_path: Path) -> None:
    files = _files(tmp_path)

    result = run_select_probe_transition(
        _run_config(files, tmp_path / "output"),
        activation_provider=_FakeProvider(),
    )

    assert result.selected_transition_layer == 2
    assert result.validation_passed is True
    weight_hashes = {seed: sha256_file(path) for seed, path in result.weights_by_seed.items()}
    assert len(set(weight_hashes.values())) == 2
    protocol = load_probe_producer_config(files["config"])
    for seed, path in result.weights_by_seed.items():
        with safe_open(path, framework="np") as handle:
            assert set(handle.keys()) == {
                f"decoder_layer.{layer}.{kind}"
                for layer in range(4)
                for kind in ("weight", "bias")
            }
            metadata = handle.metadata()
            assert metadata["seed"] == str(seed)
            assert metadata["config_sha256"] == protocol.config_sha256
            assert metadata["fit_manifest_sha256"] == sha256_file(files["fit"])
            assert metadata["class_inventory_sha256"] == sha256_file(files["classes"])
            assert metadata["model_revision"] == "a" * 40
            assert metadata["code_revision"] == "b" * 40
            assert handle.get_tensor("decoder_layer.0.weight").shape == (2, 2)
            assert handle.get_tensor("decoder_layer.0.bias").shape == (2,)
        for role, paths in (
            ("selection", result.selection_scores_by_seed),
            ("validation", result.validation_scores_by_seed),
        ):
            score = json.loads(paths[seed].read_text())
            assert score["bindings"] == {
                "model": "google/gemma-3-4b-it",
                "model_revision": "a" * 40,
                "code_revision": "b" * 40,
                "config_sha256": protocol.config_sha256,
                "class_inventory_sha256": sha256_file(files["classes"]),
                "fit_manifest_sha256": sha256_file(files["fit"]),
                "role_manifest_sha256": sha256_file(files[role]),
                "probe_weights_sha256": weight_hashes[seed],
            }
    run = json.loads(result.run_path.read_text())
    assert all(value > 0.0 for value in run["selection_ci_lower_by_seed"].values())
    assert all(value > 0.0 for value in run["validation_ci_lower_by_seed"].values())
    artifact_hash = sha256_file(result.artifact_path)
    assert result.artifact_path.name == f"probe-transition-{artifact_hash}.json"
    consumed = load_probe_transition_artifact(result.artifact_path)
    assert consumed.selected_transition_layer == 2
    assert consumed.code_revision == "b" * 40
    assert consumed.hidden_size == 2


def test_input_hash_mismatch_fails_before_provider_construction(tmp_path: Path) -> None:
    files = _files(tmp_path)
    classes = json.loads(files["classes"].read_text())
    classes["classes"][0]["label"] = "changed"
    _write_json(files["classes"], classes)
    calls = 0

    def factory(_protocol: object, _gpu_id: str) -> _FakeProvider:
        nonlocal calls
        calls += 1
        return _FakeProvider()

    with pytest.raises(ValueError, match="input hashes differ"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            provider_factory=factory,
        )
    assert calls == 0


def test_preregistered_strata_mismatch_fails_before_provider(tmp_path: Path) -> None:
    files = _files(tmp_path)
    selection = json.loads(files["selection"].read_text())
    selection["records"][0]["token_inflation_bucket"] = "unexpected"
    _write_json(files["selection"], selection)
    config = json.loads(files["config"].read_text())
    config["inputs"]["selection_manifest_sha256"] = sha256_file(files["selection"])
    _write_json(files["config"], config)
    calls = 0

    def factory(_protocol: object, _gpu_id: str) -> _FakeProvider:
        nonlocal calls
        calls += 1
        return _FakeProvider()

    with pytest.raises(ValueError, match="strata differ"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            provider_factory=factory,
        )
    assert calls == 0


def test_rejects_token_inflation_bucket_mismatching_runtime_tokenizer(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    selection = json.loads(files["selection"].read_text())
    selection["records"][0]["token_inflation_bucket"] = "plus-two-or-more"
    _write_json(files["selection"], selection)
    config = json.loads(files["config"].read_text())
    config["inputs"]["selection_manifest_sha256"] = sha256_file(files["selection"])
    strata = config["cohorts"]["stratum_counts"]["selection"]
    strata["keyboard-neighbor-substitution|1|same"] = 1
    strata["keyboard-neighbor-substitution|1|plus-two-or-more"] = 1
    _write_json(files["config"], config)

    with pytest.raises(ValueError, match="runtime tokenizer"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            activation_provider=_FakeProvider(),
        )


def test_rejects_duplicate_normalized_pair_under_distinct_source_groups(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    selection = json.loads(files["selection"].read_text())
    original = selection["records"][0]
    duplicate = selection["records"][1]
    for field in (
        "normalized_clean_sha256",
        "clean_text",
        "clean_word_char_span",
        "normalized_noisy_sha256",
        "typo_text",
        "typo_word_char_span",
        "edit_type",
        "token_inflation_bucket",
    ):
        duplicate[field] = original[field]
    _write_json(files["selection"], selection)
    config = json.loads(files["config"].read_text())
    config["inputs"]["selection_manifest_sha256"] = sha256_file(files["selection"])
    _write_json(files["config"], config)

    with pytest.raises(ValueError, match="unique within role"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            activation_provider=_FakeProvider(),
        )


def test_cross_kind_role_identity_overlap_fails_before_provider(tmp_path: Path) -> None:
    files = _files(tmp_path)
    fit = json.loads(files["fit"].read_text())
    selection = json.loads(files["selection"].read_text())
    selection["records"][0]["source_group_sha256"] = fit["records"][0][
        "parent_source_sha256"
    ]
    _write_json(files["selection"], selection)
    config = json.loads(files["config"].read_text())
    config["inputs"]["selection_manifest_sha256"] = sha256_file(files["selection"])
    _write_json(files["config"], config)
    calls = 0

    def factory(_protocol: object, _gpu_id: str) -> _FakeProvider:
        nonlocal calls
        calls += 1
        return _FakeProvider()

    with pytest.raises(ValueError, match="overlap transitively"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            provider_factory=factory,
        )
    assert calls == 0


def test_non_neighbor_substitution_is_rejected_by_producer_and_loader(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    selection = json.loads(files["selection"].read_text())
    row = selection["records"][0]
    clean_text = row["clean_text"]
    start, stop = row["clean_word_char_span"]
    typo_word = "xlpha"
    typo_text = clean_text[:start] + typo_word + clean_text[stop:]
    row["typo_text"] = typo_text
    row["typo_word_char_span"] = [start, start + len(typo_word)]
    row["normalized_noisy_sha256"] = normalized_content_sha256(typo_text)
    _write_json(files["selection"], selection)
    config = json.loads(files["config"].read_text())
    config["inputs"]["selection_manifest_sha256"] = sha256_file(files["selection"])
    _write_json(files["config"], config)

    with pytest.raises(ValueError, match="case-preserving neighbor"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            activation_provider=_FakeProvider(),
        )
    with pytest.raises(ValueError, match="case-preserving neighbor"):
        probe_artifacts._load_manifest(  # noqa: SLF001 - falsifies the loader boundary
            files["selection"],
            expected_role="selection",
            class_labels=("alpha", "beta"),
        )


def test_config_rejects_unbound_selection_or_model_identity(tmp_path: Path) -> None:
    files = _files(tmp_path)
    payload = json.loads(files["config"].read_text())
    payload["selection"]["selected_layer"] = 2
    _write_json(files["config"], payload)
    with pytest.raises(ValueError, match="selection fields differ"):
        load_probe_producer_config(files["config"])

    files = _files(tmp_path / "second")
    payload = json.loads(files["config"].read_text())
    del payload["model"]["code_revision"]
    _write_json(files["config"], payload)
    with pytest.raises(ValueError, match="model fields differ"):
        load_probe_producer_config(files["config"])


def test_provider_architecture_or_freeze_mismatch_is_rejected(tmp_path: Path) -> None:
    files = _files(tmp_path)
    provider = _FakeProvider()
    provider.base_model_frozen = False

    with pytest.raises(ValueError, match="identity or freeze contract"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            activation_provider=provider,
        )


def test_rejects_unverified_runtime_code_revision(tmp_path: Path) -> None:
    files = _files(tmp_path)
    provider = _FakeProvider()
    provider.code_revision = "c" * 40

    with pytest.raises(ValueError, match="identity or freeze contract"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            activation_provider=provider,
        )


def test_checkout_code_revision_must_be_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(probe_runtime.__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent

    def fake_run(args: list[str], **_kwargs: object) -> object:
        operation = tuple(args[1:])
        if operation == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{checkout}\n")
        if operation[0] == "ls-files":
            return SimpleNamespace(returncode=0, stdout="tracked\n")
        if operation[0] == "status":
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout="not-a-revision\n")

    monkeypatch.setattr("typo_robust_training.probe.runtime.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="cannot attest"):
        _checkout_code_revision()


def test_checkout_code_revision_rejects_dirty_executing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(probe_runtime.__file__).resolve()
    while not (checkout / ".git").exists():
        checkout = checkout.parent

    def fake_run(args: list[str], **_kwargs: object) -> object:
        operation = tuple(args[1:])
        if operation == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=f"{checkout}\n")
        if operation[0] == "ls-files":
            return SimpleNamespace(returncode=0, stdout="tracked\n")
        if operation[0] == "status":
            return SimpleNamespace(returncode=0, stdout=" M probe/runtime.py\n")
        return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")

    monkeypatch.setattr("typo_robust_training.probe.runtime.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="source tree is not clean"):
        _checkout_code_revision()


def test_model_revision_must_be_observable_and_exact() -> None:
    class _Config:
        _commit_hash: str | None = None
        text_config: object | None = None

    class _Tokenizer:
        init_kwargs: dict[str, str] = {}

    with pytest.raises(ValueError, match="not observable"):
        _require_exact_model_revision(
            model_config=_Config(),
            tokenizer=_Tokenizer(),
            expected="a" * 40,
        )
    _Tokenizer.init_kwargs = {"_commit_hash": "a" * 40}
    with pytest.raises(ValueError, match="not observable"):
        _require_exact_model_revision(
            model_config=_Config(),
            tokenizer=_Tokenizer(),
            expected="a" * 40,
        )
    _Config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="differs"):
        _require_exact_model_revision(
            model_config=_Config(),
            tokenizer=_Tokenizer(),
            expected="a" * 40,
        )
    _Config._commit_hash = "a" * 40
    _Tokenizer.init_kwargs = {"_commit_hash": "b" * 40}
    with pytest.raises(ValueError, match="tokenizer revision differs"):
        _require_exact_model_revision(
            model_config=_Config(),
            tokenizer=_Tokenizer(),
            expected="a" * 40,
        )


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (-3, "minus-two-or-more"),
        (-1, "minus-one"),
        (0, "same"),
        (1, "plus-one"),
        (3, "plus-two-or-more"),
    ],
)
def test_token_inflation_bucket_has_closed_boundaries(delta: int, expected: str) -> None:
    assert _inflation_bucket(delta) == expected


def test_rejects_identical_probe_tensors_despite_seed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _files(tmp_path)

    def identical_fit(*_args: object, **_kwargs: object) -> object:
        from typo_robust_training.probe import producer

        return producer._ProbeWeights(  # noqa: SLF001 - falsifies seed independence
            weight=np.zeros((4, 2, 2), dtype=np.float32),
            bias=np.zeros((4, 2), dtype=np.float32),
        )

    monkeypatch.setattr(
        "typo_robust_training.probe.producer._fit_probe",
        identical_fit,
    )
    with pytest.raises(ValueError, match="identical numerical tensors"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            activation_provider=_FakeProvider(),
        )


def test_rejects_numerically_identical_probe_tensors_with_signed_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _files(tmp_path)

    def signed_zero_fit(*_args: object, **kwargs: object) -> object:
        from typo_robust_training.probe import producer

        weight = np.ones((4, 2, 2), dtype=np.float32)
        weight.flat[0] = -0.0 if kwargs["seed"] == 43 else 0.0
        return producer._ProbeWeights(  # noqa: SLF001 - falsifies seed independence
            weight=weight,
            bias=np.ones((4, 2), dtype=np.float32),
        )

    monkeypatch.setattr(
        "typo_robust_training.probe.producer._fit_probe",
        signed_zero_fit,
    )
    with pytest.raises(ValueError, match="identical numerical tensors"):
        run_select_probe_transition(
            _run_config(files, tmp_path / "output"),
            activation_provider=_FakeProvider(),
        )


def test_cli_registers_every_probe_producer_input() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)

    args = parser.parse_args(
        [
            "select-probe-transition",
            "--config",
            "config.json",
            "--class-inventory",
            "classes.json",
            "--fit-manifest",
            "fit.json",
            "--selection-manifest",
            "selection.json",
            "--validation-manifest",
            "validation.json",
            "--protected-registry",
            "protected.json",
            "--gpu-id",
            "4",
            "--output-dir",
            "evidence",
        ]
    )

    assert args.command == "select-probe-transition"
    assert args.config == Path("config.json")
    assert args.class_inventory == Path("classes.json")
    assert args.fit_manifest == Path("fit.json")
    assert args.selection_manifest == Path("selection.json")
    assert args.validation_manifest == Path("validation.json")
    assert args.protected_registry == Path("protected.json")
    assert args.gpu_id == "4"
    assert args.output_dir == Path("evidence")
