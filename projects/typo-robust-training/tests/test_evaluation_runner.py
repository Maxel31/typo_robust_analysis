"""The evaluator resumes per condition/pair without changing the paired cohort."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from typo_robust_training.data.records import TypoEdit
from typo_robust_training.evaluation.checkpoints import AdapterDescriptor, PatchWindow
from typo_robust_training.evaluation.config import load_robustness_evaluation_config
from typo_robust_training.evaluation.data import (
    EvaluationCorpusBundle,
    EvaluationCorpusRecord,
    EvaluationDataBundle,
    EvaluationPair,
    FrozenEvaluationProvenance,
)
from typo_robust_training.evaluation.records import (
    CorpusEvaluationObservation,
    EvaluationObservation,
)
from typo_robust_training.evaluation.runner import (
    RobustnessEvaluationRunConfig,
    _experiment_binding,
    _validate_injected_inputs,
    _validate_production_runtime_inventory,
    _validate_production_runtime_provenance,
    run_robustness_evaluation,
)
from typo_robust_training.integrity import sha256_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/gemma4b-evaluation.yaml"
STUDY = PROJECT_ROOT / "configs/robustness-evaluation-v1.yaml"


def _pair(index: int) -> EvaluationPair:
    return EvaluationPair(
        record_id=f"{index:064x}",
        kind="synthetic",
        source="gsm8k",
        source_revision="a" * 40,
        source_split="test",
        source_id=f"gsm8k-{index}",
        group_id=f"gsm8k-{index}",
        role="tune",
        clean_text="The airport is open.",
        typo_text="The arport is open.",
        task="gsm8k",
        answer="12",
        operation="deletion",
        edits=(
            TypoEdit(
                operation="deletion",
                clean_word="airport",
                typo_word="arport",
                clean_char_span=(4, 11),
                typo_char_span=(4, 10),
            ),
        ),
        mechanistic_audit=True,
        metadata=MappingProxyType({"evaluation_condition": "random-2"}),
        strata=("same-task",),
    )


def _bundle(root: Path) -> EvaluationDataBundle:
    return EvaluationDataBundle(
        root=root,
        evaluation_role="tune",
        records=tuple(_pair(index) for index in range(3)),
        manifest_path=root / "tune_manifest.jsonl",
        manifest_sha256="a" * 64,
        evaluation_manifest_sha256="b" * 64,
        data_identity_sha256="7" * 64,
        held_out_operations=("adjacent-transposition",),
    )


def _corpus_bundle(root: Path) -> EvaluationCorpusBundle:
    edit = TypoEdit(
        operation="deletion",
        clean_word="airport",
        typo_word="arport",
        clean_char_span=(4, 11),
        typo_char_span=(4, 10),
    )
    records = (
        EvaluationCorpusRecord(
            record_id=f"{100:064x}",
            kind="clean-corpus",
            source="fineweb_edu",
            source_revision="a" * 40,
            source_split="train",
            source_id="fineweb-1",
            group_id="fineweb-1",
            role="tune",
            clean_text="The airport is open.",
            typo_text=None,
            edits=(),
            metadata=MappingProxyType({}),
        ),
        EvaluationCorpusRecord(
            record_id=f"{101:064x}",
            kind="natural",
            source="github_typo_corpus",
            source_revision="b" * 40,
            source_split="train",
            source_id="natural-1",
            group_id="repository-1",
            role="tune",
            clean_text="The airport is open.",
            typo_text="The arport is open.",
            edits=(edit,),
            metadata=MappingProxyType({}),
        ),
    )
    return EvaluationCorpusBundle(
        root=root,
        evaluation_role="tune",
        records=records,
        manifest_path=root / "tune_corpus_manifest.jsonl",
        manifest_sha256="6" * 64,
    )


def _descriptors(root: Path) -> tuple[AdapterDescriptor, ...]:
    return tuple(
        AdapterDescriptor(
            path=root / f"seed-{seed}" / "adapter",
            condition="localized-state-distillation",
            seed=seed,
            config_sha256="d" * 64,
            training_data_sha256="e" * 64,
            data_identity_sha256="c" * 64,
            localization_sha256="f" * 64,
            adapter_sha256=f"{seed:064x}",
        )
        for seed in (42, 43, 44)
    )


def _v4_descriptors(
    root: Path,
    *,
    condition: str = "probe-transition-output-matching",
    evidence_sha256: str = "8" * 64,
) -> tuple[AdapterDescriptor, ...]:
    return tuple(
        AdapterDescriptor(
            path=root / f"seed-{seed}" / "adapter",
            condition=condition,
            seed=seed,
            config_sha256="d" * 64,
            training_data_sha256="e" * 64,
            data_identity_sha256="c" * 64,
            localization_sha256=None,
            adapter_sha256=f"{seed:064x}",
            method_evidence_sha256=evidence_sha256,
        )
        for seed in (42, 43, 44)
    )


def _completed_adapter(
    root: Path,
    *,
    condition: str,
    seed: int,
    localization_sha256: str | None = None,
    method_evidence_sha256: str | None = None,
) -> Path:
    """Write the minimal completed adapter consumed by the production loader."""

    protocol = load_robustness_evaluation_config(CONFIG)
    output = root / condition / f"seed-{seed}"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": protocol.model,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "training_runtime.json").write_text(
        json.dumps(
            {
                "runtime": "HuggingFaceAdapterTrainingRuntime/v2",
                "model": protocol.model,
                "requested_revision": protocol.model_revision,
                "teacher_revision": protocol.model_revision,
                "student_revision": protocol.model_revision,
                "tokenizer_revision": protocol.model_revision,
                "code_revision": "f" * 40,
                "condition": condition,
                "seed": seed,
                "teacher_frozen": True,
                "student_base_frozen": True,
                **(
                    {"method_evidence_sha256": method_evidence_sha256}
                    if method_evidence_sha256 is not None
                    else {}
                ),
            }
        ),
        encoding="utf-8",
    )
    run = {
        "schema_version": "robustness-adapter-training-run/v1",
        "status": "completed",
        "condition": condition,
        "seed": seed,
        "adapter_sha256": None,
        "method_evidence_sha256": None,
        "config_sha256": "d" * 64,
        "training_data_sha256": "e" * 64,
        "data_identity_sha256": "c" * 64,
        "localization_sha256": localization_sha256,
        "outputs": {"adapter": {"sha256": sha256_tree(adapter)}},
    }
    if method_evidence_sha256 is not None:
        run["method_evidence_sha256"] = method_evidence_sha256
    (output / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return adapter


class _Runtime:
    def __init__(self, factory: _RuntimeFactory, descriptor: AdapterDescriptor | None) -> None:
        self.factory = factory
        self.condition = "base" if descriptor is None else descriptor.condition
        self.seed = None if descriptor is None else descriptor.seed

    def scan_pair(self, pair: EvaluationPair) -> EvaluationObservation:
        if (
            self.factory.fail_after is not None
            and len(self.factory.seen) >= self.factory.fail_after
        ):
            raise RuntimeError("injected evaluation interruption")
        self.factory.seen.append((self.condition, self.seed, pair.record_id))
        base = self.condition == "base"
        typo_correct = int(pair.record_id, 16) == 0 or not base
        patch_gain = 0.5 if base else 0.25
        return EvaluationObservation(
            record_id=pair.record_id,
            condition=self.condition,
            seed=self.seed,
            evaluation_condition=str(pair.metadata["evaluation_condition"]),
            source=pair.source,
            task=pair.task,
            operation=pair.operation,
            edit_count=len(pair.edits),
            mechanistic_audit=pair.mechanistic_audit,
            strata=pair.strata,
            clean_answer="12",
            typo_answer="12" if typo_correct else "11",
            patched_answer="12",
            clean_correct=True,
            typo_correct=typo_correct,
            patched_correct=True,
            target_token_ids=tuple(range(16)),
            untreated_kl_2_16=(1.0,) * 15,
            patched_kl_2_16=(1.0 - patch_gain,) * 15,
            kl_invalid_reason=None,
            patch_invalid_reason=None,
            clean_subtoken_counts=(1,),
            typo_subtoken_counts=(2,),
            tokenization_stratum="fragmentation-increased",
            audit={"mechanistic_audit": pair.mechanistic_audit},
        )

    def provenance(self) -> dict[str, object]:
        return {"runtime": "offline-evaluation-fixture/v1", "condition": self.condition}

    def scan_corpus(
        self,
        record: EvaluationCorpusRecord,
        *,
        max_tokens: int,
    ) -> CorpusEvaluationObservation:
        assert max_tokens == 512
        if (
            self.factory.corpus_fail_after is not None
            and len(self.factory.corpus_seen) >= self.factory.corpus_fail_after
        ):
            raise RuntimeError("injected corpus evaluation interruption")
        self.factory.corpus_seen.append((self.condition, self.seed, record.record_id))
        natural = record.kind == "natural"
        return CorpusEvaluationObservation(
            record_id=record.record_id,
            condition=self.condition,
            seed=self.seed,
            kind=record.kind,
            source=record.source,
            clean_nll_sum=10.0,
            clean_nll_tokens=10,
            typo_nll_sum=10.0 if natural else 0.0,
            typo_nll_tokens=10 if natural else 0,
            base_clean_kl_sum=0.0 if self.condition == "base" else 0.01,
            base_clean_kl_tokens=10,
            natural_clean_typo_kl_sum=(
                1.0 if natural and self.condition == "base" else 0.9 if natural else 0.0
            ),
            natural_clean_typo_kl_tokens=10 if natural else 0,
        )

    def close(self) -> None:
        self.factory.closed.append((self.condition, self.seed))


class _RuntimeFactory:
    def __init__(
        self,
        *,
        fail_after: int | None = None,
        corpus_fail_after: int | None = None,
    ) -> None:
        self.fail_after = fail_after
        self.corpus_fail_after = corpus_fail_after
        self.seen: list[tuple[str, int | None, str]] = []
        self.corpus_seen: list[tuple[str, int | None, str]] = []
        self.closed: list[tuple[str, int | None]] = []

    def __call__(self, descriptor: AdapterDescriptor | None) -> _Runtime:
        return _Runtime(self, descriptor)


def _config(root: Path, output: Path, *, resume: bool) -> RobustnessEvaluationRunConfig:
    return RobustnessEvaluationRunConfig(
        config_path=CONFIG,
        study_protocol_path=STUDY,
        training_data_dir=root,
        evaluation_data_dir=root,
        evaluation_role="tune",
        layer_selection_path=root / "layers.json",
        window_validation_path=None,
        checkpoint_paths=tuple(descriptor.path for descriptor in _descriptors(root)),
        splits=("same-task",),
        gpu_id="3",
        output_dir=output,
        confirm_sealed_role=False,
        resume=resume,
    )


def _production_runtime_provenance(
    *,
    condition: str = "base",
    seed: int | None = None,
) -> dict[str, object]:
    protocol = load_robustness_evaluation_config(CONFIG)
    return {
        "runtime": "HuggingFaceRobustnessEvaluationRuntime/v3",
        "model": protocol.model,
        "requested_revision": protocol.model_revision,
        "model_revision": protocol.model_revision,
        "tokenizer_revision": protocol.model_revision,
        "code_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "condition": condition,
        "seed": seed,
    }


@pytest.mark.parametrize(
    "field",
    ["requested_revision", "model_revision", "tokenizer_revision"],
)
def test_production_runtime_loader_rejects_missing_or_mismatched_revisions(
    field: str,
) -> None:
    protocol = load_robustness_evaluation_config(CONFIG)
    good = _production_runtime_provenance()
    _validate_production_runtime_provenance(
        good,
        protocol=protocol,
        descriptor=None,
    )

    missing = dict(good)
    missing.pop(field)
    with pytest.raises(ValueError, match="model/tokenizer provenance"):
        _validate_production_runtime_provenance(
            missing,
            protocol=protocol,
            descriptor=None,
        )

    mismatched = dict(good)
    mismatched[field] = "f" * 40
    with pytest.raises(ValueError, match="model/tokenizer provenance"):
        _validate_production_runtime_provenance(
            mismatched,
            protocol=protocol,
            descriptor=None,
        )


def test_production_runtime_loader_requires_exact_condition_inventory() -> None:
    protocol = load_robustness_evaluation_config(CONFIG)
    descriptors = _descriptors(Path("/tmp/identity-only"))
    runtime = {"base": _production_runtime_provenance()}

    with pytest.raises(ValueError, match="provenance inventory differs"):
        _validate_production_runtime_inventory(
            runtime,
            protocol=protocol,
            descriptors=descriptors,
            require_complete=True,
        )

    runtime["unexpected"] = _production_runtime_provenance()
    with pytest.raises(ValueError, match="provenance inventory differs"):
        _validate_production_runtime_inventory(
            runtime,
            protocol=protocol,
            descriptors=descriptors,
            require_complete=False,
        )


def test_production_runtime_loader_binds_adapter_and_method_evidence() -> None:
    protocol = load_robustness_evaluation_config(CONFIG)
    descriptor = _v4_descriptors(
        Path("/tmp/identity-only"),
        evidence_sha256="8" * 64,
    )[0]
    provenance = _production_runtime_provenance(
        condition=descriptor.condition,
        seed=descriptor.seed,
    )
    provenance["adapter_sha256"] = descriptor.adapter_sha256
    provenance["method_evidence_sha256"] = descriptor.method_evidence_sha256
    _validate_production_runtime_provenance(
        provenance,
        protocol=protocol,
        descriptor=descriptor,
    )

    for field in ("adapter_sha256", "method_evidence_sha256"):
        tampered = dict(provenance)
        tampered[field] = "f" * 64
        with pytest.raises(ValueError, match="model/tokenizer provenance"):
            _validate_production_runtime_provenance(
                tampered,
                protocol=protocol,
                descriptor=descriptor,
            )


def test_runner_rejects_legacy_evaluation_data_without_frozen_registry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a frozen evaluation registry"):
        run_robustness_evaluation(
            _config(tmp_path, tmp_path / "out", resume=False),
            runtime_factory=_RuntimeFactory(),
        )


def test_method_evidence_is_bound_into_evaluation_identity_without_changing_legacy(
    tmp_path: Path,
) -> None:
    protocol = load_robustness_evaluation_config(CONFIG)
    window = PatchWindow(start=0, stop=6, artifact_sha256="9" * 64)
    legacy = _experiment_binding(
        protocol=protocol,
        study_protocol_sha256="7" * 64,
        descriptors=_descriptors(tmp_path),
        patch_window=window,
        training_sources_sha256=None,
    )
    expected_legacy_payload = {
        "schema_version": "robustness-evaluation-experiment-binding/v2",
        "config_sha256": protocol.config_sha256,
        "study_protocol_sha256": "7" * 64,
        "patch_window_sha256": "9" * 64,
        "patch_layers": list(range(6)),
        "training_sources_sha256": None,
        "adapters": [
            {
                "condition_id": item.condition_id,
                "adapter_sha256": item.adapter_sha256,
                "config_sha256": item.config_sha256,
                "training_data_sha256": item.training_data_sha256,
                "data_identity_sha256": item.data_identity_sha256,
                "localization_sha256": item.localization_sha256,
            }
            for item in sorted(_descriptors(tmp_path), key=lambda item: item.condition_id)
        ],
    }
    expected_legacy = hashlib.sha256(
        json.dumps(
            expected_legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert legacy == expected_legacy
    # Adding the optional field with null must not perturb a historical binding.
    explicit_legacy = tuple(
        replace(item, method_evidence_sha256=None) for item in _descriptors(tmp_path)
    )
    assert (
        _experiment_binding(
            protocol=protocol,
            study_protocol_sha256="7" * 64,
            descriptors=explicit_legacy,
            patch_window=window,
            training_sources_sha256=None,
        )
        == legacy
    )
    first = _experiment_binding(
        protocol=protocol,
        study_protocol_sha256="7" * 64,
        descriptors=_v4_descriptors(tmp_path, evidence_sha256="8" * 64),
        patch_window=window,
        training_sources_sha256=None,
    )
    second = _experiment_binding(
        protocol=protocol,
        study_protocol_sha256="7" * 64,
        descriptors=_v4_descriptors(tmp_path, evidence_sha256="9" * 64),
        patch_window=window,
        training_sources_sha256=None,
    )
    assert first != second


def test_evaluation_resume_rejects_method_evidence_drift(tmp_path: Path) -> None:
    output = tmp_path / "method-resume"
    bundle = _bundle(tmp_path)
    corpus = _corpus_bundle(tmp_path)
    window = PatchWindow(start=0, stop=6, artifact_sha256="9" * 64)
    descriptors = _v4_descriptors(tmp_path, evidence_sha256="8" * 64)
    config = replace(
        _config(tmp_path, output, resume=False),
        checkpoint_paths=tuple(item.path for item in descriptors),
    )
    with pytest.raises(RuntimeError, match="injected evaluation interruption"):
        run_robustness_evaluation(
            config,
            runtime_factory=_RuntimeFactory(fail_after=1),
            data_bundle=bundle,
            corpus_bundle=corpus,
            descriptors=descriptors,
            patch_window=window,
        )
    failed = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert failed["adapters"][0]["method_evidence_sha256"] == "8" * 64

    changed = _v4_descriptors(tmp_path, evidence_sha256="7" * 64)
    with pytest.raises(ValueError, match="resume run binding differs"):
        run_robustness_evaluation(
            replace(
                config,
                resume=True,
                checkpoint_paths=tuple(item.path for item in changed),
            ),
            runtime_factory=_RuntimeFactory(),
            data_bundle=bundle,
            corpus_bundle=corpus,
            descriptors=changed,
            patch_window=window,
        )


def test_runner_resume_is_byte_identical_to_uninterrupted_evaluation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    descriptors = _descriptors(tmp_path)
    window = PatchWindow(
        start=0,
        stop=6,
        artifact_sha256="9" * 64,
        localization_sha256="f" * 64,
    )

    uninterrupted_factory = _RuntimeFactory()
    uninterrupted = run_robustness_evaluation(
        _config(tmp_path, tmp_path / "uninterrupted", resume=False),
        runtime_factory=uninterrupted_factory,
        data_bundle=bundle,
        corpus_bundle=_corpus_bundle(tmp_path),
        descriptors=descriptors,
        patch_window=window,
    )
    assert uninterrupted.records == 12
    assert uninterrupted.corpus_records == 8
    assert len(uninterrupted_factory.seen) == 12
    assert len(uninterrupted_factory.corpus_seen) == 8

    output = tmp_path / "resumed"
    interrupted_factory = _RuntimeFactory(fail_after=5)
    with pytest.raises(RuntimeError, match="injected evaluation interruption"):
        run_robustness_evaluation(
            _config(tmp_path, output, resume=False),
            runtime_factory=interrupted_factory,
            data_bundle=bundle,
            corpus_bundle=_corpus_bundle(tmp_path),
            descriptors=descriptors,
            patch_window=window,
        )
    failed = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["started_at"]

    resumed_factory = _RuntimeFactory()
    resumed = run_robustness_evaluation(
        _config(tmp_path, output, resume=True),
        runtime_factory=resumed_factory,
        data_bundle=bundle,
        corpus_bundle=_corpus_bundle(tmp_path),
        descriptors=descriptors,
        patch_window=window,
    )
    assert interrupted_factory.seen + resumed_factory.seen == uninterrupted_factory.seen
    assert resumed.records_path.read_bytes() == uninterrupted.records_path.read_bytes()
    assert (
        resumed.corpus_records_path.read_bytes() == uninterrupted.corpus_records_path.read_bytes()
    )
    assert resumed.report_path.read_bytes() == uninterrupted.report_path.read_bytes()
    assert resumed.corpus_records == uninterrupted.corpus_records
    assert (
        interrupted_factory.corpus_seen + resumed_factory.corpus_seen
        == uninterrupted_factory.corpus_seen
    )
    assert len(interrupted_factory.closed) == 2
    assert len(resumed_factory.closed) == 3
    completed = json.loads(resumed.run_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["started_at"] == failed["started_at"]
    assert completed["access_binding_sha256"]
    assert completed["experiment_binding_sha256"]
    assert set(completed["runtime"]) == {
        "base",
        "localized-state-distillation:seed-42",
        "localized-state-distillation:seed-43",
        "localized-state-distillation:seed-44",
    }


def test_runner_resumes_a_corpus_interruption_without_recomputing_records(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    descriptors = _descriptors(tmp_path)
    corpus_bundle = _corpus_bundle(tmp_path)
    window = PatchWindow(0, 6, "9" * 64, "f" * 64)
    reference_factory = _RuntimeFactory()
    reference = run_robustness_evaluation(
        _config(tmp_path, tmp_path / "reference", resume=False),
        runtime_factory=reference_factory,
        data_bundle=bundle,
        corpus_bundle=corpus_bundle,
        descriptors=descriptors,
        patch_window=window,
    )

    output = tmp_path / "corpus-resume"
    interrupted_factory = _RuntimeFactory(corpus_fail_after=3)
    with pytest.raises(RuntimeError, match="corpus evaluation interruption"):
        run_robustness_evaluation(
            _config(tmp_path, output, resume=False),
            runtime_factory=interrupted_factory,
            data_bundle=bundle,
            corpus_bundle=corpus_bundle,
            descriptors=descriptors,
            patch_window=window,
        )

    resumed_factory = _RuntimeFactory()
    resumed = run_robustness_evaluation(
        _config(tmp_path, output, resume=True),
        runtime_factory=resumed_factory,
        data_bundle=bundle,
        corpus_bundle=corpus_bundle,
        descriptors=descriptors,
        patch_window=window,
    )

    assert interrupted_factory.seen + resumed_factory.seen == reference_factory.seen
    assert (
        interrupted_factory.corpus_seen + resumed_factory.corpus_seen
        == reference_factory.corpus_seen
    )
    assert resumed.records_path.read_bytes() == reference.records_path.read_bytes()
    assert resumed.corpus_records_path.read_bytes() == reference.corpus_records_path.read_bytes()
    assert resumed.report_path.read_bytes() == reference.report_path.read_bytes()


def test_runner_refuses_nonempty_output_without_resume(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        run_robustness_evaluation(
            _config(tmp_path, output, resume=False),
            runtime_factory=_RuntimeFactory(),
            data_bundle=_bundle(tmp_path),
            corpus_bundle=_corpus_bundle(tmp_path),
            descriptors=_descriptors(tmp_path),
            patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
        )


def test_runner_rejects_gate_drift_from_frozen_study(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["metrics"]["bootstrap_seed"] = 43
    drifted = tmp_path / "drifted-evaluation.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    config = replace(
        _config(tmp_path, tmp_path / "drifted-output", resume=False),
        config_path=drifted,
    )

    with pytest.raises(ValueError, match="runtime and study protocols differ"):
        run_robustness_evaluation(
            config,
            runtime_factory=_RuntimeFactory(),
            data_bundle=_bundle(tmp_path),
            corpus_bundle=_corpus_bundle(tmp_path),
            descriptors=_descriptors(tmp_path),
            patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
        )


def test_runner_keeps_training_and_frozen_evaluation_identities_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_root = tmp_path / "frozen-evaluation"
    evaluation_root.mkdir()
    (evaluation_root / "registry.json").write_text("{}\n", encoding="utf-8")
    config = replace(
        _config(tmp_path, tmp_path / "evaluation-output", resume=False),
        evaluation_data_dir=evaluation_root,
    )
    bundle = _bundle(evaluation_root)
    corpus_bundle = _corpus_bundle(evaluation_root)
    descriptors = _descriptors(tmp_path)
    loader_calls: list[tuple[Path, dict[str, object]]] = []

    monkeypatch.setattr(
        "typo_robust_training.training.data.load_training_data_provenance",
        lambda root: type(
            "TrainingBundle",
            (),
            {
                "training_data_sha256": "e" * 64,
                "data_identity_sha256": "c" * 64,
                "artifact_sha256": {"training_sources.jsonl": "1" * 64},
            },
        )(),
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runner.load_frozen_evaluation_provenance",
        lambda root, **_kwargs: FrozenEvaluationProvenance(
            root=Path(root),
            data_identity_sha256="7" * 64,
            exclusion_artifact_sha256=MappingProxyType(
                {"training_sources.jsonl": "1" * 64}
            ),
        ),
    )

    def load_frozen(root: Path, **kwargs: object) -> EvaluationDataBundle:
        loader_calls.append((root, dict(kwargs)))
        return bundle

    monkeypatch.setattr(
        "typo_robust_training.evaluation.runner.load_evaluation_bundle",
        load_frozen,
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runner.load_evaluation_corpus_bundle",
        lambda _root, **_kwargs: corpus_bundle,
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runner.complete_evaluation_role",
        lambda *_args, **_kwargs: None,
    )
    result = run_robustness_evaluation(
        config,
        runtime_factory=_RuntimeFactory(),
        descriptors=descriptors,
        patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
    )

    assert result.records == 12
    assert loader_calls[0][0] == evaluation_root
    assert len(str(loader_calls[0][1]["study_protocol_sha256"])) == 64
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["training_data_identity_sha256"] == "c" * 64
    assert run["evaluation_data_identity_sha256"] == "7" * 64
    assert run["training_sources_sha256"] == "1" * 64


def test_runner_rejects_frozen_population_excluding_different_training_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_root = tmp_path / "frozen-evaluation"
    evaluation_root.mkdir()
    (evaluation_root / "registry.json").write_text("{}\n", encoding="utf-8")
    config = replace(
        _config(tmp_path, tmp_path / "evaluation-output", resume=False),
        evaluation_data_dir=evaluation_root,
    )
    monkeypatch.setattr(
        "typo_robust_training.training.data.load_training_data_provenance",
        lambda root: type(
            "TrainingBundle",
            (),
            {
                "training_data_sha256": "e" * 64,
                "data_identity_sha256": "c" * 64,
                "artifact_sha256": {"training_sources.jsonl": "1" * 64},
            },
        )(),
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runner.load_frozen_evaluation_provenance",
        lambda root, **_kwargs: FrozenEvaluationProvenance(
            root=Path(root),
            data_identity_sha256="7" * 64,
            exclusion_artifact_sha256=MappingProxyType(
                {"training_sources.jsonl": "2" * 64}
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="frozen evaluation exclusions were built from different training sources",
    ):
        run_robustness_evaluation(
            config,
            runtime_factory=_RuntimeFactory(),
            descriptors=_descriptors(tmp_path),
            patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
        )


def test_runner_rejects_evaluation_data_or_patch_evidence_drift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    descriptors = _descriptors(tmp_path)
    config = _config(tmp_path, tmp_path / "evaluation", resume=False)

    wrong_data = (replace(descriptors[0], data_identity_sha256="0" * 64), *descriptors[1:])
    with pytest.raises(ValueError, match="different training data identities"):
        run_robustness_evaluation(
            config,
            runtime_factory=_RuntimeFactory(),
            data_bundle=bundle,
            corpus_bundle=_corpus_bundle(tmp_path),
            descriptors=wrong_data,
            patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
        )

    with pytest.raises(ValueError, match="localization evidence"):
        run_robustness_evaluation(
            config,
            runtime_factory=_RuntimeFactory(),
            data_bundle=bundle,
            corpus_bundle=_corpus_bundle(tmp_path),
            descriptors=descriptors,
            patch_window=PatchWindow(0, 6, "9" * 64, "0" * 64),
        )


def test_sealed_evaluation_requires_localized_checkpoint_for_every_seed(
    tmp_path: Path,
) -> None:
    descriptors = _descriptors(tmp_path)
    config = replace(
        _config(tmp_path, tmp_path / "sealed", resume=False),
        evaluation_role="pre-pr-gate",
        checkpoint_paths=(descriptors[0].path,),
        splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
        confirm_sealed_role=True,
    )

    with pytest.raises(ValueError, match="complete seed inventory"):
        _validate_injected_inputs(
            config,
            protocol=load_robustness_evaluation_config(CONFIG),
            data_bundle=None,
            corpus_bundle=None,
            descriptors=(descriptors[0],),
            patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
        )


@pytest.mark.parametrize(
    "condition",
    (
        "probe-transition-output-matching",
        "probe-transition-single-layer-state-distillation",
        "causal-probe-subspace-distillation",
    ),
)
def test_non_tune_evaluation_requires_every_seed_for_each_v4_condition_present(
    tmp_path: Path,
    condition: str,
) -> None:
    legacy = _descriptors(tmp_path / "legacy")
    method = _v4_descriptors(tmp_path / "method", condition=condition)
    incomplete = (*legacy, method[0])
    config = replace(
        _config(tmp_path, tmp_path / "sealed", resume=False),
        evaluation_role="pre-pr-gate",
        checkpoint_paths=tuple(item.path for item in incomplete),
        splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
        confirm_sealed_role=True,
    )

    with pytest.raises(ValueError, match=rf"{condition}.*complete seed inventory"):
        _validate_injected_inputs(
            config,
            protocol=load_robustness_evaluation_config(CONFIG),
            data_bundle=None,
            corpus_bundle=None,
            descriptors=incomplete,
            patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
        )

    complete = (*legacy, *method)
    complete_config = replace(
        config,
        checkpoint_paths=tuple(item.path for item in complete),
    )
    resolved, _, _, _ = _validate_injected_inputs(
        complete_config,
        protocol=load_robustness_evaluation_config(CONFIG),
        data_bundle=None,
        corpus_bundle=None,
        descriptors=complete,
        patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
    )
    assert resolved == complete


def test_non_tune_production_loader_path_rejects_one_of_three_v4_seeds(
    tmp_path: Path,
) -> None:
    legacy_paths = tuple(
        _completed_adapter(
            tmp_path,
            condition="localized-state-distillation",
            seed=seed,
            localization_sha256="f" * 64,
        )
        for seed in (42, 43, 44)
    )
    method_path = _completed_adapter(
        tmp_path,
        condition="probe-transition-output-matching",
        seed=42,
        method_evidence_sha256="8" * 64,
    )
    checkpoint_paths = (*legacy_paths, method_path)
    config = replace(
        _config(tmp_path, tmp_path / "sealed-production", resume=False),
        evaluation_role="pre-pr-gate",
        checkpoint_paths=checkpoint_paths,
        splits=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
        confirm_sealed_role=True,
    )

    with pytest.raises(
        ValueError,
        match="probe-transition-output-matching.*complete seed inventory",
    ):
        _validate_injected_inputs(
            config,
            protocol=load_robustness_evaluation_config(CONFIG),
            data_bundle=None,
            corpus_bundle=None,
            descriptors=None,
            patch_window=PatchWindow(0, 6, "9" * 64, "f" * 64),
        )
