from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.data.records import record_id_for
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.probe.cohort_builder import (
    ProbeTransitionDataBuildConfig,
    load_probe_cohort_template,
    probe_parent_source_sha256,
    probe_source_group_sha256,
    run_build_probe_transition_data,
)
from typo_robust_training.probe.config import load_probe_producer_config
from typo_robust_training.probe.producer import _load_classes, _load_cohort
from typo_robust_training.training.pairs import TrainingSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_TEMPLATE = (
    PROJECT_ROOT / "configs" / "proposals" / "mistral7b-probe-transition-data.template.yaml"
)
SOURCE_REVISION = "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _small_template(tmp_path: Path) -> Path:
    value = json.loads(PRODUCTION_TEMPLATE.read_text(encoding="utf-8"))
    value["cohorts"].update(
        {
            "class_count": 2,
            "fit_records_per_class": 4,
            "paired_records_per_class": 9,
            "min_word_letters": 4,
            "max_word_letters": 12,
            "max_text_characters": 128,
        }
    )
    value["perturbations"]["stratum_counts"] = {
        "keyboard-neighbor-substitution|1|same": 2,
        "keyboard-neighbor-substitution|1|plus-one": 2,
        "keyboard-neighbor-substitution|1|plus-two-or-more": 2,
        "deletion|1|same": 3,
        "deletion|1|plus-one": 3,
        "duplication|1|plus-one": 3,
        "duplication|1|plus-two-or-more": 3,
    }
    return _write_json(tmp_path / "template.json", value)


def _source_row(label: str, index: int) -> dict[str, object]:
    source_id = f"fineweb_edu:{label}:{index:03d}"
    text = f"aa {label} zz {index:03d}."
    return {
        "schema_version": "robustness-clean-record/v1",
        "kind": "clean",
        "record_id": record_id_for(
            source="fineweb_edu",
            source_revision=SOURCE_REVISION,
            source_id=source_id,
        ),
        "source": "fineweb_edu",
        "source_revision": SOURCE_REVISION,
        "source_split": "train",
        "source_id": source_id,
        "group_id": f"fineweb_edu:group:{label}:{index:03d}",
        "split": "train",
        "text": text,
        "task": None,
        "answer": None,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "normalized_content_sha256": normalized_content_sha256(text),
        "metadata": {"fixture_index": index},
        "token_count": 8,
    }


def _source_manifest(tmp_path: Path, *, per_class: int = 180) -> Path:
    path = tmp_path / "clean-sources.jsonl"
    rows = [_source_row(label, index) for label in ("alpha", "beta") for index in range(per_class)]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )
    return path


def _protected_registry(
    tmp_path: Path,
    *,
    source_row: dict[str, object] | None = None,
) -> Path:
    rows = []
    for tier in ("training", "localization", "tune", "pre-pr", "sealed"):
        groups = [_sha(f"protected-group-{tier}")]
        parents = [_sha(f"protected-parent-{tier}")]
        contents = [_sha(f"protected-content-{tier}")]
        if source_row is not None and tier == "training":
            source = TrainingSource.from_dict(source_row)
            groups.append(probe_source_group_sha256(source))
            parents.append(probe_parent_source_sha256(source))
            contents.append(str(source_row["normalized_content_sha256"]))
        rows.append(
            {
                "tier": tier,
                "source_group_sha256": groups,
                "parent_source_sha256": parents,
                "normalized_content_sha256": contents,
            }
        )
    return _write_json(
        tmp_path / "protected.json",
        {"schema_version": "typo-protected-split-registry/v1", "registries": rows},
    )


def _fully_protected_registry(tmp_path: Path, source_manifest: Path) -> Path:
    tiers = ("training", "localization", "tune", "pre-pr", "sealed")
    registries = {
        tier: {
            "tier": tier,
            "source_group_sha256": [_sha(f"protected-group-{tier}")],
            "parent_source_sha256": [_sha(f"protected-parent-{tier}")],
            "normalized_content_sha256": [_sha(f"protected-content-{tier}")],
        }
        for tier in tiers
    }
    for index, line in enumerate(source_manifest.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        source = TrainingSource.from_dict(row)
        registry = registries[tiers[index % len(tiers)]]
        registry["source_group_sha256"].append(probe_source_group_sha256(source))
        registry["parent_source_sha256"].append(probe_parent_source_sha256(source))
        registry["normalized_content_sha256"].append(row["normalized_content_sha256"])
    return _write_json(
        tmp_path / "fully-protected.json",
        {
            "schema_version": "typo-protected-split-registry/v1",
            "registries": list(registries.values()),
        },
    )


class _Counter:
    def bucket(self, *, clean_text: str, typo_text: str) -> str:
        del typo_text
        index = int(clean_text.rsplit(" ", 1)[-1].removesuffix("."))
        return ("same", "plus-one", "plus-two-or-more")[index % 3]

    def provenance(self) -> dict[str, object]:
        return {
            "provider": "fixture-tokenizer-only/v1",
            "model_outputs_observed": False,
        }


def _run(
    tmp_path: Path,
    *,
    source: Path | None = None,
    protected: Path | None = None,
    output_name: str = "output",
) -> object:
    attestation = _write_json(tmp_path / f"attestation-{output_name}.json", {})
    return run_build_probe_transition_data(
        ProbeTransitionDataBuildConfig(
            template_path=_small_template(tmp_path),
            source_manifest_path=source or _source_manifest(tmp_path),
            protected_registry_path=protected or _protected_registry(tmp_path),
            tokenizer_attestation_path=attestation,
            output_dir=tmp_path / output_name,
        ),
        counter=_Counter(),
        code_revision="c" * 40,
    )


def test_builder_emits_existing_producer_schemas_without_model_outputs(tmp_path: Path) -> None:
    result = _run(tmp_path)

    protocol = load_probe_producer_config(result.producer_config_path)
    labels = _load_classes(result.class_inventory_path)
    fit = _load_cohort(result.fit_manifest_path, role="fit", labels=labels)
    selection = _load_cohort(result.selection_manifest_path, role="selection", labels=labels)
    validation = _load_cohort(result.validation_manifest_path, role="validation", labels=labels)

    assert result.classes == 2
    assert labels == ("beta", "alpha") or labels == ("alpha", "beta")
    assert len(fit) == 8
    assert len(selection) == len(validation) == 18
    assert result.records == 44
    assert all(record.typo_text is None for record in fit)
    assert set(record.edit_type for record in selection) == {
        "keyboard-neighbor-substitution",
        "deletion",
        "duplication",
    }
    assert set(record.token_inflation_bucket for record in selection) == {
        "same",
        "plus-one",
        "plus-two-or-more",
    }
    assert protocol.model == "mistralai/Mistral-7B-v0.1"
    assert protocol.model_revision == "7231864981174d9bee8c7687c24c8344414eae6b"
    assert protocol.records_per_class == {"fit": 4, "selection": 9, "validation": 9}
    assert protocol.stratum_counts["selection"] == {
        "keyboard-neighbor-substitution|1|same": 2,
        "keyboard-neighbor-substitution|1|plus-one": 2,
        "keyboard-neighbor-substitution|1|plus-two-or-more": 2,
        "deletion|1|same": 3,
        "deletion|1|plus-one": 3,
        "duplication|1|plus-one": 3,
        "duplication|1|plus-two-or-more": 3,
    }
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["model_outputs_observed"] is False
    assert run["token_counter"]["model_outputs_observed"] is False
    assert (
        run["source"]["manifest_sha256"]
        == hashlib.sha256((tmp_path / "clean-sources.jsonl").read_bytes()).hexdigest()
    )
    self_hash = run.pop("self_hash")
    assert self_hash == {
        "algorithm": "sha256",
        "canonicalization": "canonical-json-without-self-hash/v1",
        "sha256": hashlib.sha256(
            json.dumps(
                run,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    feasibility = json.loads(result.feasibility_report_path.read_text(encoding="utf-8"))
    assert feasibility["model_outputs_observed"] is False
    assert feasibility["simulated_cohort"]["paired_records"] == 18

    identities = []
    for role in (fit, selection, validation):
        identities.append(
            {
                identity
                for record in role
                for identity in (
                    record.source_group_sha256,
                    record.parent_source_sha256,
                    record.normalized_clean_sha256,
                    record.normalized_noisy_sha256,
                )
                if identity is not None
            }
        )
    assert not identities[0] & identities[1]
    assert not identities[0] & identities[2]
    assert not identities[1] & identities[2]


def test_builder_is_byte_deterministic_for_the_same_inputs(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path)
    protected = _protected_registry(tmp_path)
    first = _run(
        tmp_path,
        source=source,
        protected=protected,
        output_name="first",
    )
    second = _run(
        tmp_path,
        source=source,
        protected=protected,
        output_name="second",
    )

    for name in (
        "class_inventory_path",
        "fit_manifest_path",
        "selection_manifest_path",
        "validation_manifest_path",
        "protected_registry_path",
        "feasibility_report_path",
        "producer_config_path",
        "run_path",
    ):
        assert getattr(first, name).read_bytes() == getattr(second, name).read_bytes()


def test_builder_balances_classes_and_strata_globally_not_as_impossible_cross_product(
    tmp_path: Path,
) -> None:
    template_value = json.loads(_small_template(tmp_path).read_text(encoding="utf-8"))
    template_value["perturbations"]["stratum_counts"] = {
        f"{operation}|1|{bucket}": 3
        for operation in (
            "keyboard-neighbor-substitution",
            "deletion",
            "duplication",
        )
        for bucket in ("same", "plus-one")
    }
    template = _write_json(tmp_path / "global-strata-template.json", template_value)

    class _ClassConstrainedCounter(_Counter):
        def bucket(self, *, clean_text: str, typo_text: str) -> str:
            del typo_text
            return "same" if " alpha " in clean_text else "plus-one"

    result = run_build_probe_transition_data(
        ProbeTransitionDataBuildConfig(
            template_path=template,
            source_manifest_path=_source_manifest(tmp_path),
            protected_registry_path=_protected_registry(tmp_path),
            tokenizer_attestation_path=_write_json(tmp_path / "attestation.json", {}),
            output_dir=tmp_path / "global-strata-output",
        ),
        counter=_ClassConstrainedCounter(),
        code_revision="c" * 40,
    )
    labels = _load_classes(result.class_inventory_path)
    selection = _load_cohort(result.selection_manifest_path, role="selection", labels=labels)
    buckets_by_class = {
        class_id: {
            record.token_inflation_bucket for record in selection if record.class_id == class_id
        }
        for class_id in range(2)
    }
    assert set(map(frozenset, buckets_by_class.values())) == {
        frozenset(("same",)),
        frozenset(("plus-one",)),
    }


def test_builder_filters_protected_parent_group_and_content_transitively(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path)
    protected_row = _source_row("alpha", 0)
    protected = _protected_registry(tmp_path, source_row=protected_row)
    result = _run(tmp_path, source=source, protected=protected)
    protected_source = TrainingSource.from_dict(protected_row)
    forbidden = {
        probe_source_group_sha256(protected_source),
        probe_parent_source_sha256(protected_source),
        protected_row["normalized_content_sha256"],
    }

    for path, role in (
        (result.fit_manifest_path, "fit"),
        (result.selection_manifest_path, "selection"),
        (result.validation_manifest_path, "validation"),
    ):
        labels = _load_classes(result.class_inventory_path)
        records = _load_cohort(path, role=role, labels=labels)
        observed = {
            value
            for record in records
            for value in (
                record.source_group_sha256,
                record.parent_source_sha256,
                record.normalized_clean_sha256,
                record.normalized_noisy_sha256,
            )
            if value is not None
        }
        assert not forbidden & observed


def test_builder_fails_closed_on_insufficient_capacity_without_partial_output(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path, per_class=20)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="insufficient disjoint .*capacity"):
        _run(tmp_path, source=source)
    assert not output.exists()


def test_builder_rejects_reusing_sources_protected_across_all_five_tiers(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path, per_class=20)
    protected = _fully_protected_registry(tmp_path, source)
    output = tmp_path / "fully-protected-output"
    with pytest.raises(ValueError, match="insufficient disjoint .*capacity"):
        _run(
            tmp_path,
            source=source,
            protected=protected,
            output_name=output.name,
        )
    assert not output.exists()


def test_builder_rejects_source_content_tamper_and_symlinks(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path)
    rows = source.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0])
    changed["text"] += " changed"
    rows[0] = json.dumps(changed, sort_keys=True)
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content hash differs"):
        _run(tmp_path, source=source)

    valid = _source_manifest(tmp_path)
    link = tmp_path / "source-link.jsonl"
    link.symlink_to(valid)
    with pytest.raises(ValueError, match="must be one regular file"):
        _run(tmp_path, source=link, output_name="symlink-output")


def test_builder_rejects_protected_tier_overlap_and_counter_output_access(
    tmp_path: Path,
) -> None:
    protected = _protected_registry(tmp_path)
    value = json.loads(protected.read_text(encoding="utf-8"))
    shared = _sha("cross-tier")
    value["registries"][0]["normalized_content_sha256"].append(shared)
    value["registries"][1]["source_group_sha256"].append(shared)
    _write_json(protected, value)
    with pytest.raises(ValueError, match="overlap transitively"):
        _run(tmp_path, protected=protected)

    class _BadCounter(_Counter):
        def provenance(self) -> dict[str, object]:
            return {"provider": "bad/v1", "model_outputs_observed": True}

    with pytest.raises(ValueError, match="no model outputs"):
        run_build_probe_transition_data(
            ProbeTransitionDataBuildConfig(
                template_path=_small_template(tmp_path),
                source_manifest_path=_source_manifest(tmp_path),
                protected_registry_path=_protected_registry(tmp_path),
                tokenizer_attestation_path=_write_json(tmp_path / "attestation.json", {}),
                output_dir=tmp_path / "bad-counter-output",
            ),
            counter=_BadCounter(),
            code_revision="c" * 40,
        )


def test_template_and_cli_freeze_exact_mistral_inputs(tmp_path: Path) -> None:
    protocol = load_probe_cohort_template(PRODUCTION_TEMPLATE)
    assert protocol.model == "mistralai/Mistral-7B-v0.1"
    assert protocol.model_revision == "7231864981174d9bee8c7687c24c8344414eae6b"
    assert protocol.decoder_layers == 32
    assert protocol.hidden_size == 4096
    assert protocol.class_count == 17
    assert protocol.paired_records_per_class == 9
    assert sum(protocol.stratum_counts.values()) == 153
    assert protocol.operations == (
        "keyboard-neighbor-substitution",
        "deletion",
        "duplication",
    )

    value = json.loads(PRODUCTION_TEMPLATE.read_text(encoding="utf-8"))
    value["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        load_probe_cohort_template(_write_json(tmp_path / "extra.json", value))

    wrong_quota = json.loads(PRODUCTION_TEMPLATE.read_text(encoding="utf-8"))
    wrong_quota["perturbations"]["stratum_counts"]["keyboard-neighbor-substitution|1|same"] += 1
    with pytest.raises(ValueError, match="global stratum counts"):
        load_probe_cohort_template(_write_json(tmp_path / "wrong-quota.json", wrong_quota))

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)
    args = parser.parse_args(
        [
            "build-probe-transition-data",
            "--template",
            "template.json",
            "--source-manifest",
            "source.jsonl",
            "--protected-registry",
            "protected.json",
            "--tokenizer-attestation",
            "tokenizer.json",
            "--output-dir",
            "cohorts",
        ]
    )
    assert args.command == "build-probe-transition-data"
    assert args.template == Path("template.json")
    assert args.source_manifest == Path("source.jsonl")
    assert args.protected_registry == Path("protected.json")
    assert args.tokenizer_attestation == Path("tokenizer.json")
    assert args.output_dir == Path("cohorts")
