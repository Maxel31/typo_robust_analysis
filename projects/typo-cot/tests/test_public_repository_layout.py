"""Repository-hygiene contract for the public reproduction package."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import tomllib
from pathlib import Path

import typo_cot.cli as cli_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
PROJECT_PREFIX = "projects/typo-cot/"

LEGACY_PROJECT_DIRECTORIES = {"analysis", "configs", "scripts"}
LEGACY_PACKAGE_DIRECTORIES = {
    "analysis",
    "attribution",
    "attribution_family",
    "defense",
    "intervention",
    "nocot",
    "perturbation",
    "repair",
    "visualization",
}
LEGACY_PACKAGE_MODULES = {"config.py", "registry.py", "sharding.py"}
LEGACY_PACKAGE_ENTRIES = LEGACY_PACKAGE_DIRECTORIES | LEGACY_PACKAGE_MODULES
LEGACY_DOCUMENTS = {
    "all_results_by_setting.md",
    "architecture.drawio",
    "attnlrp_mechanism.drawio",
    "data_provenance.md",
    "dev_notes_01_03_transplant.md",
    "dev_notes_02_target_deletion.md",
    "dev_notes_04_fixed_target.md",
    "dev_notes_05_matched_control.md",
    "dev_notes_06_attribution.md",
    "dev_notes_07_correctors.md",
    "dev_notes_08_fine.md",
    "dev_notes_08_patching.md",
    "dev_notes_09_inner_repair.md",
    "dev_notes_10_scope.md",
    "dev_notes_15_patch_freegen.md",
    "dev_notes_exp10_math500.md",
    "dev_notes_exp10_natural_typo.md",
    "dev_notes_exp10_scope.md",
    "dev_notes_step0.md",
    "discussion_family_effects.md",
    "experiment_details.md",
    "experiment_flow.drawio",
    "experiment_plan.md",
    "experiments_11_18_plan.md",
    "followup_plan_20260719.md",
    "hypothesis_registry.md",
    "implementation_reference.md",
    "improvement_plan.md",
    "legacy-development-readme.md",
    "loo_definition_survey.md",
    "paper_outline.md",
    "rebuttal_draft.md",
    "size_ladder_plan.md",
    "v1_run_manifest.md",
    "work_items.md",
}
LEGACY_DEPENDENCIES = {
    "pydantic",
    "scipy",
    "pingouin",
    "pyspellchecker",
    "statsmodels",
    "typo-utils",
    "wordfreq",
}
PUBLIC_SOURCE_FILES = {
    Path("src/typo_cot/__init__.py"),
    Path("src/typo_cot/cli.py"),
    Path("src/typo_cot/data/__init__.py"),
    Path("src/typo_cot/data/loader.py"),
    Path("src/typo_cot/evaluation/__init__.py"),
    Path("src/typo_cot/evaluation/extractor.py"),
    Path("src/typo_cot/evaluation/fallback.py"),
    Path("src/typo_cot/experiments/__init__.py"),
    Path("src/typo_cot/experiments/catalog.py"),
    Path("src/typo_cot/experiments/cot_swap/__init__.py"),
    Path("src/typo_cot/experiments/cot_swap/planning.py"),
    Path("src/typo_cot/experiments/cot_swap/runner.py"),
    Path("src/typo_cot/experiments/cot_swap/runtime.py"),
    Path("src/typo_cot/experiments/fixed_window_answer_patching/__init__.py"),
    Path("src/typo_cot/experiments/fixed_window_answer_patching/metrics.py"),
    Path("src/typo_cot/experiments/fixed_window_answer_patching/patching.py"),
    Path("src/typo_cot/experiments/fixed_window_answer_patching/runner.py"),
    Path("src/typo_cot/experiments/fixed_window_answer_patching/runtime.py"),
    Path("src/typo_cot/experiments/layerwise_answer_patching/__init__.py"),
    Path("src/typo_cot/experiments/layerwise_answer_patching/metrics.py"),
    Path("src/typo_cot/experiments/layerwise_answer_patching/patching.py"),
    Path("src/typo_cot/experiments/layerwise_answer_patching/runner.py"),
    Path("src/typo_cot/experiments/layerwise_answer_patching/runtime.py"),
    Path("src/typo_cot/experiments/layerwise_kl_patching/__init__.py"),
    Path("src/typo_cot/experiments/layerwise_kl_patching/metrics.py"),
    Path("src/typo_cot/experiments/layerwise_kl_patching/patching.py"),
    Path("src/typo_cot/experiments/layerwise_kl_patching/runner.py"),
    Path("src/typo_cot/experiments/layerwise_kl_patching/runtime.py"),
    Path("src/typo_cot/experiments/patch_coordinate_controls/__init__.py"),
    Path("src/typo_cot/experiments/patch_coordinate_controls/metrics.py"),
    Path("src/typo_cot/experiments/patch_coordinate_controls/planning.py"),
    Path("src/typo_cot/experiments/patch_coordinate_controls/runner.py"),
    Path("src/typo_cot/experiments/patch_coordinate_controls/runtime.py"),
    Path("src/typo_cot/experiments/patch_position_controls/__init__.py"),
    Path("src/typo_cot/experiments/patch_position_controls/planning.py"),
    Path("src/typo_cot/experiments/patch_position_controls/runner.py"),
    Path("src/typo_cot/experiments/patch_position_controls/runtime.py"),
    Path("src/typo_cot/experiments/patch_text_combination/__init__.py"),
    Path("src/typo_cot/experiments/patch_text_combination/planning.py"),
    Path("src/typo_cot/experiments/patch_text_combination/runner.py"),
    Path("src/typo_cot/experiments/patch_text_combination/runtime.py"),
    Path("src/typo_cot/experiments/prepare_edited_pairs/__init__.py"),
    Path("src/typo_cot/experiments/prepare_edited_pairs/protocol.py"),
    Path("src/typo_cot/experiments/prepare_edited_pairs/runner.py"),
    Path("src/typo_cot/experiments/prepare_edited_pairs/runtime.py"),
    Path("src/typo_cot/experiments/targeting_fidelity_audit/__init__.py"),
    Path("src/typo_cot/experiments/targeting_fidelity_audit/runner.py"),
    Path("src/typo_cot/lrp/__init__.py"),
    Path("src/typo_cot/lrp/analyzer.py"),
    Path("src/typo_cot/models/__init__.py"),
    Path("src/typo_cot/models/prompts.py"),
    Path("src/typo_cot/models/wrapper.py"),
}


def _tracked_existing_paths() -> set[Path]:
    """Return tracked project paths that still exist in the worktree."""
    completed = subprocess.run(
        ["git", "ls-files", "--", "projects/typo-cot"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    relative_paths: set[Path] = set()
    for line in completed.stdout.splitlines():
        if not line.startswith(PROJECT_PREFIX):
            continue
        relative = Path(line.removeprefix(PROJECT_PREFIX))
        if (PROJECT_ROOT / relative).exists():
            relative_paths.add(relative)
    return relative_paths


def _dependency_name(requirement: str) -> str:
    """Extract a normalized distribution name from a simple requirement."""
    boundary = min(
        (requirement.find(marker) for marker in "[<>=!~ @" if marker in requirement),
        default=len(requirement),
    )
    return requirement[:boundary].strip().lower().replace("_", "-")


def test_legacy_experiment_bundles_are_not_tracked() -> None:
    tracked = _tracked_existing_paths()
    top_level = {path.parts[0] for path in tracked}
    assert top_level.isdisjoint(LEGACY_PROJECT_DIRECTORIES)


def test_results_directory_contains_only_its_placeholder() -> None:
    tracked_results = {path for path in _tracked_existing_paths() if path.parts[0] == "results"}
    assert tracked_results == {Path("results/.gitkeep")}


def test_only_current_public_package_modules_remain() -> None:
    package_root = PROJECT_ROOT / "src" / "typo_cot"
    present = {path.name for path in package_root.iterdir() if path.name != "__pycache__"}
    assert present.isdisjoint(LEGACY_PACKAGE_ENTRIES)


def test_public_source_tree_is_the_reviewed_runtime_closure() -> None:
    tracked_sources = {
        path
        for path in _tracked_existing_paths()
        if path.parts[:2] == ("src", "typo_cot") and path.suffix == ".py"
    }
    assert tracked_sources == PUBLIC_SOURCE_FILES


def test_retired_packages_are_not_importable() -> None:
    for package in sorted(LEGACY_PACKAGE_DIRECTORIES):
        assert importlib.util.find_spec(f"typo_cot.{package}") is None


def test_root_guide_matches_the_reduced_project_scope() -> None:
    root_readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert "│   ├── configs/" not in root_readme
    assert "`projects/typo-cot` depends on the `typo-utils`" not in root_readme


def test_local_legacy_output_locations_remain_ignored() -> None:
    local_artifacts = {
        "projects/typo-cot/outputs/example.json",
        "projects/typo-cot/datasets/example.json",
        "projects/typo-cot/logs/example.log",
    }
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPOSITORY_ROOT,
        input="\n".join(sorted(local_artifacts)),
        check=False,
        capture_output=True,
        text=True,
    )
    ignored = set(completed.stdout.splitlines())
    assert ignored == local_artifacts


def test_cli_only_exposes_reviewed_public_commands() -> None:
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "experiments",
        "cot-swap",
        "fixed-window-answer-patching",
        "layerwise-answer-patching",
        "layerwise-kl-patching",
        "patch-coordinate-controls",
        "patch-position-controls",
        "patch-text-combination",
        "prepare-edited-pairs",
        "targeting-fidelity-audit",
    }


def test_legacy_development_documents_are_not_tracked() -> None:
    tracked_docs = {
        path.name
        for path in _tracked_existing_paths()
        if len(path.parts) == 2 and path.parts[0] == "docs"
    }
    assert tracked_docs.isdisjoint(LEGACY_DOCUMENTS)


def test_legacy_only_dependencies_and_extras_are_removed() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    dependency_names = {_dependency_name(item) for item in project["dependencies"]}
    assert dependency_names.isdisjoint(LEGACY_DEPENDENCIES)
    assert set(project.get("optional-dependencies", {})) == {"lrp"}
