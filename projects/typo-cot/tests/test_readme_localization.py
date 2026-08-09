"""Contracts for the English and Japanese public reproduction guides."""

from __future__ import annotations

import re
from pathlib import Path

from typo_cot.experiments.catalog import PAPER_EXPERIMENTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def _bash_blocks(markdown: str) -> list[str]:
    """Return executable examples without translating their shell syntax."""
    return [block.rstrip() for block in re.findall(r"```bash\n(.*?)\n```", markdown, re.DOTALL)]


def test_root_readmes_are_reciprocally_linked() -> None:
    english_path = REPOSITORY_ROOT / "README.md"
    japanese_path = REPOSITORY_ROOT / "README.ja.md"

    assert japanese_path.is_file()
    english = english_path.read_text(encoding="utf-8")
    japanese = japanese_path.read_text(encoding="utf-8")

    assert "[日本語](README.ja.md)" in english
    assert "[English](README.md)" in japanese
    assert "[日本語](README.ja.md)" in japanese
    assert "[日本語README](projects/typo-cot/README.ja.md)" in japanese


def test_project_readmes_are_reciprocally_linked() -> None:
    english_path = PROJECT_ROOT / "README.md"
    japanese_path = PROJECT_ROOT / "README.ja.md"

    assert japanese_path.is_file()
    english = english_path.read_text(encoding="utf-8")
    japanese = japanese_path.read_text(encoding="utf-8")

    assert "[日本語](README.ja.md)" in english
    assert "[English](README.md)" in japanese
    assert "[日本語](README.ja.md)" in japanese


def test_localized_readmes_preserve_every_shell_example() -> None:
    for english_path, japanese_path in (
        (REPOSITORY_ROOT / "README.md", REPOSITORY_ROOT / "README.ja.md"),
        (PROJECT_ROOT / "README.md", PROJECT_ROOT / "README.ja.md"),
    ):
        english = english_path.read_text(encoding="utf-8")
        japanese = japanese_path.read_text(encoding="utf-8")
        assert _bash_blocks(japanese) == _bash_blocks(english), japanese_path


def test_japanese_project_readme_covers_every_implemented_operation() -> None:
    japanese = (PROJECT_ROOT / "README.ja.md").read_text(encoding="utf-8")

    for spec in PAPER_EXPERIMENTS:
        if spec.status != "implemented":
            continue
        assert f"`{spec.slug}`" in japanese
        for argument in spec.required_arguments:
            assert argument in japanese, f"{spec.slug}: missing {argument}"
        for output in spec.outputs:
            assert output in japanese, f"{spec.slug}: missing {output}"


def test_public_guides_lead_with_reproduction_instead_of_source_of_truth() -> None:
    public_guides = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "README.ja.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README.ja.md",
        *sorted((PROJECT_ROOT / "docs").glob("*.md")),
    )

    for path in public_guides:
        contents = path.read_text(encoding="utf-8")
        assert "Source of truth" not in contents
        assert "source of truth" not in contents

    assert "## Paper and reproduction" in (REPOSITORY_ROOT / "README.md").read_text(
        encoding="utf-8"
    )
    assert "## 論文と再現手順" in (REPOSITORY_ROOT / "README.ja.md").read_text(encoding="utf-8")
