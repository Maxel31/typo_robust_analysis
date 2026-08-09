"""Deterministic CSV, Markdown, and LaTeX rendering for Table 8."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("render input must contain JSON objects")
    return value


def _percent(value: object) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}"


def _fraction(cell: Mapping[str, object], numerator: str) -> str:
    denominator = cell.get("denominator")
    value = cell.get(numerator)
    if denominator is None or value is None:
        return ""
    return f"{value}/{denominator}"


def _csv_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    accuracy = _mapping(summary["accuracy"])
    equal = _mapping(accuracy["equal_setting_mean"])
    matched = _mapping(_mapping(accuracy["matched_items"])["conditions"])
    for count in ("0", "1", "2", "4"):
        if count in equal:
            rows.append(
                {
                    "outcome": "accuracy",
                    "scope": "equal-setting mean",
                    "setting_id": "",
                    "edit_count": count,
                    "numerator": "",
                    "denominator": accuracy["complete_settings"],
                    "rate": equal[count],
                    "percent": _percent(equal[count]),
                }
            )
        if count in matched:
            cell = _mapping(matched[count])
            rows.append(
                {
                    "outcome": "accuracy",
                    "scope": "matched items",
                    "setting_id": "",
                    "edit_count": count,
                    "numerator": cell["correct"],
                    "denominator": cell["denominator"],
                    "rate": cell["rate"],
                    "percent": _percent(cell["rate"]),
                }
            )
    restoration = _mapping(summary["restoration"])
    for row_value in restoration["settings"]:  # type: ignore[union-attr]
        row = _mapping(row_value)
        by_count = _mapping(row["by_edit_count"])
        for count in ("1", "2", "4"):
            if count not in by_count:
                continue
            cell = _mapping(by_count[count])
            rows.append(
                {
                    "outcome": "cot-swap restoration",
                    "scope": row["label"],
                    "setting_id": row["setting_id"],
                    "edit_count": count,
                    "numerator": cell["restored"],
                    "denominator": cell["denominator"],
                    "rate": cell["rate"],
                    "percent": _percent(cell["rate"]),
                }
            )
    pooled = _mapping(restoration["pooled"])
    for count in ("1", "2", "4"):
        if count not in pooled:
            continue
        cell = _mapping(pooled[count])
        rows.append(
            {
                "outcome": "cot-swap restoration",
                "scope": "six-setting pooled",
                "setting_id": "",
                "edit_count": count,
                "numerator": cell["restored"],
                "denominator": cell["denominator"],
                "rate": cell["rate"],
                "percent": _percent(cell["rate"]),
            }
        )
    return rows


def _render_csv(path: Path, summary: Mapping[str, object]) -> None:
    fieldnames = (
        "outcome",
        "scope",
        "setting_id",
        "edit_count",
        "numerator",
        "denominator",
        "rate",
        "percent",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_rows(summary))


def _markdown_accuracy(summary: Mapping[str, object]) -> list[str]:
    accuracy = _mapping(summary["accuracy"])
    equal = _mapping(accuracy["equal_setting_mean"])
    matched_payload = _mapping(accuracy["matched_items"])
    matched = _mapping(matched_payload["conditions"])
    matched_item_label = "item" if matched_payload["sample_count"] == 1 else "items"
    lines = [
        "| Outcome | Scope | 0 edits | 1 edit | 2 edits | 4 edits |",
        "|---|---|---:|---:|---:|---:|",
        "| Accuracy (%) | "
        f"{accuracy['complete_settings']}-setting mean | "
        + " | ".join(_percent(equal.get(key)) for key in ("0", "1", "2", "4"))
        + " |",
        "| Accuracy (%) | "
        f"{matched_payload['sample_count']} matched {matched_item_label} | "
        + " | ".join(
            _percent(_mapping(matched[key])["rate"]) if key in matched else "—"
            for key in ("0", "1", "2", "4")
        )
        + " |",
    ]
    return lines


def _markdown_restoration(summary: Mapping[str, object]) -> list[str]:
    restoration = _mapping(summary["restoration"])
    lines = [
        "| CoT-swap restoration | 0 edits | 1 edit | 2 edits | 4 edits |",
        "|---|---:|---:|---:|---:|",
    ]
    for value in restoration["settings"]:  # type: ignore[union-attr]
        row = _mapping(value)
        by_count = _mapping(row["by_edit_count"])
        cells = []
        for count in ("1", "2", "4"):
            if count not in by_count:
                cells.append("—")
                continue
            cell = _mapping(by_count[count])
            cells.append(f"{_percent(cell['rate'])}% ({_fraction(cell, 'restored')})")
        lines.append(f"| {row['label']} | — | " + " | ".join(cells) + " |")
    pooled = _mapping(restoration["pooled"])
    pooled_cells = []
    for count in ("1", "2", "4"):
        if count not in pooled:
            pooled_cells.append("—")
            continue
        cell = _mapping(pooled[count])
        pooled_cells.append(f"{_percent(cell['rate'])}% ({_fraction(cell, 'restored')})")
    lines.append("| Six-setting pooled | — | " + " | ".join(pooled_cells) + " |")
    return lines


def _render_markdown(path: Path, summary: Mapping[str, object]) -> None:
    coverage = _mapping(summary["coverage"])
    accuracy_coverage = _mapping(coverage["accuracy"])
    restoration_coverage = _mapping(coverage["restoration"])
    comparability = _mapping(summary["comparability"])
    lines = [
        "# Table 8: edit-count sensitivity",
        "",
        "Restoration is undefined at zero edits. Its denominator is recomputed "
        "separately at each nonzero edit count and is not a matched-item cohort.",
        "",
        *_markdown_accuracy(summary),
        "",
        *_markdown_restoration(summary),
        "",
        "## Coverage and comparability",
        "",
        f"- Status: `{comparability['status']}`",
        "- Accuracy settings: "
        f"{accuracy_coverage['complete_setting_count']}/"
        f"{accuracy_coverage['expected_setting_count']}",
        "- Restoration settings: "
        f"{restoration_coverage['complete_setting_count']}/"
        f"{restoration_coverage['expected_setting_count']}",
        f"- Complete accuracy grid: `{coverage['complete_accuracy_grid']}`",
        f"- Complete restoration grid: `{coverage['complete_restoration_grid']}`",
        "- Historical cohort identity: `false`",
        "",
        "## Historical-reference comparison",
        "",
        "The final-PDF values are retained in `edit_count_summary.json`. "
        "Comparisons are evaluated only when the corresponding grid is complete.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _render_latex(path: Path, summary: Mapping[str, object]) -> None:
    accuracy = _mapping(summary["accuracy"])
    equal = _mapping(accuracy["equal_setting_mean"])
    matched_payload = _mapping(accuracy["matched_items"])
    matched = _mapping(matched_payload["conditions"])
    matched_item_label = "item" if matched_payload["sample_count"] == 1 else "items"
    restoration = _mapping(summary["restoration"])
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Outcome & Scope & 0 & 1 & 2 & 4 \\",
        r"\midrule",
        "Accuracy (\\%) & "
        f"{accuracy['complete_settings']}-setting mean & "
        + " & ".join(_percent(equal.get(key)) for key in ("0", "1", "2", "4"))
        + r" \\",
        "Accuracy (\\%) & "
        f"{matched_payload['sample_count']} matched {matched_item_label} & "
        + " & ".join(
            _percent(_mapping(matched[key])["rate"]) if key in matched else "--"
            for key in ("0", "1", "2", "4")
        )
        + r" \\",
        r"\midrule",
    ]
    for value in restoration["settings"]:  # type: ignore[union-attr]
        row = _mapping(value)
        by_count = _mapping(row["by_edit_count"])
        cells = []
        for count in ("1", "2", "4"):
            if count not in by_count:
                cells.append("--")
                continue
            cell = _mapping(by_count[count])
            cells.append(f"{_percent(cell['rate'])} ({_fraction(cell, 'restored')})")
        lines.append(
            "CoT-swap restoration & "
            + _latex_escape(row["label"])
            + " & -- & "
            + " & ".join(cells)
            + r" \\"
        )
    pooled = _mapping(restoration["pooled"])
    cells = []
    for count in ("1", "2", "4"):
        if count not in pooled:
            cells.append("--")
            continue
        cell = _mapping(pooled[count])
        cells.append(f"{_percent(cell['rate'])} ({_fraction(cell, 'restored')})")
    lines.extend(
        [
            "CoT-swap restoration & Six-setting pooled & -- & " + " & ".join(cells) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_artifacts(output_dir: Path, summary: Mapping[str, object]) -> None:
    """Write every human-readable Table 8 artifact into a staging directory."""
    _render_csv(output_dir / "table8_edit_count.csv", summary)
    _render_markdown(output_dir / "table8_edit_count.md", summary)
    _render_latex(output_dir / "table8_edit_count.tex", summary)


__all__ = ["render_artifacts"]
