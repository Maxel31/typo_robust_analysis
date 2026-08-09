"""Deterministic machine-readable and human-readable table rendering."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from typo_cot.experiments.one_token_prefix_replacement.protocol import (
    EXTENSION_SETTINGS,
    LEGACY_SETTING_IDS,
    PRIMARY_SETTING,
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _decimal(value: object) -> object:
    return "" if value is None else value


def _table10_rows(analysis: Mapping[str, object]) -> list[dict[str, object]]:
    table = _mapping(analysis.get("table10"), field="Table 10")
    settings = _mapping(table.get("settings"), field="Table 10 settings")
    primary_id = LEGACY_SETTING_IDS[PRIMARY_SETTING]
    extension_ids = frozenset(LEGACY_SETTING_IDS[setting] for setting in EXTENSION_SETTINGS)
    rows: list[dict[str, object]] = []
    for setting_id, raw in settings.items():
        summary = _mapping(raw, field=f"Table 10 {setting_id}")
        selected = _mapping(summary.get("selected"), field="Table 10 selected")
        control = _mapping(summary.get("control"), field="Table 10 control")
        rows.append(
            {
                "scope": "setting",
                "setting_id": setting_id,
                "cohort": (
                    "primary"
                    if setting_id == primary_id
                    else "extension"
                    if setting_id in extension_ids
                    else "unknown"
                ),
                "paired_eligible": summary.get("paired_eligible"),
                "selected_numerator": selected.get("numerator"),
                "selected_denominator": selected.get("denominator"),
                "selected_rate": _decimal(selected.get("rate")),
                "control_numerator": control.get("numerator"),
                "control_denominator": control.get("denominator"),
                "control_rate": _decimal(control.get("rate")),
            }
        )
    if table.get("extension_pool") is not None:
        summary = _mapping(table["extension_pool"], field="Table 10 extension pool")
        selected = _mapping(summary.get("selected"), field="Table 10 pool selected")
        control = _mapping(summary.get("control"), field="Table 10 pool control")
        rows.append(
            {
                "scope": "fourteen-extension-pool",
                "setting_id": "all_extensions",
                "cohort": "extension",
                "paired_eligible": summary.get("paired_eligible"),
                "selected_numerator": selected.get("numerator"),
                "selected_denominator": selected.get("denominator"),
                "selected_rate": _decimal(selected.get("rate")),
                "control_numerator": control.get("numerator"),
                "control_denominator": control.get("denominator"),
                "control_rate": _decimal(control.get("rate")),
            }
        )
    return rows


def _interval(summary: Mapping[str, object]) -> tuple[object, object, object]:
    for key in ("cluster_robust_interval", "cluster_bootstrap_interval"):
        raw = summary.get(key)
        if isinstance(raw, Mapping):
            return raw.get("ci95_low", ""), raw.get("ci95_high", ""), raw.get("method", "")
    return "", "", ""


def _table11_row(
    *,
    analysis_name: str,
    scope: str,
    setting_id: str,
    stratum: str,
    summary: Mapping[str, object],
) -> dict[str, object]:
    selected = _mapping(summary.get("selected"), field="Table 11 selected")
    control = _mapping(summary.get("control"), field="Table 11 control")
    low, high, method = _interval(summary)
    return {
        "analysis": analysis_name,
        "scope": scope,
        "setting_id": setting_id,
        "stratum": stratum,
        "paired_eligible": summary.get("paired_eligible"),
        "event_opportunities_per_position": summary.get(
            "event_opportunities_per_position", summary.get("paired_eligible")
        ),
        "selected_numerator": selected.get("numerator"),
        "selected_denominator": selected.get("denominator"),
        "selected_rate": _decimal(selected.get("rate")),
        "control_numerator": control.get("numerator"),
        "control_denominator": control.get("denominator"),
        "control_rate": _decimal(control.get("rate")),
        "difference_percentage_points": _decimal(summary.get("difference_percentage_points")),
        "ci95_low_percentage_points": low,
        "ci95_high_percentage_points": high,
        "interval_method": method,
    }


def _table11_rows(analysis: Mapping[str, object]) -> list[dict[str, object]]:
    table11 = _mapping(analysis.get("table11"), field="Table 11")
    distant = _mapping(table11.get("distant"), field="Table 11 distant")
    rows: list[dict[str, object]] = []
    for definition in ("pdf_literal", "submitted_producer"):
        section = _mapping(distant.get(definition), field=f"Table 11 {definition}")
        settings = _mapping(section.get("settings"), field=f"Table 11 {definition} settings")
        for setting_id, raw in settings.items():
            rows.append(
                _table11_row(
                    analysis_name=f"distant_{definition}",
                    scope="setting",
                    setting_id=setting_id,
                    stratum="all",
                    summary=_mapping(raw, field=f"Table 11 {definition} {setting_id}"),
                )
            )
        pool = section.get("extension_pool")
        if isinstance(pool, Mapping):
            rows.append(
                _table11_row(
                    analysis_name=f"distant_{definition}",
                    scope="fourteen-extension-pool",
                    setting_id="all_extensions",
                    stratum="all",
                    summary=pool,
                )
            )
            strata = _mapping(pool.get("position_order_strata"), field="position strata")
            for stratum, raw in strata.items():
                rows.append(
                    _table11_row(
                        analysis_name=f"distant_{definition}",
                        scope="fourteen-extension-pool",
                        setting_id="all_extensions",
                        stratum=stratum,
                        summary=_mapping(raw, field=f"position stratum {stratum}"),
                    )
                )
    adjacent = _mapping(table11.get("adjacent"), field="Table 11 adjacent")
    adjacent_settings = _mapping(adjacent.get("settings"), field="adjacent settings")
    for setting_id, raw in adjacent_settings.items():
        rows.append(
            _table11_row(
                analysis_name="adjacent_same_token",
                scope="setting",
                setting_id=setting_id,
                stratum="all",
                summary=_mapping(raw, field=f"adjacent {setting_id}"),
            )
        )
    if isinstance(adjacent.get("extension_pool"), Mapping):
        rows.append(
            _table11_row(
                analysis_name="adjacent_same_token",
                scope="three-setting-pool",
                setting_id="prespecified_adjacent_settings",
                stratum="all",
                summary=adjacent["extension_pool"],  # type: ignore[arg-type]
            )
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def _difference_interval(row: Mapping[str, object]) -> str:
    difference = row.get("difference_percentage_points")
    if not isinstance(difference, (int, float)) or isinstance(difference, bool):
        return "NA [NA]"
    estimate = f"{float(difference):.1f}"
    low = row.get("ci95_low_percentage_points")
    high = row.get("ci95_high_percentage_points")
    if (
        not isinstance(low, (int, float))
        or isinstance(low, bool)
        or not isinstance(high, (int, float))
        or isinstance(high, bool)
    ):
        return f"{estimate} [NA]"
    return f"{estimate} [{float(low):.1f}, {float(high):.1f}]"


def _comparison_status(value: object) -> str:
    return str(value.get("status", "unknown")) if isinstance(value, Mapping) else "unknown"


def _historical_comparison_lines(analysis: Mapping[str, object]) -> list[str]:
    comparison = _mapping(
        analysis.get("historical_comparison"),
        field="historical comparison",
    )
    table10 = _mapping(comparison.get("table10"), field="historical Table 10 comparison")
    table11 = _mapping(comparison.get("table11"), field="historical Table 11 comparison")
    lines = [
        f"- Table 10 primary: {_comparison_status(table10.get('primary'))}",
        f"- Table 10 fourteen-extension pool: {_comparison_status(table10.get('extension_pool'))}",
    ]
    settings = _mapping(table10.get("settings"), field="historical Table 10 settings")
    lines.extend(
        f"- Table 10 {setting_id}: {_comparison_status(settings[setting_id])}"
        for setting_id in sorted(settings)
    )
    lines.extend(
        (
            f"- Table 11 distant PDF-literal: "
            f"{_comparison_status(table11.get('distant_pdf_literal'))}",
            f"- Table 11 distant submitted producer: "
            f"{_comparison_status(table11.get('distant_submitted_producer'))}",
            f"- Table 11 adjacent three-setting pool: "
            f"{_comparison_status(table11.get('adjacent'))}",
        )
    )
    adjacent = _mapping(
        table11.get("adjacent_settings"),
        field="historical adjacent settings",
    )
    lines.extend(
        f"- Table 11 adjacent {setting_id}: {_comparison_status(adjacent[setting_id])}"
        for setting_id in sorted(adjacent)
    )
    return lines


def _markdown(analysis: Mapping[str, object]) -> str:
    coverage = _mapping(analysis.get("coverage"), field="coverage")
    lines = [
        "# One-token paper tables",
        "",
        "## Table 10: one-token columns",
        "",
        "| Setting | Cohort | n1 | Selected | Control |",
        "|---|---|---:|---:|---:|",
    ]
    for row in _table10_rows(analysis):
        lines.append(
            f"| {row['setting_id']} | {row['cohort']} | {row['paired_eligible']} | "
            f"{_percent(row['selected_rate'])} | {_percent(row['control_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Table 11: position controls",
            "",
            "| Analysis | Setting | Stratum | n | Selected | Control | Difference [95% CI] (pp) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in _table11_rows(analysis):
        lines.append(
            f"| {row['analysis']} | {row['setting_id']} | {row['stratum']} | "
            f"{row['paired_eligible']} | {_percent(row['selected_rate'])} | "
            f"{_percent(row['control_rate'])} | {_difference_interval(row)} |"
        )
    missing = coverage.get("missing_settings")
    missing_text = (
        ", ".join(str(value) for value in missing)
        if isinstance(missing, list) and missing
        else "none"
    )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "- Available settings: "
            + ", ".join(str(value) for value in coverage.get("available_settings", [])),
            f"- Missing settings: {missing_text}",
            f"- Complete 15-setting grid: {coverage.get('complete_15_setting_grid')}",
            f"- Complete 14-extension grid: {coverage.get('complete_14_extension_grid')}",
            f"- Complete 3-setting adjacent grid: {coverage.get('complete_3_adjacent_grid')}",
            "",
            "Paper-labelled pools are omitted where their required coverage is incomplete.",
            "",
            "## Historical-reference comparison",
            "",
            *_historical_comparison_lines(analysis),
            "",
            "## Denominators",
            "",
            "Table 10 uses the common paired-eligible denominator n1. Distant Table 11 "
            "uses two event opportunities per position and reports both the final-PDF-literal "
            "and submitted-producer definitions. Adjacent Table 11 uses one paired event per "
            "position. Historical values are references, not acceptance thresholds.",
            "",
        ]
    )
    return "\n".join(lines)


def _tex_escape(value: object) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def _latex(analysis: Mapping[str, object]) -> str:
    lines = [
        "% Generated by typo-cot build-one-token-tables.",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Setting & Cohort & $n_1$ & Selected & Control \\",
        r"\hline",
    ]
    for row in _table10_rows(analysis):
        lines.append(
            f"{_tex_escape(row['setting_id'])} & {_tex_escape(row['cohort'])} & "
            f"{row['paired_eligible']} & {_tex_escape(_percent(row['selected_rate']))} & "
            f"{_tex_escape(_percent(row['control_rate']))} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            "",
            "% Table 11 position-control fragment.",
            r"\begin{tabular}{lllrrrr}",
            r"\hline",
            r"Analysis & Setting & Stratum & $n$ & Selected & Control & $\Delta$ [95\% CI] (pp) \\",
            r"\hline",
        ]
    )
    for row in _table11_rows(analysis):
        rendered_difference = _difference_interval(row)
        lines.append(
            f"{_tex_escape(row['analysis'])} & {_tex_escape(row['setting_id'])} & "
            f"{_tex_escape(row['stratum'])} & {row['paired_eligible']} & "
            f"{_tex_escape(_percent(row['selected_rate']))} & "
            f"{_tex_escape(_percent(row['control_rate']))} & "
            f"{_tex_escape(rendered_difference)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_artifacts(output_dir: Path, analysis: Mapping[str, object]) -> None:
    """Render the six hash-bound artifacts into an existing private directory."""

    output = Path(output_dir)
    if not output.is_dir():
        raise ValueError("render output directory must already exist")
    _write_json(output / "one_token_tables.json", analysis)
    _write_json(output / "figure5_validation.json", analysis.get("figure5_validation"))
    table10_rows = _table10_rows(analysis)
    _write_csv(
        output / "table10_one_token.csv",
        table10_rows,
        (
            "scope",
            "setting_id",
            "cohort",
            "paired_eligible",
            "selected_numerator",
            "selected_denominator",
            "selected_rate",
            "control_numerator",
            "control_denominator",
            "control_rate",
        ),
    )
    table11_rows = _table11_rows(analysis)
    _write_csv(
        output / "table11_position_controls.csv",
        table11_rows,
        (
            "analysis",
            "scope",
            "setting_id",
            "stratum",
            "paired_eligible",
            "event_opportunities_per_position",
            "selected_numerator",
            "selected_denominator",
            "selected_rate",
            "control_numerator",
            "control_denominator",
            "control_rate",
            "difference_percentage_points",
            "ci95_low_percentage_points",
            "ci95_high_percentage_points",
            "interval_method",
        ),
    )
    (output / "one_token_tables.md").write_text(_markdown(analysis), encoding="utf-8")
    (output / "one_token_tables.tex").write_text(_latex(analysis), encoding="utf-8")


__all__ = ["render_artifacts"]
