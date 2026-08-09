"""CSV, Markdown, and LaTeX renderers for final-PDF Table 9 cells."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

_BOLD_MODELS = {
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
}


def _metric(row: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = row[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a metric object")
    return value


def _percent(metric: Mapping[str, object]) -> str:
    rate = metric["rate"]
    return "NA" if rate is None else f"{100 * float(rate):.1f}%"


def _count_percent(metric: Mapping[str, object]) -> str:
    return f"{int(metric['numerator'])} ({_percent(metric)})"


def _restoration(metric: Mapping[str, object]) -> str:
    return f"{int(metric['numerator'])}/{int(metric['denominator'])} ({_percent(metric)})"


def _latex_metric(metric: Mapping[str, object], *, restoration: bool = False) -> str:
    numerator = int(metric["numerator"])
    prefix = f"{numerator}/{int(metric['denominator'])}" if restoration else str(numerator)
    rate = metric["rate"]
    percent = "NA" if rate is None else f"{100 * float(rate):.1f}\\%"
    return f"{prefix} ({percent})"


def render_artifacts(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    """Write the three deterministic human/machine-readable table fragments."""
    csv_path = output_dir / "table9_model_scale.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "model",
                "label",
                "source_records",
                "executed_pairs",
                "n_s",
                "both",
                "both_denominator",
                "both_rate",
                "question_only",
                "question_only_denominator",
                "question_only_rate",
                "cot_only",
                "cot_only_denominator",
                "cot_only_rate",
                "restored",
                "n_b",
                "restoration_rate",
            )
        )
        for row in rows:
            both = _metric(row, "both")
            question = _metric(row, "question_only")
            cot = _metric(row, "cot_only")
            restoration = _metric(row, "restoration")
            writer.writerow(
                (
                    row["model"],
                    row["label"],
                    row["source_records"],
                    row["executed_pairs"],
                    row["n_s"],
                    both["numerator"],
                    both["denominator"],
                    both["rate"],
                    question["numerator"],
                    question["denominator"],
                    question["rate"],
                    cot["numerator"],
                    cot["denominator"],
                    cot["rate"],
                    restoration["numerator"],
                    restoration["denominator"],
                    restoration["rate"],
                )
            )

    coverage = summary["coverage"]
    if not isinstance(coverage, Mapping):
        raise TypeError("summary coverage must be an object")
    markdown = [
        "# Appendix C / Table 9: model-scale CoT swap",
        "",
        f"Grid status: {'complete' if coverage['complete_grid'] else 'partial valid grid'}.",
        "",
        "| Model | n_s | Both | Question only | CoT only | Restored / n_B |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {label} | {n_s} | {both} | {question} | {cot} | {restoration} |".format(
                label=row["label"],
                n_s=row["n_s"],
                both=_count_percent(_metric(row, "both")),
                question=_count_percent(_metric(row, "question_only")),
                cot=_count_percent(_metric(row, "cot_only")),
                restoration=_restoration(_metric(row, "restoration")),
            )
        )
    markdown.extend(
        (
            "",
            "Rates for Both, Question only, and CoT only use n_s. Restoration uses n_B.",
            "Qwen2.5-72B is directional only in the published reference because n_B = 10.",
            "",
        )
    )
    (output_dir / "table9_model_scale.md").write_text("\n".join(markdown), encoding="utf-8")

    latex = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $n_s$ & Both & Question only & CoT only & Restored / $n_B$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["label"])
        cells = (
            str(row["n_s"]),
            _latex_metric(_metric(row, "both")),
            _latex_metric(_metric(row, "question_only")),
            _latex_metric(_metric(row, "cot_only")),
            _latex_metric(_metric(row, "restoration"), restoration=True),
        )
        if row["model"] in _BOLD_MODELS:
            label = rf"\textbf{{{label}}}"
            cells = tuple(rf"\textbf{{{cell}}}" for cell in cells)
        line = f"{label} & " + " & ".join(cells) + r" \\"
        latex.append(line)
    latex.extend((r"\bottomrule", r"\end{tabular}", ""))
    (output_dir / "table9_model_scale.tex").write_text("\n".join(latex), encoding="utf-8")


__all__ = ["render_artifacts"]
