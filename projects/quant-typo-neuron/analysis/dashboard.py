#!/usr/bin/env python
"""Marimo analysis dashboard: quantization x typo robustness visualization."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(app_title="Quantization x Typo Robustness Analysis")


# ---------------------------------------------------------------------------
# Cell: imports
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Quantization x Typo Robustness Analysis Dashboard

        This dashboard visualises how LLM quantization affects typo robustness.
        Results are loaded from `../results/` and aggregated on the fly.

        **Experiment matrix**: 9 models x 15 quantization conditions x 8 benchmarks x 6 typo conditions
        """
    )
    return


@app.cell
def _():
    import json
    import os
    import re
    from collections import defaultdict
    from pathlib import Path

    import altair as alt
    import nltk
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    return alt, defaultdict, go, json, nltk, os, pd, Path, px, re


# ---------------------------------------------------------------------------
# Cell: constants
# ---------------------------------------------------------------------------
@app.cell
def _(Path):
    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

    TYPO_ORDER = ["clean", "swap_n1", "swap_n2", "swap_n4", "random_n4", "replace_n4"]
    TYPO_LABELS = {
        "clean": "Clean",
        "swap_n1": "Swap n=1",
        "swap_n2": "Swap n=2",
        "swap_n4": "Swap n=4",
        "random_n4": "Random n=4",
        "replace_n4": "Replace n=4",
    }

    ALL_MODELS = [
        "Llama-3.2-1B",
        "Llama-3.2-3B",
        "Llama-3.1-8B",
        "Mistral-7B-v0.3",
        "Qwen3-4B",
        "Qwen3-8B",
        "gemma-3-1b-pt",
        "gemma-3-4b-pt",
        "gemma-3-12b-pt",
    ]

    ALL_BENCHMARKS = [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "mmlu",
        "piqa",
        "gsm8k",
        "wikitext2",
        "c4",
    ]

    PERPLEXITY_BENCHMARKS = {"wikitext2", "c4"}

    QUANT_CONDITIONS = [
        "none_w16",
        "gptq_w4_clean",
        "gptq_w4_noisy",
        "gptq_w8_clean",
        "gptq_w8_noisy",
        "awq_w4_clean",
        "awq_w4_noisy",
        "awq_w8_clean",
        "awq_w8_noisy",
        "smoothquant_w4_clean",
        "smoothquant_w4_noisy",
        "qep_w4_clean",
        "qep_w4_noisy",
        "qep_w8_clean",
        "qep_w8_noisy",
    ]

    QUANT_LABELS = {
        "none_w16": "Baseline (FP16)",
        "gptq_w4_clean": "GPTQ W4 clean",
        "gptq_w4_noisy": "GPTQ W4 noisy",
        "gptq_w8_clean": "GPTQ W8 clean",
        "gptq_w8_noisy": "GPTQ W8 noisy",
        "awq_w4_clean": "AWQ W4 clean",
        "awq_w4_noisy": "AWQ W4 noisy",
        "awq_w8_clean": "AWQ W8 clean",
        "awq_w8_noisy": "AWQ W8 noisy",
        "smoothquant_w4_clean": "SmoothQuant W4 clean",
        "smoothquant_w4_noisy": "SmoothQuant W4 noisy",
        "qep_w4_clean": "QEP W4 clean",
        "qep_w4_noisy": "QEP W4 noisy",
        "qep_w8_clean": "QEP W8 clean",
        "qep_w8_noisy": "QEP W8 noisy",
    }

    CALIBRATION_PAIRS = [
        ("gptq_w4_clean", "gptq_w4_noisy"),
        ("gptq_w8_clean", "gptq_w8_noisy"),
        ("awq_w4_clean", "awq_w4_noisy"),
        ("awq_w8_clean", "awq_w8_noisy"),
        ("smoothquant_w4_clean", "smoothquant_w4_noisy"),
        ("qep_w4_clean", "qep_w4_noisy"),
        ("qep_w8_clean", "qep_w8_noisy"),
    ]

    return (
        ALL_BENCHMARKS,
        ALL_MODELS,
        CALIBRATION_PAIRS,
        PERPLEXITY_BENCHMARKS,
        QUANT_CONDITIONS,
        QUANT_LABELS,
        RESULTS_DIR,
        TYPO_LABELS,
        TYPO_ORDER,
    )


# ---------------------------------------------------------------------------
# Cell: data loading
# ---------------------------------------------------------------------------
@app.cell
def _(RESULTS_DIR, json, pd):
    def load_all_metrics(results_dir):
        """Walk the results directory and load every metrics.json into a DataFrame."""
        records = []
        if not results_dir.exists():
            return pd.DataFrame()
        for metrics_path in results_dir.rglob("metrics.json"):
            try:
                with open(metrics_path) as f:
                    data = json.load(f)
                rel = metrics_path.relative_to(results_dir)
                parts = rel.parts
                if len(parts) >= 4:
                    data["model_short"] = parts[0]
                    data["quant_condition"] = parts[1]
                    data["typo_condition"] = parts[3]
                if "accuracy" not in data:
                    data["accuracy"] = None
                if "mean_perplexity" not in data:
                    data["mean_perplexity"] = None
                records.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    df_metrics = load_all_metrics(RESULTS_DIR)
    n_results = len(df_metrics)
    return df_metrics, load_all_metrics, n_results


@app.cell
def _(df_metrics, mo, n_results):
    mo.md(
        f"""
        ## Data Summary

        - **Results loaded**: {n_results} metric files
        - **Models found**: {', '.join(sorted(df_metrics['model_short'].unique())) if n_results > 0 else 'none'}
        - **Benchmarks found**: {', '.join(sorted(df_metrics['benchmark'].unique())) if n_results > 0 else 'none'}
        - **Quantization conditions found**: {', '.join(sorted(df_metrics['quant_condition'].unique())) if n_results > 0 else 'none'}
        """
    )
    return


# ---------------------------------------------------------------------------
# Cell: prediction loading helper
# ---------------------------------------------------------------------------
@app.cell
def _(RESULTS_DIR, json):
    def load_predictions(model_short, quant_condition, benchmark, typo_condition):
        """Load predictions.jsonl for a specific condition."""
        pred_path = (
            RESULTS_DIR
            / model_short
            / quant_condition
            / benchmark
            / typo_condition
            / "predictions.jsonl"
        )
        if not pred_path.exists():
            return []
        records = []
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    return (load_predictions,)


# ---------------------------------------------------------------------------
# Cell: interactive filters
# ---------------------------------------------------------------------------
@app.cell
def _(ALL_BENCHMARKS, ALL_MODELS, df_metrics, mo, n_results, PERPLEXITY_BENCHMARKS):
    _available_models = (
        sorted(df_metrics["model_short"].unique().tolist())
        if n_results > 0
        else ALL_MODELS
    )
    _available_benchmarks = (
        sorted(df_metrics["benchmark"].unique().tolist())
        if n_results > 0
        else ALL_BENCHMARKS
    )
    _available_quant = (
        sorted(df_metrics["quant_condition"].unique().tolist())
        if n_results > 0
        else []
    )
    _default_bench = next(
        (b for b in _available_benchmarks if b not in PERPLEXITY_BENCHMARKS),
        _available_benchmarks[0] if _available_benchmarks else None,
    )

    model_selector = mo.ui.dropdown(
        options=_available_models,
        value=_available_models[0] if _available_models else None,
        label="Model",
    )
    benchmark_selector = mo.ui.dropdown(
        options=_available_benchmarks,
        value=_default_bench,
        label="Benchmark",
    )
    quant_multiselect = mo.ui.multiselect(
        options=_available_quant,
        value=_available_quant,
        label="Quantization conditions",
    )

    mo.vstack(
        [
            mo.hstack([model_selector, benchmark_selector], justify="start", gap=1),
            quant_multiselect,
        ]
    )
    return benchmark_selector, model_selector, quant_multiselect


# ===========================================================================
# SECTION 1: Performance Overview
# ===========================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Performance Overview

        Accuracy (or perplexity) across typo conditions for each quantization method.
        Lines show how performance degrades with increasing typo severity.
        """
    )
    return


@app.cell
def _(
    benchmark_selector,
    df_metrics,
    mo,
    model_selector,
    n_results,
    PERPLEXITY_BENCHMARKS,
    px,
    quant_multiselect,
    QUANT_LABELS,
    TYPO_LABELS,
    TYPO_ORDER,
):
    def _build_performance_chart():
        if n_results == 0:
            return mo.md("**No data available.** Run experiments first.")

        _model = model_selector.value
        _bench = benchmark_selector.value
        _quants = quant_multiselect.value

        if not _model or not _bench:
            return mo.md("Please select a model and benchmark above.")

        _mask = (
            (df_metrics["model_short"] == _model)
            & (df_metrics["benchmark"] == _bench)
            & (df_metrics["quant_condition"].isin(_quants))
        )
        _df = df_metrics[_mask].copy()
        if _df.empty:
            return mo.md(
                f"No results for **{_model}** / **{_bench}** with selected quantization conditions."
            )

        _is_ppl = _bench in PERPLEXITY_BENCHMARKS
        _metric_col = "mean_perplexity" if _is_ppl else "accuracy"
        _metric_label = "Perplexity" if _is_ppl else "Accuracy"

        _df = _df.dropna(subset=[_metric_col])
        if _df.empty:
            return mo.md(f"No {_metric_label.lower()} data for **{_model}** / **{_bench}**.")

        _df["typo_label"] = _df["typo_condition"].map(TYPO_LABELS)
        _df["typo_sort"] = _df["typo_condition"].map(
            {t: i for i, t in enumerate(TYPO_ORDER)}
        )
        _df = _df.sort_values("typo_sort")
        _df["quant_label"] = (
            _df["quant_condition"].map(QUANT_LABELS).fillna(_df["quant_condition"])
        )

        fig = px.line(
            _df,
            x="typo_label",
            y=_metric_col,
            color="quant_label",
            markers=True,
            title=f"{_model} / {_bench}: {_metric_label} vs Typo Condition",
            labels={
                "typo_label": "Typo Condition",
                _metric_col: _metric_label,
                "quant_label": "Quantization",
            },
            category_orders={"typo_label": [TYPO_LABELS[t] for t in TYPO_ORDER]},
        )
        fig.update_layout(
            xaxis_title="Typo Condition",
            yaxis_title=_metric_label,
            legend_title="Quantization",
            hovermode="x unified",
        )
        if _is_ppl:
            fig.update_yaxes(type="log")

        return mo.ui.plotly(fig)

    _build_performance_chart()
    return


# ===========================================================================
# SECTION 2: Clean-Typo Gap Analysis
# ===========================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Clean-Typo Gap Analysis

        Shows the accuracy difference between clean and typo conditions.
        Positive values mean the model loses accuracy under typos.
        The key question: **does quantization widen or narrow the clean-typo gap?**
        """
    )
    return


@app.cell
def _(mo):
    gap_typo_selector = mo.ui.dropdown(
        options=["swap_n4", "random_n4", "replace_n4", "swap_n2", "swap_n1"],
        value="swap_n4",
        label="Typo condition for gap analysis",
    )
    gap_typo_selector
    return (gap_typo_selector,)


@app.cell
def _(
    benchmark_selector,
    df_metrics,
    gap_typo_selector,
    mo,
    model_selector,
    n_results,
    pd,
    PERPLEXITY_BENCHMARKS,
    px,
    quant_multiselect,
    QUANT_LABELS,
):
    def _build_gap_chart():
        if n_results == 0:
            return mo.md("**No data available.**")

        _model = model_selector.value
        _bench = benchmark_selector.value
        _typo = gap_typo_selector.value
        _quants = quant_multiselect.value
        _is_ppl = _bench in PERPLEXITY_BENCHMARKS
        _metric_col = "mean_perplexity" if _is_ppl else "accuracy"

        if not _model or not _bench:
            return mo.md("Please select a model and benchmark.")

        _mask_clean = (
            (df_metrics["model_short"] == _model)
            & (df_metrics["benchmark"] == _bench)
            & (df_metrics["typo_condition"] == "clean")
            & (df_metrics["quant_condition"].isin(_quants))
        )
        _mask_typo = (
            (df_metrics["model_short"] == _model)
            & (df_metrics["benchmark"] == _bench)
            & (df_metrics["typo_condition"] == _typo)
            & (df_metrics["quant_condition"].isin(_quants))
        )
        _df_clean = df_metrics[_mask_clean][["quant_condition", _metric_col]].rename(
            columns={_metric_col: "clean_metric"}
        )
        _df_typo = df_metrics[_mask_typo][["quant_condition", _metric_col]].rename(
            columns={_metric_col: "typo_metric"}
        )
        _merged = pd.merge(_df_clean, _df_typo, on="quant_condition")

        if _merged.empty:
            return mo.md(f"No paired clean/{_typo} data for **{_model}** / **{_bench}**.")

        if _is_ppl:
            _merged["gap"] = _merged["typo_metric"] - _merged["clean_metric"]
            _gap_label = "Perplexity Increase (typo - clean)"
        else:
            _merged["gap"] = _merged["clean_metric"] - _merged["typo_metric"]
            _gap_label = "Accuracy Drop (clean - typo)"

        _merged["quant_label"] = (
            _merged["quant_condition"].map(QUANT_LABELS).fillna(_merged["quant_condition"])
        )
        _merged = _merged.sort_values("gap", ascending=False)

        fig = px.bar(
            _merged,
            x="quant_label",
            y="gap",
            color="gap",
            color_continuous_scale="RdYlGn_r",
            title=f"{_model} / {_bench}: {_gap_label} ({_typo})",
            labels={"quant_label": "Quantization", "gap": _gap_label},
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        return mo.ui.plotly(fig)

    _build_gap_chart()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Gap Heatmap (all models x quantization)

        Overview of clean-typo accuracy drop across all available model/quantization combinations
        for the selected benchmark and typo condition.
        """
    )
    return


@app.cell
def _(
    benchmark_selector,
    df_metrics,
    gap_typo_selector,
    mo,
    n_results,
    pd,
    PERPLEXITY_BENCHMARKS,
    px,
    QUANT_LABELS,
):
    def _build_gap_heatmap():
        if n_results == 0:
            return mo.md("**No data available.**")

        _bench = benchmark_selector.value
        _typo = gap_typo_selector.value
        _is_ppl = _bench in PERPLEXITY_BENCHMARKS
        _metric_col = "mean_perplexity" if _is_ppl else "accuracy"

        if not _bench:
            return mo.md("Please select a benchmark.")

        _clean = df_metrics[
            (df_metrics["benchmark"] == _bench) & (df_metrics["typo_condition"] == "clean")
        ][["model_short", "quant_condition", _metric_col]].rename(
            columns={_metric_col: "clean_metric"}
        )
        _typo_df = df_metrics[
            (df_metrics["benchmark"] == _bench) & (df_metrics["typo_condition"] == _typo)
        ][["model_short", "quant_condition", _metric_col]].rename(
            columns={_metric_col: "typo_metric"}
        )
        _merged = pd.merge(_clean, _typo_df, on=["model_short", "quant_condition"])

        if _merged.empty:
            return mo.md(f"No paired data for heatmap on **{_bench}** / **{_typo}**.")

        if _is_ppl:
            _merged["gap"] = _merged["typo_metric"] - _merged["clean_metric"]
            _gap_label = "Perplexity increase"
        else:
            _merged["gap"] = _merged["clean_metric"] - _merged["typo_metric"]
            _gap_label = "Accuracy drop"

        _merged["quant_label"] = (
            _merged["quant_condition"].map(QUANT_LABELS).fillna(_merged["quant_condition"])
        )

        _pivot = _merged.pivot_table(
            index="model_short", columns="quant_label", values="gap", aggfunc="first"
        )

        fig = px.imshow(
            _pivot,
            text_auto=".3f",
            color_continuous_scale="RdYlGn_r",
            title=f"Clean-Typo Gap Heatmap: {_bench} / {_typo}",
            labels={"color": _gap_label},
            aspect="auto",
        )
        fig.update_layout(xaxis_tickangle=-45)
        return mo.ui.plotly(fig)

    _build_gap_heatmap()
    return


# ===========================================================================
# SECTION 3: Calibration Data Effect
# ===========================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Calibration Data Effect

        Compares **clean calibration** vs **noisy calibration** for each quantization method/bitwidth.
        The question: **does calibrating on noisy data improve typo robustness?**
        """
    )
    return


@app.cell
def _(
    benchmark_selector,
    CALIBRATION_PAIRS,
    df_metrics,
    mo,
    model_selector,
    n_results,
    pd,
    PERPLEXITY_BENCHMARKS,
    px,
    QUANT_LABELS,
    TYPO_LABELS,
    TYPO_ORDER,
):
    def _build_calibration_chart():
        if n_results == 0:
            return mo.md("**No data available.**")

        _model = model_selector.value
        _bench = benchmark_selector.value
        _is_ppl = _bench in PERPLEXITY_BENCHMARKS
        _metric_col = "mean_perplexity" if _is_ppl else "accuracy"
        _metric_label = "Perplexity" if _is_ppl else "Accuracy"

        if not _model or not _bench:
            return mo.md("Please select a model and benchmark.")

        rows = []
        for clean_q, noisy_q in CALIBRATION_PAIRS:
            method_bits = clean_q.rsplit("_", 1)[0]
            for typo_cond in TYPO_ORDER:
                for qcond, cal_type in [(clean_q, "clean"), (noisy_q, "noisy")]:
                    _mask = (
                        (df_metrics["model_short"] == _model)
                        & (df_metrics["benchmark"] == _bench)
                        & (df_metrics["quant_condition"] == qcond)
                        & (df_metrics["typo_condition"] == typo_cond)
                    )
                    _match = df_metrics[_mask]
                    if not _match.empty:
                        val = _match.iloc[0][_metric_col]
                        if val is not None and pd.notna(val):
                            rows.append(
                                {
                                    "method_bits": method_bits,
                                    "calibration": cal_type,
                                    "typo_condition": typo_cond,
                                    "typo_label": TYPO_LABELS.get(typo_cond, typo_cond),
                                    "metric": val,
                                    "label": f"{QUANT_LABELS.get(qcond, qcond)}",
                                }
                            )

        if not rows:
            return mo.md(
                f"No calibration pair data for **{_model}** / **{_bench}**. "
                "Quantized model results may not be available yet."
            )

        _df = pd.DataFrame(rows)
        fig = px.line(
            _df,
            x="typo_label",
            y="metric",
            color="label",
            line_dash="calibration",
            markers=True,
            facet_col="method_bits",
            facet_col_wrap=3,
            title=f"Calibration Effect: {_model} / {_bench}",
            labels={
                "typo_label": "Typo Condition",
                "metric": _metric_label,
                "label": "Quantization",
            },
            category_orders={
                "typo_label": [TYPO_LABELS[t] for t in TYPO_ORDER],
            },
        )
        fig.update_layout(hovermode="x unified")
        if _is_ppl:
            fig.update_yaxes(type="log")
        return mo.ui.plotly(fig)

    _build_calibration_chart()
    return


@app.cell
def _(
    benchmark_selector,
    CALIBRATION_PAIRS,
    df_metrics,
    gap_typo_selector,
    mo,
    model_selector,
    n_results,
    pd,
    PERPLEXITY_BENCHMARKS,
    px,
):
    def _build_calibration_bar():
        if n_results == 0:
            return mo.md("**No data available.**")

        _model = model_selector.value
        _bench = benchmark_selector.value
        _typo = gap_typo_selector.value
        _is_ppl = _bench in PERPLEXITY_BENCHMARKS
        _metric_col = "mean_perplexity" if _is_ppl else "accuracy"
        _metric_label = "Perplexity" if _is_ppl else "Accuracy"

        if not _model or not _bench:
            return mo.md("Please select a model and benchmark.")

        rows = []
        for clean_q, noisy_q in CALIBRATION_PAIRS:
            method_bits = clean_q.rsplit("_", 1)[0]
            for qcond, cal_label in [(clean_q, "Clean cal"), (noisy_q, "Noisy cal")]:
                _mask = (
                    (df_metrics["model_short"] == _model)
                    & (df_metrics["benchmark"] == _bench)
                    & (df_metrics["quant_condition"] == qcond)
                    & (df_metrics["typo_condition"] == _typo)
                )
                _match = df_metrics[_mask]
                if not _match.empty:
                    val = _match.iloc[0][_metric_col]
                    if val is not None and pd.notna(val):
                        rows.append(
                            {
                                "method_bits": method_bits,
                                "calibration": cal_label,
                                "metric": val,
                            }
                        )

        if not rows:
            return mo.md(
                f"No calibration pair data for **{_model}** / **{_bench}** / **{_typo}**."
            )

        _df = pd.DataFrame(rows)
        fig = px.bar(
            _df,
            x="method_bits",
            y="metric",
            color="calibration",
            barmode="group",
            title=f"Clean vs Noisy Calibration: {_model} / {_bench} / {_typo}",
            labels={
                "method_bits": "Method + Bits",
                "metric": _metric_label,
                "calibration": "Calibration Data",
            },
        )
        return mo.ui.plotly(fig)

    _build_calibration_bar()
    return


# ===========================================================================
# SECTION 4: Error Analysis
# ===========================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Error Analysis (POS Tag Analysis)

        Analyses which part-of-speech categories of typo-affected words are most
        associated with incorrect predictions.

        Uses NLTK POS tagging on the **original** (pre-typo) words that received typos.
        """
    )
    return


@app.cell
def _(nltk):
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    return


@app.cell
def _(df_metrics, mo, n_results, PERPLEXITY_BENCHMARKS):
    _err_models = (
        sorted(df_metrics["model_short"].unique().tolist())
        if n_results > 0
        else ["Llama-3.2-3B"]
    )
    _err_benchmarks = (
        sorted(
            b
            for b in df_metrics["benchmark"].unique().tolist()
            if b not in PERPLEXITY_BENCHMARKS
        )
        if n_results > 0
        else ["hellaswag"]
    )
    _err_quants = (
        sorted(df_metrics["quant_condition"].unique().tolist())
        if n_results > 0
        else ["none_w16"]
    )

    error_model_selector = mo.ui.dropdown(
        options=_err_models,
        value=_err_models[0] if _err_models else None,
        label="Model (error analysis)",
    )
    error_bench_selector = mo.ui.dropdown(
        options=_err_benchmarks,
        value=_err_benchmarks[0] if _err_benchmarks else None,
        label="Benchmark (error analysis)",
    )
    error_quant_selector = mo.ui.dropdown(
        options=_err_quants,
        value=_err_quants[0] if _err_quants else None,
        label="Quantization (error analysis)",
    )
    error_typo_selector = mo.ui.dropdown(
        options=["swap_n1", "swap_n2", "swap_n4", "random_n4", "replace_n4"],
        value="swap_n4",
        label="Typo condition (error analysis)",
    )
    mo.hstack(
        [error_model_selector, error_bench_selector, error_quant_selector, error_typo_selector],
        justify="start",
        gap=1,
    )
    return (
        error_bench_selector,
        error_model_selector,
        error_quant_selector,
        error_typo_selector,
    )


@app.cell
def _(
    error_bench_selector,
    error_model_selector,
    error_quant_selector,
    error_typo_selector,
    load_predictions,
    mo,
    nltk,
    pd,
    px,
    re,
):
    def _pos_tag_word(word):
        """POS tag a single word, stripping punctuation."""
        clean = re.sub(r"[^\w]", "", word)
        if not clean:
            return "PUNCT"
        tagged = nltk.pos_tag([clean])
        return tagged[0][1] if tagged else "UNK"

    POS_GROUPS = {
        "Noun": {"NN", "NNS", "NNP", "NNPS"},
        "Verb": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
        "Adjective": {"JJ", "JJR", "JJS"},
        "Adverb": {"RB", "RBR", "RBS"},
        "Pronoun": {"PRP", "PRP$", "WP", "WP$"},
        "Determiner": {"DT", "PDT", "WDT"},
        "Preposition": {"IN", "TO"},
        "Conjunction": {"CC"},
        "Numeral": {"CD"},
    }

    def _group_pos(tag):
        for group, tags in POS_GROUPS.items():
            if tag in tags:
                return group
        return "Other"

    def _build_error_analysis():
        _model = error_model_selector.value
        _bench = error_bench_selector.value
        _quant = error_quant_selector.value
        _typo = error_typo_selector.value

        if not all([_model, _bench, _quant, _typo]):
            return mo.md("Please select all error analysis parameters.")

        preds = load_predictions(_model, _quant, _bench, _typo)
        if not preds:
            return mo.md(
                f"No predictions found for **{_model}** / **{_quant}** / **{_bench}** / **{_typo}**. "
                "Data may not be available yet."
            )

        rows = []
        for pred in preds:
            annotations = pred.get("typo_annotations", [])
            correct = pred.get("correct", None)
            if correct is None:
                continue
            for ann in annotations:
                orig_word = ann.get("original_word", "")
                pos_tag = _pos_tag_word(orig_word)
                pos_group = _group_pos(pos_tag)
                rows.append(
                    {
                        "example_id": pred.get("example_id", ""),
                        "original_word": orig_word,
                        "typo_word": ann.get("typo_word", ""),
                        "pos_tag": pos_tag,
                        "pos_group": pos_group,
                        "correct": correct,
                    }
                )

        if not rows:
            return mo.md("No typo annotations found in predictions.")

        _df = pd.DataFrame(rows)

        _agg = (
            _df.groupby(["pos_group", "correct"])
            .size()
            .reset_index(name="count")
        )
        _agg["outcome"] = _agg["correct"].map({True: "Correct", False: "Incorrect"})

        _totals = _df.groupby("pos_group").agg(
            total=("correct", "count"),
            n_incorrect=("correct", lambda x: (~x).sum()),
        ).reset_index()
        _totals["error_rate"] = _totals["n_incorrect"] / _totals["total"]
        _totals = _totals.sort_values("error_rate", ascending=False)

        fig1 = px.bar(
            _agg,
            x="pos_group",
            y="count",
            color="outcome",
            barmode="stack",
            title=f"POS Distribution: {_model} / {_quant} / {_bench} / {_typo}",
            labels={"pos_group": "POS Group", "count": "Count", "outcome": "Outcome"},
            color_discrete_map={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
        )

        fig2 = px.bar(
            _totals,
            x="pos_group",
            y="error_rate",
            title=f"Error Rate by POS Group: {_model} / {_quant} / {_bench} / {_typo}",
            labels={"pos_group": "POS Group", "error_rate": "Error Rate"},
            color="error_rate",
            color_continuous_scale="Reds",
        )
        fig2.update_layout(showlegend=False)

        return mo.vstack(
            [
                mo.md(
                    f"**{len(preds)}** predictions loaded, "
                    f"**{len(_df)}** typo annotations, "
                    f"**{_totals['n_incorrect'].sum():.0f}** in incorrect predictions."
                ),
                mo.ui.plotly(fig1),
                mo.ui.plotly(fig2),
            ]
        )

    _build_error_analysis()
    return


# ---------------------------------------------------------------------------
# Error flip analysis: clean vs typo
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ### Error Flip Analysis

        Compare predictions between **clean** and **typo** conditions:
        which examples flip from correct to incorrect (or vice versa) when typos are introduced?
        """
    )
    return


@app.cell
def _(
    error_bench_selector,
    error_model_selector,
    error_quant_selector,
    error_typo_selector,
    load_predictions,
    mo,
    pd,
    px,
):
    def _build_flip_analysis():
        _model = error_model_selector.value
        _bench = error_bench_selector.value
        _quant = error_quant_selector.value
        _typo = error_typo_selector.value

        if not all([_model, _bench, _quant, _typo]):
            return mo.md("Please select all parameters.")

        clean_preds = load_predictions(_model, _quant, _bench, "clean")
        typo_preds = load_predictions(_model, _quant, _bench, _typo)

        if not clean_preds or not typo_preds:
            return mo.md(
                f"Need both clean and typo predictions for **{_model}** / **{_quant}** / **{_bench}**."
            )

        clean_by_id = {p["example_id"]: p.get("correct", None) for p in clean_preds}
        typo_by_id = {p["example_id"]: p.get("correct", None) for p in typo_preds}

        common_ids = set(clean_by_id.keys()) & set(typo_by_id.keys())

        categories = {
            "Correct -> Correct": 0,
            "Correct -> Incorrect": 0,
            "Incorrect -> Correct": 0,
            "Incorrect -> Incorrect": 0,
        }
        for eid in common_ids:
            c = clean_by_id[eid]
            t = typo_by_id[eid]
            if c is None or t is None:
                continue
            if c and t:
                categories["Correct -> Correct"] += 1
            elif c and not t:
                categories["Correct -> Incorrect"] += 1
            elif not c and t:
                categories["Incorrect -> Correct"] += 1
            else:
                categories["Incorrect -> Incorrect"] += 1

        _df = pd.DataFrame([{"Category": k, "Count": v} for k, v in categories.items()])

        fig = px.bar(
            _df,
            x="Category",
            y="Count",
            color="Category",
            title=f"Prediction Flips (clean vs {_typo}): {_model} / {_quant} / {_bench}",
            color_discrete_map={
                "Correct -> Correct": "#2ecc71",
                "Correct -> Incorrect": "#e74c3c",
                "Incorrect -> Correct": "#3498db",
                "Incorrect -> Incorrect": "#95a5a6",
            },
        )
        fig.update_layout(showlegend=False)

        total = sum(categories.values())
        originally_correct = categories["Correct -> Correct"] + categories["Correct -> Incorrect"]
        flip_rate = (
            categories["Correct -> Incorrect"] / originally_correct
            if originally_correct > 0
            else 0.0
        )

        return mo.vstack(
            [
                mo.md(
                    f"**{total}** common examples. "
                    f"**{categories['Correct -> Incorrect']}** flipped from correct to incorrect "
                    f"(flip rate: **{flip_rate:.1%}** of originally correct)."
                ),
                mo.ui.plotly(fig),
            ]
        )

    _build_flip_analysis()
    return


# ===========================================================================
# SECTION 5: JSON Summary Export
# ===========================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. JSON Summary Export

        Generate aggregated JSON files for downstream analysis.
        Click the button below to export summaries to `analysis/summary/`.
        """
    )
    return


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export JSON Summaries")
    export_button
    return (export_button,)


@app.cell
def _(
    CALIBRATION_PAIRS,
    df_metrics,
    error_bench_selector,
    error_model_selector,
    error_quant_selector,
    error_typo_selector,
    export_button,
    json,
    load_predictions,
    mo,
    n_results,
    nltk,
    os,
    Path,
    pd,
    PERPLEXITY_BENCHMARKS,
    re,
    TYPO_ORDER,
):
    def _pos_tag_single(word):
        clean = re.sub(r"[^\w]", "", word)
        if not clean:
            return "PUNCT"
        tagged = nltk.pos_tag([clean])
        return tagged[0][1] if tagged else "UNK"

    _EXPORT_POS_GROUPS = {
        "Noun": {"NN", "NNS", "NNP", "NNPS"},
        "Verb": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
        "Adjective": {"JJ", "JJR", "JJS"},
        "Adverb": {"RB", "RBR", "RBS"},
        "Pronoun": {"PRP", "PRP$", "WP", "WP$"},
        "Determiner": {"DT", "PDT", "WDT"},
        "Preposition": {"IN", "TO"},
        "Conjunction": {"CC"},
        "Numeral": {"CD"},
    }

    def _group_pos_export(tag):
        for group, tags in _EXPORT_POS_GROUPS.items():
            if tag in tags:
                return group
        return "Other"

    def _export_summaries():
        if not export_button.value:
            return mo.md("Click the button above to export.")

        if n_results == 0:
            return mo.md("**No data to export.**")

        summary_dir = Path(__file__).resolve().parent / "summary"
        os.makedirs(summary_dir, exist_ok=True)
        messages = []

        # --- 1. performance_by_condition.json ---
        perf_records = []
        for _, row in df_metrics.iterrows():
            rec = {
                "model": row.get("model_short", ""),
                "quant_condition": row.get("quant_condition", ""),
                "benchmark": row.get("benchmark", ""),
                "typo_condition": row.get("typo_condition", ""),
            }
            if row.get("benchmark", "") in PERPLEXITY_BENCHMARKS:
                rec["mean_perplexity"] = row.get("mean_perplexity")
            else:
                rec["accuracy"] = row.get("accuracy")
            perf_records.append(rec)

        with open(summary_dir / "performance_by_condition.json", "w") as f:
            json.dump(perf_records, f, indent=2, default=str)
        messages.append(f"performance_by_condition.json: {len(perf_records)} records")

        # --- 2. gap_analysis.json ---
        gap_records = []
        for bench in df_metrics["benchmark"].unique():
            _is_ppl = bench in PERPLEXITY_BENCHMARKS
            _metric_col = "mean_perplexity" if _is_ppl else "accuracy"
            for model in df_metrics["model_short"].unique():
                for qcond in df_metrics["quant_condition"].unique():
                    _clean_mask = (
                        (df_metrics["model_short"] == model)
                        & (df_metrics["benchmark"] == bench)
                        & (df_metrics["quant_condition"] == qcond)
                        & (df_metrics["typo_condition"] == "clean")
                    )
                    _clean_rows = df_metrics[_clean_mask]
                    if _clean_rows.empty:
                        continue
                    clean_val = _clean_rows.iloc[0][_metric_col]
                    if clean_val is None or pd.isna(clean_val):
                        continue
                    for typo_cond in TYPO_ORDER[1:]:
                        _typo_mask = (
                            (df_metrics["model_short"] == model)
                            & (df_metrics["benchmark"] == bench)
                            & (df_metrics["quant_condition"] == qcond)
                            & (df_metrics["typo_condition"] == typo_cond)
                        )
                        _typo_rows = df_metrics[_typo_mask]
                        if _typo_rows.empty:
                            continue
                        typo_val = _typo_rows.iloc[0][_metric_col]
                        if typo_val is None or pd.isna(typo_val):
                            continue
                        if _is_ppl:
                            gap = float(typo_val - clean_val)
                        else:
                            gap = float(clean_val - typo_val)
                        gap_records.append(
                            {
                                "model": model,
                                "quant_condition": qcond,
                                "benchmark": bench,
                                "typo_condition": typo_cond,
                                "clean_value": float(clean_val),
                                "typo_value": float(typo_val),
                                "gap": gap,
                                "metric_type": "perplexity" if _is_ppl else "accuracy",
                            }
                        )

        with open(summary_dir / "gap_analysis.json", "w") as f:
            json.dump(gap_records, f, indent=2, default=str)
        messages.append(f"gap_analysis.json: {len(gap_records)} records")

        # --- 3. calibration_effect.json ---
        cal_records = []
        for clean_q, noisy_q in CALIBRATION_PAIRS:
            method_bits = clean_q.rsplit("_", 1)[0]
            for model in df_metrics["model_short"].unique():
                for bench in df_metrics["benchmark"].unique():
                    _is_ppl = bench in PERPLEXITY_BENCHMARKS
                    _metric_col = "mean_perplexity" if _is_ppl else "accuracy"
                    for typo_cond in TYPO_ORDER:
                        vals = {}
                        for qcond, cal_type in [(clean_q, "clean"), (noisy_q, "noisy")]:
                            _mask = (
                                (df_metrics["model_short"] == model)
                                & (df_metrics["benchmark"] == bench)
                                & (df_metrics["quant_condition"] == qcond)
                                & (df_metrics["typo_condition"] == typo_cond)
                            )
                            _rows = df_metrics[_mask]
                            if not _rows.empty:
                                v = _rows.iloc[0][_metric_col]
                                if v is not None and pd.notna(v):
                                    vals[cal_type] = float(v)
                        if "clean" in vals and "noisy" in vals:
                            cal_records.append(
                                {
                                    "model": model,
                                    "method_bits": method_bits,
                                    "benchmark": bench,
                                    "typo_condition": typo_cond,
                                    "clean_cal_value": vals["clean"],
                                    "noisy_cal_value": vals["noisy"],
                                    "delta": vals["noisy"] - vals["clean"],
                                    "metric_type": "perplexity" if _is_ppl else "accuracy",
                                }
                            )

        with open(summary_dir / "calibration_effect.json", "w") as f:
            json.dump(cal_records, f, indent=2, default=str)
        messages.append(f"calibration_effect.json: {len(cal_records)} records")

        # --- 4. error_pos_analysis.json ---
        _model = error_model_selector.value
        _bench = error_bench_selector.value
        _quant = error_quant_selector.value
        _typo = error_typo_selector.value
        pos_records = []

        if _model and _bench and _quant and _typo:
            preds = load_predictions(_model, _quant, _bench, _typo)
            pos_counts = {}
            for pred in preds:
                correct = pred.get("correct", None)
                if correct is None:
                    continue
                for ann in pred.get("typo_annotations", []):
                    orig = ann.get("original_word", "")
                    tag = _pos_tag_single(orig)
                    group = _group_pos_export(tag)
                    key = (group, correct)
                    pos_counts[key] = pos_counts.get(key, 0) + 1

            for (group, correct), count in sorted(pos_counts.items()):
                pos_records.append(
                    {
                        "model": _model,
                        "quant_condition": _quant,
                        "benchmark": _bench,
                        "typo_condition": _typo,
                        "pos_group": group,
                        "correct": correct,
                        "count": count,
                    }
                )

        with open(summary_dir / "error_pos_analysis.json", "w") as f:
            json.dump(pos_records, f, indent=2, default=str)
        messages.append(f"error_pos_analysis.json: {len(pos_records)} records")

        return mo.md("**Export complete.**\n\n" + "\n".join(f"- {m}" for m in messages))

    _export_summaries()
    return


# ---------------------------------------------------------------------------
# Cell: footer
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        *Dashboard generated by `marimo`. Run with:*

        ```bash
        uv run --package quant-typo-neuron --extra analysis marimo run projects/quant-typo-neuron/analysis/dashboard.py
        ```
        """
    )
    return


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
