#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Source-side Predictive Representativity diagnostics.

This script extracts one unique source-side performance estimate and confidence
interval per model--seed--metric from the interval TAC/ETC analysis and
summarizes the source ORP precision across the 25 locked model replicas.

Input:
  outputs/tables/interval_tac_etc_by_seed.csv

Outputs:
  outputs/tables/source_pr_diagnostics_by_model_seed.csv
  outputs/publication_tables/table_07_source_pr_diagnostics.csv
  outputs/publication_tables/table_07_source_pr_diagnostics.tex

Important:
  interval_tac_etc_by_seed.csv repeats the same source model--seed--metric
  information once per target condition: BOSQUE overall, BOSQUE light, and
  BOSQUE dark. For source-side PR diagnostics, this script keeps only the
  BOSQUE overall copy so each source model--seed--metric is counted once.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PUB = ROOT / "outputs" / "publication_tables"

TABLES.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)

INPUT = TABLES / "interval_tac_etc_by_seed.csv"

METRIC_ORDER = ["recall", "auc_pr", "f1", "precision"]

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
}

# Precision tolerance for source-side PR.
# This is not a clinical adequacy threshold. It is a documentation threshold
# for the maximum acceptable CI half-width of the source performance estimate.
SOURCE_HALF_WIDTH_TOLERANCE = {
    "recall": 0.075,
    "auc_pr": 0.075,
    "f1": 0.075,
    "precision": 0.075,
}


def latex_escape(value):
    """Minimal LaTeX escaping for table text."""
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def fmt(x, digits=3):
    """Format numeric output for LaTeX."""
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def build_latex_table(summary):
    """Build publication-ready LaTeX table manually."""

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Source-side Predictive Representativity diagnostics for the HAM10000 internal evaluation ORP.}"
    )
    lines.append(r"\label{tab:source-pr-diagnostics}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metric} & "
        r"\textbf{$n$ decisions} & "
        r"\textbf{$n_S$} & "
        r"\textbf{Mean source} & "
        r"\textbf{SD seeds} & "
        r"\textbf{Mean SE} & "
        r"\textbf{Max half-width} & "
        r"\textbf{Precision adequate} \\"
    )
    lines.append(r"\midrule")

    for _, row in summary.iterrows():
        lines.append(
            f"{latex_escape(row['Metric'])} & "
            f"{int(row['n_model_seed'])} & "
            f"{int(row['n_source'])} & "
            f"{fmt(row['mean_source'])} & "
            f"{fmt(row['sd_across_seeds'])} & "
            f"{fmt(row['mean_se'])} & "
            f"{fmt(row['max_half_width'])} & "
            f"{int(row['n_precision_adequate'])}/{int(row['n_model_seed'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: Each row summarizes 25 source-side estimates, corresponding to "
        r"five architectures trained under five random seeds. Source uncertainty "
        r"is computed on the non-augmented HAM10000 internal test ORP for each "
        r"locked model. The approximate standard error is obtained from the "
        r"bootstrap confidence interval half-width divided by 1.96. The "
        r"precision-adequacy count uses a documentation tolerance of CI half-width "
        r"$\leq 0.075$; this tolerance is not a clinical performance threshold "
        r"and can be modified in the script. Because the interval TAC/ETC table "
        r"repeats source estimates across target conditions, only the BOSQUE "
        r"overall rows are retained for this source-side diagnostic."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "model",
        "seed",
        "metric",
        "target_condition",
        "n_source",
        "source_performance",
        "source_ci_low",
        "source_ci_high",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT}: {missing}")

    # ------------------------------------------------------------------
    # Important correction:
    # interval_tac_etc_by_seed.csv contains repeated source estimates:
    # one copy for BOSQUE overall, one for BOSQUE light, one for BOSQUE dark.
    # Source-side PR should count each model--seed--metric once only.
    # ------------------------------------------------------------------
    df_src = df[df["target_condition"] == "BOSQUE overall"].copy()

    src = (
        df_src[
            [
                "model",
                "seed",
                "metric",
                "n_source",
                "source_performance",
                "source_ci_low",
                "source_ci_high",
            ]
        ]
        .drop_duplicates(subset=["model", "seed", "metric"])
        .copy()
    )

    expected_n = 25 * len(METRIC_ORDER)
    if len(src) != expected_n:
        print(
            "WARNING: Expected "
            f"{expected_n} unique source model--seed--metric rows "
            f"but found {len(src)}."
        )
        print("Counts by metric:")
        print(src["metric"].value_counts().to_string())

    src["source_half_width"] = (src["source_ci_high"] - src["source_ci_low"]) / 2
    src["source_se_approx"] = src["source_half_width"] / 1.96
    src["source_precision_tolerance"] = src["metric"].map(
        SOURCE_HALF_WIDTH_TOLERANCE
    )
    src["source_precision_adequate"] = (
        src["source_half_width"] <= src["source_precision_tolerance"]
    )

    src["metric"] = pd.Categorical(
        src["metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    src = src.sort_values(["metric", "model", "seed"]).reset_index(drop=True)

    by_seed_path = TABLES / "source_pr_diagnostics_by_model_seed.csv"
    src.to_csv(by_seed_path, index=False)

    summary = (
        src.groupby("metric", observed=False)
        .agg(
            n_model_seed=("source_performance", "size"),
            n_source=("n_source", "first"),
            mean_source=("source_performance", "mean"),
            sd_across_seeds=("source_performance", "std"),
            mean_se=("source_se_approx", "mean"),
            max_se=("source_se_approx", "max"),
            mean_half_width=("source_half_width", "mean"),
            max_half_width=("source_half_width", "max"),
            n_precision_adequate=("source_precision_adequate", "sum"),
        )
        .reset_index()
    )

    summary["Metric"] = summary["metric"].map(METRIC_LABELS)

    summary = summary[
        [
            "Metric",
            "n_model_seed",
            "n_source",
            "mean_source",
            "sd_across_seeds",
            "mean_se",
            "max_se",
            "mean_half_width",
            "max_half_width",
            "n_precision_adequate",
        ]
    ]

    summary_path = PUB / "table_07_source_pr_diagnostics.csv"
    tex_path = PUB / "table_07_source_pr_diagnostics.tex"

    summary.to_csv(summary_path, index=False)
    tex_path.write_text(build_latex_table(summary), encoding="utf-8")

    print(f"Saved: {by_seed_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {tex_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
