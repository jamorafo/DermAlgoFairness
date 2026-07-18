#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finite-ORP precision diagnostics for the HAM10000 held-out source ORP.

The script extracts one source estimate and percentile-bootstrap interval per
locked model--seed system and metric. It evaluates only interval half-width
against a documentation tolerance; it does not establish full Predictive
Representativity, clinical adequacy, bias control, or population alignment.

Input:
  outputs/tables/interval_tac_etc_by_seed.csv

Outputs:
  outputs/tables/source_pr_diagnostics_by_model_seed.csv
  outputs/publication_tables/table_07_source_pr_diagnostics.csv
  outputs/publication_tables/table_07_source_pr_diagnostics.tex
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PUB = ROOT / "outputs" / "publication_tables"

TABLES.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)

INPUT = TABLES / "interval_tac_etc_by_seed.csv"

PRIMARY_METRICS = ["recall", "auc_pr", "f1", "precision"]
SECONDARY_METRICS = ["accuracy", "specificity", "auc_roc"]
METRIC_ORDER = PRIMARY_METRICS + SECONDARY_METRICS

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
    "accuracy": "Accuracy",
    "specificity": "Specificity",
    "auc_roc": "AUC-ROC",
}

SOURCE_HALF_WIDTH_TOLERANCE = {
    metric: 0.075 for metric in METRIC_ORDER
}


def latex_escape(value):
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


def fmt(value, digits=3):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def build_latex_table(summary):
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        (
            r"\caption{Finite-ORP precision of HAM10000 held-out "
            r"source performance estimates.}"
        ),
        r"\label{tab:source-pr-diagnostics}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Metric} & "
            r"\textbf{$n$ systems} & "
            r"\textbf{$n_S$} & "
            r"\textbf{Mean performance} & "
            r"\textbf{SD systems} & "
            r"\textbf{Mean half-width} & "
            r"\textbf{Max half-width} & "
            r"\textbf{Within tolerance} \\"
        ),
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{latex_escape(row['Metric'])} & "
            f"{int(row['n_model_seed'])} & "
            f"{int(row['n_source'])} & "
            f"{fmt(row['mean_source'])} & "
            f"{fmt(row['sd_across_systems'])} & "
            f"{fmt(row['mean_half_width'])} & "
            f"{fmt(row['max_half_width'])} & "
            f"{int(row['n_within_tolerance'])}/"
            f"{int(row['n_model_seed'])} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: Each row summarizes 25 locked architecture--seed "
                r"systems evaluated on the non-augmented HAM10000 held-out "
                r"source ORP ($n_S=1002$). Mean performance and SD describe "
                r"variation across the 25 locked systems; the SD therefore "
                r"combines architecture and training-seed variation. Precision "
                r"is summarized by the half-width of the 95\% image-level "
                r"percentile-bootstrap interval based on 2000 replicates. "
                r"``Within tolerance'' denotes a half-width $\leq 0.075$. "
                r"This documentation tolerance is not a clinical threshold, "
                r"and satisfying it establishes only finite-ORP precision, "
                r"not full Predictive Representativity, bias control, interval "
                r"coverage, or population-level generalizability."
            ),
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

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
        raise ValueError(
            f"Missing required columns in {INPUT}: {sorted(missing)}"
        )

    # Source estimates are repeated once for each BOSQUE target condition.
    # Keep one copy per locked model--seed system and metric.
    source = (
        df.loc[
            df["target_condition"] == "BOSQUE overall",
            [
                "model",
                "seed",
                "metric",
                "n_source",
                "source_performance",
                "source_ci_low",
                "source_ci_high",
            ],
        ]
        .drop_duplicates(["model", "seed", "metric"])
        .copy()
    )

    if source.duplicated(["model", "seed", "metric"]).any():
        raise RuntimeError("Duplicate source model--seed--metric rows remain.")

    expected_rows = 25 * len(METRIC_ORDER)
    if len(source) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} source rows, found {len(source)}."
        )

    if set(source["n_source"]) != {1002}:
        raise RuntimeError(
            f"Unexpected source sizes: {sorted(source['n_source'].unique())}"
        )

    source["source_half_width"] = (
        source["source_ci_high"] - source["source_ci_low"]
    ) / 2

    source["source_half_width_tolerance"] = source["metric"].map(
        SOURCE_HALF_WIDTH_TOLERANCE
    )

    if source["source_half_width_tolerance"].isna().any():
        unknown = source.loc[
            source["source_half_width_tolerance"].isna(),
            "metric",
        ].unique()
        raise RuntimeError(f"Missing tolerances for metrics: {unknown}")

    source["source_within_tolerance"] = (
        source["source_half_width"]
        <= source["source_half_width_tolerance"]
    )

    source["metric"] = pd.Categorical(
        source["metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    source = source.sort_values(
        ["metric", "model", "seed"]
    ).reset_index(drop=True)

    by_system_path = (
        TABLES / "source_pr_diagnostics_by_model_seed.csv"
    )
    source.to_csv(by_system_path, index=False)

    summary = (
        source.groupby("metric", observed=True)
        .agg(
            n_model_seed=("source_performance", "size"),
            n_source=("n_source", "first"),
            mean_source=("source_performance", "mean"),
            sd_across_systems=("source_performance", "std"),
            mean_half_width=("source_half_width", "mean"),
            max_half_width=("source_half_width", "max"),
            half_width_tolerance=(
                "source_half_width_tolerance",
                "first",
            ),
            n_within_tolerance=(
                "source_within_tolerance",
                "sum",
            ),
        )
        .reset_index()
    )

    if len(summary) != len(METRIC_ORDER):
        raise RuntimeError(
            f"Expected {len(METRIC_ORDER)} summary rows, "
            f"found {len(summary)}."
        )

    if not (summary["n_model_seed"] == 25).all():
        raise RuntimeError(
            "At least one source metric does not contain 25 systems."
        )

    summary["Metric"] = (
        summary["metric"].astype(str).map(METRIC_LABELS)
    )

    summary = summary[
        [
            "Metric",
            "n_model_seed",
            "n_source",
            "mean_source",
            "sd_across_systems",
            "mean_half_width",
            "max_half_width",
            "half_width_tolerance",
            "n_within_tolerance",
        ]
    ]

    summary_path = (
        PUB / "table_07_source_pr_diagnostics.csv"
    )
    tex_path = (
        PUB / "table_07_source_pr_diagnostics.tex"
    )

    summary.to_csv(summary_path, index=False)
    tex_path.write_text(
        build_latex_table(summary),
        encoding="utf-8",
    )

    print(f"Saved: {by_system_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {tex_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
