#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Target-side Predictive Representativity diagnostics.

This script extracts target-side performance estimates and confidence intervals
from the interval TAC/ETC analysis and summarizes target ORP precision by
metric and target condition.

Input:
  outputs/tables/interval_tac_etc_by_seed.csv

Outputs:
  outputs/tables/target_pr_diagnostics_by_model_seed.csv
  outputs/publication_tables/table_08_target_pr_diagnostics.csv
  outputs/publication_tables/table_08_target_pr_diagnostics.tex

Purpose:
  This table documents whether BOSQUE overall, BOSQUE light, and BOSQUE dark
  provide sufficiently precise target-side estimates for the selected metrics.
  It complements the source-side PR diagnostic table.
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
TARGET_ORDER = ["BOSQUE overall", "BOSQUE light", "BOSQUE dark"]

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
    "accuracy": "Accuracy",
    "specificity": "Specificity",
    "auc_roc": "AUC-ROC",
}

METRIC_GROUPS = {
    "recall": "Primary",
    "auc_pr": "Primary",
    "f1": "Primary",
    "precision": "Primary",
    "accuracy": "Secondary",
    "specificity": "Secondary",
    "auc_roc": "Secondary",
}

TARGET_LABELS = {
    "BOSQUE overall": "Overall",
    "BOSQUE light": "Light phototype",
    "BOSQUE dark": "Dark phototype",
}

# Documentation tolerance for CI half-width.
# This is not a clinical adequacy threshold. It is a PR documentation threshold
# for finite-ORP precision of the target estimate.
TARGET_HALF_WIDTH_TOLERANCE = {
    "recall": 0.100,
    "auc_pr": 0.100,
    "f1": 0.100,
    "precision": 0.100,
    "accuracy": 0.100,
    "specificity": 0.100,
    "auc_roc": 0.100,
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
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def pr_interpretation(row):
    """
    Conservative PR interpretation based on precision-adequacy counts.

    This does not decide clinical adequacy. It only describes whether the
    finite BOSQUE ORP provides stable enough estimates under the chosen
    half-width tolerance.
    """

    n = int(row["n_model_seed"])
    adequate = int(row["n_precision_adequate"])

    if adequate == n:
        return "PR-adequate for BOSQUE-condition estimation"

    if adequate >= 0.8 * n:
        return "Mostly PR-adequate; some estimates imprecise"

    if adequate >= 0.5 * n:
        return "Partially PR-adequate; uncertainty material"

    return "Evidentially limited by finite-ORP uncertainty"


def build_latex_table(summary):
    lines = []

    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Target-side Predictive Representativity diagnostics for the BOSQUE external ORP.}"
    )
    lines.append(r"\label{tab:target-pr-diagnostics}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{llrrrrrrp{4.1cm}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metric} & "
        r"\textbf{Target} & "
        r"\textbf{$n$ decisions} & "
        r"\textbf{$n_T$} & "
        r"\textbf{Mean target} & "
        r"\textbf{SD seeds} & "
        r"\textbf{Mean SE} & "
        r"\textbf{Max half-width} & "
        r"\textbf{PR interpretation} \\"
    )
    lines.append(r"\midrule")

    previous_metric = None

    for _, row in summary.iterrows():
        metric = latex_escape(row["Metric"])
        metric_cell = metric if metric != previous_metric else ""

        lines.append(
            f"{metric_cell} & "
            f"{latex_escape(row['Target condition'])} & "
            f"{int(row['n_model_seed'])} & "
            f"{int(row['n_target'])} & "
            f"{fmt(row['mean_target'])} & "
            f"{fmt(row['sd_across_seeds'])} & "
            f"{fmt(row['mean_se'])} & "
            f"{fmt(row['max_half_width'])} & "
            f"{latex_escape(row['PR interpretation'])} \\\\"
        )

        previous_metric = metric

        if row["Target condition"] == "Dark phototype":
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: Each row summarizes 25 model--seed target estimates, "
        r"corresponding to five architectures trained under five random seeds. "
        r"Target uncertainty is computed on the non-augmented BOSQUE external ORP. "
        r"The approximate standard error is obtained from the bootstrap confidence "
        r"interval half-width divided by 1.96. The PR interpretation concerns "
        r"finite-ORP precision under the BOSQUE evaluation condition; it is not a "
        r"clinical adequacy decision and does not imply population-level "
        r"generalization beyond BOSQUE. The precision-adequacy count uses a "
        r"documentation tolerance of CI half-width $\leq 0.100$, which can be "
        r"modified in the script."
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
        "n_target",
        "target_performance",
        "target_ci_low",
        "target_ci_high",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT}: {missing}")

    tgt = df[
        [
            "model",
            "seed",
            "metric",
            "target_condition",
            "n_target",
            "target_performance",
            "target_ci_low",
            "target_ci_high",
        ]
    ].copy()

    tgt = tgt.drop_duplicates(
        subset=["model", "seed", "metric", "target_condition"]
    ).copy()

    tgt["target_half_width"] = (tgt["target_ci_high"] - tgt["target_ci_low"]) / 2
    tgt["target_se_approx"] = tgt["target_half_width"] / 1.96
    tgt["target_precision_tolerance"] = tgt["metric"].map(
        TARGET_HALF_WIDTH_TOLERANCE
    )
    tgt["target_precision_adequate"] = (
        tgt["target_half_width"] <= tgt["target_precision_tolerance"]
    )

    tgt["metric"] = pd.Categorical(
        tgt["metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    tgt["target_condition"] = pd.Categorical(
        tgt["target_condition"],
        categories=TARGET_ORDER,
        ordered=True,
    )

    tgt = tgt.sort_values(["metric", "target_condition", "model", "seed"]).reset_index(
        drop=True
    )

    by_seed_path = TABLES / "target_pr_diagnostics_by_model_seed.csv"
    tgt.to_csv(by_seed_path, index=False)

    summary = (
        tgt.groupby(["metric", "target_condition"], observed=False)
        .agg(
            n_model_seed=("target_performance", "size"),
            n_target=("n_target", "first"),
            mean_target=("target_performance", "mean"),
            sd_across_seeds=("target_performance", "std"),
            mean_se=("target_se_approx", "mean"),
            max_se=("target_se_approx", "max"),
            mean_half_width=("target_half_width", "mean"),
            max_half_width=("target_half_width", "max"),
            n_precision_adequate=("target_precision_adequate", "sum"),
        )
        .reset_index()
    )

    summary["Metric"] = summary["metric"].map(METRIC_LABELS)
    summary["Target condition"] = summary["target_condition"].map(TARGET_LABELS)
    summary["PR interpretation"] = summary.apply(pr_interpretation, axis=1)

    summary = summary[
        [
            "Metric",
            "Target condition",
            "n_model_seed",
            "n_target",
            "mean_target",
            "sd_across_seeds",
            "mean_se",
            "max_se",
            "mean_half_width",
            "max_half_width",
            "n_precision_adequate",
            "PR interpretation",
        ]
    ]

    summary_path = PUB / "table_08_target_pr_diagnostics.csv"
    tex_path = PUB / "table_08_target_pr_diagnostics.tex"

    summary.to_csv(summary_path, index=False)
    tex_path.write_text(build_latex_table(summary), encoding="utf-8")

    print(f"Saved: {by_seed_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {tex_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
