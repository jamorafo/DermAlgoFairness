#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finite-ORP precision diagnostics for the BOSQUE target ORPs.

The script summarizes target interval half-widths for BOSQUE overall, light
phototype, and dark phototype conditions. It evaluates finite-ORP precision
only; it does not establish full Predictive Representativity, clinical
adequacy, bias control, or population alignment.

Input:
  outputs/tables/interval_tac_etc_by_seed.csv

Outputs:
  outputs/tables/target_pr_diagnostics_by_model_seed.csv
  outputs/publication_tables/table_08_target_pr_diagnostics.csv
  outputs/publication_tables/table_08_target_pr_diagnostics.tex
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

TARGET_ORDER = [
    "BOSQUE overall",
    "BOSQUE light",
    "BOSQUE dark",
]

TARGET_SIZES = {
    "BOSQUE overall": 151,
    "BOSQUE light": 105,
    "BOSQUE dark": 46,
}

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
    "accuracy": "Accuracy",
    "specificity": "Specificity",
    "auc_roc": "AUC-ROC",
}

TARGET_LABELS = {
    "BOSQUE overall": "Overall",
    "BOSQUE light": "Light phototype",
    "BOSQUE dark": "Dark phototype",
}

TARGET_HALF_WIDTH_TOLERANCE = {
    metric: 0.100 for metric in METRIC_ORDER
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
        r"\scriptsize",
        (
            r"\caption{Finite-ORP precision of BOSQUE overall and "
            r"phototype-specific target performance estimates.}"
        ),
        r"\label{tab:target-pr-diagnostics}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Metric} & "
            r"\textbf{Target} & "
            r"\textbf{$n$ systems} & "
            r"\textbf{$n_T$} & "
            r"\textbf{Mean performance} & "
            r"\textbf{SD systems} & "
            r"\textbf{Mean half-width} & "
            r"\textbf{Max half-width} & "
            r"\textbf{Within tolerance} \\"
        ),
        r"\midrule",
    ]

    previous_metric = None

    for _, row in summary.iterrows():
        metric = latex_escape(row["Metric"])
        metric_cell = (
            metric if metric != previous_metric else ""
        )

        lines.append(
            f"{metric_cell} & "
            f"{latex_escape(row['Target condition'])} & "
            f"{int(row['n_model_seed'])} & "
            f"{int(row['n_target'])} & "
            f"{fmt(row['mean_target'])} & "
            f"{fmt(row['sd_across_systems'])} & "
            f"{fmt(row['mean_half_width'])} & "
            f"{fmt(row['max_half_width'])} & "
            f"{int(row['n_within_tolerance'])}/"
            f"{int(row['n_model_seed'])} \\\\"
        )

        previous_metric = metric

        if row["Target condition"] == "Dark phototype":
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: Each row summarizes 25 locked architecture--seed "
                r"systems evaluated on a non-augmented BOSQUE target condition. "
                r"Mean performance and SD describe variation across the 25 "
                r"locked systems; the SD therefore combines architecture and "
                r"training-seed variation. Precision is summarized by the "
                r"half-width of the 95\% image-level percentile-bootstrap "
                r"interval based on 2000 replicates. ``Within tolerance'' "
                r"denotes a half-width $\leq 0.100$. This documentation "
                r"tolerance is not a clinical threshold. Satisfying it "
                r"establishes only finite-ORP precision under the observed "
                r"BOSQUE condition, not full Predictive Representativity, "
                r"bias control, interval coverage, or generalizability beyond "
                r"BOSQUE."
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
        "n_target",
        "target_performance",
        "target_ci_low",
        "target_ci_high",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {INPUT}: {sorted(missing)}"
        )

    target = (
        df[
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
        ]
        .drop_duplicates(
            ["model", "seed", "metric", "target_condition"]
        )
        .copy()
    )

    keys = [
        "model",
        "seed",
        "metric",
        "target_condition",
    ]

    if target.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate target model--seed--metric--condition rows remain."
        )

    expected_rows = 25 * len(METRIC_ORDER) * len(TARGET_ORDER)
    if len(target) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} target rows, found {len(target)}."
        )

    for condition, expected_n in TARGET_SIZES.items():
        observed_n = set(
            target.loc[
                target["target_condition"] == condition,
                "n_target",
            ]
        )

        if observed_n != {expected_n}:
            raise RuntimeError(
                f"{condition}: expected n={expected_n}, "
                f"found {sorted(observed_n)}."
            )

    target["target_half_width"] = (
        target["target_ci_high"] - target["target_ci_low"]
    ) / 2

    target["target_half_width_tolerance"] = target["metric"].map(
        TARGET_HALF_WIDTH_TOLERANCE
    )

    if target["target_half_width_tolerance"].isna().any():
        unknown = target.loc[
            target["target_half_width_tolerance"].isna(),
            "metric",
        ].unique()
        raise RuntimeError(f"Missing tolerances for metrics: {unknown}")

    target["target_within_tolerance"] = (
        target["target_half_width"]
        <= target["target_half_width_tolerance"]
    )

    target["metric"] = pd.Categorical(
        target["metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    target["target_condition"] = pd.Categorical(
        target["target_condition"],
        categories=TARGET_ORDER,
        ordered=True,
    )

    target = target.sort_values(
        ["metric", "target_condition", "model", "seed"]
    ).reset_index(drop=True)

    by_system_path = (
        TABLES / "target_pr_diagnostics_by_model_seed.csv"
    )
    target.to_csv(by_system_path, index=False)

    summary = (
        target.groupby(
            ["metric", "target_condition"],
            observed=True,
        )
        .agg(
            n_model_seed=("target_performance", "size"),
            n_target=("n_target", "first"),
            mean_target=("target_performance", "mean"),
            sd_across_systems=("target_performance", "std"),
            mean_half_width=("target_half_width", "mean"),
            max_half_width=("target_half_width", "max"),
            half_width_tolerance=(
                "target_half_width_tolerance",
                "first",
            ),
            n_within_tolerance=(
                "target_within_tolerance",
                "sum",
            ),
        )
        .reset_index()
    )

    if len(summary) != 21:
        raise RuntimeError(
            f"Expected 21 target summary rows, found {len(summary)}."
        )

    if not (summary["n_model_seed"] == 25).all():
        raise RuntimeError(
            "At least one target metric-condition does not contain "
            "25 systems."
        )

    summary["Metric"] = (
        summary["metric"].astype(str).map(METRIC_LABELS)
    )

    summary["Target condition"] = (
        summary["target_condition"]
        .astype(str)
        .map(TARGET_LABELS)
    )

    summary = summary[
        [
            "Metric",
            "Target condition",
            "n_model_seed",
            "n_target",
            "mean_target",
            "sd_across_systems",
            "mean_half_width",
            "max_half_width",
            "half_width_tolerance",
            "n_within_tolerance",
        ]
    ]

    summary_path = (
        PUB / "table_08_target_pr_diagnostics.csv"
    )
    tex_path = (
        PUB / "table_08_target_pr_diagnostics.tex"
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
