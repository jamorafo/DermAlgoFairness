#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a publication-ready interval adequacy and preservation table.

Input:
  outputs/tables/interval_tac_etc_consistency.csv

Outputs:
  outputs/publication_tables/table_04_interval_tac_etc_consistency.csv
  outputs/publication_tables/table_04_interval_tac_etc_consistency.tex

This table reports primary and secondary metrics separately.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PUB = ROOT / "outputs" / "publication_tables"
PUB.mkdir(parents=True, exist_ok=True)

INPUT = TABLES / "interval_tac_etc_consistency.csv"

PRIMARY_METRICS = ["recall", "auc_pr", "f1", "precision"]
SECONDARY_METRICS = ["accuracy", "specificity", "auc_roc"]
METRIC_ORDER = PRIMARY_METRICS + SECONDARY_METRICS

TARGET_ORDER = ["BOSQUE overall", "BOSQUE light", "BOSQUE dark"]

REGION_ORDER = [
    "adequate and transported",
    "inconclusive",
    "not adequate and not transported",
    "transported but inadequate",
    "adequate but not transported",
    "evidentially unresolved",
]

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


def build_publication_table(df):
    df = df.copy()

    required = {
        "metric",
        "target_condition",
        "joint_region",
        "n_model_seed_decisions",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    df["metric"] = df["metric"].astype(str)
    df["target_condition"] = df["target_condition"].astype(str)
    df["joint_region"] = df["joint_region"].astype(str)
    df["n_model_seed_decisions"] = pd.to_numeric(
        df["n_model_seed_decisions"],
        errors="coerce",
    ).fillna(0)

    # Important: aggregate first. This prevents non-unique multi-index errors
    # if the consistency table contains repeated rows from previous versions.
    df_agg = (
        df.groupby(
            ["metric", "target_condition", "joint_region"],
            as_index=False,
        )["n_model_seed_decisions"]
        .sum()
    )

    full_index = pd.MultiIndex.from_product(
        [METRIC_ORDER, TARGET_ORDER, REGION_ORDER],
        names=["metric", "target_condition", "joint_region"],
    )

    df_full = (
        df_agg.set_index(["metric", "target_condition", "joint_region"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    wide = (
        df_full.pivot_table(
            index=["metric", "target_condition"],
            columns="joint_region",
            values="n_model_seed_decisions",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for col in REGION_ORDER:
        if col not in wide.columns:
            wide[col] = 0

    wide["metric_group"] = wide["metric"].map(METRIC_GROUPS)

    wide["metric_group"] = pd.Categorical(
        wide["metric_group"],
        categories=["Primary", "Secondary"],
        ordered=True,
    )

    wide["metric"] = pd.Categorical(
        wide["metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    wide["target_condition"] = pd.Categorical(
        wide["target_condition"],
        categories=TARGET_ORDER,
        ordered=True,
    )

    wide = wide.sort_values(
        ["metric_group", "metric", "target_condition"]
    ).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Metric group": wide["metric_group"].astype(str),
            "Metric": wide["metric"].astype(str).map(METRIC_LABELS),
            "Target condition": wide["target_condition"].astype(str).map(TARGET_LABELS),
            "Adequate and preserved": wide["adequate and transported"].astype(int),
            "Inconclusive": wide["inconclusive"].astype(int),
            "Not adequate and not preserved": wide[
                "not adequate and not transported"
            ].astype(int),
            "Preserved but inadequate": wide["transported but inadequate"].astype(int),
            "Adequate but not preserved": wide["adequate but not transported"].astype(int),
            "Evidentially unresolved": wide["evidentially unresolved"].astype(int),
        }
    )

    decision_cols = [
        "Adequate and preserved",
        "Inconclusive",
        "Not adequate and not preserved",
        "Preserved but inadequate",
        "Adequate but not preserved",
        "Evidentially unresolved",
    ]

    out["Total model--seed decisions"] = out[decision_cols].sum(axis=1)

    return out


def build_latex_table(out):
    lines = []

    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Interval-based target adequacy and performance-preservation decisions across locked model--seed systems.}"
    )
    lines.append(r"\label{tab:interval-tac-etc-consistency}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lllrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Group} & "
        r"\textbf{Metric} & "
        r"\textbf{Target} & "
        r"\textbf{Adeq. + pres.} & "
        r"\textbf{Inconc.} & "
        r"\textbf{Not adeq. + not pres.} & "
        r"\textbf{Pres. but inad.} & "
        r"\textbf{Adeq. but not pres.} & "
        r"\textbf{Unresolved} & "
        r"\textbf{Total} \\"
    )
    lines.append(r"\midrule")

    previous_group = None
    previous_metric = None

    for _, row in out.iterrows():
        group = latex_escape(row["Metric group"])
        metric = latex_escape(row["Metric"])

        group_cell = group if group != previous_group else ""
        metric_cell = metric if metric != previous_metric else ""

        lines.append(
            f"{group_cell} & "
            f"{metric_cell} & "
            f"{latex_escape(row['Target condition'])} & "
            f"{int(row['Adequate and preserved'])} & "
            f"{int(row['Inconclusive'])} & "
            f"{int(row['Not adequate and not preserved'])} & "
            f"{int(row['Preserved but inadequate'])} & "
            f"{int(row['Adequate but not preserved'])} & "
            f"{int(row['Evidentially unresolved'])} & "
            f"{int(row['Total model--seed decisions'])} \\\\"
        )

        previous_group = group
        previous_metric = metric

        if row["Target condition"] == "Dark phototype":
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: Each row summarizes 25 locked model--seed decisions, "
        r"corresponding to five architectures and five random seeds. For the "
        r"overall BOSQUE condition, preservation denotes ETC transportability "
        r"relative to the HAM10000 held-out source estimate. For the light- and "
        r"dark-phototype conditions, preservation denotes comparison with the "
        r"overall HAM10000 source benchmark and is not subgroup "
        r"transportability. Primary metrics drive the main interpretation; "
        r"secondary metrics are descriptive. Decisions are based on 95\% "
        r"intervals."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT}")

    df = pd.read_csv(INPUT)
    out = build_publication_table(df)

    csv_path = PUB / "table_04_interval_tac_etc_consistency.csv"
    tex_path = PUB / "table_04_interval_tac_etc_consistency.tex"

    out.to_csv(csv_path, index=False)
    tex_path.write_text(build_latex_table(out), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
