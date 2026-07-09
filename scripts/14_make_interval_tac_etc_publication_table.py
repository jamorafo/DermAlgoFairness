#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create publication-ready interval TAC/ETC consistency tables.

Input:
  outputs/tables/interval_tac_etc_consistency.csv

Outputs:
  outputs/publication_tables/table_04_interval_tac_etc_consistency.csv
  outputs/publication_tables/table_04_interval_tac_etc_consistency.tex

This version avoids pandas.to_latex() so it does not require jinja2.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PUB = ROOT / "outputs" / "publication_tables"

PUB.mkdir(parents=True, exist_ok=True)

INPUT = TABLES / "interval_tac_etc_consistency.csv"

METRIC_ORDER = ["recall", "auc_pr", "f1", "precision"]
TARGET_ORDER = ["BOSQUE overall", "BOSQUE light", "BOSQUE dark"]

REGION_ORDER = [
    "adequate and transported",
    "inconclusive",
    "not adequate and not transported",
]

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
}

TARGET_LABELS = {
    "BOSQUE overall": "Overall",
    "BOSQUE light": "Light phototype",
    "BOSQUE dark": "Dark phototype",
}


def latex_escape(value):
    """Minimal LaTeX escaping for table text."""
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
    """Convert long consistency table into publication-wide format."""

    full_index = pd.MultiIndex.from_product(
        [METRIC_ORDER, TARGET_ORDER, REGION_ORDER],
        names=["metric", "target_condition", "joint_region"],
    )

    df_full = (
        df.set_index(["metric", "target_condition", "joint_region"])
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

    wide = wide.sort_values(["metric", "target_condition"]).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Metric": wide["metric"].map(METRIC_LABELS),
            "Target condition": wide["target_condition"].map(TARGET_LABELS),
            "Adequate and transported": wide["adequate and transported"].astype(int),
            "Inconclusive": wide["inconclusive"].astype(int),
            "Not adequate and not transported": wide[
                "not adequate and not transported"
            ].astype(int),
        }
    )

    out["Total model--seed decisions"] = (
        out["Adequate and transported"]
        + out["Inconclusive"]
        + out["Not adequate and not transported"]
    )

    return out


def build_latex_table(out):
    """Build a LaTeX table manually, without pandas.to_latex()."""

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Interval-based TAC/ETC decision consistency across model--seed replicas.}"
    )
    lines.append(r"\label{tab:interval-tac-etc-consistency}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metric} & "
        r"\textbf{Target condition} & "
        r"\textbf{Adeq. + transp.} & "
        r"\textbf{Inconc.} & "
        r"\textbf{Not adeq. + not transp.} & "
        r"\textbf{Total} \\"
    )
    lines.append(r"\midrule")

    previous_metric = None

    for _, row in out.iterrows():
        metric = latex_escape(row["Metric"])
        target = latex_escape(row["Target condition"])

        # Print metric only once per block for a cleaner table.
        metric_cell = metric if metric != previous_metric else ""

        lines.append(
            f"{metric_cell} & "
            f"{target} & "
            f"{int(row['Adequate and transported'])} & "
            f"{int(row['Inconclusive'])} & "
            f"{int(row['Not adequate and not transported'])} & "
            f"{int(row['Total model--seed decisions'])} \\\\"
        )

        previous_metric = metric

        if target == "Dark phototype":
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: Each row summarizes 25 model--seed decisions, corresponding to "
        r"five architectures trained under five random seeds. TAC denotes the "
        r"Target Adequacy Criterion and ETC denotes the External Transportability "
        r"Criterion. Decisions are interval-based. ``Adeq. + transp.'' indicates "
        r"that the lower confidence bound for target performance exceeded the "
        r"adequacy threshold and the upper confidence bound for source-to-target "
        r"degradation did not exceed the admissible degradation margin. "
        r"``Not adeq. + not transp.'' indicates that the upper confidence bound "
        r"for target performance remained below the adequacy threshold and the "
        r"lower confidence bound for degradation exceeded the admissible margin. "
        r"All other cases are classified as inconclusive."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "metric",
        "target_condition",
        "joint_region",
        "n_model_seed_decisions",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    out = build_publication_table(df)

    csv_path = PUB / "table_04_interval_tac_etc_consistency.csv"
    tex_path = PUB / "table_04_interval_tac_etc_consistency.tex"

    out.to_csv(csv_path, index=False)

    latex = build_latex_table(out)
    tex_path.write_text(latex, encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
