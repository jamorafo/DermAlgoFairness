#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate LaTeX versions of publication Tables 1--2.

Inputs:
  outputs/publication_tables/table_01_internal_external_performance.csv
  outputs/publication_tables/table_02_bosque_subgroup_performance.csv

Outputs:
  outputs/publication_tables/table_01_internal_external_performance.tex
  outputs/publication_tables/table_02_bosque_subgroup_performance.tex

The script avoids pandas.to_latex() to prevent optional jinja2 dependency issues.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "outputs" / "publication_tables"
PUB.mkdir(parents=True, exist_ok=True)


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
        .replace("±", r"$\pm$")
    )


def write_text(path, content):
    path.write_text(content, encoding="utf-8")
    print(f"Saved: {path}")


def make_table_01():
    """
    Internal HAM10000 versus external BOSQUE performance.
    """

    input_path = PUB / "table_01_internal_external_performance.csv"
    output_path = PUB / "table_01_internal_external_performance.tex"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    # Expected columns:
    # model, dataset, accuracy, precision, recall, specificity, f1, auc_roc, auc_pr
    expected = [
        "model",
        "dataset",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc_roc",
        "auc_pr",
    ]

    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Table 1 missing columns: {missing}")

    dataset_labels = {
        "HAM10000_internal_test": "HAM10000 internal",
        "BOSQUE_public": "BOSQUE external",
    }

    df = df.copy()
    df["dataset"] = df["dataset"].replace(dataset_labels)

    # Sort models in preferred order if present.
    model_order = [
        "ResNet50",
        "DenseNet121",
        "MobileNetV2",
        "EfficientNetV2B0",
        "VGG16",
    ]

    df["model_order"] = df["model"].apply(
        lambda x: model_order.index(x) if x in model_order else 999
    )
    df["dataset_order"] = df["dataset"].apply(
        lambda x: 0 if "HAM10000" in str(x) else 1
    )
    df = df.sort_values(["model_order", "dataset_order"]).drop(
        columns=["model_order", "dataset_order"]
    )

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Internal source and external target performance across five training seeds.}"
    )
    lines.append(r"\label{tab:internal-external-performance}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{llrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Evaluation ORP} & "
        r"\textbf{Acc.} & \textbf{Prec.} & \textbf{Recall} & "
        r"\textbf{Spec.} & \textbf{F1} & \textbf{AUC-ROC} & \textbf{AUC-PR} \\"
    )
    lines.append(r"\midrule")

    previous_model = None

    for _, row in df.iterrows():
        model = latex_escape(row["model"])
        model_cell = model if model != previous_model else ""

        lines.append(
            f"{model_cell} & "
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['accuracy'])} & "
            f"{latex_escape(row['precision'])} & "
            f"{latex_escape(row['recall'])} & "
            f"{latex_escape(row['specificity'])} & "
            f"{latex_escape(row['f1'])} & "
            f"{latex_escape(row['auc_roc'])} & "
            f"{latex_escape(row['auc_pr'])} \\\\"
        )

        previous_model = model

        if "BOSQUE" in str(row["dataset"]):
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: Values are reported as mean $\pm$ standard deviation across five "
        r"independently trained model replicas. HAM10000 internal denotes the "
        r"source-side internal evaluation ORP. BOSQUE external denotes the "
        r"external target-side ORP. Metrics are computed for the malignant class "
        r"under the binary diagnostic task."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    write_text(output_path, "\n".join(lines))


def make_table_02():
    """
    BOSQUE subgroup performance by light/dark phototype.
    """

    input_path = PUB / "table_02_bosque_subgroup_performance.csv"
    output_path = PUB / "table_02_bosque_subgroup_performance.tex"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    expected = [
        "model",
        "subgroup",
        "n",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc_roc",
        "auc_pr",
    ]

    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Table 2 missing columns: {missing}")

    df = df.copy()
    df["subgroup"] = df["subgroup"].replace(
        {
            "light": "Light phototype",
            "dark": "Dark phototype",
        }
    )

    model_order = [
        "ResNet50",
        "DenseNet121",
        "MobileNetV2",
        "EfficientNetV2B0",
        "VGG16",
    ]

    subgroup_order = {
        "Light phototype": 0,
        "Dark phototype": 1,
    }

    df["model_order"] = df["model"].apply(
        lambda x: model_order.index(x) if x in model_order else 999
    )
    df["subgroup_order"] = df["subgroup"].map(subgroup_order).fillna(999)

    df = df.sort_values(["model_order", "subgroup_order"]).drop(
        columns=["model_order", "subgroup_order"]
    )

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{BOSQUE target-side subgroup performance by phototype group.}"
    )
    lines.append(r"\label{tab:bosque-subgroup-performance}")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{llrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Subgroup} & \textbf{$n$} & "
        r"\textbf{Acc.} & \textbf{Prec.} & \textbf{Recall} & "
        r"\textbf{Spec.} & \textbf{F1} & \textbf{AUC-ROC} & \textbf{AUC-PR} \\"
    )
    lines.append(r"\midrule")

    previous_model = None

    for _, row in df.iterrows():
        model = latex_escape(row["model"])
        model_cell = model if model != previous_model else ""

        lines.append(
            f"{model_cell} & "
            f"{latex_escape(row['subgroup'])} & "
            f"{int(row['n'])} & "
            f"{latex_escape(row['accuracy'])} & "
            f"{latex_escape(row['precision'])} & "
            f"{latex_escape(row['recall'])} & "
            f"{latex_escape(row['specificity'])} & "
            f"{latex_escape(row['f1'])} & "
            f"{latex_escape(row['auc_roc'])} & "
            f"{latex_escape(row['auc_pr'])} \\\\"
        )

        previous_model = model

        if str(row["subgroup"]) == "Dark phototype":
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: Values are reported as mean $\pm$ standard deviation across five "
        r"independently trained model replicas. Subgroup performance was estimated "
        r"on the BOSQUE external target ORP using the available phototype grouping. "
        r"The subgroup sample size $n$ refers to the number of BOSQUE images in "
        r"each target subgroup."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    write_text(output_path, "\n".join(lines))




def main():
    make_table_01()
    make_table_02()
    print("Table 3 is generated by scripts/20_make_seed_aware_gap_publication_outputs.py")


if __name__ == "__main__":
    main()
