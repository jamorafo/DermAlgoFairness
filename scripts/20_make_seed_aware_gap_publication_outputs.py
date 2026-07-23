#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create publication outputs for the seed-aware BOSQUE light--dark gap analysis.

Inputs:
  outputs/tables/bosque_light_dark_gap_by_model_seed.csv
  outputs/tables/bosque_light_dark_gap_summary_by_architecture.csv

Outputs:
  outputs/publication_tables/table_03_seed_aware_light_dark_gaps.csv
  outputs/publication_tables/table_03_seed_aware_light_dark_gaps.tex
  outputs/figures/figure_bosque_<metric>_gap_seed_aware.png
  outputs/figures/figure_bosque_<metric>_gap_seed_aware.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TABLES = ROOT / "outputs" / "tables"
PUB = ROOT / "outputs" / "publication_tables"
FIGURES = ROOT / "outputs" / "figures"

PUB.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

BY_SEED_INPUT = (
    TABLES / "bosque_light_dark_gap_by_model_seed.csv"
)

SUMMARY_INPUT = (
    TABLES / "bosque_light_dark_gap_summary_by_architecture.csv"
)

PRIMARY_METRICS = [
    "recall",
    "auc_pr",
    "f1",
    "precision",
]

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
}

MODEL_ORDER = [
    "resnet50",
    "densenet121",
    "mobilenetv2",
    "efficientnetv2b0",
    "vgg16",
]

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenetv2": "MobileNetV2",
    "efficientnetv2b0": "EfficientNetV2B0",
    "vgg16": "VGG16",
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


def prepare_publication_table(summary):
    primary = summary[
        summary["metric"].isin(PRIMARY_METRICS)
    ].copy()

    primary["metric_order"] = primary["metric"].map(
        {
            metric: index
            for index, metric in enumerate(PRIMARY_METRICS)
        }
    )

    primary["model_order"] = primary["model"].map(
        {
            model: index
            for index, model in enumerate(MODEL_ORDER)
        }
    )

    primary = primary.sort_values(
        ["metric_order", "model_order"]
    ).reset_index(drop=True)

    output = pd.DataFrame(
        {
            "Metric": primary["metric"].map(METRIC_LABELS),
            "Architecture": primary["model"].map(MODEL_LABELS),
            "n_seeds": primary["n_seeds"].astype(int),
            "mean_light": primary["mean_light"],
            "mean_dark": primary["mean_dark"],
            "mean_gap": primary["mean_gap"],
            "sd_gap": primary["sd_gap"],
            "n_positive_gap": primary["n_gap_positive"].astype(int),
            "n_ci_light_higher": (
                primary["n_interval_light_higher"].astype(int)
            ),
            "n_ci_includes_zero": (
                primary["n_interval_includes_zero"].astype(int)
            ),
            "n_ci_dark_higher": (
                primary["n_interval_dark_higher"].astype(int)
            ),
        }
    )

    output["ci_direction_counts"] = (
        output["n_ci_light_higher"].astype(str)
        + "/"
        + output["n_ci_includes_zero"].astype(str)
        + "/"
        + output["n_ci_dark_higher"].astype(str)
    )

    return output


def build_latex_table(table):
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Seed-aware BOSQUE light--dark performance "
            r"gaps for the primary metrics.}"
        ),
        r"\label{tab:seed-aware-light-dark-gaps}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Metric} & "
            r"\textbf{Architecture} & "
            r"\textbf{Light} & "
            r"\textbf{Dark} & "
            r"\textbf{Mean gap} & "
            r"\textbf{SD gap} & "
            r"\textbf{Positive} & "
            r"\textbf{CI $+/0/-$} \\"
        ),
        r"\midrule",
    ]

    previous_metric = None

    for _, row in table.iterrows():
        metric = latex_escape(row["Metric"])
        metric_cell = (
            metric if metric != previous_metric else ""
        )

        lines.append(
            f"{metric_cell} & "
            f"{latex_escape(row['Architecture'])} & "
            f"{fmt(row['mean_light'])} & "
            f"{fmt(row['mean_dark'])} & "
            f"{fmt(row['mean_gap'])} & "
            f"{fmt(row['sd_gap'])} & "
            f"{int(row['n_positive_gap'])}/"
            f"{int(row['n_seeds'])} & "
            f"{latex_escape(row['ci_direction_counts'])} \\\\"
        )

        previous_metric = metric

        if row["Architecture"] == "VGG16":
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: The gap is light-phototype performance minus "
                r"dark-phototype performance. Light, dark, mean gap, and "
                r"SD gap summarize the five independently trained seeds "
                r"within each architecture. ``Positive'' gives the number "
                r"of seeds with a positive point-estimate gap. CI $+/0/-$ "
                r"gives the number of seed-specific 95\% percentile-bootstrap "
                r"intervals lying entirely above zero, including zero, or "
                r"lying entirely below zero, respectively. Each interval "
                r"uses 10{,}000 image-level bootstrap replicates within the "
                r"observed BOSQUE light ($n=105$) and dark ($n=46$) ORPs. "
                r"Intervals are conditional on the locked system and on "
                r"image-level independence. No $p$-values or multiplicity "
                r"adjustments are used."
            ),
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def make_metric_forest(by_seed, metric):
    data = by_seed[
        by_seed["metric"] == metric
    ].copy()

    data["model_order"] = data["model"].map(
        {
            model: index
            for index, model in enumerate(MODEL_ORDER)
        }
    )

    data = data.sort_values(
        ["model_order", "seed"]
    ).reset_index(drop=True)

    if len(data) != 25:
        raise RuntimeError(
            f"{metric}: expected 25 seed-specific rows, "
            f"found {len(data)}."
        )

    estimates = data["gap_light_minus_dark"].to_numpy()
    lower = data["bootstrap_ci_low"].to_numpy()
    upper = data["bootstrap_ci_high"].to_numpy()

    lower_error = np.maximum(estimates - lower, 0)
    upper_error = np.maximum(upper - estimates, 0)

    y = np.arange(len(data))

    labels = [
        f"{MODEL_LABELS[model]} · seed {int(seed)}"
        for model, seed in zip(data["model"], data["seed"])
    ]

    figure, axis = plt.subplots(figsize=(7.2, 9.0))

    axis.errorbar(
        estimates,
        y,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        capsize=2,
        markersize=4,
        linewidth=0.9,
    )

    axis.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()

    axis.set_xlabel(
        "Light − dark performance gap"
    )

    axis.set_title(
        (
            f"BOSQUE {METRIC_LABELS[metric]} gap "
            "by locked architecture–seed system"
        )
    )

    axis.grid(
        axis="x",
        linewidth=0.4,
        alpha=0.5,
    )

    figure.tight_layout()

    stem = (
        FIGURES
        / f"figure_bosque_{metric}_gap_seed_aware"
    )

    figure.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", stem.with_suffix(".png"))
    print("Saved:", stem.with_suffix(".pdf"))


def main():
    if not BY_SEED_INPUT.exists():
        raise FileNotFoundError(
            f"Missing input: {BY_SEED_INPUT}"
        )

    if not SUMMARY_INPUT.exists():
        raise FileNotFoundError(
            f"Missing input: {SUMMARY_INPUT}"
        )

    by_seed = pd.read_csv(BY_SEED_INPUT)
    summary = pd.read_csv(SUMMARY_INPUT)

    required_by_seed = {
        "model",
        "seed",
        "metric",
        "gap_light_minus_dark",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    }

    required_summary = {
        "model",
        "metric",
        "n_seeds",
        "mean_light",
        "mean_dark",
        "mean_gap",
        "sd_gap",
        "n_gap_positive",
        "n_interval_light_higher",
        "n_interval_includes_zero",
        "n_interval_dark_higher",
    }

    missing_seed = required_by_seed - set(by_seed.columns)
    missing_summary = required_summary - set(summary.columns)

    if missing_seed:
        raise ValueError(
            f"By-seed input missing: {sorted(missing_seed)}"
        )

    if missing_summary:
        raise ValueError(
            f"Summary input missing: {sorted(missing_summary)}"
        )

    table = prepare_publication_table(summary)

    if len(table) != 20:
        raise RuntimeError(
            f"Expected 20 primary table rows, found {len(table)}."
        )

    if not (table["n_seeds"] == 5).all():
        raise RuntimeError(
            "At least one table row does not contain five seeds."
        )

    direction_total = (
        table["n_ci_light_higher"]
        + table["n_ci_includes_zero"]
        + table["n_ci_dark_higher"]
    )

    if not (direction_total == 5).all():
        raise RuntimeError(
            "At least one CI-direction total is not five."
        )

    csv_path = (
        PUB / "table_03_seed_aware_light_dark_gaps.csv"
    )

    tex_path = (
        PUB / "table_03_seed_aware_light_dark_gaps.tex"
    )

    table.to_csv(csv_path, index=False)

    tex_path.write_text(
        build_latex_table(table),
        encoding="utf-8",
    )

    print("Saved:", csv_path)
    print("Saved:", tex_path)

    for metric in PRIMARY_METRICS:
        make_metric_forest(by_seed, metric)

    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
