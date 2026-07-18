#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate descriptive performance figures.

The BOSQUE light--dark heatmap uses architecture-level mean gaps across five
locked seeds. It contains no p-values, FDR stars, or significance filtering.

Seed-specific subgroup-gap forest plots are generated separately by:
  scripts/20_make_seed_aware_gap_publication_outputs.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
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

METRICS = [
    "recall",
    "auc_pr",
    "f1",
    "precision",
    "accuracy",
    "specificity",
    "auc_roc",
]

METRIC_LABELS = {
    "recall": "Recall",
    "auc_pr": "AUC-PR",
    "f1": "F1",
    "precision": "Precision",
    "accuracy": "Accuracy",
    "specificity": "Specificity",
    "auc_roc": "AUC-ROC",
}


def savefig(path):
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    print("Saved:", path)
    print("Saved:", path.with_suffix(".pdf"))

    plt.close()


def load_all_seed_metrics():
    rows = []

    for model in MODELS:
        path = (
            TABLES_DIR
            / f"{model}_all_seed_metrics.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing input: {path}"
            )

        frame = pd.read_csv(path)
        frame["model"] = model
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def load_seed_aware_gap_summary():
    path = (
        TABLES_DIR
        / "bosque_light_dark_gap_summary_by_architecture.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "model",
        "metric",
        "mean_gap",
        "n_seeds",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Gap summary missing columns: {sorted(missing)}"
        )

    if not (frame["n_seeds"] == 5).all():
        raise ValueError(
            "At least one architecture--metric row "
            "does not contain five seeds."
        )

    return frame


def plot_internal_external_metric(metric):
    frame = load_all_seed_metrics()

    summary = (
        frame.groupby(
            ["model", "dataset"],
            observed=True,
        )[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    model_labels = [
        MODEL_LABELS[model]
        for model in MODELS
    ]

    x = np.arange(len(MODELS))
    width = 0.36

    ham = (
        summary[
            summary["dataset"]
            == "HAM10000_internal_test"
        ]
        .set_index("model")
        .reindex(MODELS)
    )

    bosque = (
        summary[
            summary["dataset"]
            == "BOSQUE_public"
        ]
        .set_index("model")
        .reindex(MODELS)
    )

    if ham[["mean", "std"]].isna().any().any():
        raise RuntimeError(
            f"Incomplete HAM10000 summary for {metric}."
        )

    if bosque[["mean", "std"]].isna().any().any():
        raise RuntimeError(
            f"Incomplete BOSQUE summary for {metric}."
        )

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        ham["mean"],
        width,
        yerr=ham["std"],
        capsize=4,
        label="HAM10000 held-out source ORP",
    )

    plt.bar(
        x + width / 2,
        bosque["mean"],
        width,
        yerr=bosque["std"],
        capsize=4,
        label="BOSQUE target ORP",
    )

    plt.xticks(
        x,
        model_labels,
        rotation=20,
        ha="right",
    )

    plt.ylabel(METRIC_LABELS[metric])
    plt.ylim(0, 1)

    plt.title(
        "Source and target performance: "
        f"{METRIC_LABELS[metric]}"
    )

    plt.legend(frameon=False)

    savefig(
        FIGURES_DIR
        / f"figure_internal_external_{metric}.png"
    )


def plot_gap_heatmap():
    frame = load_seed_aware_gap_summary()

    pivot = (
        frame.pivot(
            index="model",
            columns="metric",
            values="mean_gap",
        )
        .reindex(
            index=MODELS,
            columns=METRICS,
        )
    )

    if pivot.isna().any().any():
        raise RuntimeError(
            "The seed-aware gap heatmap contains "
            "missing architecture--metric cells."
        )

    values = pivot.to_numpy(dtype=float)

    limit = max(
        0.05,
        float(np.nanmax(np.abs(values))),
    )

    plt.figure(figsize=(11, 5.5))

    image = plt.imshow(
        values,
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )

    plt.colorbar(
        image,
        label="Mean light − dark performance gap",
    )

    plt.xticks(
        np.arange(len(METRICS)),
        [
            METRIC_LABELS[metric]
            for metric in METRICS
        ],
        rotation=35,
        ha="right",
    )

    plt.yticks(
        np.arange(len(MODELS)),
        [
            MODEL_LABELS[model]
            for model in MODELS
        ],
    )

    plt.title(
        "BOSQUE light–dark gaps averaged across five seeds"
    )

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            plt.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    savefig(
        FIGURES_DIR
        / "figure_bosque_light_dark_mean_gap_heatmap.png"
    )


def main():
    plot_internal_external_metric("f1")
    plot_internal_external_metric("auc_pr")
    plot_gap_heatmap()


if __name__ == "__main__":
    main()
