#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["resnet50", "densenet121", "mobilenetv2", "efficientnetv2b0", "vgg16"]

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenetv2": "MobileNetV2",
    "efficientnetv2b0": "EfficientNetV2B0",
    "vgg16": "VGG16",
}

METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
}


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print("Wrote:", path)
    print("Wrote:", path.with_suffix(".pdf"))
    plt.close()


def load_all_seed_metrics() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        path = TABLES_DIR / f"{model}_all_seed_metrics.csv"
        df = pd.read_csv(path)
        df["model"] = model
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_subgroup_summary() -> pd.DataFrame:
    path = TABLES_DIR / "bosque_light_dark_bootstrap_comparison_readable.csv"
    return pd.read_csv(path)


def plot_internal_external_metric(metric: str) -> None:
    df = load_all_seed_metrics()

    summary = (
        df.groupby(["model", "dataset"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    model_labels = [MODEL_LABELS[m] for m in MODELS]
    x = np.arange(len(MODELS))
    width = 0.36

    ham = summary[summary["dataset"] == "HAM10000_internal_test"].set_index("model").loc[MODELS]
    bosque = summary[summary["dataset"] == "BOSQUE_public"].set_index("model").loc[MODELS]

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, ham["mean"], width, yerr=ham["std"], capsize=4, label="HAM10000 internal test")
    plt.bar(x + width / 2, bosque["mean"], width, yerr=bosque["std"], capsize=4, label="BOSQUE external test")

    plt.xticks(x, model_labels, rotation=20, ha="right")
    plt.ylabel(METRIC_LABELS[metric])
    plt.ylim(0, 1)
    plt.title(f"Internal vs external performance: {METRIC_LABELS[metric]}")
    plt.legend(frameon=False)

    savefig(FIGURES_DIR / f"figure_internal_external_{metric}.png")


def plot_gap_heatmap() -> None:
    df = load_subgroup_summary()

    pivot = df.pivot(index="model", columns="metric", values="gap_light_minus_dark").loc[MODELS, METRICS]
    sig = df.pivot(index="model", columns="metric", values="significant_fdr_0_05").loc[MODELS, METRICS]

    values = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(11, 5.5))
    im = plt.imshow(values, aspect="auto", vmin=-0.30, vmax=0.30)

    plt.colorbar(im, label="Light - dark performance gap")
    plt.xticks(np.arange(len(METRICS)), [METRIC_LABELS[m] for m in METRICS], rotation=35, ha="right")
    plt.yticks(np.arange(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    plt.title("BOSQUE light–dark subgroup gaps")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            star = "*" if bool(sig.iloc[i, j]) else ""
            plt.text(j, i, f"{values[i, j]:.2f}{star}", ha="center", va="center", fontsize=9)

    savefig(FIGURES_DIR / "figure_bosque_light_dark_gap_heatmap.png")


def plot_forest_significant_gaps() -> None:
    df = load_subgroup_summary()
    df = df[df["significant_fdr_0_05"]].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["metric_label"] = df["metric"].map(METRIC_LABELS)
    df["label"] = df["model_label"] + " — " + df["metric_label"]

    df = df.sort_values(["model", "metric"]).reset_index(drop=True)

    y = np.arange(len(df))
    x = df["gap_light_minus_dark"].to_numpy()
    low = df["bootstrap_ci_low"].to_numpy()
    high = df["bootstrap_ci_high"].to_numpy()

    xerr = np.vstack([x - low, high - x])

    height = max(6, 0.35 * len(df))
    plt.figure(figsize=(10, height))
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    plt.axvline(0, linestyle="--", linewidth=1)

    plt.yticks(y, df["label"])
    plt.xlabel("Light - dark performance gap")
    plt.title("FDR-significant BOSQUE subgroup gaps with bootstrap 95% CIs")
    plt.gca().invert_yaxis()

    savefig(FIGURES_DIR / "figure_bosque_significant_gap_forest.png")


def main() -> None:
    plot_internal_external_metric("f1")
    plot_internal_external_metric("auc_pr")
    plot_gap_heatmap()
    plot_forest_significant_gaps()


if __name__ == "__main__":
    main()
