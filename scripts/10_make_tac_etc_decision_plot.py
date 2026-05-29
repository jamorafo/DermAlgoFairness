#!/usr/bin/env python3
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenetv2": "MobileNetV2",
    "efficientnetv2b0": "EfficientNetV2B0",
    "vgg16": "VGG16",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1-score",
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
}


def load_targets(metric: str) -> pd.DataFrame:
    overall = pd.read_csv(TABLES_DIR / "all_models_all_seed_metrics.csv")
    subgroup = pd.read_csv(TABLES_DIR / "all_models_all_seed_subgroup_metrics.csv")

    source = (
        overall[overall["dataset"] == "HAM10000_internal_test"]
        .groupby("model")[metric]
        .mean()
        .rename("source")
        .reset_index()
    )

    bosq_overall = (
        overall[overall["dataset"] == "BOSQUE_public"]
        .groupby("model")[metric]
        .mean()
        .rename("target")
        .reset_index()
    )
    bosq_overall["target_condition"] = "BOSQUE overall"

    bosq_subgroup = (
        subgroup
        .groupby(["model", "subgroup"])[metric]
        .mean()
        .rename("target")
        .reset_index()
    )
    bosq_subgroup["target_condition"] = "BOSQUE " + bosq_subgroup["subgroup"].astype(str)
    bosq_subgroup = bosq_subgroup.drop(columns=["subgroup"])

    targets = pd.concat([bosq_overall, bosq_subgroup], ignore_index=True)
    out = targets.merge(source, on="model", how="left")

    out["degradation"] = out["source"] - out["target"]
    out["model_label"] = out["model"].map(MODEL_LABELS).fillna(out["model"])

    return out


def classify(row, tau: float, epsilon: float) -> str:
    tac_pass = row["target"] >= tau
    etc_pass = row["degradation"] <= epsilon

    if tac_pass and etc_pass:
        return "maintained"
    if tac_pass and not etc_pass:
        return "adequate_but_degraded"
    if (not tac_pass) and etc_pass:
        return "transported_but_inadequate"
    return "restricted"


def plot_tac_etc(df: pd.DataFrame, metric: str, tau: float, epsilon: float, include_labels: bool) -> None:
    df = df.copy()
    df["decision"] = df.apply(classify, axis=1, tau=tau, epsilon=epsilon)

    x = df["degradation"]
    y = df["target"]

    x_min = min(-0.05, x.min() - 0.03)
    x_max = max(epsilon + 0.05, x.max() + 0.03)
    y_min = max(0, min(tau - 0.20, y.min() - 0.05))
    y_max = min(1.02, max(tau + 0.12, y.max() + 0.05))

    fig, ax = plt.subplots(figsize=(11, 7))

    markers = {
        "BOSQUE overall": "o",
        "BOSQUE light": "^",
        "BOSQUE dark": "s",
    }

    for target_condition, g in df.groupby("target_condition"):
        ax.scatter(
            g["degradation"],
            g["target"],
            s=80,
            marker=markers.get(target_condition, "o"),
            label=target_condition,
        )

        if include_labels:
            for _, row in g.iterrows():
                ax.annotate(
                    row["model_label"],
                    (row["degradation"], row["target"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.axhline(tau, linestyle="--", linewidth=1.5)
    ax.axvline(epsilon, linestyle="--", linewidth=1.5)

    ax.text(
        x_min + 0.01, y_max - 0.03,
        "maintained\nTAC pass + ETC pass",
        ha="left", va="top", fontsize=10,
    )
    ax.text(
        epsilon + 0.01, y_max - 0.03,
        "adequate but degraded\nTAC pass + ETC fail",
        ha="left", va="top", fontsize=10,
    )
    ax.text(
        x_min + 0.01, y_min + 0.02,
        "transported but inadequate\nTAC fail + ETC pass",
        ha="left", va="bottom", fontsize=10,
    )
    ax.text(
        epsilon + 0.01, y_min + 0.02,
        "restricted / rejected\nTAC fail + ETC fail",
        ha="left", va="bottom", fontsize=10,
    )

    ax.text(x_min + 0.002, tau + 0.005, rf"TAC threshold $\tau={tau:.2f}$", fontsize=10)
    ax.text(epsilon + 0.003, y_min + 0.005, rf"ETC margin $\epsilon={epsilon:.2f}$", rotation=90, fontsize=10)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(f"Source-to-BOSQUE degradation: HAM10000 internal − target {METRIC_LABELS[metric]}")
    ax.set_ylabel(f"BOSQUE target {METRIC_LABELS[metric]}")
    ax.set_title(f"TAC–ETC decision plot for {METRIC_LABELS[metric]}")

    ax.legend(frameon=False, loc="lower right")

    out_png = FIGURES_DIR / f"figure_tac_etc_decision_plot_{metric}.png"
    out_pdf = FIGURES_DIR / f"figure_tac_etc_decision_plot_{metric}.pdf"

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="auc_pr", choices=list(METRIC_LABELS.keys()))
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    df = load_targets(args.metric)
    df = plot_tac_etc(
        df,
        metric=args.metric,
        tau=args.tau,
        epsilon=args.epsilon,
        include_labels=not args.no_labels,
    )

    out_csv = TABLES_DIR / f"tac_etc_decision_table_{args.metric}.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    print()
    print(
        df[
            [
                "model",
                "target_condition",
                "source",
                "target",
                "degradation",
                "decision",
            ]
        ]
        .sort_values(["target_condition", "model"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
