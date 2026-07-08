#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

INPUT = TABLES / "interval_tac_etc_consistency.csv"

METRIC_ORDER = ["recall", "auc_pr", "f1", "precision"]
TARGET_ORDER = ["BOSQUE overall", "BOSQUE light", "BOSQUE dark"]

REGION_ORDER = [
    "adequate and transported",
    "inconclusive",
    "not adequate and not transported",
]

REGION_LABELS = {
    "adequate and transported": "Adequate + transported",
    "inconclusive": "Inconclusive",
    "not adequate and not transported": "Not adequate + not transported",
}

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
}


def main():
    df = pd.read_csv(INPUT)

    full_index = pd.MultiIndex.from_product(
        [METRIC_ORDER, TARGET_ORDER, REGION_ORDER],
        names=["metric", "target_condition", "joint_region"],
    )

    df_full = (
        df.set_index(["metric", "target_condition", "joint_region"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    plot_df = (
        df_full.pivot_table(
            index=["metric", "target_condition"],
            columns="joint_region",
            values="n_model_seed_decisions",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for region in REGION_ORDER:
        if region not in plot_df.columns:
            plot_df[region] = 0

    plot_df["metric"] = pd.Categorical(
        plot_df["metric"], categories=METRIC_ORDER, ordered=True
    )
    plot_df["target_condition"] = pd.Categorical(
        plot_df["target_condition"], categories=TARGET_ORDER, ordered=True
    )

    plot_df = plot_df.sort_values(["metric", "target_condition"]).reset_index(drop=True)

    labels = [
        f"{METRIC_LABELS.get(row.metric, row.metric)}\n{row.target_condition.replace('BOSQUE ', '')}"
        for row in plot_df.itertuples()
    ]

    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for region in REGION_ORDER:
        values = plot_df[region].to_numpy()
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=REGION_LABELS[region],
            edgecolor="black",
            linewidth=0.4,
        )

        for i, value in enumerate(values):
            if value > 0:
                ax.text(
                    x[i],
                    bottom[i] + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

        bottom += values

    ax.axhline(25, linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 27)
    ax.set_ylabel("Number of model–seed decisions")
    ax.set_title("Interval-based TAC/ETC decision consistency across model–seed replicas")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

    ax.text(
        0.99,
        0.97,
        "Each bar summarizes 25 decisions\n5 architectures × 5 seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()

    out_png = FIGURES / "figure_interval_tac_etc_consistency.png"
    out_pdf = FIGURES / "figure_interval_tac_etc_consistency.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
