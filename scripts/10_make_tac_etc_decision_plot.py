#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc_roc",
    "auc_pr",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1-score",
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
}

TARGET_ORDER = ["BOSQUE dark", "BOSQUE light", "BOSQUE overall"]

TARGET_MARKERS = {
    "BOSQUE dark": "s",
    "BOSQUE light": "^",
    "BOSQUE overall": "o",
}

TARGET_COLORS = {
    "BOSQUE dark": "#1f77b4",
    "BOSQUE light": "#ff7f0e",
    "BOSQUE overall": "#2ca02c",
}


def normalize_model(x):
    s = str(x).strip()

    mapping = {
        "resnet50": "ResNet50",
        "resnet_50": "ResNet50",
        "densenet121": "DenseNet121",
        "dense_net121": "DenseNet121",
        "mobilenetv2": "MobileNetV2",
        "mobile_net_v2": "MobileNetV2",
        "efficientnetv2b0": "EfficientNetV2B0",
        "efficient_net_v2_b0": "EfficientNetV2B0",
        "vgg16": "VGG16",
    }

    key = s.lower().replace("-", "").replace("_", "")
    return mapping.get(key, s)


def normalize_dataset(x):
    s = str(x).strip().lower()
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")

    if "ham" in s:
        return "HAM10000 internal"

    if "bosque" in s:
        return "BOSQUE overall"

    return str(x).strip()


def normalize_subgroup(x):
    s = str(x).strip().lower()
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")

    if "dark" in s or "darker" in s:
        return "BOSQUE dark"

    if "light" in s or "lighter" in s:
        return "BOSQUE light"

    if "overall" in s or "all" in s or "total" in s:
        return "BOSQUE overall"

    return None


def standardize_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "model_name": "model",
        "architecture": "model",
        "dataset_name": "dataset",
        "data": "dataset",
        "metric_name": "metric",
        "mean_metric": "mean",
        "metric_mean": "mean",
        "mean_value": "mean",
        "value_mean": "mean",
        "group": "subgroup",
        "skin_tone_group": "subgroup",
        "phototype_group": "subgroup",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def wide_or_long_to_long(df):
    df = standardize_columns(df)

    if "model" not in df.columns:
        raise ValueError(f"No model column found. Available columns: {list(df.columns)}")

    df["model"] = df["model"].apply(normalize_model)

    # Already long
    if {"model", "metric", "mean"}.issubset(df.columns):
        keep = ["model", "metric", "mean"]
        if "dataset" in df.columns:
            keep.append("dataset")
        if "subgroup" in df.columns:
            keep.append("subgroup")

        out = df[keep].copy()
        out["metric"] = out["metric"].astype(str).str.strip().str.lower()
        out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
        return out.dropna(subset=["mean"]).reset_index(drop=True)

    # Wide
    available_metrics = [m for m in METRICS if m in df.columns]

    if not available_metrics:
        raise ValueError(
            "No metric columns found.\n"
            f"Expected one or more of: {METRICS}\n"
            f"Available columns: {list(df.columns)}"
        )

    id_vars = ["model"]
    if "dataset" in df.columns:
        id_vars.append("dataset")
    if "subgroup" in df.columns:
        id_vars.append("subgroup")

    out = df.melt(
        id_vars=id_vars,
        value_vars=available_metrics,
        var_name="metric",
        value_name="mean",
    )

    out["metric"] = out["metric"].astype(str).str.strip().str.lower()
    out["mean"] = pd.to_numeric(out["mean"], errors="coerce")

    return out.dropna(subset=["mean"]).reset_index(drop=True)


def load_overall_table():
    path = TABLES / "all_models_all_seed_metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df = wide_or_long_to_long(df)

    if "dataset" not in df.columns:
        raise ValueError(f"Overall summary table has no dataset column: {path}")

    df["dataset_norm"] = df["dataset"].apply(normalize_dataset)

    return df


def load_subgroup_table():
    path = TABLES / "all_models_all_seed_subgroup_metrics_summary.csv"

    if not path.exists():
        warnings.warn(f"Subgroup summary file not found: {path}")
        return pd.DataFrame(columns=["model", "metric", "mean", "target_condition"])

    df = pd.read_csv(path)
    df = wide_or_long_to_long(df)

    # Try to identify subgroup condition
    if "subgroup" in df.columns:
        df["target_condition"] = df["subgroup"].apply(normalize_subgroup)
    elif "dataset" in df.columns:
        df["target_condition"] = df["dataset"].apply(normalize_subgroup)
    else:
        df["target_condition"] = None

    # Keep only dark/light target rows
    df = df[df["target_condition"].isin(["BOSQUE dark", "BOSQUE light"])].copy()

    return df


def build_decision_table(metric, tau, epsilon):
    metric = metric.lower()

    overall = load_overall_table()

    # Source = HAM10000 internal
    source = (
        overall[
            (overall["dataset_norm"] == "HAM10000 internal")
            & (overall["metric"] == metric)
        ][["model", "mean"]]
        .rename(columns={"mean": "source_performance"})
        .copy()
    )

    # Overall BOSQUE target
    bosque_overall = (
        overall[
            (overall["dataset_norm"] == "BOSQUE overall")
            & (overall["metric"] == metric)
        ][["model", "mean"]]
        .rename(columns={"mean": "target_performance"})
        .copy()
    )
    bosque_overall["target_condition"] = "BOSQUE overall"

    # Subgroup BOSQUE targets
    subgroups = load_subgroup_table()
    bosque_subgroups = (
        subgroups[
            (subgroups["target_condition"].isin(["BOSQUE dark", "BOSQUE light"]))
            & (subgroups["metric"] == metric)
        ][["model", "target_condition", "mean"]]
        .rename(columns={"mean": "target_performance"})
        .copy()
    )

    target = pd.concat([bosque_subgroups, bosque_overall], ignore_index=True)

    if source.empty:
        raise ValueError(f"No HAM10000 internal source rows found for metric={metric}")

    if target.empty:
        raise ValueError(f"No BOSQUE target rows found for metric={metric}")

    decision = target.merge(source, on="model", how="left")

    if decision["source_performance"].isna().any():
        print("\nWARNING: Some target models have no HAM10000 source match:")
        print(decision[decision["source_performance"].isna()][["model", "target_condition"]])
        decision = decision.dropna(subset=["source_performance"]).copy()

    decision["degradation"] = (
        decision["source_performance"] - decision["target_performance"]
    )

    decision["tac_pass"] = decision["target_performance"] >= tau
    decision["etc_pass"] = decision["degradation"] <= epsilon

    def classify(row):
        if row["tac_pass"] and row["etc_pass"]:
            return "maintained"
        if row["tac_pass"] and not row["etc_pass"]:
            return "adequate but degraded"
        if (not row["tac_pass"]) and row["etc_pass"]:
            return "transported but inadequate"
        return "restricted / rejected"

    decision["decision_region"] = decision.apply(classify, axis=1)
    decision["metric"] = metric
    decision["tau"] = tau
    decision["epsilon"] = epsilon

    decision["target_condition"] = pd.Categorical(
        decision["target_condition"],
        categories=TARGET_ORDER,
        ordered=True,
    )

    decision = decision.sort_values(["target_condition", "model"]).reset_index(drop=True)

    if decision.empty:
        raise ValueError("Decision table is empty after merging source and target rows.")

    return decision


def choose_axis_limits(decision, tau, epsilon):
    x = decision["degradation"].dropna().to_numpy()
    y = decision["target_performance"].dropna().to_numpy()

    xmin = min(np.min(x), 0, epsilon) - 0.035
    xmax = max(np.max(x), 0, epsilon) + 0.035

    ymin = max(0.0, min(np.min(y), tau) - 0.04)
    ymax = min(1.02, max(np.max(y), tau) + 0.04)

    return xmin, xmax, ymin, ymax


def add_quadrant_labels(ax, tau, epsilon):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_left = xmin + 0.03 * (xmax - xmin)
    x_right = epsilon + 0.03 * (xmax - xmin)
    y_top = ymax - 0.08 * (ymax - ymin)
    y_bottom = ymin + 0.07 * (ymax - ymin)

    kw = dict(
        fontsize=9.5,
        color="black",
        alpha=0.65,
        ha="left",
        va="top",
        linespacing=1.05,
    )

    ax.text(x_left, y_top, "maintained\nTAC pass + ETC pass", **kw)
    ax.text(x_right, y_top, "adequate but degraded\nTAC pass + ETC fail", **kw)
    ax.text(x_left, y_bottom, "transported but inadequate\nTAC fail + ETC pass", **kw)
    ax.text(x_right, y_bottom, "restricted / rejected\nTAC fail + ETC fail", **kw)


def add_threshold_labels(ax, tau, epsilon):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.text(
        xmin + 0.005 * (xmax - xmin),
        tau + 0.012 * (ymax - ymin),
        rf"TAC threshold $\tau = {tau:.2f}$",
        fontsize=9.5,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=2),
    )

    ax.text(
        epsilon + 0.006 * (xmax - xmin),
        ymax - 0.03 * (ymax - ymin),
        rf"ETC margin $\epsilon = {epsilon:.2f}$",
        fontsize=9.5,
        rotation=90,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=2),
    )


def plot_decision(decision, metric, tau, epsilon, show_labels=True):
    metric_label = METRIC_LABELS.get(metric, metric)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    xmin, xmax, ymin, ymax = choose_axis_limits(decision, tau, epsilon)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.axhline(tau, linestyle="--", linewidth=1.5, color="#1f77b4", zorder=0)
    ax.axvline(epsilon, linestyle="--", linewidth=1.5, color="#1f77b4", zorder=0)
    ax.axvline(0, linestyle="-", linewidth=0.7, color="0.80", zorder=0)

    texts = []

    for target in TARGET_ORDER:
        sub = decision[decision["target_condition"] == target].copy()

        if sub.empty:
            continue

        ax.scatter(
            sub["degradation"],
            sub["target_performance"],
            label=target,
            marker=TARGET_MARKERS[target],
            color=TARGET_COLORS[target],
            s=90,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        if show_labels:
            for _, row in sub.iterrows():
                label = str(row["model"])
                texts.append(
                    ax.text(
                        row["degradation"],
                        row["target_performance"],
                        label,
                        fontsize=8.5,
                        ha="left",
                        va="center",
                        zorder=4,
                    )
                )

    if show_labels and texts and HAS_ADJUST_TEXT:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, alpha=0.6),
        )
    elif show_labels and texts and not HAS_ADJUST_TEXT:
        warnings.warn("adjustText is not installed. Run: python3 -m pip install adjustText")

    add_quadrant_labels(ax, tau, epsilon)
    add_threshold_labels(ax, tau, epsilon)

    ax.set_title(f"TAC–ETC decision plot for {metric_label}", fontsize=15, pad=10)

    ax.set_xlabel(
        f"Source-to-target difference: HAM10000 internal − BOSQUE {metric_label}\n"
        "Negative values indicate higher BOSQUE performance than internal validation",
        fontsize=10.5,
    )

    ax.set_ylabel(f"BOSQUE target {metric_label}", fontsize=10.5)

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)

    leg = ax.legend(
        title="Target condition",
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.5,
        title_fontsize=9.5,
    )
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, choices=METRICS)
    parser.add_argument("--tau", required=True, type=float)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    decision = build_decision_table(args.metric, args.tau, args.epsilon)

    table_path = TABLES / f"tac_etc_decision_table_{args.metric}.csv"
    decision.to_csv(table_path, index=False)

    fig, ax = plot_decision(
        decision=decision,
        metric=args.metric,
        tau=args.tau,
        epsilon=args.epsilon,
        show_labels=not args.no_labels,
    )

    png_path = FIGURES / f"figure_tac_etc_decision_plot_{args.metric}.png"
    pdf_path = FIGURES / f"figure_tac_etc_decision_plot_{args.metric}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved decision table: {table_path}")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print("\nDecision table preview:")
    print(decision[["model", "target_condition", "source_performance", "target_performance", "degradation", "decision_region"]].to_string(index=False))


if __name__ == "__main__":
    main()
