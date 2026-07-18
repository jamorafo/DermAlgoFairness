#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

MODEL_ORDER = [
    "ResNet50",
    "DenseNet121",
    "MobileNetV2",
    "EfficientNetV2B0",
    "VGG16",
]

TARGET_ORDER = [
    "BOSQUE dark",
    "BOSQUE light",
    "BOSQUE overall",
]


def normalize_model(x):
    s = str(x).strip()
    key = s.lower().replace("-", "").replace("_", "").replace(" ", "")

    mapping = {
        "resnet50": "ResNet50",
        "densenet121": "DenseNet121",
        "mobilenetv2": "MobileNetV2",
        "efficientnetv2b0": "EfficientNetV2B0",
        "vgg16": "VGG16",
    }

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
        "metric": "metric",
        "mean_metric": "value",
        "metric_mean": "value",
        "mean_value": "value",
        "value_mean": "value",
        "score": "value",
        "performance": "value",
        "group": "subgroup",
        "skin_tone_group": "subgroup",
        "phototype_group": "subgroup",
        "random_seed": "seed",
        "run_seed": "seed",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def wide_or_long_to_long(df):
    df = standardize_columns(df)

    if "model" not in df.columns:
        raise ValueError(f"No model column found. Available columns: {list(df.columns)}")

    df["model"] = df["model"].apply(normalize_model)

    if "seed" not in df.columns:
        possible_seed_cols = [c for c in df.columns if "seed" in c.lower()]
        if possible_seed_cols:
            df = df.rename(columns={possible_seed_cols[0]: "seed"})
        else:
            df["seed"] = 1

    # Already long
    if {"model", "metric"}.issubset(df.columns) and ("value" in df.columns or "mean" in df.columns):
        if "value" not in df.columns and "mean" in df.columns:
            df = df.rename(columns={"mean": "value"})

        keep = ["model", "seed", "metric", "value"]

        if "dataset" in df.columns:
            keep.append("dataset")
        if "subgroup" in df.columns:
            keep.append("subgroup")

        out = df[keep].copy()
        out["metric"] = out["metric"].astype(str).str.strip().str.lower()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

        return out.dropna(subset=["value"]).reset_index(drop=True)

    # Wide
    available_metrics = [m for m in METRICS if m in df.columns]

    if not available_metrics:
        raise ValueError(
            "No metric columns found.\n"
            f"Expected one or more of: {METRICS}\n"
            f"Available columns: {list(df.columns)}"
        )

    id_vars = ["model", "seed"]

    if "dataset" in df.columns:
        id_vars.append("dataset")
    if "subgroup" in df.columns:
        id_vars.append("subgroup")

    out = df.melt(
        id_vars=id_vars,
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    )

    out["metric"] = out["metric"].astype(str).str.strip().str.lower()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    return out.dropna(subset=["value"]).reset_index(drop=True)


def load_overall_seed_table():
    path = TABLES / "all_models_all_seed_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df = wide_or_long_to_long(df)

    if "dataset" not in df.columns:
        raise ValueError(f"Overall seed table has no dataset column: {path}")

    df["target_condition"] = df["dataset"].apply(normalize_dataset)

    return df


def load_subgroup_seed_table():
    path = TABLES / "all_models_all_seed_subgroup_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df = wide_or_long_to_long(df)

    if "subgroup" in df.columns:
        df["target_condition"] = df["subgroup"].apply(normalize_subgroup)
    elif "dataset" in df.columns:
        df["target_condition"] = df["dataset"].apply(normalize_subgroup)
    else:
        raise ValueError(
            f"Subgroup seed table has no subgroup or dataset column: {path}"
        )

    df = df[df["target_condition"].isin(["BOSQUE dark", "BOSQUE light"])].copy()

    return df


def build_seed_level_dataset(metric):
    metric = metric.lower()

    overall = load_overall_seed_table()
    subgroups = load_subgroup_seed_table()

    overall = overall[overall["metric"] == metric].copy()
    subgroups = subgroups[subgroups["metric"] == metric].copy()

    # BOSQUE overall target
    bosque_overall = overall[
        overall["target_condition"] == "BOSQUE overall"
    ][["model", "seed", "metric", "target_condition", "value"]].copy()

    # HAM10000 source
    source = overall[
        overall["target_condition"] == "HAM10000 internal"
    ][["model", "seed", "value"]].rename(columns={"value": "source_performance"})

    # BOSQUE subgroup targets
    bosque_subgroups = subgroups[
        subgroups["target_condition"].isin(["BOSQUE dark", "BOSQUE light"])
    ][["model", "seed", "metric", "target_condition", "value"]].copy()

    target = pd.concat([bosque_subgroups, bosque_overall], ignore_index=True)
    target = target.rename(columns={"value": "target_performance"})

    merged = target.merge(source, on=["model", "seed"], how="left")

    if merged["source_performance"].isna().any():
        print("WARNING: Some target rows do not have matching HAM10000 source values.")
        print(merged[merged["source_performance"].isna()][["model", "seed", "target_condition"]])
        merged = merged.dropna(subset=["source_performance"]).copy()

    merged["degradation"] = merged["source_performance"] - merged["target_performance"]

    merged["model"] = pd.Categorical(
        merged["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    merged["target_condition"] = pd.Categorical(
        merged["target_condition"],
        categories=TARGET_ORDER,
        ordered=True,
    )

    merged = merged.sort_values(["model", "seed", "target_condition"]).reset_index(drop=True)

    return merged


def build_gap_table(seed_df):
    wide = seed_df[
        seed_df["target_condition"].isin(["BOSQUE dark", "BOSQUE light"])
    ].pivot_table(
        index=["model", "seed", "metric"],
        columns="target_condition",
        values="target_performance",
        aggfunc="mean",
    ).reset_index()

    wide.columns.name = None

    if "BOSQUE light" not in wide.columns or "BOSQUE dark" not in wide.columns:
        raise ValueError("Could not construct light-dark gap table.")

    wide["light_minus_dark_gap"] = wide["BOSQUE light"] - wide["BOSQUE dark"]

    wide["model"] = pd.Categorical(
        wide["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return wide.sort_values(["model", "seed"]).reset_index(drop=True)


def make_seed_summary(seed_df, gap_df, metric):
    perf_summary = (
        seed_df
        .groupby(["model", "target_condition"], observed=True)
        .agg(
            mean_target=("target_performance", "mean"),
            sd_target=("target_performance", "std"),
            min_target=("target_performance", "min"),
            max_target=("target_performance", "max"),
            mean_degradation=("degradation", "mean"),
            max_degradation=("degradation", "max"),
            min_degradation=("degradation", "min"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    gap_summary = (
        gap_df
        .groupby("model", observed=True)
        .agg(
            mean_light_dark_gap=("light_minus_dark_gap", "mean"),
            sd_light_dark_gap=("light_minus_dark_gap", "std"),
            min_light_dark_gap=("light_minus_dark_gap", "min"),
            max_light_dark_gap=("light_minus_dark_gap", "max"),
            n_seeds_gap=("seed", "nunique"),
        )
        .reset_index()
    )

    perf_path = TABLES / f"five_seed_performance_summary_{metric}.csv"
    gap_path = TABLES / f"five_seed_light_dark_gap_summary_{metric}.csv"

    perf_summary.to_csv(perf_path, index=False)
    gap_summary.to_csv(gap_path, index=False)

    print(f"Saved performance summary: {perf_path}")
    print(f"Saved light-dark gap summary: {gap_path}")

    print("\nPerformance summary:")
    print(perf_summary.to_string(index=False))

    print("\nLight-dark gap summary:")
    print(gap_summary.to_string(index=False))


def plot_seed_performance(seed_df, metric, tau=None):
    metric_label = METRIC_LABELS.get(metric, metric)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    x_positions = {model: i for i, model in enumerate(MODEL_ORDER)}
    offsets = {
        "BOSQUE dark": -0.22,
        "BOSQUE light": 0.00,
        "BOSQUE overall": 0.22,
    }
    markers = {
        "BOSQUE dark": "s",
        "BOSQUE light": "^",
        "BOSQUE overall": "o",
    }

    for target in TARGET_ORDER:
        sub = seed_df[seed_df["target_condition"] == target].copy()

        if sub.empty:
            continue

        xs = sub["model"].astype(str).map(x_positions).astype(float) + offsets[target]

        # small deterministic seed jitter
        seed_jitter = (sub["seed"].astype(float) - sub["seed"].astype(float).mean()) * 0.012
        xs = xs + seed_jitter

        ax.scatter(
            xs,
            sub["target_performance"],
            label=target,
            marker=markers[target],
            s=65,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.6,
        )

        # mean bar per model-target
        means = sub.groupby("model", observed=True)["target_performance"].mean()
        for model, y in means.items():
            x = x_positions[str(model)] + offsets[target]
            ax.hlines(y, x - 0.08, x + 0.08, linewidth=2.0)

    if tau is not None:
        ax.axhline(tau, linestyle="--", linewidth=1.3)
        ax.text(
            -0.45,
            tau + 0.01,
            rf"TAC threshold $\tau={tau:.2f}$",
            fontsize=9,
            va="bottom",
        )

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")

    ax.set_ylim(
        max(0, seed_df["target_performance"].min() - 0.08),
        min(1.02, seed_df["target_performance"].max() + 0.08),
    )

    ax.set_title(f"Five-seed BOSQUE target performance: {metric_label}", fontsize=14)
    ax.set_ylabel(f"BOSQUE target {metric_label}")
    ax.set_xlabel("Architecture")

    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(title="Target condition", frameon=True, fontsize=9)

    fig.tight_layout()

    png_path = FIGURES / f"figure_five_seed_target_performance_{metric}.png"
    pdf_path = FIGURES / f"figure_five_seed_target_performance_{metric}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def plot_light_dark_gap(gap_df, metric, epsilon_gap=None):
    metric_label = METRIC_LABELS.get(metric, metric)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)

    x_positions = {model: i for i, model in enumerate(MODEL_ORDER)}

    xs = gap_df["model"].astype(str).map(x_positions).astype(float)
    seed_jitter = (gap_df["seed"].astype(float) - gap_df["seed"].astype(float).mean()) * 0.025
    xs = xs + seed_jitter

    ax.scatter(
        xs,
        gap_df["light_minus_dark_gap"],
        s=70,
        alpha=0.80,
        edgecolor="white",
        linewidth=0.6,
    )

    means = gap_df.groupby("model", observed=True)["light_minus_dark_gap"].mean()
    for model, y in means.items():
        x = x_positions[str(model)]
        ax.hlines(y, x - 0.18, x + 0.18, linewidth=2.3)

    ax.axhline(0, linewidth=0.9)

    if epsilon_gap is not None:
        ax.axhline(epsilon_gap, linestyle="--", linewidth=1.3)
        ax.text(
            -0.45,
            epsilon_gap + 0.01,
            rf"Gap tolerance $\epsilon_G={epsilon_gap:.2f}$",
            fontsize=9,
            va="bottom",
        )

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")

    ymin = min(0, gap_df["light_minus_dark_gap"].min()) - 0.06
    ymax = max(0, gap_df["light_minus_dark_gap"].max()) + 0.06
    ax.set_ylim(ymin, ymax)

    ax.set_title(f"Five-seed light-minus-dark BOSQUE gap: {metric_label}", fontsize=14)
    ax.set_ylabel(f"Light minus dark {metric_label}")
    ax.set_xlabel("Architecture")

    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.45)

    fig.tight_layout()

    png_path = FIGURES / f"figure_five_seed_light_dark_gap_{metric}.png"
    pdf_path = FIGURES / f"figure_five_seed_light_dark_gap_{metric}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def plot_degradation(seed_df, metric, epsilon_transport=None):
    metric_label = METRIC_LABELS.get(metric, metric)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)

    x_positions = {model: i for i, model in enumerate(MODEL_ORDER)}
    offsets = {
        "BOSQUE dark": -0.18,
        "BOSQUE light": 0.00,
        "BOSQUE overall": 0.18,
    }
    markers = {
        "BOSQUE dark": "s",
        "BOSQUE light": "^",
        "BOSQUE overall": "o",
    }

    for target in TARGET_ORDER:
        sub = seed_df[seed_df["target_condition"] == target].copy()

        if sub.empty:
            continue

        xs = sub["model"].astype(str).map(x_positions).astype(float) + offsets[target]
        seed_jitter = (sub["seed"].astype(float) - sub["seed"].astype(float).mean()) * 0.010
        xs = xs + seed_jitter

        ax.scatter(
            xs,
            sub["degradation"],
            label=target,
            marker=markers[target],
            s=65,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.6,
        )

        means = sub.groupby("model", observed=True)["degradation"].mean()
        for model, y in means.items():
            x = x_positions[str(model)] + offsets[target]
            ax.hlines(y, x - 0.08, x + 0.08, linewidth=2.0)

    ax.axhline(0, linewidth=0.9)

    if epsilon_transport is not None:
        ax.axhline(epsilon_transport, linestyle="--", linewidth=1.3)
        ax.text(
            -0.45,
            epsilon_transport + 0.01,
            rf"ETC margin $\epsilon_T={epsilon_transport:.2f}$",
            fontsize=9,
            va="bottom",
        )

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")

    ymin = min(0, seed_df["degradation"].min()) - 0.06
    ymax = max(0, seed_df["degradation"].max()) + 0.06
    ax.set_ylim(ymin, ymax)

    ax.set_title(f"Five-seed HAM10000-to-BOSQUE degradation: {metric_label}", fontsize=14)
    ax.set_ylabel(f"HAM10000 internal − BOSQUE {metric_label}")
    ax.set_xlabel("Architecture")

    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(title="Target condition", frameon=True, fontsize=9)

    fig.tight_layout()

    png_path = FIGURES / f"figure_five_seed_degradation_{metric}.png"
    pdf_path = FIGURES / f"figure_five_seed_degradation_{metric}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, choices=METRICS)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--epsilon-transport", type=float, default=None)
    parser.add_argument("--epsilon-gap", type=float, default=None)
    args = parser.parse_args()

    metric = args.metric.lower()

    seed_df = build_seed_level_dataset(metric)
    gap_df = build_gap_table(seed_df)

    seed_path = TABLES / f"five_seed_analysis_long_{metric}.csv"
    gap_path = TABLES / f"five_seed_light_dark_gap_seedlevel_{metric}.csv"

    seed_df.to_csv(seed_path, index=False)
    gap_df.to_csv(gap_path, index=False)

    print(f"Saved seed-level table: {seed_path}")
    print(f"Saved seed-level gap table: {gap_path}")

    make_seed_summary(seed_df, gap_df, metric)

    plot_seed_performance(seed_df, metric, tau=args.tau)
    plot_light_dark_gap(gap_df, metric, epsilon_gap=args.epsilon_gap)
    plot_degradation(seed_df, metric, epsilon_transport=args.epsilon_transport)


if __name__ == "__main__":
    main()
