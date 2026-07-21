#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seed-aware BOSQUE light--dark performance-gap analysis.

Each locked architecture--seed system is analyzed separately. Image-level
bootstrap resampling is performed independently within the light and dark
BOSQUE subgroups. The resulting intervals are conditional on the locked
system and on image-level independence within the observed BOSQUE ORP.

No bootstrap p-values or FDR corrections are produced.

Outputs:
  outputs/logs/bosque_gap_prediction_manifest.csv
  outputs/tables/bosque_light_dark_gap_by_model_seed.csv
  outputs/tables/bosque_light_dark_gap_summary_by_architecture.csv
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

from dermalgo.seeds import get_analysis_seed, get_training_seeds

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
TABLES = ROOT / "outputs" / "tables"
LOGS = ROOT / "outputs" / "logs"

TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

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

PRIMARY_METRICS = {
    "recall",
    "auc_pr",
    "f1",
    "precision",
}

EXPECTED_SEEDS = set(get_training_seeds())

FILE_RE = re.compile(
    r"^bosque_public_predictions_"
    r"(resnet50|densenet121|mobilenetv2|efficientnetv2b0|vgg16)"
    r"_seed(\d+)_(\d{8}T\d{6}Z)\.csv$"
)


def compute_metrics(y_true, y_score, threshold=0.5):
    """Compute all metrics for one subgroup."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    has_two_classes = len(np.unique(y_true)) == 2

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "auc_roc": (
            roc_auc_score(y_true, y_score)
            if has_two_classes
            else np.nan
        ),
        "auc_pr": (
            average_precision_score(y_true, y_score)
            if has_two_classes
            else np.nan
        ),
    }


def discover_canonical_files():
    """
    Select the latest timestamped BOSQUE file for every model--seed pair.
    """

    grouped = {}

    for path in sorted(PRED_DIR.glob(
        "bosque_public_predictions_*_seed*.csv"
    )):
        match = FILE_RE.match(path.name)

        if match is None:
            continue

        model = match.group(1)
        seed = int(match.group(2))
        timestamp = match.group(3)

        grouped.setdefault(
            (model, seed),
            [],
        ).append(
            {
                "timestamp": timestamp,
                "path": path,
            }
        )

    selected = {}
    manifest_rows = []

    for model in MODELS:
        model_seeds = {
            seed
            for candidate_model, seed in grouped
            if candidate_model == model
        }

        if model_seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{model}: expected seeds {sorted(EXPECTED_SEEDS)}, "
                f"found {sorted(model_seeds)}."
            )

        for seed in sorted(EXPECTED_SEEDS):
            candidates = grouped[(model, seed)]

            latest = max(
                candidates,
                key=lambda item: item["timestamp"],
            )

            path = latest["path"]
            selected[(model, seed)] = path

            manifest_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "seed": seed,
                    "timestamp": latest["timestamp"],
                    "n_candidate_files": len(candidates),
                    "prediction_file": str(
                        path.relative_to(ROOT)
                    ),
                }
            )

    if len(selected) != 25:
        raise RuntimeError(
            f"Expected 25 canonical BOSQUE files, found {len(selected)}."
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = (
        LOGS / "bosque_gap_prediction_manifest.csv"
    )
    manifest.to_csv(manifest_path, index=False)

    print("Canonical BOSQUE files:", len(selected))
    print("Saved manifest:", manifest_path)

    return selected


def read_and_validate(path):
    """Read one canonical BOSQUE prediction file."""

    df = pd.read_csv(path)

    required = {
        "y_true",
        "y_score",
        "skin_group",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    if len(df) != 151:
        raise ValueError(
            f"{path} contains {len(df)} rows; expected 151."
        )

    df = df.copy()

    df["skin_group"] = (
        df["skin_group"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if df[["y_true", "y_score", "skin_group"]].isna().any().any():
        raise ValueError(
            f"{path} contains missing required values."
        )

    if not set(df["y_true"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(
            f"{path} contains non-binary y_true values."
        )

    counts = df["skin_group"].value_counts().to_dict()

    if counts.get("light", 0) != 105:
        raise ValueError(
            f"{path}: expected 105 light images, "
            f"found {counts.get('light', 0)}."
        )

    if counts.get("dark", 0) != 46:
        raise ValueError(
            f"{path}: expected 46 dark images, "
            f"found {counts.get('dark', 0)}."
        )

    return df


def bootstrap_locked_system(
    df,
    model,
    seed,
    prediction_file,
    n_boot,
    threshold,
    random_seed,
):
    """
    Bootstrap light-minus-dark gaps for one locked model--seed system.
    """

    dark = df[df["skin_group"] == "dark"]
    light = df[df["skin_group"] == "light"]

    y_dark = dark["y_true"].to_numpy(dtype=int)
    s_dark = dark["y_score"].to_numpy(dtype=float)

    y_light = light["y_true"].to_numpy(dtype=int)
    s_light = light["y_score"].to_numpy(dtype=float)

    observed_dark = compute_metrics(
        y_dark,
        s_dark,
        threshold,
    )

    observed_light = compute_metrics(
        y_light,
        s_light,
        threshold,
    )

    rng = np.random.default_rng(random_seed)

    bootstrap_gaps = {
        metric: []
        for metric in METRICS
    }

    for _ in range(n_boot):
        dark_index = rng.integers(
            0,
            len(y_dark),
            size=len(y_dark),
        )

        light_index = rng.integers(
            0,
            len(y_light),
            size=len(y_light),
        )

        dark_metrics = compute_metrics(
            y_dark[dark_index],
            s_dark[dark_index],
            threshold,
        )

        light_metrics = compute_metrics(
            y_light[light_index],
            s_light[light_index],
            threshold,
        )

        for metric in METRICS:
            gap = (
                light_metrics[metric]
                - dark_metrics[metric]
            )

            if np.isfinite(gap):
                bootstrap_gaps[metric].append(gap)

    rows = []

    for metric in METRICS:
        gaps = np.asarray(
            bootstrap_gaps[metric],
            dtype=float,
        )

        if len(gaps) < max(100, int(0.9 * n_boot)):
            raise RuntimeError(
                f"{model} seed {seed}, {metric}: "
                f"only {len(gaps)} valid bootstrap replicates."
            )

        ci_low, ci_high = np.percentile(
            gaps,
            [2.5, 97.5],
        )

        observed_gap = (
            observed_light[metric]
            - observed_dark[metric]
        )

        if ci_low > 0:
            interval_direction = (
                "light higher"
            )
        elif ci_high < 0:
            interval_direction = (
                "dark higher"
            )
        else:
            interval_direction = (
                "includes zero"
            )

        rows.append(
            {
                "metric_group": (
                    "Primary"
                    if metric in PRIMARY_METRICS
                    else "Secondary"
                ),
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seed": seed,
                "metric": metric,
                "n_light": len(light),
                "n_dark": len(dark),
                "light_performance": observed_light[metric],
                "dark_performance": observed_dark[metric],
                "gap_light_minus_dark": observed_gap,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "interval_direction": interval_direction,
                "n_boot_valid": len(gaps),
                "prediction_file": str(
                    prediction_file.relative_to(ROOT)
                ),
            }
        )

    return rows


def summarize_by_architecture(by_seed):
    """Summarize the five locked seeds for each architecture and metric."""

    summary = (
        by_seed.groupby(
            [
                "metric_group",
                "model",
                "model_label",
                "metric",
            ],
            observed=True,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_light=("light_performance", "mean"),
            sd_light=("light_performance", "std"),
            mean_dark=("dark_performance", "mean"),
            sd_dark=("dark_performance", "std"),
            mean_gap=("gap_light_minus_dark", "mean"),
            sd_gap=("gap_light_minus_dark", "std"),
            min_gap=("gap_light_minus_dark", "min"),
            max_gap=("gap_light_minus_dark", "max"),
            n_gap_positive=(
                "gap_light_minus_dark",
                lambda x: int((x > 0).sum()),
            ),
            n_interval_light_higher=(
                "interval_direction",
                lambda x: int((x == "light higher").sum()),
            ),
            n_interval_dark_higher=(
                "interval_direction",
                lambda x: int((x == "dark higher").sum()),
            ),
            n_interval_includes_zero=(
                "interval_direction",
                lambda x: int((x == "includes zero").sum()),
            ),
            min_boot_valid=("n_boot_valid", "min"),
        )
        .reset_index()
    )

    if len(summary) != 35:
        raise RuntimeError(
            f"Expected 35 architecture--metric rows, "
            f"found {len(summary)}."
        )

    if not (summary["n_seeds"] == 5).all():
        raise RuntimeError(
            "At least one architecture--metric combination "
            "does not contain five seeds."
        )

    interval_totals = (
        summary["n_interval_light_higher"]
        + summary["n_interval_dark_higher"]
        + summary["n_interval_includes_zero"]
    )

    if not (interval_totals == 5).all():
        raise RuntimeError(
            "At least one interval-direction total is not five."
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Seed-aware BOSQUE light--dark gap analysis."
        )
    )

    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=get_analysis_seed("seed_aware_subgroup_gap"),
    )

    args = parser.parse_args()

    canonical_files = discover_canonical_files()

    rows = []

    for model_index, model in enumerate(MODELS):
        for seed in sorted(EXPECTED_SEEDS):
            path = canonical_files[(model, seed)]

            print(
                f"Processing {MODEL_LABELS[model]} "
                f"seed {seed}..."
            )

            df = read_and_validate(path)

            system_seed = (
                args.seed
                + model_index * 1000
                + seed
            )

            rows.extend(
                bootstrap_locked_system(
                    df=df,
                    model=model,
                    seed=seed,
                    prediction_file=path,
                    n_boot=args.n_boot,
                    threshold=args.threshold,
                    random_seed=system_seed,
                )
            )

    by_seed = pd.DataFrame(rows)

    expected_rows = (
        len(MODELS)
        * len(EXPECTED_SEEDS)
        * len(METRICS)
    )

    if len(by_seed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} by-seed rows, "
            f"found {len(by_seed)}."
        )

    keys = [
        "model",
        "seed",
        "metric",
    ]

    if by_seed.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate model--seed--metric rows detected."
        )

    model_order = {
        model: index
        for index, model in enumerate(MODELS)
    }

    metric_order = {
        metric: index
        for index, metric in enumerate(METRICS)
    }

    by_seed["_model_order"] = (
        by_seed["model"].map(model_order)
    )

    by_seed["_metric_order"] = (
        by_seed["metric"].map(metric_order)
    )

    by_seed = (
        by_seed.sort_values(
            [
                "_metric_order",
                "_model_order",
                "seed",
            ]
        )
        .drop(
            columns=[
                "_model_order",
                "_metric_order",
            ]
        )
        .reset_index(drop=True)
    )

    summary = summarize_by_architecture(by_seed)

    summary["_model_order"] = (
        summary["model"].map(model_order)
    )

    summary["_metric_order"] = (
        summary["metric"].map(metric_order)
    )

    summary = (
        summary.sort_values(
            [
                "_metric_order",
                "_model_order",
            ]
        )
        .drop(
            columns=[
                "_model_order",
                "_metric_order",
            ]
        )
        .reset_index(drop=True)
    )

    by_seed_path = (
        TABLES
        / "bosque_light_dark_gap_by_model_seed.csv"
    )

    summary_path = (
        TABLES
        / "bosque_light_dark_gap_summary_by_architecture.csv"
    )

    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print("Saved:", by_seed_path)
    print("Shape:", by_seed.shape)

    print("Saved:", summary_path)
    print("Shape:", summary.shape)

    print()
    print("Primary-metric architecture summary:")

    print(
        summary[
            summary["metric_group"] == "Primary"
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
