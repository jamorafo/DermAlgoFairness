#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interval-based TAC/ETC analysis for image-based diagnostic AI.

This script implements a Chapter-5-consistent TAC/ETC analysis using
prediction-level files rather than summary means.

It estimates, for each locked model replica f_{a,s}:

  theta_S^M       = source overall performance
  theta_T^M       = target overall performance
  theta_{T,g}^M   = target subgroup performance

and the benchmark-preservation degradation:

  Delta_{S->T,g}^M = theta_S^M - theta_{T,g}^M

Positive degradation means target performance is lower than source performance.

Metrics are divided into:
  - primary validation metrics used for the main PR/TAC/ETC interpretation
  - secondary descriptive metrics reported for completeness
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from dermalgo.seeds import get_analysis_seed, get_training_seeds

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------

PRIMARY_METRICS = ["recall", "auc_pr", "f1", "precision"]
SECONDARY_METRICS = ["accuracy", "specificity", "auc_roc"]

METRIC_ORDER = PRIMARY_METRICS + SECONDARY_METRICS

METRIC_CONFIG = {
    # Primary validation metrics
    "recall": {
        "tau": 0.70,
        "epsilon": 0.05,
        "label": "Recall / sensitivity",
        "metric_group": "Primary",
    },
    "auc_pr": {
        "tau": 0.85,
        "epsilon": 0.05,
        "label": "AUC-PR",
        "metric_group": "Primary",
    },
    "f1": {
        "tau": 0.70,
        "epsilon": 0.05,
        "label": "F1-score",
        "metric_group": "Primary",
    },
    "precision": {
        "tau": 0.80,
        "epsilon": 0.05,
        "label": "Precision",
        "metric_group": "Primary",
    },

    # Secondary descriptive metrics
    "accuracy": {
        "tau": 0.80,
        "epsilon": 0.05,
        "label": "Accuracy",
        "metric_group": "Secondary",
    },
    "specificity": {
        "tau": 0.80,
        "epsilon": 0.05,
        "label": "Specificity",
        "metric_group": "Secondary",
    },
    "auc_roc": {
        "tau": 0.85,
        "epsilon": 0.05,
        "label": "AUC-ROC",
        "metric_group": "Secondary",
    },
}

N_BOOT = 10000
TRAINING_SEEDS = get_training_seeds()
RANDOM_STATE = get_analysis_seed("primary_interval_bootstrap")
THRESHOLD = 0.5

SOURCE_SPLIT_MANIFEST = (
    ROOT / "splits" / "ham10000_lesion_grouped_fixed.csv"
)

if not SOURCE_SPLIT_MANIFEST.exists():
    raise FileNotFoundError(
        f"Missing fixed source split manifest: {SOURCE_SPLIT_MANIFEST}"
    )

_source_split_frame = pd.read_csv(
    SOURCE_SPLIT_MANIFEST,
    usecols=["split"],
)

SOURCE_EXPECTED_N = int(
    (
        _source_split_frame["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "test"
    ).sum()
)


MODEL_ORDER = [
    "ResNet50",
    "DenseNet121",
    "MobileNetV2",
    "EfficientNetV2B0",
    "VGG16",
]

TARGET_ORDER = ["BOSQUE overall", "BOSQUE light", "BOSQUE dark"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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


def parse_model_seed(path):
    name = path.name.lower()

    model = None
    for candidate in [
        "efficientnetv2b0",
        "densenet121",
        "mobilenetv2",
        "resnet50",
        "vgg16",
    ]:
        if candidate in name:
            model = normalize_model(candidate)
            break

    m = re.search(r"seed(\d+)", name)
    seed = int(m.group(1)) if m else None

    return model, seed



def discover_prediction_files():
    """
    Discover one canonical source and target prediction file per model--seed pair.

    Primary source:
      ham10000_internal_test_predictions_*.csv

    Primary target:
      bosque_public_predictions_*.csv

    Whole-HAM10000 predictions are deliberately excluded from the primary
    source-to-target TAC/ETC analysis.

    When duplicate files exist for the same model--seed pair, the file with the
    latest timestamp in its filename is selected.
    """

    expected_pairs = {
        (model, seed)
        for model in MODEL_ORDER
        for seed in TRAINING_SEEDS
    }

    timestamp_re = re.compile(r"_(\d{8}T\d{6}Z)\.csv$")

    def latest_file_map(pattern, label, expected_n, require_skin_group=False):
        grouped = {}

        for path in sorted(PRED_DIR.glob(pattern)):
            model, seed = parse_model_seed(path)

            if model is None or seed is None:
                warnings.warn(f"Could not parse {label} file: {path}")
                continue

            timestamp_match = timestamp_re.search(path.name)
            if timestamp_match is None:
                warnings.warn(f"Could not parse timestamp from {path.name}")
                continue

            timestamp = timestamp_match.group(1)
            key = (model, seed)

            grouped.setdefault(key, []).append(
                {
                    "timestamp": timestamp,
                    "path": path,
                }
            )

        selected = {}
        manifest_rows = []

        for key, candidates in sorted(grouped.items()):
            latest = max(
                candidates,
                key=lambda item: item["timestamp"],
            )

            path = latest["path"]
            df = pd.read_csv(path)

            required = {"y_true", "y_score"}
            if require_skin_group:
                required.add("skin_group")

            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {sorted(missing)}"
                )

            if len(df) != expected_n:
                raise ValueError(
                    f"{path} contains {len(df)} rows; expected {expected_n}."
                )

            if df["y_true"].isna().any():
                raise ValueError(f"{path} contains missing y_true values.")

            if df["y_score"].isna().any():
                raise ValueError(f"{path} contains missing y_score values.")

            selected[key] = path

            y_true = pd.to_numeric(
                df["y_true"],
                errors="raise",
            ).astype(int)

            manifest_rows.append(
                {
                    "dataset": label,
                    "model": key[0],
                    "seed": key[1],
                    "timestamp": latest["timestamp"],
                    "n": len(df),
                    "n_negative": int((y_true == 0).sum()),
                    "n_positive": int((y_true == 1).sum()),
                    "prediction_file": str(path.relative_to(ROOT)),
                    "n_duplicate_candidates": len(candidates),
                }
            )

        found_pairs = set(selected)
        missing_pairs = expected_pairs - found_pairs
        unexpected_pairs = found_pairs - expected_pairs

        if missing_pairs:
            raise RuntimeError(
                f"Missing {label} model--seed pairs: {sorted(missing_pairs)}"
            )

        if unexpected_pairs:
            raise RuntimeError(
                f"Unexpected {label} model--seed pairs: "
                f"{sorted(unexpected_pairs)}"
            )

        if len(selected) != 25:
            raise RuntimeError(
                f"Expected 25 canonical {label} files, found {len(selected)}."
            )

        return selected, manifest_rows

    source_map, source_manifest = latest_file_map(
        pattern="ham10000_internal_test_predictions_*_seed*.csv",
        label="HAM10000 internal test",
        expected_n=SOURCE_EXPECTED_N,
        require_skin_group=False,
    )

    target_map, target_manifest = latest_file_map(
        pattern="bosque_public_predictions_*_seed*.csv",
        label="BOSQUE external",
        expected_n=151,
        require_skin_group=True,
    )

    common = sorted(set(source_map).intersection(target_map))

    if set(common) != expected_pairs:
        raise RuntimeError(
            "The matched source--target pairs do not equal the expected "
            "5 architectures x 5 seeds."
        )

    manifest = pd.DataFrame(source_manifest + target_manifest)
    manifest = manifest.sort_values(
        ["dataset", "model", "seed"]
    ).reset_index(drop=True)

    logs_dir = ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = logs_dir / "primary_prediction_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Canonical source prediction files: {len(source_map)}")
    print(f"Canonical target prediction files: {len(target_map)}")
    print(f"Matched model--seed pairs: {len(common)}")
    print(f"Saved canonical input manifest: {manifest_path}")

    return source_map, target_map, common


def read_predictions(path):
    df = pd.read_csv(path)

    required = {"y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype("Int64")
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")

    df = df.dropna(subset=["y_true", "y_score"]).copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = (df["y_score"] >= THRESHOLD).astype(int)

    return df


def metric_value(y_true, y_score, metric):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= THRESHOLD).astype(int)

    if metric in {"auc_roc", "auc_pr"}:
        if len(np.unique(y_true)) < 2:
            return np.nan

    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)

    if metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)

    if metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)

    if metric == "specificity":
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return np.nan if (tn + fp) == 0 else tn / (tn + fp)

    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)

    if metric == "auc_roc":
        return roc_auc_score(y_true, y_score)

    if metric == "auc_pr":
        return average_precision_score(y_true, y_score)

    raise ValueError(f"Unsupported metric: {metric}")


def percentile_interval(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size == 0:
        return np.nan, np.nan

    return (
        np.percentile(values, 100 * alpha / 2),
        np.percentile(values, 100 * (1 - alpha / 2)),
    )


def bootstrap_degradation_interval(source_df, target_df, metric, rng, n_boot=N_BOOT):
    ns = len(source_df)
    nt = len(target_df)

    y_s = source_df["y_true"].to_numpy()
    p_s = source_df["y_score"].to_numpy()

    y_t = target_df["y_true"].to_numpy()
    p_t = target_df["y_score"].to_numpy()

    deltas = []
    source_vals = []
    target_vals = []

    for _ in range(n_boot):
        idx_s = rng.integers(0, ns, size=ns)
        idx_t = rng.integers(0, nt, size=nt)

        theta_s = metric_value(y_s[idx_s], p_s[idx_s], metric)
        theta_t = metric_value(y_t[idx_t], p_t[idx_t], metric)

        if np.isnan(theta_s) or np.isnan(theta_t):
            continue

        source_vals.append(theta_s)
        target_vals.append(theta_t)
        deltas.append(theta_s - theta_t)

    return (
        np.asarray(source_vals, dtype=float),
        np.asarray(target_vals, dtype=float),
        np.asarray(deltas, dtype=float),
    )


def tac_decision(l_theta, u_theta, tau):
    if np.isnan(l_theta) or np.isnan(u_theta):
        return "unresolved"

    if l_theta >= tau:
        return "adequate"

    if u_theta < tau:
        return "not adequate"

    return "inconclusive"


def etc_decision(l_delta, u_delta, epsilon):
    if np.isnan(l_delta) or np.isnan(u_delta):
        return "unresolved"

    if u_delta <= epsilon:
        return "transported"

    if l_delta > epsilon:
        return "not transported"

    return "inconclusive"


def joint_region(tac, etc):
    if tac == "adequate" and etc == "transported":
        return "adequate and transported"

    if tac == "adequate" and etc == "not transported":
        return "adequate but not transported"

    if tac == "not adequate" and etc == "transported":
        return "transported but inadequate"

    if tac == "not adequate" and etc == "not transported":
        return "not adequate and not transported"

    if "unresolved" in {tac, etc}:
        return "evidentially unresolved"

    return "inconclusive"


def make_target_subsets(target_df):
    out = {
        "BOSQUE overall": target_df.copy(),
    }

    if "skin_group" in target_df.columns:
        for group in ["light", "dark"]:
            subset = target_df[
                target_df["skin_group"].astype(str).str.lower() == group
            ].copy()
            if len(subset) > 0:
                out[f"BOSQUE {group}"] = subset

    return out


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_STATE)

    source_map, target_map, pairs = discover_prediction_files()

    rows = []

    for model, seed in pairs:
        print(f"Processing {model} seed {seed}...")

        source_path = source_map[(model, seed)]
        target_path = target_map[(model, seed)]

        source_df = read_predictions(source_path)
        target_df = read_predictions(target_path)

        target_subsets = make_target_subsets(target_df)

        for metric in METRIC_ORDER:
            cfg = METRIC_CONFIG[metric]
            tau = cfg["tau"]
            epsilon = cfg["epsilon"]
            metric_group = cfg["metric_group"]

            theta_s = metric_value(
                source_df["y_true"].to_numpy(),
                source_df["y_score"].to_numpy(),
                metric,
            )

            for target_condition, target_part in target_subsets.items():
                theta_t = metric_value(
                    target_part["y_true"].to_numpy(),
                    target_part["y_score"].to_numpy(),
                    metric,
                )

                source_boot, target_boot, delta_boot = bootstrap_degradation_interval(
                    source_df=source_df,
                    target_df=target_part,
                    metric=metric,
                    rng=rng,
                    n_boot=N_BOOT,
                )

                l_source, u_source = percentile_interval(source_boot)
                l_theta, u_theta = percentile_interval(target_boot)
                l_delta, u_delta = percentile_interval(delta_boot)

                delta_hat = theta_s - theta_t

                tac = tac_decision(l_theta, u_theta, tau)
                etc = etc_decision(l_delta, u_delta, epsilon)
                region = joint_region(tac, etc)

                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "metric": metric,
                        "metric_group": metric_group,
                        "target_condition": target_condition,
                        "n_source": len(source_df),
                        "n_target": len(target_part),
                        "source_performance": theta_s,
                        "source_ci_low": l_source,
                        "source_ci_high": u_source,
                        "target_performance": theta_t,
                        "target_ci_low": l_theta,
                        "target_ci_high": u_theta,
                        "degradation": delta_hat,
                        "degradation_ci_low": l_delta,
                        "degradation_ci_high": u_delta,
                        "tau": tau,
                        "epsilon": epsilon,
                        "tac_decision": tac,
                        "etc_decision": etc,
                        "joint_region": region,
                        "source_file": str(source_path.relative_to(ROOT)),
                        "target_file": str(target_path.relative_to(ROOT)),
                    }
                )

    by_seed = pd.DataFrame(rows)

    by_seed["model"] = pd.Categorical(
        by_seed["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    by_seed["metric"] = pd.Categorical(
        by_seed["metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    by_seed["target_condition"] = pd.Categorical(
        by_seed["target_condition"],
        categories=TARGET_ORDER,
        ordered=True,
    )

    by_seed = by_seed.sort_values(
        ["metric_group", "metric", "target_condition", "model", "seed"]
    ).reset_index(drop=True)

    by_seed_path = TABLES / "interval_tac_etc_by_seed.csv"
    by_seed.to_csv(by_seed_path, index=False)

    print(f"\nSaved: {by_seed_path}")
    print("Shape:", by_seed.shape)

    summary = (
        by_seed
        .groupby(["metric_group", "model", "metric", "target_condition"], observed=True)
        .agg(
            n_seeds=("seed", "nunique"),
            n_target=("n_target", "first"),
            source_mean=("source_performance", "mean"),
            source_sd=("source_performance", "std"),
            target_mean=("target_performance", "mean"),
            target_sd=("target_performance", "std"),
            degradation_mean=("degradation", "mean"),
            degradation_sd=("degradation", "std"),
            tac_adequate_n=("tac_decision", lambda x: (x == "adequate").sum()),
            tac_not_adequate_n=("tac_decision", lambda x: (x == "not adequate").sum()),
            tac_inconclusive_n=("tac_decision", lambda x: (x == "inconclusive").sum()),
            etc_transported_n=("etc_decision", lambda x: (x == "transported").sum()),
            etc_not_transported_n=("etc_decision", lambda x: (x == "not transported").sum()),
            etc_inconclusive_n=("etc_decision", lambda x: (x == "inconclusive").sum()),
            most_common_joint_region=("joint_region", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )

    summary_path = TABLES / "interval_tac_etc_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {summary_path}")
    print("Shape:", summary.shape)

    consistency = (
        by_seed
        .groupby(["metric_group", "metric", "target_condition", "joint_region"], observed=True)
        .size()
        .reset_index(name="n_model_seed_decisions")
        .sort_values(
            ["metric_group", "metric", "target_condition", "n_model_seed_decisions"],
            ascending=[True, True, True, False],
        )
    )

    consistency_path = TABLES / "interval_tac_etc_consistency.csv"
    consistency.to_csv(consistency_path, index=False)

    print(f"Saved: {consistency_path}")
    print("Shape:", consistency.shape)

    print("\nPreview: interval_tac_etc_consistency.csv")
    print(consistency.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
