#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interval-based TAC/ETC analysis for image-based diagnostic AI.

This script implements a Chapter-5-consistent TAC/ETC analysis using
prediction-level files rather than summary means. Source uncertainty is
estimated by resampling HAM10000 lesions as clusters, while BOSQUE is
resampled at the image level because it contains one image per lesion.

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
    Select the exact source and target prediction files recorded in the
    verified evaluation matrix.

    Historical prediction files remain preserved but are excluded unless
    explicitly listed in the evaluation-status manifest.
    """

    status_path = (
        ROOT
        / "outputs"
        / "logs"
        / "evaluation_matrix_status_3a403f3.tsv"
    )

    if not status_path.exists():
        raise FileNotFoundError(
            "Missing verified evaluation-status manifest: "
            f"{status_path}"
        )

    status = pd.read_csv(
        status_path,
        sep="\t",
    )

    required_status_columns = {
        "architecture",
        "seed",
        "evaluation_status",
        "evaluation_run_id",
    }

    missing = (
        required_status_columns
        - set(status.columns)
    )

    if missing:
        raise ValueError(
            "The evaluation-status manifest is missing columns: "
            f"{sorted(missing)}"
        )

    if len(status) != 25:
        raise RuntimeError(
            "Expected 25 verified evaluation records, "
            f"found {len(status)}."
        )

    status = status.copy()

    status["architecture"] = (
        status["architecture"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    status["model"] = (
        status["architecture"]
        .map(normalize_model)
    )

    status["seed"] = (
        status["seed"]
        .astype(int)
    )

    status["evaluation_run_id"] = (
        status["evaluation_run_id"]
        .astype(str)
        .str.strip()
    )

    if status[
        ["model", "seed"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate model--seed rows were found in "
            "the evaluation-status manifest."
        )

    allowed_statuses = {
        "evaluated_and_verified",
        "already_verified",
    }

    if not set(
        status["evaluation_status"]
    ).issubset(allowed_statuses):
        raise RuntimeError(
            "The evaluation-status manifest contains an "
            "unverified evaluation status."
        )

    expected_pairs = {
        (model, seed)
        for model in MODEL_ORDER
        for seed in TRAINING_SEEDS
    }

    observed_pairs = set(
        zip(
            status["model"],
            status["seed"],
        )
    )

    if observed_pairs != expected_pairs:
        raise RuntimeError(
            "The verified evaluation matrix does not match "
            "the expected 5 architectures x 5 seeds."
        )

    source_map = {}
    target_map = {}
    manifest_rows = []

    def validate_prediction_file(
        prediction_path,
        dataset_label,
        expected_n,
        require_lesion_id=False,
        require_skin_group=False,
    ):
        if (
            not prediction_path.exists()
            or prediction_path.stat().st_size == 0
        ):
            raise FileNotFoundError(
                f"Missing prediction file: {prediction_path}"
            )

        frame = pd.read_csv(
            prediction_path
        )

        required = {
            "y_true",
            "y_score",
        }

        if require_lesion_id:
            required.add("lesion_id")

        if require_skin_group:
            required.add("skin_group")

        missing_columns = (
            required - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{prediction_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if len(frame) != expected_n:
            raise ValueError(
                f"{prediction_path} contains {len(frame)} rows; "
                f"expected {expected_n}."
            )

        if frame[
            ["y_true", "y_score"]
        ].isna().any().any():
            raise ValueError(
                f"{prediction_path} contains missing "
                "outcomes or scores."
            )

        y_true = pd.to_numeric(
            frame["y_true"],
            errors="raise",
        ).astype(int)

        if not set(
            y_true.unique()
        ).issubset({0, 1}):
            raise ValueError(
                f"{prediction_path} contains non-binary outcomes."
            )

        if require_lesion_id:
            if frame["lesion_id"].isna().any():
                raise ValueError(
                    f"{prediction_path} contains missing lesion IDs."
                )

            if frame["lesion_id"].nunique() != 747:
                raise ValueError(
                    f"{prediction_path} contains "
                    f"{frame['lesion_id'].nunique()} lesions; "
                    "expected 747."
                )

        if require_skin_group:
            skin_counts = (
                frame["skin_group"]
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
                .to_dict()
            )

            if skin_counts != {
                "light": 105,
                "dark": 46,
            }:
                raise ValueError(
                    f"{prediction_path} has unexpected "
                    f"skin-group counts: {skin_counts}"
                )

        return frame, y_true

    for row in status.itertuples(index=False):
        architecture = str(
            row.architecture
        )

        model = str(
            row.model
        )

        seed = int(
            row.seed
        )

        evaluation_run_id = str(
            row.evaluation_run_id
        )

        expected_prefix = (
            f"{architecture}_seed{seed}_"
        )

        if not evaluation_run_id.startswith(
            expected_prefix
        ):
            raise RuntimeError(
                "Evaluation run-ID mismatch for "
                f"{architecture}, seed {seed}: "
                f"{evaluation_run_id}"
            )

        source_path = (
            PRED_DIR
            / (
                "ham10000_internal_test_predictions_"
                f"{evaluation_run_id}.csv"
            )
        )

        target_path = (
            PRED_DIR
            / (
                "bosque_public_predictions_"
                f"{evaluation_run_id}.csv"
            )
        )

        source_frame, source_y = (
            validate_prediction_file(
                source_path,
                dataset_label="HAM10000 internal test",
                expected_n=SOURCE_EXPECTED_N,
                require_lesion_id=True,
            )
        )

        target_frame, target_y = (
            validate_prediction_file(
                target_path,
                dataset_label="BOSQUE external",
                expected_n=151,
                require_skin_group=True,
            )
        )

        key = (
            model,
            seed,
        )

        source_map[key] = source_path
        target_map[key] = target_path

        timestamp = (
            evaluation_run_id
            .rsplit("_", maxsplit=1)[-1]
        )

        source_candidates = list(
            PRED_DIR.glob(
                "ham10000_internal_test_predictions_"
                f"{architecture}_seed{seed}_*.csv"
            )
        )

        target_candidates = list(
            PRED_DIR.glob(
                "bosque_public_predictions_"
                f"{architecture}_seed{seed}_*.csv"
            )
        )

        manifest_rows.extend(
            [
                {
                    "dataset": "HAM10000 internal test",
                    "model": model,
                    "seed": seed,
                    "timestamp": timestamp,
                    "evaluation_run_id": evaluation_run_id,
                    "n": len(source_frame),
                    "n_clusters": (
                        source_frame["lesion_id"].nunique()
                    ),
                    "n_negative": int(
                        (source_y == 0).sum()
                    ),
                    "n_positive": int(
                        (source_y == 1).sum()
                    ),
                    "prediction_file": str(
                        source_path.relative_to(ROOT)
                    ),
                    "n_duplicate_candidates": len(
                        source_candidates
                    ),
                    "selection_source": str(
                        status_path.relative_to(ROOT)
                    ),
                },
                {
                    "dataset": "BOSQUE external",
                    "model": model,
                    "seed": seed,
                    "timestamp": timestamp,
                    "evaluation_run_id": evaluation_run_id,
                    "n": len(target_frame),
                    "n_clusters": len(target_frame),
                    "n_negative": int(
                        (target_y == 0).sum()
                    ),
                    "n_positive": int(
                        (target_y == 1).sum()
                    ),
                    "prediction_file": str(
                        target_path.relative_to(ROOT)
                    ),
                    "n_duplicate_candidates": len(
                        target_candidates
                    ),
                    "selection_source": str(
                        status_path.relative_to(ROOT)
                    ),
                },
            ]
        )

    model_index = {
        model: index
        for index, model in enumerate(MODEL_ORDER)
    }

    seed_index = {
        seed: index
        for index, seed in enumerate(TRAINING_SEEDS)
    }

    pairs = sorted(
        expected_pairs,
        key=lambda item: (
            model_index[item[0]],
            seed_index[item[1]],
        ),
    )

    if len(source_map) != 25:
        raise RuntimeError(
            f"Expected 25 source files, found {len(source_map)}."
        )

    if len(target_map) != 25:
        raise RuntimeError(
            f"Expected 25 target files, found {len(target_map)}."
        )

    manifest = (
        pd.DataFrame(manifest_rows)
        .sort_values(
            [
                "dataset",
                "model",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )

    logs_dir = (
        ROOT
        / "outputs"
        / "logs"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        logs_dir
        / "primary_prediction_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    print(
        f"Canonical source prediction files: {len(source_map)}"
    )
    print(
        f"Canonical target prediction files: {len(target_map)}"
    )
    print(
        f"Matched model--seed pairs: {len(pairs)}"
    )
    print(
        f"Saved canonical input manifest: {manifest_path}"
    )

    return (
        source_map,
        target_map,
        pairs,
    )


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


def metric_value(
    y_true,
    y_score,
    metric,
    sample_weight=None,
):
    """
    Compute one performance metric.

    Bootstrap frequency weights are equivalent to explicitly duplicating
    sampled observations or sampled lesion clusters.
    """

    y_true = np.asarray(
        y_true,
    ).astype(int)

    y_score = np.asarray(
        y_score,
    ).astype(float)

    if sample_weight is None:
        weight = None
        observed_y = y_true
    else:
        weight = np.asarray(
            sample_weight,
            dtype=float,
        )

        if weight.shape != y_true.shape:
            raise ValueError(
                "sample_weight must have the same shape "
                "as y_true."
            )

        if (
            not np.isfinite(weight).all()
            or (weight < 0).any()
        ):
            raise ValueError(
                "sample_weight contains invalid values."
            )

        observed_y = y_true[
            weight > 0
        ]

    if observed_y.size == 0:
        return np.nan

    y_pred = (
        y_score >= THRESHOLD
    ).astype(int)

    if metric in {
        "auc_roc",
        "auc_pr",
    }:
        if len(
            np.unique(observed_y)
        ) < 2:
            return np.nan

    if metric == "accuracy":
        return accuracy_score(
            y_true,
            y_pred,
            sample_weight=weight,
        )

    if metric == "precision":
        return precision_score(
            y_true,
            y_pred,
            sample_weight=weight,
            zero_division=0,
        )

    if metric == "recall":
        return recall_score(
            y_true,
            y_pred,
            sample_weight=weight,
            zero_division=0,
        )

    if metric == "specificity":
        if weight is None:
            working_weight = np.ones(
                len(y_true),
                dtype=float,
            )
        else:
            working_weight = weight

        negative = (
            y_true == 0
        )

        tn = working_weight[
            negative
            & (y_pred == 0)
        ].sum()

        fp = working_weight[
            negative
            & (y_pred == 1)
        ].sum()

        return (
            np.nan
            if (tn + fp) == 0
            else tn / (tn + fp)
        )

    if metric == "f1":
        return f1_score(
            y_true,
            y_pred,
            sample_weight=weight,
            zero_division=0,
        )

    if metric == "auc_roc":
        return roc_auc_score(
            y_true,
            y_score,
            sample_weight=weight,
        )

    if metric == "auc_pr":
        return average_precision_score(
            y_true,
            y_score,
            sample_weight=weight,
        )

    raise ValueError(
        f"Unsupported metric: {metric}"
    )


def percentile_interval(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size == 0:
        return np.nan, np.nan

    return (
        np.percentile(values, 100 * alpha / 2),
        np.percentile(values, 100 * (1 - alpha / 2)),
    )


def make_cluster_codes(
    frame,
    cluster_column,
):
    """
    Encode the cluster membership of every row as integers 0,...,G-1.
    """

    if cluster_column not in frame.columns:
        raise ValueError(
            f"Missing cluster column: {cluster_column}"
        )

    if frame[
        cluster_column
    ].isna().any():
        raise ValueError(
            f"{cluster_column} contains missing values."
        )

    codes, unique_clusters = pd.factorize(
        frame[cluster_column],
        sort=False,
    )

    if (codes < 0).any():
        raise RuntimeError(
            "Invalid cluster codes were generated."
        )

    n_clusters = int(
        len(unique_clusters)
    )

    if n_clusters < 1:
        raise RuntimeError(
            "No source clusters were found."
        )

    return (
        codes.astype(int),
        n_clusters,
    )


def bootstrap_degradation_interval(
    source_df,
    target_df,
    metric,
    rng,
    source_cluster_codes,
    n_source_clusters,
    n_boot=N_BOOT,
):
    """
    Bootstrap source and target performance and their degradation gap.

    HAM10000 lesions are sampled as clusters. Every image belonging to a
    sampled lesion receives the frequency with which that lesion was drawn.
    BOSQUE rows are sampled independently because the public target ORP
    contains one image per lesion.
    """

    ns = len(source_df)
    nt = len(target_df)

    if len(
        source_cluster_codes
    ) != ns:
        raise ValueError(
            "Source cluster codes do not align with source rows."
        )

    if n_source_clusters != len(
        np.unique(source_cluster_codes)
    ):
        raise ValueError(
            "The source-cluster count is inconsistent "
            "with the cluster codes."
        )

    y_s = (
        source_df["y_true"]
        .to_numpy(dtype=int)
    )

    p_s = (
        source_df["y_score"]
        .to_numpy(dtype=float)
    )

    y_t = (
        target_df["y_true"]
        .to_numpy(dtype=int)
    )

    p_t = (
        target_df["y_score"]
        .to_numpy(dtype=float)
    )

    deltas = []
    source_vals = []
    target_vals = []

    for _ in range(n_boot):
        sampled_source_clusters = rng.integers(
            0,
            n_source_clusters,
            size=n_source_clusters,
        )

        source_cluster_counts = np.bincount(
            sampled_source_clusters,
            minlength=n_source_clusters,
        )

        source_weights = source_cluster_counts[
            source_cluster_codes
        ]

        sampled_target_rows = rng.integers(
            0,
            nt,
            size=nt,
        )

        target_weights = np.bincount(
            sampled_target_rows,
            minlength=nt,
        )

        theta_s = metric_value(
            y_s,
            p_s,
            metric,
            sample_weight=source_weights,
        )

        theta_t = metric_value(
            y_t,
            p_t,
            metric,
            sample_weight=target_weights,
        )

        if (
            np.isnan(theta_s)
            or np.isnan(theta_t)
        ):
            continue

        source_vals.append(
            theta_s
        )

        target_vals.append(
            theta_t
        )

        deltas.append(
            theta_s - theta_t
        )

    return (
        np.asarray(
            source_vals,
            dtype=float,
        ),
        np.asarray(
            target_vals,
            dtype=float,
        ),
        np.asarray(
            deltas,
            dtype=float,
        ),
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

        (
            source_cluster_codes,
            n_source_clusters,
        ) = make_cluster_codes(
            source_df,
            cluster_column="lesion_id",
        )

        if n_source_clusters != 747:
            raise RuntimeError(
                f"{model} seed {seed}: expected 747 source lesions, "
                f"found {n_source_clusters}."
            )

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

                (
                    source_boot,
                    target_boot,
                    delta_boot,
                ) = bootstrap_degradation_interval(
                    source_df=source_df,
                    target_df=target_part,
                    metric=metric,
                    rng=rng,
                    source_cluster_codes=source_cluster_codes,
                    n_source_clusters=n_source_clusters,
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
                        "n_source_clusters": n_source_clusters,
                        "n_target": len(target_part),
                        "source_bootstrap_unit": "lesion",
                        "target_bootstrap_unit": "image_or_lesion",
                        "n_boot_requested": N_BOOT,
                        "n_boot_valid": len(delta_boot),
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
