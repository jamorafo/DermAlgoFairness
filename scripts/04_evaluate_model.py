#!/usr/bin/env python3
"""Evaluate a locked model on fixed HAM10000 and BOSQUE ORPs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dermalgo.config import (
    load_config,
    require_keys,
)
from dermalgo.data import (
    load_bosque_public,
    load_ham10000,
)
from dermalgo.fixed_split import (
    load_fixed_ham10000_split,
)
from dermalgo.models import (
    get_preprocess_input,
)
from dermalgo.paths import ensure_dir
from dermalgo.tfdata import make_image_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained model on the immutable "
            "HAM10000 source ORP and fixed BOSQUE target ORP."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--model-path",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Training-randomization value associated with "
            "the locked model. It does not determine the "
            "evaluation datasets."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate prespecified binary-classification metrics."""
    y_pred = (
        y_score >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if precision + recall > 0:
        f1_score = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1_score = 0.0

    return {
        "n": int(
            len(y_true)
        ),
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "specificity": float(
            tn / (tn + fp)
        )
        if (tn + fp) > 0
        else float("nan"),
        "f1": float(
            f1_score
        ),
        "auc_roc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )
        if len(set(y_true)) == 2
        else float("nan"),
        "auc_pr": float(
            average_precision_score(
                y_true,
                y_score,
            )
        )
        if len(set(y_true)) == 2
        else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_dataframe(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    preprocess_input,
) -> pd.DataFrame:
    """Generate image-level predictions without shuffling."""
    dataset = make_image_dataset(
        frame,
        image_size=image_size,
        batch_size=batch_size,
        preprocess_input=preprocess_input,
        shuffle=False,
    )

    scores = (
        model.predict(
            dataset
        )
        .reshape(-1)
    )

    output = frame.copy()

    output["y_true"] = (
        output["label_binary"]
        .astype(int)
    )

    output["y_score"] = scores

    return output


def main() -> None:
    """Evaluate one locked architecture-run system."""
    args = parse_args()

    config = load_config(
        args.config
    )

    require_keys(
        config,
        [
            "model",
            "data",
            "split",
            "training",
            "evaluation",
            "outputs",
        ],
        context=args.config,
    )

    if "manifest_path" not in config["split"]:
        raise ValueError(
            "The configuration must define "
            "split.manifest_path."
        )

    model_path = Path(
        args.model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    model_name = config["model"]["name"]

    image_size = tuple(
        config["model"]["input_size"]
    )

    batch_size = int(
        config["training"]["batch_size"]
    )

    preprocess_input = get_preprocess_input(
        config["model"]["keras_application"]
    )

    predictions_dir = ensure_dir(
        config["outputs"]["predictions_dir"]
    )

    tables_dir = ensure_dir(
        config["outputs"]["tables_dir"]
    )

    logs_dir = ensure_dir(
        config["outputs"]["logs_dir"]
    )

    run_timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_id = (
        f"{model_name}_seed{args.seed}_"
        f"{run_timestamp}"
    )

    ham_df = load_ham10000(
        ham10000_root=(
            config["data"]["ham10000_root"]
        ),
        metadata_filename=Path(
            config["data"]["ham10000_metadata"]
        ).name,
        benign_labels=(
            config["data"]["benign_labels"]
        ),
        malignant_labels=(
            config["data"]["malignant_labels"]
        ),
    )

    (
        _,
        _,
        ham_test_df,
        split_manifest_path,
        split_manifest_sha256,
    ) = load_fixed_ham10000_split(
        ham_df,
        manifest_path=(
            config["split"]["manifest_path"]
        ),
    )

    bosque_df = load_bosque_public(
        bosque_root=(
            config["data"]["bosque_root"]
        ),
        light_phototypes=(
            config["data"].get(
                "light_phototypes",
                [
                    1,
                    2,
                    3,
                ],
            )
        ),
        dark_phototypes=(
            config["data"].get(
                "dark_phototypes",
                [
                    4,
                    5,
                    6,
                ],
            )
        ),
    )

    ham_predictions = predict_dataframe(
        model,
        ham_test_df,
        image_size,
        batch_size,
        preprocess_input,
    )

    bosque_predictions = predict_dataframe(
        model,
        bosque_df,
        image_size,
        batch_size,
        preprocess_input,
    )

    ham_predictions_path = (
        predictions_dir
        / (
            "ham10000_internal_test_predictions_"
            f"{run_id}.csv"
        )
    )

    bosque_predictions_path = (
        predictions_dir
        / (
            "bosque_public_predictions_"
            f"{run_id}.csv"
        )
    )

    ham_predictions.to_csv(
        ham_predictions_path,
        index=False,
    )

    bosque_predictions.to_csv(
        bosque_predictions_path,
        index=False,
    )

    metric_rows = []

    for dataset_name, prediction_frame in [
        (
            "HAM10000_internal_test",
            ham_predictions,
        ),
        (
            "BOSQUE_public",
            bosque_predictions,
        ),
    ]:
        metrics = compute_binary_metrics(
            prediction_frame[
                "y_true"
            ].to_numpy(),
            prediction_frame[
                "y_score"
            ].to_numpy(),
            threshold=args.threshold,
        )

        metric_rows.append(
            {
                "dataset": dataset_name,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_path = (
        tables_dir
        / f"evaluation_metrics_{run_id}.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    subgroup_rows = []

    for subgroup_name, subgroup_frame in (
        bosque_predictions.groupby(
            "skin_group"
        )
    ):
        metrics = compute_binary_metrics(
            subgroup_frame[
                "y_true"
            ].to_numpy(),
            subgroup_frame[
                "y_score"
            ].to_numpy(),
            threshold=args.threshold,
        )

        subgroup_rows.append(
            {
                "dataset": "BOSQUE_public",
                "subgroup_column": "skin_group",
                "subgroup": subgroup_name,
                **metrics,
            }
        )

    subgroup_metrics_df = pd.DataFrame(
        subgroup_rows
    )

    subgroup_metrics_path = (
        tables_dir
        / (
            "evaluation_metrics_by_subgroup_"
            f"{run_id}.csv"
        )
    )

    subgroup_metrics_df.to_csv(
        subgroup_metrics_path,
        index=False,
    )

    evaluation_manifest = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "run_id": run_id,
        "config_path": str(
            args.config
        ),
        "model_path": str(
            model_path
        ),
        "model_name": model_name,
        "training_randomization_seed": int(
            args.seed
        ),
        "threshold": float(
            args.threshold
        ),
        "source_orp": (
            "fixed lesion-grouped HAM10000 "
            "internal-test partition"
        ),
        "source_prediction_unit": "image",
        "source_cluster_variable": "lesion_id",
        "source_n_images": int(
            len(ham_test_df)
        ),
        "source_n_lesions": int(
            ham_test_df[
                "lesion_id"
            ].nunique()
        ),
        "split_manifest_path": str(
            split_manifest_path
        ),
        "split_manifest_sha256": (
            split_manifest_sha256
        ),
        "target_orp": (
            "BOSQUE external image collection"
        ),
        "target_prediction_unit": "image",
        "target_n_images": int(
            len(bosque_df)
        ),
        "ham10000_predictions_path": str(
            ham_predictions_path
        ),
        "bosque_predictions_path": str(
            bosque_predictions_path
        ),
        "metrics_path": str(
            metrics_path
        ),
        "subgroup_metrics_path": str(
            subgroup_metrics_path
        ),
    }

    evaluation_manifest_path = (
        logs_dir
        / f"evaluation_manifest_{run_id}.json"
    )

    evaluation_manifest_path.write_text(
        json.dumps(
            evaluation_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Evaluation ORPs")
    print("---------------")
    print(
        "Fixed HAM10000 source ORP: "
        f"{len(ham_test_df)} images, "
        f"{ham_test_df['lesion_id'].nunique()} lesions"
    )
    print(
        "BOSQUE target ORP: "
        f"{len(bosque_df)} images"
    )
    print(
        "Fixed split SHA-256: "
        f"{split_manifest_sha256}"
    )

    print()
    print("Evaluation metrics")
    print("------------------")
    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Wrote HAM10000 predictions: "
        f"{ham_predictions_path}"
    )
    print(
        f"Wrote BOSQUE predictions: "
        f"{bosque_predictions_path}"
    )
    print(
        f"Wrote metrics: {metrics_path}"
    )
    print(
        "Wrote subgroup metrics: "
        f"{subgroup_metrics_path}"
    )
    print(
        "Wrote evaluation manifest: "
        f"{evaluation_manifest_path}"
    )


if __name__ == "__main__":
    main()
