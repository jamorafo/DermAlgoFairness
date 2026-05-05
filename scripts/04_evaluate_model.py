#!/usr/bin/env python3
"""Evaluate a trained dermatology classifier on HAM10000 and public BOSQUE."""

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

from dermalgo.config import load_config
from dermalgo.data import load_bosque_public, load_ham10000, split_ham10000
from dermalgo.models import get_preprocess_input
from dermalgo.paths import ensure_dir
from dermalgo.tfdata import make_image_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(
            2
            * precision_score(y_true, y_pred, zero_division=0)
            * recall_score(y_true, y_pred, zero_division=0)
            / (
                precision_score(y_true, y_pred, zero_division=0)
                + recall_score(y_true, y_pred, zero_division=0)
            )
        )
        if (precision_score(y_true, y_pred, zero_division=0) + recall_score(y_true, y_pred, zero_division=0))
        else 0.0,
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_score)) if len(set(y_true)) == 2 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out


def predict_dataframe(
    model: tf.keras.Model,
    df: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    preprocess_input,
) -> pd.DataFrame:
    dataset = make_image_dataset(
        df,
        image_size=image_size,
        batch_size=batch_size,
        preprocess_input=preprocess_input,
        shuffle=False,
    )
    scores = model.predict(dataset).reshape(-1)

    out = df.copy()
    out["y_true"] = out["label_binary"].astype(int)
    out["y_score"] = scores
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = Path(args.model_path)
    model = tf.keras.models.load_model(model_path)

    model_name = config["model"]["name"]
    image_size = tuple(config["model"]["input_size"])
    batch_size = int(config["training"]["batch_size"])
    preprocess = get_preprocess_input(config["model"]["keras_application"])

    predictions_dir = ensure_dir(config["outputs"]["predictions_dir"])
    tables_dir = ensure_dir(config["outputs"]["tables_dir"])
    logs_dir = ensure_dir(config["outputs"]["logs_dir"])

    run_id = f"{model_name}_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    ham_df = load_ham10000(
        ham10000_root=config["data"]["ham10000_root"],
        metadata_filename=Path(config["data"]["ham10000_metadata"]).name,
        benign_labels=config["data"]["benign_labels"],
        malignant_labels=config["data"]["malignant_labels"],
    )
    _, _, ham_test_df = split_ham10000(
        ham_df,
        test_size=float(config["split"]["test_size"]),
        validation_fraction_of_temp=float(config["split"]["validation_fraction_of_temp"]),
        seed=args.seed,
    )

    bosque_df = load_bosque_public(bosque_root=config["data"]["bosque_root"])

    ham_pred = predict_dataframe(model, ham_test_df, image_size, batch_size, preprocess)
    bosque_pred = predict_dataframe(model, bosque_df, image_size, batch_size, preprocess)

    ham_pred_path = predictions_dir / f"ham10000_internal_test_predictions_{run_id}.csv"
    bosque_pred_path = predictions_dir / f"bosque_public_predictions_{run_id}.csv"

    ham_pred.to_csv(ham_pred_path, index=False)
    bosque_pred.to_csv(bosque_pred_path, index=False)

    rows = []
    for dataset_name, pred_df in [
        ("HAM10000_internal_test", ham_pred),
        ("BOSQUE_public", bosque_pred),
    ]:
        metrics = compute_binary_metrics(
            pred_df["y_true"].to_numpy(),
            pred_df["y_score"].to_numpy(),
            threshold=args.threshold,
        )
        rows.append({"dataset": dataset_name, **metrics})

    metrics_df = pd.DataFrame(rows)
    metrics_path = tables_dir / f"evaluation_metrics_{run_id}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    subgroup_rows = []
    for subgroup_name, subgroup_df in bosque_pred.groupby("skin_group"):
        metrics = compute_binary_metrics(
            subgroup_df["y_true"].to_numpy(),
            subgroup_df["y_score"].to_numpy(),
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

    subgroup_metrics_df = pd.DataFrame(subgroup_rows)
    subgroup_metrics_path = tables_dir / f"evaluation_metrics_by_subgroup_{run_id}.csv"
    subgroup_metrics_df.to_csv(subgroup_metrics_path, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config_path": str(args.config),
        "model_path": str(model_path),
        "seed": args.seed,
        "threshold": args.threshold,
        "ham10000_predictions_path": str(ham_pred_path),
        "bosque_predictions_path": str(bosque_pred_path),
        "metrics_path": str(metrics_path),
        "subgroup_metrics_path": str(subgroup_metrics_path),
    }
    manifest_path = logs_dir / f"evaluation_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Evaluation metrics:")
    print(metrics_df.to_string(index=False))
    print()
    print(f"Wrote HAM10000 predictions: {ham_pred_path}")
    print(f"Wrote BOSQUE predictions: {bosque_pred_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote subgroup metrics: {subgroup_metrics_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
