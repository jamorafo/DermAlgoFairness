#!/usr/bin/env python3
"""Train a dermatology classifier from a YAML configuration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tensorflow as tf

from dermalgo.config import load_config, require_keys
from dermalgo.data import (
    load_ham10000,
    oversample_training_dataframe,
    split_ham10000,
    summarize_split,
)
from dermalgo.models import build_binary_classifier, get_preprocess_input
from dermalgo.paths import ensure_dir
from dermalgo.tfdata import make_image_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from a YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Optional debugging limit for training batches per epoch.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Optional debugging limit for validation batches.",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    require_keys(
        config,
        ["model", "data", "split", "training", "evaluation", "outputs"],
        context=args.config,
    )

    seeds = config["training"].get("seeds", [])
    seed = int(args.seed if args.seed is not None else seeds[0])

    set_reproducibility(seed)

    model_name = config["model"]["name"]
    batch_size = int(config["training"]["batch_size"])
    image_size = tuple(config["model"]["input_size"])

    models_dir = ensure_dir(config["outputs"]["models_dir"])
    logs_dir = ensure_dir(config["outputs"]["logs_dir"])
    tables_dir = ensure_dir(config["outputs"]["tables_dir"])

    run_id = f"{model_name}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    df = load_ham10000(
        ham10000_root=config["data"]["ham10000_root"],
        metadata_filename=Path(config["data"]["ham10000_metadata"]).name,
        benign_labels=config["data"]["benign_labels"],
        malignant_labels=config["data"]["malignant_labels"],
    )

    train_df, val_df, test_df = split_ham10000(
        df,
        test_size=float(config["split"]["test_size"]),
        validation_fraction_of_temp=float(config["split"]["validation_fraction_of_temp"]),
        seed=seed,
    )

    split_summary = summarize_split(train_df, val_df, test_df)
    split_summary_path = tables_dir / f"ham10000_split_summary_{run_id}.csv"
    split_summary.to_csv(split_summary_path, index=False)

    if config["training"].get("oversampling") == "train_only":
        train_fit_df = oversample_training_dataframe(train_df, seed=seed)
    else:
        train_fit_df = train_df.copy()

    oversampled_counts_path = tables_dir / f"ham10000_training_counts_after_oversampling_{run_id}.csv"
    (
        train_fit_df["label_binary"]
        .value_counts()
        .sort_index()
        .rename_axis("label_binary")
        .reset_index(name="n")
        .to_csv(oversampled_counts_path, index=False)
    )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config_path": str(args.config),
        "model_name": model_name,
        "seed": seed,
        "dry_run": bool(args.dry_run),
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "n_raw": len(df),
        "n_train_before_oversampling": len(train_df),
        "n_validation": len(val_df),
        "n_internal_test": len(test_df),
        "n_train_after_oversampling": len(train_fit_df),
        "split_summary_path": str(split_summary_path),
        "oversampled_counts_path": str(oversampled_counts_path),
        "model": config["model"],
        "training": config["training"],
    }

    manifest_path = logs_dir / f"train_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Input size: {image_size}")
    print(f"Train before oversampling: {len(train_df)}")
    print(f"Train after oversampling: {len(train_fit_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Internal test: {len(test_df)}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote split summary: {split_summary_path}")

    if args.dry_run:
        return

    preprocess = get_preprocess_input(config["model"]["keras_application"])

    train_ds = make_image_dataset(
        train_fit_df,
        image_size=image_size,
        batch_size=batch_size,
        preprocess_input=preprocess,
        shuffle=True,
        seed=seed,
    )
    val_ds = make_image_dataset(
        val_df,
        image_size=image_size,
        batch_size=batch_size,
        preprocess_input=preprocess,
        shuffle=False,
        seed=seed,
    )

    if args.limit_train_batches is not None:
        train_ds = train_ds.take(args.limit_train_batches)
    if args.limit_val_batches is not None:
        val_ds = val_ds.take(args.limit_val_batches)

    model = build_binary_classifier(config)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=config["training"].get("early_stopping_monitor", "val_loss"),
            patience=int(config["training"].get("early_stopping_patience", 15)),
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["training"]["epochs"]),
        callbacks=callbacks,
    )

    model_path = models_dir / f"{run_id}.keras"
    history_path = logs_dir / f"training_history_{run_id}.csv"

    model.save(model_path)
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    print(f"Wrote model: {model_path}")
    print(f"Wrote history: {history_path}")


if __name__ == "__main__":
    main()
