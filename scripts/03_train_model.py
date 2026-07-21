#!/usr/bin/env python3
"""Train a dermatology classifier using the fixed HAM10000 split."""

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
)
from dermalgo.fixed_split import (
    load_fixed_ham10000_split,
    summarize_fixed_split,
)
from dermalgo.seeds import get_training_seeds
from dermalgo.models import (
    build_binary_classifier,
    get_preprocess_input,
)
from dermalgo.paths import ensure_dir
from dermalgo.tfdata import make_image_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a model using the immutable lesion-grouped "
            "HAM10000 split."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Training-randomization value. It affects model "
            "initialization, oversampling, and training-data order, "
            "but not the HAM10000 partition."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help=(
            "Optional debugging limit for training batches "
            "per epoch."
        ),
    )

    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help=(
            "Optional debugging limit for validation batches."
        ),
    )

    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    """Set TensorFlow, NumPy, and Python random states."""
    tf.keras.utils.set_random_seed(seed)


def main() -> None:
    """Train one architecture under one randomized training run."""
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

    configured_seeds = (
        config["training"]
        .get("seeds", [])
    )

    if args.seed is not None:
        training_seed = int(args.seed)
    elif configured_seeds:
        training_seed = int(
            configured_seeds[0]
        )
    else:
        raise ValueError(
            "No training seed was supplied and no seeds are "
            "defined in the configuration."
        )


    allowed_training_seeds = get_training_seeds()

    configured_training_seeds = [
        int(value)
        for value in configured_seeds
    ]

    if configured_training_seeds != allowed_training_seeds:
        raise ValueError(
            "The training seeds in the architecture configuration "
            "do not match config/random_seeds.json. "
            f"Configured: {configured_training_seeds}; "
            f"policy: {allowed_training_seeds}."
        )

    if training_seed not in allowed_training_seeds:
        raise ValueError(
            f"Training seed {training_seed} is not part of the "
            f"prespecified seed policy: {allowed_training_seeds}."
        )

    set_reproducibility(
        training_seed
    )

    model_name = config["model"]["name"]

    batch_size = int(
        config["training"]["batch_size"]
    )

    image_size = tuple(
        config["model"]["input_size"]
    )

    models_dir = ensure_dir(
        config["outputs"]["models_dir"]
    )

    logs_dir = ensure_dir(
        config["outputs"]["logs_dir"]
    )

    tables_dir = ensure_dir(
        config["outputs"]["tables_dir"]
    )

    run_timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_id = (
        f"{model_name}_seed{training_seed}_"
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
        train_df,
        validation_df,
        test_df,
        split_manifest_path,
        split_manifest_sha256,
    ) = load_fixed_ham10000_split(
        ham_df,
        manifest_path=(
            config["split"]["manifest_path"]
        ),
    )

    split_summary = summarize_fixed_split(
        train_df,
        validation_df,
        test_df,
    )

    split_summary_path = (
        tables_dir
        / f"ham10000_fixed_split_summary_{run_id}.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    balancing_method = (
        config["training"]
        .get("oversampling")
    )

    if balancing_method == "train_only":
        train_fit_df = (
            oversample_training_dataframe(
                train_df,
                seed=training_seed,
            )
        )
    elif balancing_method in {
        None,
        "none",
    }:
        train_fit_df = (
            train_df.copy()
        )
    else:
        raise ValueError(
            "Unsupported training balancing method: "
            f"{balancing_method}"
        )

    oversampled_counts_path = (
        tables_dir
        / (
            "ham10000_training_counts_after_balancing_"
            f"{run_id}.csv"
        )
    )

    (
        train_fit_df["label_binary"]
        .value_counts()
        .sort_index()
        .rename_axis("label_binary")
        .reset_index(name="n")
        .to_csv(
            oversampled_counts_path,
            index=False,
        )
    )

    training_manifest = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "run_id": run_id,
        "config_path": str(
            args.config
        ),
        "model_name": model_name,
        "training_randomization_seed": (
            training_seed
        ),
        "split_manifest_path": str(
            split_manifest_path
        ),
        "split_manifest_sha256": (
            split_manifest_sha256
        ),
        "split_grouping_variable": (
            "lesion_id"
        ),
        "prediction_unit": "image",
        "dry_run": bool(
            args.dry_run
        ),
        "limit_train_batches": (
            args.limit_train_batches
        ),
        "limit_val_batches": (
            args.limit_val_batches
        ),
        "n_ham10000_images": int(
            len(ham_df)
        ),
        "n_train_before_balancing": int(
            len(train_df)
        ),
        "n_validation": int(
            len(validation_df)
        ),
        "n_internal_test": int(
            len(test_df)
        ),
        "n_train_after_balancing": int(
            len(train_fit_df)
        ),
        "training_balancing_method": (
            balancing_method
        ),
        "split_summary_path": str(
            split_summary_path
        ),
        "balanced_counts_path": str(
            oversampled_counts_path
        ),
        "model": config["model"],
        "training": config["training"],
    }

    training_manifest_path = (
        logs_dir
        / f"train_manifest_{run_id}.json"
    )

    training_manifest_path.write_text(
        json.dumps(
            training_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Training run")
    print("------------")
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print(
        "Training-randomization seed: "
        f"{training_seed}"
    )
    print(
        "Fixed split manifest: "
        f"{split_manifest_path}"
    )
    print(
        "Fixed split SHA-256: "
        f"{split_manifest_sha256}"
    )
    print(f"Input size: {image_size}")
    print(
        "Training before balancing: "
        f"{len(train_df)}"
    )
    print(
        "Training after balancing: "
        f"{len(train_fit_df)}"
    )
    print(
        f"Validation: {len(validation_df)}"
    )
    print(
        f"Internal test: {len(test_df)}"
    )
    print()
    print(split_summary.to_string(index=False))
    print()
    print(
        f"Wrote training manifest: "
        f"{training_manifest_path}"
    )

    if args.dry_run:
        print()
        print(
            "Dry run completed. No model was trained."
        )
        return

    preprocess_input = get_preprocess_input(
        config["model"]["keras_application"]
    )

    train_dataset = make_image_dataset(
        train_fit_df,
        image_size=image_size,
        batch_size=batch_size,
        preprocess_input=preprocess_input,
        shuffle=True,
        seed=training_seed,
    )

    validation_dataset = make_image_dataset(
        validation_df,
        image_size=image_size,
        batch_size=batch_size,
        preprocess_input=preprocess_input,
        shuffle=False,
        seed=training_seed,
    )

    if args.limit_train_batches is not None:
        train_dataset = train_dataset.take(
            args.limit_train_batches
        )

    if args.limit_val_batches is not None:
        validation_dataset = (
            validation_dataset.take(
                args.limit_val_batches
            )
        )

    model = build_binary_classifier(
        config
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=(
                config["training"].get(
                    "early_stopping_monitor",
                    "val_loss",
                )
            ),
            patience=int(
                config["training"].get(
                    "early_stopping_patience",
                    15,
                )
            ),
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(
            config["training"]["epochs"]
        ),
        callbacks=callbacks,
    )

    model_path = (
        models_dir
        / f"{run_id}.keras"
    )

    history_path = (
        logs_dir
        / f"training_history_{run_id}.csv"
    )

    model.save(
        model_path
    )

    pd.DataFrame(
        history.history
    ).to_csv(
        history_path,
        index=False,
    )

    print()
    print(f"Wrote model: {model_path}")
    print(
        f"Wrote training history: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()
